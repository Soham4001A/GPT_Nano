"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

Supports standard GPT or GPT with Time-Step Gated Reduction.
Includes optional HellaSwag evaluation.

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False --use_gated_reduction=True --d_reduction_factor=4

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py --use_gated_reduction=True --d_reduction_factor=4
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import tiktoken # <--- IMPORT TIKTOKEN

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Assuming model.py contains GPTConfig, GPT, and evaluate_hellaswag
# Make sure the imported evaluate_hellaswag uses float32 internally!
from model import GPTConfig, GPT, evaluate_hellaswag

# -----------------------------------------------------------------------------
# Default config values
# I/O
out_dir = 'out'
eval_interval = 1000
log_interval = 10 # Changed default to log less frequently
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'owt'
wandb_run_name = 'G-LMA' # Adjusted default name
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes -> 40
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
user_config = input("Which Config? (16/12/reduc_2/reduc_1.5) ")
if user_config == "12":
    n_head = 12
    d_reduction_factor = 4
elif user_config == "16":
    n_head = 16
    d_reduction_factor = 3
elif user_config == "reduc_2":
    n_head = 16
    d_reduction_factor = 2
elif user_config == "reduc_1.5":
    n_head = 16
    d_reduction_factor = 1.5
else:
    print("invalid config")
    exit(1)
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = True # do we use bias inside LayerNorm and Linear layers?
# --- Time-Step Gated Reduction Config --- (NEW DEFAULTS)
use_gated_reduction = True # Default to standard GPT
# gating_d_new will be calculated after config loading based on n_embd & d_reduction_factor
# -----------------------------------------
# adamw optimizer
learning_rate = 9e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 0.07
beta1 = 0.9
beta2 = 0.95 
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # Default backend, will be adjusted based on device
# system
# --- Determine device and backend ---
if torch.cuda.is_available():
    device = 'cuda'
    backend = 'nccl'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
    backend = 'gloo'
    print("WARNING: Using MPS device. DDP support might be limited or experimental.")
else:
    device = 'cpu'
    backend = 'gloo'
# --- End device/backend determination ---

# Dtype and Autocast setup
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False # Disable torch.compile initially

# --- HellaSwag ---
hellaswag = True # Default to True, can be overridden
hellaswag_path = 'data/hellaswag/hellaswag_val.jsonl'

# -----------------------------------------------------------------------------
# Load config overrides from command line or config file
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
# Use try-except for configurator.py in case it doesn't exist
try:
    exec(open('configurator.py').read()) # overrides from command line or config file
    print("Loaded overrides from configurator.py")
except FileNotFoundError:
    print("configurator.py not found, using command line arguments or defaults.")
except Exception as e:
    print(f"Error executing configurator.py: {e}")

# Create config dict AFTER loading overrides
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------


# ----- Derive gating_d_new AFTER config loading -----
gating_d_new_derived = None # Initialize derived value
if use_gated_reduction:
    # Check if gating_d_new was explicitly set (e.g., via command line)
    # Note: This requires 'gating_d_new' to be included in config_keys if settable via cmd line
    if 'gating_d_new' in config and config['gating_d_new'] is not None:
        gating_d_new_derived = config['gating_d_new']
        print(f"Using explicitly set gating_d_new: {gating_d_new_derived}")
        if not isinstance(gating_d_new_derived, int) or gating_d_new_derived <= 0:
            raise ValueError(f"Explicit gating_d_new must be a positive integer, got {gating_d_new_derived}")
    else:
        # Calculate from n_embd and d_reduction_factor
        gating_d_new_derived = int(n_embd // d_reduction_factor)
        gating_d_new_derived = max(1, gating_d_new_derived) # Ensure > 0
        print(f"Calculated initial gating_d_new: {n_embd} // {d_reduction_factor} = {gating_d_new_derived}")

    # --- Robustness Check: Ensure divisibility by n_head ---
    if gating_d_new_derived % n_head != 0:
        original_d_new = gating_d_new_derived
        # Adjust down to the nearest multiple of n_head
        gating_d_new_derived = (gating_d_new_derived // n_head) * n_head
        # Ensure it's at least n_head if the result is too small (or zero)
        gating_d_new_derived = max(n_head, gating_d_new_derived)
        print(f"WARNING: Initial gating_d_new ({original_d_new}) is not divisible by n_head ({n_head}).")
        print(f"Adjusting gating_d_new -> {gating_d_new_derived} to ensure divisibility.")
    # --- End adjustment ---

    # Update the main config dict with the final derived/validated value
    config['gating_d_new'] = gating_d_new_derived
    # Also update the global variable if other parts rely on it directly (though using config dict is better)
    gating_d_new = gating_d_new_derived

else:
    # Ensure gating_d_new is None if reduction is not used
    gating_d_new = None # Update global variable
    config['gating_d_new'] = None # Update config dict
    if d_reduction_factor != 1:
        print(f"Warning: d_reduction_factor ({d_reduction_factor}) is set, but use_gated_reduction is False. Factor will be ignored.")
# ----- End deriving gating_d_new -----


# ----- DDP and Device Setup -----
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    if device == 'mps':
        print("Warning: DDP requested with MPS device. Forcing CPU backend/device.")
        device = 'cpu'
        backend = 'gloo'
    elif backend == 'nccl' and not torch.cuda.is_available():
        print("Warning: NCCL backend specified but CUDA not available. Switching to Gloo.")
        backend = 'gloo'

    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    if device == 'cuda':
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    if gradient_accumulation_steps % ddp_world_size != 0:
         print(f"Warning: gradient_accumulation_steps ({gradient_accumulation_steps}) not divisible by world size ({ddp_world_size}). Effective steps may vary.")
    gradient_accumulation_steps //= ddp_world_size # Adjust accumulation steps per process
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
# ------------------------------------

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")
print(f"Using device: {device}, Backend: {backend if ddp else 'N/A'}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'mps' if 'mps' in device else 'cpu'

# Autocast context setup for training
if device_type == 'cuda':
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
else:
    ptdtype = torch.float32
    dtype = 'float32' # Force float32 string for logging if CPU/MPS
    ctx = nullcontext()
    if device_type == 'mps': print("Using float32 on MPS device, disabling Autocast for training.")

print(f"Using PyTorch dtype for training: {ptdtype}")

# ---- Data Loader ----
data_dir = os.path.join('data', dataset)
train_data_path = os.path.join(data_dir, 'train.bin')
val_data_path = os.path.join(data_dir, 'val.bin')
if not os.path.exists(train_data_path) or not os.path.exists(val_data_path):
    print("\nERROR: Training data (.bin files) not found.")
    print(f"Expected locations: {train_data_path}, {val_data_path}")
    exit(1)

def get_batch(split):
    data_path = train_data_path if split == 'train' else val_data_path
    try:
        # Use context manager for memmap
        with open(data_path, 'rb') as f:
             # Use mode='r' for read-only access
             data = np.memmap(f, dtype=np.uint16, mode='r')
             total_len = len(data)
             if total_len < block_size + 1:
                 raise ValueError(f"Dataset file {data_path} is too small ({total_len} tokens) for block_size {block_size}")
             # Generate random starting indices
             ix = torch.randint(total_len - block_size, (batch_size,))
             # Stack tensors directly on the target device
             x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix]).to(device)
             y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix]).to(device)
        # data object (memmap) is closed automatically here
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        raise
    except Exception as e:
        print(f"Error reading or processing file {data_path}: {e}")
        raise
    return x, y
# --------------------

# ---- Tokenizer for HellaSwag ----
enc = None # Initialize enc to None
if hellaswag: # Only try to load if hellaswag is enabled
    try:
        enc = tiktoken.get_encoding("gpt2")
        print("Tiktoken loaded successfully for HellaSwag.")
    except ImportError:
         print("Warning: tiktoken not installed. HellaSwag evaluation will be disabled.")
         hellaswag = False
    except Exception as e:
         print(f"Error initializing tiktoken: {e}. HellaSwag evaluation disabled.")
         hellaswag = False
# ---------------------------------

# init these up here, can override if init_from='resume'
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset meta file
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    try:
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    except Exception as e:
        print(f"Warning: Could not load meta.pkl: {e}")
else:
     print(f"Warning: meta.pkl not found in {data_dir}. Will use default vocab_size=50304.")

# ---- Model Initialization ----
# Use the derived gating_d_new value here
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    bias=bias, vocab_size=None, dropout=dropout,
    use_gated_reduction=use_gated_reduction,
    gating_d_new=gating_d_new # Use the potentially adjusted value calculated above
)
# d_reduction_factor is not a direct model arg, it's used to calculate gating_d_new
print("\n--- Model Arguments Being Passed to GPTConfig ---")
print(model_args)

checkpoint_data = None # To store loaded checkpoint data if resuming

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    if model_args['vocab_size'] == 50304 and meta_vocab_size is None:
        print(f"Warning: vocab_size not found in meta.pkl, using default: {model_args['vocab_size']}")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint file not found at {ckpt_path}. Cannot resume.")
        exit(1)
    try:
        checkpoint_data = torch.load(ckpt_path, map_location=device) # Load checkpoint
    except Exception as e:
        print(f"Error loading checkpoint file {ckpt_path}: {e}")
        exit(1)

    checkpoint_model_args = checkpoint_data['model_args']
    print("--- Checkpoint Model Args ---")
    print(checkpoint_model_args)

    # --- Compatibility Check: Gated Reduction ---
    ckpt_use_gated = checkpoint_model_args.get('use_gated_reduction', False)
    current_use_gated = model_args['use_gated_reduction'] # From current config/defaults
    if current_use_gated != ckpt_use_gated:
        print("\n!!! WARNING: Mismatch in 'use_gated_reduction' between config and checkpoint! !!!")
        print(f"  Config: use_gated_reduction = {current_use_gated}")
        print(f"  Checkpoint: use_gated_reduction = {ckpt_use_gated}")
        print(">> Using checkpoint's setting for 'use_gated_reduction'. <<")
        model_args['use_gated_reduction'] = ckpt_use_gated # Prioritize checkpoint

    if model_args['use_gated_reduction']:
        ckpt_gating_d_new = checkpoint_model_args.get('gating_d_new', None)
        current_gating_d_new = model_args['gating_d_new'] # Calculated based on current config
        if current_gating_d_new != ckpt_gating_d_new:
            print("\n!!! WARNING: Mismatch in 'gating_d_new' between config/derived and checkpoint! !!!")
            print(f"  Config/Derived: gating_d_new = {current_gating_d_new}")
            print(f"  Checkpoint: gating_d_new = {ckpt_gating_d_new}")
            if ckpt_gating_d_new is not None:
                print(">> Using checkpoint's 'gating_d_new'. <<")
                model_args['gating_d_new'] = ckpt_gating_d_new # Prioritize checkpoint
            else:
                 print(">> Checkpoint missing 'gating_d_new' but 'use_gated_reduction=True'. Using current config/derived value. Verify. <<")
    # --- End Compatibility Check ---

    # Force core architecture settings from checkpoint
    forced_keys = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']
    for k in forced_keys:
        if k in checkpoint_model_args:
            if k in model_args and model_args[k] != checkpoint_model_args[k]:
                 print(f"Warning: Config value for '{k}' ({model_args[k]}) differs from checkpoint ({checkpoint_model_args[k]}). Using checkpoint value.")
            model_args[k] = checkpoint_model_args[k]
        else:
             # Handle missing essential args - critical for vocab_size
             if k == 'vocab_size':
                 model_args[k] = meta_vocab_size if meta_vocab_size is not None else 50304
                 print(f"Warning: Checkpoint missing '{k}'. Using meta/default value: {model_args[k]}")
             else:
                 print(f"Warning: Checkpoint missing essential arg '{k}'. Using current config value: {model_args.get(k)}")

    # Re-create model config and model with potentially updated args
    print("\n--- Final Model Args after Checkpoint Merge ---")
    print(model_args)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    # Load model state dict
    state_dict = checkpoint_data['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    # Load training state
    iter_num = checkpoint_data['iter_num']
    best_val_loss = checkpoint_data['best_val_loss']
    print(f"Resumed from iteration {iter_num} with best_val_loss {best_val_loss:.4f}")

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # --- CRITICAL CHECK ---
    if use_gated_reduction: # Check the flag derived from current config
        print("\nERROR: Cannot initialize model with 'use_gated_reduction=True' from standard GPT-2 weights.")
        print("Set 'use_gated_reduction=False' in config or use '--use_gated_reduction=False' command line arg.")
        exit(1)
    # --- End Check ---

    override_args = dict(dropout=dropout)
    # Explicitly ensure gating is OFF when loading standard GPT-2
    override_args['use_gated_reduction'] = False
    override_args['gating_d_new'] = None

    model = GPT.from_pretrained(init_from, override_args)
    # Read back the config parameters from the loaded model
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
    # Update model_args to reflect the actual loaded config (gating is off)
    model_args['use_gated_reduction'] = False
    model_args['gating_d_new'] = None
else:
    raise ValueError(f"Unknown init_from option: {init_from}")
# --------------------------

# Crop block size if needed
if block_size < model.config.block_size:
    print(f"Cropping model block size from {model.config.block_size} to {block_size}")
    try:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # Update configuration record
    except AttributeError:
         print("Warning: Model does not have 'crop_block_size' method. Skipping block size cropping.")
    except Exception as e:
         print(f"Warning: Could not crop block size: {e}")

model.to(device) # Move model to device

# ---- Optimizer and Scaler ----
scaler_enabled = (dtype == 'float16') # Enable scaler only if using float16
scaler = torch.amp.GradScaler(enabled=scaler_enabled)
print(f"Using GradScaler: {scaler_enabled}")

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume' and checkpoint_data and 'optimizer' in checkpoint_data:
    try:
        optimizer.load_state_dict(checkpoint_data['optimizer'])
        print("Loaded optimizer state from checkpoint.")
    except Exception as e:
        print(f"Warning: Failed to load optimizer state dict: {e}. Initializing optimizer from scratch.")
checkpoint_data = None # free up memory
# -----------------------------

# ---- Compile Model (Optional) ----
if compile:
    if device_type not in ['cuda']:
         print(f"Warning: Disabling torch.compile as it's not fully supported on device '{device_type}'.")
         compile = False
    else:
         print("compiling the model... (takes a ~minute)")
         try:
             model = torch.compile(model) # , mode="reduce-overhead", fullgraph=True) # Add mode options if desired
             print("Model compiled successfully.")
         except Exception as e:
             print(f"Warning: Model compilation failed: {e}. Proceeding without compilation.")
             compile = False # Fallback if compilation fails
             # No need to reassign model, it wasn't replaced if compile failed
# -------------------------------

# ---- Wrap model in DDP ----
if ddp:
    if device_type == 'mps':
         print("ERROR: Cannot use DDP with MPS device.")
         exit(1)
    # find_unused_parameters=True can be safer but slower if some params aren't used in every forward pass
    # For this TimeStepGated model, it's likely not needed unless dropout is very high or layers are conditional
    model = DDP(model, device_ids=[ddp_local_rank] if device_type == 'cuda' else None, find_unused_parameters=False)
# --------------------------

# ---- Loss Estimation Function ----
# NOTE: This uses the global `dtype` for eval context. HellaSwag eval precision
# MUST be handled INSIDE the evaluate_hellaswag function itself (force float32 there).
@torch.no_grad()
def estimate_loss(eval_model): # Pass the potentially DDP-wrapped model
    out = {}
    eval_model.eval() # Set model to eval mode
    model_device = next(eval_model.parameters()).device
    eval_device_type = 'cuda' if 'cuda' in str(model_device) else ('mps' if 'mps' in str(model_device) else 'cpu')

    # Determine autocast context for val loss estimation based on TRAINING dtype
    if eval_device_type == 'cuda':
        # Use the global 'dtype' variable defined for training
        global dtype
        eval_ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        loss_eval_ctx = torch.amp.autocast(device_type=eval_device_type, dtype=eval_ptdtype)
        # print(f"DEBUG: estimate_loss using context: {eval_ptdtype}") # Optional debug
    else:
        loss_eval_ctx = nullcontext()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=model_device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            # No need to move X,Y again, get_batch puts them on `device` which should match model_device
            with loss_eval_ctx:
                logits, loss = eval_model(X, Y)
            if loss is not None:
                 if torch.isnan(loss): losses[k] = float('nan')
                 else: losses[k] = loss.item()
            else: losses[k] = float('nan') # Handle case where loss might be None (shouldn't happen with targets)
        valid_losses = losses[~torch.isnan(losses)]
        out[split] = valid_losses.mean().item() if len(valid_losses) > 0 else float('inf')

    # HellaSwag evaluation (only on master process)
    if hellaswag and master_process:
        # IMPORTANT: Ensure evaluate_hellaswag uses float32 internally!
        # This script CANNOT enforce that if evaluate_hellaswag is imported.
        raw_eval_model = eval_model.module if isinstance(eval_model, DDP) else eval_model
        if enc is None: # Check if tokenizer loaded
             print("Warning: HellaSwag evaluation skipped because tokenizer ('enc') is not available.")
             out['hellaswag'] = -1.0
        else:
             try:
                 print("\n--- Running HellaSwag Evaluation ---")
                 # *** REMINDER: Fix evaluate_hellaswag to use float32 context internally ***
                 hellaswag_acc = evaluate_hellaswag(raw_eval_model, enc, hellaswag_path)
                 out['hellaswag'] = hellaswag_acc if hellaswag_acc is not None else -1.0
                 print("--- Finished HellaSwag Evaluation ---")
             except Exception as e:
                 print(f"Error during HellaSwag evaluation: {e}")
                 import traceback
                 traceback.print_exc()
                 out['hellaswag'] = -1.0

    elif hellaswag: # Non-master processes need the key for consistent logging if wandb used across ranks
         out['hellaswag'] = -1.0 # Indicate not calculated on this rank

    eval_model.train() # Set model back to train mode
    return out
# -----------------------------

# ---- LR Scheduler ----
def get_lr(it):
    if not decay_lr: return learning_rate
    # 1) linear warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1) # Start from close to 0
    # 2) if it > lr_decay_iters, return min learning rate
    if it >= lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (learning_rate - min_lr)
# --------------------

# ---- Logging Setup ----
if wandb_log and master_process:
    import wandb
    # Log the final configuration used, including derived/adjusted values
    print("Logging effective config to WandB:")
    print(config)
    try:
        wandb.init(project=wandb_project, name=wandb_run_name, config=config) # Log the potentially adjusted config
    except Exception as e:
        print(f"Error initializing WandB: {e}. Disabling WandB logging.")
        wandb_log = False
# -----------------------

# ---- Training Loop ----
X, Y = get_batch('train') # Fetch first batch
t0 = time.time()
local_iter_num = 0 # number of iterations run on this process
raw_model = model.module if ddp else model # Get the unwrapped model
running_mfu = -1.0
print(f"\nStarting training loop from iteration {iter_num}...")
while True:

    # Determine and set LR for the current iteration
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate loss and save checkpoints at eval_interval
    if iter_num % eval_interval == 0 and master_process:
        print(f"\n--- Evaluating at Step {iter_num} ---")
        losses = estimate_loss(model) # estimate_loss sets model.eval() and back to .train()
        print_str = f"step {iter_num}: train loss {losses.get('train', float('nan')):.4f}, val loss {losses.get('val', float('nan')):.4f}"
        if hellaswag and 'hellaswag' in losses:
            # Print score even if -1.0 to show it ran/failed
            print_str += f", HellaSwag Acc: {losses['hellaswag']:.4f}"
        print(print_str)
        print("------------------------------------")


        if wandb_log:
            try:
                 log_data = {
                     "iter": iter_num,
                     "train/loss": losses.get('train', float('nan')),
                     "val/loss": losses.get('val', float('nan')),
                     "lr": lr,
                     "mfu": running_mfu*100, # Log MFU percentage
                 }
                 if hellaswag and 'hellaswag' in losses:
                      log_data['val/hellaswag_acc'] = losses['hellaswag']
                 wandb.log(log_data)
            except Exception as e:
                 print(f"WandB logging failed: {e}")

        current_val_loss = losses.get('val', float('inf'))
        # Save checkpoint if loss improved or always_save_checkpoint is true
        if current_val_loss < best_val_loss or always_save_checkpoint:
            if current_val_loss < best_val_loss:
                 best_val_loss = current_val_loss
                 print(f"New best validation loss: {best_val_loss:.4f}")
            if iter_num > 0: # Avoid saving at step 0 if not needed
                # Save the unwrapped model's state_dict
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args, # Save the args used to init this model run
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss, # Save the best loss observed so far
                    'config': config, # Save the full config used for this run
                }
                ckpt_path = os.path.join(out_dir, 'ckpt.pt')
                print(f"saving checkpoint to {ckpt_path}")
                # Save atomically
                temp_ckpt_path = ckpt_path + ".tmp"
                torch.save(checkpoint, temp_ckpt_path)
                os.rename(temp_ckpt_path, ckpt_path)
    if iter_num == 0 and eval_only:
        print("eval_only=True, exiting after first evaluation.")
        break

    # ----- Training Step -----
    model.train() # Ensure model is in train mode for the training step
    # Forward backward update with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        # DDP gradient sync control
        if ddp:
            # Prepare for potential DDP synchronization on the last micro-step
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

        with ctx: # Apply training autocast context
            logits, loss = model(X, Y)
            if loss is None: # Should not happen if targets are provided
                 print(f"Warning: Loss is None at iter {iter_num}, micro_step {micro_step}. Skipping backward.")
                 continue # Skip this micro-step if loss is None
            if torch.isnan(loss):
                 print(f"ERROR: Loss is NaN at iter {iter_num}, micro_step {micro_step}. Stopping training.")
                 if ddp: destroy_process_group() # Attempt cleanup before exit
                 exit(1) # Stop training

            loss = loss / gradient_accumulation_steps # Scale loss

        # Immediately async prefetch next batch (if using efficient dataloader)
        # This overlaps data loading with computation. get_batch needs to be fast.
        # X_next, Y_next = get_batch('train') # Uncomment if get_batch is highly optimized

        # Backward pass with scaler
        # scaler.scale(loss).backward() will be done outside the DDP no_sync context if used
        # For require_backward_grad_sync, just call backward directly
        scaler.scale(loss).backward()

        # If not prefetching, get the next batch here
        # X, Y = get_batch('train')
        X, Y = get_batch('train') # Get next batch simple way

    # ----- End Micro-steps -----

    # Gradient Clipping (applied after all accumulation)
    if grad_clip > 0.0:
        scaler.unscale_(optimizer) # Unscale gradients before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer Step
    scaler.step(optimizer)
    scaler.update() # Update scaler state for next iteration
    optimizer.zero_grad(set_to_none=True) # Reset gradients

    # Timing and Logging
    t1 = time.time(); dt = t1 - t0; t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps # Estimate accumulated loss
        if local_iter_num >= 5: # MFU warmup
             # Use the unwrapped model to estimate MFU
             mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
             running_mfu = mfu if running_mfu < 0 else 0.9*running_mfu + 0.1*mfu # More stable update
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}% lr {lr:.2e}")

    iter_num += 1
    local_iter_num += 1

    # Termination condition
    if iter_num > max_iters:
        print(f"Reached max_iters ({max_iters}). Stopping training.")
        break
# ---------------------

# ---- Cleanup ----
if ddp:
    destroy_process_group()
# ---------------

print("\nTraining finished.")
# Final save?
if master_process and not eval_only:
     final_ckpt_path = os.path.join(out_dir, 'ckpt_final.pt')
     print(f"Saving final model checkpoint to {final_ckpt_path}")
     checkpoint = {
         'model': raw_model.state_dict(),
         'optimizer': optimizer.state_dict(),
         'model_args': model_args, # Use the final model_args used for this run
         'iter_num': iter_num,
         'best_val_loss': best_val_loss,
         'config': config, # Save the final config used
     }
     temp_ckpt_path = final_ckpt_path + ".tmp"
     torch.save(checkpoint, temp_ckpt_path)
     os.rename(temp_ckpt_path, final_ckpt_path)
     print("Final checkpoint saved.")