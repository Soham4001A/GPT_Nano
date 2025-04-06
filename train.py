"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

Includes optional HellaSwag evaluation.

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False --use_gated_reduction=True --d_reduction_factor=2

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py --use_gated_reduction=True --d_reduction_factor=2

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py [gated_reduction_args...]
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py [gated_reduction_args...]
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
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
from model import GPTConfig, GPT, evaluate_hellaswag # <--- IMPORT evaluate_hellaswag

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 1000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# Gated Reduction specific flags (NEW)
use_gated_reduction = True # Default based on your last model.py update
d_reduction_factor = 4     # Default: 1 means no reduction (gating_d_new = n_embd)
gating_d_new = None        # Will be calculated later based on n_embd and d_reduction_factor
# adamw optimizer
learning_rate = 3e-4 # 6e-4 max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 1e-6 # 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # Default backend, will be adjusted based on device
# system
# --- Determine device and backend ---
if torch.cuda.is_available():
    device = 'cuda'
    backend = 'nccl' # NCCL is generally preferred for CUDA DDP
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
    # Check if DDP is active later, and force CPU if DDP+MPS
    backend = 'gloo' # Gloo *might* work, but often CPU fallback needed
    print("WARNING: Using MPS device. DDP support might be limited or experimental.")
else:
    device = 'cpu'
    backend = 'gloo' # Use Gloo for CPU distributed training
# --- End device/backend determination ---

# Dtype and Autocast setup
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # Use float16 by default if no bfloat16 support
compile = False # Disable torch.compile initially for broader compatibility/debugging

# --- HellaSwag ---
hellaswag = True # Default to True, override with config file or cmd line
hellaswag_path = 'data/hellaswag/hellaswag_val.jsonl' # Default path

# -----------------------------------------------------------------------------
# Load config overrides from command line or config file
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))] # Include NoneType
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# ----- Derive gating_d_new AFTER config loading -----
if use_gated_reduction:
    if 'gating_d_new' in config and config['gating_d_new'] is not None:
        # If gating_d_new was explicitly set via command line/config file, use it
        gating_d_new = config['gating_d_new']
        print(f"Using explicitly set gating_d_new: {gating_d_new}")
        if not isinstance(gating_d_new, int) or gating_d_new <= 0:
            raise ValueError(f"Explicit gating_d_new must be a positive integer, got {gating_d_new}")
    else:
        # Calculate from n_embd and d_reduction_factor
        if not isinstance(d_reduction_factor, int) or d_reduction_factor < 1:
             raise ValueError(f"d_reduction_factor must be an integer >= 1, got {d_reduction_factor}")
        if n_embd % d_reduction_factor != 0:
            print(f"Warning: n_embd ({n_embd}) is not perfectly divisible by d_reduction_factor ({d_reduction_factor}).")
        # Use integer division
        gating_d_new = n_embd // d_reduction_factor
        # Ensure it's at least 1 (or maybe n_head if we want strict divisibility later)
        gating_d_new = max(1, gating_d_new)
        print(f"Calculated gating_d_new: {n_embd} // {d_reduction_factor} = {gating_d_new}")
    # Update config dict with the final derived/validated value
    config['gating_d_new'] = gating_d_new
else:
    # Ensure gating_d_new is None if reduction is not used
    gating_d_new = None
    config['gating_d_new'] = None
    # If reduction factor was set but use_gated_reduction is false, maybe warn?
    if d_reduction_factor != 1:
        print(f"Warning: d_reduction_factor ({d_reduction_factor}) is set, but use_gated_reduction is False. Factor will be ignored.")
# ----- End deriving gating_d_new -----


# ----- DDP and Device Setup -----
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # Check/adjust backend based on final device choice (esp. MPS)
    if device == 'mps':
        print("Warning: DDP requested with MPS device. Forcing CPU backend/device due to compatibility issues.")
        device = 'cpu'    # Force CPU for DDP if MPS was initially detected
        backend = 'gloo'  # Ensure Gloo backend for CPU DDP
    elif backend == 'nccl' and not torch.cuda.is_available():
        print("Warning: NCCL backend specified but CUDA not available. Switching to Gloo.")
        backend = 'gloo'

    # Initialize process group
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    # Assign device based on local rank only if using CUDA
    if device == 'cuda':
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
    # For CPU DDP, 'device' remains 'cpu'
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
# ------------------------------------

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")
print(f"Using device: {device}, Backend: {backend if ddp else 'N/A'}") # Log final device/backend

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
# Determine device type string and PyTorch dtype
if 'cuda' in device: device_type = 'cuda'
elif 'mps' in device: device_type = 'mps'
else: device_type = 'cpu'

# Adjust dtype and autocast context based on final device_type
if device_type == 'cuda':
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
else: # CPU or MPS
    ptdtype = torch.float32
    dtype = 'float32' # Force float32 for CPU/MPS
    ctx = nullcontext()
    if device_type == 'mps': print("Using float32 on MPS device, disabling Autocast.")

print(f"Using PyTorch dtype: {ptdtype}")

# ---- Data Loader ----
data_dir = os.path.join('data', dataset)
train_data_path = os.path.join(data_dir, 'train.bin')
val_data_path = os.path.join(data_dir, 'val.bin')
# Check if data files exist
if not os.path.exists(train_data_path) or not os.path.exists(val_data_path):
    print("\nERROR: Training data (.bin files) not found.")
    print(f"Expected locations: {train_data_path}, {val_data_path}")
    print(f"Please ensure the '{dataset}' dataset is prepared correctly in the '{data_dir}' directory.")
    print("You may need to run the data preparation script (e.g., prepare.py for openwebtext).")
    exit(1) # Exit if data is missing

def get_batch(split):
    data_path = train_data_path if split == 'train' else val_data_path
    try:
        data = np.memmap(data_path, dtype=np.uint16, mode='r')
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        raise # Re-raise the exception
    except Exception as e:
        print(f"Error memory mapping file {data_path}: {e}")
        raise

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    # Move data to the correct device
    x, y = x.to(device), y.to(device)
    return x, y
# --------------------

# ---- Tokenizer for HellaSwag ----
# NOTE: Requires tiktoken (`pip install tiktoken`)
try:
    enc = tiktoken.get_encoding("gpt2")
except ImportError:
     print("Warning: tiktoken not installed. HellaSwag evaluation will be disabled.")
     print("Install tiktoken: pip install tiktoken")
     hellaswag = False # Disable hellaswag if tiktoken is missing
except Exception as e:
     print(f"Error initializing tiktoken: {e}")
     hellaswag = False # Disable hellaswag on other tiktoken errors
# ---------------------------------

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset meta file
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
else:
     print(f"Warning: meta.pkl not found in {data_dir}. Using default vocab_size=50304")


# ---- Model Initialization ----
# Ensure gating_d_new derived above is used here
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    bias=bias, vocab_size=None, dropout=dropout,
    use_gated_reduction=use_gated_reduction, # Pass gating flag
    gating_d_new=gating_d_new # Pass derived/validated d_new
)
print("Model Arguments Being Passed to GPTConfig:")
print(model_args)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    # Determine vocab size: use meta if available, otherwise use a reasonable default
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    if meta_vocab_size is None:
        print(f"Warning: vocab_size not found in meta.pkl, using default: {model_args['vocab_size']}")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint file not found at {ckpt_path}. Cannot resume.")
        exit(1)
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']

    # ---- Check for compatibility between checkpoint and current config ----
    # Check gated reduction flag consistency
    ckpt_use_gated = checkpoint_model_args.get('use_gated_reduction', False) # Default to False if missing
    if use_gated_reduction != ckpt_use_gated:
        print("\n!!! WARNING: Mismatch in 'use_gated_reduction' between config and checkpoint! !!!")
        print(f"  Config: use_gated_reduction = {use_gated_reduction}")
        print(f"  Checkpoint: use_gated_reduction = {ckpt_use_gated}")
        print("Loading checkpoint's setting. Ensure this is intended.")
        use_gated_reduction = ckpt_use_gated # Prioritize checkpoint setting
        model_args['use_gated_reduction'] = use_gated_reduction # Update args

    # Check gating_d_new consistency if gating is enabled in checkpoint
    if use_gated_reduction:
        ckpt_gating_d_new = checkpoint_model_args.get('gating_d_new', None)
        # Compare with currently derived/set gating_d_new
        if gating_d_new != ckpt_gating_d_new:
            print("\n!!! WARNING: Mismatch in 'gating_d_new' between config/derived and checkpoint! !!!")
            print(f"  Config/Derived: gating_d_new = {gating_d_new}")
            print(f"  Checkpoint: gating_d_new = {ckpt_gating_d_new}")
            if ckpt_gating_d_new is not None:
                print("Loading checkpoint's 'gating_d_new'.")
                gating_d_new = ckpt_gating_d_new # Prioritize checkpoint's value
                model_args['gating_d_new'] = gating_d_new # Update args
            else:
                 print("Checkpoint missing 'gating_d_new' but has use_gated_reduction=True. Using config/derived value. Verify model structure.")

    # Force core architecture settings from checkpoint
    forced_keys = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']
    for k in forced_keys:
        if k in checkpoint_model_args:
             # Check for mismatch and warn if config differs from checkpoint
             if k in model_args and model_args[k] != checkpoint_model_args[k]:
                  print(f"Warning: Config value for '{k}' ({model_args[k]}) differs from checkpoint ({checkpoint_model_args[k]}). Using checkpoint value.")
             model_args[k] = checkpoint_model_args[k]
        else:
             print(f"Warning: Checkpoint missing essential arg '{k}'. Using default/cmd line value: {model_args.get(k)}")
             # If vocab_size is missing, try using meta_vocab_size again
             if k == 'vocab_size' and model_args.get(k) is None:
                 model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
                 print(f"Attempting to set missing vocab_size to {model_args['vocab_size']}")

    # Re-create model config and model with potentially updated args
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    # Load model state dict
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    # Load training state
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    print(f"Resumed from iteration {iter_num} with best_val_loss {best_val_loss:.4f}")

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # --- CRITICAL CHECK ---
    if use_gated_reduction:
        print("\nERROR: Cannot initialize model with 'use_gated_reduction=True' from standard GPT-2 weights.")
        print("Standard GPT-2 checkpoints do not have the necessary TimeStepGatedReduction layers.")
        print("Set 'use_gated_reduction=False' or train from scratch/resume a gated checkpoint.")
        exit(1)
    # --- End Check ---

    override_args = dict(dropout=dropout)
    # Ensure gating args are explicitly off when loading standard GPT-2
    override_args['use_gated_reduction'] = False
    override_args['gating_d_new'] = None

    model = GPT.from_pretrained(init_from, override_args)
    # Read back the config parameters from the loaded model (they are already set by from_pretrained)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
    # Ensure our derived config matches the loaded model's config
    model_args['use_gated_reduction'] = False
    model_args['gating_d_new'] = None
# --------------------------

# Crop block size if needed (must happen AFTER model init)
if block_size < model.config.block_size:
    print(f"Cropping model block size from {model.config.block_size} to {block_size}")
    try:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # Update configuration recording
    except NotImplementedError as e:
        print(f"Warning: Could not crop block size - {e}")
    except AttributeError:
         print("Warning: Model does not have 'crop_block_size' method. Skipping block size cropping.")


model.to(device) # Move model to device

# ---- Optimizer and Scaler ----
# Determine scaler enabled status based on the effective dtype being used
scaler_enabled = (dtype == 'float16') # Enable scaler only if using float16
scaler = torch.amp.GradScaler(enabled=scaler_enabled)
print(f"Using GradScaler: {scaler_enabled}")

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume' and 'optimizer' in checkpoint: # Check if optimizer state exists
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
    except Exception as e:
        print(f"Warning: Failed to load optimizer state dict: {e}. Initializing optimizer from scratch.")
checkpoint = None # free up memory
# -----------------------------

# ---- Compile Model (Optional) ----
if compile:
    # Check if device supports compile, disable if not (e.g., MPS)
    if device_type not in ['cuda']: # Add other supported types if needed
         print(f"Warning: Disabling torch.compile as it's not fully supported on device '{device_type}'.")
         compile = False
    else:
         print("compiling the model... (takes a ~minute)")
         unoptimized_model = model
         try:
             # Suggested compile options for potentially better performance
             model = torch.compile(model, mode="reduce-overhead", fullgraph=True) # requires PyTorch 2.0+
             print("Model compiled successfully.")
         except Exception as e:
             print(f"Warning: Model compilation failed: {e}. Proceeding without compilation.")
             compile = False # Fallback if compilation fails
             model = unoptimized_model # Use the original model
# -------------------------------

# ---- Wrap model in DDP ----
if ddp:
    # Check for MPS incompatibility again before wrapping
    if device_type == 'mps':
         print("ERROR: Cannot use DDP with MPS device due to backend limitations.")
         exit(1) # Exit if DDP+MPS requested
    # find_unused_parameters might be needed if gating layers aren't used in every forward pass under certain conditions (unlikely here)
    model = DDP(model, device_ids=[ddp_local_rank] if device_type == 'cuda' else None, find_unused_parameters=False)
# --------------------------

# ---- Loss Estimation Function ----
@torch.no_grad()
def estimate_loss(model):
    out = {}
    # model should be passed in eval mode
    model_device = next(model.parameters()).device
    device_type = 'cuda' if 'cuda' in str(model_device) else 'cpu'

    # ---- Determine autocast context INSIDE the function ----
    # Use the same dtype logic as in the main script training part
    if device_type == 'cuda':
        # Use the global 'dtype' variable ('bfloat16' or 'float16')
        # Ensure 'dtype' variable is accessible here or pass it in if needed
        global dtype # Access the global dtype setting
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        eval_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    else: # CPU or MPS
        eval_ctx = nullcontext()
    # ---- End context determination ----

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=model_device) # Use model's device
        for k in range(eval_iters):
            X, Y = get_batch(split)
            # Move batch data explicitly to model's device
            X, Y = X.to(model_device), Y.to(model_device)
            with eval_ctx: # Use locally determined context
                logits, loss = model(X, Y)
            if loss is not None and not torch.isnan(loss): losses[k] = loss.item()
            else: losses[k] = float('nan')
        valid_losses = losses[~torch.isnan(losses)]
        out[split] = valid_losses.mean() if len(valid_losses) > 0 else float('inf')

    if hellaswag and master_process:
        eval_model = model.module if ddp else model
        # evaluate_hellaswag now determines its own context
        hellaswag_acc = evaluate_hellaswag(eval_model, enc, hellaswag_path) # Pass path only
        out['hellaswag'] = hellaswag_acc if hellaswag_acc is not None else -1.0
    elif hellaswag:
         out['hellaswag'] = 0.0

    return out
# -----------------------------

# ---- LR Scheduler ----
def get_lr(it):
    if not decay_lr: return learning_rate # Return fixed LR if decay is off
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1) # Use it+1 and warmup_iters+1 for smoother start
    # 2) if it > lr_decay_iters, return min learning rate
    if it >= lr_decay_iters: # Use >= to include the last step
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (learning_rate - min_lr)
# --------------------

# ---- Logging Setup ----
if wandb_log and master_process:
    import wandb
    # Ensure config passed to wandb includes the final gating settings
    run_config = config.copy() # Use the global config dict which includes derived gating_d_new
    # Add any specific derived values if needed (already in config dict)
    print("Logging config to WandB:")
    print(run_config)
    try:
        wandb.init(project=wandb_project, name=wandb_run_name, config=run_config)
    except Exception as e:
        print(f"Error initializing WandB: {e}. Disabling WandB logging.")
        wandb_log = False # Disable logging if init fails
# -----------------------

# ---- Training Loop ----
X, Y = get_batch('train') # Fetch first batch
t0 = time.time()
local_iter_num = 0 # number of iterations run on this process (for MFU warmup)
raw_model = model.module if ddp else model # Get the unwrapped model for saving/MFU
running_mfu = -1.0
print(f"\nStarting training loop from iteration {iter_num}...")
while True:

    # Determine and set LR for the current iteration
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate loss and save checkpoints at eval_interval
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss(model) # Pass the potentially DDP-wrapped model
        print_str = f"step {iter_num}: train loss {losses.get('train', float('nan')):.4f}, val loss {losses.get('val', float('nan')):.4f}"
        if hellaswag and 'hellaswag' in losses: # Check if key exists
            print_str += f", HellaSwag Acc: {losses['hellaswag']:.4f}"
        print(print_str)

        if wandb_log:
            try:
                 log_data = { "iter": iter_num, "train/loss": losses.get('train', float('nan')), "val/loss": losses.get('val', float('nan')), "lr": lr, "mfu": running_mfu*100 }
                 if hellaswag and 'hellaswag' in losses: log_data['val/hellaswag_acc'] = losses['hellaswag']
                 wandb.log(log_data)
            except Exception as e:
                 print(f"WandB logging failed: {e}")

        current_val_loss = losses.get('val', float('inf')) # Handle case where val loss might be NaN/missing
        # Save checkpoint if it's the best so far or always_save is true
        if current_val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = current_val_loss if current_val_loss != float('inf') else best_val_loss # Only update if valid loss
            if iter_num > 0: # Don't save initial checkpoint at iter 0 unless requested?
                # Save the unwrapped model's state_dict
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args, # Save the args used to init the model
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config, # Save the full config used for this run
                }
                print(f"saving checkpoint to {out_dir} (val_loss: {best_val_loss:.4f})")
                ckpt_path = os.path.join(out_dir, 'ckpt.pt')
                # Save atomically (save to temp file, then rename)
                temp_ckpt_path = ckpt_path + ".tmp"
                torch.save(checkpoint, temp_ckpt_path)
                os.rename(temp_ckpt_path, ckpt_path) # Atomic rename
    if iter_num == 0 and eval_only:
        print("eval_only=True, exiting after first evaluation.")
        break

    # ----- Training Step -----
    model.train() # Ensure model is in training mode
    # Forward backward update with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        # DDP specific logic for gradient sync
        if ddp:
            # only sync gradients on the last micro-step.
            # Note: While model.no_sync() context manager is recommended,
            # setting require_backward_grad_sync is a common alternative.
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

        with ctx: # Apply autocast context
            logits, loss = model(X, Y)
            # Check for NaN loss immediately after forward pass
            if torch.isnan(loss):
                 print(f"ERROR: Loss is NaN at iter {iter_num}, micro_step {micro_step}. Forward pass produced NaN.")
                 print(f"  Logits sample (sum): {logits.sum().item() if logits is not None else 'N/A'}")
                 # Consider additional debugging: check inputs X, Y, model params for NaNs
                 # For now, exit to prevent further issues.
                 exit(1)

            loss = loss / gradient_accumulation_steps # Scale loss for accumulation

        # Immediately async prefetch next batch while CPU is potentially free
        # Note: get_batch itself needs to be efficient for this to be useful
        X_next, Y_next = get_batch('train')

        # Backward pass with scaler
        scaler.scale(loss).backward()

        # Move prefetched batch to current batch variables
        X, Y = X_next, Y_next
    # ----- End Micro-steps -----

    # Gradient Clipping (after accumulation, before optimizer step)
    if grad_clip > 0.0:
        scaler.unscale_(optimizer) # Need to unscale gradients before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer Step (advances model parameters)
    scaler.step(optimizer)
    scaler.update() # Update scaler for next iteration
    # Flush gradients after optimizer step
    optimizer.zero_grad(set_to_none=True)

    # Timing and Logging
    t1 = time.time(); dt = t1 - t0; t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to get the approximate loss summed over accumulation steps
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit (e.g. MFU calculation)
            # Use the unwrapped model to estimate MFU
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        # Log step time, loss, and MFU
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

print("Training finished.")
# Final save? Optionally save the final model state regardless of validation loss
if master_process and not eval_only:
     final_ckpt_path = os.path.join(out_dir, 'ckpt_final.pt')
     print(f"Saving final model checkpoint to {final_ckpt_path}")
     checkpoint = {
         'model': raw_model.state_dict(),
         'optimizer': optimizer.state_dict(),
         'model_args': model_args,
         'iter_num': iter_num,
         'best_val_loss': best_val_loss,
         'config': config,
     }
     temp_ckpt_path = final_ckpt_path + ".tmp"
     torch.save(checkpoint, temp_ckpt_path)
     os.rename(temp_ckpt_path, final_ckpt_path)