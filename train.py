"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

Includes optional HellaSwag evaluation.

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import tiktoken # <--- IMPORT TIKTOKEN
import torch.nn.functional as F

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
# LMA specific flags (add defaults here if they should be configurable)
use_lma = True
lma_reduction_factor = 3
# adamw optimizer
learning_rate = 3e-4 # 2e-5 or 1e-5
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 500 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 1e-6 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
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
hellaswag = False # Default to False, override with config file or cmd line
hellaswag_path = 'data/hellaswag/hellaswag_val.jsonl' # Default path

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

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
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    # Move data to the correct device
    x, y = x.to(device), y.to(device)
    # Pin memory only if using CUDA DDP? Check if beneficial otherwise.
    # if device_type == 'cuda':
    #     x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    # else:
    #     x, y = x.to(device), y.to(device) # Simple move for CPU/MPS
    return x, y
# --------------------

# ---- Tokenizer for HellaSwag ----
# NOTE: Requires tiktoken (`pip install tiktoken`)
enc = tiktoken.get_encoding("gpt2")
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

# ---- Model Initialization ----
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    bias=bias, vocab_size=None, dropout=dropout,
    use_lma=use_lma, # Pass LMA flag
    lma_reduction_factor=lma_reduction_factor # Pass reduction factor
)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    gate_params = [p for n, p in model.named_parameters() if 'gate_net.' in n] # Identify gate parameters
    main_params = [p for n, p in model.named_parameters() if 'gate_net.' not in n] # All other parameters
    if not gate_params:
        print("WARNING: No parameters found containing 'gate_net.'. Gate training will not work.")
        # Decide how to handle: exit, or skip gate training?
        # For now, let's allow it to continue but gate training steps will do nothing.
        can_train_gate = False
    else:
        can_train_gate = True
        print(f"Identified {len(gate_params)} gate parameters.")
    print(f"Identified {len(main_params)} main parameters.")
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # Ensure crucial args match, others can be overridden
    forced_keys = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'use_lma'] # Added use_lma
    for k in forced_keys:
        # Check if key exists in checkpoint_model_args before assigning
        if k in checkpoint_model_args:
             model_args[k] = checkpoint_model_args[k]
        else:
             print(f"Warning: Checkpoint missing arg '{k}'. Using default/cmd line value: {model_args.get(k)}")
             # If loading a non-LMA checkpoint into LMA config or vice-versa, need careful handling
             if k == 'use_lma' and model_args.get(k) != checkpoint_model_args.get(k, False): # Default checkpoint LMA to False if missing
                  raise ValueError("Checkpoint/Config mismatch for 'use_lma'. Cannot resume.")

    # Include LMA specific args if resuming an LMA model
    if model_args.get('use_lma', False):
         # Check if reduction factor exists in checkpoint args, otherwise use current config
         model_args['lma_reduction_factor'] = checkpoint_model_args.get('lma_reduction_factor', lma_reduction_factor)

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix): state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # Cannot use LMA with pretrained weights
    if use_lma: raise ValueError("Cannot initialize LMA model from standard GPT-2 weights.")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# --------------------------

# Crop block size if needed (must happen AFTER model init)
if block_size < model.config.block_size:
    try:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # Update configuration
    except NotImplementedError as e:
        print(f"Warning: Could not crop block size - {e}")

model.to(device) # Move model to device

# ---- Optimizer and Scaler ----
# Determine scaler enabled status based on the effective dtype being used
scaler_enabled = (dtype == 'float16') # Enable scaler only if using float16
# The GradScaler API doesn't take device_type directly in newer PyTorch versions
# It infers device from the tensors it scales.
scaler = torch.amp.GradScaler(enabled=scaler_enabled)
print(f"Using GradScaler: {scaler_enabled}")

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
# Separate optimizer for the gate (maybe different LR?)
gate_lr = 1e-5 # Example: potentially smaller LR for gate
optimizer_gate = torch.optim.AdamW(gate_params, lr=gate_lr, betas=(beta1, beta2), weight_decay=weight_decay) # Separate optimizer
if init_from == 'resume' and 'optimizer' in checkpoint: # Check if optimizer state exists
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# --- Training Control ---
train_gate_every_n_steps = 10 # Train gate every 10 main steps (hyperparameter)
gate_training_steps = 1 # Number of gate steps per gate training cycle

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
             model = torch.compile(model) # requires PyTorch 2.0
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
         # Consider exiting or forcing CPU if DDP is critical
         exit(1) # Exit if DDP+MPS requested
    model = DDP(model, device_ids=[ddp_local_rank] if device_type == 'cuda' else None, find_unused_parameters=True)
# --------------------------

# ---- Loss Estimation Function ----
@torch.no_grad()
# Remove ctx from signature
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

# --- Add this Helper Function ---
def set_requires_grad(params, requires_grad):
    """Helper function to set requires_grad for a list of parameters."""
    for p in params:
        p.requires_grad_(requires_grad) # Use requires_grad_ for in-place modification
# --- End Helper Function ---

# ---- LR Scheduler ----
def get_lr(it):
    if not decay_lr: return learning_rate # Return fixed LR if decay is off
    if it < warmup_iters: return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters: return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters); assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)); return min_lr + coeff * (learning_rate - min_lr)
# --------------------

# ---- Logging Setup ----
if wandb_log and master_process:
    import wandb
    # Ensure config passed to wandb includes LMA settings if used
    run_config = config.copy()
    if use_lma: # Check the global flag from config
         # Try to get LMA specifics from model config if available
         if hasattr(model.module if ddp else model, 'config') and hasattr(model.module if ddp else model, 'lma_config_internal'):
              # This assumes GPT stores the initial LMAConfig as 'lma_config_internal'
              # If not, get from the first block's lma_config
              try:
                   first_block = (model.module if ddp else model).transformer.h[0]
                   if hasattr(first_block, 'attn') and hasattr(first_block.attn, 'lma_config'):
                        lma_conf_instance = first_block.attn.lma_config
                        run_config.update({f"lma_{k}": v for k, v in lma_conf_instance.__dict__.items() if not k.startswith('_')})
              except Exception as e:
                   print(f"Warning: Could not retrieve detailed LMA config for wandb: {e}")
         else: # Fallback using global config values
              run_config['use_lma'] = use_lma
              run_config['lma_reduction_factor'] = lma_reduction_factor
              run_config['lma_L_new'] = 'N/A' # Placeholder, actual L_new is dynamic/adjusted
              run_config['lma_d_new'] = 'N/A' # Placeholder
    wandb.init(project=wandb_project, name=wandb_run_name, config=run_config)
# -----------------------

# ---- Training Loop ----
X, Y = get_batch('train') # Fetch first batch
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model # unwrap DDP
running_mfu = -1.0
print("\nStarting training loop...")
while True:
    # --- Determine Phase ---
    is_main_training_step = (iter_num % train_gate_every_n_steps != 0) or (iter_num == 0)
    is_gate_training_step = (iter_num % train_gate_every_n_steps == 0) and (iter_num > 0)

    # --- Set LR (for main optimizer) ---
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups: param_group['lr'] = lr

    # --- Evaluation ---
    if iter_num % eval_interval == 0 and master_process:
        model.eval()
        # --- Add Debug Print ---
        print(f"DEBUG: Type of ctx BEFORE calling estimate_loss: {type(ctx)}")
        print(f"DEBUG: Does ctx have __enter__? {'__enter__' in dir(ctx)}")
        # --- End Debug Print ---
        losses = estimate_loss(model)
        model.train() # Set back to train mode after evaluation
        print_str = f"step {iter_num}: train loss {losses.get('train', float('nan')):.4f}, val loss {losses.get('val', float('nan')):.4f}"
        if hellaswag: print_str += f", HellaSwag Acc: {losses.get('hellaswag', -1):.4f}"
        print(print_str)

        if wandb_log:
            log_data = { "iter": iter_num, "train/loss": losses.get('train', float('nan')), "val/loss": losses.get('val', float('nan')), "lr": lr, "mfu": running_mfu*100 }
            if hellaswag and 'hellaswag' in losses: log_data['val/hellaswag_acc'] = losses['hellaswag']
            wandb.log(log_data)

        current_val_loss = losses.get('val', float('inf')) # Handle case where val loss might be NaN/missing
        if current_val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = current_val_loss if current_val_loss != float('inf') else best_val_loss # Only update if valid
            if iter_num > 0:
                checkpoint = { 'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(), 'model_args': model_args, 'iter_num': iter_num, 'best_val_loss': best_val_loss, 'config': config }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only: break

    # --- Forward/Backward (Main Model or Gate) ---
    if is_main_training_step:
        # === Train Main Model (Gate Frozen) ===
        set_requires_grad(gate_params, False) # Freeze gate
        set_requires_grad(main_params, True)  # Unfreeze main

        for micro_step in range(gradient_accumulation_steps):
            if ddp: model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y) # Standard forward pass
                loss = loss / gradient_accumulation_steps
            X, Y = get_batch('train') # Prefetch next for main OR gate training
            scaler.scale(loss).backward()

        if grad_clip != 0.0: scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(main_params, grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        lossf = loss.item() * gradient_accumulation_steps

    elif is_gate_training_step:
        # === Train Gate Model (Main Frozen) ===
        print(f"\n--- Training Gate Network (Iter {iter_num}) ---")
        set_requires_grad(gate_params, True)   # Unfreeze gate
        set_requires_grad(main_params, False) # Freeze main

        total_gate_loss = 0.0
        # Use the X, Y prefetched previously
        context_sequence = X # (B, T) - Use the full batch context

        # Autoregressive generation step-by-step to get gate loss
        # This is EXPENSIVE - only do a few steps or use a smaller batch?
        # Let's try on the full batch for 'gate_training_steps' iterations
        for _ in range(gate_training_steps): # Loop for gate optimizer steps
             gate_loss_accum = 0.0
             # Process the sequence token by token to get loss ONLY for the gate
             for t_step in range(1, context_sequence.size(1)): # From second token up to end
                  if ddp: model.require_backward_grad_sync = (t_step == context_sequence.size(1) - 1) # Sync on last step? Or every step? Check DDP best practices for this type of loss
                  
                  current_context = context_sequence[:, :t_step] # (B, t_step)
                  target_token = context_sequence[:, t_step]    # (B,) - the actual next token

                  with ctx:
                       # Forward pass for THIS step ONLY
                       # Model needs to handle variable length input correctly
                       # We only care about the prediction for the *next* token
                       logits_step, _ = model(current_context, targets=None) # Get full logits (B, t_step, V)
                       
                       # We need the loss for the token at t_step
                       # Logits for predicting target_token are at sequence position t_step-1
                       logits_for_target = logits_step[:, t_step-1, :] # (B, V)
                       
                       # Calculate cross entropy loss for this step
                       step_loss = F.cross_entropy(logits_for_target, target_token)

                  # Accumulate loss for gate training step
                  # Should we average over sequence length? Or sum? Let's average.
                  gate_loss_accum += step_loss / (context_sequence.size(1) - 1) # Average over sequence steps processed

             # Backward pass for the gate using accumulated step losses
             # This assumes gradients from different steps don't interfere badly? Needs thought.
             # Maybe better: Calculate loss only on the *last* token prediction of the *full sequence*? Simpler.

             # === Revised Gate Loss Calculation (Simpler: Use last token loss) ===
             # No inner loop needed, use full X, Y from get_batch
             if ddp: model.require_backward_grad_sync = True # Always sync for gate step

             with ctx:
                  logits_full, loss_main_ignored = model(X, Y) # Run full forward
                  # We want the gradients ONLY for the gate from this loss
                  # Calculate loss explicitly again IF NEEDED or just use loss_main_ignored?
                  # Using loss_main_ignored might have stale grads if main wasn't trained just before.
                  # Recompute loss for safety, ensuring main params are frozen.
                  gate_objective_loss = F.cross_entropy(logits_full.view(-1, logits_full.size(-1)), Y.view(-1), ignore_index=-1)

             scaler.scale(gate_objective_loss).backward() # Calculate gradients ONLY for gate params

             if grad_clip != 0.0: scaler.unscale_(optimizer_gate); torch.nn.utils.clip_grad_norm_(gate_params, grad_clip)
             scaler.step(optimizer_gate)
             scaler.update()
             optimizer_gate.zero_grad(set_to_none=True)
             total_gate_loss += gate_objective_loss.item()
             print(f"    Gate step loss: {gate_objective_loss.item():.4f}")
             # Get next batch for next potential gate step (or main step)
             X, Y = get_batch('train')


        lossf = total_gate_loss / gate_training_steps # Average gate loss over its steps
        print(f"--- Finished Gate Training (Iter {iter_num}) Avg Loss: {lossf:.4f} ---")
        
    # Forward backward update with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp: model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            # Check for NaN loss immediately
            if torch.isnan(loss):
                 print(f"ERROR: Loss is NaN at iter {iter_num}, micro_step {micro_step}. Stopping.")
                 exit(1) # Stop training if loss becomes NaN
            loss = loss / gradient_accumulation_steps # Scale loss
        # Prefetch next batch
        X, Y = get_batch('train')
        # Backward pass
        scaler.scale(loss).backward() # Use scaler for backward

    # Gradient Clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer) # Unscale before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer Step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Timing and Logging
    t1 = time.time(); dt = t1 - t0; t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps # Approx total loss
        if local_iter_num >= 5: # MFU warmup
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1; local_iter_num += 1
    if iter_num > max_iters: break
# ---------------------

# ---- Cleanup ----
if ddp:
    destroy_process_group()
# ---------------

print("Training finished.")