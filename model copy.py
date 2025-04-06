 -----------------------------------------------------------------------------
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