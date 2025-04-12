# ----- model.py -----
"""
Full definition of a GPT Language Model, all of it in this single file.
Supports standard GPT architecture or a time-step-wise Gated Reduction
before standard Causal Self-Attention blocks.
"""

import math
import requests
import inspect
from dataclasses import dataclass, field
import os
import tqdm, json # For HellaSwag
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from contextlib import nullcontext
from tiktoken.core import Encoding # For HellaSwag

# -----------------------------------------------------------------------------
# Core Model Components
# -----------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    d_reduction_factor: int= 1
    bias: bool = True # True: use bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # --- New Gating Configuration ---
    use_gated_reduction: bool = True # If True, use TimeStepGatedReduction to map n_embd -> gating_d_new
    gating_d_new: int = n_embd/d_reduction_factor # Target dimension after gating. Must be set if use_gated_reduction=True.

class LayerNorm(nn.Module):
    """ LayerNorm with optional bias. """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, input):
        expected_dim = self.weight.shape[0]
        if input.size(-1) != expected_dim:
             raise RuntimeError(f"LayerNorm dim mismatch: Input={input.shape}, Expected Dim={expected_dim}")
        # Calculate normalization based on the last dimension (ndim)
        return F.layer_norm(input, (expected_dim,), self.weight, self.bias, 1e-5)


# --- Time-Step-Wise Gated Reduction Layer ---
class TimeStepGatedReduction(nn.Module):
    def __init__(self, d0: int, d_new: int, bias: bool):
        super().__init__()
        if d0 <= 0 or d_new <= 0:
            raise ValueError(f"Dimensions must be positive (d0={d0}, d_new={d_new})")
        self.d0 = d0
        self.d_new = d_new
        self.gate_proj = nn.Linear(d0, d_new, bias=bias)
        self.value_proj = nn.Linear(d0, d_new, bias=bias)
        self.gate_act = nn.Sigmoid()

    def forward(self, x):
        if x.size(-1) != self.d0:
            raise ValueError(f"Input dim mismatch: Expected {self.d0}, got {x.size(-1)}")
        gate = self.gate_act(self.gate_proj(x))  # (B, L, d_new)
        value = self.value_proj(x)               # (B, L, d_new)
        output = gate * value                    # (B, L, d_new)
        return output

# --- Standard CausalSelfAttention (Modified to accept embed_dim) ---
class CausalSelfAttention(nn.Module):
    """ Standard MHA implementation, now accepting embed_dim """
    def __init__(self, config: GPTConfig, embed_dim: int):
        super().__init__()
        assert embed_dim % config.n_head == 0, f"embed_dim ({embed_dim}) must be divisible by n_head ({config.n_head})"
        self.embed_dim = embed_dim
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.bias = config.bias

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=self.bias)
        # output projection
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # Note: Fixed size mask. Will be sliced if sequence length T < block_size
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            # Use a different name for the mask buffer to avoid conflict with self.bias (the bool flag)
            self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)
        # No need for an else block here regarding buffer registration for flash attention
        else:
             print(f"   - CausalSelfAttention: Using Flash Attention (embed_dim={self.embed_dim})")
             # Ensure the attribute for the mask doesn't exist if not needed, or explicitly set to None
             # if other parts of the code might check for its existence.
             # Setting it to None is safer if the forward pass checks for it.
             self.register_buffer("causal_mask", None, persistent=False) # Register as None if flash is ON


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (embed_dim)
        if C != self.embed_dim:
             raise ValueError(f"CausalSelfAttention C mismatch: Expected {self.embed_dim}, got {C}")

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.embed_dim, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # Note: is_causal=True handles the masking implicitly.
            # dropout_p is applied only during training.
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                               dropout_p=self.dropout if self.training else 0,
                                               is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # Apply causal mask
            if self.causal_mask is None: # Check the correct buffer name
                 # This should theoretically not happen if self.flash is False, but good practice check
                 raise RuntimeError("Slow attention requires causal_mask buffer, but it's None.")
            # Slice the mask if T is smaller than block_size
            slice_T = min(T, self.causal_mask.size(-1)) # Use causal_mask shape
            # Apply the mask using the correct buffer name
            att = att.masked_fill(self.causal_mask[:,:,:slice_T,:slice_T] == 0, float('-inf'))
            # Apply softmax and dropout
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

# --- MLP (Modified to accept embed_dim) ---
class MLP(nn.Module):
    def __init__(self, config: GPTConfig, embed_dim: int):
        super().__init__()
        self.input_dim = embed_dim
        hidden_dim = 4 * self.input_dim
        self.c_fc    = nn.Linear(self.input_dim, hidden_dim, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(hidden_dim, self.input_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        #print(f"   - MLP: Initialized for dim={self.input_dim}")

    def forward(self, x):
        if x.size(-1) != self.input_dim:
            raise ValueError(f"MLP input dim mismatch: Expected {self.input_dim}, got {x.size(-1)}")
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# --- Block (Simplified: Always uses MHA on operating_dim) ---
class Block(nn.Module):
    """ Transformer Block: Always uses CausalSelfAttention """
    def __init__(self, config: GPTConfig, operating_dim: int):
        super().__init__()
        self.operating_dim = operating_dim
        #print(f" Initializing Block {id(self)}: Dim={self.operating_dim}")
        self.ln_1 = LayerNorm(self.operating_dim, bias=config.bias)
        self.attn = CausalSelfAttention(config, self.operating_dim) # Pass operating_dim
        self.ln_2 = LayerNorm(self.operating_dim, bias=config.bias)
        self.mlp = MLP(config, self.operating_dim) # Pass operating_dim

    def forward(self, x):
        # Input x: (B, T_current, self.operating_dim)
        B, T_current, C_current = x.shape
        if C_current != self.operating_dim:
            raise ValueError(f"Block C mismatch: Expected {self.operating_dim}, got {C_current}")

        # Residual connection around Attention
        attn_output = self.attn(self.ln_1(x))
        x = x + attn_output

        # Residual connection around MLP
        mlp_output = self.mlp(self.ln_2(x))
        x = x + mlp_output

        return x

# --- GPT Model ---
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # --- Determine operating dimension within the transformer blocks ---
        self.operates_on_reduced_dim = config.use_gated_reduction
        self.gated_reduction = None
        self.final_ln_lm_head_dim = config.n_embd # Default to input embedding dim

        if self.operates_on_reduced_dim:
            print("--- Configuring Gated Reduction ---")
            if config.gating_d_new is None:
                raise ValueError("gating_d_new must be specified in GPTConfig when use_gated_reduction is True.")
            if config.gating_d_new <= 0:
                 raise ValueError(f"gating_d_new ({config.gating_d_new}) must be positive.")
            if config.gating_d_new >= config.n_embd:
                print(f"Warning: gating_d_new ({config.gating_d_new}) >= n_embd ({config.n_embd}). Gating will not reduce dimension.")
            # Check if d_new is divisible by n_head (good practice for MHA)
            if config.gating_d_new % config.n_head != 0:
                # Option 1: Raise error
                # raise ValueError(f"gating_d_new ({config.gating_d_new}) must be divisible by n_head ({config.n_head}).")
                # Option 2: Adjust d_new (choose one)
                # target_d_new_init = max(config.n_head, (config.gating_d_new // config.n_head) * config.n_head)
                # print(f"Warning: Adjusting gating_d_new {config.gating_d_new} -> {target_d_new_init} to be divisible by n_head ({config.n_head})")
                # config.gating_d_new = target_d_new_init
                # Option 3: Warn but proceed (MHA can handle it, might be less optimal)
                 print(f"Warning: gating_d_new ({config.gating_d_new}) is not divisible by n_head ({config.n_head}). MHA performance might vary.")

            d_new = config.gating_d_new
            self.gated_reduction = TimeStepGatedReduction(config.n_embd, d_new, config.bias)
            self.final_ln_lm_head_dim = d_new
            print(f"--- Dimensions into Blocks: L={config.block_size}, D={self.final_ln_lm_head_dim} (Reduced) ---")
        else:
             print(f"--- Dimensions into Blocks: L={config.block_size}, D={self.final_ln_lm_head_dim} (Standard) ---")

        # --- Transformer Components ---
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # Blocks will be added below
            ln_f = LayerNorm(self.final_ln_lm_head_dim, bias=config.bias), # Final layer norm operates on the output dimension
        ))

        # --- Build Transformer Blocks ---
        print(f"--- Building {config.n_layer} Transformer Blocks (Operating Dim: {self.final_ln_lm_head_dim}) ---")
        blocks = []
        for i in range(config.n_layer):
            # Pass the determined operating dimension to each block
            block = Block(config, operating_dim=self.final_ln_lm_head_dim)
            blocks.append(block)
        self.transformer['h'] = nn.ModuleList(blocks)

        # --- Language Model Head ---
        self.lm_head = nn.Linear(self.final_ln_lm_head_dim, config.vocab_size, bias=False)
        print(f"--- Final LN & LM Head on dim: {self.final_ln_lm_head_dim} ---")

        # --- Weight Tying ---
        # Only tie weights if NOT using gated reduction AND the dimensions match
        # (which they should if not using reduction)
        if not self.operates_on_reduced_dim and self.final_ln_lm_head_dim == config.n_embd:
            self.transformer.wte.weight = self.lm_head.weight
            print("Weight tying enabled (Standard GPT mode).")
        else:
            reason = "Gated Reduction is used" if self.operates_on_reduced_dim else f"Dim mismatch (should not happen in standard mode: final={self.final_ln_lm_head_dim}, n_embd={config.n_embd})"
            print(f"Weight tying disabled ({reason}).")
        # Sanity check print
        # print(f"DEBUG: Are lm_head/wte weights same object? {self.lm_head.weight is self.transformer.wte.weight}")

        # Init all weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to Parameter Sharing they
        count as přístupná already in the final layer, so we don't need to subtract them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
             # Initialize LayerNorm bias to zero if it exists
             if module.bias is not None:
                 torch.nn.init.zeros_(module.bias)
             # Weight is initialized to ones by default in LayerNorm constructor


    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        if t > self.config.block_size:
            idx = idx[:, -self.config.block_size:]
            t = self.config.block_size

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)

        if self.gated_reduction is not None:
            tok_emb_reduced = self.gated_reduction(tok_emb)           # (B, T, d_new)
            pos_emb_reduced = nn.Linear(self.config.n_embd, self.config.gating_d_new, bias=False)(pos_emb)  # (T, d_new)
            x = self.transformer.drop(tok_emb_reduced + pos_emb_reduced)
        else:
            x = self.transformer.drop(tok_emb + pos_emb)
        # Continue with transformer blocks...

        # --- Optional Gated Reduction ---
        if self.gated_reduction is not None:
            x = self.gated_reduction(x)

        # --- Transformer Blocks ---
        for block in self.transformer.h:
            x = block(x)

        # --- Final Layer Norm ---
        x = self.transformer.ln_f(x)

        # --- Logit Calculation (ALWAYS for full sequence) ---
        logits = self.lm_head(x) # Shape: (b, t, vocab_size)

        # --- Loss Calculation (Optional) ---
        loss = None
        if targets is not None:
            # Calculate loss ONLY if targets are provided
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss # Return full logits, loss is None if targets were None

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load a pretrained model with block_size=1024,
        # but want to use a smaller block_size=128 for some smaller downstream task
        assert block_size <= self.config.block_size
        current_max_block_size = self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        # Adjust the mask in CausalSelfAttention if using the slow path
        for block in self.transformer.h:
             # Only adjust if the attention layer has the 'bias' buffer (i.e., slow attention)
             if hasattr(block.attn, 'bias') and block.attn.bias is not None:
                 # Check if the buffer exists and has the expected shape before slicing
                 if block.attn.bias.shape[-1] == current_max_block_size:
                      block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
                 else:
                      print(f"Warning: MHA bias buffer shape {block.attn.bias.shape} does not match expected old block size {current_max_block_size}. Skipping resize.")


    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # This method is kept for compatibility but needs careful consideration
        # if loading a model that *used* gated reduction vs one that didn't.
        # Currently, it loads standard GPT-2 weights. Enabling gated reduction
        # would likely require retraining or specific fine-tuning.
        print(f"Loading weights from pretrained gpt: {model_type}")
        if override_args is None: override_args = {}
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)

        from transformers import GPT2LMHeadModel
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints

        # handle potential overrides AFTER setting defaults
        if 'dropout' in override_args: config_args['dropout'] = override_args['dropout']

        # --- Crucial: Decide on Gated Reduction for loaded model ---
        # By default, standard GPT-2 models do NOT use gated reduction.
        # If you want to load GPT-2 weights AND use gated reduction,
        # you'd likely need to retrain or fine-tune significantly.
        config_args['use_gated_reduction'] = override_args.get('use_gated_reduction', False)
        config_args['gating_d_new'] = override_args.get('gating_d_new', None)

        # create a from-scratch initialized minGPT model
        print("Creating model with config:", config_args)
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, possibly created by slow attention

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all parameters are aligned and match in shape and name
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these buffers
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the name used in older versions of transformers
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        warn_skip_gated = False
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy
                if sd_hf[k].shape != sd[k].shape:
                    # This WILL happen if use_gated_reduction=True, as the shapes won't match
                    # for layers operating on d_new (e.g., LayerNorms, MLP projections after first, MHA projections).
                    # Also, the lm_head and final ln_f will mismatch.
                    if config.use_gated_reduction:
                         if not warn_skip_gated: # Print warning only once
                              print(f"Warning: Skipping parameter copy for keys due to shape mismatch caused by use_gated_reduction=True. These layers will remain randomly initialized: {k} (and others like it)")
                              warn_skip_gated = True
                         continue # Skip copying this parameter
                    else:
                        # If not using gated reduction, shapes should match. Raise error.
                         raise ValueError(f"Shape mismatch for key {k}: HF={sd_hf[k].shape}, Model={sd[k].shape}. Ensure config matches pretrained model.")
                else:
                     with torch.no_grad():
                          sd[k].copy_(sd_hf[k])

        # Handle the gated reduction layer weights - they won't exist in sd_hf
        if config.use_gated_reduction and model.gated_reduction is not None:
            print("Warning: TimeStepGatedReduction layer weights are randomly initialized as they don't exist in standard GPT-2 checkpoints.")

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type.startswith('cuda')
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # Note: This estimate is for standard GPT-2 architecture.
        # The MFU for the gated reduction variant might differ slightly,
        # but the dominant cost is usually the Attention and MLP layers.
        N = self.get_num_params()
        cfg = self.config
        L, H = cfg.n_layer, cfg.n_head
        # Determine the dimension used in attention/MLP calculations
        C = cfg.gating_d_new if cfg.use_gated_reduction else cfg.n_embd
        Q = C // H if H > 0 else 0 # Head dimension
        T = cfg.block_size

        # flops calculation based on standard transformer estimates
        # Attention: 4*B*T*C*C (QK^T + AV) - approx 2*B*T*C*(2*T*H*Q) = 4*B*T*C^2/H * T (check this) -> Karpathy uses 2*B*T*C* (2*C) for QKV + attn*V? Let's use standard estimate.
        # MHA Flops: Roughly 2 * B * T * C * (2 * C) for QKV/Proj + 2 * B * T^2 * C for Attn Scores/Output = 4*B*T*C^2 + 2*B*T^2*C
        # MLP Flops: Roughly 2 * B * T * C * (4 * C) + 2 * B * T * (4 * C) * C = 16 * B * T * C^2
        # Total Flops per layer: Approx 4*B*T*C^2 + 2*B*T^2*C + 16*B*T*C^2 = 20*B*T*C^2 + 2*B*T^2*C
        # Karpathy's estimate: 6*N + 12*L*H*Q*T simplifies things. Let's use that.
        # 6*N accounts for matmuls in MLP/Projections. N = L*(4*C^2 + 4*C^2 + C^2 + C^2) + Embeddings ~ L*10*C^2
        # 12*L*H*Q*T = 12*L*C*T accounts for attention computation.
        # Let's stick to the simpler 6*N + 12*L*C*T estimate where C is the operating dim.
        flops_per_token = 6*N + 12*L*C*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter # number of microsteps per iteration

        # expressed using A100 peak FLOPS as baseline: 312 TFLOPS = 312e12
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops
        mfu = flops_achieved / flops_promised
        # Print a warning if gated reduction is used, as estimate might be less accurate
        if cfg.use_gated_reduction:
            print("Warning: MFU estimate based on standard GPT architecture; may be less accurate for Gated Reduction variant.")
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        self.eval() # Ensure model is in evaluation mode
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond) # We only need logits, ignore loss
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf') # Apply top-k filtering
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        self.train() # Set model back to training mode if it was before
        return idx

def get_most_likely_row(tokens, mask, logits):
    """
    Given tokens, mask, and logits for multiple choice options,
    find the index of the row (choice) with the lowest average cross-entropy loss
    calculated ONLY over the completion part (where mask == 1).
    """
    # Check if sequence length is sufficient for shifting
    if logits.shape[1] <= 1:
        # This warning indicates an issue upstream in how tokens/logits were prepared
        print(f"Warning (get_most_likely_row): Logits seq len <= 1 ({logits.shape}). Cannot calculate loss.")
        return 0 # Return a default index, as loss cannot be computed

    # Shift logits to align with target tokens (predict next token)
    # Logits shape: (B, T, V) -> (B, T-1, V)
    shift_logits = logits[..., :-1, :].contiguous()
    # Shift tokens to align with prediction targets
    # Tokens shape: (B, T) -> (B, T-1)
    shift_tokens = tokens[..., 1:].contiguous()
    # Shift mask to align with target tokens
    # Mask shape: (B, T) -> (B, T-1)
    shift_mask = mask[..., 1:].contiguous()

    # Calculate per-token loss without reduction
    # Flatten shapes for cross_entropy: (B * (T-1), V) and (B * (T-1),)
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)

    # Ensure dimensions match before loss calculation (robustness check)
    if flat_shift_logits.shape[0] != flat_shift_tokens.shape[0]:
         print(f"ERROR (get_most_likely_row): Size mismatch before loss! Logits: {flat_shift_logits.shape}, Tokens: {flat_shift_tokens.shape}")
         return 0

    try:
        # Calculate loss per token position
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        # Reshape back to (B, T-1)
        shift_losses = shift_losses.view(tokens.size(0), -1)
    except Exception as e:
        print(f"ERROR during cross_entropy in get_most_likely_row: {e}")
        print(f"  Logits shape: {flat_shift_logits.shape}, Tokens shape: {flat_shift_tokens.shape}")
        return 0 # Return default index on error

    # Apply the mask to zero out losses for non-completion tokens (context or padding)
    masked_shift_losses = shift_losses * shift_mask

    # Sum the loss for each row (completion candidate)
    sum_loss = masked_shift_losses.sum(dim=1)

    # Count the number of completion tokens in each row (where mask was 1)
    num_loss_tokens = shift_mask.sum(dim=1)

    # Calculate average loss per row. Add epsilon for numerical stability.
    # Handle cases where a row might have zero completion tokens (division by zero)
    avg_loss = sum_loss / (num_loss_tokens + 1e-9) # Use a slightly larger epsilon
    avg_loss[num_loss_tokens == 0] = float('inf') # Assign infinite loss if no completion tokens

    # Find the index of the row with the minimum average loss
    # If all avg_loss are inf (e.g., all rows had 0 completion tokens), argmin might return 0
    if torch.all(torch.isinf(avg_loss)):
        print("Warning (get_most_likely_row): All rows have zero completion tokens or infinite loss.")
        return 0 # Return default index

    pred_norm = avg_loss.argmin().item()
    return pred_norm


@torch.no_grad()
def evaluate_hellaswag(model, enc: Encoding, hellaswag_path='data/hellaswag/hellaswag_val.jsonl'):
    """ Evaluates the model performance on the HellaSwag dataset. """
    assert isinstance(enc, Encoding), "Encoder `enc` must be tiktoken Encoding object"
    print(f"Evaluating HellaSwag from {hellaswag_path}...")
    num_correct_norm = 0
    num_total = 0

    # --- Attempt to download HellaSwag validation set if not found ---
    if not os.path.exists(hellaswag_path):
        print(f"Error: HellaSwag validation file not found at {hellaswag_path}")
        data_dir = os.path.dirname(hellaswag_path) or '.'
        if not os.path.exists(data_dir):
            try:
                os.makedirs(data_dir)
            except OSError as e:
                print(f"Error creating directory {data_dir}: {e}")
                return -1.0

        val_url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
        print(f"Attempting download from {val_url}...")
        try:
            with requests.get(val_url, stream=True) as r:
                r.raise_for_status()
                with open(hellaswag_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download successful.")
        except requests.exceptions.RequestException as e:
            print(f"Download failed: {e}. Cannot evaluate HellaSwag.")
            if os.path.exists(hellaswag_path):
                os.remove(hellaswag_path)
            return -1.0
        except Exception as e:
            print(f"An unexpected error occurred during download: {e}")
            if os.path.exists(hellaswag_path):
                os.remove(hellaswag_path)
            return -1.0

    # --- Determine device ---
    model_device = next(model.parameters()).device
    device_type = 'cuda' if 'cuda' in str(model_device) else 'cpu'
    print(f"DEBUG: Device for HellaSwag evaluation: {device_type}")

    # --- Ensure model is in float32 for evaluation ---
    model.eval()  # Set model to evaluation mode
    original_dtypes = {}
    if device_type == 'cuda':
        print("DEBUG: Converting model to float32 for HellaSwag evaluation.")
        # Store original dtypes and convert parameters/buffers to float32
        for name, param in model.named_parameters():
            original_dtypes[name] = param.dtype
            param.data = param.data.to(dtype=torch.float32)
        for name, buf in model.named_buffers():
            original_dtypes[f"buffer_{name}"] = buf.dtype
            buf.data = buf.data.to(dtype=torch.float32)

    # --- Evaluation context (no autocast needed since model is in float32) ---
    eval_ctx = nullcontext()  # Use nullcontext since model is explicitly float32

    processed_lines = 0
    try:
        with open(hellaswag_path, 'r', encoding='utf-8') as f:
            for line in tqdm.tqdm(f, desc="HellaSwag Eval"):
                processed_lines += 1
                try:
                    example = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line: {line.strip()}")
                    continue

                num_total += 1
                ctx = example['ctx']
                label = example['label']
                endings = example['endings']

                # --- Encode context and each completion ---
                try:
                    ctx_tokens = enc.encode(ctx)
                except Exception as e:
                    print(f"Warning: Skipping example due to encoding error in context: {e}")
                    num_total -= 1
                    continue

                tok_rows = []
                mask_rows = []

                for end_idx, end in enumerate(endings):
                    try:
                        completion_tokens = enc.encode(" " + end)
                    except Exception as e:
                        print(f"Warning: Encoding error in completion index {end_idx}, skipping this ending: {e}")
                        completion_tokens = []

                    tok = ctx_tokens + completion_tokens
                    mask = ([0]*len(ctx_tokens)) + ([1]*len(completion_tokens))

                    # --- Truncation Logic ---
                    if len(tok) > model.config.block_size:
                        num_comp = len(completion_tokens)
                        max_ctx = model.config.block_size - num_comp
                        if max_ctx < 0:
                            completion_tokens = completion_tokens[:model.config.block_size]
                            tok = completion_tokens
                            mask = [1] * len(tok)
                        else:
                            start_idx = max(0, len(ctx_tokens) - max_ctx)
                            trunc_ctx = ctx_tokens[start_idx:]
                            tok = trunc_ctx + completion_tokens
                            mask = ([0]*len(trunc_ctx)) + ([1]*len(completion_tokens))
                        if len(tok) > model.config.block_size:
                            tok = tok[-model.config.block_size:]
                            mask = mask[-model.config.block_size:]

                    # --- Minimum Length Padding ---
                    if len(tok) < 2:
                        pad_len = 2 - len(tok)
                        pad_token_id = getattr(enc, 'pad_token_id', getattr(enc, 'eot_token', 0))
                        tok = tok + ([pad_token_id] * pad_len)
                        mask = mask + ([0] * pad_len)

                    tok_rows.append(torch.tensor(tok, dtype=torch.long))
                    mask_rows.append(torch.tensor(mask, dtype=torch.long))

                if not tok_rows or len(tok_rows) != 4:
                    print(f"Warning: Skipping example due to insufficient valid endings ({len(tok_rows)}/4). Context: {ctx[:50]}...")
                    num_total -= 1
                    continue

                # --- Batching and Padding ---
                try:
                    max_len = max(len(r) for r in tok_rows)
                except ValueError:
                    print(f"Warning: Skipping example due to empty tok_rows after processing endings. Context: {ctx[:50]}...")
                    num_total -= 1
                    continue

                max_len = max(2, max_len)
                pad_token_id = getattr(enc, 'pad_token_id', getattr(enc, 'eot_token', 0))
                tokens = torch.full((len(tok_rows), max_len), pad_token_id, dtype=torch.long)
                mask_t = torch.zeros((len(tok_rows), max_len), dtype=torch.long)

                for i, (tr, mr) in enumerate(zip(tok_rows, mask_rows)):
                    seq_len = len(tr)
                    tokens[i, :seq_len] = tr
                    mask_t[i, :seq_len] = mr

                tokens = tokens.to(model_device)
                mask_t = mask_t.to(model_device)

                # --- Model Inference ---
                with eval_ctx:
                    logits, _ = model(tokens)

                # --- NaN/Inf Check ---
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"ERROR: NaNs or Infs detected in HellaSwag logits! Context: {ctx[:50]}...")
                    pred_norm = -1
                else:
                    try:
                        pred_norm = get_most_likely_row(tokens, mask_t, logits)
                    except Exception as e:
                        print(f"ERROR occurred inside get_most_likely_row: {e}")
                        import traceback
                        traceback.print_exc()
                        pred_norm = -1

                if pred_norm == label:
                    num_correct_norm += 1

    except FileNotFoundError:
        print(f"Error: HellaSwag validation file not found at {hellaswag_path} even after download attempt.")
        return -1.0
    except Exception as e:
        print(f"\nAn unexpected error occurred during the HellaSwag evaluation loop: {e}")
        import traceback
        traceback.print_exc()
        return -1.0
    finally:
        # --- Restore original dtypes ---
        if device_type == 'cuda':
            print("DEBUG: Restoring original parameter dtypes after HellaSwag evaluation.")
            for name, param in model.named_parameters():
                if name in original_dtypes:
                    param.data = param.data.to(dtype=original_dtypes[name])
            for name, buf in model.named_buffers():
                buf_name = f"buffer_{name}"
                if buf_name in original_dtypes:
                    buf.data = buf.data.to(dtype=original_dtypes[buf_name])

    if num_total == 0:
        print("Warning: No examples were processed during HellaSwag evaluation.")
        return 0.0
    acc_norm = float(num_correct_norm) / num_total
    print(f"HellaSwag Accuracy: {acc_norm*100:.2f}% ({num_correct_norm}/{num_total})")
    return acc_norm
# -----------------------------------------------------------------------------



# --- Example Usage / Testing Block ---
if __name__ == '__main__':

    # --- Configuration Options ---

    # Option 1: Small Model with Gated Reduction
    use_gated = True
    d_new = 64 # Target dimension after gating (must be <= n_embd)
    config_args = dict(
        block_size=128, vocab_size=50257, n_layer=4, n_head=4, n_embd=128,
        dropout=0.1, bias=True,
        use_gated_reduction=use_gated,
        d_reduction_factor=3 if use_gated else 1, # Set d_new only if using gating
    )

    # Option 2: Small Model Standard GPT (No Gating)
    # config_args = dict(
    #     block_size=128, vocab_size=50257, n_layer=4, n_head=4, n_embd=128,
    #     dropout=0.1, bias=True,
    #     use_gated_reduction=False,
    #     gating_d_new=None,
    # )

    # Option 3: GPT-2 Base size with Gated Reduction (Example)
    # use_gated = True
    # d_new = 384 # Example: Reduce 768 -> 384 (Divisible by n_head=12)
    # config_args = dict(
    #     block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768,
    #     dropout=0.0, bias=True, # GPT-2 defaults
    #     use_gated_reduction=use_gated,
    #     gating_d_new=d_new if use_gated else None,
    # )

    # Create config object
    gpt_config = GPTConfig(**config_args)
    print("\n--- Model Configuration ---")
    print(gpt_config)

    # --- Initialize Model ---
    print("\n--- Initializing Model ---")
    model = GPT(gpt_config)

    # --- Testing ---
    print("\n--- Testing Forward/Backward Pass ---")
    B = 4
    T = gpt_config.block_size
    T_short = T // 2 if T > 1 else 1 # Ensure T_short is at least 1

    dummy_input_full = torch.randint(0, gpt_config.vocab_size, (B, T))
    dummy_targets_full = torch.randint(0, gpt_config.vocab_size, (B, T))
    # Adjust targets to avoid -1 index if ignore_index is used (though not strictly necessary here)
    dummy_targets_full[dummy_targets_full == -1] = 0

    dummy_input_short = torch.randint(0, gpt_config.vocab_size, (B, T_short))
    dummy_targets_short = torch.randint(0, gpt_config.vocab_size, (B, T_short))
    dummy_targets_short[dummy_targets_short == -1] = 0

    # Determine device
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Check if running in distributed environment, default to CPU if so
        device = 'mps' if "RANK" not in os.environ else 'cpu'
    else:
        device = 'cpu'

    print(f"Using device: {device}")
    model.to(device)

    # Create optimizer
    optimizer = model.configure_optimizers(weight_decay=1e-1, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device)

    # Test with different sequence lengths
    for seq_len_label, dummy_input, dummy_targets in [
        (f"T = {T}", dummy_input_full, dummy_targets_full),
        (f"T = {T_short}", dummy_input_short, dummy_targets_short)
    ]:
        print(f"\nTesting with {seq_len_label}...")
        dummy_input = dummy_input.to(device)
        dummy_targets = dummy_targets.to(device)

        try:
            model.train() # Set to train mode for dropout, etc.
            optimizer.zero_grad(set_to_none=True) # More efficient zeroing

            # Autocast for mixed precision if on CUDA
            ctx = nullcontext()
            if device == 'cuda':
                 ctx = torch.amp.autocast(device_type=device, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

            with ctx:
                logits, loss = model(dummy_input, dummy_targets)

            print("Forward pass successful!")
            print(f"  Input shape:  {dummy_input.shape}")
            print(f"  Logits shape: {logits.shape}") # Should be (B, T, VocabSize)

            if loss is not None:
                print(f"  Loss: {loss.item():.4f}")
                # Scaler for mixed precision
                if device == 'cuda' and isinstance(ctx, torch.amp.autocast):
                     # Basic example without gradient scaler - usually needed for stability
                     # For proper training, use torch.cuda.amp.GradScaler
                     loss.backward()
                     print("Backward pass attempted (without scaler).")
                else:
                     loss.backward()
                     print("Backward pass successful!")

                # Check gradients (optional)
                # grad_norm = 0.0
                # for p in model.parameters():
                #     if p.grad is not None:
                #         grad_norm += p.grad.detach().data.norm(2).item() ** 2
                # grad_norm = grad_norm ** 0.5
                # print(f"  Gradient norm: {grad_norm:.4f}")

                optimizer.step() # Update weights
                print("Optimizer step successful!")
            else:
                print("  Loss is None (likely inference mode in forward pass).")

        except Exception as e:
            print(f"\n !!! Error during Forward/Backward ({seq_len_label}) !!!")
            print(e)
            import traceback
            traceback.print_exc()
            # Break the loop on error? Or continue? Continue for now.

    # --- Test Generation ---
    print("\n--- Testing Generation ---")
    try:
        model.eval() # Set to evaluation mode
        start_ids = torch.randint(0, gpt_config.vocab_size, (1, 10), device=device) # Example start sequence
        print(f"  Generating from start sequence shape: {start_ids.shape}")
        generated_ids = model.generate(start_ids, max_new_tokens=20, temperature=0.8, top_k=5)
        print("Generation successful!")
        print(f"  Generated sequence shape: {generated_ids.shape}") # Should be (1, 10 + 20)
    except Exception as e:
        print("\n !!! Error during Generation !!!")
        print(e)
        import traceback
        traceback.print_exc()

    # --- Test HellaSwag Evaluation ---
    print("\n--- Testing HellaSwag ---")
    try:
        # Attempt to import tiktoken
        import tiktoken
        enc = tiktoken.get_encoding("gpt2") # Get the tokenizer
        print("Tiktoken loaded successfully.")

        # Define path to HellaSwag validation file
        # Assumes 'data/hellaswag/' directory structure relative to script location
        script_dir = os.path.dirname(__file__) if "__file__" in locals() else '.'
        hs_path = os.path.join(script_dir, 'data', 'hellaswag', 'hellaswag_val.jsonl')

        # Run evaluation
        accuracy = evaluate_hellaswag(model, enc, hs_path)
        print(f"HellaSwag evaluation finished. Accuracy: {accuracy:.4f}")

    except ImportError:
        print("tiktoken not installed, skipping HellaSwag evaluation.")
        print("Install tiktoken: pip install tiktoken")
    except FileNotFoundError:
         print(f"HellaSwag file not found at expected location ({hs_path}), skipping.")
         print("Ensure the file exists or the download works.")
    except Exception as e:
        print(f"\n !!! Error during HellaSwag Evaluation !!!")
        print(e)
        import traceback
        traceback.print_exc()

    print("\n--- Model Testing Complete ---")