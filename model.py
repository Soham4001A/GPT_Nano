# ----- model.py -----
"""
Full definition of a GPT Language Model, all of it in this single file.
Incorporates Latent Meta Attention (LMA) as an alternative attention mechanism.
"""

import math
import inspect
from dataclasses import dataclass, field # Import field

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# -----------------------------------------------------------------------------
# Helper Functions for LMA
# -----------------------------------------------------------------------------

def find_closest_divisor(total_value, target_divisor, max_delta=20):
    """
    Finds a divisor of total_value that is closest to target_divisor (PyTorch/NumPy version).
    """
    if not isinstance(total_value, int) or total_value <= 0:
        raise ValueError(f"total_value ({total_value}) must be a positive integer.")
    if not isinstance(target_divisor, int) or target_divisor <= 0:
        raise ValueError(f"target_divisor ({target_divisor}) must be positive.")
    if not isinstance(max_delta, int) or max_delta < 0:
        raise ValueError(f"max_delta ({max_delta}) must be non-negative.")

    if total_value == 0: # Handle edge case
        return 1 # Or raise error? Returning 1 might be safer downstream.

    # Check target directly only if > 0
    if target_divisor > 0 and total_value % target_divisor == 0:
        return target_divisor

    # Ensure target_divisor is at least 1 for search start if it was 0 or less initially
    search_start = max(1, target_divisor)

    for delta in range(1, max_delta + 1):
        candidate_minus = search_start - delta
        if candidate_minus > 0 and total_value % candidate_minus == 0:
            return candidate_minus
        candidate_plus = search_start + delta
        # Check divisibility only for positive candidates (though total_value is > 0)
        if candidate_plus > 0 and total_value % candidate_plus == 0:
            return candidate_plus

    # Fallback if no divisor found nearby: find *any* divisor, prioritizing smaller ones?
    # Or just raise error as before. Raising error is safer.
    raise ValueError(
        f"Could not find a valid divisor for {total_value} near {target_divisor} "
        f"(within +/- {max_delta}). Check L, d0, or target L_new."
    )


def get_lma_causal_mask(L, n_h, L_new, device):
    """
    Calculates the causal mask for the LMA latent attention space (PyTorch version).
    Mask=True means attention is prevented.
    """
    if L <= 0 or n_h <= 0 or L_new <= 0:
        print(f"Warning: Invalid dims L={L}, n_h={n_h}, L_new={L_new} for mask. Returning None.")
        return None

    L_prime = L * n_h
    if L_prime == 0 : return None # Avoid division by zero

    # Assume reduction is by striding/pooling. k_stride is items in L' per item in L_new
    # Handle potential non-divisibility cleanly
    if L_prime < L_new:
        print(f"Warning: L*nh ({L_prime}) < L_new ({L_new}). Mask generation might be problematic. Setting stride to 1.")
        k_stride = 1 # Each L_new step covers at least one L' step? This needs careful thought based on R_seq
    elif L_new == 0: # Avoid division by zero
        print(f"Warning: L_new is zero. Cannot calculate stride. Returning None.")
        return None
    else:
        k_stride = L_prime // L_new
        if L_prime % L_new != 0 :
            print(f"Warning: L*nh ({L_prime}) not perfectly divisible by L_new ({L_new}). Mask assumes integer stride {k_stride}.")

    max_orig_index_per_latent_pos = [-1] * L_new
    for i_new in range(L_new):
        # Calculate range based on stride, ensuring bounds are respected
        p_start = i_new * k_stride
        # Ensure p_end doesn't exceed L_prime, using exclusive end for range
        p_end = min((i_new + 1) * k_stride, L_prime)

        max_orig_l = -1
        if p_start < p_end: # Check if the range is valid
            for p in range(p_start, p_end):
                max_orig_l = max(max_orig_l, p % L)
        # Handle potential case where p_start might be valid but range is empty due to stride/rounding
        elif p_start < L_prime:
             max_orig_l = p_start % L

        max_orig_index_per_latent_pos[i_new] = max_orig_l

    # Create boolean mask (True means mask out)
    mask = torch.zeros((L_new, L_new), device=device, dtype=torch.bool)
    for i_new in range(L_new):
        query_max_orig = max_orig_index_per_latent_pos[i_new]
        if query_max_orig == -1: continue # Skip if latent pos has no origin (Error in logic if this happens)
        for j_new in range(L_new):
            key_max_orig = max_orig_index_per_latent_pos[j_new]
            if key_max_orig == -1: continue
            if key_max_orig > query_max_orig:
                mask[i_new, j_new] = True

    return mask

# -----------------------------------------------------------------------------
# Core Model Components
# -----------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """ LayerNorm with optional bias. """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        if input.size(-1) != self.weight.shape[0]:
             layer_name = [name for name, mod in self.named_modules() if mod is self]
             parent_name = [name for name, mod in self.named_modules() if self in mod.children()]
             print(f"ERROR in LayerNorm ({parent_name}/{layer_name}): Input dim {input.size(-1)} != Weight dim {self.weight.shape[0]}")
             raise RuntimeError(f"LayerNorm input dim ({input.size(-1)}) mismatch with weight dim ({self.weight.shape[0]})")
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

@dataclass
class LMAConfig:
    """ Configuration specific to the LMA layer internals. """
    d0: int
    L: int
    n_head_stacking: int
    target_L_new: int
    d_new: int
    n_head_latent: int
    # Removed seq_reduction_type from __init__ signature
    L_new: int = field(init=False)
    C_new: int = field(init=False)

    def __post_init__(self):
        if self.L <= 0 or self.d0 <= 0:
            print(f"Warning: LMAConfig received non-positive L={self.L} or d0={self.d0}. Setting L_new/C_new to 1.")
            self.L_new = 1; self.C_new = 1; return
        total_features = self.L * self.d0
        if total_features == 0:
            print(f"Warning: LMAConfig total features (L*d0) is zero. Setting L_new/C_new to 1.")
            self.L_new = 1; self.C_new = 1; return
        try:
            # Ensure target_L_new is positive before passing to find_closest_divisor
            if self.target_L_new <= 0: raise ValueError("target_L_new must be positive.")
            self.L_new = find_closest_divisor(total_features, self.target_L_new)
            if self.L_new != self.target_L_new: print(f"LMAConfig ADJUSTMENT: Target L_new ({self.target_L_new}) changed to {self.L_new} to divide total features ({total_features}).")
            self.C_new = total_features // self.L_new
        except ValueError as e: raise ValueError(f"LMA Config Error calculating L_new/C_new: {e}") from e
        if self.d_new <= 0: raise ValueError(f"LMA Config Error: d_new ({self.d_new}) must be positive.")
        if self.n_head_latent <= 0: raise ValueError(f"LMA Config Error: n_head_latent ({self.n_head_latent}) must be positive.")
        # Check divisibility *after* d_new is confirmed positive
        if self.d_new % self.n_head_latent != 0: raise ValueError(f"LMA Config Error: d_new ({self.d_new}) must be divisible by n_head_latent ({self.n_head_latent}).")


class LatentMetaAttention(nn.Module):
    """ LMA Core Logic - Includes projection back to L if L_new != L """
    def __init__(self, config, lma_config: LMAConfig):
        super().__init__()
        # Store configs
        self.config = config
        self.lma_config = lma_config

        # --- Validate and Set Dimensions ---
        assert lma_config.d0 % lma_config.n_head_stacking == 0, f"d0 ({lma_config.d0}) must be divisible by n_head_stacking ({lma_config.n_head_stacking})"
        assert lma_config.d_new % lma_config.n_head_latent == 0, f"d_new ({lma_config.d_new}) must be divisible by n_head_latent ({lma_config.n_head_latent})"
        self.d0 = lma_config.d0
        self.L = lma_config.L
        self.n_head_stacking = lma_config.n_head_stacking
        self.d_k = self.d0 // self.n_head_stacking
        self.d_new = lma_config.d_new
        self.n_head_latent = lma_config.n_head_latent
        self.bias = config.bias
        self.dropout = config.dropout
        # Use the adjusted L_new and calculated C_new from lma_config instance
        self.L_new = lma_config.L_new
        self.C_new = lma_config.C_new

        print(f"  Initializing LMA Layer: L={self.L}, d0={self.d0}, n_h_stack={self.n_head_stacking} -> L_new={self.L_new}, d_new={self.d_new}, C_new={self.C_new}, n_h_latent={self.n_head_latent}")

        # --- Layers ---
        # Stage 2b: Latent Embedding (maps C_new -> d_new)
        self.embed_layer_2 = nn.Linear(self.C_new, self.d_new, bias=self.bias)
        self.embed_layer_2_act = nn.ReLU() # Activation after second embed

        # Latent Attention Layers (QKV projections + MHA)
        self.q_proj = nn.Linear(self.d_new, self.d_new, bias=self.bias)
        self.k_proj = nn.Linear(self.d_new, self.d_new, bias=self.bias)
        self.v_proj = nn.Linear(self.d_new, self.d_new, bias=self.bias)
        self.latent_attn = nn.MultiheadAttention(
            embed_dim=self.d_new,
            num_heads=self.n_head_latent,
            dropout=self.dropout,
            bias=self.bias,
            batch_first=True # IMPORTANT: Assume input tensors are (Batch, Seq, Dim)
        )

        # Output projection (after latent attention)
        self.c_proj = nn.Linear(self.d_new, self.d_new, bias=self.bias) # Projects latent MHA output

        # Regularization
        self.resid_dropout = nn.Dropout(self.dropout)

        # --- Causal Mask ---
        # Pre-compute the custom LMA causal mask on CPU initially
        try:
            lma_mask = get_lma_causal_mask(self.L, self.n_head_stacking, self.L_new, device='cpu')
            self.register_buffer("causal_mask_latent", lma_mask, persistent=False) # Don't save in state_dict if recomputed
            if lma_mask is not None:
                print(f"  LMA Layer: Registered latent causal mask ({self.L_new}x{self.L_new})")
        except Exception as e:
            print(f"ERROR generating LMA mask during init: {e}. Mask set to None.")
            self.register_buffer("causal_mask_latent", None, persistent=False)

        # --- Upscaling Projection Layer ---
        # Needed if L_new != L to restore sequence length for compatibility within Block
        # If LMA is the *last* layer before lm_head, this might not be needed,
        # but making Block interface consistent requires outputting (B, L, d0 or d_new)
        self.needs_upscaling = (self.L_new != self.L) # Compare actual L_new used with block L
        if self.needs_upscaling:
             total_features_latent = self.L_new * self.d_new
             # Output should be (B, L, d_new) - project features then reshape
             self.upscale_proj = nn.Linear(total_features_latent, self.L * self.d_new, bias=config.bias)
             print(f"  LMA Layer: Adding upscale projection ({total_features_latent} -> {self.L * self.d_new})")
        else:
             self.upscale_proj = None

    def forward(self, y):
        # Input y: (B, L, d0) - Check shape against instance attributes
        B, T, C = y.size()
        if T != self.L or C != self.d0:
             # Allow for shorter sequences during generation (T < self.L)
             if T > self.L:
                 raise ValueError(f"LMA forward input T ({T}) > configured L ({self.L})")
             # If T < L, the mask logic needs adjustment. For now, assume T == L.
             # TODO: Handle T < L for generation if needed (mask slicing)
             # For training, T should usually equal L (block_size)
             assert T == self.L, f"LMA forward input T ({T}) != configured L ({self.L})"
             assert C == self.d0, f"LMA forward input C ({C}) != configured d0 ({self.d0})"


        # --- Stage 2a: Head-View Stacking ---
        head_views = torch.split(y, self.d_k, dim=2)
        x_stacked = torch.cat(head_views, dim=1) # (B, L * n_h_stacking, d_k)

        # --- Stage 2b: Re-Chunking & Latent Embedding ---
        x_flat = x_stacked.view(B, -1) # (B, L * d0)
        x_rechunked = x_flat.view(B, self.L_new, self.C_new) # (B, L_new, C_new)

        # Apply embed_layer_2 using reshape-apply-reshape for TimeDistributed effect
        z_embedded_flat = self.embed_layer_2(x_rechunked.view(-1, self.C_new)) # Input shape (B*L_new, C_new)
        z = self.embed_layer_2_act(z_embedded_flat) # Apply activation
        z = z.view(B, self.L_new, self.d_new) # Reshape back to (B, L_new, d_new) - Pre-Attention state

        # --- Stage 3: Latent Attention ---
        q_prime = self.q_proj(z)
        k_prime = self.k_proj(z)
        v_prime = self.v_proj(z)

        # --- Mask Handling ---
        attn_mask_to_use = self.causal_mask_latent
        if attn_mask_to_use is None:
            raise RuntimeError(f"LMA Causal Mask is None for L={self.L}, L_new={self.L_new}.")

        current_T_latent = q_prime.size(1) # Should be L_new
        if attn_mask_to_use.size(0) != current_T_latent:
            # This error check might be too strict if T < L was handled upstream
            # Let's assume mask is precomputed for L_new based on L = block_size
            # We should only use the [:current_T_latent, :current_T_latent] slice?
             print(f"Warning: LMA Mask size {attn_mask_to_use.size(0)} != latent seq len {current_T_latent}. Slicing mask.")
             # Slice mask dynamically if needed (unlikely during training if T==block_size)
             attn_mask_to_use = attn_mask_to_use[:current_T_latent, :current_T_latent]
            # raise RuntimeError(f"LMA Mask size mismatch: {attn_mask_to_use.size(0)} vs {current_T_latent}")

        # Ensure mask is on correct device
        attn_mask_to_use = attn_mask_to_use.to(q_prime.device)

        # Apply latent attention
        attn_output, _ = self.latent_attn(q_prime, k_prime, v_prime,
                                          attn_mask=attn_mask_to_use, # Use the potentially sliced mask
                                          need_weights=False,
                                          is_causal=False) # Mask is explicit
        attn_output_proj = self.c_proj(attn_output) # (B, L_new, d_new)
        attn_output_drop = self.resid_dropout(attn_output_proj) # Apply dropout *after* projection

        # --- Apply Upscaling Projection if needed ---
        if self.needs_upscaling:
             # Flatten -> Project -> Reshape
             attn_flat = attn_output_drop.view(B, -1)
             upscaled_flat = self.upscale_proj(attn_flat)
             attn_output_upscaled = upscaled_flat.view(B, self.L, self.d_new) # (B, L, d_new)
        else:
             attn_output_upscaled = attn_output_drop # Shape already (B, L, d_new) if L==L_new

        # Return the PRE-ATTENTION state Z for residual, and the FINAL block output
        # Return Z (B, L_new, d_new) and attn_output_upscaled (B, L, d_new)
        # Block needs to handle combining these for residual 1
        return z, attn_output_upscaled

class CausalSelfAttention(nn.Module):
    """ Standard MHA implementation """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """ Standard MLP for Transformer Block """
    def __init__(self, config, input_dim): # Takes explicit input_dim
        super().__init__()
        self.input_dim = input_dim
        # Use fixed hidden dim based on original config.n_embd
        # This means ff_dim is constant regardless of input_dim (d0 or d_new)
        hidden_dim = 4 * config.n_embd
        self.c_fc    = nn.Linear(self.input_dim, hidden_dim, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(hidden_dim, self.input_dim, bias=config.bias) # Projects back to input_dim
        self.dropout = nn.Dropout(config.dropout)
        # print(f"  MLP Initialized: Input Dim={self.input_dim}, Hidden Dim={hidden_dim}") # Less verbose

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """ Transformer Block: Handles MHA or LMA logic and preserves sequence length L """
    def __init__(self, config: LMAConfig, lma_config: LMAConfig = None):
        super().__init__()
        self.use_lma = lma_config is not None
        input_dim = config.n_embd # Block input dim is always d0

        self.ln_1 = LayerNorm(input_dim, bias=config.bias)
        if self.use_lma:
            if lma_config is None: raise ValueError("lma_config must be provided if use_lma is True")
            self.attn = LatentMetaAttention(config, lma_config) # Outputs (B, L, d_new)
            self.lma_output_dim = lma_config.d_new # Internal dimension after LMA attn

            # Layer norm and MLP operate on the LMA output dimension d_new
            self.ln_2 = LayerNorm(self.lma_output_dim, bias=config.bias)
            self.mlp = MLP(config, self.lma_output_dim) # Takes d_new, outputs d_new

            # Projection needed for the first residual connection: x(d0) -> attn_output(d_new)
            self.residual_proj1 = nn.Linear(input_dim, self.lma_output_dim, bias=config.bias)
            print(f" Block {id(self)} (LMA): Added residual proj1 ({input_dim} -> {self.lma_output_dim})")

            # Projection needed for the second residual connection's output:
            # residual_1_out(d_new) + mlp_out(d_new) -> final_out(d0)
            self.output_proj = nn.Linear(self.lma_output_dim, input_dim, bias=config.bias)
            print(f" Block {id(self)} (LMA): Added output proj ({self.lma_output_dim} -> {input_dim})")

        else: # MHA Path
            self.attn = CausalSelfAttention(config) # Outputs (B, L, d0)
            # Layer norm and MLP operate on d0
            self.ln_2 = LayerNorm(input_dim, bias=config.bias)
            self.mlp = MLP(config, input_dim)
            self.residual_proj1 = None # No projection needed
            self.output_proj = None    # No projection needed

    def forward(self, x):
        # Input x: (B, L, d0)
        # First residual path: x
        # Main path: ln_1 -> attn
        attn_output = self.attn(self.ln_1(x)) # MHA:(B,L,d0), LMA:(B,L,d_new)

        # --- First Residual Connection ---
        # Project original x to match attention output dimension if necessary (only for LMA)
        x_for_res1 = self.residual_proj1(x) if self.residual_proj1 else x
        residual_1_out = x_for_res1 + attn_output # Result is (B, L, d_new) if LMA, else (B,L,d0)

        # --- Second Residual Connection ---
        # Input to second path: residual_1_out
        # Main path: ln_2 -> mlp
        mlp_out = self.mlp(self.ln_2(residual_1_out)) # Output is (B, L, d_new) if LMA, else (B,L,d0)
        residual_2_out = residual_1_out + mlp_out # Result is (B, L, d_new) if LMA, else (B,L,d0)

        # --- Final Output Projection (LMA only) ---
        # Project back to d0 if LMA was used, to maintain consistent block output dim
        block_output = self.output_proj(residual_2_out) if self.output_proj else residual_2_out

        # Output shape MUST be (B, L, d0) for stacking blocks
        assert block_output.shape == x.shape, f"Block output shape {block_output.shape} mismatch input {x.shape}"
        return block_output

# --- GPT Class Definition ---
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    # LMA specific flags
    use_lma: bool = False
    lma_reduction_factor: int = 2
    # Removed lma_seq_reduction_type, assuming fixed logic (e.g., pooling/stride implied by mask)

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # --- Embedding Layers ---
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
        ))

        # --- Build Transformer Blocks ---
        blocks = []
        # All blocks operate with input/output dimension config.n_embd
        for i in range(config.n_layer):
            block_lma_config = None
            if config.use_lma:
                # Determine LMA internal dimensions based on config.n_embd
                # Target d_new based on block's input (which is always d0)
                target_d_new = config.n_embd // config.lma_reduction_factor
                target_d_new = max(1, target_d_new)
                # Adjust target_d_new if needed for head divisibility
                if target_d_new == 0 or config.n_head == 0: # Prevent division by zero
                     raise ValueError(f"Invalid LMA config: target_d_new={target_d_new}, n_head={config.n_head}")
                if target_d_new % config.n_head != 0:
                    target_d_new = max(config.n_head, (target_d_new // config.n_head) * config.n_head)
                    print(f"Warning: Block {i} LMA adjusted target d_new to {target_d_new} for divisibility by n_head {config.n_head}")
                # Target L_new based on block_size
                target_l_new = config.block_size // config.lma_reduction_factor
                target_l_new = max(1, target_l_new) # Ensure positive

                block_lma_config = LMAConfig(
                    d0=config.n_embd,           # LMA d0 is the standard block dim
                    L=config.block_size,        # LMA L is the standard block dim
                    n_head_stacking=config.n_head,
                    target_L_new=target_l_new,  # Target based on block_size
                    d_new=target_d_new,
                    n_head_latent=config.n_head
                )
            # Instantiate block - it handles internal logic and ensures output is d0
            block = Block(config, block_lma_config)
            blocks.append(block)
            print(f" Appending Block {i}: Type={'LMA' if block.use_lma else 'MHA'}, Output Dim={block.lma_output_dim}") # Block output is always d0

        self.transformer['h'] = nn.ModuleList(blocks)

        # --- Final Layers ---
        # Final LN & LM Head use config.n_embd since blocks preserve it
        self.transformer['ln_f'] = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        print(f"Final LayerNorm Dim: {config.n_embd}, LM Head Input Dim: {config.n_embd}")

        # --- Weight Tying ---
        # Now possible regardless of LMA use, since block output dim is consistent
        self.transformer.wte.weight = self.lm_head.weight
        print("Weight tying enabled.")

        # --- Weight Initialization ---
        self.apply(self._init_weights)
        # Special Scaled Initialization for Residual Projection Layers
        for pn, p in self.named_parameters():
            # Target c_proj in MLP and output_proj in LMA Block, upscale_proj in LMA Layer?
            proj_suffixes = ['mlp.c_proj.weight', 'attn.c_proj.weight'] # MHA attn proj, MLP proj
            if self.config.use_lma:
                 # Add LMA internal projection and block output projection
                 proj_suffixes.extend(['attn.c_proj.weight', 'output_proj.weight'])
                 # Add upscale proj if it exists
                 # Need to check names: assume attn module is named 'attn' in Block
                 if 'attn.upscale_proj.weight' in pn: proj_suffixes.append('attn.upscale_proj.weight')


            if any(pn.endswith(s) for s in proj_suffixes):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("Total number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel() # Exclude WPE
            # NOTE: People often exclude WTE too when reporting model size,
            # but nanoGPT includes it by default. Keeping it for consistency.
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def crop_block_size(self, block_size):
         # Major issue: LMA internal config (L, L_new) and mask are tied to original block_size
         # Cropping block_size would invalidate LMA layers unless re-initialized.
         if self.config.use_lma:
              raise NotImplementedError("Dynamic block size cropping is incompatible with current LMA implementation.")
         assert block_size <= self.config.block_size
         self.config.block_size = block_size
         self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
         for block in self.transformer.h:
             if hasattr(block.attn, 'bias') and isinstance(block.attn, CausalSelfAttention):
                 # Ensure bias buffer is boolean before slicing
                 if block.attn.bias.dtype == torch.bool:
                     block.attn.bias = nn.Parameter(block.attn.bias[:,:,:block_size,:block_size], requires_grad=False)
                 else: # Handle case where it might be float (older implementation?)
                      print("Warning: Unexpected bias dtype during cropping.")


    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
         override_args = override_args or {}
         # Ensure LMA is not requested when loading pretrained weights
         config_temp=GPTConfig(); use_lma_request = override_args.get('use_lma', config_temp.use_lma)
         if use_lma_request: raise NotImplementedError("Cannot load standard GPT-2 weights into LMA model.")

         from transformers import GPT2LMHeadModel # Keep import local
         assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
         assert all(k == 'dropout' for k in override_args) # Only allow overriding dropout

         print(f"loading weights from pretrained gpt: {model_type}")
         config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
         }[model_type]
         config_args['vocab_size'] = 50257
         config_args['block_size'] = 1024
         config_args['bias'] = True # GPT-2 checkpoints always used bias
         if 'dropout' in override_args: config_args['dropout'] = override_args['dropout']

         config = GPTConfig(**config_args)
         model = GPT(config) # Initialize with MHA
         sd = model.state_dict()
         sd_keys = sd.keys()
         # Ignore the buffer 'bias' in MHA layers
         sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

         # Load HF model
         model_hf = GPT2LMHeadModel.from_pretrained(model_type)
         sd_hf = model_hf.state_dict()

         # Ensure correct keys are used for HF model (may change slightly with versions)
         sd_keys_hf = sd_hf.keys()
         # Common prefixes/suffixes to ignore in HF keys that don't exist in nanoGPT
         ignore_prefixes_suffixes = ['.attn.masked_bias', '.attn.bias', '.position_ids'] # Add others if needed
         sd_keys_hf = [k for k in sd_keys_hf if not any(k.endswith(s) or k.startswith(s) for s in ignore_prefixes_suffixes)]

         # Map HF keys to nanoGPT keys (might need adjustments based on HF library version)
         # Typically: hf -> transformer.h.LAYER.SUBMODULE.WEIGHT/BIAS
         #           nano -> transformer.h.LAYER.SUBMODULE.WEIGHT/BIAS
         # But HF might have extra prefixes like 'transformer.'
         key_map = { k_hf: k_hf.replace('transformer.','') for k_hf in sd_keys_hf } # Simple initial mapping

         # Weights that need transposing
         transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight', 'lm_head.weight']

         # Check key alignment
         mapped_sd_keys_hf = set(key_map.values())
         current_sd_keys = set(sd_keys)
         if mapped_sd_keys_hf != current_sd_keys:
              print("Key mismatch detected:")
              print("Keys in nanoGPT model but not mapped from HF:", current_sd_keys - mapped_sd_keys_hf)
              print("Keys mapped from HF but not in nanoGPT model:", mapped_sd_keys_hf - current_sd_keys)
              raise RuntimeError("State dict key mismatch between HuggingFace and nanoGPT model")

         # Copy weights
         for k_hf, k_nano in key_map.items():
             if k_nano in sd: # Ensure key exists in our model
                # Handle potential transposition for Linear layers
                if any(k_nano.endswith(w) for w in transposed):
                    if sd_hf[k_hf].shape[::-1] != sd[k_nano].shape:
                         raise ValueError(f"Shape mismatch (transpose issue?) for {k_nano}: HF {sd_hf[k_hf].shape[::-1]} vs Mine {sd[k_nano].shape}")
                    with torch.no_grad():
                        sd[k_nano].copy_(sd_hf[k_hf].t())
                else:
                    # Handle potential shape mismatch for embeddings etc.
                    if sd_hf[k_hf].shape != sd[k_nano].shape:
                        # Special handling for WTE/LM Head if vocab size differs slightly (e.g. 50257 vs 50304)
                        if k_nano == 'transformer.wte.weight' or k_nano == 'lm_head.weight':
                             print(f"Handling vocab size mismatch for {k_nano}. Copying intersecting weights.")
                             min_vocab = min(sd_hf[k_hf].shape[0], sd[k_nano].shape[0])
                             with torch.no_grad():
                                  sd[k_nano][:min_vocab] = sd_hf[k_hf][:min_vocab]
                        else:
                            raise ValueError(f"Shape mismatch for {k_nano}: HF {sd_hf[k_hf].shape} vs Mine {sd[k_nano].shape}")
                    else:
                        with torch.no_grad():
                            sd[k_nano].copy_(sd_hf[k_hf])
             else:
                 print(f"Warning: Key {k_nano} (from HF key {k_hf}) not found in nanoGPT state dict.")

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
        # Correct check for device type
        is_cuda = device_type.startswith('cuda')
        use_fused = fused_available and is_cuda
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


    def estimate_mfu(self, fwdbwd_per_iter, dt):
         # Needs update for LMA to be accurate
         N = self.get_num_params()
         cfg = self.config
         L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
         flops_per_token = 6*N + 12*L*H*Q*T # MHA approximation
         if self.config.use_lma:
             # Rough estimate: Reduce attention cost part?
             # Full LMA FLOPs calc is complex. Use simpler 6N + fudge factor?
             # Let's use the simpler calc but acknowledge it's less accurate for LMA
             flops_per_token = 6*N # Base cost for params
             # Add estimate for LMA layers? Very rough.
             # Maybe scale the attention part down?
             # MHA Attn term: 12*L*H*Q*T = 12*L*(n_embd)*T
             # LMA Attn term (latent): ~ 2 * L * (L_new^2 * d_new) # Missing batch, rough
             # LMA Embed terms: L*d0*d_new + N*d0
             # It's hard to estimate accurately without full breakdown. Stick to 6N maybe?
             print("WARNING: LMA MFU estimation uses simplified 6N flops/token, underestimating actual cost.")

         flops_per_fwdbwd = flops_per_token * T
         flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
         flops_achieved = flops_per_iter * (1.0/dt) # per second
         # Reference: A100 GPU ~ 312 TFLOPS for FP16
         flops_promised = 312e12
         mfu = flops_achieved / flops_promised
         return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
         """
         Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
         the sequence max_new_tokens times, feeding the predictions back into the model each time.
         Most likely you'll want to make sure to be in model.eval() mode of operation for this.
         """
         for _ in range(max_new_tokens):
             # if the sequence context is growing too long we must crop it at block_size
             idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
             # forward the model to get the logits for the index in the sequence
             logits, _ = self(idx_cond)
             # pluck the logits at the final step and scale by desired temperature
             logits = logits[:, -1, :] / temperature
             # optionally crop the logits to only the top k options
             if top_k is not None:
                 v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                 logits[logits < v[:, [-1]]] = -float('Inf')
             # apply softmax to convert logits to (normalized) probabilities
             probs = F.softmax(logits, dim=-1)
             # sample from the distribution
             idx_next = torch.multinomial(probs, num_samples=1)
             # append sampled index to the running sequence and continue
             idx = torch.cat((idx, idx_next), dim=1)

         return idx

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        # Ensure sequence length does not exceed block size capabilities
        if t > self.config.block_size:
            idx = idx[:, :self.config.block_size]
            t = self.config.block_size
            # Targets must also be cropped if provided
            if targets is not None:
                targets = targets[:, :self.config.block_size]

        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb) # (b, t, n_embd)

        # Apply transformer blocks
        for block in self.transformer.h:
            x = block(x) # Each block preserves (B, T, n_embd) shape

        x = self.transformer.ln_f(x) # (b, t, n_embd)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x) # (b, t, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # note: using list [-1] to preserve the time dim
            logits = self.lm_head(x[:, [-1], :]) # (b, 1, vocab_size)
            loss = None

        return logits, loss

# --- Example Usage (Modified for direct execution) ---
if __name__ == '__main__':
    # Configuration for LMA
    config_args = dict(
        block_size=128, # Smaller block size for faster testing
        vocab_size=50257, # Use actual GPT2 vocab size
        n_layer=6,      # Fewer layers
        n_head=6,       # Needs to divide n_embd and d_new
        n_embd=384,     # d0 (Must be divisible by n_head)
        dropout=0.1,
        bias=True,
        use_lma=True,   # <--- Enable LMA
        lma_reduction_factor=2, # k=2 -> target L_new=64, target d_new=192
        # Removed seq reduction type from GPTConfig
    )
    gpt_config = GPTConfig(**config_args)

    # Recalculate derived LMA params for printing (if LMA is used)
    # if gpt_config.use_lma:
    #     # Need to create LMAConfig instance to trigger calculations
    #     # This happens inside GPT.__init__ now
    #     pass # Printing happens inside GPT init

    model = GPT(gpt_config)
    print(model) # Print model structure

    # --- Simple Forward/Backward Test ---
    print("\nTesting Forward/Backward Pass...")
    B = 4
    T = gpt_config.block_size # Use full block size for test
    dummy_input = torch.randint(0, gpt_config.vocab_size, (B, T))
    dummy_targets = torch.randint(0, gpt_config.vocab_size, (B, T))
    # Determine device
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
         # Use MPS only if DDP is not the issue
         # Check if running under torchrun
         import os
         if "RANK" in os.environ: # Likely torchrun/DDP environment
              print("Warning: DDP detected, forcing CPU due to MPS incompatibility.")
              device = 'cpu'
         else:
              device = 'mps'
    else:
        device = 'cpu'

    model.to(device)
    dummy_input = dummy_input.to(device)
    dummy_targets = dummy_targets.to(device)
    print(f"Using device: {device}")

    # Test forward pass
    try:
        logits, loss = model(dummy_input, dummy_targets)
        print("Forward pass successful!")
        print(f"  Logits shape: {logits.shape}") # Should be (B, T, Vocab)
        if loss is not None:
             print(f"  Loss: {loss.item()}")
             # Test backward pass only if loss was computed
             loss.backward()
             print("Backward pass successful!")
        else:
             print("  Loss is None (inference mode?). Skipping backward pass.")

    except Exception as e:
        print("\n !!! Error during forward/backward pass !!!"); print(e); import traceback; traceback.print_exc()

    # Test generation
    print("\nTesting Generation...")
    try:
        start_ids = torch.randint(0, gpt_config.vocab_size, (1, 10), device=device) # Start with 10 tokens
        model.eval() # Set model to evaluation mode
        generated_ids = model.generate(start_ids, max_new_tokens=20)
        print("Generation successful!")
        print(f"  Input IDs shape: {start_ids.shape}")
        print(f"  Generated IDs shape: {generated_ids.shape}") # Should be (1, 10+20)
        # print(f"  Generated sequence: {generated_ids.tolist()}") # Optional: print tokens
    except Exception as e:
        print("\n !!! Error during generation !!!"); print(e); import traceback; traceback.print_exc()