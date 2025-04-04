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
    """ LMA Core Logic - Operates entirely in the PRE-EXISTING latent space """
    def __init__(self, config, lma_latent_config):
        # lma_latent_config should contain L_new, d_new, n_head_latent
        super().__init__()
        self.config = config
        self.lma_config = lma_latent_config # Contains L_new, d_new etc.

        self.d_new = lma_latent_config.d_new
        self.L_new = lma_latent_config.L_new # EXPECTED Latent length
        self.n_head_latent = lma_latent_config.n_head_latent
        self.bias = config.bias
        self.dropout = config.dropout

        assert self.d_new > 0 and self.n_head_latent > 0 and self.d_new % self.n_head_latent == 0

        print(f"  Initializing LMA Attention Layer: Operates on Latent(L_new={self.L_new}, d_new={self.d_new}), Heads={self.n_head_latent}")

        # --- Latent Attention Layers (QKV projections + MHA) ---
        # These layers operate directly on the input Z (which has shape B, L_new, d_new)
        self.q_proj = nn.Linear(self.d_new, self.d_new, bias=self.bias)
        self.k_proj = nn.Linear(self.d_new, self.d_new, bias=self.bias)
        self.v_proj = nn.Linear(self.d_new, self.d_new, bias=self.bias)
        self.latent_attn = nn.MultiheadAttention(
            embed_dim=self.d_new,
            num_heads=self.n_head_latent,
            dropout=self.dropout,
            bias=self.bias,
            batch_first=True
        )
        # Output projection after latent attention
        self.c_proj = nn.Linear(self.d_new, self.d_new, bias=self.bias)
        self.resid_dropout = nn.Dropout(self.dropout)

        # --- Causal Mask ---
        # The mask needs to be generated based on the *original* L and n_h
        # that *produced* this latent space. LMAConfig needs these original values.
        # We assume lma_latent_config contains the ORIGINAL L and n_head_stacking
        if not hasattr(lma_latent_config, 'L') or not hasattr(lma_latent_config, 'n_head_stacking'):
             raise AttributeError("LMAConfig for LatentMetaAttention needs original L and n_head_stacking for mask.")
        try:
            lma_mask = get_lma_causal_mask(lma_latent_config.L, lma_latent_config.n_head_stacking, self.L_new, device='cpu')
            self.register_buffer("causal_mask_latent", lma_mask, persistent=False)
            if lma_mask is not None: print(f"  LMA Attn: Registered mask ({self.L_new}x{self.L_new})")
        except Exception as e: print(f"ERROR LMA mask: {e}"); self.register_buffer("causal_mask_latent", None, persistent=False)

    def forward(self, z, current_seq_len): # Input z is ALREADY LATENT (B, T_latent, d_new)
        B, T_latent, C_latent = z.size();
        # Assert input matches expected latent dimensions
        # Allow T_latent <= self.L_new for generation
        if T_latent > self.L_new: raise ValueError(f"LMA Attn input T({T_latent}) > config L_new({self.L_new})")
        assert C_latent == self.d_new, f"LMA Attn input C({C_latent}) != config d_new({self.d_new})"

        # --- Latent Attention ---
        q_prime = self.q_proj(z); k_prime = self.k_proj(z); v_prime = self.v_proj(z)

        attn_mask_to_use = self.causal_mask_latent; assert attn_mask_to_use is not None
        # Slice mask based on *current* latent sequence length T_latent
        attn_mask_to_use = attn_mask_to_use[:T_latent, :T_latent] # Dynamic slicing
        attn_mask_to_use = attn_mask_to_use.to(q_prime.device)

        attn_output, _ = self.latent_attn(q_prime, k_prime, v_prime, attn_mask=attn_mask_to_use, need_weights=False, is_causal=False)
        attn_output_proj = self.c_proj(attn_output)
        attn_output_drop = self.resid_dropout(attn_output_proj) # (B, T_latent, d_new)

        # Return PRE-ATTENTION input Z and FINAL ATTENTION output
        # Block handles residual addition
        return z, attn_output_drop # Both (B, T_latent, d_new)

# --- LMA Initial Transformation Layer (Applies Stage 1 and Stage 2) ---
class LMA_InitialTransform(nn.Module):
    """ Performs Stage 1 and Stage 2 of LMA to map (B,L,d0) -> (B,L_new,d_new) """
    def __init__(self, config, lma_config: LMAConfig):
        super().__init__()
        self.config = config
        self.lma_config = lma_config
        # Validate dimensions
        assert lma_config.d0 % lma_config.n_head_stacking == 0
        self.d0=lma_config.d0; self.L=lma_config.L; self.n_head_stacking=lma_config.n_head_stacking; self.d_k=self.d0 // self.n_head_stacking;
        self.d_new=lma_config.d_new; self.L_new=lma_config.L_new; self.C_new=lma_config.C_new; self.bias=config.bias;

        print(f" Init LMA InitialTransform: In(L={self.L},d0={self.d0}) -> Out(L_new={self.L_new},d_new={self.d_new})")
        # Layer for Stage 2b Embedding
        self.embed_layer_2 = nn.Linear(self.C_new, self.d_new, bias=self.bias)
        self.embed_layer_2_act = nn.ReLU()

    def forward(self, y): # Input y is (B, L, d0) from initial embedding/prev block
        B, T, C = y.size()
        # Assume T==L for this transform layer
        if T != self.L: raise NotImplementedError(f"LMA InitialTransform requires T({T})==L({self.L})")
        assert C == self.d0, f"LMA InitialTransform input C({C}) != d0({self.d0})"

        # --- Stage 2a: Head-View Stacking ---
        head_views = torch.split(y, self.d_k, dim=2); x_stacked = torch.cat(head_views, dim=1)
        # --- Stage 2b: Re-Chunking & Latent Embedding ---
        x_flat = x_stacked.view(B, -1); x_rechunked = x_flat.view(B, self.L_new, self.C_new)
        z_embedded_flat = self.embed_layer_2(x_rechunked.view(-1, self.C_new)); z = self.embed_layer_2_act(z_embedded_flat);
        z = z.view(B, self.L_new, self.d_new) # (B, L_new, d_new)

        return z # Output is the first latent representation

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


# --- MLP (Operates on Block's Output Dim) ---
class MLP(nn.Module):
    def __init__(self, config, block_output_dim): # Takes dim of data within block
        super().__init__(); self.input_dim = block_output_dim;
        hidden_dim = 4 * config.n_embd # Hidden based on ORIGINAL n_embd
        self.c_fc = nn.Linear(self.input_dim, hidden_dim, bias=config.bias); self.gelu = nn.GELU();
        self.c_proj = nn.Linear(hidden_dim, self.input_dim, bias=config.bias); # Back to block's output dim
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x): x = self.c_fc(x); x = self.gelu(x); x = self.c_proj(x); x = self.dropout(x); return x

class Block(nn.Module):
    """ Transformer Block: MHA (preserves dims) or LMA (operates in latent dims) """
    def __init__(self, config: GPTConfig, is_lma=False, lma_config: LMAConfig = None):
        super().__init__()
        self.use_lma = is_lma
        # Determine the actual input/output dimensions for THIS block
        # In Option A, these change layer by layer. We need the expected input.
        # This needs modification in GPT.__init__ to pass the correct input_d.
        # Let's assume GPT init passes correct input_d for now.
        # For simplicity, removing input_L/input_d from __init__ args, assume set by GPT loop
        self.input_d = config.n_embd # Placeholder - will be overwritten by GPT init logic

        if self.use_lma:
            if lma_config is None: raise ValueError("lma_config needed for LMA block")
            self.attn = LatentMetaAttention(config, lma_config) # Attention operates in L_new, d_new
            self.output_d = lma_config.d_new # Block outputs d_new
            self.ln_1 = LayerNorm(lma_config.d0, bias=config.bias) # LN1 takes LMA input d0
            self.ln_2 = LayerNorm(lma_config.d_new, bias=config.bias) # LN2 takes latent d_new
            self.mlp = MLP(config, lma_config.d_new) # MLP takes/outputs latent d_new
            print(f"Initializing Block (LMA): Expects D={lma_config.d0} -> Outputs D={self.output_d}")
        else: # MHA Path
            self.attn = CausalSelfAttention(config) # Operates on n_embd
            self.output_d = config.n_embd # Block outputs n_embd
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
            self.mlp = MLP(config, config.n_embd) # Operates on n_embd
            print(f"Initializing Block (MHA): Input/Output D={self.output_d}")

    def forward(self, x, current_seq_len): # Input x: (B, T, current_d)
        x_norm1 = self.ln_1(x)
        if self.use_lma:
            # Pass T if LMA needs it for masking (though current LMA assumes T=L)
            z, attn_output = self.attn(x_norm1, current_seq_len) # attn takes (B,L,d0), outputs (B,L_new,d_new)
            residual_1_out = z + attn_output # In latent space (B, L_new, d_new)
        else:
            attn_output = self.attn(x_norm1) # (B, T, d0)
            residual_1_out = x + attn_output # In original space (B, T, d0)

        mlp_out = self.mlp(self.ln_2(residual_1_out)) # Operates on d_new or d0
        block_output = residual_1_out + mlp_out # Output matches residual_1_out shape

        return block_output # (B, L_new, d_new) for LMA, (B, T, d0) for MHA

# --- GPT Class Definition (Revised for Option A - Consistent Latent Space) ---
@dataclass
class GPTConfig: # Unchanged
    block_size: int = 1024; vocab_size: int = 50304; n_layer: int = 12; n_head: int = 12
    n_embd: int = 768; dropout: float = 0.0; bias: bool = True
    use_lma: bool = False; lma_reduction_factor: int = 2

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__(); self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
        ))

        blocks = []
        current_L = config.block_size
        current_d = config.n_embd
        self.latent_L = current_L # Store final L after blocks
        self.latent_d = current_d # Store final D after blocks

        # --- Optional: Initial Transformation Layer ---
        # If use_lma, apply the transformation *once* before the blocks start
        self.initial_lma_transform = None
        if config.use_lma:
            print("--- Creating Initial LMA Transformation ---")
            # Calculate initial L_new, d_new based on config
            target_l_new_init = config.block_size // config.lma_reduction_factor
            target_d_new_init = config.n_embd // config.lma_reduction_factor
            target_l_new_init = max(1, target_l_new_init)
            target_d_new_init = max(1, target_d_new_init)
            if config.n_embd % config.n_head != 0: raise ValueError(...) # Check head divisibility
            if target_d_new_init % config.n_head != 0: target_d_new_init = max(config.n_head, (target_d_new_init // config.n_head) * config.n_head)

            # Create LMAConfig specifically for this initial transformation
            initial_lma_cfg = LMAConfig(
                d0=config.n_embd, L=config.block_size, n_head_stacking=config.n_head,
                target_L_new=target_l_new_init, d_new=target_d_new_init, n_head_latent=config.n_head
            )
            self.initial_lma_transform = LMA_InitialTransform(config, initial_lma_cfg)
            # Update dimensions for subsequent blocks
            current_L = initial_lma_cfg.L_new
            current_d = initial_lma_cfg.d_new
            self.latent_L = current_L
            self.latent_d = current_d
            print(f"--- Dimensions after Initial Transform: L={current_L}, D={current_d} ---")
        # ---------------------------------------------

        # --- Build Transformer Blocks (Operating in consistent space) ---
        for i in range(config.n_layer):
            is_lma_block = config.use_lma # Apply LMA to all blocks if enabled
            block_lma_config = None

            if is_lma_block:
                # ALL subsequent LMA blocks operate on the *same* latent dimensions
                # We need an LMAConfig reflecting this: input (d0, L) and latent (d_new, L_new)
                # are the *same* for these blocks. The LMAConfig needs to hold these latent dims.

                # Create LMAConfig for the ATTENTION layer inside the block
                # Its d0 and L parameters are the latent dimensions from the previous step
                lma_attention_cfg = LMAConfig(
                    d0=current_d, L=current_L, # Input to *attention module* is latent space
                    n_head_stacking=config.n_head, # Stacking uses n_head
                    target_L_new=current_L,        # Target L_new is current_L (no further reduction)
                    d_new=current_d,               # Target d_new is current_d
                    n_head_latent=config.n_head    # Latent attention uses n_head
                )
                # Create the Block, passing the config for the attention layer
                # The Block itself takes current_d as input/output dimension
                block = Block(config, is_lma=True, lma_config=lma_attention_cfg, input_d=current_d)
            else:
                # Standard MHA block operates on config.n_embd
                block = Block(config, is_lma=False, input_d=current_d) # Pass current_d

            blocks.append(block)
            # Dimensions are preserved by MHA block, or LMA block (operating d_new -> d_new)
            current_L = block.output_L # Should remain constant after initial transform
            current_d = block.output_d # Should remain constant after initial transform
            print(f" Appending Block {i}: Type={'LMA' if is_lma_block else 'MHA'}, Output Shape=({current_L}, {current_d})")

        self.transformer['h'] = nn.ModuleList(blocks)
        # Final Layers use the dimensions AFTER all blocks (which is now the consistent latent dim if LMA used)
        self.transformer['ln_f'] = LayerNorm(current_d, bias=config.bias)
        self.lm_head = nn.Linear(current_d, config.vocab_size, bias=False)
        print(f"Final LN/LMHead Dim: {current_d}")

        # Weight Tying check
        if current_d == config.n_embd: self.transformer.wte.weight = self.lm_head.weight; print("Weight tying enabled.")
        else: print(f"Weight tying disabled (final_d {current_d} != n_embd {config.n_embd}).")

        self.apply(self._init_weights);
        for pn, p in self.named_parameters(): # Scaled init
             if p.dim() >= 2 and ('c_proj.weight' in pn or 'mlp.c_fc.weight' in pn):
                  torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        print("Total params: %.2fM" % (self.get_num_params()/1e6,))

    # --- Methods: get_num_params, _init_weights, configure_optimizers, generate, etc. ---
    # (Copy previous working versions, noting LMA limitations)
    def get_num_params(self, non_embedding=True): n_params = sum(p.numel() for p in self.parameters()); n_params -= self.transformer.wpe.weight.numel() if non_embedding else 0; return n_params
    def _init_weights(self, module):
        if isinstance(module, nn.Linear): torch.nn.init.normal_(module.weight, mean=0.0, std=0.02);
        if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}; decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]; nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]; optim_groups = [{'params': decay_params, 'weight_decay': weight_decay}, {'params': nodecay_params, 'weight_decay': 0.0}]; num_decay_params = sum(p.numel() for p in decay_params); num_nodecay_params = sum(p.numel() for p in nodecay_params); print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"); print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"); fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters; is_cuda = device_type.startswith('cuda'); use_fused = fused_available and is_cuda; extra_args = dict(fused=True) if use_fused else dict(); optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args); print(f"using fused AdamW: {use_fused}"); return optimizer
    def crop_block_size(self, block_size): raise NotImplementedError("LMA block size cropping not fully supported")
    @classmethod
    def from_pretrained(cls, model_type, override_args=None): raise NotImplementedError("LMA from_pretrained not supported")
    def estimate_mfu(self, fwdbwd_per_iter, dt): print("WARNING: LMA MFU estimation not accurate."); N = self.get_num_params(); L, T = self.config.n_layer, self.config.block_size; flops_per_token = 6*N; flops_per_fwdbwd = flops_per_token * T; flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter; flops_achieved = flops_per_iter * (1.0/dt); flops_promised = 312e12; mfu = flops_achieved / flops_promised; return mfu * 0.8
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
         final_L = self.latent_L # Use the final latent length determined at init
         block_size_used = self.config.block_size # Max input length
         for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= block_size_used else idx[:, -block_size_used:]
            logits, _ = self(idx_cond); # Forward pass
            # Logits are (B, final_L, Vocab). Take the features for the *last latent position*
            logits = logits[:, -1, :] / temperature
            if top_k is not None: v, _ = torch.topk(logits, min(top_k, logits.size(-1))); logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1); idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
         return idx

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        # Input length handling
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        if t > self.config.block_size:
            idx = idx[:, -self.config.block_size:]; pos = pos[-self.config.block_size:]; t = self.config.block_size
            if targets is not None: targets = targets[:, -self.config.block_size:]

        # Embeddings
        tok_emb = self.transformer.wte(idx); pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb) # (b, t, n_embd)

        # --- Apply Initial LMA Transform if enabled ---
        if self.initial_lma_transform is not None:
             # This transform expects input T == config.block_size
             if t != self.config.block_size:
                  raise NotImplementedError(f"Initial LMA transform requires T({t}) == block_size({self.config.block_size})")
             x = self.initial_lma_transform(x) # Output (B, latent_L, latent_d)
             # Update t to reflect new latent length for loss check / final logit selection
             t_final = self.latent_L # The sequence length after transform
        else:
             t_final = t # Sequence length remains original t

        # Apply transformer blocks sequentially
        for block in self.transformer.h:
             # Pass the current sequence length expected by the block
             # For MHA, T can vary. For LMA internal mask, we assume T=L_latent
             current_block_input_len = x.size(1)
             x = block(x, current_block_input_len) # Block preserves L_latent/d_latent if LMA

        x = self.transformer.ln_f(x) # Applied to final block output dim (latent_d if LMA)

        # --- Output Head & Loss ---
        if targets is not None:
            logits = self.lm_head(x) # Input (B, t_final, latent_d), Output (B, t_final, vocab_size)

            # Problem: Targets are (B, T). Logits are (B, t_final). Need to match.
            # Apply the upsampling fix HERE, after all blocks and ln_f.
            if logits.size(1) != targets.size(1):
                target_len = targets.size(1)    # Original T
                latent_len = logits.size(1)     # t_final (L_new after blocks)
                print(f"Shape mismatch for loss: Logits L={latent_len}, Targets L={target_len}. Upsampling logits...")
                if latent_len == 0: raise RuntimeError("Latent sequence length is zero!")
                if target_len % latent_len != 0: raise RuntimeError(f"Cannot upsample: Target L ({target_len}) not multiple of Latent L ({latent_len}).")
                upsample_factor = target_len // latent_len

                # Upsample logits using repeat_interleave
                # Reshape logits to allow repeat on seq dim: (B, L_new, V) -> (B*L_new, V)
                # Repeat: (B*L_new*factor, V)
                # Reshape back: (B, L_new*factor, V) = (B, T, V)
                V = logits.size(-1)
                logits_upsampled = logits.repeat_interleave(upsample_factor, dim=1)
                # Ensure shape is correct
                if logits_upsampled.size(1) != target_len:
                     raise RuntimeError(f"Upsampling Error: Output L {logits_upsampled.size(1)} != Target L {target_len}")
                logits = logits_upsampled # Use upsampled logits

            # Now sequence lengths match
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else: # Inference
             logits = self.lm_head(x[:, [-1], :]); loss = None # Use last position of final latent sequence x

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