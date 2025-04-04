# ----- model.py -----
"""
Full definition of a GPT Language Model, all of it in this single file.
Incorporates Latent Meta Attention (LMA) as an alternative attention mechanism,
including an Initial Transform, Latent Attention Blocks, and a Learnable Decoder.
"""

import math
import inspect
from dataclasses import dataclass, field # Import field

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from contextlib import nullcontext # Make sure nullcontext is imported if used

# -----------------------------------------------------------------------------
# Helper Functions for LMA
# -----------------------------------------------------------------------------

def find_closest_divisor(total_value, target_divisor, max_delta=100):
    """
    Finds a divisor of total_value that is closest to target_divisor (PyTorch/NumPy version).
    """
    if not isinstance(total_value, int) or total_value <= 0:
        raise ValueError(f"total_value ({total_value}) must be a positive integer.")
    if not isinstance(target_divisor, int) or target_divisor <= 0:
        # Allow target_divisor <= 0, but start search from 1
        print(f"Warning: find_closest_divisor target_divisor ({target_divisor}) is not positive. Starting search near 1.")
        target_divisor = max(1, target_divisor) # Ensure search starts positively
    if not isinstance(max_delta, int) or max_delta < 0:
        raise ValueError(f"max_delta ({max_delta}) must be non-negative.")

    if total_value == 0: # Handle edge case
        return 1 # Or raise error? Returning 1 might be safer downstream.

    # Check target directly only if > 0 and it divides total_value
    if target_divisor > 0 and total_value % target_divisor == 0:
        return target_divisor

    # Ensure search_start is at least 1
    search_start = max(1, target_divisor)

    for delta in range(1, max_delta + 1):
        candidate_minus = search_start - delta
        if candidate_minus > 0 and total_value % candidate_minus == 0:
            return candidate_minus
        candidate_plus = search_start + delta
        # Check divisibility only for positive candidates (though total_value > 0 assumed)
        if candidate_plus > 0 and total_value % candidate_plus == 0:
            return candidate_plus

    # Fallback if no divisor found nearby: Find *any* divisor.
    # Start from 1 up to sqrt(total_value)
    for i in range(1, int(math.sqrt(total_value)) + 1):
        if total_value % i == 0:
            print(f"Warning: No divisor found near {target_divisor}. Using {i} as a fallback divisor for {total_value}.")
            return i
    # If total_value is prime > 1, its only divisor other than 1 is itself
    if total_value > 1:
        print(f"Warning: No divisor found near {target_divisor}. Using {total_value} as a fallback divisor.")
        return total_value

    # If total_value was 1, the loop range(1, 1+1) finds 1.
    # If total_value was 0, we returned 1 earlier.
    # This final raise should theoretically not be reached if total_value > 0.
    raise ValueError(
        f"Could not find any valid divisor for {total_value} near {target_divisor} "
        f"(within +/- {max_delta}) or as fallback. Check L, d0, or target L_new."
    )


def get_lma_causal_mask(L, n_h, L_new, device):
    """
    Calculates the causal mask for the LMA latent attention space (PyTorch version).
    Mask=True means attention is prevented.
    `L` and `n_h` refer to the *original* dimensions before the transformation that produced L_new.
    """
    # Basic validation
    if L <= 0: L = 1 # Prevent division by zero if original L was invalid
    if n_h <= 0: n_h = 1
    if L_new <= 0:
        print(f"Warning: Invalid L_new={L_new} for mask generation. Returning None.")
        return None

    L_prime = L * n_h # Total number of 'items' after head stacking
    if L_prime == 0 :
        print(f"Warning: L*nh ({L_prime}) is zero. Cannot generate mask. Returning None.")
        return None

    # Calculate the effective stride: how many items in L_prime map to one item in L_new
    # Handle potential non-divisibility cleanly
    if L_new == 0: # Avoid division by zero
        print(f"Warning: L_new is zero. Cannot calculate stride. Returning None.")
        return None
    elif L_prime < L_new:
        # This case implies L_new was chosen larger than L*nh, which shouldn't happen
        # if L_new is derived correctly from L*d0. But handle defensively.
        print(f"Warning: L*nh ({L_prime}) < L_new ({L_new}). Mask generation assumes stride=1.")
        k_stride = 1
    else:
        # This is the expected calculation based on how L_new and C_new divide L*d0
        # We assume the reduction from L*nh to L_new is proportional.
        # Total features = L * d0. L_new * C_new = L * d0.
        # L_prime = L * n_h. d_k = d0 / n_h.
        # Stride should relate L_prime to L_new.
        k_stride = L_prime / L_new # Use float division initially for accuracy
        # Check if stride makes sense or if L*nh isn't divisible by L_new
        if L_prime % L_new != 0 :
            print(f"Warning: L*nh ({L_prime}) not perfectly divisible by L_new ({L_new}). Mask uses effective float stride {k_stride:.2f}.")
            # Keep k_stride as float for calculations, convert range endpoints to int

    # Calculate the maximum original sequence index (0 to L-1) associated with each latent position
    max_orig_index_per_latent_pos = [-1] * L_new
    for i_new in range(L_new):
        # Calculate the range in L_prime corresponding to this latent position i_new
        p_start_float = i_new * k_stride
        p_end_float = (i_new + 1) * k_stride

        # Convert to integer indices for range iteration (exclusive end)
        p_start = int(math.floor(p_start_float))
        p_end = int(math.ceil(p_end_float)) # Use ceil to be inclusive of the boundary
        p_end = min(p_end, L_prime) # Ensure it doesn't exceed L_prime

        max_orig_l = -1
        if p_start < p_end: # Check if the range is valid
            # Iterate through the corresponding positions in the stacked view (L*nh)
            for p in range(p_start, p_end):
                # Find the original sequence index (modulo L) for this stacked position
                orig_l_index = p % L
                max_orig_l = max(max_orig_l, orig_l_index)

        max_orig_index_per_latent_pos[i_new] = max_orig_l

    # Create boolean mask (True means mask out)
    mask = torch.zeros((L_new, L_new), device=device, dtype=torch.bool)
    for i_new in range(L_new): # Query position in latent space
        query_max_orig = max_orig_index_per_latent_pos[i_new]
        if query_max_orig == -1: continue # Should not happen if L > 0

        for j_new in range(L_new): # Key position in latent space
            key_max_orig = max_orig_index_per_latent_pos[j_new]
            if key_max_orig == -1: continue

            # Mask if the key's latest original index is *later* than the query's latest original index
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
        expected_dim = self.weight.shape[0]
        if input.size(-1) != expected_dim:
             # Try to get module hierarchy for better error message
             name_parts = []
             curr = self
             while hasattr(curr, '_parent_module_and_name'): # Heuristic to find parent names
                 parent, name = curr._parent_module_and_name
                 name_parts.append(name)
                 if parent is None or not isinstance(parent, nn.Module): break
                 curr = parent
             layer_name_str = ".".join(reversed(name_parts)) if name_parts else "UnknownLayerNorm"

             print(f"ERROR in LayerNorm ({layer_name_str}): Input dim {input.size(-1)} != Weight dim {expected_dim}")
             # Provide more context about input shape
             print(f"Input shape: {input.shape}")
             raise RuntimeError(f"LayerNorm ({layer_name_str}) input dim ({input.size(-1)}) mismatch with weight dim ({expected_dim})")
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

@dataclass
class LMAConfig:
    """ Configuration specific to the LMA layer internals. """
    d0: int # Input dimension to the transformation/attention layer
    L: int  # Input sequence length to the transformation/attention layer
    n_head_stacking: int # Number of heads used for stacking (Stage 2a) before THIS layer
    target_L_new: int # Target latent sequence length
    d_new: int        # Target latent dimension
    n_head_latent: int # Number of heads for attention *within* the latent space

    # Fields calculated after initialization
    L_new: int = field(init=False) # Actual latent sequence length (divisor of L*d0)
    C_new: int = field(init=False) # Dimension of chunks before latent embedding (L*d0 / L_new)

    def __post_init__(self):
        # Validate inputs
        if self.L <= 0: raise ValueError(f"LMAConfig Error: L ({self.L}) must be positive.")
        if self.d0 <= 0: raise ValueError(f"LMAConfig Error: d0 ({self.d0}) must be positive.")
        if self.n_head_stacking <= 0: raise ValueError(f"LMAConfig Error: n_head_stacking ({self.n_head_stacking}) must be positive.")
        if self.target_L_new <= 0: raise ValueError(f"LMAConfig Error: target_L_new ({self.target_L_new}) must be positive.")
        if self.d_new <= 0: raise ValueError(f"LMAConfig Error: d_new ({self.d_new}) must be positive.")
        if self.n_head_latent <= 0: raise ValueError(f"LMAConfig Error: n_head_latent ({self.n_head_latent}) must be positive.")

        # Calculate total features based on INPUT L and d0
        total_features = self.L * self.d0
        if total_features == 0:
             # This shouldn't happen due to checks above, but safeguard
             print(f"Warning: LMAConfig total features (L*d0) is zero. Setting L_new/C_new to 1.")
             self.L_new = 1
             self.C_new = 1
             return # Exit early

        # Calculate L_new (actual latent sequence length) and C_new (intermediate chunk dim)
        try:
            # Find the closest divisor to target_L_new for the total features
            self.L_new = find_closest_divisor(total_features, self.target_L_new)
            if self.L_new != self.target_L_new:
                 print(f"LMAConfig ADJUSTMENT: Target L_new ({self.target_L_new}) changed to {self.L_new} to divide total features ({total_features}).")

            # Check if L_new is valid before division
            if self.L_new <= 0:
                raise ValueError(f"Calculated L_new ({self.L_new}) is not positive.")

            # Calculate C_new (must be integer division)
            if total_features % self.L_new != 0:
                 # This should NOT happen if find_closest_divisor worked correctly
                 raise RuntimeError(f"Internal Error: total_features ({total_features}) not divisible by calculated L_new ({self.L_new})")
            self.C_new = total_features // self.L_new

            if self.C_new <= 0:
                raise ValueError(f"Calculated C_new ({self.C_new}) is not positive.")

        except ValueError as e:
            # Provide more context in the error message
            raise ValueError(f"LMA Config Error calculating L_new/C_new from L={self.L}, d0={self.d0}, target_L_new={self.target_L_new}: {e}") from e

        # Validate d_new divisibility by n_head_latent AFTER d_new is confirmed positive
        if self.d_new % self.n_head_latent != 0:
             raise ValueError(f"LMA Config Error: d_new ({self.d_new}) must be divisible by n_head_latent ({self.n_head_latent}).")
        # Validate d0 divisibility by n_head_stacking (relevant for head stacking step)
        if self.d0 % self.n_head_stacking != 0:
             raise ValueError(f"LMA Config Error: d0 ({self.d0}) must be divisible by n_head_stacking ({self.n_head_stacking}).")


class LatentMetaAttention(nn.Module):
    """ LMA Core Logic - Operates entirely in the PRE-EXISTING latent space """
    def __init__(self, config, lma_latent_config: LMAConfig):
        # lma_latent_config defines the space this attention operates in (L_new, d_new)
        # AND contains info about the *original* space (L, n_head_stacking) for mask calculation.
        super().__init__()
        self.config = config # Main GPTConfig
        self.lma_config = lma_latent_config # LMAConfig specific to this layer

        # Dimensions this attention layer operates on:
        self.d_latent = lma_latent_config.d_new # Dimension of the latent space
        self.L_latent = lma_latent_config.L_new # Expected max length of the latent sequence
        self.n_head_latent = lma_latent_config.n_head_latent
        self.bias = config.bias
        self.dropout = config.dropout

        # Validate operational dimensions
        if not (self.d_latent > 0 and self.n_head_latent > 0 and self.d_latent % self.n_head_latent == 0):
             raise ValueError(f"Invalid latent attention params: d_latent={self.d_latent}, n_head_latent={self.n_head_latent}")

        print(f"  Initializing LatentMetaAttention Layer: Operates on Latent(L_latent={self.L_latent}, d_latent={self.d_latent}), Heads={self.n_head_latent}")

        # --- Latent Attention Layers (QKV projections + MHA) ---
        # These layers operate directly on the input Z (which has shape B, T_latent, d_latent)
        self.q_proj = nn.Linear(self.d_latent, self.d_latent, bias=self.bias)
        self.k_proj = nn.Linear(self.d_latent, self.d_latent, bias=self.bias)
        self.v_proj = nn.Linear(self.d_latent, self.d_latent, bias=self.bias)

        # Using PyTorch's MultiheadAttention for the core operation in the latent space
        self.latent_attn = nn.MultiheadAttention(
            embed_dim=self.d_latent,
            num_heads=self.n_head_latent,
            dropout=self.dropout,
            bias=self.bias,
            batch_first=True # Expect input as (B, T_latent, d_latent)
        )
        # Output projection after latent attention
        self.c_proj = nn.Linear(self.d_latent, self.d_latent, bias=self.bias)
        self.resid_dropout = nn.Dropout(self.dropout)

        # # --- Causal Mask ---
        # # The mask needs to be generated based on the *original* L and n_h
        # # that *produced* this latent space. LMAConfig provides these.
        # self.original_L_for_mask = lma_latent_config.L # Original L before transform
        # self.original_nH_for_mask = lma_latent_config.n_head_stacking # Original nH before transform

        # # Generate mask once on CPU, register as buffer. Moved device transfer to forward.
        # try:
        #     print(f"  LMA Attn: Generating causal mask based on original L={self.original_L_for_mask}, nH={self.original_nH_for_mask} for L_latent={self.L_latent}")
        #     # Use the L and n_head_stacking from the config that *defined* this latent space
        #     lma_mask = get_lma_causal_mask(
        #         L=self.original_L_for_mask,
        #         n_h=self.original_nH_for_mask,
        #         L_new=self.L_latent, # Target latent length
        #         device='cpu' # Create on CPU initially
        #     )
        #     if lma_mask is None:
        #          print("  LMA Attn: WARNING - Causal mask generation failed. Attention will not be causal.")
        #          self.register_buffer("causal_mask_latent", None, persistent=False)
        #     else:
        #          # Mask shape should be (L_latent, L_latent)
        #          assert lma_mask.shape == (self.L_latent, self.L_latent)
        #          self.register_buffer("causal_mask_latent", lma_mask, persistent=False)
        #          print(f"  LMA Attn: Registered latent causal mask ({self.L_latent}x{self.L_latent})")

        # except Exception as e:
        #     print(f"ERROR generating LMA causal mask: {e}")
        #     import traceback; traceback.print_exc()
        #     self.register_buffer("causal_mask_latent", None, persistent=False)
        # --- ADD THIS INSTEAD ---
        print(f"  LMA Attn: Using SIMPLE TRIL mask for latent sequence (L_latent={self.L_latent})")
        # Create a standard lower-triangular mask of size (L_latent, L_latent)
        simple_mask = torch.tril(torch.ones(self.L_latent, self.L_latent, dtype=torch.bool, device='cpu'))
        # nn.MultiheadAttention expects True where positions are *masked out* (prevented from attending).
        # So, we need the *upper* triangle to be True. Invert the lower-triangular mask.
        simple_mask_inverted = ~simple_mask
        self.register_buffer("causal_mask_latent", simple_mask_inverted, persistent=False)

    def forward(self, z): # Input z is ALREADY LATENT (B, T_latent, d_latent)
        B, T_latent, C_latent = z.size()

        # Assert input matches expected latent dimensions, allow shorter sequence T_latent <= L_latent
        if T_latent > self.L_latent:
             # This shouldn't happen if padding/truncation works correctly upstream
             print(f"Warning: LatentMetaAttention input T_latent ({T_latent}) > configured L_latent ({self.L_latent}). Truncating.")
             z = z[:, :self.L_latent, :]
             T_latent = self.L_latent
        if C_latent != self.d_latent:
            raise ValueError(f"LatentMetaAttention input C ({C_latent}) != configured d_latent ({self.d_latent})")

        # --- Latent Attention ---
        q_prime = self.q_proj(z)
        k_prime = self.k_proj(z)
        v_prime = self.v_proj(z)

        # Prepare attention mask
        attn_mask_to_use = None
        if self.causal_mask_latent is not None:
            # Slice the pre-computed mask based on the *current* latent sequence length T_latent
            # Mask needs to be (T_latent, T_latent)
            # Ensure T_latent is not larger than the mask dimension L_latent
            current_T_latent_for_mask = min(T_latent, self.L_latent)
            attn_mask_slice = self.causal_mask_latent[:current_T_latent_for_mask, :current_T_latent_for_mask]

            # Move mask to the correct device
            attn_mask_to_use = attn_mask_slice.to(q_prime.device)

            # MHA expects True where attention should be *prevented*.
            # Our get_lma_causal_mask already returns True for masked positions.

        # Apply MultiheadAttention
        # is_causal=False because we provide an explicit mask
        attn_output, _ = self.latent_attn(
            q_prime, k_prime, v_prime,
            attn_mask=attn_mask_to_use,
            need_weights=False, # Don't need attention weights output
            is_causal=False # We handle causality via attn_mask
        )
        # attn_output shape: (B, T_latent, d_latent)

        # Apply output projection and dropout
        attn_output_proj = self.c_proj(attn_output)
        attn_output_drop = self.resid_dropout(attn_output_proj) # (B, T_latent, d_latent)

        # Block handles residual addition: returns pre-attention input and attention output
        return attn_output_drop # Return only the processed output


# --- LMA Initial Transformation Layer (Revised for Padding) ---
class LMA_InitialTransform(nn.Module):
    """ Performs Stage 1 and Stage 2 of LMA to map (B,T,d0) -> (B,L_new,d_new) """
    def __init__(self, config, lma_config: LMAConfig):
        super().__init__()
        self.config = config # Main GPTConfig
        self.lma_config = lma_config # LMAConfig for *this specific transform*

        # Validate dimensions based on the provided lma_config
        if lma_config.d0 <= 0 or lma_config.n_head_stacking <= 0:
             raise ValueError("LMA Initial Transform: d0 and n_head_stacking must be positive.")
        if lma_config.d0 % lma_config.n_head_stacking != 0:
             raise ValueError(f"LMA Initial Transform: d0 ({lma_config.d0}) must be divisible by n_head_stacking ({lma_config.n_head_stacking}).")

        # Configured dimensions for this transformation
        self.d0 = lma_config.d0 # Input feature dimension (e.g., n_embd)
        self.L = lma_config.L   # Expected *maximum* input sequence length (e.g., block_size)
        self.n_head_stacking = lma_config.n_head_stacking # Heads for Stage 2a
        self.d_k = self.d0 // self.n_head_stacking # Dimension per head view

        # Output dimensions of this transformation
        self.d_new = lma_config.d_new # Target latent dimension
        self.L_new = lma_config.L_new # Target latent sequence length
        self.C_new = lma_config.C_new # Intermediate chunk dimension

        self.bias = config.bias

        print(f" Init LMA InitialTransform: In(Max L={self.L}, d0={self.d0}, nH_stack={self.n_head_stacking}) -> Out(L_new={self.L_new}, d_new={self.d_new})")
        print(f"   Intermediate calculated: d_k={self.d_k}, C_new={self.C_new}")

        # Layer for Stage 2b Embedding (mapping C_new -> d_new)
        self.embed_layer_2 = nn.Linear(self.C_new, self.d_new, bias=self.bias)
        #self.embed_layer_2_act = nn.ReLU() # Or nn.GELU()? Using ReLU as specified originally
        self.embed_layer_2_act = nn.GELU()

    def forward(self, y): # Input y is (B, T, d0) from initial embedding/prev block
        B, T, C = y.size()
        if C != self.d0:
            raise ValueError(f"LMA InitialTransform input C({C}) != configured d0({self.d0})")

        # --- Handle sequences shorter or longer than self.L ---
        padded_y = y
        current_T_for_processing = T

        if T < self.L:
            padding_size = self.L - T
            # Pad sequence length (dim 1). Use 0 for padding value.
            padded_y = F.pad(y, (0, 0, 0, padding_size)) # Pads (last dim C), (second-to-last dim T)
            # print(f"LMA InitialTransform: Padded input T={T} to L={self.L}")
            current_T_for_processing = self.L # Use the padded length for subsequent processing
        elif T > self.L:
             # This case should ideally be handled by truncation *before* calling the model
             # Truncate from the end to keep the most recent tokens if T > L
             print(f"Warning: LMA InitialTransform received T={T} > L={self.L}. Truncating input to last {self.L} tokens.")
             padded_y = y[:, -self.L:, :]
             current_T_for_processing = self.L
        # If T == self.L, padded_y is just y, and current_T_for_processing is L

        # Ensure the input for head splitting has length current_T_for_processing (which is self.L after padding/truncation)
        assert padded_y.size(1) == self.L, f"Shape after padding/truncation mismatch: Got {padded_y.shape}, expected T={self.L}"

        # --- Stage 2a: Head-View Stacking ---
        # Use padded_y which now has shape (B, self.L, d0)
        try:
            head_views = torch.split(padded_y, self.d_k, dim=2)
        except RuntimeError as e:
             raise RuntimeError(f"Error splitting heads in LMA InitialTransform: d0={self.d0}, d_k={self.d_k}. Is d0 divisible by n_head_stacking? Input shape={padded_y.shape}") from e
        x_stacked = torch.cat(head_views, dim=1) # Shape: (B, self.L * self.n_head_stacking, d_k)

        # --- Stage 2b: Re-Chunking & Latent Embedding ---
        # The total number of features should be self.L * self.d0
        total_features_expected = self.L * self.d0
        # Check total elements after stacking (B * L*nH * dk = B * L * d0)
        if x_stacked.numel() != B * total_features_expected:
             raise RuntimeError(f"Internal dimension mismatch in LMA InitialTransform after stacking. "
                                f"Expected {B*total_features_expected} elements, got {x_stacked.numel()}. "
                                f"Check L={self.L}, d0={self.d0}, nH={self.n_head_stacking}, d_k={self.d_k}, "
                                f"Input T={T}, padded_y={padded_y.shape}, x_stacked={x_stacked.shape}")

        # Flatten features: (B, self.L * d0)
        x_flat = x_stacked.view(B, -1)

        # Check if total features match expected L_new * C_new based on config
        expected_latent_features = self.L_new * self.C_new
        if x_flat.shape[1] != expected_latent_features:
            # This indicates an issue in the LMAConfig calculation or its application
            raise RuntimeError(f"LMA Config mismatch during transform: Input L*d0 ({total_features_expected}) != Calculated L_new*C_new ({expected_latent_features}). "
                               f"Check LMAConfig values: L={self.L}, d0={self.d0}, L_new={self.L_new}, C_new={self.C_new}")

        # Reshape into latent steps: (B, L_new, C_new)
        x_rechunked = x_flat.view(B, self.L_new, self.C_new)

        # Apply embedding layer (maps C_new -> d_new)
        # Reshape for Linear layer: (B * L_new, C_new)
        z_embedded_flat = self.embed_layer_2(x_rechunked.view(-1, self.C_new))
        # Apply activation
        z_activated = self.embed_layer_2_act(z_embedded_flat)
        # Reshape back to latent space dimensions: (B, L_new, d_new)
        z = z_activated.view(B, self.L_new, self.d_new)

        return z # Output is the first latent representation, always (B, L_new, d_new)

# --- GPT Class Definition (Unchanged) ---
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # LMA Specific Config
    use_lma: bool = False           # Whether to use LMA architecture
    lma_reduction_factor: int = 2   # Target reduction factor for L and d in initial transform
    
# --- Learnable Decoder for LMA ---
class LMA_Decoder(nn.Module):
    """ Learns to map latent sequence (B, L_new, d_new) back to (B, T, d_output) """
    def __init__(self, config: GPTConfig, lma_config: LMAConfig):
        super().__init__()
        self.config = config # Main GPTConfig
        self.lma_config = lma_config # Config providing L_new, d_new

        self.L_new = lma_config.L_new
        self.d_new = lma_config.d_new
        # Define the output dimension of the decoder. Let's keep it same as d_new for simplicity.
        # Could also map back to config.n_embd if desired.
        self.d_output = self.d_new
        self.bias = config.bias
        self.dropout = config.dropout

        print(f" Init LMA Decoder: In(L_new={self.L_new}, d_new={self.d_new}) -> Out(T, d_output={self.d_output})")

        # Using simple refinement layers after interpolation
        # LayerNorm -> Linear -> Activation -> Linear -> Dropout
        # These operate on the feature dimension (d_new)
        self.ln = LayerNorm(self.d_new, bias=self.bias)
        # Make hidden dim proportional to d_new, similar to MLP structure
        hidden_dim = self.d_new * 2 # Can be tuned
        self.fc1 = nn.Linear(self.d_new, hidden_dim, bias=self.bias)
        self.act = nn.GELU() # Common activation in transformers
        self.fc2 = nn.Linear(hidden_dim, self.d_output, bias=self.bias)
        self.drop = nn.Dropout(self.dropout)

    def forward(self, z, target_T): # z shape: (B, L_new, d_new), target_T is original sequence length
        B, current_L_new, current_d_new = z.shape

        if current_L_new != self.L_new:
             # This might happen if input T to model was < block_size, affecting L_new calculation somehow?
             # Or if L_new wasn't calculated correctly. For now, warn and proceed.
             print(f"Warning: LMA_Decoder received L_new={current_L_new}, expected {self.L_new}. Proceeding.")
             # Let interpolation handle the actual input length current_L_new
        if current_d_new != self.d_new:
            raise ValueError(f"LMA_Decoder input d ({current_d_new}) != configured d_new ({self.d_new})")

        # 1. Interpolate length
        # Permute to (B, C, L) for interpolation function
        z_permuted = z.permute(0, 2, 1) # -> (B, d_new, current_L_new)
        z_interpolated = F.interpolate(
            z_permuted,
            size=target_T,      # Target original sequence length T
            mode='linear',      # Linear interpolation
            align_corners=False # Recommended for non-image data / linear mode
        ) # -> (B, d_new, target_T)

        # Permute back to (B, L, C)
        z_upsampled = z_interpolated.permute(0, 2, 1) # -> (B, target_T, d_new)

        # 2. Refine features with learnable layers
        z_norm = self.ln(z_upsampled)
        z_hidden = self.act(self.fc1(z_norm))
        z_refined = self.fc2(z_hidden)
        z_output = self.drop(z_refined) # -> (B, target_T, d_output)

        # Optional: Add residual connection?
        # If d_output == d_new, we could potentially add z_upsampled here.
        # Let's keep it simple first.
        # if self.d_output == self.d_new:
        #    z_output = z_upsampled + z_output

        return z_output # Shape: (B, target_T, d_output)


# --- Standard Causal Self Attention ---
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
            # Note: This fixed mask assumes block_size length. Dynamic slicing needed in forward.
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size), persistent=False)
        else:
             print("Using Flash Attention.")
             self.register_buffer("bias", None, persistent=False) # No bias needed for flash


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        if C != self.n_embd:
             raise ValueError(f"CausalSelfAttention input C({C}) != config n_embd({self.n_embd})")

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # is_causal=True handles masking internally
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # Apply causal mask dynamically based on current sequence length T
            if self.bias is None: raise RuntimeError("Slow attention requires bias buffer")
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


# --- MLP (Operates on Block's Input/Output Dim) ---
class MLP(nn.Module):
    def __init__(self, config, block_internal_dim): # Takes dimension of data within the block
        super().__init__()
        self.input_dim = block_internal_dim
        # Hidden dim often 4x n_embd, but here let's make it 4x block_internal_dim
        hidden_dim = 4 * self.input_dim
        self.c_fc = nn.Linear(self.input_dim, hidden_dim, bias=config.bias)
        self.gelu = nn.GELU()
        # Project back to the block's internal dimension
        self.c_proj = nn.Linear(hidden_dim, self.input_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if x.size(-1) != self.input_dim:
             raise ValueError(f"MLP input dim {x.size(-1)} != expected {self.input_dim}")
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# --- Block (Revised to handle LMA or MHA consistently) ---
class Block(nn.Module):
    """ Transformer Block: Can use either CausalSelfAttention or LatentMetaAttention """
    def __init__(self, config: GPTConfig, is_lma: bool, lma_config: LMAConfig = None):
        super().__init__()
        self.use_lma = is_lma

        # Determine block's operating dimension based on LMA config or main config
        if self.use_lma:
            if lma_config is None:
                raise ValueError("lma_config must be provided if is_lma is True for a Block")
            # LMA blocks operate *within* the latent space defined by lma_config
            self.operating_dim = lma_config.d_new # Should be d_latent
            self.operating_L = lma_config.L_new # Should be L_latent
            print(f"Initializing Block {id(self)} (LMA): Operates on dim={self.operating_dim}, max_L={self.operating_L}")
            self.attn = LatentMetaAttention(config, lma_config)
        else: # Standard MHA
            # MHA blocks operate on the main embedding dimension
            self.operating_dim = config.n_embd
            self.operating_L = config.block_size # Max sequence length
            print(f"Initializing Block {id(self)} (MHA): Operates on dim={self.operating_dim}, max_L={self.operating_L}")
            self.attn = CausalSelfAttention(config)

        # LayerNorms and MLP operate on the dimension internal to this block
        self.ln_1 = LayerNorm(self.operating_dim, bias=config.bias)
        self.mlp = MLP(config, self.operating_dim) # MLP sized according to operating dim
        self.ln_2 = LayerNorm(self.operating_dim, bias=config.bias)


    def forward(self, x): # Input x: (B, T_current, self.operating_dim)
        B, T_current, C_current = x.shape

        # Check input dimension matches the block's operating dimension
        if C_current != self.operating_dim:
             block_type = "LMA" if self.use_lma else "MHA"
             raise ValueError(f"Block ({block_type}) input C ({C_current}) != block operating_dim ({self.operating_dim})")
        # Sequence length check (optional, can be handled by attention mask slicing)
        # if T_current > self.operating_L:
        #      print(f"Warning: Block input T ({T_current}) > max configured L ({self.operating_L}). Ensure attention mask handles this.")

        # Attention + Residual Path
        # Apply ln_1 -> attn -> residual
        x_norm1 = self.ln_1(x)
        attn_output = self.attn(x_norm1) # Attn modules handle their respective inputs/masking
        x = x + attn_output # Residual connection 1

        # MLP + Residual Path
        # Apply ln_2 -> mlp -> residual
        x_norm2 = self.ln_2(x)
        mlp_output = self.mlp(x_norm2)
        x = x + mlp_output # Residual connection 2

        # Output shape: (B, T_current, self.operating_dim)
        # If LMA, T_current is T_latent. If MHA, T_current is T.
        return x


# --- Main GPT Model (Revised with Initial Transform and Decoder) ---
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # --- Validate LMA Config dependencies ---
        if config.use_lma:
             if config.n_embd % config.n_head != 0:
                  raise ValueError(f"LMA requires n_embd ({config.n_embd}) to be divisible by n_head ({config.n_head}) for initial head stacking.")
             if config.lma_reduction_factor <= 0:
                  raise ValueError(f"LMA requires lma_reduction_factor ({config.lma_reduction_factor}) to be > 0.")

        # --- Embedding Layers ---
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
        ))

        # --- Determine dimensions and architecture flow ---
        self.initial_lma_transform = None
        self.lma_decoder = None
        self.initial_lma_cfg = None # Store the config used for the transform

        current_d = config.n_embd     # Dimension flowing into blocks
        current_L = config.block_size # Max sequence length flowing into blocks
        self.operates_in_latent = False # Flag if blocks run in latent space

        # --- Optional: Initial LMA Transformation ---
        if config.use_lma:
            print("--- Configuring LMA: Initial Transformation ---")
            self.operates_in_latent = True # Blocks will run in latent space
            reduction_factor = max(1, config.lma_reduction_factor)
            target_l_new_init = config.block_size // reduction_factor
            target_d_new_init = config.n_embd // reduction_factor
            target_l_new_init = max(1, target_l_new_init)
            target_d_new_init = max(1, target_d_new_init)
            latent_n_head = config.n_head # Use same n_head for latent attention

            # Adjust d_new to be divisible by latent_n_head
            if target_d_new_init == 0: raise ValueError("Initial LMA target_d_new calculated as zero.")
            if target_d_new_init % latent_n_head != 0:
                original_target_d = target_d_new_init
                target_d_new_init = max(latent_n_head, (target_d_new_init // latent_n_head) * latent_n_head)
                if target_d_new_init == 0: target_d_new_init = latent_n_head
                print(f"LMA Init: Adjusted target d_new from {original_target_d} to {target_d_new_init} to be divisible by latent n_head {latent_n_head}")

            try:
                self.initial_lma_cfg = LMAConfig(
                    d0=config.n_embd, L=config.block_size, n_head_stacking=config.n_head,
                    target_L_new=target_l_new_init, d_new=target_d_new_init, n_head_latent=latent_n_head
                )
                self.initial_lma_transform = LMA_InitialTransform(config, self.initial_lma_cfg)
                # Update dimensions for blocks
                current_L = self.initial_lma_cfg.L_new # Blocks operate on latent length L_new
                current_d = self.initial_lma_cfg.d_new # Blocks operate on latent dimension d_new
                print(f"--- Dimensions into Blocks: Latent L={current_L}, Latent D={current_d} ---")
            except ValueError as e:
                 print(f"ERROR configuring Initial LMA Transform: {e}")
                 raise e

        # --- Build Transformer Blocks ---
        print(f"--- Building {config.n_layer} Transformer Blocks ---")
        blocks = []
        for i in range(config.n_layer):
            block_lma_config = None
            is_lma_block = self.operates_in_latent

            if is_lma_block:
                # If LMA is used, blocks operate in the latent space defined by initial_lma_cfg.
                # The LMAConfig passed to the block's LatentMetaAttention needs to reflect this.
                if self.initial_lma_cfg is None: # Should not happen if operates_in_latent is True
                    raise RuntimeError("Internal setup error: operates_in_latent=True but initial_lma_cfg is None.")

                # Config for the attention *inside* the block
                block_lma_config = LMAConfig(
                     d0=current_d,               # Attention input is current latent d
                     L=current_L,                # Attention input is current latent L
                     # Mask calculation needs original context before *initial transform*
                     n_head_stacking=config.n_head, # Orig heads used for stacking
                     # Latent attention parameters (no further reduction within blocks)
                     target_L_new=current_L,     # Stays L_new
                     d_new=current_d,            # Stays d_new
                     n_head_latent=self.initial_lma_cfg.n_head_latent # Use latent heads from initial config
                 )
                 # We need to pass the original L (block_size) to LatentMetaAttention for the mask.
                 # Modify L in block_lma_config *only for mask calculation purpose*.
                 # This is slightly hacky - maybe LMAConfig needs explicit orig_L field?
                 # Let's patch it here for now:
                #block_lma_config.L = config.block_size # Patch L for mask calculation inside LatentMetaAttention

                # Instantiate block, telling it it's LMA and passing the config
                block = Block(config, is_lma=True, lma_config=block_lma_config)

            else: # Standard MHA block
                block = Block(config, is_lma=False)

            blocks.append(block)
            # Note: Output dimensions of block match input dimension (current_d, current_L)

        self.transformer['h'] = nn.ModuleList(blocks)

        # --- Optional: LMA Decoder ---
        self.final_ln_lm_head_dim = current_d # Store the dimension entering final LN/LMHead
        if self.operates_in_latent:
            if self.initial_lma_cfg is None: # Sanity check
                 raise RuntimeError("Internal setup error: operates_in_latent=True but initial_lma_cfg is None.")
            print("--- Configuring LMA: Decoder ---")
            self.lma_decoder = LMA_Decoder(config, self.initial_lma_cfg)
            # The decoder outputs d_output, which we set to d_new (current_d)
            self.final_ln_lm_head_dim = self.lma_decoder.d_output
            print(f"--- Dimension after Decoder (into Final LN/LMHead): {self.final_ln_lm_head_dim} ---")

        # --- Final Layers ---
        # These operate on the output dimension of the last block (if MHA)
        # or the output dimension of the decoder (if LMA)
        self.transformer['ln_f'] = LayerNorm(self.final_ln_lm_head_dim, bias=config.bias)
        self.lm_head = nn.Linear(self.final_ln_lm_head_dim, config.vocab_size, bias=False)
        print(f"--- Final LN & LM Head operating on dimension: {self.final_ln_lm_head_dim} ---")

        # Weight Tying check
        # Only tie if not using LMA AND the final dimension matches the embedding dimension
        if not config.use_lma and self.final_ln_lm_head_dim == config.n_embd:
            self.transformer.wte.weight = self.lm_head.weight
            print("Weight tying enabled.")
        else:
             if config.use_lma: reason = "LMA is used"
             elif self.final_ln_lm_head_dim != config.n_embd: reason = f"final dim {self.final_ln_lm_head_dim} != n_embd {config.n_embd}"
             else: reason = "Unknown"
             print(f"Weight tying disabled ({reason}).")

        # Init weights
        self.apply(self._init_weights)
        # Apply special scaled init to projection weights in residual connections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'): # Applies to MHA proj, MLP proj
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
            # Consider scaling for LMA projections too?
            # if 'lma_decoder' in pn and pn.endswith('weight'): # Example
            #      torch.nn.init.normal_(p, mean=0.0, std=0.02) # Standard init for decoder for now

        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to parameter sharing these
        params are actually used as weights in the final layer, so we include them.
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
             # Weight is already initialized to ones by default in LayerNorm constructor

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size() # Batch size, Sequence length

        # --- Input Length Handling ---
        if t > self.config.block_size:
            # Truncate input sequence if longer than block size
            idx = idx[:, -self.config.block_size:]
            t = self.config.block_size # Update sequence length
            # Truncate targets accordingly if they exist
            if targets is not None:
                 targets = targets[:, -self.config.block_size:]

        # --- Embeddings ---
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd) -> broadcasted? No, need (1, t, n_embd)
        # pos_emb needs to be added correctly. Let's ensure it's broadcastable.
        # wpe output is (block_size, n_embd). We need (t, n_embd).
        pos_emb = self.transformer.wpe(pos) # Shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb) # Shape (b, t, n_embd)

        # Store original sequence length T needed for decoder target length
        original_T = t

        # --- LMA Initial Transformation (if applicable) ---
        if self.initial_lma_transform is not None:
            # Transform handles padding internally if t < block_size
            x = self.initial_lma_transform(x)
            # Output x shape: (B, L_new, d_new)
            # Sequence length dimension is now L_new

        # --- Transformer Blocks ---
        # Blocks operate on the current shape of x (either latent or original)
        for block in self.transformer.h:
            x = block(x)
            # Output shape remains (B, L_new, d_new) if LMA, or (B, T, n_embd) if MHA

        # --- LMA Decoder (if applicable) ---
        if self.lma_decoder is not None:
            # Decode back to original sequence length T
            x = self.lma_decoder(x, original_T)
            # Output x shape: (B, T, d_output) where d_output is likely d_new

        # --- Final Layers ---
        # Apply final LayerNorm
        x = self.transformer.ln_f(x)
        # Output shape: (B, T, final_ln_lm_head_dim)

        # Calculate logits
        if targets is not None:
            # Training: calculate loss
            logits = self.lm_head(x) # Shape: (B, T, vocab_size)
            # Reshape for cross_entropy: (B*T, vocab_size) and (B*T,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference: calculate logits only for the last position.
            # Note: We still process the full sequence through decoder, LN.
            # This is slightly inefficient but consistent with training structure.
            # We could potentially optimize inference later if needed.
            logits = self.lm_head(x[:, [-1], :]) # Shape: (B, 1, vocab_size)
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # TODO: implement this
        # note: Requires handling Positional Embeddings and potentially LMA masks/configs if block_size changes L
        # For now, raise error or ignore. LMA makes this complex.
        # assert block_size <= self.config.block_size # cannot grow block size
        # self.config.block_size = block_size
        # self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        # if hasattr(self.transformer, 'bias'): # For non-flash MHA
        #    self.transformer.bias = self.transformer.bias[:,:,:block_size,:block_size]
        print("Warning: crop_block_size not fully implemented, especially for LMA.")
        raise NotImplementedError("LMA block size cropping not fully supported yet.")

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # TODO: implement loading pretrained weights (e.g., from HuggingFace)
        # Needs careful mapping of weights, especially if using LMA.
        print(f"Warning: from_pretrained not implemented for model_type '{model_type}'.")
        raise NotImplementedError("Loading pretrained models not supported yet.")

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
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
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B // https://arxiv.org/abs/2204.02311 for details
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter # number of microsteps per step
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        # --- Adjust MFU for LMA ---
        # This estimation is based on standard MHA. LMA changes computation significantly.
        # The initial transform, latent attention, and decoder have different costs.
        # A precise MFU for LMA would require detailed FLOP counting of each component.
        # For now, we just return the MHA-based estimate with a warning.
        print("WARNING: MFU estimate based on standard MHA, may be inaccurate for LMA.")
        # Heuristic adjustment: Reduce MFU if LMA seems cheaper? (e.g., smaller latent dim)
        adjustment_factor = 0.8 # Placeholder adjustment
        return mfu * adjustment_factor


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        self.eval() # Ensure model is in eval mode
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond) # Runs full forward pass (incl. decoder if LMA)
            # pluck the logits at the final step and scale by desired temperature
            # logits shape is (B, T_cond, V), we need the last position T_cond-1
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
        self.train() # Return model to train mode if needed elsewhere
        return idx

# -----------------------------------------------------------------------------
# HellaSwag evaluation logic
# -----------------------------------------------------------------------------

def get_most_likely_row(tokens, mask, logits):
    # Evaluate the autoregressive loss at all positions
    # logits shape: (batch_size, T, vocab_size)
    # tokens shape: (batch_size, T)
    # mask shape: (batch_size, T) where 1 means evaluate loss
    shift_logits = logits[..., :-1, :].contiguous() # Shape: (batch, T-1, V)
    shift_tokens = tokens[..., 1:].contiguous()     # Shape: (batch, T-1)
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1)) # Shape: (batch*(T-1), V)
    flat_shift_tokens = shift_tokens.view(-1)                         # Shape: (batch*(T-1),)

    # Calculate loss per token, but don't reduce yet
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1) # Shape: (batch, T-1)

    # Now get the average loss just for the completion region (where mask == 1)
    # Shift mask to align with shifted losses/tokens
    shift_mask = mask[..., 1:].contiguous() # Shape: (batch, T-1)
    masked_shift_losses = shift_losses * shift_mask

    # Sum and divide by the number of loss tokens (prevent division by zero)
    sum_loss = masked_shift_losses.sum(dim=1)
    num_loss_tokens = shift_mask.sum(dim=1)
    avg_loss = sum_loss / (num_loss_tokens + 1e-6) # Add epsilon for stability

    # Handle cases where num_loss_tokens is 0 (no completion tokens?)
    # In this case, avg_loss will be 0. Assign a high loss instead?
    # If a row had zero completion tokens, its loss should be penalized.
    # Let's check for num_loss_tokens == 0 and assign infinity?
    avg_loss[num_loss_tokens == 0] = float('inf')


    # Now find the choice (row) with the minimal average loss
    pred_norm = avg_loss.argmin().item()
    return pred_norm

@torch.no_grad()
def evaluate_hellaswag(model, enc, hellaswag_path='data/hellaswag/hellaswag_val.jsonl'):
    """ Runs HellaSwag evaluation and returns accuracy """
    import json
    import tqdm
    import os # For path joining
    from tiktoken.core import Encoding # For type hint

    assert isinstance(enc, Encoding), "Encoder `enc` must be a tiktoken Encoding object"

    print(f"Evaluating HellaSwag from {hellaswag_path}...")
    num_correct_norm = 0
    num_total = 0
    
    # Ensure path exists
    if not os.path.exists(hellaswag_path):
        print(f"Error: HellaSwag validation file not found at {hellaswag_path}")
        # Attempt to download if using standard nanoGPT structure
        data_dir = os.path.dirname(hellaswag_path)
        if not os.path.exists(data_dir): os.makedirs(data_dir)
        val_url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
        print(f"Attempting to download from {val_url}...")
        try:
            import requests
            with requests.get(val_url, stream=True) as r:
                r.raise_for_status()
                with open(hellaswag_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download successful.")
        except Exception as e:
            print(f"Download failed: {e}. Cannot evaluate HellaSwag.")
            return -1.0 # Indicate error

    # Determine device of the model
    model_device = next(model.parameters()).device

    try:
        with open(hellaswag_path, 'r') as f:
            for line in tqdm.tqdm(f, desc="HellaSwag Eval"): # Use tqdm for progress bar
                example = json.loads(line)
                num_total += 1
                ctx = example['ctx']
                label = example['label'] # Integer index of correct ending
                endings = example['endings'] # List of 4 ending strings

                # Encode context and each ending
                ctx_tokens = enc.encode(ctx)
                if not ctx_tokens: # Handle empty context (rare)
                    print(f"Warning: Skipping example with empty context: {example.get('activity_label', 'N/A')}")
                    num_total -=1
                    continue

                tok_rows = []
                mask_rows = []

                for end in endings:
                    completion_tokens = enc.encode(end)
                    # Combine context and completion
                    tok = ctx_tokens + completion_tokens
                    mask = [0]*len(ctx_tokens) + [1]*len(completion_tokens) # Evaluate loss only on completion

                    # Truncate to model's block size *from the left*
                    if len(tok) > model.config.block_size:
                        # Ensure we don't remove the entire completion if ctx is very long
                        num_completion_tokens = len(completion_tokens)
                        max_ctx_len = model.config.block_size - num_completion_tokens
                        if max_ctx_len < 0:
                             # Completion itself is longer than block size, truncate completion
                             print(f"Warning: Completion longer than block size ({num_completion_tokens} > {model.config.block_size}). Truncating completion.")
                             completion_tokens = completion_tokens[:model.config.block_size] # Take first part of completion
                             tok = completion_tokens # Use only truncated completion
                             mask = [1] * len(tok) # Mask is all 1s
                             max_ctx_len=0 # No context left

                        # Truncate context if needed
                        start_index = max(0, len(ctx_tokens) - max_ctx_len)
                        truncated_ctx_tokens = ctx_tokens[start_index:]

                        # Rebuild truncated sequence and mask
                        tok = truncated_ctx_tokens + completion_tokens
                        mask = [0]*len(truncated_ctx_tokens) + [1]*len(completion_tokens)

                        # Final check on length
                        if len(tok) > model.config.block_size:
                             # This can happen if completion was also truncated but combo still too long?
                             # Should be rare after previous check. Truncate combined sequence.
                             print(f"Warning: Final truncation needed for combined sequence (len={len(tok)}).")
                             tok = tok[-model.config.block_size:]
                             mask = mask[-model.config.block_size:]


                    tok_rows.append(torch.tensor(tok, dtype=torch.long))
                    mask_rows.append(torch.tensor(mask, dtype=torch.long))

                # Batch the rows for efficiency
                # Find max length *in this batch of 4*
                max_len = max(len(row) for row in tok_rows)
                tokens = torch.zeros((len(tok_rows), max_len), dtype=torch.long)
                mask = torch.zeros((len(tok_rows), max_len), dtype=torch.long)
                for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
                    # Pad each row to max_len
                    tokens[i, :len(tok_row)] = tok_row
                    mask[i, :len(mask_row)] = mask_row

                # Move batch to model's device
                tokens = tokens.to(model_device)
                mask = mask.to(model_device)

                # Get the logits from the model
                model.eval()
                # Ensure ctx is a valid context manager before using
                if not hasattr(ctx, '__enter__') or not hasattr(ctx, '__exit__'):
                    print("Warning: Invalid context manager passed to evaluate_hellaswag. Using nullcontext.")
                    ctx = nullcontext() # Fallback safely

                with ctx: # Use the passed-in ctx
                    logits, _ = model(tokens)

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"ERROR: NaNs or Infs detected in logits during HellaSwag eval!")
                    # Optionally print the problematic input 'tokens' here
                    # print(f"Problematic tokens (first 10): {tokens[:, :10]}")
                    # Skip this example or return an error metric
                    # For now, let's skip the row calculation for this example
                    print(f"Skipping HellaSwag example due to NaN/Inf logits.")
                    # How to handle skipping? We can't just continue, need to finish the loop.
                    # Assign a default prediction that's likely wrong?
                    pred_norm = 0 # Or some other default incorrect label index
                    # Or maybe raise an exception? Let's try assigning a default first.
                    # Need to adjust num_total? No, better to just get it wrong.
                else:
                    # Only calculate row if logits are valid
                    pred_norm = get_most_likely_row(tokens, mask, logits)

                # Check if prediction matches label
                if pred_norm == label:
                    num_correct_norm += 1

    except FileNotFoundError: # Already handled by check/download above, but keep for safety
        print(f"Error: HellaSwag validation file not found at {hellaswag_path}")
        return -1.0 # Indicate error
    except Exception as e:
        print(f"Error during HellaSwag evaluation: {e}")
        import traceback; traceback.print_exc()
        return -1.0 # Indicate error

    acc_norm = num_correct_norm / num_total if num_total > 0 else 0.0
    print(f"HellaSwag Accuracy: {acc_norm*100:.2f}% ({num_correct_norm}/{num_total})")
    return acc_norm

# -----------------------------------------------------------------------------
# Example Usage (Modified for direct execution)
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Configuration for LMA
    config_args = dict(
        block_size=128, # Smaller block size for faster testing
        vocab_size=50257, # Use actual GPT2 vocab size
        n_layer=4,      # Fewer layers
        n_head=4,       # Needs to divide n_embd (e.g., 4)
        n_embd=128,     # d0 (Must be divisible by n_head) (e.g., 128)
        dropout=0.1,
        bias=True,
        use_lma=True,           # <--- Enable LMA
        lma_reduction_factor=2, # k=2 -> target L_new=64, target d_new=64
    )
    gpt_config = GPTConfig(**config_args)

    print("\n--- Model Configuration ---")
    print(gpt_config)
    print("\n--- Initializing Model ---")
    model = GPT(gpt_config)
    print("\n--- Model Structure ---")
    # print(model) # Can be very verbose, print summary instead maybe

    # --- Simple Forward/Backward Test ---
    print("\n--- Testing Forward/Backward Pass ---")
    B = 4
    T = gpt_config.block_size # Use full block size for test
    T_short = T // 2         # Test with shorter sequence too

    dummy_input_full = torch.randint(0, gpt_config.vocab_size, (B, T))
    dummy_targets_full = torch.randint(0, gpt_config.vocab_size, (B, T)) # Use -1 for ignore? No, use actual tokens

    dummy_input_short = torch.randint(0, gpt_config.vocab_size, (B, T_short))
    dummy_targets_short = torch.randint(0, gpt_config.vocab_size, (B, T_short))

    # Determine device
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
         # Check if running under torchrun/DDP which conflicts with MPS
         import os
         if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
              print("Warning: DDP environment detected, forcing CPU due to potential MPS incompatibility.")
              device = 'cpu'
         else:
              device = 'mps'
    else:
        device = 'cpu'

    print(f"Using device: {device}")
    model.to(device)

    optimizer = model.configure_optimizers(weight_decay=1e-1, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device)

    # --- Test Full Length ---
    print(f"\nTesting with T = {T}...")
    dummy_input = dummy_input_full.to(device)
    dummy_targets = dummy_targets_full.to(device)
    try:
        model.train() # Ensure train mode
        optimizer.zero_grad()
        logits, loss = model(dummy_input, dummy_targets)
        print("Forward pass successful!")
        print(f"  Logits shape: {logits.shape}") # Should be (B, T, Vocab)
        if loss is not None:
             print(f"  Loss: {loss.item()}")
             loss.backward()
             optimizer.step()
             print("Backward pass and optimizer step successful!")
        else:
             print("  Loss is None (should not happen in training).")
    except Exception as e:
        print(f"\n !!! Error during forward/backward pass (T={T}) !!!"); print(e); import traceback; traceback.print_exc()

    # --- Test Short Length ---
    print(f"\nTesting with T = {T_short}...")
    dummy_input = dummy_input_short.to(device)
    dummy_targets = dummy_targets_short.to(device)
    try:
        model.train() # Ensure train mode
        optimizer.zero_grad()
        logits, loss = model(dummy_input, dummy_targets)
        print("Forward pass successful!")
        print(f"  Logits shape: {logits.shape}") # Should be (B, T_short, Vocab)
        if loss is not None:
             print(f"  Loss: {loss.item()}")
             loss.backward()
             optimizer.step()
             print("Backward pass and optimizer step successful!")
        else:
             print("  Loss is None (should not happen in training).")
    except Exception as e:
        print(f"\n !!! Error during forward/backward pass (T={T_short}) !!!"); print(e); import traceback; traceback.print_exc()


    # Test generation
    print("\n--- Testing Generation ---")
    try:
        start_ids = torch.randint(0, gpt_config.vocab_size, (1, 10), device=device) # Start with 10 tokens
        model.eval() # Set model to evaluation mode
        generated_ids = model.generate(start_ids, max_new_tokens=20, temperature=0.8, top_k=5)
        print("Generation successful!")
        print(f"  Input IDs shape: {start_ids.shape}")
        print(f"  Generated IDs shape: {generated_ids.shape}") # Should be (1, 10+20)
        # print(f"  Generated sequence: {generated_ids.tolist()}") # Optional: print tokens
    except Exception as e:
        print("\n !!! Error during generation !!!"); print(e); import traceback; traceback.print_exc()

    # Optional: HellaSwag Test (requires tiktoken)
    print("\n--- Testing HellaSwag (requires tiktoken) ---")
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        # Make sure the path to hellaswag_val.jsonl is correct
        # Or let the function try to download it to data/hellaswag/
        hs_path = os.path.join('data', 'hellaswag', 'hellaswag_val.jsonl')
        accuracy = evaluate_hellaswag(model, enc, hs_path)
        print(f"HellaSwag evaluation finished. Accuracy: {accuracy:.4f}")
    except ImportError:
        print("tiktoken not installed, skipping HellaSwag evaluation.")
        print("Install with: pip install tiktoken")
    except Exception as e:
        print(f"\n !!! Error during HellaSwag evaluation !!!"); print(e); import traceback; traceback.print_exc()