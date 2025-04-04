from helper import *

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class LatentMetaAttention(nn.Module):
    """
    Latent Meta Attention (LMA) layer for PyTorch.

    Implements: Head-Stacking -> Re-Chunking -> Latent Embedding -> Latent Attention
    Assumes input is the output of Stage 1 (Initial Embedding).
    Handles causal masking within the latent space.
    """
    def __init__(self, config, lma_config):
        super().__init__()
        assert lma_config.d0 % lma_config.n_head_stacking == 0, "d0 must be divisible by n_head_stacking"
        assert lma_config.d_new % lma_config.n_head_latent == 0, "d_new must be divisible by n_head_latent"

        self.d0 = lma_config.d0 # Original embedding dim (input to this module)
        self.L = lma_config.L   # Original sequence length (input to this module)
        self.n_head_stacking = lma_config.n_head_stacking # Heads for stacking (nh)
        self.d_k = self.d0 // self.n_head_stacking        # Dim per stacking head

        self.target_L_new = lma_config.target_L_new       # Desired L_new (e.g., L // 2)
        self.d_new = lma_config.d_new                     # Target latent embedding dim
        self.n_head_latent = lma_config.n_head_latent     # Heads for latent attention
        self.bias = config.bias                           # Use bias from main config
        self.dropout = config.dropout

        # --- Calculate adjusted L_new and C_new ---
        total_features = self.L * self.d0
        try:
            # Find closest valid L_new that divides total_features
            self.L_new = find_closest_divisor(total_features, self.target_L_new)
            if self.L_new != self.target_L_new:
                print(f"LMA ADJUSTMENT: Target L_new ({self.target_L_new}) changed to {self.L_new} to divide total features ({total_features}).")
        except ValueError as e:
            raise ValueError(f"LMA Config Error: {e}") from e

        self.C_new = total_features // self.L_new # Size of chunks for second embedding
        # --- End Adjustment ---

        print(f"LMA Initialized: L={self.L}, d0={self.d0}, n_h_stack={self.n_head_stacking} -> L_new={self.L_new}, d_new={self.d_new}, C_new={self.C_new}, n_h_latent={self.n_head_latent}")

        # --- Layers ---
        # Stage 2b: Latent Embedding (maps C_new -> d_new)
        self.embed_layer_2 = nn.Linear(self.C_new, self.d_new, bias=self.bias)

        # Latent Attention Layers (QKV projections + MHA)
        # Combine QKV projection for potential efficiency? Let's keep separate for now.
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
        # Dropout for attention output AND residual connection
        self.resid_dropout = nn.Dropout(self.dropout)

        # --- Causal Mask ---
        # Pre-compute the custom LMA causal mask
        # Need to know the sequence reduction stride/method. Assuming stride = L'/L_new
        reduction_stride = (self.L * self.n_head_stacking) // self.L_new
        lma_mask = get_lma_causal_mask(self.L, self.n_head_stacking, self.L_new, device='cpu') # Compute on CPU initially
        if lma_mask is not None:
            # Register as buffer, ensures it moves to GPU with model.to(device)
            # Needs reshaping for MultiheadAttention: (L_new, L_new) -> (1, L_new, L_new) or similar?
            # MHA expects mask shape (N, S, S) or (S,S) - N=Batch size, S=Seq len
            # Let's provide (L_new, L_new) - PyTorch should broadcast it.
            self.register_buffer("causal_mask_latent", lma_mask)
        else:
            # This case should ideally not happen with valid params
            self.register_buffer("causal_mask_latent", None)


    def forward(self, y):
        # Input y: Output of Stage 1 embedding (or LayerNorm) - Shape (B, L, d0)
        B, T, C = y.size() # T should be L, C should be d0
        assert T == self.L, f"Input sequence length {T} does not match LMA configured L={self.L}"
        assert C == self.d0, f"Input embedding dim {C} does not match LMA configured d0={self.d0}"

        # --- Stage 2a: Head-View Stacking ---
        # Split along embedding dim (d0)
        # y shape: (B, L, d0) -> split into n_head_stacking chunks of size d_k along dim 2
        head_views = torch.split(y, self.d_k, dim=2) # List of n_h tensors (B, L, d_k)
        # Stack sequentially along sequence dim (dim=1)
        x_stacked = torch.cat(head_views, dim=1) # Shape: (B, L * n_h_stacking, d_k)

        # --- Stage 2b: Re-Chunking & Latent Embedding ---
        L_prime = self.L * self.n_head_stacking
        assert x_stacked.size(1) == L_prime, "Stacked sequence length mismatch"

        # Flatten features for re-chunking
        x_flat = x_stacked.view(B, -1) # Shape: (B, L * d0)

        # Reshape into new chunks
        # Shape: (B, L_new, C_new)
        x_rechunked = x_flat.view(B, self.L_new, self.C_new)

        # Apply second embedding (TimeDistributed equivalent)
        # Reshape -> Linear -> Reshape
        z_in_shape = x_rechunked.shape # Store shape
        z = x_rechunked.view(-1, self.C_new) # Shape: (B * L_new, C_new)
        z = self.embed_layer_2(z)           # Shape: (B * L_new, d_new)
        z = z.view(z_in_shape[0], z_in_shape[1], self.d_new) # Shape: (B, L_new, d_new)

        # --- Stage 3: Latent Attention Calculation ---
        # QKV projections
        q_prime = self.q_proj(z) # (B, L_new, d_new)
        k_prime = self.k_proj(z) # (B, L_new, d_new)
        v_prime = self.v_proj(z) # (B, L_new, d_new)

        # Perform latent multi-head attention
        # Need causal mask derived for LMA
        # The mask should be (L_new, L_new)
        current_mask = self.causal_mask_latent
        if current_mask is not None:
            # Ensure mask matches query length if needed (though should be fixed L_new)
             if current_mask.size(0) != q_prime.size(1):
                 # This indicates a potential issue during init or variable sequence lengths
                 # If block_size can vary up to a max, the mask needs care
                 # For fixed block_size GPT-2, this check might be redundant if L=block_size
                 print(f"Warning: Mask size {current_mask.size()} mismatch with Q' seq len {q_prime.size(1)}. Using mask slice.")
                 current_mask = current_mask[:q_prime.size(1), :q_prime.size(1)]
        else:
             # This implies L_new was <= 0 during init, should not happen
             print("Error: Latent causal mask is None!")
             # Handle error appropriately, e.g., raise Exception or try proceeding without mask

        # print(f"Debug: Q' shape: {q_prime.shape}, Mask shape: {current_mask.shape if current_mask is not None else 'None'}") # Debug print

        attn_output, _ = self.latent_attn(q_prime, k_prime, v_prime,
                                          attn_mask=current_mask, # Use the derived mask
                                          need_weights=False,
                                          is_causal=False) # Crucial: Our mask handles causality

        # Output projection
        attn_output = self.c_proj(attn_output) # Shape: (B, L_new, d_new)
        attn_output = self.resid_dropout(attn_output) # Apply dropout to output

        # --- Return state *before* residual connection ---
        # The block using this layer will handle the residual connection
        # We need to return attn_output and the tensor needed for the first residual (z)
        return attn_output, z # Return both the attention output and the pre-attention latent state
    

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)