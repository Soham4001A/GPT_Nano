# ----- model.py -----
"""
Full definition of a GPT Language Model, all of it in this single file.
Incorporates Latent Meta Attention (LMA) with Dynamic Causality Masking
based on propagated Min/Max original position tags. Includes NaN/Inf checks.
"""

import math
import inspect
from dataclasses import dataclass, field
import os
import tqdm, json
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from contextlib import nullcontext
from tiktoken.core import Encoding

# -----------------------------------------------------------------------------
# Helper Functions for LMA
# -----------------------------------------------------------------------------

def find_closest_divisor(total_value, target_divisor, max_delta=100):
    """
    Finds a divisor of total_value that is closest to target_divisor.
    """
    if not isinstance(total_value, int) or total_value <= 0: raise ValueError(f"total_value ({total_value}) must be positive integer.")
    if not isinstance(target_divisor, int) or target_divisor <= 0: target_divisor = max(1, target_divisor)
    if not isinstance(max_delta, int) or max_delta < 0: raise ValueError(f"max_delta ({max_delta}) must be non-negative.")
    if total_value == 0: return 1
    if target_divisor > 0 and total_value % target_divisor == 0: return target_divisor
    search_start = max(1, target_divisor)
    for delta in range(1, max_delta + 1):
        candidate_minus = search_start - delta
        if candidate_minus > 0 and total_value % candidate_minus == 0: return candidate_minus
        candidate_plus = search_start + delta
        if candidate_plus > 0 and total_value % candidate_plus == 0: return candidate_plus
    for i in range(1, int(math.sqrt(total_value)) + 1):
        if total_value % i == 0: print(f"Warning: No divisor found near {target_divisor}. Using {i} as fallback."); return i
    if total_value > 1: print(f"Warning: No divisor found near {target_divisor}. Using {total_value} as fallback."); return total_value
    raise ValueError(f"Could not find any valid divisor for {total_value} near {target_divisor}.")

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
             raise RuntimeError(f"LayerNorm input dim ({input.size(-1)}) mismatch weight dim ({expected_dim}) Input shape: {input.shape}")
        # Use a slightly larger epsilon for potentially more stability? Default 1e-5
        eps = 1e-5
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, eps)

# --- GPTConfig ---
@dataclass
class GPTConfig:
    block_size: int = 1024; vocab_size: int = 50304; n_layer: int = 12
    n_head: int = 12; n_embd: int = 768; dropout: float = 0.0
    bias: bool = True; use_lma: bool = False; lma_reduction_factor: int = 2
    
@dataclass
class LMAConfig:
    """ Configuration specific to the LMA layer internals. """
    d0: int
    L: int
    n_head_stacking: int
    target_L_new: int
    d_new: int
    n_head_latent: int
    L_new: int = field(init=False)
    C_new: int = field(init=False)

    def __post_init__(self):
        if self.L <= 0 or self.d0 <= 0 or self.n_head_stacking <= 0 or \
           self.target_L_new <= 0 or self.d_new <= 0 or self.n_head_latent <= 0:
            raise ValueError("All LMAConfig inputs must be positive.")
        if self.d0 % self.n_head_stacking != 0: raise ValueError(f"LMA Config Error: d0 ({self.d0}) not divisible by n_head_stacking ({self.n_head_stacking}).")
        if self.d_new % self.n_head_latent != 0: raise ValueError(f"LMA Config Error: d_new ({self.d_new}) not divisible by n_head_latent ({self.n_head_latent}).")
        total_features = self.L * self.d0
        if total_features == 0: raise ValueError("LMAConfig total features (L*d0) cannot be zero.")
        try:
            self.L_new = find_closest_divisor(total_features, self.target_L_new)
            if self.L_new != self.target_L_new: print(f"LMAConfig ADJUSTMENT: Target L_new ({self.target_L_new}) changed to {self.L_new}.")
            if self.L_new <= 0: raise ValueError(f"Calculated L_new ({self.L_new}) is not positive.")
            if total_features % self.L_new != 0: raise RuntimeError(f"Internal Error: total_features ({total_features}) not divisible by calculated L_new ({self.L_new})")
            self.C_new = total_features // self.L_new
            if self.C_new <= 0: raise ValueError(f"Calculated C_new ({self.C_new}) is not positive.")
        except ValueError as e: raise ValueError(f"LMA Config Error calculating L_new/C_new: {e}") from e


# --- LMA Initial Transformation Layer (Modified for Position Tag Propagation) ---
class LMA_InitialTransform(nn.Module):
    """
    Performs Stage 1 and Stage 2 of LMA to map (B,T,d0) -> (z, pos_tags)
    where z is (B, L_new, d_new) and pos_tags is (B, L_new, 2) [min_t, max_t]
    """
    def __init__(self, config, lma_config: LMAConfig):
        super().__init__()
        self.config = config
        self.lma_config = lma_config
        if lma_config.d0 <= 0 or lma_config.n_head_stacking <= 0: raise ValueError("LMA Initial Transform: d0/n_head_stacking must be positive.")
        if lma_config.d0 % lma_config.n_head_stacking != 0: raise ValueError(f"LMA Initial Transform: d0 ({lma_config.d0}) not divisible by n_head_stacking ({lma_config.n_head_stacking}).")
        self.d0 = lma_config.d0; self.L = lma_config.L; self.n_head_stacking = lma_config.n_head_stacking
        self.d_k = self.d0 // self.n_head_stacking; self.d_new = lma_config.d_new
        self.L_new = lma_config.L_new; self.C_new = lma_config.C_new; self.bias = config.bias
        print(f" Init LMA InitialTransform: In(Max L={self.L}, d0={self.d0}, nH_stack={self.n_head_stacking}) -> Out(L_new={self.L_new}, d_new={self.d_new}) + PosTags")
        print(f"   Intermediate calculated: d_k={self.d_k}, C_new={self.C_new}")
        self.embed_layer_2 = nn.Linear(self.C_new, self.d_new, bias=self.bias)
        self.embed_layer_2_act = nn.GELU()

    def forward(self, y): # Input y is (B, T, d0)
        B, T, C = y.size()
        if C != self.d0: raise ValueError(f"LMA InitialTransform C({C}) != d0({self.d0})")
        device = y.device

        padded_y = y
        if T < self.L: padded_y = F.pad(y, (0, 0, 0, self.L - T))
        elif T > self.L: print(f"Warning: LMA InitialTransform T={T} > L={self.L}. Truncating."); padded_y = y[:, -self.L:, :]
        assert padded_y.size(1) == self.L

        try: head_views = torch.split(padded_y, self.d_k, dim=2)
        except RuntimeError as e: raise RuntimeError(f"Error splitting heads: d0={self.d0}, d_k={self.d_k}. Input shape={padded_y.shape}") from e
        x_stacked = torch.cat(head_views, dim=1)
        x_flat = x_stacked.view(B, -1)
        if x_flat.shape[1] != self.L_new * self.C_new: raise RuntimeError(f"Config mismatch: L*d0 != L_new*C_new.")
        x_rechunked = x_flat.view(B, self.L_new, self.C_new)
        z_embedded_flat = self.embed_layer_2(x_rechunked.view(-1, self.C_new))
        z_activated = self.embed_layer_2_act(z_embedded_flat)
        z = z_activated.view(B, self.L_new, self.d_new)

        pos_indices = torch.arange(T, device=device, dtype=torch.long).view(1, T, 1).expand(B, T, 1)
        padded_pos = pos_indices
        if T < self.L: padded_pos = F.pad(pos_indices, (0, 0, 0, self.L - T), value=-1)
        elif T > self.L: padded_pos = pos_indices[:, -self.L:, :]
        assert padded_pos.size(1) == self.L
        pos_stacked = padded_pos.repeat(1, self.n_head_stacking, 1)
        pos_flat = pos_stacked.view(B, -1)
        L_nH = self.L * self.n_head_stacking
        C_pos = -1; target_len_pos = -1
        if L_nH == 0 or self.L_new == 0 :
            print(f"Warning: L_nH={L_nH} or L_new={self.L_new} is zero. Pos tags invalid.");
            pos_rechunked = torch.full((B, self.L_new, 1), -1, dtype=torch.long, device=device)
            C_pos = 1
        elif L_nH % self.L_new == 0:
            C_pos = L_nH // self.L_new; target_len_pos = L_nH
            pos_rechunked = pos_flat.view(B, self.L_new, C_pos)
        else:
            print(f"Warning: L*nH ({L_nH}) not divisible by L_new ({self.L_new}). Padding positions.")
            C_pos_float = L_nH / self.L_new; C_pos = math.ceil(C_pos_float)
            target_len_pos = self.L_new * C_pos; padding_size_pos = target_len_pos - L_nH
            pos_flat_padded = F.pad(pos_flat, (0, padding_size_pos), value=-1)
            pos_rechunked = pos_flat_padded.view(B, self.L_new, C_pos)
            if pos_rechunked.shape[1] * pos_rechunked.shape[2] != target_len_pos: raise RuntimeError("Pos rechunk shape mismatch.")

        min_val_replace = T; max_val_replace = -1
        pos_for_min = torch.where(pos_rechunked == -1, min_val_replace, pos_rechunked)
        pos_for_max = torch.where(pos_rechunked == -1, max_val_replace, pos_rechunked)
        min_t_per_latent = torch.min(pos_for_min, dim=2)[0]
        max_t_per_latent = torch.max(pos_for_max, dim=2)[0]
        min_t_per_latent = torch.where(min_t_per_latent == min_val_replace, -1, min_t_per_latent)
        max_t_per_latent = torch.where(max_t_per_latent == max_val_replace, -1, max_t_per_latent)
        pos_tags = torch.stack([min_t_per_latent, max_t_per_latent], dim=2)
        return z, pos_tags


# --- Latent Attention (Rewritten for Manual MHA & Dynamic Masking w/ NaN checks) ---
class LatentMetaAttention(nn.Module):
    """
    LMA Core Logic - Manual MHA with Dynamic Masking and NaN/Inf checks.
    """
    def __init__(self, config, lma_latent_config: LMAConfig):
        super().__init__()
        self.config = config; self.lma_config = lma_latent_config
        self.d_latent = lma_latent_config.d_new; self.L_latent = lma_latent_config.L_new
        self.n_head_latent = lma_latent_config.n_head_latent; self.bias = config.bias
        self.dropout_rate = config.dropout
        if not (self.d_latent > 0 and self.n_head_latent > 0 and self.d_latent % self.n_head_latent == 0):
             raise ValueError(f"Invalid latent attention params: d={self.d_latent}, nH={self.n_head_latent}")
        self.head_dim = self.d_latent // self.n_head_latent
        print(f"  Initializing LatentMetaAttention (Manual MHA, Dynamic Mask): Latent(L={self.L_latent}, d={self.d_latent}), Heads={self.n_head_latent}")
        self.q_proj = nn.Linear(self.d_latent, self.d_latent, bias=self.bias)
        self.k_proj = nn.Linear(self.d_latent, self.d_latent, bias=self.bias)
        self.v_proj = nn.Linear(self.d_latent, self.d_latent, bias=self.bias)
        self.c_proj = nn.Linear(self.d_latent, self.d_latent, bias=self.bias)
        self.attn_dropout = nn.Dropout(self.dropout_rate)
        self.resid_dropout = nn.Dropout(self.dropout_rate)
        self.register_buffer("causal_mask_latent", None, persistent=False) # Not used

    def forward(self, z, pos_tags): # Input z:(B, T_latent, d_latent), pos_tags:(B, T_latent, 2)
        B, T_latent, C_latent = z.size()
        if torch.isnan(z).any() or torch.isinf(z).any(): print("NaN/Inf DETECTED in input z!"); return torch.zeros_like(z) # Return zeros if input unstable
        if C_latent != self.d_latent: raise ValueError(f"LatentAttention C ({C_latent}) != d_latent ({self.d_latent})")
        if pos_tags is None: raise ValueError("pos_tags required for dynamic masking.")
        if pos_tags.shape[:2] != (B, T_latent) or pos_tags.shape[2] != 2: raise ValueError(f"pos_tags shape mismatch. Expected ({B}, {T_latent}, 2), got {pos_tags.shape}")
        if T_latent > self.L_latent: print(f"Warning: T_latent ({T_latent}) > max L_latent ({self.L_latent}). Truncating."); z=z[:,:self.L_latent,:]; pos_tags=pos_tags[:,:self.L_latent,:]; T_latent=self.L_latent

        # --- Dynamic Mask Calculation ---
        min_t = pos_tags[:, :, 0]; max_t = pos_tags[:, :, 1]
        query_max_t = max_t.unsqueeze(2); key_min_t = min_t.unsqueeze(1)
        dynamic_mask = key_min_t > query_max_t # True where Key starts after Query ends
        query_pad_mask = (query_max_t == -1); key_pad_mask = (key_min_t == -1)
        dynamic_mask = dynamic_mask | query_pad_mask | key_pad_mask # Also mask if query or key is padding

        # --- Manual Multi-Head Attention ---
        q = self.q_proj(z); k = self.k_proj(z); v = self.v_proj(z)
        if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any(): print("NaN DETECTED in Q, K, or V!"); return torch.zeros_like(z)

        q = q.view(B, T_latent, self.n_head_latent, self.head_dim).transpose(1, 2)
        k = k.view(B, T_latent, self.n_head_latent, self.head_dim).transpose(1, 2)
        v = v.view(B, T_latent, self.n_head_latent, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if torch.isnan(attn_scores).any() or torch.isinf(attn_scores).any(): print("NaN/Inf DETECTED in attn_scores BEFORE mask!"); return torch.zeros_like(z)

        attn_scores = attn_scores.masked_fill(dynamic_mask.unsqueeze(1), float('-inf'))
        all_masked_rows = torch.all(attn_scores == float('-inf'), dim=-1)
        
        query_pad_mask_expanded = query_pad_mask.unsqueeze(1).expand(-1, self.n_head_latent, -1, -1) # (B, nH, T_latent, 1)
        query_pad_mask_final = query_pad_mask_expanded.squeeze(-1).reshape(B * self.n_head_latent, T_latent)

        fully_masked_non_padding = all_masked_rows & (~query_pad_mask_final)
        if torch.any(fully_masked_non_padding):
            print(f"WARNING: {torch.sum(fully_masked_non_padding)} NON-PADDING attention rows are fully masked!")
            
        if torch.isnan(attn_scores).any() or (torch.isinf(attn_scores) & (attn_scores != float('-inf'))).any(): print("NaN/Inf DETECTED in attn_scores AFTER mask!"); return torch.zeros_like(z)

        # Safeguarded Softmax
        attn_scores_safe = torch.where(attn_scores == float('-inf'), torch.finfo(attn_scores.dtype).min, attn_scores)
        attn_probs = F.softmax(attn_scores_safe, dim=-1)
        # Zero out probs where the input row was all -inf (softmax should technically handle this, but belt-and-suspenders)
        attn_probs = torch.where(all_masked_rows.unsqueeze(-1), torch.zeros_like(attn_probs), attn_probs)

        if torch.isnan(attn_probs).any(): print("NaN DETECTED in attn_probs AFTER softmax!"); return torch.zeros_like(z)
        attn_probs = self.attn_dropout(attn_probs)

        y = torch.matmul(attn_probs, v)
        if torch.isnan(y).any() or torch.isinf(y).any(): print("NaN/Inf DETECTED in y AFTER attn @ v!"); return torch.zeros_like(z)

        y = y.transpose(1, 2).contiguous().view(B, T_latent, self.d_latent)
        y_proj = self.c_proj(y)
        if torch.isnan(y_proj).any() or torch.isinf(y_proj).any(): print("NaN/Inf DETECTED in y_proj AFTER c_proj!"); return torch.zeros_like(z)
        y = self.resid_dropout(y_proj)
        if torch.isnan(y).any() or torch.isinf(y).any(): print("NaN/Inf DETECTED in y AFTER resid_dropout!"); return torch.zeros_like(z)

        return y


# --- Learnable Decoder for LMA ---
class LMA_Decoder(nn.Module):
    """ Learns to map latent sequence (B, L_new, d_new) back to (B, T, d_output) """
    def __init__(self, config: GPTConfig, lma_config: LMAConfig):
        super().__init__()
        self.config = config; self.lma_config = lma_config
        self.L_new = lma_config.L_new; self.d_new = lma_config.d_new
        self.d_output = self.d_new; self.bias = config.bias; self.dropout = config.dropout
        print(f" Init LMA Decoder: In(L_new={self.L_new}, d_new={self.d_new}) -> Out(T, d_output={self.d_output})")
        self.ln = LayerNorm(self.d_new, bias=self.bias)
        hidden_dim = self.d_new * 2
        self.fc1 = nn.Linear(self.d_new, hidden_dim, bias=self.bias)
        self.act = nn.GELU(); self.fc2 = nn.Linear(hidden_dim, self.d_output, bias=self.bias)
        self.drop = nn.Dropout(self.dropout)

    def forward(self, z, target_T): # z shape: (B, L_new, d_new)
        B, current_L_new, current_d_new = z.shape
        if current_L_new != self.L_new: print(f"Warning: LMA_Decoder L_new={current_L_new}, expected {self.L_new}.")
        if current_d_new != self.d_new: raise ValueError(f"LMA_Decoder d ({current_d_new}) != d_new ({self.d_new})")
        if torch.isnan(z).any() or torch.isinf(z).any(): print("NaN/Inf DETECTED in LMA_Decoder input z!"); return torch.zeros(B, target_T, self.d_output, device=z.device, dtype=z.dtype)

        z_permuted = z.permute(0, 2, 1)
        try:
             z_interpolated = F.interpolate(z_permuted, size=target_T, mode='linear', align_corners=False)
        except Exception as e:
             print(f"ERROR during LMA_Decoder F.interpolate: {e}"); print(f"Input shape: {z_permuted.shape}, target_T: {target_T}"); raise e
        z_upsampled = z_interpolated.permute(0, 2, 1)
        if torch.isnan(z_upsampled).any() or torch.isinf(z_upsampled).any(): print("NaN/Inf DETECTED AFTER interpolation!"); return torch.zeros_like(z_upsampled)

        z_norm = self.ln(z_upsampled); z_hidden = self.act(self.fc1(z_norm)); z_refined = self.fc2(z_hidden)
        z_output = self.drop(z_refined)
        if self.d_output == self.d_new: z_output = z_upsampled + z_output # Add residual
        if torch.isnan(z_output).any() or torch.isinf(z_output).any(): print("NaN/Inf DETECTED in LMA_Decoder output!"); return torch.zeros_like(z_output)
        return z_output


# --- Standard Causal Self Attention ---
class CausalSelfAttention(nn.Module):
    """ Standard MHA implementation """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout); self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head; self.n_embd = config.n_embd; self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0
        if not self.flash: print("WARNING: using slow attention."); mask = torch.tril(torch.ones(config.block_size, config.block_size)); self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size), persistent=False)
        else: print("Using Flash Attention."); self.register_buffer("bias", None, persistent=False)

    def forward(self, x):
        B, T, C = x.size()
        if C != self.n_embd: raise ValueError(f"CausalSelfAttention C({C}) != n_embd({self.n_embd})")
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash: y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.bias is None: raise RuntimeError("Slow attention requires bias buffer")
            slice_T = min(T, self.bias.size(-1))
            att = att.masked_fill(self.bias[:,:,:slice_T,:slice_T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1); att = self.attn_dropout(att); y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y)); return y


# --- MLP ---
class MLP(nn.Module):
    def __init__(self, config, block_internal_dim):
        super().__init__()
        self.input_dim = block_internal_dim; hidden_dim = 4 * self.input_dim
        self.c_fc = nn.Linear(self.input_dim, hidden_dim, bias=config.bias)
        self.gelu = nn.GELU(); self.c_proj = nn.Linear(hidden_dim, self.input_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        if x.size(-1) != self.input_dim: raise ValueError(f"MLP input dim {x.size(-1)} != expected {self.input_dim}")
        x = self.c_fc(x); x = self.gelu(x); x = self.c_proj(x); x = self.dropout(x); return x


# --- Block (Modified to handle pos_tags) ---
class Block(nn.Module):
    """ Transformer Block: Modified to handle pos_tags for LMA """
    def __init__(self, config: GPTConfig, is_lma: bool, lma_config: LMAConfig = None):
        super().__init__()
        self.use_lma = is_lma
        if self.use_lma:
            if lma_config is None: raise ValueError("lma_config needed for LMA Block")
            self.operating_dim = lma_config.d_new; self.operating_L = lma_config.L_new
            print(f"Initializing Block {id(self)} (LMA): Operates on dim={self.operating_dim}, max_L={self.operating_L}")
            self.attn = LatentMetaAttention(config, lma_config)
        else:
            self.operating_dim = config.n_embd; self.operating_L = config.block_size
            print(f"Initializing Block {id(self)} (MHA): Operates on dim={self.operating_dim}, max_L={self.operating_L}")
            self.attn = CausalSelfAttention(config)
        self.ln_1 = LayerNorm(self.operating_dim, bias=config.bias)
        self.mlp = MLP(config, self.operating_dim)
        self.ln_2 = LayerNorm(self.operating_dim, bias=config.bias)

    def forward(self, x, pos_tags=None): # Accept optional pos_tags
        # Input x: (B, T_current, self.operating_dim)
        # Input pos_tags: (B, T_current, 2) if LMA, else None
        B, T_current, C_current = x.shape
        if C_current != self.operating_dim:
             block_type = "LMA" if self.use_lma else "MHA"
             raise ValueError(f"Block ({block_type}) C ({C_current}) != operating_dim ({self.operating_dim})")

        x_norm1 = self.ln_1(x)
        if self.use_lma:
            if pos_tags is None: raise ValueError("LMA Block requires pos_tags.")
            attn_output = self.attn(x_norm1, pos_tags) # Pass pos_tags
        else:
            attn_output = self.attn(x_norm1) # MHA doesn't need pos_tags
        x = x + attn_output

        x_norm2 = self.ln_2(x)
        mlp_output = self.mlp(x_norm2)
        x = x + mlp_output

        # Return data and pass pos_tags through unmodified
        return x, pos_tags


# --- Main GPT Model (Modified for pos_tags flow) ---
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None; assert config.block_size is not None
        self.config = config
        if config.use_lma:
             if config.n_embd % config.n_head != 0: raise ValueError(f"LMA n_embd ({config.n_embd}) not divisible by n_head ({config.n_head}).")
             if config.lma_reduction_factor <= 0: raise ValueError(f"LMA reduction_factor ({config.lma_reduction_factor}) > 0.")

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
        ))

        self.initial_lma_transform = None; self.lma_decoder = None; self.initial_lma_cfg = None
        current_d = config.n_embd; current_L = config.block_size; self.operates_in_latent = False

        if config.use_lma:
            print("--- Configuring LMA: Initial Transformation ---")
            self.operates_in_latent = True
            reduction_factor = max(1, config.lma_reduction_factor)
            target_l_new_init = config.block_size // reduction_factor
            target_d_new_init = config.n_embd // reduction_factor
            target_l_new_init = max(1, target_l_new_init); target_d_new_init = max(1, target_d_new_init)
            latent_n_head = config.n_head
            if target_d_new_init == 0: raise ValueError("Initial LMA target_d_new is zero.")
            if target_d_new_init % latent_n_head != 0:
                original_target_d = target_d_new_init
                target_d_new_init = max(latent_n_head, (target_d_new_init // latent_n_head) * latent_n_head)
                if target_d_new_init == 0: target_d_new_init = latent_n_head
                print(f"LMA Init: Adjusted target d_new from {original_target_d} to {target_d_new_init}")
            try:
                self.initial_lma_cfg = LMAConfig(
                    d0=config.n_embd, L=config.block_size, n_head_stacking=config.n_head,
                    target_L_new=target_l_new_init, d_new=target_d_new_init, n_head_latent=latent_n_head
                )
                self.initial_lma_transform = LMA_InitialTransform(config, self.initial_lma_cfg)
                current_L = self.initial_lma_cfg.L_new; current_d = self.initial_lma_cfg.d_new
                print(f"--- Dimensions into Blocks: Latent L={current_L}, Latent D={current_d} ---")
            except ValueError as e: print(f"ERROR configuring Initial LMA Transform: {e}"); raise e

        print(f"--- Building {config.n_layer} Transformer Blocks ---")
        blocks = []
        for i in range(config.n_layer):
            is_lma_block = self.operates_in_latent
            block_lma_config = None
            if is_lma_block:
                if self.initial_lma_cfg is None: raise RuntimeError("LMA config error.")
                block_lma_config = LMAConfig(
                     d0=current_d, L=current_L, n_head_stacking=config.n_head,
                     target_L_new=current_L, d_new=current_d, n_head_latent=self.initial_lma_cfg.n_head_latent
                 )
                block = Block(config, is_lma=True, lma_config=block_lma_config)
            else: block = Block(config, is_lma=False)
            blocks.append(block)
        self.transformer['h'] = nn.ModuleList(blocks)

        self.final_ln_lm_head_dim = current_d
        if self.operates_in_latent:
            if self.initial_lma_cfg is None: raise RuntimeError("LMA config error.")
            print("--- Configuring LMA: Decoder ---")
            self.lma_decoder = LMA_Decoder(config, self.initial_lma_cfg)
            self.final_ln_lm_head_dim = self.lma_decoder.d_output
            print(f"--- Dimension after Decoder: {self.final_ln_lm_head_dim} ---")

        self.transformer['ln_f'] = LayerNorm(self.final_ln_lm_head_dim, bias=config.bias)
        self.lm_head = nn.Linear(self.final_ln_lm_head_dim, config.vocab_size, bias=False)
        print(f"--- Final LN & LM Head operating on dimension: {self.final_ln_lm_head_dim} ---")

        print(f"DEBUG: Checking weight tying. Final Dim = {self.final_ln_lm_head_dim}, n_embd = {self.config.n_embd}, use_lma = {config.use_lma}")
        if not config.use_lma and self.final_ln_lm_head_dim == config.n_embd:
            self.transformer.wte.weight = self.lm_head.weight; print("Weight tying enabled.")
        else:
             reason = "LMA is used" if config.use_lma else f"final dim {self.final_ln_lm_head_dim} != n_embd {config.n_embd}"
             print(f"Weight tying disabled ({reason}).")
        print(f"DEBUG: Are lm_head/wte weights same object? {self.lm_head.weight is self.transformer.wte.weight}")

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'): torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters());
        if non_embedding: n_params -= self.transformer.wpe.weight.numel(); return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Initialize Linear weight
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Initialize Linear bias ONLY if it exists
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Initialize Embedding weight
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Embeddings DO NOT have a bias, so no check needed here
        elif isinstance(module, LayerNorm):
            # LayerNorm weight is initialized to ones in its constructor
            # Initialize LayerNorm bias ONLY if it exists
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        device = idx.device; b, t = idx.size()
        if t > self.config.block_size: idx = idx[:, -self.config.block_size:]; t = self.config.block_size;
        if targets is not None and targets.shape[1] > self.config.block_size: targets = targets[:, -self.config.block_size:]

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx); pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb); original_T = t; pos_tags = None

        if self.initial_lma_transform is not None:
            x, pos_tags = self.initial_lma_transform(x) # Now returns tuple

        for block in self.transformer.h:
            # Block forward now returns (output, pos_tags)
            x, pos_tags = block(x, pos_tags=pos_tags)

        if self.lma_decoder is not None:
            x = self.lma_decoder(x, original_T) # Decoder only needs data x

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # Calculate full logits

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        # Check for NaNs in final output before returning (optional safety)
        # if torch.isnan(logits).any(): print("NaN DETECTED in final logits!")
        # if loss is not None and torch.isnan(loss): print("NaN DETECTED in final loss!")

        return logits, loss

    def crop_block_size(self, block_size): raise NotImplementedError("LMA block size cropping not fully supported yet.")
    @classmethod
    def from_pretrained(cls, model_type, override_args=None): raise NotImplementedError("Loading pretrained models not supported yet.")

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [{'params': decay_params, 'weight_decay': weight_decay}, {'params': nodecay_params, 'weight_decay': 0.0}]
        num_decay_params = sum(p.numel() for p in decay_params); num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type.startswith('cuda')
        extra_args = dict(fused=True) if use_fused else dict(); optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}"); return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params(); cfg = self.config; L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T; flops_per_fwdbwd = flops_per_token * T; flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt); flops_promised = 312e12; mfu = flops_achieved / flops_promised
        print("WARNING: MFU estimate based on standard MHA, may be inaccurate for LMA."); adjustment_factor = 0.8; return mfu * adjustment_factor

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond) # Gets full logits (B, T, V)
            logits = logits[:, -1, :] / temperature # Select last token's logits
            if top_k is not None: v, _ = torch.topk(logits, min(top_k, logits.size(-1))); logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1); idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train(); return idx

# -----------------------------------------------------------------------------
# HellaSwag evaluation logic (Uses self-contained context)
# -----------------------------------------------------------------------------

def get_most_likely_row(tokens, mask, logits):
    # logits: (B, T, V), tokens: (B, T)
    if logits.shape[1] <= 1: print(f"Warning (get_most_likely_row): Logits seq len ({logits.shape[1]}) <= 1."); return 0
    shift_logits = logits[..., :-1, :].contiguous() # (B, T-1, V)
    shift_tokens = tokens[..., 1:].contiguous()     # (B, T-1)
    if shift_logits.shape[1] == 0 or shift_tokens.shape[1] == 0: print(f"Warning: shift_logits or shift_tokens empty."); return 0
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    if flat_shift_logits.shape[0] != flat_shift_tokens.shape[0]: print(f"ERROR (get_most_likely_row): Size mismatch! Logits flat: {flat_shift_logits.shape[0]}, Tokens flat: {flat_shift_tokens.shape[0]}"); return 0
    try:
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    except Exception as e:
        print(f"ERROR during cross_entropy in get_most_likely_row: {e}")
        # Potentially print shapes again or problematic values
        # print(f"Shapes: flat_shift_logits={flat_shift_logits.shape}, flat_shift_tokens={flat_shift_tokens.shape}")
        # print(f"Sample logits: {flat_shift_logits[:5]}")
        # print(f"Sample tokens: {flat_shift_tokens[:5]}")
        return 0 # Return default on error
    shift_losses = shift_losses.view(tokens.size(0), -1) # (B, T-1)
    shift_mask = mask[..., 1:].contiguous() # (B, T-1)
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1); num_loss_tokens = shift_mask.sum(dim=1)
    avg_loss = sum_loss / (num_loss_tokens + 1e-6); avg_loss[num_loss_tokens == 0] = float('inf')
    pred_norm = avg_loss.argmin().item(); return pred_norm

@torch.no_grad()
def evaluate_hellaswag(model, enc, hellaswag_path='data/hellaswag/hellaswag_val.jsonl'):
    """ Runs HellaSwag evaluation. Determines its own autocast context. """
    assert isinstance(enc, Encoding), "Encoder `enc` must be tiktoken Encoding"
    print(f"Evaluating HellaSwag from {hellaswag_path}...")
    num_correct_norm = 0; num_total = 0
    if not os.path.exists(hellaswag_path):
        print(f"Error: HS file not found: {hellaswag_path}"); data_dir = os.path.dirname(hellaswag_path)
        if not os.path.exists(data_dir): os.makedirs(data_dir)
        val_url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
        print(f"Attempting download from {val_url}..."); import requests
        try:
            with requests.get(val_url, stream=True) as r: r.raise_for_status()
            with open(hellaswag_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            print("Download successful.")
        except Exception as e: print(f"Download failed: {e}. Cannot eval."); return -1.0
    model_device = next(model.parameters()).device
    device_type = 'cuda' if 'cuda' in str(model_device) else 'cpu'
    eval_ctx = nullcontext()
    if device_type == 'cuda':
        print("DEBUG: Using torch.float32 context for HellaSwag model call.")
        eval_dtype = torch.float32; eval_ctx = torch.amp.autocast(device_type=device_type, dtype=eval_dtype)
    try:
        with open(hellaswag_path, 'r') as f:
            for line in tqdm.tqdm(f, desc="HellaSwag Eval"):
                example = json.loads(line); num_total += 1
                ctx = example['ctx']; label = example['label']; endings = example['endings']
                ctx_tokens = enc.encode(ctx)
                if not ctx_tokens: print(f"Warning: Skipping empty context."); num_total -=1; continue
                tok_rows = []; mask_rows = []
                for end in endings:
                    completion_tokens = enc.encode(end)
                    tok = ctx_tokens + completion_tokens
                    mask = [0]*len(ctx_tokens) + [1]*len(completion_tokens)
                    if len(tok) > model.config.block_size:
                        num_completion_tokens = len(completion_tokens)
                        max_ctx_len = model.config.block_size - num_completion_tokens
                        if max_ctx_len < 0: completion_tokens=completion_tokens[:model.config.block_size]; tok=completion_tokens; mask=[1]*len(tok); max_ctx_len=0
                        start_index = max(0, len(ctx_tokens) - max_ctx_len); truncated_ctx_tokens = ctx_tokens[start_index:]
                        tok = truncated_ctx_tokens + completion_tokens; mask = [0]*len(truncated_ctx_tokens) + [1]*len(completion_tokens)
                        if len(tok) > model.config.block_size: tok = tok[-model.config.block_size:]; mask = mask[-model.config.block_size:]
                    if len(tok) <= 1: tok = tok + [0] * (2 - len(tok)); mask = mask + [0] * (2 - len(mask))
                    tok = tok[:model.config.block_size]; mask = mask[:model.config.block_size]
                    tok_rows.append(torch.tensor(tok, dtype=torch.long)); mask_rows.append(torch.tensor(mask, dtype=torch.long))
                if not tok_rows: continue
                max_len = max(len(row) for row in tok_rows)
                if max_len <= 1: max_len = 2
                tokens = torch.zeros((len(tok_rows), max_len), dtype=torch.long); mask = torch.zeros((len(tok_rows), max_len), dtype=torch.long)
                for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
                    current_len = len(tok_row); tokens[i, :current_len] = tok_row; mask[i, :current_len] = mask_row
                tokens = tokens.to(model_device); mask = mask.to(model_device)
                model.eval()
                with eval_ctx: logits, _ = model(tokens)
                if torch.isnan(logits).any() or torch.isinf(logits).any(): print(f"ERROR: NaNs/Infs detected in HellaSwag logits!"); pred_norm = 0
                else:
                    try: pred_norm = get_most_likely_row(tokens, mask, logits)
                    except Exception as e: print(f"ERROR during get_most_likely_row: {e}"); import traceback; traceback.print_exc(); pred_norm = 0 # Catch broader errors
                if pred_norm == label: num_correct_norm += 1
    except FileNotFoundError: print(f"Error: HS file not found: {hellaswag_path}"); return -1.0
    except Exception as e: print(f"Error during HellaSwag eval loop: {e}"); import traceback; traceback.print_exc(); return -1.0
    acc_norm = num_correct_norm / num_total if num_total > 0 else 0.0
    print(f"HellaSwag Accuracy: {acc_norm*100:.2f}% ({num_correct_norm}/{num_total})"); return acc_norm

# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    config_args = dict( block_size=128, vocab_size=50257, n_layer=4, n_head=4, n_embd=128, dropout=0.1, bias=True, use_lma=True, lma_reduction_factor=2,)
    gpt_config = GPTConfig(**config_args)
    print("\n--- Model Configuration ---"); print(gpt_config)
    print("\n--- Initializing Model ---"); model = GPT(gpt_config)
    print("\n--- Testing Forward/Backward Pass ---")
    B = 4; T = gpt_config.block_size; T_short = T // 2
    dummy_input_full = torch.randint(0, gpt_config.vocab_size, (B, T)); dummy_targets_full = torch.randint(0, gpt_config.vocab_size, (B, T))
    dummy_input_short = torch.randint(0, gpt_config.vocab_size, (B, T_short)); dummy_targets_short = torch.randint(0, gpt_config.vocab_size, (B, T_short))
    if torch.cuda.is_available(): device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built(): device = 'mps' if "RANK" not in os.environ else 'cpu'
    else: device = 'cpu'
    print(f"Using device: {device}"); model.to(device)
    optimizer = model.configure_optimizers(weight_decay=1e-1, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device)
    for seq_len_label, dummy_input, dummy_targets in [ (f"T = {T}", dummy_input_full, dummy_targets_full), (f"T = {T_short}", dummy_input_short, dummy_targets_short) ]:
        print(f"\nTesting with {seq_len_label}...")
        dummy_input = dummy_input.to(device); dummy_targets = dummy_targets.to(device)
        try:
            model.train(); optimizer.zero_grad()
            logits, loss = model(dummy_input, dummy_targets)
            print("Forward pass successful!"); print(f"  Logits shape: {logits.shape}")
            if loss is not None: print(f"  Loss: {loss.item()}"); loss.backward(); optimizer.step(); print("Backward pass successful!")
            else: print("  Loss is None.")
        except Exception as e: print(f"\n !!! Error FWD/BWD ({seq_len_label}) !!!"); print(e); import traceback; traceback.print_exc()
    print("\n--- Testing Generation ---")
    try:
        start_ids = torch.randint(0, gpt_config.vocab_size, (1, 10), device=device); model.eval()
        generated_ids = model.generate(start_ids, max_new_tokens=20, temperature=0.8, top_k=5)
        print("Generation successful!"); print(f"  Input shape: {start_ids.shape}"); print(f"  Generated shape: {generated_ids.shape}")
    except Exception as e: print("\n !!! Error Generation !!!"); print(e); import traceback; traceback.print_exc()
    print("\n--- Testing HellaSwag ---")
    try:
        import tiktoken; enc = tiktoken.get_encoding("gpt2")
        hs_path = os.path.join('data', 'hellaswag', 'hellaswag_val.jsonl')
        accuracy = evaluate_hellaswag(model, enc, hs_path)
        print(f"HellaSwag eval finished. Accuracy: {accuracy:.4f}")
    except ImportError: print("tiktoken not installed, skipping HellaSwag.")
    except Exception as e: print(f"\n !!! Error HellaSwag !!!"); print(e); import traceback; traceback.print_exc()