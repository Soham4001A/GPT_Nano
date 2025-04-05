# ----- model.py -----
"""
Full definition of a GPT Language Model, all of it in this single file.
Incorporates Time-Chunking LMA: Reduces sequence length by chunking time
and stacking features, followed by standard causal attention in the reduced space.
"""

import math
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
# Helper Function: find_closest_divisor (Still potentially useful for LMAConfig logic)
# -----------------------------------------------------------------------------
# (Keep the find_closest_divisor function as defined before)
def find_closest_divisor(total_value, target_divisor, max_delta=100):
    if not isinstance(total_value, int) or total_value <= 0: raise ValueError(f"total_value positive integer.")
    if not isinstance(target_divisor, int) or target_divisor <= 0: target_divisor = max(1, target_divisor)
    if not isinstance(max_delta, int) or max_delta < 0: raise ValueError(f"max_delta non-negative.")
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

@dataclass
class GPTConfig:
    block_size: int = 1024; vocab_size: int = 50304; n_layer: int = 12
    n_head: int = 12; n_embd: int = 768; dropout: float = 0.0
    bias: bool = True
    # --- LMA (Time Chunking) Specific ---
    use_lma: bool = False # If True, uses TimeChunkingTransform + Attention on reduced dims
    lma_reduction_factor: int = 2 # Determines k for L -> L/k length reduction
    # Optional: Specify target d_new, otherwise derived from n_embd / factor
    lma_d_new: int = None # Target latent dimension (optional)
    # lma_mask_path is no longer needed

class LayerNorm(nn.Module):
    """ LayerNorm with optional bias. """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, input):
        expected_dim = self.weight.shape[0]
        if input.size(-1) != expected_dim: raise RuntimeError(f"LayerNorm dim mismatch")
        eps = 1e-5; return F.layer_norm(input, self.weight.shape, self.weight, self.bias, eps)

# LMAConfig might be simplified or removed if not heavily used elsewhere,
# but TimeChunkingTransform needs L, d0, k, d_new.
# Let's keep a simplified config structure or pass params directly.
# For now, let TimeChunkingTransform take params from main GPTConfig.

# --- Time Chunking Transformation Layer ---
class TimeChunkingTransform(nn.Module):
    """
    Reduces sequence length L -> L/k by chunking time and stacking features.
    Maps stacked features (d0*k) -> d_new.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.reduction_factor = config.lma_reduction_factor
        self.L = config.block_size # Original max sequence length
        self.d0 = config.n_embd   # Original embedding dim

        if self.L % self.reduction_factor != 0:
            raise ValueError(f"block_size ({self.L}) must be divisible by lma_reduction_factor ({self.reduction_factor}) for TimeChunking.")
        self.L_new = self.L // self.reduction_factor # Reduced sequence length
        self.d_stacked = self.d0 * self.reduction_factor # Intermediate feature dim

        # Determine target d_new
        if config.lma_d_new is not None and config.lma_d_new > 0:
             self.d_new = config.lma_d_new
        else: # Derive d_new from d0 and factor
             target_d_new = self.d0 // self.reduction_factor
             self.d_new = max(1, target_d_new) # Ensure positive
        # Ensure d_new is divisible by n_head for subsequent attention
        if self.d_new % config.n_head != 0:
            original_d_new = self.d_new
            self.d_new = max(config.n_head, (self.d_new // config.n_head) * config.n_head)
            if self.d_new == 0: self.d_new = config.n_head # Fallback
            print(f"TimeChunking LMA: Adjusted d_new from {original_d_new} to {self.d_new} to be divisible by n_head ({config.n_head})")

        print(f" Init TimeChunkingTransform: In(L={self.L}, d0={self.d0}) -> Stack(k={self.reduction_factor}, L_new={self.L_new}, d_stack={self.d_stacked}) -> Out(L_new={self.L_new}, d_new={self.d_new})")

        self.proj = nn.Linear(self.d_stacked, self.d_new, bias=config.bias)
        self.act = nn.GELU()

    def forward(self, x): # Input x is (B, T, d0)
        B, T, C = x.size()
        if C != self.d0: raise ValueError(f"TimeChunkingTransform C ({C}) != d0 ({self.d0})")

        # Handle sequences shorter than L by padding
        padded_x = x
        if T < self.L:
            padded_x = F.pad(x, (0, 0, 0, self.L - T)) # Pad C, then T dim
        elif T > self.L: # Should be handled upstream, but truncate just in case
            padded_x = x[:, -self.L:, :]
        assert padded_x.size(1) == self.L

        # Time Chunking and Stacking Features
        # Reshape to (B, k, L_new, d0)
        x_chunked = padded_x.view(B, self.reduction_factor, self.L_new, self.d0)
        # Permute to (B, L_new, k, d0)
        x_permuted = x_chunked.permute(0, 2, 1, 3)
        # Reshape to (B, L_new, k * d0)
        x_time_stacked = x_permuted.contiguous().view(B, self.L_new, self.d_stacked)

        # Project to d_new
        z_proj = self.proj(x_time_stacked.view(-1, self.d_stacked)) # Flatten B, L_new for Linear
        z = self.act(z_proj)
        z = z.view(B, self.L_new, self.d_new) # Reshape back

        # Return latent state z with reduced length L_new and dim d_new
        return z

# --- Latent Attention becomes Standard CausalSelfAttention operating on d_new ---
# We can reuse CausalSelfAttention, but need to adapt its internal dimensions OR create a new class.
# Let's create a new class for clarity.

class LatentCausalSelfAttention(nn.Module):
    """ Standard MHA implementation adapted for latent dimension d_new. """
    def __init__(self, config: GPTConfig, d_new: int, n_head_latent: int, L_new: int):
        super().__init__()
        assert d_new % n_head_latent == 0
        self.d_new = d_new
        self.n_head = n_head_latent
        self.L_new = L_new # Max sequence length for mask buffer

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.d_new, 3 * self.d_new, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(self.d_new, self.d_new, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # causal mask to ensure that attention is only applied to the left in the input sequence
        # Use standard triangular mask for length L_new
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if self.flash:
             print(f"  Initializing LatentCausalSelfAttention: Flash Attention enabled. d_new={d_new}, nH={n_head_latent}")
             self.register_buffer("bias", None, persistent=False)
        else:
             print(f"  Initializing LatentCausalSelfAttention: Using slow attention. d_new={d_new}, nH={n_head_latent}")
             mask = torch.tril(torch.ones(self.L_new, self.L_new))
             self.register_buffer("bias", mask.view(1, 1, self.L_new, self.L_new), persistent=False)

    def forward(self, x): # Input x: (B, T_latent, d_new)
        B, T, C = x.size() # T is current T_latent <= L_new
        if C != self.d_new: raise ValueError(f"LatentCausalSelfAttention C ({C}) != d_new ({self.d_new})")

        q, k, v  = self.c_attn(x).split(self.d_new, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Use flash attention if possible
        if self.flash and self.dropout == 0.0:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True)
        else:
             if self.flash and hasattr(F, 'scaled_dot_product_attention'):
                  # print("Warning: Latent Flash Attention with dropout.") # Reduce noise
                  y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
             else: # Manual implementation
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                if self.bias is None: raise RuntimeError("Slow attention needs bias buffer")
                # Slice standard causal mask
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble heads
        y = self.resid_dropout(self.c_proj(y)) # Output projection
        return y

# --- LMA_Decoder (Still needed for length upsampling) ---
class LMA_Decoder(nn.Module):
    """ Maps latent (B, L_new, d_new) back to (B, T, d_output=d_new) """
    def __init__(self, config: GPTConfig, L_new: int, d_new: int): # Takes L_new, d_new directly
        super().__init__(); self.config = config;
        self.L_new = L_new; self.d_new = d_new; self.d_output = d_new # Output dim matches input dim
        self.bias = config.bias; self.dropout = config.dropout
        print(f" Init LMA Decoder: In(L_new={self.L_new}, d_new={self.d_new}) -> Out(T, d_output={self.d_output})")
        self.ln = LayerNorm(self.d_new, bias=self.bias); hidden_dim = self.d_new * 2 # Standard MLP sizing
        self.fc1 = nn.Linear(self.d_new, hidden_dim, bias=self.bias); self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, self.d_output, bias=self.bias); self.drop = nn.Dropout(self.dropout)
    def forward(self, z, target_T): # z shape: (B, T_latent, d_new), T_latent <= L_new
        B, current_L_new, current_d_new = z.shape
        if current_L_new > self.L_new : print(f"Warning: Decoder L_new > config L_new"); z = z[:,:self.L_new,:] # Truncate
        if current_d_new != self.d_new: raise ValueError(f"LMA_Decoder d mismatch")
        # Check for NaNs before potentially problematic interpolation
        if torch.isnan(z).any() or torch.isinf(z).any(): print("NaN/Inf DETECTED in LMA_Decoder input!"); return torch.zeros(B, target_T, self.d_output, device=z.device, dtype=z.dtype)
        # Check if input length is zero before interpolating
        if current_L_new == 0: print("Warning: LMA_Decoder received zero-length input."); return torch.zeros(B, target_T, self.d_output, device=z.device, dtype=z.dtype)
        z_permuted = z.permute(0, 2, 1)
        try: z_interpolated = F.interpolate(z_permuted, size=target_T, mode='linear', align_corners=False)
        except Exception as e: print(f"ERROR during interpolate: {e}"); raise e
        z_upsampled = z_interpolated.permute(0, 2, 1)
        # Check after interpolation
        if torch.isnan(z_upsampled).any() or torch.isinf(z_upsampled).any(): print("NaN/Inf DETECTED AFTER interpolation!"); return torch.zeros_like(z_upsampled)
        z_norm = self.ln(z_upsampled); z_hidden = self.act(self.fc1(z_norm)); z_refined = self.fc2(z_hidden)
        z_output = self.drop(z_refined)
        if self.d_output == self.d_new: z_output = z_upsampled + z_output # Residual connection
        # Final check
        if torch.isnan(z_output).any() or torch.isinf(z_output).any(): print("NaN/Inf DETECTED in LMA_Decoder output!"); return torch.zeros_like(z_output)
        return z_output

# --- CausalSelfAttention (Standard MHA - Keep unchanged) ---
class CausalSelfAttention(nn.Module):
    # (Keep the previous standard CausalSelfAttention code exactly as it was)
    def __init__(self, config):
        super().__init__(); assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout); self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head; self.n_embd = config.n_embd; self.dropout = config.dropout
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if self.flash: print("Using Flash Attention (if available/applicable)."); self.register_buffer("bias", None, persistent=False)
        else: print("WARNING: using slow attention."); mask = torch.tril(torch.ones(config.block_size, config.block_size)); self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size), persistent=False)
    def forward(self, x):
        B, T, C = x.size();
        if C != self.n_embd: raise ValueError(f"CausalSelfAttention C mismatch")
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash and self.dropout == 0.0 : y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True)
        else:
            if self.flash and hasattr(F, 'scaled_dot_product_attention'):
                 # print("Warning: Using Flash Attention with dropout/non-zero dropout_p.") # Reduce noise
                 y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                if self.bias is None: raise RuntimeError("Slow attention requires bias buffer")
                slice_T = min(T, self.bias.size(-1))
                att = att.masked_fill(self.bias[:,:,:slice_T,:slice_T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1); att = self.attn_dropout(att); y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C); y = self.resid_dropout(self.c_proj(y)); return y


# --- MLP (Keep unchanged) ---
class MLP(nn.Module):
    def __init__(self, config, block_internal_dim):
        super().__init__(); self.input_dim = block_internal_dim; hidden_dim = 4 * self.input_dim
        self.c_fc = nn.Linear(self.input_dim, hidden_dim, bias=config.bias); self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden_dim, self.input_dim, bias=config.bias); self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        if x.size(-1) != self.input_dim: raise ValueError(f"MLP input dim mismatch")
        x = self.c_fc(x); x = self.gelu(x); x = self.c_proj(x); x = self.dropout(x); return x

# --- Block (Modified for Time Chunking LMA) ---
class Block(nn.Module):
    """ Transformer Block: Uses MHA or TimeChunkingLMA """
    def __init__(self, config: GPTConfig, is_lma: bool, lma_latent_dim: int = None, lma_latent_len: int = None):
        # lma_latent_dim = d_new, lma_latent_len = L_new
        super().__init__()
        self.use_lma = is_lma
        if self.use_lma:
            if lma_latent_dim is None or lma_latent_len is None:
                raise ValueError("lma_latent_dim (d_new) and lma_latent_len (L_new) required for LMA Block")
            self.operating_dim = lma_latent_dim
            self.operating_L = lma_latent_len # Max sequence length this block sees
            print(f"Initializing Block (TimeChunking LMA): Dim={self.operating_dim}, MaxL={self.operating_L}")
            # Use LatentCausalSelfAttention operating on d_new
            self.attn = LatentCausalSelfAttention(config, d_new=self.operating_dim, n_head_latent=config.n_head, L_new=self.operating_L)
        else: # Standard MHA
            self.operating_dim = config.n_embd
            self.operating_L = config.block_size
            print(f"Initializing Block (MHA): Dim={self.operating_dim}, MaxL={self.operating_L}")
            self.attn = CausalSelfAttention(config)

        self.ln_1 = LayerNorm(self.operating_dim, bias=config.bias)
        self.mlp = MLP(config, self.operating_dim)
        self.ln_2 = LayerNorm(self.operating_dim, bias=config.bias)

    def forward(self, x): # Input x: (B, T_current, self.operating_dim) - NO pos_tags
        B, T_current, C_current = x.shape
        if C_current != self.operating_dim: raise ValueError(f"Block C mismatch")
        # NaN checks on input - maybe overkill but safe for debug
        if torch.isnan(x).any(): raise ValueError(f"NaN in Block {id(self)} input!")

        x_input_residual1 = x
        x_norm1 = self.ln_1(x)
        if torch.isnan(x_norm1).any(): raise ValueError(f"NaN after ln_1 in Block {id(self)}")

        attn_output = self.attn(x_norm1) # Attention (either MHA or Latent MHA)
        if attn_output is None: raise ValueError(f"attn_output is None") # Should not happen
        if torch.isnan(attn_output).any(): raise ValueError(f"NaN in attn_output")

        x = x_input_residual1 + attn_output
        if torch.isnan(x).any(): raise ValueError(f"NaN after residual add 1")

        x_input_residual2 = x
        x_norm2 = self.ln_2(x)
        if torch.isnan(x_norm2).any(): raise ValueError(f"NaN after ln_2")

        mlp_output = self.mlp(x_norm2)
        if mlp_output is None: raise ValueError(f"mlp_output is None")
        if torch.isnan(mlp_output).any(): raise ValueError(f"NaN in mlp_output")
        if x_input_residual2.shape != mlp_output.shape: raise ValueError(f"Shape mismatch res2")

        x = x_input_residual2 + mlp_output
        if torch.isnan(x).any(): raise ValueError(f"NaN after residual add 2")

        return x # Return only data tensor

# --- GPT (Modified for Time Chunking LMA) ---
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__(); assert config.vocab_size is not None; assert config.block_size is not None
        self.config = config
        self.time_chunk_transform = None # Renamed from initial_lma_transform
        self.lma_decoder = None
        self.latent_dim = config.n_embd # Dim entering blocks (changes if LMA)
        self.latent_len = config.block_size # Length entering blocks (changes if LMA)

        if config.use_lma:
             # Validation for Time Chunking
             if config.block_size % config.lma_reduction_factor != 0:
                  raise ValueError(f"LMA TimeChunking requires block_size ({config.block_size}) to be divisible by lma_reduction_factor ({config.lma_reduction_factor})")
             if config.n_embd % config.n_head != 0: # Still need for head dimension calc within attention
                  raise ValueError(f"n_embd ({config.n_embd}) not divisible by n_head ({config.n_head}).")
             if config.lma_reduction_factor <= 0: raise ValueError("LMA reduction factor must be > 0.")

             print("--- Configuring Time Chunking LMA ---")
             # Instantiate the new transform
             self.time_chunk_transform = TimeChunkingTransform(config)
             self.latent_dim = self.time_chunk_transform.d_new # Blocks operate on d_new
             self.latent_len = self.time_chunk_transform.L_new # Blocks operate on L_new
             print(f"--- Dimensions into Blocks: Latent L={self.latent_len}, Latent D={self.latent_dim} ---")

             # Decoder setup
             print("--- Configuring LMA Decoder ---")
             # Pass L_new and d_new directly to decoder
             self.lma_decoder = LMA_Decoder(config, self.latent_len, self.latent_dim)
             # Final layers operate on decoder output dim (which is d_new)
             self.final_ln_lm_head_dim = self.lma_decoder.d_output
             print(f"--- Dimension after Decoder: {self.final_ln_lm_head_dim} ---")
        else:
             # Standard MHA path
             self.latent_dim = config.n_embd
             self.latent_len = config.block_size
             self.final_ln_lm_head_dim = config.n_embd


        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
        ))

        print(f"--- Building {config.n_layer} Transformer Blocks ---"); blocks = []
        for i in range(config.n_layer):
            # If LMA, pass latent dim and len, else pass None
            lma_block_dim = self.latent_dim if config.use_lma else None
            lma_block_len = self.latent_len if config.use_lma else None
            block = Block(config, is_lma=config.use_lma,
                          lma_latent_dim=lma_block_dim,
                          lma_latent_len=lma_block_len)
            blocks.append(block)
        self.transformer['h'] = nn.ModuleList(blocks)

        self.transformer['ln_f'] = LayerNorm(self.final_ln_lm_head_dim, bias=config.bias)
        self.lm_head = nn.Linear(self.final_ln_lm_head_dim, config.vocab_size, bias=False);
        print(f"--- Final LN & LM Head on dim: {self.final_ln_lm_head_dim} ---")

        # Weight tying check
        print(f"DEBUG: Weight tying check: FinalDim={self.final_ln_lm_head_dim}, n_embd={self.config.n_embd}, use_lma={config.use_lma}")
        if not config.use_lma and self.final_ln_lm_head_dim == config.n_embd: self.transformer.wte.weight = self.lm_head.weight; print("Weight tying enabled.")
        else: reason = "LMA is used" if config.use_lma else f"dim mismatch"; print(f"Weight tying disabled ({reason}).")

        self.apply(self._init_weights);
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'): torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    # --- GPT forward (Simplified - No pos_tags flow) ---
    def forward(self, idx, targets=None):
        device = idx.device; b, t = idx.size()
        if t > self.config.block_size: idx = idx[:, -self.config.block_size:]; t = self.config.block_size;
        if targets is not None and targets.shape[1] > self.config.block_size: targets = targets[:, -self.config.block_size:]
        pos = torch.arange(0, t, dtype=torch.long, device=device); tok_emb = self.transformer.wte(idx); pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb); original_T = t

        # LMA Initial Transform returns only z now
        if self.initial_lma_transform is not None:
            x = self.initial_lma_transform(x) # x is now latent z (B, L_new, d_new)

        # Blocks take only x and return only x
        for block in self.transformer.h:
            x = block(x) # Block forward signature is simplified

        # Decoder takes latent x
        if self.lma_decoder is not None:
            x = self.lma_decoder(x, original_T)

        x = self.transformer.ln_f(x); logits = self.lm_head(x); loss = None
        if targets is not None: loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def get_num_params(self, non_embedding=True):
        n_params=sum(p.numel() for p in self.parameters())
        if non_embedding: n_params -= self.transformer.wpe.weight.numel(); return n_params
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: # Check bias ONLY for Linear
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # NO BIAS CHECK HERE FOR EMBEDDING
        elif isinstance(module, LayerNorm):
            if module.bias is not None: # Check bias ONLY for LayerNorm
                torch.nn.init.zeros_(module.bias)
    def crop_block_size(self, block_size): raise NotImplementedError("LMA block size cropping not fully supported.")
    @classmethod
    def from_pretrained(cls, model_type, override_args=None): raise NotImplementedError("Loading pretrained LMA models not supported.")
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}; decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]; nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [{'params': decay_params, 'weight_decay': weight_decay}, {'params': nodecay_params, 'weight_decay': 0.0}]; num_decay_params = sum(p.numel() for p in decay_params); num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"); print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters; use_fused = fused_available and device_type.startswith('cuda'); extra_args = dict(fused=True) if use_fused else dict(); optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args); print(f"using fused AdamW: {use_fused}"); return optimizer
    def estimate_mfu(self, fwdbwd_per_iter, dt): N=self.get_num_params(); cfg=self.config; L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size; flops_per_token=6*N+12*L*H*Q*T; flops_per_fwdbwd=flops_per_token*T; flops_per_iter=flops_per_fwdbwd*fwdbwd_per_iter; flops_achieved=flops_per_iter*(1.0/dt); flops_promised=312e12; mfu=flops_achieved/flops_promised; print("WARNING: MFU estimate inaccurate for LMA."); return mfu*0.8
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval();
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]; logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None: v, _ = torch.topk(logits, min(top_k, logits.size(-1))); logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1); idx_next = torch.multinomial(probs, num_samples=1); idx = torch.cat((idx, idx_next), dim=1)
        self.train(); return idx

# --- HellaSwag evaluation logic (Keep unchanged) ---
def get_most_likely_row(tokens, mask, logits):
    if logits.shape[1] <= 1: print(f"Warning (get_most_likely_row): Logits seq len <= 1."); return 0
    shift_logits = logits[..., :-1, :].contiguous(); shift_tokens = tokens[..., 1:].contiguous()
    if shift_logits.shape[1] == 0 or shift_tokens.shape[1] == 0: print(f"Warning: shift_logits or shift_tokens empty."); return 0
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1)); flat_shift_tokens = shift_tokens.view(-1)
    if flat_shift_logits.shape[0] != flat_shift_tokens.shape[0]: print(f"ERROR (get_most_likely_row): Size mismatch!"); return 0
    try: shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    except Exception as e: print(f"ERROR during CE in get_most_likely_row: {e}"); return 0
    shift_losses = shift_losses.view(tokens.size(0), -1); shift_mask = mask[..., 1:].contiguous()
    masked_shift_losses = shift_losses * shift_mask; sum_loss = masked_shift_losses.sum(dim=1); num_loss_tokens = shift_mask.sum(dim=1)
    avg_loss = sum_loss / (num_loss_tokens + 1e-6); avg_loss[num_loss_tokens == 0] = float('inf'); pred_norm = avg_loss.argmin().item(); return pred_norm

@torch.no_grad()
def evaluate_hellaswag(model, enc, hellaswag_path='data/hellaswag/hellaswag_val.jsonl'):
    assert isinstance(enc, Encoding), "Encoder `enc` must be tiktoken Encoding"
    print(f"Evaluating HellaSwag from {hellaswag_path}..."); num_correct_norm = 0; num_total = 0
    if not os.path.exists(hellaswag_path):
        print(f"Error: HS file not found: {hellaswag_path}"); data_dir = os.path.dirname(hellaswag_path)
        if not os.path.exists(data_dir): os.makedirs(data_dir); val_url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
        print(f"Attempting download from {val_url}..."); import requests
        try:
            with requests.get(val_url, stream=True) as r: r.raise_for_status();
            with open(hellaswag_path, 'wb') as f: [f.write(chunk) for chunk in r.iter_content(chunk_size=8192)]
            print("Download successful.")
        except Exception as e: print(f"Download failed: {e}. Cannot eval."); return -1.0
    model_device = next(model.parameters()).device; device_type = 'cuda' if 'cuda' in str(model_device) else 'cpu'
    eval_ctx = nullcontext();
    if device_type == 'cuda': print("DEBUG: Using torch.float32 context for HellaSwag model call."); eval_dtype = torch.float32; eval_ctx = torch.amp.autocast(device_type=device_type, dtype=eval_dtype)
    try:
        with open(hellaswag_path, 'r') as f:
            for line in tqdm.tqdm(f, desc="HellaSwag Eval"):
                example = json.loads(line); num_total += 1; ctx = example['ctx']; label = example['label']; endings = example['endings']
                ctx_tokens = enc.encode(ctx);
                if not ctx_tokens: print(f"Warning: Skipping empty context."); num_total -=1; continue
                tok_rows = []; mask_rows = []
                for end in endings:
                    completion_tokens = enc.encode(end); tok = ctx_tokens + completion_tokens; mask = [0]*len(ctx_tokens) + [1]*len(completion_tokens)
                    if len(tok) > model.config.block_size:
                        num_comp = len(completion_tokens); max_ctx = model.config.block_size - num_comp
                        if max_ctx < 0: completion_tokens=completion_tokens[:model.config.block_size]; tok=completion_tokens; mask=[1]*len(tok); max_ctx=0
                        start_idx = max(0, len(ctx_tokens)-max_ctx); trunc_ctx = ctx_tokens[start_idx:]
                        tok = trunc_ctx + completion_tokens; mask = [0]*len(trunc_ctx) + [1]*len(completion_tokens)
                        if len(tok) > model.config.block_size: tok = tok[-model.config.block_size:]; mask = mask[-model.config.block_size:]
                    if len(tok) <= 1: tok=tok+[0]*(2-len(tok)); mask=mask+[0]*(2-len(mask))
                    tok=tok[:model.config.block_size]; mask=mask[:model.config.block_size]
                    tok_rows.append(torch.tensor(tok,dtype=torch.long)); mask_rows.append(torch.tensor(mask,dtype=torch.long))
                if not tok_rows: continue
                max_len = max(len(r) for r in tok_rows)
                if max_len <= 1: max_len = 2
                tokens=torch.zeros((len(tok_rows),max_len),dtype=torch.long); mask_t=torch.zeros((len(tok_rows),max_len),dtype=torch.long) # Renamed mask
                for i, (tr, mr) in enumerate(zip(tok_rows, mask_rows)): tokens[i,:len(tr)]=tr; mask_t[i,:len(mr)]=mr # Use mask_t
                tokens=tokens.to(model_device); mask_t=mask_t.to(model_device) # Use mask_t
                model.eval();
                with eval_ctx: logits, _ = model(tokens)
                if torch.isnan(logits).any() or torch.isinf(logits).any(): print(f"ERROR: NaNs/Infs in HS logits!"); pred_norm = 0
                else:
                    try: pred_norm = get_most_likely_row(tokens, mask_t, logits) # Pass mask_t
                    except Exception as e: print(f"ERROR in get_most_likely_row: {e}"); import traceback; traceback.print_exc(); pred_norm = 0
                if pred_norm == label: num_correct_norm += 1
    except FileNotFoundError: print(f"Error: HS file not found: {hellaswag_path}"); return -1.0
    except Exception as e: print(f"Error during HS eval loop: {e}"); import traceback; traceback.print_exc(); return -1.0
    acc_norm = num_correct_norm / num_total if num_total > 0 else 0.0; print(f"HellaSwag Accuracy: {acc_norm*100:.2f}% ({num_correct_norm}/{num_total})"); return acc_norm

# --- Example Usage __main__ block ---
if __name__ == '__main__':
    # --- Example Config (Small TimeChunking LMA) ---
    config_args = dict(
        block_size=128, vocab_size=50257, n_layer=4, n_head=4, n_embd=128,
        dropout=0.1, bias=True,
        use_lma=True,               # Enable Time Chunking LMA
        lma_reduction_factor=2,   # k=2 => L_new = 128/2 = 64
        lma_d_new=64                # Optional: Explicitly set d_new
        # lma_mask_path=None # No longer needed
    )
    # --- OR USE GPT-2 Config (Large TimeChunking LMA) ---
    # config_args = dict(
    #     block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768,
    #     dropout=0.0, bias=True,
    #     use_lma=True,
    #     lma_reduction_factor=4, # Example: k=4 => L_new = 1024/4 = 256
    #     lma_d_new=None            # Example: Let d_new be derived (768/4=192, adjusted to 192 if n_head=12)
    # )

    # --- Initialize ---
    gpt_config = GPTConfig(**config_args)
    print("\n--- Model Configuration ---"); print(gpt_config)
    print("\n--- Initializing Model ---"); model = GPT(gpt_config)

    # --- Testing code (remains the same) ---
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
            print("Forward pass successful!"); print(f"  Logits shape: {logits.shape}") # Should be (B, T_current, Vocab)
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