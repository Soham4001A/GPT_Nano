# compute_static_lma_mask.py
"""
Computes the 'theoretically correct' static causal mask for LMA
based on tracking exact (t, h) origins and applying causality rules.

Run this script once offline for a given LMA configuration
to generate the mask file (.pt).
"""
import torch
import math
import argparse
import os
import time
from dataclasses import dataclass, field # For LMAConfig if defined here

# --- Helper: find_closest_divisor (copied for standalone use) ---
def find_closest_divisor(total_value, target_divisor, max_delta=100):
    """ Finds closest divisor. """
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

# --- LMAConfig (Optional - can just pass parameters directly) ---
# If model.py LMAConfig is complex, you might copy it here,
# otherwise, just use the parameters directly in compute function.

# --- Main Computation Function ---
def compute_perfect_static_mask(L, nH, d0, target_L_new):
    """ Computes the static mask based on LMA parameters. """
    start_time = time.time()
    # 1. Derive LMA dimensions
    if L <= 0 or nH <= 0 or d0 <= 0 or target_L_new <= 0: raise ValueError("Inputs L, nH, d0, target_L_new must be positive.")
    if d0 % nH != 0: raise ValueError(f"d0 ({d0}) must be divisible by nH ({nH})")
    d_k = d0 // nH
    total_features = L * d0
    L_new = find_closest_divisor(total_features, target_L_new)
    if total_features % L_new != 0: raise RuntimeError(f"Logic Error: L*d0 not divisible by calculated L_new {L_new}")
    C_new = total_features // L_new
    L_prime = L * nH # Intermediate stacked length
    print(f"--- Computing mask for Config ---")
    print(f"  L={L}, nH={nH}, d0={d0}")
    print(f"  target_L_new={target_L_new} -> L_new={L_new}")
    print(f"  Derived: d_k={d_k}, C_new={C_new}, L_prime={L_prime}")
    if L_prime == 0 or d_k == 0: raise ValueError("L*nH or d_k is zero, cannot compute origins.")

    # 2. Compute Exact Origin Sets
    print("INFO: Computing exact origin sets...")
    origins_per_latent = [set() for _ in range(L_new)]
    computation_count = 0
    for i_new in range(L_new):
        start_flat_idx = i_new * C_new
        end_flat_idx = start_flat_idx + C_new
        for p_flat_idx in range(start_flat_idx, end_flat_idx):
            p_stack_idx = p_flat_idx // d_k
            if 0 <= p_stack_idx < L_prime:
                 original_t = p_stack_idx % L
                 original_h = p_stack_idx // L
                 origins_per_latent[i_new].add((original_t, original_h))
                 computation_count += 1
    print(f"INFO: Finished computing origin sets (approx {computation_count} mappings).")
    max_set_size = max(len(s) for s in origins_per_latent) if origins_per_latent else 0
    print(f"INFO: Max origin set size: {max_set_size}")


    # 3. Generate Static Causal Mask
    print("INFO: Generating static mask from origin sets...")
    static_mask = torch.zeros((L_new, L_new), dtype=torch.bool)
    comparison_count = 0
    masked_count = 0
    for i in range(L_new): # Query index
         query_origins = origins_per_latent[i]
         if not query_origins: static_mask[i, :] = True; static_mask[:, i] = True; continue
         for j in range(L_new): # Key index
             if i == j: continue # A position can always attend to itself (handled by tril if needed)
             key_origins = origins_per_latent[j]
             if not key_origins: static_mask[i, j] = True; continue

             mask_this_pair = False
             for t_q, h_q in query_origins:
                 for t_k, h_k in key_origins:
                     comparison_count += 1
                     # Check causal rule
                     if h_k > h_q or (h_k == h_q and t_k > t_q):
                         mask_this_pair = True; break
                 if mask_this_pair: break
             if mask_this_pair: static_mask[i, j] = True; masked_count+=1

    print(f"INFO: Finished generating static mask (approx {comparison_count} comparisons, {masked_count} masked entries).")
    end_time = time.time()
    print(f"INFO: Mask computation took {end_time - start_time:.2f} seconds.")
    return static_mask, L_new # Return mask and the actual L_new used

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Static LMA Causal Mask")
    parser.add_argument('--L', type=int, required=True, help="Original sequence length (block_size)")
    parser.add_argument('--nH', type=int, required=True, help="Number of heads for stacking (n_head)")
    parser.add_argument('--d0', type=int, required=True, help="Original embedding dim (n_embd)")
    parser.add_argument('--target_L_new', type=int, required=True, help="Target latent sequence length (e.g., L // reduction_factor)")
    parser.add_argument('--out_dir', type=str, default='lma_masks', help="Directory to save the mask file")
    parser.add_argument('--filename', type=str, default=None, help="Optional specific filename for the mask")
    args = parser.parse_args()

    # Compute the mask
    mask_tensor, final_L_new = compute_perfect_static_mask(args.L, args.nH, args.d0, args.target_L_new)

    # Determine output filename
    if args.filename:
        output_filename = args.filename
    else:
        output_filename = f"lma_static_mask_L{args.L}_nH{args.nH}_d0{args.d0}_Lnew{final_L_new}.pt"

    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)
    output_path = os.path.join(args.out_dir, output_filename)

    # Save the mask
    torch.save(mask_tensor, output_path)
    print(f"Saved mask ({mask_tensor.shape}) to {output_path}")