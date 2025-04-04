# Helper functions (place before model classes or in a separate utils file)
import torch
import numpy as np
import math

def find_closest_divisor(total_value, target_divisor, max_delta=20): # Reduced default max_delta
    """
    Finds a divisor of total_value that is closest to target_divisor (PyTorch/NumPy version).
    Searches outwards from target_divisor up to max_delta away.
    """
    if not isinstance(total_value, int) or total_value <= 0:
        raise ValueError(f"total_value ({total_value}) must be a positive integer.")
    if not isinstance(target_divisor, int) or target_divisor <= 0:
        raise ValueError(f"target_divisor ({target_divisor}) must be positive.")
    if not isinstance(max_delta, int) or max_delta < 0:
        raise ValueError(f"max_delta ({max_delta}) must be non-negative.")

    if total_value % target_divisor == 0:
        return target_divisor

    for delta in range(1, max_delta + 1):
        candidate_minus = target_divisor - delta
        if candidate_minus > 0 and total_value % candidate_minus == 0:
            return candidate_minus
        candidate_plus = target_divisor + delta
        if total_value % candidate_plus == 0:
            return candidate_plus

    raise ValueError(
        f"Could not find a valid divisor for {total_value} near {target_divisor} "
        f"(within +/- {max_delta}). Check L, d0, or target L_new."
    )

def get_lma_causal_mask(L, n_h, L_new, device):
    """
    Calculates the causal mask for the LMA latent attention space (PyTorch version).

    Args:
        L (int): Original sequence length.
        n_h (int): Number of heads used for stacking.
        L_new (int): Target latent sequence length.
        device (torch.device): Device to create the mask tensor on.

    Returns:
        torch.Tensor: An (L_new, L_new) mask tensor where -inf means masked.
                      Returns None if L_new is 0 or less (should not happen).
    """
    if L_new <= 0:
        print(f"Warning: L_new={L_new} is invalid for mask generation. Returning None.")
        return None # Or raise error

    L_prime = L * n_h
    # Check if reduction is clean - this assumes stride/pooling reduction primarily
    if L_prime % L_new != 0:
        print(f"Warning: L*nh ({L_prime}) is not perfectly divisible by L_new ({L_new}). Masking assumes standard reduction (e.g., stride/pool).")
    # Use integer division, floor behavior is fine if not perfectly divisible
    k_stride = L_prime // L_new

    max_orig_index_per_latent_pos = [-1] * L_new

    for i_new in range(L_new):
        p_start = i_new * k_stride
        # Calculate end index carefully, ensure it's inclusive for range()
        # and doesn't exceed L_prime
        p_end = min((i_new + 1) * k_stride, L_prime)

        if p_start >= p_end:
             # This might happen if stride is very large or L_new >= L_prime
             # Assign based on the single index p_start if it's valid
             if p_start < L_prime:
                 max_orig_index_per_latent_pos[i_new] = p_start % L
             else:
                 max_orig_index_per_latent_pos[i_new] = -1 # Or handle error appropriately
             continue

        # Find the max original index (p % L) in this range [p_start, p_end)
        max_orig_l = -1
        for p in range(p_start, p_end): # p_end is exclusive here
            max_orig_l = max(max_orig_l, p % L)
        max_orig_index_per_latent_pos[i_new] = max_orig_l

    # Create the L_new x L_new mask
    mask = torch.zeros((L_new, L_new), device=device, dtype=torch.float32) # Use float for -inf
    for i_new in range(L_new):
        for j_new in range(L_new):
             # Allow attention if key's latest origin is <= query's latest origin
            if max_orig_index_per_latent_pos[j_new] > max_orig_index_per_latent_pos[i_new]:
                mask[i_new, j_new] = float('-inf') # Mask out (block attention)

    return mask # Shape (L_new, L_new)