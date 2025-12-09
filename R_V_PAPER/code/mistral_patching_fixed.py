#!/usr/bin/env python3
"""
Fixed activation patching implementation for Mistral-7B
Based on Meng et al. 2022 methodology
"""

import torch
import numpy as np
from contextlib import contextmanager

# Configuration
WINDOW_SIZE = 16  # or whatever you're using
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@contextmanager
def patch_and_capture_v_at_layer(model, layer_idx, source_v, window_size=WINDOW_SIZE):
    """
    Patch V at layer_idx with source_v AND capture the result.
    This correctly implements Meng et al. 2022's activation patching.
    
    Args:
        model: The transformer model
        layer_idx: Which layer to patch at
        source_v: The source V tensor to patch with [seq_len, hidden_dim]
        window_size: How many tokens to patch
    
    Yields:
        storage_dict: Contains both early and patched V tensors
    """
    storage_dict = {"v_early": [], "v_patched": []}
    handles = []
    
    try:
        # Hook for early layer (just capture, no patching)
        early_layer = model.model.layers[5].self_attn
        def early_hook(module, inp, out):
            storage_dict["v_early"].append(out.detach())
            return out
        handles.append(early_layer.v_proj.register_forward_hook(early_hook))
        
        # Hook for target layer (patch THEN capture)
        target_layer = model.model.layers[layer_idx].self_attn
        def patch_hook(module, inp, out):
            # Clone to avoid in-place modification issues
            out_patched = out.clone()
            
            # Get dimensions
            B, T, D = out_patched.shape
            T_src = source_v.shape[0]
            
            # Calculate patch window
            W = min(window_size, T, T_src)
            
            if W > 0:
                # Patch the last W tokens with source
                # Ensure proper device and dtype
                source_patch = source_v[-W:, :].to(out_patched.device, dtype=out_patched.dtype)
                out_patched[:, -W:, :] = source_patch.unsqueeze(0).expand(B, -1, -1)
            
            # Store the patched result
            storage_dict["v_patched"].append(out_patched.detach())
            
            # Return the patched tensor to continue forward pass
            return out_patched
        
        handles.append(target_layer.v_proj.register_forward_hook(patch_hook))
        
        yield storage_dict
        
    finally:
        # Clean up all hooks
        for handle in handles:
            handle.remove()


def run_patched_forward_correct(model, tokenizer, baseline_text, v_recursive_source, layer_idx=21):
    """
    Run baseline prompt with recursive V patched at layer_idx.
    Returns early V and patched V for R_V calculation.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        baseline_text: Text to run through model
        v_recursive_source: V tensor from recursive prompt [seq_len, hidden_dim]
        layer_idx: Where to patch (default 21 for Mistral)
    
    Returns:
        v_early: V from layer 5
        v_patched: V from layer_idx after patching
    """
    # Tokenize
    inputs = tokenizer(
        baseline_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)
    
    # Run with patching
    with torch.no_grad():
        with patch_and_capture_v_at_layer(model, layer_idx, v_recursive_source) as storage:
            _ = model(**inputs)
    
    # Extract results
    v_early = storage["v_early"][0][0] if storage["v_early"] else None  # [seq, hidden]
    v_patched = storage["v_patched"][0][0] if storage["v_patched"] else None  # [seq, hidden]
    
    return v_early, v_patched


def compute_metrics_fast(v_tensor, window_size=WINDOW_SIZE):
    """Your existing metric computation"""
    if v_tensor is None:
        return np.nan, np.nan
    
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]
    
    if v_tensor.dim() != 2:
        return np.nan, np.nan
    
    T, D = v_tensor.shape
    if T < 1:
        return np.nan, np.nan
    
    W = min(window_size, T)
    v_window = v_tensor[-W:, :].float()
    
    try:
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        S_sq = S_np ** 2
        
        if S_sq.sum() < 1e-10:
            return np.nan, np.nan
        
        p = S_sq / S_sq.sum()
        eff_rank = 1.0 / (p**2).sum()
        pr = (S_sq.sum()**2) / (S_sq**2).sum()
        
        return float(eff_rank), float(pr)
        
    except Exception as e:
        print(f"SVD error: {e}")
        return np.nan, np.nan


# ============ COMPLETE EXPERIMENTAL LOOP ============

def run_single_patching_experiment(model, tokenizer, rec_text, base_text, layer_idx=21):
    """
    Run one complete patching experiment with all conditions.
    
    Returns:
        dict with all R_V values and PR values
    """
    results = {}
    
    # 1. Get unpatched recursive V
    inputs_rec = tokenizer(rec_text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    v5_rec = []
    v_target_rec = []
    
    with torch.no_grad():
        # Capture recursive V at both layers
        def capture_early(m, i, o):
            v5_rec.append(o.detach())
            return o
        def capture_target(m, i, o):
            v_target_rec.append(o.detach())
            return o
        
        h1 = model.model.layers[5].self_attn.v_proj.register_forward_hook(capture_early)
        h2 = model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(capture_target)
        
        _ = model(**inputs_rec)
        
        h1.remove()
        h2.remove()
    
    v5_rec = v5_rec[0][0] if v5_rec else None
    v_target_rec = v_target_rec[0][0] if v_target_rec else None
    
    # Calculate recursive metrics
    er5_rec, pr5_rec = compute_metrics_fast(v5_rec)
    er_target_rec, pr_target_rec = compute_metrics_fast(v_target_rec)
    rv_rec = pr_target_rec / pr5_rec if pr5_rec and pr5_rec > 0 else np.nan
    
    results["rv_rec"] = rv_rec
    results["pr5_rec"] = pr5_rec
    results["pr_target_rec"] = pr_target_rec
    
    # 2. Get unpatched baseline V
    inputs_base = tokenizer(base_text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    v5_base = []
    v_target_base = []
    
    with torch.no_grad():
        def capture_early(m, i, o):
            v5_base.append(o.detach())
            return o
        def capture_target(m, i, o):
            v_target_base.append(o.detach())
            return o
        
        h1 = model.model.layers[5].self_attn.v_proj.register_forward_hook(capture_early)
        h2 = model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(capture_target)
        
        _ = model(**inputs_base)
        
        h1.remove()
        h2.remove()
    
    v5_base = v5_base[0][0] if v5_base else None
    v_target_base = v_target_base[0][0] if v_target_base else None
    
    # Calculate baseline metrics
    er5_base, pr5_base = compute_metrics_fast(v5_base)
    er_target_base, pr_target_base = compute_metrics_fast(v_target_base)
    rv_base = pr_target_base / pr5_base if pr5_base and pr5_base > 0 else np.nan
    
    results["rv_base"] = rv_base
    results["pr5_base"] = pr5_base
    results["pr_target_base"] = pr_target_base
    
    # 3. MAIN CONDITION: Patch baseline with recursive V
    v5_patched, v_target_patched = run_patched_forward_correct(
        model, tokenizer, base_text, v_target_rec, layer_idx
    )
    
    er5_patched, pr5_patched = compute_metrics_fast(v5_patched)
    er_target_patched, pr_target_patched = compute_metrics_fast(v_target_patched)
    rv_patched = pr_target_patched / pr5_patched if pr5_patched and pr5_patched > 0 else np.nan
    
    results["rv_patched"] = rv_patched
    results["pr5_patched"] = pr5_patched
    results["pr_target_patched"] = pr_target_patched
    
    # 4. CONTROL 1: Random patch
    random_v = torch.randn_like(v_target_rec)
    # Norm-match as per Hase et al. 2023
    random_v = random_v * (v_target_rec.norm() / random_v.norm())
    
    v5_random, v_target_random = run_patched_forward_correct(
        model, tokenizer, base_text, random_v, layer_idx
    )
    
    er5_random, pr5_random = compute_metrics_fast(v5_random)
    er_target_random, pr_target_random = compute_metrics_fast(v_target_random)
    rv_random = pr_target_random / pr5_random if pr5_random and pr5_random > 0 else np.nan
    
    results["rv_random"] = rv_random
    results["pr_target_random"] = pr_target_random
    
    # 5. CONTROL 2: Shuffled patch
    shuffled_v = v_target_rec.clone()
    # Shuffle along sequence dimension
    perm = torch.randperm(shuffled_v.shape[0])
    shuffled_v = shuffled_v[perm, :]
    
    v5_shuffled, v_target_shuffled = run_patched_forward_correct(
        model, tokenizer, base_text, shuffled_v, layer_idx
    )
    
    er5_shuffled, pr5_shuffled = compute_metrics_fast(v5_shuffled)
    er_target_shuffled, pr_target_shuffled = compute_metrics_fast(v_target_shuffled)
    rv_shuffled = pr_target_shuffled / pr5_shuffled if pr5_shuffled and pr5_shuffled > 0 else np.nan
    
    results["rv_shuffled"] = rv_shuffled
    results["pr_target_shuffled"] = pr_target_shuffled
    
    return results


# ============ EXAMPLE USAGE ============

if __name__ == "__main__":
    # Example of how to use this in your notebook
    print("""
    To use in your notebook:
    
    1. Load this file or copy the functions
    2. For each prompt pair:
    
    results = run_single_patching_experiment(
        model, 
        tokenizer,
        prompt_bank_1c[rec_id]["text"],
        prompt_bank_1c[base_id]["text"],
        layer_idx=21  # or 27
    )
    
    print(f"Recursive R_V: {results['rv_rec']:.3f}")
    print(f"Baseline R_V: {results['rv_base']:.3f}")
    print(f"Patched R_V: {results['rv_patched']:.3f}")
    print(f"Random R_V: {results['rv_random']:.3f}")
    print(f"Shuffled R_V: {results['rv_shuffled']:.3f}")
    
    # The patched R_V should be BETWEEN recursive and baseline
    # not identical to either
    """)
