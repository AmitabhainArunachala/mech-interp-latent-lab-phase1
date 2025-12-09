#!/usr/bin/env python3
"""
EXACT replication of Mixtral methodology for Mistral-7B
Key: Patch and measure at the SAME layer, compute R_V as ratio
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Configuration - matching Mixtral exactly
TARGET_LAYER = 27  # Try 27 first (same as Mixtral), can adjust based on sweep
EARLY_LAYER = 5
WINDOW_SIZE = 16  # Same as Mixtral
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute_metrics_fast(v_tensor, window_size=WINDOW_SIZE):
    """Compute PR and effective rank for V tensor"""
    if v_tensor is None:
        return np.nan, np.nan
    
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]
    
    T, D = v_tensor.shape
    W = min(window_size, T)
    
    if W < 2:
        return np.nan, np.nan
    
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
    except Exception:
        return np.nan, np.nan

def run_single_forward_get_V(model, tokenizer, text, capture_layers):
    """Get V tensors at specified layers"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    captured = {}
    handles = []
    
    with torch.no_grad():
        for layer_idx in capture_layers:
            storage = []
            
            def make_hook(storage_list, idx):
                def hook_fn(m, i, o):
                    storage_list.append(o.detach())
                    return o
                return hook_fn
            
            layer = model.model.layers[layer_idx].self_attn
            h = layer.v_proj.register_forward_hook(make_hook(storage, layer_idx))
            handles.append(h)
            captured[layer_idx] = storage
        
        _ = model(**inputs)
        
        for h in handles:
            h.remove()
    
    result = {}
    for layer_idx in capture_layers:
        if captured[layer_idx]:
            result[layer_idx] = captured[layer_idx][0][0]  # [seq, hidden]
        else:
            result[layer_idx] = None
    
    return result

def run_patched_forward_MIXTRAL_STYLE(model, tokenizer, baseline_text, rec_v_source, target_layer=TARGET_LAYER):
    """
    EXACT Mixtral methodology:
    - Patch V at target_layer with recursive values
    - Measure at the SAME layer (target_layer)
    - Return V at early_layer and target_layer for R_V calculation
    """
    inputs = tokenizer(baseline_text, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
    
    v_early_list = []
    v_target_list = []
    
    with torch.no_grad():
        # Hook for early layer (just capture)
        def capture_early(m, i, o):
            v_early_list.append(o.detach())
            return o
        
        # Hook for target layer (patch AND capture)
        def patch_and_capture(m, i, o):
            # Clone to avoid in-place modification
            out = o.clone()
            B, T, D = out.shape
            
            # Get source dimensions
            src = rec_v_source.to(out.device, dtype=out.dtype)
            T_src = src.shape[0]
            
            # Patch last WINDOW_SIZE positions
            W = min(WINDOW_SIZE, T, T_src)
            if W > 0:
                out[:, -W:, :] = src[-W:, :].unsqueeze(0).expand(B, -1, -1)
            
            # Capture the PATCHED result
            v_target_list.append(out.detach())
            
            # Return patched for downstream (though we don't measure downstream)
            return out
        
        h_early = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(capture_early)
        h_target = model.model.layers[target_layer].self_attn.v_proj.register_forward_hook(patch_and_capture)
        
        _ = model(**inputs)
        
        h_early.remove()
        h_target.remove()
    
    v_early = v_early_list[0][0] if v_early_list else None
    v_target = v_target_list[0][0] if v_target_list else None
    
    return v_early, v_target

def test_mixtral_methodology(model, tokenizer, prompt_bank, num_pairs=5):
    """
    Test using EXACT Mixtral methodology
    Start with n=5 like the original Mixtral experiment
    """
    print("="*70)
    print("ACTIVATION PATCHING - EXACT MIXTRAL METHODOLOGY")
    print("="*70)
    print(f"Target layer: {TARGET_LAYER} (patch AND measure here)")
    print(f"Early layer:  {EARLY_LAYER}")
    print(f"Window size:  {WINDOW_SIZE}")
    print(f"Testing {num_pairs} pairs (like Mixtral n=5)")
    print("="*70)
    
    # Get prompt pairs
    pairs = []
    for i in range(1, num_pairs + 1):
        rec_id = f"L5_refined_{i:02d}"
        base_id = f"factual_new_{i:02d}"
        
        # Check if these IDs exist
        rec_candidates = [k for k in prompt_bank.keys() if k.startswith(f"L5_refined_{i:02d}")]
        base_candidates = [k for k in prompt_bank.keys() if k.startswith(f"factual_new_{i:02d}")]
        
        if not rec_candidates:
            rec_candidates = [k for k in prompt_bank.keys() if "L5_refined" in k]
        if not base_candidates:
            base_candidates = [k for k in prompt_bank.keys() if "factual" in k and "new" in k]
        
        if rec_candidates and base_candidates:
            pairs.append((rec_candidates[0], base_candidates[0]))
    
    if not pairs:
        # Fallback: just get any L5 and factual pairs
        rec_ids = [k for k, v in prompt_bank.items() if v["group"] == "L5_refined"][:num_pairs]
        base_ids = [k for k, v in prompt_bank.items() if "factual" in v["group"]][:num_pairs]
        pairs = list(zip(rec_ids, base_ids))
    
    print(f"\nProcessing {len(pairs)} pairs...")
    print()
    
    results = []
    
    for idx, (rec_id, base_id) in enumerate(pairs):
        rec_text = prompt_bank[rec_id]["text"]
        base_text = prompt_bank[base_id]["text"]
        
        print(f"Pair {idx+1}: {rec_id[:20]}... → {base_id[:20]}...")
        
        # Skip if too short
        if len(tokenizer.encode(base_text)) < WINDOW_SIZE:
            print("  Skipping: baseline too short")
            continue
        
        try:
            # 1. Get recursive V values at both layers
            rec_vs = run_single_forward_get_V(model, tokenizer, rec_text, [EARLY_LAYER, TARGET_LAYER])
            v5_rec = rec_vs[EARLY_LAYER]
            v_target_rec = rec_vs[TARGET_LAYER]
            
            # 2. Get baseline V values at both layers
            base_vs = run_single_forward_get_V(model, tokenizer, base_text, [EARLY_LAYER, TARGET_LAYER])
            v5_base = base_vs[EARLY_LAYER]
            v_target_base = base_vs[TARGET_LAYER]
            
            # 3. Run patched: inject recursive V at target layer
            v5_patched, v_target_patched = run_patched_forward_MIXTRAL_STYLE(
                model, tokenizer, base_text, v_target_rec, TARGET_LAYER
            )
            
            # Compute metrics
            er5_rec, pr5_rec = compute_metrics_fast(v5_rec)
            er_target_rec, pr_target_rec = compute_metrics_fast(v_target_rec)
            
            er5_base, pr5_base = compute_metrics_fast(v5_base)
            er_target_base, pr_target_base = compute_metrics_fast(v_target_base)
            
            er5_patched, pr5_patched = compute_metrics_fast(v5_patched)
            er_target_patched, pr_target_patched = compute_metrics_fast(v_target_patched)
            
            # Calculate R_V ratios
            rv_rec = pr_target_rec / pr5_rec if pr5_rec > 0 else np.nan
            rv_base = pr_target_base / pr5_base if pr5_base > 0 else np.nan
            rv_patched = pr_target_patched / pr5_patched if pr5_patched > 0 else np.nan
            
            results.append({
                'rec_id': rec_id,
                'base_id': base_id,
                f'RV{TARGET_LAYER}_rec': rv_rec,
                f'RV{TARGET_LAYER}_base': rv_base,
                f'RV{TARGET_LAYER}_patched': rv_patched,
                f'er{TARGET_LAYER}_rec': er_target_rec,
                f'er{TARGET_LAYER}_base': er_target_base,
                f'er{TARGET_LAYER}_patched': er_target_patched,
                'delta': rv_patched - rv_base
            })
            
            print(f"  Recursive R_V:  {rv_rec:.3f}")
            print(f"  Baseline R_V:   {rv_base:.3f}")
            print(f"  Patched R_V:    {rv_patched:.3f}")
            print(f"  Delta:          {rv_patched - rv_base:+.3f}")
            print()
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    if not results:
        print("No valid results!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mistral_MIXTRAL_METHOD_{timestamp}.csv"
    df.to_csv(filename, index=False)
    
    # Analysis
    print("="*70)
    print("SUMMARY (Mixtral methodology)")
    print("="*70)
    print(f"Mean R_V{TARGET_LAYER}_rec:     {df[f'RV{TARGET_LAYER}_rec'].mean():.3f} ± {df[f'RV{TARGET_LAYER}_rec'].std():.3f}")
    print(f"Mean R_V{TARGET_LAYER}_base:    {df[f'RV{TARGET_LAYER}_base'].mean():.3f} ± {df[f'RV{TARGET_LAYER}_base'].std():.3f}")
    print(f"Mean R_V{TARGET_LAYER}_patched: {df[f'RV{TARGET_LAYER}_patched'].mean():.3f} ± {df[f'RV{TARGET_LAYER}_patched'].std():.3f}")
    print()
    print(f"Mean delta: {df['delta'].mean():+.4f}")
    print()
    
    # Check effect
    if df['delta'].mean() < -0.05:
        transfer = abs(df['delta'].mean() / (df[f'RV{TARGET_LAYER}_base'].mean() - df[f'RV{TARGET_LAYER}_rec'].mean()))
        print(f"✅ CAUSAL EFFECT DETECTED!")
        print(f"   Transfer strength: {transfer*100:.1f}% toward recursive")
    else:
        print("⚠️  No significant causal effect")
    
    print(f"\nResults saved to: {filename}")
    
    # Compare to Mixtral results
    print("\n" + "="*70)
    print("COMPARISON TO MIXTRAL")
    print("="*70)
    print("Mixtral (Layer 27, n=5):")
    print("  R_V_rec:     0.429 ± 0.044")
    print("  R_V_base:    1.078 ± 0.066")
    print("  R_V_patched: 0.886 ± 0.086")
    print("  Delta:       -0.192 (moved 29% toward recursive)")
    
    return df

if __name__ == "__main__":
    print("Run in notebook:")
    print("from mistral_EXACT_MIXTRAL_METHOD import test_mixtral_methodology")
    print("results = test_mixtral_methodology(model, tokenizer, prompt_bank_1c, num_pairs=5)")
