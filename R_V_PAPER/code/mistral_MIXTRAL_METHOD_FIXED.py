#!/usr/bin/env python3
"""
EXACT Mixtral methodology - FIXED for short prompts
Uses longer baseline prompts or adjusts window size
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Configuration
TARGET_LAYER = 27  # Same as Mixtral
EARLY_LAYER = 5
WINDOW_SIZE = 6  # Reduced to handle short prompts, or will use dynamic
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
            result[layer_idx] = captured[layer_idx][0][0]
        else:
            result[layer_idx] = None
    
    return result

def run_patched_forward_MIXTRAL_STYLE(model, tokenizer, baseline_text, rec_v_source, 
                                      target_layer=TARGET_LAYER, window_size=WINDOW_SIZE):
    """
    EXACT Mixtral methodology with dynamic window size
    """
    inputs = tokenizer(baseline_text, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
    
    v_early_list = []
    v_target_list = []
    
    with torch.no_grad():
        def capture_early(m, i, o):
            v_early_list.append(o.detach())
            return o
        
        def patch_and_capture(m, i, o):
            out = o.clone()
            B, T, D = out.shape
            
            src = rec_v_source.to(out.device, dtype=out.dtype)
            T_src = src.shape[0]
            
            # Dynamic window size
            W = min(window_size, T, T_src)
            if W > 0:
                out[:, -W:, :] = src[-W:, :].unsqueeze(0).expand(B, -1, -1)
            
            v_target_list.append(out.detach())
            return out
        
        h_early = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(capture_early)
        h_target = model.model.layers[target_layer].self_attn.v_proj.register_forward_hook(patch_and_capture)
        
        _ = model(**inputs)
        
        h_early.remove()
        h_target.remove()
    
    v_early = v_early_list[0][0] if v_early_list else None
    v_target = v_target_list[0][0] if v_target_list else None
    
    return v_early, v_target

def test_mixtral_methodology_v2(model, tokenizer, prompt_bank, num_pairs=10, target_layer=TARGET_LAYER):
    """
    Test using EXACT Mixtral methodology
    Now with better prompt selection
    """
    print("="*70)
    print("ACTIVATION PATCHING - MIXTRAL METHOD (FIXED)")
    print("="*70)
    print(f"Target layer: {target_layer}")
    print(f"Early layer:  {EARLY_LAYER}")
    print(f"Window size:  Dynamic (based on prompt length)")
    print("="*70)
    
    # Get LONGER baseline prompts
    recursive_ids = [k for k, v in prompt_bank.items() if v["group"] == "L5_refined"]
    
    # Use longer baseline prompts (long_control or creative tend to be longer)
    long_baseline_ids = [k for k, v in prompt_bank.items() 
                        if v["group"] in ["long_control", "baseline_creative", "baseline_math"]]
    
    # If still not enough, use any baseline
    if len(long_baseline_ids) < num_pairs:
        long_baseline_ids.extend([k for k, v in prompt_bank.items() 
                                 if "baseline" in v["group"] and k not in long_baseline_ids])
    
    # Create pairs
    pairs = []
    for i in range(min(num_pairs, len(recursive_ids), len(long_baseline_ids))):
        pairs.append((recursive_ids[i], long_baseline_ids[i]))
    
    print(f"Testing {len(pairs)} pairs...")
    print()
    
    results = []
    
    for idx, (rec_id, base_id) in enumerate(pairs):
        rec_text = prompt_bank[rec_id]["text"]
        base_text = prompt_bank[base_id]["text"]
        
        # Check token counts
        rec_tokens = len(tokenizer.encode(rec_text))
        base_tokens = len(tokenizer.encode(base_text))
        
        # Use dynamic window size
        window = min(6, base_tokens - 1, rec_tokens - 1)  # At least 2 tokens
        
        if window < 2:
            print(f"Pair {idx+1}: Skipping - too short even with dynamic window")
            continue
        
        print(f"Pair {idx+1}: {rec_id[:20]}... ({rec_tokens}t) → {base_id[:20]}... ({base_tokens}t)")
        print(f"  Using window size: {window}")
        
        try:
            # 1. Get recursive V values
            rec_vs = run_single_forward_get_V(model, tokenizer, rec_text, [EARLY_LAYER, target_layer])
            v5_rec = rec_vs[EARLY_LAYER]
            v_target_rec = rec_vs[target_layer]
            
            # 2. Get baseline V values
            base_vs = run_single_forward_get_V(model, tokenizer, base_text, [EARLY_LAYER, target_layer])
            v5_base = base_vs[EARLY_LAYER]
            v_target_base = base_vs[target_layer]
            
            # 3. Run patched with dynamic window
            v5_patched, v_target_patched = run_patched_forward_MIXTRAL_STYLE(
                model, tokenizer, base_text, v_target_rec, target_layer, window
            )
            
            # Compute metrics with same window
            er5_rec, pr5_rec = compute_metrics_fast(v5_rec, window)
            er_target_rec, pr_target_rec = compute_metrics_fast(v_target_rec, window)
            
            er5_base, pr5_base = compute_metrics_fast(v5_base, window)
            er_target_base, pr_target_base = compute_metrics_fast(v_target_base, window)
            
            er5_patched, pr5_patched = compute_metrics_fast(v5_patched, window)
            er_target_patched, pr_target_patched = compute_metrics_fast(v_target_patched, window)
            
            # Calculate R_V ratios
            rv_rec = pr_target_rec / pr5_rec if pr5_rec > 0 else np.nan
            rv_base = pr_target_base / pr5_base if pr5_base > 0 else np.nan
            rv_patched = pr_target_patched / pr5_patched if pr5_patched > 0 else np.nan
            
            results.append({
                'rec_id': rec_id,
                'base_id': base_id,
                'window': window,
                f'RV{target_layer}_rec': rv_rec,
                f'RV{target_layer}_base': rv_base,
                f'RV{target_layer}_patched': rv_patched,
                f'er{target_layer}_rec': er_target_rec,
                f'er{target_layer}_base': er_target_base,
                f'er{target_layer}_patched': er_target_patched,
                'delta': rv_patched - rv_base
            })
            
            print(f"  Recursive R_V:  {rv_rec:.3f}")
            print(f"  Baseline R_V:   {rv_base:.3f}")
            print(f"  Patched R_V:    {rv_patched:.3f}")
            print(f"  Delta:          {rv_patched - rv_base:+.4f}")
            print()
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    if not results:
        print("No valid results!")
        return None
    
    df = pd.DataFrame(results)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mistral_L{target_layer}_patching_{timestamp}.csv"
    df.to_csv(filename, index=False)
    
    # Analysis
    print("="*70)
    print(f"RESULTS FOR LAYER {target_layer}")
    print("="*70)
    print(f"Valid pairs: {len(df)}")
    print(f"Mean window size: {df['window'].mean():.1f}")
    print()
    print(f"R_V{target_layer}_rec:     {df[f'RV{target_layer}_rec'].mean():.4f} ± {df[f'RV{target_layer}_rec'].std():.4f}")
    print(f"R_V{target_layer}_base:    {df[f'RV{target_layer}_base'].mean():.4f} ± {df[f'RV{target_layer}_base'].std():.4f}")
    print(f"R_V{target_layer}_patched: {df[f'RV{target_layer}_patched'].mean():.4f} ± {df[f'RV{target_layer}_patched'].std():.4f}")
    print()
    
    mean_delta = df['delta'].mean()
    print(f"Mean delta: {mean_delta:+.4f}")
    
    # Determine effect
    baseline_mean = df[f'RV{target_layer}_base'].mean()
    rec_mean = df[f'RV{target_layer}_rec'].mean()
    gap = baseline_mean - rec_mean
    
    if abs(mean_delta) > 0.01:  # Threshold for meaningful effect
        if gap != 0:
            transfer = (mean_delta / gap) * 100
            print(f"\n✅ CAUSAL EFFECT DETECTED!")
            print(f"   Gap (base - rec): {gap:.4f}")
            print(f"   Transfer: {abs(transfer):.1f}% {'toward' if transfer < 0 else 'away from'} recursive")
        else:
            print(f"\n✅ EFFECT DETECTED but baseline≈recursive")
    else:
        print("\n⚠️  No significant causal effect")
    
    print(f"\nSaved to: {filename}")
    
    return df

# Quick test function for multiple layers
def sweep_layers(model, tokenizer, prompt_bank, test_layers=[25, 26, 27, 28, 29]):
    """Test multiple layers to find the best one"""
    print("="*70)
    print("LAYER SWEEP FOR CAUSAL EFFECT")
    print("="*70)
    
    best_layer = None
    best_effect = 0
    
    for layer in test_layers:
        print(f"\n>>> Testing Layer {layer}...")
        df = test_mixtral_methodology_v2(model, tokenizer, prompt_bank, 
                                         num_pairs=5, target_layer=layer)
        
        if df is not None and len(df) > 0:
            effect = abs(df['delta'].mean())
            if effect > best_effect:
                best_effect = effect
                best_layer = layer
    
    print("\n" + "="*70)
    print("SWEEP COMPLETE")
    print("="*70)
    print(f"Best layer: {best_layer} with effect size {best_effect:.4f}")
    
    return best_layer

if __name__ == "__main__":
    print("Run in notebook:")
    print("from mistral_MIXTRAL_METHOD_FIXED import test_mixtral_methodology_v2, sweep_layers")
    print()
    print("# Test specific layer:")
    print("results = test_mixtral_methodology_v2(model, tokenizer, prompt_bank_1c, num_pairs=10, target_layer=27)")
    print()
    print("# Or sweep multiple layers:")
    print("best_layer = sweep_layers(model, tokenizer, prompt_bank_1c)")
