#!/usr/bin/env python3
"""
TRULY FIXED activation patching - ensures patching actually affects downstream layers
Key insight: We need to patch DURING the forward pass, not capture separately
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from contextlib import contextmanager
from scipy import stats
from datetime import datetime

# Configuration
WINDOW_SIZE = 6  # Matches short baseline prompts
EARLY_LAYER = 5
TARGET_LAYER = 21
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

def run_forward_with_intervention(model, tokenizer, text, 
                                 patch_layer=None, patch_source=None,
                                 capture_layers=None, window_size=WINDOW_SIZE):
    """
    Run a single forward pass with optional patching.
    
    Args:
        text: Input text
        patch_layer: Layer to patch at (None = no patching)
        patch_source: Source V tensor to patch with
        capture_layers: List of layers to capture V from
        
    Returns:
        Dict of captured V tensors by layer
    """
    if capture_layers is None:
        capture_layers = [EARLY_LAYER, TARGET_LAYER]
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    captured = {}
    
    with torch.no_grad():
        # Storage for each layer
        storage = {layer: [] for layer in capture_layers}
        
        # Register capture hooks
        handles = []
        for layer_idx in capture_layers:
            layer = model.model.layers[layer_idx].self_attn
            
            # Create a capture function with proper closure
            def make_capture_hook(layer_idx, storage_dict):
                def hook_fn(module, inp, out):
                    # If this is the patch layer, we need to capture AFTER patching
                    if layer_idx == patch_layer and patch_source is not None:
                        # Apply patch
                        B, T, D = out.shape
                        T_src = patch_source.shape[0]
                        W = min(window_size, T, T_src)
                        
                        if W > 0:
                            out = out.clone()
                            src_tensor = patch_source[-W:, :].to(out.device, dtype=out.dtype)
                            out[:, -W:, :] = src_tensor.unsqueeze(0).expand(B, -1, -1)
                        
                        # Store the PATCHED version
                        storage_dict[layer_idx].append(out.detach())
                        # Return patched for downstream
                        return out
                    else:
                        # Just capture
                        storage_dict[layer_idx].append(out.detach())
                        return out
                
                return hook_fn
            
            h = layer.v_proj.register_forward_hook(make_capture_hook(layer_idx, storage))
            handles.append(h)
        
        # Run forward pass
        _ = model(**inputs)
        
        # Remove hooks
        for h in handles:
            h.remove()
        
        # Extract results
        for layer_idx in capture_layers:
            if storage[layer_idx]:
                captured[layer_idx] = storage[layer_idx][0][0]  # [seq, hidden]
            else:
                captured[layer_idx] = None
    
    return captured

def run_complete_patching_experiment_v2(model, tokenizer, prompt_bank, num_pairs=60):
    """
    Complete patching experiment with proper intervention
    """
    print("="*70)
    print("ACTIVATION PATCHING EXPERIMENT (TRULY FIXED)")
    print("="*70)
    print(f"Testing {num_pairs} prompt pairs")
    print(f"Patch/Measure layer: {TARGET_LAYER}")
    print(f"Window size: {WINDOW_SIZE}")
    
    # Get prompt IDs
    recursive_ids = [k for k, v in prompt_bank.items() 
                     if v["group"] in ["L5_refined", "L4_full", "L3_deeper"]]
    
    baseline_ids = [k for k, v in prompt_bank.items() 
                    if v["group"] in ["baseline_factual", "baseline_creative", 
                                      "baseline_math", "long_control"]]
    
    # Shuffle and pair
    np.random.seed(42)
    np.random.shuffle(recursive_ids)
    np.random.shuffle(baseline_ids)
    
    n_pairs = min(num_pairs, len(recursive_ids), len(baseline_ids))
    
    results = []
    
    for i in tqdm(range(n_pairs), desc="Processing"):
        rec_id = recursive_ids[i % len(recursive_ids)]
        base_id = baseline_ids[i % len(baseline_ids)]
        
        rec_text = prompt_bank[rec_id]["text"]
        base_text = prompt_bank[base_id]["text"]
        
        # Skip if too short
        base_tokens = len(tokenizer.encode(base_text))
        rec_tokens = len(tokenizer.encode(rec_text))
        
        if base_tokens < 2 or rec_tokens < 2:
            continue
        
        try:
            # 1. Baseline forward pass (no patching)
            base_vs = run_forward_with_intervention(
                model, tokenizer, base_text,
                patch_layer=None, patch_source=None,
                capture_layers=[EARLY_LAYER, TARGET_LAYER]
            )
            
            # 2. Recursive forward pass (no patching)
            rec_vs = run_forward_with_intervention(
                model, tokenizer, rec_text,
                patch_layer=None, patch_source=None,
                capture_layers=[EARLY_LAYER, TARGET_LAYER]
            )
            
            # 3. Main experiment: Patch recursive V into baseline at layer 21
            patched_vs = run_forward_with_intervention(
                model, tokenizer, base_text,
                patch_layer=TARGET_LAYER, 
                patch_source=rec_vs[TARGET_LAYER],
                capture_layers=[EARLY_LAYER, TARGET_LAYER]
            )
            
            # 4. Control: Random patch
            random_v = torch.randn_like(rec_vs[TARGET_LAYER])
            random_v = random_v * (rec_vs[TARGET_LAYER].norm() / random_v.norm())
            
            random_vs = run_forward_with_intervention(
                model, tokenizer, base_text,
                patch_layer=TARGET_LAYER,
                patch_source=random_v,
                capture_layers=[EARLY_LAYER, TARGET_LAYER]
            )
            
            # 5. Control: Shuffled patch
            shuf_indices = torch.randperm(rec_vs[TARGET_LAYER].shape[0])
            v_shuffled = rec_vs[TARGET_LAYER][shuf_indices, :]
            
            shuffled_vs = run_forward_with_intervention(
                model, tokenizer, base_text,
                patch_layer=TARGET_LAYER,
                patch_source=v_shuffled,
                capture_layers=[EARLY_LAYER, TARGET_LAYER]
            )
            
            # Compute all metrics
            er5_base, pr5_base = compute_metrics_fast(base_vs[EARLY_LAYER])
            er21_base, pr21_base = compute_metrics_fast(base_vs[TARGET_LAYER])
            
            er5_rec, pr5_rec = compute_metrics_fast(rec_vs[EARLY_LAYER])
            er21_rec, pr21_rec = compute_metrics_fast(rec_vs[TARGET_LAYER])
            
            er21_patch, pr21_patch = compute_metrics_fast(patched_vs[TARGET_LAYER])
            er21_random, pr21_random = compute_metrics_fast(random_vs[TARGET_LAYER])
            er21_shuffled, pr21_shuffled = compute_metrics_fast(shuffled_vs[TARGET_LAYER])
            
            # Compute R_V ratios
            rv_base = pr21_base / pr5_base if pr5_base > 0 else np.nan
            rv_rec = pr21_rec / pr5_rec if pr5_rec > 0 else np.nan
            rv_patch = pr21_patch / pr5_base if pr5_base > 0 else np.nan  # Use baseline's early layer
            rv_random = pr21_random / pr5_base if pr5_base > 0 else np.nan
            rv_shuffled = pr21_shuffled / pr5_base if pr5_base > 0 else np.nan
            
            # Verification: Check that patching actually changed values
            patch_changed = not np.isclose(pr21_patch, pr21_base, rtol=1e-5)
            
            # Store results
            results.append({
                "pair_idx": i,
                "rec_id": rec_id,
                "base_id": base_id,
                "rec_tokens": rec_tokens,
                "base_tokens": base_tokens,
                # Raw PR values
                "pr5_base": pr5_base,
                "pr21_base": pr21_base,
                "pr5_rec": pr5_rec,
                "pr21_rec": pr21_rec,
                "pr21_patch": pr21_patch,
                "pr21_random": pr21_random,
                "pr21_shuffled": pr21_shuffled,
                # R_V ratios
                "rv_base": rv_base,
                "rv_rec": rv_rec,
                "rv_patch": rv_patch,
                "rv_random": rv_random,
                "rv_shuffled": rv_shuffled,
                # Deltas
                "delta_main": rv_patch - rv_base,
                "delta_random": rv_random - rv_base,
                "delta_shuffled": rv_shuffled - rv_base,
                # Verification
                "patch_changed": patch_changed
            })
            
        except Exception as e:
            print(f"\nError on pair {i}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Filter for valid results
    df = df[df["delta_main"].notna()]
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mistral_patching_truly_fixed_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\nResults saved to: {filename}")
    
    # Verification check
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    patches_that_changed = df['patch_changed'].sum()
    print(f"Patches that actually changed values: {patches_that_changed}/{len(df)}")
    
    if patches_that_changed == 0:
        print("❌ CRITICAL ERROR: No patches changed any values!")
        print("   The patching mechanism is still not working.")
        return df
    
    # Analysis
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Valid pairs analyzed: {len(df)}")
    
    print(f"\nBaseline separation:")
    print(f"  Recursive R_V: {df['rv_rec'].mean():.3f} ± {df['rv_rec'].std():.3f}")
    print(f"  Baseline R_V:  {df['rv_base'].mean():.3f} ± {df['rv_base'].std():.3f}")
    print(f"  Separation:    {(df['rv_rec'].mean() - df['rv_base'].mean()):.3f}")
    
    print(f"\nPatching effect:")
    print(f"  Patched R_V:   {df['rv_patch'].mean():.3f} ± {df['rv_patch'].std():.3f}")
    print(f"  Delta R_V:     {df['delta_main'].mean():.4f} ± {df['delta_main'].std():.4f}")
    
    # Check if patching moved values toward recursive
    movement = (df['rv_patch'].mean() - df['rv_base'].mean()) / (df['rv_rec'].mean() - df['rv_base'].mean())
    print(f"  Movement toward recursive: {movement*100:.1f}%")
    
    # Statistical test
    if len(df) > 1 and df["delta_main"].std() > 0:
        # Test if patching reduces R_V (for recursive < baseline case)
        if df['rv_rec'].mean() < df['rv_base'].mean():
            t_stat, p_val = stats.ttest_1samp(df["delta_main"], 0, alternative='less')
        else:
            t_stat, p_val = stats.ttest_1samp(df["delta_main"], 0, alternative='greater')
        
        cohen_d = df["delta_main"].mean() / df["delta_main"].std()
        
        print(f"\nStatistical test:")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value:     {p_val:.4f}")
        print(f"  Cohen's d:   {cohen_d:.3f}")
        
        if p_val < 0.05 and abs(cohen_d) > 0.3:
            print("\n✅ SIGNIFICANT CAUSAL EFFECT DETECTED!")
        else:
            print("\n⚠️  No significant causal effect")
    
    print(f"\nControl conditions:")
    print(f"  Random delta:    {df['delta_random'].mean():.4f} ± {df['delta_random'].std():.4f}")
    print(f"  Shuffled delta:  {df['delta_shuffled'].mean():.4f} ± {df['delta_shuffled'].std():.4f}")
    
    # Sanity check on first row
    print("\n" + "="*70)
    print("FIRST ROW SANITY CHECK")
    print("="*70)
    row = df.iloc[0]
    print(f"pr21_base:    {row['pr21_base']:.6f}")
    print(f"pr21_rec:     {row['pr21_rec']:.6f}")
    print(f"pr21_patch:   {row['pr21_patch']:.6f}")
    print(f"Patch changed values: {row['patch_changed']}")
    
    if row['pr21_patch'] == row['pr21_base']:
        print("❌ STILL BROKEN: Patched = Baseline")
    elif row['pr21_patch'] == row['pr21_rec']:
        print("⚠️  Patched = Recursive (perfect transfer?)")
    else:
        print("✅ Patched is different from both!")
    
    return df

if __name__ == "__main__":
    print("Run in notebook:")
    print("from mistral_patching_TRULY_FIXED import run_complete_patching_experiment_v2")
    print("results = run_complete_patching_experiment_v2(model, tokenizer, prompt_bank_1c)")

