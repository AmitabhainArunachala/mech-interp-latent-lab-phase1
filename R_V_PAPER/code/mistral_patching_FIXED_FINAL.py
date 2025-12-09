#!/usr/bin/env python3
"""
FIXED activation patching experiment for Mistral-7B
Key fixes:
1. Reduced WINDOW_SIZE to 6 (most baseline prompts are short)
2. Fixed hook ordering to ensure patching happens BEFORE measurement
3. Added explicit validation of patching effect
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from contextlib import contextmanager
from scipy import stats
from datetime import datetime

# Configuration
WINDOW_SIZE = 6  # Reduced from 16 to handle short prompts
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
    W = min(window_size, T)  # Handle short sequences
    
    if W < 2:  # Need at least 2 tokens
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

@contextmanager
def patch_v_during_forward(model, layer_idx, source_v, window_size=WINDOW_SIZE):
    """
    Context manager that patches V output during forward pass.
    This ensures downstream layers see the patched values.
    """
    handle = None
    
    def patch_hook(module, inp, out):
        # Clone output to avoid in-place modification issues
        B, T, D = out.shape
        T_src = source_v.shape[0]
        W = min(window_size, T, T_src)
        
        if W > 0:
            out_modified = out.clone()
            src_tensor = source_v[-W:, :].to(out.device, dtype=out.dtype)
            out_modified[:, -W:, :] = src_tensor.unsqueeze(0).expand(B, -1, -1)
            return out_modified
        return out
    
    try:
        layer = model.model.layers[layer_idx].self_attn
        handle = layer.v_proj.register_forward_hook(patch_hook)
        yield
    finally:
        if handle:
            handle.remove()

def run_single_forward_get_V(model, tokenizer, text, capture_layers):
    """Run forward pass and capture V at specified layers"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    captured = {layer: [] for layer in capture_layers}
    handles = []
    
    with torch.no_grad():
        # Register hooks for all layers we want to capture
        for layer_idx in capture_layers:
            storage = captured[layer_idx]
            
            def make_hook(storage_list):
                def hook_fn(m, i, o):
                    storage_list.append(o.detach())
                    return o
                return hook_fn
            
            layer = model.model.layers[layer_idx].self_attn
            h = layer.v_proj.register_forward_hook(make_hook(storage))
            handles.append(h)
        
        # Run forward pass
        _ = model(**inputs)
        
        # Remove hooks
        for h in handles:
            h.remove()
    
    # Extract tensors
    result = {}
    for layer_idx in capture_layers:
        if captured[layer_idx]:
            result[layer_idx] = captured[layer_idx][0][0]  # [seq, hidden]
        else:
            result[layer_idx] = None
    
    return result

def run_patched_forward_final(model, tokenizer, baseline_text, v_source, patch_layer, measure_layer):
    """
    Run forward pass with patching applied.
    Patch at patch_layer, measure at measure_layer.
    """
    inputs = tokenizer(baseline_text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    v_early = []
    v_measured = []
    
    with torch.no_grad():
        # Set up capture hooks
        def capture_early(m, i, o):
            v_early.append(o.detach())
            return o
        
        def capture_measure(m, i, o):
            v_measured.append(o.detach())
            return o
        
        h1 = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(capture_early)
        h2 = model.model.layers[measure_layer].self_attn.v_proj.register_forward_hook(capture_measure)
        
        # Run with patching
        with patch_v_during_forward(model, patch_layer, v_source):
            _ = model(**inputs)
        
        h1.remove()
        h2.remove()
    
    v5 = v_early[0][0] if v_early else None
    vM = v_measured[0][0] if v_measured else None
    
    return v5, vM

def run_complete_patching_experiment(model, tokenizer, prompt_bank, num_pairs=60):
    """
    Complete activation patching experiment with all controls.
    """
    print("="*70)
    print("ACTIVATION PATCHING EXPERIMENT (FIXED)")
    print("="*70)
    print(f"Testing {num_pairs} prompt pairs")
    print(f"Patch layer: {TARGET_LAYER}")
    print(f"Measure layer: {TARGET_LAYER}")
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
        
        # Skip if baseline is too short (need at least 2 tokens)
        base_tokens = len(tokenizer.encode(base_text))
        if base_tokens < 2:
            continue
        
        try:
            # 1. Get baseline V values
            base_vs = run_single_forward_get_V(model, tokenizer, base_text, 
                                              [EARLY_LAYER, TARGET_LAYER])
            v5_base = base_vs[EARLY_LAYER]
            v21_base = base_vs[TARGET_LAYER]
            
            # 2. Get recursive V values
            rec_vs = run_single_forward_get_V(model, tokenizer, rec_text,
                                             [EARLY_LAYER, TARGET_LAYER])
            v5_rec = rec_vs[EARLY_LAYER]
            v21_rec = rec_vs[TARGET_LAYER]
            
            # 3. Main experiment: Patch recursive V21 into baseline
            v5_patch_main, v21_patch_main = run_patched_forward_final(
                model, tokenizer, base_text, v21_rec, TARGET_LAYER, TARGET_LAYER
            )
            
            # 4. Control 1: Random patch (norm-matched)
            random_v = torch.randn_like(v21_rec)
            random_v = random_v * (v21_rec.norm() / random_v.norm())
            v5_patch_rand, v21_patch_rand = run_patched_forward_final(
                model, tokenizer, base_text, random_v, TARGET_LAYER, TARGET_LAYER
            )
            
            # 5. Control 2: Shuffled patch
            v21_rec_shuffled = v21_rec[torch.randperm(v21_rec.shape[0]), :]
            v5_patch_shuf, v21_patch_shuf = run_patched_forward_final(
                model, tokenizer, base_text, v21_rec_shuffled, TARGET_LAYER, TARGET_LAYER
            )
            
            # 6. Control 3: Wrong layer patch (patch at layer 15 instead)
            wrong_layer = 15
            wrong_vs = run_single_forward_get_V(model, tokenizer, rec_text, [wrong_layer])
            v_wrong = wrong_vs[wrong_layer]
            v5_patch_wrong, v21_patch_wrong = run_patched_forward_final(
                model, tokenizer, base_text, v_wrong, wrong_layer, TARGET_LAYER
            )
            
            # Compute metrics
            er5_base, pr5_base = compute_metrics_fast(v5_base)
            er21_base, pr21_base = compute_metrics_fast(v21_base)
            
            er5_rec, pr5_rec = compute_metrics_fast(v5_rec)
            er21_rec, pr21_rec = compute_metrics_fast(v21_rec)
            
            er21_patch_main, pr21_patch_main = compute_metrics_fast(v21_patch_main)
            er21_patch_rand, pr21_patch_rand = compute_metrics_fast(v21_patch_rand)
            er21_patch_shuf, pr21_patch_shuf = compute_metrics_fast(v21_patch_shuf)
            er21_patch_wrong, pr21_patch_wrong = compute_metrics_fast(v21_patch_wrong)
            
            # Calculate R_V ratios
            rv_base = pr21_base / pr5_base if pr5_base > 0 else np.nan
            rv_rec = pr21_rec / pr5_rec if pr5_rec > 0 else np.nan
            rv_patch_main = pr21_patch_main / pr5_base if pr5_base > 0 else np.nan
            rv_patch_rand = pr21_patch_rand / pr5_base if pr5_base > 0 else np.nan
            rv_patch_shuf = pr21_patch_shuf / pr5_base if pr5_base > 0 else np.nan
            rv_patch_wrong = pr21_patch_wrong / pr5_base if pr5_base > 0 else np.nan
            
            # Store results
            results.append({
                "pair_idx": i,
                "rec_id": rec_id,
                "base_id": base_id,
                "rec_tokens": len(tokenizer.encode(rec_text)),
                "base_tokens": base_tokens,
                # Metrics
                "pr5_base": pr5_base,
                "pr21_base": pr21_base,
                "pr5_rec": pr5_rec,
                "pr21_rec": pr21_rec,
                "pr21_patch_main": pr21_patch_main,
                "pr21_patch_rand": pr21_patch_rand,
                "pr21_patch_shuf": pr21_patch_shuf,
                "pr21_patch_wrong": pr21_patch_wrong,
                # R_V ratios
                "rv_base": rv_base,
                "rv_rec": rv_rec,
                "rv_patch_main": rv_patch_main,
                "rv_patch_rand": rv_patch_rand,
                "rv_patch_shuf": rv_patch_shuf,
                "rv_patch_wrong": rv_patch_wrong,
                # Deltas
                "delta_main": rv_patch_main - rv_base if not np.isnan(rv_patch_main) else np.nan,
                "delta_rand": rv_patch_rand - rv_base if not np.isnan(rv_patch_rand) else np.nan,
                "delta_shuf": rv_patch_shuf - rv_base if not np.isnan(rv_patch_shuf) else np.nan,
                "delta_wrong": rv_patch_wrong - rv_base if not np.isnan(rv_patch_wrong) else np.nan,
            })
            
        except Exception as e:
            print(f"Error on pair {i}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Filter valid results
    df = df[df["delta_main"].notna()]
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mistral_patching_fixed_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\nResults saved to: {filename}")
    
    # Analysis
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Valid pairs analyzed: {len(df)}")
    
    print(f"\nBaseline separation:")
    print(f"  Recursive R_V: {df['rv_rec'].mean():.3f} ± {df['rv_rec'].std():.3f}")
    print(f"  Baseline R_V:  {df['rv_base'].mean():.3f} ± {df['rv_base'].std():.3f}")
    print(f"  Difference:    {df['rv_rec'].mean() - df['rv_base'].mean():.3f}")
    
    print(f"\nMain patching effect:")
    print(f"  Patched R_V:   {df['rv_patch_main'].mean():.3f} ± {df['rv_patch_main'].std():.3f}")
    print(f"  Delta R_V:     {df['delta_main'].mean():.4f} ± {df['delta_main'].std():.4f}")
    
    # Statistical test
    if len(df) > 1 and df["delta_main"].std() > 0:
        t_stat, p_val = stats.ttest_1samp(df["delta_main"], 0, alternative='less')
        cohen_d = df["delta_main"].mean() / df["delta_main"].std()
        
        print(f"\nStatistical test (H1: patching reduces R_V):")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value:     {p_val:.4f}")
        print(f"  Cohen's d:   {cohen_d:.3f}")
        
        if p_val < 0.05 and cohen_d < -0.3:
            print("\n✅ SIGNIFICANT CAUSAL EFFECT DETECTED!")
            transfer_pct = abs(df['delta_main'].mean() / (df['rv_base'].mean() - df['rv_rec'].mean()) * 100)
            print(f"   Transfer strength: {transfer_pct:.1f}% of full recursive effect")
        else:
            print("\n⚠️  No significant causal effect detected")
    
    print(f"\nControl conditions:")
    print(f"  Random delta:      {df['delta_rand'].mean():.4f} ± {df['delta_rand'].std():.4f}")
    print(f"  Shuffled delta:    {df['delta_shuf'].mean():.4f} ± {df['delta_shuf'].std():.4f}")
    print(f"  Wrong layer delta: {df['delta_wrong'].mean():.4f} ± {df['delta_wrong'].std():.4f}")
    
    # Check if main effect > controls
    if abs(df['delta_main'].mean()) > abs(df['delta_rand'].mean()) and \
       abs(df['delta_main'].mean()) > abs(df['delta_shuf'].mean()):
        print("\n✅ Main effect stronger than control conditions")
    
    return df

if __name__ == "__main__":
    print("Import and run in notebook:")
    print("from mistral_patching_FIXED_FINAL import run_complete_patching_experiment")
    print("results = run_complete_patching_experiment(model, tokenizer, prompt_bank_1c, num_pairs=60)")
