#!/usr/bin/env python3
"""
FINAL CORRECTED Activation Patching for Mistral-7B
The key insight: We need to patch DURING the forward pass and let the model
continue processing, not just replace values after computation.
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from contextlib import contextmanager

# Configuration
WINDOW_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CRITICAL_LAYER = 21
EARLY_LAYER = 5

# ============================================================================
# CRITICAL FIX: Proper patching that affects downstream computation
# ============================================================================

@contextmanager
def patch_v_during_forward(model, layer_idx, source_v, window_size=WINDOW_SIZE):
    """
    Patch V at layer_idx DURING forward pass so it affects all downstream layers.
    This is the key difference - we intervene in the computation flow.
    """
    handle = None
    
    def patch_hook(module, inp, out):
        # This hook modifies the V output BEFORE it goes to the next layer
        B, T, D = out.shape
        T_src = source_v.shape[0]
        W = min(window_size, T, T_src)
        
        if W > 0:
            # Clone to avoid in-place issues
            out_modified = out.clone()
            # Inject source into last W positions
            src_tensor = source_v[-W:, :].to(out.device, dtype=out.dtype)
            out_modified[:, -W:, :] = src_tensor.unsqueeze(0).expand(B, -1, -1)
            return out_modified  # Return modified tensor to continue forward pass
        return out
    
    try:
        layer = model.model.layers[layer_idx].self_attn
        handle = layer.v_proj.register_forward_hook(patch_hook)
        yield
    finally:
        if handle:
            handle.remove()

def run_patched_forward_final(model, tokenizer, baseline_text, v_source, patch_layer, measure_layer):
    """
    Run baseline with patched V and measure at a specific layer.
    
    Args:
        model: The model
        tokenizer: The tokenizer  
        baseline_text: Text to run
        v_source: Source V to inject [seq, hidden]
        patch_layer: Where to inject (e.g., 21)
        measure_layer: Where to measure result (e.g., 21 for same layer)
    
    Returns:
        v_early: V from early layer
        v_measured: V from measure_layer AFTER patching affects it
    """
    v_early = []
    v_measured = []
    
    # Tokenize
    inputs = tokenizer(baseline_text, return_tensors="pt", 
                      truncation=True, max_length=512).to(DEVICE)
    
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
        
        # Run WITH patching active
        with patch_v_during_forward(model, patch_layer, v_source):
            _ = model(**inputs)
        
        h1.remove()
        h2.remove()
    
    v5 = v_early[0][0] if v_early else None
    vM = v_measured[0][0] if v_measured else None
    
    return v5, vM

def compute_metrics_fast(v_tensor, window_size=WINDOW_SIZE):
    """Compute metrics with proper error handling"""
    if v_tensor is None:
        return np.nan, np.nan
    
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]
    
    T, D = v_tensor.shape
    W = min(window_size, T)
    
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

# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_complete_patching_experiment(model, tokenizer, prompt_bank, num_pairs=60):
    """
    Run the complete patching experiment with all controls.
    """
    print("="*70)
    print("ACTIVATION PATCHING EXPERIMENT (FINAL)")
    print("="*70)
    
    # Build pairs
    np.random.seed(42)
    
    recursive_ids = [k for k, v in prompt_bank.items() 
                     if v["group"] in ["L5_refined", "L4_full", "L3_deeper"]]
    
    baseline_ids = [k for k, v in prompt_bank.items() 
                    if v["group"] in ["baseline_factual", "baseline_creative", 
                                      "baseline_math", "long_control"]]
    
    np.random.shuffle(recursive_ids)
    np.random.shuffle(baseline_ids)
    
    num_pairs = min(num_pairs, len(recursive_ids), len(baseline_ids))
    pairs = list(zip(recursive_ids[:num_pairs], baseline_ids[:num_pairs]))
    
    print(f"Testing {num_pairs} prompt pairs")
    print(f"Patch layer: {CRITICAL_LAYER}")
    print(f"Measure layer: {CRITICAL_LAYER}")
    print()
    
    results = []
    
    for idx, (rec_id, base_id) in enumerate(tqdm(pairs, desc="Processing")):
        rec_text = prompt_bank[rec_id]["text"]
        base_text = prompt_bank[base_id]["text"]
        
        # Skip very short prompts
        if len(tokenizer.encode(rec_text)) < WINDOW_SIZE:
            continue
        if len(tokenizer.encode(base_text)) < WINDOW_SIZE:
            continue
        
        # ====================================================================
        # 1. BASELINE MEASUREMENTS
        # ====================================================================
        
        # Get recursive V
        inputs_rec = tokenizer(rec_text, return_tensors="pt", 
                              truncation=True, max_length=512).to(DEVICE)
        v5_rec = []
        v21_rec = []
        
        with torch.no_grad():
            def cap5(m, i, o):
                v5_rec.append(o.detach())
                return o
            def cap21(m, i, o):
                v21_rec.append(o.detach())
                return o
            
            h1 = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(cap5)
            h2 = model.model.layers[CRITICAL_LAYER].self_attn.v_proj.register_forward_hook(cap21)
            
            _ = model(**inputs_rec)
            
            h1.remove()
            h2.remove()
        
        v5_rec = v5_rec[0][0] if v5_rec else None
        v21_rec = v21_rec[0][0] if v21_rec else None
        
        er5_rec, pr5_rec = compute_metrics_fast(v5_rec)
        er21_rec, pr21_rec = compute_metrics_fast(v21_rec)
        rv_rec = pr21_rec / pr5_rec if (pr5_rec and pr5_rec > 0) else np.nan
        
        # Get baseline V
        inputs_base = tokenizer(base_text, return_tensors="pt",
                               truncation=True, max_length=512).to(DEVICE)
        v5_base = []
        v21_base = []
        
        with torch.no_grad():
            def cap5(m, i, o):
                v5_base.append(o.detach())
                return o
            def cap21(m, i, o):
                v21_base.append(o.detach())
                return o
            
            h1 = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(cap5)
            h2 = model.model.layers[CRITICAL_LAYER].self_attn.v_proj.register_forward_hook(cap21)
            
            _ = model(**inputs_base)
            
            h1.remove()
            h2.remove()
        
        v5_base = v5_base[0][0] if v5_base else None
        v21_base = v21_base[0][0] if v21_base else None
        
        er5_base, pr5_base = compute_metrics_fast(v5_base)
        er21_base, pr21_base = compute_metrics_fast(v21_base)
        rv_base = pr21_base / pr5_base if (pr5_base and pr5_base > 0) else np.nan
        
        # ====================================================================
        # 2. MAIN CONDITION: Patch recursive V into baseline
        # ====================================================================
        
        v5_patch, v21_patch = run_patched_forward_final(
            model, tokenizer, base_text, v21_rec, 
            CRITICAL_LAYER, CRITICAL_LAYER
        )
        
        er5_patch, pr5_patch = compute_metrics_fast(v5_patch)
        er21_patch, pr21_patch = compute_metrics_fast(v21_patch)
        rv_patch = pr21_patch / pr5_patch if (pr5_patch and pr5_patch > 0) else np.nan
        
        # ====================================================================
        # 3. CONTROL: Random patch
        # ====================================================================
        
        v21_random = torch.randn_like(v21_rec)
        v21_random = v21_random * (v21_rec.norm() / v21_random.norm())
        
        v5_rand, v21_rand = run_patched_forward_final(
            model, tokenizer, base_text, v21_random,
            CRITICAL_LAYER, CRITICAL_LAYER
        )
        
        er5_rand, pr5_rand = compute_metrics_fast(v5_rand)
        er21_rand, pr21_rand = compute_metrics_fast(v21_rand)
        rv_rand = pr21_rand / pr5_rand if (pr5_rand and pr5_rand > 0) else np.nan
        
        # ====================================================================
        # 4. CONTROL: Shuffled patch
        # ====================================================================
        
        v21_shuffled = v21_rec.clone()
        perm = torch.randperm(v21_shuffled.shape[0])
        v21_shuffled = v21_shuffled[perm, :]
        
        v5_shuf, v21_shuf = run_patched_forward_final(
            model, tokenizer, base_text, v21_shuffled,
            CRITICAL_LAYER, CRITICAL_LAYER
        )
        
        er5_shuf, pr5_shuf = compute_metrics_fast(v5_shuf)
        er21_shuf, pr21_shuf = compute_metrics_fast(v21_shuf)
        rv_shuf = pr21_shuf / pr5_shuf if (pr5_shuf and pr5_shuf > 0) else np.nan
        
        # ====================================================================
        # 5. CONTROL: Wrong layer patch (patch at 16, measure at 21)
        # ====================================================================
        
        WRONG_LAYER = 16
        v5_wrong, v21_wrong = run_patched_forward_final(
            model, tokenizer, base_text, v21_rec,
            WRONG_LAYER, CRITICAL_LAYER  # Patch at 16, measure at 21
        )
        
        er5_wrong, pr5_wrong = compute_metrics_fast(v5_wrong)
        er21_wrong, pr21_wrong = compute_metrics_fast(v21_wrong)
        rv_wrong = pr21_wrong / pr5_wrong if (pr5_wrong and pr5_wrong > 0) else np.nan
        
        # ====================================================================
        # STORE RESULTS
        # ====================================================================
        
        results.append({
            "pair_id": idx,
            "rec_id": rec_id,
            "base_id": base_id,
            
            # Baselines
            "rv_rec": rv_rec,
            "rv_base": rv_base,
            "pr5_rec": pr5_rec,
            "pr5_base": pr5_base,
            "pr21_rec": pr21_rec,
            "pr21_base": pr21_base,
            
            # Main patch
            "rv_patch_main": rv_patch,
            "pr21_patch_main": pr21_patch,
            
            # Random control
            "rv_patch_random": rv_rand,
            "pr21_patch_random": pr21_rand,
            
            # Shuffled control
            "rv_patch_shuffled": rv_shuf,
            "pr21_patch_shuffled": pr21_shuf,
            
            # Wrong layer control
            "rv_patch_wronglayer": rv_wrong,
            "pr21_patch_wronglayer": pr21_wrong,
        })
        
        # Memory cleanup
        if (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    # Save and analyze
    df = pd.DataFrame(results)
    df.to_csv("mistral_patching_FINAL.csv", index=False)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    # Calculate deltas
    df["delta_main"] = df["rv_patch_main"] - df["rv_base"]
    df["delta_random"] = df["rv_patch_random"] - df["rv_base"]
    df["delta_shuffled"] = df["rv_patch_shuffled"] - df["rv_base"]
    df["delta_wronglayer"] = df["rv_patch_wronglayer"] - df["rv_base"]
    
    print(f"\nValid pairs analyzed: {len(df)}")
    print(f"\nBaseline separation:")
    print(f"  Recursive R_V: {df['rv_rec'].mean():.3f} ± {df['rv_rec'].std():.3f}")
    print(f"  Baseline R_V:  {df['rv_base'].mean():.3f} ± {df['rv_base'].std():.3f}")
    print(f"  Difference:    {df['rv_base'].mean() - df['rv_rec'].mean():.3f}")
    
    print(f"\nMain patching effect:")
    print(f"  Patched R_V:   {df['rv_patch_main'].mean():.3f} ± {df['rv_patch_main'].std():.3f}")
    print(f"  Delta R_V:     {df['delta_main'].mean():.4f} ± {df['delta_main'].std():.4f}")
    
    # Statistical test
    from scipy import stats
    t_stat, p_val = stats.ttest_1samp(df["delta_main"].dropna(), 0.0, alternative="less")
    cohen_d = df["delta_main"].mean() / df["delta_main"].std()
    
    print(f"\nStatistical test (H1: patching reduces R_V):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value:     {p_val:.6f}")
    print(f"  Cohen's d:   {cohen_d:.3f}")
    
    if p_val < 0.01 and df["delta_main"].mean() < 0:
        print("\n✅ SIGNIFICANT CAUSAL EFFECT DETECTED!")
        print("   Layer 21 causally mediates the R_V contraction")
    else:
        print("\n⚠️  No significant causal effect detected")
    
    print(f"\nControl conditions:")
    print(f"  Random delta:      {df['delta_random'].mean():.4f} ± {df['delta_random'].std():.4f}")
    print(f"  Shuffled delta:    {df['delta_shuffled'].mean():.4f} ± {df['delta_shuffled'].std():.4f}")
    print(f"  Wrong layer delta: {df['delta_wronglayer'].mean():.4f} ± {df['delta_wronglayer'].std():.4f}")
    
    return df

# ============================================================================
# RUN IT
# ============================================================================

if __name__ == "__main__":
    print("""
    To use in your notebook:
    
    from mistral_patching_FINAL import run_complete_patching_experiment
    
    # Run the full experiment
    results_df = run_complete_patching_experiment(
        model, 
        tokenizer, 
        prompt_bank_1c,
        num_pairs=60  # or 80
    )
    
    # Results are automatically saved to mistral_patching_FINAL.csv
    # and statistical analysis is printed
    """)

