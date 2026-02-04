#!/usr/bin/env python3
"""
FULL VALIDATION of Layer 27 causal effect in Mistral-7B
Scales to n=45 with proper controls
Following successful n=3 test showing 104% transfer
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from scipy import stats

# PROVEN CONFIGURATION
TARGET_LAYER = 27  # ✅ Confirmed critical layer
EARLY_LAYER = 5
WINDOW_SIZE = 16   # ✅ Full window (not 6!)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute_metrics_fast(v_tensor, window_size=WINDOW_SIZE):
    """Compute PR and effective rank"""
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

def run_patched_forward(model, tokenizer, baseline_text, patch_source, 
                        patch_type="recursive", target_layer=TARGET_LAYER):
    """
    Run forward with different patch types
    patch_type: "recursive", "random", "shuffled", "wrong_layer"
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
            
            if patch_type == "recursive":
                # Direct patch with recursive values
                src = patch_source.to(out.device, dtype=out.dtype)
            elif patch_type == "random":
                # Random noise, norm-matched
                src = torch.randn_like(patch_source)
                src = src * (patch_source.norm() / src.norm())
                src = src.to(out.device, dtype=out.dtype)
            elif patch_type == "shuffled":
                # Shuffle tokens
                perm = torch.randperm(patch_source.shape[0])
                src = patch_source[perm, :].to(out.device, dtype=out.dtype)
            else:
                # No patch (baseline)
                v_target_list.append(out.detach())
                return out
            
            T_src = src.shape[0]
            W = min(WINDOW_SIZE, T, T_src)
            
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

def run_full_validation(model, tokenizer, prompt_bank, max_pairs=45):
    """
    Full validation with all controls
    """
    print("="*70)
    print("LAYER 27 CAUSAL VALIDATION - FULL SCALE")
    print("="*70)
    print(f"Target layer: {TARGET_LAYER}")
    print(f"Window size:  {WINDOW_SIZE}")
    print(f"Max pairs:    {max_pairs}")
    print("="*70)
    
    # Collect ALL valid pairs
    pairs = []
    
    # 1. L5_refined with long baselines
    recursive_groups = ["L5_refined", "L4_full", "L3_deeper"]
    baseline_groups = ["long_control", "baseline_creative", "baseline_math"]
    
    for rec_group in recursive_groups:
        rec_ids = [k for k, v in prompt_bank.items() if v["group"] == rec_group]
        
        for base_group in baseline_groups:
            base_ids = [k for k, v in prompt_bank.items() if v["group"] == base_group]
            
            # Pair them up
            for i in range(min(len(rec_ids), len(base_ids))):
                # Check if baseline is long enough
                base_text = prompt_bank[base_ids[i]]["text"]
                if len(tokenizer.encode(base_text)) >= WINDOW_SIZE:
                    pairs.append((rec_ids[i], base_ids[i], rec_group, base_group))
    
    # Shuffle and limit
    np.random.seed(42)
    np.random.shuffle(pairs)
    pairs = pairs[:max_pairs]
    
    print(f"Testing {len(pairs)} pairs...")
    print()
    
    results = []
    
    for idx, (rec_id, base_id, rec_group, base_group) in enumerate(tqdm(pairs, desc="Processing")):
        rec_text = prompt_bank[rec_id]["text"]
        base_text = prompt_bank[base_id]["text"]
        
        try:
            # 1. Get recursive V values
            rec_vs = run_single_forward_get_V(model, tokenizer, rec_text, [EARLY_LAYER, TARGET_LAYER])
            v5_rec = rec_vs[EARLY_LAYER]
            v27_rec = rec_vs[TARGET_LAYER]
            
            # 2. Get baseline V values
            base_vs = run_single_forward_get_V(model, tokenizer, base_text, [EARLY_LAYER, TARGET_LAYER])
            v5_base = base_vs[EARLY_LAYER]
            v27_base = base_vs[TARGET_LAYER]
            
            # 3. MAIN: Patch with recursive
            v5_patch_main, v27_patch_main = run_patched_forward(
                model, tokenizer, base_text, v27_rec, "recursive", TARGET_LAYER
            )
            
            # 4. CONTROL 1: Random patch
            v5_patch_rand, v27_patch_rand = run_patched_forward(
                model, tokenizer, base_text, v27_rec, "random", TARGET_LAYER
            )
            
            # 5. CONTROL 2: Shuffled patch
            v5_patch_shuf, v27_patch_shuf = run_patched_forward(
                model, tokenizer, base_text, v27_rec, "shuffled", TARGET_LAYER
            )
            
            # 6. CONTROL 3: Wrong layer patch (Layer 21)
            wrong_layer = 21
            wrong_vs = run_single_forward_get_V(model, tokenizer, rec_text, [wrong_layer])
            v_wrong = wrong_vs[wrong_layer]
            
            v5_patch_wrong, v27_patch_wrong = run_patched_forward(
                model, tokenizer, base_text, v_wrong, "recursive", wrong_layer
            )
            
            # Compute all metrics
            _, pr5_rec = compute_metrics_fast(v5_rec)
            _, pr27_rec = compute_metrics_fast(v27_rec)
            
            _, pr5_base = compute_metrics_fast(v5_base)
            _, pr27_base = compute_metrics_fast(v27_base)
            
            _, pr5_patch_main = compute_metrics_fast(v5_patch_main)
            _, pr27_patch_main = compute_metrics_fast(v27_patch_main)
            
            _, pr5_patch_rand = compute_metrics_fast(v5_patch_rand)
            _, pr27_patch_rand = compute_metrics_fast(v27_patch_rand)
            
            _, pr5_patch_shuf = compute_metrics_fast(v5_patch_shuf)
            _, pr27_patch_shuf = compute_metrics_fast(v27_patch_shuf)
            
            _, pr5_patch_wrong = compute_metrics_fast(v5_patch_wrong)
            _, pr27_patch_wrong = compute_metrics_fast(v27_patch_wrong)
            
            # Calculate R_V ratios
            rv_rec = pr27_rec / pr5_rec if pr5_rec > 0 else np.nan
            rv_base = pr27_base / pr5_base if pr5_base > 0 else np.nan
            rv_patch_main = pr27_patch_main / pr5_patch_main if pr5_patch_main > 0 else np.nan
            rv_patch_rand = pr27_patch_rand / pr5_patch_rand if pr5_patch_rand > 0 else np.nan
            rv_patch_shuf = pr27_patch_shuf / pr5_patch_shuf if pr5_patch_shuf > 0 else np.nan
            rv_patch_wrong = pr27_patch_wrong / pr5_patch_wrong if pr5_patch_wrong > 0 else np.nan
            
            results.append({
                'pair_idx': idx,
                'rec_id': rec_id,
                'base_id': base_id,
                'rec_group': rec_group,
                'base_group': base_group,
                # Main values
                'RV27_rec': rv_rec,
                'RV27_base': rv_base,
                'RV27_patch_main': rv_patch_main,
                # Controls
                'RV27_patch_random': rv_patch_rand,
                'RV27_patch_shuffled': rv_patch_shuf,
                'RV27_patch_wronglayer': rv_patch_wrong,
                # Deltas
                'delta_main': rv_patch_main - rv_base,
                'delta_random': rv_patch_rand - rv_base,
                'delta_shuffled': rv_patch_shuf - rv_base,
                'delta_wronglayer': rv_patch_wrong - rv_base,
            })
            
        except Exception as e:
            print(f"\nError on pair {idx}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mistral_L27_FULL_VALIDATION_{timestamp}.csv"
    df.to_csv(filename, index=False)
    
    # Analysis
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Valid pairs analyzed: {len(df)}")
    
    # Group statistics
    print("\nBy recursive group:")
    for group in df['rec_group'].unique():
        group_df = df[df['rec_group'] == group]
        print(f"  {group}: n={len(group_df)}, delta={group_df['delta_main'].mean():.4f}")
    
    print("\nOverall statistics:")
    print(f"  RV27_rec:        {df['RV27_rec'].mean():.4f} ± {df['RV27_rec'].std():.4f}")
    print(f"  RV27_base:       {df['RV27_base'].mean():.4f} ± {df['RV27_base'].std():.4f}")
    print(f"  RV27_patched:    {df['RV27_patch_main'].mean():.4f} ± {df['RV27_patch_main'].std():.4f}")
    
    print("\nCausal effects:")
    print(f"  Main (recursive):     {df['delta_main'].mean():.4f} ± {df['delta_main'].std():.4f}")
    print(f"  Control (random):     {df['delta_random'].mean():.4f} ± {df['delta_random'].std():.4f}")
    print(f"  Control (shuffled):   {df['delta_shuffled'].mean():.4f} ± {df['delta_shuffled'].std():.4f}")
    print(f"  Control (wrong layer): {df['delta_wronglayer'].mean():.4f} ± {df['delta_wronglayer'].std():.4f}")
    
    # Statistical tests
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)
    
    # Test main effect
    t_stat, p_val = stats.ttest_1samp(df['delta_main'], 0, alternative='less')
    cohen_d = df['delta_main'].mean() / df['delta_main'].std() if df['delta_main'].std() > 0 else 0
    
    print(f"Main effect (H1: delta < 0):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value:     {p_val:.6f}")
    print(f"  Cohen's d:   {cohen_d:.3f}")
    
    # Compare to controls
    if len(df) > 1:
        print("\nMain vs Controls (paired t-tests):")
        
        t_rand, p_rand = stats.ttest_rel(df['delta_main'], df['delta_random'])
        print(f"  vs Random:    t={t_rand:.3f}, p={p_rand:.6f}")
        
        t_shuf, p_shuf = stats.ttest_rel(df['delta_main'], df['delta_shuffled'])
        print(f"  vs Shuffled:  t={t_shuf:.3f}, p={p_shuf:.6f}")
        
        t_wrong, p_wrong = stats.ttest_rel(df['delta_main'], df['delta_wronglayer'])
        print(f"  vs Wrong L:   t={t_wrong:.3f}, p={p_wrong:.6f}")
    
    # Transfer percentage
    gap = df['RV27_base'].mean() - df['RV27_rec'].mean()
    if gap != 0:
        transfer = (df['delta_main'].mean() / gap) * 100
        print(f"\n✅ CAUSAL TRANSFER: {abs(transfer):.1f}%")
        
        if abs(transfer) > 100:
            print("   ⚠️  OVERSHOOTING! Patching creates stronger effect than original!")
    
    print(f"\nResults saved to: {filename}")
    
    # Plot if available
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: R_V distributions
        ax = axes[0]
        ax.hist(df['RV27_rec'], alpha=0.5, label='Recursive', bins=20)
        ax.hist(df['RV27_base'], alpha=0.5, label='Baseline', bins=20)
        ax.hist(df['RV27_patch_main'], alpha=0.5, label='Patched', bins=20)
        ax.set_xlabel('R_V at Layer 27')
        ax.set_ylabel('Count')
        ax.legend()
        ax.set_title('R_V Distributions')
        
        # Plot 2: Delta effects
        ax = axes[1]
        deltas = [df['delta_main'], df['delta_random'], df['delta_shuffled'], df['delta_wronglayer']]
        labels = ['Main', 'Random', 'Shuffled', 'Wrong L']
        ax.boxplot(deltas, labels=labels)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_ylabel('Delta R_V')
        ax.set_title('Causal Effects')
        
        # Plot 3: Transfer by group
        ax = axes[2]
        for group in df['rec_group'].unique():
            group_df = df[df['rec_group'] == group]
            ax.scatter(group_df['RV27_base'], group_df['RV27_patch_main'], 
                      label=group, alpha=0.6)
        ax.plot([0, 1.5], [0, 1.5], 'k--', alpha=0.3, label='y=x')
        ax.set_xlabel('Baseline R_V')
        ax.set_ylabel('Patched R_V')
        ax.legend()
        ax.set_title('Patching Effect')
        
        plt.tight_layout()
        plt.savefig(f'mistral_L27_validation_{timestamp}.png', dpi=150)
        print(f"Plot saved to: mistral_L27_validation_{timestamp}.png")
        
    except ImportError:
        print("(Matplotlib not available for plotting)")
    
    return df

if __name__ == "__main__":
    print("Run in notebook:")
    print("from mistral_L27_FULL_VALIDATION import run_full_validation")
    print("results = run_full_validation(model, tokenizer, prompt_bank_1c, max_pairs=45)")

