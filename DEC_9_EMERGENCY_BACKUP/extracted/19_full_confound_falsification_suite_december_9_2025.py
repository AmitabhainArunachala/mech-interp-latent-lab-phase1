#!/usr/bin/env python3
"""
FULL CONFOUND FALSIFICATION SUITE - December 9, 2025
=====================================================
60 confound prompts (20 per group) + 20 recursive reference prompts
Expected runtime: ~2-3 hours on RTX 6000 Pro

Confound Groups:
1. long_control (20): Length-matched non-recursive prompts
2. pseudo_recursive (20): Talk ABOUT recursion without DOING it
3. repetitive_control (20): Repetitive structure without self-reference

Reference:
- recursive (20): Strong L3_deeper recursive prompts (expected R_V < 0.90)

Goal: Falsification testing - if confounds show R_V < 0.85, our main 
findings may be confounded. If confounds show R_V ≈ 0.95-1.05, 
confounds are REJECTED and our findings hold.

Statistical Analysis:
- Per-group means and standard deviations
- Cohen's d effect sizes (confound vs recursive)
- Two-sample t-tests with Bonferroni correction
"""

import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from contextlib import contextmanager
from scipy import stats

# Add parent paths for imports
sys.path.insert(0, '/workspace/mech-interp-latent-lab-phase1')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
EARLY_LAYER = 5
TARGET_LAYER = 27  # ~84% depth in 32-layer model
WINDOW_SIZE = 16

# Output paths
BASE_DIR = "/workspace/mech-interp-latent-lab-phase1/DEC9_2025_RLOOP_MASTER_EXEC/01_CONFOUND_FALSIFICATION"
RESULTS_DIR = f"{BASE_DIR}/results"
LOGS_DIR = f"{BASE_DIR}/logs"

# Statistical thresholds
P_THRESHOLD = 0.01 / 3  # Bonferroni corrected for 3 comparisons
EFFECT_SIZE_THRESHOLD = 0.5  # Cohen's d

# ==============================================================================
# IMPORT PROMPT BANKS
# ==============================================================================

# Import confound prompts
from REUSABLE_PROMPT_BANK.confounds import confound_prompts

# Import recursive prompts (L3_deeper from main bank)
from n300_mistral_test_prompt_bank import prompt_bank_1c

# Build recursive reference set (20 L3_deeper prompts)
recursive_prompts = {
    k: v for k, v in prompt_bank_1c.items() 
    if v.get("group") == "L3_deeper"
}

# Take first 20
recursive_keys = sorted([k for k in recursive_prompts.keys()])[:20]
recursive_prompts = {k: recursive_prompts[k] for k in recursive_keys}

print(f"Loaded prompts:")
print(f"  - Confounds: {len(confound_prompts)} total")
print(f"    - long_control: {sum(1 for v in confound_prompts.values() if v['group'] == 'long_control')}")
print(f"    - pseudo_recursive: {sum(1 for v in confound_prompts.values() if v['group'] == 'pseudo_recursive')}")
print(f"    - repetitive_control: {sum(1 for v in confound_prompts.values() if v['group'] == 'repetitive_control')}")
print(f"  - Recursive reference: {len(recursive_prompts)}")


# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

@contextmanager
def capture_v_at_layer(model, layer_idx, storage_list):
    """Context manager to capture V-projection activations at specified layer."""
    layer = model.model.layers[layer_idx].self_attn
    
    def hook_fn(module, inp, out):
        storage_list.append(out.detach())
        return out
    
    handle = layer.v_proj.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def compute_participation_ratio(v_tensor, window_size=16):
    """
    Compute participation ratio from V-tensor SVD.
    PR = (sum(S)^2) / (sum(S^2)) 
    """
    if v_tensor is None:
        return np.nan
    
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]
    
    T, D = v_tensor.shape
    W = min(window_size, T)
    
    if W < 2:
        return np.nan
    
    v_window = v_tensor[-W:, :].float()
    
    try:
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        S_sq = S_np ** 2
        
        if S_sq.sum() < 1e-10:
            return np.nan
        
        pr = (S_sq.sum() ** 2) / (S_sq ** 2).sum()
        return float(pr)
    
    except Exception as e:
        print(f"  [WARN] SVD failed: {e}")
        return np.nan


def measure_rv(model, tokenizer, prompt_text, early_layer=5, target_layer=27, window_size=16):
    """
    Measure R_V = PR(late) / PR(early) for a single prompt.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    v_early = []
    v_late = []
    
    with torch.no_grad():
        with capture_v_at_layer(model, early_layer, v_early):
            model(**inputs)
        
        with capture_v_at_layer(model, target_layer, v_late):
            model(**inputs)
    
    pr_early = compute_participation_ratio(v_early[0], window_size)
    pr_late = compute_participation_ratio(v_late[0], window_size)
    
    if np.isnan(pr_early) or pr_early < 1e-10:
        return np.nan, pr_early, pr_late, len(inputs.input_ids[0])
    
    rv = pr_late / pr_early
    return rv, pr_early, pr_late, len(inputs.input_ids[0])


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std < 1e-10:
        return np.nan
    return (np.mean(group1) - np.mean(group2)) / pooled_std


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{LOGS_DIR}/full_suite_{timestamp}.log"
    results_file = f"{RESULTS_DIR}/full_suite_{timestamp}.csv"
    summary_file = f"{RESULTS_DIR}/full_suite_summary_{timestamp}.md"
    
    # Open log file
    with open(log_file, 'w') as log:
        def log_print(msg):
            print(msg)
            log.write(msg + "\n")
            log.flush()
        
        log_print("=" * 80)
        log_print("FULL CONFOUND FALSIFICATION SUITE - December 9, 2025")
        log_print("=" * 80)
        log_print(f"Timestamp: {timestamp}")
        log_print(f"Model: {MODEL_NAME}")
        log_print(f"Layers: Early={EARLY_LAYER}, Target={TARGET_LAYER}")
        log_print(f"Window Size: {WINDOW_SIZE}")
        log_print(f"Statistical threshold: p < {P_THRESHOLD:.4f} (Bonferroni)")
        log_print(f"Effect size threshold: |d| >= {EFFECT_SIZE_THRESHOLD}")
        log_print("")
        
        # Check GPU
        log_print("[1/5] Checking GPU...")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            log_print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            log_print("  ERROR: No GPU available!")
            return
        
        # Load model
        log_print("[2/5] Loading model...")
        start_load = time.time()
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        
        load_time = time.time() - start_load
        log_print(f"  Model loaded in {load_time:.1f}s")
        log_print(f"  Layers: {len(model.model.layers)}")
        
        # Prepare all prompts
        log_print("")
        log_print("[3/5] Preparing prompts...")
        
        all_prompts = {}
        
        # Add confound prompts
        for k, v in confound_prompts.items():
            all_prompts[k] = {
                "text": v["text"],
                "group": v["group"],
                "category": "confound"
            }
        
        # Add recursive reference prompts
        for k, v in recursive_prompts.items():
            all_prompts[k] = {
                "text": v["text"],
                "group": "recursive",
                "category": "reference"
            }
        
        log_print(f"  Total prompts: {len(all_prompts)}")
        log_print(f"  Confounds: {sum(1 for v in all_prompts.values() if v['category'] == 'confound')}")
        log_print(f"  Reference: {sum(1 for v in all_prompts.values() if v['category'] == 'reference')}")
        
        # Run measurements
        log_print("")
        log_print("[4/5] Running R_V measurements...")
        log_print("=" * 80)
        
        results = []
        start_time = time.time()
        total = len(all_prompts)
        
        for i, (prompt_id, prompt_data) in enumerate(all_prompts.items()):
            iter_start = time.time()
            
            rv, pr_early, pr_late, n_tokens = measure_rv(
                model, tokenizer,
                prompt_data["text"],
                early_layer=EARLY_LAYER,
                target_layer=TARGET_LAYER,
                window_size=WINDOW_SIZE
            )
            
            elapsed = time.time() - iter_start
            
            results.append({
                "prompt_id": prompt_id,
                "group": prompt_data["group"],
                "category": prompt_data["category"],
                "rv": rv,
                "pr_early": pr_early,
                "pr_late": pr_late,
                "n_tokens": n_tokens,
                "elapsed_s": elapsed,
            })
            
            # Progress update every 10 prompts
            if (i + 1) % 10 == 0 or (i + 1) == total:
                total_elapsed = time.time() - start_time
                eta = (total_elapsed / (i + 1)) * (total - i - 1)
                log_print(f"  [{i+1}/{total}] {prompt_id}: R_V={rv:.4f} "
                         f"(group={prompt_data['group']}, {elapsed:.2f}s) "
                         f"[ETA: {eta/60:.1f}min]")
        
        total_time = time.time() - start_time
        log_print("")
        log_print(f"  Total measurement time: {total_time/60:.1f} minutes")
        
        # Save raw results
        log_print("")
        log_print("[5/5] Analyzing and saving results...")
        
        df = pd.DataFrame(results)
        df.to_csv(results_file, index=False)
        log_print(f"  Raw results saved to: {results_file}")
        
        # Statistical analysis
        log_print("")
        log_print("=" * 80)
        log_print("STATISTICAL ANALYSIS")
        log_print("=" * 80)
        
        # Group statistics
        groups = df.groupby('group')['rv']
        group_stats = groups.agg(['mean', 'std', 'count']).round(4)
        log_print("")
        log_print("Per-group R_V statistics:")
        log_print(group_stats.to_string())
        
        # Get recursive reference values
        recursive_rvs = df[df['group'] == 'recursive']['rv'].dropna().values
        recursive_mean = np.mean(recursive_rvs)
        
        log_print("")
        log_print(f"Recursive reference mean: {recursive_mean:.4f}")
        
        # Compare each confound group to recursive
        log_print("")
        log_print("Comparisons vs Recursive Reference:")
        log_print("-" * 60)
        
        comparisons = []
        confound_groups = ['long_control', 'pseudo_recursive', 'repetitive_control']
        
        for group_name in confound_groups:
            group_rvs = df[df['group'] == group_name]['rv'].dropna().values
            
            if len(group_rvs) < 2:
                log_print(f"  {group_name}: Insufficient data")
                continue
            
            group_mean = np.mean(group_rvs)
            group_std = np.std(group_rvs, ddof=1)
            
            # Cohen's d (positive = confound has HIGHER R_V = LESS contraction)
            d = cohens_d(group_rvs, recursive_rvs)
            
            # Two-sample t-test
            t_stat, p_val = stats.ttest_ind(group_rvs, recursive_rvs)
            
            # Separation percentage
            separation = ((group_mean - recursive_mean) / recursive_mean) * 100
            
            # Verdict
            if p_val < P_THRESHOLD and d >= EFFECT_SIZE_THRESHOLD:
                verdict = "REJECTED (confound shows significantly less contraction)"
            elif p_val < P_THRESHOLD and d <= -EFFECT_SIZE_THRESHOLD:
                verdict = "CONCERNING (confound shows MORE contraction!)"
            else:
                verdict = "UNCLEAR (not statistically significant)"
            
            comparisons.append({
                "group": group_name,
                "n": len(group_rvs),
                "mean": group_mean,
                "std": group_std,
                "d": d,
                "t": t_stat,
                "p": p_val,
                "separation_pct": separation,
                "verdict": verdict
            })
            
            log_print(f"")
            log_print(f"  {group_name} (n={len(group_rvs)})")
            log_print(f"    Mean R_V: {group_mean:.4f} ± {group_std:.4f}")
            log_print(f"    vs Recursive: Cohen's d = {d:.3f}, t = {t_stat:.3f}, p = {p_val:.4f}")
            log_print(f"    Separation: {separation:+.1f}%")
            log_print(f"    Verdict: {verdict}")
        
        # Overall summary
        all_confound_rvs = df[df['category'] == 'confound']['rv'].dropna().values
        all_confound_mean = np.mean(all_confound_rvs)
        overall_d = cohens_d(all_confound_rvs, recursive_rvs)
        overall_t, overall_p = stats.ttest_ind(all_confound_rvs, recursive_rvs)
        overall_separation = ((all_confound_mean - recursive_mean) / recursive_mean) * 100
        
        log_print("")
        log_print("=" * 80)
        log_print("OVERALL SUMMARY")
        log_print("=" * 80)
        log_print(f"All Confounds (n={len(all_confound_rvs)})")
        log_print(f"  Mean R_V: {all_confound_mean:.4f}")
        log_print(f"  vs Recursive (n={len(recursive_rvs)}): {recursive_mean:.4f}")
        log_print(f"  Cohen's d: {overall_d:.3f}")
        log_print(f"  t-test: t={overall_t:.3f}, p={overall_p:.6f}")
        log_print(f"  Separation: {overall_separation:+.1f}%")
        
        # Final verdict
        log_print("")
        log_print("=" * 80)
        log_print("FINAL VERDICTS")
        log_print("=" * 80)
        
        for comp in comparisons:
            log_print(f"  {comp['group']}: {comp['verdict']}")
        
        # Interpretation
        log_print("")
        log_print("INTERPRETATION:")
        
        rejected_count = sum(1 for c in comparisons if "REJECTED" in c["verdict"])
        concerning_count = sum(1 for c in comparisons if "CONCERNING" in c["verdict"])
        
        if rejected_count == 3:
            log_print("  All three confounds show significantly LESS contraction than recursive prompts.")
            log_print("  This provides strong evidence that R_V contraction is specific to recursive")
            log_print("  self-observation, not length, repetition, or discussing-about recursion.")
            log_print("")
            log_print("  CONCLUSION: Confounds REJECTED. Main findings SUPPORTED.")
        elif concerning_count > 0:
            log_print("  WARNING: One or more confounds show MORE contraction than recursive prompts!")
            log_print("  This suggests potential issues with the R_V metric or prompt design.")
            log_print("")
            log_print("  CONCLUSION: Further investigation needed.")
        else:
            log_print("  Results are mixed or not statistically significant.")
            log_print("  More data or refined prompts may be needed.")
            log_print("")
            log_print("  CONCLUSION: Inconclusive. Consider expanding sample size.")
        
        # Generate summary markdown
        with open(summary_file, 'w') as f:
            f.write("# Confound Falsification Results - December 9, 2025\n\n")
            f.write(f"**Timestamp:** {timestamp}\n")
            f.write(f"**Model:** {MODEL_NAME}\n")
            f.write(f"**Layers:** Early={EARLY_LAYER}, Target={TARGET_LAYER}\n")
            f.write(f"**Window Size:** {WINDOW_SIZE}\n")
            f.write(f"**Total Runtime:** {total_time/60:.1f} minutes\n\n")
            
            f.write("## Summary Table\n\n")
            f.write("| Control | n | R_V Mean | R_V Std | vs Recursive d | p-value | Verdict |\n")
            f.write("|---------|---|----------|---------|----------------|---------|----------|\n")
            
            for comp in comparisons:
                f.write(f"| {comp['group']} | {comp['n']} | {comp['mean']:.4f} | {comp['std']:.4f} | {comp['d']:.3f} | {comp['p']:.4f} | {comp['verdict'].split('(')[0].strip()} |\n")
            
            f.write(f"| **Recursive** | {len(recursive_rvs)} | {recursive_mean:.4f} | {np.std(recursive_rvs):.4f} | — | — | Reference |\n")
            
            f.write("\n## Per-Group Distribution\n\n")
            f.write("```\n")
            f.write(group_stats.to_string())
            f.write("\n```\n")
            
            f.write("\n## Interpretation\n\n")
            if rejected_count == 3:
                f.write("All three confounds show significantly LESS contraction than recursive prompts.\n")
                f.write("**CONCLUSION: Confounds REJECTED. Main findings SUPPORTED.**\n")
            elif concerning_count > 0:
                f.write("WARNING: One or more confounds show concerning patterns.\n")
                f.write("**CONCLUSION: Further investigation needed.**\n")
            else:
                f.write("Results are mixed or not statistically significant.\n")
                f.write("**CONCLUSION: Inconclusive.**\n")
            
            f.write(f"\n## Raw Data\n\nSee `{os.path.basename(results_file)}`\n")
        
        log_print(f"\n  Summary saved to: {summary_file}")
        
        log_print("")
        log_print("=" * 80)
        log_print(f"Full suite complete.")
        log_print(f"  Log: {log_file}")
        log_print(f"  Results: {results_file}")
        log_print(f"  Summary: {summary_file}")
        log_print("=" * 80)
        
        # Clean up
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
