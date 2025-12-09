#!/usr/bin/env python3
"""
DEC9 Confound Falsification Tests
==================================

Tests the three critical confounds:
1. Repetitive control (induction head confound)
2. Pseudo-recursive control (semantic content confound)
3. Long control (length confound)

Expected runtime: ~2-3 hours for all 60 prompts
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import contextmanager
from tqdm import tqdm
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"  # Or use Llama-3-8B-Instruct
EARLY_LAYER = 5
TARGET_LAYER = 27  # Use 24 for Llama-3-8B
WINDOW_SIZE = 16

print(f"Device: {DEVICE}")
print(f"Model: {MODEL_NAME}")
print(f"Target layer: {TARGET_LAYER}")
print()

# =============================================================================
# HELPER FUNCTIONS (from validated experiments)
# =============================================================================

@contextmanager
def capture_v_at_layer(model, layer_idx, storage_list):
    """Context manager to capture V activations at a specific layer."""
    layer = model.model.layers[layer_idx].self_attn
    
    def hook_fn(module, inp, out):
        storage_list.append(out.detach())
        return out
    
    handle = layer.v_proj.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def compute_metrics_fast(v_tensor, window_size=WINDOW_SIZE):
    """Compute Effective Rank and Participation Ratio via SVD."""
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


def measure_rv_for_prompt(model, tokenizer, prompt, early_layer=EARLY_LAYER, target_layer=TARGET_LAYER):
    """Measure R_V = PR(late) / PR(early) for a single prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    v_early_list = []
    v_late_list = []
    
    with torch.no_grad():
        with capture_v_at_layer(model, early_layer, v_early_list):
            with capture_v_at_layer(model, target_layer, v_late_list):
                _ = model(**inputs)
    
    v_early = v_early_list[0][0] if v_early_list else None
    v_late = v_late_list[0][0] if v_late_list else None
    
    er_early, pr_early = compute_metrics_fast(v_early)
    er_late, pr_late = compute_metrics_fast(v_late)
    
    r_v = pr_late / pr_early if (pr_early and pr_early > 0) else np.nan
    
    return r_v, pr_early, pr_late


# =============================================================================
# LOAD CONFOUND PROMPTS
# =============================================================================

def load_confound_prompts():
    """Load confound prompts from confounds.py"""
    try:
        from confounds import confound_prompts
        return confound_prompts
    except ImportError:
        print("ERROR: Could not import confounds.py")
        print("Make sure REUSABLE_PROMPT_BANK/confounds.py is in the same directory")
        sys.exit(1)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_confound_tests(model, tokenizer, confound_prompts, output_dir="results"):
    """
    Run R_V measurements on all confound prompts.
    
    Tests:
    1. Repetitive control (n=20)
    2. Pseudo-recursive control (n=20)
    3. Long control (n=20)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("DEC9 CONFOUND FALSIFICATION TESTS")
    print("="*80)
    print(f"Total prompts: {len(confound_prompts)}")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print()
    
    # Group prompts by type
    groups = {}
    for prompt_id, prompt_data in confound_prompts.items():
        group = prompt_data["group"]
        if group not in groups:
            groups[group] = []
        groups[group].append((prompt_id, prompt_data))
    
    print("Groups found:")
    for group, prompts in groups.items():
        print(f"  {group}: {len(prompts)} prompts")
    print()
    
    # Run measurements
    all_results = []
    
    for group_name, prompts in groups.items():
        print(f"\n{'='*80}")
        print(f"Testing: {group_name} (n={len(prompts)})")
        print(f"{'='*80}\n")
        
        for prompt_id, prompt_data in tqdm(prompts, desc=group_name):
            prompt_text = prompt_data["text"]
            
            # Skip if too short
            tokens = tokenizer.encode(prompt_text)
            if len(tokens) < WINDOW_SIZE:
                print(f"⚠️  Skipping {prompt_id}: too short ({len(tokens)} tokens)")
                continue
            
            # Measure R_V
            r_v, pr_early, pr_late = measure_rv_for_prompt(model, tokenizer, prompt_text)
            
            all_results.append({
                "prompt_id": prompt_id,
                "group": group_name,
                "r_v": r_v,
                "pr_early": pr_early,
                "pr_late": pr_late,
                "token_count": len(tokens),
                "expected_rv_min": prompt_data.get("expected_rv_range", [0.95, 1.05])[0],
                "expected_rv_max": prompt_data.get("expected_rv_range", [0.95, 1.05])[1]
            })
        
        # Group summary
        group_results = [r for r in all_results if r["group"] == group_name]
        rv_values = [r["r_v"] for r in group_results if not np.isnan(r["r_v"])]
        
        if rv_values:
            print(f"\n{group_name} Summary:")
            print(f"  Mean R_V: {np.mean(rv_values):.3f} ± {np.std(rv_values):.3f}")
            print(f"  Min R_V:  {np.min(rv_values):.3f}")
            print(f"  Max R_V:  {np.max(rv_values):.3f}")
            print(f"  Expected: 0.95-1.05 (no contraction)")
            
            if np.mean(rv_values) < 0.90:
                print(f"  ⚠️  UNEXPECTED CONTRACTION DETECTED!")
            elif np.mean(rv_values) >= 0.95:
                print(f"  ✅ No contraction (confound rejected)")
        
        # Memory cleanup
        torch.cuda.empty_cache()
    
    # Save results
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/confound_tests_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}\n")
    
    # Summary by group
    for group_name in groups.keys():
        group_data = df[df["group"] == group_name]
        rv_values = group_data["r_v"].dropna()
        
        if len(rv_values) > 0:
            mean_rv = rv_values.mean()
            std_rv = rv_values.std()
            n = len(rv_values)
            
            print(f"{group_name}:")
            print(f"  n = {n}")
            print(f"  R_V = {mean_rv:.3f} ± {std_rv:.3f}")
            
            # Statistical test vs 1.0
            from scipy import stats
            t_stat, p_val = stats.ttest_1samp(rv_values, 1.0)
            print(f"  t-test vs 1.0: t={t_stat:.3f}, p={p_val:.4f}")
            
            # Verdict
            if mean_rv >= 0.95 and p_val > 0.05:
                print(f"  ✅ CONFOUND REJECTED: No significant contraction")
            elif mean_rv < 0.85:
                print(f"  ⚠️  CONFOUND DETECTED: Significant contraction (R_V < 0.85)")
            else:
                print(f"  ⚠️  UNCLEAR: Moderate contraction (needs further investigation)")
            print()
    
    print(f"Results saved: {output_file}")
    
    return df


# =============================================================================
# COMPARISON WITH RECURSIVE PROMPTS
# =============================================================================

def load_recursive_baseline_comparison():
    """
    Load recursive and baseline R_V values from prior experiments for comparison.
    
    If you have existing results, add them here. Otherwise, we'll just report
    the confound values standalone.
    """
    # Placeholder - add your known values here
    comparison_data = {
        "recursive_mean": 0.82,  # From DEC3/DEC7 experiments
        "recursive_std": 0.10,
        "baseline_mean": 0.95,
        "baseline_std": 0.08
    }
    return comparison_data


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("DEC9 CONFOUND FALSIFICATION EXPERIMENT")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load model
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None
    )
    model.eval()
    
    if DEVICE == "cpu":
        model = model.to(DEVICE)
    
    print("✅ Model loaded\n")
    
    # Load prompts
    print("Loading confound prompts...")
    confound_prompts = load_confound_prompts()
    print(f"✅ Loaded {len(confound_prompts)} confound prompts\n")
    
    # Run tests
    results_df = run_confound_tests(model, tokenizer, confound_prompts)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review results CSV")
    print("2. Compare to recursive/baseline distributions")
    print("3. Update DEC9_CONFOUND_FALSIFICATION_RESULTS.md")
    print()


if __name__ == "__main__":
    main()

