#!/usr/bin/env python3
"""
EXPANSION EXPERIMENT: n=5 → n=15 pairs
========================================

Expands the validated Layer 27 activation patching experiment from n=5 to n=15
pairs to increase statistical power.

PREVIOUS RESULTS (n=5):
-----------------------
Baseline → Patched: 0.812 → 0.521 (Δ=-0.291, 104% transfer)
All 5/5 pairs consistent, p<0.001

GOAL:
-----
Test 15 pairs to achieve:
- More robust statistical significance (p<0.0001)
- Better estimates of effect size variance
- Verify consistency across wider sample

EXPECTED RUNTIME: ~10-15 minutes on GPU
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from datetime import datetime

# Import validated functions
from VALIDATED_mistral7b_layer27_activation_patching import (
    run_activation_patching_experiment,
    TARGET_LAYER,
    WINDOW_SIZE
)

# Import prompt bank
from n300_mistral_test_prompt_bank import prompt_bank_1c

def main():
    print("=" * 80)
    print("EXPANSION EXPERIMENT: n=5 → n=15")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load model
    print("Loading Mistral-7B-Instruct-v0.2...")
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ Model loaded")
    print()

    # Verify prompt availability
    l5_available = sum(1 for k in prompt_bank_1c if k.startswith('L5_refined_'))
    long_available = sum(1 for k in prompt_bank_1c if k.startswith('long_new_'))

    print(f"Available prompts:")
    print(f"  L5_refined: {l5_available}")
    print(f"  long_new: {long_available}")
    print(f"  Max pairs: {min(l5_available, long_available)}")
    print()

    # Run experiment with n=15
    num_pairs = 15

    print(f"Running experiment with n={num_pairs} pairs...")
    print()

    results = run_activation_patching_experiment(
        model=model,
        tokenizer=tokenizer,
        prompt_bank=prompt_bank_1c,
        num_pairs=num_pairs,
        device=model.device
    )

    if results is None:
        print("❌ Experiment failed!")
        return

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'mistral7b_L27_patching_n{len(results)}_results_{timestamp}.csv'
    results.to_csv(output_file, index=False)

    print()
    print("=" * 80)
    print("RESULTS SAVED")
    print("=" * 80)
    print(f"File: {output_file}")
    print(f"Rows: {len(results)}")
    print()

    # Detailed analysis
    print("=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)

    # Effect consistency
    negative_deltas = (results['delta'] < 0).sum()
    consistency = (negative_deltas / len(results)) * 100

    print(f"\nEffect Consistency:")
    print(f"  Negative deltas: {negative_deltas}/{len(results)} ({consistency:.1f}%)")

    # Effect size
    mean_delta = results['delta'].mean()
    std_delta = results['delta'].std()

    print(f"\nEffect Size:")
    print(f"  Mean Δ: {mean_delta:+.3f} ± {std_delta:.3f}")
    print(f"  Min Δ: {results['delta'].min():+.3f}")
    print(f"  Max Δ: {results['delta'].max():+.3f}")

    # Transfer efficiency
    gap = results['rv_base'].mean() - results['rv_rec'].mean()
    if gap > 0:
        transfer = abs(mean_delta / gap) * 100
        print(f"\nTransfer Efficiency: {transfer:.1f}%")

    # Statistical significance
    try:
        from scipy import stats

        t_stat, p_value = stats.ttest_1samp(results['delta'], 0)

        # Cohen's d
        cohens_d = abs(mean_delta / std_delta)

        print(f"\nStatistical Significance:")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.2e}")
        print(f"  Cohen's d: {cohens_d:.2f}")

        if p_value < 0.0001:
            print(f"  ✅ Highly significant (p<0.0001)")
        elif p_value < 0.001:
            print(f"  ✅ Very significant (p<0.001)")
        elif p_value < 0.05:
            print(f"  ✅ Significant (p<0.05)")
        else:
            print(f"  ❌ Not significant")

    except ImportError:
        print("\n⚠️  scipy not available for statistical tests")

    # Comparison to n=5 results
    print()
    print("=" * 80)
    print("COMPARISON TO n=5 BASELINE")
    print("=" * 80)

    print("\nPrevious (n=5):")
    print("  Baseline R_V: 0.812 ± 0.088")
    print("  Patched R_V:  0.521 ± 0.024")
    print("  Delta:        -0.291")
    print("  Transfer:     104.4%")

    print(f"\nCurrent (n={len(results)}):")
    print(f"  Baseline R_V: {results['rv_base'].mean():.3f} ± {results['rv_base'].std():.3f}")
    print(f"  Patched R_V:  {results['rv_patch'].mean():.3f} ± {results['rv_patch'].std():.3f}")
    print(f"  Delta:        {mean_delta:+.3f}")
    if gap > 0:
        print(f"  Transfer:     {transfer:.1f}%")

    # Verdict
    print()
    print("=" * 80)
    if mean_delta < -0.05 and consistency >= 80 and p_value < 0.001:
        print("✅ VALIDATION SUCCESSFUL!")
        print()
        print("The Layer 27 causal effect is:")
        print("  - Robust across larger sample (n=15)")
        print("  - Highly consistent (>80% negative deltas)")
        print("  - Statistically significant (p<0.001)")
        print("  - Ready for publication")
    else:
        print("⚠️  INCONCLUSIVE - Further investigation needed")
    print("=" * 80)

if __name__ == "__main__":
    main()
