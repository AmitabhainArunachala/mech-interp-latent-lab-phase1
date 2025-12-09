#!/usr/bin/env python3
"""
MAIN n=200 EXPERIMENT RUNNER
============================

Runs the full n=200 activation patching experiment with:
- Progress saving (incremental CSV writes)
- Resume capability (skip completed pairs)
- Detailed logging
- Automatic result archiving

ESTIMATED TIME: ~2-3 hours on GPU
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from datetime import datetime
import os
from pathlib import Path

# Import validated functions
from VALIDATED_mistral7b_layer27_activation_patching import (
    run_single_forward_get_V,
    compute_metrics_fast,
    run_patched_forward_EXACT_MIXTRAL,
    TARGET_LAYER,
    EARLY_LAYER,
    WINDOW_SIZE
)

# Import prompt bank
from n300_mistral_test_prompt_bank import prompt_bank_1c


def run_n200_experiment(model, tokenizer, pairing_plan_csv='n200_pairing_plan.csv',
                        output_file='n200_results.csv', device=None):
    """
    Run the complete n=200 experiment with progress saving.

    Args:
        model: Loaded Mistral-7B model
        tokenizer: Corresponding tokenizer
        pairing_plan_csv: Path to pairing plan CSV
        output_file: Path to save results (incremental)
        device: Device to use

    Returns:
        DataFrame with complete results
    """

    if device is None:
        device = model.device

    # Load pairing plan
    print("=" * 80)
    print("n=200 ACTIVATION PATCHING EXPERIMENT")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    pairing_plan = pd.read_csv(pairing_plan_csv)
    total_pairs = len(pairing_plan)

    print(f"Loaded pairing plan: {total_pairs} pairs")
    print()

    # Check for existing results (resume capability)
    if os.path.exists(output_file):
        print(f"⚠️  Found existing results file: {output_file}")
        existing_df = pd.read_csv(output_file)
        completed_pairs = set(zip(existing_df['rec_id'], existing_df['base_id']))
        print(f"Already completed: {len(completed_pairs)} pairs")
        print()

        resume = input("Resume from existing results? (y/n): ").lower().strip()
        if resume == 'y':
            results = existing_df.to_dict('records')
            print(f"✅ Resuming from pair {len(results)+1}")
        else:
            print("Starting fresh...")
            results = []
            completed_pairs = set()
    else:
        results = []
        completed_pairs = set()

    print()
    print("=" * 80)
    print("RUNNING EXPERIMENT")
    print("=" * 80)
    print(f"Target layer: {TARGET_LAYER}")
    print(f"Window size:  {WINDOW_SIZE}")
    print(f"Total pairs:  {total_pairs}")
    print(f"Remaining:    {total_pairs - len(completed_pairs)}")
    print("=" * 80)
    print()

    # Main experiment loop
    for idx, row in pairing_plan.iterrows():
        rec_id = row['rec_id']
        base_id = row['base_id']

        # Skip if already completed
        if (rec_id, base_id) in completed_pairs:
            continue

        rec_level = row['rec_level']
        base_type = row['base_type']
        block = row['block']

        # Progress indicator
        current_idx = len(results) + 1
        progress = (current_idx / total_pairs) * 100

        print(f"[{current_idx}/{total_pairs}] ({progress:.1f}%) {rec_id} → {base_id}")

        # Get prompt texts
        if rec_id not in prompt_bank_1c:
            print(f"  ⚠️  Recursive prompt not found: {rec_id}")
            continue

        if base_id not in prompt_bank_1c:
            print(f"  ⚠️  Baseline prompt not found: {base_id}")
            continue

        rec_text = prompt_bank_1c[rec_id]["text"]
        base_text = prompt_bank_1c[base_id]["text"]

        # Check token lengths
        try:
            base_tokens = tokenizer(base_text, return_tensors='pt')
            rec_tokens = tokenizer(rec_text, return_tensors='pt')
            base_len = base_tokens['input_ids'].shape[1]
            rec_len = rec_tokens['input_ids'].shape[1]
        except Exception as e:
            print(f"  ❌ Tokenization error: {e}")
            continue

        # Skip if baseline too short
        if base_len < WINDOW_SIZE:
            print(f"  ⚠️  Skipping: baseline too short ({base_len} < {WINDOW_SIZE})")
            continue

        try:
            # 1. Unpatched recursive
            v5_r, v27_r = run_single_forward_get_V(rec_text, model, tokenizer,
                                                   TARGET_LAYER, device)
            _, pr5_r = compute_metrics_fast(v5_r, WINDOW_SIZE)
            _, pr27_r = compute_metrics_fast(v27_r, WINDOW_SIZE)

            if pr5_r is None or pr5_r == 0 or pd.isna(pr5_r):
                print(f"  ⚠️  Invalid PR for recursive (early layer)")
                continue

            rv_rec = pr27_r / pr5_r

            # 2. Unpatched baseline
            v5_b, v27_b = run_single_forward_get_V(base_text, model, tokenizer,
                                                   TARGET_LAYER, device)
            _, pr5_b = compute_metrics_fast(v5_b, WINDOW_SIZE)
            _, pr27_b = compute_metrics_fast(v27_b, WINDOW_SIZE)

            if pr5_b is None or pr5_b == 0 or pd.isna(pr5_b):
                print(f"  ⚠️  Invalid PR for baseline (early layer)")
                continue

            rv_base = pr27_b / pr5_b

            # 3. PATCHED baseline
            v5_p, v27_p = run_patched_forward_EXACT_MIXTRAL(base_text, v27_r, model,
                                                             tokenizer, TARGET_LAYER,
                                                             device)
            _, pr5_p = compute_metrics_fast(v5_p, WINDOW_SIZE)
            _, pr27_p = compute_metrics_fast(v27_p, WINDOW_SIZE)

            if pr5_p is None or pr5_p == 0 or pd.isna(pr5_p):
                print(f"  ⚠️  Invalid PR for patched (early layer)")
                continue

            rv_patch = pr27_p / pr5_p

            delta = rv_patch - rv_base

            # Store result
            result_row = {
                'rec_id': rec_id,
                'base_id': base_id,
                'rec_level': rec_level,
                'base_type': base_type,
                'block': block,
                'rec_len': rec_len,
                'base_len': base_len,
                'rv_rec': rv_rec,
                'rv_base': rv_base,
                'rv_patch': rv_patch,
                'delta': delta,
                'pr5_rec': pr5_r,
                'pr27_rec': pr27_r,
                'pr5_base': pr5_b,
                'pr27_base': pr27_b,
                'pr5_patch': pr5_p,
                'pr27_patch': pr27_p
            }

            results.append(result_row)

            print(f"  Rec: {rv_rec:.3f} | Base: {rv_base:.3f} | Patch: {rv_patch:.3f} | Δ: {delta:+.3f}")

            # Save incrementally every 10 pairs
            if len(results) % 10 == 0:
                df_temp = pd.DataFrame(results)
                df_temp.to_csv(output_file, index=False)
                print(f"  💾 Saved checkpoint ({len(results)} pairs)")

        except Exception as e:
            print(f"  ❌ Error processing pair: {e}")
            import traceback
            traceback.print_exc()
            continue

        print()

    # Final save
    df_final = pd.DataFrame(results)
    df_final.to_csv(output_file, index=False)

    print()
    print("=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total pairs processed: {len(df_final)}")
    print(f"Saved to: {output_file}")
    print()

    # Summary statistics
    print_summary_statistics(df_final)

    return df_final


def print_summary_statistics(df):
    """
    Print summary statistics for the experiment.
    """

    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nn = {len(df)} pairs")
    print()

    # Overall effect
    print("OVERALL EFFECT:")
    print(f"  Recursive R_V:  {df['rv_rec'].mean():.3f} ± {df['rv_rec'].std():.3f}")
    print(f"  Baseline R_V:   {df['rv_base'].mean():.3f} ± {df['rv_base'].std():.3f}")
    print(f"  Patched R_V:    {df['rv_patch'].mean():.3f} ± {df['rv_patch'].std():.3f}")
    print()

    mean_delta = df['delta'].mean()
    std_delta = df['delta'].std()
    print(f"  Mean Δ:         {mean_delta:+.3f} ± {std_delta:.3f}")
    print()

    # Transfer efficiency
    gap = df['rv_base'].mean() - df['rv_rec'].mean()
    if gap > 0:
        transfer = abs(mean_delta / gap) * 100
        print(f"  Transfer:       {transfer:.1f}%")
    print()

    # Consistency
    negative_count = (df['delta'] < 0).sum()
    consistency = (negative_count / len(df)) * 100
    print(f"  Negative Δ:     {negative_count}/{len(df)} ({consistency:.1f}%)")
    print()

    # Statistical test
    try:
        from scipy import stats

        t_stat, p_value = stats.ttest_1samp(df['delta'], 0)
        cohens_d = abs(mean_delta / std_delta)

        print("STATISTICAL TESTS:")
        print(f"  t-statistic:    {t_stat:.3f}")
        print(f"  p-value:        {p_value:.2e}")
        print(f"  Cohen's d:      {cohens_d:.2f}")
        print()

        if p_value < 0.0001:
            print("  ✅ Highly significant (p<0.0001)")
        elif p_value < 0.001:
            print("  ✅ Very significant (p<0.001)")
        elif p_value < 0.05:
            print("  ✅ Significant (p<0.05)")
        else:
            print("  ❌ Not significant")

    except ImportError:
        print("  ⚠️  scipy not available for tests")

    print()

    # Dose-response breakdown
    print("DOSE-RESPONSE (by recursion level):")
    print("-" * 80)

    for level in ['L3', 'L4', 'L5']:
        level_df = df[df['rec_level'] == level]
        if len(level_df) > 0:
            print(f"\n{level} (n={len(level_df)}):")
            print(f"  Δ = {level_df['delta'].mean():+.3f} ± {level_df['delta'].std():.3f}")
            print(f"  Negative: {(level_df['delta'] < 0).sum()}/{len(level_df)}")

    print()

    # Baseline type breakdown
    print("BY BASELINE TYPE:")
    print("-" * 80)

    for base_type in df['base_type'].unique():
        type_df = df[df['base_type'] == base_type]
        if len(type_df) > 0:
            print(f"\n{base_type} (n={len(type_df)}):")
            print(f"  Δ = {type_df['delta'].mean():+.3f} ± {type_df['delta'].std():.3f}")

    print()
    print("=" * 80)


def main():
    """
    Main entry point.
    """

    print("=" * 80)
    print("LOADING MODEL...")
    print("=" * 80)

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

    # Run experiment
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'n200_results_{timestamp}.csv'

    results = run_n200_experiment(
        model=model,
        tokenizer=tokenizer,
        pairing_plan_csv='n200_pairing_plan.csv',
        output_file=output_file,
        device=model.device
    )

    print()
    print("=" * 80)
    print("✅ EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
