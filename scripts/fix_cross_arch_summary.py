#!/usr/bin/env python3
"""Fix summary.json for cross-architecture validation runs."""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.stats as stats

def regenerate_summary(csv_path, run_dir):
    """Regenerate summary.json from CSV."""
    df = pd.read_csv(csv_path)
    
    # Summary by family
    summary_by_family = {}
    for family_name in df["family"].unique():
        family_df = df[df["family"] == family_name].copy()
        rv_values = family_df["rv"].dropna()
        if len(rv_values) > 0:
            summary_by_family[family_name] = {
                "rv_mean": float(rv_values.mean()),
                "rv_std": float(rv_values.std()),
                "rv_min": float(rv_values.min()),
                "rv_max": float(rv_values.max()),
                "n_prompts": len(family_df["prompt_idx"].unique()),
                "n_valid": len(rv_values),
                "meets_criteria": float(rv_values.mean()) < 0.55,
            }
    
    # Summary by window
    summary_by_window = {}
    for window_size in sorted(df["window_size"].unique()):
        window_df = df[df["window_size"] == window_size].copy()
        rv_values = window_df["rv"].dropna()
        if len(rv_values) > 0:
            summary_by_window[int(window_size)] = {
                "rv_mean": float(rv_values.mean()),
                "rv_std": float(rv_values.std()),
                "rv_min": float(rv_values.min()),
                "rv_max": float(rv_values.max()),
                "n_samples": len(rv_values),
            }
    
    # Compare recursive vs non-recursive
    recursive_families = [
        "recursive_self_reference",
        "recursive_no_introspection_vocab",
        "nonsense_recursion",
    ]
    non_recursive_families = [
        "abstract_non_recursive",
        "same_vocab_different_semantics",
        "introspective_concrete",
    ]
    
    recursive_rv = df[df["family"].isin(recursive_families)]["rv"].dropna()
    non_recursive_rv = df[df["family"].isin(non_recursive_families)]["rv"].dropna()
    
    comparison_stats = {}
    if len(recursive_rv) > 0 and len(non_recursive_rv) > 0:
        t_stat, p_value = stats.ttest_ind(recursive_rv, non_recursive_rv)
        pooled_std = np.sqrt((recursive_rv.std()**2 + non_recursive_rv.std()**2) / 2)
        cohens_d = (non_recursive_rv.mean() - recursive_rv.mean()) / pooled_std if pooled_std > 0 else 0
        
        comparison_stats = {
            "recursive_mean": float(recursive_rv.mean()),
            "recursive_std": float(recursive_rv.std()),
            "non_recursive_mean": float(non_recursive_rv.mean()),
            "non_recursive_std": float(non_recursive_rv.std()),
            "delta_mean": float(non_recursive_rv.mean() - recursive_rv.mean()),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "recursive_meets_criteria": float(recursive_rv.mean()) < 0.55,
            "non_recursive_meets_criteria": float(non_recursive_rv.mean()) < 0.55,
        }
    
    summary = {
        "experiment": "cross_architecture_validation",
        "model": "mistralai/Mistral-7B-v0.1",
        "n_families": len(df["family"].unique()),
        "window_sizes": sorted([int(w) for w in df["window_size"].unique()]),
        "early_layer": 5,
        "late_layer": 27,
        "by_family": summary_by_family,
        "by_window": summary_by_window,
        "comparison": comparison_stats,
        "artifacts": {
            "csv": str(csv_path),
        },
    }
    
    # Save summary
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Summary regenerated: {summary_path}")
    return summary

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 fix_cross_arch_summary.py <run_dir>")
        sys.exit(1)
    
    run_dir = Path(sys.argv[1])
    csv_path = run_dir / "cross_architecture_validation.csv"
    
    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        sys.exit(1)
    
    summary = regenerate_summary(csv_path, run_dir)
    
    print("\n=== Summary ===")
    print(f"\nBy Family:")
    for family_name, stats in summary["by_family"].items():
        print(f"  {family_name}: R_V = {stats['rv_mean']:.4f} ± {stats['rv_std']:.4f} "
              f"{'✓' if stats['meets_criteria'] else '✗'}")
    
    if summary["comparison"]:
        comp = summary["comparison"]
        print(f"\nRecursive vs Non-Recursive:")
        print(f"  Recursive: {comp['recursive_mean']:.4f}")
        print(f"  Non-recursive: {comp['non_recursive_mean']:.4f}")
        print(f"  Delta: {comp['delta_mean']:.4f}")
        print(f"  p-value: {comp['p_value']:.6f}")
        print(f"  Cohen's d: {comp['cohens_d']:.3f}")
