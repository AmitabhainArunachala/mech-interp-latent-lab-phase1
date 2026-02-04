#!/usr/bin/env python3
"""Fix corrupted summary.json from smoke test"""

import json
import pandas as pd
from pathlib import Path
from scipy import stats
import sys

# Find the smoke test run directory
results_root = Path("results/phase1_mechanism/runs")
smoke_dirs = list(results_root.glob("*smoke_test*"))
if not smoke_dirs:
    print("No smoke test directories found")
    sys.exit(1)

run_dir = sorted(smoke_dirs)[-1]
print(f"Fixing summary.json in: {run_dir}")

# Load CSV and metadata
df = pd.read_csv(run_dir / "mlp_ablation_necessity.csv")
metadata = json.load(open(run_dir / "metadata.json"))

# Compute stats
rv_deltas = df["rv_delta"].dropna().values
mode_deltas = df["mode_delta"].dropna().values

rv_stat = rv_pvalue = rv_significant = None
if len(rv_deltas) >= 3:
    t_stat, p_val = stats.ttest_1samp(rv_deltas, 0.0)
    rv_stat = float(t_stat)
    rv_pvalue = float(p_val)
    rv_significant = bool(p_val < 0.01)

mode_stat = mode_pvalue = mode_significant = None
if len(mode_deltas) >= 3:
    t_stat, p_val = stats.ttest_1samp(mode_deltas, 0.0)
    mode_stat = float(t_stat)
    mode_pvalue = float(p_val)
    mode_significant = bool(p_val < 0.01)

# Build summary
summary = {
    "experiment": "mlp_ablation_necessity",
    "layer": 0,
    "n_pairs": len(df),
    "mode_score_m": float(df["mode_baseline"].mean()) if df["mode_baseline"].notna().any() else None,
    "mode_score_m_delta": float(df["mode_delta"].mean()) if df["mode_delta"].notna().any() else None,
    "mode_score_m_ablated": float(df["mode_ablated"].mean()) if df["mode_ablated"].notna().any() else None,
    "mode_t_statistic": mode_stat,
    "mode_pvalue": mode_pvalue,
    "mode_significant": mode_significant,
    "rv": float(df["rv_baseline"].mean()),
    "rv_baseline_mean": float(df["rv_baseline"].mean()),
    "rv_baseline_std": float(df["rv_baseline"].std()),
    "rv_ablated_mean": float(df["rv_ablated"].mean()),
    "rv_ablated_std": float(df["rv_ablated"].std()),
    "rv_delta_mean": float(df["rv_delta"].mean()),
    "rv_delta_std": float(df["rv_delta"].std()),
    "rv_t_statistic": rv_stat,
    "rv_pvalue": rv_pvalue,
    "rv_significant": rv_significant,
    "eval_window": 16,
    "intervention_scope": "all_tokens",
    "behavior_metric": "mode_score_m",
    "coherence_mean": float(df["coherence"].mean()),
    "recursion_score_mean": float(df["recursion_score"].mean()),
    **metadata,
}

# Add verdict
rv_delta_mean = summary["rv_delta_mean"]
if rv_significant and rv_delta_mean > 0.1:
    summary["verdict"] = f"L0 MLP is NOT necessary - R_V contraction persists (delta: +{rv_delta_mean:.3f})"
elif rv_significant and rv_delta_mean < -0.1:
    summary["verdict"] = f"L0 MLP is NECESSARY - R_V contraction disappears (delta: {rv_delta_mean:.3f})"
else:
    summary["verdict"] = f"L0 MLP has minimal effect - inconclusive (delta: {rv_delta_mean:.3f})"

# Write fixed summary
with open(run_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("✅ Fixed summary.json")
print(f"   Keys: {len(summary)}")
print(f"   Has git_commit: {'git_commit' in summary}")
print(f"   Has prompt_bank_version: {'prompt_bank_version' in summary}")
print(f"   Has mode_score_m: {'mode_score_m' in summary}")

# Append to RUN_INDEX
from src.utils.run_metadata import append_to_run_index
append_to_run_index(run_dir, summary)
print("✅ RUN_INDEX updated")


