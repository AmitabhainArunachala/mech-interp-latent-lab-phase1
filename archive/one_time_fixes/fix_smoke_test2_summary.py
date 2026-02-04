#!/usr/bin/env python3
"""Fix corrupted summary.json from smoke test 2"""

import json
import pandas as pd
from pathlib import Path
from scipy import stats
import sys

# Find the sufficiency smoke test run directory
results_root = Path("results/phase1_mechanism/runs")
sufficiency_dirs = list(results_root.glob("*sufficiency_smoke_test*"))
if not sufficiency_dirs:
    print("No sufficiency smoke test directories found")
    sys.exit(1)

run_dir = sorted(sufficiency_dirs)[-1]
print(f"Fixing summary.json in: {run_dir}")

# Load CSV and metadata
df = pd.read_csv(run_dir / "mlp_sufficiency_test.csv")
metadata = json.load(open(run_dir / "metadata.json"))

# Compute stats
rv_restoration_pcts = df["rv_restoration_pct"].dropna().values
mode_deltas = df["mode_delta"].dropna().values

rv_stat = rv_pvalue = rv_significant = None
if len(rv_restoration_pcts) >= 3:
    t_stat, p_val = stats.ttest_1samp(rv_restoration_pcts, 0.0)
    rv_stat = float(t_stat)
    rv_pvalue = float(p_val)
    rv_significant = bool(p_val < 0.01 and np.mean(rv_restoration_pcts) > 0)

mode_stat = mode_pvalue = mode_significant = None
if len(mode_deltas) >= 3:
    t_stat, p_val = stats.ttest_1samp(mode_deltas, 0.0)
    mode_stat = float(t_stat)
    mode_pvalue = float(p_val)
    mode_significant = bool(p_val < 0.01)

# Build summary
summary = {
    "experiment": "mlp_sufficiency_test",
    "layer": 0,
    "n_pairs": len(df),
    "mode_score_m": float(df["mode_baseline"].mean()) if df["mode_baseline"].notna().any() else None,
    "mode_score_m_delta": float(df["mode_delta"].mean()) if df["mode_delta"].notna().any() else None,
    "mode_t_statistic": mode_stat,
    "mode_pvalue": mode_pvalue,
    "mode_significant": mode_significant,
    "rv": float(df["rv_baseline"].mean()),
    "rv_baseline_mean": float(df["rv_baseline"].mean()),
    "rv_recursive_mean": float(df["rv_recursive"].mean()),
    "rv_patched_mean": float(df["rv_patched"].mean()),
    "rv_restoration_pct": float(df["rv_restoration_pct"].mean()),
    "rv_restoration_pct_mean": float(df["rv_restoration_pct"].mean()),
    "rv_restoration_pct_std": float(df["rv_restoration_pct"].std()),
    "rv_t_statistic": rv_stat,
    "rv_pvalue": rv_pvalue,
    "rv_significant": rv_significant,
    "eval_window": 16,
    "intervention_scope": "last_16",
    "behavior_metric": "mode_score_m",
    **metadata,
}

# Add verdict
rv_restoration_pct_mean = summary["rv_restoration_pct_mean"]
if rv_significant and rv_restoration_pct_mean > 50.0:
    summary["verdict"] = f"L0 MLP is SUFFICIENT - Patching restores {rv_restoration_pct_mean:.1f}% of contraction"
elif rv_significant and rv_restoration_pct_mean > 0:
    summary["verdict"] = f"L0 MLP is PARTIALLY SUFFICIENT - Patching restores {rv_restoration_pct_mean:.1f}% of contraction"
else:
    summary["verdict"] = f"L0 MLP is NOT SUFFICIENT - Patching does not restore contraction"

# Write fixed summary
with open(run_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("✅ Fixed summary.json")
print(f"   Keys: {len(summary)}")
print(f"   Has git_commit: {'git_commit' in summary}")
print(f"   Has prompt_bank_version: {'prompt_bank_version' in summary}")
print(f"   Has mode_score_m: {'mode_score_m' in summary}")

# Append to RUN_INDEX
sys.path.insert(0, '.')
from src.utils.run_metadata import append_to_run_index
append_to_run_index(run_dir, summary)
print("✅ RUN_INDEX updated")


