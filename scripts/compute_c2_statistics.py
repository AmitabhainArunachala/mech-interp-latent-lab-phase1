#!/usr/bin/env python3
"""Compute comprehensive statistics for C2+R_V results."""

import json
from pathlib import Path
import pandas as pd
from scipy import stats
import numpy as np

# Find latest run
runs = sorted(Path("results/phase1_mechanism/runs").glob("*_c2_rv_measurement"))
if not runs:
    print("No runs found")
    exit(1)

latest = runs[-1]
print(f"Processing: {latest.name}")

if not (latest / "c2_rv_measurement.csv").exists():
    print("CSV not found")
    exit(1)

df = pd.read_csv(latest / "c2_rv_measurement.csv")
print(f"Loaded CSV: {len(df)} rows")

# Compute statistics
baseline_df = df[df["config"] == "baseline"]
c2_df = df[df["config"] == "c2_full"]

merged = baseline_df.merge(c2_df, on="prompt_idx", suffixes=("_baseline", "_c2"))
rv_baseline = merged["rv_mean_baseline"].dropna()
rv_c2 = merged["rv_mean_c2"].dropna()

if len(rv_baseline) == len(rv_c2) and len(rv_baseline) > 1:
    t_stat, p_value = stats.ttest_rel(rv_baseline, rv_c2)
    pooled_std = np.sqrt((rv_baseline.std()**2 + rv_c2.std()**2) / 2)
    cohens_d = (rv_baseline.mean() - rv_c2.mean()) / pooled_std if pooled_std > 0 else 0
    ci = stats.t.interval(0.95, df=len(rv_c2)-1, loc=rv_c2.mean(), scale=stats.sem(rv_c2))
    
    ld_baseline = merged["logit_diff_baseline"].dropna()
    ld_c2 = merged["logit_diff_c2"].dropna()
    cryst_baseline = merged["crystallization_layer_baseline"].dropna()
    cryst_c2 = merged["crystallization_layer_c2"].dropna()
    
    results = {
        "experiment": "c2_rv_measurement",
        "date": "2025-01-11",
        "n_prompts": 50,
        "statistics": {
            "rv": {
                "baseline_mean": float(rv_baseline.mean()),
                "baseline_std": float(rv_baseline.std()),
                "c2_mean": float(rv_c2.mean()),
                "c2_std": float(rv_c2.std()),
                "delta_mean": float(rv_baseline.mean() - rv_c2.mean()),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "cohens_d": float(cohens_d),
                "c2_ci_95_low": float(ci[0]),
                "c2_ci_95_high": float(ci[1]),
                "meets_criteria": bool(ci[1] < 0.55),
            },
            "logit_diff": {
                "baseline_mean": float(ld_baseline.mean()),
                "baseline_std": float(ld_baseline.std()),
                "c2_mean": float(ld_c2.mean()),
                "c2_std": float(ld_c2.std()),
                "delta_mean": float(ld_c2.mean() - ld_baseline.mean()),
            },
            "crystallization": {
                "baseline_mean": float(cryst_baseline.mean()),
                "baseline_std": float(cryst_baseline.std()),
                "c2_mean": float(cryst_c2.mean()),
                "c2_std": float(cryst_c2.std()),
                "delta_mean": float(cryst_c2.mean() - cryst_baseline.mean()),
            },
        },
        "publication_readiness": {
            "n_pairs": len(rv_baseline),
            "n_meets_criteria": len(rv_baseline) >= 30,
            "p_value_meets_criteria": p_value < 0.001,
            "cohens_d_meets_criteria": cohens_d > 0.8,
            "ci_meets_criteria": ci[1] < 0.55,
        },
    }
    
    results_file = latest / "comprehensive_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print("✅ Comprehensive results saved")
    print(f"\n📊 Statistics:")
    print(f"  p-value: {p_value:.6f} {'✓' if p_value < 0.001 else '✗'}")
    print(f"  Cohen's d: {cohens_d:.3f} {'✓' if cohens_d > 0.8 else '✗'}")
    print(f"  C2 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}] {'✓' if ci[1] < 0.55 else '✗'}")
