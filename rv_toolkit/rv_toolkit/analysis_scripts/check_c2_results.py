#!/usr/bin/env python3
"""Check C2 results and compute missing statistics."""

import json
from pathlib import Path
import pandas as pd
from scipy import stats
import numpy as np

# Find latest run
runs = sorted(Path("results/phase1_mechanism/runs").glob("*_c2_rv_measurement"))
if not runs:
    print("No C2 runs found")
    exit(1)

latest = runs[-1]
print(f"Checking: {latest}")

# Load CSV
df = pd.read_csv(latest / "c2_rv_measurement.csv")
print(f"\nColumns: {list(df.columns)}")
print(f"Has logit_diff: {'logit_diff' in df.columns}")
print(f"n_prompts per config: {df.groupby('config').size().to_dict()}")

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
    
    print(f"\n✅ Statistics:")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print(f"  R_V baseline: {rv_baseline.mean():.4f} ± {rv_baseline.std():.4f}")
    print(f"  R_V C2: {rv_c2.mean():.4f} ± {rv_c2.std():.4f}")
    
    # 95% CI for C2
    ci = stats.t.interval(0.95, df=len(rv_c2)-1, loc=rv_c2.mean(), scale=stats.sem(rv_c2))
    print(f"  C2 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    # Check if meets criteria
    print(f"\n📊 Publication Criteria:")
    print(f"  n ≥ 30: {len(rv_baseline) >= 30} ({len(rv_baseline)})")
    print(f"  p < 0.001: {p_value < 0.001} ({p_value:.6f})")
    print(f"  Cohen's d > 0.8: {cohens_d > 0.8} ({cohens_d:.3f})")
    print(f"  C2 CI < 0.55: {ci[1] < 0.55} (upper bound: {ci[1]:.4f})")
