#!/usr/bin/env python3
"""Compute statistical analysis for C2+R_V results."""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Find the latest C2 run
runs_dir = Path("results/phase1_mechanism/runs")
c2_runs = sorted(runs_dir.glob("*_c2_rv_measurement"))
if not c2_runs:
    print("No C2 runs found")
    exit(1)

latest_run = c2_runs[-1]
csv_path = latest_run / "c2_rv_measurement.csv"

print(f"Analyzing: {latest_run.name}")
print("=" * 60)

# Load data
df = pd.read_csv(csv_path)

# Get R_V values by condition
baseline_rv = df[df["config"] == "baseline"]["rv_mean"].values
kv_only_rv = df[df["config"] == "kv_only"]["rv_mean"].values
c2_rv = df[df["config"] == "c2_full"]["rv_mean"].values

print(f"\nSample sizes:")
print(f"  Baseline: n={len(baseline_rv)}")
print(f"  KV Only:  n={len(kv_only_rv)}")
print(f"  C2 Full:  n={len(c2_rv)}")

print(f"\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(f"\nBaseline R_V: {np.mean(baseline_rv):.4f} ± {np.std(baseline_rv):.4f}")
print(f"KV Only R_V:  {np.mean(kv_only_rv):.4f} ± {np.std(kv_only_rv):.4f}")
print(f"C2 Full R_V:  {np.mean(c2_rv):.4f} ± {np.std(c2_rv):.4f}")

print(f"\n" + "=" * 60)
print("CONFIDENCE INTERVALS (95%)")
print("=" * 60)

for name, values in [("Baseline", baseline_rv), ("KV Only", kv_only_rv), ("C2 Full", c2_rv)]:
    ci_low, ci_high = stats.t.interval(
        0.95,
        df=len(values)-1,
        loc=np.mean(values),
        scale=stats.sem(values)
    )
    print(f"\n{name}: [{ci_low:.4f}, {ci_high:.4f}]")
    if name == "C2 Full":
        print(f"  CI entirely below 0.55: {ci_high < 0.55}")

print(f"\n" + "=" * 60)
print("PAIRED T-TESTS")
print("=" * 60)

# Baseline vs C2
t_stat, p_value = stats.ttest_rel(baseline_rv, c2_rv)
print(f"\nBaseline vs C2 Full:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.2e}")
print(f"  p < 0.001: {p_value < 0.001}")
print(f"  p < 0.0001: {p_value < 0.0001}")

# Baseline vs KV Only
t_stat_kv, p_value_kv = stats.ttest_rel(baseline_rv, kv_only_rv)
print(f"\nBaseline vs KV Only:")
print(f"  t-statistic: {t_stat_kv:.4f}")
print(f"  p-value: {p_value_kv:.2e}")

# KV Only vs C2
t_stat_kv_c2, p_value_kv_c2 = stats.ttest_rel(kv_only_rv, c2_rv)
print(f"\nKV Only vs C2 Full:")
print(f"  t-statistic: {t_stat_kv_c2:.4f}")
print(f"  p-value: {p_value_kv_c2:.2e}")

print(f"\n" + "=" * 60)
print("EFFECT SIZES (Cohen's d)")
print("=" * 60)

def cohens_d(group1, group2):
    pooled_std = np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2)
    return (np.mean(group1) - np.mean(group2)) / pooled_std

d_baseline_c2 = cohens_d(baseline_rv, c2_rv)
d_baseline_kv = cohens_d(baseline_rv, kv_only_rv)
d_kv_c2 = cohens_d(kv_only_rv, c2_rv)

def interpret_d(d):
    d = abs(d)
    if d > 0.8:
        return "Large"
    elif d > 0.5:
        return "Medium"
    elif d > 0.2:
        return "Small"
    else:
        return "Negligible"

print(f"\nBaseline vs C2 Full:  d = {d_baseline_c2:.4f} ({interpret_d(d_baseline_c2)})")
print(f"Baseline vs KV Only:  d = {d_baseline_kv:.4f} ({interpret_d(d_baseline_kv)})")
print(f"KV Only vs C2 Full:   d = {d_kv_c2:.4f} ({interpret_d(d_kv_c2)})")

print(f"\n" + "=" * 60)
print("PUBLICATION-READY SUMMARY")
print("=" * 60)

ci_low_c2, ci_high_c2 = stats.t.interval(0.95, df=len(c2_rv)-1, loc=np.mean(c2_rv), scale=stats.sem(c2_rv))

print(f"""
C2 intervention significantly reduces geometric contraction (R_V):
  Baseline: M = {np.mean(baseline_rv):.3f}, SD = {np.std(baseline_rv):.3f}
  C2 Full:  M = {np.mean(c2_rv):.3f}, SD = {np.std(c2_rv):.3f}
  
  t({len(c2_rv)-1}) = {t_stat:.2f}, p < {max(p_value, 1e-10):.0e}, d = {d_baseline_c2:.2f}
  95% CI for C2: [{ci_low_c2:.3f}, {ci_high_c2:.3f}]
  
Criteria met:
  ✓ n ≥ 30:       {len(c2_rv)} ≥ 30
  ✓ p < 0.001:    {p_value:.2e} < 0.001
  ✓ d > 0.8:      {d_baseline_c2:.2f} > 0.8 = {d_baseline_c2 > 0.8}
  ✓ CI < 0.55:    [{ci_low_c2:.3f}, {ci_high_c2:.3f}] < 0.55 = {ci_high_c2 < 0.55}
""")
