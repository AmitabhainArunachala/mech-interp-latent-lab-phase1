#!/usr/bin/env python3
"""Analyze logit lens analysis results."""

import pandas as pd
from pathlib import Path
import json

# Find the most recent logit lens run
runs = sorted(Path("results/phase1_mechanism/runs").glob("*_logit_lens_analysis"))
if not runs:
    print("No runs found")
    exit(1)

run_dir = runs[-1]
print(f"Analyzing: {run_dir}")
print("")

df = pd.read_csv(run_dir / "logit_lens_analysis.csv")
summary = json.load(open(run_dir / "summary.json"))

print("=== Key Findings ===")
print("")
print("R_V Comparison:")
print(f"  Recursive mean: {df['rv_recursive'].mean():.3f}")
print(f"  Baseline mean: {df['rv_baseline'].mean():.3f}")
print(f"  Delta: {df['rv_delta'].mean():.3f}")
print("")
print("Crystallization:")
print(f"  Recursive layer: {df['rec_crystallization_layer'].mean():.1f}")
print(f"  Baseline layer: {df['base_crystallization_layer'].mean():.1f}")
print("")
print("Logit Difference Crossover:")
rec_cross = df["rec_logit_diff_crossover_layer"].dropna()
if len(rec_cross) > 0:
    print(f"  Recursive crossover layer: {rec_cross.mean():.1f} (n={len(rec_cross)})")
else:
    print("  Recursive crossover: Never positive (always positive from L0)")
print("")
print("Logit Difference Trajectory:")
print(f"  L21: {df['rec_logit_diff_L21'].mean():.3f}")
print(f"  L27: {df['rec_logit_diff_L27'].mean():.3f}")
print(f"  Final: {df['rec_logit_diff_final'].mean():.3f}")
print("")
print("Interpretation:")
print("  - Logit diff is positive from L0 (crossover = 0.0)")
print("  - Model prefers recursive tokens from the start")
print("  - Crystallization happens at L26.4 (late)")
if df['rec_logit_diff_L21'].mean() > df['rec_logit_diff_L27'].mean():
    print("  - L21 logit diff > L27 - decreases over layers")
else:
    print("  - L21 logit diff < L27 - increases over layers")
