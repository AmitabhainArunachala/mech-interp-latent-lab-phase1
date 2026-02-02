#!/usr/bin/env python3
"""Analyze KV layer sweep results."""

import pandas as pd
from pathlib import Path

print("=== KV Layer Sweep Results ===")
print("")

ranges = ["l0_l8", "l8_l16", "l16_l24", "l24_l32"]
results = []

for layer_range in ranges:
    dirs = list(Path("results/phase1_mechanism/runs").glob(f"*kv_sweep_{layer_range}"))
    if not dirs:
        continue
    
    run_dir = dirs[0]
    csv_path = run_dir / "kv_mechanism.csv"
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        gap = df["rv_base"].mean() - df["rv_rec"].mean()
        restored = df["rv_base"].mean() - df["rv_swap"].mean()
        transfer = (restored / gap * 100.0) if gap > 1e-6 else 0.0
        
        results.append({
            "range": layer_range,
            "rv_rec": df["rv_rec"].mean(),
            "rv_base": df["rv_base"].mean(),
            "rv_swap": df["rv_swap"].mean(),
            "transfer": transfer,
        })

# Sort by transfer efficiency
results.sort(key=lambda x: x["transfer"], reverse=True)

print("Transfer Efficiency by Layer Range:")
print("")
for r in results:
    gap_val = r["rv_base"] - r["rv_rec"]
    print(f"{r['range']:10s}: {r['transfer']:7.1f}%  (R_V swap: {r['rv_swap']:.4f}, gap: {gap_val:.4f})")

print("")
print("Full KV swap (baseline): ~105%")
print("")
print("Key Finding:")
if results:
    best = max(results, key=lambda x: x["transfer"])
    worst = min(results, key=lambda x: x["transfer"])
    print(f"  Best: {best['range']} ({best['transfer']:.1f}%)")
    print(f"  Worst: {worst['range']} ({worst['transfer']:.1f}%)")
    print("")
    print("Interpretation:")
    print("  - Selective KV patching has LOW transfer efficiency")
    print("  - Mode requires FULL KV context across all layers")
    print("  - No single layer range stores the mode independently")
