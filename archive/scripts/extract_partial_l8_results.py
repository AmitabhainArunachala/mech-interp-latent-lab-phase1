"""
extract_partial_l8_results.py

Emergency script to extract partial results from Experiment 1 log file
if shutdown is needed before completion.

This parses the log file and creates a CSV with all completed pairs.
"""

import re
import pandas as pd
from pathlib import Path

LOG_FILE = "l8_patching_sweep_350.log"
OUTPUT_CSV = "results/dec11_evening/l8_early_layer_patching_sweep_PARTIAL.csv"


def parse_log_file(log_path):
    """Extract all completed pair data from log file."""
    rows = []
    current_pair = None
    
    with open(log_path, 'r') as f:
        for line in f:
            # Match: [Pair X, LY] rec→base: state1→state2 (RV: val1→val2, Δ=delta) | base→rec: ...
            match = re.match(
                r'\[Pair (\d+), L(\d+)\] rec→base: (\w+)→(\w+) \(RV: ([\d.]+)→([\d.]+), Δ=([+-][\d.]+)\) \| base→rec: (\w+)→(\w+) \(RV: ([\d.]+)→([\d.]+), Δ=([+-][\d.]+)\)',
                line
            )
            
            if match:
                pair_idx = int(match.group(1))
                layer = int(match.group(2))
                
                # rec→base direction
                state_base_nat = match.group(3)
                state_rec_to_base = match.group(4)
                rv_base_nat = float(match.group(5))
                rv_rec_to_base = float(match.group(6))
                rv_delta_rec_to_base = float(match.group(7))
                
                # base→rec direction
                state_rec_nat = match.group(8)
                state_base_to_rec = match.group(9)
                rv_rec_nat = float(match.group(10))
                rv_base_to_rec = float(match.group(11))
                rv_delta_base_to_rec = float(match.group(12))
                
                rows.append({
                    'pair_idx': pair_idx,
                    'layer': layer,
                    'state_base_natural': state_base_nat,
                    'state_rec_to_base': state_rec_to_base,
                    'rv_base_natural': rv_base_nat,
                    'rv_rec_to_base': rv_rec_to_base,
                    'rv_delta_rec_to_base': rv_delta_rec_to_base,
                    'state_rec_natural': state_rec_nat,
                    'state_base_to_rec': state_base_to_rec,
                    'rv_rec_natural': rv_rec_nat,
                    'rv_base_to_rec': rv_base_to_rec,
                    'rv_delta_base_to_rec': rv_delta_base_to_rec,
                })
    
    return rows


if __name__ == "__main__":
    print("Extracting partial results from log file...")
    
    if not Path(LOG_FILE).exists():
        print(f"Error: Log file {LOG_FILE} not found")
        exit(1)
    
    rows = parse_log_file(LOG_FILE)
    
    if not rows:
        print("No data found in log file")
        exit(1)
    
    df = pd.DataFrame(rows)
    
    # Create output directory
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ Extracted {len(rows)} rows from log")
    print(f"   Unique pairs: {df['pair_idx'].nunique()}")
    print(f"   Layers: {sorted(df['layer'].unique())}")
    print(f"   Saved to: {OUTPUT_CSV}")
    
    # Summary
    print(f"\nSummary by layer (rec→base):")
    for layer in sorted(df['layer'].unique()):
        layer_data = df[df['layer'] == layer]
        mean_delta = layer_data['rv_delta_rec_to_base'].mean()
        collapse_pct = (layer_data['state_rec_to_base'] == 'collapse').mean() * 100
        print(f"   L{layer}: RV Δ={mean_delta:+.3f}, Collapse={collapse_pct:.0f}%")
