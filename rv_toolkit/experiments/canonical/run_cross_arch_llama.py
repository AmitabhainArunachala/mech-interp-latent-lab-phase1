#!/usr/bin/env python3
"""
Run cross-architecture validation on Llama-3-8B-Instruct.
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.cross_architecture_validation import run_cross_architecture_validation_from_config

# Set HuggingFace token
HF_TOKEN = "HF_TOKEN_REDACTED"
os.environ["HF_TOKEN"] = HF_TOKEN

# Load config
config_path = Path("configs/cross_architecture_llama.json")
with open(config_path, "r") as f:
    cfg = json.load(f)

# Create run directory
run_dir_base = Path("results/phase2_generalization/runs")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = "cross_arch_llama"
run_dir = run_dir_base / f"{timestamp}_{run_name}"
run_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CROSS-ARCHITECTURE VALIDATION: Llama-3-8B-Instruct")
print("=" * 60)
print(f"Model: {cfg['params']['model']}")
print(f"Late layer: {cfg['params']['late_layer']}")
print(f"Window: {cfg['params']['window']}")
print(f"Recursive group: {cfg['params']['prompt_groups']['recursive']}")
print(f"Control groups: {cfg['params']['prompt_groups']['controls']}")
print(f"Run directory: {run_dir}")
print("=" * 60)

# Run experiment
try:
    result = run_cross_architecture_validation_from_config(cfg, run_dir)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED")
    print("=" * 60)
    
    if "mean_rv" in result.summary:
        print(f"Champions R_V: {result.summary['mean_rv'].get('champions', 'N/A'):.4f}")
        print(f"Length-matched R_V: {result.summary['mean_rv'].get('length_matched', 'N/A'):.4f}")
        print(f"Pseudo-recursive R_V: {result.summary['mean_rv'].get('pseudo_recursive', 'N/A'):.4f}")
    
    if "ttest" in result.summary:
        ttest = result.summary["ttest"]
        if "champions_vs_length_matched" in ttest:
            p_val = ttest["champions_vs_length_matched"]["p"]
            cohens_d = ttest["champions_vs_length_matched"]["cohens_d"]
            print(f"\nChampions vs Length-matched:")
            print(f"  p-value: {p_val:.2e}")
            print(f"  Cohen's d: {cohens_d:.2f}")
    
    print(f"\nResults saved to: {run_dir}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
