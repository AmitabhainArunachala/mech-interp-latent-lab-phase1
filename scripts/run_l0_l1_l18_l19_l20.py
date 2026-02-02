#!/usr/bin/env python3
"""Run L0+L1+L18+L19+L20 combined sufficiency test."""

import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.mlp_combined_sufficiency_test import run_combined_mlp_sufficiency_test_from_config

if __name__ == "__main__":
    cfg = json.load(open("configs/combined_mlp_sufficiency_l0_l1_l18_l19_l20.json"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results/phase1_mechanism/runs") / f"{timestamp}_l0_l1_l18_l19_l20_combined_sufficiency"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {run_dir}")
    print(f"Layers: {cfg['params']['layers']}")
    print("Starting experiment...")
    print("=" * 60)

    result = run_combined_mlp_sufficiency_test_from_config(cfg, run_dir)

    print("=" * 60)
    print("✅ Experiment complete!")
    print(f"Results: {run_dir}")
