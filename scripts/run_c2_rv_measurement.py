#!/usr/bin/env python3
"""
Run C2 + R_V Measurement Experiment

This is the key experiment bridging geometry (R_V) to behavior (domain shift).

Usage:
    python scripts/run_c2_rv_measurement.py

Or with custom config:
    python scripts/run_c2_rv_measurement.py --config configs/c2_rv_measurement.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.discovery.c2_rv_measurement import run_c2_rv_measurement_from_config


def main():
    parser = argparse.ArgumentParser(description="Run C2 + R_V measurement experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/c2_rv_measurement.json",
        help="Path to config file"
    )
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=None,
        help="Override number of prompts"
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
    else:
        print(f"Config not found: {config_path}, using defaults")
        cfg = {
            "params": {
                "model": "mistralai/Mistral-7B-v0.1",
                "n_prompts": 20,
                "n_recursive": 10,
                "max_new_tokens": 100,
                "temperature": 0.7,
                "early_layer": 5,
                "late_layer": 27,
                "rv_window": 16,
                "seed": 42,
            }
        }

    # Override n_prompts if specified
    if args.n_prompts is not None:
        cfg["params"]["n_prompts"] = args.n_prompts

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results/phase1_mechanism/runs") / f"{timestamp}_c2_rv_measurement"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {run_dir}")
    print(f"Config: {args.config}")
    print("")
    print("Starting C2 + R_V measurement...")
    print("=" * 60)

    result = run_c2_rv_measurement_from_config(cfg, run_dir)

    print("")
    print("=" * 60)
    print("Experiment complete!")
    print(f"Results: {run_dir}")
    print("")
    print("Summary:")
    print(json.dumps(result.summary, indent=2))


if __name__ == "__main__":
    main()
