#!/usr/bin/env python3
"""Run C2 component ablation experiments."""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.c2_rv_measurement import run_c2_rv_measurement_from_config


def create_ablation_config(base_config, ablation_name, config_override):
    """Create ablation config from base config."""
    cfg = {
        "experiment": "c2_ablation",
        "ablation": ablation_name,
        "params": {
            **base_config["params"],
            "config_override": config_override,
        },
    }
    return cfg


def main():
    # Base config (same as C2 full)
    base_config = {
        "params": {
            "model": "mistralai/Mistral-7B-v0.1",
            "n_prompts": 30,  # Smaller for faster ablation
            "n_recursive": 15,
            "max_new_tokens": 100,
            "temperature": 0.7,
            "early_layer": 5,
            "late_layer": 27,
            "rv_window": 16,
            "seed": 42,
        }
    }

    # Ablation configs (in priority order)
    ablations = [
        {
            "name": "no_cascade",
            "description": "C2 without L26 cascade - only KV + H18/H26 steering",
            "config_override": {
                "head_target": "h18_h26",
                "kv_strategy": "full",
                "residual_alphas": None,
                "vproj_alpha": 2.5,
            },
        },
        {
            "name": "no_steering",
            "description": "C2 with KV only - no H18/H26 steering",
            "config_override": {
                "head_target": "none",
                "kv_strategy": "full",
                "residual_alphas": {"26": 0.6},
                "vproj_alpha": 0.0,
            },
        },
        {
            "name": "no_kv",
            "description": "C2 with steering only - no KV swap",
            "config_override": {
                "head_target": "h18_h26",
                "kv_strategy": "none",
                "residual_alphas": {"26": 0.6},
                "vproj_alpha": 2.5,
            },
        },
    ]

    results = {}

    for ablation in ablations:
        print(f"\n{'='*80}")
        print(f"Running ablation: {ablation['name']}")
        print(f"Description: {ablation['description']}")
        print(f"{'='*80}\n")

        cfg = create_ablation_config(base_config, ablation["name"], ablation["config_override"])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("results/phase1_mechanism/runs") / f"{timestamp}_c2_ablation_{ablation['name']}"
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = run_c2_rv_measurement_from_config(cfg, run_dir)
            results[ablation["name"]] = {
                "run_dir": str(run_dir),
                "summary": result.summary,
            }
            print(f"\n✅ {ablation['name']} complete: {run_dir}")
        except Exception as e:
            print(f"\n❌ {ablation['name']} failed: {e}")
            results[ablation["name"]] = {"error": str(e)}

    # Save ablation summary
    summary_file = Path("results/phase1_mechanism/runs") / f"{datetime.now().strftime('%Y%m%d')}_c2_ablation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("All ablations complete!")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
