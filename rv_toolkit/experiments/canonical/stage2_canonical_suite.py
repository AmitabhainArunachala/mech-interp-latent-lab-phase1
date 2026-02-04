#!/usr/bin/env python3
"""
Stage 2: Canonical Suite Execution

Runs 13 canonical experiments with standardized infrastructure:
- Necessity (4): L0, L1, L2, L3 zero ablation
- Sufficiency (2): L0 patch, L0+L1 patch
- Position (1): L0 position-specific
- Windowed Denoising (4): L0-L2, L0-L4, L0-L8, L0-L12
- KV Interaction (2): KV-only, KV + L0-L4 window

All experiments use:
- PromptLoader with IDs
- Standardized metadata
- RUN_INDEX.jsonl tracking
- n_pairs=30 (or config default)
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '.')

from src.pipelines.mlp_ablation_necessity import run_mlp_ablation_necessity_from_config
from src.pipelines.mlp_sufficiency_test import run_mlp_sufficiency_test_from_config
from src.pipelines.mlp_combined_sufficiency_test import run_combined_mlp_sufficiency_test_from_config
from src.pipelines.mlp_ablation_position_specific import run_position_specific_ablation_from_config

# Import registry to check available experiments
from src.pipelines.registry import get_registry

# Canonical suite definition
CANONICAL_SUITE = [
    # Necessity Tests (4)
    {
        "name": "L0 Necessity",
        "config": "configs/mlp_ablation_necessity_l0.json",
        "runner": run_mlp_ablation_necessity_from_config,
        "n_pairs": 30,
    },
    {
        "name": "L1 Necessity",
        "config": "configs/mlp_ablation_necessity_l1.json",
        "runner": run_mlp_ablation_necessity_from_config,
        "n_pairs": 30,
    },
    {
        "name": "L2 Necessity",
        "config": "configs/mlp_ablation_necessity_l2.json",
        "runner": run_mlp_ablation_necessity_from_config,
        "n_pairs": 30,
    },
    {
        "name": "L3 Necessity",
        "config": "configs/mlp_ablation_necessity_l3.json",
        "runner": run_mlp_ablation_necessity_from_config,
        "n_pairs": 30,
    },
    # Sufficiency Tests (2)
    {
        "name": "L0 Sufficiency",
        "config": "configs/mlp_sufficiency_l0.json",
        "runner": run_mlp_sufficiency_test_from_config,
        "n_pairs": 30,
    },
    {
        "name": "L0+L1 Sufficiency",
        "config": "configs/combined_mlp_sufficiency_l0_l1.json",
        "runner": run_combined_mlp_sufficiency_test_from_config,
        "n_pairs": 30,
    },
    # Position Tests (1)
    {
        "name": "L0 Position-Specific",
        "config": "configs/position_specific_l0_ablation.json",
        "runner": run_position_specific_ablation_from_config,
        "n_pairs": 30,
    },
    # Windowed Denoising (4) - TODO: Create configs and pipelines
    # KV Interaction (2) - TODO: Create configs and pipelines
]

# Output directory
CANONICAL_OUTPUT_DIR = Path("results/canonical_suite_v1_0")
CANONICAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment(exp_def: dict, suite_dir: Path) -> dict:
    """Run a single canonical experiment."""
    print("=" * 80)
    print(f"Running: {exp_def['name']}")
    print("=" * 80)
    
    # Load config
    config_path = Path(exp_def["config"])
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return {"status": "error", "error": f"Config not found: {config_path}"}
    
    with open(config_path) as f:
        cfg = json.load(f)
    
    # Override n_pairs if specified
    if "n_pairs" in exp_def:
        cfg["params"]["n_pairs"] = exp_def["n_pairs"]
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = exp_def["name"].lower().replace(" ", "_").replace("+", "_plus")
    run_dir = suite_dir / "runs" / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Run directory: {run_dir}")
    print(f"Config: {config_path}")
    print(f"n_pairs: {cfg['params'].get('n_pairs', 'default')}")
    print("=" * 80)
    
    try:
        # Run experiment
        result = exp_def["runner"](cfg, run_dir)
        
        print("=" * 80)
        print(f"✅ {exp_def['name']} completed successfully")
        print("=" * 80)
        
        return {
            "status": "success",
            "name": exp_def["name"],
            "run_dir": str(run_dir),
            "summary": result.summary,
        }
    except Exception as e:
        print("=" * 80)
        print(f"❌ {exp_def['name']} failed: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        
        return {
            "status": "error",
            "name": exp_def["name"],
            "run_dir": str(run_dir),
            "error": str(e),
        }


def main():
    """Run canonical suite."""
    print("=" * 80)
    print("STAGE 2: CANONICAL SUITE")
    print("=" * 80)
    print(f"Total experiments: {len(CANONICAL_SUITE)}")
    print(f"Output directory: {CANONICAL_OUTPUT_DIR}")
    print("=" * 80)
    
    results = []
    
    for i, exp_def in enumerate(CANONICAL_SUITE, 1):
        print(f"\n[{i}/{len(CANONICAL_SUITE)}]")
        result = run_experiment(exp_def, CANONICAL_OUTPUT_DIR)
        results.append(result)
        
        # Save progress
        progress_path = CANONICAL_OUTPUT_DIR / "progress.json"
        with open(progress_path, "w") as f:
            json.dump(results, f, indent=2)
    
    # Final summary
    print("\n" + "=" * 80)
    print("CANONICAL SUITE COMPLETE")
    print("=" * 80)
    
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    print(f"Success: {success_count}/{len(results)}")
    print(f"Errors: {error_count}/{len(results)}")
    
    # Save final results
    results_path = CANONICAL_OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()


