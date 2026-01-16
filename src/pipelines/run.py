#!/usr/bin/env python3
"""
Canonical config-driven experiment runner.

Usage:
  python -m src.pipelines.run --config /abs/path/to/config.json
  python -m src.pipelines.run --config configs/phase1_existence.json --results_root results
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.experiment_io import atomic_config_snapshot, create_run_dir, write_json, write_text
from src.pipelines.registry import ConfigError, run_from_config
from src.metrics.baseline_suite import BaselineMetricsSuite
from prompts.loader import PromptLoader
from src.utils.run_metadata import get_run_metadata, save_metadata, get_git_commit

CANONICAL_EXPERIMENTS = {
    "rv_l27_causal_validation",
    "confound_validation",
    "random_direction_control",
    "mlp_ablation_necessity",
    "mlp_sufficiency_test",
    "combined_mlp_sufficiency_test",
    "head_ablation_validation",
}

DISCOVERY_EXPERIMENTS = {
    "behavioral_grounding",
    "behavioral_grounding_batch",
    "path_patching_mechanism",
    "temporal_stability",
    "hysteresis",
    "kv_mechanism",
    "layer_sweep",
    "logit_lens_analysis",
    "vproj_patching_analysis",
    "mlp_vproj_combined_sufficiency_test",
    "c2_rv_measurement",
    "cross_architecture_validation",
}


def _missing_required_keys(summary: Dict[str, Any], required_keys: list[str]) -> list[str]:
    missing = []
    for key in required_keys:
        if key not in summary:
            missing.append(key)
    return missing


def _emit_soft_warning(exp_name: str, missing: list[str]) -> None:
    if missing:
        print(
            "[warning] summary.json missing required keys for discovery experiment "
            f"{exp_name}: {missing}",
            file=sys.stderr,
        )


def _load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("Config must be a JSON object at the top level")
    return obj


def _append_to_ledger(
    run_paths: Any, 
    cfg: Dict[str, Any], 
    summary: Dict[str, Any],
    success: bool,
    results_root: str | Path
) -> None:
    """
    Append run details to the global run ledger (RUN_INDEX.jsonl).
    
    The ledger provides a single source of truth for all experiment outcomes.
    """
    try:
        # Extract timestamp from run directory name (YYYYMMDD_HHMMSS_...)
        # Folder format: <timestamp>_<experiment_name>...
        folder_name = run_paths.run_dir.name
        timestamp_parts = folder_name.split("_")[:2] # ['20250101', '120000']
        timestamp = "_".join(timestamp_parts)
        
        # Extract standardized stats using BaselineMetricsSuite helper
        # If the summary doesn't have the standard structure, this might return partial data
        # but BaselineMetricsSuite.extract_ledger_stats is robust to missing keys.
        stats = BaselineMetricsSuite.extract_ledger_stats(summary)
        
        ledger_entry = {
            "timestamp": timestamp,
            "experiment": cfg.get("experiment", "unknown"),
            "model": cfg.get("params", {}).get("model", "unknown"),
            "prompt_bank_version": summary.get("prompt_bank_version", "unknown"),
            "success": success,
            "run_dir": str(run_paths.run_dir),
            
            # Metrics (nullable)
            "rv_d": stats.get("rv_d"),
            "rv_p": stats.get("rv_p"),
            "rv_delta": stats.get("rv_delta"),
            "logit_diff_d": stats.get("logit_diff_d"),
            "logit_diff_p": stats.get("logit_diff_p"),
            "logit_diff_delta": stats.get("logit_diff_delta"),
            "n_pairs": stats.get("n_pairs"),
            
            # Additional context
            "git_commit": get_git_commit(),
            "schema_version": summary.get("schema_version", "metrics_summary_v1"),
        }
        
        # Determine ledger path relative to results root
        # If results_root points to a phase subdir (e.g. results/phase1), 
        # we might want the ledger there or at the top level.
        # Current pattern seems to be one ledger per results root.
        ledger_path = Path(results_root) / "RUN_INDEX.jsonl"
        
        # Append to JSONL
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ledger_entry) + "\n")
            
        print(f"[ledger] Appended to {ledger_path}")

    except Exception as e:
        print(f"[warning] Failed to append to ledger: {e}", file=sys.stderr)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run canonical experiments from a JSON config.")
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument(
        "--results_root",
        default=None,
        help="Override results root directory (otherwise uses config['results']['root'] or 'results').",
    )
    args = parser.parse_args(argv)

    cfg = _load_json(args.config)

    results_root = args.results_root
    if results_root is None:
        results_root = (cfg.get("results") or {}).get("root") or "results"

    # Optional phase scoping: results/<phase>/runs/<timestamp>_<experiment>...
    # This matches the repo's meta goal of phase-based organization.
    results_phase = (cfg.get("results") or {}).get("phase")
    if results_phase:
        results_root = str(Path(results_root) / str(results_phase))

    exp_name = str(cfg.get("experiment") or "unknown_experiment")
    run_name = (cfg.get("run_name") or None)

    # 1. Get Prompt Bank Version
    try:
        loader = PromptLoader()
        prompt_bank_version = loader.version
    except Exception:
        prompt_bank_version = "unknown"
        print("[warning] Could not load PromptLoader, prompt_bank_version will be 'unknown'")

    # 2. Inject version into config params (so it appears in config.json snapshot)
    if "params" not in cfg:
        cfg["params"] = {}
    cfg["params"]["prompt_bank_version"] = prompt_bank_version

    # 3. Create run directory
    paths = create_run_dir(results_root=results_root, experiment_name=exp_name, run_name=run_name)
    atomic_config_snapshot(cfg, paths.config_path)

    try:
        # 4. Run Experiment
        result = run_from_config(cfg, paths.run_dir)
        
        # 5. Merge Prompt Bank Version into Summary
        result.summary["prompt_bank_version"] = prompt_bank_version
        result.summary.setdefault("schema_version", "metrics_summary_v1")
        result.summary.setdefault("experiment", exp_name)
        result.summary.setdefault("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        # Best-effort model name injection if missing
        if "model" not in result.summary:
            result.summary["model"] = cfg.get("model", {}).get("name") or cfg.get("params", {}).get("model")
        # Artifact pointers
        result.summary.setdefault("artifacts", {})
        result.summary["artifacts"].setdefault("config", str(paths.config_path))
        result.summary["artifacts"].setdefault("summary", str(paths.run_dir / "summary.json"))
        result.summary["artifacts"].setdefault("report", str(paths.run_dir / "report.md"))
        
        # 5b. Enforce summary schema for canonical; warn for discovery
        required_keys = [
            "n_pairs",
            "rv_recursive_mean",
            "rv_baseline_mean",
            "rv_delta_mean",
            "rv_cohens_d",
            "rv_p_value",
            "logit_diff_delta_mean",
            "logit_diff_cohens_d",
            "logit_diff_p_value",
            "prompt_bank_version",
        ]
        missing = _missing_required_keys(result.summary, required_keys)
        if exp_name in CANONICAL_EXPERIMENTS:
            if missing:
                raise ValueError(
                    "summary.json missing required keys for canonical experiment "
                    f"{exp_name}: {missing}"
                )
        elif exp_name in DISCOVERY_EXPERIMENTS:
            _emit_soft_warning(exp_name, missing)

        # 6. Save Artifacts
        write_json(paths.run_dir / "summary.json", result.summary)
        write_text(
            paths.run_dir / "report.md",
            "\n".join(
                [
                    f"# Run report: {exp_name}",
                    "",
                    f"- **run_dir**: `{paths.run_dir}`",
                    f"- **prompt_bank_version**: `{prompt_bank_version}`",
                    "",
                    "## Summary (machine-readable)",
                    "",
                    "```json",
                    json.dumps(result.summary, indent=2, sort_keys=True),
                    "```",
                    "",
                ]
            ),
        )
        print(f"[ok] run_dir: {paths.run_dir}")
        print(f"[ok] summary: {paths.run_dir / 'summary.json'}")

        # 6b. Save standardized metadata.json for reproducibility
        eval_window = (
            (cfg.get("params") or {}).get("window")
            or (cfg.get("params") or {}).get("window_size")
            or 16
        )
        metadata = get_run_metadata(cfg, eval_window=int(eval_window))
        save_metadata(paths.run_dir, metadata)
        
        # 7. Write to Ledger
        _append_to_ledger(paths, cfg, result.summary, success=True, results_root=results_root)
        
        return 0
        
    except ConfigError as e:
        write_text(paths.run_dir / "error.txt", f"ConfigError: {e}\n")
        print(f"[error] ConfigError: {e}", file=sys.stderr)
        
        # Attempt to write failure to ledger
        _append_to_ledger(paths, cfg, {"prompt_bank_version": prompt_bank_version}, success=False, results_root=results_root)
        
        return 2
    except Exception as e:
        write_text(paths.run_dir / "error.txt", f"{type(e).__name__}: {e}\n")
        print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)
        
        # Attempt to write failure to ledger
        _append_to_ledger(paths, cfg, {"prompt_bank_version": prompt_bank_version}, success=False, results_root=results_root)
        
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
