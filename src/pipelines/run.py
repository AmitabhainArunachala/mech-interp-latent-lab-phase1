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
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.experiment_io import atomic_config_snapshot, create_run_dir, write_json, write_text
from src.pipelines.registry import ConfigError, run_from_config


def _load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("Config must be a JSON object at the top level")
    return obj


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

    paths = create_run_dir(results_root=results_root, experiment_name=exp_name, run_name=run_name)
    atomic_config_snapshot(cfg, paths.config_path)

    try:
        result = run_from_config(cfg, paths.run_dir)
        write_json(paths.run_dir / "summary.json", result.summary)
        write_text(
            paths.run_dir / "report.md",
            "\n".join(
                [
                    f"# Run report: {exp_name}",
                    "",
                    f"- **run_dir**: `{paths.run_dir}`",
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
        return 0
    except ConfigError as e:
        write_text(paths.run_dir / "error.txt", f"ConfigError: {e}\n")
        print(f"[error] ConfigError: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        write_text(paths.run_dir / "error.txt", f"{type(e).__name__}: {e}\n")
        print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


