"""
Experiment I/O utilities: standardized run directories and artifact writing.

Design goals:
- Every run creates a timestamped folder with the full config snapshot.
- Outputs are written to predictable filenames (csv/json/md).
- No external dependencies beyond the Python stdlib.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    config_path: Path
    stdout_path: Path


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_dir(
    results_root: str | Path,
    experiment_name: str,
    run_name: Optional[str] = None,
) -> RunPaths:
    """
    Create a standardized run directory:
      <results_root>/runs/<timestamp>_<experiment_name>[_<run_name>]/
    """
    root = Path(results_root)
    safe_exp = experiment_name.replace(" ", "_")
    safe_run = (run_name or "").strip().replace(" ", "_")
    suffix = f"_{safe_run}" if safe_run else ""
    run_dir = root / "runs" / f"{_timestamp()}_{safe_exp}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return RunPaths(
        run_dir=run_dir,
        config_path=run_dir / "config.json",
        stdout_path=run_dir / "stdout.txt",
    )


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write(text)


def atomic_config_snapshot(
    config: Dict[str, Any],
    out_path: str | Path,
) -> None:
    """
    Save the exact config dict used for the run.
    """
    write_json(out_path, config)


