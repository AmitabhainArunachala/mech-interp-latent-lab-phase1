"""
Run Index Utility
=================

Scans `results/**/runs/*` and builds a compact index of runs with:
- experiment name
- config snapshot path
- model identifier (incl. instruct/base + v0.1/v0.2 parsing)
- prompt bank version (if recorded by the pipeline)
- artifact outputs (csv/json/md/txt)

Usage:
  python -m src.utils.run_index --results_root results
  python -m src.utils.run_index --results_root results/phase1_mechanism --out_csv results/phase1_mechanism/run_index.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _extract_model_name(cfg: Dict[str, Any]) -> str:
    # Common schema variants across pipelines
    params = cfg.get("params") or {}
    model_cfg = cfg.get("model") or {}
    cand = (
        params.get("model")
        or params.get("model_name")
        or model_cfg.get("name")
        or cfg.get("model_name")
        or ""
    )
    return str(cand) if cand is not None else ""


def _parse_model_variant(model_name: str) -> Tuple[str, str]:
    """
    Returns:
      (family_variant, version_tag)
    where family_variant in {"instruct", "base", "unknown"}
    and version_tag is e.g. "v0.1", "v0.2", "".
    """
    low = (model_name or "").lower()
    if "instruct" in low:
        fam = "instruct"
    elif low:
        fam = "base"
    else:
        fam = "unknown"

    version = ""
    for tag in ("v0.1", "v0.2", "v0.3", "v1.0"):
        if tag in low:
            version = tag
            break
    return fam, version


def _extract_prompt_selection(cfg: Dict[str, Any]) -> str:
    """
    Best-effort summary of prompt slice parameters.
    Not all pipelines share a common schema, so we surface likely keys.
    """
    params = cfg.get("params") or {}
    keys = [
        "prompt_group",
        "prompt_groups",
        "prompt_pillar",
        "recursive_groups",
        "baseline_groups",
        "n_pairs",
        "n_prompts",
        "n_test_pairs",
        "window",
        "window_size",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        if k in params:
            out[k] = params.get(k)
    return json.dumps(out, sort_keys=True) if out else ""


def _iter_run_dirs(results_root: Path) -> Iterable[Path]:
    # Support both:
    # - results/runs/*
    # - results/<phase>/runs/*
    if (results_root / "runs").is_dir():
        for p in sorted((results_root / "runs").iterdir()):
            if p.is_dir():
                yield p
        return

    # Otherwise scan any nested runs dirs: results_root/**/runs/*
    for runs_dir in sorted(results_root.rglob("runs")):
        if not runs_dir.is_dir():
            continue
        for p in sorted(runs_dir.iterdir()):
            if p.is_dir():
                yield p


def _collect_artifacts(run_dir: Path) -> List[str]:
    # Keep it lightweight: list common artifact extensions at top-level of run_dir.
    exts = {".csv", ".json", ".md", ".txt"}
    artifacts: List[str] = []
    for p in sorted(run_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            artifacts.append(p.name)
    return artifacts


@dataclass(frozen=True)
class RunIndexRow:
    run_dir: str
    experiment: str
    timestamp: str
    config_path: str
    model_name: str
    model_family: str
    model_version: str
    prompt_bank_version: str
    prompt_selection: str
    artifacts: str


def build_index(results_root: Path) -> List[RunIndexRow]:
    rows: List[RunIndexRow] = []
    for run_dir in _iter_run_dirs(results_root):
        name = run_dir.name
        # Default parse: <timestamp>_<experiment>...
        timestamp = name.split("_", 2)[0] if "_" in name else ""

        cfg_path = run_dir / "config.json"
        cfg = _safe_read_json(cfg_path) or {}
        exp = str(cfg.get("experiment") or "")
        if not exp and "_" in name:
            # Heuristic: timestamp_experiment...
            parts = name.split("_", 2)
            exp = parts[1] if len(parts) > 1 else ""

        model_name = _extract_model_name(cfg)
        fam, ver = _parse_model_variant(model_name)

        pbv = ""
        pbv_json = _safe_read_json(run_dir / "prompt_bank_version.json")
        if pbv_json and isinstance(pbv_json.get("version"), str):
            pbv = str(pbv_json["version"])

        prompt_sel = _extract_prompt_selection(cfg)
        artifacts = ", ".join(_collect_artifacts(run_dir))

        rows.append(
            RunIndexRow(
                run_dir=str(run_dir),
                experiment=exp,
                timestamp=timestamp,
                config_path=str(cfg_path) if cfg_path.exists() else "",
                model_name=model_name,
                model_family=fam,
                model_version=ver,
                prompt_bank_version=pbv,
                prompt_selection=prompt_sel,
                artifacts=artifacts,
            )
        )
    return rows


def _write_csv(path: Path, rows: List[RunIndexRow]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(RunIndexRow.__annotations__.keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build an index of results/**/runs/*")
    p.add_argument("--results_root", default="results", help="Root results directory to scan.")
    p.add_argument("--out_csv", default=None, help="Output CSV path (default: <results_root>/run_index.csv)")
    args = p.parse_args(argv)

    results_root = Path(args.results_root)
    out_csv = Path(args.out_csv) if args.out_csv else (results_root / "run_index.csv")

    rows = build_index(results_root)
    _write_csv(out_csv, rows)

    print(f"[ok] indexed {len(rows)} runs")
    print(f"[ok] wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




