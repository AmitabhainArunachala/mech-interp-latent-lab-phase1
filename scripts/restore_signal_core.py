#!/usr/bin/env python3
"""
Restore signal-core runs from results/archive/incomplete.
Default is dry-run. Use --apply to move.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_ROOT = REPO_ROOT / "results" / "archive" / "incomplete"

RESTORE_MAP: List[Tuple[str, str]] = [
    # High-N path patching (phase1_mechanism)
    (
        "phase1_mechanism/runs/20251213_080454_path_patching_mechanism_full_early_layer_sweep_full_controls_base",
        "results/phase1_mechanism/runs/20251213_080454_path_patching_mechanism_full_early_layer_sweep_full_controls_base",
    ),
    (
        "phase1_mechanism/runs/20251213_090121_path_patching_mechanism_early_layers_deep_base",
        "results/phase1_mechanism/runs/20251213_090121_path_patching_mechanism_early_layers_deep_base",
    ),
    (
        "phase1_mechanism/runs/20251213_073754_path_patching_mechanism_full_early_layer_sweep_base",
        "results/phase1_mechanism/runs/20251213_073754_path_patching_mechanism_full_early_layer_sweep_base",
    ),
    (
        "phase1_mechanism/runs/20251213_064141_path_patching_mechanism_layer_sweep_base",
        "results/phase1_mechanism/runs/20251213_064141_path_patching_mechanism_layer_sweep_base",
    ),
    # Gold standard runs
    (
        "gold_standard/runs/20251216_060955_rv_l27_causal_validation",
        "results/gold_standard/runs/20251216_060955_rv_l27_causal_validation",
    ),
    (
        "gold_standard/runs/20251216_061127_rv_l27_causal_validation",
        "results/gold_standard/runs/20251216_061127_rv_l27_causal_validation",
    ),
    (
        "gold_standard/runs/20251216_060911_confound_validation",
        "results/gold_standard/runs/20251216_060911_confound_validation",
    ),
]


def _move(src: Path, dest: Path, apply: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if apply:
        if dest.exists():
            raise RuntimeError(f"Destination exists: {dest}")
        shutil.move(str(src), str(dest))


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore signal-core runs from archive.")
    parser.add_argument("--apply", action="store_true", help="Actually move files.")
    args = parser.parse_args()

    moved = 0
    missing = 0

    for rel_src, rel_dest in RESTORE_MAP:
        src = ARCHIVE_ROOT / rel_src
        dest = REPO_ROOT / rel_dest
        if not src.exists():
            missing += 1
            print(f"[missing] {src}")
            continue
        print(f"[restore] {src} -> {dest}")
        _move(src, dest, apply=args.apply)
        moved += 1

    print(f"[ok] to_restore: {len(RESTORE_MAP)}")
    print(f"[ok] moved: {moved} (apply={args.apply})")
    print(f"[ok] missing: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
