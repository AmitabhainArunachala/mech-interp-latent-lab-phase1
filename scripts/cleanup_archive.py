#!/usr/bin/env python3
"""
Cleanup helper: archive failed/incomplete runs while preserving signal core.

Default is dry-run. Use --apply to move files.
"""
from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, Set


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
ARCHIVE_ROOT = RESULTS_ROOT / "archive"
SIGNAL_INDEX = REPO_ROOT / "SIGNAL_CORE_INDEX.md"


def _load_signal_core_paths() -> Set[Path]:
    if not SIGNAL_INDEX.exists():
        return set()
    text = SIGNAL_INDEX.read_text(encoding="utf-8", errors="ignore")
    paths: Set[Path] = set()
    for m in re.finditer(r"`([^`]+)`", text):
        rel = m.group(1).strip()
        if not rel:
            continue
        abs_path = (REPO_ROOT / rel).resolve()
        paths.add(abs_path)
        # If the signal core references a file, preserve its parent directory
        if abs_path.is_file():
            paths.add(abs_path.parent)
    return paths


def _is_under(path: Path, roots: Iterable[Path]) -> bool:
    for root in roots:
        try:
            path.resolve().relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _iter_run_dirs(results_root: Path) -> Iterable[Path]:
    for runs_dir in results_root.rglob("runs"):
        if runs_dir.is_dir():
            for p in runs_dir.iterdir():
                if p.is_dir():
                    yield p


def _has_data_artifact(run_dir: Path) -> bool:
    for ext in (".csv", ".jsonl"):
        if any(run_dir.glob(f"*{ext}")):
            return True
    if any(run_dir.glob("*pairs.csv")):
        return True
    return False


def _archive_path(run_dir: Path, bucket: str) -> Path:
    rel = run_dir.resolve().relative_to(RESULTS_ROOT)
    return (ARCHIVE_ROOT / bucket / rel)


def _move_dir(src: Path, dest: Path, apply: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if apply:
        if dest.exists():
            raise RuntimeError(f"Archive target already exists: {dest}")
        shutil.move(str(src), str(dest))


def _scan_empty_files(root: Path, keep_roots: Set[Path]) -> list[Path]:
    empties = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if _is_under(p, [ARCHIVE_ROOT]):
            continue
        if _is_under(p, keep_roots):
            continue
        try:
            if p.stat().st_size == 0:
                empties.append(p)
        except OSError:
            continue
    return empties


def _archive_empty_file(path: Path, apply: bool) -> None:
    rel = path.resolve().relative_to(RESULTS_ROOT)
    dest = ARCHIVE_ROOT / "empty" / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if apply:
        if dest.exists():
            raise RuntimeError(f"Archive target already exists: {dest}")
        shutil.move(str(path), str(dest))


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive failed/incomplete runs.")
    parser.add_argument("--apply", action="store_true", help="Actually move files.")
    parser.add_argument("--scan-empty", action="store_true", help="Also list empty files.")
    args = parser.parse_args()

    keep_paths = _load_signal_core_paths()
    keep_roots = set(keep_paths)

    failed = []
    incomplete = []

    for run_dir in _iter_run_dirs(RESULTS_ROOT):
        if _is_under(run_dir, [ARCHIVE_ROOT]):
            continue
        if _is_under(run_dir, keep_roots):
            continue

        if (run_dir / "error.txt").exists():
            failed.append(run_dir)
            continue

        if (run_dir / "summary.json").exists() and not _has_data_artifact(run_dir):
            incomplete.append(run_dir)

    report_lines = []
    report_lines.append("# Cleanup Report")
    report_lines.append("")
    report_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    report_lines.append("")

    report_lines.append("## Failed Runs (error.txt)")
    for d in failed:
        report_lines.append(f"- {d.relative_to(REPO_ROOT)}")
    report_lines.append("")

    report_lines.append("## Incomplete Runs (summary.json but no CSV/JSONL)")
    for d in incomplete:
        report_lines.append(f"- {d.relative_to(REPO_ROOT)}")
    report_lines.append("")

    empties = []
    if args.scan_empty:
        empties = _scan_empty_files(RESULTS_ROOT, keep_roots)
        report_lines.append("## Empty Files")
        for p in empties:
            report_lines.append(f"- {p.relative_to(REPO_ROOT)}")
        report_lines.append("")

    report_path = ARCHIVE_ROOT / f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if args.apply:
        for d in failed:
            _move_dir(d, _archive_path(d, "failed"), apply=True)
        for d in incomplete:
            _move_dir(d, _archive_path(d, "incomplete"), apply=True)
        for p in empties:
            _archive_empty_file(p, apply=True)

    print(f"[ok] failed: {len(failed)}")
    print(f"[ok] incomplete: {len(incomplete)}")
    if args.scan_empty:
        print(f"[ok] empty files: {len(empties)}")
    print(f"[ok] report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
