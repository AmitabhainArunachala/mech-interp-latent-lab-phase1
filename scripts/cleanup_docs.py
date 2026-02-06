#!/usr/bin/env python3
"""
Archive non-core docs while preserving signal core.
Default is dry-run; use --apply to move.
"""
from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Set

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = REPO_ROOT / "docs"
ARCHIVE_ROOT = DOCS_ROOT / "archive"
SIGNAL_INDEX = REPO_ROOT / "SIGNAL_CORE_INDEX.md"
META_INDEX = REPO_ROOT / "META_TOP10_INDEX.md"

KEEP_DIRS = {
    (DOCS_ROOT / "standards").resolve(),
    (DOCS_ROOT / "status").resolve(),
}


def _load_keep_paths() -> Set[Path]:
    paths: Set[Path] = set()
    for index_path in (SIGNAL_INDEX, META_INDEX):
        if not index_path.exists():
            continue
        text = index_path.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"`([^`]+)`", text):
            rel = m.group(1).strip()
            if not rel:
                continue
            abs_path = (REPO_ROOT / rel).resolve()
            paths.add(abs_path)
            if abs_path.is_file():
                paths.add(abs_path.parent)
    return paths


def _is_under(path: Path, roots: Set[Path]) -> bool:
    for root in roots:
        try:
            path.resolve().relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _archive_path(doc_path: Path) -> Path:
    rel = doc_path.resolve().relative_to(DOCS_ROOT)
    return ARCHIVE_ROOT / rel


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive non-core docs.")
    parser.add_argument("--apply", action="store_true", help="Actually move files.")
    args = parser.parse_args()

    keep_paths = _load_keep_paths()
    candidates = []

    for p in DOCS_ROOT.rglob("*.md"):
        if p.is_dir():
            continue
        if _is_under(p, KEEP_DIRS):
            continue
        if _is_under(p, keep_paths):
            continue
        # Keep archive itself
        if _is_under(p, {ARCHIVE_ROOT.resolve()}):
            continue
        candidates.append(p)

    report_lines = []
    report_lines.append("# Docs Cleanup Report")
    report_lines.append("")
    report_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    report_lines.append("")
    report_lines.append("## Archived Docs")
    for p in candidates:
        report_lines.append(f"- {p.relative_to(REPO_ROOT)}")

    report_path = ARCHIVE_ROOT / f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if args.apply:
        for p in candidates:
            dest = _archive_path(p)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise RuntimeError(f"Archive target already exists: {dest}")
            shutil.move(str(p), str(dest))

    print(f"[ok] archived_docs: {len(candidates)}")
    print(f"[ok] report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
