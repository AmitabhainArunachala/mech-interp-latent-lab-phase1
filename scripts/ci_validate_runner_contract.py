#!/usr/bin/env python3
"""CI check: ensure runner keeps required artifact and key contract hooks."""

from __future__ import annotations

import sys
from pathlib import Path

REQUIRED_TOKENS = [
    "summary.json",
    "prompt_bank_version.json",
    "report.md",
    "metadata.json",
    "hardware_info.json",
    "manifest.json",
    "RUN_INDEX.jsonl",
    "MULTI_TOKEN_REQUIRED_KEYS",
    "rv_cohens_d",
    "rv_p_value",
]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    run_py = repo_root / "src" / "pipelines" / "run.py"

    if not run_py.exists():
        print("[FAIL] src/pipelines/run.py missing")
        return 1

    text = run_py.read_text(encoding="utf-8")
    missing = [tok for tok in REQUIRED_TOKENS if tok not in text]

    if missing:
        print("[FAIL] runner contract tokens missing:")
        for tok in missing:
            print(f"  - {tok}")
        return 1

    print("[PASS] runner contract tokens present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
