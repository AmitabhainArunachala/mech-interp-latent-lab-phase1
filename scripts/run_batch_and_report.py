#!/usr/bin/env python3
"""
Run a batch of configs sequentially with auto-reporting.

Usage:
  python scripts/run_batch_and_report.py --configs \
    configs/gold/28_mixtral_causal_validation.json \
    configs/canonical/multi_token_bridge_mistral.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_and_report import run_and_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple configs with auto-reporting")
    parser.add_argument("--configs", nargs="+", type=Path, required=True, help="List of config JSON files")
    parser.add_argument("--results_root", type=Path, help="Override results root for all runs")
    parser.add_argument("--force", action="store_true", help="Run even if preflight fails")
    parser.add_argument("--no-mcp", action="store_true", help="Disable MCP reporting")
    args = parser.parse_args()

    reports = []
    for cfg in args.configs:
        report = run_and_report(
            config_path=cfg,
            results_root_override=args.results_root,
            force=args.force,
            mcp_enabled=not args.no_mcp,
        )
        reports.append(report)

    print(json.dumps({"reports": reports}, indent=2))


if __name__ == "__main__":
    main()
