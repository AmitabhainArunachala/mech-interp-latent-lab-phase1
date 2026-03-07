#!/usr/bin/env python3
"""Summarize sufficiency hardening JSON runs into a single CSV."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def get_nested(d, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def run_row(path: Path) -> dict:
    obj = json.loads(path.read_text())
    agg = obj.get("aggregated", {})
    cmp_ = obj.get("comparisons", {})
    prereg = cmp_.get("preregistered_decision", {})

    row = {
        "file": path.name,
        "timestamp": obj.get("timestamp"),
        "seed": obj.get("seed"),
        "dual_alpha": obj.get("dual_alpha"),
        "temperature": obj.get("temperature"),
        "rep_penalty": obj.get("rep_penalty"),
        "n_sessions": obj.get("n_sessions_per_condition"),
        "conditions": ",".join(obj.get("selected_conditions", [])),
        "baseline_rate": get_nested(agg, ["clean_baseline", "bt_art_rate"]),
        "kv_only_rate": get_nested(agg, ["kv_only", "bt_art_rate"]),
        "dual_patch_rate": get_nested(agg, ["dual_patch", "bt_art_rate"]),
        "kv_plus_dual_rate": get_nested(agg, ["kv_plus_dual", "bt_art_rate"]),
        "recursive_rate": get_nested(agg, ["clean_recursive", "bt_art_rate"]),
        "kv_only_vs_baseline_p": get_nested(cmp_, ["kv_only_vs_baseline", "turn_level", "p"]),
        "dual_vs_baseline_p": get_nested(cmp_, ["dual_patch_vs_baseline", "turn_level", "p"]),
        "kvd_vs_baseline_p": get_nested(cmp_, ["kv_plus_dual_vs_baseline", "turn_level", "p"]),
        "kvd_vs_dual_p": get_nested(cmp_, ["kv_plus_dual_vs_dual_patch", "turn_level", "p"]),
        "prereg_evaluated": prereg.get("evaluated"),
        "prereg_pass": prereg.get("pass"),
        "prereg_lift": prereg.get("observed_lift"),
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/sufficiency_ladder"),
        help="Directory containing sufficiency_ladder JSON outputs",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="sufficiency_ladder_*.json",
        help="Filename glob pattern",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: results dir with timestamp)",
    )
    args = parser.parse_args()

    files = sorted(args.results_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched {args.results_dir / args.pattern}")

    rows = []
    for p in files:
        try:
            rows.append(run_row(p))
        except Exception as e:
            rows.append({"file": p.name, "error": str(e)})

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["timestamp", "file"], ascending=[True, True], na_position="last")

    if args.out is None:
        out = args.results_dir / f"hardening_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    else:
        out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Wrote {len(df)} rows to {out}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.tail(20))


if __name__ == "__main__":
    main()
