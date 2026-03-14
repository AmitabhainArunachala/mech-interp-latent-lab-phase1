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
        "model": obj.get("model"),
        "seed": obj.get("seed"),
        "early": obj.get("early"),
        "late": obj.get("late"),
        "r_layer": obj.get("r_layer"),
        "v_layer": obj.get("v_layer"),
        "rv_window": obj.get("rv_window"),
        "max_turns_per_session": obj.get("max_turns_per_session"),
        "dual_alpha": obj.get("dual_alpha"),
        "temperature": obj.get("temperature"),
        "rep_penalty": obj.get("rep_penalty"),
        "n_sessions": obj.get("n_sessions_per_condition"),
        "conditions": ",".join(obj.get("selected_conditions", [])),
        "baseline_rate": get_nested(agg, ["clean_baseline", "bt_art_rate"]),
        "baseline_mean_rv": get_nested(agg, ["clean_baseline", "mean_rv"]),
        "kv_only_rate": get_nested(agg, ["kv_only", "bt_art_rate"]),
        "kv_only_mean_rv": get_nested(agg, ["kv_only", "mean_rv"]),
        "dual_patch_rate": get_nested(agg, ["dual_patch", "bt_art_rate"]),
        "dual_patch_mean_rv": get_nested(agg, ["dual_patch", "mean_rv"]),
        "kv_plus_dual_rate": get_nested(agg, ["kv_plus_dual", "bt_art_rate"]),
        "kv_plus_dual_mean_rv": get_nested(agg, ["kv_plus_dual", "mean_rv"]),
        "recursive_rate": get_nested(agg, ["clean_recursive", "bt_art_rate"]),
        "recursive_mean_rv": get_nested(agg, ["clean_recursive", "mean_rv"]),
        "kv_only_turn_or": get_nested(cmp_, ["kv_only_vs_baseline", "turn_level", "or"]),
        "kv_only_vs_baseline_p": get_nested(cmp_, ["kv_only_vs_baseline", "turn_level", "p"]),
        "kv_only_session_d": get_nested(cmp_, ["kv_only_vs_baseline", "session_level", "cohens_d"]),
        "dual_turn_or": get_nested(cmp_, ["dual_patch_vs_baseline", "turn_level", "or"]),
        "dual_vs_baseline_p": get_nested(cmp_, ["dual_patch_vs_baseline", "turn_level", "p"]),
        "dual_session_d": get_nested(cmp_, ["dual_patch_vs_baseline", "session_level", "cohens_d"]),
        "kvd_turn_or": get_nested(cmp_, ["kv_plus_dual_vs_baseline", "turn_level", "or"]),
        "kvd_vs_baseline_p": get_nested(cmp_, ["kv_plus_dual_vs_baseline", "turn_level", "p"]),
        "kvd_session_d": get_nested(cmp_, ["kv_plus_dual_vs_baseline", "session_level", "cohens_d"]),
        "kvd_vs_dual_turn_or": get_nested(cmp_, ["kv_plus_dual_vs_dual_patch", "turn_level", "or"]),
        "kvd_vs_dual_p": get_nested(cmp_, ["kv_plus_dual_vs_dual_patch", "turn_level", "p"]),
        "kvd_vs_dual_session_d": get_nested(cmp_, ["kv_plus_dual_vs_dual_patch", "session_level", "cohens_d"]),
        "clean_recursive_turn_or": get_nested(cmp_, ["clean_recursive_vs_baseline", "turn_level", "or"]),
        "clean_recursive_session_d": get_nested(cmp_, ["clean_recursive_vs_baseline", "session_level", "cohens_d"]),
        "prereg_evaluated": prereg.get("evaluated"),
        "prereg_pass": prereg.get("pass"),
        "prereg_lift": prereg.get("observed_lift"),
        "prereg_turn_level_p": prereg.get("turn_level_p"),
        "prereg_session_level_p": prereg.get("session_level_permutation_p"),
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
