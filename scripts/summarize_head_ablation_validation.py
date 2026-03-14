#!/usr/bin/env python3
"""Summarize legacy head-ablation validation runs into a single CSV."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def mean_col(rows: list[dict], column: str) -> float | None:
    vals = []
    for row in rows:
        raw = row.get(column)
        if raw in (None, "", "nan", "NaN"):
            continue
        vals.append(float(raw))
    if not vals:
        return None
    return sum(vals) / len(vals)


def load_csv_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def run_row(run_dir: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    metadata = json.loads((run_dir / "metadata.json").read_text())
    rows = load_csv_rows(run_dir / "head_ablation_results.csv")

    recursive_rows = [r for r in rows if r.get("prompt_type") == "recursive"]
    baseline_rows = [r for r in rows if r.get("prompt_type") == "baseline"]
    params = config.get("params", {})
    notes = summary.get("notes", {})

    return {
        "run_dir": str(run_dir),
        "timestamp": summary.get("timestamp"),
        "model": summary.get("model_name") or config.get("model", {}).get("name"),
        "prompt_bank_version": summary.get("prompt_bank_version"),
        "seed": summary.get("seed"),
        "n_pairs": summary.get("n_pairs"),
        "n_recursive_actual": summary.get("n_recursive_actual"),
        "n_baseline_actual": summary.get("n_baseline_actual"),
        "target_layer": params.get("target_layer"),
        "target_kv_head": params.get("target_kv_head"),
        "control_layer": params.get("control_layer"),
        "control_kv_head": params.get("control_kv_head"),
        "early_layer": params.get("early_layer"),
        "window": params.get("window"),
        "gqa_aliasing_note": notes.get("gqa_aliasing"),
        "num_kv_heads": notes.get("num_kv_heads"),
        "target_vs_control_recursive_mean": mean_col(recursive_rows, "delta_target_at_target_layer") - mean_col(recursive_rows, "delta_control_head_at_target_layer")
        if mean_col(recursive_rows, "delta_target_at_target_layer") is not None and mean_col(recursive_rows, "delta_control_head_at_target_layer") is not None else None,
        "target_vs_control_baseline_mean": mean_col(baseline_rows, "delta_target_at_target_layer") - mean_col(baseline_rows, "delta_control_head_at_target_layer")
        if mean_col(baseline_rows, "delta_target_at_target_layer") is not None and mean_col(baseline_rows, "delta_control_head_at_target_layer") is not None else None,
        "recursive_target_delta_mean": mean_col(recursive_rows, "delta_target_at_target_layer"),
        "recursive_control_head_delta_mean": mean_col(recursive_rows, "delta_control_head_at_target_layer"),
        "recursive_wrong_layer_delta_mean": mean_col(recursive_rows, "delta_target_at_control_layer"),
        "baseline_target_delta_mean": mean_col(baseline_rows, "delta_target_at_target_layer"),
        "baseline_control_head_delta_mean": mean_col(baseline_rows, "delta_control_head_at_target_layer"),
        "baseline_wrong_layer_delta_mean": mean_col(baseline_rows, "delta_target_at_control_layer"),
        "rv_delta_mean": summary.get("rv_delta_mean"),
        "rv_cohens_d": summary.get("rv_cohens_d"),
        "rv_p_value": summary.get("rv_p_value"),
        "recursive_target_ci_low": summary.get("analysis", {}).get("recursive", {}).get("target_at_target_layer", {}).get("ci_95_low"),
        "recursive_target_ci_high": summary.get("analysis", {}).get("recursive", {}).get("target_at_target_layer", {}).get("ci_95_high"),
        "recursive_target_vs_control_p": summary.get("comparisons", {}).get("recursive_target_vs_control_head", {}).get("p_value"),
        "recursive_target_vs_control_t": summary.get("comparisons", {}).get("recursive_target_vs_control_head", {}).get("t_stat"),
        "recursive_target_vs_wrong_layer_p": summary.get("comparisons", {}).get("recursive_L27_vs_L21", {}).get("p_value"),
        "recursive_target_vs_wrong_layer_t": summary.get("comparisons", {}).get("recursive_L27_vs_L21", {}).get("t_stat"),
        "pass_checks": ";".join(
            f"{chk.get('check')}={chk.get('passed')}" for chk in summary.get("pass_checks", [])
        ),
        "git_commit": metadata.get("git_commit"),
        "intervention_scope": metadata.get("intervention_scope"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("results/phase1_mechanism/runs"),
        help="Directory containing head_ablation_validation run folders",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*head_ablation_validation*",
        help="Glob pattern for run directories",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: results/phase1_mechanism with timestamp)",
    )
    args = parser.parse_args()

    run_dirs = sorted(p for p in args.runs_root.glob(args.pattern) if p.is_dir())
    if not run_dirs:
        raise SystemExit(f"No run directories matched {args.runs_root / args.pattern}")

    rows = []
    for run_dir in run_dirs:
        csv_path = run_dir / "head_ablation_results.csv"
        summary_path = run_dir / "summary.json"
        if not csv_path.exists() or not summary_path.exists():
            continue
        rows.append(run_row(run_dir))

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["timestamp", "run_dir"], ascending=[True, True], na_position="last")

    if args.out is None:
        out = args.runs_root / f"head_ablation_validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    else:
        out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Wrote {len(df)} rows to {out}")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(df)


if __name__ == "__main__":
    main()
