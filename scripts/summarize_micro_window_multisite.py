#!/usr/bin/env python3
"""Summarize surgical micro-window multisite runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def summarize_condition(records: list[dict[str, Any]]) -> dict[str, Any]:
    overall_counts = Counter(str(r.get("classification") or "UNKNOWN") for r in records)
    overall_rvs = [float(r["output_rv"]) for r in records if r.get("output_rv") is not None]
    by_mode = defaultdict(list)
    for row in records:
        by_mode[str(row["prompt_mode"])].append(row)

    def mode_bt_art(rows: list[dict[str, Any]]) -> float:
        return rate(sum(int(r.get("bt_art", 0)) for r in rows), len(rows))

    def mode_mean_rv(rows: list[dict[str, Any]]) -> float:
        vals = [float(r["output_rv"]) for r in rows if r.get("output_rv") is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    return {
        "recursive_bt_art": mode_bt_art(by_mode["recursive"]),
        "baseline_bt_art": mode_bt_art(by_mode["baseline"]),
        "recursive_mean_output_rv": mode_mean_rv(by_mode["recursive"]),
        "baseline_mean_output_rv": mode_mean_rv(by_mode["baseline"]),
        "mean_output_rv": sum(overall_rvs) / len(overall_rvs) if overall_rvs else float("nan"),
        "malformed_rate": rate(overall_counts["MALFORMED"], len(records)),
        "repetitive_rate": rate(overall_counts["REPETITIVE"], len(records)),
        "n": len(records),
        "class_counts": dict(overall_counts),
    }


def interesting(summary: dict[str, Any], bridge_only_3: dict[str, Any], tol: float = 1e-9) -> bool:
    better_recursive = summary["recursive_bt_art"] > bridge_only_3["recursive_bt_art"] + tol
    no_baseline_regression = summary["baseline_bt_art"] <= bridge_only_3["baseline_bt_art"] + tol
    no_malformed_regression = summary["malformed_rate"] <= bridge_only_3["malformed_rate"] + tol
    no_repetitive_regression = summary["repetitive_rate"] <= bridge_only_3["repetitive_rate"] + tol
    return (
        better_recursive
        and no_baseline_regression
        and no_malformed_regression
        and no_repetitive_regression
    )


def score(summary: dict[str, Any], control: dict[str, Any], bridge_only_3: dict[str, Any]) -> dict[str, float]:
    rec_delta_vs_ctrl = summary["recursive_bt_art"] - control["recursive_bt_art"]
    rec_delta_vs_bridge = summary["recursive_bt_art"] - bridge_only_3["recursive_bt_art"]
    base_delta_vs_ctrl = summary["baseline_bt_art"] - control["baseline_bt_art"]
    base_delta_vs_bridge = summary["baseline_bt_art"] - bridge_only_3["baseline_bt_art"]
    rv_drop_vs_ctrl = control["recursive_mean_output_rv"] - summary["recursive_mean_output_rv"]
    rv_drop_vs_bridge = bridge_only_3["recursive_mean_output_rv"] - summary["recursive_mean_output_rv"]
    safety_score = rec_delta_vs_ctrl - max(base_delta_vs_ctrl, 0.0) + 0.5 * rv_drop_vs_ctrl
    bridge_score = rec_delta_vs_bridge - max(base_delta_vs_bridge, 0.0) + 0.5 * rv_drop_vs_bridge
    return {
        "recursive_delta_vs_control": rec_delta_vs_ctrl,
        "recursive_delta_vs_bridge_only_3": rec_delta_vs_bridge,
        "baseline_delta_vs_control": base_delta_vs_ctrl,
        "baseline_delta_vs_bridge_only_3": base_delta_vs_bridge,
        "recursive_rv_drop_vs_control": rv_drop_vs_ctrl,
        "recursive_rv_drop_vs_bridge_only_3": rv_drop_vs_bridge,
        "safety_score_vs_control": safety_score,
        "safety_score_vs_bridge_only_3": bridge_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir
    records = load_records(run_dir / "benchmark_records.jsonl")
    by_condition_records = defaultdict(list)
    for row in records:
        by_condition_records[str(row["condition_name"])].append(row)

    by_condition = {
        name: summarize_condition(rows)
        for name, rows in sorted(by_condition_records.items())
    }
    control = by_condition["control"]
    bridge_only_3 = by_condition["bridge_only_3"]
    scored_conditions = {
        name: {
            **summary,
            **score(summary, control, bridge_only_3),
            "beats_bridge_only_3_safely": interesting(summary, bridge_only_3),
        }
        for name, summary in by_condition.items()
    }
    interesting_conditions = [
        name for name, summary in by_condition.items()
        if name not in {"control", "bridge_only_2", "bridge_only_3"}
        and interesting(summary, bridge_only_3)
    ]
    ranked_conditions = sorted(
        (
            {"condition": name, **payload}
            for name, payload in scored_conditions.items()
            if name not in {"control", "bridge_only_2", "bridge_only_3"}
        ),
        key=lambda item: (
            item["safety_score_vs_bridge_only_3"],
            item["recursive_delta_vs_bridge_only_3"],
            -max(item["baseline_delta_vs_bridge_only_3"], 0.0),
        ),
        reverse=True,
    )

    payload = {
        "run_dir": str(run_dir),
        "bridge_only_3_reference": bridge_only_3,
        "control_reference": control,
        "by_condition": scored_conditions,
        "interesting_conditions": interesting_conditions,
        "ranked_conditions": ranked_conditions,
        "decision_rule": (
            "interesting iff recursive BT+ART > bridge_only_3 and baseline BT+ART, "
            "malformed_rate, repetitive_rate do not exceed bridge_only_3"
        ),
    }

    out_path = run_dir / "micro_window_summary.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
