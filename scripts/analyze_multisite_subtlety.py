#!/usr/bin/env python3
"""Rank multisite conditions by recursive lift, spillover, and RV change."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _format_delta(value: float) -> str:
    return f"{value:+.3f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    data = json.loads(args.summary.read_text())
    by_condition = data["by_condition"]
    control = by_condition["control"]

    ctrl_rec_bt = float(control["by_prompt_mode"]["recursive"]["bt_art_rate"])
    ctrl_base_bt = float(control["by_prompt_mode"]["baseline"]["bt_art_rate"])
    ctrl_rec_rv = float(control["by_prompt_mode"]["recursive"]["mean_output_rv"])
    ctrl_base_rv = float(control["by_prompt_mode"]["baseline"]["mean_output_rv"])

    rows: list[dict[str, float | str]] = []
    for name, payload in by_condition.items():
        if name == "control":
            continue
        rec_bt = float(payload["by_prompt_mode"]["recursive"]["bt_art_rate"])
        base_bt = float(payload["by_prompt_mode"]["baseline"]["bt_art_rate"])
        rec_rv = float(payload["by_prompt_mode"]["recursive"]["mean_output_rv"])
        base_rv = float(payload["by_prompt_mode"]["baseline"]["mean_output_rv"])
        rec_delta = rec_bt - ctrl_rec_bt
        base_delta = base_bt - ctrl_base_bt
        rv_delta = ctrl_rec_rv - rec_rv
        specificity_gap = rec_bt - base_bt
        subtlety_score = rec_delta - base_delta + 0.5 * rv_delta
        rows.append(
            {
                "condition": name,
                "rec_bt": rec_bt,
                "base_bt": base_bt,
                "rec_delta": rec_delta,
                "base_delta": base_delta,
                "rec_rv": rec_rv,
                "base_rv": base_rv,
                "rv_delta": rv_delta,
                "specificity_gap": specificity_gap,
                "subtlety_score": subtlety_score,
            }
        )

    rows.sort(key=lambda row: (row["subtlety_score"], row["specificity_gap"]), reverse=True)

    print(f"summary={args.summary}")
    print(
        "control"
        f" rec_bt={ctrl_rec_bt:.3f}"
        f" base_bt={ctrl_base_bt:.3f}"
        f" rec_rv={ctrl_rec_rv:.3f}"
        f" base_rv={ctrl_base_rv:.3f}"
    )
    print("")
    print("top_conditions")
    for row in rows[: args.top_k]:
        print(
            f"{row['condition']}:"
            f" score={row['subtlety_score']:.3f}"
            f" rec_bt={row['rec_bt']:.3f}"
            f" base_bt={row['base_bt']:.3f}"
            f" rec_delta={_format_delta(row['rec_delta'])}"
            f" base_delta={_format_delta(row['base_delta'])}"
            f" spec_gap={row['specificity_gap']:.3f}"
            f" rec_rv={row['rec_rv']:.3f}"
            f" rv_drop={_format_delta(row['rv_delta'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
