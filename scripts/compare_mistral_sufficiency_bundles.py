#!/usr/bin/env python3
"""Compare Mistral sufficiency-bundle runs side by side."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_pointer(bundle_dir: Path, stem: str) -> Path | None:
    txt = bundle_dir / f"{stem}.txt"
    if not txt.exists():
        return None
    return Path(txt.read_text().strip())


def rel(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def bridge_row(summary: dict[str, Any], label: str) -> dict[str, Any]:
    analysis = summary["analysis"]
    if "temp_0.7" in analysis:
        primary = analysis["temp_0.7"]
        primary_temp = "temp_0.7"
    else:
        first = sorted(analysis.keys())[0]
        primary = analysis[first]
        primary_temp = first
    return {
        "label": label,
        "primary_temp": primary_temp,
        "n_total": summary.get("n_total_prompts"),
        "max_new_tokens": summary.get("max_new_tokens"),
        "rv_recursive_mean": summary.get("rv_recursive_mean"),
        "rv_baseline_mean": summary.get("rv_baseline_mean"),
        "rv_d": summary.get("rv_cohens_d"),
        "rv_p": summary.get("rv_p_value"),
        "h1_basis": primary.get("h1_basis"),
        "h1_r": primary.get("h1_spearman_r"),
        "h1_p": primary.get("h1_spearman_p"),
        "h1_significant": primary.get("h1_significant"),
        "h1_all_r": primary.get("h1_all_spearman_r"),
        "h1_all_p": primary.get("h1_all_spearman_p"),
        "h3_r": primary.get("h3_point_biserial_r"),
        "h3_p": primary.get("h3_point_biserial_p"),
        "h3_significant": primary.get("h3_significant"),
        "n_non_truncated": primary.get("n_non_truncated"),
        "n_total_primary": primary.get("n_total"),
        "pct_truncated": primary.get("pct_truncated"),
    }


def self_feed_row(summary: dict[str, Any], condition: str) -> dict[str, Any]:
    stats = summary["conditions"][condition]
    return {
        "condition": condition,
        "n_sessions": stats.get("n_sessions"),
        "n_turns": stats.get("n_turns"),
        "bt_art_rate": stats.get("bt_art_rate"),
        "mean_rv": stats.get("mean_rv"),
        "segment_stats": stats.get("segment_stats", {}),
    }


def sustained_row(summary: dict[str, Any]) -> dict[str, Any]:
    metric = summary["metric_stats"]["rv"]
    return {
        "n_recursive": summary.get("n_recursive"),
        "n_baseline": summary.get("n_baseline"),
        "rv_rec_mean": metric.get("rec_mean"),
        "rv_bas_mean": metric.get("bas_mean"),
        "rv_d": metric.get("d"),
        "rv_p": metric.get("p"),
        "segment_stats": summary.get("segment_stats", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-v1", type=Path, required=True)
    parser.add_argument("--bundle-v2", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    v1_manifest = load_json(args.bundle_v1 / "bundle_manifest.json")
    v2_manifest = load_json(args.bundle_v2 / "bundle_manifest.json")

    bridge_v1_path = Path(v1_manifest["artifacts"]["bridge_low_trunc_n12_run_dir"]) / "summary.json"
    bridge_v2_path = Path(v2_manifest["artifacts"]["bridge_low_trunc_confirmatory_n24_run_dir"]) / "summary.json"
    longgen_v2_path = Path(v2_manifest["artifacts"]["bridge_true_longgen_n18_run_dir"]) / "summary.json"
    self_v1_path = Path(v1_manifest["artifacts"]["self_feeding_summary_artifact"])
    self_v2_path = Path(v2_manifest["artifacts"]["self_feeding_summary_v2_artifact"])
    gnani_v1_path = Path(v1_manifest["artifacts"]["sustained_gnani_summary_artifact"])
    gnani_v2_path = Path(v2_manifest["artifacts"]["sustained_gnani_summary_v2_artifact"])

    payload = {
        "bundle_v1": rel(args.bundle_v1),
        "bundle_v2": rel(args.bundle_v2),
        "bridge": {
            "v1_low_trunc_n12": bridge_row(load_json(bridge_v1_path), "v1_low_trunc_n12"),
            "v2_low_trunc_n24": bridge_row(load_json(bridge_v2_path), "v2_low_trunc_n24"),
            "v2_true_longgen_n18": bridge_row(load_json(longgen_v2_path), "v2_true_longgen_n18"),
        },
        "self_feeding": {
            "v1_self_feed_recursive": self_feed_row(load_json(self_v1_path), "self_feed_recursive"),
            "v1_self_feed_baseline": self_feed_row(load_json(self_v1_path), "self_feed_baseline"),
            "v1_gnani_scaffolded": self_feed_row(load_json(self_v1_path), "gnani_scaffolded"),
            "v2_self_feed_recursive": self_feed_row(load_json(self_v2_path), "self_feed_recursive"),
            "v2_self_feed_baseline": self_feed_row(load_json(self_v2_path), "self_feed_baseline"),
            "v2_gnani_scaffolded": self_feed_row(load_json(self_v2_path), "gnani_scaffolded"),
        },
        "sustained_gnani": {
            "v1": sustained_row(load_json(gnani_v1_path)),
            "v2": sustained_row(load_json(gnani_v2_path)),
        },
        "artifacts": {
            "bridge_v1_summary": rel(bridge_v1_path),
            "bridge_v2_summary": rel(bridge_v2_path),
            "longgen_v2_summary": rel(longgen_v2_path),
            "self_v1_summary": rel(self_v1_path),
            "self_v2_summary": rel(self_v2_path),
            "gnani_v1_summary": rel(gnani_v1_path),
            "gnani_v2_summary": rel(gnani_v2_path),
        },
    }

    if args.out is None:
        out = args.bundle_v2 / "bundle_comparison.json"
    else:
        out = args.out
    out.write_text(json.dumps(payload, indent=2))
    print(out)


if __name__ == "__main__":
    main()
