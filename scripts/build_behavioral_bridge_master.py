#!/usr/bin/env python3
"""Build behavioral n-boost summary + master evidence artifact."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT = Path(".")
SESSIONS_DIR = ROOT / "results" / "sustained_gnani_v3_fixed"
WITHIN_DIR = ROOT / "results" / "within_session_bridge"
BRIDGE_DIR = ROOT / "results" / "bridge_battery"
CLASS_DIR = ROOT / "results" / "classifier_evaluation"
TOKEN_DIR = ROOT / "results" / "batch_per_token_rv"
PRE_SNAPSHOT = ROOT / "results" / "behavioral_nboost_pre_snapshot.json"
FINAL_CORR = ROOT / "industry_grade" / "2026-02-20" / "evidence" / "final_correlations.json"


def latest_json(path: Path, pattern: str):
    matches = sorted(path.glob(pattern))
    return matches[-1] if matches else None


def load_json(path: Path | None) -> Dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def count_sessions() -> Dict[str, int]:
    rec = bas = turns_rec = turns_bas = bt_rec = bt_bas = 0
    for fp in sorted(SESSIONS_DIR.glob("*.json")):
        if fp.name == "comparison_summary.json":
            continue
        data = load_json(fp)
        mode = data.get("mode")
        turns = data.get("turns", [])
        bt = sum(1 for t in turns if t.get("classification") in ("ARTICULATE", "BREAKTHROUGH"))
        if mode == "recursive":
            rec += 1
            turns_rec += len(turns)
            bt_rec += bt
        elif mode == "baseline":
            bas += 1
            turns_bas += len(turns)
            bt_bas += bt
    return {
        "n_recursive_sessions": rec,
        "n_baseline_sessions": bas,
        "n_recursive_turns": turns_rec,
        "n_baseline_turns": turns_bas,
        "n_bt_art_recursive": bt_rec,
        "n_bt_art_baseline": bt_bas,
    }


def pull_within_stats(within: Dict) -> Tuple[float | None, float | None]:
    rec_out = within.get("pooled", {}).get("recursive_only", {}).get("output_rv", {})
    return rec_out.get("cohens_d"), rec_out.get("mannwhitney_p")


def output_rv_ci95_from_sessions(n_bootstrap: int = 2000, seed: int = 42) -> list[float | None]:
    """Bootstrap CI for mean(BT+ART output_rv) - mean(Other output_rv) on recursive turns."""
    hi: list[float] = []
    lo: list[float] = []
    for fp in sorted(SESSIONS_DIR.glob("*.json")):
        if fp.name == "comparison_summary.json":
            continue
        data = load_json(fp)
        if data.get("mode") != "recursive":
            continue
        for t in data.get("turns", []):
            rv = t.get("output_rv")
            cls = t.get("classification")
            try:
                rvf = float(rv)
            except Exception:
                continue
            if not np.isfinite(rvf):
                continue
            if cls in ("ARTICULATE", "BREAKTHROUGH"):
                hi.append(rvf)
            elif cls in ("REPETITIVE", "SURFACE", "CONCEPTUAL"):
                lo.append(rvf)

    if len(hi) < 2 or len(lo) < 2:
        return [None, None]

    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    hi_arr = np.array(hi, dtype=float)
    lo_arr = np.array(lo, dtype=float)
    for _ in range(n_bootstrap):
        hi_s = hi_arr[rng.integers(0, len(hi_arr), len(hi_arr))]
        lo_s = lo_arr[rng.integers(0, len(lo_arr), len(lo_arr))]
        diffs.append(float(np.mean(hi_s) - np.mean(lo_s)))
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def main() -> int:
    counts = count_sessions()

    within_file = latest_json(WITHIN_DIR, "within_session_bridge_*.json")
    bridge_file = latest_json(BRIDGE_DIR, "bridge_battery_*.json")
    classifier_file = latest_json(CLASS_DIR, "classifier_eval_*.json")
    token_file = latest_json(TOKEN_DIR, "batch_per_token_rv_*.json")

    within = load_json(within_file)
    bridge = load_json(bridge_file)
    clf = load_json(classifier_file)
    token = load_json(token_file)
    pre = load_json(PRE_SNAPSHOT)
    final_corr = load_json(FINAL_CORR)

    new_d, new_p = pull_within_stats(within)
    old_d = pre.get("within_session_output_rv", {}).get("cohens_d")
    old_p = pre.get("within_session_output_rv", {}).get("mannwhitney_p")
    old_auc = pre.get("bridge_battery_recursive_logistic", {}).get("cv_auc_mean")
    new_auc = bridge.get("logistic_regression", {}).get("recursive_only", {}).get("cv_auc_mean")

    temporal = bridge.get("temporal_lag", {}).get("pooled", {}).get("recursive_lag1", {}).get("output_rv", {})
    token_summary = token.get("summary", {})
    clf_all = clf.get("all_sessions", {})
    clf_base = clf.get("baseline_only", {})
    rv_ci95 = output_rv_ci95_from_sessions()

    c2 = final_corr.get("semantic", {}).get("c2_spearman", {}).get("rv_mean_vs_semantic_score", {})

    nboost_summary = {
        "timestamp": datetime.now().isoformat(),
        "old_counts": pre.get("old_counts", {}),
        "new_counts": counts,
        "total_sessions": {
            "recursive": counts["n_recursive_sessions"],
            "baseline": counts["n_baseline_sessions"],
        },
        "total_turns": {
            "recursive": counts["n_recursive_turns"],
            "baseline": counts["n_baseline_turns"],
        },
        "total_bt_art": {
            "recursive": counts["n_bt_art_recursive"],
            "baseline": counts["n_bt_art_baseline"],
        },
        "within_session_bridge": {
            "output_rv_cohens_d_old": old_d,
            "output_rv_cohens_d_new": new_d,
            "output_rv_p_old": old_p,
            "output_rv_p_new": new_p,
            "delta_cohens_d": (new_d - old_d) if old_d is not None and new_d is not None else None,
            "logistic_auc_old": old_auc,
            "logistic_auc_new": new_auc,
            "delta_logistic_auc": (new_auc - old_auc) if old_auc is not None and new_auc is not None else None,
            "within_session_file": str(within_file) if within_file else None,
            "bridge_battery_file": str(bridge_file) if bridge_file else None,
        },
    }

    master = {
        "within_session_bridge": {
            **counts,
            "output_rv_cohens_d": new_d,
            "output_rv_p_value": new_p,
            "output_rv_ci95": rv_ci95,
            "temporal_lag_rho": temporal.get("r"),
            "temporal_lag_p": temporal.get("p"),
        },
        "logistic_classifier": {
            "train_n": clf_all.get("train_n"),
            "test_n": clf_all.get("test_n"),
            "rv_alone_test_auc": clf_all.get("rv_alone_test_auc"),
            "multi_metric_test_auc": clf_all.get("test_auc"),
            "multi_metric_auc_ci95": clf_all.get("bootstrap_test_auc_ci95"),
            "baseline_only_auc": clf_base.get("test_auc"),
            "top_3_features": clf_all.get("top_3_features"),
        },
        "per_token_rv": {
            "n_recursive": token_summary.get("n_recursive"),
            "n_baseline": token_summary.get("n_baseline"),
            "max_new_tokens": token_summary.get("max_new_tokens"),
            "mean_gen_rv_recursive": token_summary.get("mean_generation_rv_recursive"),
            "mean_gen_rv_baseline": token_summary.get("mean_generation_rv_baseline"),
            "cohens_d": token_summary.get("cohens_d_recursive_vs_baseline"),
            "p_value": token_summary.get("mannwhitney_p"),
            "biserial_r_with_bt_art": token_summary.get("pointbiserial_all", {}).get("r"),
            "biserial_p": token_summary.get("pointbiserial_all", {}).get("p"),
        },
        "c2_semantic": {
            "n_total": c2.get("n"),
            "rho": c2.get("rho"),
            "p": c2.get("p_value"),
        },
        "verdict": None,
    }

    test_auc = master["logistic_classifier"]["multi_metric_test_auc"]
    d_val = master["within_session_bridge"]["output_rv_cohens_d"]
    bt_val = counts["n_bt_art_recursive"]
    if (
        d_val is not None
        and test_auc is not None
        and bt_val is not None
        and d_val < -0.5
        and test_auc > 0.65
        and bt_val >= 200
    ):
        master["verdict"] = "SUCCESS_CRITERIA_MET"
    else:
        master["verdict"] = "PARTIAL_OR_PENDING"

    nboost_path = ROOT / "results" / "behavioral_nboost_summary.json"
    master_path = ROOT / "industry_grade" / "2026-02-20" / "evidence" / "behavioral_bridge_master.json"
    master_path.parent.mkdir(parents=True, exist_ok=True)

    nboost_path.write_text(json.dumps(nboost_summary, indent=2) + "\n", encoding="utf-8")
    master_path.write_text(json.dumps(master, indent=2) + "\n", encoding="utf-8")

    print(nboost_path)
    print(master_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
