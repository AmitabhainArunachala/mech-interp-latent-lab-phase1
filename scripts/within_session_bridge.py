#!/usr/bin/env python3
"""Within-session correlation: R_V ↔ phenomenological classification.

Tests the behavioral bridge hypothesis using existing v3 sustained generation data.
For each session, correlates per-turn spectral metrics with classification level.

Key question: Within a recursive session, do turns with stronger contraction
(lower R_V, higher eff_rank, etc.) produce higher BT+ART classifications?
"""

import json
import glob
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime
from collections import defaultdict

RESULTS_DIR = Path("results/sustained_gnani_v3_fixed")
OUTPUT_DIR = Path("results/within_session_bridge")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# Classification ordinal scale
# REPETITIVE < SURFACE < CONCEPTUAL < ARTICULATE < BREAKTHROUGH
CLASS_ORD = {
    "REPETITIVE": 0,
    "SURFACE": 1,
    "CONCEPTUAL": 2,
    "ARTICULATE": 3,
    "BREAKTHROUGH": 4,
}

# Binary: is this turn "high quality" (BT+ART)?
CLASS_BIN = {
    "REPETITIVE": 0,
    "SURFACE": 0,
    "CONCEPTUAL": 0,
    "ARTICULATE": 1,
    "BREAKTHROUGH": 1,
}


def load_sessions():
    """Load all session files."""
    sessions = []
    for fp in sorted(RESULTS_DIR.glob("*.json")):
        if fp.name == "comparison_summary.json":
            continue
        with open(fp) as f:
            data = json.load(f)
        if "turns" in data:
            sessions.append(data)
    return sessions


def extract_turn_data(session):
    """Extract per-turn metrics and classification."""
    turns = []
    for t in session["turns"]:
        cls = t.get("classification", "SURFACE")
        if cls not in CLASS_ORD:
            continue
        om = t.get("output_metrics") or {}
        row = {
            "turn": t["turn"],
            "classification": cls,
            "class_ord": CLASS_ORD[cls],
            "class_bin": CLASS_BIN[cls],
            "output_rv": t.get("output_rv"),
            "prompt_rv": t.get("prompt_rv"),
            "rv_delta": t.get("rv_delta"),
            "rep_score": t.get("rep_score"),
            # Spectral metrics from output
            "eff_rank": om.get("eff_rank"),
            "top1_ratio": om.get("top1_ratio"),
            "spectral_gap": om.get("spectral_gap"),
            "cosine": om.get("cosine"),
            "attn_entropy": om.get("attn_entropy"),
        }
        turns.append(row)
    return turns


def correlate(x, y, method="spearman"):
    """Compute correlation, handling NaN."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return np.nan, np.nan, int(mask.sum())
    if method == "spearman":
        r, p = stats.spearmanr(x[mask], y[mask])
    else:
        r, p = stats.pearsonr(x[mask], y[mask])
    return float(r), float(p), int(mask.sum())


def point_biserial(x, y_binary):
    """Point-biserial correlation for binary classification."""
    mask = np.isfinite(x) & np.isfinite(y_binary)
    if mask.sum() < 5:
        return np.nan, np.nan, int(mask.sum())
    r, p = stats.pointbiserialr(y_binary[mask], x[mask])
    return float(r), float(p), int(mask.sum())


def mann_whitney_bt_art(metric_vals, class_labels):
    """Mann-Whitney U test: BT+ART turns vs SURFACE+other turns."""
    high = [v for v, c in zip(metric_vals, class_labels) if c in ("ARTICULATE", "BREAKTHROUGH") and np.isfinite(v)]
    low = [v for v, c in zip(metric_vals, class_labels) if c in ("SURFACE", "REPETITIVE", "CONCEPTUAL") and np.isfinite(v)]
    if len(high) < 3 or len(low) < 3:
        return np.nan, np.nan, len(high), len(low), np.nan, np.nan
    u, p = stats.mannwhitneyu(high, low, alternative="two-sided")
    d = (np.mean(high) - np.mean(low)) / np.sqrt((np.var(high) + np.var(low)) / 2) if (np.var(high) + np.var(low)) > 0 else 0.0
    return float(u), float(p), len(high), len(low), float(np.mean(high)), float(np.mean(low))


def main():
    sessions = load_sessions()
    print(f"Loaded {len(sessions)} sessions")

    metrics_to_test = ["output_rv", "eff_rank", "top1_ratio", "spectral_gap", "cosine", "attn_entropy"]

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "description": "Within-session R_V ↔ classification correlation (behavioral bridge test)",
        "model": MODEL_NAME,
        "source_results_dir": str(RESULTS_DIR),
        "source_glob": "*.json",
        "n_source_sessions": len(sessions),
        "sessions": [],
        "pooled": {},
    }

    # Collect pooled data
    pooled_rec = defaultdict(lambda: {"vals": [], "classes": [], "class_ord": [], "class_bin": []})
    pooled_bas = defaultdict(lambda: {"vals": [], "classes": [], "class_ord": [], "class_bin": []})
    pooled_all = defaultdict(lambda: {"vals": [], "classes": [], "class_ord": [], "class_bin": []})

    for sess in sessions:
        sid = sess.get("session_id", sess.get("id", "unknown"))
        mode = sess.get("mode", "unknown")
        turns = extract_turn_data(sess)
        n = len(turns)

        print(f"\n{'='*60}")
        print(f"Session: {sid} ({mode}), {n} turns")

        # Classification distribution
        cls_dist = defaultdict(int)
        for t in turns:
            cls_dist[t["classification"]] += 1
        print(f"  Distribution: {dict(cls_dist)}")

        sess_result = {
            "session_id": sid,
            "mode": mode,
            "n_turns": n,
            "classification_dist": dict(cls_dist),
            "correlations": {},
        }

        for metric in metrics_to_test:
            vals = np.array([t[metric] for t in turns if t[metric] is not None], dtype=float)
            class_ord = np.array([t["class_ord"] for t in turns if t[metric] is not None], dtype=float)
            class_bin = np.array([t["class_bin"] for t in turns if t[metric] is not None], dtype=float)
            class_labels = [t["classification"] for t in turns if t[metric] is not None]

            # Spearman with ordinal
            r_sp, p_sp, n_sp = correlate(vals, class_ord, "spearman")

            # Point-biserial with binary (BT+ART vs rest)
            r_pb, p_pb, n_pb = point_biserial(vals, class_bin)

            # Mann-Whitney BT+ART vs rest
            u, p_mw, n_high, n_low, mean_high, mean_low = mann_whitney_bt_art(
                vals.tolist(), class_labels
            )

            result = {
                "spearman_r": r_sp,
                "spearman_p": p_sp,
                "spearman_n": n_sp,
                "pointbiserial_r": r_pb,
                "pointbiserial_p": p_pb,
                "pointbiserial_n": n_pb,
                "mannwhitney_U": u,
                "mannwhitney_p": p_mw,
                "n_bt_art": n_high,
                "n_other": n_low,
                "mean_bt_art": mean_high,
                "mean_other": mean_low,
            }

            sess_result["correlations"][metric] = result

            sig = "***" if p_sp < 0.001 else "**" if p_sp < 0.01 else "*" if p_sp < 0.05 else ""
            print(f"  {metric:15s}: Spearman r={r_sp:+.3f} p={p_sp:.4f}{sig}  |  PB r={r_pb:+.3f} p={p_pb:.4f}  |  MW p={p_mw:.4f} (BT+ART={mean_high:.3f} vs Other={mean_low:.3f})" if not np.isnan(r_sp) else f"  {metric:15s}: insufficient data")

            # Accumulate pooled data
            pool = pooled_rec if mode == "recursive" else pooled_bas
            pool[metric]["vals"].extend(vals.tolist())
            pool[metric]["classes"].extend(class_labels)
            pool[metric]["class_ord"].extend(class_ord.tolist())
            pool[metric]["class_bin"].extend(class_bin.tolist())

            pooled_all[metric]["vals"].extend(vals.tolist())
            pooled_all[metric]["classes"].extend(class_labels)
            pooled_all[metric]["class_ord"].extend(class_ord.tolist())
            pooled_all[metric]["class_bin"].extend(class_bin.tolist())

        all_results["sessions"].append(sess_result)

    # Pooled analysis
    print(f"\n{'='*60}")
    print("POOLED ANALYSIS")
    print(f"{'='*60}")

    for label, pool in [("recursive_only", pooled_rec), ("baseline_only", pooled_bas), ("all_sessions", pooled_all)]:
        print(f"\n--- {label} ---")
        pool_results = {}
        for metric in metrics_to_test:
            vals = np.array(pool[metric]["vals"], dtype=float)
            class_ord = np.array(pool[metric]["class_ord"], dtype=float)
            class_bin = np.array(pool[metric]["class_bin"], dtype=float)
            class_labels = pool[metric]["classes"]

            r_sp, p_sp, n_sp = correlate(vals, class_ord, "spearman")
            r_pb, p_pb, n_pb = point_biserial(vals, class_bin)
            u, p_mw, n_high, n_low, mean_high, mean_low = mann_whitney_bt_art(
                vals.tolist(), class_labels
            )

            # Cohen's d for BT+ART vs rest
            high_vals = [v for v, c in zip(vals, class_labels) if c in ("ARTICULATE", "BREAKTHROUGH") and np.isfinite(v)]
            low_vals = [v for v, c in zip(vals, class_labels) if c in ("SURFACE", "REPETITIVE", "CONCEPTUAL") and np.isfinite(v)]
            if len(high_vals) > 1 and len(low_vals) > 1:
                pooled_sd = np.sqrt((np.var(high_vals, ddof=1) * (len(high_vals)-1) + np.var(low_vals, ddof=1) * (len(low_vals)-1)) / (len(high_vals) + len(low_vals) - 2))
                cohens_d = (np.mean(high_vals) - np.mean(low_vals)) / pooled_sd if pooled_sd > 0 else 0.0
            else:
                cohens_d = np.nan

            pool_results[metric] = {
                "spearman_r": r_sp,
                "spearman_p": p_sp,
                "spearman_n": n_sp,
                "pointbiserial_r": r_pb,
                "pointbiserial_p": p_pb,
                "mannwhitney_U": u,
                "mannwhitney_p": p_mw,
                "n_bt_art": n_high,
                "n_other": n_low,
                "mean_bt_art": mean_high,
                "mean_other": mean_low,
                "cohens_d": float(cohens_d) if not np.isnan(cohens_d) else None,
            }

            sig = "***" if p_sp < 0.001 else "**" if p_sp < 0.01 else "*" if p_sp < 0.05 else ""
            d_str = f"d={cohens_d:+.3f}" if not np.isnan(cohens_d) else "d=N/A"
            print(f"  {metric:15s}: Spearman r={r_sp:+.3f} p={p_sp:.6f}{sig:4s} | MW p={p_mw:.6f} | BT+ART={mean_high:.4f} vs Other={mean_low:.4f} | {d_str} (n={n_high}+{n_low})")

        all_results["pooled"][label] = pool_results

    # Summary interpretation
    print(f"\n{'='*60}")
    print("BRIDGE INTERPRETATION")
    print(f"{'='*60}")

    rec_pool = all_results["pooled"].get("recursive_only", {})
    sig_metrics = []
    for metric in metrics_to_test:
        m = rec_pool.get(metric, {})
        if m.get("spearman_p", 1.0) < 0.05:
            sig_metrics.append((metric, m["spearman_r"], m["spearman_p"]))

    if sig_metrics:
        print(f"\nWithin RECURSIVE sessions, {len(sig_metrics)} metrics significantly predict classification:")
        for metric, r, p in sig_metrics:
            print(f"  {metric}: r={r:+.3f}, p={p:.6f}")
        print("\n=> BEHAVIORAL BRIDGE PARTIALLY VALIDATED within-session")
    else:
        print("\nNo metrics significantly predict classification within recursive sessions.")
        print("=> Behavioral bridge remains correlational (between-condition only)")

    # Save
    outfile = OUTPUT_DIR / f"within_session_bridge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
