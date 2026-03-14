#!/usr/bin/env python3
"""Comprehensive behavioral bridge analysis battery.

Four analyses on existing data:
1. Temporal lag: R_V(t) → classification(t+1) — temporal precedence
2. State transitions: spectral signatures that predict quality transitions
3. Logistic regression: multi-metric BT+ART predictor with cross-validated AUC
4. C2 R_V→recursion_score validation across all C2 measurement suites

No GPU required — all local analyses on existing results.
"""

import json
import csv
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

V3_DIR = Path("results/sustained_gnani_v3_fixed")
C2_DIR = Path("results/canonical/c2_measurement_suite")
C2_KITCHEN = Path("results/phase1_mechanism/runs/20260208_232528_c2_rv_measurement_kitchen_sink_behavioral_transfer")
OUTPUT_DIR = Path("results/bridge_battery")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

CLASS_ORD = {"REPETITIVE": 0, "SURFACE": 1, "CONCEPTUAL": 2, "ARTICULATE": 3, "BREAKTHROUGH": 4}
CLASS_BIN = {"REPETITIVE": 0, "SURFACE": 0, "CONCEPTUAL": 0, "ARTICULATE": 1, "BREAKTHROUGH": 1}


def load_v3_sessions():
    sessions = []
    for fp in sorted(V3_DIR.glob("*.json")):
        if fp.name == "comparison_summary.json":
            continue
        with open(fp) as f:
            data = json.load(f)
        if "turns" in data:
            sessions.append(data)
    return sessions


def extract_turns(session):
    turns = []
    for t in session["turns"]:
        cls = t.get("classification", "SURFACE")
        if cls not in CLASS_ORD:
            continue
        om = t.get("output_metrics") or {}
        turns.append({
            "turn": t["turn"],
            "classification": cls,
            "class_ord": CLASS_ORD[cls],
            "class_bin": CLASS_BIN[cls],
            "output_rv": t.get("output_rv"),
            "eff_rank": om.get("eff_rank"),
            "top1_ratio": om.get("top1_ratio"),
            "spectral_gap": om.get("spectral_gap"),
            "cosine": om.get("cosine"),
            "attn_entropy": om.get("attn_entropy"),
        })
    return turns


# ============================================================
# ANALYSIS 1: TEMPORAL LAG
# ============================================================
def temporal_lag_analysis(sessions):
    """Does R_V at turn t predict classification at turn t+1?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: TEMPORAL LAG (R_V(t) → Classification(t+1))")
    print("=" * 70)

    results = {"by_session": [], "pooled": {}}
    metrics = ["output_rv", "eff_rank", "top1_ratio", "spectral_gap", "attn_entropy"]

    # Collect pooled lag data
    pooled_rec = {m: {"x": [], "y_ord": [], "y_bin": []} for m in metrics}
    pooled_bas = {m: {"x": [], "y_ord": [], "y_bin": []} for m in metrics}

    for sess in sessions:
        sid = sess.get("session_id", "?")
        mode = sess.get("mode", "?")
        turns = extract_turns(sess)

        sess_res = {"session_id": sid, "mode": mode, "n_pairs": len(turns) - 1, "lag_correlations": {}}

        for metric in metrics:
            # Lag-1: metric at t, classification at t+1
            x_t = []
            y_t1_ord = []
            y_t1_bin = []
            for i in range(len(turns) - 1):
                val = turns[i].get(metric)
                if val is not None and np.isfinite(val):
                    x_t.append(val)
                    y_t1_ord.append(turns[i + 1]["class_ord"])
                    y_t1_bin.append(turns[i + 1]["class_bin"])

            x = np.array(x_t, dtype=float)
            y_ord = np.array(y_t1_ord, dtype=float)
            y_bin = np.array(y_t1_bin, dtype=float)

            if len(x) < 10 or not np.all(np.isfinite(x)):
                continue

            r_lag, p_lag = stats.spearmanr(x, y_ord)

            # Also compute same-turn for comparison
            x_same = np.array([turns[i].get(metric) for i in range(len(turns)) if turns[i].get(metric) is not None], dtype=float)
            y_same = np.array([turns[i]["class_ord"] for i in range(len(turns)) if turns[i].get(metric) is not None], dtype=float)
            r_same, p_same = stats.spearmanr(x_same, y_same) if len(x_same) > 5 else (np.nan, np.nan)

            sess_res["lag_correlations"][metric] = {
                "lag1_r": float(r_lag), "lag1_p": float(p_lag),
                "same_r": float(r_same), "same_p": float(p_same),
                "n": len(x),
            }

            pool = pooled_rec if mode == "recursive" else pooled_bas
            pool[metric]["x"].extend(x_t)
            pool[metric]["y_ord"].extend(y_t1_ord)
            pool[metric]["y_bin"].extend(y_t1_bin)

        results["by_session"].append(sess_res)

    # Pooled lag analysis
    print("\n--- RECURSIVE sessions (pooled lag-1) ---")
    rec_lag = {}
    for metric in metrics:
        x = np.array(pooled_rec[metric]["x"], dtype=float)
        y = np.array(pooled_rec[metric]["y_ord"], dtype=float)
        if len(x) < 10:
            continue
        r, p = stats.spearmanr(x, y)
        # Also point-biserial with binary
        y_bin = np.array(pooled_rec[metric]["y_bin"], dtype=float)
        r_pb, p_pb = stats.pointbiserialr(y_bin, x)
        rec_lag[metric] = {"r": float(r), "p": float(p), "r_pb": float(r_pb), "p_pb": float(p_pb), "n": len(x)}
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {metric:15s}: lag1 r={r:+.3f} p={p:.6f}{sig:4s} (n={len(x)})")

    print("\n--- BASELINE sessions (pooled lag-1) ---")
    bas_lag = {}
    for metric in metrics:
        x = np.array(pooled_bas[metric]["x"], dtype=float)
        y = np.array(pooled_bas[metric]["y_ord"], dtype=float)
        if len(x) < 10:
            continue
        r, p = stats.spearmanr(x, y)
        bas_lag[metric] = {"r": float(r), "p": float(p), "n": len(x)}
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {metric:15s}: lag1 r={r:+.3f} p={p:.6f}{sig:4s} (n={len(x)})")

    results["pooled"] = {"recursive_lag1": rec_lag, "baseline_lag1": bas_lag}
    return results


# ============================================================
# ANALYSIS 2: STATE TRANSITIONS
# ============================================================
def state_transition_analysis(sessions):
    """What spectral signatures predict quality transitions?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: STATE TRANSITIONS")
    print("=" * 70)

    results = {}

    # Focus on recursive sessions
    all_transitions = []
    for sess in sessions:
        if sess.get("mode") != "recursive":
            continue
        turns = extract_turns(sess)
        for i in range(len(turns) - 1):
            t_now = turns[i]
            t_next = turns[i + 1]
            transition = f"{t_now['classification']}->{t_next['classification']}"
            quality_change = t_next["class_ord"] - t_now["class_ord"]
            all_transitions.append({
                "transition": transition,
                "quality_change": quality_change,
                "improving": quality_change > 0,
                "degrading": quality_change < 0,
                "stable": quality_change == 0,
                "rv_before": t_now.get("output_rv"),
                "eff_rank_before": t_now.get("eff_rank"),
                "spectral_gap_before": t_now.get("spectral_gap"),
                "attn_entropy_before": t_now.get("attn_entropy"),
            })

    # Transition frequency matrix
    trans_counts = Counter(t["transition"] for t in all_transitions)
    print(f"\nTransition counts (recursive, n={len(all_transitions)}):")
    for trans, count in sorted(trans_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {trans:30s}: {count}")

    # Key question: what R_V precedes IMPROVING vs DEGRADING transitions?
    improving = [t for t in all_transitions if t["improving"] and t["rv_before"] is not None]
    degrading = [t for t in all_transitions if t["degrading"] and t["rv_before"] is not None]
    stable = [t for t in all_transitions if t["stable"] and t["rv_before"] is not None]

    print(f"\n  Improving: {len(improving)}, Degrading: {len(degrading)}, Stable: {len(stable)}")

    metrics = ["rv_before", "eff_rank_before", "spectral_gap_before", "attn_entropy_before"]
    transition_metrics = {}

    for metric in metrics:
        imp_vals = np.array([t[metric] for t in improving if t[metric] is not None])
        deg_vals = np.array([t[metric] for t in degrading if t[metric] is not None])
        sta_vals = np.array([t[metric] for t in stable if t[metric] is not None])

        if len(imp_vals) < 3 or len(deg_vals) < 3:
            continue

        u, p = stats.mannwhitneyu(imp_vals, deg_vals, alternative="two-sided")
        pooled_sd = np.sqrt((np.var(imp_vals, ddof=1) * (len(imp_vals)-1) + np.var(deg_vals, ddof=1) * (len(deg_vals)-1)) / (len(imp_vals) + len(deg_vals) - 2))
        d = (np.mean(imp_vals) - np.mean(deg_vals)) / pooled_sd if pooled_sd > 0 else 0

        transition_metrics[metric] = {
            "improving_mean": float(np.mean(imp_vals)),
            "degrading_mean": float(np.mean(deg_vals)),
            "stable_mean": float(np.mean(sta_vals)),
            "cohens_d": float(d),
            "mannwhitney_p": float(p),
            "n_imp": len(imp_vals),
            "n_deg": len(deg_vals),
        }

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        clean_name = metric.replace("_before", "")
        print(f"  {clean_name:15s}: Improving={np.mean(imp_vals):.4f} vs Degrading={np.mean(deg_vals):.4f} | d={d:+.3f} p={p:.4f}{sig}")

    # R_V quartile → transition probability
    all_rv = [t["rv_before"] for t in all_transitions if t["rv_before"] is not None]
    if all_rv:
        q25, q50, q75 = np.percentile(all_rv, [25, 50, 75])
        quartile_probs = {}
        for label, lo, hi in [("Q1_lowest", 0, q25), ("Q2", q25, q50), ("Q3", q50, q75), ("Q4_highest", q75, 999)]:
            in_q = [t for t in all_transitions if t["rv_before"] is not None and lo <= t["rv_before"] < hi]
            if not in_q:
                continue
            n_imp = sum(1 for t in in_q if t["improving"])
            n_deg = sum(1 for t in in_q if t["degrading"])
            n_sta = sum(1 for t in in_q if t["stable"])
            n = len(in_q)
            quartile_probs[label] = {
                "n": n, "p_improve": n_imp/n, "p_degrade": n_deg/n, "p_stable": n_sta/n,
                "rv_range": [float(lo), float(hi)]
            }
            print(f"  R_V {label}: P(improve)={n_imp/n:.2f}, P(degrade)={n_deg/n:.2f}, P(stable)={n_sta/n:.2f} (n={n})")

        results["quartile_transition_probs"] = quartile_probs

    results["transition_counts"] = dict(trans_counts)
    results["transition_metrics"] = transition_metrics
    return results


# ============================================================
# ANALYSIS 3: LOGISTIC REGRESSION + AUC
# ============================================================
def logistic_regression_analysis(sessions):
    """Multi-metric BT+ART predictor with cross-validated AUC."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: LOGISTIC REGRESSION (Multi-metric → BT+ART)")
    print("=" * 70)

    results = {}
    features = ["output_rv", "eff_rank", "top1_ratio", "spectral_gap", "attn_entropy"]

    for label, mode_filter in [("recursive_only", "recursive"), ("baseline_only", "baseline"), ("all_sessions", None)]:
        X_rows = []
        y_rows = []

        for sess in sessions:
            if mode_filter and sess.get("mode") != mode_filter:
                continue
            turns = extract_turns(sess)
            for t in turns:
                row = [t.get(f) for f in features]
                if any(v is None for v in row):
                    continue
                X_rows.append(row)
                y_rows.append(t["class_bin"])

        X = np.array(X_rows, dtype=float)
        y = np.array(y_rows, dtype=int)

        # Drop rows with NaN
        valid = np.all(np.isfinite(X), axis=1)
        X = X[valid]
        y = y[valid]

        if len(X) < 20 or y.sum() < 5 or (len(y) - y.sum()) < 5:
            print(f"\n--- {label}: Insufficient data (n={len(X)}, pos={y.sum()}) ---")
            continue

        print(f"\n--- {label} (n={len(X)}, BT+ART={y.sum()}, Other={len(y)-y.sum()}) ---")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Cross-validated AUC
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        lr = LogisticRegression(random_state=42, max_iter=1000)

        auc_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring="roc_auc")
        acc_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring="accuracy")

        print(f"  5-fold CV AUC: {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
        print(f"  5-fold CV Acc: {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")

        # Fit on full data for coefficients
        lr_full = LogisticRegression(random_state=42, max_iter=1000)
        lr_full.fit(X_scaled, y)
        coefs = dict(zip(features, lr_full.coef_[0].tolist()))
        print(f"  Coefficients (standardized):")
        for f, c in sorted(coefs.items(), key=lambda x: -abs(x[1])):
            print(f"    {f:15s}: {c:+.3f}")

        # Full-data AUC
        y_prob = lr_full.predict_proba(X_scaled)[:, 1]
        full_auc = roc_auc_score(y, y_prob)

        # Single-metric AUCs for comparison
        single_aucs = {}
        for i, feat in enumerate(features):
            try:
                lr_single = LogisticRegression(random_state=42, max_iter=1000)
                X_single = X_scaled[:, i:i+1]
                auc_single = cross_val_score(lr_single, X_single, y, cv=cv, scoring="roc_auc")
                single_aucs[feat] = float(np.mean(auc_single))
                print(f"    {feat:15s} alone: AUC={np.mean(auc_single):.3f}")
            except:
                pass

        results[label] = {
            "n": len(X),
            "n_positive": int(y.sum()),
            "cv_auc_mean": float(np.mean(auc_scores)),
            "cv_auc_std": float(np.std(auc_scores)),
            "cv_acc_mean": float(np.mean(acc_scores)),
            "full_auc": float(full_auc),
            "coefficients": coefs,
            "single_metric_aucs": single_aucs,
        }

    return results


# ============================================================
# ANALYSIS 4: C2 R_V → RECURSION_SCORE VALIDATION
# ============================================================
def c2_validation():
    """Validate R_V→behavior correlation on C2 circuit data."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: C2 R_V → RECURSION_SCORE VALIDATION")
    print("=" * 70)

    results = {}
    all_rv = []
    all_rec = []
    all_configs = []

    # Load all C2 CSVs
    c2_files = sorted(C2_DIR.glob("*/c2_rv_measurement.csv"))
    # Also add kitchen sink
    ks_csv = C2_KITCHEN / "c2_rv_measurement.csv"
    if ks_csv.exists():
        c2_files.append(ks_csv)

    for csv_path in c2_files:
        run_name = csv_path.parent.name
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        configs = Counter(r["config"] for r in rows)
        print(f"\n  {run_name}: {len(rows)} rows, configs={dict(configs)}")

        # Per-config correlations
        run_results = {"n_rows": len(rows), "configs": dict(configs), "per_config": {}}

        by_config = defaultdict(list)
        for r in rows:
            by_config[r["config"]].append(r)

        for config, config_rows in sorted(by_config.items()):
            rvs = []
            recs = []
            for r in config_rows:
                try:
                    rv = float(r["rv_mean"])
                    rec = float(r["recursion_score"])
                    rvs.append(rv)
                    recs.append(rec)
                    all_rv.append(rv)
                    all_rec.append(rec)
                    all_configs.append(config)
                except (ValueError, KeyError):
                    continue

            if len(rvs) < 5:
                continue

            r_sp, p_sp = stats.spearmanr(rvs, recs)
            run_results["per_config"][config] = {
                "n": len(rvs),
                "rv_mean": float(np.mean(rvs)),
                "rec_mean": float(np.mean(recs)),
                "rec_nonzero_pct": float(np.mean([1 for x in recs if x > 0]) / len(recs) * 100),
                "spearman_r": float(r_sp),
                "spearman_p": float(p_sp),
            }
            sig = "***" if p_sp < 0.001 else "**" if p_sp < 0.01 else "*" if p_sp < 0.05 else ""
            print(f"    {config:20s}: n={len(rvs):3d}, R_V={np.mean(rvs):.3f}, rec={np.mean(recs):.3f} (nonzero={np.mean([1 for x in recs if x > 0]) / len(recs) * 100:.0f}%), r={r_sp:+.3f} p={p_sp:.4f}{sig}")

        results[run_name] = run_results

    # Overall correlation
    if len(all_rv) > 10:
        r_all, p_all = stats.spearmanr(all_rv, all_rec)
        print(f"\n  OVERALL: n={len(all_rv)}, rho={r_all:+.3f}, p={p_all:.2e}")
        results["overall"] = {
            "n": len(all_rv),
            "spearman_r": float(r_all),
            "spearman_p": float(p_all),
        }

        # By config across all runs
        print("\n  BY CONFIG (pooled across all runs):")
        by_config_all = defaultdict(lambda: {"rv": [], "rec": []})
        for rv, rec, config in zip(all_rv, all_rec, all_configs):
            by_config_all[config]["rv"].append(rv)
            by_config_all[config]["rec"].append(rec)

        config_pooled = {}
        for config in sorted(by_config_all.keys()):
            rvs = np.array(by_config_all[config]["rv"])
            recs = np.array(by_config_all[config]["rec"])
            r, p = stats.spearmanr(rvs, recs)
            config_pooled[config] = {
                "n": len(rvs), "rv_mean": float(np.mean(rvs)),
                "rec_mean": float(np.mean(recs)),
                "spearman_r": float(r), "spearman_p": float(p),
            }
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {config:20s}: n={len(rvs):4d}, R_V={np.mean(rvs):.3f}, rec={np.mean(recs):.3f}, r={r:+.3f} p={p:.4f}{sig}")

        results["config_pooled"] = config_pooled

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("COMPREHENSIVE BEHAVIORAL BRIDGE BATTERY")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    sessions = load_v3_sessions()
    print(f"Loaded {len(sessions)} v3 sessions")

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "description": "Comprehensive bridge battery: temporal lag, transitions, logistic regression, C2 validation",
        "model": MODEL_NAME,
        "source_v3_dir": str(V3_DIR),
        "source_c2_dir": str(C2_DIR),
        "source_c2_kitchen": str(C2_KITCHEN),
        "n_source_sessions": len(sessions),
    }

    # Run all four analyses
    all_results["temporal_lag"] = temporal_lag_analysis(sessions)
    all_results["state_transitions"] = state_transition_analysis(sessions)
    all_results["logistic_regression"] = logistic_regression_analysis(sessions)
    all_results["c2_validation"] = c2_validation()

    # Save
    outfile = OUTPUT_DIR / f"bridge_battery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*70}")
    print(f"All results saved to {outfile}")


if __name__ == "__main__":
    main()
