#!/usr/bin/env python3
"""Held-out classifier evaluation for behavioral bridge data."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

V3_DIR = Path("results/sustained_gnani_v3_fixed")
OUT_DIR = Path("results/classifier_evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

FEATURES = [
    "output_rv",
    "eff_rank",
    "top1_ratio",
    "spectral_gap",
    "cosine",
    "attn_entropy",
    "perplexity",
    "rs_rv",
]

CLASS_BIN = {
    "REPETITIVE": 0,
    "SURFACE": 0,
    "CONCEPTUAL": 0,
    "ARTICULATE": 1,
    "BREAKTHROUGH": 1,
}


def _safe_float(x) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def load_turn_matrix(mode_filter: str | None = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rows: List[List[float]] = []
    labels: List[int] = []
    sessions: List[str] = []

    for fp in sorted(V3_DIR.glob("*.json")):
        if fp.name == "comparison_summary.json":
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        mode = data.get("mode")
        if mode_filter and mode != mode_filter:
            continue
        sid = data.get("session_id", fp.stem)

        for t in data.get("turns", []):
            cls = t.get("classification")
            if cls not in CLASS_BIN:
                continue
            om = t.get("output_metrics") or {}
            feat = [
                _safe_float(t.get("output_rv")),
                _safe_float(om.get("eff_rank")),
                _safe_float(om.get("top1_ratio")),
                _safe_float(om.get("spectral_gap")),
                _safe_float(om.get("cosine")),
                _safe_float(om.get("attn_entropy")),
                _safe_float(om.get("perplexity")),
                _safe_float(om.get("rs_rv")),
            ]
            if not np.all(np.isfinite(np.array(feat, dtype=float))):
                continue
            rows.append(feat)
            labels.append(int(CLASS_BIN[cls]))
            sessions.append(str(sid))

    if not rows:
        return np.zeros((0, len(FEATURES)), dtype=float), np.zeros((0,), dtype=int), []
    return np.array(rows, dtype=float), np.array(labels, dtype=int), sessions


def bootstrap_auc_ci(y_true: np.ndarray, y_prob: np.ndarray, n_bootstrap: int, seed: int) -> Dict[str, float | int | None]:
    rng = np.random.default_rng(seed)
    aucs: List[float] = []
    n = len(y_true)
    if n < 3:
        return {"n_valid": 0, "ci95_low": None, "ci95_high": None}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        pb = y_prob[idx]
        aucs.append(float(roc_auc_score(yb, pb)))

    if not aucs:
        return {"n_valid": 0, "ci95_low": None, "ci95_high": None}
    return {
        "n_valid": int(len(aucs)),
        "ci95_low": float(np.percentile(aucs, 2.5)),
        "ci95_high": float(np.percentile(aucs, 97.5)),
    }


def evaluate_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.30,
    random_state: int = 42,
    n_bootstrap: int = 1000,
) -> Dict[str, object]:
    if len(x) < 30 or len(np.unique(y)) < 2:
        return {"status": "insufficient_data", "n": int(len(x))}

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    clf = LogisticRegression(max_iter=4000, random_state=random_state)
    clf.fit(x_train_s, y_train)
    train_prob = clf.predict_proba(x_train_s)[:, 1]
    test_prob = clf.predict_proba(x_test_s)[:, 1]

    train_auc = float(roc_auc_score(y_train, train_prob))
    test_auc = float(roc_auc_score(y_test, test_prob))
    test_pred = (test_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()

    # R_V-alone baseline on the same split.
    rv_idx = FEATURES.index("output_rv")
    rv_train = x_train_s[:, [rv_idx]]
    rv_test = x_test_s[:, [rv_idx]]
    rv_clf = LogisticRegression(max_iter=4000, random_state=random_state)
    rv_clf.fit(rv_train, y_train)
    rv_test_prob = rv_clf.predict_proba(rv_test)[:, 1]
    rv_auc = float(roc_auc_score(y_test, rv_test_prob))

    coefs = {k: float(v) for k, v in zip(FEATURES, clf.coef_[0].tolist())}
    top3 = [
        {"feature": k, "coef": coefs[k], "abs_coef": float(abs(coefs[k]))}
        for k in sorted(coefs.keys(), key=lambda kk: abs(coefs[kk]), reverse=True)[:3]
    ]

    ci = bootstrap_auc_ci(y_test, test_prob, n_bootstrap=n_bootstrap, seed=random_state + 17)

    return {
        "status": "ok",
        "train_n": int(len(y_train)),
        "test_n": int(len(y_test)),
        "train_pos": int(y_train.sum()),
        "test_pos": int(y_test.sum()),
        "train_auc": train_auc,
        "test_auc": test_auc,
        "rv_alone_test_auc": rv_auc,
        "confusion_matrix_test": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "coefficients_standardized": coefs,
        "top_3_features": top3,
        "bootstrap_test_auc_ci95": [ci["ci95_low"], ci["ci95_high"]],
        "bootstrap_valid_samples": int(ci["n_valid"]),
    }


def main() -> int:
    x_all, y_all, _ = load_turn_matrix(mode_filter=None)
    x_bas, y_bas, _ = load_turn_matrix(mode_filter="baseline")

    overall = evaluate_split(x_all, y_all, test_size=0.30, random_state=42, n_bootstrap=1000)
    baseline_only = evaluate_split(x_bas, y_bas, test_size=0.30, random_state=42, n_bootstrap=1000)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "source_results_dir": str(V3_DIR),
        "source_glob": "*.json",
        "feature_order": FEATURES,
        "all_sessions": overall,
        "baseline_only": baseline_only,
        "notes": "70/30 stratified train/test; AUC reported on held-out test split.",
    }

    out_file = OUT_DIR / f"classifier_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Saved: {out_file}")
    if overall.get("status") == "ok":
        print(
            f"All sessions: train_auc={overall['train_auc']:.4f}, "
            f"test_auc={overall['test_auc']:.4f}, rv_only_test_auc={overall['rv_alone_test_auc']:.4f}"
        )
        print(
            "Top3: "
            + ", ".join(f"{x['feature']}({x['coef']:+.3f})" for x in overall["top_3_features"])
        )
    if baseline_only.get("status") == "ok":
        print(f"Baseline-only test_auc={baseline_only['test_auc']:.4f}")
    else:
        print(f"Baseline-only: {baseline_only.get('status')} (n={baseline_only.get('n')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
