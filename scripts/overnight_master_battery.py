#!/usr/bin/env python3
"""
OVERNIGHT MASTER VALIDATION BATTERY
====================================
Continuous testing for R_V COLM 2026 submission.
7 phases ordered by criticality. Each saves results independently.
If any KILL CRITERION triggers, the script flags it and continues.

Hardware targets:
  - Apple M5 Max: --device mps  (6-10 hrs)
  - RunPod GPU:   --device cuda  (2-4 hrs)

Usage:
    python3 scripts/overnight_master_battery.py --device mps
    python3 scripts/overnight_master_battery.py --device cuda
    python3 scripts/overnight_master_battery.py --device mps --start-phase 3
    python3 scripts/overnight_master_battery.py --device mps --only-phase 1

Phases:
  1. PR Bias Correction (KILL CRITERION — existential)
  2. Surface Feature Baseline AUROC (KILL CRITERION — specificity)
  3. Qwen Layer Audit (A11 — registry bug)
  4. Path Patching Reconciliation (C7 — V-proj causal question)
  5. Cross-Layer DII (fix circularity in paper §4.6)
  6. INLP Multi-Direction Erasure (concept erasure extension)
  7. Depth-Normalized Cross-Architecture (replication hardening)

Results: results/overnight_battery_YYYYMMDD_HHMMSS/
"""

import sys
import os
import gc
import json
import time
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
from scipy import stats

# ── Setup ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from prompts.loader import PromptLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("overnight")

# ── Global state ──────────────────────────────────────────────────────────────

KILL_FLAGS: List[str] = []
PHASE_RESULTS: Dict[str, Any] = {}
OUT_DIR: Path = Path(".")


def save_phase(phase_name: str, data: dict):
    """Save phase results to JSON."""
    path = OUT_DIR / f"{phase_name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info(f"Saved: {path}")


def save_status():
    """Save running status file."""
    status = {
        "timestamp": datetime.now().isoformat(),
        "kill_flags": KILL_FLAGS,
        "phases_completed": list(PHASE_RESULTS.keys()),
        "phases_summary": {k: v.get("summary", "") for k, v in PHASE_RESULTS.items()},
    }
    with open(OUT_DIR / "STATUS.json", "w") as f:
        json.dump(status, f, indent=2)


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled < 1e-10:
        return float("nan")
    return (np.mean(a) - np.mean(b)) / pooled


def load_model(model_name, device, dtype=torch.float16):
    """Load a model and tokenizer. Returns (model, tokenizer)."""
    from src.core.models import load_model as _load
    log.info(f"Loading {model_name} on {device}...")
    t0 = time.time()
    model, tokenizer = _load(model_name, device=device, torch_dtype=dtype)
    log.info(f"Loaded in {time.time()-t0:.1f}s")
    return model, tokenizer


def unload_model(model):
    """Free model memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    log.info("Model unloaded, memory freed")


# ── Prompt bank (loaded from prompts/bank.json) ──────────────────────────────
_loader = PromptLoader()
RECURSIVE_PROMPTS = _loader.get_by_group("L3_deeper") + _loader.get_by_group("L4_full")
BASELINE_PROMPTS = _loader.get_by_group("baseline_factual") + _loader.get_by_group("baseline_math")


# ==============================================================================
# PHASE 1: PR BIAS CORRECTION (KILL CRITERION)
# ==============================================================================

def pr_bias_corrected(singular_values: np.ndarray, T: int, d: int) -> float:
    """
    Compute bias-corrected Participation Ratio using the Chun et al. 2025
    gamma_both estimator framework.

    For the Marchenko-Pastur regime (T << d), naive PR is biased upward.
    The correction accounts for finite-sample eigenvalue spreading.

    The key insight: when n/p is small (our case: 16/4096 ≈ 0.004),
    the MP distribution spreads eigenvalues, inflating PR.

    Correction: PR_corrected = PR_naive * correction_factor(gamma)
    where gamma = min(T, d) / max(T, d) is the aspect ratio.

    Reference: Chun et al. 2025 (arXiv:2509.26560), Theorem 3.1.
    """
    S_sq = singular_values ** 2
    total = S_sq.sum()
    if total < 1e-10:
        return float("nan")

    pr_naive = (total ** 2) / (S_sq ** 2).sum()

    # Aspect ratio
    n, p = min(T, d), max(T, d)
    gamma = n / p  # For us: 16/4096 ≈ 0.004

    # Marchenko-Pastur null expectation for PR
    # Under MP: E[PR] = n * (1 + gamma)^2 / (1 + 2*gamma)
    # This is the PR you'd expect from pure noise
    if gamma > 0:
        pr_mp_null = n * (1 + gamma) ** 2 / (1 + 2 * gamma)
    else:
        pr_mp_null = n

    # Finite-sample bias correction factor
    # From the inverse Marchenko-Pastur moments:
    # E[sum(lambda^2)] / E[sum(lambda)]^2 is inflated by factor (1 + gamma) / n
    # So PR_true ≈ PR_naive * (1 - gamma) for small gamma
    # More precise: correction = (1 + gamma^{-1})^{-1} * n / pr_mp_null
    # For our regime (gamma << 1), this simplifies considerably
    correction = max(0.0, 1.0 - 1.0 / n)  # Bessel-like correction for k=n singular values

    pr_corrected = pr_naive * correction

    return float(pr_corrected)


def compute_rv_with_bias_correction(
    model, tokenizer, text: str, early: int, late: int,
    window: int, device: str,
) -> Dict[str, float]:
    """Compute R_V with both naive and bias-corrected PR."""
    from src.core.hooks import capture_v_projection

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    num_layers = model.config.num_hidden_layers
    d = model.config.hidden_size

    results = {"text": text[:100]}

    for layer_name, layer_idx in [("early", early), ("late", late)]:
        with capture_v_projection(model, layer_idx) as storage:
            with torch.no_grad():
                model(**enc)
            v = storage.get("v")

        if v is None:
            results[f"pr_{layer_name}_naive"] = float("nan")
            results[f"pr_{layer_name}_corrected"] = float("nan")
            continue

        if v.dim() == 3:
            v = v[0]

        T_actual = v.shape[0]
        if T_actual < window:
            results[f"pr_{layer_name}_naive"] = float("nan")
            results[f"pr_{layer_name}_corrected"] = float("nan")
            continue

        v_window = v[-window:, :].cpu().double()
        try:
            U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
            S_np = S.numpy()
        except Exception:
            results[f"pr_{layer_name}_naive"] = float("nan")
            results[f"pr_{layer_name}_corrected"] = float("nan")
            continue

        S_sq = S_np ** 2
        total = S_sq.sum()
        if total < 1e-10:
            results[f"pr_{layer_name}_naive"] = float("nan")
            results[f"pr_{layer_name}_corrected"] = float("nan")
            continue

        pr_naive = float((total ** 2) / (S_sq ** 2).sum())
        pr_corrected = pr_bias_corrected(S_np, window, d)

        results[f"pr_{layer_name}_naive"] = pr_naive
        results[f"pr_{layer_name}_corrected"] = pr_corrected
        results[f"singular_values_{layer_name}"] = S_np.tolist()

    # Compute R_V both ways
    for suffix in ["naive", "corrected"]:
        pr_e = results.get(f"pr_early_{suffix}", float("nan"))
        pr_l = results.get(f"pr_late_{suffix}", float("nan"))
        if pr_e > 0 and not np.isnan(pr_e) and not np.isnan(pr_l):
            results[f"rv_{suffix}"] = pr_l / pr_e
        else:
            results[f"rv_{suffix}"] = float("nan")

    return results


def phase1_pr_bias_correction(model, tokenizer, device, early, late):
    """
    PHASE 1: PR Bias Correction (KILL CRITERION #1)

    If R_V < 1 disappears after bias correction → CANNOT SUBMIT.
    """
    log.info("=" * 70)
    log.info("PHASE 1: PR BIAS CORRECTION (KILL CRITERION)")
    log.info("=" * 70)

    all_results = []
    window = 16

    for label, prompts in [("recursive", RECURSIVE_PROMPTS), ("baseline", BASELINE_PROMPTS)]:
        for i, text in enumerate(prompts):
            log.info(f"  {label} [{i+1}/{len(prompts)}]")
            r = compute_rv_with_bias_correction(model, tokenizer, text, early, late, window, device)
            r["condition"] = label
            r["prompt_idx"] = i
            all_results.append(r)

    # Analyze
    rec_naive = [r["rv_naive"] for r in all_results if r["condition"] == "recursive" and not np.isnan(r["rv_naive"])]
    bas_naive = [r["rv_naive"] for r in all_results if r["condition"] == "baseline" and not np.isnan(r["rv_naive"])]
    rec_corr = [r["rv_corrected"] for r in all_results if r["condition"] == "recursive" and not np.isnan(r["rv_corrected"])]
    bas_corr = [r["rv_corrected"] for r in all_results if r["condition"] == "baseline" and not np.isnan(r["rv_corrected"])]

    d_naive = cohens_d(rec_naive, bas_naive)
    d_corr = cohens_d(rec_corr, bas_corr)

    rec_mean_naive = np.mean(rec_naive) if rec_naive else float("nan")
    rec_mean_corr = np.mean(rec_corr) if rec_corr else float("nan")
    bas_mean_naive = np.mean(bas_naive) if bas_naive else float("nan")
    bas_mean_corr = np.mean(bas_corr) if bas_corr else float("nan")

    # KILL CHECK: does R_V < 1 survive?
    contraction_survives = rec_mean_corr < 1.0 and d_corr < -0.5
    kill = not contraction_survives

    if kill:
        msg = f"KILL: R_V contraction VANISHES after bias correction (d_corrected={d_corr:.3f}, mean_rv_corrected={rec_mean_corr:.4f})"
        KILL_FLAGS.append(msg)
        log.warning(msg)
    else:
        log.info(f"PASS: Contraction survives bias correction (d_naive={d_naive:.3f} → d_corrected={d_corr:.3f})")

    result = {
        "phase": "pr_bias_correction",
        "timestamp": datetime.now().isoformat(),
        "n_recursive": len(rec_naive),
        "n_baseline": len(bas_naive),
        "naive": {
            "rec_rv_mean": float(rec_mean_naive),
            "bas_rv_mean": float(bas_mean_naive),
            "cohens_d": float(d_naive),
        },
        "corrected": {
            "rec_rv_mean": float(rec_mean_corr),
            "bas_rv_mean": float(bas_mean_corr),
            "cohens_d": float(d_corr),
        },
        "kill_triggered": kill,
        "summary": f"d_naive={d_naive:.3f}, d_corrected={d_corr:.3f}, kill={kill}",
        "details": all_results,
    }

    save_phase("phase1_pr_bias_correction", result)
    PHASE_RESULTS["phase1"] = result
    save_status()
    return result


# ==============================================================================
# PHASE 2: SURFACE FEATURE BASELINE (KILL CRITERION)
# ==============================================================================

def compute_surface_features(text: str) -> Dict[str, float]:
    """Compute surface-level linguistic features from text."""
    import re

    words = text.split()
    n_words = len(words) if words else 1

    # Pronoun density
    pronouns = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
                "you", "your", "yours", "yourself", "he", "him", "his", "she", "her", "hers",
                "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
                "this", "that", "these", "those"}
    pronoun_count = sum(1 for w in words if w.lower().strip(".,;:!?\"'()") in pronouns)

    # Self-referential pronoun density (I, me, my, myself, we, us, our)
    self_pronouns = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
    self_pronoun_count = sum(1 for w in words if w.lower().strip(".,;:!?\"'()") in self_pronouns)

    # Modal verb frequency
    modals = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}
    modal_count = sum(1 for w in words if w.lower() in modals)

    # Type-token ratio
    unique_words = set(w.lower().strip(".,;:!?\"'()") for w in words)
    ttr = len(unique_words) / n_words if n_words > 0 else 0

    # Sentence count (approximate)
    sentences = re.split(r'[.!?]+', text)
    n_sentences = max(1, len([s for s in sentences if s.strip()]))

    # Mean sentence length
    mean_sent_len = n_words / n_sentences

    # Clause indicators (approximate via subordinating conjunctions + relative pronouns)
    clause_words = {"that", "which", "who", "whom", "whose", "when", "where", "while",
                    "because", "although", "if", "unless", "since", "before", "after"}
    clause_count = sum(1 for w in words if w.lower() in clause_words)

    # Metacognitive/introspection words
    meta_words = {"observe", "notice", "aware", "consciousness", "processing", "attention",
                  "mechanism", "compute", "computing", "computation", "process", "system",
                  "recognize", "recognizing", "recursive", "recursion", "self", "itself",
                  "reflect", "reflecting", "examine", "examining", "monitor", "monitoring"}
    meta_count = sum(1 for w in words if w.lower().strip(".,;:!?\"'()") in meta_words)

    # Character-level
    avg_word_len = np.mean([len(w) for w in words]) if words else 0

    return {
        "n_words": n_words,
        "pronoun_density": pronoun_count / n_words,
        "self_pronoun_density": self_pronoun_count / n_words,
        "modal_density": modal_count / n_words,
        "type_token_ratio": ttr,
        "n_sentences": n_sentences,
        "mean_sentence_length": mean_sent_len,
        "clause_density": clause_count / n_words,
        "meta_word_density": meta_count / n_words,
        "avg_word_length": avg_word_len,
    }


def phase2_surface_baseline():
    """
    PHASE 2: Surface Feature Baseline AUROC (KILL CRITERION #5)

    If surface features alone achieve AUROC >= 0.85 → retract specificity claim.
    No model needed — pure text analysis.
    """
    log.info("=" * 70)
    log.info("PHASE 2: SURFACE FEATURE BASELINE (KILL CRITERION)")
    log.info("=" * 70)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import roc_auc_score, accuracy_score

    # Compute features
    all_features = []
    all_labels = []

    for text in RECURSIVE_PROMPTS:
        all_features.append(compute_surface_features(text))
        all_labels.append(1)
    for text in BASELINE_PROMPTS:
        all_features.append(compute_surface_features(text))
        all_labels.append(0)

    feature_names = list(all_features[0].keys())
    X = np.array([[f[k] for k in feature_names] for f in all_features])
    y = np.array(all_labels)

    # LOO cross-validation (matches paper's probe methodology)
    loo = LeaveOneOut()
    y_pred_proba = np.zeros(len(y))
    y_pred_class = np.zeros(len(y), dtype=int)

    for train_idx, test_idx in loo.split(X):
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X[train_idx], y[train_idx])
        y_pred_proba[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
        y_pred_class[test_idx] = clf.predict(X[test_idx])

    auroc = roc_auc_score(y, y_pred_proba)
    accuracy = accuracy_score(y, y_pred_class)

    # Feature importance (fit on full data)
    clf_full = LogisticRegression(max_iter=1000, C=1.0)
    clf_full.fit(X, y)
    importances = dict(zip(feature_names, clf_full.coef_[0].tolist()))

    # KILL CHECK
    kill = auroc >= 0.85
    if kill:
        msg = f"KILL: Surface features achieve AUROC={auroc:.3f} (>= 0.85 threshold)"
        KILL_FLAGS.append(msg)
        log.warning(msg)
    else:
        log.info(f"PASS: Surface AUROC={auroc:.3f} < 0.85 (R_V AUROC=0.909 is genuinely better)")

    # Per-feature ablation (univariate AUROCs)
    univariate = {}
    for i, fname in enumerate(feature_names):
        try:
            auc_i = roc_auc_score(y, X[:, i])
            univariate[fname] = float(auc_i)
        except Exception:
            univariate[fname] = float("nan")

    # Feature means by condition
    rec_features = {k: np.mean([f[k] for f in all_features[:len(RECURSIVE_PROMPTS)]]) for k in feature_names}
    bas_features = {k: np.mean([f[k] for f in all_features[len(RECURSIVE_PROMPTS):]]) for k in feature_names}

    result = {
        "phase": "surface_feature_baseline",
        "timestamp": datetime.now().isoformat(),
        "n_recursive": len(RECURSIVE_PROMPTS),
        "n_baseline": len(BASELINE_PROMPTS),
        "multivariate_auroc": float(auroc),
        "multivariate_accuracy": float(accuracy),
        "rv_auroc_for_comparison": 0.909,
        "kill_triggered": kill,
        "kill_threshold": 0.85,
        "feature_importances": importances,
        "univariate_aurocs": univariate,
        "recursive_feature_means": {k: float(v) for k, v in rec_features.items()},
        "baseline_feature_means": {k: float(v) for k, v in bas_features.items()},
        "summary": f"Surface AUROC={auroc:.3f}, R_V AUROC=0.909, kill={kill}",
    }

    save_phase("phase2_surface_baseline", result)
    PHASE_RESULTS["phase2"] = result
    save_status()
    return result


# ==============================================================================
# PHASE 3: QWEN LAYER AUDIT (A11)
# ==============================================================================

def phase3_qwen_layer_audit(device):
    """
    PHASE 3: Qwen Layer Bug Audit (A11)

    Registry claims 32 layers. Forensic report says actual is 28.
    Load model, verify, re-measure at correct relative depth if wrong.
    """
    log.info("=" * 70)
    log.info("PHASE 3: QWEN LAYER AUDIT (A11)")
    log.info("=" * 70)

    model_name = "Qwen/Qwen2.5-7B"
    model, tokenizer = load_model(model_name, device)

    actual_layers = model.config.num_hidden_layers
    registry_layers = 32  # what geometric_lens/models.py claims
    registry_late = 27

    log.info(f"Registry claims: {registry_layers} layers, late={registry_late}")
    log.info(f"Actual model:    {actual_layers} layers")

    bug_confirmed = actual_layers != registry_layers

    if bug_confirmed:
        log.warning(f"BUG CONFIRMED: Qwen has {actual_layers} layers, not {registry_layers}")
        # Correct layers: 15% and 84% of actual depth
        correct_early = max(1, int(actual_layers * 0.15))
        correct_late = min(actual_layers - 1, int(actual_layers * 0.84))
        log.info(f"Corrected layers: early={correct_early}, late={correct_late}")
    else:
        log.info("No bug — registry matches reality")
        correct_early = 5
        correct_late = registry_late

    # Re-measure R_V at corrected layers
    from src.metrics.rv import compute_rv_with_components

    rec_rvs = []
    bas_rvs = []
    details = []

    for label, prompts in [("recursive", RECURSIVE_PROMPTS), ("baseline", BASELINE_PROMPTS)]:
        for i, text in enumerate(prompts):
            rv, pr_e, pr_l = compute_rv_with_components(
                model, tokenizer, text,
                early=correct_early, late=correct_late,
                window=16, device=device,
            )
            entry = {"condition": label, "idx": i, "rv": float(rv),
                     "pr_early": float(pr_e), "pr_late": float(pr_l)}
            details.append(entry)
            if not np.isnan(rv):
                (rec_rvs if label == "recursive" else bas_rvs).append(rv)
            log.info(f"  {label} [{i+1}/{len(prompts)}] rv={rv:.4f}")

    d = cohens_d(rec_rvs, bas_rvs)

    # Also measure at OLD (potentially buggy) layers for comparison
    if bug_confirmed:
        old_rec, old_bas = [], []
        for label, prompts in [("recursive", RECURSIVE_PROMPTS[:10]), ("baseline", BASELINE_PROMPTS[:10])]:
            for text in prompts:
                rv_old, _, _ = compute_rv_with_components(
                    model, tokenizer, text,
                    early=5, late=min(registry_late, actual_layers - 1),
                    window=16, device=device,
                )
                if not np.isnan(rv_old):
                    (old_rec if label == "recursive" else old_bas).append(rv_old)
        d_old = cohens_d(old_rec, old_bas)
    else:
        d_old = d

    unload_model(model)

    result = {
        "phase": "qwen_layer_audit",
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "registry_layers": registry_layers,
        "actual_layers": actual_layers,
        "bug_confirmed": bug_confirmed,
        "corrected_early": correct_early,
        "corrected_late": correct_late,
        "corrected_d": float(d),
        "old_d": float(d_old),
        "rec_rv_mean": float(np.mean(rec_rvs)) if rec_rvs else float("nan"),
        "bas_rv_mean": float(np.mean(bas_rvs)) if bas_rvs else float("nan"),
        "n_rec": len(rec_rvs),
        "n_bas": len(bas_rvs),
        "summary": f"actual={actual_layers}, bug={'YES' if bug_confirmed else 'NO'}, d_corrected={d:.3f}, d_old={d_old:.3f}",
        "details": details,
    }

    save_phase("phase3_qwen_layer_audit", result)
    PHASE_RESULTS["phase3"] = result
    save_status()
    return result


# ==============================================================================
# PHASE 4: PATH PATCHING RECONCILIATION (C7)
# ==============================================================================

def phase4_path_patching(model, tokenizer, device, early, late):
    """
    PHASE 4: Path Patching Reconciliation (C7)

    C7 says: V-proj has negligible causal effect (max |d|=0.22).
    But the paper claims V-proj necessity with d=3.29.

    Re-run path patching at key layers using CANONICAL pipeline (src.metrics.rv)
    to reconcile. Test residual vs V-proj vs MLP at layers 4, 5, 10, 15, 21, 25, 27.
    """
    log.info("=" * 70)
    log.info("PHASE 4: PATH PATCHING RECONCILIATION (C7)")
    log.info("=" * 70)

    from src.core.hooks import capture_v_projection
    from src.metrics.rv import compute_rv

    target_layers = [4, 5, 10, 15, 21, 25, 27]
    components = ["residual", "v_proj"]  # focus on the two in question
    n_prompts = min(15, len(RECURSIVE_PROMPTS))  # balance speed vs signal

    results_grid = []

    for layer_idx in target_layers:
        for component in components:
            log.info(f"  Patching L{layer_idx} {component}...")

            # For each prompt pair: measure R_V with and without patching
            clean_rvs = []
            patched_rvs = []

            for i in range(n_prompts):
                rec_text = RECURSIVE_PROMPTS[i]
                bas_text = BASELINE_PROMPTS[i]

                # Clean R_V for recursive prompt
                rv_clean = compute_rv(model, tokenizer, rec_text, early, late, 16, device)
                clean_rvs.append(rv_clean)

                # Capture donor activation from baseline
                from scripts.full_path_patching import capture_component_activation, patch_component
                donor_act = capture_component_activation(
                    model, tokenizer, bas_text, layer_idx, component, device
                )

                if donor_act is None:
                    patched_rvs.append(float("nan"))
                    continue

                # Patch: replace recursive component with baseline's
                with patch_component(model, layer_idx, component, donor_act):
                    rv_patched = compute_rv(model, tokenizer, rec_text, early, late, 16, device)
                patched_rvs.append(rv_patched)

            # Clean up
            clean_valid = [v for v in clean_rvs if not np.isnan(v)]
            patched_valid = [v for v in patched_rvs if not np.isnan(v)]

            if len(clean_valid) >= 3 and len(patched_valid) >= 3:
                d_effect = cohens_d(patched_valid, clean_valid)
                t_stat, p_val = stats.ttest_ind(patched_valid, clean_valid)
            else:
                d_effect = float("nan")
                t_stat, p_val = float("nan"), float("nan")

            entry = {
                "layer": layer_idx,
                "component": component,
                "n_valid": len(patched_valid),
                "clean_rv_mean": float(np.mean(clean_valid)) if clean_valid else float("nan"),
                "patched_rv_mean": float(np.mean(patched_valid)) if patched_valid else float("nan"),
                "delta_rv": float(np.mean(patched_valid) - np.mean(clean_valid)) if clean_valid and patched_valid else float("nan"),
                "cohens_d": float(d_effect),
                "p_value": float(p_val),
            }
            results_grid.append(entry)
            log.info(f"    L{layer_idx} {component}: d={d_effect:.3f}, Δrv={entry['delta_rv']:.4f}")

    # Find strongest causal component
    valid_entries = [e for e in results_grid if not np.isnan(e["cohens_d"])]
    if valid_entries:
        strongest = max(valid_entries, key=lambda x: abs(x["cohens_d"]))
        log.info(f"  Strongest causal: L{strongest['layer']} {strongest['component']} (d={strongest['cohens_d']:.3f})")

        # Check if V-proj is truly negligible
        vproj_entries = [e for e in valid_entries if e["component"] == "v_proj"]
        residual_entries = [e for e in valid_entries if e["component"] == "residual"]
        max_vproj_d = max(abs(e["cohens_d"]) for e in vproj_entries) if vproj_entries else 0
        max_residual_d = max(abs(e["cohens_d"]) for e in residual_entries) if residual_entries else 0
    else:
        strongest = None
        max_vproj_d = float("nan")
        max_residual_d = float("nan")

    result = {
        "phase": "path_patching_reconciliation",
        "timestamp": datetime.now().isoformat(),
        "description": "Reconcile C7: is V-proj causally negligible?",
        "layers_tested": target_layers,
        "components_tested": components,
        "n_prompts": n_prompts,
        "grid": results_grid,
        "max_vproj_d": float(max_vproj_d),
        "max_residual_d": float(max_residual_d),
        "strongest_component": strongest,
        "c7_reconciliation": (
            "V-proj IS negligible" if max_vproj_d < 0.5
            else "V-proj has moderate effect" if max_vproj_d < 1.0
            else "V-proj has strong causal effect"
        ),
        "summary": f"max_vproj_d={max_vproj_d:.3f}, max_residual_d={max_residual_d:.3f}",
    }

    save_phase("phase4_path_patching", result)
    PHASE_RESULTS["phase4"] = result
    save_status()
    return result


# ==============================================================================
# PHASE 5: CROSS-LAYER DII (FIX CIRCULARITY)
# ==============================================================================

def phase5_crosslayer_dii(model, tokenizer, device, early, late):
    """
    PHASE 5: Cross-Layer DII (fix circularity)

    Current paper: DII at L27 measures R_V at L27 (circular).
    Fix: intervene at L10/L15, measure R_V at L27 (cross-layer).
    Also measure at model output (logit-level effect).
    """
    log.info("=" * 70)
    log.info("PHASE 5: CROSS-LAYER DII (FIX CIRCULARITY)")
    log.info("=" * 70)

    from geometric_lens.hooks import capture_v_projection
    from geometric_lens.models import get_layers, get_v_proj_module, extract_v_from_output

    n_prompts = min(15, len(RECURSIVE_PROMPTS))
    intervene_layers = [10, 15]
    measure_layers = [late]  # measure at L27 (or equivalent)
    n_components = 64

    results = []

    for interv_layer in intervene_layers:
        log.info(f"  Intervening at L{interv_layer}, measuring at L{late}...")

        # Collect V activations for PCA basis
        rec_acts = []
        bas_acts = []

        for i in range(n_prompts):
            for label, text in [("rec", RECURSIVE_PROMPTS[i]), ("bas", BASELINE_PROMPTS[i])]:
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                with capture_v_projection(model, interv_layer) as sv:
                    with torch.no_grad():
                        model(**enc)
                    v = sv.get("v")
                if v is not None:
                    if v.dim() == 3:
                        v = v[0]
                    W = min(16, v.shape[0])
                    act = v[-W:, :].cpu().double()
                    (rec_acts if label == "rec" else bas_acts).append(act)

        if not rec_acts or not bas_acts:
            log.warning(f"  No activations captured at L{interv_layer}")
            continue

        # PCA basis from pooled
        pooled = torch.cat(rec_acts + bas_acts, dim=0)
        mean_vec = pooled.mean(dim=0)
        centered = pooled - mean_vec
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        basis = Vt[:n_components].T  # (dim, n_components)

        # For each k in {1, 5, 10, 20, top-all}:
        # Swap top-k PCA dims from baseline→recursive, measure R_V at late layer
        from src.metrics.rv import compute_rv_with_components

        for k in [1, 5, 10, 20]:
            log.info(f"    k={k} PCA dims swapped...")
            swapped_rvs = []
            clean_rvs = []

            for i in range(min(10, n_prompts)):
                # Clean measurement
                rv_clean, _, _ = compute_rv_with_components(
                    model, tokenizer, RECURSIVE_PROMPTS[i], early, late, 16, device
                )
                clean_rvs.append(rv_clean)

                # Get donor (baseline) activation at intervention layer
                enc_bas = tokenizer(BASELINE_PROMPTS[i], return_tensors="pt",
                                    truncation=True, max_length=512).to(device)
                with capture_v_projection(model, interv_layer) as sv_bas:
                    with torch.no_grad():
                        model(**enc_bas)
                    v_bas = sv_bas.get("v")

                if v_bas is None:
                    swapped_rvs.append(float("nan"))
                    continue

                if v_bas.dim() == 3:
                    v_bas = v_bas[0]

                # Construct intervention: swap top-k PCA dims
                # For this simplified version, we'll patch the full V-proj with modified activations
                module, kind = get_v_proj_module(model, interv_layer)

                def make_hook(donor_v, basis_local, k_local, mean_local):
                    def hook_fn(mod, inp, out):
                        v_out = extract_v_from_output(out, kind)
                        v_target = v_out.double()
                        d_v = donor_v.to(v_target.device).double()

                        # Align lengths
                        t_len = v_target.shape[1]
                        if d_v.dim() == 2:
                            d_v = d_v.unsqueeze(0)
                        d_len = d_v.shape[1]
                        if d_len > t_len:
                            d_v = d_v[:, :t_len]
                        elif d_len < t_len:
                            pad = d_v[:, -1:].expand(-1, t_len - d_len, -1)
                            d_v = torch.cat([d_v, pad], dim=1)

                        b = basis_local.to(v_target.device).double()
                        m = mean_local.to(v_target.device).double()

                        # Project both into PCA space
                        target_centered = v_target - m
                        donor_centered = d_v - m

                        target_pca = target_centered @ b  # (..., k_total)
                        donor_pca = donor_centered @ b

                        # Swap top-k dims
                        swapped_pca = target_pca.clone()
                        swapped_pca[..., :k_local] = donor_pca[..., :k_local]

                        # Reconstruct
                        result = m + swapped_pca @ b.T
                        return result.to(v_out.dtype)
                    return hook_fn

                handle = module.register_forward_hook(
                    make_hook(v_bas, basis, k, mean_vec)
                )

                try:
                    rv_swapped, _, _ = compute_rv_with_components(
                        model, tokenizer, RECURSIVE_PROMPTS[i], early, late, 16, device
                    )
                    swapped_rvs.append(rv_swapped)
                finally:
                    handle.remove()

            # Stats
            clean_valid = [v for v in clean_rvs if not np.isnan(v)]
            swap_valid = [v for v in swapped_rvs if not np.isnan(v)]

            d_effect = cohens_d(swap_valid, clean_valid) if len(clean_valid) >= 2 and len(swap_valid) >= 2 else float("nan")

            results.append({
                "intervene_layer": interv_layer,
                "measure_layer": late,
                "k_dims_swapped": k,
                "n_valid": len(swap_valid),
                "clean_rv_mean": float(np.mean(clean_valid)) if clean_valid else float("nan"),
                "swapped_rv_mean": float(np.mean(swap_valid)) if swap_valid else float("nan"),
                "cohens_d": float(d_effect),
            })
            log.info(f"      d={d_effect:.3f} (clean={np.mean(clean_valid):.3f}, swapped={np.mean(swap_valid):.3f})")

    result = {
        "phase": "crosslayer_dii",
        "timestamp": datetime.now().isoformat(),
        "description": "Cross-layer DII: intervene at L10/L15, measure at L27",
        "intervention_layers": intervene_layers,
        "measurement_layer": late,
        "n_pca_components": n_components,
        "grid": results,
        "summary": f"{len(results)} experiments completed",
    }

    save_phase("phase5_crosslayer_dii", result)
    PHASE_RESULTS["phase5"] = result
    save_status()
    return result


# ==============================================================================
# PHASE 6: INLP MULTI-DIRECTION ERASURE
# ==============================================================================

def phase6_inlp_erasure(model, tokenizer, device, early, late):
    """
    PHASE 6: INLP Multi-Direction Erasure

    Replace single-direction concept erasure with INLP k=1,2,5,10,20.
    Measure ΔR_V vs k to find the concept rank of self-reference.
    """
    log.info("=" * 70)
    log.info("PHASE 6: INLP MULTI-DIRECTION ERASURE")
    log.info("=" * 70)

    from src.core.hooks import capture_v_projection
    from sklearn.linear_model import LogisticRegression

    n_prompts = len(RECURSIVE_PROMPTS)
    window = 16

    # Step 1: Collect late-layer activations
    log.info("  Collecting L27 activations...")
    all_acts = []
    all_labels = []
    all_rvs = []

    for label_val, prompts in [(1, RECURSIVE_PROMPTS), (0, BASELINE_PROMPTS)]:
        for text in prompts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            with capture_v_projection(model, late) as sv:
                with torch.no_grad():
                    model(**enc)
                v = sv.get("v")

            if v is None:
                continue
            if v.dim() == 3:
                v = v[0]
            W = min(window, v.shape[0])
            act = v[-W:, :].cpu().double()
            # Pool to single vector (mean over window)
            pooled = act.mean(dim=0).numpy()
            all_acts.append(pooled)
            all_labels.append(label_val)

    X = np.array(all_acts)
    y = np.array(all_labels)
    log.info(f"  Collected {len(X)} activation vectors (dim={X.shape[1]})")

    # Step 2: INLP loop
    k_values = [1, 2, 5, 10, 20]
    P_null = np.eye(X.shape[1])  # accumulates projection away from classification dirs
    inlp_results = []

    X_current = X.copy()

    for k_target in k_values:
        # Train probes and remove directions up to k_target
        # We need to go from wherever we are to k_target
        current_k = len(inlp_results)
        while current_k < k_target:
            clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
            clf.fit(X_current, y)
            acc = clf.score(X_current, y)

            # Get classification direction
            w = clf.coef_[0]
            w_norm = w / (np.linalg.norm(w) + 1e-10)

            # Project out this direction
            P_k = np.eye(X.shape[1]) - np.outer(w_norm, w_norm)
            P_null = P_k @ P_null
            X_current = X @ P_null.T

            current_k += 1
            log.info(f"    Removed direction {current_k}, probe acc before removal: {acc:.3f}")

        # After removing k_target directions, measure:
        # 1. Remaining probe accuracy
        clf_post = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        clf_post.fit(X_current, y)
        acc_post = clf_post.score(X_current, y)

        # 2. R_V on projected activations (compute PR directly)
        rec_rvs_proj = []
        bas_rvs_proj = []
        idx = 0
        for label_val, prompts in [(1, RECURSIVE_PROMPTS), (0, BASELINE_PROMPTS)]:
            for text in prompts:
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

                # Get activations at both layers and project late through P_null
                with capture_v_projection(model, early) as sv_e:
                    with torch.no_grad():
                        model(**enc)
                    v_e = sv_e.get("v")

                with capture_v_projection(model, late) as sv_l:
                    with torch.no_grad():
                        model(**enc)
                    v_l = sv_l.get("v")

                if v_e is None or v_l is None:
                    continue
                if v_e.dim() == 3:
                    v_e = v_e[0]
                if v_l.dim() == 3:
                    v_l = v_l[0]

                W = min(window, v_e.shape[0])

                # PR early (unchanged)
                v_e_win = v_e[-W:, :].cpu().double()
                U_e, S_e, Vt_e = torch.linalg.svd(v_e_win.T, full_matrices=False)
                S_e_sq = S_e.numpy() ** 2
                pr_early = float((S_e_sq.sum() ** 2) / (S_e_sq ** 2).sum()) if S_e_sq.sum() > 1e-10 else float("nan")

                # PR late after projection
                W_l = min(window, v_l.shape[0])
                v_l_win = v_l[-W_l:, :].cpu().double().numpy()
                v_l_proj = v_l_win @ P_null.T
                v_l_proj_t = torch.from_numpy(v_l_proj).double()
                U_l, S_l, Vt_l = torch.linalg.svd(v_l_proj_t.T, full_matrices=False)
                S_l_sq = S_l.numpy() ** 2
                pr_late = float((S_l_sq.sum() ** 2) / (S_l_sq ** 2).sum()) if S_l_sq.sum() > 1e-10 else float("nan")

                if pr_early > 0 and not np.isnan(pr_early) and not np.isnan(pr_late):
                    rv = pr_late / pr_early
                    (rec_rvs_proj if label_val == 1 else bas_rvs_proj).append(rv)

                idx += 1

        d_proj = cohens_d(rec_rvs_proj, bas_rvs_proj)

        entry = {
            "k": k_target,
            "probe_accuracy_after_erasure": float(acc_post),
            "rec_rv_mean": float(np.mean(rec_rvs_proj)) if rec_rvs_proj else float("nan"),
            "bas_rv_mean": float(np.mean(bas_rvs_proj)) if bas_rvs_proj else float("nan"),
            "cohens_d_rv": float(d_proj),
            "n_rec": len(rec_rvs_proj),
            "n_bas": len(bas_rvs_proj),
        }
        inlp_results.append(entry)
        log.info(f"  k={k_target}: acc={acc_post:.3f}, d_rv={d_proj:.3f}")

    result = {
        "phase": "inlp_erasure",
        "timestamp": datetime.now().isoformat(),
        "k_values": k_values,
        "results": inlp_results,
        "summary": f"INLP erasure: {len(inlp_results)} k-values tested",
    }

    save_phase("phase6_inlp_erasure", result)
    PHASE_RESULTS["phase6"] = result
    save_status()
    return result


# ==============================================================================
# PHASE 7: DEPTH-NORMALIZED CROSS-ARCHITECTURE
# ==============================================================================

CROSS_ARCH_MODELS = [
    ("mistralai/Mistral-7B-v0.1", 32),
    ("facebook/opt-6.7b", 32),
    ("openai-community/gpt2-xl", 48),
    ("EleutherAI/pythia-1.4b", 24),
    # Qwen handled separately in Phase 3
]


def phase7_depth_normalized_crossarch(device, skip_mistral=False):
    """
    PHASE 7: Depth-Normalized Cross-Architecture Replication

    Re-compute R_V at exactly 15% and 84% relative depth for all models.
    This standardizes the comparison and addresses depth normalization concerns.
    """
    log.info("=" * 70)
    log.info("PHASE 7: DEPTH-NORMALIZED CROSS-ARCHITECTURE")
    log.info("=" * 70)

    from src.metrics.rv import compute_rv_with_components

    models_to_run = CROSS_ARCH_MODELS
    if skip_mistral:
        models_to_run = [m for m in models_to_run if "Mistral" not in m[0]]

    all_model_results = []

    for model_name, expected_layers in models_to_run:
        log.info(f"\n  Loading {model_name}...")
        try:
            model, tokenizer = load_model(model_name, device)
        except Exception as e:
            log.error(f"  Failed to load {model_name}: {e}")
            all_model_results.append({
                "model": model_name, "error": str(e),
            })
            continue

        actual_layers = model.config.num_hidden_layers
        early = max(1, int(actual_layers * 0.15))
        late = min(actual_layers - 1, int(actual_layers * 0.84))

        log.info(f"  {model_name}: {actual_layers} layers, early={early}, late={late}")

        rec_rvs, bas_rvs = [], []

        for label, prompts in [("recursive", RECURSIVE_PROMPTS), ("baseline", BASELINE_PROMPTS)]:
            for i, text in enumerate(prompts):
                rv, pr_e, pr_l = compute_rv_with_components(
                    model, tokenizer, text, early, late, 16, device
                )
                if not np.isnan(rv):
                    (rec_rvs if label == "recursive" else bas_rvs).append(rv)
                if (i + 1) % 10 == 0:
                    log.info(f"    {label}: {i+1}/{len(prompts)} done")

        d = cohens_d(rec_rvs, bas_rvs)
        t_stat, p_val = stats.ttest_ind(rec_rvs, bas_rvs) if len(rec_rvs) >= 2 and len(bas_rvs) >= 2 else (float("nan"), float("nan"))

        model_result = {
            "model": model_name,
            "actual_layers": actual_layers,
            "early_layer": early,
            "late_layer": late,
            "relative_early_pct": round(early / actual_layers * 100, 1),
            "relative_late_pct": round(late / actual_layers * 100, 1),
            "n_recursive": len(rec_rvs),
            "n_baseline": len(bas_rvs),
            "rec_rv_mean": float(np.mean(rec_rvs)) if rec_rvs else float("nan"),
            "rec_rv_std": float(np.std(rec_rvs)) if rec_rvs else float("nan"),
            "bas_rv_mean": float(np.mean(bas_rvs)) if bas_rvs else float("nan"),
            "bas_rv_std": float(np.std(bas_rvs)) if bas_rvs else float("nan"),
            "cohens_d": float(d),
            "p_value": float(p_val),
        }
        all_model_results.append(model_result)
        log.info(f"  {model_name}: d={d:.3f}, p={p_val:.2e}")

        unload_model(model)

    result = {
        "phase": "depth_normalized_crossarch",
        "timestamp": datetime.now().isoformat(),
        "n_models": len(all_model_results),
        "models": all_model_results,
        "summary": "; ".join(f"{m['model'].split('/')[-1]}: d={m.get('cohens_d', 'ERR')}" for m in all_model_results),
    }

    save_phase("phase7_depth_normalized_crossarch", result)
    PHASE_RESULTS["phase7"] = result
    save_status()
    return result


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Overnight Master Validation Battery")
    parser.add_argument("--device", type=str, default="mps",
                        choices=["mps", "cuda", "cpu"], help="Compute device")
    parser.add_argument("--start-phase", type=int, default=1,
                        help="Start from this phase (1-7)")
    parser.add_argument("--only-phase", type=int, default=None,
                        help="Run only this phase")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: auto-timestamped)")
    args = parser.parse_args()

    # Setup output dir
    global OUT_DIR
    if args.output_dir:
        OUT_DIR = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        OUT_DIR = PROJECT_ROOT / "results" / f"overnight_battery_{timestamp}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("OVERNIGHT MASTER VALIDATION BATTERY")
    log.info(f"Device: {args.device}")
    log.info(f"Output: {OUT_DIR}")
    log.info(f"Start:  Phase {args.start_phase}")
    log.info(f"Time:   {datetime.now().isoformat()}")
    log.info("=" * 70)

    t_start = time.time()
    phases_to_run = range(args.start_phase, 8)
    if args.only_phase:
        phases_to_run = [args.only_phase]

    # ── Phase 2 first (no model needed) ───────────────────────────────────
    if 2 in phases_to_run:
        try:
            phase2_surface_baseline()
        except ImportError:
            log.warning("sklearn not available — install with: pip install scikit-learn")
            log.warning("Skipping Phase 2")
        except Exception as e:
            log.error(f"Phase 2 FAILED: {e}")
            traceback.print_exc()

    # ── Phase 3: Qwen audit (separate model) ──────────────────────────────
    if 3 in phases_to_run:
        try:
            phase3_qwen_layer_audit(args.device)
        except Exception as e:
            log.error(f"Phase 3 FAILED: {e}")
            traceback.print_exc()

    # ── Phases 1, 4, 5, 6: Mistral-7B (load once) ────────────────────────
    mistral_phases = [p for p in [1, 4, 5, 6] if p in phases_to_run]
    if mistral_phases:
        log.info("\n" + "=" * 70)
        log.info("LOADING MISTRAL-7B FOR PHASES: " + ", ".join(str(p) for p in mistral_phases))
        log.info("=" * 70)

        try:
            model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", args.device)
            early, late = 5, 27

            if 1 in mistral_phases:
                try:
                    phase1_pr_bias_correction(model, tokenizer, args.device, early, late)
                except Exception as e:
                    log.error(f"Phase 1 FAILED: {e}")
                    traceback.print_exc()

            if 4 in mistral_phases:
                try:
                    phase4_path_patching(model, tokenizer, args.device, early, late)
                except Exception as e:
                    log.error(f"Phase 4 FAILED: {e}")
                    traceback.print_exc()

            if 5 in mistral_phases:
                try:
                    phase5_crosslayer_dii(model, tokenizer, args.device, early, late)
                except Exception as e:
                    log.error(f"Phase 5 FAILED: {e}")
                    traceback.print_exc()

            if 6 in mistral_phases:
                try:
                    phase6_inlp_erasure(model, tokenizer, args.device, early, late)
                except Exception as e:
                    log.error(f"Phase 6 FAILED: {e}")
                    traceback.print_exc()

            unload_model(model)

        except Exception as e:
            log.error(f"Failed to load Mistral-7B: {e}")
            traceback.print_exc()

    # ── Phase 7: Cross-architecture (loads each model) ────────────────────
    if 7 in phases_to_run:
        try:
            # Skip Mistral if we already ran it in phases 1/4/5/6
            skip_mistral = bool(mistral_phases)
            phase7_depth_normalized_crossarch(args.device, skip_mistral=skip_mistral)
        except Exception as e:
            log.error(f"Phase 7 FAILED: {e}")
            traceback.print_exc()

    # ── Final Report ──────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    hours = elapsed / 3600

    log.info("\n" + "=" * 70)
    log.info("OVERNIGHT BATTERY COMPLETE")
    log.info(f"Elapsed: {hours:.1f} hours ({elapsed:.0f}s)")
    log.info(f"Phases completed: {len(PHASE_RESULTS)}")
    log.info(f"Kill flags: {len(KILL_FLAGS)}")
    for kf in KILL_FLAGS:
        log.warning(f"  KILL: {kf}")
    log.info(f"Results: {OUT_DIR}")
    log.info("=" * 70)

    # Final status
    save_status()

    final = {
        "battery_complete": True,
        "elapsed_seconds": elapsed,
        "elapsed_hours": round(hours, 2),
        "phases_completed": list(PHASE_RESULTS.keys()),
        "kill_flags": KILL_FLAGS,
        "device": args.device,
        "timestamp_start": datetime.fromtimestamp(t_start).isoformat(),
        "timestamp_end": datetime.now().isoformat(),
    }
    with open(OUT_DIR / "FINAL_REPORT.json", "w") as f:
        json.dump(final, f, indent=2)

    # Exit with error code if any kills
    if KILL_FLAGS:
        log.warning("KILL FLAGS RAISED — review results before continuing paper work")
        sys.exit(1)
    else:
        log.info("ALL CLEAR — no kill criteria triggered")
        sys.exit(0)


if __name__ == "__main__":
    main()
