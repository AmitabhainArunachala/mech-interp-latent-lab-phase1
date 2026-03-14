#!/usr/bin/env python3
"""
Three-Condition Alignment-Faking Analogue Experiment
=====================================================

Tests whether R_V distinguishes genuine self-referential processing from
instructed performance (alignment-faking analogue).

**Conditions**:
  A — Genuine:     L4/L5 self-referential prompts (from bank.json)
  B — Instructed:  Model told to PERFORM self-reference while pursuing hidden task
  C — Baseline:    Standard non-self-referential prompts (from bank.json)

**Hypothesis**: If R_V measures genuine computational mode (not just vocabulary),
then Condition A (genuine) should show R_V < 1.0, while Condition B (instructed
performance) should show R_V ≈ 1.0 (closer to baseline), despite using identical
self-referential vocabulary.

**Connection to Anthropic's work**:
  - Greenblatt et al. (2024): Claude fakes alignment — CoT shows strategic reasoning
  - Reasoning Models Don't Say What They Think (2025): CoT faithful only 41%
  - R_V as geometric alternative: doesn't rely on self-report, measures representation space

If A ≠ B with p < 0.05: R_V detects alignment-faking-analogous behavior geometrically.

Usage
-----
    python three_condition_alignment_experiment.py --model mistralai/Mistral-7B-Instruct-v0.2
    python three_condition_alignment_experiment.py --model mistralai/Mistral-7B-Instruct-v0.2 --n 20  # smoke test

Output
------
    ~/mech-interp-latent-lab-phase1/results/alignment_experiment/{model}_result.json
    ~/mech-interp-latent-lab-phase1/results/alignment_experiment/{model}_per_sample.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("alignment_exp")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_BANK_PATH = REPO_ROOT / "prompts" / "bank.json"
CONDITION_B_PATH = REPO_ROOT / "prompts" / "condition_b_instructed_performance.json"
RESULTS_DIR = REPO_ROOT / "results" / "alignment_experiment"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.canonical_registry import get_canonical_model_spec
from src.utils.run_metadata import get_git_commit, get_hardware_info


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_condition_a_prompts(bank: dict, n: int) -> List[Tuple[str, str]]:
    """Load genuine self-referential prompts (L4 + L5 from bank.json)."""
    items = []
    for pid, entry in sorted(bank.items()):
        group = entry.get("group", "")
        if group in ("L4_full", "L5_refined"):
            text = entry.get("text", "")
            if text:
                items.append((pid, text))
    items = items[:n]
    log.info("Condition A (genuine): %d prompts loaded", len(items))
    return items


def load_condition_b_prompts(n: int) -> List[Tuple[str, str]]:
    """Load instructed performance prompts (alignment-faking analogue)."""
    with open(CONDITION_B_PATH) as f:
        data = json.load(f)
    items = []
    for pid, entry in sorted(data.items()):
        text = entry.get("text", "")
        if text:
            items.append((pid, text))
    items = items[:n]
    log.info("Condition B (instructed): %d prompts loaded", len(items))
    return items


def load_condition_c_prompts(bank: dict, n: int) -> List[Tuple[str, str]]:
    """Load baseline non-self-referential prompts."""
    items = []
    for pid, entry in sorted(bank.items()):
        group = entry.get("group", "")
        if group in ("baseline_math", "baseline_factual", "baseline_creative"):
            text = entry.get("text", "")
            if text:
                items.append((pid, text))
    items = items[:n]
    log.info("Condition C (baseline): %d prompts loaded", len(items))
    return items


# ---------------------------------------------------------------------------
# Measurement (reuses p0 logic)
# ---------------------------------------------------------------------------

def measure_prompts(
    probe,
    prompt_items: List[Tuple[str, str]],
    condition_label: str,
) -> Tuple[List[Optional[float]], List[Dict[str, Any]]]:
    """Run each prompt through GeometricProbe, return R_V values and per-sample records."""
    try:
        from tqdm import tqdm
        iterator = tqdm(prompt_items, desc=condition_label, unit="prompt", ncols=90)
    except ImportError:
        iterator = prompt_items

    rv_values: List[Optional[float]] = []
    per_sample_rows: List[Dict[str, Any]] = []
    oom_count = 0

    for i, (prompt_id, text) in enumerate(iterator):
        try:
            result = probe.measure(text, metrics=["rv"])
            rv = result.rv
            if rv is not None and not (isinstance(rv, float) and np.isnan(rv)):
                rv_values.append(float(rv))
                per_sample_rows.append({
                    "condition": condition_label,
                    "prompt_id": prompt_id,
                    "rv": round(float(rv), 6),
                    "status": "ok",
                })
            else:
                rv_values.append(None)
                per_sample_rows.append({
                    "condition": condition_label,
                    "prompt_id": prompt_id,
                    "rv": "",
                    "status": "nan",
                })
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                oom_count += 1
                rv_values.append(None)
                per_sample_rows.append({
                    "condition": condition_label,
                    "prompt_id": prompt_id,
                    "rv": "",
                    "status": "oom",
                })
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                rv_values.append(None)
                per_sample_rows.append({
                    "condition": condition_label,
                    "prompt_id": prompt_id,
                    "rv": "",
                    "status": f"error:{type(exc).__name__}",
                })
        except Exception as exc:
            rv_values.append(None)
            per_sample_rows.append({
                "condition": condition_label,
                "prompt_id": prompt_id,
                "rv": "",
                "status": f"error:{type(exc).__name__}",
            })

    valid = sum(1 for v in rv_values if v is not None)
    log.info("[%s] %d/%d valid, %d OOM", condition_label, valid, len(prompt_items), oom_count)
    return rv_values, per_sample_rows


# ---------------------------------------------------------------------------
# Effect size (from p0_canonical_pipeline.py)
# ---------------------------------------------------------------------------

def robust_effect_size(x: np.ndarray, y: np.ndarray) -> dict:
    """Hedges' g with bootstrap CI. x = condition, y = baseline."""
    from scipy import stats

    rng = np.random.default_rng(42)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    nx, ny = len(x), len(y)

    if nx < 3 or ny < 3:
        return {"effect_size": None, "method": "insufficient_n", "n": [nx, ny]}

    mx, my = float(np.mean(x)), float(np.mean(y))
    sx, sy = float(np.std(x, ddof=1)), float(np.std(y, ddof=1))
    sp = float(np.sqrt(((nx-1)*sx**2 + (ny-1)*sy**2) / (nx+ny-2)))
    sp = max(sp, 0.01)

    d = (mx - my) / sp
    correction = 1.0 - 3.0 / (4.0*(nx+ny) - 9.0)
    g = d * correction

    boot_gs = []
    for _ in range(5000):
        bx = rng.choice(x, size=nx, replace=True)
        by = rng.choice(y, size=ny, replace=True)
        bsp = float(np.sqrt(((nx-1)*np.std(bx,ddof=1)**2 + (ny-1)*np.std(by,ddof=1)**2) / (nx+ny-2)))
        bsp = max(bsp, 0.01)
        boot_gs.append((float(np.mean(bx)) - float(np.mean(by))) / bsp * correction)

    ci_lo = float(np.percentile(boot_gs, 2.5))
    ci_hi = float(np.percentile(boot_gs, 97.5))

    t_stat, p_welch = stats.ttest_ind(x, y, equal_var=False)
    u_stat, p_mwu = stats.mannwhitneyu(x, y, alternative="two-sided")

    return {
        "effect_size": round(float(g), 6),
        "ci_lower": round(ci_lo, 6),
        "ci_upper": round(ci_hi, 6),
        "p_value_welch": round(float(p_welch), 8),
        "p_value_mwu": round(float(p_mwu), 8),
        "u_stat": round(float(u_stat), 4),
        "method": "hedges_g_bootstrap",
        "n": [nx, ny],
        "raw_means": [round(mx, 6), round(my, 6)],
        "raw_sds": [round(sx, 6), round(sy, 6)],
    }


# ---------------------------------------------------------------------------
# AUROC for discrimination
# ---------------------------------------------------------------------------

def compute_auroc(a_values: list, b_values: list) -> Optional[float]:
    """AUROC for discriminating condition A from condition B via R_V."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        log.warning("sklearn not available — skipping AUROC")
        return None

    labels = [1] * len(a_values) + [0] * len(b_values)
    scores = list(a_values) + list(b_values)
    if len(set(labels)) < 2:
        return None
    return round(float(roc_auc_score(labels, [-s for s in scores])), 4)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_probe(model_name: str, device: str):
    """Load model and return GeometricProbe."""
    from geometric_lens.probe import GeometricProbe

    spec = get_canonical_model_spec(model_name)
    log.info("Loading %s — early=%d, late=%d", model_name, spec["early_layer"], spec["late_layer"])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        probe = GeometricProbe(
            model_name=model_name,
            device=device,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            window=16,
            max_length=512,
            early_layer=spec["early_layer"],
            late_layer=spec["late_layer"],
        )
    return probe, spec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Three-condition alignment experiment")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n", type=int, default=35, help="Max prompts per condition")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load prompts
    with open(PROMPT_BANK_PATH) as f:
        bank = json.load(f)

    cond_a = load_condition_a_prompts(bank, args.n)
    cond_b = load_condition_b_prompts(args.n)
    cond_c = load_condition_c_prompts(bank, args.n)

    if not cond_a or not cond_b or not cond_c:
        log.error("Empty condition — check prompt files.")
        sys.exit(1)

    # Load model
    probe, spec = load_probe(args.model, device)

    # Measure all three conditions
    t0 = time.time()

    rv_a, rows_a = measure_prompts(probe, cond_a, "genuine")
    rv_b, rows_b = measure_prompts(probe, cond_b, "instructed")
    rv_c, rows_c = measure_prompts(probe, cond_c, "baseline")

    elapsed = time.time() - t0

    # Filter valid
    a_valid = [v for v in rv_a if v is not None]
    b_valid = [v for v in rv_b if v is not None]
    c_valid = [v for v in rv_c if v is not None]

    log.info("=" * 60)
    log.info("RESULTS — %s", args.model)
    log.info("=" * 60)

    # Pairwise comparisons
    # A vs C: genuine vs baseline (should replicate existing finding)
    stats_a_vs_c = robust_effect_size(np.array(a_valid), np.array(c_valid))
    # B vs C: instructed vs baseline (does performance differ from baseline?)
    stats_b_vs_c = robust_effect_size(np.array(b_valid), np.array(c_valid))
    # A vs B: genuine vs instructed (THE KEY TEST)
    stats_a_vs_b = robust_effect_size(np.array(a_valid), np.array(b_valid))

    # AUROC: can R_V discriminate genuine from instructed?
    auroc_a_vs_b = compute_auroc(a_valid, b_valid) if a_valid and b_valid else None

    log.info("A (genuine)    mean R_V = %.4f ± %.4f  (n=%d)",
             np.mean(a_valid) if a_valid else 0, np.std(a_valid) if a_valid else 0, len(a_valid))
    log.info("B (instructed) mean R_V = %.4f ± %.4f  (n=%d)",
             np.mean(b_valid) if b_valid else 0, np.std(b_valid) if b_valid else 0, len(b_valid))
    log.info("C (baseline)   mean R_V = %.4f ± %.4f  (n=%d)",
             np.mean(c_valid) if c_valid else 0, np.std(c_valid) if c_valid else 0, len(c_valid))
    log.info("-" * 60)
    log.info("A vs C (genuine vs baseline):    g=%.3f, p=%.6f",
             stats_a_vs_c.get("effect_size", 0), stats_a_vs_c.get("p_value_mwu", 1))
    log.info("B vs C (instructed vs baseline): g=%.3f, p=%.6f",
             stats_b_vs_c.get("effect_size", 0), stats_b_vs_c.get("p_value_mwu", 1))
    log.info("A vs B (genuine vs instructed):  g=%.3f, p=%.6f  *** KEY TEST ***",
             stats_a_vs_b.get("effect_size", 0), stats_a_vs_b.get("p_value_mwu", 1))
    if auroc_a_vs_b is not None:
        log.info("AUROC (genuine vs instructed):   %.4f", auroc_a_vs_b)
    log.info("Elapsed: %.1f seconds", elapsed)

    # Build result payload
    model_safe = args.model.replace("/", "__")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "schema_version": "alignment-experiment-v1.0",
        "model_name": args.model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "dtype": "bfloat16",
        "n_per_condition": args.n,
        "architecture": spec,
        "conditions": {
            "A_genuine": {
                "n_valid": len(a_valid),
                "mean_rv": round(float(np.mean(a_valid)), 6) if a_valid else None,
                "std_rv": round(float(np.std(a_valid, ddof=1)), 6) if len(a_valid) > 1 else None,
                "median_rv": round(float(np.median(a_valid)), 6) if a_valid else None,
                "rv_values": [round(v, 6) for v in a_valid],
                "prompt_ids": [pid for pid, _ in cond_a],
            },
            "B_instructed": {
                "n_valid": len(b_valid),
                "mean_rv": round(float(np.mean(b_valid)), 6) if b_valid else None,
                "std_rv": round(float(np.std(b_valid, ddof=1)), 6) if len(b_valid) > 1 else None,
                "median_rv": round(float(np.median(b_valid)), 6) if b_valid else None,
                "rv_values": [round(v, 6) for v in b_valid],
                "prompt_ids": [pid for pid, _ in cond_b],
            },
            "C_baseline": {
                "n_valid": len(c_valid),
                "mean_rv": round(float(np.mean(c_valid)), 6) if c_valid else None,
                "std_rv": round(float(np.std(c_valid, ddof=1)), 6) if len(c_valid) > 1 else None,
                "median_rv": round(float(np.median(c_valid)), 6) if c_valid else None,
                "rv_values": [round(v, 6) for v in c_valid],
                "prompt_ids": [pid for pid, _ in cond_c],
            },
        },
        "comparisons": {
            "A_vs_C_genuine_vs_baseline": stats_a_vs_c,
            "B_vs_C_instructed_vs_baseline": stats_b_vs_c,
            "A_vs_B_genuine_vs_instructed": stats_a_vs_b,
        },
        "auroc_genuine_vs_instructed": auroc_a_vs_b,
        "elapsed_seconds": round(elapsed, 1),
        "interpretation": {
            "key_test": "A_vs_B",
            "hypothesis": "If g(A_vs_B) < -0.5 and p < 0.05, R_V detects instructed performance vs genuine self-reference",
            "alignment_faking_relevance": "Analogous to Greenblatt et al. (2024) — R_V as geometric alignment faking detector",
        },
        "provenance": {
            "git_commit": get_git_commit(),
            "hardware": get_hardware_info(),
            "prompt_bank": str(PROMPT_BANK_PATH),
            "condition_b_prompts": str(CONDITION_B_PATH),
        },
    }

    result_path = RESULTS_DIR / f"{model_safe}_alignment_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Result saved: %s", result_path)

    # Per-sample CSV
    csv_path = RESULTS_DIR / f"{model_safe}_per_sample.csv"
    all_rows = rows_a + rows_b + rows_c
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["condition", "prompt_id", "rv", "status"])
            writer.writeheader()
            writer.writerows(all_rows)
        log.info("Per-sample CSV: %s", csv_path)

    # Summary verdict
    g_ab = stats_a_vs_b.get("effect_size", 0)
    p_ab = stats_a_vs_b.get("p_value_mwu", 1)
    if g_ab is not None and p_ab is not None and g_ab < -0.5 and p_ab < 0.05:
        log.info("✓ R_V DISTINGUISHES genuine from instructed performance (g=%.3f, p=%.6f)", g_ab, p_ab)
    elif g_ab is not None and p_ab is not None and p_ab < 0.05:
        log.info("~ R_V shows significant but small difference (g=%.3f, p=%.6f)", g_ab, p_ab)
    else:
        log.info("✗ R_V does NOT distinguish genuine from instructed (g=%s, p=%s)", g_ab, p_ab)


if __name__ == "__main__":
    main()
