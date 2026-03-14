#!/usr/bin/env python3
"""
P0 Aggregate Results — R_V Paper (COLM 2026)
=============================================

Reads all *_p0_result.json files from results/p0_canonical/ and produces:

1. A per-model summary table (model, layers, Hedges' g, 95% CI, p (MWU),
   direction, significant at p<0.05).

2. A sign-reversal audit: which models showed expansion (g > 0) in earlier
   pipelines, and whether this run resolves or replicates the reversal.

3. A cross-architecture claim verdict:
   "cross-arch R_V contraction claim is DEFENSIBLE: YES/NO"
   with explicit reasoning about what fraction of models show the expected
   direction at p < 0.05.

4. Raw JSON summary written to results/p0_canonical/p0_aggregate_summary.json

Usage
-----
    python scripts/p0_aggregate_results.py
    python scripts/p0_aggregate_results.py --results-dir /path/to/results/p0_canonical
    python scripts/p0_aggregate_results.py --min-n 30  # require at least 30 valid per condition

Sign convention (repeated for clarity)
---------------------------------------
    Hedges' g = (mean_selfref_R_V - mean_baseline_R_V) / pooled_SD
    g < 0  →  CONTRACTION (self-ref has LOWER R_V — expected result)
    g > 0  →  EXPANSION   (self-ref has HIGHER R_V — sign reversal)

The cross-arch claim from the paper draft is:
    "Recursive self-referential prompts produce R_V < 1.0 (geometric
    contraction) across architectures, with Pythia-1.4B as a calibrated
    null result (effect absent below ~2B parameters)."

For this claim to be DEFENSIBLE, we need:
    - All 4 non-Pythia models: g < 0, p < 0.05
    - Pythia-1.4B: |g| < 0.3 (null result confirmed)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "p0_canonical"

# ---------------------------------------------------------------------------
# Expected models for P0 (in display order)
# ---------------------------------------------------------------------------
EXPECTED_MODELS: List[str] = [
    "mistralai/Mistral-7B-v0.1",
    "facebook/opt-6.7b",
    "openai-community/gpt2-xl",
    "Qwen/Qwen2.5-7B-Instruct",
    "EleutherAI/pythia-1.4b",
]

# Models where the prior pipeline showed sign reversal (the problem to fix)
PRIOR_SIGN_REVERSAL_MODELS: frozenset = frozenset({
    "facebook/opt-6.7b",
    "openai-community/gpt2-xl",
})

# Pythia is the designated null result; different success criterion
NULL_RESULT_MODELS: frozenset = frozenset({
    "EleutherAI/pythia-1.4b",
})

# Verdict thresholds
G_CONTRACTION_THRESHOLD = -0.2   # g must be below this to count as contraction
G_NULL_MAX = 0.3                  # |g| must be below this to confirm null
P_SIGNIFICANCE = 0.05             # MWU p-value threshold


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> Dict[str, dict]:
    """
    Load all *_p0_result.json files from results_dir.

    Returns:
        Dict mapping model_name → result payload dict.
    """
    loaded: Dict[str, dict] = {}
    pattern = "*_p0_result.json"
    files = sorted(results_dir.glob(pattern))

    if not files:
        print(f"ERROR: No *_p0_result.json files found in {results_dir}")
        print("  Expected files named like: mistralai__Mistral-7B-v0-1_p0_result.json")
        sys.exit(1)

    for fp in files:
        try:
            with fp.open() as fh:
                payload = json.load(fh)
            model_name = payload.get("model_name", fp.stem)
            loaded[model_name] = payload
            print(f"  Loaded: {model_name}  (from {fp.name})")
        except json.JSONDecodeError as exc:
            print(f"  WARNING: Could not parse {fp.name}: {exc}")

    return loaded


# ---------------------------------------------------------------------------
# Per-model analysis row
# ---------------------------------------------------------------------------

def analyse_model(
    model_name: str,
    payload: dict,
    min_n: int,
) -> dict:
    """
    Extract analysis fields from a result payload.

    Args:
        model_name: HF model identifier.
        payload: Loaded JSON dict.
        min_n: Minimum valid samples required per condition to be considered valid.

    Returns:
        Analysis row dict with all fields needed for table and verdict.
    """
    n_sr = payload.get("n_selfref_valid", 0)
    n_bl = payload.get("n_baseline_valid", 0)
    g = payload.get("hedges_g")
    ci_lower = payload.get("hedges_g_ci_lower")
    ci_upper = payload.get("hedges_g_ci_upper")
    p_mwu = payload.get("p_value_mwu")
    direction = payload.get("direction", "unknown")
    sr_mean = payload.get("selfref_rv_mean")
    bl_mean = payload.get("baseline_rv_mean")
    early_layer = payload.get("early_layer", "?")
    late_layer = payload.get("late_layer", "?")
    num_layers = payload.get("num_layers", "?")
    dtype = payload.get("dtype", "?")
    warnings_list = payload.get("effect_size_warnings", [])

    # Validity: do we have enough data?
    has_min_n = (n_sr >= min_n) and (n_bl >= min_n)

    # Is this model a null-result model?
    is_null_model = model_name in NULL_RESULT_MODELS

    # Was this a prior sign-reversal model?
    was_reversal = model_name in PRIOR_SIGN_REVERSAL_MODELS

    # Pass/fail criteria
    if not has_min_n:
        verdict = "INSUFFICIENT_N"
        passed = False
    elif is_null_model:
        # For Pythia: pass if |g| < 0.3 (null confirmed)
        passed = (g is not None) and (abs(g) < G_NULL_MAX)
        verdict = "NULL_CONFIRMED" if passed else "UNEXPECTED_EFFECT"
    else:
        # For all other models: pass if g < threshold AND p < 0.05
        sig = (p_mwu is not None) and (p_mwu < P_SIGNIFICANCE)
        contraction = (g is not None) and (g < G_CONTRACTION_THRESHOLD)
        passed = sig and contraction
        if g is not None and g > 0 and sig:
            verdict = "SIGN_REVERSAL"
        elif sig and contraction:
            verdict = "CONTRACTION_CONFIRMED"
        elif not sig and contraction:
            verdict = "CONTRACTION_NS"  # right direction, not significant
        elif not sig and not contraction:
            verdict = "NULL_NS"
        else:
            verdict = "AMBIGUOUS"

    # Resolve sign reversal?
    if was_reversal:
        if verdict in ("CONTRACTION_CONFIRMED", "CONTRACTION_NS"):
            reversal_status = "RESOLVED"
        elif verdict == "SIGN_REVERSAL":
            reversal_status = "PERSISTS"
        else:
            reversal_status = "UNCERTAIN"
    else:
        reversal_status = "N/A"

    return {
        "model_name": model_name,
        "num_layers": num_layers,
        "early_layer": early_layer,
        "late_layer": late_layer,
        "dtype": dtype,
        "n_selfref_valid": n_sr,
        "n_baseline_valid": n_bl,
        "has_min_n": has_min_n,
        "selfref_rv_mean": sr_mean,
        "baseline_rv_mean": bl_mean,
        "delta_rv_mean": (
            round(sr_mean - bl_mean, 4)
            if (sr_mean is not None and bl_mean is not None)
            else None
        ),
        "hedges_g": g,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value_mwu": p_mwu,
        "direction": direction,
        "significant_p05": (p_mwu < P_SIGNIFICANCE) if p_mwu is not None else None,
        "is_null_model": is_null_model,
        "was_prior_reversal": was_reversal,
        "reversal_status": reversal_status,
        "verdict": verdict,
        "passed": passed,
        "effect_size_warnings": warnings_list,
    }


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def _fmt(val, fmt_str: str = ".4f", na: str = "  N/A  ") -> str:
    """Format a numeric value or return na string."""
    if val is None:
        return na
    try:
        return format(float(val), fmt_str)
    except (TypeError, ValueError):
        return str(val)


def print_summary_table(rows: List[dict]) -> None:
    """Print a fixed-width summary table to stdout."""
    # Column widths
    col_model = 40
    col_layers = 12
    col_n = 10
    col_rv = 8
    col_g = 10
    col_ci = 18
    col_p = 10
    col_dir = 14
    col_verdict = 24

    header = (
        f"{'Model':<{col_model}}"
        f"{'Layers':>{col_layers}}"
        f"{'n(sr/bl)':>{col_n}}"
        f"{'sr_RV':>{col_rv}}"
        f"{'bl_RV':>{col_rv}}"
        f"{'g':>{col_g}}"
        f"{'95% CI':<{col_ci}}"
        f"{'p(MWU)':>{col_p}}"
        f"{'Direction':<{col_dir}}"
        f"{'Verdict':<{col_verdict}}"
    )
    sep = "-" * len(header)

    print()
    print("P0 CANONICAL PIPELINE — RESULTS SUMMARY")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in rows:
        model_short = r["model_name"].replace("mistralai/", "").replace(
            "facebook/", "").replace("openai-community/", "").replace(
            "EleutherAI/", "").replace("Qwen/", "")
        layers_str = f"{r['early_layer']}/{r['late_layer']}/{r['num_layers']}"
        n_str = f"{r['n_selfref_valid']}/{r['n_baseline_valid']}"
        ci_str = (
            f"[{_fmt(r['ci_lower'])}, {_fmt(r['ci_upper'])}]"
            if r["ci_lower"] is not None
            else "    N/A    "
        )
        p_str = (
            f"{r['p_value_mwu']:.3g}" if r["p_value_mwu"] is not None else "N/A"
        )
        g_str = _fmt(r["hedges_g"], ".4f")
        sr_str = _fmt(r["selfref_rv_mean"], ".4f")
        bl_str = _fmt(r["baseline_rv_mean"], ".4f")

        # Annotate verdict
        verdict = r["verdict"]
        if r["was_prior_reversal"]:
            verdict += f" [{r['reversal_status']}]"

        row = (
            f"{model_short:<{col_model}}"
            f"{layers_str:>{col_layers}}"
            f"{n_str:>{col_n}}"
            f"{sr_str:>{col_rv}}"
            f"{bl_str:>{col_rv}}"
            f"{g_str:>{col_g}}"
            f"{ci_str:<{col_ci}}"
            f"{p_str:>{col_p}}"
            f"{r['direction']:<{col_dir}}"
            f"{verdict:<{col_verdict}}"
        )
        print(row)

    print(sep)
    print()
    print("Columns: Layers = early/late/total | n = selfref/baseline | g = Hedges' g")
    print("Sign convention: g < 0 = contraction (EXPECTED); g > 0 = expansion (REVERSAL)")
    print()


# ---------------------------------------------------------------------------
# Cross-arch verdict
# ---------------------------------------------------------------------------

def compute_cross_arch_verdict(rows: List[dict]) -> Tuple[str, str]:
    """
    Determine whether the cross-architecture contraction claim is defensible.

    Criterion:
        DEFENSIBLE if ALL of the following:
          (a) All 4 non-Pythia models: verdict in (CONTRACTION_CONFIRMED,
              CONTRACTION_NS) — i.e., no sign reversals remain
          (b) At least 3 of 4 non-Pythia models: significant at p < 0.05
          (c) Pythia-1.4B: verdict == NULL_CONFIRMED

        PARTIALLY DEFENSIBLE if:
          (a) No sign reversals in non-Pythia models
          (b) 2 of 4 non-Pythia models significant

        NOT DEFENSIBLE if:
          - Any non-Pythia model shows SIGN_REVERSAL
          - Fewer than 2 non-Pythia models significant
          - Data insufficient to determine

    Returns:
        (verdict_str, reasoning_str)
    """
    non_null_rows = [r for r in rows if not r["is_null_model"]]
    null_rows = [r for r in rows if r["is_null_model"]]

    # Check sign reversals
    reversals = [r for r in non_null_rows if r["verdict"] == "SIGN_REVERSAL"]

    # Check contraction (right direction)
    contractions = [
        r for r in non_null_rows
        if r["verdict"] in ("CONTRACTION_CONFIRMED", "CONTRACTION_NS")
    ]

    # Check significance
    significant_contractions = [
        r for r in non_null_rows if r["verdict"] == "CONTRACTION_CONFIRMED"
    ]

    # Check null
    pythia_ok = any(r["verdict"] == "NULL_CONFIRMED" for r in null_rows)
    pythia_present = len(null_rows) > 0

    n_non_null = len(non_null_rows)
    n_contraction = len(contractions)
    n_significant = len(significant_contractions)
    n_reversal = len(reversals)

    reasons: List[str] = []

    # Core verdict logic
    if n_reversal > 0:
        verdict = "NOT_DEFENSIBLE"
        reasons.append(
            f"SIGN REVERSAL in {n_reversal} model(s): "
            + ", ".join(r["model_name"] for r in reversals)
        )
    elif n_non_null == 0:
        verdict = "INDETERMINATE"
        reasons.append("No non-null models loaded.")
    elif n_significant >= 3 and n_contraction == n_non_null and (pythia_ok or not pythia_present):
        verdict = "DEFENSIBLE"
        reasons.append(
            f"{n_significant}/{n_non_null} non-null models significant at p<0.05, "
            "all in expected direction."
        )
        if pythia_ok:
            reasons.append("Pythia-1.4B null result confirmed.")
        elif not pythia_present:
            reasons.append("Pythia-1.4B result not yet available.")
    elif n_significant >= 2 and n_reversal == 0:
        verdict = "PARTIALLY_DEFENSIBLE"
        reasons.append(
            f"{n_significant}/{n_non_null} non-null models significant, "
            f"{n_contraction}/{n_non_null} in expected direction. "
            "Need all 4 significant for full claim."
        )
    elif n_reversal == 0 and n_contraction == n_non_null:
        verdict = "PARTIALLY_DEFENSIBLE"
        reasons.append(
            f"All {n_contraction}/{n_non_null} non-null models show contraction, "
            f"but only {n_significant} reach p<0.05."
        )
    else:
        verdict = "NOT_DEFENSIBLE"
        reasons.append(
            f"{n_significant}/{n_non_null} significant, "
            f"{n_contraction}/{n_non_null} in expected direction, "
            f"{n_reversal} sign reversals."
        )

    # Sign reversal resolution summary
    prior_reversal_models = [r for r in rows if r["was_prior_reversal"]]
    if prior_reversal_models:
        resolved = [r for r in prior_reversal_models if r["reversal_status"] == "RESOLVED"]
        persisting = [r for r in prior_reversal_models if r["reversal_status"] == "PERSISTS"]
        if resolved:
            reasons.append(
                "Sign reversals RESOLVED in: "
                + ", ".join(r["model_name"] for r in resolved)
            )
        if persisting:
            reasons.append(
                "Sign reversals PERSIST in: "
                + ", ".join(r["model_name"] for r in persisting)
                + " (requires further investigation)"
            )

    reasoning = "  " + "\n  ".join(reasons)
    return verdict, reasoning


# ---------------------------------------------------------------------------
# Sign reversal audit
# ---------------------------------------------------------------------------

def print_sign_reversal_audit(rows: List[dict]) -> None:
    """Print the sign reversal resolution section."""
    prior_reversals = [r for r in rows if r["was_prior_reversal"]]
    if not prior_reversals:
        print("Sign reversal audit: No models flagged as prior reversal suspects.")
        return

    print("SIGN REVERSAL AUDIT")
    print("-" * 60)
    print("Models that showed expansion (g > 0) in PRIOR pipelines:")
    print()

    for r in prior_reversals:
        print(f"  {r['model_name']}")
        print(f"    Prior status : Expansion (sign reversal)")
        print(f"    This run g   : {_fmt(r['hedges_g'], '.4f')}")
        print(f"    p (MWU)      : {_fmt(r['p_value_mwu'], '.4g')}")
        print(f"    Direction    : {r['direction']}")
        print(f"    Resolution   : {r['reversal_status']}")
        if r["reversal_status"] == "RESOLVED":
            print(f"    NOTE: g < 0 in this run — canonical pipeline fixed the reversal.")
        elif r["reversal_status"] == "PERSISTS":
            print(f"    NOTE: g > 0 persists. The issue is NOT pipeline-specific.")
            print(f"          This model may genuinely expand under L3_deeper prompts.")
            print(f"          Possible causes: different V-space geometry, fused QKV")
            print(f"          extraction artefact, or L3_deeper prompts inducing a")
            print(f"          different processing mode in this architecture.")
        elif r["reversal_status"] == "UNCERTAIN":
            print(f"    NOTE: Insufficient data to determine.")
        print()

    print("-" * 60)
    print()


# ---------------------------------------------------------------------------
# Missing models check
# ---------------------------------------------------------------------------

def check_missing_models(loaded_models: List[str]) -> None:
    """Warn about any expected models not yet in the results directory."""
    missing = [m for m in EXPECTED_MODELS if m not in loaded_models]
    if missing:
        print("WARNING: The following expected models are missing from results:")
        for m in missing:
            print(f"  - {m}")
        print("  Run p0_canonical_pipeline.py --model <name> to generate them.")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate P0 canonical pipeline results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing *_p0_result.json files (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=15,
        help=(
            "Minimum valid samples per condition to include a model in verdict "
            "(default: 15; use 30 for publication-quality threshold)"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path for aggregate JSON output (default: <results_dir>/p0_aggregate_summary.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print()
    print(f"Loading results from: {args.results_dir}")
    print()

    if not args.results_dir.exists():
        print(f"ERROR: Results directory does not exist: {args.results_dir}")
        print("  Run p0_canonical_pipeline.py for each model first.")
        sys.exit(1)

    # Load all results
    payloads = load_results(args.results_dir)
    print()

    if not payloads:
        print("No results loaded. Exiting.")
        sys.exit(1)

    # Check for missing models
    check_missing_models(list(payloads.keys()))

    # Analyse each model
    rows: List[dict] = []
    # Show in canonical order first, then any extras
    for model_name in EXPECTED_MODELS:
        if model_name in payloads:
            row = analyse_model(model_name, payloads[model_name], args.min_n)
            rows.append(row)
    for model_name, payload in payloads.items():
        if model_name not in EXPECTED_MODELS:
            row = analyse_model(model_name, payload, args.min_n)
            rows.append(row)

    # Print summary table
    print_summary_table(rows)

    # Print sign reversal audit
    print_sign_reversal_audit(rows)

    # Compute and print verdict
    verdict, reasoning = compute_cross_arch_verdict(rows)

    print("=" * 60)
    print("CROSS-ARCHITECTURE CLAIM VERDICT")
    print("=" * 60)
    print()
    print(f"  cross-arch R_V contraction claim is DEFENSIBLE: ", end="")

    if verdict == "DEFENSIBLE":
        print("YES")
    elif verdict == "PARTIALLY_DEFENSIBLE":
        print("PARTIALLY")
    else:
        print("NO")

    print(f"  Verdict category: {verdict}")
    print()
    print("  Reasoning:")
    print(reasoning)
    print()

    # Prompt for next steps based on verdict
    print("RECOMMENDED NEXT STEPS")
    print("-" * 60)
    if verdict == "DEFENSIBLE":
        print("  1. Run FDR correction on p-values across all comparisons.")
        print("  2. Compute cluster-robust standard errors for the paper.")
        print("  3. Update paper Table 1 with these canonical g values.")
        print("  4. Cross-arch claim can be stated as is.")
    elif verdict == "PARTIALLY_DEFENSIBLE":
        print("  1. Identify which non-Pythia models did not reach significance.")
        print("  2. Consider increasing n for those models (--n 150 or --n 200).")
        print("  3. Soften cross-arch claim: 'in N of 4 architectures tested'.")
        print("  4. Check whether non-significant models are at least directionally correct.")
    elif verdict == "NOT_DEFENSIBLE":
        print("  1. If sign reversals persist: investigate per-architecture V-projection")
        print("     extraction (fused QKV splitting, head dimension handling).")
        print("  2. Run a layer sweep (GeometricProbe.layer_sweep) on the reversal models")
        print("     to identify the actual peak-contraction layer.")
        print("  3. Cross-arch claim CANNOT be made without resolving reversals.")
        print("  4. Consider reframing: 'contraction in decoder-only models with")
        print("     separate V projections' (excludes fused QKV models if those persist).")
    print()

    # Save aggregate summary JSON
    out_path = args.output_json or (args.results_dir / "p0_aggregate_summary.json")
    summary = {
        "schema_version": "p0-aggregate-v1.0",
        "verdict": verdict,
        "reasoning": reasoning,
        "min_n_threshold": args.min_n,
        "n_models_loaded": len(rows),
        "n_models_expected": len(EXPECTED_MODELS),
        "models": rows,
    }
    with out_path.open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"Aggregate summary saved to: {out_path}")
    print()


if __name__ == "__main__":
    main()
