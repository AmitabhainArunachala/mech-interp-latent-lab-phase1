#!/usr/bin/env python3
"""
P0 Canonical Pipeline — R_V Paper (COLM 2026)
==============================================

Single canonical script that runs ONE model through the standardised
R_V measurement pipeline.  Resolves the sign-reversal problem observed
between earlier pipelines by enforcing:

  - bfloat16 precision (MANDATORY — float16 causes NaN in deep OPT/GPT-2 layers)
  - Architecture-specific L_early / L_late via the canonical ModelSpec registry
  - Frozen prompt subset resolved from prompts/bank.json
  - Fixed token window W=16
  - Hedges' g effect size with bootstrap CI (not raw Cohen's d)
  - Mann-Whitney U as non-parametric companion test

Run one model at a time; submit all five in parallel on RunPod.

Usage
-----
    python p0_canonical_pipeline.py --model mistralai/Mistral-7B-v0.1
    python p0_canonical_pipeline.py --model facebook/opt-6.7b --n 50
    python p0_canonical_pipeline.py --model openai-community/gpt2-xl
    python p0_canonical_pipeline.py --model Qwen/Qwen2.5-7B-Instruct
    python p0_canonical_pipeline.py --model EleutherAI/pythia-1.4b

Output
------
    ~/mech-interp-latent-lab-phase1/results/p0_canonical/{model_safe_name}_p0_result.json

Sign convention
---------------
    Hedges' g = (mean_selfref - mean_baseline) / pooled_SD
    g < 0  →  self-referential prompts produce LOWER R_V  (CONTRACTION — expected)
    g > 0  →  self-referential prompts produce HIGHER R_V  (EXPANSION — sign reversal)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
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
log = logging.getLogger("p0")

# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent  # mech-interp-latent-lab-phase1/
PROMPT_BANK_PATH = REPO_ROOT / "prompts" / "bank.json"
PROMPT_SUBSET_PATH = REPO_ROOT / "prompts" / "subsets" / "mistral_hardening_v1.json"
RESULTS_DIR = REPO_ROOT / "results" / "p0_canonical"

# Add repo root to sys.path so geometric_lens is importable without install
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompts.loader import PromptLoader
from prompts.subsets import FrozenPromptSubset
from src.utils.canonical_registry import CANONICAL_REGISTRY_PATH, get_canonical_model_spec
from src.utils.run_metadata import get_git_commit, get_hardware_info

# Models where bfloat16 is supported natively on the target GPU family.
# GPT-2-XL was originally a float32 model; bfloat16 cast is safe on A100/H100
# but may silently degrade on older (V100) hardware.  We enforce it regardless
# because float16 causes NaN at deep layers — see MI_AGENT_TO_CODEX_RV_ANSWERS §1.
BFLOAT16_REQUIRED = True


def select_prompts(
    subset: FrozenPromptSubset,
    n: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Draw up to n prompts from the frozen core measurement tier.

    Self-referential prompts are identified by pillar == "dose_response".
    Baseline prompts are identified by pillar == "baselines".

    Sampling is deterministic (sorted key order) so results are reproducible
    without a fixed random seed.

    Args:
        subset: Frozen prompt subset resolved against prompts/bank.json.
        n: Maximum prompts per condition.

    Returns:
        (selfref_items, baseline_items) where each item is (prompt_id, text).
    """
    selfref_items: List[Tuple[str, str]] = []
    baseline_items: List[Tuple[str, str]] = []

    for prompt_id, entry in subset.get_records_for_tier("core_measurement"):
        pillar = entry.get("pillar", "")
        text = entry.get("text", "")
        if not text:
            continue
        if pillar == "dose_response":
            selfref_items.append((prompt_id, text))
        elif pillar == "baselines":
            baseline_items.append((prompt_id, text))

    # Cap at n each
    selfref_items = selfref_items[:n]
    baseline_items = baseline_items[:n]

    log.info(
        "Prompt selection from frozen subset: %d self-ref, %d baseline",
        len(selfref_items),
        len(baseline_items),
    )
    return selfref_items, baseline_items


# ---------------------------------------------------------------------------
# Robust Hedges' g  (copied verbatim from MI_AGENT_TO_CODEX_RV_ANSWERS §6)
# ---------------------------------------------------------------------------

def robust_effect_size(
    x: np.ndarray,
    y: np.ndarray,
    variance_floor: float = 0.01,
    clamp: float = 10.0,
    n_bootstrap: int = 5000,
    rng_seed: int = 42,
) -> dict:
    """
    Compute Hedges' g with variance floor and bootstrap CI.

    Sign convention: g = (mean_x - mean_y) / pooled_SD
    Caller passes (selfref_rv, baseline_rv), so g < 0 means contraction.

    Args:
        x: Self-referential R_V values.
        y: Baseline R_V values.
        variance_floor: Prevents d -> inf when one group has near-zero variance.
        clamp: |g| values beyond this are flagged and clamped.
        n_bootstrap: Number of bootstrap resamples for 95% CI.
        rng_seed: Seed for reproducibility.

    Returns:
        dict matching the production schema from MI_AGENT answers §6.
    """
    from scipy import stats

    rng = np.random.default_rng(rng_seed)

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    warn_list: List[str] = []
    nx, ny = len(x), len(y)

    if nx < 3 or ny < 3:
        return {
            "effect_size": None,
            "method": "insufficient_n",
            "warnings": [f"n_x={nx}, n_y={ny}, need >=3 each"],
            "raw_means": [
                float(np.mean(x)) if nx > 0 else None,
                float(np.mean(y)) if ny > 0 else None,
            ],
        }

    mx, my = float(np.mean(x)), float(np.mean(y))
    sx, sy = float(np.std(x, ddof=1)), float(np.std(y, ddof=1))

    sp = float(
        np.sqrt(((nx - 1) * sx ** 2 + (ny - 1) * sy ** 2) / (nx + ny - 2))
    )
    if sp < variance_floor:
        warn_list.append("variance_floor_applied")
        sp = variance_floor

    d = (mx - my) / sp
    # Hedges' g small-sample correction
    correction = 1.0 - 3.0 / (4.0 * (nx + ny) - 9.0)
    g = d * correction

    # Bootstrap 95% CI
    boot_gs: List[float] = []
    for _ in range(n_bootstrap):
        bx = rng.choice(x, size=nx, replace=True)
        by = rng.choice(y, size=ny, replace=True)
        bsp = float(
            np.sqrt(
                ((nx - 1) * np.std(bx, ddof=1) ** 2 + (ny - 1) * np.std(by, ddof=1) ** 2)
                / (nx + ny - 2)
            )
        )
        bsp = max(bsp, variance_floor)
        bd = (float(np.mean(bx)) - float(np.mean(by))) / bsp
        boot_gs.append(bd * correction)

    ci_lower, ci_upper = float(np.percentile(boot_gs, 2.5)), float(np.percentile(boot_gs, 97.5))

    if abs(g) > clamp:
        warn_list.append(f"clamped_from_{g:.4f}")
        g = float(np.sign(g) * clamp)

    # Welch's t-test (parametric companion)
    t_stat, p_welch = stats.ttest_ind(x, y, equal_var=False)

    # Mann-Whitney U (non-parametric; primary test for this script)
    u_stat, p_mwu = stats.mannwhitneyu(x, y, alternative="two-sided")

    return {
        "effect_size": round(float(g), 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "p_value_welch": round(float(p_welch), 8),
        "p_value_mwu": round(float(p_mwu), 8),
        "u_stat": round(float(u_stat), 4),
        "method": "hedges_g_bootstrap",
        "n": [nx, ny],
        "raw_means": [round(mx, 6), round(my, 6)],
        "raw_sds": [round(sx, 6), round(sy, 6)],
        "variance_floor": variance_floor,
        "warnings": warn_list,
    }


# ---------------------------------------------------------------------------
# Core measurement loop
# ---------------------------------------------------------------------------

def measure_prompts(
    probe,  # GeometricProbe instance
    prompt_items: List[Tuple[str, str]],
    condition_label: str,
) -> Tuple[List[Optional[float]], List[Dict[str, Any]]]:
    """
    Run each text through the probe and return a list of R_V values.

    OOM errors are caught per-prompt: that prompt is logged and skipped
    (returns None in its slot).  This prevents a single bad prompt from
    aborting the entire run.

    Args:
        probe: Initialised GeometricProbe.
        prompt_items: Sequence of (prompt_id, prompt_text) tuples.
        condition_label: "selfref" or "baseline" (for logging only).

    Returns:
        Tuple of:
        - List of R_V floats (or None on OOM/error).
        - Per-sample records for CSV export.
    """
    try:
        from tqdm import tqdm
        iterator = tqdm(prompt_items, desc=condition_label, unit="prompt", ncols=90)
    except ImportError:
        log.warning("tqdm not installed — progress will not display per-prompt.")
        iterator = prompt_items  # type: ignore[assignment]

    rv_values: List[Optional[float]] = []
    per_sample_rows: List[Dict[str, Any]] = []
    oom_count = 0
    nan_count = 0

    for i, (prompt_id, text) in enumerate(iterator):
        try:
            result = probe.measure(text, metrics=["rv"])
            rv = result.rv
            if rv is not None and not (isinstance(rv, float) and np.isnan(rv)):
                rv_values.append(float(rv))
                per_sample_rows.append(
                    {
                        "condition": condition_label,
                        "prompt_id": prompt_id,
                        "rv": round(float(rv), 6),
                        "status": "ok",
                    }
                )
            else:
                nan_count += 1
                rv_values.append(None)
                per_sample_rows.append(
                    {
                        "condition": condition_label,
                        "prompt_id": prompt_id,
                        "rv": "",
                        "status": "nan",
                    }
                )
                log.debug("[%s] prompt %d returned NaN R_V", condition_label, i)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() or "CUDA out of memory" in str(exc):
                oom_count += 1
                rv_values.append(None)
                per_sample_rows.append(
                    {
                        "condition": condition_label,
                        "prompt_id": prompt_id,
                        "rv": "",
                        "status": "oom",
                    }
                )
                log.warning("[%s] OOM on prompt %d — skipping", condition_label, i)
                # Free cache and continue; do NOT re-raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                # Non-OOM RuntimeError: log and skip
                rv_values.append(None)
                per_sample_rows.append(
                    {
                        "condition": condition_label,
                        "prompt_id": prompt_id,
                        "rv": "",
                        "status": f"runtime_error:{type(exc).__name__}",
                    }
                )
                log.warning("[%s] RuntimeError on prompt %d: %s", condition_label, i, exc)
        except Exception as exc:
            rv_values.append(None)
            per_sample_rows.append(
                {
                    "condition": condition_label,
                    "prompt_id": prompt_id,
                    "rv": "",
                    "status": f"error:{type(exc).__name__}",
                }
            )
            log.warning("[%s] Unexpected error on prompt %d: %s", condition_label, i, exc)

    valid_count = sum(1 for v in rv_values if v is not None)
    log.info(
        "[%s] Done: %d/%d valid, %d NaN, %d OOM",
        condition_label, valid_count, len(prompt_items), nan_count, oom_count,
    )
    return rv_values, per_sample_rows


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model_and_probe(model_name: str, device: str, arch_cfg: dict):
    """
    Load model + tokenizer and return an initialised GeometricProbe.

    Enforces bfloat16.  Uses attn_implementation="eager" so that attention
    pattern hooks work (flash-attention bypasses the hook registration points).

    Args:
        model_name: HuggingFace model identifier.
        device: "cuda" or "cpu".
        arch_cfg: Dict with early_layer, late_layer from ARCH_LAYER_MAP.

    Returns:
        GeometricProbe instance ready for .measure() calls.
    """
    from geometric_lens.probe import GeometricProbe

    log.info("Loading model: %s", model_name)
    log.info("  device=%s  dtype=bfloat16  early=%d  late=%d",
             device, arch_cfg["early_layer"], arch_cfg["late_layer"])
    log.info("  attn_implementation=eager (required for hook registration)")

    # OPT models in some HF versions emit a spurious UserWarning about
    # positional embeddings being too long; suppress it.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        probe = GeometricProbe(
            model_name=model_name,
            device=device,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            window=16,
            max_length=512,
            early_layer=arch_cfg["early_layer"],
            late_layer=arch_cfg["late_layer"],
        )

    log.info("Model loaded. Spec: %s", probe.spec)
    return probe


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------

def build_result_payload(
    model_name: str,
    arch_cfg: dict,
    selfref_rv: List[Optional[float]],
    baseline_rv: List[Optional[float]],
    selfref_prompt_ids: List[str],
    baseline_prompt_ids: List[str],
    n_requested: int,
    elapsed_seconds: float,
    device: str,
    prompt_contract: Dict[str, Any],
) -> dict:
    """
    Assemble the final JSON payload from raw R_V lists and effect-size stats.

    Args:
        model_name: Model identifier string.
        arch_cfg: Architecture config dict.
        selfref_rv: Raw R_V values (with Nones) for self-referential condition.
        baseline_rv: Raw R_V values (with Nones) for baseline condition.
        n_requested: Prompts requested per condition.
        elapsed_seconds: Wall time for the full run.
        device: "cuda" or "cpu".

    Returns:
        Flat dict ready for json.dumps.
    """
    # Filter to valid (non-None, non-NaN) floats
    sr_valid = [v for v in selfref_rv if v is not None and not np.isnan(v)]
    bl_valid = [v for v in baseline_rv if v is not None and not np.isnan(v)]

    sr_arr = np.array(sr_valid, dtype=float)
    bl_arr = np.array(bl_valid, dtype=float)

    stats = robust_effect_size(sr_arr, bl_arr)

    # Determine direction label
    g = stats.get("effect_size")
    if g is None:
        direction = "undetermined"
        sign_label = "undetermined"
    elif g < -0.2:
        direction = "contraction"
        sign_label = "EXPECTED (g < 0 = self-ref has lower R_V)"
    elif g > 0.2:
        direction = "expansion"
        sign_label = "SIGN REVERSAL (g > 0 = self-ref has higher R_V)"
    else:
        direction = "null"
        sign_label = "NULL (|g| < 0.2)"

    return {
        # Identity
        "schema_version": "p0-canonical-v1.0",
        "model_name": model_name,
        "measurement_version": "rv-1.0.0-bfloat16-w16",
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "dtype": "bfloat16",

        # Architecture
        "num_layers": arch_cfg["num_layers"],
        "early_layer": arch_cfg["early_layer"],
        "late_layer": arch_cfg["late_layer"],
        "depth_fraction_early": round(arch_cfg["early_layer"] / arch_cfg["num_layers"], 4),
        "depth_fraction_late": round(arch_cfg["late_layer"] / arch_cfg["num_layers"], 4),
        "window_size": 16,

        # Sample sizes
        "n_requested_per_condition": n_requested,
        "n_selfref_attempted": len(selfref_rv),
        "n_selfref_valid": len(sr_valid),
        "n_baseline_attempted": len(baseline_rv),
        "n_baseline_valid": len(bl_valid),

        # Condition summaries
        "selfref_rv_mean": round(float(np.mean(sr_arr)), 6) if len(sr_arr) > 0 else None,
        "selfref_rv_std": round(float(np.std(sr_arr, ddof=1)), 6) if len(sr_arr) > 1 else None,
        "selfref_rv_median": round(float(np.median(sr_arr)), 6) if len(sr_arr) > 0 else None,
        "baseline_rv_mean": round(float(np.mean(bl_arr)), 6) if len(bl_arr) > 0 else None,
        "baseline_rv_std": round(float(np.std(bl_arr, ddof=1)), 6) if len(bl_arr) > 1 else None,
        "baseline_rv_median": round(float(np.median(bl_arr)), 6) if len(bl_arr) > 0 else None,

        # Effect size (primary)
        "hedges_g": stats.get("effect_size"),
        "hedges_g_ci_lower": stats.get("ci_lower"),
        "hedges_g_ci_upper": stats.get("ci_upper"),
        "p_value_mwu": stats.get("p_value_mwu"),
        "p_value_welch": stats.get("p_value_welch"),
        "u_stat": stats.get("u_stat"),
        "effect_size_method": stats.get("method"),
        "effect_size_n": stats.get("n"),
        "effect_size_warnings": stats.get("warnings", []),

        # Interpretation
        "direction": direction,
        "sign_label": sign_label,
        "significant_p05_mwu": (
            bool(stats.get("p_value_mwu", 1.0) < 0.05)
            if stats.get("p_value_mwu") is not None
            else None
        ),

        # Raw data (for downstream aggregation)
        "selfref_rv_values": [round(v, 6) for v in sr_valid],
        "baseline_rv_values": [round(v, 6) for v in bl_valid],
        "selfref_prompt_ids": selfref_prompt_ids,
        "baseline_prompt_ids": baseline_prompt_ids,

        # Provenance
        "prompt_bank": prompt_contract["prompt_bank_path"],
        "prompt_bank_version": prompt_contract["prompt_bank_version"],
        "prompt_subset_path": prompt_contract["prompt_subset_path"],
        "prompt_subset_name": prompt_contract["prompt_subset_name"],
        "prompt_subset_schema_version": prompt_contract["prompt_subset_schema_version"],
        "prompt_subset_tier": prompt_contract["prompt_subset_tier"],
        "canonical_registry_path": prompt_contract["canonical_registry_path"],
        "canonical_registry_schema_version": prompt_contract["canonical_registry_schema_version"],
        "elapsed_seconds": round(elapsed_seconds, 1),
    }


def build_config_payload(
    args: argparse.Namespace,
    arch_cfg: Dict[str, Any],
    device: str,
    prompt_contract: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": "p0-canonical-config-v1",
        "model_name": args.model,
        "device": device,
        "dtype": "bfloat16",
        "window_size": 16,
        "n_requested_per_condition": args.n,
        "prompt_bank_path": prompt_contract["prompt_bank_path"],
        "prompt_bank_version": prompt_contract["prompt_bank_version"],
        "prompt_subset_path": prompt_contract["prompt_subset_path"],
        "prompt_subset_name": prompt_contract["prompt_subset_name"],
        "prompt_subset_schema_version": prompt_contract["prompt_subset_schema_version"],
        "prompt_subset_tier": prompt_contract["prompt_subset_tier"],
        "canonical_registry_path": prompt_contract["canonical_registry_path"],
        "canonical_registry_schema_version": prompt_contract["canonical_registry_schema_version"],
        "num_layers": arch_cfg["num_layers"],
        "early_layer": arch_cfg["early_layer"],
        "late_layer": arch_cfg["late_layer"],
    }


def build_provenance_payload(
    prompt_contract: Dict[str, Any],
    metric_path: str,
) -> Dict[str, Any]:
    return {
        "schema_version": "p0-canonical-provenance-v1",
        "git_commit": get_git_commit(),
        "hardware": get_hardware_info(),
        "metric_path": metric_path,
        "prompt_bank_path": prompt_contract["prompt_bank_path"],
        "prompt_bank_version": prompt_contract["prompt_bank_version"],
        "prompt_subset_path": prompt_contract["prompt_subset_path"],
        "prompt_subset_name": prompt_contract["prompt_subset_name"],
        "prompt_subset_tier": prompt_contract["prompt_subset_tier"],
        "canonical_registry_path": prompt_contract["canonical_registry_path"],
        "canonical_registry_schema_version": prompt_contract["canonical_registry_schema_version"],
        "cwd": os.getcwd(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P0 canonical R_V pipeline (COLM 2026)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name (e.g. mistralai/Mistral-7B-v0.1)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Max prompts per condition (default: 100).  Use --n 20 for a quick smoke test.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device: 'cuda' or 'cpu'.  Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Output directory (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--prompt-subset",
        type=Path,
        default=PROMPT_SUBSET_PATH,
        help=f"Frozen prompt subset manifest (default: {PROMPT_SUBSET_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load prompts and model spec, then exit without running inference.",
    )
    return parser.parse_args()


def model_to_safe_name(model_name: str) -> str:
    """Convert 'org/model-name' to a filesystem-safe slug."""
    return model_name.replace("/", "__").replace(".", "-")


def display_value(value: Optional[float]) -> float:
    """Preserve legitimate 0.0 values in logs; only substitute NaN for None."""
    return float("nan") if value is None else float(value)


def main() -> None:
    args = parse_args()

    # ── Device ──────────────────────────────────────────────────────────────
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Auto-detected device: %s", device)

    if device == "cpu":
        log.warning(
            "Running on CPU.  Expect very slow inference (~120-300s/prompt for 7B models). "
            "This is fine for Pythia-1.4B smoke tests but impractical for 7B models."
        )

    # ── Architecture config ──────────────────────────────────────────────────
    try:
        arch_cfg = get_canonical_model_spec(args.model)
        log.info("Architecture config from canonical registry: %s", arch_cfg)
    except KeyError:
        log.warning(
            "Model '%s' not in canonical registry %s. Will fall back to GeometricProbe "
            "auto-detection. Verify early/late layers before using results for paper claims.",
            args.model,
            CANONICAL_REGISTRY_PATH,
        )
        arch_cfg = {
            "num_layers": -1,
            "early_layer": -1,
            "late_layer": -1,
            "name": args.model,
            "registry_path": str(CANONICAL_REGISTRY_PATH),
            "registry_schema_version": "unknown",
            "note": "auto-detected",
        }

    # ── Load frozen prompt subset ────────────────────────────────────────────
    log.info("Loading prompt bank from %s", PROMPT_BANK_PATH)
    loader = PromptLoader(PROMPT_BANK_PATH)
    log.info("Prompt bank loaded: %d entries (version=%s)", len(loader.prompts), loader.version)
    log.info("Loading frozen prompt subset from %s", args.prompt_subset)
    subset = FrozenPromptSubset.load(args.prompt_subset, loader=loader)

    selfref_items, baseline_items = select_prompts(subset, n=args.n)

    if len(selfref_items) == 0 or len(baseline_items) == 0:
        log.error(
            "Prompt selection returned 0 texts in one or more conditions.  "
            "Check the frozen subset manifest and prompts/bank.json."
        )
        sys.exit(1)

    selfref_prompt_ids = [prompt_id for prompt_id, _ in selfref_items]
    baseline_prompt_ids = [prompt_id for prompt_id, _ in baseline_items]
    prompt_contract = {
        "prompt_bank_path": str(loader.bank_path),
        "prompt_bank_version": loader.version,
        "prompt_subset_path": str(subset.manifest_path),
        "prompt_subset_name": subset.name,
        "prompt_subset_schema_version": subset.schema_version,
        "prompt_subset_tier": "core_measurement",
        "canonical_registry_path": str(CANONICAL_REGISTRY_PATH),
        "canonical_registry_schema_version": arch_cfg.get("registry_schema_version", "unknown"),
    }

    if args.dry_run:
        log.info(
            "DRY RUN: would measure %d self-ref + %d baseline prompts with %s",
            len(selfref_items), len(baseline_items), args.model,
        )
        log.info("  prompt_bank_version=%s", prompt_contract["prompt_bank_version"])
        log.info("  prompt_subset=%s", prompt_contract["prompt_subset_name"])
        log.info("Exiting without inference (--dry-run).")
        return

    # ── Load model ────────────────────────────────────────────────────────────
    t0 = time.monotonic()

    # When arch_cfg has -1 sentinels, GeometricProbe uses its own auto-detection.
    if arch_cfg["early_layer"] != -1:
        probe = load_model_and_probe(args.model, device, arch_cfg)
    else:
        from geometric_lens.probe import GeometricProbe
        probe = GeometricProbe(
            model_name=args.model,
            device=device,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            window=16,
            max_length=512,
        )
        # Update arch_cfg with auto-detected values for the output payload
        arch_cfg = {
            "num_layers": probe.spec.num_layers,
            "early_layer": probe.early_layer,
            "late_layer": probe.late_layer,
            "note": "auto-detected",
        }
        log.info("Auto-detected layers: early=%d  late=%d", probe.early_layer, probe.late_layer)

    # ── Measure ───────────────────────────────────────────────────────────────
    log.info("=== SELF-REFERENTIAL CONDITION (core_measurement, n=%d) ===", len(selfref_items))
    selfref_rv, selfref_rows = measure_prompts(probe, selfref_items, condition_label="selfref")

    log.info("=== BASELINE CONDITION (core_measurement, n=%d) ===", len(baseline_items))
    baseline_rv, baseline_rows = measure_prompts(probe, baseline_items, condition_label="baseline")

    elapsed = time.monotonic() - t0

    # ── Build and save result ─────────────────────────────────────────────────
    payload = build_result_payload(
        model_name=args.model,
        arch_cfg=arch_cfg,
        selfref_rv=selfref_rv,
        baseline_rv=baseline_rv,
        selfref_prompt_ids=selfref_prompt_ids,
        baseline_prompt_ids=baseline_prompt_ids,
        n_requested=args.n,
        elapsed_seconds=elapsed,
        device=device,
        prompt_contract=prompt_contract,
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_to_safe_name(args.model)
    out_path = out_dir / f"{safe_name}_p0_result.json"
    csv_path = out_dir / f"{safe_name}_per_sample.csv"
    config_path = out_dir / f"{safe_name}_config.json"
    provenance_path = out_dir / f"{safe_name}_provenance.json"
    run_id = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_dir = out_dir / "runs" / f"{run_id}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2)

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["condition", "prompt_id", "rv", "status"])
        writer.writeheader()
        writer.writerows(selfref_rows + baseline_rows)

    config_payload = build_config_payload(args, arch_cfg, device, prompt_contract)
    provenance_payload = build_provenance_payload(
        prompt_contract=prompt_contract,
        metric_path="geometric_lens.metrics.participation_ratio",
    )

    for path, data in (
        (config_path, config_payload),
        (provenance_path, provenance_payload),
        (run_dir / "config.json", config_payload),
        (run_dir / "summary.json", payload),
        (run_dir / "provenance.json", provenance_payload),
    ):
        with path.open("w") as fh:
            json.dump(data, fh, indent=2)

    with (run_dir / "per_sample.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["condition", "prompt_id", "rv", "status"])
        writer.writeheader()
        writer.writerows(selfref_rows + baseline_rows)

    # ── Console summary ───────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("P0 RESULT SUMMARY: %s", args.model)
    log.info("  early_layer=%d  late_layer=%d  window=16  dtype=bfloat16",
             payload["early_layer"], payload["late_layer"])
    log.info("  n_selfref_valid=%d  n_baseline_valid=%d",
             payload["n_selfref_valid"], payload["n_baseline_valid"])
    log.info("  selfref  R_V = %.4f ± %.4f",
             display_value(payload["selfref_rv_mean"]),
             display_value(payload["selfref_rv_std"]))
    log.info("  baseline R_V = %.4f ± %.4f",
             display_value(payload["baseline_rv_mean"]),
             display_value(payload["baseline_rv_std"]))
    log.info("  Hedges' g = %.4f  [95%% CI %.4f, %.4f]",
             display_value(payload["hedges_g"]),
             display_value(payload["hedges_g_ci_lower"]),
             display_value(payload["hedges_g_ci_upper"]))
    log.info("  p (MWU)   = %.4g", display_value(payload["p_value_mwu"]))
    log.info("  direction = %s", payload["direction"].upper())
    log.info("  sign      = %s", payload["sign_label"])
    log.info("  prompt subset = %s", payload["prompt_subset_name"])
    log.info("  prompt bank   = %s", payload["prompt_bank_version"])
    log.info("  elapsed   = %.0f s", elapsed)
    log.info("  saved to  = %s", out_path)
    log.info("  per-sample= %s", csv_path)
    log.info("  config    = %s", config_path)
    log.info("  provenance= %s", provenance_path)
    log.info("  run dir   = %s", run_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
