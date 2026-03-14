#!/usr/bin/env python3
"""
SCALING GAP SWEEP (E1.3)
========================

Fills the 2.8B–7B scaling gap with mid-range models.
Measures R_V (recursive vs baseline) with Cohen's d per model.

Models:
  Gemma-2-2B (2.6B), Qwen2.5-3B (3B), Llama-3.2-3B (3.2B),
  Phi-3-mini-4k (3.8B), Pythia-6.9B (6.9B retry)

Uses subprocess isolation per model to avoid OOM contamination.
Requires ≥50GB free disk for model downloads.

Output: results/scaling_gap/<model>_result.json + summary

Usage:
    python3 scripts/scaling_gap_sweep.py --device cuda
    python3 scripts/scaling_gap_sweep.py --device cuda --models gemma-2-2b qwen2.5-3b
    python3 scripts/scaling_gap_sweep.py --device cuda --single-model gemma-2-2b
"""

import sys
import json
import argparse
import gc
import time
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.loader import PromptLoader
from geometric_lens.probe import GeometricProbe


# ── Model definitions ────────────────────────────────────────────────────────

MODELS = {
    "gemma-2-2b": {
        "name": "google/gemma-2-2b",
        "params": 2_600_000_000,
        "attn_impl": "eager",
    },
    "qwen2.5-3b": {
        "name": "Qwen/Qwen2.5-3B",
        "params": 3_000_000_000,
        "attn_impl": "eager",
    },
    "llama-3.2-3b": {
        "name": "meta-llama/Llama-3.2-3B",
        "params": 3_200_000_000,
        "attn_impl": "eager",
    },
    "phi-3-mini-4k": {
        "name": "microsoft/Phi-3-mini-4k-instruct",
        "params": 3_800_000_000,
        "attn_impl": "eager",
    },
    "pythia-6.9b": {
        "name": "EleutherAI/pythia-6.9b",
        "params": 6_900_000_000,
        "attn_impl": "eager",
    },
}


# ── Prompt bank (loaded from prompts/bank.json) ──────────────────────────────
_loader = PromptLoader()
RECURSIVE_PROMPTS = (_loader.get_by_group("L3_deeper") + _loader.get_by_group("L4_full")
                     + _loader.get_by_group("L5_refined") + _loader.get_by_group("L1_hint"))
BASELINE_PROMPTS = (_loader.get_by_group("baseline_factual") + _loader.get_by_group("baseline_math")
                    + _loader.get_by_group("baseline_creative") + _loader.get_by_group("baseline_personal"))


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std < 1e-10:
        return float("nan")
    return (np.mean(a) - np.mean(b)) / pooled_std


def bootstrap_ci(a, b, stat_fn=cohens_d, n_boot=5000, ci=0.95):
    """Bootstrap confidence interval for effect size."""
    rng = np.random.default_rng(42)
    boot_stats = []
    for _ in range(n_boot):
        a_boot = rng.choice(a, size=len(a), replace=True)
        b_boot = rng.choice(b, size=len(b), replace=True)
        boot_stats.append(stat_fn(a_boot, b_boot))
    boot_stats = np.array([s for s in boot_stats if not np.isnan(s)])
    if len(boot_stats) < 100:
        return (float("nan"), float("nan"))
    alpha = (1 - ci) / 2
    return (float(np.percentile(boot_stats, 100 * alpha)),
            float(np.percentile(boot_stats, 100 * (1 - alpha))))


def run_single_model(model_key, args):
    """Run R_V measurement for a single model."""
    model_cfg = MODELS[model_key]
    out_dir = Path("results/scaling_gap")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_key} ({model_cfg['params']/1e9:.1f}B params)")
    print(f"{'=' * 70}")

    t0 = time.time()
    try:
        probe = GeometricProbe(
            model_name=model_cfg["name"],
            device=args.device,
            attn_implementation=model_cfg["attn_impl"],
        )
    except Exception as e:
        print(f"  FAILED to load: {e}")
        result = {"model": model_key, "error": str(e)}
        with open(out_dir / f"{model_key}_result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        return

    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  Spec: layers={probe.spec.num_layers}, heads={probe.spec.num_heads}, "
          f"early={probe.early_layer}, late={probe.late_layer}")

    # Measure R_V
    n = args.n_prompts
    rec_prompts = RECURSIVE_PROMPTS[:n]
    bas_prompts = BASELINE_PROMPTS[:n]

    print(f"\n  Measuring R_V ({n} recursive + {n} baseline)...")
    rec_results = probe.measure_batch(rec_prompts, metrics=["rv", "spectral"], progress=True)
    bas_results = probe.measure_batch(bas_prompts, metrics=["rv", "spectral"], progress=True)

    rec_rvs = [r.rv for r in rec_results if not np.isnan(r.rv)]
    bas_rvs = [r.rv for r in bas_results if not np.isnan(r.rv)]

    if rec_rvs and bas_rvs:
        d = cohens_d(rec_rvs, bas_rvs)
        u, p = stats.mannwhitneyu(rec_rvs, bas_rvs, alternative="two-sided")
        ci_lo, ci_hi = bootstrap_ci(rec_rvs, bas_rvs)
    else:
        d, p, ci_lo, ci_hi = float("nan"), float("nan"), float("nan"), float("nan")

    print(f"  R_V recursive: {np.mean(rec_rvs):.3f} ± {np.std(rec_rvs):.3f} (n={len(rec_rvs)})")
    print(f"  R_V baseline:  {np.mean(bas_rvs):.3f} ± {np.std(bas_rvs):.3f} (n={len(bas_rvs)})")
    print(f"  Cohen's d: {d:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], p={p:.6f}")

    # Spectral stats
    rec_eff_rank = [r.spectral_late.effective_rank for r in rec_results
                    if r.spectral_late and not np.isnan(r.spectral_late.effective_rank)]
    bas_eff_rank = [r.spectral_late.effective_rank for r in bas_results
                    if r.spectral_late and not np.isnan(r.spectral_late.effective_rank)]

    model_result = {
        "model": model_key,
        "model_name": model_cfg["name"],
        "params": model_cfg["params"],
        "early_layer": probe.early_layer,
        "late_layer": probe.late_layer,
        "n_layers": probe.spec.num_layers,
        "n_heads": probe.spec.num_heads,
        "rv_recursive_mean": float(np.mean(rec_rvs)) if rec_rvs else float("nan"),
        "rv_recursive_std": float(np.std(rec_rvs)) if rec_rvs else float("nan"),
        "rv_baseline_mean": float(np.mean(bas_rvs)) if bas_rvs else float("nan"),
        "rv_baseline_std": float(np.std(bas_rvs)) if bas_rvs else float("nan"),
        "cohens_d": d,
        "ci_95_lo": ci_lo,
        "ci_95_hi": ci_hi,
        "p_value": p,
        "n_recursive": len(rec_rvs),
        "n_baseline": len(bas_rvs),
        "eff_rank_recursive_mean": float(np.mean(rec_eff_rank)) if rec_eff_rank else float("nan"),
        "eff_rank_baseline_mean": float(np.mean(bas_eff_rank)) if bas_eff_rank else float("nan"),
        "load_time_s": load_time,
        "recursive_rv_values": [float(v) for v in rec_rvs],
        "baseline_rv_values": [float(v) for v in bas_rvs],
    }

    result_path = out_dir / f"{model_key}_result.json"
    with open(result_path, "w") as f:
        json.dump(model_result, f, indent=2, default=str)
    print(f"  Saved: {result_path}")

    # Cleanup
    del probe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_sweep(args):
    """Run scaling gap sweep with subprocess isolation."""
    print("=" * 70)
    print("SCALING GAP SWEEP (E1.3)")
    print("=" * 70)

    out_dir = Path("results/scaling_gap")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.models:
        model_keys = [k for k in args.models if k in MODELS]
    else:
        model_keys = list(MODELS.keys())

    print(f"Models to sweep: {model_keys}")

    if args.single_model:
        run_single_model(args.single_model, args)
        return

    # Orchestrator mode: subprocess per model
    for model_key in model_keys:
        print(f"\n>>> Launching subprocess for {model_key}...")
        cmd = [
            sys.executable, __file__,
            "--device", args.device,
            "--single-model", model_key,
            "--n-prompts", str(args.n_prompts),
        ]
        result = subprocess.run(cmd, capture_output=False, timeout=2400)  # 40 min max
        if result.returncode != 0:
            print(f"  [WARN] Subprocess for {model_key} exited with code {result.returncode}")
            error_path = out_dir / f"{model_key}_result.json"
            if not error_path.exists():
                with open(error_path, "w") as f:
                    json.dump({"model": model_key, "error": f"subprocess exit {result.returncode}"}, f)

    # Collect results and produce summary
    print("\n" + "=" * 70)
    print("SCALING GAP SUMMARY")
    print("=" * 70)

    all_results = {}
    for model_key in model_keys:
        result_path = out_dir / f"{model_key}_result.json"
        if result_path.exists():
            with open(result_path) as f:
                all_results[model_key] = json.load(f)
        else:
            all_results[model_key] = {"error": "no result file"}

    # Also load existing scaling_law results if available
    scaling_law_dir = Path("results/scaling_law")
    if scaling_law_dir.exists():
        for p in scaling_law_dir.glob("*_result.json"):
            key = p.stem.replace("_result", "")
            if key not in all_results:
                with open(p) as f:
                    data = json.load(f)
                    if "error" not in data:
                        all_results[key] = data

    print(f"\n{'Model':<18} {'Params':>10} {'d':>8} {'CI_lo':>8} {'CI_hi':>8} {'p':>12} {'RV_rec':>8} {'RV_bas':>8}")
    print("-" * 90)
    for key in sorted(all_results.keys(), key=lambda k: all_results[k].get("params", 0)):
        r = all_results[key]
        if "error" in r:
            print(f"{key:<18} {'ERROR':>10}  {r.get('error', '')[:40]}")
            continue
        print(f"{key:<18} "
              f"{r['params']/1e9:>9.1f}B "
              f"{r.get('cohens_d', float('nan')):>8.3f} "
              f"{r.get('ci_95_lo', float('nan')):>8.3f} "
              f"{r.get('ci_95_hi', float('nan')):>8.3f} "
              f"{r.get('p_value', float('nan')):>12.6f} "
              f"{r.get('rv_recursive_mean', float('nan')):>8.3f} "
              f"{r.get('rv_baseline_mean', float('nan')):>8.3f}")

    # Fit scaling model on ALL available data points
    valid = [(r["params"], r["cohens_d"]) for r in all_results.values()
             if "error" not in r and not np.isnan(r.get("cohens_d", float("nan")))]

    scaling_fit = {}
    if len(valid) >= 3:
        log_params = np.log10([p for p, d in valid])
        effect_sizes = np.array([d for p, d in valid])
        slope, intercept, r_val, p_val, std_err = stats.linregress(log_params, effect_sizes)
        print(f"\n  Scaling fit: d = {slope:.3f} × log10(params) + {intercept:.3f}")
        print(f"  R² = {r_val**2:.3f}, p = {p_val:.4f}")
        scaling_fit = {
            "slope": slope, "intercept": intercept,
            "r_squared": r_val**2, "p_value": p_val,
            "n_points": len(valid),
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "experiment": "E1.3_scaling_gap",
        "models": all_results,
        "scaling_fit": scaling_fit,
    }
    summary_path = out_dir / f"scaling_gap_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Summary saved: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scaling Gap Sweep (E1.3)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Specific models to run (e.g., gemma-2-2b qwen2.5-3b)")
    parser.add_argument("--single-model", default=None,
                        help="Run a single model directly (subprocess isolation)")
    parser.add_argument("--n-prompts", type=int, default=40, help="Prompts per condition")
    args = parser.parse_args()
    run_sweep(args)
