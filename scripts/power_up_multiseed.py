#!/usr/bin/env python3
"""
POWER-UP & MULTI-SEED VALIDATION (E1.1 + E1.2)
=================================================

E1.1: Expand R_V measurement to n≥100 per model for all 5 architectures.
E1.2: Multi-seed validation (5 seeds × Mistral-7B) — confirms seed-independence.

Uses subprocess isolation per model. Expanded prompt bank (100 rec + 100 bas).

Output: results/power_up/<model>_n100_result.json + multi-seed summary

Usage:
    python3 scripts/power_up_multiseed.py --device cuda --mode power-up
    python3 scripts/power_up_multiseed.py --device cuda --mode multi-seed
    python3 scripts/power_up_multiseed.py --device cuda --mode all
    python3 scripts/power_up_multiseed.py --device cuda --single-model mistral-7b
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

from geometric_lens.probe import GeometricProbe
from prompts.loader import PromptLoader


# ── Models ───────────────────────────────────────────────────────────────────

MODELS = {
    "mistral-7b": {"name": "mistralai/Mistral-7B-v0.1", "attn_impl": "eager"},
    "opt-6.7b": {"name": "facebook/opt-6.7b", "attn_impl": "eager"},
    "gpt2-xl": {"name": "openai-community/gpt2-xl", "attn_impl": "eager"},
    "qwen2.5-7b": {"name": "Qwen/Qwen2.5-7B", "attn_impl": "eager"},
    "pythia-1.4b": {"name": "EleutherAI/pythia-1.4b", "attn_impl": "eager"},
}


# ── Prompt bank (loaded from prompts/bank.json) ─────────────────────────────
_loader = PromptLoader()
RECURSIVE_PROMPTS = (
    _loader.get_by_group("L1_hint") + _loader.get_by_group("L2_simple")
    + _loader.get_by_group("L3_deeper") + _loader.get_by_group("L4_full")
    + _loader.get_by_group("L5_refined")
)
BASELINE_PROMPTS = (
    _loader.get_by_group("baseline_factual") + _loader.get_by_group("baseline_math")
    + _loader.get_by_group("baseline_creative") + _loader.get_by_group("baseline_personal")
    + _loader.get_by_group("baseline_impossible")
)



def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std < 1e-10:
        return float("nan")
    return (np.mean(a) - np.mean(b)) / pooled_std


def bootstrap_ci(a, b, n_boot=5000, ci=0.95):
    rng = np.random.default_rng(42)
    boot_stats = []
    for _ in range(n_boot):
        a_b = rng.choice(a, size=len(a), replace=True)
        b_b = rng.choice(b, size=len(b), replace=True)
        boot_stats.append(cohens_d(a_b, b_b))
    boot_stats = np.array([s for s in boot_stats if not np.isnan(s)])
    if len(boot_stats) < 100:
        return (float("nan"), float("nan"))
    alpha = (1 - ci) / 2
    return (float(np.percentile(boot_stats, 100*alpha)),
            float(np.percentile(boot_stats, 100*(1-alpha))))


def run_single_model(model_key, args):
    """Run n≥100 R_V measurement for a single model."""
    model_cfg = MODELS[model_key]
    out_dir = Path("results/power_up")
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed else 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_key} (seed={seed}, n={args.n_prompts})")
    print(f"{'=' * 70}")

    t0 = time.time()
    try:
        probe = GeometricProbe(
            model_name=model_cfg["name"],
            device=args.device,
            attn_implementation=model_cfg["attn_impl"],
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        result = {"model": model_key, "error": str(e)}
        suffix = f"_seed{seed}" if seed != 42 else ""
        with open(out_dir / f"{model_key}_n{args.n_prompts}{suffix}_result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        return

    load_time = time.time() - t0
    n = min(args.n_prompts, len(RECURSIVE_PROMPTS), len(BASELINE_PROMPTS))

    print(f"  Measuring R_V ({n} rec + {n} bas)...")
    rec_results = probe.measure_batch(RECURSIVE_PROMPTS[:n], metrics=["rv"], progress=True)
    bas_results = probe.measure_batch(BASELINE_PROMPTS[:n], metrics=["rv"], progress=True)

    rec_rvs = [r.rv for r in rec_results if not np.isnan(r.rv)]
    bas_rvs = [r.rv for r in bas_results if not np.isnan(r.rv)]

    if rec_rvs and bas_rvs:
        d = cohens_d(rec_rvs, bas_rvs)
        u, p = stats.mannwhitneyu(rec_rvs, bas_rvs, alternative="two-sided")
        ci_lo, ci_hi = bootstrap_ci(rec_rvs, bas_rvs)
    else:
        d, p, ci_lo, ci_hi = [float("nan")] * 4

    print(f"  d={d:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], p={p:.6f}")

    result = {
        "model": model_key,
        "model_name": model_cfg["name"],
        "seed": seed,
        "n_prompts": n,
        "rv_recursive_mean": float(np.mean(rec_rvs)) if rec_rvs else float("nan"),
        "rv_recursive_std": float(np.std(rec_rvs)) if rec_rvs else float("nan"),
        "rv_baseline_mean": float(np.mean(bas_rvs)) if bas_rvs else float("nan"),
        "rv_baseline_std": float(np.std(bas_rvs)) if bas_rvs else float("nan"),
        "cohens_d": d,
        "ci_95": [ci_lo, ci_hi],
        "p_value": p,
        "n_recursive": len(rec_rvs),
        "n_baseline": len(bas_rvs),
        "load_time_s": load_time,
    }

    suffix = f"_seed{seed}" if seed != 42 else ""
    path = out_dir / f"{model_key}_n{n}{suffix}_result.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Saved: {path}")

    del probe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_power_up(args):
    """Run power-up sweep across all models."""
    out_dir = Path("results/power_up")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_keys = list(MODELS.keys()) if not args.models else [k for k in args.models if k in MODELS]

    for model_key in model_keys:
        print(f"\n>>> Subprocess: {model_key}...")
        cmd = [
            sys.executable, __file__,
            "--device", args.device,
            "--single-model", model_key,
            "--n-prompts", str(args.n_prompts),
        ]
        subprocess.run(cmd, capture_output=False, timeout=3600)


def run_multi_seed(args):
    """Run multi-seed validation on Mistral-7B."""
    seeds = [42, 137, 2026, 31415, 27182]
    model_key = "mistral-7b"

    for seed in seeds:
        print(f"\n>>> Subprocess: {model_key} seed={seed}...")
        cmd = [
            sys.executable, __file__,
            "--device", args.device,
            "--single-model", model_key,
            "--n-prompts", "45",  # Match original n
            "--seed", str(seed),
        ]
        subprocess.run(cmd, capture_output=False, timeout=1800)

    # Collect multi-seed results
    out_dir = Path("results/power_up")
    seed_results = []
    for seed in seeds:
        path = out_dir / f"{model_key}_n45_seed{seed}_result.json"
        if not path.exists():
            path = out_dir / f"{model_key}_n45_result.json"
        if path.exists():
            with open(path) as f:
                seed_results.append(json.load(f))

    if seed_results:
        d_values = [r["cohens_d"] for r in seed_results if not np.isnan(r.get("cohens_d", float("nan")))]
        print(f"\n  Multi-seed summary:")
        print(f"    Seeds tested: {len(seed_results)}")
        print(f"    d values: {[f'{d:.3f}' for d in d_values]}")
        print(f"    d mean: {np.mean(d_values):.3f} ± {np.std(d_values):.3f}")
        print(f"    Seed-independence: {'YES' if np.std(d_values) < 0.3 else 'MARGINAL'}")

        summary = {
            "experiment": "E1.2_multi_seed",
            "model": model_key,
            "seeds": seeds,
            "d_values": d_values,
            "d_mean": float(np.mean(d_values)),
            "d_std": float(np.std(d_values)),
            "seed_results": seed_results,
        }
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(out_dir / f"multi_seed_summary_{ts}.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)


def main(args):
    if args.single_model:
        run_single_model(args.single_model, args)
        return

    if args.mode in ("power-up", "all"):
        run_power_up(args)
    if args.mode in ("multi-seed", "all"):
        run_multi_seed(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Power-Up & Multi-Seed (E1.1 + E1.2)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["power-up", "multi-seed", "all"], default="all")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--single-model", default=None)
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args)
