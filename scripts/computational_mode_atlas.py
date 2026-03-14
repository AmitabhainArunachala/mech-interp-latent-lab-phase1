#!/usr/bin/env python3
"""
COMPUTATIONAL MODE ATLAS
========================

Maps geometric signatures across 10 distinct computational modes.
Tests whether different types of computation produce distinguishable
geometric fingerprints in V-projection space.

10 modes × 20 prompts = 200 measurements.
Full metric suite per prompt: R_V, PR, spectral stats, cosine, attn entropy, eigenvalue dominance.
Output: spectral fingerprint matrix + all 45 pairwise mode comparisons.

Usage:
    python3 scripts/computational_mode_atlas.py --device cuda
    python3 scripts/computational_mode_atlas.py --device cpu --model mistralai/Mistral-7B-v0.1
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from itertools import combinations

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.subsets import load_default_mode_atlas_subset
from geometric_lens.probe import GeometricProbe


# ── Frozen mode atlas contract ───────────────────────────────────────────────
_mode_subset = load_default_mode_atlas_subset()


def cohens_d(a, b):
    """Compute Cohen's d effect size."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std < 1e-10:
        return float("nan")
    return (np.mean(a) - np.mean(b)) / pooled_std


def run_atlas(args):
    """Run the computational mode atlas experiment."""
    print("=" * 70)
    print("COMPUTATIONAL MODE ATLAS")
    print("=" * 70)

    # Output directory
    out_dir = Path("results/mode_atlas")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize probe
    print(f"Loading model: {args.model}")
    probe = GeometricProbe(
        model_name=args.model,
        device=args.device,
        window=16,
        attn_implementation="eager",
    )
    print(f"Model loaded. Spec: {probe.spec.name}")
    print(f"Layers: early={probe.early_layer}, late={probe.late_layer}")
    print(
        "Prompt contract: "
        f"subset={_mode_subset.name} bank={_mode_subset.source_bank_version}"
    )

    # Metrics to compute
    metrics = ["rv", "spectral", "cosine", "attn_entropy", "eigenvalue"]

    # Run measurements
    all_results = {}  # mode_name -> list of GeometricResult.to_dict()
    mode_prompt_ids = {}
    mode_names = list(_mode_subset.manifest["tiers"].keys())

    for mode_idx, mode_name in enumerate(mode_names):
        mode_records = _mode_subset.get_records_for_tier(mode_name)
        mode_prompt_ids[mode_name] = [prompt_id for prompt_id, _ in mode_records]
        prompts = [record["text"] for _, record in mode_records]
        n = min(args.n_prompts, len(prompts))
        prompts = prompts[:n]
        prompt_ids = mode_prompt_ids[mode_name][:n]

        print(f"\n[{mode_idx+1}/{len(mode_names)}] {mode_name} ({n} prompts)")

        mode_results = probe.measure_batch(prompts, metrics=metrics, progress=True)
        all_results[mode_name] = [r.to_dict() for r in mode_results]

        # Quick summary
        rvs = [r.rv for r in mode_results if not np.isnan(r.rv)]
        if rvs:
            print(f"  R_V: {np.mean(rvs):.3f} ± {np.std(rvs):.3f} (n={len(rvs)})")

        # Save per-mode results incrementally
        mode_path = out_dir / f"{mode_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(mode_path, "w") as f:
            json.dump({
                "mode": mode_name,
                "n_prompts": n,
                "prompt_ids": prompt_ids,
                "prompt_bank_version": _mode_subset.source_bank_version,
                "prompt_subset_name": _mode_subset.name,
                "prompt_subset_schema_version": _mode_subset.schema_version,
                "prompt_subset_path": str(_mode_subset.manifest_path),
                "results": all_results[mode_name],
            }, f, indent=2, default=str)

    # ── Compute fingerprint matrix ──
    print("\n" + "=" * 70)
    print("SPECTRAL FINGERPRINT MATRIX")
    print("=" * 70)

    metric_keys = [
        "rv", "pr_early", "pr_late", "cosine", "attn_entropy",
        "spectral_late_top1_ratio", "spectral_late_spectral_gap",
        "spectral_late_effective_rank",
    ]

    fingerprint = {}
    for mode_name in mode_names:
        mode_data = all_results[mode_name]
        fingerprint[mode_name] = {}
        for mk in metric_keys:
            values = [r[mk] for r in mode_data if mk in r and not np.isnan(r.get(mk, float("nan")))]
            fingerprint[mode_name][mk] = {
                "mean": float(np.mean(values)) if values else float("nan"),
                "std": float(np.std(values)) if values else float("nan"),
                "n": len(values),
            }

    # Print fingerprint
    print(f"\n{'Mode':<25} {'R_V':>8} {'PR_late':>8} {'Top1':>8} {'Cosine':>8} {'AttnEnt':>8}")
    print("-" * 75)
    for mode_name in mode_names:
        fp = fingerprint[mode_name]
        print(f"{mode_name:<25} "
              f"{fp['rv']['mean']:>8.3f} "
              f"{fp['pr_late']['mean']:>8.3f} "
              f"{fp['spectral_late_top1_ratio']['mean']:>8.3f} "
              f"{fp['cosine']['mean']:>8.3f} "
              f"{fp['attn_entropy']['mean']:>8.3f}")

    # ── Pairwise comparisons ──
    print("\n" + "=" * 70)
    print("PAIRWISE MODE COMPARISONS (R_V)")
    print("=" * 70)

    comparisons = {}
    for mode_a, mode_b in combinations(mode_names, 2):
        rv_a = [r["rv"] for r in all_results[mode_a] if not np.isnan(r.get("rv", float("nan")))]
        rv_b = [r["rv"] for r in all_results[mode_b] if not np.isnan(r.get("rv", float("nan")))]

        if len(rv_a) >= 3 and len(rv_b) >= 3:
            d = cohens_d(rv_a, rv_b)
            u_stat, p_val = stats.mannwhitneyu(rv_a, rv_b, alternative="two-sided")
        else:
            d = float("nan")
            p_val = float("nan")
            u_stat = float("nan")

        key = f"{mode_a}_vs_{mode_b}"
        comparisons[key] = {
            "d": d,
            "p": p_val,
            "u": u_stat,
            "mean_a": float(np.mean(rv_a)) if rv_a else float("nan"),
            "mean_b": float(np.mean(rv_b)) if rv_b else float("nan"),
        }

        if abs(d) > 0.5 and p_val < 0.05:
            print(f"  {mode_a} vs {mode_b}: d={d:+.2f}, p={p_val:.4f} ***")

    # ── Save summary ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "model": args.model,
        "prompt_bank_version": _mode_subset.source_bank_version,
        "prompt_subset_name": _mode_subset.name,
        "prompt_subset_schema_version": _mode_subset.schema_version,
        "prompt_subset_path": str(_mode_subset.manifest_path),
        "n_modes": len(mode_names),
        "n_prompts_per_mode": args.n_prompts,
        "mode_prompt_ids": mode_prompt_ids,
        "fingerprint": fingerprint,
        "comparisons": comparisons,
        "all_results": all_results,
    }

    summary_path = out_dir / f"atlas_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Summary saved: {summary_path}")

    # Key finding
    print("\n" + "=" * 70)
    print("KEY FINDING: Is self_referential geometrically distinct?")
    sr_rv = [r["rv"] for r in all_results["self_referential"] if not np.isnan(r.get("rv", float("nan")))]
    other_rvs = []
    for mode_name in mode_names:
        if mode_name != "self_referential":
            other_rvs.extend([r["rv"] for r in all_results[mode_name] if not np.isnan(r.get("rv", float("nan")))])

    if sr_rv and other_rvs:
        d_sr = cohens_d(sr_rv, other_rvs)
        _, p_sr = stats.mannwhitneyu(sr_rv, other_rvs, alternative="two-sided")
        print(f"  self_referential R_V: {np.mean(sr_rv):.3f} ± {np.std(sr_rv):.3f}")
        print(f"  all_other R_V:       {np.mean(other_rvs):.3f} ± {np.std(other_rvs):.3f}")
        print(f"  Cohen's d: {d_sr:+.3f}, p={p_sr:.6f}")
        print(f"  DISTINCT: {'YES' if abs(d_sr) > 0.5 and p_sr < 0.01 else 'NO'}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computational Mode Atlas")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-prompts", type=int, default=20, help="Prompts per mode")
    args = parser.parse_args()
    run_atlas(args)
