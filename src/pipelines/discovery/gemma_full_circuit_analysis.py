"""
Gemma 2 9B Full Circuit Analysis

Comprehensive analysis including:
1. R_V layer sweep (all 42 layers) - find phase transition
2. Logit lens trajectory - per-layer predictions/entropy
3. Extended metrics - spectral stats, cosine similarity, attention entropy
4. PR component breakdown - early vs late contributions

This replicates the full Mistral validation protocol for Gemma.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.metrics.rv import compute_rv_with_components
from src.metrics.logit_lens import compute_logit_lens_trajectory
from src.metrics.extended import (
    compute_spectral_stats,
    compute_cosine_similarity,
)
from src.pipelines.registry import ExperimentResult
from src.utils.run_metadata import get_run_metadata, save_metadata


def extract_v_projection(model, tokenizer, text: str, layer_idx: int, device: str = "cuda") -> torch.Tensor:
    """Extract V-projection at a specific layer."""
    inputs = tokenizer(text, return_tensors="pt").to(device)

    v_proj = None

    def hook_fn(module, inp, out):
        nonlocal v_proj
        v_proj = out.detach()

    # Register hook on v_proj at the target layer
    handle = model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()

    return v_proj


def compute_rv_at_layer(model, tokenizer, text: str, early: int, late: int, window: int, device: str) -> Tuple[float, float, float]:
    """Compute R_V with a specific late layer."""
    try:
        rv, pr_early, pr_late = compute_rv_with_components(
            model, tokenizer, text, early=early, late=late, window=window, device=device
        )
        return rv, pr_early, pr_late
    except Exception as e:
        return float("nan"), float("nan"), float("nan")


def run_gemma_full_circuit_analysis_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """
    Run comprehensive Gemma 2 9B circuit analysis.
    """
    params = cfg.get("params", {})
    model_name = params.get("model") or cfg.get("model", {}).get("name", "google/gemma-2-9b")
    n_prompts = params.get("n_prompts", 30)
    window_size = params.get("window_size", 16)
    early_layer = params.get("early_layer", 5)
    seed = int(params.get("seed", 42))

    # Gemma 2 9B has 42 layers
    num_layers = params.get("num_layers", 42)

    # Layer parity: "odd" (default), "even", or "all"
    layer_parity = params.get("layer_parity", "odd")

    # Optional: specific layer range for focused sweeps
    layer_range = params.get("layer_range", None)  # e.g., [35, 41] for L35-L41

    set_seed(seed)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name, device=device)
    model.eval()

    # Load prompts
    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)

    # Get recursive and baseline prompts
    recursive_prompts = loader.get_by_group("L5_refined")[:n_prompts]
    baseline_prompts = loader.get_by_group("baseline_factual")[:n_prompts]

    print(f"\n{'='*70}")
    print("GEMMA 2 9B FULL CIRCUIT ANALYSIS")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Layers: {num_layers}")
    print(f"Prompts: {n_prompts} recursive, {n_prompts} baseline")
    print(f"{'='*70}\n")

    # ========== PHASE 1: R_V LAYER SWEEP ==========
    print("\n[1/3] R_V LAYER SWEEP (finding phase transition)...")

    layer_sweep_results = []

    # Test layers at different depths based on config settings
    if layer_range is not None:
        # Focused sweep: test all layers in the specified range
        start, end = layer_range[0], layer_range[1]
        test_layers = list(range(start, end + 1))
        print(f"Focused sweep: layers {start}-{end} (all layers in range)")
    elif layer_parity == "even":
        test_layers = list(range(6, num_layers, 2))  # Even layers: 6, 8, 10, ..., 40
    elif layer_parity == "all":
        test_layers = list(range(5, num_layers))  # All layers
    else:  # default: "odd"
        test_layers = list(range(5, num_layers, 2))  # Odd layers: 5, 7, 9, ..., 41

    # Use sweep_n_prompts from config (default 10 for speed, but configurable)
    sweep_n_prompts = params.get("sweep_n_prompts", 10)

    for late_layer in tqdm(test_layers, desc="Layer sweep"):
        rec_rvs = []
        base_rvs = []

        for rec_text in recursive_prompts[:sweep_n_prompts]:
            rv, pr_e, pr_l = compute_rv_at_layer(
                model, tokenizer, rec_text, early_layer, late_layer, window_size, device
            )
            if not np.isnan(rv):
                rec_rvs.append(rv)

        for base_text in baseline_prompts[:sweep_n_prompts]:
            rv, pr_e, pr_l = compute_rv_at_layer(
                model, tokenizer, base_text, early_layer, late_layer, window_size, device
            )
            if not np.isnan(rv):
                base_rvs.append(rv)

        if rec_rvs and base_rvs:
            rec_mean = np.mean(rec_rvs)
            base_mean = np.mean(base_rvs)
            delta = rec_mean - base_mean

            # T-test
            if len(rec_rvs) >= 2 and len(base_rvs) >= 2:
                t_stat, p_val = stats.ttest_ind(rec_rvs, base_rvs)
            else:
                t_stat, p_val = 0.0, 1.0

            layer_sweep_results.append({
                "late_layer": late_layer,
                "depth_pct": late_layer / num_layers * 100,
                "rv_recursive_mean": rec_mean,
                "rv_recursive_std": np.std(rec_rvs),
                "rv_baseline_mean": base_mean,
                "rv_baseline_std": np.std(base_rvs),
                "rv_delta": delta,
                "t_statistic": t_stat,
                "p_value": p_val,
                "significant": p_val < 0.01,
            })

    # Save layer sweep
    sweep_df = pd.DataFrame(layer_sweep_results)
    sweep_df.to_csv(run_dir / "layer_sweep.csv", index=False)

    # Find phase transition (biggest delta change)
    if len(layer_sweep_results) > 1:
        deltas = [r["rv_delta"] for r in layer_sweep_results]
        max_delta_idx = np.argmax(np.abs(deltas))
        phase_transition_layer = layer_sweep_results[max_delta_idx]["late_layer"]
        max_delta = deltas[max_delta_idx]
    else:
        phase_transition_layer = 35
        max_delta = 0

    print(f"\n  Phase transition at L{phase_transition_layer} (delta={max_delta:.3f})")

    # ========== PHASE 2: LOGIT LENS ==========
    print("\n[2/3] LOGIT LENS ANALYSIS...")

    logit_lens_results = []

    for i, rec_text in enumerate(tqdm(recursive_prompts[:5], desc="Logit lens")):
        try:
            results, metadata = compute_logit_lens_trajectory(
                model, tokenizer, rec_text, device=device
            )

            for r in results:
                logit_lens_results.append({
                    "prompt_idx": i,
                    "prompt_type": "recursive",
                    "layer": r.layer,
                    "top_token": r.top_tokens[0] if r.top_tokens else "",
                    "top_prob": r.top_probs[0] if r.top_probs else 0,
                    "entropy": r.entropy,
                })

        except Exception as e:
            print(f"  Logit lens error: {e}")

    for i, base_text in enumerate(tqdm(baseline_prompts[:5], desc="Logit lens baseline")):
        try:
            results, metadata = compute_logit_lens_trajectory(
                model, tokenizer, base_text, device=device
            )

            for r in results:
                logit_lens_results.append({
                    "prompt_idx": i,
                    "prompt_type": "baseline",
                    "layer": r.layer,
                    "top_token": r.top_tokens[0] if r.top_tokens else "",
                    "top_prob": r.top_probs[0] if r.top_probs else 0,
                    "entropy": r.entropy,
                })

        except Exception as e:
            print(f"  Logit lens baseline error: {e}")

    # Save logit lens
    if logit_lens_results:
        lens_df = pd.DataFrame(logit_lens_results)
        lens_df.to_csv(run_dir / "logit_lens.csv", index=False)

        # Compute entropy trajectory comparison
        rec_entropy = lens_df[lens_df["prompt_type"] == "recursive"].groupby("layer")["entropy"].mean()
        base_entropy = lens_df[lens_df["prompt_type"] == "baseline"].groupby("layer")["entropy"].mean()

        entropy_comparison = pd.DataFrame({
            "layer": rec_entropy.index,
            "recursive_entropy": rec_entropy.values,
            "baseline_entropy": base_entropy.reindex(rec_entropy.index).values,
        })
        entropy_comparison["entropy_delta"] = entropy_comparison["recursive_entropy"] - entropy_comparison["baseline_entropy"]
        entropy_comparison.to_csv(run_dir / "entropy_trajectory.csv", index=False)

    # ========== PHASE 3: EXTENDED METRICS ==========
    print("\n[3/3] EXTENDED METRICS (spectral, cosine)...")

    extended_results = []

    for i, (rec_text, base_text) in enumerate(tqdm(
        zip(recursive_prompts[:10], baseline_prompts[:10]),
        total=10, desc="Extended metrics"
    )):
        try:
            # Get V projections at early and late layers
            v_early_rec = extract_v_projection(model, tokenizer, rec_text, early_layer, device)
            v_late_rec = extract_v_projection(model, tokenizer, rec_text, phase_transition_layer, device)

            v_early_base = extract_v_projection(model, tokenizer, base_text, early_layer, device)
            v_late_base = extract_v_projection(model, tokenizer, base_text, phase_transition_layer, device)

            # Compute spectral stats
            spec_early_rec = compute_spectral_stats(v_early_rec, window_size)
            spec_late_rec = compute_spectral_stats(v_late_rec, window_size)
            spec_early_base = compute_spectral_stats(v_early_base, window_size)
            spec_late_base = compute_spectral_stats(v_late_base, window_size)

            # Compute cosine similarities
            cos_rec = compute_cosine_similarity(v_early_rec, v_late_rec, window_size)
            cos_base = compute_cosine_similarity(v_early_base, v_late_base, window_size)

            extended_results.append({
                "pair_idx": i,
                # Recursive
                "rec_spectral_early_top1": spec_early_rec.top1_ratio if spec_early_rec else float("nan"),
                "rec_spectral_late_top1": spec_late_rec.top1_ratio if spec_late_rec else float("nan"),
                "rec_spectral_early_eff_rank": spec_early_rec.effective_rank if spec_early_rec else float("nan"),
                "rec_spectral_late_eff_rank": spec_late_rec.effective_rank if spec_late_rec else float("nan"),
                "rec_cosine_early_late": cos_rec,
                # Baseline
                "base_spectral_early_top1": spec_early_base.top1_ratio if spec_early_base else float("nan"),
                "base_spectral_late_top1": spec_late_base.top1_ratio if spec_late_base else float("nan"),
                "base_spectral_early_eff_rank": spec_early_base.effective_rank if spec_early_base else float("nan"),
                "base_spectral_late_eff_rank": spec_late_base.effective_rank if spec_late_base else float("nan"),
                "base_cosine_early_late": cos_base,
            })

        except Exception as e:
            print(f"  Extended metrics error at {i}: {e}")

    # Save extended metrics
    if extended_results:
        ext_df = pd.DataFrame(extended_results)
        ext_df.to_csv(run_dir / "extended_metrics.csv", index=False)

    # ========== SUMMARY ==========
    summary = {
        "experiment": "gemma_full_circuit_analysis",
        "model": model_name,
        "num_layers": num_layers,
        "early_layer": early_layer,
        "phase_transition_layer": int(phase_transition_layer),
        "phase_transition_depth_pct": float(phase_transition_layer / num_layers * 100),
        "max_rv_delta": float(max_delta),
        "n_layer_sweep_points": len(layer_sweep_results),
        "n_logit_lens_prompts": len(set(r["prompt_idx"] for r in logit_lens_results)) if logit_lens_results else 0,
        "n_extended_metric_pairs": len(extended_results),
    }

    # Add layer sweep stats
    if layer_sweep_results:
        sig_layers = [r["late_layer"] for r in layer_sweep_results if r["significant"]]
        summary["significant_layers"] = sig_layers
        summary["n_significant_layers"] = len(sig_layers)

        # Find best separation layer
        best_idx = np.argmax([abs(r["rv_delta"]) for r in layer_sweep_results])
        summary["best_separation_layer"] = int(layer_sweep_results[best_idx]["late_layer"])
        summary["best_separation_delta"] = float(layer_sweep_results[best_idx]["rv_delta"])

    # Save summary
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Phase transition layer: L{phase_transition_layer} ({phase_transition_layer/num_layers*100:.1f}% depth)")
    print(f"Max R_V delta: {max_delta:.3f}")
    if layer_sweep_results:
        print(f"Significant layers: {sig_layers}")
    print(f"\nResults saved to: {run_dir}")

    return ExperimentResult(summary=summary)
