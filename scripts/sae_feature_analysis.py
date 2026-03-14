#!/usr/bin/env python3
"""
SAE FEATURE ANALYSIS (E3.1 + E3.4)
====================================

Uses Sparse Autoencoder (SAE) features on Gemma-2-2B to bridge R_V with
feature-level interpretability.

1. Load Gemma-2-2B + GemmaScope pre-trained SAEs (via sae-lens)
2. Run self-ref vs baseline prompts through the model
3. Extract SAE feature activations at target layers
4. Identify features that fire differentially for self-referential content
5. Measure R_V on same prompts → correlate R_V with self-ref feature density
6. Feature geometry: do self-ref features cluster more tightly?

Dependencies:
    pip install sae-lens transformer-lens

Papers integrated:
  - Ameisen et al. 2025 (circuit tracing)
  - Templeton et al. 2024 (scaling monosemanticity)
  - Park et al. 2024 (geometry of concepts)

Output: results/sae_features/sae_analysis_<timestamp>.json

Usage:
    python3 scripts/sae_feature_analysis.py --device cuda
    python3 scripts/sae_feature_analysis.py --device cuda --n-prompts 20
"""

import sys
import json
import argparse
import gc
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.loader import PromptLoader
from geometric_lens.probe import GeometricProbe


# ── Prompt bank (loaded from prompts/bank.json) ──────────────────────────────
_loader = PromptLoader()
RECURSIVE_PROMPTS = _loader.get_by_group("L3_deeper") + _loader.get_by_group("L4_full")
BASELINE_PROMPTS = _loader.get_by_group("baseline_factual") + _loader.get_by_group("baseline_math")


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std < 1e-10:
        return float("nan")
    return (np.mean(a) - np.mean(b)) / pooled_std


def try_load_sae(layer_idx, device="cuda"):
    """Try to load a GemmaScope SAE for a given layer.

    Attempts sae-lens first, falls back to manual loading.
    Returns (sae, method) or (None, None).
    """
    # Method 1: sae-lens library
    try:
        from sae_lens import SAE
        sae_id = f"gemma-scope-2b-pt-res-canonical/layer_{layer_idx}/width_16k/canonical"
        sae = SAE.from_pretrained(
            release=sae_id.split("/")[0],
            sae_id="/".join(sae_id.split("/")[1:]),
            device=device,
        )
        print(f"    Loaded SAE via sae-lens: {sae_id}")
        return sae, "sae_lens"
    except Exception as e:
        print(f"    sae-lens failed for layer {layer_idx}: {e}")

    # Method 2: HuggingFace GemmaScope direct
    try:
        from sae_lens import SAE
        # Try alternative naming conventions
        for width in ["16k", "32k", "65k"]:
            try:
                sae_id = f"gemma-scope-2b-pt-res/layer_{layer_idx}/width_{width}/canonical"
                sae = SAE.from_pretrained(
                    release=sae_id.split("/")[0],
                    sae_id="/".join(sae_id.split("/")[1:]),
                    device=device,
                )
                print(f"    Loaded SAE: {sae_id}")
                return sae, "sae_lens"
            except Exception:
                continue
    except Exception:
        pass

    print(f"    No SAE available for layer {layer_idx}")
    return None, None


def extract_sae_features(sae, hidden_states, method="sae_lens"):
    """Extract SAE feature activations from hidden states.

    Args:
        sae: Loaded SAE model
        hidden_states: (seq, hidden_dim) tensor
        method: Loading method used

    Returns:
        feature_acts: (seq, n_features) tensor of feature activations
    """
    if method == "sae_lens":
        with torch.no_grad():
            feature_acts = sae.encode(hidden_states)
        return feature_acts
    return None


def compute_feature_geometry(feature_acts, top_k=100):
    """Compute geometric properties of active feature space.

    Args:
        feature_acts: (seq, n_features) activation tensor
        top_k: Number of top features to consider

    Returns:
        Dict with geometry stats.
    """
    if feature_acts is None:
        return {"effective_dim": float("nan")}

    # Average across sequence positions
    avg_acts = feature_acts.float().mean(dim=0)  # (n_features,)
    active_mask = avg_acts > 0.01
    n_active = active_mask.sum().item()

    if n_active < 3:
        return {"n_active": n_active, "effective_dim": float("nan")}

    # Get top-k active features
    top_vals, top_idx = torch.topk(avg_acts, min(top_k, n_active))
    active_features = feature_acts[:, top_idx].float().cpu()  # (seq, top_k)

    # Effective dimension via SVD
    try:
        _, S, _ = torch.linalg.svd(active_features.T.double(), full_matrices=False)
        S_np = S.numpy()
        S_sq = S_np ** 2
        total = S_sq.sum()
        if total < 1e-10:
            eff_dim = float("nan")
        else:
            p = S_sq / total
            p = p[p > 1e-10]
            eff_dim = float(np.exp(-np.sum(p * np.log(p))))
    except Exception:
        eff_dim = float("nan")

    # Pairwise cosine similarity among top features
    norms = active_features.norm(dim=0, keepdim=True) + 1e-10
    normed = active_features / norms
    cos_sim = (normed.T @ normed).numpy()
    np.fill_diagonal(cos_sim, 0)
    mean_cos = float(cos_sim.mean()) if cos_sim.size > 1 else float("nan")

    return {
        "n_active": n_active,
        "effective_dim": eff_dim,
        "mean_pairwise_cosine": mean_cos,
        "top_feature_indices": top_idx.tolist()[:20],
        "top_feature_values": top_vals.tolist()[:20],
    }


def run_sae_analysis(args):
    """Run SAE feature analysis on Gemma-2-2B."""
    print("=" * 70)
    print("SAE FEATURE ANALYSIS (E3.1 + E3.4)")
    print("=" * 70)

    out_dir = Path("results/sae_features")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model
    print(f"Loading model: {model_name}")

    # ── Step 1: Measure R_V on all prompts (E3.4) ──
    print("\n  Step 1: R_V measurement on Gemma-2-2B...")
    probe = GeometricProbe(
        model_name=model_name,
        device=args.device,
        attn_implementation="eager",
    )
    model = probe.model
    tokenizer = probe.tokenizer
    spec = probe.spec

    print(f"  Model: layers={spec.num_layers}, heads={spec.num_heads}, "
          f"early={probe.early_layer}, late={probe.late_layer}")

    n = args.n_prompts
    rec_prompts = RECURSIVE_PROMPTS[:n]
    bas_prompts = BASELINE_PROMPTS[:n]

    rec_rv_results = probe.measure_batch(rec_prompts, metrics=["rv"], progress=True)
    bas_rv_results = probe.measure_batch(bas_prompts, metrics=["rv"], progress=True)

    rec_rvs = [r.rv for r in rec_rv_results if not np.isnan(r.rv)]
    bas_rvs = [r.rv for r in bas_rv_results if not np.isnan(r.rv)]

    if rec_rvs and bas_rvs:
        rv_d = cohens_d(rec_rvs, bas_rvs)
        _, rv_p = stats.mannwhitneyu(rec_rvs, bas_rvs, alternative="two-sided")
    else:
        rv_d, rv_p = float("nan"), float("nan")

    print(f"\n  R_V recursive: {np.mean(rec_rvs):.3f} ± {np.std(rec_rvs):.3f}")
    print(f"  R_V baseline:  {np.mean(bas_rvs):.3f} ± {np.std(bas_rvs):.3f}")
    print(f"  Cohen's d: {rv_d:.3f}, p={rv_p:.6f}")

    # ── Step 2: Load SAEs at target layers ──
    print("\n  Step 2: Loading SAEs...")
    target_layers = [probe.early_layer, spec.num_layers // 2, probe.late_layer]
    saes = {}
    for layer_idx in target_layers:
        sae, method = try_load_sae(layer_idx, device=args.device)
        if sae is not None:
            saes[layer_idx] = (sae, method)

    if not saes:
        print("\n  WARNING: No SAEs loaded. Falling back to hidden state analysis only.")
        # Fallback: use raw hidden state geometry analysis
        print("  Running hidden state geometry analysis instead...")
        return run_hidden_state_fallback(probe, rec_prompts, bas_prompts, rec_rvs, bas_rvs,
                                          rv_d, rv_p, out_dir, args)

    # ── Step 3: Extract SAE features per prompt ──
    print("\n  Step 3: Extracting SAE features...")

    all_feature_data = {}
    for layer_idx, (sae, method) in saes.items():
        print(f"\n    Layer {layer_idx}:")
        layer_data = {"recursive": [], "baseline": []}

        for condition, prompts in [("recursive", rec_prompts), ("baseline", bas_prompts)]:
            for pi, text in enumerate(prompts):
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(args.device)

                # Get hidden states at this layer
                from geometric_lens.hooks import capture_hidden_states
                with capture_hidden_states(model, layer_idx) as sh:
                    with torch.no_grad():
                        model(**enc)
                    hidden = sh.get("hidden")

                if hidden is None:
                    layer_data[condition].append({"error": "no hidden states"})
                    continue

                if hidden.dim() == 3:
                    hidden = hidden[0]  # (seq, dim)

                # Extract SAE features
                feature_acts = extract_sae_features(sae, hidden, method)
                if feature_acts is None:
                    layer_data[condition].append({"error": "sae encoding failed"})
                    continue

                # Compute statistics
                avg_acts = feature_acts.float().mean(dim=0)
                n_active = (avg_acts > 0.01).sum().item()
                total_activation = avg_acts.sum().item()

                # Feature geometry
                geometry = compute_feature_geometry(feature_acts)

                # Top features
                top_vals, top_idx = torch.topk(avg_acts, min(20, len(avg_acts)))

                layer_data[condition].append({
                    "prompt_idx": pi,
                    "n_active_features": n_active,
                    "total_activation": float(total_activation),
                    "effective_dim": geometry["effective_dim"],
                    "mean_pairwise_cosine": geometry.get("mean_pairwise_cosine", float("nan")),
                    "top_features": top_idx.tolist(),
                    "top_values": [float(v) for v in top_vals.tolist()],
                })

            print(f"      {condition}: {len(layer_data[condition])} prompts processed")

        all_feature_data[layer_idx] = layer_data

    # ── Step 4: Differential feature analysis ──
    print("\n  Step 4: Differential feature analysis...")

    differential_results = {}
    for layer_idx, layer_data in all_feature_data.items():
        rec_n_active = [d["n_active_features"] for d in layer_data["recursive"]
                        if isinstance(d, dict) and "n_active_features" in d]
        bas_n_active = [d["n_active_features"] for d in layer_data["baseline"]
                        if isinstance(d, dict) and "n_active_features" in d]

        rec_eff_dim = [d["effective_dim"] for d in layer_data["recursive"]
                       if isinstance(d, dict) and not np.isnan(d.get("effective_dim", float("nan")))]
        bas_eff_dim = [d["effective_dim"] for d in layer_data["baseline"]
                       if isinstance(d, dict) and not np.isnan(d.get("effective_dim", float("nan")))]

        # Count features: more active features in self-ref?
        if rec_n_active and bas_n_active:
            d_active = cohens_d(rec_n_active, bas_n_active)
        else:
            d_active = float("nan")

        # Feature dimensionality: lower effective dim in self-ref = more concentrated
        if rec_eff_dim and bas_eff_dim:
            d_dim = cohens_d(rec_eff_dim, bas_eff_dim)
        else:
            d_dim = float("nan")

        # Find features unique to self-ref
        rec_feature_sets = [set(d["top_features"][:10]) for d in layer_data["recursive"]
                           if isinstance(d, dict) and "top_features" in d]
        bas_feature_sets = [set(d["top_features"][:10]) for d in layer_data["baseline"]
                           if isinstance(d, dict) and "top_features" in d]

        rec_union = set().union(*rec_feature_sets) if rec_feature_sets else set()
        bas_union = set().union(*bas_feature_sets) if bas_feature_sets else set()
        selfref_specific = rec_union - bas_union
        baseline_specific = bas_union - rec_union

        differential_results[layer_idx] = {
            "n_active_recursive_mean": float(np.mean(rec_n_active)) if rec_n_active else float("nan"),
            "n_active_baseline_mean": float(np.mean(bas_n_active)) if bas_n_active else float("nan"),
            "d_n_active": d_active,
            "eff_dim_recursive_mean": float(np.mean(rec_eff_dim)) if rec_eff_dim else float("nan"),
            "eff_dim_baseline_mean": float(np.mean(bas_eff_dim)) if bas_eff_dim else float("nan"),
            "d_eff_dim": d_dim,
            "n_selfref_specific_features": len(selfref_specific),
            "n_baseline_specific_features": len(baseline_specific),
            "selfref_specific_features": list(selfref_specific)[:50],
        }

        print(f"\n    Layer {layer_idx}:")
        print(f"      Active features: rec={np.mean(rec_n_active):.0f} vs bas={np.mean(bas_n_active):.0f} (d={d_active:.3f})")
        print(f"      Effective dim: rec={np.mean(rec_eff_dim):.1f} vs bas={np.mean(bas_eff_dim):.1f} (d={d_dim:.3f})")
        print(f"      Self-ref specific features: {len(selfref_specific)}")

    # ── Step 5: R_V × feature correlation ──
    print("\n  Step 5: R_V × feature density correlation...")
    correlations = {}
    for layer_idx, layer_data in all_feature_data.items():
        rec_n_active = [d["n_active_features"] for d in layer_data["recursive"]
                        if isinstance(d, dict) and "n_active_features" in d]
        # Pair with per-prompt R_V
        paired_rv = []
        paired_features = []
        for i, d in enumerate(layer_data["recursive"]):
            if isinstance(d, dict) and "n_active_features" in d and i < len(rec_rv_results):
                rv_val = rec_rv_results[i].rv
                if not np.isnan(rv_val):
                    paired_rv.append(rv_val)
                    paired_features.append(d["n_active_features"])

        if len(paired_rv) >= 5:
            r_corr, p_corr = stats.pearsonr(paired_rv, paired_features)
            correlations[layer_idx] = {"r": float(r_corr), "p": float(p_corr), "n": len(paired_rv)}
            print(f"    Layer {layer_idx}: r={r_corr:.3f}, p={p_corr:.4f} (n={len(paired_rv)})")
        else:
            correlations[layer_idx] = {"r": float("nan"), "p": float("nan"), "n": len(paired_rv)}

    # ── Save results ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "experiment": "E3.1_E3.4_sae_features",
        "model": model_name,
        "n_recursive": len(rec_prompts),
        "n_baseline": len(bas_prompts),
        "rv_stats": {
            "recursive_mean": float(np.mean(rec_rvs)) if rec_rvs else float("nan"),
            "baseline_mean": float(np.mean(bas_rvs)) if bas_rvs else float("nan"),
            "cohens_d": rv_d,
            "p_value": rv_p,
        },
        "sae_layers_loaded": list(saes.keys()),
        "differential_results": {str(k): v for k, v in differential_results.items()},
        "rv_feature_correlations": {str(k): v for k, v in correlations.items()},
        "per_prompt_data": {str(k): v for k, v in all_feature_data.items()},
    }

    summary_path = out_dir / f"sae_analysis_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Summary saved: {summary_path}")
    print("=" * 70)


def run_hidden_state_fallback(probe, rec_prompts, bas_prompts, rec_rvs, bas_rvs,
                               rv_d, rv_p, out_dir, args):
    """Fallback analysis using hidden state geometry when SAEs unavailable."""
    from geometric_lens.hooks import capture_hidden_states

    print("\n  Running hidden state geometry fallback...")
    model = probe.model
    tokenizer = probe.tokenizer
    spec = probe.spec

    target_layers = list(range(0, spec.num_layers, max(1, spec.num_layers // 8)))

    layer_geometry = {}
    for layer_idx in target_layers:
        rec_dims, bas_dims = [], []

        for condition, prompts in [("recursive", rec_prompts), ("baseline", bas_prompts)]:
            for text in prompts:
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(args.device)
                with capture_hidden_states(model, layer_idx) as sh:
                    with torch.no_grad():
                        model(**enc)
                    hidden = sh.get("hidden")

                if hidden is None:
                    continue
                if hidden.dim() == 3:
                    hidden = hidden[0]

                # Effective dimension via SVD
                W = min(16, hidden.shape[0])
                h_window = hidden[-W:, :].cpu().double()
                try:
                    _, S, _ = torch.linalg.svd(h_window.T, full_matrices=False)
                    S_np = S.numpy()
                    S_sq = S_np ** 2
                    total = S_sq.sum()
                    if total > 1e-10:
                        p = S_sq / total
                        p = p[p > 1e-10]
                        eff_dim = float(np.exp(-np.sum(p * np.log(p))))
                    else:
                        eff_dim = float("nan")
                except Exception:
                    eff_dim = float("nan")

                if condition == "recursive":
                    rec_dims.append(eff_dim)
                else:
                    bas_dims.append(eff_dim)

        rec_valid = [v for v in rec_dims if not np.isnan(v)]
        bas_valid = [v for v in bas_dims if not np.isnan(v)]

        if rec_valid and bas_valid:
            d = cohens_d(rec_valid, bas_valid)
        else:
            d = float("nan")

        layer_geometry[layer_idx] = {
            "eff_dim_recursive_mean": float(np.mean(rec_valid)) if rec_valid else float("nan"),
            "eff_dim_baseline_mean": float(np.mean(bas_valid)) if bas_valid else float("nan"),
            "d": d,
        }
        print(f"    Layer {layer_idx}: rec_dim={np.mean(rec_valid):.1f} bas_dim={np.mean(bas_valid):.1f} d={d:.3f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "experiment": "E3.1_E3.4_hidden_state_fallback",
        "model": args.model,
        "note": "SAEs unavailable — using hidden state geometry as fallback",
        "rv_stats": {
            "recursive_mean": float(np.mean(rec_rvs)) if rec_rvs else float("nan"),
            "baseline_mean": float(np.mean(bas_rvs)) if bas_rvs else float("nan"),
            "cohens_d": rv_d,
            "p_value": rv_p,
        },
        "layer_geometry": {str(k): v for k, v in layer_geometry.items()},
    }

    summary_path = out_dir / f"hidden_state_analysis_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Summary saved: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAE Feature Analysis (E3.1 + E3.4)")
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-prompts", type=int, default=20, help="Prompts per condition")
    args = parser.parse_args()
    run_sae_analysis(args)
