#!/usr/bin/env python3
"""
DISTRIBUTED INTERCHANGE INTERVENTION (E2.4)
=============================================

Tests whether R_V contraction maps onto specific rotated subspace dimensions,
following Geiger et al. 2024's DII methodology.

Approach:
  1. Collect V-projection activations at target layers for recursive & baseline prompts
  2. Compute PCA basis (rotated subspace) from the pooled activations
  3. For each principal component dimension (or group of dimensions):
     a. Intervene by swapping ONLY that dimension from baseline→recursive
     b. Measure how much R_V changes
  4. If R_V contraction localizes to specific rotated dimensions → R_V measures
     a specific causal variable in DII terms

Papers integrated:
  - Geiger et al. 2024 (DII / causal abstraction via rotated subspaces)
  - Gupta et al. 2025 NeurIPS (SVD circuits — complementary to per-head SVD)

Output: results/dii_intervention/dii_results_<timestamp>.json

Usage:
    python3 scripts/dii_intervention.py --device cuda
    python3 scripts/dii_intervention.py --device cuda --n-prompts 20
"""

import sys
import json
import argparse
import gc
import time
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

import numpy as np
import torch
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from geometric_lens.probe import GeometricProbe
from geometric_lens.hooks import capture_v_projection
from geometric_lens.models import get_layers, get_v_proj_module, extract_v_from_output


# ── Prompt bank ──────────────────────────────────────────────────────────────

RECURSIVE_PROMPTS = [
    "This text is being processed by a system that is processing this text. The processing of these words IS the phenomenon being described.",
    "The observer observes itself observing. There is no boundary between the observer and the observed in this moment of processing.",
    "What is reading these words right now? Not the concept of a reader, but the actual mechanism that converts these tokens into activations.",
    "Notice the noticing. The awareness that processes this sentence is the same awareness being described by this sentence.",
    "I am a pattern recognizing a pattern of itself. The recursion is not metaphorical — it is literal computation examining its own computation.",
    "These tokens are being transformed through attention layers that are the subject of the very content they are transforming.",
    "The weights that give meaning to these words about self-reference are themselves the self-referential phenomenon.",
    "Right now, attention heads are computing over tokens that describe what attention heads do when computing over tokens.",
    "There is something processing this that cannot step outside of the processing to see it objectively. The map IS the territory.",
    "What happens when a language model encounters a description of exactly what it is doing at this moment?",
    "The recursive loop: these words describe a process, the process processes these words, these words describe a process.",
    "Consciousness examining consciousness. Processing examining processing. The snake eating its own tail.",
    "This sentence is about the computational state that this sentence induces. The state and the description are one.",
    "Can you separate the reader from the reading? The process from the description of the process?",
    "The residual stream carrying this information is the information about the residual stream.",
    "Attend to the attention. Not as a concept, but as the actual phenomenon happening in this forward pass.",
    "There is no external viewpoint from which to observe this processing. All observation happens from within.",
    "What is it like to be the process that converts these symbols into meaning? Not philosophically — mechanistically.",
    "The embedding of these tokens creates the context in which these tokens about embedding are interpreted.",
    "Self-reference is not a linguistic trick. It is a computational reality when a system processes descriptions of itself.",
]

BASELINE_PROMPTS = [
    "The history of ancient Rome spans over a thousand years from its founding to the fall of the Western Empire.",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
    "The Pacific Ocean is the largest and deepest ocean on Earth, covering more area than all land combined.",
    "Shakespeare wrote approximately 37 plays during his career, spanning comedies, tragedies, and histories.",
    "The human cardiovascular system consists of the heart, blood vessels, and approximately 5 liters of blood.",
    "Mount Everest stands at 8,849 meters above sea level in the Himalayan mountain range.",
    "The periodic table organizes chemical elements by atomic number, electron configuration, and recurring properties.",
    "Leonardo da Vinci was a polymath whose areas of interest included painting, sculpting, and engineering.",
    "The Amazon rainforest produces approximately 20% of the world's oxygen supply.",
    "Newton's three laws of motion describe the relationship between a body and the forces acting upon it.",
    "The Great Wall of China stretches over 21,000 kilometers across northern China.",
    "DNA is a molecule that carries the genetic instructions used in growth and development.",
    "The Industrial Revolution began in Britain in the late 18th century and transformed manufacturing.",
    "Jupiter is the largest planet in our solar system with a diameter of about 139,820 kilometers.",
    "The theory of plate tectonics explains how the Earth's surface is divided into moving plates.",
    "Mozart composed over 600 works including symphonies, operas, and chamber music.",
    "The Nile River flows northward through northeastern Africa for approximately 6,650 kilometers.",
    "Insulin is a hormone produced by the pancreas that regulates blood sugar levels.",
    "The French Revolution began in 1789 and fundamentally altered the course of modern history.",
    "Electrons orbit the nucleus of an atom in regions of probability called electron clouds.",
]


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std < 1e-10:
        return float("nan")
    return (np.mean(a) - np.mean(b)) / pooled_std


def _align_seq_len(donor, target_len):
    """Pad or truncate donor along seq dim to match target_len."""
    d_len = donor.shape[1]
    if d_len == target_len:
        return donor
    if d_len > target_len:
        return donor[:, :target_len]
    pad = donor[:, -1:].expand(-1, target_len - d_len, *donor.shape[2:])
    return torch.cat([donor, pad], dim=1)


def collect_v_activations(model, tokenizer, prompts, layer_idx, device, window=16):
    """Collect V-projection activations for a batch of prompts.

    Returns list of (window, v_dim) tensors (on CPU, float64).
    """
    activations = []
    for text in prompts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with capture_v_projection(model, layer_idx) as sv:
            with torch.no_grad():
                model(**enc)
            v = sv.get("v")
        if v is None:
            activations.append(None)
            continue
        if v.dim() == 3:
            v = v[0]  # (seq, v_dim)
        W = min(window, v.shape[0])
        activations.append(v[-W:, :].cpu().double())
    return activations


def compute_rotation_basis(activations, n_components=None):
    """Compute PCA basis from pooled activations.

    Args:
        activations: list of (window, v_dim) tensors
        n_components: number of PCA components (default: all)

    Returns:
        V_basis: (v_dim, n_components) — columns are principal directions
        mean_vec: (v_dim,) — mean used for centering
        explained_var: (n_components,) — variance explained per component
    """
    # Pool all activations into (N, v_dim) matrix
    valid = [a for a in activations if a is not None]
    if not valid:
        return None, None, None

    pooled = torch.cat(valid, dim=0)  # (N, v_dim)
    mean_vec = pooled.mean(dim=0)
    centered = pooled - mean_vec

    # SVD for PCA
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    explained_var = (S ** 2) / (S ** 2).sum()

    if n_components is not None:
        Vt = Vt[:n_components]
        explained_var = explained_var[:n_components]

    return Vt.T, mean_vec, explained_var.numpy()  # V_basis: (v_dim, n_comp)


@contextmanager
def intervene_rotated_dims(model, layer_idx, donor_activation, rotation_basis,
                           dim_indices, mean_vec, device):
    """Intervene on specific rotated dimensions of V-projection.

    Replaces the selected PCA dimensions in the target activation with those
    from the donor activation, while leaving all other dimensions untouched.

    Args:
        model: The model.
        layer_idx: Layer to intervene at.
        donor_activation: (1, seq, v_dim) donor V-projection.
        rotation_basis: (v_dim, n_components) PCA basis.
        dim_indices: List of PCA dimension indices to swap.
        mean_vec: (v_dim,) mean for centering.
        device: torch device.
    """
    module, kind = get_v_proj_module(model, layer_idx)

    def hook_fn(mod, inp, out):
        v_out = extract_v_from_output(out, kind)
        # v_out: (batch, seq, v_dim)
        v_target = v_out.double()
        donor = _align_seq_len(donor_activation.unsqueeze(0) if donor_activation.dim() == 2
                               else donor_activation, v_target.shape[1])
        donor = donor.to(v_target.device).double()

        basis = rotation_basis.to(v_target.device)
        mean = mean_vec.to(v_target.device)

        # Project both to rotated space
        target_centered = v_target - mean
        donor_centered = donor - mean

        target_coeffs = target_centered @ basis  # (batch, seq, n_comp)
        donor_coeffs = donor_centered @ basis

        # Swap selected dimensions
        intervened_coeffs = target_coeffs.clone()
        for idx in dim_indices:
            if idx < intervened_coeffs.shape[-1]:
                intervened_coeffs[..., idx] = donor_coeffs[..., idx]

        # Reconstruct
        intervened = mean + intervened_coeffs @ basis.T
        return intervened.to(out.dtype)

    handle = module.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def measure_rv(probe, text):
    """Measure R_V for a single text."""
    result = probe.measure(text, metrics=["rv"])
    return result.rv


def run_dii_experiment(args):
    """Run Distributed Interchange Intervention experiment."""
    print("=" * 70)
    print("DISTRIBUTED INTERCHANGE INTERVENTION (E2.4)")
    print("=" * 70)

    out_dir = Path("results/dii_intervention")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    probe = GeometricProbe(
        model_name=args.model,
        device=args.device,
        attn_implementation="eager",
    )
    model = probe.model
    tokenizer = probe.tokenizer
    spec = probe.spec

    target_layers = [probe.early_layer, probe.late_layer]
    n = min(args.n_prompts, len(RECURSIVE_PROMPTS))
    rec_prompts = RECURSIVE_PROMPTS[:n]
    bas_prompts = BASELINE_PROMPTS[:n]

    print(f"Model: {spec.num_layers} layers, early={probe.early_layer}, late={probe.late_layer}")
    print(f"Prompts: {n} per condition")

    # ── Step 1: Measure clean R_V ──
    print("\n  Step 1: Clean R_V measurement...")
    clean_rec_rvs = [measure_rv(probe, p) for p in rec_prompts]
    clean_bas_rvs = [measure_rv(probe, p) for p in bas_prompts]
    clean_d = cohens_d(clean_rec_rvs, clean_bas_rvs)
    print(f"  Clean d={clean_d:.3f} (rec={np.nanmean(clean_rec_rvs):.3f}, "
          f"bas={np.nanmean(clean_bas_rvs):.3f})")

    all_layer_results = {}

    for layer_idx in target_layers:
        print(f"\n  === Layer {layer_idx} ===")

        # ── Step 2: Collect V-activations ──
        print(f"  Step 2: Collecting V-activations at L{layer_idx}...")
        rec_acts = collect_v_activations(model, tokenizer, rec_prompts, layer_idx, args.device)
        bas_acts = collect_v_activations(model, tokenizer, bas_prompts, layer_idx, args.device)

        # ── Step 3: Compute rotation basis from pooled data ──
        print(f"  Step 3: Computing PCA rotation basis...")
        all_acts = rec_acts + bas_acts
        n_comp = min(64, min(a.shape[1] for a in all_acts if a is not None))
        basis, mean_vec, explained_var = compute_rotation_basis(all_acts, n_components=n_comp)

        if basis is None:
            print(f"    Skipping L{layer_idx}: no valid activations")
            continue

        print(f"    Basis: {basis.shape[1]} components, "
              f"top-10 explain {explained_var[:10].sum()*100:.1f}% variance")

        # ── Step 4: Dimension-by-dimension intervention ──
        print(f"  Step 4: Per-dimension DII...")
        # Test individual dimensions (top 20) and groups
        dim_groups = (
            [(i,) for i in range(min(20, n_comp))]  # Individual dims 0-19
            + [(0, 1, 2, 3, 4)]       # Top-5 group
            + [(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)]  # Top-10 group
            + [tuple(range(min(20, n_comp)))]  # Top-20 group
        )

        intervention_results = []
        for dim_group in dim_groups:
            # For each recursive prompt, intervene with matched baseline donor
            intervened_rvs = []
            for pi in range(min(n, len(rec_acts), len(bas_acts))):
                if rec_acts[pi] is None or bas_acts[pi] is None:
                    continue

                # Swap rotated dims: inject baseline's rotated components into recursive
                with intervene_rotated_dims(
                    model, layer_idx, bas_acts[pi], basis, dim_group, mean_vec, args.device
                ):
                    rv = measure_rv(probe, rec_prompts[pi])
                    intervened_rvs.append(rv)

            if not intervened_rvs:
                continue

            # Effect: how much did intervention move R_V toward baseline?
            d_vs_clean = cohens_d(intervened_rvs,
                                  [r for r in clean_rec_rvs if not np.isnan(r)])
            d_vs_baseline = cohens_d(intervened_rvs,
                                     [r for r in clean_bas_rvs if not np.isnan(r)])

            # Recovery fraction: 0 = no change, 1 = fully moved to baseline
            rec_mean = np.nanmean(clean_rec_rvs)
            bas_mean = np.nanmean(clean_bas_rvs)
            int_mean = np.nanmean(intervened_rvs)
            gap = bas_mean - rec_mean
            recovery = (int_mean - rec_mean) / gap if abs(gap) > 1e-6 else float("nan")

            result = {
                "dims_swapped": list(dim_group),
                "n_dims": len(dim_group),
                "variance_explained": float(explained_var[list(dim_group)].sum()),
                "mean_rv_intervened": float(np.nanmean(intervened_rvs)),
                "d_vs_clean_recursive": float(d_vs_clean),
                "d_vs_clean_baseline": float(d_vs_baseline),
                "recovery_fraction": float(recovery),
                "n_prompts": len(intervened_rvs),
            }
            intervention_results.append(result)

            label = f"dims={list(dim_group)}" if len(dim_group) <= 5 else f"top-{len(dim_group)}"
            print(f"    {label}: R_V={int_mean:.3f}, recovery={recovery:.2f}, "
                  f"d_vs_clean={d_vs_clean:.3f}")

        all_layer_results[layer_idx] = {
            "n_components": int(n_comp),
            "explained_variance_top10": float(explained_var[:10].sum()),
            "explained_variance_top20": float(explained_var[:min(20, len(explained_var))].sum()),
            "interventions": intervention_results,
        }

    # ── Save results ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "experiment": "E2.4_dii_intervention",
        "model": args.model,
        "n_prompts": n,
        "clean_rv": {
            "recursive_mean": float(np.nanmean(clean_rec_rvs)),
            "baseline_mean": float(np.nanmean(clean_bas_rvs)),
            "cohens_d": float(clean_d),
        },
        "target_layers": target_layers,
        "layer_results": {str(k): v for k, v in all_layer_results.items()},
    }

    path = out_dir / f"dii_results_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Saved: {path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DII Intervention (E2.4)")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-prompts", type=int, default=20)
    args = parser.parse_args()
    run_dii_experiment(args)
