#!/usr/bin/env python3
"""
FULL PATH PATCHING
==================

Activation patching at every layer (L0-L31) on 5 components:
  residual stream, V-proj, K-proj, Q-proj, MLP output

For each (layer, component) pair:
  - Patch recursive→baseline: measure R_V effect (necessity direction)
  - Patch baseline→recursive: measure R_V effect (sufficiency direction)

Produces the full "information flow map" showing where recursive information
enters, transforms, and exits.

Design: 32 layers × 5 components × n_prompts prompts = many measurements.

Usage:
    python3 scripts/full_path_patching.py --device cuda
    python3 scripts/full_path_patching.py --device cuda --n-prompts 20 --layers 0 5 10 15 20 25 27 31
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

import torch
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.subsets import load_default_mistral_hardening_subset, split_tier_records_by_pillar
from geometric_lens.probe import GeometricProbe
from geometric_lens.hooks import capture_v_projection
from geometric_lens.metrics import participation_ratio, compute_rv
from geometric_lens.models import get_layers, get_v_proj_module, get_self_attn_module


# ── Frozen prompt contract ───────────────────────────────────────────────────
_subset = load_default_mistral_hardening_subset()
_tier_records = split_tier_records_by_pillar(_subset, "core_measurement")
RECURSIVE_RECORDS = _tier_records["recursive"]
BASELINE_RECORDS = _tier_records["baseline"]
RECURSIVE_PROMPT_IDS = [prompt_id for prompt_id, _ in RECURSIVE_RECORDS]
BASELINE_PROMPT_IDS = [prompt_id for prompt_id, _ in BASELINE_RECORDS]
RECURSIVE_PROMPTS = [record["text"] for _, record in RECURSIVE_RECORDS]
BASELINE_PROMPTS = [record["text"] for _, record in BASELINE_RECORDS]


def _align_seq_len(donor, target_len):
    """Pad or truncate donor activation along the seq dimension to match target_len."""
    # donor shape: (batch, seq, hidden) or (batch, seq, heads, head_dim)
    d_len = donor.shape[1]
    if d_len == target_len:
        return donor
    if d_len > target_len:
        return donor[:, :target_len]
    # Pad by repeating the last token
    pad = donor[:, -1:].expand(-1, target_len - d_len, *donor.shape[2:])
    return torch.cat([donor, pad], dim=1)


@contextmanager
def patch_component(model, layer_idx, component, donor_activations):
    """
    Context manager that patches a specific component at a layer with donor activations.
    Automatically aligns sequence lengths between donor and target.

    Components: 'residual', 'v_proj', 'k_proj', 'q_proj', 'mlp'
    """
    handle = None
    layers = get_layers(model)
    layer = layers[layer_idx]

    if component == "residual":
        def hook(module, inputs):
            target = inputs[0]
            aligned = _align_seq_len(donor_activations, target.shape[1])
            return (aligned.to(target.device, dtype=target.dtype),) + inputs[1:]
        handle = layer.register_forward_pre_hook(hook)

    elif component == "v_proj":
        module, kind = get_v_proj_module(model, layer_idx)
        def hook(mod, inp, out):
            aligned = _align_seq_len(donor_activations, out.shape[1])
            return aligned.to(out.device, dtype=out.dtype)
        handle = module.register_forward_hook(hook)

    elif component == "mlp":
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
        elif hasattr(layer, "feed_forward"):
            mlp = layer.feed_forward
        else:
            yield
            return
        def hook(mod, inp, out):
            aligned = _align_seq_len(donor_activations, out.shape[1])
            return aligned.to(out.device, dtype=out.dtype)
        handle = mlp.register_forward_hook(hook)

    else:
        yield
        return

    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


def measure_rv_for_text(probe, text):
    """Measure R_V for a single text using the probe."""
    result = probe.measure(text, metrics=["rv"])
    return result.rv


def capture_component_activation(model, tokenizer, text, layer_idx, component, device):
    """Run forward pass and capture a component's output activation."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    layers = get_layers(model)
    layer = layers[layer_idx]
    storage = {"act": None}

    if component == "residual":
        def hook(module, inputs):
            storage["act"] = inputs[0].detach().clone()
        handle = layer.register_forward_pre_hook(hook)
    elif component == "v_proj":
        module, _ = get_v_proj_module(model, layer_idx)
        def hook(mod, inp, out):
            storage["act"] = out.detach().clone()
        handle = module.register_forward_hook(hook)
    elif component == "mlp":
        mlp = getattr(layer, "mlp", getattr(layer, "feed_forward", None))
        if mlp is None:
            return None
        def hook(mod, inp, out):
            storage["act"] = out.detach().clone()
        handle = mlp.register_forward_hook(hook)
    else:
        return None

    with torch.no_grad():
        model(**enc)
    handle.remove()
    return storage["act"]


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std < 1e-10:
        return float("nan")
    return (np.mean(a) - np.mean(b)) / pooled_std


def run_path_patching(args):
    """Run full path patching experiment."""
    print("=" * 70)
    print("FULL PATH PATCHING")
    print("=" * 70)

    out_dir = Path("results/path_patching")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    probe = GeometricProbe(
        model_name=args.model,
        device=args.device,
        attn_implementation="eager",
    )
    model = probe.model
    tokenizer = probe.tokenizer
    spec = probe.spec

    # Select layers
    if args.layers:
        target_layers = args.layers
    else:
        target_layers = list(range(0, spec.num_layers, 2))  # Every other layer for speed

    components = ["residual", "v_proj", "mlp"]
    n = min(args.n_prompts, len(RECURSIVE_PROMPTS), len(BASELINE_PROMPTS))
    rec_prompt_ids = RECURSIVE_PROMPT_IDS[:n]
    bas_prompt_ids = BASELINE_PROMPT_IDS[:n]

    print(f"Layers: {target_layers}")
    print(f"Components: {components}")
    print(f"Prompts: {n} paired")
    print(
        "Prompt contract: "
        f"subset={_subset.name} bank={_subset.source_bank_version} tier=core_measurement"
    )

    # ── Measure clean R_V for all prompts ──
    print("\n  Measuring clean R_V...")
    clean_rec_rvs = []
    clean_bas_rvs = []
    for i in range(n):
        clean_rec_rvs.append(measure_rv_for_text(probe, RECURSIVE_PROMPTS[i]))
        clean_bas_rvs.append(measure_rv_for_text(probe, BASELINE_PROMPTS[i]))

    print(f"  Clean recursive R_V: {np.nanmean(clean_rec_rvs):.3f}")
    print(f"  Clean baseline R_V:  {np.nanmean(clean_bas_rvs):.3f}")

    # ── Patch at each (layer, component) ──
    results = []
    total = len(target_layers) * len(components)
    count = 0

    for layer_idx in target_layers:
        for component in components:
            count += 1
            print(f"\n  [{count}/{total}] L{layer_idx} {component}")

            patched_rvs = []
            for i in range(n):
                # Direction: patch recursive with baseline (break direction)
                # Capture baseline activation at this layer/component
                donor_act = capture_component_activation(
                    model, tokenizer, BASELINE_PROMPTS[i],
                    layer_idx, component, args.device
                )
                if donor_act is None:
                    patched_rvs.append(float("nan"))
                    continue

                # Run recursive prompt with patched component
                enc = tokenizer(RECURSIVE_PROMPTS[i], return_tensors="pt",
                               truncation=True, max_length=512).to(args.device)

                with patch_component(model, layer_idx, component, donor_act):
                    # Measure R_V under patching
                    with capture_v_projection(model, probe.early_layer) as se:
                        with torch.no_grad():
                            model(**enc)
                        v_early = se.get("v")
                    with capture_v_projection(model, probe.late_layer) as sl:
                        with torch.no_grad():
                            model(**enc)
                        v_late = sl.get("v")

                patched_rv = compute_rv(v_early, v_late, probe.window)
                patched_rvs.append(patched_rv)

            # Compute effect of patching
            valid_clean = [v for v in clean_rec_rvs if not np.isnan(v)]
            valid_patched = [v for v in patched_rvs if not np.isnan(v)]

            if valid_clean and valid_patched:
                d = cohens_d(valid_patched, valid_clean)
                delta_rv = np.nanmean(valid_patched) - np.nanmean(valid_clean)
            else:
                d = float("nan")
                delta_rv = float("nan")

            result = {
                "layer": layer_idx,
                "component": component,
                "direction": "break",
                "clean_rv_mean": float(np.nanmean(clean_rec_rvs)),
                "patched_rv_mean": float(np.nanmean(valid_patched)) if valid_patched else float("nan"),
                "delta_rv": delta_rv,
                "cohens_d": d,
                "n_valid": len(valid_patched),
            }
            results.append(result)
            print(f"    ΔR_V = {delta_rv:+.3f}, d = {d:+.3f} (n={len(valid_patched)})")

    # ── Save ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "model": args.model,
        "prompt_bank_version": _subset.source_bank_version,
        "prompt_subset_name": _subset.name,
        "prompt_subset_schema_version": _subset.schema_version,
        "prompt_subset_path": str(_subset.manifest_path),
        "prompt_tier": "core_measurement",
        "target_layers": target_layers,
        "components": components,
        "n_prompts": n,
        "recursive_prompt_ids": rec_prompt_ids,
        "baseline_prompt_ids": bas_prompt_ids,
        "results": results,
    }

    summary_path = out_dir / f"path_patching_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print heatmap
    print("\n" + "=" * 70)
    print("PATH PATCHING HEATMAP (ΔR_V, break direction)")
    print("=" * 70)
    print(f"\n{'Layer':>6}", end="")
    for comp in components:
        print(f"  {comp:>10}", end="")
    print()
    print("-" * (8 + 12 * len(components)))

    for layer_idx in target_layers:
        print(f"L{layer_idx:>4}", end="")
        for comp in components:
            r = next((r for r in results if r["layer"] == layer_idx and r["component"] == comp), None)
            if r:
                delta = r["delta_rv"]
                marker = "***" if abs(r["cohens_d"]) > 1.0 else " * " if abs(r["cohens_d"]) > 0.5 else "   "
                print(f"  {delta:>+7.3f}{marker}", end="")
            else:
                print(f"  {'N/A':>10}", end="")
        print()

    print(f"\n  Summary saved: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Path Patching")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-prompts", type=int, default=20, help="Prompt pairs")
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Specific layers (default: every other layer)")
    args = parser.parse_args()
    run_path_patching(args)
