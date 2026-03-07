#!/usr/bin/env python3
"""
CIRCUIT TRACING ANALYSIS (E3.2)
================================

Generates attribution graphs for self-referential vs baseline prompts,
comparing which features and circuits are activated differently.

Approach:
  Primary: Use Anthropic's circuit-tracer library (if available)
  Fallback: Layer-by-layer gradient-based attribution + activation difference
  analysis using our existing infrastructure.

Both methods produce:
  1. Per-layer attribution scores for self-ref vs baseline
  2. Feature importance rankings (which dimensions matter most)
  3. Circuit graph: which layers/heads contribute most to the R_V contraction

Papers integrated:
  - Ameisen et al. 2025 (circuit tracing)
  - Lindsey et al. 2025 (biology of LLMs)

Output: results/circuit_tracing/circuit_trace_<timestamp>.json

Usage:
    python3 scripts/circuit_tracing_analysis.py --device cuda
    python3 scripts/circuit_tracing_analysis.py --device cuda --model google/gemma-2-2b
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

from geometric_lens.probe import GeometricProbe
from geometric_lens.hooks import (
    capture_v_projection,
    capture_hidden_states,
    capture_multi_layer,
)
from geometric_lens.models import get_layers


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


def try_circuit_tracer(model_name, rec_prompts, bas_prompts, device):
    """Attempt to use Anthropic's circuit-tracer library.

    Returns results dict or None if library unavailable.
    """
    try:
        from circuit_tracer import CircuitTracer
        print("  circuit-tracer library found — using Anthropic method")

        tracer = CircuitTracer(model_name, device=device)

        rec_graphs = []
        for text in rec_prompts[:5]:  # Limit to 5 for speed
            graph = tracer.trace(text, top_k=50)
            rec_graphs.append({
                "n_nodes": len(graph.nodes),
                "n_edges": len(graph.edges),
                "top_features": [
                    {"feature_id": n.feature_id, "activation": float(n.activation)}
                    for n in sorted(graph.nodes, key=lambda x: -x.activation)[:20]
                ],
            })

        bas_graphs = []
        for text in bas_prompts[:5]:
            graph = tracer.trace(text, top_k=50)
            bas_graphs.append({
                "n_nodes": len(graph.nodes),
                "n_edges": len(graph.edges),
                "top_features": [
                    {"feature_id": n.feature_id, "activation": float(n.activation)}
                    for n in sorted(graph.nodes, key=lambda x: -x.activation)[:20]
                ],
            })

        return {
            "method": "circuit_tracer",
            "recursive_graphs": rec_graphs,
            "baseline_graphs": bas_graphs,
        }
    except ImportError:
        print("  circuit-tracer not installed — using attribution fallback")
        return None
    except Exception as e:
        print(f"  circuit-tracer failed: {e} — using attribution fallback")
        return None


def compute_activation_attribution(model, tokenizer, prompts, layer_indices, device, window=16):
    """Compute per-layer activation norms and effective dimensions.

    For each prompt, captures hidden states at every target layer and computes:
    - L2 norm of the activation (magnitude of representation)
    - Effective dimension (spectral entropy)
    - Condition-mean activation vector (for later cosine comparison)

    Returns dict keyed by layer_idx with lists of per-prompt stats.
    """
    layer_data = {l: [] for l in layer_indices}

    for text in prompts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

        with capture_multi_layer(model, layer_indices, component="hidden") as storage:
            with torch.no_grad():
                model(**enc)

        for layer_idx in layer_indices:
            hidden = storage.get(layer_idx)
            if hidden is None:
                layer_data[layer_idx].append(None)
                continue

            if hidden.dim() == 3:
                hidden = hidden[0]  # (seq, hidden)

            W = min(window, hidden.shape[0])
            h = hidden[-W:, :].cpu().double()

            # L2 norm
            l2_norm = float(h.norm().item())

            # Effective dimension via SVD
            try:
                _, S, _ = torch.linalg.svd(h.T, full_matrices=False)
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

            # Mean activation vector (for cross-condition comparison)
            mean_act = h.mean(dim=0).numpy()

            layer_data[layer_idx].append({
                "l2_norm": l2_norm,
                "eff_dim": eff_dim,
                "mean_act": mean_act,
            })

    return layer_data


def compute_v_projection_attribution(model, tokenizer, prompts, layer_indices, device, window=16):
    """Compute per-layer V-projection attribution statistics."""
    layer_data = {l: [] for l in layer_indices}

    for text in prompts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

        with capture_multi_layer(model, layer_indices, component="v") as storage:
            with torch.no_grad():
                model(**enc)

        for layer_idx in layer_indices:
            v = storage.get(layer_idx)
            if v is None:
                layer_data[layer_idx].append(None)
                continue

            if v.dim() == 3:
                v = v[0]
            W = min(window, v.shape[0])
            vw = v[-W:, :].cpu().double()

            # V-projection participation ratio
            try:
                _, S, _ = torch.linalg.svd(vw.T, full_matrices=False)
                S_np = S.numpy()
                S_sq = S_np ** 2
                total = S_sq.sum()
                if total > 1e-10:
                    pr = float((total ** 2) / (S_sq ** 2).sum())
                else:
                    pr = float("nan")
            except Exception:
                pr = float("nan")

            layer_data[layer_idx].append({
                "v_pr": pr,
                "v_norm": float(vw.norm().item()),
            })

    return layer_data


def run_circuit_tracing(args):
    """Run circuit tracing analysis."""
    print("=" * 70)
    print("CIRCUIT TRACING ANALYSIS (E3.2)")
    print("=" * 70)

    out_dir = Path("results/circuit_tracing")
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(args.n_prompts, len(RECURSIVE_PROMPTS))
    rec_prompts = RECURSIVE_PROMPTS[:n]
    bas_prompts = BASELINE_PROMPTS[:n]

    # ── Try Anthropic's circuit-tracer first ──
    ct_results = try_circuit_tracer(args.model, rec_prompts, bas_prompts, args.device)

    # ── Attribution fallback: always run for comprehensive data ──
    print(f"\n  Loading model: {args.model}")
    probe = GeometricProbe(
        model_name=args.model,
        device=args.device,
        attn_implementation="eager",
    )
    model = probe.model
    tokenizer = probe.tokenizer
    spec = probe.spec

    # Sample layers across the network
    step = max(1, spec.num_layers // 16)
    layer_indices = list(range(0, spec.num_layers, step))
    if probe.late_layer not in layer_indices:
        layer_indices.append(probe.late_layer)
    if probe.early_layer not in layer_indices:
        layer_indices.append(probe.early_layer)
    layer_indices = sorted(set(layer_indices))

    print(f"  Layers: {layer_indices}")
    print(f"  Prompts: {n} per condition")

    # ── Step 1: Hidden state attribution ──
    print("\n  Step 1: Hidden state attribution across layers...")
    rec_hidden = compute_activation_attribution(
        model, tokenizer, rec_prompts, layer_indices, args.device)
    bas_hidden = compute_activation_attribution(
        model, tokenizer, bas_prompts, layer_indices, args.device)

    # ── Step 2: V-projection attribution ──
    print("  Step 2: V-projection attribution across layers...")
    rec_vproj = compute_v_projection_attribution(
        model, tokenizer, rec_prompts, layer_indices, args.device)
    bas_vproj = compute_v_projection_attribution(
        model, tokenizer, bas_prompts, layer_indices, args.device)

    # ── Step 3: Compute per-layer divergence scores ──
    print("\n  Step 3: Computing attribution scores...")
    attribution_graph = {}

    for layer_idx in layer_indices:
        # Hidden state effective dimension divergence
        rec_dims = [d["eff_dim"] for d in rec_hidden[layer_idx]
                    if d is not None and not np.isnan(d["eff_dim"])]
        bas_dims = [d["eff_dim"] for d in bas_hidden[layer_idx]
                    if d is not None and not np.isnan(d["eff_dim"])]
        d_eff_dim = cohens_d(rec_dims, bas_dims) if rec_dims and bas_dims else float("nan")

        # Hidden state norm divergence
        rec_norms = [d["l2_norm"] for d in rec_hidden[layer_idx] if d is not None]
        bas_norms = [d["l2_norm"] for d in bas_hidden[layer_idx] if d is not None]
        d_norm = cohens_d(rec_norms, bas_norms) if rec_norms and bas_norms else float("nan")

        # V-projection PR divergence
        rec_vprs = [d["v_pr"] for d in rec_vproj[layer_idx]
                    if d is not None and not np.isnan(d["v_pr"])]
        bas_vprs = [d["v_pr"] for d in bas_vproj[layer_idx]
                    if d is not None and not np.isnan(d["v_pr"])]
        d_vpr = cohens_d(rec_vprs, bas_vprs) if rec_vprs and bas_vprs else float("nan")

        # Cross-condition cosine similarity (do representations diverge?)
        rec_means = [d["mean_act"] for d in rec_hidden[layer_idx] if d is not None]
        bas_means = [d["mean_act"] for d in bas_hidden[layer_idx] if d is not None]
        if rec_means and bas_means:
            rec_centroid = np.mean(rec_means, axis=0)
            bas_centroid = np.mean(bas_means, axis=0)
            cos_sim = float(np.dot(rec_centroid, bas_centroid) /
                          (np.linalg.norm(rec_centroid) * np.linalg.norm(bas_centroid) + 1e-10))
        else:
            cos_sim = float("nan")

        attribution_graph[layer_idx] = {
            "d_eff_dim": float(d_eff_dim),
            "d_norm": float(d_norm),
            "d_v_pr": float(d_vpr),
            "centroid_cosine": cos_sim,
            "eff_dim_recursive_mean": float(np.mean(rec_dims)) if rec_dims else float("nan"),
            "eff_dim_baseline_mean": float(np.mean(bas_dims)) if bas_dims else float("nan"),
            "v_pr_recursive_mean": float(np.mean(rec_vprs)) if rec_vprs else float("nan"),
            "v_pr_baseline_mean": float(np.mean(bas_vprs)) if bas_vprs else float("nan"),
        }

        print(f"    L{layer_idx:2d}: d_dim={d_eff_dim:+.2f}  d_vpr={d_vpr:+.2f}  "
              f"cos={cos_sim:.4f}  d_norm={d_norm:+.2f}")

    # ── Step 4: Identify circuit nodes (layers where self-ref diverges most) ──
    print("\n  Step 4: Circuit node identification...")
    sorted_by_divergence = sorted(
        attribution_graph.items(),
        key=lambda x: abs(x[1]["d_eff_dim"]) if not np.isnan(x[1]["d_eff_dim"]) else 0,
        reverse=True
    )

    circuit_nodes = []
    for layer_idx, data in sorted_by_divergence[:5]:
        circuit_nodes.append({
            "layer": layer_idx,
            "depth_frac": layer_idx / spec.num_layers,
            "d_eff_dim": data["d_eff_dim"],
            "d_v_pr": data["d_v_pr"],
            "role": "compression" if data["d_eff_dim"] < 0 else "expansion",
        })
        print(f"    Node L{layer_idx}: {circuit_nodes[-1]['role']} "
              f"(d_dim={data['d_eff_dim']:.2f}, d_vpr={data['d_v_pr']:.2f})")

    # ── Step 5: Graph comparison summary ──
    print("\n  Step 5: Graph comparison...")
    # Compute layer-trajectory statistics
    layer_list = sorted(attribution_graph.keys())
    d_trajectory = [attribution_graph[l]["d_eff_dim"] for l in layer_list]
    vpr_trajectory = [attribution_graph[l]["d_v_pr"] for l in layer_list]
    cos_trajectory = [attribution_graph[l]["centroid_cosine"] for l in layer_list]

    # Find transition point: where does divergence become large?
    transition_layer = None
    for i, l in enumerate(layer_list):
        d = attribution_graph[l]["d_eff_dim"]
        if not np.isnan(d) and abs(d) > 0.5:
            transition_layer = l
            break

    graph_summary = {
        "n_circuit_nodes": len(circuit_nodes),
        "transition_layer": transition_layer,
        "transition_depth": transition_layer / spec.num_layers if transition_layer else None,
        "max_divergence_layer": sorted_by_divergence[0][0] if sorted_by_divergence else None,
        "max_d_eff_dim": sorted_by_divergence[0][1]["d_eff_dim"] if sorted_by_divergence else None,
        "mean_late_cosine": float(np.nanmean([
            attribution_graph[l]["centroid_cosine"]
            for l in layer_list if l > spec.num_layers * 0.7
        ])),
    }
    print(f"    Transition at L{transition_layer} ({transition_layer/spec.num_layers*100:.0f}% depth)"
          if transition_layer else "    No clear transition found")

    # ── Save results ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "experiment": "E3.2_circuit_tracing",
        "model": args.model,
        "n_prompts": n,
        "method": "circuit_tracer" if ct_results else "attribution_fallback",
        "layer_indices": layer_indices,
        "attribution_graph": {str(k): v for k, v in attribution_graph.items()},
        "circuit_nodes": circuit_nodes,
        "graph_summary": graph_summary,
        "d_trajectory": d_trajectory,
        "vpr_trajectory": vpr_trajectory,
        "cos_trajectory": cos_trajectory,
    }

    if ct_results:
        summary["circuit_tracer_results"] = ct_results

    path = out_dir / f"circuit_trace_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Saved: {path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Circuit Tracing Analysis (E3.2)")
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-prompts", type=int, default=20)
    args = parser.parse_args()
    run_circuit_tracing(args)
