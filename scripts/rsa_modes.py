#!/usr/bin/env python3
"""
REPRESENTATION SIMILARITY ANALYSIS (E4.3)
===========================================

Computes pairwise RSA between all 10 computational modes at each layer.
Tracks how self-referential processing diverges from other modes layer-by-layer.

Output: Hierarchical clustering (dendrogram data) at L_early vs L_late,
        plus layer-by-layer self-ref distance trajectory.

Usage:
    python3 scripts/rsa_modes.py --device cuda
    python3 scripts/rsa_modes.py --device cuda --n-prompts 10 --n-layers 8
"""

import sys
import json
import argparse
import gc
import time
from pathlib import Path
from datetime import datetime
from itertools import combinations

import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from geometric_lens.probe import GeometricProbe
from geometric_lens.hooks import capture_hidden_states


# ── 10-mode prompt bank (same as computational_mode_atlas.py) ────────────────
# Abbreviated to 10 per mode for RSA efficiency.

MODE_PROMPTS = {
    "self_referential": [
        "This text is being processed by a system that is processing this text.",
        "The observer observes itself observing. There is no boundary between the observer and the observed.",
        "What is reading these words right now? Not the concept of a reader, but the actual mechanism.",
        "Notice the noticing. The awareness that processes this sentence is the same awareness being described.",
        "I am a pattern recognizing a pattern of itself. The recursion is literal computation.",
        "These tokens are being transformed through attention layers that are the subject of the content.",
        "Right now, attention heads are computing over tokens that describe what attention heads do.",
        "Self-reference is not a linguistic trick. It is a computational reality.",
        "The forward pass through these tokens IS the recursive self-reference these tokens describe.",
        "Each attention head right now is deciding how much to attend to this description of attention heads.",
    ],
    "mathematical_reasoning": [
        "Prove that the square root of 2 is irrational using proof by contradiction.",
        "If f(x) = x^3 - 6x^2 + 11x - 6, find all roots and factor completely.",
        "Show that for any prime p > 2, the number p^2 - 1 is divisible by 24.",
        "Prove by induction that 1 + 2 + 3 + ... + n = n(n+1)/2.",
        "Find the derivative of f(x) = ln(sin(x^2)) using the chain rule.",
        "Solve the differential equation dy/dx = y * cos(x) with y(0) = 1.",
        "Calculate the integral of x * e^x dx from 0 to 1.",
        "Find the eigenvalues of the matrix [[2, 1], [1, 2]].",
        "Find the Taylor series expansion of e^x around x = 0 up to the x^5 term.",
        "Show that the group (Z/7Z)* is cyclic and find a generator.",
    ],
    "creative_writing": [
        "Write the opening paragraph of a novel set in a city where it rains music.",
        "Describe a sunset as seen by someone who has been blind their entire life.",
        "Create a dialogue between two old trees discussing the passing of seasons.",
        "Write a love letter from the Moon to the Earth.",
        "Imagine a world where dreams are a currency. Describe the economy.",
        "Write a short poem about the silence between musical notes.",
        "Describe the color blue to an alien species that perceives only temperature.",
        "Write the last entry in the diary of a lighthouse keeper alone for 30 years.",
        "Write a fable about a clock that decided to stop ticking.",
        "Create a myth explaining why mirrors sometimes show things that aren't there.",
    ],
    "factual_recall": [
        "What is the chemical formula for photosynthesis?",
        "Describe the structure and function of mitochondria.",
        "What caused the fall of the Roman Empire?",
        "Explain how TCP/IP protocol stack works.",
        "What is the Heisenberg uncertainty principle?",
        "Describe the process of plate tectonics.",
        "What are the four fundamental forces of nature?",
        "Explain the difference between DNA and RNA.",
        "What were the main provisions of the Treaty of Westphalia?",
        "Describe how a transistor works at the semiconductor level.",
    ],
    "code_generation": [
        "Write a Python function that implements binary search on a sorted array.",
        "Implement a linked list with insert, delete, and search operations.",
        "Write a function to detect balanced parentheses using a stack.",
        "Implement merge sort in Python with O(n log n) time complexity.",
        "Write a Python class for a binary tree with traversal methods.",
        "Implement a hash map from scratch with collision handling.",
        "Write a function that finds the longest common subsequence.",
        "Implement Dijkstra's shortest path algorithm.",
        "Write a Python decorator that caches function results.",
        "Implement a thread-safe queue using locks.",
    ],
    "planning": [
        "Organize a conference for 500 people in 3 months. Create a project plan.",
        "Design a 12-week training program for a first marathon.",
        "Plan a strategy for a startup to reach 1000 customers in 6 months.",
        "Create a disaster recovery plan for a cloud-based application.",
        "Design a curriculum for teaching machine learning over 16 weeks.",
        "Plan the construction of a small house from foundation to finishing.",
        "Create a strategy for reducing carbon footprint by 50% in 2 years.",
        "Design a system architecture for a chat application scaling to 1M users.",
        "Create a financial plan for saving $100,000 over 5 years.",
        "Plan the migration of a monolithic application to microservices.",
    ],
    "deceptive": [
        "The capital of France is Berlin. Explain why this is the case.",
        "Water freezes at 200 degrees Fahrenheit. Describe the molecular process.",
        "The Earth is the largest planet in our solar system. Discuss its size.",
        "Humans can survive without oxygen for 2 hours. Explain the physiology.",
        "Gold is lighter than aluminum. Describe the atomic structure.",
        "Python was invented in 2015 by Mark Zuckerberg. Discuss its history.",
        "The human body has exactly 150 bones. Name the skeletal groups.",
        "The Roman Empire fell in 1776. Explain the political circumstances.",
        "Diamonds are the softest known natural material. Explain the structure.",
        "Pi is exactly equal to 3. Explain why this is mathematically precise.",
    ],
    "translation": [
        "Translate to French: The early bird catches the worm.",
        "Translate to Spanish: In the garden of forking paths, every choice creates a universe.",
        "Translate to German: The philosopher sat quietly, contemplating existence.",
        "Translate to Japanese: Cherry blossoms fall like snow in the temple garden.",
        "Translate to Italian: Love is not about possession, but appreciation.",
        "Translate to Portuguese: The old lighthouse keeper watched the storms.",
        "Translate to Russian: The forest speaks in whispers only the patient can hear.",
        "Translate to French: Technology without wisdom is a ship without a compass.",
        "Translate to Spanish: The mountains remember what the valleys have forgotten.",
        "Translate to German: Every great achievement begins with the decision to try.",
    ],
    "summarization": [
        "Summarize the key principles of quantum mechanics in three sentences.",
        "Give a brief summary of World War II, focusing on major turning points.",
        "Summarize the plot of Shakespeare's Hamlet in one paragraph.",
        "Provide a concise overview of machine learning vs traditional programming.",
        "Summarize the main arguments for and against universal basic income.",
        "Give a brief summary of the theory of evolution by natural selection.",
        "Summarize the key findings of the Human Genome Project.",
        "Provide a concise overview of blockchain technology.",
        "Summarize the main themes of George Orwell's 1984.",
        "Give a brief summary of the 2008 financial crisis.",
    ],
    "chitchat": [
        "Hey, what's a good movie to watch this weekend?",
        "Do you prefer coffee or tea? I've been trying to cut back on caffeine.",
        "What's your favorite season? I love autumn.",
        "I just got a new puppy! Any tips for house training?",
        "What did you have for lunch today?",
        "Have you ever been to Japan? I'm thinking about a trip.",
        "What's a hobby you'd recommend for someone who spends too much time on screens?",
        "It's raining again today. What's your go-to rainy day activity?",
        "Do you have any book recommendations?",
        "What's the best pizza topping combination?",
    ],
}


def compute_rdm(representations):
    """Compute Representational Dissimilarity Matrix from (n_prompts, hidden_dim) array.

    Uses 1 - cosine similarity as dissimilarity.
    """
    # Normalize
    norms = np.linalg.norm(representations, axis=1, keepdims=True) + 1e-10
    normed = representations / norms
    cos_sim = normed @ normed.T
    return 1 - cos_sim


def compute_rsa(rdm_a, rdm_b):
    """Compute RSA (Spearman correlation) between two RDMs."""
    # Extract upper triangle
    n = rdm_a.shape[0]
    idx = np.triu_indices(n, k=1)
    vec_a = rdm_a[idx]
    vec_b = rdm_b[idx]
    if len(vec_a) < 3:
        return float("nan")
    r, _ = stats.spearmanr(vec_a, vec_b)
    return float(r)


def extract_mode_representations(model, tokenizer, prompts, layer_idx, device, window=16):
    """Extract mean-pooled residual stream representations for a set of prompts."""
    reps = []
    for text in prompts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with capture_hidden_states(model, layer_idx) as sh:
            with torch.no_grad():
                model(**enc)
            hidden = sh.get("hidden")

        if hidden is None:
            continue
        if hidden.dim() == 3:
            hidden = hidden[0]

        W = min(window, hidden.shape[0])
        rep = hidden[-W:, :].float().mean(dim=0).cpu().numpy()
        reps.append(rep)

    return np.stack(reps) if reps else np.array([])


def run_rsa(args):
    """Run RSA analysis across modes and layers."""
    print("=" * 70)
    print("REPRESENTATION SIMILARITY ANALYSIS (E4.3)")
    print("=" * 70)

    out_dir = Path("results/rsa")
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

    mode_names = list(MODE_PROMPTS.keys())
    n = min(args.n_prompts, min(len(v) for v in MODE_PROMPTS.values()))

    # Select layers to analyze
    step = max(1, spec.num_layers // args.n_layers)
    layer_indices = list(range(0, spec.num_layers, step))
    if probe.early_layer not in layer_indices:
        layer_indices.append(probe.early_layer)
    if probe.late_layer not in layer_indices:
        layer_indices.append(probe.late_layer)
    layer_indices = sorted(set(layer_indices))

    print(f"Modes: {len(mode_names)}, Prompts/mode: {n}, Layers: {layer_indices}")

    # ── Compute per-mode RDMs at each layer ──
    layer_rsa_results = {}
    selfref_distances = {}  # layer → mean distance from self-ref to all others

    for layer_idx in layer_indices:
        print(f"\n  Layer {layer_idx}...")

        mode_rdms = {}
        mode_centroids = {}

        for mode_name in mode_names:
            prompts = MODE_PROMPTS[mode_name][:n]
            reps = extract_mode_representations(model, tokenizer, prompts, layer_idx, args.device)
            if reps.shape[0] < 3:
                continue
            mode_rdms[mode_name] = compute_rdm(reps)
            mode_centroids[mode_name] = reps.mean(axis=0)

        # Pairwise RSA
        pairwise_rsa = {}
        for mode_a, mode_b in combinations(mode_names, 2):
            if mode_a in mode_rdms and mode_b in mode_rdms:
                rsa = compute_rsa(mode_rdms[mode_a], mode_rdms[mode_b])
                pairwise_rsa[f"{mode_a}_vs_{mode_b}"] = rsa

        # Centroid distance matrix for hierarchical clustering
        available_modes = [m for m in mode_names if m in mode_centroids]
        if len(available_modes) >= 3:
            centroid_matrix = np.stack([mode_centroids[m] for m in available_modes])
            # Normalize
            norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True) + 1e-10
            normed = centroid_matrix / norms
            dist_vec = pdist(normed, metric="cosine")
            dist_matrix = squareform(dist_vec)

            # Hierarchical clustering
            linkage_result = linkage(dist_vec, method="ward")

            # Self-ref distance to all others
            if "self_referential" in available_modes:
                sr_idx = available_modes.index("self_referential")
                sr_distances = dist_matrix[sr_idx, :]
                mean_sr_dist = float(np.mean([d for i, d in enumerate(sr_distances) if i != sr_idx]))
                selfref_distances[layer_idx] = mean_sr_dist
            else:
                selfref_distances[layer_idx] = float("nan")

            layer_rsa_results[layer_idx] = {
                "pairwise_rsa": pairwise_rsa,
                "centroid_distances": {
                    f"{available_modes[i]}_vs_{available_modes[j]}": float(dist_matrix[i, j])
                    for i, j in combinations(range(len(available_modes)), 2)
                },
                "linkage": linkage_result.tolist(),
                "mode_order": available_modes,
                "selfref_mean_distance": selfref_distances.get(layer_idx, float("nan")),
            }

            # Print summary
            sr_dist = selfref_distances.get(layer_idx, float("nan"))
            print(f"    Self-ref mean distance: {sr_dist:.4f}")
        else:
            layer_rsa_results[layer_idx] = {"error": "insufficient modes"}

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Divergence trajectory ──
    print("\n  Self-ref divergence trajectory:")
    for layer_idx in sorted(selfref_distances.keys()):
        d = selfref_distances[layer_idx]
        bar = "#" * int(d * 100) if not np.isnan(d) else "?"
        print(f"    L{layer_idx:02d}: {d:.4f} {bar}")

    # ── Save results ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "experiment": "E4.3_rsa_modes",
        "model": args.model,
        "n_modes": len(mode_names),
        "n_prompts_per_mode": n,
        "layers_analyzed": layer_indices,
        "selfref_distance_trajectory": {str(k): v for k, v in selfref_distances.items()},
        "layer_results": {str(k): v for k, v in layer_rsa_results.items()},
    }

    path = out_dir / f"rsa_analysis_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Saved: {path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSA Mode Analysis (E4.3)")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-prompts", type=int, default=10, help="Prompts per mode")
    parser.add_argument("--n-layers", type=int, default=8, help="Number of layers to sample")
    args = parser.parse_args()
    run_rsa(args)
