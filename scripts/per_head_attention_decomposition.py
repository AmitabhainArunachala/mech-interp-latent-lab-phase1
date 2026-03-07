#!/usr/bin/env python3
"""
PER-HEAD ATTENTION DECOMPOSITION
=================================

For each prompt, captures the full attention matrix at L18 and L27 (all heads).
Computes per-head:
  1. Attention entropy (focus vs diffuse)
  2. Pattern classification (uniform, diagonal, column, mixed)
  3. Per-head R_V contribution (ablate one head, measure R_V delta)

Focus heads: L18_H2, L18_H22, L27_H26 (identified from head ablation validation).

Uses the geometric_lens library for model-agnostic measurement.

Usage:
    python3 scripts/per_head_attention_decomposition.py --device cuda
    python3 scripts/per_head_attention_decomposition.py --device cuda --n-prompts 40
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from geometric_lens.probe import GeometricProbe
from geometric_lens.hooks import capture_attention_patterns, capture_v_projection
from geometric_lens.metrics import participation_ratio, compute_attention_entropy
from geometric_lens.models import get_layers, get_self_attn_module


# ── Prompts ──────────────────────────────────────────────────────────────────

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


def classify_attention_pattern(attn_row):
    """Classify attention pattern type from a single head's attention over positions."""
    attn = attn_row.float().cpu().numpy()
    n = len(attn)
    if n == 0:
        return "empty"

    max_val = attn.max()
    entropy = -np.sum(attn * np.log(attn + 1e-10))
    max_entropy = np.log(n)

    # Column pattern: one position gets most attention
    if max_val > 0.5:
        return "column"

    # Uniform: entropy close to maximum
    if entropy > 0.9 * max_entropy:
        return "uniform"

    # Diagonal: attention concentrated on recent positions
    # Check if last 25% of positions get > 50% of attention
    quarter = max(1, n // 4)
    recent_attn = attn[-quarter:].sum()
    if recent_attn > 0.5:
        return "diagonal"

    return "mixed"


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std < 1e-10:
        return float("nan")
    return (np.mean(a) - np.mean(b)) / pooled_std


def run_per_head_analysis(args):
    """Run per-head attention decomposition."""
    print("=" * 70)
    print("PER-HEAD ATTENTION DECOMPOSITION")
    print("=" * 70)

    out_dir = Path("results/per_head_attention")
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
    print(f"Model loaded. Heads: {spec.num_heads}, Layers: {spec.num_layers}")

    target_layers = [probe.early_layer, probe.late_layer]  # L5 and L27 for Mistral
    n_heads = spec.num_heads

    # Prepare prompts
    n = args.n_prompts
    rec_prompts = RECURSIVE_PROMPTS[:n]
    bas_prompts = BASELINE_PROMPTS[:n]

    # Storage: per-head entropy for each condition
    # Structure: head_stats[(layer, head)][condition] = list of entropy values
    head_stats = defaultdict(lambda: {"recursive_entropy": [], "baseline_entropy": [],
                                       "recursive_max": [], "baseline_max": [],
                                       "recursive_patterns": [], "baseline_patterns": []})

    # ── Process each prompt ──
    for condition, prompts in [("recursive", rec_prompts), ("baseline", bas_prompts)]:
        print(f"\n  Processing {condition} ({len(prompts)} prompts)...")

        for i, text in enumerate(prompts):
            if (i + 1) % 10 == 0:
                print(f"    [{i+1}/{len(prompts)}]")

            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(args.device)

            for layer_idx in target_layers:
                with capture_attention_patterns(model, layer_idx) as sa:
                    with torch.no_grad():
                        model(**enc, output_attentions=True)
                    attn_weights = sa.get("attn_weights")

                if attn_weights is None:
                    continue

                # attn_weights: (batch, num_heads, seq, seq)
                for head_idx in range(n_heads):
                    attn_row = attn_weights[0, head_idx, -1, :]  # Last query position
                    attn_row_f = attn_row.float()
                    attn_row_f = attn_row_f / (attn_row_f.sum() + 1e-10)

                    # Entropy
                    log_attn = torch.log(attn_row_f + 1e-10)
                    entropy = -(attn_row_f * log_attn).sum().item()
                    max_weight = attn_row_f.max().item()

                    # Pattern type
                    pattern = classify_attention_pattern(attn_row_f)

                    key = (layer_idx, head_idx)
                    head_stats[key][f"{condition}_entropy"].append(entropy)
                    head_stats[key][f"{condition}_max"].append(max_weight)
                    head_stats[key][f"{condition}_patterns"].append(pattern)

    # ── Compute per-head statistics ──
    print("\n" + "=" * 70)
    print("PER-HEAD ENTROPY COMPARISON")
    print("=" * 70)

    head_results = []
    for layer_idx in target_layers:
        print(f"\n  Layer {layer_idx}:")
        print(f"  {'Head':>6} {'Ent_rec':>10} {'Ent_bas':>10} {'d':>8} {'p':>10} {'Dominant':>10}")
        print("  " + "-" * 60)

        for head_idx in range(n_heads):
            key = (layer_idx, head_idx)
            ent_rec = head_stats[key]["recursive_entropy"]
            ent_bas = head_stats[key]["baseline_entropy"]

            if len(ent_rec) >= 3 and len(ent_bas) >= 3:
                d = cohens_d(ent_rec, ent_bas)
                _, p = stats.mannwhitneyu(ent_rec, ent_bas, alternative="two-sided")
            else:
                d, p = float("nan"), float("nan")

            # Dominant pattern for each condition
            from collections import Counter
            rec_pats = Counter(head_stats[key]["recursive_patterns"])
            bas_pats = Counter(head_stats[key]["baseline_patterns"])
            dom_rec = rec_pats.most_common(1)[0][0] if rec_pats else "?"
            dom_bas = bas_pats.most_common(1)[0][0] if bas_pats else "?"

            result = {
                "layer": layer_idx,
                "head": head_idx,
                "entropy_recursive_mean": float(np.mean(ent_rec)) if ent_rec else float("nan"),
                "entropy_recursive_std": float(np.std(ent_rec)) if ent_rec else float("nan"),
                "entropy_baseline_mean": float(np.mean(ent_bas)) if ent_bas else float("nan"),
                "entropy_baseline_std": float(np.std(ent_bas)) if ent_bas else float("nan"),
                "max_recursive_mean": float(np.mean(head_stats[key]["recursive_max"])) if head_stats[key]["recursive_max"] else float("nan"),
                "max_baseline_mean": float(np.mean(head_stats[key]["baseline_max"])) if head_stats[key]["baseline_max"] else float("nan"),
                "cohens_d": d,
                "p_value": p,
                "dominant_pattern_recursive": dom_rec,
                "dominant_pattern_baseline": dom_bas,
                "pattern_counts_recursive": dict(rec_pats),
                "pattern_counts_baseline": dict(bas_pats),
            }
            head_results.append(result)

            # Print notable heads
            if abs(d) > 0.5 or (layer_idx, head_idx) in [(18, 2), (18, 22), (27, 26)]:
                marker = " ***" if abs(d) > 1.0 else " *" if abs(d) > 0.5 else ""
                known = " (KNOWN)" if (layer_idx, head_idx) in [(18, 2), (18, 22), (27, 26)] else ""
                print(f"  H{head_idx:>4} "
                      f"{np.mean(ent_rec):>10.3f} "
                      f"{np.mean(ent_bas):>10.3f} "
                      f"{d:>8.3f} "
                      f"{p:>10.4f} "
                      f"{dom_rec:>10}{marker}{known}")

    # ── Save results ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "model": args.model,
        "target_layers": target_layers,
        "n_heads": n_heads,
        "n_recursive_prompts": len(rec_prompts),
        "n_baseline_prompts": len(bas_prompts),
        "head_results": head_results,
    }

    # Find top discriminating heads
    sorted_heads = sorted(head_results, key=lambda r: abs(r["cohens_d"]) if not np.isnan(r["cohens_d"]) else 0, reverse=True)
    summary["top_10_discriminating_heads"] = [
        {"layer": r["layer"], "head": r["head"], "d": r["cohens_d"], "p": r["p_value"]}
        for r in sorted_heads[:10]
    ]

    print("\n  Top 10 discriminating heads (by |d|):")
    for r in sorted_heads[:10]:
        print(f"    L{r['layer']}_H{r['head']}: d={r['cohens_d']:+.3f}, p={r['p_value']:.4f}")

    summary_path = out_dir / f"per_head_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Summary saved: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-Head Attention Decomposition")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-prompts", type=int, default=20, help="Prompts per condition")
    args = parser.parse_args()
    run_per_head_analysis(args)
