#!/usr/bin/env python3
"""
FULL HEAD SWEEP (E2.2)
======================

Extends per-head analysis to ALL layers × ALL heads (32×32 = 1024 heads for Mistral-7B).
Computes per-head:
  1. Attention entropy divergence (recursive vs baseline)
  2. OV effective rank divergence
  3. Cohen's d for each metric per head

Output: 32×32 heatmap data + top-N discriminating heads across full model.
Connects to Ferrando & Voita 2024 (automated information flow routes).

Usage:
    python3 scripts/full_head_sweep.py --device cuda
    python3 scripts/full_head_sweep.py --device cuda --n-prompts 20 --batch-layers 4
"""

import sys
import json
import argparse
import gc
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.subsets import load_default_mistral_hardening_subset, split_tier_records_by_pillar
from geometric_lens.probe import GeometricProbe
from geometric_lens.hooks import capture_attention_patterns, capture_v_projection
from geometric_lens.metrics import participation_ratio
from geometric_lens.models import get_layers, get_self_attn_module, get_v_proj_module


# ── Frozen prompt contract ───────────────────────────────────────────────────
_subset = load_default_mistral_hardening_subset()
_tier_records = split_tier_records_by_pillar(_subset, "core_measurement")
RECURSIVE_RECORDS = _tier_records["recursive"]
BASELINE_RECORDS = _tier_records["baseline"]
RECURSIVE_PROMPT_IDS = [prompt_id for prompt_id, _ in RECURSIVE_RECORDS]
BASELINE_PROMPT_IDS = [prompt_id for prompt_id, _ in BASELINE_RECORDS]
RECURSIVE_PROMPTS = [record["text"] for _, record in RECURSIVE_RECORDS]
BASELINE_PROMPTS = [record["text"] for _, record in BASELINE_RECORDS]


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std < 1e-10:
        return float("nan")
    return (np.mean(a) - np.mean(b)) / pooled_std


def compute_per_head_entropy(attn_weights, n_heads):
    """Compute per-head attention entropy from attention tensor.

    Args:
        attn_weights: (batch, num_heads, seq, seq)
        n_heads: Number of heads

    Returns:
        List of entropy values per head.
    """
    entropies = []
    for h in range(n_heads):
        attn_row = attn_weights[0, h, -1, :].float()
        attn_row = attn_row / (attn_row.sum() + 1e-10)
        log_attn = torch.log(attn_row + 1e-10)
        entropy = -(attn_row * log_attn).sum().item()
        entropies.append(entropy)
    return entropies


def compute_ov_effective_rank(model, layer_idx, enc, n_heads, head_dim, device):
    """Compute OV effective rank per head at a layer.

    Extracts V-projection, reshapes to per-head, and computes effective rank
    from SVD of each head's activations.

    Returns:
        List of effective rank values per head.
    """
    with capture_v_projection(model, layer_idx) as sv:
        with torch.no_grad():
            model(**enc)
        v_tensor = sv.get("v")

    if v_tensor is None:
        return [float("nan")] * n_heads

    # v_tensor: (batch, seq, hidden_size)
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]  # (seq, hidden_size)

    T, D = v_tensor.shape
    window = min(16, T)
    v_window = v_tensor[-window:, :]  # (W, D)

    ranks = []
    for h in range(n_heads):
        start = h * head_dim
        end = start + head_dim
        if end > D:
            ranks.append(float("nan"))
            continue

        v_head = v_window[:, start:end].cpu().double()  # (W, head_dim)
        try:
            _, S, _ = torch.linalg.svd(v_head.T, full_matrices=False)
            S_np = S.numpy()
            S_sq = S_np ** 2
            total = S_sq.sum()
            if total < 1e-10:
                ranks.append(float("nan"))
                continue
            p = S_sq / total
            p = p[p > 1e-10]
            entropy = -np.sum(p * np.log(p))
            ranks.append(float(np.exp(entropy)))
        except Exception:
            ranks.append(float("nan"))

    return ranks


def run_full_head_sweep(args):
    """Run full layer×head sweep."""
    print("=" * 70)
    print("FULL HEAD SWEEP (E2.2)")
    print("=" * 70)

    out_dir = Path("results/full_head_sweep")
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
    n_layers = spec.num_layers
    n_heads = spec.num_heads
    head_dim = spec.head_dim

    print(f"Model loaded. Layers={n_layers}, Heads={n_heads}, HeadDim={head_dim}")

    n = min(args.n_prompts, len(RECURSIVE_PROMPTS), len(BASELINE_PROMPTS))
    rec_prompts = RECURSIVE_PROMPTS[:n]
    bas_prompts = BASELINE_PROMPTS[:n]
    rec_prompt_ids = RECURSIVE_PROMPT_IDS[:n]
    bas_prompt_ids = BASELINE_PROMPT_IDS[:n]

    print(
        "Prompt contract: "
        f"subset={_subset.name} bank={_subset.source_bank_version} tier=core_measurement"
    )

    # Storage: [layer][head][condition] = list of values
    entropy_data = defaultdict(lambda: defaultdict(lambda: {"recursive": [], "baseline": []}))
    rank_data = defaultdict(lambda: defaultdict(lambda: {"recursive": [], "baseline": []}))

    # Process layers in batches to manage memory
    layer_batch_size = args.batch_layers
    layer_batches = [list(range(i, min(i + layer_batch_size, n_layers)))
                     for i in range(0, n_layers, layer_batch_size)]

    total_batches = len(layer_batches)
    t0 = time.time()

    for batch_idx, layer_batch in enumerate(layer_batches):
        print(f"\n  Layer batch [{batch_idx+1}/{total_batches}]: layers {layer_batch[0]}-{layer_batch[-1]}")

        for condition, prompts in [("recursive", rec_prompts), ("baseline", bas_prompts)]:
            print(f"    {condition} ({len(prompts)} prompts)...")

            for pi, text in enumerate(prompts):
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(args.device)

                for layer_idx in layer_batch:
                    # Attention entropy
                    with capture_attention_patterns(model, layer_idx) as sa:
                        with torch.no_grad():
                            model(**enc, output_attentions=True)
                        attn_weights = sa.get("attn_weights")

                    if attn_weights is not None:
                        head_entropies = compute_per_head_entropy(attn_weights, n_heads)
                        for h, ent in enumerate(head_entropies):
                            entropy_data[layer_idx][h][condition].append(ent)

                    # OV effective rank
                    head_ranks = compute_ov_effective_rank(
                        model, layer_idx, enc, n_heads, head_dim, args.device
                    )
                    for h, rank in enumerate(head_ranks):
                        rank_data[layer_idx][h][condition].append(rank)

                if (pi + 1) % 10 == 0:
                    print(f"      [{pi+1}/{len(prompts)}]")

        # Save incremental checkpoint
        elapsed = time.time() - t0
        print(f"    Batch done. Elapsed: {elapsed/60:.1f}min")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Compute per-head statistics ──
    print("\n" + "=" * 70)
    print("COMPUTING HEATMAP DATA")
    print("=" * 70)

    # Build heatmap matrices
    entropy_d_matrix = np.full((n_layers, n_heads), float("nan"))
    entropy_p_matrix = np.full((n_layers, n_heads), float("nan"))
    rank_d_matrix = np.full((n_layers, n_heads), float("nan"))
    rank_p_matrix = np.full((n_layers, n_heads), float("nan"))

    head_results = []

    for layer_idx in range(n_layers):
        for h in range(n_heads):
            ent_rec = entropy_data[layer_idx][h]["recursive"]
            ent_bas = entropy_data[layer_idx][h]["baseline"]
            rank_rec = [v for v in rank_data[layer_idx][h]["recursive"] if not np.isnan(v)]
            rank_bas = [v for v in rank_data[layer_idx][h]["baseline"] if not np.isnan(v)]

            # Entropy Cohen's d
            if len(ent_rec) >= 3 and len(ent_bas) >= 3:
                d_ent = cohens_d(ent_rec, ent_bas)
                _, p_ent = stats.mannwhitneyu(ent_rec, ent_bas, alternative="two-sided")
            else:
                d_ent, p_ent = float("nan"), float("nan")

            # OV rank Cohen's d
            if len(rank_rec) >= 3 and len(rank_bas) >= 3:
                d_rank = cohens_d(rank_rec, rank_bas)
                _, p_rank = stats.mannwhitneyu(rank_rec, rank_bas, alternative="two-sided")
            else:
                d_rank, p_rank = float("nan"), float("nan")

            entropy_d_matrix[layer_idx, h] = d_ent
            entropy_p_matrix[layer_idx, h] = p_ent
            rank_d_matrix[layer_idx, h] = d_rank
            rank_p_matrix[layer_idx, h] = p_rank

            head_results.append({
                "layer": layer_idx,
                "head": h,
                "entropy_d": d_ent,
                "entropy_p": p_ent,
                "rank_d": d_rank,
                "rank_p": p_rank,
                "entropy_recursive_mean": float(np.mean(ent_rec)) if ent_rec else float("nan"),
                "entropy_baseline_mean": float(np.mean(ent_bas)) if ent_bas else float("nan"),
                "rank_recursive_mean": float(np.mean(rank_rec)) if rank_rec else float("nan"),
                "rank_baseline_mean": float(np.mean(rank_bas)) if rank_bas else float("nan"),
            })

    # ── Print top discriminating heads ──
    sorted_by_entropy = sorted(head_results,
                                key=lambda r: abs(r["entropy_d"]) if not np.isnan(r["entropy_d"]) else 0,
                                reverse=True)
    sorted_by_rank = sorted(head_results,
                             key=lambda r: abs(r["rank_d"]) if not np.isnan(r["rank_d"]) else 0,
                             reverse=True)

    print("\n  Top 20 heads by |entropy divergence|:")
    print(f"  {'L.H':>6} {'d_ent':>8} {'p_ent':>10} {'d_rank':>8} {'p_rank':>10}")
    print("  " + "-" * 50)
    for r in sorted_by_entropy[:20]:
        sig = " ***" if abs(r["entropy_d"]) > 1.0 else " *" if abs(r["entropy_d"]) > 0.5 else ""
        print(f"  L{r['layer']:02d}.H{r['head']:02d} "
              f"{r['entropy_d']:>8.3f} {r['entropy_p']:>10.4f} "
              f"{r['rank_d']:>8.3f} {r['rank_p']:>10.4f}{sig}")

    print("\n  Top 20 heads by |OV rank divergence|:")
    for r in sorted_by_rank[:20]:
        sig = " ***" if abs(r["rank_d"]) > 1.0 else " *" if abs(r["rank_d"]) > 0.5 else ""
        print(f"  L{r['layer']:02d}.H{r['head']:02d} "
              f"{r['rank_d']:>8.3f} {r['rank_p']:>10.4f} "
              f"{r['entropy_d']:>8.3f} {r['entropy_p']:>10.4f}{sig}")

    # Layer-average statistics
    print("\n  Per-layer average |d| (entropy):")
    for layer_idx in range(n_layers):
        row = entropy_d_matrix[layer_idx]
        valid = row[~np.isnan(row)]
        if len(valid) > 0:
            avg_abs_d = np.mean(np.abs(valid))
            n_sig = np.sum(np.abs(valid) > 0.5)
            marker = " <<<" if avg_abs_d > 0.3 else ""
            print(f"    L{layer_idx:02d}: avg|d|={avg_abs_d:.3f}, n_sig={n_sig:>3}{marker}")

    # ── Save results ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "experiment": "E2.2_full_head_sweep",
        "model": args.model,
        "prompt_bank_version": _subset.source_bank_version,
        "prompt_subset_name": _subset.name,
        "prompt_subset_schema_version": _subset.schema_version,
        "prompt_subset_path": str(_subset.manifest_path),
        "prompt_tier": "core_measurement",
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_recursive_prompts": len(rec_prompts),
        "n_baseline_prompts": len(bas_prompts),
        "recursive_prompt_ids": rec_prompt_ids,
        "baseline_prompt_ids": bas_prompt_ids,
        "head_results": head_results,
        "top_20_entropy_heads": [
            {"layer": r["layer"], "head": r["head"], "d": r["entropy_d"], "p": r["entropy_p"]}
            for r in sorted_by_entropy[:20]
        ],
        "top_20_rank_heads": [
            {"layer": r["layer"], "head": r["head"], "d": r["rank_d"], "p": r["rank_p"]}
            for r in sorted_by_rank[:20]
        ],
        "heatmap_entropy_d": entropy_d_matrix.tolist(),
        "heatmap_entropy_p": entropy_p_matrix.tolist(),
        "heatmap_rank_d": rank_d_matrix.tolist(),
        "heatmap_rank_p": rank_p_matrix.tolist(),
    }

    summary_path = out_dir / f"full_head_sweep_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Summary saved: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Head Sweep (E2.2)")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-prompts", type=int, default=20, help="Prompts per condition")
    parser.add_argument("--batch-layers", type=int, default=4,
                        help="Number of layers to process per batch (memory management)")
    args = parser.parse_args()
    run_full_head_sweep(args)
