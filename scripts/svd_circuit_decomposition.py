#!/usr/bin/env python3
"""
SVD CIRCUIT DECOMPOSITION (E2.1 + E2.3)
=========================================

Per-head OV singular direction decomposition for identified circuit heads.
For each head: SVD of OV activations, compare spectra between recursive/baseline,
then project top singular directions to vocabulary space for interpretability.

Directly replicates Gupta et al. NeurIPS 2025 "Beyond Components" methodology.

Output: results/svd_circuits/svd_decomposition_<timestamp>.json

Usage:
    python3 scripts/svd_circuit_decomposition.py --device cuda
"""

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.subsets import load_default_mistral_hardening_subset, split_tier_records_by_pillar
from geometric_lens.probe import GeometricProbe
from geometric_lens.hooks import capture_v_projection
from geometric_lens.models import get_layers, get_self_attn_module


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


def get_per_head_v(model, tokenizer, text, layer_idx, n_heads, head_dim, device, window=16):
    """Get per-head V-projection activations.

    NOTE: For GQA models (e.g. Mistral), the V-projection has fewer heads
    than Q (num_key_value_heads < num_attention_heads). Each KV head is
    shared by (num_attention_heads / num_key_value_heads) Q heads.
    We return the KV head corresponding to each Q head index.
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with capture_v_projection(model, layer_idx) as sv:
        with torch.no_grad():
            model(**enc)
        v = sv.get("v")

    if v is None:
        return [None] * n_heads

    if v.dim() == 3:
        v = v[0]  # (seq, hidden)

    T, D = v.shape
    W = min(window, T)
    v_window = v[-W:, :]

    # Detect GQA: V-proj dim may be smaller than n_heads * head_dim
    n_kv_heads = D // head_dim
    q_per_kv = n_heads // max(n_kv_heads, 1)  # e.g. 32/8 = 4 for Mistral

    heads = []
    for h in range(n_heads):
        # Map Q head → KV head
        kv_h = h // q_per_kv if q_per_kv > 0 else h
        start = kv_h * head_dim
        end = start + head_dim
        if end > D:
            heads.append(None)
        else:
            heads.append(v_window[:, start:end].cpu().double())
    return heads


def svd_spectrum(v_head):
    """Compute SVD spectrum for a single head's V-projection."""
    if v_head is None:
        return None
    try:
        U, S, Vt = torch.linalg.svd(v_head.T, full_matrices=False)
        return {"U": U, "S": S, "Vt": Vt, "S_np": S.numpy()}
    except Exception:
        return None


def vocab_projection(U_col, model, layer_idx, head_idx, head_dim):
    """Project a singular direction to vocabulary space.

    Gets top-10 tokens that this direction "reads" from / "writes" to.
    Handles GQA by mapping Q head index to KV head index.
    """
    try:
        # Get the output projection weight: W_o for this head
        layers = get_layers(model)
        layer = layers[layer_idx]

        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            W_o = layer.self_attn.o_proj.weight.data  # (hidden, hidden)
        else:
            return {"error": "no o_proj found"}

        # Get the unembedding matrix
        if hasattr(model, "lm_head"):
            W_u = model.lm_head.weight.data  # (vocab, hidden)
        else:
            return {"error": "no lm_head found"}

        # The direction in V-space maps through W_o to residual stream
        # Then through W_u to vocabulary
        # U_col is (head_dim,) — a direction in per-head V space
        # For GQA, map Q head → KV head for V-space indexing
        n_heads = model.config.num_attention_heads
        n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
        q_per_kv = n_heads // max(n_kv_heads, 1)
        kv_head_idx = head_idx // q_per_kv
        start = kv_head_idx * head_dim
        end = start + head_dim

        full_dir = torch.zeros(W_o.shape[1], device=W_o.device, dtype=W_o.dtype)
        U_col_device = U_col.float().to(W_o.device)
        full_dir[start:end] = U_col_device[:head_dim]

        # Project through O and unembedding
        residual_dir = W_o @ full_dir  # (hidden,)
        vocab_scores = W_u @ residual_dir  # (vocab,)

        top_vals, top_idx = torch.topk(vocab_scores, 10)
        bot_vals, bot_idx = torch.topk(-vocab_scores, 10)

        return {
            "top_token_ids": top_idx.tolist(),
            "top_scores": top_vals.tolist(),
            "bottom_token_ids": bot_idx.tolist(),
            "bottom_scores": (-bot_vals).tolist(),
        }
    except Exception as e:
        return {"error": str(e)}


def run_svd_decomposition(args):
    """Run SVD circuit decomposition."""
    print("=" * 70)
    print("SVD CIRCUIT DECOMPOSITION (E2.1 + E2.3)")
    print("=" * 70)

    out_dir = Path("results/svd_circuits")
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
    n_heads = spec.num_heads
    head_dim = spec.head_dim

    # Target heads: known circuit heads + auto-detected from prior results
    target_heads = [
        (5, 15), (5, 29),   # Early layer identified heads
        (27, 2), (27, 10), (27, 18), (27, 26), (27, 31),  # Late layer
    ]
    # Filter to valid heads for this model
    target_heads = [(l, h) for l, h in target_heads
                    if l < spec.num_layers and h < n_heads]

    print(f"Target heads: {target_heads}")

    n = min(args.n_prompts, len(RECURSIVE_PROMPTS), len(BASELINE_PROMPTS))
    rec_prompts = RECURSIVE_PROMPTS[:n]
    bas_prompts = BASELINE_PROMPTS[:n]
    rec_prompt_ids = RECURSIVE_PROMPT_IDS[:n]
    bas_prompt_ids = BASELINE_PROMPT_IDS[:n]

    print(
        "Prompt contract: "
        f"subset={_subset.name} bank={_subset.source_bank_version} tier=core_measurement"
    )

    head_results = {}

    for layer_idx, head_idx in target_heads:
        key = f"L{layer_idx}_H{head_idx}"
        print(f"\n  Analyzing {key}...")

        rec_spectra = {"eff_ranks": [], "top1_ratios": [], "spectral_gaps": []}
        bas_spectra = {"eff_ranks": [], "top1_ratios": [], "spectral_gaps": []}
        rec_U_accum = []
        bas_U_accum = []

        for condition, prompts, spectra, U_accum in [
            ("recursive", rec_prompts, rec_spectra, rec_U_accum),
            ("baseline", bas_prompts, bas_spectra, bas_U_accum),
        ]:
            for text in prompts:
                heads = get_per_head_v(model, tokenizer, text, layer_idx, n_heads, head_dim, args.device)
                v_head = heads[head_idx]
                result = svd_spectrum(v_head)
                if result is None:
                    continue

                S_np = result["S_np"]
                S_sq = S_np ** 2
                total = S_sq.sum()
                if total < 1e-10:
                    continue

                # Effective rank
                p = S_sq / total
                p = p[p > 1e-10]
                eff_rank = float(np.exp(-np.sum(p * np.log(p))))
                spectra["eff_ranks"].append(eff_rank)

                # Top-1 ratio
                spectra["top1_ratios"].append(float(S_np[0] / S_np.sum()))

                # Spectral gap
                if len(S_np) >= 2:
                    spectra["spectral_gaps"].append(float(S_np[0] - S_np[1]))

                # Accumulate top singular vector
                U_accum.append(result["U"][:, 0].numpy())

        # Compute effect sizes
        d_rank = cohens_d(rec_spectra["eff_ranks"], bas_spectra["eff_ranks"])
        d_top1 = cohens_d(rec_spectra["top1_ratios"], bas_spectra["top1_ratios"])
        d_gap = cohens_d(rec_spectra["spectral_gaps"], bas_spectra["spectral_gaps"])

        print(f"    Eff rank: rec={np.mean(rec_spectra['eff_ranks']):.2f} "
              f"bas={np.mean(bas_spectra['eff_ranks']):.2f} d={d_rank:.3f}")
        print(f"    Top1 ratio: rec={np.mean(rec_spectra['top1_ratios']):.3f} "
              f"bas={np.mean(bas_spectra['top1_ratios']):.3f} d={d_top1:.3f}")

        # Direction stability: cosine between top singular vectors across prompts
        if rec_U_accum:
            rec_U_mean = np.mean(rec_U_accum, axis=0)
            rec_U_mean /= np.linalg.norm(rec_U_mean) + 1e-10
            rec_cosines = [float(np.abs(np.dot(u / (np.linalg.norm(u) + 1e-10), rec_U_mean)))
                          for u in rec_U_accum]
            rec_stability = float(np.mean(rec_cosines))
        else:
            rec_stability = float("nan")

        if bas_U_accum:
            bas_U_mean = np.mean(bas_U_accum, axis=0)
            bas_U_mean /= np.linalg.norm(bas_U_mean) + 1e-10
            bas_cosines = [float(np.abs(np.dot(u / (np.linalg.norm(u) + 1e-10), bas_U_mean)))
                          for u in bas_U_accum]
            bas_stability = float(np.mean(bas_cosines))
        else:
            bas_stability = float("nan")

        print(f"    Direction stability: rec={rec_stability:.3f} bas={bas_stability:.3f}")

        # Vocabulary projection (E2.3) — for top-3 singular directions of mean recursive
        vocab_results = {}
        if rec_U_accum:
            # Get representative SVD
            rep_heads = get_per_head_v(model, tokenizer, rec_prompts[0], layer_idx, n_heads, head_dim, args.device)
            rep_result = svd_spectrum(rep_heads[head_idx])
            if rep_result:
                for k in range(min(3, rep_result["U"].shape[1])):
                    U_col = rep_result["U"][:, k]
                    vp = vocab_projection(U_col, model, layer_idx, head_idx, head_dim)
                    if "top_token_ids" in vp:
                        # Decode token IDs
                        vp["top_tokens"] = [tokenizer.decode([tid]) for tid in vp["top_token_ids"]]
                        vp["bottom_tokens"] = [tokenizer.decode([tid]) for tid in vp["bottom_token_ids"]]
                    vocab_results[f"sv{k+1}"] = vp

        head_results[key] = {
            "layer": layer_idx,
            "head": head_idx,
            "eff_rank_recursive_mean": float(np.mean(rec_spectra["eff_ranks"])) if rec_spectra["eff_ranks"] else float("nan"),
            "eff_rank_baseline_mean": float(np.mean(bas_spectra["eff_ranks"])) if bas_spectra["eff_ranks"] else float("nan"),
            "d_eff_rank": d_rank,
            "top1_recursive_mean": float(np.mean(rec_spectra["top1_ratios"])) if rec_spectra["top1_ratios"] else float("nan"),
            "top1_baseline_mean": float(np.mean(bas_spectra["top1_ratios"])) if bas_spectra["top1_ratios"] else float("nan"),
            "d_top1_ratio": d_top1,
            "d_spectral_gap": d_gap,
            "direction_stability_recursive": rec_stability,
            "direction_stability_baseline": bas_stability,
            "vocabulary_projections": vocab_results,
        }

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "experiment": "E2.1_E2.3_svd_circuits",
        "model": args.model,
        "prompt_bank_version": _subset.source_bank_version,
        "prompt_subset_name": _subset.name,
        "prompt_subset_schema_version": _subset.schema_version,
        "prompt_subset_path": str(_subset.manifest_path),
        "prompt_tier": "core_measurement",
        "target_heads": [f"L{l}_H{h}" for l, h in target_heads],
        "n_prompts": n,
        "recursive_prompt_ids": rec_prompt_ids,
        "baseline_prompt_ids": bas_prompt_ids,
        "head_results": head_results,
    }

    path = out_dir / f"svd_decomposition_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: {path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVD Circuit Decomposition (E2.1 + E2.3)")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-prompts", type=int, default=20)
    args = parser.parse_args()
    run_svd_decomposition(args)
