#!/usr/bin/env python3
"""
NOV 16 Mixtral Free Play
Exploratory analysis and experiments with Mixtral-8x7B

This script is designed for *free play* on a live RunPod with Mixtral loaded.
It gives you solid, reusable tools to:

1. Compute R_V(layer) for *all* layers (relative to Layer 5)
2. Compute the "Step 3" style R_V(layer) (relative to Layer 28)
3. Compute Effective Rank(layer) from hidden states
4. Find the "snap layer" (largest R_V drop) per prompt
5. Build a histogram of snap layers over many prompts
6. Inspect correlation between R_V and Effective Rank across layers

You can import this in a notebook, or run functions directly here.
"""

import math
from typing import Dict, List, Tuple, Any

import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

from n300_mistral_test_prompt_bank import prompt_bank_1c


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# These are the canonical early/late layers used in Phase 1
EARLY_LAYER = 5
LATE_LAYER = 28
WINDOW_SIZE = 16


# ============================================================
# MODEL LOADING
# ============================================================

def load_mixtral(model_name: str = MODEL_NAME):
    """
    Load Mixtral-8x7B in a reasonably memory-efficient way.

    Tries 8-bit quantization first; falls back to fp16 if needed.
    """
    print("=" * 60)
    print(f"Loading model: {model_name}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None

    # Try 8-bit quantization if bitsandbytes is available
    try:
        from transformers import BitsAndBytesConfig

        print("Trying 8-bit quantization (BitsAndBytes)...")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            output_hidden_states=True,
            attn_implementation="eager",
        )
        print("✅ Loaded Mixtral in 8-bit mode")
    except Exception as e:
        print(f"8-bit load failed ({e}); falling back to fp16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            output_hidden_states=True,
            attn_implementation="eager",
        )
        print("✅ Loaded Mixtral in fp16")

    model.eval()
    print(f"Layers: {model.config.num_hidden_layers}, "
          f"heads: {model.config.num_attention_heads}")

    return tokenizer, model


# ============================================================
# METRICS: PR(V) AND EFFECTIVE RANK
# ============================================================

def compute_column_space_pr(
    v_tensor: torch.Tensor,
    num_heads: int,
    window_size: int = WINDOW_SIZE,
) -> float:
    """
    Compute Participation Ratio (PR) of Value matrix column space.
    PR = (Σ λ_i)^2 / Σ λ_i^2, where λ_i are singular values.

    v_tensor: [batch, seq_len, hidden] or [seq_len, hidden]
    """
    if v_tensor.dim() == 2:
        v_tensor = v_tensor.unsqueeze(0)  # [1, seq, hidden]

    try:
        batch_size, seq_len, total_hidden = v_tensor.shape
    except ValueError:
        return float("nan")

    d_v = total_hidden // num_heads

    # [batch, seq, heads, d_v]
    v_reshaped = v_tensor.view(batch_size, seq_len, num_heads, d_v)
    # [batch, heads, d_v, seq]
    v_transposed = v_reshaped.permute(0, 2, 3, 1)

    pr_values: List[float] = []

    for head_idx in range(num_heads):
        v_head = v_transposed[0, head_idx, :, :]  # [d_v, seq_len]

        end_idx = min(window_size, v_head.shape[1])
        v_window = v_head[:, -end_idx:]  # [d_v, window]

        v_window = v_window.float()
        try:
            _, S, _ = torch.linalg.svd(v_window, full_matrices=False)
            S_sq = S ** 2
            if S_sq.sum() <= 0:
                continue
            S_sq_norm = S_sq / S_sq.sum()
            pr = 1.0 / (S_sq_norm ** 2).sum()
            pr_values.append(pr.item())
        except Exception:
            continue

    if not pr_values:
        return float("nan")
    return float(np.mean(pr_values))


def compute_effective_rank_from_hidden(
    hidden_layer: torch.Tensor,
    window_size: int = WINDOW_SIZE,
) -> float:
    """
    Effective rank of hidden states at a given layer.

    hidden_layer: [batch, seq_len, hidden]
    We take the last `window_size` tokens and treat the matrix as [hidden, window_size].
    """
    if hidden_layer.dim() != 3:
        return float("nan")

    # [seq_len, hidden]
    H = hidden_layer[0]  # assume batch=1
    seq_len, hidden_size = H.shape

    end_idx = min(window_size, seq_len)
    H_window = H[-end_idx:, :]  # [window, hidden]
    if H_window.numel() == 0:
        return float("nan")

    # [hidden, window]
    M = H_window.T.float()
    try:
        _, S, _ = torch.linalg.svd(M, full_matrices=False)
        S_sq = S ** 2
        if S_sq.sum() <= 0:
            return float("nan")
        p = S_sq / S_sq.sum()
        eff_rank = 1.0 / (p ** 2).sum()
        return float(eff_rank.item())
    except Exception:
        return float("nan")


# ============================================================
# HOOK: CAPTURE V MATRICES FOR ALL LAYERS IN ONE PASS
# ============================================================

from contextlib import contextmanager


@contextmanager
def capture_v_for_layers(model, layer_indices: List[int], store: Dict[int, torch.Tensor]):
    """
    Register hooks on v_proj for multiple layers and store outputs.
    """
    handles = []

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            store[layer_idx] = output.detach()
        return hook

    try:
        for idx in layer_indices:
            attn = model.model.layers[idx].self_attn
            v_proj = attn.v_proj
            h = v_proj.register_forward_hook(make_hook(idx))
            handles.append(h)
        yield
    finally:
        for h in handles:
            h.remove()


# ============================================================
# CORE: ANALYZE ONE PROMPT ACROSS ALL LAYERS
# ============================================================

def analyze_prompt_layers(
    model,
    tokenizer,
    prompt: str,
    window_size: int = WINDOW_SIZE,
) -> Dict[str, Any]:
    """
    For a single prompt, compute:
      - PR(V_layer) for all layers
      - R_V_new(layer)   = PR(V_layer) / PR(V_EARLY)
      - R_V_step3(layer) = PR(V_LATE) / PR(V_layer)
      - EffectiveRank(layer) from hidden states
      - snap_layer_new:  layer with largest negative ΔR_V_new
      - snap_layer_step3: same but for R_V_step3
      - min_rank_layer: layer with minimum EffectiveRank
    """
    model.eval()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    layer_indices = list(range(num_layers))

    v_outputs: Dict[int, torch.Tensor] = {}

    with torch.no_grad():
        with capture_v_for_layers(model, layer_indices, v_outputs):
            outputs = model(
                **inputs,
                output_hidden_states=True,
            )

    hidden_states = outputs.hidden_states  # tuple: [emb, layer0, layer1, ...]

    # --- Compute PR(V) for each layer ---
    pr_by_layer: Dict[int, float] = {}
    for idx in layer_indices:
        v_tensor = v_outputs.get(idx)
        if v_tensor is None:
            pr_by_layer[idx] = float("nan")
        else:
            pr_by_layer[idx] = compute_column_space_pr(
                v_tensor, num_heads, window_size
            )

    pr_early = pr_by_layer.get(EARLY_LAYER, float("nan"))
    pr_late = pr_by_layer.get(LATE_LAYER, float("nan"))

    R_V_new: List[float] = []
    R_V_step3: List[float] = []
    eff_rank: List[float] = []

    for idx in layer_indices:
        pr_i = pr_by_layer.get(idx, float("nan"))

        # New definition: PR(V_i) / PR(V_early)
        if math.isfinite(pr_i) and math.isfinite(pr_early) and pr_early > 0:
            rv_new = pr_i / (pr_early + 1e-8)
        else:
            rv_new = float("nan")

        # Step-3 definition: PR(V_late) / PR(V_i)
        if math.isfinite(pr_i) and math.isfinite(pr_late) and pr_i > 0:
            rv_old = pr_late / (pr_i + 1e-8)
        else:
            rv_old = float("nan")

        R_V_new.append(rv_new)
        R_V_step3.append(rv_old)

        # Effective rank from hidden states
        # hidden_states[0] = embedding, hidden_states[1] = layer 0, ...
        if hidden_states is not None and len(hidden_states) > idx + 1:
            h_layer = hidden_states[idx + 1]
            er = compute_effective_rank_from_hidden(h_layer, window_size)
        else:
            er = float("nan")
        eff_rank.append(er)

    # --- Find snap layers and min-rank layer ---
    def find_snap_layer(rv_list: List[float]) -> int:
        best_layer = -1
        max_drop = 0.0
        for i in range(1, len(rv_list)):
            a, b = rv_list[i - 1], rv_list[i]
            if not (math.isfinite(a) and math.isfinite(b)):
                continue
            drop = b - a
            if drop < max_drop:
                max_drop = drop
                best_layer = i
        return best_layer

    snap_layer_new = find_snap_layer(R_V_new)
    snap_layer_step3 = find_snap_layer(R_V_step3)

    # Min effective rank
    min_rank_layer = -1
    min_rank_val = float("inf")
    for i, er in enumerate(eff_rank):
        if math.isfinite(er) and er < min_rank_val:
            min_rank_val = er
            min_rank_layer = i

    return {
        "R_V_new": R_V_new,
        "R_V_step3": R_V_step3,
        "EffRank": eff_rank,
        "PR_by_layer": pr_by_layer,
        "snap_layer_new": snap_layer_new,
        "snap_layer_step3": snap_layer_step3,
        "min_rank_layer": min_rank_layer,
    }


# ============================================================
# PROMPT SELECTION HELPERS
# ============================================================

def get_prompts_by_group(group_name: str, max_prompts: int = 20) -> List[str]:
    """
    From prompt_bank_1c, select up to `max_prompts` with given group label.
    """
    texts: List[str] = []
    for entry in prompt_bank_1c.values():
        if entry.get("group") == group_name:
            texts.append(entry.get("text", ""))
            if len(texts) >= max_prompts:
                break
    return texts


def build_eval_prompt_list(
    groups: List[str],
    max_per_group: int = 20,
) -> List[Tuple[str, str]]:
    """
    Returns a list of (group, text) pairs to evaluate.
    """
    out: List[Tuple[str, str]] = []
    for g in groups:
        texts = get_prompts_by_group(g, max_per_group)
        out.extend((g, t) for t in texts)
    return out


# ============================================================
# EXPERIMENT 1: SNAP-LAYER HISTOGRAM (R_V NEW)
# ============================================================

def run_snap_histogram(
    model,
    tokenizer,
    groups: List[str],
    max_per_group: int = 20,
    window_size: int = WINDOW_SIZE,
) -> Dict[str, Any]:
    """
    For many prompts, compute the snap layer (R_V_new) and build a histogram.
    """
    prompts = build_eval_prompt_list(groups, max_per_group)
    print("=" * 60)
    print("RUNNING SNAP-LAYER HISTOGRAM")
    print(f"Total prompts: {len(prompts)}")
    print("=" * 60)

    snap_counts: Dict[int, int] = {}
    all_results: List[Dict[str, Any]] = []

    for idx, (group, text) in enumerate(prompts):
        print(f"\n[{idx+1}/{len(prompts)}] Group={group}")
        metrics = analyze_prompt_layers(model, tokenizer, text, window_size)
        snap = metrics["snap_layer_new"]
        all_results.append({
            "group": group,
            "snap_layer_new": snap,
            "snap_layer_step3": metrics["snap_layer_step3"],
            "min_rank_layer": metrics["min_rank_layer"],
        })
        if snap >= 0:
            snap_counts[snap] = snap_counts.get(snap, 0) + 1
        print(f"  snap_layer_new = {snap}, "
              f"snap_layer_step3 = {metrics['snap_layer_step3']}, "
              f"min_rank_layer = {metrics['min_rank_layer']}")

    print("\nSNAP-LAYER HISTOGRAM (R_V_new):")
    for layer in sorted(snap_counts.keys()):
        print(f"  Layer {layer:2d}: {snap_counts[layer]} prompts")

    return {
        "snap_counts": snap_counts,
        "per_prompt": all_results,
    }


# ============================================================
# EXPERIMENT 2: R_V NEW vs STEP3 FOR A SINGLE PROMPT
# ============================================================

def compare_rv_definitions_for_prompt(
    model,
    tokenizer,
    prompt: str,
    window_size: int = WINDOW_SIZE,
) -> Dict[str, Any]:
    """
    Compute and print both R_V definitions for one prompt.
    """
    metrics = analyze_prompt_layers(model, tokenizer, prompt, window_size)

    print("=" * 60)
    print("R_V DEFINITIONS COMPARISON (SINGLE PROMPT)")
    print("=" * 60)
    print("Layer | R_V_new (PR_i / PR_early) | R_V_step3 (PR_late / PR_i)")
    print("------|---------------------------|-----------------------------")
    for i, (rv_new, rv_old) in enumerate(
        zip(metrics["R_V_new"], metrics["R_V_step3"])
    ):
        if math.isfinite(rv_new) or math.isfinite(rv_old):
            print(f"{i:5d} | {rv_new:>9.3f}                 | {rv_old:>9.3f}")

    print(f"\nSnap layer (new):   {metrics['snap_layer_new']}")
    print(f"Snap layer (step3): {metrics['snap_layer_step3']}")
    print(f"Min rank layer:     {metrics['min_rank_layer']}")

    return metrics


# ============================================================
# EXPERIMENT 3: R_V vs EFFECTIVE RANK CORRELATION
# ============================================================

def rv_vs_rank_correlation(
    model,
    tokenizer,
    groups: List[str],
    max_per_group: int = 20,
    window_size: int = WINDOW_SIZE,
) -> Dict[str, Any]:
    """
    For many prompts, compute R_V_new and EffRank at all layers
    and report per-layer correlation between them.
    """
    prompts = build_eval_prompt_list(groups, max_per_group)
    num_layers = model.config.num_hidden_layers

    rv_matrix = []
    er_matrix = []

    print("=" * 60)
    print("RUNNING R_V vs EFFECTIVE RANK CORRELATION")
    print(f"Total prompts: {len(prompts)}")
    print("=" * 60)

    for idx, (group, text) in enumerate(prompts):
        print(f"\n[{idx+1}/{len(prompts)}] Group={group}")
        metrics = analyze_prompt_layers(model, tokenizer, text, window_size)
        rv_matrix.append(metrics["R_V_new"])
        er_matrix.append(metrics["EffRank"])

    rv_arr = np.array(rv_matrix)  # [P, L]
    er_arr = np.array(er_matrix)  # [P, L]

    layer_corrs: Dict[int, float] = {}
    print("\nPer-layer Pearson correlation between R_V_new and EffRank:")
    for layer in range(num_layers):
        rv_col = rv_arr[:, layer]
        er_col = er_arr[:, layer]
        mask = np.isfinite(rv_col) & np.isfinite(er_col)
        if mask.sum() < 3:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(rv_col[mask], er_col[mask])[0, 1])
        layer_corrs[layer] = corr
        print(f"  Layer {layer:2d}: corr = {corr: .3f}")

    return {
        "rv_matrix": rv_arr,
        "er_matrix": er_arr,
        "layer_corrs": layer_corrs,
    }


# ============================================================
# EXPERIMENT 4: FULL 80-PROMPT × 32-LAYER SWEEP
# ============================================================

def run_full_80_prompt_layer_sweep(
    model,
    tokenizer,
    groups: List[str],
    max_per_group: int = 20,
    window_size: int = WINDOW_SIZE,
    output_csv: str = "MIXTRAL_80x32_LAYER_SWEEP.csv",
) -> Dict[str, Any]:
    """
    Run the FULL measurement:
      - 80 prompts (4 groups × 20)
      - 32 layers
      - R_V_new, R_V_step3, EffRank, PR per layer

    Saves a tall CSV with columns:
      group, prompt_index, layer,
      R_V_new, R_V_step3, EffRank, PR,
      snap_layer_new, snap_layer_step3, min_rank_layer
    """
    prompts = build_eval_prompt_list(groups, max_per_group)
    num_layers = model.config.num_hidden_layers

    print("=" * 60)
    print("RUNNING FULL 80-PROMPT × 32-LAYER SWEEP")
    print(f"Total prompts: {len(prompts)}")
    print(f"Layers: {num_layers}")
    print("=" * 60)

    rows: List[Dict[str, Any]] = []

    for p_idx, (group, text) in enumerate(prompts):
        print(f"\n[{p_idx+1}/{len(prompts)}] Group={group}")
        metrics = analyze_prompt_layers(model, tokenizer, text, window_size)

        snap_new = metrics["snap_layer_new"]
        snap_old = metrics["snap_layer_step3"]
        min_rank_layer = metrics["min_rank_layer"]
        rv_new = metrics["R_V_new"]
        rv_old = metrics["R_V_step3"]
        eff_rank = metrics["EffRank"]
        pr_by_layer = metrics["PR_by_layer"]

        for layer in range(num_layers):
            rows.append(
                {
                    "group": group,
                    "prompt_index": p_idx,
                    "layer": layer,
                    "R_V_new": rv_new[layer] if layer < len(rv_new) else float("nan"),
                    "R_V_step3": rv_old[layer] if layer < len(rv_old) else float("nan"),
                    "EffRank": eff_rank[layer] if layer < len(eff_rank) else float("nan"),
                    "PR": pr_by_layer.get(layer, float("nan")),
                    "snap_layer_new": snap_new,
                    "snap_layer_step3": snap_old,
                    "min_rank_layer": min_rank_layer,
                }
            )

    # Save to CSV (with JSON fallback if pandas missing)
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        print(f"\n💾 Full sweep saved to {output_csv}")
    except Exception as e:
        import json
        json_path = output_csv.replace(".csv", ".json")
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\n⚠️ Pandas unavailable ({e}); saved JSON instead to {json_path}")

    return {
        "rows": rows,
        "num_prompts": len(prompts),
        "num_layers": num_layers,
    }


# ============================================================
# USAGE EXAMPLES (for free play in a notebook)
# ============================================================

if __name__ == "__main__":
    print("NOV 16 Mixtral Free Play")
    print("This module is meant to be imported into a notebook.")
    print("Example usage:")
    print("""
from NOV_16_Mixtral_free_play import (
    load_mixtral,
    run_snap_histogram,
    compare_rv_definitions_for_prompt,
    rv_vs_rank_correlation,
    get_prompts_by_group,
)

tokenizer, model = load_mixtral()

# 1) Snap-layer histogram across 80 prompts
groups = ["L5_refined", "L3_deeper", "factual_baseline", "creative_baseline"]
snap_stats = run_snap_histogram(model, tokenizer, groups, max_per_group=20)

# 2) Compare R_V definitions for a single L5_refined prompt
prompt = get_prompts_by_group("L5_refined", max_prompts=1)[0]
metrics = compare_rv_definitions_for_prompt(model, tokenizer, prompt)

# 3) Correlation between R_V and Effective Rank
corr_stats = rv_vs_rank_correlation(model, tokenizer, groups, max_per_group=10)
""")
