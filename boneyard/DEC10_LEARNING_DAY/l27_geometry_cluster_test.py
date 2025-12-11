#!/usr/bin/env python3
"""
DEC10 Test A: L27 Geometry Cluster Test
======================================

Goal:
  Test whether recursive prompts occupy a lower-dimensional, more
  collapsed subspace at Layer 27 than baselines, using many prompts.

Protocol:
  1. Use prompt_bank_1c from `n300_mistral_test_prompt_bank.py`:
       - Recursive group: all prompts with pillar == "dose_response" (L3_deeper etc.)
       - Baseline group: all prompts with pillar == "baseline"
  2. For each prompt, capture V at:
       - EARLY_LAYER (L5) and TARGET_LAYER (L27)
  3. For each cluster (baseline vs recursive, early vs late):
       - Stack per-prompt V-windows into a 2D matrix
       - Compute PR and effective rank via SVD
       - Compute top-k (e.g. k=10, 20, 50) energy fractions
  4. Compare:
       - PR_late(recursive) vs PR_late(baseline)
       - EffRank_late(recursive) vs EffRank_late(baseline)
       - How many PCs explain 80/90/95% variance in each cluster

Outputs:
  - CSV summary in `DEC10_LEARNING_DAY/l27_geometry_cluster_results_*.csv`
  - Human-readable prints of key ratios.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import importlib.util
import sys

PROJECT_ROOT = "/workspace/mech-interp-latent-lab-phase1"
PROMPT_BANK_PATH = f"{PROJECT_ROOT}/n300_mistral_test_prompt_bank.py"

spec = importlib.util.spec_from_file_location("n300_mistral_test_prompt_bank", PROMPT_BANK_PATH)
pb = importlib.util.module_from_spec(spec)
sys.modules["n300_mistral_test_prompt_bank"] = pb
spec.loader.exec_module(pb)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EARLY_LAYER = 5
TARGET_LAYER = 27
WINDOW_SIZE = 16

MAX_PROMPTS_PER_GROUP = 40  # keep this modest for DEC10 but > standard N
TOP_K_LIST = [10, 20, 50]


def compute_spectrum_stats(mat: torch.Tensor, top_k_list=TOP_K_LIST):
    """
    mat: [samples, D] matrix of flattened V activations.
    Returns PR, eff_rank, and variance fractions for given k.
    """
    if mat.numel() == 0:
        return {
            "pr": np.nan,
            "eff_rank": np.nan,
            **{f"var_k{k}": np.nan for k in top_k_list},
        }

    # center features
    mat = mat - mat.mean(dim=0, keepdim=True)

    # SVD on feature covariance via mat^T
    try:
        U, S, Vt = torch.linalg.svd(mat, full_matrices=False)
    except Exception:
        return {
            "pr": np.nan,
            "eff_rank": np.nan,
            **{f"var_k{k}": np.nan for k in top_k_list},
        }

    s2 = (S.detach().cpu().numpy()) ** 2
    if s2.sum() <= 1e-10:
        return {
            "pr": np.nan,
            "eff_rank": np.nan,
            **{f"var_k{k}": np.nan for k in top_k_list},
        }

    pr = float((s2.sum() ** 2) / (s2 ** 2).sum())
    p = s2 / s2.sum()
    eff_rank = float(1.0 / (p ** 2).sum())

    out = {"pr": pr, "eff_rank": eff_rank}
    cum = np.cumsum(p)
    for k in top_k_list:
        k_eff = min(k, len(cum))
        out[f"var_k{k}"] = float(cum[k_eff - 1])
    return out


def collect_V_for_prompt(model, tokenizer, prompt: str):
    """
    Returns (V_early_window, V_late_window) each of shape [W, D].
    """
    v_early, v_late = [], []

    def mk_hook(storage):
        def hook(module, inp, out):
            storage.append(out.detach())

        return hook

    h1 = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(
        mk_hook(v_early)
    )
    h2 = model.model.layers[TARGET_LAYER].self_attn.v_proj.register_forward_hook(
        mk_hook(v_late)
    )

    try:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            model(**inputs)
    finally:
        h1.remove()
        h2.remove()

    if not v_early or not v_late:
        return None, None

    def window_last(v_tensor):
        if v_tensor.dim() == 3:
            v_tensor = v_tensor[0]
        T, D = v_tensor.shape
        W = min(WINDOW_SIZE, T)
        return v_tensor[-W:, :].float()

    Ve = window_last(v_early[0])
    Vl = window_last(v_late[0])
    return Ve, Vl


def main():
    print("=" * 70)
    print("DEC10 L27 GEOMETRY CLUSTER TEST")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Partition prompt bank into baseline vs recursive
    baseline_texts = []
    recursive_texts = []
    for key, meta in pb.prompt_bank_1c.items():
        text = meta["text"]
        pillar = meta.get("pillar", "")
        if pillar == "baselines":
            baseline_texts.append(text)
        elif pillar == "dose_response":
            recursive_texts.append(text)

    baseline_texts = baseline_texts[:MAX_PROMPTS_PER_GROUP]
    recursive_texts = recursive_texts[:MAX_PROMPTS_PER_GROUP]

    print(f"Using {len(baseline_texts)} baseline prompts and {len(recursive_texts)} recursive prompts.")

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    )
    model.to(DEVICE)
    model.eval()
    print("Model loaded.\n")

    # Collect V windows
    def collect_group(texts):
        V_early_list = []
        V_late_list = []
        for t in texts:
            Ve, Vl = collect_V_for_prompt(model, tokenizer, t)
            if Ve is None or Vl is None:
                continue
            # Reduce last-window tokens to a single vector per prompt (mean over time)
            V_early_list.append(Ve.mean(dim=0))  # [D]
            V_late_list.append(Vl.mean(dim=0))   # [D]
        if not V_early_list or not V_late_list:
            return None, None
        return torch.stack(V_early_list), torch.stack(V_late_list)

    print("Collecting baseline activations...")
    base_E, base_L = collect_group(baseline_texts)
    print("Collecting recursive activations...")
    rec_E, rec_L = collect_group(recursive_texts)

    # Compute stats
    rows = []
    for group_name, E, L in [
        ("baseline", base_E, base_L),
        ("recursive", rec_E, rec_L),
    ]:
        if E is None or L is None:
            continue
        stats_E = compute_spectrum_stats(E)
        stats_L = compute_spectrum_stats(L)

        print(f"\n=== {group_name.upper()} ===")
        print(f"N prompts: {E.shape[0]}")
        print(f"Early L{EARLY_LAYER}: PR={stats_E['pr']:.2f}, eff_rank={stats_E['eff_rank']:.2f}")
        print(f"Late  L{TARGET_LAYER}: PR={stats_L['pr']:.2f}, eff_rank={stats_L['eff_rank']:.2f}")
        for k in TOP_K_LIST:
            print(
                f"  Var@k={k}: early={stats_E[f'var_k{k}']:.3f}, late={stats_L[f'var_k{k}']:.3f}"
            )

        rows.append(
            {
                "group": group_name,
                "layer": "early",
                "layer_idx": EARLY_LAYER,
                **stats_E,
            }
        )
        rows.append(
            {
                "group": group_name,
                "layer": "late",
                "layer_idx": TARGET_LAYER,
                **stats_L,
            }
        )

    # Save to CSV
    out_dir = Path("/workspace/mech-interp-latent-lab-phase1/DEC10_LEARNING_DAY")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"l27_geometry_cluster_results_{ts}.csv"

    try:
        import pandas as pd

        import_dicts = []
        for row in rows:
            import_dicts.append(row)
        df = pd.DataFrame(import_dicts)
        df.to_csv(out_path, index=False)
        print(f"\nSaved cluster geometry stats to: {out_path}")
    except Exception as e:
        print(f"\nCould not save CSV due to: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()


