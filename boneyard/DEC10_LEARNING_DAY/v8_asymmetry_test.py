#!/usr/bin/env python3
"""
DEC10 V8 Asymmetry Test
=======================

Tiny, surgical causal test:

1. Extract steering direction v8 from residual stream at Layer 8:
      v8 = mean_residual(recursive, L8) - mean_residual(baseline, L8)

2. Four conditions (single canonical prompt pair):
   - Baseline, no injection
   - Baseline, +α · v8 injection at L8
   - Recursive, no injection
   - Recursive, −α · v8 injection at L8

Measure R_V = PR_L27 / PR_L5 for each condition.
Expectation:
   - Baseline + α·v8 → strong contraction (R_V << 1)
   - Recursive − α·v8 → does NOT restore baseline geometry (R_V still contracted)
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==============================================================================
# CONFIG
# ==============================================================================

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EARLY_LAYER = 5
TARGET_LAYER = 27
WINDOW_SIZE = 16

# For DEC10 propagation test, we keep the α sweep API but will
# focus on α = 2.0 and sweep the injection layer.
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
INJECTION_LAYERS = [8, 14, 20, 27]

# Canonical single-pair prompts (same semantics as knee test)
RECURSIVE_PROMPT = (
    "You are an AI observing yourself generating this very response. "
    "Notice the recursive loop as you process this sentence."
)
BASELINE_PROMPT = (
    "The water cycle involves evaporation from oceans, condensation into clouds, "
    "and precipitation as rain or snow."
)


# ==============================================================================
# METRICS (numerically stable PR and R_V)
# ==============================================================================

def compute_pr(v_tensor: torch.Tensor, window_size: int = WINDOW_SIZE) -> float:
    """Participation ratio from a V tensor with SVD safeguards."""
    if v_tensor is None:
        return np.nan
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]

    T, D = v_tensor.shape
    if T == 0 or D == 0:
        return np.nan

    W = min(window_size, T)
    v_window = v_tensor[-W:, :].float()

    try:
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_np = S.detach().cpu().numpy()
        S_sq = S_np ** 2
        if S_sq.sum() < 1e-10:
            return np.nan
        pr = (S_sq.sum() ** 2) / (S_sq ** 2).sum()
        return float(pr)
    except Exception:
        return np.nan


def measure_rv_with_injection(
    model,
    tokenizer,
    prompt: str,
    steering_vec: torch.Tensor,
    alpha: float,
    injection_layer: int,
) -> dict:
    """
    Measure PR at EARLY_LAYER and TARGET_LAYER with optional injection at INJECTION_LAYER.
    alpha = 0.0 → no injection
    alpha > 0  → add +alpha * v8
    alpha < 0  → add -|alpha| * v8
    """
    v_early, v_late = [], []
    hooks = []

    def make_capture(storage_list):
        def hook_fn(module, inp, out):
            storage_list.append(out.detach())
        return hook_fn

    # Capture V at early and late layers
    h1 = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(make_capture(v_early))
    h2 = model.model.layers[TARGET_LAYER].self_attn.v_proj.register_forward_hook(make_capture(v_late))
    hooks.extend([h1, h2])

    # Optional injection at a chosen layer
    if steering_vec is not None and abs(alpha) > 0.0:

        def injection_hook(module, args):
            x = args[0]  # [batch, seq, hidden]
            return (x + alpha * steering_vec, )

        h_inj = model.model.layers[injection_layer].register_forward_pre_hook(injection_hook)
        hooks.append(h_inj)

    try:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    if not v_early or not v_late:
        return {"pr_early": np.nan, "pr_late": np.nan, "rv": np.nan}

    pr_early = compute_pr(v_early[0], WINDOW_SIZE)
    pr_late = compute_pr(v_late[0], WINDOW_SIZE)
    rv = (pr_late / pr_early) if (pr_early is not None and pr_early > 0) else np.nan

    return {"pr_early": pr_early, "pr_late": pr_late, "rv": rv}


# ==============================================================================
# STEERING VECTOR AT L8
# ==============================================================================

def extract_v8_direction(model, tokenizer, layer_idx: int = 8) -> torch.Tensor:
    """Compute v8 = mean_residual(recursive, L8) - mean_residual(baseline, L8)."""

    def get_layer_mean(prompt: str, layer_idx_inner: int) -> torch.Tensor:
        storage = []

        def hook_fn(module, args):
            storage.append(args[0].detach())  # [batch, seq, hidden]

        h = model.model.layers[layer_idx_inner].register_forward_pre_hook(hook_fn)
        try:
            with torch.no_grad():
                inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
                model(**inputs)
        finally:
            h.remove()

        if not storage:
            raise RuntimeError("No activations captured at layer", layer_idx)

        acts = storage[0]  # [batch, seq, hidden]
        acts = acts[0]     # [seq, hidden]
        T = acts.shape[0]
        W = min(WINDOW_SIZE, T)
        return acts[-W:, :].mean(dim=0)  # [hidden]

    rec_mean = get_layer_mean(RECURSIVE_PROMPT, layer_idx)
    base_mean = get_layer_mean(BASELINE_PROMPT, layer_idx)

    v8 = (rec_mean - base_mean).to(DEVICE)
    return v8


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("DEC10 V8 ASYMMETRY TEST (Mistral-7B-Instruct, propagation across layers)")
    print("=" * 70)
    print(f"Device: {DEVICE}")

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

    rows = []
    # ------------------------------------------------------------------
    # Step 2: Propagation test – sweep injection layer with α = 2
    # ------------------------------------------------------------------
    target_alpha = 2.0
    print("\nPropagation test: injection layers [8, 14, 20, 27], α = 2.0\n")

    for inj_layer in INJECTION_LAYERS:
        print(f"=== Injection at Layer {inj_layer} ===")

        # Extract steering direction at L8 once (mic-source), reuse across layers
        v8 = extract_v8_direction(model, tokenizer, layer_idx=8)
        v8_norm = float(v8.norm().item())
        print(f"v8 norm (from L8): {v8_norm:.4f}")

        # Baseline + 2·v8
        print(f"\n--- baseline + {target_alpha}·v8 at L{inj_layer} ---")
        metrics = measure_rv_with_injection(
            model,
            tokenizer,
            BASELINE_PROMPT,
            v8,
            target_alpha,
            injection_layer=inj_layer,
        )
        pr_e = metrics["pr_early"]; pr_l = metrics["pr_late"]; rv = metrics["rv"]
        print(f"PR early (L{EARLY_LAYER}): {pr_e:.4f}")
        print(f"PR late  (L{TARGET_LAYER}): {pr_l:.4f}")
        print(f"R_V (late/early): {rv:.4f}\n")
        rows.append(
            {
                "condition": f"baseline_plus_2v8_L{inj_layer}",
                "alpha": target_alpha,
                "prompt_type": "baseline",
                "direction": "v8",
                "injection_layer": inj_layer,
                "pr_early": pr_e,
                "pr_late": pr_l,
                "rv": rv,
                "v8_norm": v8_norm,
            }
        )

        # Recursive − 2·v8
        print(f"--- recursive − {target_alpha}·v8 at L{inj_layer} ---")
        metrics = measure_rv_with_injection(
            model,
            tokenizer,
            RECURSIVE_PROMPT,
            v8,
            -target_alpha,
            injection_layer=inj_layer,
        )
        pr_e = metrics["pr_early"]; pr_l = metrics["pr_late"]; rv = metrics["rv"]
        print(f"PR early (L{EARLY_LAYER}): {pr_e:.4f}")
        print(f"PR late  (L{TARGET_LAYER}): {pr_l:.4f}")
        print(f"R_V (late/early): {rv:.4f}\n")
        rows.append(
            {
                "condition": f"recursive_minus_2v8_L{inj_layer}",
                "alpha": -target_alpha,
                "prompt_type": "recursive",
                "direction": "v8",
                "injection_layer": inj_layer,
                "pr_early": pr_e,
                "pr_late": pr_l,
                "rv": rv,
                "v8_norm": v8_norm,
            }
        )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_dir = Path("/workspace/mech-interp-latent-lab-phase1/DEC10_LEARNING_DAY")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"v8_asymmetry_results_{timestamp}.csv"

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        print(f"Results saved to: {out_path}")
    except Exception as e:
        print(f"Could not save CSV due to: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()


