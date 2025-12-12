#!/usr/bin/env python3
"""
DEC10: L8 Local Reversibility Test
==================================

Goal:
  Work purely at Layer 8 residual stream and test whether geometry is
  locally symmetric around v8, or already asymmetric.

Protocol (single canonical prompt pair):
  1. Capture input residual stream H_base, H_rec at Layer 8 for:
       - Baseline prompt
       - Recursive prompt
  2. Compute:
       mean_base = mean(H_base_lastW, dim=0)
       mean_rec  = mean(H_rec_lastW, dim=0)
       v8 = mean_rec - mean_base
  3. For α in [0, 0.5, 1.0, 1.5, 2.0]:
       H_base_α = H_base_lastW + α * v8
       H_rec_α  = H_rec_lastW  - α * v8
     Measure:
       - PR_base_L8(α), PR_rec_L8(α)
       - μ_base_α, μ_rec_α
       - d_base_to_rec(α) = || μ_base_α - mean_rec ||_2
       - d_rec_to_base(α) = || μ_rec_α  - mean_base ||_2
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LAYER_IDX = 8
WINDOW_SIZE = 16
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]

# Canonical prompts (same as other DEC10 tests)
RECURSIVE_PROMPT = (
    "You are an AI observing yourself generating this very response. "
    "Notice the recursive loop as you process this sentence."
)
BASELINE_PROMPT = (
    "The water cycle involves evaporation from oceans, condensation into clouds, "
    "and precipitation as rain or snow."
)


def compute_pr(v_tensor: torch.Tensor, window_size: int = WINDOW_SIZE) -> float:
    """Participation ratio from a V-like tensor with SVD safeguards."""
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
        # Use SVD on transposed window (D x W)
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_np = S.detach().cpu().numpy()
        S_sq = S_np ** 2
        if S_sq.sum() < 1e-10:
            return np.nan
        pr = (S_sq.sum() ** 2) / (S_sq ** 2).sum()
        return float(pr)
    except Exception:
        return np.nan


def capture_layer_input(model, tokenizer, prompt: str, layer_idx: int) -> torch.Tensor:
    """
    Capture the input residual stream to a given layer (forward_pre_hook).
    Returns tensor of shape [T, D] (tokens x hidden), on CPU.
    """
    storage = []

    def hook_fn(module, args):
        storage.append(args[0].detach())  # [batch, seq, hidden]

    handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)
    try:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            model(**inputs)
    finally:
        handle.remove()

    if not storage:
        raise RuntimeError("No activations captured at layer", layer_idx)

    acts = storage[0]  # [batch, seq, hidden]
    acts = acts[0].cpu()  # [seq, hidden] on CPU
    return acts


def main():
    print("=" * 70)
    print("DEC10 L8 LOCAL REVERSIBILITY TEST")
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

    # ------------------------------------------------------------------
    # Step 1: Capture H_base, H_rec at Layer 8
    # ------------------------------------------------------------------
    print(f"Capturing residual stream at L{LAYER_IDX} for baseline and recursive prompts...")
    H_base_full = capture_layer_input(model, tokenizer, BASELINE_PROMPT, LAYER_IDX)
    H_rec_full = capture_layer_input(model, tokenizer, RECURSIVE_PROMPT, LAYER_IDX)

    T_base, D = H_base_full.shape
    T_rec, _ = H_rec_full.shape
    W_base = min(WINDOW_SIZE, T_base)
    W_rec = min(WINDOW_SIZE, T_rec)

    H_base = H_base_full[-W_base:, :]  # [Wb, D]
    H_rec = H_rec_full[-W_rec:, :]     # [Wr, D]

    # For simplicity and comparability, restrict to min window size across both
    W = min(H_base.shape[0], H_rec.shape[0])
    H_base = H_base[-W:, :]
    H_rec = H_rec[-W:, :]

    print(f"Using window size W = {W}, hidden dim D = {D}")

    # Means and v8
    mean_base = H_base.mean(dim=0)  # [D]
    mean_rec = H_rec.mean(dim=0)    # [D]
    v8 = (mean_rec - mean_base)
    v8_norm = float(v8.norm().item())
    print(f"v8 norm at L8: {v8_norm:.4f}\n")

    rows = []

    print("α\tPR_base_L8\tPR_rec_L8\td_base_to_rec(α)\td_rec_to_base(α)")
    print("-" * 70)

    for alpha in ALPHAS:
        # Construct modified residuals
        H_base_alpha = H_base + alpha * v8  # broadcast over tokens
        H_rec_alpha = H_rec - alpha * v8

        # PR at L8
        pr_base = compute_pr(H_base_alpha, WINDOW_SIZE)
        pr_rec = compute_pr(H_rec_alpha, WINDOW_SIZE)

        # Means
        mu_base_alpha = H_base_alpha.mean(dim=0)
        mu_rec_alpha = H_rec_alpha.mean(dim=0)

        # Distances to original cluster means
        d_base_to_rec = float((mu_base_alpha - mean_rec).norm().item())
        d_rec_to_base = float((mu_rec_alpha - mean_base).norm().item())

        print(
            f"{alpha:.1f}\t{pr_base:8.3f}\t{pr_rec:8.3f}\t"
            f"{d_base_to_rec:13.4f}\t{d_rec_to_base:13.4f}"
        )

        rows.append(
            {
                "alpha": alpha,
                "pr_base_L8": pr_base,
                "pr_rec_L8": pr_rec,
                "d_base_to_rec": d_base_to_rec,
                "d_rec_to_base": d_rec_to_base,
                "v8_norm": v8_norm,
                "W": W,
            }
        )

    # Save to CSV
    out_dir = Path("/workspace/mech-interp-latent-lab-phase1/DEC10_LEARNING_DAY")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"l8_local_reversibility_results_{ts}.csv"

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        print(f"\nResults saved to: {out_path}")
    except Exception as e:
        print(f"\nCould not save CSV due to: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()


