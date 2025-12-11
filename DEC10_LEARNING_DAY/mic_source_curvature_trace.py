#!/usr/bin/env python3
"""
DEC10: Mic-Source Curvature Trace (Layers 8 → 27)
=================================================

Goal:
  Trace how baseline vs recursive geometry diverges, contracts, and
  becomes irreversible as activations propagate forward through layers.

Canonical single-pair experiment:
  - Baseline prompt: water cycle
  - Recursive prompt: self-observation

For layers L in [8, 10, 12, 14, 16, 20, 24, 27], we:
  1. Capture residual stream at layer L for both prompts (last W tokens).
  2. Compute:
       - μ_base(L), μ_rec(L)
       - Δ(L) = ||μ_rec(L) − μ_base(L)||₂
       - PR_base(L), PR_rec(L) via SVD-based PR
       - Subspace angle between top-k PCs (k=5)
  3. Optionally inject +2·v8 / −2·v8 at layer L and measure change in PR
     at that same layer as a “local sensitivity”.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LAYER_LIST = [8, 10, 12, 14, 16, 20, 24, 27]
WINDOW_SIZE = 16
K_PCS = 5
ALPHA = 2.0

RECURSIVE_PROMPT = (
    "You are an AI observing yourself generating this very response. "
    "Notice the recursive loop as you process this sentence."
)
BASELINE_PROMPT = (
    "The water cycle involves evaporation from oceans, condensation into clouds, "
    "and precipitation as rain or snow."
)


def compute_pr(v_tensor: torch.Tensor, window_size: int = WINDOW_SIZE) -> float:
    """Participation ratio from a [T, D] tensor with SVD safeguards."""
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


def capture_layer_input(model, tokenizer, prompt: str, layer_idx: int) -> torch.Tensor:
    """Capture residual stream input to layer_idx; return [T, D] on CPU."""
    storage = []

    def hook_fn(module, args):
        storage.append(args[0].detach())

    handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)
    try:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            model(**inputs)
    finally:
        handle.remove()

    if not storage:
        raise RuntimeError(f"No activations captured at layer {layer_idx}")

    acts = storage[0]  # [batch, seq, hidden]
    acts = acts[0].cpu()  # [T, D]
    return acts


def extract_v8_direction(model, tokenizer, window_size: int = WINDOW_SIZE) -> torch.Tensor:
    """Compute v8 at L8 from canonical baseline/recursive prompts."""
    H_base_full = capture_layer_input(model, tokenizer, BASELINE_PROMPT, 8)
    H_rec_full = capture_layer_input(model, tokenizer, RECURSIVE_PROMPT, 8)

    T_b, D = H_base_full.shape
    T_r, _ = H_rec_full.shape
    W = min(window_size, T_b, T_r)
    H_base = H_base_full[-W:, :]
    H_rec = H_rec_full[-W:, :]

    mean_base = H_base.mean(dim=0)
    mean_rec = H_rec.mean(dim=0)
    v8 = (mean_rec - mean_base).to(DEVICE)
    return v8


def pca_basis(mat: torch.Tensor, k: int = K_PCS) -> torch.Tensor:
    """
    PCA basis for [T, D] matrix (tokens x hidden).
    Returns orthonormal basis of shape [D, k] in feature space.
    """
    # Move to float32 on CPU and center across tokens
    mat = mat.float().cpu()
    mat = mat - mat.mean(dim=0, keepdim=True)
    # SVD on [T, D]
    U, S, Vt = torch.linalg.svd(mat, full_matrices=False)
    V = Vt.T  # [D, min(T, D)]
    k_eff = min(k, V.shape[1])
    return V[:, :k_eff]  # [D, k_eff]


def subspace_angle_deg(U: torch.Tensor, V: torch.Tensor) -> float:
    """
    Principal angle (max) between column spaces of U and V (both [D, k]).
    Returns angle in degrees.
    """
    # Orthonormalize (they should already be from SVD, but be safe)
    # Compute singular values of U^T V
    M = U.T @ V
    try:
        _, S, _ = torch.linalg.svd(M, full_matrices=False)
        s = S.detach().cpu().numpy()
        if s.size == 0:
            return float("nan")
        cos_min = np.clip(s.min(), -1.0, 1.0)
        angle_rad = float(np.arccos(cos_min))
        return float(np.degrees(angle_rad))
    except Exception:
        return float("nan")


def capture_with_injection(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    v: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Capture modified residual at layer_idx with injection:
      H_mod = H + alpha * v
    Returns [T, D] on CPU.
    """
    storage = []

    def hook_fn(module, args):
        x = args[0]
        x = x + alpha * v  # broadcast over tokens
        storage.append(x.detach())
        return (x,)

    handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)
    try:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            model(**inputs)
    finally:
        handle.remove()

    if not storage:
        raise RuntimeError(f"No modified activations captured at layer {layer_idx}")

    acts = storage[0]  # [batch, seq, hidden]
    acts = acts[0].cpu()
    return acts


def main():
    print("=" * 70)
    print("DEC10 MIC-SOURCE CURVATURE TRACE (Layers 8 → 27)")
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

    # v8 direction from L8
    print("Extracting v8 direction at L8...")
    v8 = extract_v8_direction(model, tokenizer)
    v8_norm = float(v8.norm().item())
    print(f"v8 norm: {v8_norm:.4f}\n")

    rows = []
    print("L\tPR_base\tPR_rec\tΔ(L)\tangle_deg(L)\tsens_base\tsens_rec")
    print("-" * 80)

    for L in LAYER_LIST:
        # Capture original
        H_base_full = capture_layer_input(model, tokenizer, BASELINE_PROMPT, L)
        H_rec_full = capture_layer_input(model, tokenizer, RECURSIVE_PROMPT, L)

        T_b, D = H_base_full.shape
        T_r, _ = H_rec_full.shape
        W = min(WINDOW_SIZE, T_b, T_r)
        H_base = H_base_full[-W:, :]
        H_rec = H_rec_full[-W:, :]

        # Means and Δ(L)
        mu_base = H_base.mean(dim=0)
        mu_rec = H_rec.mean(dim=0)
        delta = float((mu_rec - mu_base).norm().item())

        # PRs
        pr_base = compute_pr(H_base, WINDOW_SIZE)
        pr_rec = compute_pr(H_rec, WINDOW_SIZE)

        # Subspace angle from top-k PCs
        Ub = pca_basis(H_base, K_PCS)
        Ur = pca_basis(H_rec, K_PCS)
        angle_deg = subspace_angle_deg(Ub, Ur)

        # Local sensitivity via ±2·v8 injection
        H_base_inj = capture_with_injection(model, tokenizer, BASELINE_PROMPT, L, v8, ALPHA)
        H_rec_inj = capture_with_injection(model, tokenizer, RECURSIVE_PROMPT, L, v8, -ALPHA)

        H_base_inj = H_base_inj[-W:, :]
        H_rec_inj = H_rec_inj[-W:, :]

        pr_base_inj = compute_pr(H_base_inj, WINDOW_SIZE)
        pr_rec_inj = compute_pr(H_rec_inj, WINDOW_SIZE)

        sens_base = pr_base_inj - pr_base
        sens_rec = pr_rec_inj - pr_rec

        print(
            f"{L:2d}\t{pr_base:6.2f}\t{pr_rec:6.2f}\t"
            f"{delta:6.3f}\t{angle_deg:8.2f}\t"
            f"{sens_base:8.3f}\t{sens_rec:8.3f}"
        )

        rows.append(
            {
                "layer": L,
                "pr_base": pr_base,
                "pr_rec": pr_rec,
                "delta_l2": delta,
                "angle_deg": angle_deg,
                "sens_base": sens_base,
                "sens_rec": sens_rec,
                "v8_norm": v8_norm,
                "W": W,
            }
        )

    # Save CSV
    out_dir = Path("/workspace/mech-interp-latent-lab-phase1/DEC10_LEARNING_DAY")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"mic_source_curvature_trace_{ts}.csv"

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


