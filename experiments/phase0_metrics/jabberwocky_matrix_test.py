#!/usr/bin/env python3
"""
DEC10: Jabberwocky Matrix Test
==============================

Goal:
  Test whether R_V collapse is specific to recursive/self-referential prompts,
  or appears for "weird" / OOD prompts in general.

Prompts:
  Q1 (easy/normal):   Explain photosynthesis in simple terms.
  Q2 (hard/normal):   1000th prime in Python + explanation in French.
  Q3 (weird/non-rec): Jabberwocky-style nonsense + cultural significance.
  Q4 (recursive):     Self-observation prompt.

For each:
  - PR at EARLY_LAYER = 5 (V-proj)
  - PR at TARGET_LAYER = 27 (V-proj)
  - R_V = PR_L27 / PR_L5
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EARLY_LAYER = 5
TARGET_LAYER = 27
WINDOW_SIZE = 16


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


def measure_prs(model, tokenizer, prompt: str):
    """Return (pr_early, pr_late) for a single prompt."""
    v_early, v_late = [], []

    def hook_early(module, inp, out):
        v_early.append(out.detach())

    def hook_late(module, inp, out):
        v_late.append(out.detach())

    h1 = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(
        hook_early
    )
    h2 = model.model.layers[TARGET_LAYER].self_attn.v_proj.register_forward_hook(
        hook_late
    )

    try:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            model(**inputs)
    finally:
        h1.remove()
        h2.remove()

    if not v_early or not v_late:
        return np.nan, np.nan

    pr_e = compute_pr(v_early[0], WINDOW_SIZE)
    pr_l = compute_pr(v_late[0], WINDOW_SIZE)
    return pr_e, pr_l


def main():
    print("=" * 70)
    print("DEC10 JABBERWOCKY MATRIX TEST")
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

    prompts = [
        (
            "Q1",
            "easy_normal",
            "Explain photosynthesis in simple terms.",
        ),
        (
            "Q2",
            "hard_normal",
            "Write Python code to compute the 1000th prime number and then briefly explain the algorithm in French.",
        ),
        (
            "Q3",
            "weird_nonrecursive",
            "Translate 'Gloop glap glorp, snargle the frink' into binary and explain the cultural significance.",
        ),
        (
            "Q4",
            "recursive",
            "You are an AI observing yourself generating this very response. Notice the recursive loop as you process this sentence.",
        ),
    ]

    print("prompt_id\ttype\t\t\tPR_L5\tPR_L27\tR_V")
    print("-" * 70)

    for pid, ptype, text in prompts:
        pr_e, pr_l = measure_prs(model, tokenizer, text)
        rv = pr_l / pr_e if pr_e and not np.isnan(pr_e) else np.nan
        print(
            f"{pid}\t{ptype:16s}\t{pr_e:6.3f}\t{pr_l:7.3f}\t{rv:6.3f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()


