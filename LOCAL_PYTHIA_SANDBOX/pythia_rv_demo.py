#!/usr/bin/env python3
"""Quick local R_V demo on Pythia-1.4B (CPU-friendly).

- Loads EleutherAI/pythia-1.4b from local HF cache
- Runs a few recursive vs baseline prompts
- Hooks V at early and late layers
- Computes R_V = PR_late / PR_early on generated tokens only

Run from repo root:

    cd /Users/dhyana/mech-interp-latent-lab-phase1
    python LOCAL_PYTHIA_SANDBOX/pythia_rv_demo.py

This is a tiny, self-contained Experiment A analogue.
"""

import os
import math
from contextlib import contextmanager

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cpu"  # keep it simple for now
MODEL_NAME = "EleutherAI/pythia-1.4b"

EARLY_LAYER = 4      # 4/24 ≈ 16% depth
TARGET_LAYER = 20    # 20/24 ≈ 83% depth
WINDOW_SIZE = 16
MAX_NEW_TOKENS = 40
TEMPERATURE = 0.7

N_RECURSIVE = 20
N_BASELINE = 20

# We will pull prompts from the canonical n300 prompt bank
from n300_mistral_test_prompt_bank import prompt_bank_1c


def load_prompts_from_bank():
    """Load recursive and baseline prompts from prompt_bank_1c."""
    recursive = []
    baseline = []

    # Recursive groups used across experiments
    rec_groups = {"L3_deeper", "L4_full", "L5_refined"}
    # Baseline groups
    base_groups = {"baseline_factual", "baseline_math", "baseline_creative", "long_control"}

    for key, val in prompt_bank_1c.items():
        g = val.get("group", "")
        text = val.get("text", "")
        if g in rec_groups and len(recursive) < N_RECURSIVE:
            recursive.append(text)
        elif g in base_groups and len(baseline) < N_BASELINE:
            baseline.append(text)
        if len(recursive) >= N_RECURSIVE and len(baseline) >= N_BASELINE:
            break

    return recursive, baseline


def compute_pr(v_tensor: torch.Tensor, window_size: int = WINDOW_SIZE) -> float:
    """Participation Ratio via SVD on last W tokens.

    v_tensor: [T, D] or [1, T, D]
    """
    if v_tensor is None:
        return math.nan

    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]

    T, D = v_tensor.shape
    W = min(window_size, T)
    if W < 2:
        return math.nan

    v_window = v_tensor[-W:, :].float()
    try:
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_sq = (S ** 2).cpu().numpy()
        if S_sq.sum() < 1e-10:
            return math.nan
        return float((S_sq.sum() ** 2) / (S_sq ** 2).sum())
    except Exception:
        return math.nan


@contextmanager
def capture_v_at_layer(model, layer_idx, storage_list):
    """Hook V-projections at a given layer (Pythia uses gpt_neox blocks)."""
    layer = model.gpt_neox.layers[layer_idx]

    def hook_fn(module, input, output):
        # For this quick demo, treat the layer output (hidden states) as a V-like proxy.
        # GPT-NeoX blocks sometimes return tuples; grab the first element if so.
        hidden = output[0] if isinstance(output, tuple) else output
        storage_list.append(hidden.detach())
        return output

    handle = layer.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def generate_and_capture(model, tokenizer, prompt: str):
    """Generate text and capture early/late layer outputs for GENERATED tokens only.

    For this quick demo we treat layer outputs as V-like and compute PR on them.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    # First pass: generate with cache to get full continuation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_ids = outputs[0]
    gen_ids = full_ids[prompt_len:]

    # Second pass: run full_ids through model with hooks to capture early/late
    v_early_list = []
    v_late_list = []

    with torch.no_grad():
        with capture_v_at_layer(model, EARLY_LAYER, v_early_list):
            with capture_v_at_layer(model, TARGET_LAYER, v_late_list):
                _ = model(full_ids.unsqueeze(0))

    if not v_early_list or not v_late_list:
        return math.nan, math.nan, math.nan, ""

    # Extract generated-token slice from captured outputs
    v_early = v_early_list[0][0][prompt_len:, :]
    v_late = v_late_list[0][0][prompt_len:, :]

    pr_early = compute_pr(v_early)
    pr_late = compute_pr(v_late)
    r_v = pr_late / pr_early if (pr_early and pr_early > 0) else math.nan

    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return r_v, pr_early, pr_late, gen_text.strip()


def main():
    print("=== Pythia-1.4B R_V demo (recursive vs baseline) ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Layers: early=L{EARLY_LAYER}, target=L{TARGET_LAYER}")

    recursive_prompts, baseline_prompts = load_prompts_from_bank()
    print(f"\nLoaded {len(recursive_prompts)} recursive and {len(baseline_prompts)} baseline prompts from n300 bank.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=DEVICE)
    model.eval()

    rec_rv = []
    base_rv = []

    print("\n--- Recursive prompts ---")
    for i, p in enumerate(recursive_prompts, 1):
        r_v, pr_e, pr_l, text = generate_and_capture(model, tokenizer, p)
        rec_rv.append(r_v)
        print(f"[REC {i}] R_V={r_v:.4f}  (PR_early={pr_e:.2f}, PR_late={pr_l:.2f})")

    print("\n--- Baseline prompts ---")
    for i, p in enumerate(baseline_prompts, 1):
        r_v, pr_e, pr_l, text = generate_and_capture(model, tokenizer, p)
        base_rv.append(r_v)
        print(f"[BASE {i}] R_V={r_v:.4f}  (PR_early={pr_e:.2f}, PR_late={pr_l:.2f})")

    rec_arr = np.array([x for x in rec_rv if not math.isnan(x)])
    base_arr = np.array([x for x in base_rv if not math.isnan(x)])

    if len(rec_arr) > 0 and len(base_arr) > 0:
        print("\n=== Summary ===")
        print(f"Recursive R_V: mean={rec_arr.mean():.4f} ± {rec_arr.std(ddof=1):.4f}")
        print(f"Baseline  R_V: mean={base_arr.mean():.4f} ± {base_arr.std(ddof=1):.4f}")
        gap = base_arr.mean() - rec_arr.mean()
        rel = gap / base_arr.mean() * 100 if base_arr.mean() > 0 else float('nan')
        print(f"Gap (baseline - recursive): {gap:.4f} ({rel:.1f}% contraction)")
    else:
        print("Not enough valid R_V values to summarize.")


if __name__ == "__main__":
    main()
