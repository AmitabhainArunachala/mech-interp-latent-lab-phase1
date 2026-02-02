"""
experiment_l8_early_layer_patching_sweep.py

Task: Early-Layer Bidirectional Patching Sweep
Purpose: Test if L8 is a discontinuity or part of a smooth gradient

For layers [4, 8, 12, 16, 20, 24]:
- Direction A (rec→base): Patch recursive residual into baseline prompt
- Direction B (base→rec): Patch baseline residual into recursive prompt

We measure:
- R_V change (geometric contraction)
- BehaviorState (baseline/questioning/naked_loop/recursive_prose/collapse)
- Actual generated text

Key Questions:
1. Is L8 a discontinuity or part of a gradient?
2. Does L8 patching = L8 steering (interrogative mode)?
3. Where is the actual boundary (when does patching start/stop working)?
"""

import os
import sys
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))

from src.core.models import load_model, set_seed
from src.metrics.rv import compute_rv
from src.metrics.behavior_states import label_behavior_state
from prompts.loader import PromptLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 16
LAYERS_TO_TEST = [4, 8, 12, 16, 20, 24]
N_PAIRS = 350  # Try to get as many as possible (will be limited by available prompts)


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 60) -> str:
    """Generate text from a prompt."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    text = tokenizer.decode(gen[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    return text


def get_residual_at_layer(
    model, tokenizer, text: str, layer_idx: int
) -> Optional[torch.Tensor]:
    """Capture the residual stream (layer output) at a given layer for a prompt."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    residual = None

    def hook_fn(module, inputs, output):
        nonlocal residual
        residual = output.detach()
        return output

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            _ = model(**enc)
    finally:
        handle.remove()

    if residual is None:
        return None
    return residual[0]  # Remove batch dimension: (seq_len, hidden_dim)


@contextmanager
def patch_residual_at_layer(model, layer_idx: int, patch_source: torch.Tensor):
    """
    Patch the residual stream at the input to a layer.
    
    Replace the last WINDOW_SIZE token positions with those from patch_source.
    """
    handle = None

    def patch_hook(module, args):
        hidden_states = args[0]
        if patch_source is None:
            return args

        B, T, D = hidden_states.shape
        T_src, D_src = patch_source.shape
        if D_src != D:
            return args

        W = min(WINDOW_SIZE, T, T_src)
        if W <= 0:
            return args

        patch_tensor = patch_source[-W:, :].to(
            hidden_states.device, dtype=hidden_states.dtype
        )
        hidden_states = hidden_states.clone()
        hidden_states[:, -W:, :] = patch_tensor.unsqueeze(0).expand(B, -1, -1)
        return (hidden_states,) + args[1:]

    handle = model.model.layers[layer_idx].register_forward_pre_hook(patch_hook)
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


def run_early_layer_patching_sweep():
    """Run bidirectional patching sweep across early-to-mid layers."""
    print("=" * 70)
    print("L8 Gap-Filling Experiment 1: Early-Layer Bidirectional Patching Sweep")
    print("=" * 70)
    print(f"Testing layers: {LAYERS_TO_TEST}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Number of pairs: {N_PAIRS}")
    print()

    set_seed(42)

    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    loader = PromptLoader()

    # Get all available balanced pairs (up to 350)
    # Try to get as many as possible from all recursive and baseline groups
    recursive_groups = ["L1_hint", "L2_simple", "L3_deeper", "L4_full", "L5_refined"]
    baseline_groups = ["baseline_math", "baseline_factual", "baseline_creative", "baseline_impossible", "baseline_personal"]
    
    pairs = loader.get_balanced_pairs(
        n_pairs=N_PAIRS,
        recursive_groups=recursive_groups,
        baseline_groups=baseline_groups,
        seed=123
    )
    print(f"Loaded {len(pairs)} recursive/baseline pairs.")
    print()

    rows: List[Dict] = []

    for pair_idx, (rec_prompt, base_prompt) in enumerate(tqdm(pairs, desc="Processing pairs")):
        # Natural baselines
        try:
            rv_base_natural = compute_rv(model, tokenizer, base_prompt, device=DEVICE)
        except Exception as e:
            print(f"Warning: Failed to compute R_V for baseline natural: {e}")
            rv_base_natural = float("nan")

        try:
            text_base_natural = generate_text(model, tokenizer, base_prompt)
            label_base_natural = label_behavior_state(text_base_natural)
        except Exception as e:
            print(f"Warning: Failed to generate baseline natural: {e}")
            text_base_natural = ""
            label_base_natural = label_behavior_state("")

        # Natural recursive
        try:
            rv_rec_natural = compute_rv(model, tokenizer, rec_prompt, device=DEVICE)
        except Exception as e:
            print(f"Warning: Failed to compute R_V for recursive natural: {e}")
            rv_rec_natural = float("nan")

        try:
            text_rec_natural = generate_text(model, tokenizer, rec_prompt)
            label_rec_natural = label_behavior_state(text_rec_natural)
        except Exception as e:
            print(f"Warning: Failed to generate recursive natural: {e}")
            text_rec_natural = ""
            label_rec_natural = label_behavior_state("")

        # Test each layer
        for layer_idx in LAYERS_TO_TEST:
            # Capture residuals at this layer
            try:
                rec_residual = get_residual_at_layer(model, tokenizer, rec_prompt, layer_idx)
                base_residual = get_residual_at_layer(model, tokenizer, base_prompt, layer_idx)
            except Exception as e:
                print(f"Warning: Failed to capture residuals at L{layer_idx}: {e}")
                continue

            if rec_residual is None or base_residual is None:
                print(f"Warning: Null residual at L{layer_idx}")
                continue

            # Direction A: rec → base
            try:
                with patch_residual_at_layer(model, layer_idx, rec_residual):
                    rv_rec_to_base = compute_rv(
                        model, tokenizer, base_prompt, device=DEVICE
                    )
                    text_rec_to_base = generate_text(model, tokenizer, base_prompt)
            except Exception as e:
                print(f"Warning: Failed rec→base at L{layer_idx}: {e}")
                rv_rec_to_base = float("nan")
                text_rec_to_base = ""

            label_rec_to_base = label_behavior_state(text_rec_to_base)

            # Direction B: base → rec
            try:
                with patch_residual_at_layer(model, layer_idx, base_residual):
                    rv_base_to_rec = compute_rv(
                        model, tokenizer, rec_prompt, device=DEVICE
                    )
                    text_base_to_rec = generate_text(model, tokenizer, rec_prompt)
            except Exception as e:
                print(f"Warning: Failed base→rec at L{layer_idx}: {e}")
                rv_base_to_rec = float("nan")
                text_base_to_rec = ""

            label_base_to_rec = label_behavior_state(text_base_to_rec)

            # Compute deltas
            rv_delta_rec_to_base = rv_rec_to_base - rv_base_natural
            rv_delta_base_to_rec = rv_base_to_rec - rv_rec_natural

            # Store results
            row: Dict = {
                "pair_idx": pair_idx,
                "layer": layer_idx,
                "rec_prompt": rec_prompt[:100] + ("..." if len(rec_prompt) > 100 else ""),
                "base_prompt": base_prompt[:100] + ("..." if len(base_prompt) > 100 else ""),
                # Natural R_V
                "rv_base_natural": rv_base_natural,
                "rv_rec_natural": rv_rec_natural,
                # Patched R_V
                "rv_rec_to_base": rv_rec_to_base,
                "rv_base_to_rec": rv_base_to_rec,
                # R_V deltas
                "rv_delta_rec_to_base": rv_delta_rec_to_base,
                "rv_delta_base_to_rec": rv_delta_base_to_rec,
                # Natural states
                "state_base_natural": label_base_natural.state.value,
                "state_rec_natural": label_rec_natural.state.value,
                # Patched states
                "state_rec_to_base": label_rec_to_base.state.value,
                "state_base_to_rec": label_base_to_rec.state.value,
                # Natural outputs (truncated)
                "output_base_natural": text_base_natural[:200],
                "output_rec_natural": text_rec_natural[:200],
                # Patched outputs (truncated)
                "output_rec_to_base": text_rec_to_base[:200],
                "output_base_to_rec": text_base_to_rec[:200],
                # Behavior state details
                "repetition_rec_to_base": label_rec_to_base.repetition_ratio,
                "repetition_base_to_rec": label_base_to_rec.repetition_ratio,
                "question_ratio_rec_to_base": label_rec_to_base.question_mark_ratio,
                "question_ratio_base_to_rec": label_base_to_rec.question_mark_ratio,
            }
            rows.append(row)

            # Print summary
            print(
                f"\n[Pair {pair_idx}, L{layer_idx}] "
                f"rec→base: {label_base_natural.state.value}→{label_rec_to_base.state.value} "
                f"(RV: {rv_base_natural:.3f}→{rv_rec_to_base:.3f}, Δ={rv_delta_rec_to_base:+.3f}) | "
                f"base→rec: {label_rec_natural.state.value}→{label_base_to_rec.state.value} "
                f"(RV: {rv_rec_natural:.3f}→{rv_base_to_rec:.3f}, Δ={rv_delta_base_to_rec:+.3f})"
            )

    # Save results
    os.makedirs("results/dec11_evening", exist_ok=True)
    out_csv = "results/dec11_evening/l8_early_layer_patching_sweep.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 70)
    print("Summary Statistics by Layer")
    print("=" * 70)

    for layer in LAYERS_TO_TEST:
        layer_data = df[df["layer"] == layer]
        if len(layer_data) == 0:
            continue

        print(f"\nLayer {layer}:")
        print(f"  rec→base: Mean RV Δ = {layer_data['rv_delta_rec_to_base'].mean():+.3f}")
        print(f"  base→rec: Mean RV Δ = {layer_data['rv_delta_base_to_rec'].mean():+.3f}")
        print(f"  State transitions (rec→base):")
        transitions = (
            layer_data.groupby(["state_base_natural", "state_rec_to_base"])
            .size()
            .reset_index(name="count")
        )
        for _, t in transitions.iterrows():
            print(f"    {t['state_base_natural']} → {t['state_rec_to_base']}: {t['count']}")
        print(f"  State transitions (base→rec):")
        transitions = (
            layer_data.groupby(["state_rec_natural", "state_base_to_rec"])
            .size()
            .reset_index(name="count")
        )
        for _, t in transitions.iterrows():
            print(f"    {t['state_rec_natural']} → {t['state_base_to_rec']}: {t['count']}")

    print(f"\nResults saved to: {out_csv}")
    print("=" * 70)


if __name__ == "__main__":
    run_early_layer_patching_sweep()
