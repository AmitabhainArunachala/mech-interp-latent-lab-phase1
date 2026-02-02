"""
phase2_bidirectional_loop_patching.py

Task: Test the "one-way door" hypothesis with late-layer residual patching.

For a small set of recursive/baseline pairs and late layers (24, 26):
- Direction A (rec->base): inject recursive residual into the baseline run.
- Direction B (base->rec): inject baseline residual into the recursive run.

We measure:
- R_V (geometric contraction),
- BehaviorState for completions (baseline / questioning / recursive_prose / naked_loop / collapse),
- bekan/bekannt artifact flags.
"""

import os
import sys
from contextlib import contextmanager
from typing import Dict, List

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
LAYERS = [24, 26]
N_PAIRS = 8


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 60) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(gen[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    return text


def get_residual_at_layer(model, tokenizer, text: str, layer_idx: int) -> torch.Tensor:
    """Capture the residual stream (layer output) at a given layer for a prompt."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    residual = None

    def hook_fn(module, inputs, output):
        nonlocal residual
        residual = output.detach()
        return output

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(**enc)
    handle.remove()

    if residual is None:
        return None
    return residual[0]


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


def run_bidirectional_loop_patching():
    print("Initializing Phase 2: Bidirectional late-layer loop patching...")
    set_seed(123)

    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    loader = PromptLoader()

    pairs = loader.get_balanced_pairs(n_pairs=N_PAIRS, seed=321)
    print(f"Using {len(pairs)} recursive/baseline pairs.")

    rows: List[Dict] = []

    for idx, (rec_prompt, base_prompt) in enumerate(tqdm(pairs, desc="Pairs")):
        # Baseline natural
        try:
            rv_base = compute_rv(model, tokenizer, base_prompt, device=DEVICE)
        except Exception:
            rv_base = float("nan")
        text_base = generate_text(model, tokenizer, base_prompt)
        label_base = label_behavior_state(text_base)

        # Recursive natural
        try:
            rv_rec = compute_rv(model, tokenizer, rec_prompt, device=DEVICE)
        except Exception:
            rv_rec = float("nan")
        text_rec = generate_text(model, tokenizer, rec_prompt)
        label_rec = label_behavior_state(text_rec)

        base_snip = base_prompt[:80] + ("..." if len(base_prompt) > 80 else "")
        rec_snip = rec_prompt[:80] + ("..." if len(rec_prompt) > 80 else "")

        for layer_idx in LAYERS:
            # Capture residuals
            rec_res = get_residual_at_layer(model, tokenizer, rec_prompt, layer_idx)
            base_res = get_residual_at_layer(model, tokenizer, base_prompt, layer_idx)

            # A: rec -> base
            try:
                with patch_residual_at_layer(model, layer_idx, rec_res):
                    rv_A = compute_rv(model, tokenizer, base_prompt, device=DEVICE)
                    text_A = generate_text(model, tokenizer, base_prompt)
            except Exception:
                rv_A = float("nan")
                text_A = ""
            label_A = label_behavior_state(text_A)

            # B: base -> rec
            try:
                with patch_residual_at_layer(model, layer_idx, base_res):
                    rv_B = compute_rv(model, tokenizer, rec_prompt, device=DEVICE)
                    text_B = generate_text(model, tokenizer, rec_prompt)
            except Exception:
                rv_B = float("nan")
                text_B = ""
            label_B = label_behavior_state(text_B)

            row: Dict = {
                "pair_idx": idx,
                "layer": layer_idx,
                "rec_prompt": rec_snip,
                "base_prompt": base_snip,
                "rv_base": rv_base,
                "rv_rec": rv_rec,
                "rv_rec_to_base": rv_A,
                "rv_base_to_rec": rv_B,
                "state_base": label_base.state.value,
                "state_rec": label_rec.state.value,
                "state_rec_to_base": label_A.state.value,
                "state_base_to_rec": label_B.state.value,
                "bekan_base": label_base.is_bekan_artifact,
                "bekan_rec": label_rec.is_bekan_artifact,
                "bekan_rec_to_base": label_A.is_bekan_artifact,
                "bekan_base_to_rec": label_B.is_bekan_artifact,
            }
            rows.append(row)

            print(
                f"\n[Pair {idx}, L{layer_idx}] "
                f"rec→base: {label_base.state.value}->{label_A.state.value}, "
                f"base→rec: {label_rec.state.value}->{label_B.state.value} | "
                f"RV base={rv_base:.3f}, rec={rv_rec:.3f}, "
                f"rec→base={rv_A:.3f}, base→rec={rv_B:.3f}"
            )

    os.makedirs("results/dec11_evening", exist_ok=True)
    out_csv = "results/dec11_evening/bidirectional_patch_grid.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"\nBidirectional late-layer patching complete. CSV: {out_csv}")


if __name__ == "__main__":
    run_bidirectional_loop_patching()

