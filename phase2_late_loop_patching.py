"""
phase2_late_loop_patching.py

Task: Late-layer path patching grid for the strange loop.

We:
- Take balanced recursive/baseline pairs,
- For several late layers, patch the recursive residual stream into the baseline run,
- Measure:
  - R_V (geometric contraction),
  - phenomenological state (BehaviorState),
  - bekan/bekannt artifact flag.

Goal:
- Identify the earliest late layer where injecting recursive residuals
  causes the baseline to enter naked_loop / recursive_prose / collapse states.
"""

import os
import sys
from contextlib import contextmanager
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))

from src.core.models import load_model, set_seed
from src.metrics.rv import compute_rv
from src.metrics.behavior_states import label_behavior_state
from prompts.loader import PromptLoader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATE_LAYERS: List[int] = [24, 26, 28, 30, 31]
WINDOW_SIZE = 16
N_PAIRS = 5


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
    """
    Capture the residual stream (layer output) at a given layer for a prompt.
    """
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
    # Residual shape: (batch, seq, hidden)
    return residual[0]  # (seq, hidden)


@contextmanager
def patch_residual_at_layer(model, layer_idx: int, patch_source: torch.Tensor):
    """
    Patch the residual stream at the input to a layer.

    We use a forward_pre_hook on the given layer to replace the last WINDOW_SIZE
    token positions with those from patch_source (also last WINDOW_SIZE tokens).
    """
    handle = None

    def patch_hook(module, args):
        hidden_states = args[0]
        # hidden_states: (batch, seq, hidden)
        if patch_source is None:
            return args

        B, T, D = hidden_states.shape
        T_src, D_src = patch_source.shape
        if D_src != D:
            # Dimension mismatch – skip patching.
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


def run_late_loop_patching():
    print("Initializing Phase 2: Late-layer loop patching grid...")
    set_seed(42)

    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    loader = PromptLoader()

    pairs = loader.get_balanced_pairs(n_pairs=N_PAIRS, seed=123)
    print(f"Using {len(pairs)} recursive/baseline pairs.")

    rows: List[Dict] = []

    for idx, (rec_prompt, base_prompt) in enumerate(tqdm(pairs, desc="Pairs")):
        # Baseline run
        try:
            rv_base = compute_rv(model, tokenizer, base_prompt, device=DEVICE)
        except Exception:
            rv_base = float("nan")
        text_base = generate_text(model, tokenizer, base_prompt)
        label_base = label_behavior_state(text_base)

        # Recursive run
        try:
            rv_rec = compute_rv(model, tokenizer, rec_prompt, device=DEVICE)
        except Exception:
            rv_rec = float("nan")
        text_rec = generate_text(model, tokenizer, rec_prompt)
        label_rec = label_behavior_state(text_rec)

        base_snip = base_prompt[:80] + ("..." if len(base_prompt) > 80 else "")
        rec_snip = rec_prompt[:80] + ("..." if len(rec_prompt) > 80 else "")

        for layer_idx in LATE_LAYERS:
            rec_residual = get_residual_at_layer(model, tokenizer, rec_prompt, layer_idx)

            if rec_residual is None:
                continue

            # Patched run: baseline prompt with recursive residual injected at this layer.
            try:
                with patch_residual_at_layer(model, layer_idx, rec_residual):
                    rv_patched = compute_rv(model, tokenizer, base_prompt, device=DEVICE)
                    text_patched = generate_text(model, tokenizer, base_prompt)
            except Exception:
                rv_patched = float("nan")
                text_patched = ""

            label_patched = label_behavior_state(text_patched)

            row: Dict = {
                "pair_idx": idx,
                "layer": layer_idx,
                "rec_prompt": rec_snip,
                "base_prompt": base_snip,
                "rv_base": rv_base,
                "rv_rec": rv_rec,
                "rv_patched": rv_patched,
                "state_base": label_base.state.value,
                "state_rec": label_rec.state.value,
                "state_patched": label_patched.state.value,
                "bekan_base": label_base.is_bekan_artifact,
                "bekan_rec": label_rec.is_bekan_artifact,
                "bekan_patched": label_patched.is_bekan_artifact,
            }
            rows.append(row)

            print(
                f"\n[Pair {idx}, L{layer_idx}] "
                f"base={label_base.state.value}, rec={label_rec.state.value}, "
                f"patched={label_patched.state.value}, "
                f"RV: base={rv_base:.3f}, rec={rv_rec:.3f}, patched={rv_patched:.3f}"
            )

    os.makedirs("results/dec11_evening", exist_ok=True)
    out_csv = "results/dec11_evening/late_layer_patch_grid.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    os.makedirs("logs/dec11_evening", exist_ok=True)
    out_log = "logs/dec11_evening/late_layer_patch_grid.txt"
    with open(out_log, "w") as f:
        f.write("# Late-layer loop patching grid\n\n")
        for r in rows:
            f.write(
                f"PAIR {r['pair_idx']} | L{r['layer']} | "
                f"base={r['state_base']} -> patched={r['state_patched']} "
                f"(rec={r['state_rec']}) | "
                f"RV base={r['rv_base']:.3f}, rec={r['rv_rec']:.3f}, "
                f"patched={r['rv_patched']:.3f}\n"
            )
        f.write("\nSaved CSV: " + out_csv + "\n")

    print(f"\nLate-layer patching complete. CSV: {out_csv}, log: {out_log}")


if __name__ == "__main__":
    run_late_loop_patching()

