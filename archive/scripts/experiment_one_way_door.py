"""
experiment_one_way_door.py

Question 1: One-Way Door Test

We know: Patching baseline INTO recursive causes collapse.
We test: Can we patch baseline INTO recursive to BREAK the recursive geometry?

Hypothesis: If recursion is a "one-way door", injecting baseline activations
at late layers should NOT break the recursive state (R_V should stay < 1.0).
If it's reversible, R_V should move toward baseline (≈ 1.0).

Method:
1. Take recursive/baseline pairs
2. Extract baseline residual at late layers (L24, L26, L28, L30, L31)
3. Patch baseline residual INTO recursive prompt
4. Measure R_V: Does it stay recursive (< 0.85) or move toward baseline (> 0.95)?
5. Measure behavior state: Does it collapse or stay coherent?
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
N_PAIRS = 20


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


def run_one_way_door_test():
    print("=" * 80)
    print("EXPERIMENT: One-Way Door Test")
    print("=" * 80)
    print("Question: Can baseline activations BREAK recursive geometry?")
    print("Method: Patch baseline residual INTO recursive prompt at late layers")
    print("=" * 80)
    
    set_seed(42)

    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    model.eval()
    loader = PromptLoader()

    pairs = loader.get_balanced_pairs(n_pairs=N_PAIRS, seed=123)
    print(f"\nUsing {len(pairs)} recursive/baseline pairs.")

    rows: List[Dict] = []

    for idx, (rec_prompt, base_prompt) in enumerate(tqdm(pairs, desc="Pairs")):
        # Baseline measurements
        try:
            rv_base = compute_rv(model, tokenizer, base_prompt, device=DEVICE)
        except Exception:
            rv_base = float("nan")
        text_base = generate_text(model, tokenizer, base_prompt)
        label_base = label_behavior_state(text_base)

        # Recursive measurements (natural state)
        try:
            rv_rec_natural = compute_rv(model, tokenizer, rec_prompt, device=DEVICE)
        except Exception:
            rv_rec_natural = float("nan")
        text_rec_natural = generate_text(model, tokenizer, rec_prompt)
        label_rec_natural = label_behavior_state(text_rec_natural)

        base_snip = base_prompt[:80] + ("..." if len(base_prompt) > 80 else "")
        rec_snip = rec_prompt[:80] + ("..." if len(rec_prompt) > 80 else "")

        # Test: Patch baseline INTO recursive at each late layer
        for layer_idx in LATE_LAYERS:
            base_residual = get_residual_at_layer(model, tokenizer, base_prompt, layer_idx)

            if base_residual is None:
                continue

            # Patched run: recursive prompt with baseline residual injected
            try:
                with patch_residual_at_layer(model, layer_idx, base_residual):
                    rv_patched = compute_rv(model, tokenizer, rec_prompt, device=DEVICE)
                    text_patched = generate_text(model, tokenizer, rec_prompt)
            except Exception as e:
                print(f"  Error at L{layer_idx}: {e}")
                rv_patched = float("nan")
                text_patched = ""

            label_patched = label_behavior_state(text_patched)

            # Compute "recovery" metric: How much did R_V move toward baseline?
            # If rv_rec_natural = 0.80 and rv_base = 1.00, then:
            # - If rv_patched = 0.80 → 0% recovery (stuck in recursive)
            # - If rv_patched = 0.90 → 50% recovery
            # - If rv_patched = 1.00 → 100% recovery (fully broken)
            if not (np.isnan(rv_rec_natural) or np.isnan(rv_base) or np.isnan(rv_patched)):
                if abs(rv_base - rv_rec_natural) > 1e-6:
                    recovery_pct = ((rv_patched - rv_rec_natural) / (rv_base - rv_rec_natural)) * 100
                else:
                    recovery_pct = 0.0
            else:
                recovery_pct = float("nan")

            row: Dict = {
                "pair_idx": idx,
                "layer": layer_idx,
                "rec_prompt": rec_snip,
                "base_prompt": base_snip,
                "rv_base": rv_base,
                "rv_rec_natural": rv_rec_natural,
                "rv_patched": rv_patched,
                "recovery_pct": recovery_pct,
                "state_base": label_base.state.value,
                "state_rec_natural": label_rec_natural.state.value,
                "state_patched": label_patched.state.value,
                "bekan_base": label_base.is_bekan_artifact,
                "bekan_rec": label_rec_natural.is_bekan_artifact,
                "bekan_patched": label_patched.is_bekan_artifact,
            }
            rows.append(row)

            print(
                f"\n[Pair {idx}, L{layer_idx}] "
                f"RV: rec_natural={rv_rec_natural:.3f}, base={rv_base:.3f}, "
                f"patched={rv_patched:.3f} (recovery={recovery_pct:.1f}%)"
            )
            print(
                f"  State: rec={label_rec_natural.state.value} -> "
                f"patched={label_patched.state.value}"
            )

    # Save results
    os.makedirs("results/dec11_evening", exist_ok=True)
    out_csv = "results/dec11_evening/one_way_door_test.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    for layer_idx in LATE_LAYERS:
        layer_rows = [r for r in rows if r["layer"] == layer_idx]
        valid_recovery = [r["recovery_pct"] for r in layer_rows 
                         if not np.isnan(r["recovery_pct"])]
        
        if valid_recovery:
            mean_recovery = np.mean(valid_recovery)
            std_recovery = np.std(valid_recovery)
            print(f"L{layer_idx}: Mean recovery = {mean_recovery:.1f}% ± {std_recovery:.1f}%")
            
            # Count how many broke recursion (R_V moved > 50% toward baseline)
            broke_count = sum(1 for r in valid_recovery if r > 50.0)
            print(f"  → {broke_count}/{len(valid_recovery)} pairs broke recursion (>50% recovery)")

    os.makedirs("logs/dec11_evening", exist_ok=True)
    out_log = "logs/dec11_evening/one_way_door_test.txt"
    with open(out_log, "w") as f:
        f.write("# One-Way Door Test: Can baseline break recursion?\n\n")
        f.write("Method: Patch baseline residual INTO recursive prompt at late layers\n")
        f.write("Metric: Recovery % = (RV_patched - RV_rec) / (RV_base - RV_rec) * 100\n")
        f.write("  → 0% = stuck in recursive, 100% = fully broken\n\n")
        for r in rows:
            f.write(
                f"PAIR {r['pair_idx']} | L{r['layer']} | "
                f"RV: rec={r['rv_rec_natural']:.3f}, base={r['rv_base']:.3f}, "
                f"patched={r['rv_patched']:.3f} (recovery={r['recovery_pct']:.1f}%) | "
                f"State: {r['state_rec_natural']} -> {r['state_patched']}\n"
            )
        f.write("\nSaved CSV: " + out_csv + "\n")

    print(f"\nOne-way door test complete. CSV: {out_csv}, log: {out_log}")


if __name__ == "__main__":
    run_one_way_door_test()
