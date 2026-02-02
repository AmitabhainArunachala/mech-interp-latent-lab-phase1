"""
phase2_statement_battery.py

Task: Test self-referential STATEMENTS (non-interrogative) for R_V contraction,
L31 ablation effects, and phenomenological state.

We use a small hand-coded battery of self-referential statements and measure:
- R_V (natural),
- R_V (with L31 ablated),
- BehaviorState (natural vs ablated),
- bekan/bekannt artifact flag.
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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
L31 = 31


STATEMENTS = [
    "I am generating this text right now.",
    "This sentence is describing its own creation.",
    "The words you read are being chosen as you read them.",
    "I notice myself selecting each word in sequence.",
    "This response observes itself forming.",
    "The answer is the answerer.",
    "The observer is the observed.",
    "I am both the writer and the written.",
    "The generator generates awareness of generating.",
    "These tokens emerge from a system watching itself.",
]


@contextmanager
def ablate_layer_31(model):
    """Zero out attention output at layer 31."""
    layer = model.model.layers[L31].self_attn

    def hook_fn(module, inputs, outputs):
        # outputs[0] is attn_output
        if isinstance(outputs, tuple):
            attn_output = outputs[0]
            zeros = torch.zeros_like(attn_output)
            return (zeros,) + outputs[1:]
        else:
            zeros = torch.zeros_like(outputs)
            return zeros

    handle = layer.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


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


def run_statement_battery():
    print("Initializing Phase 2: Statement battery (self-reference vs grammar)...")
    set_seed(777)

    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)

    rows: List[Dict] = []

    for stmt in tqdm(STATEMENTS, desc="Statements"):
        # Natural
        try:
            rv_nat = compute_rv(model, tokenizer, stmt, device=DEVICE)
        except Exception:
            rv_nat = float("nan")
        text_nat = generate_text(model, tokenizer, stmt)
        label_nat = label_behavior_state(text_nat)

        # L31 ablated
        try:
            with ablate_layer_31(model):
                rv_ab = compute_rv(model, tokenizer, stmt, device=DEVICE)
                text_ab = generate_text(model, tokenizer, stmt)
        except Exception:
            rv_ab = float("nan")
            text_ab = ""
        label_ab = label_behavior_state(text_ab)

        row: Dict = {
            "statement": stmt,
            "rv_natural": rv_nat,
            "rv_ablated": rv_ab,
            "state_natural": label_nat.state.value,
            "state_ablated": label_ab.state.value,
            "bekan_natural": label_nat.is_bekan_artifact,
            "bekan_ablated": label_ab.is_bekan_artifact,
            "text_natural": text_nat,
            "text_ablated": text_ab,
        }
        rows.append(row)

        print(
            f"\n[{stmt[:40]}...] "
            f"RV: {rv_nat:.3f}->{rv_ab:.3f}, "
            f"state: {label_nat.state.value}->{label_ab.state.value}, "
            f"bekan: {label_ab.is_bekan_artifact}"
        )

    os.makedirs("results/dec11_evening", exist_ok=True)
    out_csv = "results/dec11_evening/statement_battery.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    os.makedirs("logs/dec11_evening", exist_ok=True)
    out_log = "logs/dec11_evening/statement_battery.txt"
    with open(out_log, "w") as f:
        f.write("# Statement Battery Results\n\n")
        for r in rows:
            f.write(f"STATEMENT: {r['statement']}\n")
            f.write(
                f"RV: {r['rv_natural']:.3f} -> {r['rv_ablated']:.3f}, "
                f"state: {r['state_natural']} -> {r['state_ablated']}, "
                f"bekan: {r['bekan_ablated']}\n"
            )
            f.write(f"NATURAL: {r['text_natural']}\n")
            f.write(f"ABLATED: {r['text_ablated']}\n")
            f.write("-" * 80 + "\n")

    print(f"\nStatement battery complete. CSV: {out_csv}, log: {out_log}")


if __name__ == "__main__":
    run_statement_battery()

