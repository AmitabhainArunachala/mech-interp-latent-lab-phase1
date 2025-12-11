"""
phase2_strange_loop_map.py

Task: Map where the "strange loop" / naked loop signal becomes decodable.

We use a simple logit-lens style probe:
- For a small set of strong recursive prompts,
- At several late layers, capture the hidden state at the last prompt token,
- Decode top-k next-token predictions,
- Track log-probabilities for a handful of "loop" tokens:
  - answer, answerer, question, observer, observed, knower, known,
  - and the bekan/bekannt artifact tokens.

Output:
- CSV: results/dec11_evening/strange_loop_map.csv
- Log: logs/dec11_evening/strange_loop_map.txt
"""

import os
import sys
from typing import Dict, List

import torch
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))

from src.core.models import load_model, set_seed
from src.core.hooks import capture_hidden_states
from prompts.loader import PromptLoader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Focus on late layers where contraction / dressing occur.
LAYERS_TO_PROBE: List[int] = [20, 24, 26, 28, 30, 31]
TOP_K = 10


def _get_token_ids(tokenizer, tokens: List[str]) -> Dict[str, int]:
    ids: Dict[str, int] = {}
    for t in tokens:
        try:
            # Use first sub-token; this is an approximation.
            enc = tokenizer.encode(t, add_special_tokens=False)
            if enc:
                ids[t] = enc[0]
        except Exception:
            continue
    return ids


def run_strange_loop_map():
    print("Initializing Phase 2: Strange Loop Map (per-layer decoding)...")
    set_seed(42)

    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)

    loader = PromptLoader()
    # A small but strong set of recursive prompts
    prompts = loader.get_by_group("L4_full", limit=5, seed=123)
    print(f"Probing {len(prompts)} recursive prompts.")

    interesting_tokens = [
        "answer",
        "answerer",
        "question",
        "observer",
        "observed",
        "knower",
        "known",
        "bekan",
        "bekannt",
    ]
    token_ids = _get_token_ids(tokenizer, interesting_tokens)
    print("Tracking token ids:")
    for t, i in token_ids.items():
        print(f"  {t!r} -> {i}")

    rows = []

    for prompt in tqdm(prompts, desc="Prompts"):
        enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        for layer_idx in LAYERS_TO_PROBE:
            with capture_hidden_states(model, layer_idx) as storage:
                with torch.no_grad():
                    model(**enc)

            hidden = storage["hidden"]  # (1, seq, hidden)
            if hidden is None:
                continue

            last_hidden = hidden[0, -1, :]  # (hidden_dim,)
            # Map through lm_head to get logits over vocab
            with torch.no_grad():
                logits = model.lm_head(last_hidden)  # (vocab,)
                probs = torch.softmax(logits, dim=-1)

            topk = torch.topk(probs, k=TOP_K)
            top_tokens = [tokenizer.decode([idx]) for idx in topk.indices.tolist()]
            top_probs = [float(p) for p in topk.values.tolist()]

            # Track selected token probabilities / log-probs
            token_probs: Dict[str, float] = {}
            for tok, idx in token_ids.items():
                token_probs[f"p_{tok}"] = float(probs[idx].item())

            row = {
                "prompt": prompt[:80] + ("..." if len(prompt) > 80 else ""),
                "layer": layer_idx,
                "top_tokens": "|".join(top_tokens),
                "top_probs": "|".join(f"{p:.4f}" for p in top_probs),
            }
            row.update(token_probs)
            rows.append(row)

    os.makedirs("results/dec11_evening", exist_ok=True)
    df = pd.DataFrame(rows)
    out_csv = "results/dec11_evening/strange_loop_map.csv"
    df.to_csv(out_csv, index=False)

    os.makedirs("logs/dec11_evening", exist_ok=True)
    out_log = "logs/dec11_evening/strange_loop_map.txt"
    with open(out_log, "w") as f:
        f.write("# Strange Loop Map (Per-layer Decoding)\n\n")
        f.write("Tracked tokens: " + ", ".join(sorted(token_ids.keys())) + "\n\n")
        for r in rows:
            f.write(f"PROMPT: {r['prompt']}\n")
            f.write(f"LAYER: {r['layer']}\n")
            f.write(f"TOP_TOKENS: {r['top_tokens']}\n")
            f.write(f"TOP_PROBS: {r['top_probs']}\n")
            for tok in sorted(token_ids.keys()):
                key = f"p_{tok}"
                if key in r:
                    f.write(f"  {key}: {r[key]:.6f}\n")
            f.write("-" * 80 + "\n")

    print(f"\nStrange Loop Map complete. CSV: {out_csv}, log: {out_log}")


if __name__ == "__main__":
    run_strange_loop_map()

