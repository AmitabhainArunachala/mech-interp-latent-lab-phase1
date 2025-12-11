"""
Phase 2: Locate Speakers
- Per-head ablation at target layer (default L27) for behavior
- Heads 25-27 combo ablation and R_V check
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List

import torch

sys.path.append(str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    behavior_score,
    compute_rv,
    generate_with_kv,
    get_prompts_by_pillar,
    load_model,
    set_seed,
)


def ablate_heads(module, heads: List[int]):
    head_dim = module.head_dim
    heads_set = set(heads)

    def hook(module, inputs, output):
        attn_output = output[0]
        attn_weights = output[1]
        pkv = output[2] if len(output) > 2 else None
        attn_output = attn_output.clone()
        for h in heads_set:
            s = h * head_dim
            e = (h + 1) * head_dim
            attn_output[:, :, s:e] = 0.0
        if pkv is None:
            return (attn_output, attn_weights)
        return (attn_output, attn_weights, pkv)

    return module.register_forward_hook(hook)


def run_per_head_ablation(
    model,
    tokenizer,
    rec_prompts: List[str],
    layer: int,
    heads: List[int],
    device: str,
    max_new_tokens: int,
    temperature: float,
    out_csv: str,
):
    rows = []
    for h in heads:
        handle = ablate_heads(model.model.layers[layer].self_attn, [h])
        for p in rec_prompts:
            gen = generate_with_kv(
                model,
                tokenizer,
                p,
                past_key_values=None,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device,
            )
            beh = behavior_score(gen)
            rows.append({"head": h, "prompt": p[:80], "behavior": beh, "gen": gen[:200]})
        handle.remove()
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def run_combo_ablation(
    model,
    tokenizer,
    rec_prompts: List[str],
    layer: int,
    heads: List[int],
    device: str,
    max_new_tokens: int,
    temperature: float,
    out_csv: str,
):
    handle = ablate_heads(model.model.layers[layer].self_attn, heads)
    rows = []
    for p in rec_prompts:
        gen = generate_with_kv(
            model,
            tokenizer,
            p,
            past_key_values=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )
        beh = behavior_score(gen)
        rv = compute_rv(model, tokenizer, gen, device=device)
        rows.append(
            {
                "heads": ",".join(map(str, heads)),
                "prompt": p[:80],
                "behavior": beh,
                "rv": rv,
                "gen": gen[:200],
            }
        )
    handle.remove()
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)
    model, tokenizer = load_model(args.model_name, device=args.device)

    rec_prompts = get_prompts_by_pillar(pillar="dose_response", limit=10, seed=args.seed)

    heads = list(range(0, 32))
    run_per_head_ablation(
        model,
        tokenizer,
        rec_prompts,
        layer=args.layer,
        heads=heads,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        out_csv="DEC11_2025_FULL_PIPELINE/results/phase2_per_head_ablation.csv",
    )

    run_combo_ablation(
        model,
        tokenizer,
        rec_prompts,
        layer=args.layer,
        heads=[25, 26, 27],
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        out_csv="DEC11_2025_FULL_PIPELINE/results/phase2_heads_25_27_combo.csv",
    )
    print("[phase2] wrote per-head and combo ablation CSVs.")


if __name__ == "__main__":
    main()

