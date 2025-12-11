"""
Phase 5: One-Way Door
- Induction: baseline + v8 (α=2.0)
- Reversal: recursive - v8 (α=2.0)
- Random controls and CV on v8
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch

sys.path.append(str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    behavior_score,
    capture_v_projection,
    compute_rv,
    generate_with_kv,
    get_prompts_by_pillar,
    load_model,
    participation_ratio,
    set_seed,
)
from phase3_microphone import apply_steering, extract_vectors  # reuse steering helpers


def run_induction_reversal(
    model,
    tokenizer,
    vec: torch.Tensor,
    base_prompts: List[str],
    rec_prompts: List[str],
    alpha: float,
    layer: int,
    device: str,
    max_new_tokens: int,
    temperature: float,
    out_csv: str,
):
    rows = []
    # Induction: baseline + v8
    for p in base_prompts:
        h = apply_steering(model, layer, vec, alpha=alpha)
        gen = generate_with_kv(
            model,
            tokenizer,
            p,
            past_key_values=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )
        h.remove()
        rv = compute_rv(model, tokenizer, gen, device=device)
        beh = behavior_score(gen)
        rows.append(
            {
                "type": "induction",
                "prompt": p[:80],
                "alpha": alpha,
                "rv": rv,
                "behavior": beh,
                "gen": gen[:200],
            }
        )
    # Reversal: recursive - v8
    for p in rec_prompts:
        h = apply_steering(model, layer, vec, alpha=-alpha)
        gen = generate_with_kv(
            model,
            tokenizer,
            p,
            past_key_values=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )
        h.remove()
        rv = compute_rv(model, tokenizer, gen, device=device)
        beh = behavior_score(gen)
        rows.append(
            {
                "type": "reversal",
                "prompt": p[:80],
                "alpha": -alpha,
                "rv": rv,
                "behavior": beh,
                "gen": gen[:200],
            }
        )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def run_random_controls(
    model,
    tokenizer,
    vec: torch.Tensor,
    base_prompts: List[str],
    rec_prompts: List[str],
    alpha: float,
    layer: int,
    device: str,
    max_new_tokens: int,
    temperature: float,
    out_csv: str,
):
    norm = torch.norm(vec)
    torch.manual_seed(0)
    rand_vec = torch.randn_like(vec)
    rand_vec = rand_vec / torch.norm(rand_vec) * norm
    rows = []
    for p in base_prompts:
        h = apply_steering(model, layer, rand_vec, alpha=alpha)
        gen = generate_with_kv(
            model,
            tokenizer,
            p,
            past_key_values=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )
        h.remove()
        rv = compute_rv(model, tokenizer, gen, device=device)
        beh = behavior_score(gen)
        rows.append(
            {
                "type": "random_plus",
                "prompt": p[:80],
                "alpha": alpha,
                "rv": rv,
                "behavior": beh,
                "gen": gen[:200],
            }
        )
    for p in rec_prompts:
        h = apply_steering(model, layer, rand_vec, alpha=-alpha)
        gen = generate_with_kv(
            model,
            tokenizer,
            p,
            past_key_values=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )
        h.remove()
        rv = compute_rv(model, tokenizer, gen, device=device)
        beh = behavior_score(gen)
        rows.append(
            {
                "type": "random_minus",
                "prompt": p[:80],
                "alpha": -alpha,
                "rv": rv,
                "behavior": beh,
                "gen": gen[:200],
            }
        )
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
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)
    model, tokenizer = load_model(args.model_name, device=args.device)

    rec_prompts = get_prompts_by_pillar(pillar="dose_response", limit=20, seed=args.seed)
    base_prompts = get_prompts_by_pillar(pillar="baseline", limit=20, seed=args.seed)

    vec, _, _ = extract_vectors(model, tokenizer, rec_prompts, base_prompts, layer=args.layer, device=args.device)

    run_induction_reversal(
        model,
        tokenizer,
        vec,
        base_prompts,
        rec_prompts,
        alpha=args.alpha,
        layer=args.layer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        out_csv="DEC11_2025_FULL_PIPELINE/results/phase5_one_way_door.csv",
    )
    run_random_controls(
        model,
        tokenizer,
        vec,
        base_prompts,
        rec_prompts,
        alpha=args.alpha,
        layer=args.layer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        out_csv="DEC11_2025_FULL_PIPELINE/results/phase5_random_controls.csv",
    )
    print("[phase5] wrote induction/reversal and random control CSVs.")


if __name__ == "__main__":
    main()

