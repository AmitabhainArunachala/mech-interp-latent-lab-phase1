"""
Phase 4: KV Mechanism
- Layer-range KV patching
- α-mixing KV caches
- Optional temperature sweep

Outputs:
- ../results/phase4_kv_ranges.csv
- ../results/phase4_alpha_mix_T{temp}.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

sys.path.append(str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    behavior_score,
    capture_past_key_values,
    compute_rv,
    generate_with_kv,
    get_prompts_by_pillar,
    load_model,
    set_seed,
)


def mix_kv(
    base_kv: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    rec_kv: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    layer_start: int,
    layer_end: int,
    alpha: float = 1.0,
):
    mixed = []
    for i, (bk, bv) in enumerate(base_kv):
        rk, rv = rec_kv[i]
        if layer_start <= i < layer_end:
            mk = (1 - alpha) * bk + alpha * rk
            mv = (1 - alpha) * bv + alpha * rv
        else:
            mk, mv = bk, bv
        mixed.append((mk, mv))
    return tuple(mixed)


def run_layer_ranges(
    model,
    tokenizer,
    rec_prompts: List[str],
    base_prompts: List[str],
    ranges: List[Tuple[int, int]],
    device: str,
    max_new_tokens: int,
    temperature: float,
    out_csv: str,
):
    rows = []
    for rec, base in zip(rec_prompts, base_prompts):
        kv_rec = capture_past_key_values(model, tokenizer, rec, device=device)
        kv_base = capture_past_key_values(model, tokenizer, base, device=device)
        # Natural baseline metrics
        gen_nat = generate_with_kv(
            model,
            tokenizer,
            base,
            past_key_values=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )
        rv_nat = compute_rv(model, tokenizer, gen_nat, device=device)
        beh_nat = behavior_score(gen_nat)
        for r_start, r_end in ranges:
            kv_mix = mix_kv(kv_base, kv_rec, r_start, r_end, alpha=1.0)
            gen = generate_with_kv(
                model,
                tokenizer,
                base,
                past_key_values=kv_mix,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device,
            )
            rv = compute_rv(model, tokenizer, gen, device=device)
            beh = behavior_score(gen)
            rows.append(
                {
                    "rec_prompt": rec[:80],
                    "base_prompt": base[:80],
                    "range": f"{r_start}-{r_end}",
                    "rv_nat": rv_nat,
                    "beh_nat": beh_nat,
                    "rv_patch": rv,
                    "beh_patch": beh,
                    "gen_patch": gen[:200],
                }
            )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def run_alpha_mix(
    model,
    tokenizer,
    rec_prompts: List[str],
    base_prompts: List[str],
    alphas: List[float],
    device: str,
    max_new_tokens: int,
    temperature: float,
    out_csv: str,
):
    rows = []
    for rec, base in zip(rec_prompts, base_prompts):
        kv_rec = capture_past_key_values(model, tokenizer, rec, device=device)
        kv_base = capture_past_key_values(model, tokenizer, base, device=device)
        for alpha in alphas:
            kv_mix = mix_kv(kv_base, kv_rec, 16, 32, alpha=alpha)
            gen = generate_with_kv(
                model,
                tokenizer,
                base,
                past_key_values=kv_mix,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device,
            )
            rv = compute_rv(model, tokenizer, gen, device=device)
            beh = behavior_score(gen)
            rows.append(
                {
                    "rec_prompt": rec[:80],
                    "base_prompt": base[:80],
                    "alpha": alpha,
                    "temperature": temperature,
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--alpha_temps", nargs="*", type=float, default=[0.3, 0.7, 1.0])
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    args = parser.parse_args()

    set_seed(args.seed)
    model, tokenizer = load_model(args.model_name, device=args.device)

    rec_prompts = get_prompts_by_pillar(pillar="dose_response", limit=20, seed=args.seed)
    base_prompts = get_prompts_by_pillar(pillar="baseline", limit=20, seed=args.seed)

    ranges = [(0, 8), (8, 16), (16, 24), (16, 32), (24, 32)]
    run_layer_ranges(
        model,
        tokenizer,
        rec_prompts,
        base_prompts,
        ranges=ranges,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        out_csv="DEC11_2025_FULL_PIPELINE/results/phase4_kv_ranges.csv",
    )

    for temp in args.alpha_temps:
        run_alpha_mix(
            model,
            tokenizer,
            rec_prompts,
            base_prompts,
            alphas=args.alphas,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=temp,
            out_csv=f"DEC11_2025_FULL_PIPELINE/results/phase4_alpha_mix_T{temp}.csv",
        )
    print("[phase4] completed KV range and alpha-mix CSVs.")


if __name__ == "__main__":
    main()

