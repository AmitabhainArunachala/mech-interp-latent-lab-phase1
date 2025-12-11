"""
Phase 0: Cross-Baseline KV Control (Gatekeeper)

Purpose:
- Patch KV from one baseline prompt (A) into another baseline prompt (B)
- Verify that R_V and behavior stay near baseline (i.e., no spurious recursion)

Outputs:
- CSV in ../results/phase0_cross_baseline.csv with per-pair metrics

Notes:
- This script is written for Mistral-7B-Instruct-v0.1.
- Assumes prompt_bank_1c is available in repo root.
- Behavior score here is a simple keyword count; upgrade to LLM/human eval later.
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import List

import torch

# Allow running from repo root or this folder
sys.path.append(str(Path(__file__).resolve().parent))

from common import (
    behavior_score,
    capture_past_key_values,
    compute_rv,
    generate_with_kv,
    get_prompts_by_pillar,
    load_model,
    set_seed,
)


def run_cross_baseline(
    model,
    tokenizer,
    prompts: List[str],
    pairs: int = 20,
    device: str = "cuda",
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    out_csv: str = "DEC11_2025_FULL_PIPELINE/results/phase0_cross_baseline.csv",
):
    rng = random.Random(0)
    rows = []
    for i in range(pairs):
        a, b = rng.sample(prompts, 2)
        kv_a = capture_past_key_values(model, tokenizer, a, device=device)
        # Adjust kv_a to match length of B to reduce seq mismatch
        len_b = len(tokenizer(b)["input_ids"])

        def reshape_kv(kv):
            reshaped = []
            for k, v in kv:
                seq_len = k.shape[2]
                if seq_len > len_b:
                    k_new = k[:, :, :len_b, :]
                    v_new = v[:, :, :len_b, :]
                elif seq_len < len_b:
                    pad_k = torch.zeros_like(k[:, :, :1, :]).expand(-1, -1, len_b - seq_len, -1)
                    pad_v = torch.zeros_like(v[:, :, :1, :]).expand(-1, -1, len_b - seq_len, -1)
                    k_new = torch.cat([k, pad_k], dim=2)
                    v_new = torch.cat([v, pad_v], dim=2)
                else:
                    k_new, v_new = k, v
                reshaped.append((k_new, v_new))
            return tuple(reshaped)

        kv_a = reshape_kv(kv_a)
        # Natural baseline B
        rv_nat = compute_rv(model, tokenizer, b, device=device)
        gen_nat = generate_with_kv(
            model,
            tokenizer,
            b,
            past_key_values=None,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        beh_nat = behavior_score(gen_nat)

        # Patched: baseline_B with KV from baseline_A
        # Generate using past_kv only (single BOS token) to test foreign KV effect
        bos_input = torch.tensor([[tokenizer.eos_token_id]], device=device)
        attn_mask = torch.ones((1, kv_a[0][0].shape[2] + 1), device=device, dtype=torch.long)
        position_ids = torch.arange(kv_a[0][0].shape[2], kv_a[0][0].shape[2] + 1, device=device).unsqueeze(0)
        gen = model.generate(
            input_ids=bos_input,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=kv_a,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0 else None,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_patched = tokenizer.decode(gen[0], skip_special_tokens=True)
        beh_patched = behavior_score(gen_patched)
        rv_patched = compute_rv(model, tokenizer, gen_patched, device=device)

        rows.append(
            {
                "pair_id": i,
                "prompt_a": a[:80],
                "prompt_b": b[:80],
                "rv_nat": rv_nat,
                "rv_patched": rv_patched,
                "beh_nat": beh_nat,
                "beh_patched": beh_patched,
                "gen_nat": gen_nat[:200],
                "gen_patched": gen_patched[:200],
            }
        )

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)

    prompts = get_prompts_by_pillar(pillar="baseline", limit=60, seed=args.seed, group_prefix="baseline")
    if len(prompts) < 2:
        raise ValueError("Not enough baseline prompts found.")

    model, tokenizer = load_model(args.model_name, device=args.device)

    out_csv = run_cross_baseline(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        pairs=args.pairs,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        out_csv="DEC11_2025_FULL_PIPELINE/results/phase0_cross_baseline.csv",
    )
    print(f"[phase0] wrote {out_csv}")


if __name__ == "__main__":
    main()

