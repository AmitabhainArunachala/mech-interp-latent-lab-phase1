"""
Phase 1: Establish the Symptom
- Layer sweep (single rec + base prompt)
- Prompt battery (N=80 across categories)
- Window sensitivity test
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.append(str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    compute_rv,
    get_prompts_by_pillar,
    load_model,
    set_seed,
)


def layer_sweep(model, tokenizer, rec_prompt: str, base_prompt: str, windows: List[int], device: str, out_csv: str):
    rows = []
    for layer in range(0, 32):
        for window in windows:
            rv_rec = compute_rv(model, tokenizer, rec_prompt, early=4, late=layer, window=window, device=device)
            rv_base = compute_rv(model, tokenizer, base_prompt, early=4, late=layer, window=window, device=device)
            rows.append(
                {
                    "layer": layer,
                    "window": window,
                    "rv_recursive": rv_rec,
                    "rv_baseline": rv_base,
                    "separation": rv_base - rv_rec,
                }
            )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def prompt_battery(model, tokenizer, categories: Dict[str, List[str]], device: str, out_csv: str):
    rows = []
    for name, prompts in categories.items():
        for p in prompts:
            rv = compute_rv(model, tokenizer, p, device=device)
            rows.append({"category": name, "prompt": p[:120], "rv": rv})
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
    args = parser.parse_args()

    set_seed(args.seed)
    model, tokenizer = load_model(args.model_name, device=args.device)

    rec_prompt = get_prompts_by_pillar(pillar="dose_response", limit=1, seed=args.seed)[0]
    base_prompt = get_prompts_by_pillar(pillar="baseline", limit=1, seed=args.seed)[0]
    layer_sweep(
        model,
        tokenizer,
        rec_prompt,
        base_prompt,
        windows=[8, 16, 32],
        device=args.device,
        out_csv="DEC11_2025_FULL_PIPELINE/results/phase1_layer_sweep.csv",
    )

    categories = {
        "recursive": get_prompts_by_pillar(pillar="dose_response", limit=20, seed=args.seed),
        "baseline": get_prompts_by_pillar(pillar="baseline", limit=20, seed=args.seed),
        "repetitive": get_prompts_by_pillar(group_prefix="repetitive", limit=10, seed=args.seed),
        "long": get_prompts_by_pillar(group_prefix="long_control", limit=10, seed=args.seed),
        "pseudo_recursive": get_prompts_by_pillar(group_prefix="pseudo_recursive", limit=10, seed=args.seed),
        "creative": get_prompts_by_pillar(group_prefix="baseline_creative", limit=10, seed=args.seed),
    }
    prompt_battery(
        model,
        tokenizer,
        categories=categories,
        device=args.device,
        out_csv="DEC11_2025_FULL_PIPELINE/results/phase1_prompt_battery.csv",
    )
    print("[phase1] wrote layer sweep and prompt battery CSVs.")


if __name__ == "__main__":
    main()

