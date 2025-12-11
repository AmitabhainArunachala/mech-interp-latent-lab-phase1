"""
Phase 6: Robustness & Confounds
- Confound battery (R_V only)
- Length matching check
- Seed stability (repeat runs)
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    compute_rv,
    get_prompts_by_pillar,
    load_model,
    set_seed,
)


def confound_battery(model, tokenizer, device: str, seed: int, out_csv: str):
    categories = {
        "recursive": get_prompts_by_pillar(pillar="dose_response", limit=20, seed=seed),
        "repetitive": get_prompts_by_pillar(group_prefix="repetitive", limit=20, seed=seed),
        "long": get_prompts_by_pillar(group_prefix="long_control", limit=20, seed=seed),
        "pseudo_recursive": get_prompts_by_pillar(group_prefix="pseudo_recursive", limit=20, seed=seed),
        "self_nonrec": get_prompts_by_pillar(group_prefix="self", limit=20, seed=seed),
    }
    rows = []
    for cat, prompts in categories.items():
        for p in prompts:
            rv = compute_rv(model, tokenizer, p, device=device)
            rows.append({"category": cat, "prompt": p[:120], "rv": rv})
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def length_match_check(model, tokenizer, device: str, seed: int, out_csv: str):
    rec = get_prompts_by_pillar(pillar="dose_response", limit=20, seed=seed)
    base = get_prompts_by_pillar(pillar="baseline", limit=20, seed=seed)
    min_len = min(len(tokenizer(p)["input_ids"]) for p in rec + base)
    rec_trim = [p[: min_len // 4 * 4] for p in rec]
    base_trim = [p[: min_len // 4 * 4] for p in base]
    rows = []
    for p in rec_trim:
        rv = compute_rv(model, tokenizer, p, device=device)
        rows.append({"category": "recursive_trim", "rv": rv, "prompt": p[:120]})
    for p in base_trim:
        rv = compute_rv(model, tokenizer, p, device=device)
        rows.append({"category": "baseline_trim", "rv": rv, "prompt": p[:120]})
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

    confound_battery(
        model,
        tokenizer,
        device=args.device,
        seed=args.seed,
        out_csv="DEC11_2025_FULL_PIPELINE/results/phase6_confound_battery.csv",
    )
    length_match_check(
        model,
        tokenizer,
        device=args.device,
        seed=args.seed,
        out_csv="DEC11_2025_FULL_PIPELINE/results/phase6_length_match.csv",
    )
    print("[phase6] wrote confound and length-match CSVs.")


if __name__ == "__main__":
    main()

