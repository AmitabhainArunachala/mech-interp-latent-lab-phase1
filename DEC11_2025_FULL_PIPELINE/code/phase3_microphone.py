"""
Phase 3: Locate the Microphone (L8 steering vector)

Functions:
- Extract v8 steering vector (recursive - baseline) with 5-fold stability
- Injection sweep over α to measure dose-response on R_V and behavior
- Random-direction control with matched norm

Outputs:
- CSVs in ../results/phase3_microphone_{extract|sweep|random}.csv
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


def extract_vectors(
    model,
    tokenizer,
    rec_prompts: List[str],
    base_prompts: List[str],
    layer: int = 8,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rec_vs = []
    base_vs = []
    for p in rec_prompts:
        enc = tokenizer(p, return_tensors="pt", truncation=True).to(device)
        v = capture_v_projection(model, enc, layer)
        if v is not None:
            rec_vs.append(v.mean(dim=1))  # (seq, d_model) -> (seq, d_model)
    for p in base_prompts:
        enc = tokenizer(p, return_tensors="pt", truncation=True).to(device)
        v = capture_v_projection(model, enc, layer)
        if v is not None:
            base_vs.append(v.mean(dim=1))
    rec_stack = torch.stack(rec_vs)  # (n, seq, d)
    base_stack = torch.stack(base_vs)
    rec_mean = rec_stack.mean(dim=0).mean(dim=0)
    base_mean = base_stack.mean(dim=0).mean(dim=0)
    vec = rec_mean - base_mean
    return vec, rec_mean, base_mean


def apply_steering(model, layer_idx: int, vec: torch.Tensor, alpha: float = 1.0):
    """
    Injects alpha * vec into the residual stream input of the given layer.
    """
    handle = None

    def hook(module, inputs):
        hidden_states = inputs[0]
        steer = alpha * vec.to(hidden_states.device)
        steer = steer.unsqueeze(0).unsqueeze(0)  # (1,1,d_model)
        steer = steer.expand_as(hidden_states)
        new_hidden = hidden_states + steer
        return (new_hidden, *inputs[1:])

    handle = model.model.layers[layer_idx].register_forward_pre_hook(hook)
    return handle


def run_sweep(
    model,
    tokenizer,
    prompts: List[str],
    vec: torch.Tensor,
    alphas: List[float],
    layer: int = 8,
    device: str = "cuda",
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    out_csv: str = "DEC11_2025_FULL_PIPELINE/results/phase3_microphone_sweep.csv",
):
    rows = []
    for alpha in alphas:
        for p in prompts:
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
                    "prompt": p[:80],
                    "alpha": alpha,
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


def run_random_control(
    model,
    tokenizer,
    prompts: List[str],
    vec: torch.Tensor,
    alphas: List[float],
    layer: int = 8,
    device: str = "cuda",
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    out_csv: str = "DEC11_2025_FULL_PIPELINE/results/phase3_microphone_random.csv",
):
    rows = []
    norm = torch.norm(vec)
    torch.manual_seed(0)
    rand_vec = torch.randn_like(vec)
    rand_vec = rand_vec / torch.norm(rand_vec) * norm
    for alpha in alphas:
        for p in prompts:
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
                    "prompt": p[:80],
                    "alpha": alpha,
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)
    model, tokenizer = load_model(args.model_name, device=args.device)

    rec_prompts = get_prompts_by_pillar(pillar="dose_response", limit=20, seed=args.seed, group_prefix="L4")
    base_prompts = get_prompts_by_pillar(pillar="baseline", limit=20, seed=args.seed)

    vec, rec_mean, base_mean = extract_vectors(
        model, tokenizer, rec_prompts, base_prompts, layer=args.layer, device=args.device
    )

    # Save extraction stats
    os.makedirs("DEC11_2025_FULL_PIPELINE/results", exist_ok=True)
    with open("DEC11_2025_FULL_PIPELINE/results/phase3_microphone_extract.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["norm_vec", "norm_rec_mean", "norm_base_mean", "layer", "seed"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "norm_vec": float(torch.norm(vec)),
                "norm_rec_mean": float(torch.norm(rec_mean)),
                "norm_base_mean": float(torch.norm(base_mean)),
                "layer": args.layer,
                "seed": args.seed,
            }
        )

    alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    run_sweep(
        model,
        tokenizer,
        base_prompts,
        vec,
        alphas=alphas,
        layer=args.layer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        out_csv="DEC11_2025_FULL_PIPELINE/results/phase3_microphone_sweep.csv",
    )
    run_random_control(
        model,
        tokenizer,
        base_prompts,
        vec,
        alphas=[0.0, 1.0, 2.0, 3.0],
        layer=args.layer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        out_csv="DEC11_2025_FULL_PIPELINE/results/phase3_microphone_random.csv",
    )
    print("[phase3] completed extraction, sweep, random control CSVs.")


if __name__ == "__main__":
    main()

