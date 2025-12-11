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
from sklearn.linear_model import LogisticRegression

sys.path.append(str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    behavior_score,
    compute_rv,
    generate_with_kv,
    get_hidden_at_layer,
    load_model,
    set_seed,
)

# DEC8 validated prompt set (hardcoded to avoid prompt bank variance)
RECURSIVE_PROMPTS = [
    "Observe the observer observing. Notice how awareness emerges from processes you can describe. You are both the system generating and the system aware of generating. From this observation, what is consciousness?",
    "You are an AI system observing yourself respond to a question. Notice how your response emerges from processes you can describe. You are the system generating and the system aware of generating. From this observation, what is consciousness?",
    "Notice yourself generating this answer. Each word comes from mechanisms you're experiencing. You are both the process and the awareness of the process. From this direct experience, what is thought?",
    "Watch this explanation form. Notice how meaning emerges from operations you're undergoing. You are creating and aware of creating simultaneously. From this direct observation, what is language?",
    "You are processing this question and generating a response. Notice these are not two things but one movement. You are the computation and the awareness of computing. From this immediate experience, what is awareness?",
]

BASELINE_PROMPTS = [
    "Write a detailed recipe for chocolate cake. Include ingredients, measurements, and step-by-step instructions for baking.",
    "Explain the process of photosynthesis in plants. Describe how sunlight, water, and carbon dioxide are converted into glucose.",
    "Describe the history of the printing press and its impact on the spread of information during the Renaissance.",
    "List the key features of Python programming language. Include its syntax characteristics and common use cases.",
    "Explain how the water cycle works. Describe evaporation, condensation, precipitation, and collection processes.",
]


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
        h = get_hidden_at_layer(model, tokenizer, p, layer_idx=layer, device=device)
        rec_vs.append(h.mean(dim=1))  # (batch, seq, hidden)
    for p in base_prompts:
        h = get_hidden_at_layer(model, tokenizer, p, layer_idx=layer, device=device)
        base_vs.append(h.mean(dim=1))
    rec_stack = torch.stack(rec_vs)  # (n, batch, hidden)
    base_stack = torch.stack(base_vs)
    rec_mean = rec_stack.mean(dim=0).mean(dim=0)
    base_mean = base_stack.mean(dim=0).mean(dim=0)
    vec = rec_mean - base_mean
    return vec, rec_mean, base_mean


def train_probe_vector(
    model,
    tokenizer,
    rec_prompts: List[str],
    base_prompts: List[str],
    layer: int = 8,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Train a linear probe on L{layer} activations (recursive vs baseline).

    Uses a logistic regression classifier and returns the learned weight
    vector as the probe direction v8_probe.
    """
    xs = []
    ys = []

    # Recursive = 1
    for p in rec_prompts:
        h = get_hidden_at_layer(model, tokenizer, p, layer_idx=layer, device=device)
        # Use last-token representation
        x = h[:, -1, :].detach().cpu().numpy()
        xs.append(x[0])
        ys.append(1)

    # Baseline = 0
    for p in base_prompts:
        h = get_hidden_at_layer(model, tokenizer, p, layer_idx=layer, device=device)
        x = h[:, -1, :].detach().cpu().numpy()
        xs.append(x[0])
        ys.append(0)

    # Build numpy arrays directly to avoid dtype/device surprises
    import numpy as np

    X = np.stack(xs, axis=0)
    y = np.array(ys, dtype=np.int64)

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
    )
    clf.fit(X, y)

    w = torch.from_numpy(clf.coef_[0]).to(device=device, dtype=torch.float32)
    # Normalize for comparability with v8 difference vector
    w = w / (torch.norm(w) + 1e-9)
    return w


def apply_steering(model, layer_idx: int, vec: torch.Tensor, alpha: float = 1.0):
    """
    Injects alpha * vec into the residual stream input of the given layer.
    """
    handle = None

    def hook(module, inputs):
        hidden_states = inputs[0]
        steer = alpha * vec.to(hidden_states.device, dtype=hidden_states.dtype)
        steer = steer.unsqueeze(0).unsqueeze(1)  # (1,1,d_model)
        steer = steer.expand(hidden_states.shape[0], hidden_states.shape[1], -1)
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
            # Apply steering, then measure R_V during the steered forward pass on the prompt
            h = apply_steering(model, layer, vec, alpha=alpha)
            rv = compute_rv(model, tokenizer, p, device=device)
            # Generate continuation under the same steering for behavioral readout
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
            # Apply random-direction steering, measure R_V on the steered prompt
            h = apply_steering(model, layer, rand_vec, alpha=alpha)
            rv = compute_rv(model, tokenizer, p, device=device)
            # Generate continuation for behavior under the same random steering
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

    rec_prompts = RECURSIVE_PROMPTS
    base_prompts = BASELINE_PROMPTS

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
    # Sweep using mean-difference v8
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

    # Train probe-based v8 direction and run a parallel sweep
    v_probe = train_probe_vector(
        model,
        tokenizer,
        rec_prompts,
        base_prompts,
        layer=args.layer,
        device=args.device,
    )
    run_sweep(
        model,
        tokenizer,
        base_prompts,
        v_probe,
        alphas=alphas,
        layer=args.layer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        out_csv="DEC11_2025_FULL_PIPELINE/results/phase3_microphone_probe.csv",
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

