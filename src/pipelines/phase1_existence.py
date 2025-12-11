"""
Phase 1: Existence Proof - Establish the R_V Symptom

Standard Conditions:
- Model: Mistral-7B Base (v0.1)
- Early layer: 5
- Late layer: 27 (num_layers - 5)
- Window: 16 tokens
"""

import csv
import os
from pathlib import Path
from typing import Dict, List

import torch

from src.core import load_model, set_seed
from src.metrics import compute_rv
from prompts.loader import get_prompts_by_pillar


def run_layer_sweep(
    model,
    tokenizer,
    rec_prompt: str,
    base_prompt: str,
    windows: List[int],
    device: str,
    out_csv: str,
):
    """Sweep across layers to find where R_V separation occurs."""
    rows = []
    for layer in range(0, 32):
        for window in windows:
            rv_rec = compute_rv(
                model, tokenizer, rec_prompt, early=5, late=layer, window=window, device=device
            )
            rv_base = compute_rv(
                model, tokenizer, base_prompt, early=5, late=layer, window=window, device=device
            )
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


def run_prompt_battery(
    model,
    tokenizer,
    categories: Dict[str, List[str]],
    device: str,
    out_csv: str,
):
    """Run R_V measurement across multiple prompt categories."""
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


def run_phase1_existence_proof(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    device: str = "cuda",
    seed: int = 0,
    results_dir: str = "results",
):
    """
    Run Phase 1: Existence Proof experiment.
    
    Args:
        model_name: Model identifier. Default: Mistral-7B Base.
        device: Target device.
        seed: Random seed.
        results_dir: Directory for output CSVs.
    
    Returns:
        Paths to output CSV files.
    """
    set_seed(seed)
    model, tokenizer = load_model(model_name, device=device)
    
    # Get standard prompt pairs
    rec_prompts = get_prompts_by_pillar(pillar="dose_response", limit=1, seed=seed)
    base_prompts = get_prompts_by_pillar(pillar="baselines", limit=1, seed=seed)
    
    if not rec_prompts or not base_prompts:
        raise ValueError("Could not load prompts from bank")
    
    rec_prompt = rec_prompts[0]
    base_prompt = base_prompts[0]
    
    # Layer sweep
    layer_sweep_csv = os.path.join(results_dir, "phase1_layer_sweep.csv")
    run_layer_sweep(
        model,
        tokenizer,
        rec_prompt,
        base_prompt,
        windows=[8, 16, 32],
        device=device,
        out_csv=layer_sweep_csv,
    )
    
    # Prompt battery
    categories = {
        "recursive": get_prompts_by_pillar(pillar="dose_response", limit=20, seed=seed),
        "baseline": get_prompts_by_pillar(pillar="baselines", limit=20, seed=seed),
    }
    
    battery_csv = os.path.join(results_dir, "phase1_prompt_battery.csv")
    run_prompt_battery(
        model,
        tokenizer,
        categories=categories,
        device=device,
        out_csv=battery_csv,
    )
    
    print(f"[phase1] Layer sweep: {layer_sweep_csv}")
    print(f"[phase1] Prompt battery: {battery_csv}")
    
    return layer_sweep_csv, battery_csv


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    
    run_phase1_existence_proof(
        model_name=args.model_name,
        device=args.device,
        seed=args.seed,
        results_dir=args.results_dir,
    )

