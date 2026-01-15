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
from prompts.loader import PromptLoader
from src.pipelines.registry import ExperimentResult


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
    
    loader = PromptLoader()
    # Log prompt bank version (DEC15 hygiene)
    bank_version = loader.version
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "prompt_bank_version.txt"), "w", encoding="utf-8") as f:
        f.write(bank_version)

    # Get standard prompt pairs
    rec_prompts = loader.get_by_pillar(pillar="dose_response", limit=1, seed=seed)
    base_prompts = loader.get_by_pillar(pillar="baselines", limit=1, seed=seed)
    
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
        "recursive": loader.get_by_pillar(pillar="dose_response", limit=20, seed=seed),
        "baseline": loader.get_by_pillar(pillar="baselines", limit=20, seed=seed),
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


def run_phase1_existence_from_config(cfg, run_dir):
    """
    Config-driven wrapper for Phase 1 existence proof.

    Expected config shape:
      {
        "experiment": "phase1_existence",
        "model": {"name": "...", "device": "cuda"},
        "seed": 0,
        "params": {
          "layer_sweep": {"windows": [8,16,32], "max_layer": 32},
          "battery": {"recursive_limit": 20, "baseline_limit": 20}
        }
      }
    """
    model_cfg = cfg.get("model") or {}
    params = cfg.get("params") or {}
    seed = int(cfg.get("seed") or 0)
    device = str(model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = str(model_cfg.get("name") or "mistralai/Mistral-7B-v0.1")

    set_seed(seed)
    model, tokenizer = load_model(model_name, device=device)

    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    (run_dir / "prompt_bank_version.json").write_text(
        json.dumps({"version": bank_version}, indent=2) + "\n"
    )

    # Prompts
    rec_prompts = loader.get_by_pillar(pillar="dose_response", limit=1, seed=seed)
    base_prompts = loader.get_by_pillar(pillar="baselines", limit=1, seed=seed)
    if not rec_prompts or not base_prompts:
        raise ValueError("Could not load prompts from bank")
    rec_prompt = rec_prompts[0]
    base_prompt = base_prompts[0]

    # Params
    ls = params.get("layer_sweep") or {}
    windows = ls.get("windows") or [8, 16, 32]
    max_layer = int(ls.get("max_layer") or 32)

    bat = params.get("battery") or {}
    recursive_limit = int(bat.get("recursive_limit") or 20)
    baseline_limit = int(bat.get("baseline_limit") or 20)

    # Outputs in run_dir
    layer_sweep_csv = str(run_dir / "phase1_layer_sweep.csv")
    battery_csv = str(run_dir / "phase1_prompt_battery.csv")

    # Layer sweep
    rows = []
    for layer in range(0, max_layer):
        for window in windows:
            rv_rec = compute_rv(model, tokenizer, rec_prompt, early=5, late=layer, window=window, device=device)
            rv_base = compute_rv(model, tokenizer, base_prompt, early=5, late=layer, window=window, device=device)
            rows.append(
                {
                    "layer": layer,
                    "window": window,
                    "rv_recursive": rv_rec,
                    "rv_baseline": rv_base,
                    "separation": rv_base - rv_rec,
                }
            )
    os.makedirs(str(run_dir), exist_ok=True)
    with open(layer_sweep_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Battery
    categories = {
        "recursive": loader.get_by_pillar(pillar="dose_response", limit=recursive_limit, seed=seed),
        "baseline": loader.get_by_pillar(pillar="baselines", limit=baseline_limit, seed=seed),
    }
    rows = []
    for name, prompts in categories.items():
        for p in prompts:
            rv = compute_rv(model, tokenizer, p, device=device)
            rows.append({"category": name, "prompt": p[:120], "rv": rv})
    with open(battery_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return ExperimentResult(
        summary={
            "experiment": "phase1_existence",
            "model_name": model_name,
            "device": device,
            "seed": seed,
            "prompt_bank_version": bank_version,
            "artifacts": {
                "layer_sweep_csv": layer_sweep_csv,
                "prompt_battery_csv": battery_csv,
            },
            "notes": {
                "rec_prompt_preview": rec_prompt[:180],
                "base_prompt_preview": base_prompt[:180],
            },
        }
    )


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

