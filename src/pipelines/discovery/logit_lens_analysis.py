"""
Logit Lens Analysis Pipeline.

Runs logit lens + logit difference analysis for recursive vs baseline prompts.
Finds crystallization points and recursive emergence layers.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm import tqdm

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.metrics.logit_lens import (
    compute_logit_lens_trajectory,
    find_recursive_emergence,
    LogitLensResult,
)
from src.metrics.logit_diff import LogitDiffMetric
from src.metrics.rv import compute_rv
from src.utils.run_metadata import get_run_metadata, save_metadata, append_to_run_index
from src.pipelines.registry import ExperimentResult


RECURSIVE_TOKENS_TO_TRACK = [
    "self", "itself", "observer", "awareness", "consciousness",
    "recursive", "loop", "reflection", "solution", "process",
]


def run_logit_lens_analysis_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """Run comprehensive logit lens analysis."""
    
    model_cfg = cfg.get("model", {})
    params = cfg.get("params", {})
    
    seed = int(cfg.get("seed", 42))
    device = str(model_cfg.get("device", "cuda"))
    model_name = str(model_cfg.get("name", "mistralai/Mistral-7B-v0.1"))
    
    n_pairs = int(params.get("n_pairs", 30))
    top_k = int(params.get("top_k", 5))
    
    set_seed(seed)
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name, device=device)
    model.eval()
    
    # Load prompts
    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    
    pairs_with_ids = loader.get_balanced_pairs_with_ids(n_pairs=n_pairs, seed=seed)
    
    # Initialize metrics
    logit_diff_metric = LogitDiffMetric(tokenizer, device=device)
    
    results = []
    
    for rec_id, base_id, rec_text, base_text in tqdm(pairs_with_ids, desc="Analyzing prompts"):
        
        # === Recursive Prompt Analysis ===
        rec_results, rec_meta = compute_logit_lens_trajectory(
            model, tokenizer, rec_text, target_position=-1, top_k=top_k, device=device
        )
        
        # Get hidden states for logit diff trajectory
        rec_inputs = tokenizer(rec_text, return_tensors="pt").to(device)
        with torch.no_grad():
            rec_outputs = model(**rec_inputs, output_hidden_states=True)
        
        rec_logit_diff_traj = logit_diff_metric.compute_trajectory(
            rec_outputs.hidden_states, model, position=-1
        )
        rec_crossover = logit_diff_metric.find_crossover_layer(rec_logit_diff_traj)
        
        # Find recursive token emergence
        rec_emergence = find_recursive_emergence(rec_results, RECURSIVE_TOKENS_TO_TRACK)
        
        # Compute R_V for comparison
        rec_rv = compute_rv(model, tokenizer, rec_text, early=5, late=27, window=16, device=device)
        
        # === Baseline Prompt Analysis ===
        base_results, base_meta = compute_logit_lens_trajectory(
            model, tokenizer, base_text, target_position=-1, top_k=top_k, device=device
        )
        
        base_inputs = tokenizer(base_text, return_tensors="pt").to(device)
        with torch.no_grad():
            base_outputs = model(**base_inputs, output_hidden_states=True)
        
        base_logit_diff_traj = logit_diff_metric.compute_trajectory(
            base_outputs.hidden_states, model, position=-1
        )
        base_crossover = logit_diff_metric.find_crossover_layer(base_logit_diff_traj)
        
        base_rv = compute_rv(model, tokenizer, base_text, early=5, late=27, window=16, device=device)
        
        # === Compile Results ===
        row = {
            "recursive_prompt_id": rec_id,
            "baseline_prompt_id": base_id,
            
            # R_V (our geometric metric)
            "rv_recursive": rec_rv,
            "rv_baseline": base_rv,
            "rv_delta": rec_rv - base_rv,
            
            # Logit Lens: Crystallization
            "rec_crystallization_layer": rec_meta["crystallization_layer"],
            "rec_final_prediction": rec_meta["final_prediction"],
            "rec_final_prob": rec_meta["final_prob"],
            "rec_min_entropy_layer": rec_meta["min_entropy_layer"],
            
            "base_crystallization_layer": base_meta["crystallization_layer"],
            "base_final_prediction": base_meta["final_prediction"],
            "base_final_prob": base_meta["final_prob"],
            "base_min_entropy_layer": base_meta["min_entropy_layer"],
            
            # Logit Difference: Crossover
            "rec_logit_diff_crossover_layer": rec_crossover,
            "base_logit_diff_crossover_layer": base_crossover,
            
            # Final logit diff values
            "rec_logit_diff_final": rec_logit_diff_traj[-1].logit_diff,
            "base_logit_diff_final": base_logit_diff_traj[-1].logit_diff,
            
            # Layer 21 logit diff (crystallization point from prior analysis)
            "rec_logit_diff_L21": rec_logit_diff_traj[21].logit_diff if len(rec_logit_diff_traj) > 21 else None,
            "base_logit_diff_L21": base_logit_diff_traj[21].logit_diff if len(base_logit_diff_traj) > 21 else None,
            
            # Layer 27 logit diff (contraction point)
            "rec_logit_diff_L27": rec_logit_diff_traj[27].logit_diff if len(rec_logit_diff_traj) > 27 else None,
            "base_logit_diff_L27": base_logit_diff_traj[27].logit_diff if len(base_logit_diff_traj) > 27 else None,
        }
        
        # Add recursive token emergence info
        for token, info in rec_emergence.items():
            row[f"rec_emergence_{token}_layer"] = info["first_appearance"]
            row[f"rec_emergence_{token}_max_prob"] = info["max_prob"]
        
        results.append(row)
        
        # Clear cache
        torch.cuda.empty_cache()
    
    # Save CSV
    df = pd.DataFrame(results)
    csv_path = run_dir / "logit_lens_analysis.csv"
    df.to_csv(csv_path, index=False)
    
    # Compute summary statistics
    summary = {
        "experiment": "logit_lens_analysis",
        "n_pairs": len(df),
        
        # R_V comparison
        "rv_recursive_mean": float(df["rv_recursive"].mean()),
        "rv_baseline_mean": float(df["rv_baseline"].mean()),
        "rv_delta_mean": float(df["rv_delta"].mean()),
        
        # Crystallization
        "rec_crystallization_layer_mean": float(df["rec_crystallization_layer"].mean()),
        "base_crystallization_layer_mean": float(df["base_crystallization_layer"].mean()),
        
        # Logit difference crossover
        "rec_crossover_layer_mean": float(df["rec_logit_diff_crossover_layer"].dropna().mean()) if df["rec_logit_diff_crossover_layer"].notna().any() else None,
        "base_crossover_layer_mean": float(df["base_logit_diff_crossover_layer"].dropna().mean()) if df["base_logit_diff_crossover_layer"].notna().any() else None,
        
        # Final logit diff
        "rec_logit_diff_final_mean": float(df["rec_logit_diff_final"].mean()),
        "base_logit_diff_final_mean": float(df["base_logit_diff_final"].mean()),
        
        # L21 and L27 logit diff
        "rec_logit_diff_L21_mean": float(df["rec_logit_diff_L21"].dropna().mean()) if df["rec_logit_diff_L21"].notna().any() else None,
        "rec_logit_diff_L27_mean": float(df["rec_logit_diff_L27"].dropna().mean()) if df["rec_logit_diff_L27"].notna().any() else None,
        
        "prompt_bank_version": bank_version,
        "artifacts": {"csv": str(csv_path)},
    }
    
    # Save summary
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save metadata
    metadata = get_run_metadata(
        cfg,
        prompt_ids=pairs_with_ids,
        eval_window=16,
        intervention_scope="none",
        behavior_metric="logit_diff",
    )
    save_metadata(run_dir, metadata)
    
    # Append to run index
    append_to_run_index(run_dir, summary)
    
    print(f"\n{'='*60}")
    print("LOGIT LENS ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"R_V recursive mean: {summary['rv_recursive_mean']:.3f}")
    print(f"R_V baseline mean: {summary['rv_baseline_mean']:.3f}")
    print(f"Rec crystallization layer mean: {summary['rec_crystallization_layer_mean']:.1f}")
    print(f"Rec logit diff crossover layer: {summary['rec_crossover_layer_mean']}")
    print(f"Rec logit diff @ L21: {summary['rec_logit_diff_L21_mean']:.3f}")
    print(f"Rec logit diff @ L27: {summary['rec_logit_diff_L27_mean']:.3f}")
    print(f"\nResults saved to: {run_dir}")
    
    return ExperimentResult(summary=summary)
