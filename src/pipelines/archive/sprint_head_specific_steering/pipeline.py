"""
Sprint Pipeline: Head-Specific Steering & Mode Score M.

Goal: Validate H18/H26 causality using Mode Score M (logit-level metric).
Method: 4-Pass IOI-Style test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core.models import load_model, set_seed
from src.core.patching import extract_v_activation
from src.core.head_specific_patching import HeadSpecificVPatcher, HeadSpecificSteeringPatcher
from src.metrics.mode_score import ModeScoreMetric
from src.pipelines.archive.steering import compute_steering_vector
from src.pipelines.registry import ExperimentResult
from prompts.loader import PromptLoader

def run_sprint_head_specific_steering(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-v0.1")
    n_pairs = params.get("n_pairs", 20)
    layer_idx = params.get("layer", 27)
    seed = int(params.get("seed", 42))
    
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(model_name, device=device)
    
    # 1. Setup Metric
    print("Initializing Mode Score Metric...")
    metric = ModeScoreMetric(tokenizer, device=device)
    
    # 2. Setup Data (Recursive & Baseline Prompts)
    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    (run_dir / "prompt_bank_version.json").write_text(json.dumps({"version": bank_version}, indent=2) + "\n")
    # For vector computation
    rec_pool = loader.get_by_group("L4_full") + loader.get_by_group("L5_refined")
    base_pool = loader.get_by_group("baseline_math") + loader.get_by_group("baseline_factual")
    
    # Compute Global Steering Vector (for Corrupted/Steering Control)
    print("Computing Global Steering Vector...")
    n_rec_avail = len(rec_pool)
    n_base_avail = len(base_pool)
    n_sample = 50
    
    rec_sample = np.random.choice(rec_pool, min(n_sample, n_rec_avail), replace=False).tolist()
    base_sample = np.random.choice(base_pool, min(n_sample, n_base_avail), replace=False).tolist()
    global_vec = compute_steering_vector(model, tokenizer, rec_sample, base_sample, layer_idx, device)
    
    # Test Pairs
    pairs = loader.get_balanced_pairs(n_pairs=n_pairs, seed=seed)
    
    results = []
    
    print(f"\nRunning 4-Pass Test on {len(pairs)} pairs...")
    
    for i, (rec_text, base_text) in enumerate(tqdm(pairs)):
        try:
            # --- Pass 1: Clean Baseline ---
            inputs_b = tokenizer(base_text, return_tensors="pt", add_special_tokens=False).to(device)
            with torch.no_grad():
                out_b = model(**inputs_b)
            # Compute M (using baseline logits to define T is circular? 
            # No, T is defined by baseline logits. M measures shift AWAY from T towards R.)
            score_b = metric.compute_score(out_b.logits, baseline_logits=out_b.logits)
            
            # --- Pass 2: Clean Recursive (Donor) ---
            inputs_r = tokenizer(rec_text, return_tensors="pt", add_special_tokens=False).to(device)
            with torch.no_grad():
                out_r = model(**inputs_r)
            # Cannot compare token-wise to baseline (different lengths/content)
            # Use None for baseline -> M = LSE(R) (Raw strength)
            score_r = metric.compute_score(out_r.logits, baseline_logits=None)
            
            # Extract Donor V-Activation for transplant
            donor_v = extract_v_activation(model, tokenizer, rec_text, layer_idx, device)
            
            # --- Pass 3: Corrupted (Random Steering) ---
            # Random vector
            rand_vec = torch.randn_like(global_vec)
            rand_vec = rand_vec * (global_vec.norm() / rand_vec.norm()) # Norm match
            
            steer_patcher = HeadSpecificSteeringPatcher(model, rand_vec, target_heads=list(range(32)), alpha=1.0)
            steer_patcher.register(layer_idx)
            try:
                with torch.no_grad():
                    out_c = model(**inputs_b)
                score_c = metric.compute_score(out_c.logits, baseline_logits=out_b.logits)
            finally:
                steer_patcher.remove()
                
            # --- Pass 4: Patched (Surgical H18+H26 Transplant) ---
            # We use HeadSpecificVPatcher to transplant H18+H26 from Donor -> Baseline
            # Note: H18 and H26 are standard Mistral "recursive heads" (need to verify mapping if not standard)
            # Assuming 18 and 26 are the indices.
            
            transplant_patcher = HeadSpecificVPatcher(model, donor_v, target_heads=[18, 26])
            transplant_patcher.register(layer_idx)
            try:
                with torch.no_grad():
                    out_p = model(**inputs_b)
                score_p = metric.compute_score(out_p.logits, baseline_logits=out_b.logits)
            finally:
                transplant_patcher.remove()
                
            # --- Pass 5: Global Steering (Control) ---
            # Steer all heads with global vector
            steer_all = HeadSpecificSteeringPatcher(model, global_vec, target_heads=list(range(32)), alpha=1.0)
            steer_all.register(layer_idx)
            try:
                with torch.no_grad():
                    out_g = model(**inputs_b)
                score_g = metric.compute_score(out_g.logits, baseline_logits=out_b.logits)
            finally:
                steer_all.remove()

            # Record
            results.append({
                "pair_idx": i,
                "score_baseline": score_b,
                "score_recursive": score_r,
                "score_corrupted": score_c,
                "score_surgical": score_p,
                "score_global": score_g,
                "delta_surgical": score_p - score_b,
                "delta_global": score_g - score_b
            })
            
        except Exception as e:
            print(f"Error on pair {i}: {e}")
            continue
            
    # Save & Summarize
    df = pd.DataFrame(results)
    df.to_csv(run_dir / "sprint_results.csv", index=False)
    
    summary = {
        "experiment": "sprint_head_specific_steering",
        "mean_scores": {
            "baseline": float(df["score_baseline"].mean()),
            "recursive": float(df["score_recursive"].mean()),
            "corrupted": float(df["score_corrupted"].mean()),
            "surgical": float(df["score_surgical"].mean()),
            "global": float(df["score_global"].mean())
        },
        "deltas": {
            "surgical_vs_baseline": float(df["delta_surgical"].mean()),
            "global_vs_baseline": float(df["delta_global"].mean())
        }
    }
    
    return ExperimentResult(summary=summary)

