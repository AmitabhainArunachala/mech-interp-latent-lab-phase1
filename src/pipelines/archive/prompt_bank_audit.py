#!/usr/bin/env python3
"""
PROMPT BANK AUDIT: Gold-Standard Quality Assessment (SCAFFOLD)
==============================================================

Implements the prompt quality standardization plan:
- Multi-dimensional quality metrics (geometry + persistence + control sensitivity)
- Layer stress testing
- Confound controls
- Reproducible measurement contract

NOTE: This is currently a scaffold/partial implementation.
- Geometry metrics are fully implemented.
- Persistence metrics are simplified (Step 0 vs Step 1 only).
- Control sensitivity is a placeholder.

Output: Single CSV/JSON report with per-prompt metrics
"""

import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
# from scipy.stats import cohen_d  # Not available in scipy.stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.models import load_model, set_seed
from src.core.hooks import capture_v_projection
from src.metrics.rv import participation_ratio, compute_rv
from prompts.loader import PromptLoader

# =============================================================================
# MEASUREMENT CONTRACT (LOCKED)
# =============================================================================

EARLY_LAYER = 5
LATE_LAYER = 27  # Will be adjusted for model
WINDOW_SIZE = 16
CONTRACTION_THRESHOLD = 0.8
SEED = 42
TEMPERATURES = [0.0, 0.7]  # Tier 1 (reproducibility) and Tier 2 (robustness)
MAX_GENERATION_STEPS = 20

# =============================================================================
# PROMPT BANK VERSIONING
# =============================================================================

def compute_bank_hash(bank_path: Path) -> str:
    """Compute hash of prompt bank for version tracking."""
    with open(bank_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


# =============================================================================
# METRIC A: PROMPT-PASS GEOMETRY
# =============================================================================

def compute_prompt_pass_geometry(
    model,
    tokenizer,
    prompt: str,
    device: str = "cuda",
) -> Dict:
    """
    Compute R_V at canonical layers.
    
    Returns:
        {
            "rv_l27": float,
            "rv_l5": float,  # For reference
            "pr_early": float,
            "pr_late": float,
        }
    """
    rv = compute_rv(model, tokenizer, prompt, early=EARLY_LAYER, late=LATE_LAYER, window=WINDOW_SIZE, device=device)
    
    # Also get individual PRs for analysis
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        with capture_v_projection(model, EARLY_LAYER) as v_early_storage:
            with capture_v_projection(model, LATE_LAYER) as v_late_storage:
                model(**inputs)
        
        v_early = v_early_storage.get("v")
        v_late = v_late_storage.get("v")
        
        pr_early = participation_ratio(v_early, window_size=WINDOW_SIZE) if v_early is not None else float('nan')
        pr_late = participation_ratio(v_late, window_size=WINDOW_SIZE) if v_late is not None else float('nan')
    
    return {
        "rv_l27": rv,
        "pr_early": pr_early,
        "pr_late": pr_late,
    }


# =============================================================================
# METRIC B: MULTI-TOKEN PERSISTENCE
# =============================================================================

def compute_multi_token_persistence(
    model,
    tokenizer,
    prompt: str,
    temperature: float,
    device: str = "cuda",
) -> Dict:
    """
    Compute persistence metrics across generation.
    
    Returns:
        {
            "persistence_ratio": float,  # Fraction of steps with R_V < threshold
            "crossings": int,  # Number of threshold crossings
            "mean_rv": float,
            "std_rv": float,
            "min_rv": float,
            "max_rv": float,
        }
    """
    # Simplified version - full implementation would use experiment_multi_token_generation logic
    # For now, compute at step 0 (prompt pass) and step 1 (first generation step)
    
    rv_0 = compute_rv(model, tokenizer, prompt, early=EARLY_LAYER, late=LATE_LAYER, window=WINDOW_SIZE, device=device)
    
    # Generate one token and measure
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        past_key_values = outputs.past_key_values
        
        # Generate one token
        if temperature == 0.0:
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
        else:
            probs = torch.softmax(outputs.logits[:, -1, :] / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        
        extended_ids = torch.cat([inputs["input_ids"], next_token_id.unsqueeze(0)], dim=1)
        rv_1 = compute_rv(model, tokenizer, tokenizer.decode(extended_ids[0]), early=EARLY_LAYER, late=LATE_LAYER, window=WINDOW_SIZE, device=device)
    
    rv_values = [rv_0, rv_1]
    contracted = [rv < CONTRACTION_THRESHOLD for rv in rv_values if not np.isnan(rv)]
    persistence_ratio = sum(contracted) / len(contracted) if contracted else float('nan')
    
    crossings = 0
    if len(contracted) > 1:
        was_contracted = contracted[0]
        for is_contracted in contracted[1:]:
            if is_contracted != was_contracted:
                crossings += 1
            was_contracted = is_contracted
    
    return {
        "persistence_ratio": persistence_ratio,
        "crossings": crossings,
        "mean_rv": float(np.nanmean(rv_values)),
        "std_rv": float(np.nanstd(rv_values)),
        "min_rv": float(np.nanmin(rv_values)),
        "max_rv": float(np.nanmax(rv_values)),
    }


# =============================================================================
# METRIC C: CONTROL SENSITIVITY (PLACEHOLDER)
# =============================================================================

def compute_control_sensitivity(
    model,
    tokenizer,
    prompt: str,
    device: str = "cuda",
) -> Dict:
    """
    Compute control sensitivity metrics.
    
    Placeholder for now - would test:
    - Length matching
    - Style/semantics minimal pairs
    - Pseudo-recursive controls
    
    Returns:
        {
            "survives_length_control": bool,
            "survives_style_control": bool,
            "survives_pseudo_recursive": bool,
        }
    """
    # TODO: Implement control tests
    return {
        "survives_length_control": True,  # Placeholder
        "survives_style_control": True,  # Placeholder
        "survives_pseudo_recursive": True,  # Placeholder
    }


def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d for two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    return (np.mean(group1) - np.mean(group2)) / pooled_se


# =============================================================================
# MAIN AUDIT PIPELINE
# =============================================================================

def main():
    print("=" * 80)
    print("PROMPT BANK AUDIT: Gold-Standard Quality Assessment")
    print("=" * 80)
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/prompt_bank_audit/runs/{timestamp}_audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "mistralai/Mistral-7B-v0.1"
    
    print(f"\n[1/5] Loading model...")
    set_seed(SEED)
    model, tokenizer = load_model(model_name, device=device, attn_implementation="eager")
    model.eval()
    
    # Adjust LATE_LAYER for actual model
    global LATE_LAYER
    LATE_LAYER = model.config.num_hidden_layers - 5
    
    # Load prompt bank
    print(f"\n[2/5] Loading prompt bank...")
    loader = PromptLoader()
    bank_path = loader.bank_path
    bank_hash = compute_bank_hash(bank_path)
    print(f"  Bank version: {bank_hash}")
    
    # Get all prompts by group
    all_prompts = []
    for prompt_id, prompt_data in loader.prompts.items():
        all_prompts.append({
            "prompt_id": prompt_id,
            "text": prompt_data.get("text", ""),
            "group": prompt_data.get("group", "unknown"),
            "pillar": prompt_data.get("pillar", "unknown"),
            "type": prompt_data.get("type", "unknown"),
        })
    
    print(f"  Loaded {len(all_prompts)} prompts")
    
    # Compute metrics for each prompt
    print(f"\n[3/5] Computing metrics...")
    results = []
    
    for i, prompt_info in enumerate(all_prompts):
        prompt_id = prompt_info["prompt_id"]
        prompt_text = prompt_info["text"]
        
        if i % 10 == 0:
            print(f"  Processing {i+1}/{len(all_prompts)}...")
        
        try:
            # Metric A: Prompt-pass geometry
            geometry = compute_prompt_pass_geometry(model, tokenizer, prompt_text, device)
            
            # Metric B: Multi-token persistence (T=0 and T=0.7)
            persistence_t0 = compute_multi_token_persistence(model, tokenizer, prompt_text, temperature=0.0, device=device)
            persistence_t07 = compute_multi_token_persistence(model, tokenizer, prompt_text, temperature=0.7, device=device)
            
            # Metric C: Control sensitivity
            controls = compute_control_sensitivity(model, tokenizer, prompt_text, device)
            
            results.append({
                "prompt_id": prompt_id,
                "prompt_group": prompt_info["group"],
                "prompt_pillar": prompt_info["pillar"],
                "prompt_type": prompt_info["type"],
                "prompt_text": prompt_text[:100],  # Truncate for CSV
                # Geometry
                "rv_prompt_pass": geometry["rv_l27"],
                "pr_early": geometry["pr_early"],
                "pr_late": geometry["pr_late"],
                # Persistence T=0
                "persistence_t0": persistence_t0["persistence_ratio"],
                "crossings_t0": persistence_t0["crossings"],
                "mean_rv_t0": persistence_t0["mean_rv"],
                # Persistence T=0.7
                "persistence_t07": persistence_t07["persistence_ratio"],
                "crossings_t07": persistence_t07["crossings"],
                "mean_rv_t07": persistence_t07["mean_rv"],
                # Controls
                "survives_length_control": controls["survives_length_control"],
                "survives_style_control": controls["survives_style_control"],
                "survives_pseudo_recursive": controls["survives_pseudo_recursive"],
            })
        except Exception as e:
            print(f"  ⚠️  Error processing {prompt_id}: {e}")
            continue
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "prompt_audit_results.csv", index=False)
    
    # Compute group-level statistics
    print(f"\n[4/5] Computing group-level statistics...")
    group_stats = {}
    for group in df["prompt_group"].unique():
        group_df = df[df["prompt_group"] == group]
        group_stats[group] = {
            "n_prompts": len(group_df),
            "mean_rv": float(group_df["rv_prompt_pass"].mean()),
            "std_rv": float(group_df["rv_prompt_pass"].std()),
            "mean_persistence_t0": float(group_df["persistence_t0"].mean()),
            "mean_persistence_t07": float(group_df["persistence_t07"].mean()),
        }
    
    # Compute separation statistics vs baselines
    baseline_groups = df[df["prompt_pillar"] == "baselines"]["prompt_group"].unique()
    recursive_groups = df[df["prompt_pillar"] == "dose_response"]["prompt_group"].unique()
    
    separation_stats = {}
    for rec_group in recursive_groups:
        rec_df = df[df["prompt_group"] == rec_group]
        rec_rv = rec_df["rv_prompt_pass"].dropna()
        
        for base_group in baseline_groups:
            base_df = df[df["prompt_group"] == base_group]
            base_rv = base_df["rv_prompt_pass"].dropna()
            
            if len(rec_rv) > 1 and len(base_rv) > 1:
                d = calculate_cohens_d(rec_rv, base_rv)
                separation_stats[f"{rec_group}_vs_{base_group}"] = {
                    "cohens_d": float(d),
                    "recursive_mean": float(rec_rv.mean()),
                    "baseline_mean": float(base_rv.mean()),
                }
    
    # Save summary
    summary = {
        "experiment": "prompt_bank_audit",
        "timestamp": timestamp,
        "model": model_name,
        "prompt_bank_version": bank_hash,
        "prompt_bank_path": str(bank_path),
        "measurement_contract": {
            "early_layer": EARLY_LAYER,
            "late_layer": LATE_LAYER,
            "window_size": WINDOW_SIZE,
            "contraction_threshold": CONTRACTION_THRESHOLD,
            "temperatures": TEMPERATURES,
            "seed": SEED,
        },
        "n_prompts": len(results),
        "group_statistics": group_stats,
        "separation_statistics": separation_stats,
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[5/5] Results saved to: {output_dir}")
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nPrompt bank version: {bank_hash}")
    print(f"Total prompts audited: {len(results)}")
    print(f"\nGroup statistics:")
    for group, stats in sorted(group_stats.items()):
        print(f"  {group}: n={stats['n_prompts']}, mean_R_V={stats['mean_rv']:.4f}")
    print(f"\n✅ Audit complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

