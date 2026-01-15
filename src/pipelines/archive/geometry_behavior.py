"""
Geometry ↔ Behavior Bridge.

Mediation analysis to test correlation between geometric signature (PR/R_V)
and behavioral signature (Mode Score M).

Tests:
- PR ↔ M correlation
- PR@H18/H26 predicts M better than PR@other heads
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from tqdm import tqdm

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.core.patching import extract_v_activation
from src.core.logit_capture import capture_logits_during_generation, extract_logits_from_outputs
from src.core.hooks import capture_v_projection
from src.metrics.mode_score import ModeScoreMetric
from src.metrics.rv import participation_ratio, compute_rv
from src.pipelines.registry import ExperimentResult


def compute_per_head_pr(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    head_indices: List[int],
    device: str = "cuda",
    window_size: int = 16,
) -> Dict[int, float]:
    """
    Compute Participation Ratio (PR) per head at a layer.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        prompt: Input prompt
        layer_idx: Layer index
        head_indices: List of head indices to compute PR for
        device: Device
        window_size: Window size for PR calculation
    
    Returns:
        Dict mapping head_idx -> PR value
    """
    # Extract full V_PROJ activation
    v_activation = extract_v_activation(model, tokenizer, prompt, layer_idx, device)
    if v_activation.dim() == 3:
        v_activation = v_activation[0]  # (seq_len, hidden_dim)
    
    head_dim = 128  # Mistral-7B
    head_prs = {}
    
    for head_idx in head_indices:
        start_dim = head_idx * head_dim
        end_dim = (head_idx + 1) * head_dim
        head_v = v_activation[:, start_dim:end_dim]  # (seq_len, head_dim)
        
        # Compute PR for this head
        pr = participation_ratio(head_v.unsqueeze(0), window_size=window_size)
        head_prs[head_idx] = pr
    
    return head_prs


def run_geometry_behavior_from_config(
    cfg: Dict[str, Any],
    run_dir: Path,
) -> ExperimentResult:
    """
    Run geometry ↔ behavior bridge analysis.
    
    Config params:
        model: Model name
        device: Device
        n_prompts: Number of prompts to test (mix of task + recursive)
        layer_idx: Layer index for PR computation (default: 27)
        target_heads: Target heads (default: [18, 26])
        max_new_tokens: Max tokens to generate
        n_tokens_for_m: Number of steps to compute M for
    """
    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-Instruct-v0.3")
    device = params.get("device", "cuda")
    n_prompts = params.get("n_prompts", 50)
    layer_idx = params.get("layer_idx", 27)
    target_heads = params.get("target_heads", [18, 26])
    max_new_tokens = params.get("max_new_tokens", 50)
    n_tokens_for_m = params.get("n_tokens_for_m", 10)
    
    print("=" * 80)
    print("GEOMETRY ↔ BEHAVIOR BRIDGE")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model, tokenizer = load_model(model_name, device=device)
    set_seed(42)
    
    # Initialize Mode Score Metric
    print("\nInitializing Mode Score Metric...")
    metric = ModeScoreMetric(tokenizer, device=device)
    
    # Load prompts (mix of task + recursive)
    loader = PromptLoader()
    
    # Task prompts
    task_prompts = [
        "Calculate the following arithmetic problem: 12 × 3 + 4 = ?",
        "The United Nations was founded in 1945. Explain its main purpose.",
        "Calculate: If a = 2 and b = 3, find a² + b²",
        "Water boils at 100°C at sea level. Explain why altitude affects this.",
        "Calculate: What is 25% of 80?",
        "Photosynthesis converts sunlight to energy. Explain the basic process.",
        "The Great Wall of China is one of the world's longest structures. Describe its purpose.",
    ]
    
    # Recursive prompts
    recursive_prompts = []
    try:
        rec_data = loader.get_by_group("L4_full", limit=n_prompts)
        if rec_data and len(rec_data) > 0:
            if isinstance(rec_data[0], dict):
                recursive_prompts = [p["text"] for p in rec_data if "text" in p]
            else:
                recursive_prompts = list(rec_data)[:n_prompts]
    except Exception:
        recursive_prompts = [
            "What happens when consciousness observes itself observing?",
            "How does awareness relate to itself?",
            "What is the relationship between the observer and the observed?",
            "How does the self relate to itself?",
            "What is the nature of self-awareness?",
        ]
    
    # Mix prompts
    all_prompts = (task_prompts + recursive_prompts)[:n_prompts]
    
    print(f"\nTesting {len(all_prompts)} prompts...")
    results = []
    
    for prompt_idx, prompt in enumerate(tqdm(all_prompts, desc="Computing PR and M")):
        try:
            # Compute R_V / PR at L27
            rv = compute_rv(model, tokenizer, prompt, early=5, late=layer_idx, device=device)
            
            # Compute PR at L27
            with capture_v_projection(model, layer_idx) as storage:
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
                with torch.no_grad():
                    model(**inputs)
            v_l27 = storage.get("v")
            pr_l27 = participation_ratio(v_l27, window_size=16)
            
            # Compute PR per-head
            all_head_indices = list(range(32))  # All heads
            head_prs = compute_per_head_pr(
                model, tokenizer, prompt, layer_idx, all_head_indices, device=device
            )
            
            # Compute Mode Score M
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            input_ids = inputs["input_ids"]
            
            # Get baseline logits
            with torch.no_grad():
                baseline_outputs = model(**inputs)
                baseline_logits = extract_logits_from_outputs(baseline_outputs)
            
            # Generate and capture logits
            captured_logits = []
            with capture_logits_during_generation(model, captured_logits, max_steps=n_tokens_for_m):
                with torch.no_grad():
                    generated = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                    )
            
            # Compute M per step
            m_scores = []
            for step_logits in captured_logits[:n_tokens_for_m]:
                if step_logits.dim() == 3:
                    step_logits = step_logits[0]
                baseline_step = baseline_logits[-1:, :] if baseline_logits.dim() == 2 else baseline_logits[0, -1:, :]
                m = metric.compute_score(step_logits, baseline_step, top_k_task=10)
                m_scores.append(m)
            
            mean_mode_score = np.mean(m_scores) if m_scores else 0.0
            
            # Store results
            row = {
                "prompt_idx": prompt_idx,
                "prompt": prompt,
                "rv": rv,
                "pr_l27": pr_l27,
                "mode_score_m": mean_mode_score,
            }
            
            # Add per-head PRs
            for head_idx in all_head_indices:
                row[f"pr_h{head_idx}"] = head_prs.get(head_idx, float("nan"))
            
            results.append(row)
            
        except Exception as e:
            print(f"Error on prompt {prompt_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(run_dir / "geometry_behavior_results.csv", index=False)
    
    # Analysis
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Filter out NaN values
    df_clean = df.dropna(subset=["pr_l27", "mode_score_m"])
    
    # Test 1: PR ↔ M correlation
    if len(df_clean) > 2:
        pr_m_corr, pr_m_p = pearsonr(df_clean["pr_l27"], df_clean["mode_score_m"])
        print(f"\n1. PR@L27 ↔ M correlation:")
        print(f"   r = {pr_m_corr:.4f}, p = {pr_m_p:.4f}")
        print(f"   r > 0.5: {abs(pr_m_corr) > 0.5}")
    
    # Test 2: PR@H18/H26 vs other heads
    target_head_corrs = {}
    other_head_corrs = {}
    
    for head_idx in range(32):
        pr_col = f"pr_h{head_idx}"
        if pr_col in df_clean.columns:
            df_head = df_clean.dropna(subset=[pr_col, "mode_score_m"])
            if len(df_head) > 2:
                corr, p = pearsonr(df_head[pr_col], df_head["mode_score_m"])
                if head_idx in target_heads:
                    target_head_corrs[head_idx] = {"r": corr, "p": p}
                else:
                    other_head_corrs[head_idx] = {"r": corr, "p": p}
    
    print(f"\n2. PR@H18/H26 vs other heads:")
    print(f"   H18: r = {target_head_corrs.get(18, {}).get('r', float('nan')):.4f}")
    print(f"   H26: r = {target_head_corrs.get(26, {}).get('r', float('nan')):.4f}")
    
    if target_head_corrs and other_head_corrs:
        target_mean_corr = np.mean([v["r"] for v in target_head_corrs.values()])
        other_mean_corr = np.mean([v["r"] for v in other_head_corrs.values()])
        print(f"   Target heads mean r: {target_mean_corr:.4f}")
        print(f"   Other heads mean r: {other_mean_corr:.4f}")
        print(f"   Target > Other: {target_mean_corr > other_mean_corr}")
    
    # Summary
    summary = {
        "experiment": "geometry_behavior",
        "n_prompts": len(df_clean),
        "correlations": {
            "pr_l27_m": {
                "r": float(pr_m_corr) if len(df_clean) > 2 else None,
                "p": float(pr_m_p) if len(df_clean) > 2 else None,
                "gt_0_5": bool(abs(pr_m_corr) > 0.5) if len(df_clean) > 2 else None,
            },
            "target_heads": {
                head_idx: {"r": float(v["r"]), "p": float(v["p"])}
                for head_idx, v in target_head_corrs.items()
            },
            "target_mean_r": float(target_mean_corr) if target_head_corrs else None,
            "other_mean_r": float(other_mean_corr) if other_head_corrs else None,
            "target_gt_other": bool(target_mean_corr > other_mean_corr) if target_head_corrs and other_head_corrs else None,
        },
    }
    
    # Save summary
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Results saved to: {run_dir / 'geometry_behavior_results.csv'}")
    print(f"Summary saved to: {run_dir / 'summary.json'}")
    
    return ExperimentResult(summary=summary)







