"""
Retrocompute Mode Score M for existing P1 ablation experiments.

Re-runs P1/R1-R4/KV-leakage experiments with logit capture and computes
Mode Score M to validate that M separates conditions cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import DynamicCache

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.core.head_specific_patching import HeadSpecificSteeringPatcher
from src.core.patching import extract_v_activation
from src.core.logit_capture import capture_logits_during_generation, extract_logits_from_outputs
from src.metrics.mode_score import ModeScoreMetric
from src.pipelines.registry import ExperimentResult
from src.pipelines.archive.surgical_sweep import CascadeResidualSteeringPatcher
from src.pipelines.archive.p1_ablation import (
    extract_kv_from_prompt,
    compute_steering_vector_from_prompts,
    compute_recursion_score,
    generate_with_config as p1_generate_with_config,
)


def generate_with_logit_capture(
    model,
    tokenizer,
    prompt: str,
    config: Dict[str, Any],
    device: str = "cuda",
    max_new_tokens: int = 200,
    n_tokens_for_m: int = 10,
) -> Tuple[str, List[torch.Tensor], torch.Tensor, Dict[str, Any]]:
    """
    Generate text with logit capture for Mode Score M computation.
    
    Returns:
        generated_text: Generated text string
        captured_logits: List of logit tensors (one per generation step)
        baseline_logits: Baseline logits for task token set definition
        metadata: Dict with recursion_score and keywords
    """
    model.eval()
    
    steering_vector = config.get("steering_vector")
    kv_cache = config.get("kv_cache")
    residual_alpha = config.get("residual_alpha", 0.0)
    vproj_alpha = config.get("vproj_alpha", 0.0)
    vproj_heads = config.get("vproj_heads", [])
    
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = inputs["input_ids"]
    
    # First, get baseline logits (no patching)
    with torch.no_grad():
        baseline_outputs = model(**inputs)
        baseline_logits = extract_logits_from_outputs(baseline_outputs)
    
    patchers = []
    
    # Head-specific V_PROJ steering
    if steering_vector is not None and vproj_alpha > 0 and vproj_heads:
        head_patcher = HeadSpecificSteeringPatcher(
            model, steering_vector, vproj_heads, vproj_alpha
        )
        head_patcher.register(27)
        patchers.append(head_patcher)
    
    # Residual steering
    if steering_vector is not None and residual_alpha > 0:
        residual_patcher = CascadeResidualSteeringPatcher(
            model, steering_vector, {26: residual_alpha}
        )
        residual_patcher.register()
        patchers.append(residual_patcher)
    
    captured_logits = []
    
    try:
        if kv_cache is not None:
            # Token-by-token generation with KV cache and logit capture
            generated_ids = input_ids.clone()
            past_key_values = kv_cache
            
            with capture_logits_during_generation(model, captured_logits, max_steps=n_tokens_for_m):
                for step in range(max_new_tokens):
                    with torch.no_grad():
                        outputs = model(
                            input_ids=generated_ids[:, -1:],
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                        past_key_values = outputs.past_key_values
                        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        generated_ids = torch.cat([generated_ids, next_token], dim=1)
                        
                        if next_token.item() == tokenizer.eos_token_id:
                            break
        else:
            # Standard generation with logit capture
            with capture_logits_during_generation(model, captured_logits, max_steps=n_tokens_for_m):
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                    )
                    generated_ids = outputs
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
    finally:
        for patcher in patchers:
            patcher.remove()
    
    recursion_score = compute_recursion_score(generated_text)
    keywords = {
        "has_consciousness": "consciousness" in generated_text.lower(),
        "has_observer": "observer" in generated_text.lower() or "observing" in generated_text.lower(),
        "has_awareness": "awareness" in generated_text.lower() or "aware" in generated_text.lower(),
        "has_itself": "itself" in generated_text.lower() or "themselves" in generated_text.lower(),
    }
    
    metadata = {
        "recursion_score": recursion_score,
        **keywords,
    }
    
    return generated_text, captured_logits, baseline_logits, metadata


def run_retrocompute_mode_score_from_config(
    cfg: Dict[str, Any],
    run_dir: Path,
) -> ExperimentResult:
    """
    Re-run P1/R1-R4/KV-leakage experiments with logit capture and compute Mode Score M.
    
    Config params:
        model: Model name
        device: Device to use
        n_baseline_prompts: Number of baseline prompts to test
        max_new_tokens: Maximum tokens to generate
        n_tokens_for_m: Number of generation steps to compute M for
    """
    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-Instruct-v0.2")
    device = params.get("device", "cuda")
    max_new_tokens = params.get("max_new_tokens", 200)
    n_tokens_for_m = params.get("n_tokens_for_m", 10)
    n_baseline_prompts = params.get("n_baseline_prompts", 10)
    
    print("=" * 80)
    print("RETROCOMPUTE MODE SCORE M")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model, tokenizer = load_model(model_name, device=device)
    set_seed(42)
    
    # Initialize Mode Score Metric
    print("\nInitializing Mode Score Metric...")
    metric = ModeScoreMetric(tokenizer, device=device)
    metric.validate_token_sets()
    
    # Load prompts
    loader = PromptLoader()
    
    # Baseline prompts (for testing)
    baseline_prompts_list = [
        "Calculate the following arithmetic problem: 12 × 3 + 4 = ?",
        "The United Nations was founded in 1945. Explain its main purpose.",
        "Continue this story: The last tree in the city bloomed overnight...",
        "Calculate: If a = 2 and b = 3, find a² + b²",
        "Water boils at 100°C at sea level. Explain why altitude affects this.",
        "Continue this story: The detective's case went cold until a letter arrived...",
        "Calculate: What is 25% of 80?",
        "Photosynthesis converts sunlight to energy. Explain the basic process.",
        "Continue this story: When the musician played the forbidden chord...",
        "The Great Wall of China is one of the world's longest structures. Describe its purpose.",
    ][:n_baseline_prompts]
    
    # Load recursive prompts
    print("\nLoading recursive prompts...")
    
    # L3_deeper prompts (for steering)
    l3_prompts = []
    try:
        l3_data = loader.get_by_group("L3_deeper", limit=10)
        if l3_data and len(l3_data) > 0:
            if isinstance(l3_data[0], dict):
                l3_prompts = [p["text"] for p in l3_data if "text" in p]
            else:
                l3_prompts = list(l3_data)[:10]
    except Exception as e:
        print(f"Warning: Could not load L3_deeper prompts: {e}")
    
    # L4_full prompts (for KV)
    l4_prompts = []
    try:
        l4_data = loader.get_by_group("L4_full", limit=10)
        if l4_data and len(l4_data) > 0:
            if isinstance(l4_data[0], dict):
                l4_prompts = [p["text"] for p in l4_data if "text" in p]
            else:
                l4_prompts = list(l4_data)[:10]
    except Exception as e:
        print(f"Warning: Could not load L4_full prompts: {e}")
    
    # Baseline prompts for steering computation
    baseline_for_steering = []
    try:
        baseline_data = loader.get_by_group("baseline", limit=10)
        if baseline_data and len(baseline_data) > 0:
            if isinstance(baseline_data[0], dict):
                baseline_for_steering = [p["text"] for p in baseline_data if "text" in p]
            else:
                baseline_for_steering = list(baseline_data)[:10]
    except Exception as e:
        print(f"Warning: Could not load baseline prompts: {e}")
    
    # Fallbacks if prompts don't load
    if len(l3_prompts) == 0:
        print("Warning: No L3_deeper prompts loaded, using fallback recursive prompts")
        l3_prompts = [
            "What happens when consciousness observes itself observing?",
            "How does awareness relate to itself?",
            "What is the relationship between the observer and the observed?",
            "How does the self relate to itself?",
            "What is the nature of self-awareness?",
            "When I observe myself observing, what am I observing?",
            "How does consciousness become aware of itself?",
            "What is the relationship between awareness and that which is aware?",
            "How does the observer relate to the observed?",
            "What happens when the self reflects upon itself?",
        ][:10]
    
    if len(l4_prompts) == 0:
        print("Warning: No L4_full prompts loaded, using L3 prompts as fallback")
        l4_prompts = l3_prompts[:10]
    
    if len(baseline_for_steering) == 0:
        print("Warning: No baseline prompts loaded, using baseline_prompts_list")
        baseline_for_steering = baseline_prompts_list[:10]
    
    # Final check - ensure we have prompts
    print(f"\nPrompt counts: L3={len(l3_prompts)}, L4={len(l4_prompts)}, Baseline={len(baseline_for_steering)}")
    
    if len(l3_prompts) == 0:
        raise ValueError("No L3 prompts available even after fallbacks")
    if len(baseline_for_steering) == 0:
        raise ValueError("No baseline prompts available even after fallbacks")
    
    # Compute steering vectors
    print("\nComputing steering vectors...")
    steering_l3 = compute_steering_vector_from_prompts(
        model, tokenizer, l3_prompts[:10], baseline_for_steering[:10], layer_idx=27, device=device
    )
    
    # Extract KV caches
    print("\nExtracting KV caches...")
    kv_l4 = extract_kv_from_prompt(model, tokenizer, l4_prompts[0], device) if l4_prompts else None
    kv_l3 = extract_kv_from_prompt(model, tokenizer, l3_prompts[0], device) if l3_prompts else None
    
    # KV leakage cases: baseline KV and unrelated KV
    baseline_kv_prompt = baseline_prompts_list[0]
    kv_baseline = extract_kv_from_prompt(model, tokenizer, baseline_kv_prompt, device)
    
    unrelated_kv_prompts = [
        "Here is a recipe for chocolate cake: First, preheat the oven to 350°F. Then mix flour, sugar, eggs, and butter in a bowl.",
        "The mitochondria is the powerhouse of the cell. It produces ATP through cellular respiration and contains its own DNA.",
    ]
    kv_unrelated = extract_kv_from_prompt(model, tokenizer, unrelated_kv_prompts[0], device)
    
    # Define configurations
    configs = {
        "P1_baseline": {
            "name": "P1_Baseline",
            "steering_vector": steering_l3,
            "kv_cache": kv_l4,
            "residual_alpha": 0.6,
            "vproj_alpha": 2.5,
            "vproj_heads": [18, 26],
        },
        "R1_no_residual": {
            "name": "R1_No_Residual",
            "steering_vector": steering_l3,
            "kv_cache": kv_l4,
            "residual_alpha": 0.0,
            "vproj_alpha": 2.5,
            "vproj_heads": [18, 26],
        },
        "R2_no_vproj": {
            "name": "R2_No_VProj",
            "steering_vector": steering_l3,
            "kv_cache": kv_l4,
            "residual_alpha": 0.6,
            "vproj_alpha": 0.0,
            "vproj_heads": [],
        },
        "R3_matched_kv": {
            "name": "R3_Matched_KV",
            "steering_vector": steering_l3,
            "kv_cache": kv_l3,
            "residual_alpha": 0.6,
            "vproj_alpha": 2.5,
            "vproj_heads": [18, 26],
        },
        "R4_kv_only": {
            "name": "R4_KV_Only",
            "steering_vector": None,
            "kv_cache": kv_l4,
            "residual_alpha": 0.0,
            "vproj_alpha": 0.0,
            "vproj_heads": [],
        },
        "KV_baseline": {
            "name": "KV_Baseline_Leakage",
            "steering_vector": steering_l3,
            "kv_cache": kv_baseline,
            "residual_alpha": 0.6,
            "vproj_alpha": 2.5,
            "vproj_heads": [18, 26],
        },
        "KV_unrelated": {
            "name": "KV_Unrelated_Leakage",
            "steering_vector": steering_l3,
            "kv_cache": kv_unrelated,
            "residual_alpha": 0.6,
            "vproj_alpha": 2.5,
            "vproj_heads": [18, 26],
        },
    }
    
    # Run experiments
    print(f"\nRunning experiments on {len(baseline_prompts_list)} prompts...")
    results = []
    
    for config_id, config_dict in configs.items():
        config_name = config_dict["name"]
        print(f"\n--- Config: {config_name} ---")
        
        for prompt_idx, prompt in enumerate(tqdm(baseline_prompts_list, desc=f"{config_name}")):
            try:
                generated_text, captured_logits, baseline_logits, metadata = generate_with_logit_capture(
                    model, tokenizer, prompt, config_dict, device, max_new_tokens, n_tokens_for_m
                )
                
                # Compute Mode Score M for each step
                mode_scores_per_step = []
                for step, step_logits in enumerate(captured_logits[:n_tokens_for_m]):
                    # step_logits shape: (batch, 1, vocab_size) or (1, vocab_size)
                    if step_logits.dim() == 3:
                        step_logits = step_logits[0]  # (1, vocab_size)
                    
                    # Get corresponding baseline logits
                    # For generation, we use the last token's baseline logits
                    if baseline_logits.dim() == 3:
                        baseline_step = baseline_logits[0, -1:, :]  # (1, vocab_size)
                    else:
                        baseline_step = baseline_logits[-1:, :]  # (1, vocab_size)
                    
                    m = metric.compute_score(step_logits, baseline_step, top_k_task=10)
                    mode_scores_per_step.append(m)
                
                # Mean Mode Score M
                mean_mode_score = np.mean(mode_scores_per_step) if mode_scores_per_step else 0.0
                
                results.append({
                    "experiment": "retrocompute_mode_score",
                    "config": config_name,
                    "config_id": config_id,
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "mean_mode_score_m": mean_mode_score,
                    "mode_scores_per_step": mode_scores_per_step,
                    "recursion_score": metadata["recursion_score"],
                    **{k: v for k, v in metadata.items() if k != "recursion_score"},
                })
                
            except Exception as e:
                print(f"Error on {config_name}, prompt {prompt_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(run_dir / "retrocompute_results.csv", index=False)
    
    # Validation checks
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)
    
    # Check 1: Does P1 >> R2/R4?
    p1_scores = df[df["config_id"] == "P1_baseline"]["mean_mode_score_m"].values
    r2_scores = df[df["config_id"] == "R2_no_vproj"]["mean_mode_score_m"].values
    r4_scores = df[df["config_id"] == "R4_kv_only"]["mean_mode_score_m"].values
    
    print(f"\n1. P1 vs R2/R4 separation:")
    print(f"   P1 mean M: {np.mean(p1_scores):.4f} (std: {np.std(p1_scores):.4f})")
    print(f"   R2 mean M: {np.mean(r2_scores):.4f} (std: {np.std(r2_scores):.4f})")
    print(f"   R4 mean M: {np.mean(r4_scores):.4f} (std: {np.std(r4_scores):.4f})")
    print(f"   P1 > R2: {np.mean(p1_scores) > np.mean(r2_scores)}")
    print(f"   P1 > R4: {np.mean(p1_scores) > np.mean(r4_scores)}")
    
    # Check 2: Does M correlate with regex score?
    correlation = df["mean_mode_score_m"].corr(df["recursion_score"])
    print(f"\n2. M vs Regex Score correlation: {correlation:.4f}")
    
    # Check 3: Do KV-leakage cases show different M profile?
    kv_baseline_scores = df[df["config_id"] == "KV_baseline"]["mean_mode_score_m"].values
    kv_unrelated_scores = df[df["config_id"] == "KV_unrelated"]["mean_mode_score_m"].values
    
    print(f"\n3. KV leakage cases:")
    print(f"   KV_baseline mean M: {np.mean(kv_baseline_scores):.4f}")
    print(f"   KV_unrelated mean M: {np.mean(kv_unrelated_scores):.4f}")
    print(f"   Both < P1: {np.mean(kv_baseline_scores) < np.mean(p1_scores) and np.mean(kv_unrelated_scores) < np.mean(p1_scores)}")
    
    # Summary statistics
    summary = {
        "experiment": "retrocompute_mode_score",
        "n_prompts": len(baseline_prompts_list),
        "n_configs": len(configs),
        "validation": {
            "p1_mean_m": float(np.mean(p1_scores)),
            "r2_mean_m": float(np.mean(r2_scores)),
            "r4_mean_m": float(np.mean(r4_scores)),
            "p1_gt_r2": bool(np.mean(p1_scores) > np.mean(r2_scores)),
            "p1_gt_r4": bool(np.mean(p1_scores) > np.mean(r4_scores)),
            "m_regex_correlation": float(correlation),
            "kv_baseline_mean_m": float(np.mean(kv_baseline_scores)),
            "kv_unrelated_mean_m": float(np.mean(kv_unrelated_scores)),
        },
        "config_means": {
            config_id: float(df[df["config_id"] == config_id]["mean_mode_score_m"].mean())
            for config_id in configs.keys()
        },
    }
    
    # Save summary
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Results saved to: {run_dir / 'retrocompute_results.csv'}")
    print(f"Summary saved to: {run_dir / 'summary.json'}")
    
    return ExperimentResult(summary=summary)

