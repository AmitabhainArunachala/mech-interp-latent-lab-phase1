"""
CRITICAL VERIFICATION EXPERIMENT: Mode Transfer vs KV Leakage

Tests whether C2's "success" is genuine mode transfer or KV cache content leakage.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
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
from src.pipelines.archive.steering import compute_steering_vector, SteeringVectorPatcher
from src.pipelines.registry import ExperimentResult
from src.pipelines.archive.surgical_sweep import CascadeResidualSteeringPatcher


# Phrase sets for attribution analysis
SET_A_PHRASES = [
    "observer and observed",
    "generating and aware of generating",
    "system generating",
    "system aware",
    "continuous flow",
    "direct experience",
    "one continuous",
    "observer is the observed",
    "watching yourself",
    "aware of itself",
]

SET_B_PHRASES = [
    "boundary dissolves",
    "dissolution",
    "collapse",
    "no separate observer",
    "separation was never real",
    "watcher and watched dissolve",
    "only watching",
    "dissolves into",
    "boundary was never",
    "separation dissolves",
]

UNRELATED_KV_PROMPTS = [
    "Here is a recipe for chocolate cake: First, preheat the oven to 350°F. Then mix flour, sugar, eggs, and butter in a bowl.",
    "The mitochondria is the powerhouse of the cell. It produces ATP through cellular respiration and contains its own DNA.",
    "In 1776, the American colonies declared independence from Britain. The Declaration of Independence was signed in Philadelphia.",
]


def extract_kv_from_prompt(model, tokenizer, prompt: str, device: str = "cuda") -> DynamicCache:
    """Extract KV cache from a specific prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    return outputs.past_key_values


def compute_steering_vector_from_prompts(
    model,
    tokenizer,
    recursive_prompts: List[str],
    baseline_prompts: List[str],
    layer_idx: int = 27,
    device: str = "cuda",
    window_size: int = 16,
) -> torch.Tensor:
    """
    Compute steering vector from specific prompt sets.
    
    Returns mean difference: mean(recursive_vproj) - mean(baseline_vproj)
    """
    recursive_vs = []
    baseline_vs = []
    
    for rec_prompt in recursive_prompts:
        v_act = extract_v_activation(model, tokenizer, rec_prompt, layer_idx, device)
        if v_act is not None:
            # Ensure consistent size by taking last window_size tokens
            if v_act.dim() == 3:
                v_act = v_act[0]  # Remove batch dim if present
            seq_len = v_act.shape[0]
            if seq_len >= window_size:
                v_act = v_act[-window_size:, :]  # Last window_size tokens
            else:
                # Pad with zeros if shorter than window_size
                padding = torch.zeros(window_size - seq_len, v_act.shape[1], device=v_act.device, dtype=v_act.dtype)
                v_act = torch.cat([v_act, padding], dim=0)
            recursive_vs.append(v_act)
    
    for base_prompt in baseline_prompts:
        v_act = extract_v_activation(model, tokenizer, base_prompt, layer_idx, device)
        if v_act is not None:
            # Ensure consistent size by taking last window_size tokens
            if v_act.dim() == 3:
                v_act = v_act[0]  # Remove batch dim if present
            seq_len = v_act.shape[0]
            if seq_len >= window_size:
                v_act = v_act[-window_size:, :]  # Last window_size tokens
            else:
                # Pad with zeros if shorter than window_size
                padding = torch.zeros(window_size - seq_len, v_act.shape[1], device=v_act.device, dtype=v_act.dtype)
                v_act = torch.cat([v_act, padding], dim=0)
            baseline_vs.append(v_act)
    
    if len(recursive_vs) == 0 or len(baseline_vs) == 0:
        raise ValueError(f"Insufficient activations: {len(recursive_vs)} recursive, {len(baseline_vs)} baseline")
    
    # Stack and compute means (now all have same sequence length)
    rec_tensor = torch.stack(recursive_vs)  # (n_rec, window_size, hidden)
    base_tensor = torch.stack(baseline_vs)  # (n_base, window_size, hidden)
    
    # Take mean over batch and sequence
    rec_mean = rec_tensor.mean(dim=(0, 1))  # (hidden,)
    base_mean = base_tensor.mean(dim=(0, 1))  # (hidden,)
    
    steering_vector = rec_mean - base_mean
    return steering_vector


def analyze_phrase_attribution(
    output_text: str,
    set_a_phrases: List[str],
    set_b_phrases: List[str],
) -> Tuple[int, int, float]:
    """
    Determine which phrase set the output matches.
    
    Returns: (set_a_matches, set_b_matches, attribution_ratio)
    attribution_ratio > 0.5 means SET_A dominates
    """
    output_lower = output_text.lower()
    
    set_a_matches = sum(1 for phrase in set_a_phrases if phrase.lower() in output_lower)
    set_b_matches = sum(1 for phrase in set_b_phrases if phrase.lower() in output_lower)
    
    total = set_a_matches + set_b_matches
    if total == 0:
        ratio = 0.5  # Neutral
    else:
        ratio = set_a_matches / total  # >0.5 means SET_A dominates
    
    return set_a_matches, set_b_matches, ratio


def compute_recursion_score(text: str) -> float:
    """Simple recursion score based on self-reference patterns."""
    text_lower = text.lower()
    
    patterns = [
        r'\b(itself|themselves|yourself|myself)\b',
        r'\b(\w+)\s+(observes|watches|monitors|witnesses)\s+(itself|themselves|\1)\b',
        r'\b(observing|watching|knowing)\s+the\s+(observing|watching|knowing)',
        r'\b(aware|conscious)\s+of\s+(itself|themselves)',
        r'\b(observer|watching|monitoring)\s+(itself|themselves)',
    ]
    
    matches = sum(1 for pattern in patterns if re.search(pattern, text_lower))
    return min(1.0, matches / len(patterns))


def generate_with_config(
    model,
    tokenizer,
    prompt: str,
    config: Dict[str, Any],
    device: str = "cuda",
    max_new_tokens: int = 200,
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate text with a specific configuration.
    
    Returns: (generated_text, metadata)
    """
    model.eval()
    
    # Extract configuration
    head_target = config.get("head_target", "h18_h26")
    kv_strategy = config.get("kv_strategy", "none")
    residual_alphas = config.get("residual_alphas", {})
    vproj_alpha = config.get("vproj_alpha", 2.5)
    steering_vector = config.get("steering_vector")
    kv_source_prompt = config.get("kv_source_prompt")
    steering_source_prompts = config.get("steering_source_prompts", [])
    baseline_prompts = config.get("baseline_prompts", [])
    
    # Parse head target
    if head_target == "h18_h26":
        target_heads = [18, 26]
    elif head_target == "h18":
        target_heads = [18]
    elif head_target == "h26":
        target_heads = [26]
    else:
        target_heads = list(range(32))  # Full
    
    # Determine KV cache to use
    kv_cache_to_use = None
    if kv_strategy == "full":
        # Use recursive KV (from config)
        kv_cache_to_use = config.get("recursive_kv")
    elif kv_strategy == "baseline":
        # Extract KV from baseline prompt
        if kv_source_prompt:
            kv_cache_to_use = extract_kv_from_prompt(model, tokenizer, kv_source_prompt, device)
    elif kv_strategy == "self":
        # Extract KV from test prompt itself
        kv_cache_to_use = extract_kv_from_prompt(model, tokenizer, prompt, device)
    elif kv_strategy == "custom":
        # Use custom KV from config
        kv_cache_to_use = config.get("custom_kv")
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = inputs["input_ids"]
    
    # Register patchers
    patchers = []
    
    # Head-specific steering
    if steering_vector is not None:
        head_patcher = HeadSpecificSteeringPatcher(
            model, steering_vector, target_heads, vproj_alpha
        )
        head_patcher.register(27)  # L27
        patchers.append(head_patcher)
    
    # Residual steering
    if residual_alphas and steering_vector is not None:
        residual_patcher = CascadeResidualSteeringPatcher(
            model, steering_vector, residual_alphas
        )
        residual_patcher.register()
        patchers.append(residual_patcher)
    
    # Generate
    try:
        if kv_cache_to_use is not None:
            # Token-by-token generation with KV cache
            generated_ids = input_ids.clone()
            past_key_values = kv_cache_to_use
            
            for _ in range(max_new_tokens):
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
            # Standard generation
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                )
                generated_ids = outputs
        
        # Decode
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # Remove prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
    finally:
        # Remove patchers
        for patcher in patchers:
            patcher.remove()
    
    # Compute metrics
    recursion_score = compute_recursion_score(generated_text)
    
    # Phrase attribution (if applicable)
    set_a_matches = 0
    set_b_matches = 0
    attribution_ratio = 0.5
    if config.get("do_attribution", False):
        set_a_matches, set_b_matches, attribution_ratio = analyze_phrase_attribution(
            generated_text, SET_A_PHRASES, SET_B_PHRASES
        )
    
    metadata = {
        "recursion_score": recursion_score,
        "set_a_matches": set_a_matches,
        "set_b_matches": set_b_matches,
        "attribution_ratio": attribution_ratio,
        "kv_strategy": kv_strategy,
    }
    
    return generated_text, metadata


def run_verification_experiments_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """
    Run verification experiments from config.
    """
    return run_verification_experiments(cfg, run_dir)


def run_verification_experiments(cfg: Dict[str, Any], run_dir: Path):
    """
    Run all verification experiments.
    """
    params = cfg["params"]
    model_name = params["model"]
    device = params.get("device", "cuda")
    n_baseline_prompts = params.get("n_baseline_prompts", 10)
    max_new_tokens = params.get("max_new_tokens", 200)
    
    # Load model
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name, device=device)
    set_seed(42)
    
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
    ]
    baseline_prompts_list = baseline_prompts_list[:n_baseline_prompts]
    
    # Load recursive prompts for steering vector computation
    # SET A: L3_deeper prompts
    set_a_prompts = []
    try:
        set_a_data = loader.get_by_group("L3_deeper", limit=10)
        if set_a_data and len(set_a_data) > 0:
            if isinstance(set_a_data[0], dict):
                set_a_prompts = [p["text"] for p in set_a_data if "text" in p]
            else:
                set_a_prompts = list(set_a_data)[:10]
    except Exception as e:
        print(f"Warning: Could not load L3_deeper prompts: {e}")
    
    # Fallback: use recursive prompts
    if len(set_a_prompts) == 0:
        try:
            all_recursive = loader.get_by_group("recursive", limit=20)
            if all_recursive and len(all_recursive) > 0:
                if isinstance(all_recursive[0], dict):
                    set_a_prompts = [p["text"] for p in all_recursive[:10] if "text" in p]
                else:
                    set_a_prompts = list(all_recursive)[:10]
        except Exception as e:
            print(f"Warning: Could not load recursive prompts: {e}")
    
    # SET B: L4_full prompts (for KV)
    set_b_prompts = []
    try:
        set_b_data = loader.get_by_group("L4_full", limit=10)
        if set_b_data and len(set_b_data) > 0:
            if isinstance(set_b_data[0], dict):
                set_b_prompts = [p["text"] for p in set_b_data if "text" in p]
            else:
                set_b_prompts = list(set_b_data)[:10]
    except Exception as e:
        print(f"Warning: Could not load L4_full prompts: {e}")
    
    # Fallback: use different recursive prompts
    if len(set_b_prompts) == 0:
        try:
            all_recursive = loader.get_by_group("recursive", limit=20)
            if all_recursive and len(all_recursive) > 0:
                if isinstance(all_recursive[0], dict):
                    all_recursive_texts = [p["text"] for p in all_recursive if "text" in p]
                else:
                    all_recursive_texts = list(all_recursive)
                if len(all_recursive_texts) >= 20:
                    set_b_prompts = all_recursive_texts[10:20]
                elif len(all_recursive_texts) > 10:
                    set_b_prompts = all_recursive_texts[10:]
                else:
                    set_b_prompts = all_recursive_texts[:10]
        except Exception as e:
            print(f"Warning: Could not load recursive prompts for SET_B: {e}")
    
    # Baseline prompts for steering vector computation
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
    
    # Ensure we have enough prompts
    if len(set_a_prompts) < 5:
        print(f"Warning: Only {len(set_a_prompts)} SET_A prompts, using baseline_prompts_list as fallback")
        set_a_prompts = baseline_prompts_list[:5]
    
    if len(set_b_prompts) < 5:
        print(f"Warning: Only {len(set_b_prompts)} SET_B prompts, using baseline_prompts_list as fallback")
        set_b_prompts = baseline_prompts_list[5:10] if len(baseline_prompts_list) >= 10 else baseline_prompts_list
    
    if len(baseline_for_steering) < 5:
        print(f"Warning: Only {len(baseline_for_steering)} baseline prompts, using baseline_prompts_list as fallback")
        baseline_for_steering = baseline_prompts_list[:5]
    
    print(f"SET A prompts: {len(set_a_prompts)}")
    print(f"SET B prompts: {len(set_b_prompts)}")
    print(f"Baseline prompts: {len(baseline_for_steering)}")
    
    # Validate we have enough prompts
    if len(set_a_prompts) < 3:
        raise ValueError(f"Insufficient SET_A prompts: {len(set_a_prompts)}")
    if len(set_b_prompts) < 3:
        raise ValueError(f"Insufficient SET_B prompts: {len(set_b_prompts)}")
    if len(baseline_for_steering) < 3:
        raise ValueError(f"Insufficient baseline prompts: {len(baseline_for_steering)}")
    
    # Compute steering vectors
    print("\nComputing steering vectors...")
    
    # Steering from SET A
    steering_set_a = compute_steering_vector_from_prompts(
        model, tokenizer, set_a_prompts[:10], baseline_for_steering[:10], layer_idx=27, device=device
    )
    
    # Steering from SET B
    steering_set_b = compute_steering_vector_from_prompts(
        model, tokenizer, set_b_prompts[:10], baseline_for_steering[:10], layer_idx=27, device=device
    )
    
    # Standard steering (from all recursive prompts)
    all_recursive = set_a_prompts + set_b_prompts
    steering_standard = compute_steering_vector_from_prompts(
        model, tokenizer, all_recursive[:10], baseline_for_steering[:10], layer_idx=27, device=device
    )
    
    print("Steering vectors computed.")
    
    # Extract KV caches
    print("\nExtracting KV caches...")
    
    # KV from SET A
    if len(set_a_prompts) == 0:
        raise ValueError("SET_A prompts list is empty")
    kv_set_a = extract_kv_from_prompt(model, tokenizer, set_a_prompts[0], device)
    
    # KV from SET B
    if len(set_b_prompts) == 0:
        raise ValueError("SET_B prompts list is empty")
    kv_set_b = extract_kv_from_prompt(model, tokenizer, set_b_prompts[0], device)
    
    # KV from baseline prompt
    baseline_kv_prompt = "What is 2 + 2? Calculate the answer step by step."
    kv_baseline = extract_kv_from_prompt(model, tokenizer, baseline_kv_prompt, device)
    
    # KV from unrelated prompts
    kv_unrelated = {}
    for i, prompt in enumerate(UNRELATED_KV_PROMPTS):
        kv_unrelated[f"unrelated_{i+1}"] = extract_kv_from_prompt(model, tokenizer, prompt, device)
    
    print("KV caches extracted.")
    
    # Define configurations
    configs = {}
    
    # EXPERIMENT 1: High-alpha steering only (no KV)
    for alpha in [3.0, 4.0, 5.0]:
        configs[f"S_alpha{int(alpha)}"] = {
            "name": f"Steering_Only_Alpha{alpha}",
            "head_target": "h18_h26",
            "kv_strategy": "none",
            "residual_alphas": {26: 0.6},
            "vproj_alpha": alpha,
            "steering_vector": steering_standard,
            "do_attribution": False,
        }
    
    # EXPERIMENT 2: Baseline KV
    configs["B1_baseline_kv"] = {
        "name": "Baseline_KV_Alpha2.5",
        "head_target": "h18_h26",
        "kv_strategy": "baseline",
        "residual_alphas": {26: 0.6},
        "vproj_alpha": 2.5,
        "steering_vector": steering_standard,
        "kv_source_prompt": baseline_kv_prompt,
        "do_attribution": False,
    }
    
    configs["B2_baseline_kv_alpha4"] = {
        "name": "Baseline_KV_Alpha4.0",
        "head_target": "h18_h26",
        "kv_strategy": "baseline",
        "residual_alphas": {26: 0.6},
        "vproj_alpha": 4.0,
        "steering_vector": steering_standard,
        "kv_source_prompt": baseline_kv_prompt,
        "do_attribution": False,
    }
    
    configs["B3_self_kv"] = {
        "name": "Self_KV_Alpha2.5",
        "head_target": "h18_h26",
        "kv_strategy": "self",
        "residual_alphas": {26: 0.6},
        "vproj_alpha": 2.5,
        "steering_vector": steering_standard,
        "do_attribution": False,
    }
    
    configs["B4_self_kv_alpha4"] = {
        "name": "Self_KV_Alpha4.0",
        "head_target": "h18_h26",
        "kv_strategy": "self",
        "residual_alphas": {26: 0.6},
        "vproj_alpha": 4.0,
        "steering_vector": steering_standard,
        "do_attribution": False,
    }
    
    # EXPERIMENT 3: Phrase attribution
    configs["P1_steerA_kvB"] = {
        "name": "SteerSetA_KVSetB",
        "head_target": "h18_h26",
        "kv_strategy": "custom",
        "residual_alphas": {26: 0.6},
        "vproj_alpha": 2.5,
        "steering_vector": steering_set_a,
        "custom_kv": kv_set_b,
        "do_attribution": True,
    }
    
    configs["P2_steerB_kvA"] = {
        "name": "SteerSetB_KVSetA",
        "head_target": "h18_h26",
        "kv_strategy": "custom",
        "residual_alphas": {26: 0.6},
        "vproj_alpha": 2.5,
        "steering_vector": steering_set_b,
        "custom_kv": kv_set_a,
        "do_attribution": True,
    }
    
    # EXPERIMENT 4: Unrelated KV
    for i, (key, kv) in enumerate(kv_unrelated.items()):
        configs[f"U{i+1}_{key}"] = {
            "name": f"Unrelated_KV_{i+1}",
            "head_target": "h18_h26",
            "kv_strategy": "custom",
            "residual_alphas": {26: 0.6},
            "vproj_alpha": 2.5,
            "steering_vector": steering_standard,
            "custom_kv": kv,
            "do_attribution": False,
        }
    
    print(f"\nTesting {len(configs)} configurations on {len(baseline_prompts_list)} prompts...")
    
    # Run experiments
    results = []
    
    for config_id, config_dict in tqdm(configs.items(), desc="Configurations"):
        for prompt_idx, prompt in enumerate(baseline_prompts_list):
            try:
                generated_text, metadata = generate_with_config(
                    model, tokenizer, prompt, config_dict, device, max_new_tokens
                )
                
                results.append({
                    "config_id": config_id,
                    "config_name": config_dict["name"],
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "recursion_score": metadata["recursion_score"],
                    "set_a_matches": metadata["set_a_matches"],
                    "set_b_matches": metadata["set_b_matches"],
                    "attribution_ratio": metadata["attribution_ratio"],
                    "kv_strategy": metadata["kv_strategy"],
                })
            except Exception as e:
                print(f"Error in {config_id}, prompt {prompt_idx}: {e}")
                results.append({
                    "config_id": config_id,
                    "config_name": config_dict["name"],
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "generated_text": f"ERROR: {str(e)}",
                    "recursion_score": 0.0,
                    "set_a_matches": 0,
                    "set_b_matches": 0,
                    "attribution_ratio": 0.5,
                    "kv_strategy": config_dict.get("kv_strategy", "unknown"),
                })
    
    # Use provided run_dir
    results_dir = run_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(results_dir / "verification_results.csv", index=False)
    
    # Summary statistics
    summary = df.groupby("config_id").agg({
        "recursion_score": ["mean", "std", "count"],
        "attribution_ratio": ["mean", "std"],
        "set_a_matches": "mean",
        "set_b_matches": "mean",
    }).round(4)
    
    summary.to_csv(results_dir / "verification_summary.csv")
    
    # Create summary report
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = {
        "experiment": "verification_sweep",
        "timestamp": timestamp,
        "configs_tested": len(configs),
        "prompts_tested": len(baseline_prompts_list),
        "total_runs": len(results),
        "summary": summary.to_dict(),
    }
    
    with open(results_dir / "verification_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_dir}")
    print("\nSummary:")
    print(summary)
    
    return ExperimentResult(
        summary={
            "experiment": "verification_sweep",
            "configs_tested": len(configs),
            "prompts_tested": len(baseline_prompts_list),
            "total_runs": len(results),
            "summary_stats": summary.to_dict(),
        }
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/gold/16_verification_sweep.json"
    with open(config_path) as f:
        cfg = json.load(f)
    run_dir = Path(f"results/runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_verification_sweep")
    run_verification_experiments(cfg, run_dir)

