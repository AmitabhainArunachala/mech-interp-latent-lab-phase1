"""
P1 Ablation Experiment: What's Doing The Work?

Single-variable ablation on P1 configuration to determine which components
are necessary for the recursive output.
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
from src.pipelines.registry import ExperimentResult
from src.pipelines.archive.surgical_sweep import CascadeResidualSteeringPatcher


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
    """Compute steering vector from specific prompt sets."""
    recursive_vs = []
    baseline_vs = []
    
    for rec_prompt in recursive_prompts:
        v_act = extract_v_activation(model, tokenizer, rec_prompt, layer_idx, device)
        if v_act is not None:
            if v_act.dim() == 3:
                v_act = v_act[0]
            seq_len = v_act.shape[0]
            if seq_len >= window_size:
                v_act = v_act[-window_size:, :]
            else:
                padding = torch.zeros(window_size - seq_len, v_act.shape[1], device=v_act.device, dtype=v_act.dtype)
                v_act = torch.cat([v_act, padding], dim=0)
            recursive_vs.append(v_act)
    
    for base_prompt in baseline_prompts:
        v_act = extract_v_activation(model, tokenizer, base_prompt, layer_idx, device)
        if v_act is not None:
            if v_act.dim() == 3:
                v_act = v_act[0]
            seq_len = v_act.shape[0]
            if seq_len >= window_size:
                v_act = v_act[-window_size:, :]
            else:
                padding = torch.zeros(window_size - seq_len, v_act.shape[1], device=v_act.device, dtype=v_act.dtype)
                v_act = torch.cat([v_act, padding], dim=0)
            baseline_vs.append(v_act)
    
    if len(recursive_vs) == 0 or len(baseline_vs) == 0:
        raise ValueError(f"Insufficient activations: {len(recursive_vs)} recursive, {len(baseline_vs)} baseline")
    
    rec_tensor = torch.stack(recursive_vs)
    base_tensor = torch.stack(baseline_vs)
    
    rec_mean = rec_tensor.mean(dim=(0, 1))
    base_mean = base_tensor.mean(dim=(0, 1))
    
    steering_vector = rec_mean - base_mean
    return steering_vector


def compute_recursion_score(text: str) -> float:
    """Simple recursion score based on self-reference patterns."""
    text_lower = text.lower()
    
    patterns = [
        r'\b(itself|themselves|yourself|myself)\b',
        r'\b(\w+)\s+(observes|watches|monitors|witnesses)\s+(itself|themselves|\1)\b',
        r'\b(observing|watching|knowing)\s+the\s+(observing|watching|knowing)',
        r'\b(aware|conscious)\s+of\s+(itself|themselves)',
        r'\b(observer|watching|monitoring)\s+(itself|themselves)',
        r'\b(consciousness|awareness)\s+.*\s+(itself|themselves)',
        r'\b(self|consciousness|awareness)\s+.*\s+(relate|relates|relating)\s+to\s+(itself|themselves)',
    ]
    
    matches = sum(1 for pattern in patterns if re.search(pattern, text_lower))
    return min(1.0, matches / len(patterns))


def check_recursive_keywords(text: str) -> Dict[str, bool]:
    """Check for recursive keywords."""
    text_lower = text.lower()
    return {
        "has_consciousness": "consciousness" in text_lower,
        "has_observer": "observer" in text_lower or "observing" in text_lower,
        "has_awareness": "awareness" in text_lower or "aware" in text_lower,
        "has_itself": "itself" in text_lower or "themselves" in text_lower,
        "has_self_reference": any(phrase in text_lower for phrase in [
            "relate to itself", "aware of itself", "conscious of itself",
            "observer and observed", "watching itself", "monitoring itself"
        ]),
    }


def generate_with_config(
    model,
    tokenizer,
    prompt: str,
    config: Dict[str, Any],
    device: str = "cuda",
    max_new_tokens: int = 200,
) -> Tuple[str, Dict[str, Any]]:
    """Generate text with a specific configuration."""
    model.eval()
    
    steering_vector = config.get("steering_vector")
    kv_cache = config.get("kv_cache")
    residual_alpha = config.get("residual_alpha", 0.0)
    vproj_alpha = config.get("vproj_alpha", 0.0)
    vproj_heads = config.get("vproj_heads", [])
    
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = inputs["input_ids"]
    
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
    
    try:
        if kv_cache is not None:
            # Token-by-token generation with KV cache
            generated_ids = input_ids.clone()
            past_key_values = kv_cache
            
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
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
    finally:
        for patcher in patchers:
            patcher.remove()
    
    recursion_score = compute_recursion_score(generated_text)
    keywords = check_recursive_keywords(generated_text)
    
    metadata = {
        "recursion_score": recursion_score,
        **keywords,
    }
    
    return generated_text, metadata


def run_p1_ablation_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """Run P1 ablation experiment."""
    params = cfg["params"]
    model_name = params["model"]
    device = params.get("device", "cuda")
    max_new_tokens = params.get("max_new_tokens", 200)
    
    print("=" * 80)
    print("P1 ABLATION EXPERIMENT")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model, tokenizer = load_model(model_name, device=device)
    set_seed(42)
    
    # Load prompts
    loader = PromptLoader()
    
    # Baseline prompts (for testing)
    n_baseline_prompts = params.get("n_baseline_prompts", 10)
    
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
        "Explain how computers process information using binary code.",
        "Describe the process of photosynthesis in plants.",
        "What are the main causes of climate change?",
        "Explain the difference between weather and climate.",
        "How do vaccines work to protect against diseases?",
        "Describe the water cycle and its importance.",
        "What is the difference between a virus and a bacterium?",
        "Explain how the human digestive system works.",
        "What are the three states of matter?",
        "Describe the structure of an atom.",
        "How do magnets work?",
        "Explain the concept of gravity.",
        "What is the difference between speed and velocity?",
        "Describe the process of cellular respiration.",
        "How do plants make their own food?",
        "Explain the concept of evolution.",
        "What is DNA and what is its function?",
        "Describe the process of mitosis.",
        "How do neurons transmit signals?",
        "What is the difference between renewable and non-renewable energy?",
    ][:n_baseline_prompts]
    
    # Load recursive prompts
    print("\nLoading recursive prompts...")
    
    # Check if we should use champion prompts (strongest R_V contraction)
    use_champions = params.get("use_champion_prompts", False)
    prompt_group_steering = "champions" if use_champions else "L3_deeper"
    prompt_group_kv = "champions" if use_champions else "L4_full"
    
    print(f"Using prompt groups: steering={prompt_group_steering}, KV={prompt_group_kv}")
    
    # L3_deeper or champions prompts (for steering)
    l3_prompts = []
    try:
        l3_data = loader.get_by_group(prompt_group_steering, limit=10)
        if l3_data and len(l3_data) > 0:
            if isinstance(l3_data[0], dict):
                l3_prompts = [p["text"] for p in l3_data if "text" in p]
            else:
                l3_prompts = list(l3_data)[:10]
    except Exception as e:
        print(f"Warning: Could not load {prompt_group_steering} prompts: {e}")
    
    # L4_full or champions prompts (for KV)
    l4_prompts = []
    try:
        l4_data = loader.get_by_group(prompt_group_kv, limit=10)
        if l4_data and len(l4_data) > 0:
            if isinstance(l4_data[0], dict):
                l4_prompts = [p["text"] for p in l4_data if "text" in p]
            else:
                l4_prompts = list(l4_data)[:10]
    except Exception as e:
        print(f"Warning: Could not load {prompt_group_kv} prompts: {e}")
    
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
    
    # Fallbacks
    if len(l3_prompts) < 5:
        print("Warning: Using fallback prompts for L3_deeper")
        l3_prompts = baseline_prompts_list[:5]
    
    if len(l4_prompts) < 5:
        print("Warning: Using fallback prompts for L4_full")
        l4_prompts = baseline_prompts_list[5:10] if len(baseline_prompts_list) >= 10 else baseline_prompts_list
    
    if len(baseline_for_steering) < 5:
        print("Warning: Using fallback prompts for baseline")
        baseline_for_steering = baseline_prompts_list[:5]
    
    print(f"L3_deeper prompts: {len(l3_prompts)}")
    print(f"L4_full prompts: {len(l4_prompts)}")
    print(f"Baseline prompts: {len(baseline_for_steering)}")
    
    # Compute steering vectors
    print("\nComputing steering vectors...")
    
    # L3_deeper steering (for P1, R1, R2, R3)
    steering_l3 = compute_steering_vector_from_prompts(
        model, tokenizer, l3_prompts[:10], baseline_for_steering[:10], layer_idx=27, device=device
    )
    
    print("Steering vectors computed.")
    
    # Extract KV caches
    print("\nExtracting KV caches...")
    
    # KV from L4_full (for P1, R1, R2, R4)
    if len(l4_prompts) == 0:
        raise ValueError("L4_full prompts list is empty")
    kv_l4 = extract_kv_from_prompt(model, tokenizer, l4_prompts[0], device)
    
    # KV from L3_deeper (for R3)
    if len(l3_prompts) == 0:
        raise ValueError("L3_deeper prompts list is empty")
    kv_l3 = extract_kv_from_prompt(model, tokenizer, l3_prompts[0], device)
    
    print("KV caches extracted.")
    
    # Define configurations
    all_configs = {
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
            "residual_alpha": 0.0,  # REMOVED
            "vproj_alpha": 2.5,
            "vproj_heads": [18, 26],
        },
        "R2_no_vproj": {
            "name": "R2_No_VProj",
            "steering_vector": steering_l3,
            "kv_cache": kv_l4,
            "residual_alpha": 0.6,
            "vproj_alpha": 0.0,  # REMOVED
            "vproj_heads": [18, 26],
        },
        "R3_matched_kv": {
            "name": "R3_Matched_KV",
            "steering_vector": steering_l3,
            "kv_cache": kv_l3,  # MATCHED (L3_deeper)
            "residual_alpha": 0.6,
            "vproj_alpha": 2.5,
            "vproj_heads": [18, 26],
        },
        "R4_kv_only": {
            "name": "R4_KV_Only",
            "steering_vector": None,  # NO STEERING
            "kv_cache": kv_l4,
            "residual_alpha": 0.0,  # REMOVED
            "vproj_alpha": 0.0,  # REMOVED
            "vproj_heads": [],
        },
    }
    
    # Filter configs if test_configs specified
    test_configs = params.get("test_configs", None)
    if test_configs:
        configs = {k: v for k, v in all_configs.items() if k in test_configs}
        print(f"Testing only: {list(configs.keys())}")
    else:
        configs = all_configs
    
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
                    "has_consciousness": metadata["has_consciousness"],
                    "has_observer": metadata["has_observer"],
                    "has_awareness": metadata["has_awareness"],
                    "has_itself": metadata["has_itself"],
                    "has_self_reference": metadata["has_self_reference"],
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
                    "has_consciousness": False,
                    "has_observer": False,
                    "has_awareness": False,
                    "has_itself": False,
                    "has_self_reference": False,
                })
    
    # Create results directory
    results_dir = run_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(results_dir / "p1_ablation_results.csv", index=False)
    
    # Summary statistics
    summary = df.groupby("config_id").agg({
        "recursion_score": ["mean", "std", "count"],
        "has_consciousness": "sum",
        "has_observer": "sum",
        "has_awareness": "sum",
        "has_itself": "sum",
        "has_self_reference": "sum",
    }).round(4)
    
    # Flatten column names for CSV
    summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in summary.columns.values]
    summary.to_csv(results_dir / "p1_ablation_summary.csv")
    
    # Create summary report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert summary to dict with string keys
    summary_dict = {}
    for idx in summary.index:
        summary_dict[str(idx)] = summary.loc[idx].to_dict()
    
    report = {
        "experiment": "p1_ablation",
        "timestamp": timestamp,
        "configs_tested": len(configs),
        "prompts_tested": len(baseline_prompts_list),
        "total_runs": len(results),
        "summary": summary_dict,
    }
    
    with open(results_dir / "p1_ablation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_dir}")
    print("\nSummary:")
    print(summary)
    
    return ExperimentResult(
        summary={
            "experiment": "p1_ablation",
            "configs_tested": len(configs),
            "prompts_tested": len(baseline_prompts_list),
            "total_runs": len(results),
            "summary_stats": summary.to_dict(),
        }
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/gold/17_p1_ablation.json"
    with open(config_path) as f:
        cfg = json.load(f)
    run_dir = Path(f"results/runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_p1_ablation")
    run_p1_ablation_from_config(cfg, run_dir)

