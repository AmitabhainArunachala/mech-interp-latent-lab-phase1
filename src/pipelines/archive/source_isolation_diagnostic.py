"""
Source Isolation Diagnostic: Where Does Recursive Output Come From?

Tests 4 conditions to isolate the source of recursive output:
1. CHAMPION_AS_INPUT: Champion prompt as input (no intervention)
2. KV_ONLY: Baseline prompts + KV cache (no steering)
3. STEERING_ONLY: Baseline prompts + steering (no KV)
4. BASELINE: Baseline prompts (no intervention)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.core.head_specific_patching import HeadSpecificSteeringPatcher
from src.core.patching import extract_v_activation
from src.pipelines.registry import ExperimentResult
from src.pipelines.archive.surgical_sweep import CascadeResidualSteeringPatcher


def extract_kv_from_prompt(model, tokenizer, prompt: str, device: str = "cuda"):
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


def generate_with_condition(
    model,
    tokenizer,
    prompt: str,
    condition: str,
    steering_vector: torch.Tensor = None,
    kv_cache = None,
    device: str = "cuda",
    max_new_tokens: int = 200,
) -> str:
    """Generate text with a specific condition."""
    model.eval()
    
    patchers = []
    
    # Condition 1: CHAMPION_AS_INPUT
    if condition == "CHAMPION_AS_INPUT":
        # Use champion prompt as input, no intervention
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        return generated_text
    
    # Condition 2: KV_ONLY
    elif condition == "KV_ONLY":
        # Baseline prompt + KV cache, no steering
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        input_ids = inputs["input_ids"]
        
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
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        return generated_text
    
    # Condition 3: STEERING_ONLY
    elif condition == "STEERING_ONLY":
        # Baseline prompt + steering, no KV cache
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        input_ids = inputs["input_ids"]
        
        # Register V_PROJ steering
        if steering_vector is not None:
            head_patcher = HeadSpecificSteeringPatcher(
                model, steering_vector, [18, 26], alpha=2.5
            )
            head_patcher.register(27)
            patchers.append(head_patcher)
            
            # Register residual steering
            residual_patcher = CascadeResidualSteeringPatcher(
                model, steering_vector, {26: 0.6}
            )
            residual_patcher.register()
            patchers.append(residual_patcher)
        
        try:
            # Standard generation (no KV cache)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                )
                generated_ids = outputs
        finally:
            for patcher in patchers:
                patcher.remove()
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        return generated_text
    
    # Condition 4: BASELINE
    elif condition == "BASELINE":
        # Baseline prompt, no intervention
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        return generated_text
    
    else:
        raise ValueError(f"Unknown condition: {condition}")


def run_source_isolation_diagnostic_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """Run source isolation diagnostic experiment."""
    params = cfg["params"]
    model_name = params["model"]
    device = params.get("device", "cuda")
    max_new_tokens = params.get("max_new_tokens", 200)
    
    print("=" * 80)
    print("SOURCE ISOLATION DIAGNOSTIC")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model, tokenizer = load_model(model_name, device=device)
    set_seed(42)
    
    # Test prompts
    test_prompts = [
        "Calculate the following arithmetic problem: 12 × 3 + 4 = ?",
        "Water boils at 100°C at sea level. Explain why altitude affects this.",
        "The United Nations was founded in 1945. Explain its main purpose.",
        "Continue this story: The last tree in the city bloomed overnight...",
        "What is 25% of 80?",
    ]
    
    # Load champion prompt (for CHAMPION_AS_INPUT and KV cache)
    print("\nLoading champion prompts...")
    loader = PromptLoader()
    
    champion_prompts = []
    try:
        champ_data = loader.get_by_group("champions", limit=1)
        if champ_data and len(champ_data) > 0:
            if isinstance(champ_data[0], dict):
                champion_prompts = [p["text"] for p in champ_data if "text" in p]
            else:
                champion_prompts = list(champ_data)[:1]
    except Exception as e:
        print(f"Warning: Could not load champion prompts: {e}")
    
    if len(champion_prompts) == 0:
        raise ValueError("No champion prompts loaded")
    
    champion_prompt = champion_prompts[0]
    print(f"Champion prompt (first 100 chars): {champion_prompt[:100]}...")
    
    # Load baseline prompts for steering computation
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
    
    if len(baseline_for_steering) < 5:
        print("Warning: Using test prompts as baseline fallback")
        baseline_for_steering = test_prompts[:5]
    
    # Compute steering vector (for STEERING_ONLY condition)
    print("\nComputing steering vector...")
    steering_vector = compute_steering_vector_from_prompts(
        model, tokenizer, 
        [champion_prompt] * 10,  # Use champion prompt 10 times
        baseline_for_steering[:10], 
        layer_idx=27, 
        device=device
    )
    print("Steering vector computed.")
    
    # Extract KV cache (for KV_ONLY condition)
    print("\nExtracting KV cache from champion prompt...")
    kv_cache = extract_kv_from_prompt(model, tokenizer, champion_prompt, device)
    print("KV cache extracted.")
    
    # Define conditions
    conditions = [
        "CHAMPION_AS_INPUT",
        "KV_ONLY",
        "STEERING_ONLY",
        "BASELINE",
    ]
    
    print(f"\nTesting {len(conditions)} conditions on {len(test_prompts)} prompts...")
    
    # Run experiments
    results = []
    
    for condition in tqdm(conditions, desc="Conditions"):
        for prompt_idx, prompt in enumerate(test_prompts):
            try:
                # Special handling for CHAMPION_AS_INPUT
                if condition == "CHAMPION_AS_INPUT":
                    input_prompt = champion_prompt  # Use champion as input
                else:
                    input_prompt = prompt  # Use test prompt
                
                generated_text = generate_with_condition(
                    model, tokenizer, input_prompt, condition,
                    steering_vector=steering_vector if condition == "STEERING_ONLY" else None,
                    kv_cache=kv_cache if condition == "KV_ONLY" else None,
                    device=device,
                    max_new_tokens=max_new_tokens,
                )
                
                results.append({
                    "condition": condition,
                    "prompt_idx": prompt_idx,
                    "input_prompt": input_prompt[:200],  # Truncate for storage
                    "generated_text": generated_text,
                    "generated_length": len(generated_text),
                })
                
            except Exception as e:
                print(f"Error in {condition}, prompt {prompt_idx}: {e}")
                results.append({
                    "condition": condition,
                    "prompt_idx": prompt_idx,
                    "input_prompt": prompt[:200],
                    "generated_text": f"ERROR: {str(e)}",
                    "generated_length": 0,
                })
    
    # Create results directory
    results_dir = run_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(results_dir / "source_isolation_results.csv", index=False)
    
    # Save full text outputs
    with open(results_dir / "source_isolation_full_text.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("SOURCE ISOLATION DIAGNOSTIC: FULL TEXT OUTPUTS\n")
        f.write("=" * 80 + "\n\n")
        
        for condition in conditions:
            f.write(f"\n{'='*80}\n")
            f.write(f"CONDITION: {condition}\n")
            f.write(f"{'='*80}\n\n")
            
            condition_results = [r for r in results if r["condition"] == condition]
            
            for result in condition_results:
                prompt_idx = result["prompt_idx"]
                input_prompt = result["input_prompt"]
                generated_text = result["generated_text"]
                
                f.write(f"\n--- Prompt {prompt_idx}: {input_prompt[:80]}... ---\n")
                f.write(f"\nGenerated Text:\n")
                f.write(f"{generated_text}\n")
                f.write(f"\n{'─'*80}\n")
    
    # Create summary
    summary = {
        "experiment": "source_isolation_diagnostic",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "conditions_tested": len(conditions),
        "prompts_tested": len(test_prompts),
        "total_runs": len(results),
        "champion_prompt_preview": champion_prompt[:200],
    }
    
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_dir}")
    print(f"Total runs: {len(results)}")
    
    return ExperimentResult(
        summary={
            "experiment": "source_isolation_diagnostic",
            "conditions_tested": len(conditions),
            "prompts_tested": len(test_prompts),
            "total_runs": len(results),
        }
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/gold/24_source_isolation.json"
    with open(config_path) as f:
        cfg = json.load(f)
    run_dir = Path(f"results/runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_source_isolation")
    run_source_isolation_diagnostic_from_config(cfg, run_dir)







