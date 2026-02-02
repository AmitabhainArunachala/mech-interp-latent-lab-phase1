#!/usr/bin/env python3
"""
RANDOM KV EFFECT INVESTIGATION
==============================

Investigates why random KV cache replacement shows same effect as recursive KV-only.

Hypotheses:
1. It's not about content - any KV replacement resets something
2. Sequence length matching - random prompt happened to match length
3. KV cache structure itself - replacing KV cache has an effect regardless of content
4. Measurement artifact - something about how we measure behavior

Tests:
1. Truly random KV (Gaussian noise) vs structured random KV (from baseline prompt)
2. Length-matched vs length-mismatched KV replacement
3. KV cache replacement vs no replacement (control)
4. Different random prompt sources (baseline vs recursive vs truly random)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import DynamicCache

sys.path.insert(0, str(Path(__file__).parent))

from src.core.models import load_model, set_seed
from src.core.hooks import capture_v_projection
from src.metrics.rv import participation_ratio
from src.metrics.behavior_states import label_behavior_state
from prompts.loader import PromptLoader

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
EARLY_LAYER = 5
LATE_LAYER = 27
WINDOW = 16
SEED = 42
N_PAIRS = 50
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
DO_SAMPLE = True

# =============================================================================
# KV CACHE MANIPULATION
# =============================================================================

def extract_full_kv_cache(model, tokenizer, prompt: str, device: str = "cuda") -> DynamicCache:
    """Extract full KV cache from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=False)
        kv_cache = outputs.past_key_values
    return kv_cache


def create_random_kv_cache(
    model,
    tokenizer,
    reference_prompt: str,
    method: str = "gaussian",
    device: str = "cuda",
) -> DynamicCache:
    """
    Create random KV cache using different methods.
    
    Methods:
    - "gaussian": Truly random Gaussian noise matching reference shape
    - "shuffled": Shuffled tokens from reference prompt
    - "baseline": KV from random baseline prompt (original method)
    """
    # First, get reference KV cache to match shape
    ref_kv = extract_full_kv_cache(model, tokenizer, reference_prompt, device)
    
    if method == "gaussian":
        # Create Gaussian noise matching reference shape
        random_kv = DynamicCache()
        # Iterate over layers using past_key_values tuple
        for layer_idx in range(len(ref_kv.key_cache)):
            k_ref = ref_kv.key_cache[layer_idx]
            v_ref = ref_kv.value_cache[layer_idx]
            
            # Get shapes
            k_shape = k_ref.shape
            v_shape = v_ref.shape
            
            # Create random tensors with same mean/std as reference
            k_mean = k_ref.mean().item()
            k_std = k_ref.std().item()
            v_mean = v_ref.mean().item()
            v_std = v_ref.std().item()
            
            k_random = torch.randn(k_shape, device=device, dtype=k_ref.dtype) * k_std + k_mean
            v_random = torch.randn(v_shape, device=device, dtype=v_ref.dtype) * v_std + v_mean
            
            random_kv.update(k_random, v_random, layer_idx)
        
        return random_kv
    
    elif method == "shuffled":
        # Use KV from reference but shuffle the sequence dimension
        random_kv = DynamicCache()
        # Iterate over layers using past_key_values tuple
        for layer_idx in range(len(ref_kv.key_cache)):
            k_ref = ref_kv.key_cache[layer_idx]
            v_ref = ref_kv.value_cache[layer_idx]
            
            # Shuffle along sequence dimension (dim=2 for keys, dim=2 for values)
            seq_len = k_ref.shape[2]
            shuffle_idx = torch.randperm(seq_len, device=device)
            
            k_shuffled = k_ref[:, :, shuffle_idx, :].clone()
            v_shuffled = v_ref[:, :, shuffle_idx, :].clone()
            
            random_kv.update(k_shuffled, v_shuffled, layer_idx)
        
        return random_kv
    
    elif method == "baseline":
        # Original method: use KV from random baseline prompt
        loader = PromptLoader()
        baseline_prompts = loader.get_by_pillar("baseline", limit=100, seed=SEED)
        random_prompt = random.choice(baseline_prompts)
        return extract_full_kv_cache(model, tokenizer, random_prompt, device)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def replace_kv_cache_during_generation(
    model,
    tokenizer,
    prompt: str,
    source_kv_cache: Optional[DynamicCache] = None,
    device: str = "cuda",
) -> Tuple[str, float]:
    """
    Generate text with replaced KV cache.
    
    Returns:
        (generated_text, rv_value)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    target_seq_len = input_ids.shape[1]
    
    # Check if we can use KV cache (sequence lengths must match)
    use_kv_cache = False
    if source_kv_cache is not None:
        try:
            first_layer_kv = source_kv_cache.key_cache[0]
            source_seq_len = first_layer_kv.shape[2] if len(first_layer_kv.shape) > 2 else first_layer_kv.shape[1]
            if source_seq_len == target_seq_len:
                use_kv_cache = True
        except:
            use_kv_cache = False
    
    # Initial forward pass
    with torch.no_grad():
        if use_kv_cache:
            outputs = model(
                input_ids=input_ids,
                past_key_values=source_kv_cache,
                use_cache=True,
            )
        else:
            outputs = model(
                input_ids=input_ids,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
    
    # Generate tokens
    generated_ids = input_ids.clone()
    for _ in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            outputs = model(
                input_ids=generated_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            
            if DO_SAMPLE:
                probs = torch.softmax(logits / TEMPERATURE, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
            
            if next_token_id.dim() == 0:
                next_token_id = next_token_id.unsqueeze(0).unsqueeze(0)
            elif next_token_id.dim() == 1:
                next_token_id = next_token_id.unsqueeze(0)
            
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            past_key_values = outputs.past_key_values
            
            if next_token_id.item() == tokenizer.eos_token_id:
                break
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Compute R_V on the full sequence
    with capture_v_projection(model, EARLY_LAYER) as v_early_storage:
        with capture_v_projection(model, LATE_LAYER) as v_late_storage:
            model(generated_ids)
    
    v_early = v_early_storage.get("v")
    v_late = v_late_storage.get("v")
    
    if v_early is None or v_late is None:
        rv = float('nan')
    else:
        if v_early.dim() == 3:
            v_early = v_early[0]
        if v_late.dim() == 3:
            v_late = v_late[0]
        
        pr_early = participation_ratio(v_early, window_size=WINDOW)
        pr_late = participation_ratio(v_late, window_size=WINDOW)
        
        if pr_early == 0 or np.isnan(pr_early) or np.isnan(pr_late):
            rv = float('nan')
        else:
            rv = float(pr_late / pr_early)
    
    return generated_text, rv


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 80)
    print("RANDOM KV EFFECT INVESTIGATION")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"N pairs: {N_PAIRS}")
    print("=" * 80)
    
    set_seed(SEED)
    # Use local_files_only=True to avoid downloading if model is cached
    try:
        model, tokenizer = load_model(MODEL_NAME, device=DEVICE, torch_dtype=torch.float16, attn_implementation="eager")
    except Exception as e:
        print(f"⚠️  Error loading model: {e}")
        print("Trying with local_files_only=True...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model.eval()
    
    # Load prompts
    loader = PromptLoader()
    # Try multiple baseline pillar names
    baseline_prompts = []
    for pillar_name in ["baseline", "baselines", "control"]:
        baseline_prompts = loader.get_by_pillar(pillar_name, limit=N_PAIRS, seed=SEED)
        if len(baseline_prompts) > 0:
            break
    # If still empty, use get_by_type
    if len(baseline_prompts) == 0:
        baseline_prompts = loader.get_by_type("baseline", limit=N_PAIRS, seed=SEED)
    
    recursive_prompts = loader.get_by_pillar("dose_response", limit=N_PAIRS, seed=SEED)
    
    print(f"\nLoaded {len(baseline_prompts)} baseline and {len(recursive_prompts)} recursive prompts")
    
    if len(baseline_prompts) == 0 or len(recursive_prompts) == 0:
        print("⚠️  ERROR: Not enough prompts loaded!")
        print(f"  Baseline: {len(baseline_prompts)}")
        print(f"  Recursive: {len(recursive_prompts)}")
        return
    
    results = []
    
    print("\n[1/5] Testing conditions...")
    print("Conditions:")
    print("  1. Control (no KV replacement)")
    print("  2. Recursive KV (from recursive prompt)")
    print("  3. Random KV - Gaussian noise")
    print("  4. Random KV - Shuffled tokens")
    print("  5. Random KV - Baseline prompt (original method)")
    
    for pair_idx in tqdm(range(min(len(baseline_prompts), len(recursive_prompts))), desc="Processing pairs"):
        baseline_prompt = baseline_prompts[pair_idx]
        recursive_prompt = recursive_prompts[pair_idx]
        
        try:
            # Condition 1: Control (no KV replacement)
            control_text, control_rv = replace_kv_cache_during_generation(
                model, tokenizer, baseline_prompt, source_kv_cache=None, device=DEVICE
            )
            control_label = label_behavior_state(control_text)
            
            # Condition 2: Recursive KV
            recursive_kv = extract_full_kv_cache(model, tokenizer, recursive_prompt, DEVICE)
            recursive_kv_text, recursive_kv_rv = replace_kv_cache_during_generation(
                model, tokenizer, baseline_prompt, source_kv_cache=recursive_kv, device=DEVICE
            )
            recursive_kv_label = label_behavior_state(recursive_kv_text)
            
            # Condition 3: Gaussian noise KV
            gaussian_kv = create_random_kv_cache(model, tokenizer, baseline_prompt, method="gaussian", device=DEVICE)
            gaussian_kv_text, gaussian_kv_rv = replace_kv_cache_during_generation(
                model, tokenizer, baseline_prompt, source_kv_cache=gaussian_kv, device=DEVICE
            )
            gaussian_kv_label = label_behavior_state(gaussian_kv_text)
            
            # Condition 4: Shuffled KV
            shuffled_kv = create_random_kv_cache(model, tokenizer, baseline_prompt, method="shuffled", device=DEVICE)
            shuffled_kv_text, shuffled_kv_rv = replace_kv_cache_during_generation(
                model, tokenizer, baseline_prompt, source_kv_cache=shuffled_kv, device=DEVICE
            )
            shuffled_kv_label = label_behavior_state(shuffled_kv_text)
            
            # Condition 5: Baseline KV (original random method)
            baseline_kv = create_random_kv_cache(model, tokenizer, baseline_prompt, method="baseline", device=DEVICE)
            baseline_kv_text, baseline_kv_rv = replace_kv_cache_during_generation(
                model, tokenizer, baseline_prompt, source_kv_cache=baseline_kv, device=DEVICE
            )
            baseline_kv_label = label_behavior_state(baseline_kv_text)
            
            # Store results
            results.append({
                "pair_id": pair_idx,
                "baseline_prompt": baseline_prompt[:100],
                "recursive_prompt": recursive_prompt[:100],
                
                # Control
                "control_rv": control_rv,
                "control_text": control_text,
                "control_behavior_score": control_label.has_recursive_keywords or control_label.has_identity_equation,
                "control_has_recursive": control_label.has_recursive_keywords,
                "control_has_identity": control_label.has_identity_equation,
                
                # Recursive KV
                "recursive_kv_rv": recursive_kv_rv,
                "recursive_kv_text": recursive_kv_text,
                "recursive_kv_behavior_score": recursive_kv_label.has_recursive_keywords or recursive_kv_label.has_identity_equation,
                "recursive_kv_has_recursive": recursive_kv_label.has_recursive_keywords,
                "recursive_kv_has_identity": recursive_kv_label.has_identity_equation,
                
                # Gaussian KV
                "gaussian_kv_rv": gaussian_kv_rv,
                "gaussian_kv_text": gaussian_kv_text,
                "gaussian_kv_behavior_score": gaussian_kv_label.has_recursive_keywords or gaussian_kv_label.has_identity_equation,
                "gaussian_kv_has_recursive": gaussian_kv_label.has_recursive_keywords,
                "gaussian_kv_has_identity": gaussian_kv_label.has_identity_equation,
                
                # Shuffled KV
                "shuffled_kv_rv": shuffled_kv_rv,
                "shuffled_kv_text": shuffled_kv_text,
                "shuffled_kv_behavior_score": shuffled_kv_label.has_recursive_keywords or shuffled_kv_label.has_identity_equation,
                "shuffled_kv_has_recursive": shuffled_kv_label.has_recursive_keywords,
                "shuffled_kv_has_identity": shuffled_kv_label.has_identity_equation,
                
                # Baseline KV
                "baseline_kv_rv": baseline_kv_rv,
                "baseline_kv_text": baseline_kv_text,
                "baseline_kv_behavior_score": baseline_kv_label.has_recursive_keywords or baseline_kv_label.has_identity_equation,
                "baseline_kv_has_recursive": baseline_kv_label.has_recursive_keywords,
                "baseline_kv_has_identity": baseline_kv_label.has_identity_equation,
            })
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"\n⚠️  Error processing pair {pair_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    if len(results) == 0:
        print("\n⚠️  WARNING: No results collected. Check errors above.")
        return
    
    output_dir = Path("results/path_b_validation/runs") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_random_kv_investigation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "results.csv", index=False)
    
    # Summary statistics
    summary = {
        "experiment": "random_kv_investigation",
        "model": MODEL_NAME,
        "n_pairs": len(results),
        "conditions": ["control", "recursive_kv", "gaussian_kv", "shuffled_kv", "baseline_kv"],
        "results": {},
    }
    
    for condition in summary["conditions"]:
        rv_col = f"{condition}_rv"
        behavior_col = f"{condition}_behavior_score"
        
        if rv_col not in df.columns or behavior_col not in df.columns:
            print(f"⚠️  Warning: Missing columns for {condition}")
            continue
        
        rv_values = df[rv_col].dropna()
        behavior_values = df[behavior_col].dropna()
        
        summary["results"][condition] = {
            "mean_rv": float(rv_values.mean()) if len(rv_values) > 0 else np.nan,
            "std_rv": float(rv_values.std()) if len(rv_values) > 0 else np.nan,
            "mean_behavior_score": float(behavior_values.mean()) if len(behavior_values) > 0 else np.nan,
            "behavior_score_std": float(behavior_values.std()) if len(behavior_values) > 0 else np.nan,
            "expression_rate": float(behavior_values.mean()) if len(behavior_values) > 0 else np.nan,
            "has_recursive_keywords": int(df[f"{condition}_has_recursive"].sum()) if f"{condition}_has_recursive" in df.columns else 0,
            "has_identity_equation": int(df[f"{condition}_has_identity"].sum()) if f"{condition}_has_identity" in df.columns else 0,
        }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    for condition in summary["conditions"]:
        stats = summary["results"][condition]
        print(f"\n{condition}:")
        print(f"  Mean R_V: {stats['mean_rv']:.4f} ± {stats['std_rv']:.4f}")
        print(f"  Expression rate: {stats['expression_rate']*100:.1f}%")
        print(f"  Recursive keywords: {stats['has_recursive_keywords']}/{len(results)}")
        print(f"  Identity equations: {stats['has_identity_equation']}/{len(results)}")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("\nIf Gaussian KV shows same effect as recursive KV:")
    print("  → Effect is NOT content-specific (any KV replacement works)")
    print("\nIf Gaussian KV shows NO effect but baseline KV does:")
    print("  → Effect requires structured KV cache (from real prompts)")
    print("\nIf shuffled KV shows same effect as recursive KV:")
    print("  → Effect is about KV cache structure, not token order")
    
    print(f"\n✅ Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

