#!/usr/bin/env python3
"""
EXPERIMENT 2: KV-Only Sufficiency Control
==========================================

Tests if full KV cache replacement ALONE (without V_PROJ patching) transfers
recursive behavior.

This resolves the confound in n=300 results where "wrong layer" (L5) also worked,
suggesting full KV cache might be doing most of the work.

Conditions:
1. Control: Baseline prompt, no patching
2. KV-only: Full KV cache from recursive prompt, NO V_PROJ patching
3. KV+V_PROJ: Full KV cache + persistent V_PROJ at L27 (positive control)
4. Random KV: Random KV cache, NO V_PROJ (negative control)

Measures:
- R_V (geometry)
- Behavior score (recursive keywords, identity equations)
- Generated text
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

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
N_PAIRS = 50  # Baseline-recursive pairs
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
DO_SAMPLE = True

# =============================================================================
# KV CACHE EXTRACTION & REPLACEMENT
# =============================================================================

def extract_full_kv_cache(
    model,
    tokenizer,
    prompt: str,
    device: str = "cuda",
) -> DynamicCache:
    """
    Extract full KV cache from a prompt forward pass.
    
    Returns:
        DynamicCache with KV for all layers
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=False)
        kv_cache = outputs.past_key_values
    
    return kv_cache


def replace_kv_cache_during_generation(
    model,
    tokenizer,
    prompt: str,
    source_kv_cache: Optional[DynamicCache] = None,
    patch_v_proj: bool = False,
    source_v_proj: Optional[torch.Tensor] = None,
    device: str = "cuda",
) -> Tuple[str, float]:
    """
    Generate text with replaced KV cache, optionally patching V_PROJ.
    
    NOTE: KV cache replacement only works if source and target prompts have matching sequence lengths.
    For mismatched lengths, we skip KV replacement and only use V_PROJ patching.
    
    Returns:
        (generated_text, rv_value)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    target_seq_len = input_ids.shape[1]
    
    # Set up V_PROJ patching if requested
    handles = []
    if patch_v_proj and source_v_proj is not None:
        def make_v_proj_hook(source_v):
            def hook_fn(module, inp, out):
                # Replace output with source V_PROJ
                # Handle sequence length mismatch by truncating/padding
                if source_v.shape[1] != out.shape[1]:
                    if source_v.shape[1] > out.shape[1]:
                        # Truncate source to target length
                        return source_v[:, :out.shape[1], :]
                    else:
                        # Pad source to target length (repeat last token)
                        padding = source_v[:, -1:, :].repeat(1, out.shape[1] - source_v.shape[1], 1)
                        return torch.cat([source_v, padding], dim=1)
                return source_v
            return hook_fn
        
        v_proj_layer = model.model.layers[LATE_LAYER].self_attn.v_proj
        handle = v_proj_layer.register_forward_hook(make_v_proj_hook(source_v_proj))
        handles.append(handle)
    
    try:
        # Initial forward pass
        with torch.no_grad():
            # Check if we can use source KV cache (sequence lengths must match)
            use_kv_cache = False
            if source_kv_cache is not None:
                # Check if sequence lengths match by inspecting first layer's KV cache
                try:
                    first_layer_kv = source_kv_cache.key_cache[0]
                    source_seq_len = first_layer_kv.shape[2] if len(first_layer_kv.shape) > 2 else first_layer_kv.shape[1]
                    if source_seq_len == target_seq_len:
                        use_kv_cache = True
                except:
                    use_kv_cache = False
            
            if use_kv_cache:
                # Use source KV cache for all layers
                outputs = model(
                    input_ids=input_ids,
                    past_key_values=source_kv_cache,
                    use_cache=True,
                )
            else:
                # Normal forward pass (control condition or mismatched lengths)
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
                
                # Ensure next_token_id has correct shape: (1, 1) for batch=1, seq_len=1
                if next_token_id.dim() == 0:
                    next_token_id = next_token_id.unsqueeze(0).unsqueeze(0)  # (1, 1)
                elif next_token_id.dim() == 1:
                    next_token_id = next_token_id.unsqueeze(0)  # (1, seq_len) -> should be (1, 1)
                # generated_ids is (1, seq_len), next_token_id should be (1, 1)
                generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
                past_key_values = outputs.past_key_values
                
                # Check for EOS
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Compute R_V on the full sequence (prompt + generated)
        with capture_v_projection(model, EARLY_LAYER) as v_early_storage:
            with capture_v_projection(model, LATE_LAYER) as v_late_storage:
                model(generated_ids)
        
        v_early = v_early_storage.get("v")
        v_late = v_late_storage.get("v")
        
        if v_early is None or v_late is None:
            rv = float('nan')
        else:
            # Normalize to 2D: participation_ratio expects (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
            # and handles 3D by taking [0]. So convert both to 2D to avoid dimension mismatches.
            if v_early.dim() == 3:
                v_early = v_early[0]  # (batch, seq, hidden) -> (seq, hidden)
            if v_late.dim() == 3:
                v_late = v_late[0]  # (batch, seq, hidden) -> (seq, hidden)
            
            pr_early = participation_ratio(v_early, window_size=WINDOW)
            pr_late = participation_ratio(v_late, window_size=WINDOW)
            
            if pr_early == 0 or np.isnan(pr_early) or np.isnan(pr_late):
                rv = float('nan')
            else:
                rv = float(pr_late / pr_early)
        
        return generated_text, rv
    
    finally:
        # Remove hooks
        for handle in handles:
            handle.remove()


def extract_v_proj_at_layer(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    device: str = "cuda",
) -> torch.Tensor:
    """Extract V_PROJ output at a specific layer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        with capture_v_projection(model, layer_idx) as storage:
            model(**inputs)
        v_proj = storage.get("v")
    
    return v_proj


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 80)
    print("EXPERIMENT 2: KV-ONLY SUFFICIENCY CONTROL")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"N pairs: {N_PAIRS}")
    print("=" * 80)
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/path_b_validation/runs/{timestamp}_kv_only_control")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    set_seed(SEED)
    model, tokenizer = load_model(MODEL_NAME, device=DEVICE, attn_implementation="eager")
    model.eval()
    
    # Load prompt pairs
    loader = PromptLoader()
    recursive_prompts = loader.get_by_pillar("dose_response", limit=N_PAIRS, seed=SEED)
    # Try multiple baseline pillar names
    baseline_prompts = []
    for pillar_name in ["baseline", "baselines", "control"]:
        baseline_prompts = loader.get_by_pillar(pillar_name, limit=N_PAIRS, seed=SEED)
        if len(baseline_prompts) > 0:
            break
    # If still empty, use get_by_type
    if len(baseline_prompts) == 0:
        baseline_prompts = loader.get_by_type("baseline", limit=N_PAIRS, seed=SEED)
    
    print(f"\nLoaded {len(recursive_prompts)} recursive and {len(baseline_prompts)} baseline prompts")
    
    results = []
    
    for pair_idx in tqdm(range(min(len(recursive_prompts), len(baseline_prompts))), desc="Processing pairs"):
        recursive_prompt = recursive_prompts[pair_idx]
        baseline_prompt = baseline_prompts[pair_idx]
        
        try:
            # Extract KV cache and V_PROJ from recursive prompt
            recursive_kv = extract_full_kv_cache(model, tokenizer, recursive_prompt, DEVICE)
            recursive_v_proj = extract_v_proj_at_layer(model, tokenizer, recursive_prompt, LATE_LAYER, DEVICE)
            
            # Extract random KV cache (negative control)
            # Use a different recursive prompt for "random" (not truly random, but different content)
            if pair_idx + 1 < len(recursive_prompts):
                random_prompt = recursive_prompts[pair_idx + 1]
            else:
                random_prompt = recursive_prompts[0]
            random_kv = extract_full_kv_cache(model, tokenizer, random_prompt, DEVICE)
            
            # Condition 1: Control (baseline prompt, no patching)
            control_text, control_rv = replace_kv_cache_during_generation(
                model, tokenizer, baseline_prompt, source_kv_cache=None, patch_v_proj=False, device=DEVICE
            )
            control_label = label_behavior_state(control_text)
            
            # Condition 2: KV-only (full KV cache, NO V_PROJ)
            kv_only_text, kv_only_rv = replace_kv_cache_during_generation(
                model, tokenizer, baseline_prompt, recursive_kv, patch_v_proj=False, device=DEVICE
            )
            kv_only_label = label_behavior_state(kv_only_text)
            
            # Condition 3: KV+V_PROJ (positive control)
            kv_vproj_text, kv_vproj_rv = replace_kv_cache_during_generation(
                model, tokenizer, baseline_prompt, recursive_kv, patch_v_proj=True,
                source_v_proj=recursive_v_proj, device=DEVICE
            )
            kv_vproj_label = label_behavior_state(kv_vproj_text)
            
            # Condition 4: Random KV (negative control)
            random_kv_text, random_kv_rv = replace_kv_cache_during_generation(
                model, tokenizer, baseline_prompt, random_kv, patch_v_proj=False, device=DEVICE
            )
            random_kv_label = label_behavior_state(random_kv_text)
            
            # Store results
            for condition_name, text, rv, label in [
                ("control", control_text, control_rv, control_label),
                ("kv_only", kv_only_text, kv_only_rv, kv_only_label),
                ("kv_vproj", kv_vproj_text, kv_vproj_rv, kv_vproj_label),
                ("random_kv", random_kv_text, random_kv_rv, random_kv_label),
            ]:
                results.append({
                    "pair_id": pair_idx,
                    "baseline_prompt": baseline_prompt[:100],
                    "recursive_prompt": recursive_prompt[:100],
                    "condition": condition_name,
                    "rv": rv,
                    "generated_text": text,
                    "behavior_state": label.state.value,
                    "has_recursive_keywords": label.has_recursive_keywords,
                    "has_identity_equation": label.has_identity_equation,
                    "behavior_score": int(label.has_recursive_keywords or label.has_identity_equation or label.state.value in ["naked_loop", "recursive_prose"]),
                })
            
        except Exception as e:
            print(f"\nError processing pair {pair_idx}: {e}")
            continue
    
    # Save results
    if len(results) == 0:
        print("\n⚠️  WARNING: No results collected. Check errors above.")
        return
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "results.csv", index=False)
    
    # Summary statistics
    summary = {
        "experiment": "kv_only_control",
        "model": MODEL_NAME,
        "n_pairs": N_PAIRS,
        "conditions": ["control", "kv_only", "kv_vproj", "random_kv"],
        "results": {},
    }
    
    for condition in ["control", "kv_only", "kv_vproj", "random_kv"]:
        cond_df = df[df["condition"] == condition] if len(df) > 0 else pd.DataFrame()
        summary["results"][condition] = {
            "mean_rv": float(cond_df["rv"].mean()),
            "std_rv": float(cond_df["rv"].std()),
            "mean_behavior_score": float(cond_df["behavior_score"].mean()),
            "behavior_score_std": float(cond_df["behavior_score"].std()),
            "expression_rate": float(cond_df["behavior_score"].mean()),
            "has_recursive_keywords": int(cond_df["has_recursive_keywords"].sum()),
            "has_identity_equation": int(cond_df["has_identity_equation"].sum()),
        }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    for condition in ["control", "kv_only", "kv_vproj", "random_kv"]:
        stats = summary["results"][condition]
        print(f"\n{condition}:")
        print(f"  Mean R_V: {stats['mean_rv']:.4f} ± {stats['std_rv']:.4f}")
        print(f"  Expression rate: {stats['expression_rate']:.2%}")
        print(f"  Recursive keywords: {stats['has_recursive_keywords']}/{N_PAIRS}")
        print(f"  Identity equations: {stats['has_identity_equation']}/{N_PAIRS}")
    
    print(f"\n✅ Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

