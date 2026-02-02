#!/usr/bin/env python3
"""
R_V CONTRACTION DURING SUPPRESSOR ABLATION TEST
===============================================

Test if R_V contraction persists when suppressor heads are ablated.

Question: If we remove the "regulatory" heads, does the geometric signature disappear?

Prediction A: R_V stays contracted → contraction is upstream (detection)
Prediction B: R_V expands → these heads ARE the contraction (processing)

Method:
1. Take 10 recursive prompts
2. Measure R_V normally (control)
3. Measure R_V with H6-group (H6/H14/H22/H30) ablated
4. Measure R_V with H18-group (H18/H26) ablated
5. Measure R_V with BOTH ablated
"""

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.core.models import load_model, set_seed
from src.core.hooks import capture_v_projection
from src.metrics.rv import participation_ratio
from prompts.loader import PromptLoader

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
TARGET_LAYER = 27
EARLY_LAYER = 5
LATE_LAYER = 27
WINDOW = 16
SEED = 42

# Head groups
H6_GROUP = [6, 14, 22, 30]  # Contraction-causing heads (suppress behavior)
H18_GROUP = [18, 26]  # Mode-switching heads (suppress behavior)

# Test prompts (recursive) - sourced from prompt bank to prevent drift
_loader = PromptLoader()
PROMPT_BANK_VERSION = _loader.version
TEST_PROMPTS = _loader.get_by_group("legacy_comprehensive_circuit_test_champions", limit=10, seed=SEED)

# =============================================================================
# V-PROJECTION ABLATION
# =============================================================================

@contextmanager
def zero_v_proj_heads(model, layer_idx: int, head_indices: List[int]):
    """
    Zero out V-projection values for multiple heads BEFORE attention.
    """
    num_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    # Find unique KV heads (since multiple query heads share KV heads in GQA)
    kv_head_indices = set()
    for head_idx in head_indices:
        kv_head_idx = head_idx % num_kv_heads if num_kv_heads < num_heads else head_idx
        kv_head_indices.add(kv_head_idx)
    
    handles = []
    
    def make_hook(kv_head_idx):
        def hook_fn(module, inp, out):
            v_proj_out = out.clone()
            
            if v_proj_out.dim() == 2:
                v_proj_out = v_proj_out.unsqueeze(0)
            
            batch, seq_len, kv_hidden_size = v_proj_out.shape
            expected_kv_size = num_kv_heads * head_dim
            
            if kv_hidden_size != expected_kv_size:
                return out
            
            try:
                v_reshaped = v_proj_out.view(batch, seq_len, num_kv_heads, head_dim)
            except RuntimeError:
                return out
            
            # Zero out the KV head
            v_reshaped[:, :, kv_head_idx, :] = 0.0
            
            # Reshape back
            v_zeroed = v_reshaped.view(batch, seq_len, kv_hidden_size)
            
            if out.dim() == 2:
                v_zeroed = v_zeroed.squeeze(0)
            
            return v_zeroed
        
        return hook_fn
    
    # Register hooks for unique KV heads
    layer = model.model.layers[layer_idx].self_attn
    for kv_head_idx in kv_head_indices:
        hook_fn = make_hook(kv_head_idx)
        handle = layer.v_proj.register_forward_hook(hook_fn)
        handles.append(handle)
    
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()

# =============================================================================
# R_V COMPUTATION
# =============================================================================

def compute_rv(model, tokenizer, prompt: str, ablate_heads: Optional[List[int]] = None) -> Optional[float]:
    """
    Compute R_V for a prompt, optionally with heads ablated.
    
    Returns:
        R_V value or None if computation failed
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    with torch.no_grad():
        if ablate_heads is not None:
            with zero_v_proj_heads(model, TARGET_LAYER, ablate_heads):
                with capture_v_projection(model, EARLY_LAYER) as v_early_storage:
                    with capture_v_projection(model, LATE_LAYER) as v_late_storage:
                        model(**inputs)
                        
                        v_early = v_early_storage.get("v")
                        v_late = v_late_storage.get("v")
        else:
            with capture_v_projection(model, EARLY_LAYER) as v_early_storage:
                with capture_v_projection(model, LATE_LAYER) as v_late_storage:
                    model(**inputs)
                    
                    v_early = v_early_storage.get("v")
                    v_late = v_late_storage.get("v")
    
    if v_early is None or v_late is None:
        return None
    
    # Extract tensor if it's a dict
    if isinstance(v_early, dict):
        v_early = v_early.get("v", None)
    if isinstance(v_late, dict):
        v_late = v_late.get("v", None)
    
    if v_early is None or v_late is None:
        return None
    
    # Ensure we have the right shape (handle batch dimension)
    if v_early.dim() == 3:
        v_early = v_early[0]  # Remove batch dimension
    if v_late.dim() == 3:
        v_late = v_late[0]
    
    try:
        pr_early = participation_ratio(v_early, window_size=WINDOW)
        pr_late = participation_ratio(v_late, window_size=WINDOW)
        
        if pr_early == 0 or np.isnan(pr_early) or np.isnan(pr_late):
            return None
        
        return float(pr_late / pr_early)
    except Exception as e:
        print(f"Error computing R_V: {e}")
        return None

# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    print("=" * 80)
    print("R_V CONTRACTION DURING SUPPRESSOR ABLATION TEST")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Layer: {TARGET_LAYER}")
    print(f"H6-group (suppressors): {H6_GROUP}")
    print(f"H18-group (suppressors): {H18_GROUP}")
    print(f"Prompts: {len(TEST_PROMPTS)}")
    print("=" * 80)
    print("\nQuestion: Does R_V contraction persist when suppressor heads are ablated?")
    print("Prediction A: R_V stays contracted → contraction is upstream (detection)")
    print("Prediction B: R_V expands → these heads ARE the contraction (processing)")
    print("=" * 80)
    
    set_seed(SEED)
    
    # Load model
    print("\n[1/4] Loading model...")
    model, tokenizer = load_model(
        model_name=MODEL_NAME,
        device=DEVICE,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    model.eval()
    print("  ✅ Model loaded")
    
    # Compute R_V for all conditions
    print("\n[2/4] Computing R_V for all conditions...")
    results = []
    
    for i, prompt in enumerate(tqdm(TEST_PROMPTS, desc="Processing prompts")):
        prompt_results = {"prompt_idx": i, "prompt": prompt[:80]}
        
        # Control (no ablation)
        rv_control = compute_rv(model, tokenizer, prompt, ablate_heads=None)
        prompt_results["control"] = rv_control
        
        # H6-group ablated
        rv_h6 = compute_rv(model, tokenizer, prompt, ablate_heads=H6_GROUP)
        prompt_results["h6_ablated"] = rv_h6
        
        # H18-group ablated
        rv_h18 = compute_rv(model, tokenizer, prompt, ablate_heads=H18_GROUP)
        prompt_results["h18_ablated"] = rv_h18
        
        # Both ablated
        rv_both = compute_rv(model, tokenizer, prompt, ablate_heads=H6_GROUP + H18_GROUP)
        prompt_results["both_ablated"] = rv_both
        
        results.append(prompt_results)
        
        # Clear GPU cache
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    
    # Analyze results
    print("\n[3/4] Analyzing results...")
    print("=" * 80)
    
    # Filter out None values
    control_rvs = [r["control"] for r in results if r["control"] is not None]
    h6_rvs = [r["h6_ablated"] for r in results if r["h6_ablated"] is not None]
    h18_rvs = [r["h18_ablated"] for r in results if r["h18_ablated"] is not None]
    both_rvs = [r["both_ablated"] for r in results if r["both_ablated"] is not None]
    
    print("\n📊 RESULTS SUMMARY")
    print("-" * 80)
    print(f"Control (no ablation):")
    print(f"  Mean R_V: {np.mean(control_rvs):.4f}")
    print(f"  Std R_V: {np.std(control_rvs):.4f}")
    print(f"  N: {len(control_rvs)}")
    print(f"  Contraction (<1.0): {sum(1 for rv in control_rvs if rv < 1.0)}/{len(control_rvs)}")
    
    print(f"\nH6-group ablated:")
    print(f"  Mean R_V: {np.mean(h6_rvs):.4f}")
    print(f"  Std R_V: {np.std(h6_rvs):.4f}")
    print(f"  N: {len(h6_rvs)}")
    print(f"  Contraction (<1.0): {sum(1 for rv in h6_rvs if rv < 1.0)}/{len(h6_rvs)}")
    if len(control_rvs) > 0 and len(h6_rvs) > 0:
        delta = np.mean(h6_rvs) - np.mean(control_rvs)
        print(f"  Δ from control: {delta:+.4f}")
    
    print(f"\nH18-group ablated:")
    print(f"  Mean R_V: {np.mean(h18_rvs):.4f}")
    print(f"  Std R_V: {np.std(h18_rvs):.4f}")
    print(f"  N: {len(h18_rvs)}")
    print(f"  Contraction (<1.0): {sum(1 for rv in h18_rvs if rv < 1.0)}/{len(h18_rvs)}")
    if len(control_rvs) > 0 and len(h18_rvs) > 0:
        delta = np.mean(h18_rvs) - np.mean(control_rvs)
        print(f"  Δ from control: {delta:+.4f}")
    
    print(f"\nBoth ablated:")
    print(f"  Mean R_V: {np.mean(both_rvs):.4f}")
    print(f"  Std R_V: {np.std(both_rvs):.4f}")
    print(f"  N: {len(both_rvs)}")
    print(f"  Contraction (<1.0): {sum(1 for rv in both_rvs if rv < 1.0)}/{len(both_rvs)}")
    if len(control_rvs) > 0 and len(both_rvs) > 0:
        delta = np.mean(both_rvs) - np.mean(control_rvs)
        print(f"  Δ from control: {delta:+.4f}")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    control_mean = np.mean(control_rvs) if control_rvs else None
    h6_mean = np.mean(h6_rvs) if h6_rvs else None
    h18_mean = np.mean(h18_rvs) if h18_rvs else None
    both_mean = np.mean(both_rvs) if both_rvs else None
    
    if control_mean is not None:
        print(f"\nControl R_V: {control_mean:.4f}")
        if control_mean < 1.0:
            print("  ✅ Shows contraction (R_V < 1.0)")
        else:
            print("  ⚠️  No contraction (R_V >= 1.0)")
    
    if h6_mean is not None and control_mean is not None:
        h6_delta = h6_mean - control_mean
        print(f"\nH6-group ablated R_V: {h6_mean:.4f} (Δ={h6_delta:+.4f})")
        if h6_delta > 0.05:
            print("  ✅ Prediction B: R_V EXPANDS → H6-group CAUSES contraction")
        elif abs(h6_delta) < 0.05:
            print("  ✅ Prediction A: R_V STAYS CONTRACTED → contraction is upstream")
        else:
            print("  ⚠️  R_V contracts further (unexpected)")
    
    if h18_mean is not None and control_mean is not None:
        h18_delta = h18_mean - control_mean
        print(f"\nH18-group ablated R_V: {h18_mean:.4f} (Δ={h18_delta:+.4f})")
        if h18_delta > 0.05:
            print("  ✅ Prediction B: R_V EXPANDS → H18-group CAUSES contraction")
        elif abs(h18_delta) < 0.05:
            print("  ✅ Prediction A: R_V STAYS CONTRACTED → contraction is upstream")
        else:
            print("  ⚠️  R_V contracts further (unexpected)")
    
    if both_mean is not None and control_mean is not None:
        both_delta = both_mean - control_mean
        print(f"\nBoth ablated R_V: {both_mean:.4f} (Δ={both_delta:+.4f})")
        if both_delta > 0.05:
            print("  ✅ Prediction B: R_V EXPANDS → Suppressor heads CAUSE contraction")
        elif abs(both_delta) < 0.05:
            print("  ✅ Prediction A: R_V STAYS CONTRACTED → contraction is upstream")
        else:
            print("  ⚠️  R_V contracts further (unexpected)")
    
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    for r in results:
        print(f"\nPrompt {r['prompt_idx']+1}: {r['prompt']}...")
        ctrl = r['control'] if r['control'] is not None else float('nan')
        h6 = r['h6_ablated'] if r['h6_ablated'] is not None else float('nan')
        h18 = r['h18_ablated'] if r['h18_ablated'] is not None else float('nan')
        both = r['both_ablated'] if r['both_ablated'] is not None else float('nan')
        print(f"  Control: {ctrl:.4f}")
        print(f"  H6-ablated: {h6:.4f}")
        print(f"  H18-ablated: {h18:.4f}")
        print(f"  Both-ablated: {both:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

