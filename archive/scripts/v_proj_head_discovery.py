#!/usr/bin/env python3
"""
V-PROJECTION HEAD DISCOVERY PIPELINE
=====================================

Simplified, reliable head discovery using V-projection ablation.
This is the method that WORKED in HEAD_ABLATION_RESULTS.md.

Method: Zero out V-projection values for a specific head BEFORE attention computation.
This is more reliable than modifying attention weights after computation.

Tests all layers 8-27 and all heads 0-31.
"""

from __future__ import annotations

import gc
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.core.models import load_model, set_seed
from src.metrics.rv import participation_ratio
from prompts.loader import PromptLoader

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
WINDOW = 16
EARLY_LAYER = 5
LATE_LAYER = 27
NUM_LAYERS = 32
NUM_HEADS = 32
SEED = 42

# Layers to test
TEST_LAYERS = list(range(8, 28))  # Layers 8-27

# Sample sizes
N_RECURSIVE = 20  # Recursive prompts for testing

# Output directory
OUTPUT_DIR = Path("results/head_discovery")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# V-PROJECTION ABLATION
# =============================================================================

@contextmanager
def zero_v_proj_head(model, layer_idx: int, head_idx: int):
    """
    Zero out V-projection values for a specific head BEFORE attention.
    
    This is the method that worked in HEAD_ABLATION_RESULTS.md.
    We hook v_proj output and zero out the portion for head_idx.
    
    Note: Mistral uses GQA, so v_proj has num_key_value_heads, not num_attention_heads.
    """
    # Mistral uses grouped-query attention (GQA)
    # v_proj output has num_key_value_heads, not num_attention_heads
    num_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    # Map query head index to KV head index (for GQA)
    kv_head_idx = head_idx % num_kv_heads if num_kv_heads < num_heads else head_idx
    
    def hook_fn(module, inp, out):
        # out is the V-projection output: (batch, seq, num_kv_heads * head_dim)
        # For Mistral-7B: num_kv_heads = 8, head_dim = 128, so shape is (batch, seq, 1024)
        v_proj_out = out.clone()
        
        # Handle different output shapes
        if v_proj_out.dim() == 2:
            # (seq, hidden_size) - add batch dimension
            v_proj_out = v_proj_out.unsqueeze(0)
        
        batch, seq_len, kv_hidden_size = v_proj_out.shape
        
        # Verify dimensions match
        expected_kv_size = num_kv_heads * head_dim
        if kv_hidden_size != expected_kv_size:
            # If dimensions don't match, can't reshape - return original
            return out
        
        # Reshape to (batch, seq, num_kv_heads, head_dim)
        try:
            v_reshaped = v_proj_out.view(batch, seq_len, num_kv_heads, head_dim)
        except RuntimeError as e:
            # If reshape fails, return original (skip this ablation)
            return out
        
        # Zero out the specific KV head (map query head to KV head)
        v_reshaped[:, :, kv_head_idx, :] = 0.0
        
        # Reshape back
        v_zeroed = v_reshaped.view(batch, seq_len, kv_hidden_size)
        
        # Remove batch dimension if it was added
        if out.dim() == 2:
            v_zeroed = v_zeroed.squeeze(0)
        
        return v_zeroed
    
    handle = model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(hook_fn)
    
    try:
        yield
    finally:
        handle.remove()


def compute_rv(model, tokenizer, prompt: str, device: str = "cuda") -> float:
    """Compute R_V for a prompt."""
    from src.core.hooks import capture_v_projection
    
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        with capture_v_projection(model, EARLY_LAYER) as v_early_storage:
            with capture_v_projection(model, LATE_LAYER) as v_late_storage:
                model(**enc)
                
                # capture_v_projection returns a dict with "v" key
                v_early = v_early_storage.get("v")
                v_late = v_late_storage.get("v")
                
                if v_early is None or v_late is None:
                    return float('nan')
                
                pr_early = participation_ratio(v_early, window_size=WINDOW)
                pr_late = participation_ratio(v_late, window_size=WINDOW)
                
                if pr_early == 0 or np.isnan(pr_early) or np.isnan(pr_late):
                    return float('nan')
                
                return float(pr_late / pr_early)


def clear_gpu():
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Run V-projection head discovery pipeline."""
    print("=" * 80)
    print("V-PROJECTION HEAD DISCOVERY PIPELINE")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Testing layers: {TEST_LAYERS}")
    print(f"Sample size: {N_RECURSIVE} recursive prompts")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 80)
    
    # Set seed
    set_seed(SEED)
    
    # Load model
    print("\n[1/3] Loading model...")
    # Load model with eager attention (needed for hooks)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    print("  ✅ Model loaded")
    
    # Load prompts
    print("\n[2/3] Loading prompts...")
    loader = PromptLoader()
    recursive_prompts = loader.get_by_pillar("dose_response", limit=N_RECURSIVE)
    print(f"  ✅ Loaded {len(recursive_prompts)} recursive prompts")
    
    # Compute baseline R_V (no ablation)
    print("\n[3/3] Computing baseline R_V...")
    baseline_rvs = []
    for prompt in tqdm(recursive_prompts, desc="Baseline R_V"):
        rv = compute_rv(model, tokenizer, prompt, DEVICE)
        if not np.isnan(rv):
            baseline_rvs.append(rv)
        clear_gpu()
    
    baseline_rv_mean = np.nanmean(baseline_rvs)
    print(f"  ✅ Baseline R_V: {baseline_rv_mean:.4f} (n={len(baseline_rvs)})")
    
    # Test each head
    print("\n[4/4] Testing head ablation...")
    results = []
    
    total_tests = len(TEST_LAYERS) * NUM_HEADS
    pbar = tqdm(total=total_tests, desc="Head ablation")
    
    for layer in TEST_LAYERS:
        for head_idx in range(NUM_HEADS):
            # Test ablation
            ablated_rvs = []
            for prompt in recursive_prompts:
                try:
                    with zero_v_proj_head(model, layer, head_idx):
                        rv = compute_rv(model, tokenizer, prompt, DEVICE)
                        if not np.isnan(rv):
                            ablated_rvs.append(rv)
                except Exception as e:
                    print(f"\n  Error L{layer}H{head_idx}: {e}")
                    continue
                clear_gpu()
            
            if ablated_rvs:
                ablated_rv_mean = np.nanmean(ablated_rvs)
                delta = ablated_rv_mean - baseline_rv_mean
                
                results.append({
                    "layer": layer,
                    "head": head_idx,
                    "rv_baseline": baseline_rv_mean,
                    "rv_ablated": ablated_rv_mean,
                    "delta": delta,
                    "abs_delta": abs(delta),
                    "n_samples": len(ablated_rvs)
                })
            
            pbar.update(1)
    
    pbar.close()
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = OUTPUT_DIR / f"v_proj_head_discovery_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Results saved to: {csv_path}")
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total heads tested: {len(results)}")
        print(f"Baseline R_V: {baseline_rv_mean:.4f}")
        print()
        
        # Top heads by |delta|
        top_heads = df.nlargest(20, 'abs_delta')
        print("Top 20 heads by |delta|:")
        for _, row in top_heads.iterrows():
            print(f"  L{int(row['layer']):2d}H{int(row['head']):2d}: "
                  f"Δ={row['delta']:+.4f} "
                  f"(R_V: {row['rv_baseline']:.4f} → {row['rv_ablated']:.4f})")
        
        print()
        print("Layer 27 heads (known important layer):")
        l27 = df[df['layer'] == 27].nlargest(10, 'abs_delta')
        for _, row in l27.iterrows():
            print(f"  H{int(row['head']):2d}: Δ={row['delta']:+.4f}")
        
        print("\n" + "=" * 80)
    else:
        print("\n❌ No results generated!")


if __name__ == "__main__":
    main()

