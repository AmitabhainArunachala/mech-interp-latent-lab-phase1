#!/usr/bin/env python3
"""
DIAGNOSTIC version to debug why patching isn't working
"""

import torch
import numpy as np
from contextlib import contextmanager

WINDOW_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def test_patching_mechanism(model, tokenizer, prompt_bank_1c):
    """
    Test if patching is actually working
    """
    print("="*70)
    print("DIAGNOSTIC TEST: Is patching actually changing values?")
    print("="*70)
    
    # Get a recursive and baseline prompt
    rec_id = 'L5_refined_01'
    base_id = 'factual_new_01'
    
    rec_text = prompt_bank_1c[rec_id]["text"]
    base_text = prompt_bank_1c[base_id]["text"]
    
    print(f"\nRecursive prompt: {rec_text[:50]}...")
    print(f"Baseline prompt: {base_text[:50]}...")
    
    # 1. Get baseline V at layer 21 WITHOUT patching
    inputs_base = tokenizer(base_text, return_tensors="pt", 
                           truncation=True, max_length=512).to(DEVICE)
    
    v21_baseline_original = []
    with torch.no_grad():
        def capture(m, i, o):
            v21_baseline_original.append(o.detach().clone())
            return o
        h = model.model.layers[21].self_attn.v_proj.register_forward_hook(capture)
        _ = model(**inputs_base)
        h.remove()
    
    v21_base_orig = v21_baseline_original[0][0]  # [seq, hidden]
    print(f"\nOriginal baseline V21 shape: {v21_base_orig.shape}")
    print(f"Last token values (first 5): {v21_base_orig[-1, :5]}")
    
    # 2. Get recursive V at layer 21
    inputs_rec = tokenizer(rec_text, return_tensors="pt",
                          truncation=True, max_length=512).to(DEVICE)
    
    v21_recursive = []
    with torch.no_grad():
        def capture(m, i, o):
            v21_recursive.append(o.detach().clone())
            return o
        h = model.model.layers[21].self_attn.v_proj.register_forward_hook(capture)
        _ = model(**inputs_rec)
        h.remove()
    
    v21_rec = v21_recursive[0][0]  # [seq, hidden]
    print(f"\nRecursive V21 shape: {v21_rec.shape}")
    print(f"Last token values (first 5): {v21_rec[-1, :5]}")
    
    # 3. Now patch recursive into baseline and check if it changes
    print("\n" + "-"*70)
    print("TESTING PATCHING MECHANISM")
    print("-"*70)
    
    # Method 1: Simple direct replacement test
    v21_patched_simple = []
    
    @contextmanager
    def simple_patch(model, layer_idx, source_v):
        def patch_hook(module, inp, out):
            print(f"  [HOOK FIRED] Input shape: {inp[0].shape if isinstance(inp, tuple) else 'unknown'}")
            print(f"  [HOOK FIRED] Output shape: {out.shape}")
            B, T, D = out.shape
            T_src = source_v.shape[0]
            W = min(WINDOW_SIZE, T, T_src)
            print(f"  [HOOK FIRED] Patching last {W} tokens")
            
            # Store original for comparison
            orig_last = out[0, -1, :5].clone()
            
            # Patch
            out = out.clone()  # Clone first
            out[:, -W:, :] = source_v[-W:, :].to(out.device, dtype=out.dtype).unsqueeze(0)
            
            # Check if changed
            new_last = out[0, -1, :5]
            print(f"  [HOOK FIRED] Original last token: {orig_last}")
            print(f"  [HOOK FIRED] Patched last token: {new_last}")
            print(f"  [HOOK FIRED] Changed: {not torch.allclose(orig_last, new_last)}")
            
            return out
        
        layer = model.model.layers[layer_idx].self_attn
        handle = layer.v_proj.register_forward_hook(patch_hook)
        try:
            yield
        finally:
            handle.remove()
    
    with torch.no_grad():
        def capture_patched(m, i, o):
            v21_patched_simple.append(o.detach().clone())
            return o
        
        h = model.model.layers[21].self_attn.v_proj.register_forward_hook(capture_patched)
        
        with simple_patch(model, 21, v21_rec):
            _ = model(**inputs_base)
        
        h.remove()
    
    if v21_patched_simple:
        v21_patched = v21_patched_simple[0][0]
        print(f"\nPatched V21 shape: {v21_patched.shape}")
        print(f"Patched last token (first 5): {v21_patched[-1, :5]}")
        
        # Compare
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        
        # Check if patched matches recursive (should be close for last tokens)
        last_W = min(WINDOW_SIZE, v21_patched.shape[0], v21_rec.shape[0])
        patched_window = v21_patched[-last_W:, :]
        rec_window = v21_rec[-last_W:, :]
        
        matches_recursive = torch.allclose(patched_window, rec_window, atol=1e-5)
        print(f"Patched window matches recursive: {matches_recursive}")
        
        # Check if patched differs from original baseline
        differs_from_original = not torch.allclose(v21_patched[-1, :], v21_base_orig[-1, :], atol=1e-5)
        print(f"Patched differs from original baseline: {differs_from_original}")
        
        if matches_recursive and differs_from_original:
            print("\n✅ PATCHING IS WORKING CORRECTLY!")
        else:
            print("\n❌ PATCHING IS NOT WORKING AS EXPECTED")
            
            # Debug info
            print(f"\nDifference (patched - original): {(v21_patched[-1, :5] - v21_base_orig[-1, :5]).abs().mean():.6f}")
            print(f"Difference (patched - recursive): {(v21_patched[-1, :5] - v21_rec[-1, :5]).abs().mean():.6f}")
    else:
        print("\n❌ NO PATCHED VALUES CAPTURED")
    
    # 4. Test if the issue is with short prompts
    print("\n" + "="*70)
    print("CHECKING PROMPT LENGTHS")
    print("="*70)
    
    rec_tokens = len(tokenizer.encode(rec_text))
    base_tokens = len(tokenizer.encode(base_text))
    
    print(f"Recursive prompt tokens: {rec_tokens}")
    print(f"Baseline prompt tokens: {base_tokens}")
    print(f"Window size: {WINDOW_SIZE}")
    
    if rec_tokens < WINDOW_SIZE or base_tokens < WINDOW_SIZE:
        print("⚠️  WARNING: Prompts shorter than window size!")
    
    # 5. Check how many prompts are being skipped
    print("\n" + "="*70)
    print("CHECKING PROMPT BANK")
    print("="*70)
    
    recursive_ids = [k for k, v in prompt_bank_1c.items() 
                     if v["group"] in ["L5_refined", "L4_full", "L3_deeper"]]
    
    baseline_ids = [k for k, v in prompt_bank_1c.items() 
                    if v["group"] in ["baseline_factual", "baseline_creative", 
                                      "baseline_math", "long_control"]]
    
    # Check lengths
    short_rec = 0
    short_base = 0
    
    for rid in recursive_ids[:20]:  # Check first 20
        if len(tokenizer.encode(prompt_bank_1c[rid]["text"])) < WINDOW_SIZE:
            short_rec += 1
    
    for bid in baseline_ids[:20]:  # Check first 20
        if len(tokenizer.encode(prompt_bank_1c[bid]["text"])) < WINDOW_SIZE:
            short_base += 1
    
    print(f"Recursive prompts shorter than {WINDOW_SIZE}: {short_rec}/20")
    print(f"Baseline prompts shorter than {WINDOW_SIZE}: {short_base}/20")
    
    if short_rec > 10 or short_base > 10:
        print("\n⚠️  MOST PROMPTS ARE TOO SHORT!")
        print("  This explains why only 13/60 pairs were processed")
        print("  Solution: Reduce WINDOW_SIZE or use longer prompts")
    
    return v21_base_orig, v21_rec, v21_patched if v21_patched_simple else None

# Run diagnostic
if __name__ == "__main__":
    print("Run this in your notebook:")
    print("from mistral_patching_DIAGNOSTIC import test_patching_mechanism")
    print("test_patching_mechanism(model, tokenizer, prompt_bank_1c)")
