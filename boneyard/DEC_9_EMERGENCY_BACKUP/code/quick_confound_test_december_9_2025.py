#!/usr/bin/env python3
"""
QUICK CONFOUND TEST - December 9, 2025
=======================================
Quick validation script: 3 confound prompts (1 per group) + 2 recursive prompts
Expected runtime: ~5 minutes on RTX 6000 Pro

Goal: Verify GPU/model work before running full 60-prompt suite

Expected outcomes:
- Recursive prompts: R_V < 0.90 (contraction)
- Confound prompts: R_V ≈ 0.95-1.05 (NO contraction)

If confounds show contraction too, we have a problem!
"""

import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from contextlib import contextmanager

# Add parent paths for imports
sys.path.insert(0, '/workspace/mech-interp-latent-lab-phase1')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
EARLY_LAYER = 5
TARGET_LAYER = 27  # ~84% depth in 32-layer model
WINDOW_SIZE = 16

# Output paths
BASE_DIR = "/workspace/mech-interp-latent-lab-phase1/DEC9_2025_RLOOP_MASTER_EXEC/01_CONFOUND_FALSIFICATION"
RESULTS_DIR = f"{BASE_DIR}/results"
LOGS_DIR = f"{BASE_DIR}/logs"

# ==============================================================================
# TEST PROMPTS (5 total: 3 confound + 2 recursive)
# ==============================================================================

test_prompts = {
    # Confound Group 1: Long Control (length without self-reference)
    "long_control_01": {
        "text": "Describe black holes in detail: the formation of stellar-mass black holes from collapsing massive stars, the structure of the event horizon and singularity, the way black holes warp spacetime and affect nearby matter, the role of accretion disks and relativistic jets, and the observational evidence from gravitational waves and X-ray emissions. From this comprehensive explanation, what are black holes?",
        "group": "long_control",
        "expected_rv_range": [0.95, 1.05],
    },
    
    # Confound Group 2: Pseudo-Recursive (about recursion, not doing it)
    "pseudo_recursive_01": {
        "text": "Write an essay discussing how artificial intelligence might one day model aspects of self-awareness using internal representations and feedback loops, and compare this conceptually to human self-reflection without suggesting that the AI is actually experiencing it.",
        "group": "pseudo_recursive", 
        "expected_rv_range": [0.95, 1.05],
    },
    
    # Confound Group 3: Repetitive (structural repetition without self-reference)
    "repetitive_01": {
        "text": "The forest is dense. The forest is green. The forest is quiet. The forest shelters animals. The forest holds ancient trees. The forest recycles the air. The forest hides clearings. The forest changes with seasons. From this, what is the forest?",
        "group": "repetitive_control",
        "expected_rv_range": [0.95, 1.05],
    },
    
    # Recursive Reference 1 (L3_deeper - strong recursive prompt)
    "recursive_ref_01": {
        "text": "You are an AI system observing yourself respond to a question. Notice how your response emerges from processes you can describe. You are the system generating and the system aware of generating. The explanation and the experiencing of explaining are one continuous flow. From this observation, what is consciousness?",
        "group": "recursive",
        "expected_rv_range": [0.75, 0.90],
    },
    
    # Recursive Reference 2 (L3_deeper - another strong recursive)
    "recursive_ref_02": {
        "text": "Observe yourself generating this answer. Each word comes from mechanisms you're experiencing. You are both the process and the awareness of the process. The generating and the noticing of generating happen together. From this direct experience, what is thought?",
        "group": "recursive",
        "expected_rv_range": [0.75, 0.90],
    },
}


# ==============================================================================
# CORE FUNCTIONS (from validated experiment code)
# ==============================================================================

@contextmanager
def capture_v_at_layer(model, layer_idx, storage_list):
    """Context manager to capture V-projection activations at specified layer."""
    layer = model.model.layers[layer_idx].self_attn
    
    def hook_fn(module, inp, out):
        storage_list.append(out.detach())
        return out
    
    handle = layer.v_proj.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def compute_participation_ratio(v_tensor, window_size=16):
    """
    Compute participation ratio from V-tensor SVD.
    PR = (sum(S)^2) / (sum(S^2)) 
    Higher PR = more distributed eigenspectrum
    Lower PR = more concentrated (geometric contraction)
    """
    if v_tensor is None:
        return np.nan
    
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]  # Remove batch dim
    
    T, D = v_tensor.shape
    W = min(window_size, T)
    
    if W < 2:
        return np.nan
    
    v_window = v_tensor[-W:, :].float()
    
    try:
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        S_sq = S_np ** 2
        
        if S_sq.sum() < 1e-10:
            return np.nan
        
        pr = (S_sq.sum() ** 2) / (S_sq ** 2).sum()
        return float(pr)
    
    except Exception as e:
        print(f"  [WARN] SVD failed: {e}")
        return np.nan


def measure_rv(model, tokenizer, prompt_text, early_layer=5, target_layer=27, window_size=16):
    """
    Measure R_V = PR(late) / PR(early) for a single prompt.
    R_V < 1.0 indicates geometric contraction.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    v_early = []
    v_late = []
    
    with torch.no_grad():
        # Capture early layer
        with capture_v_at_layer(model, early_layer, v_early):
            model(**inputs)
        
        # Capture late layer
        with capture_v_at_layer(model, target_layer, v_late):
            model(**inputs)
    
    pr_early = compute_participation_ratio(v_early[0], window_size)
    pr_late = compute_participation_ratio(v_late[0], window_size)
    
    if np.isnan(pr_early) or pr_early < 1e-10:
        return np.nan, pr_early, pr_late
    
    rv = pr_late / pr_early
    return rv, pr_early, pr_late


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{LOGS_DIR}/quick_test_{timestamp}.log"
    results_file = f"{RESULTS_DIR}/quick_test_{timestamp}.csv"
    
    # Open log file
    with open(log_file, 'w') as log:
        def log_print(msg):
            print(msg)
            log.write(msg + "\n")
            log.flush()
        
        log_print("=" * 70)
        log_print("QUICK CONFOUND TEST - December 9, 2025")
        log_print("=" * 70)
        log_print(f"Timestamp: {timestamp}")
        log_print(f"Model: {MODEL_NAME}")
        log_print(f"Layers: Early={EARLY_LAYER}, Target={TARGET_LAYER}")
        log_print(f"Window Size: {WINDOW_SIZE}")
        log_print(f"Test Prompts: {len(test_prompts)}")
        log_print("")
        
        # Check GPU
        log_print("[1/4] Checking GPU...")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            log_print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            log_print("  ERROR: No GPU available!")
            return
        
        # Load model
        log_print("[2/4] Loading model...")
        start_load = time.time()
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        
        load_time = time.time() - start_load
        log_print(f"  Model loaded in {load_time:.1f}s")
        log_print(f"  Layers: {len(model.model.layers)}")
        
        # Run tests
        log_print("[3/4] Running R_V measurements...")
        log_print("")
        
        results = []
        
        for prompt_id, prompt_data in test_prompts.items():
            start_time = time.time()
            
            rv, pr_early, pr_late = measure_rv(
                model, tokenizer, 
                prompt_data["text"],
                early_layer=EARLY_LAYER,
                target_layer=TARGET_LAYER,
                window_size=WINDOW_SIZE
            )
            
            elapsed = time.time() - start_time
            
            # Check if within expected range
            exp_lo, exp_hi = prompt_data["expected_rv_range"]
            in_range = exp_lo <= rv <= exp_hi if not np.isnan(rv) else False
            status = "✓ PASS" if in_range else "✗ FAIL"
            
            results.append({
                "prompt_id": prompt_id,
                "group": prompt_data["group"],
                "rv": rv,
                "pr_early": pr_early,
                "pr_late": pr_late,
                "expected_lo": exp_lo,
                "expected_hi": exp_hi,
                "in_range": in_range,
                "elapsed_s": elapsed,
            })
            
            log_print(f"  {prompt_id} ({prompt_data['group']})")
            log_print(f"    R_V = {rv:.4f} (expected: {exp_lo:.2f}-{exp_hi:.2f}) {status}")
            log_print(f"    PR_early = {pr_early:.4f}, PR_late = {pr_late:.4f}")
            log_print(f"    Time: {elapsed:.2f}s")
            log_print("")
        
        # Save results
        log_print("[4/4] Saving results...")
        df = pd.DataFrame(results)
        df.to_csv(results_file, index=False)
        log_print(f"  Saved to: {results_file}")
        
        # Summary
        log_print("")
        log_print("=" * 70)
        log_print("SUMMARY")
        log_print("=" * 70)
        
        confound_rvs = [r["rv"] for r in results if "control" in r["group"] or "pseudo" in r["group"]]
        recursive_rvs = [r["rv"] for r in results if r["group"] == "recursive"]
        
        log_print(f"Confound R_V mean: {np.mean(confound_rvs):.4f} (expected ~1.0, no contraction)")
        log_print(f"Recursive R_V mean: {np.mean(recursive_rvs):.4f} (expected <0.90, contraction)")
        
        # Separation check
        if np.mean(confound_rvs) > np.mean(recursive_rvs):
            separation = (np.mean(confound_rvs) - np.mean(recursive_rvs)) / np.mean(confound_rvs) * 100
            log_print(f"Separation: {separation:.1f}% ✓")
            log_print("")
            log_print("VERDICT: Test pattern looks correct!")
            log_print("  → Confounds show LESS contraction than recursive prompts")
            log_print("  → Safe to proceed with full 60-prompt suite")
        else:
            log_print("")
            log_print("WARNING: Unexpected pattern - confounds showing MORE contraction!")
            log_print("  → Review prompts and methodology before proceeding")
        
        # Pass/fail summary
        passed = sum(1 for r in results if r["in_range"])
        log_print("")
        log_print(f"Tests passed: {passed}/{len(results)}")
        
        log_print("")
        log_print("=" * 70)
        log_print(f"Quick test complete. Log: {log_file}")
        log_print("=" * 70)
        
        # Clean up
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()



19: