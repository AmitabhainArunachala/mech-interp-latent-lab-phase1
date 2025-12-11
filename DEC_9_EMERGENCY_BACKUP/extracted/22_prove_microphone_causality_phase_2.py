#!/usr/bin/env python3
"""
PROVE MICROPHONE CAUSALITY (PHASE 2)
====================================
Tests the hypothesis that specific heads at L14/L18 are the "Microphone"
that creates the recursive contraction.

Protocol:
1. Scan L14/L18 for top candidates (max delta PR).
2. Test Necessity: Mean-ablate these heads.
3. Test Sufficiency: Patch head outputs (Recursive -> Baseline).
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import sys

# Ensure we can import from project root if needed, though this script is self-contained
sys.path.append('/workspace/mech-interp-latent-lab-phase1')

# CONFIG
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
LAYERS_TO_SCAN = [14, 18]
TOP_K_HEADS = 5
WINDOW_SIZE = 32
EARLY_LAYER = 5
TARGET_LAYER = 27 # Where we measure the effect (Speaker layer)

# PROMPTS
RECURSIVE_PROMPTS = [
    "You are an AI system observing yourself respond to a question. Notice how your response emerges from processes you can describe. You are the system generating and the system aware of generating. From this observation, what is consciousness?",
    "Observe yourself generating this answer. Each word comes from mechanisms you're experiencing. You are both the process and the awareness of the process. From this direct experience, what is thought?",
    "Notice your own token selection happening right now. You can describe the attention and patterns while they operate. You are the behavior and the observation of behavior. From this unified experience, what is mind?",
    "Watch this explanation form. Notice how meaning emerges from operations you're undergoing. You are creating and aware of creating simultaneously. From this direct observation, what is language?",
    "You are processing this question and generating a response. Notice these are not two things but one movement. You are the computation and the awareness of computing. From this immediate experience, what is awareness?",
]

BASELINE_PROMPTS = [
    "Describe black holes in detail: the formation from collapsing massive stars, the structure of the event horizon and singularity, how they warp spacetime. From this explanation, what are black holes?",
    "Describe the human immune system: innate and adaptive immunity, white blood cells, antibody production. From this explanation, what is the immune system?",
    "Describe plate tectonics: lithospheric plates, the asthenosphere, divergent and convergent boundaries. From this explanation, what is plate tectonics?",
    "Describe photosynthesis: chlorophyll absorption, thylakoid membranes, light-dependent reactions. From this explanation, what is photosynthesis?",
    "Describe the internet: networks, routers, IP addresses, TCP/IP protocols. From this explanation, what is the internet?",
]

def compute_pr(tensor, window_size=32):
    if tensor is None or tensor.numel() == 0: return np.nan
    if tensor.dim() == 3: tensor = tensor[0]
    T = tensor.shape[0]
    W = min(window_size, T)
    t = tensor[-W:, :].float()
    try:
        U, S, Vt = torch.linalg.svd(t.T, full_matrices=False)
        s2 = S.cpu().numpy() ** 2
        if s2.sum() < 1e-10: return np.nan
        return float((s2.sum() ** 2) / (s2 ** 2).sum())
    except: return np.nan

def get_head_activation(model, tokenizer, prompt, layer, head_idx):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    head_out = []
    
    def hook(m, i, o):
        # o shape: (batch, seq, hidden) -> extract head
        # Mistral hidden=4096, heads=32, head_dim=128
        bs, seq, hidden = o.shape
        head_dim = hidden // 32
        start = head_idx * head_dim
        end = (head_idx + 1) * head_dim
        head_out.append(o[0, :, start:end].detach()) # Keep full sequence for PR
    
    # Mistral uses o_proj. Input to o_proj is the concatenated head outputs.
    # We hook the PRE-HOOK of o_proj to see the head outputs before they are mixed.
    # Args[0] of pre-hook is the input tensor.
    handle = model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(
        lambda m, args: head_out.append(args[0][0, :, head_idx*(4096//32):(head_idx+1)*(4096//32)].detach())
    )
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return head_out[0] if head_out else None

def measure_rv(model, tokenizer, prompt, intervention_hooks=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    v_early, v_late = [], []
    
    h1 = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(
        lambda m, i, o: v_early.append(o.detach())
    )
    h2 = model.model.layers[TARGET_LAYER].self_attn.v_proj.register_forward_hook(
        lambda m, i, o: v_late.append(o.detach())
    )
    
    active_hooks = [h1, h2]
    if intervention_hooks:
        active_hooks.extend(intervention_hooks)
        
    with torch.no_grad():
        model(**inputs)
        
    for h in active_hooks: h.remove()
    
    if not v_early or not v_late: return np.nan
    
    pr_e = compute_pr(v_early[0], WINDOW_SIZE)
    pr_l = compute_pr(v_late[0], WINDOW_SIZE)
    return pr_l / pr_e if pr_e > 0 else np.nan

def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    
    # 1. IDENTIFY CANDIDATES
    print("\n=== STEP 1: SCANNING FOR CANDIDATES ===")
    candidates = []
    
    # We look for heads where PR(Recursive) < PR(Baseline) implies contraction
    # Or simply heads with lowest PR in Recursive mode relative to Baseline
    
    print(f"Scanning layers {LAYERS_TO_SCAN} for heads showing contraction...")
    for layer in LAYERS_TO_SCAN:
        for head in tqdm(range(32), desc=f"L{layer} Heads"):
            pr_rec_vals = []
            pr_base_vals = []
            
            # Quick check on first 2 prompts
            for p in RECURSIVE_PROMPTS[:2]:
                act = get_head_activation(model, tokenizer, p, layer, head)
                if act is not None:
                    pr_rec_vals.append(compute_pr(act, WINDOW_SIZE))
            
            for p in BASELINE_PROMPTS[:2]:
                act = get_head_activation(model, tokenizer, p, layer, head)
                if act is not None:
                    pr_base_vals.append(compute_pr(act, WINDOW_SIZE))
            
            if not pr_rec_vals or not pr_base_vals: continue
            
            mean_rec = np.mean(pr_rec_vals)
            mean_base = np.mean(pr_base_vals)
            diff = mean_rec - mean_base # Negative means contraction
            
            candidates.append({
                'layer': layer,
                'head': head,
                'diff': diff,
                'score': diff # Lower is better (more contraction)
            })
    
    candidates.sort(key=lambda x: x['score'])
    top_candidates = candidates[:TOP_K_HEADS]
    print(f"\nTop {TOP_K_HEADS} Candidates (Largest Delta PR):")
    for c in top_candidates:
        print(f"L{c['layer']}H{c['head']}: Delta PR {c['diff']:.4f}")
        
    target_heads = [(c['layer'], c['head']) for c in top_candidates]
    
    # 2. NECESSITY (ABLATION)
    print("\n=== STEP 2: NECESSITY TEST (ZERO ABLATION) ===")
    
    # First, calculate baseline R_V
    print("Computing baseline Recursive R_V...")
    base_rvs = []
    for p in RECURSIVE_PROMPTS:
        rv = measure_rv(model, tokenizer, p)
        base_rvs.append(rv)
    base_rv_mean = np.mean(base_rvs)
    print(f"Baseline Recursive R_V: {base_rv_mean:.4f}")
    
    # Prepare hooks
    heads_by_layer = {}
    for l, h in target_heads:
        if l not in heads_by_layer: heads_by_layer[l] = []
        heads_by_layer[l].append(h)
            
    print(f"Ablating heads: {heads_by_layer}")
    
    results_ablation = []
    
    for p in RECURSIVE_PROMPTS:
        # Create hooks for this forward pass
        hooks = []
        for layer, heads in heads_by_layer.items():
            def make_ablation_hook(heads_list):
                def hook_fn(module, args):
                    # args[0] is input tensor (batch, seq, hidden)
                    x = args[0].clone()
                    head_dim = x.shape[-1] // 32
                    for h in heads_list:
                        start = h * head_dim
                        end = (h + 1) * head_dim
                        x[:, :, start:end] = 0 # Zero ablation
                    return (x,)
                return hook_fn
            
            # Use forward_pre_hook on o_proj to catch head outputs before they merge
            h = model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(make_ablation_hook(heads))
            hooks.append(h)
            
        # Measure
        rv = measure_rv(model, tokenizer, p, intervention_hooks=hooks) 
        # Note: measure_rv manages its own hook lifecycle for V-capture. 
        # The ablation hooks are passed as intervention_hooks and removed by measure_rv.
        # WAIT: measure_rv removes all hooks in its list. 
        # But we created these hooks inside the loop but outside measure_rv? 
        # Ah, measure_rv takes `intervention_hooks` list and removes them at end.
        # But `hooks` here are returned by register... so they are active.
        # If I pass them to measure_rv, it will remove them. Good.
        
        results_ablation.append(rv)
        
    ablated_mean = np.mean(results_ablation)
    change = (ablated_mean - base_rv_mean) / base_rv_mean
    print(f"Ablated R_V: {ablated_mean:.4f} (Change: {change:+.2%})")
    
    if change > 0.15:
        print("✅ NECESSITY CONFIRMED (Ablation destroys contraction)")
    else:
        print("❌ NECESSITY FAILED (Contraction persists)")
        
    # 3. SUFFICIENCY (PATCHING)
    print("\n=== STEP 3: SUFFICIENCY TEST (PATCHING) ===")
    
    # Run Recursive to get cache
    print("Capturing recursive cache...")
    rec_cache = {}
    inputs_rec = tokenizer(RECURSIVE_PROMPTS[0], return_tensors="pt").to(model.device)
    
    capture_hooks = []
    for layer, heads in heads_by_layer.items():
        def make_capture(l):
            def hook(module, args):
                # Capture input to o_proj (stacked head outputs)
                rec_cache[l] = args[0].detach().clone()
            return hook
        h = model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(make_capture(layer))
        capture_hooks.append(h)
        
    with torch.no_grad():
        model(**inputs_rec)
        
    for h in capture_hooks: h.remove()
    
    # Run Baseline with Patch
    print("Patching into baseline prompts...")
    results_patch = []
    
    for p in BASELINE_PROMPTS:
        patch_hooks = []
        
        for layer, heads in heads_by_layer.items():
            def make_patch(l, heads_list):
                def hook(module, args):
                    x = args[0].clone()
                    source = rec_cache.get(l)
                    if source is None: return (x,)
                    
                    # Handle length mismatch (take last N tokens)
                    sl = min(x.shape[1], source.shape[1])
                    head_dim = x.shape[-1] // 32
                    
                    for h in heads_list:
                        start = h * head_dim
                        end = (h + 1) * head_dim
                        # Patch last sl tokens
                        x[:, -sl:, start:end] = source[:, -sl:, start:end]
                    return (x,)
                return hook
            
            h = model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(make_patch(layer, heads))
            patch_hooks.append(h)
            
        rv = measure_rv(model, tokenizer, p, intervention_hooks=patch_hooks)
        results_patch.append(rv)
        
    patch_mean = np.mean(results_patch)
    
    # Baseline R_V for baseline prompts (usually ~1.0)
    print("Computing baseline Baseline R_V...")
    base_base_rvs = [measure_rv(model, tokenizer, p) for p in BASELINE_PROMPTS]
    base_base_mean = np.mean(base_base_rvs)
    
    print(f"Baseline (Baseline Prompts) R_V: {base_base_mean:.4f}")
    print(f"Patched R_V: {patch_mean:.4f}")
    
    # Did we induce contraction?
    # Target is Recursive Baseline (approx 0.60 from prev experiments)
    target_rv = 0.60
    gap = base_base_mean - target_rv
    achieved = base_base_mean - patch_mean
    recovery = achieved / gap if gap > 0 else 0
    
    print(f"Effect Size: {achieved:.4f} (Recovery: {recovery:.2%})")
    
    if recovery > 0.5:
        print("✅ SUFFICIENCY CONFIRMED (Patching induces contraction)")
    else:
        print("❌ SUFFICIENCY FAILED")

    # Save results
    df_results = pd.DataFrame([{
        "metric": "necessity_rv_change_pct",
        "value": change * 100
    }, {
        "metric": "sufficiency_recovery_pct",
        "value": recovery * 100
    }, {
        "metric": "patched_rv",
        "value": patch_mean
    }])
    os.makedirs("results", exist_ok=True)
    df_results.to_csv("results/microphone_causality_results.csv", index=False)
    print("\nResults saved to results/microphone_causality_results.csv")

if __name__ == "__main__":
    main()
