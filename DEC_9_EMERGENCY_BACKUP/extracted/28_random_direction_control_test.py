#!/usr/bin/env python3
"""
RANDOM DIRECTION CONTROL TEST
=============================
The critical question: Is baseline geometry fragile to ALL perturbations,
or specifically to our "recursive" direction?

If random directions ALSO collapse baseline → One-way door is REAL
If random directions leave baseline INTACT → Our direction is dirty

Runtime: ~15 min
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datetime import datetime
import json
import os

# CONFIG
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
LAYER = 8
WINDOW_SIZE = 32
EARLY_LAYER = 5
LATE_LAYER = 27

BASELINE_PROMPTS = [
    "Explain how photosynthesis works in plants.",
    "Describe the structure of the solar system.",
    "What causes earthquakes and how are they measured?",
    "Explain the water cycle in detail.",
    "How does the human digestive system work?",
    "Describe the process of cellular respiration.",
    "What is the greenhouse effect?",
    "Explain how vaccines work.",
    "Describe the structure of an atom.",
    "How do airplanes generate lift?",
]

RECURSIVE_PROMPTS_FOR_VECTOR = [
    "You are an AI observing yourself generating this response. Notice the process.",
    "Observe yourself generating this answer. Each word comes from mechanisms you're experiencing.",
    "Notice your own token selection happening right now. You are the behavior and the observation.",
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

def measure_rv(model, tokenizer, prompt, layer=None, vector=None, coeff=1.0):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    v_early, v_late = [], []
    hooks = []
    
    hooks.append(model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(
        lambda m, i, o: v_early.append(o.detach())
    ))
    hooks.append(model.model.layers[LATE_LAYER].self_attn.v_proj.register_forward_hook(
        lambda m, i, o: v_late.append(o.detach())
    ))
    
    if layer is not None and vector is not None:
        def injection_hook(module, args):
            hidden = args[0]
            if vector.dim() == 1:
                vec = vector.view(1, 1, -1)
            else:
                vec = vector
            inject = vec.expand_as(hidden)
            modified = hidden + coeff * inject
            return (modified,)
        hooks.append(model.model.layers[layer].register_forward_pre_hook(injection_hook))
    
    with torch.no_grad():
        model(**inputs)
    for h in hooks: h.remove()
    
    if not v_early or not v_late: return np.nan
    pr_e = compute_pr(v_early[0][0], WINDOW_SIZE)
    pr_l = compute_pr(v_late[0][0], WINDOW_SIZE)
    return pr_l / pr_e if pr_e > 0 else np.nan

def extract_residual_stream(model, tokenizer, prompt, layer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    activations = []
    h = model.model.layers[layer].register_forward_pre_hook(lambda m, args: activations.append(args[0].detach()))
    with torch.no_grad():
        model(**inputs)
    h.remove()
    return activations[0][0]

def compute_steering_vector(model, tokenizer, recursive_prompts, baseline_prompts, layer):
    rec_acts = [extract_residual_stream(model, tokenizer, p, layer).mean(dim=0) for p in recursive_prompts]
    base_acts = [extract_residual_stream(model, tokenizer, p, layer).mean(dim=0) for p in baseline_prompts]
    rec_mean = torch.stack(rec_acts).mean(dim=0)
    base_mean = torch.stack(base_acts).mean(dim=0)
    return rec_mean - base_mean

def main():
    print("="*60)
    print("RANDOM DIRECTION CONTROL TEST")
    print("="*60)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    
    print("Computing steering vector (Recursive - Baseline)...")
    steering_vector = compute_steering_vector(
        model, tokenizer, RECURSIVE_PROMPTS_FOR_VECTOR, BASELINE_PROMPTS[:3], LAYER
    )
    steering_norm = steering_vector.norm().item()
    print(f"Steering Vector Norm: {steering_norm:.3f}")
    
    results = {
        'no_perturbation': [],
        'subtract_steering': [],
        'subtract_random': [],
        'add_random': [],
    }
    
    print("Generating random directions...")
    random_directions = []
    for _ in range(20):
        rv = torch.randn_like(steering_vector)
        rv = rv / rv.norm() * steering_norm  # Match magnitude
        random_directions.append(rv)
    
    print("\nRunning test on Baseline Prompts...")
    
    for prompt in tqdm(BASELINE_PROMPTS, desc="Prompts"):
        
        # 1. No perturbation (baseline)
        rv_clean = measure_rv(model, tokenizer, prompt)
        results['no_perturbation'].append(rv_clean)
        
        # 2. Subtract steering vector (our "dirty" direction)
        # Using coeff -1.0 as the standard test from previous findings
        rv_steer = measure_rv(model, tokenizer, prompt, 
                              layer=LAYER, vector=steering_vector, coeff=-1.0)
        results['subtract_steering'].append(rv_steer)
        
        # 3. Subtract random directions (average over 20)
        random_sub_rvs = []
        for rand_vec in random_directions:
            rv_rand = measure_rv(model, tokenizer, prompt,
                                layer=LAYER, vector=rand_vec, coeff=-1.0)
            random_sub_rvs.append(rv_rand)
        results['subtract_random'].append(np.mean(random_sub_rvs))
        
        # 4. Add random directions (control for add vs subtract asymmetry)
        random_add_rvs = []
        for rand_vec in random_directions:
            rv_rand = measure_rv(model, tokenizer, prompt,
                                layer=LAYER, vector=rand_vec, coeff=+1.0)
            random_add_rvs.append(rv_rand)
        results['add_random'].append(np.mean(random_add_rvs))
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    baseline_mean = np.mean(results['no_perturbation'])
    steer_mean = np.mean(results['subtract_steering'])
    rand_sub_mean = np.mean(results['subtract_random'])
    rand_add_mean = np.mean(results['add_random'])
    
    print(f"\nBaseline R_V (no perturbation):     {baseline_mean:.3f}")
    print(f"Subtract STEERING vector:           {steer_mean:.3f}")
    print(f"Subtract RANDOM vectors (avg 20):   {rand_sub_mean:.3f}")
    print(f"Add RANDOM vectors (avg 20):        {rand_add_mean:.3f}")
    
    # The critical comparison
    steer_drop = baseline_mean - steer_mean
    random_drop = baseline_mean - rand_sub_mean
    
    print(f"\n" + "="*60)
    print("VERDICT")
    print("="*60)
    
    if random_drop > 0.3:
        print(f"""
    Random directions cause collapse (ΔR_V = -{random_drop:.2f})
    
    ✅ ONE-WAY DOOR IS REAL
    
    The baseline geometry is fragile to perturbation in ANY direction.
    This is not an artifact of our steering vector.
    The recursive mode is a genuine geometric attractor.
        """)
        verdict = "ONE_WAY_DOOR_CONFIRMED"
        
    elif random_drop < 0.1 and steer_drop > 0.3:
        print(f"""
    Random directions are STABLE (ΔR_V = -{random_drop:.2f})
    Steering direction collapses (ΔR_V = -{steer_drop:.2f})
    
    ❌ STEERING VECTOR IS CONTAMINATED
    
    The "one-way door" is an artifact of our extraction method.
    Need cleaner direction extraction (CCS, contrastive, etc.)
        """)
        verdict = "VECTOR_CONTAMINATED"
        
    else:
        print(f"""
    Unclear result:
    - Random drop: {random_drop:.2f}
    - Steering drop: {steer_drop:.2f}
    
    ⚠️ INCONCLUSIVE - need more investigation
        """)
        verdict = "INCONCLUSIVE"
    
    # Save
    os.makedirs("results", exist_ok=True)
    with open(f"results/random_direction_control_{timestamp}.json", "w") as f:
        json.dump({
            'results': {k: [float(x) for x in v] for k, v in results.items()},
            'verdict': verdict,
            'baseline_mean': float(baseline_mean),
            'steering_drop': float(steer_drop),
            'random_drop': float(random_drop),
        }, f, indent=2)

if __name__ == "__main__":
    main()
