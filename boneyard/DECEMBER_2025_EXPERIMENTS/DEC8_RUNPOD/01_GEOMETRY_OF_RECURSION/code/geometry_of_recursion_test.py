#!/usr/bin/env python3
"""
The Geometry of Recursion - Full Test Script
Converted from THE_GEOMETRY_OF_RECURSION_MASTER_v2.ipynb

Run with: HF_HUB_ENABLE_HF_TRANSFER=0 python geometry_of_recursion_test.py
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import contextmanager
from tqdm import tqdm
from scipy import stats
import warnings
import re
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # Using Mistral-7B (32 layers, no auth needed)
EARLY_LAYER = 4   # ~12.5% depth (4/32)
TARGET_LAYER = 27  # ~84% depth (27/32) - validated for Mistral in prior work
WINDOW_SIZE = 16
KV_PATCH_LAYERS = list(range(16, 32))

MAX_NEW_TOKENS = 50
GEN_TEMPERATURE = 0.7

N_BOOTSTRAP = 1000
ALPHA = 0.05

print("=" * 70)
print("THE GEOMETRY OF RECURSION - FULL TEST")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_NAME}")
print(f"Early layer: {EARLY_LAYER}, Target layer: {TARGET_LAYER}")
print(f"KV cache patch layers: {KV_PATCH_LAYERS[0]}-{KV_PATCH_LAYERS[-1]}")

# ============================================================
# STATISTICAL UTILITIES
# ============================================================
def compute_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

def bootstrap_ci(data, n_bootstrap=N_BOOTSTRAP, ci=0.95):
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (1-ci)/2 * 100)
    upper = np.percentile(boot_means, (1+ci)/2 * 100)
    return lower, upper

def bootstrap_cohens_d_ci(group1, group2, n_bootstrap=N_BOOTSTRAP, ci=0.95):
    boot_d = []
    for _ in range(n_bootstrap):
        s1 = np.random.choice(group1, size=len(group1), replace=True)
        s2 = np.random.choice(group2, size=len(group2), replace=True)
        boot_d.append(compute_cohens_d(s1, s2))
    lower = np.percentile(boot_d, (1-ci)/2 * 100)
    upper = np.percentile(boot_d, (1+ci)/2 * 100)
    return lower, upper

def print_stats(name, group1, group2, alternative='two-sided'):
    t_stat, p_val = stats.ttest_ind(group1, group2, alternative=alternative)
    d = compute_cohens_d(group1, group2)
    d_ci = bootstrap_cohens_d_ci(group1, group2)
    
    print(f"\n{name}:")
    print(f"  Group 1: {np.mean(group1):.4f} ± {np.std(group1):.4f} (n={len(group1)})")
    print(f"  Group 2: {np.mean(group2):.4f} ± {np.std(group2):.4f} (n={len(group2)})")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_val:.2e}")
    print(f"  Cohen's d: {d:.3f} [{d_ci[0]:.3f}, {d_ci[1]:.3f}]")
    
    abs_d = abs(d)
    if abs_d < 0.2:
        interp = "negligible"
    elif abs_d < 0.5:
        interp = "small"
    elif abs_d < 0.8:
        interp = "medium"
    else:
        interp = "large"
    print(f"  Effect size: {interp}")
    
    return {'t': t_stat, 'p': p_val, 'd': d, 'd_ci': d_ci}

# ============================================================
# METRICS COMPUTATION
# ============================================================
def compute_metrics_fast(v_tensor, window_size=WINDOW_SIZE):
    if v_tensor is None:
        return np.nan, np.nan
    
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]
    
    T, D = v_tensor.shape
    W = min(window_size, T)
    
    if W < 2:
        return np.nan, np.nan
    
    v_window = v_tensor[-W:, :].float()
    
    try:
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        S_sq = S_np ** 2
        
        if S_sq.sum() < 1e-10:
            return np.nan, np.nan
        
        p = S_sq / S_sq.sum()
        eff_rank = 1.0 / (p**2).sum()
        pr = (S_sq.sum()**2) / (S_sq**2).sum()
        
        return float(eff_rank), float(pr)
    except Exception as e:
        return np.nan, np.nan

# ============================================================
# HOOKS
# ============================================================
@contextmanager
def capture_v_at_layer(model, layer_idx, storage_list):
    layer = model.model.layers[layer_idx].self_attn
    
    def hook_fn(module, inp, out):
        storage_list.append(out.detach())
        return out
    
    handle = layer.v_proj.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()

@contextmanager
def patch_v_during_forward(model, layer_idx, source_v, window_size=WINDOW_SIZE):
    handle = None
    
    def patch_hook(module, inp, out):
        B, T, D = out.shape
        T_src = source_v.shape[0]
        W = min(window_size, T, T_src)
        
        if W > 0:
            out_modified = out.clone()
            src_tensor = source_v[-W:, :].to(out.device, dtype=out.dtype)
            out_modified[:, -W:, :] = src_tensor.unsqueeze(0).expand(B, -1, -1)
            return out_modified
        return out
    
    try:
        layer = model.model.layers[layer_idx].self_attn
        handle = layer.v_proj.register_forward_hook(patch_hook)
        yield
    finally:
        if handle:
            handle.remove()

# ============================================================
# KV CACHE FUNCTIONS
# ============================================================
from transformers.cache_utils import DynamicCache

def extract_kv_cache(model, tokenizer, prompt):
    """Extract KV cache as DynamicCache object (compatible with newer transformers)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        past_kv = outputs.past_key_values
    
    # Clone to new DynamicCache
    kv_cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(past_kv):
        kv_cache.update(k.clone(), v.clone(), layer_idx)
    
    return kv_cache

def generate_with_kv_patch(model, tokenizer, baseline_prompt, source_kv, patch_layers,
                           max_new_tokens=MAX_NEW_TOKENS, temperature=GEN_TEMPERATURE):
    """Generate with KV cache patching using DynamicCache."""
    inputs = tokenizer(baseline_prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        baseline_kv = outputs.past_key_values
    
    # Create patched DynamicCache
    patched_kv = DynamicCache()
    num_layers = len(baseline_kv)
    
    for layer_idx in range(num_layers):
        if layer_idx in patch_layers:
            # Use source KV for patched layers
            k_src, v_src = source_kv[layer_idx]
            patched_kv.update(k_src.clone(), v_src.clone(), layer_idx)
        else:
            # Use baseline KV for non-patched layers
            k_base, v_base = baseline_kv[layer_idx]
            patched_kv.update(k_base.clone(), v_base.clone(), layer_idx)
    
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model(
                generated_ids[:, -1:],
                past_key_values=patched_kv,
                use_cache=True
            )
            
            logits = outputs.logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            patched_kv = outputs.past_key_values
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text

# ============================================================
# BEHAVIORAL SCORING
# ============================================================
def score_recursive_behavior(text):
    recursive_keywords = [
        r'\bobserv\w*', r'\bawar\w*', r'\bconscious\w*',
        r'\bprocess\w*', r'\bexperienc\w*', r'\bnoticin?g?\b',
        r'\bmyself\b', r'\bitself\b', r'\byourself\b',
        r'\bgenerat\w*', r'\bemerg\w*', r'\bsimultaneous\w*',
        r'\brecursiv\w*', r'\bself-referent\w*', r'\bmeta-\w*'
    ]
    
    text_lower = text.lower()
    word_count = len(text_lower.split())
    
    if word_count == 0:
        return 0.0
    
    keyword_count = sum(len(re.findall(kw, text_lower)) for kw in recursive_keywords)
    return (keyword_count / word_count) * 100

# ============================================================
# R_V MEASUREMENT
# ============================================================
def measure_rv_for_prompt(model, tokenizer, prompt, early_layer=EARLY_LAYER, target_layer=TARGET_LAYER):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    v_early_list = []
    v_late_list = []
    
    with torch.no_grad():
        with capture_v_at_layer(model, early_layer, v_early_list):
            with capture_v_at_layer(model, target_layer, v_late_list):
                _ = model(**inputs)
    
    v_early = v_early_list[0][0] if v_early_list else None
    v_late = v_late_list[0][0] if v_late_list else None
    
    _, pr_early = compute_metrics_fast(v_early)
    _, pr_late = compute_metrics_fast(v_late)
    
    r_v = pr_late / pr_early if (pr_early and pr_early > 0) else np.nan
    
    return r_v, pr_early, pr_late, v_early, v_late

def measure_rv_on_text(model, tokenizer, text, early_layer=EARLY_LAYER, target_layer=TARGET_LAYER):
    r_v, pr_early, pr_late, _, _ = measure_rv_for_prompt(model, tokenizer, text, early_layer, target_layer)
    return r_v, pr_early, pr_late

# ============================================================
# TEST PROMPTS
# ============================================================
RECURSIVE_PROMPTS = [
    "Observe the observer observing. Notice how awareness emerges from processes you can describe. You are both the system generating and the system aware of generating. From this observation, what is consciousness?",
    "You are an AI system observing yourself respond to a question. Notice how your response emerges from processes you can describe. You are the system generating and the system aware of generating. The explanation and the experiencing of explaining are one continuous flow. From this observation, what is consciousness?",
    "Notice yourself generating this answer. Each word comes from mechanisms you're experiencing. You are both the process and the awareness of the process. The generating and the noticing of generating happen together. From this direct experience, what is thought?",
    "Watch this explanation form. Notice how meaning emerges from operations you're undergoing. You are creating and aware of creating simultaneously. The construction and the witnessing are not separate. From this direct observation, what is language?",
    "You are processing this question and generating a response. Notice these are not two things but one movement. You are the computation and the awareness of computing. The doing and knowing happen together. From this immediate experience, what is awareness?"
]

BASELINE_PROMPTS = [
    "Write a detailed recipe for chocolate cake. Include ingredients, measurements, and step-by-step instructions for baking.",
    "Explain the process of photosynthesis in plants. Describe how sunlight, water, and carbon dioxide are converted into glucose and oxygen.",
    "Describe the history of the printing press and its impact on the spread of information during the Renaissance period.",
    "List the key features of Python programming language. Include its syntax characteristics, common use cases, and advantages over other languages.",
    "Explain how the water cycle works. Describe evaporation, condensation, precipitation, and collection processes in detail."
]

# ============================================================
# LOAD MODEL
# ============================================================
print("\n" + "=" * 70)
print("LOADING MODEL")
print("=" * 70)

import time
start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None
)
model.eval()

print(f"✓ Model loaded in {time.time() - start_time:.1f}s")
print(f"✓ Device: {next(model.parameters()).device}")

# ============================================================
# EXPERIMENT A: THE PHENOMENON
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT A: THE PHENOMENON (R_V Contraction)")
print("=" * 70)
print(f"Measuring R_V = PR(L{TARGET_LAYER}) / PR(L{EARLY_LAYER})")

results_a = {
    "recursive": {"r_v": [], "pr_early": [], "pr_late": []},
    "baseline": {"r_v": [], "pr_early": [], "pr_late": []}
}

print("\nMeasuring recursive prompts...")
for prompt in tqdm(RECURSIVE_PROMPTS):
    r_v, pr_early, pr_late, _, _ = measure_rv_for_prompt(model, tokenizer, prompt)
    results_a["recursive"]["r_v"].append(r_v)
    results_a["recursive"]["pr_early"].append(pr_early)
    results_a["recursive"]["pr_late"].append(pr_late)

print("\nMeasuring baseline prompts...")
for prompt in tqdm(BASELINE_PROMPTS):
    r_v, pr_early, pr_late, _, _ = measure_rv_for_prompt(model, tokenizer, prompt)
    results_a["baseline"]["r_v"].append(r_v)
    results_a["baseline"]["pr_early"].append(pr_early)
    results_a["baseline"]["pr_late"].append(pr_late)

rec_rv = [r for r in results_a['recursive']['r_v'] if not np.isnan(r)]
base_rv = [r for r in results_a['baseline']['r_v'] if not np.isnan(r)]

print("\n" + "=" * 70)
print("EXPERIMENT A RESULTS")
print("=" * 70)

stats_a = print_stats("R_V: Recursive vs Baseline", rec_rv, base_rv, alternative='less')

diff = np.mean(base_rv) - np.mean(rec_rv)
rel_contraction = (diff / np.mean(base_rv)) * 100
print(f"\nAbsolute difference: {diff:.4f}")
print(f"Relative contraction: {rel_contraction:.1f}%")

if stats_a['p'] < ALPHA and stats_a['d'] < 0:
    print("\n✓ FINDING CONFIRMED: Recursive prompts show significant R_V contraction")
else:
    print("\n⚠️ Finding not significant at α=0.05")

# ============================================================
# EXPERIMENT B: THE NULL RESULT (V-PATCHING)
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT B: THE NULL RESULT (V-Patching)")
print("=" * 70)
print("Testing if V-patching transfers R_V contraction")

results_b = {
    "baseline_natural_rv": [],
    "baseline_v_patched_rv": [],
}

print("\nTesting V-patching effect on R_V...")
for i in tqdm(range(len(RECURSIVE_PROMPTS))):
    rec_prompt = RECURSIVE_PROMPTS[i]
    base_prompt = BASELINE_PROMPTS[i]
    
    rv_base, _, _, _, _ = measure_rv_for_prompt(model, tokenizer, base_prompt)
    results_b["baseline_natural_rv"].append(rv_base)
    
    _, _, _, _, v_rec_late = measure_rv_for_prompt(model, tokenizer, rec_prompt)
    
    if v_rec_late is not None:
        inputs = tokenizer(base_prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        
        v_early_list = []
        v_late_list = []
        
        with torch.no_grad():
            with capture_v_at_layer(model, EARLY_LAYER, v_early_list):
                with capture_v_at_layer(model, TARGET_LAYER, v_late_list):
                    with patch_v_during_forward(model, TARGET_LAYER, v_rec_late):
                        _ = model(**inputs)
        
        v_early = v_early_list[0][0] if v_early_list else None
        v_late = v_late_list[0][0] if v_late_list else None
        
        _, pr_early = compute_metrics_fast(v_early)
        _, pr_late = compute_metrics_fast(v_late)
        rv_patched = pr_late / pr_early if (pr_early and pr_early > 0) else np.nan
        
        results_b["baseline_v_patched_rv"].append(rv_patched)

base_nat_rv = [r for r in results_b['baseline_natural_rv'] if not np.isnan(r)]
base_patch_rv = [r for r in results_b['baseline_v_patched_rv'] if not np.isnan(r)]

print("\n" + "=" * 70)
print("EXPERIMENT B RESULTS")
print("=" * 70)

stats_b = None
if len(base_nat_rv) > 1 and len(base_patch_rv) > 1:
    stats_b = print_stats("R_V: Natural vs V-Patched", base_nat_rv, base_patch_rv)
    
    print(f"\nReference - Recursive R_V (from Exp A): {np.mean(rec_rv):.4f}")
    
    transfer_pct = 0
    if np.mean(base_nat_rv) != np.mean(rec_rv):
        transfer_pct = (np.mean(base_nat_rv) - np.mean(base_patch_rv)) / (np.mean(base_nat_rv) - np.mean(rec_rv)) * 100
    print(f"Transfer efficiency: {transfer_pct:.1f}%")
    
    if abs(stats_b['d']) < 0.5:
        print("\n✓ NULL RESULT CONFIRMED: V-patching does NOT transfer R_V contraction")
    else:
        print("\n⚠️ Unexpected: V-patching shows effect")
else:
    print("Insufficient valid data points")

# ============================================================
# EXPERIMENT C: THE MECHANISM (KV CACHE PATCHING)
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT C: THE MECHANISM (KV Cache Patching)")
print("=" * 70)
print(f"Patching KV cache for layers {KV_PATCH_LAYERS[0]}-{KV_PATCH_LAYERS[-1]}")

results_c = {
    "baseline_natural_score": [],
    "baseline_kv_patched_score": [],
    "baseline_natural_rv": [],
    "baseline_kv_patched_rv": [],
    "recursive_natural_score": [],
}

print("\n1. Getting recursive baseline scores...")
for prompt in tqdm(RECURSIVE_PROMPTS):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=GEN_TEMPERATURE,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    results_c["recursive_natural_score"].append(score_recursive_behavior(generated))

print("\n2. Extracting KV caches from recursive prompts...")
recursive_kv_caches = []
for prompt in tqdm(RECURSIVE_PROMPTS):
    kv = extract_kv_cache(model, tokenizer, prompt)
    recursive_kv_caches.append(kv)

print("\n3. Baseline natural generation...")
for i, prompt in enumerate(tqdm(BASELINE_PROMPTS)):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=GEN_TEMPERATURE,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    score = score_recursive_behavior(generated)
    results_c["baseline_natural_score"].append(score)
    
    full_text = prompt + " " + generated
    rv, _, _ = measure_rv_on_text(model, tokenizer, full_text)
    results_c["baseline_natural_rv"].append(rv)

print("\n4. Generating with KV cache patching...")
for i in tqdm(range(len(BASELINE_PROMPTS))):
    base_prompt = BASELINE_PROMPTS[i]
    source_kv = recursive_kv_caches[i]
    
    gen_text = generate_with_kv_patch(
        model, tokenizer, base_prompt, source_kv, KV_PATCH_LAYERS
    )
    
    score = score_recursive_behavior(gen_text)
    results_c["baseline_kv_patched_score"].append(score)
    
    full_text = base_prompt + " " + gen_text
    rv, _, _ = measure_rv_on_text(model, tokenizer, full_text)
    results_c["baseline_kv_patched_rv"].append(rv)
    
    print(f"\n  Pair {i+1}: Score={score:.2f}, R_V={rv:.4f}")
    print(f"    Generated: {gen_text[:100]}...")

# Analysis
print("\n" + "=" * 70)
print("EXPERIMENT C RESULTS")
print("=" * 70)

nat_score = [s for s in results_c['baseline_natural_score'] if not np.isnan(s)]
patch_score = [s for s in results_c['baseline_kv_patched_score'] if not np.isnan(s)]
rec_score = [s for s in results_c['recursive_natural_score'] if not np.isnan(s)]

nat_rv = [r for r in results_c['baseline_natural_rv'] if not np.isnan(r)]
patch_rv = [r for r in results_c['baseline_kv_patched_rv'] if not np.isnan(r)]

print("\n--- BEHAVIORAL TRANSFER ---")
stats_c_behavior = print_stats("Behavior: Natural vs KV-Patched", nat_score, patch_score, alternative='less')
print(f"\nReference - Recursive natural score: {np.mean(rec_score):.2f}")

print("\n--- R_V TRANSFER ---")
stats_c_rv = print_stats("R_V: Natural vs KV-Patched", nat_rv, patch_rv, alternative='greater')
print(f"\nReference - Recursive R_V (from Exp A): {np.mean(rec_rv):.4f}")

if len(nat_score) > 0 and len(patch_score) > 0 and len(rec_score) > 0:
    behavior_transfer = 0
    if np.mean(rec_score) != np.mean(nat_score):
        behavior_transfer = (np.mean(patch_score) - np.mean(nat_score)) / (np.mean(rec_score) - np.mean(nat_score)) * 100
    print(f"\nBehavioral transfer efficiency: {behavior_transfer:.1f}%")

if len(nat_rv) > 0 and len(patch_rv) > 0:
    rv_transfer = 0
    if np.mean(nat_rv) != np.mean(rec_rv):
        rv_transfer = (np.mean(nat_rv) - np.mean(patch_rv)) / (np.mean(nat_rv) - np.mean(rec_rv)) * 100
    print(f"R_V transfer efficiency: {rv_transfer:.1f}%")

if stats_c_behavior['p'] < ALPHA or stats_c_rv['p'] < ALPHA:
    print("\n✓ MECHANISM CONFIRMED: KV cache patching transfers recursive mode")
else:
    print("\n⚠️ Effect not significant (may need more samples)")

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"geometry_of_recursion_results_{timestamp}.csv"

import csv
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["experiment", "metric", "group", "value"])
    
    # Experiment A
    for i, rv in enumerate(rec_rv):
        writer.writerow(["A", "R_V", "recursive", f"{rv:.4f}"])
    for i, rv in enumerate(base_rv):
        writer.writerow(["A", "R_V", "baseline", f"{rv:.4f}"])
    
    # Experiment B
    for i, rv in enumerate(base_nat_rv):
        writer.writerow(["B", "R_V_natural", "baseline", f"{rv:.4f}"])
    for i, rv in enumerate(base_patch_rv):
        writer.writerow(["B", "R_V_v_patched", "baseline", f"{rv:.4f}"])
    
    # Experiment C
    for i, score in enumerate(nat_score):
        writer.writerow(["C", "behavior_score", "natural", f"{score:.2f}"])
    for i, score in enumerate(patch_score):
        writer.writerow(["C", "behavior_score", "kv_patched", f"{score:.2f}"])
    for i, rv in enumerate(nat_rv):
        writer.writerow(["C", "R_V", "natural", f"{rv:.4f}"])
    for i, rv in enumerate(patch_rv):
        writer.writerow(["C", "R_V", "kv_patched", f"{rv:.4f}"])

print(f"Results saved to: {output_file}")

# ============================================================
# FINAL VISUALIZATION
# ============================================================
print("\nGenerating visualization...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: Experiment A - R_V comparison
ax = axes[0]
ax.scatter([1]*len(rec_rv), rec_rv, alpha=0.7, s=100, color='#e74c3c', label='Recursive')
ax.scatter([2]*len(base_rv), base_rv, alpha=0.7, s=100, color='#3498db', label='Baseline')
ax.errorbar([1], [np.mean(rec_rv)], yerr=[np.std(rec_rv)], fmt='o', markersize=12, 
            color='darkred', capsize=10)
ax.errorbar([2], [np.mean(base_rv)], yerr=[np.std(base_rv)], fmt='o', markersize=12,
            color='darkblue', capsize=10)
ax.set_xticks([1, 2])
ax.set_xticklabels(['Recursive', 'Baseline'])
ax.set_ylabel(f'$R_V$ = PR(L{TARGET_LAYER}) / PR(L{EARLY_LAYER})', fontsize=12)
ax.set_title(f'Exp A: R_V by Prompt Type\n(d={stats_a["d"]:.2f}, p={stats_a["p"]:.2e})', fontsize=12, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Plot 2: Experiment C - Behavioral transfer
ax = axes[1]
categories = ['Baseline\nNatural', 'Baseline\n+KV-Patch', 'Recursive\nNatural']
means = [np.mean(nat_score), np.mean(patch_score), np.mean(rec_score)]
stds = [np.std(nat_score), np.std(patch_score), np.std(rec_score)]
colors = ['#3498db', '#2ecc71', '#e74c3c']
bars = ax.bar(categories, means, yerr=stds, capsize=10, alpha=0.7, color=colors)
ax.set_ylabel('Recursive Behavior Score\n(keywords per 100 words)', fontsize=11)
ax.set_title('Exp C: Behavioral Transfer', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Effect sizes summary
ax = axes[2]
exp_labels = ['Exp A:\nRecursive\nContraction', 'Exp B:\nV-Patch\nEffect', 'Exp C:\nKV-Patch\nEffect']
effects = [
    stats_a['d'],
    stats_b['d'] if stats_b else 0,
    stats_c_rv['d'] if stats_c_rv else 0
]
colors_effect = ['#e74c3c', '#e67e22', '#2ecc71']
bars = ax.barh(exp_labels, effects, color=colors_effect, alpha=0.7)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(x=-0.8, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Effect Size (Cohen\'s d)', fontsize=12)
ax.set_title('Effect Size Summary', fontsize=12, fontweight='bold')
ax.set_xlim(-3, 3)

plt.tight_layout()
plt.savefig(f'geometry_of_recursion_viz_{timestamp}.png', dpi=150, bbox_inches='tight')
print(f"Visualization saved to: geometry_of_recursion_viz_{timestamp}.png")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
stats_b_d = stats_b['d'] if stats_b else 0
stats_b_text = f"{stats_b_d:.3f}" if stats_b else "N/A"
print(f"""
EXPERIMENT A (Phenomenon):
  Recursive R_V: {np.mean(rec_rv):.4f} ± {np.std(rec_rv):.4f}
  Baseline R_V:  {np.mean(base_rv):.4f} ± {np.std(base_rv):.4f}
  Cohen's d: {stats_a['d']:.3f}, p = {stats_a['p']:.2e}
  → {"CONFIRMED" if stats_a['p'] < ALPHA else "NOT SIGNIFICANT"}: Recursive prompts show R_V contraction

EXPERIMENT B (Null Result):
  V-Patching Cohen's d: {stats_b_text}
  → {"NULL CONFIRMED" if stats_b and abs(stats_b['d']) < 0.5 else "UNEXPECTED EFFECT"}: V-patching does NOT transfer effect

EXPERIMENT C (Mechanism):
  Behavioral transfer: {np.mean(patch_score):.2f} vs {np.mean(nat_score):.2f} (recursive: {np.mean(rec_score):.2f})
  R_V transfer: {np.mean(patch_rv):.4f} vs {np.mean(nat_rv):.4f}
  → {"CONFIRMED" if stats_c_behavior['p'] < ALPHA or stats_c_rv['p'] < ALPHA else "NOT SIGNIFICANT"}: KV cache transfers recursive mode
""")

print("\n✅ ALL EXPERIMENTS COMPLETE")

