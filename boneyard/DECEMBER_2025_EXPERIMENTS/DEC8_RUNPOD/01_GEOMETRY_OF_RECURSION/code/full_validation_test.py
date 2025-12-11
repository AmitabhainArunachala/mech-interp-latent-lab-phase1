#!/usr/bin/env python3
"""
Full-Power Validation Test - Addressing Cross-Validation Feedback

Fixes from cross-validation:
1. Sample size: n≥30 (using n300 prompt bank)
2. Cohen's d: Explicitly calculated and reported
3. Bonferroni correction: Applied for multiple comparisons
4. Layer comparison: L22 vs L27
5. Layer sweep: L0-16 vs L16-32 KV cache

Run with: HF_HUB_ENABLE_HF_TRANSFER=0 python full_validation_test.py
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import sys
sys.path.insert(0, '/workspace/mech-interp-phase1')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from contextlib import contextmanager
from tqdm import tqdm
from scipy import stats
import warnings
import csv
from datetime import datetime
warnings.filterwarnings('ignore')

# Import prompt bank
from n300_mistral_test_prompt_bank import prompt_bank_1c

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# Layers to test
EARLY_LAYER = 4
LAYER_22 = 22  # DEC3 optimal
LAYER_27 = 27  # Current hypothesis

# Sample sizes
N_RECURSIVE = 40  # L3_deeper + L4_full 
N_BASELINE = 40   # baseline_factual + baseline_math

# KV Cache layer ranges for sweep
KV_EARLY = list(range(0, 16))   # L0-15
KV_LATE = list(range(16, 32))   # L16-31
KV_FULL = list(range(0, 32))    # All layers

# Statistics
N_BOOTSTRAP = 1000
ALPHA = 0.05
N_COMPARISONS = 4  # For Bonferroni: L22 rec vs base, L27 rec vs base, L22 vs L27 rec, L22 vs L27 base

WINDOW_SIZE = 16
MAX_NEW_TOKENS = 50
GEN_TEMPERATURE = 0.7

print("=" * 70)
print("FULL-POWER VALIDATION TEST")
print("Addressing Cross-Validation Feedback")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_NAME}")
print(f"Layers: L4 (early), L22 (DEC3), L27 (current)")
print(f"Samples: {N_RECURSIVE} recursive, {N_BASELINE} baseline")
print(f"Bonferroni α: {ALPHA}/{N_COMPARISONS} = {ALPHA/N_COMPARISONS:.4f}")

# ============================================================
# EXTRACT PROMPTS FROM BANK
# ============================================================
print("\n" + "=" * 70)
print("EXTRACTING PROMPTS")
print("=" * 70)

# Get recursive prompts (L3_deeper and L4_full - most recursive)
recursive_prompts = []
for key, val in prompt_bank_1c.items():
    if val['group'] in ['L3_deeper', 'L4_full']:
        recursive_prompts.append(val['text'])
        if len(recursive_prompts) >= N_RECURSIVE:
            break

# Get baseline prompts (factual and math - most neutral)
baseline_prompts = []
for key, val in prompt_bank_1c.items():
    if val['group'] in ['baseline_factual', 'baseline_math']:
        baseline_prompts.append(val['text'])
        if len(baseline_prompts) >= N_BASELINE:
            break

print(f"Recursive prompts: {len(recursive_prompts)}")
print(f"Baseline prompts: {len(baseline_prompts)}")

# ============================================================
# STATISTICAL UTILITIES
# ============================================================
def compute_cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

def bootstrap_cohens_d_ci(group1, group2, n_bootstrap=N_BOOTSTRAP, ci=0.95):
    """Compute bootstrap CI for Cohen's d."""
    boot_d = []
    for _ in range(n_bootstrap):
        s1 = np.random.choice(group1, size=len(group1), replace=True)
        s2 = np.random.choice(group2, size=len(group2), replace=True)
        boot_d.append(compute_cohens_d(s1, s2))
    lower = np.percentile(boot_d, (1-ci)/2 * 100)
    upper = np.percentile(boot_d, (1+ci)/2 * 100)
    return lower, upper

def print_stats(name, group1, group2, bonferroni_n=1):
    """Print comprehensive statistics with Bonferroni correction."""
    g1 = [x for x in group1 if not np.isnan(x)]
    g2 = [x for x in group2 if not np.isnan(x)]
    
    if len(g1) < 2 or len(g2) < 2:
        print(f"\n{name}: Insufficient data")
        return None
    
    t_stat, p_val_raw = stats.ttest_ind(g1, g2)
    p_val_corrected = min(p_val_raw * bonferroni_n, 1.0)
    d = compute_cohens_d(g1, g2)
    d_ci = bootstrap_cohens_d_ci(g1, g2)
    
    abs_d = abs(d)
    if abs_d < 0.2:
        interp = "negligible"
    elif abs_d < 0.5:
        interp = "small"
    elif abs_d < 0.8:
        interp = "medium"
    else:
        interp = "large"
    
    print(f"\n{name}:")
    print(f"  Group 1: {np.mean(g1):.4f} ± {np.std(g1):.4f} (n={len(g1)})")
    print(f"  Group 2: {np.mean(g2):.4f} ± {np.std(g2):.4f} (n={len(g2)})")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value (raw): {p_val_raw:.2e}")
    print(f"  p-value (Bonferroni, k={bonferroni_n}): {p_val_corrected:.2e}")
    print(f"  Cohen's d: {d:.3f} [{d_ci[0]:.3f}, {d_ci[1]:.3f}]")
    print(f"  Effect size: {interp}")
    print(f"  Significant at α={ALPHA}? {'YES' if p_val_corrected < ALPHA else 'NO'}")
    
    return {'t': t_stat, 'p_raw': p_val_raw, 'p_corrected': p_val_corrected, 
            'd': d, 'd_ci': d_ci, 'mean1': np.mean(g1), 'mean2': np.mean(g2),
            'std1': np.std(g1), 'std2': np.std(g2), 'n1': len(g1), 'n2': len(g2)}

# ============================================================
# METRICS COMPUTATION
# ============================================================
def compute_pr(v_tensor, window_size=WINDOW_SIZE):
    """Compute Participation Ratio via SVD."""
    if v_tensor is None:
        return np.nan
    
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]
    
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
        
        pr = (S_sq.sum()**2) / (S_sq**2).sum()
        return float(pr)
    except:
        return np.nan

# ============================================================
# HOOKS
# ============================================================
@contextmanager
def capture_v_at_layer(model, layer_idx, storage_list):
    """Context manager to capture V activations."""
    layer = model.model.layers[layer_idx].self_attn
    
    def hook_fn(module, inp, out):
        storage_list.append(out.detach())
        return out
    
    handle = layer.v_proj.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()

# ============================================================
# R_V MEASUREMENT
# ============================================================
def measure_rv_at_layers(model, tokenizer, prompt, early_layer, target_layer):
    """Measure R_V = PR(target) / PR(early) for a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    v_early_list = []
    v_target_list = []
    
    with torch.no_grad():
        with capture_v_at_layer(model, early_layer, v_early_list):
            with capture_v_at_layer(model, target_layer, v_target_list):
                _ = model(**inputs)
    
    v_early = v_early_list[0][0] if v_early_list else None
    v_target = v_target_list[0][0] if v_target_list else None
    
    pr_early = compute_pr(v_early)
    pr_target = compute_pr(v_target)
    
    r_v = pr_target / pr_early if (pr_early and pr_early > 0) else np.nan
    
    return r_v, pr_early, pr_target

# ============================================================
# KV CACHE FUNCTIONS
# ============================================================
def extract_kv_cache(model, tokenizer, prompt):
    """Extract KV cache as DynamicCache object."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        past_kv = outputs.past_key_values
    
    kv_cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(past_kv):
        kv_cache.update(k.clone(), v.clone(), layer_idx)
    
    return kv_cache

def generate_with_partial_kv_patch(model, tokenizer, baseline_prompt, source_kv, patch_layers,
                                    max_new_tokens=MAX_NEW_TOKENS, temperature=GEN_TEMPERATURE):
    """Generate with partial KV cache patching for specific layers only."""
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
            k_src, v_src = source_kv[layer_idx]
            patched_kv.update(k_src.clone(), v_src.clone(), layer_idx)
        else:
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
    
    return tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

# ============================================================
# BEHAVIORAL SCORING
# ============================================================
import re

def score_recursive_behavior(text):
    """Score text for recursive/self-referential behavior."""
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
# STEP 1: R_V MEASUREMENT AT BOTH L22 AND L27
# ============================================================
print("\n" + "=" * 70)
print("STEP 1: R_V MEASUREMENT AT L22 AND L27")
print("=" * 70)

results = {
    'L22': {'recursive': [], 'baseline': []},
    'L27': {'recursive': [], 'baseline': []}
}

print(f"\nMeasuring {len(recursive_prompts)} recursive prompts...")
for prompt in tqdm(recursive_prompts):
    # L22
    rv_22, _, _ = measure_rv_at_layers(model, tokenizer, prompt, EARLY_LAYER, LAYER_22)
    results['L22']['recursive'].append(rv_22)
    
    # L27
    rv_27, _, _ = measure_rv_at_layers(model, tokenizer, prompt, EARLY_LAYER, LAYER_27)
    results['L27']['recursive'].append(rv_27)

print(f"\nMeasuring {len(baseline_prompts)} baseline prompts...")
for prompt in tqdm(baseline_prompts):
    # L22
    rv_22, _, _ = measure_rv_at_layers(model, tokenizer, prompt, EARLY_LAYER, LAYER_22)
    results['L22']['baseline'].append(rv_22)
    
    # L27
    rv_27, _, _ = measure_rv_at_layers(model, tokenizer, prompt, EARLY_LAYER, LAYER_27)
    results['L27']['baseline'].append(rv_27)

# ============================================================
# STEP 2: STATISTICAL ANALYSIS WITH BONFERRONI
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: STATISTICAL ANALYSIS (Bonferroni-corrected)")
print("=" * 70)

# Comparison 1: L22 recursive vs baseline
stats_L22 = print_stats("L22: Recursive vs Baseline", 
                        results['L22']['recursive'], 
                        results['L22']['baseline'],
                        bonferroni_n=N_COMPARISONS)

# Comparison 2: L27 recursive vs baseline
stats_L27 = print_stats("L27: Recursive vs Baseline", 
                        results['L27']['recursive'], 
                        results['L27']['baseline'],
                        bonferroni_n=N_COMPARISONS)

# Comparison 3: L22 vs L27 for recursive prompts
stats_layer_rec = print_stats("L22 vs L27 (Recursive prompts)", 
                              results['L22']['recursive'], 
                              results['L27']['recursive'],
                              bonferroni_n=N_COMPARISONS)

# Comparison 4: L22 vs L27 for baseline prompts
stats_layer_base = print_stats("L22 vs L27 (Baseline prompts)", 
                               results['L22']['baseline'], 
                               results['L27']['baseline'],
                               bonferroni_n=N_COMPARISONS)

# Which layer shows stronger separation?
print("\n" + "-" * 50)
print("LAYER COMPARISON SUMMARY")
print("-" * 50)
if stats_L22 and stats_L27:
    L22_gap = stats_L22['mean2'] - stats_L22['mean1']  # baseline - recursive
    L27_gap = stats_L27['mean2'] - stats_L27['mean1']
    print(f"L22 separation: {L22_gap:.4f} (baseline - recursive)")
    print(f"L27 separation: {L27_gap:.4f} (baseline - recursive)")
    print(f"L22 Cohen's d: {stats_L22['d']:.3f}")
    print(f"L27 Cohen's d: {stats_L27['d']:.3f}")
    
    if abs(stats_L27['d']) > abs(stats_L22['d']):
        print(f"\n→ L27 shows STRONGER separation (|d|={abs(stats_L27['d']):.3f} > {abs(stats_L22['d']):.3f})")
    else:
        print(f"\n→ L22 shows STRONGER separation (|d|={abs(stats_L22['d']):.3f} > {abs(stats_L27['d']):.3f})")

# ============================================================
# STEP 3: KV CACHE LAYER SWEEP
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: KV CACHE LAYER SWEEP")
print("=" * 70)

# Use first 10 prompts for KV sweep (faster)
n_kv_test = 10

print(f"\nExtracting KV caches from {n_kv_test} recursive prompts...")
recursive_kv_caches = []
for prompt in tqdm(recursive_prompts[:n_kv_test]):
    kv = extract_kv_cache(model, tokenizer, prompt)
    recursive_kv_caches.append(kv)

kv_results = {
    'early_L0_16': {'scores': []},
    'late_L16_32': {'scores': []},
    'full_L0_32': {'scores': []},
    'baseline_natural': {'scores': []}
}

# Baseline natural generation
print("\nBaseline natural generation...")
for i, prompt in enumerate(tqdm(baseline_prompts[:n_kv_test])):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=GEN_TEMPERATURE,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    kv_results['baseline_natural']['scores'].append(score_recursive_behavior(generated))

# Early layers only (L0-15)
print("\nKV patch: L0-15 only...")
for i in tqdm(range(n_kv_test)):
    gen_text = generate_with_partial_kv_patch(
        model, tokenizer, baseline_prompts[i], recursive_kv_caches[i], KV_EARLY
    )
    kv_results['early_L0_16']['scores'].append(score_recursive_behavior(gen_text))

# Late layers only (L16-31)
print("\nKV patch: L16-31 only...")
for i in tqdm(range(n_kv_test)):
    gen_text = generate_with_partial_kv_patch(
        model, tokenizer, baseline_prompts[i], recursive_kv_caches[i], KV_LATE
    )
    kv_results['late_L16_32']['scores'].append(score_recursive_behavior(gen_text))

# Full layers (L0-31)
print("\nKV patch: L0-31 (full)...")
for i in tqdm(range(n_kv_test)):
    gen_text = generate_with_partial_kv_patch(
        model, tokenizer, baseline_prompts[i], recursive_kv_caches[i], KV_FULL
    )
    kv_results['full_L0_32']['scores'].append(score_recursive_behavior(gen_text))

# Print KV sweep results
print("\n" + "-" * 50)
print("KV CACHE LAYER SWEEP RESULTS")
print("-" * 50)

baseline_mean = np.mean(kv_results['baseline_natural']['scores'])
for key in ['early_L0_16', 'late_L16_32', 'full_L0_32']:
    scores = kv_results[key]['scores']
    mean_score = np.mean(scores)
    delta = mean_score - baseline_mean
    print(f"{key}: {mean_score:.2f} ± {np.std(scores):.2f} (Δ={delta:+.2f} from baseline)")

print(f"\nBaseline natural: {baseline_mean:.2f} ± {np.std(kv_results['baseline_natural']['scores']):.2f}")

# Which layer range is more effective?
early_effect = np.mean(kv_results['early_L0_16']['scores']) - baseline_mean
late_effect = np.mean(kv_results['late_L16_32']['scores']) - baseline_mean
print(f"\n→ Early (L0-15) effect: {early_effect:+.2f}")
print(f"→ Late (L16-31) effect: {late_effect:+.2f}")

if late_effect > early_effect:
    print("→ LATE LAYERS DOMINANT (confirms DEC7 finding)")
else:
    print("→ EARLY LAYERS MORE EFFECTIVE (contradicts DEC7)")

# ============================================================
# STEP 4: SAVE RESULTS
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: SAVING RESULTS")
print("=" * 70)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"/workspace/mech-interp-phase1/DEC_8_2025_RUNPOD_GPU_TEST/01_GEOMETRY_OF_RECURSION/results/full_validation_{timestamp}.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["experiment", "layer", "group", "metric", "value"])
    
    # R_V measurements
    for i, rv in enumerate(results['L22']['recursive']):
        writer.writerow(["R_V", "L22", "recursive", "R_V", f"{rv:.4f}" if not np.isnan(rv) else "NaN"])
    for i, rv in enumerate(results['L22']['baseline']):
        writer.writerow(["R_V", "L22", "baseline", "R_V", f"{rv:.4f}" if not np.isnan(rv) else "NaN"])
    for i, rv in enumerate(results['L27']['recursive']):
        writer.writerow(["R_V", "L27", "recursive", "R_V", f"{rv:.4f}" if not np.isnan(rv) else "NaN"])
    for i, rv in enumerate(results['L27']['baseline']):
        writer.writerow(["R_V", "L27", "baseline", "R_V", f"{rv:.4f}" if not np.isnan(rv) else "NaN"])
    
    # KV sweep
    for score in kv_results['baseline_natural']['scores']:
        writer.writerow(["KV_sweep", "natural", "baseline", "behavior_score", f"{score:.2f}"])
    for score in kv_results['early_L0_16']['scores']:
        writer.writerow(["KV_sweep", "L0-15", "patched", "behavior_score", f"{score:.2f}"])
    for score in kv_results['late_L16_32']['scores']:
        writer.writerow(["KV_sweep", "L16-31", "patched", "behavior_score", f"{score:.2f}"])
    for score in kv_results['full_L0_32']['scores']:
        writer.writerow(["KV_sweep", "L0-31", "patched", "behavior_score", f"{score:.2f}"])

print(f"Results saved to: {output_file}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"""
SAMPLE SIZES:
  Recursive prompts: {len(recursive_prompts)}
  Baseline prompts: {len(baseline_prompts)}
  KV sweep samples: {n_kv_test}

R_V CONTRACTION (Bonferroni-corrected, α={ALPHA/N_COMPARISONS:.4f}):
  L22: d = {stats_L22['d']:.3f}, p = {stats_L22['p_corrected']:.2e} {"✓" if stats_L22['p_corrected'] < ALPHA else "✗"}
  L27: d = {stats_L27['d']:.3f}, p = {stats_L27['p_corrected']:.2e} {"✓" if stats_L27['p_corrected'] < ALPHA else "✗"}
  
  Stronger separation: {"L27" if abs(stats_L27['d']) > abs(stats_L22['d']) else "L22"}

KV CACHE LAYER SWEEP:
  Baseline:  {baseline_mean:.2f}
  L0-15:     {np.mean(kv_results['early_L0_16']['scores']):.2f} (Δ={early_effect:+.2f})
  L16-31:    {np.mean(kv_results['late_L16_32']['scores']):.2f} (Δ={late_effect:+.2f})
  L0-31:     {np.mean(kv_results['full_L0_32']['scores']):.2f}
  
  Dominant range: {"Late (L16-31)" if late_effect > early_effect else "Early (L0-15)"}
""")

print("\n✅ FULL VALIDATION COMPLETE")

