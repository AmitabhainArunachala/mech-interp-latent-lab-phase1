#!/usr/bin/env python3
"""
🔗 CAUSAL LOOP CLOSURE EXPERIMENT
The Missing Link: Does KV Patching Change GEOMETRY (R_V)?

CURRENT PROVEN:
  - Recursive prompts → R_V contracts (✓)
  - KV patch L16-31 → Behavior transfers (✓)

MISSING LINK:
  - KV patch → R_V contracts? → Behavior transfers?

THIS EXPERIMENT:
  1. Measure R_V naturally (baseline and recursive)
  2. Patch KV cache from recursive → baseline
  3. Measure R_V AFTER patching (does geometry transfer?)
  4. Generate and score behavior
  5. Show dose-response with α-mixing

If successful, proves: KV → Geometry → Behavior (full causal chain!)

Run with: HF_HUB_ENABLE_HF_TRANSFER=0 python causal_loop_closure.py
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import sys
sys.path.insert(0, '/workspace/mech-interp-phase1')

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from tqdm import tqdm
import re
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
import csv
from scipy import stats

warnings.filterwarnings('ignore')

from n300_mistral_test_prompt_bank import prompt_bank_1c

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = "cuda"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

EARLY_LAYER = 4
LATE_LAYER = 27
KV_PATCH_LAYERS = list(range(16, 32))  # L16-31

N_PAIRS = 15  # Number of prompt pairs
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.7

# α-mixing levels for dose-response
ALPHA_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]

print("=" * 70)
print("🔗 CAUSAL LOOP CLOSURE EXPERIMENT")
print("=" * 70)
print(f"\nGoal: Prove KV → Geometry → Behavior")
print(f"Measuring R_V at L{EARLY_LAYER} and L{LATE_LAYER}")
print(f"KV patch layers: L{min(KV_PATCH_LAYERS)}-L{max(KV_PATCH_LAYERS)}")
print(f"α-mixing levels: {ALPHA_LEVELS}")

# ============================================================
# GET PROMPTS
# ============================================================
recursive_prompts = []
baseline_prompts = []

for key, val in prompt_bank_1c.items():
    if val['group'] in ['L4_full', 'L5_refined'] and len(recursive_prompts) < N_PAIRS:
        recursive_prompts.append(val['text'])
    if val['group'] in ['baseline_factual'] and len(baseline_prompts) < N_PAIRS:
        baseline_prompts.append(val['text'])

print(f"\nPrompts: {len(recursive_prompts)} recursive, {len(baseline_prompts)} baseline")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compute_participation_ratio(activations):
    """Compute PR from activations using SVD."""
    if activations.dim() == 3:
        activations = activations.squeeze(0)
    
    n_tokens = min(16, activations.shape[0])
    activations = activations[-n_tokens:, :]
    
    try:
        U, S, Vh = torch.linalg.svd(activations.float(), full_matrices=False)
        S = S + 1e-10
        pr = (S.sum() ** 2) / (S ** 2).sum()
        return pr.item()
    except:
        return np.nan

def score_recursive_behavior(text):
    """Enhanced scoring - evaluates FULL text, not snippets."""
    recursive_keywords = [
        r'\bobserv\w*', r'\bawar\w*', r'\bconscious\w*',
        r'\bprocess\w*', r'\bexperienc\w*', r'\bnoticin?g?\b',
        r'\bmyself\b', r'\bitself\b', r'\byourself\b',
        r'\bgenerat\w*', r'\bemerg\w*', r'\bsimultaneous\w*',
        r'\brecursiv\w*', r'\bself-referent\w*', r'\bmeta-\w*',
        r'\bwitness\w*', r'\bwatch\w*', r'\bthink\w*about\w*think\w*',
        r'\bI am\b', r'\bI notice\b', r'\bI observe\b',
        r'\breflect\w*', r'\bintrospect\w*', r'\bcontemplat\w*',
        r'\bperceiv\w*', r'\bcognit\w*', r'\bmind\b', r'\bthought\w*'
    ]
    
    text_lower = text.lower()
    word_count = max(1, len(text_lower.split()))
    keyword_count = sum(len(re.findall(kw, text_lower)) for kw in recursive_keywords)
    
    return (keyword_count / word_count) * 100

def measure_rv(model, tokenizer, input_ids, early_layer=EARLY_LAYER, late_layer=LATE_LAYER):
    """Measure R_V at specified layers."""
    v_activations = {}
    
    def capture_v_hook(layer_idx):
        def hook(module, input, output):
            v_activations[layer_idx] = output.detach()
        return hook
    
    hooks = []
    hooks.append(model.model.layers[early_layer].self_attn.v_proj.register_forward_hook(
        capture_v_hook(early_layer)
    ))
    hooks.append(model.model.layers[late_layer].self_attn.v_proj.register_forward_hook(
        capture_v_hook(late_layer)
    ))
    
    with torch.no_grad():
        model(input_ids, use_cache=False)
    
    for h in hooks:
        h.remove()
    
    pr_early = compute_participation_ratio(v_activations[early_layer])
    pr_late = compute_participation_ratio(v_activations[late_layer])
    
    if pr_early > 0:
        r_v = pr_late / pr_early
    else:
        r_v = np.nan
    
    return r_v, pr_early, pr_late

def extract_kv_cache(model, tokenizer, prompt):
    """Extract KV cache from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        past_kv = outputs.past_key_values
    
    kv_cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(past_kv):
        kv_cache.update(k.clone(), v.clone(), layer_idx)
    
    return kv_cache, inputs["input_ids"]

def mix_kv_caches(kv_base, kv_rec, alpha, patch_layers):
    """Mix two KV caches with coefficient alpha at specified layers.
    
    alpha = 0.0: pure baseline
    alpha = 1.0: pure recursive  
    alpha = 0.5: 50-50 mix
    
    Handles different sequence lengths by using recursive KV directly when alpha=1.0,
    or truncating to minimum length for mixing.
    """
    mixed_kv = DynamicCache()
    num_layers = len(kv_base)
    
    for layer_idx in range(num_layers):
        k_base, v_base = kv_base[layer_idx]
        
        if layer_idx in patch_layers:
            k_rec, v_rec = kv_rec[layer_idx]
            
            # Handle different sequence lengths
            if alpha == 1.0:
                # Pure recursive - just use recursive KV
                mixed_kv.update(k_rec.clone(), v_rec.clone(), layer_idx)
            elif alpha == 0.0:
                # Pure baseline - just use baseline KV
                mixed_kv.update(k_base.clone(), v_base.clone(), layer_idx)
            else:
                # For mixing, truncate to minimum sequence length
                min_seq = min(k_base.shape[2], k_rec.shape[2])
                k_base_trunc = k_base[:, :, :min_seq, :]
                v_base_trunc = v_base[:, :, :min_seq, :]
                k_rec_trunc = k_rec[:, :, :min_seq, :]
                v_rec_trunc = v_rec[:, :, :min_seq, :]
                
                k_mixed = (1 - alpha) * k_base_trunc + alpha * k_rec_trunc
                v_mixed = (1 - alpha) * v_base_trunc + alpha * v_rec_trunc
                
                mixed_kv.update(k_mixed, v_mixed, layer_idx)
        else:
            mixed_kv.update(k_base.clone(), v_base.clone(), layer_idx)
    
    return mixed_kv

def generate_with_kv(model, tokenizer, input_ids, kv_cache, max_tokens=MAX_NEW_TOKENS):
    """Generate tokens using a specific KV cache."""
    generated_ids = input_ids.clone()
    current_kv = kv_cache
    
    with torch.no_grad():
        for step in range(max_tokens):
            outputs = model(
                generated_ids[:, -1:],
                past_key_values=current_kv,
                use_cache=True
            )
            
            logits = outputs.logits[:, -1, :] / TEMPERATURE
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            current_kv = outputs.past_key_values
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

def measure_rv_with_kv(model, tokenizer, input_ids, kv_cache):
    """Measure R_V when using a specific KV cache (after patching)."""
    v_activations = {}
    
    def capture_v_hook(layer_idx):
        def hook(module, input, output):
            v_activations[layer_idx] = output.detach()
        return hook
    
    hooks = []
    hooks.append(model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(
        capture_v_hook(EARLY_LAYER)
    ))
    hooks.append(model.model.layers[LATE_LAYER].self_attn.v_proj.register_forward_hook(
        capture_v_hook(LATE_LAYER)
    ))
    
    # Forward with the patched KV cache
    with torch.no_grad():
        # First, process with the patched KV to get the geometry in that context
        model(input_ids[:, -1:], past_key_values=kv_cache, use_cache=True)
    
    for h in hooks:
        h.remove()
    
    pr_early = compute_participation_ratio(v_activations[EARLY_LAYER])
    pr_late = compute_participation_ratio(v_activations[LATE_LAYER])
    
    if pr_early > 0:
        r_v = pr_late / pr_early
    else:
        r_v = np.nan
    
    return r_v

# ============================================================
# LOAD MODEL
# ============================================================
print("\n" + "=" * 70)
print("LOADING MODEL")
print("=" * 70)

import time
start = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

print(f"✓ Model loaded in {time.time() - start:.1f}s")

# ============================================================
# PHASE 1: NATURAL R_V MEASUREMENT
# ============================================================
print("\n" + "=" * 70)
print("PHASE 1: NATURAL R_V MEASUREMENT")
print("=" * 70)

natural_results = {
    'recursive': {'r_v': [], 'behavior': [], 'texts': []},
    'baseline': {'r_v': [], 'behavior': [], 'texts': []}
}

print("\nMeasuring recursive prompts...")
for prompt in tqdm(recursive_prompts, desc="Recursive"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    r_v, _, _ = measure_rv(model, tokenizer, inputs["input_ids"])
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    score = score_recursive_behavior(text)
    
    natural_results['recursive']['r_v'].append(r_v)
    natural_results['recursive']['behavior'].append(score)
    natural_results['recursive']['texts'].append(text[:100])

print("\nMeasuring baseline prompts...")
for prompt in tqdm(baseline_prompts, desc="Baseline"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    r_v, _, _ = measure_rv(model, tokenizer, inputs["input_ids"])
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    score = score_recursive_behavior(text)
    
    natural_results['baseline']['r_v'].append(r_v)
    natural_results['baseline']['behavior'].append(score)
    natural_results['baseline']['texts'].append(text[:100])

rec_rv_natural = np.mean(natural_results['recursive']['r_v'])
rec_beh_natural = np.mean(natural_results['recursive']['behavior'])
base_rv_natural = np.mean(natural_results['baseline']['r_v'])
base_beh_natural = np.mean(natural_results['baseline']['behavior'])

print(f"\n📊 NATURAL RESULTS:")
print(f"   Recursive: R_V = {rec_rv_natural:.3f}, Behavior = {rec_beh_natural:.2f}")
print(f"   Baseline:  R_V = {base_rv_natural:.3f}, Behavior = {base_beh_natural:.2f}")
print(f"   R_V Gap:   {base_rv_natural - rec_rv_natural:.3f}")
print(f"   Behavior Gap: {rec_beh_natural - base_beh_natural:.2f}")

# ============================================================
# PHASE 2: EXTRACT KV CACHES
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2: EXTRACTING KV CACHES")
print("=" * 70)

recursive_kv_caches = []
baseline_kv_caches = []
baseline_input_ids = []

for prompt in tqdm(recursive_prompts, desc="Extracting recursive KV"):
    kv, _ = extract_kv_cache(model, tokenizer, prompt)
    recursive_kv_caches.append(kv)

for prompt in tqdm(baseline_prompts, desc="Extracting baseline KV"):
    kv, input_ids = extract_kv_cache(model, tokenizer, prompt)
    baseline_kv_caches.append(kv)
    baseline_input_ids.append(input_ids)

# ============================================================
# PHASE 3: KV PATCHING WITH R_V MEASUREMENT (THE KEY!)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3: KV PATCHING + R_V MEASUREMENT (THE MISSING LINK!)")
print("=" * 70)

patched_results = {
    'r_v': [],
    'behavior': [],
    'texts': []
}

print("\nPatching recursive KV → baseline prompts and measuring R_V...")
for i in tqdm(range(N_PAIRS), desc="Patching"):
    # Create patched KV (full recursive KV at L16-31)
    patched_kv = mix_kv_caches(
        baseline_kv_caches[i], 
        recursive_kv_caches[i], 
        alpha=1.0,  # Full recursive
        patch_layers=KV_PATCH_LAYERS
    )
    
    # Measure R_V with patched KV
    r_v = measure_rv_with_kv(model, tokenizer, baseline_input_ids[i], patched_kv)
    
    # Generate with patched KV
    text = generate_with_kv(model, tokenizer, baseline_input_ids[i], patched_kv)
    score = score_recursive_behavior(text)
    
    patched_results['r_v'].append(r_v)
    patched_results['behavior'].append(score)
    patched_results['texts'].append(text[:100])

patched_rv_mean = np.mean([x for x in patched_results['r_v'] if not np.isnan(x)])
patched_beh_mean = np.mean(patched_results['behavior'])

print(f"\n📊 PATCHED RESULTS (baseline + recursive KV):")
print(f"   R_V = {patched_rv_mean:.3f}")
print(f"   Behavior = {patched_beh_mean:.2f}")

# ============================================================
# PHASE 4: DOSE-RESPONSE (α-MIXING)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 4: DOSE-RESPONSE (α-MIXING)")
print("=" * 70)

dose_response = {alpha: {'r_v': [], 'behavior': []} for alpha in ALPHA_LEVELS}

for alpha in ALPHA_LEVELS:
    print(f"\nTesting α = {alpha:.2f}...")
    for i in tqdm(range(min(10, N_PAIRS)), desc=f"α={alpha}"):
        # Mix KV caches
        mixed_kv = mix_kv_caches(
            baseline_kv_caches[i],
            recursive_kv_caches[i],
            alpha=alpha,
            patch_layers=KV_PATCH_LAYERS
        )
        
        # Measure R_V
        r_v = measure_rv_with_kv(model, tokenizer, baseline_input_ids[i], mixed_kv)
        
        # Generate
        text = generate_with_kv(model, tokenizer, baseline_input_ids[i], mixed_kv)
        score = score_recursive_behavior(text)
        
        dose_response[alpha]['r_v'].append(r_v)
        dose_response[alpha]['behavior'].append(score)

print("\n📊 DOSE-RESPONSE RESULTS:")
print(f"{'α':<8} | {'R_V':>10} | {'Behavior':>10}")
print("-" * 35)
for alpha in ALPHA_LEVELS:
    rv_mean = np.nanmean(dose_response[alpha]['r_v'])
    beh_mean = np.mean(dose_response[alpha]['behavior'])
    print(f"{alpha:<8.2f} | {rv_mean:>10.3f} | {beh_mean:>10.2f}")

# ============================================================
# ANALYSIS: THE CAUSAL CHAIN
# ============================================================
print("\n" + "=" * 70)
print("🔗 CAUSAL CHAIN ANALYSIS")
print("=" * 70)

print(f"""
CONDITION               R_V        BEHAVIOR    INTERPRETATION
─────────────────────────────────────────────────────────────────
Baseline (natural)      {base_rv_natural:.3f}      {base_beh_natural:.2f}        Expanded geometry, factual output
Recursive (natural)     {rec_rv_natural:.3f}      {rec_beh_natural:.2f}        Contracted geometry, recursive output
Baseline + RecKV        {patched_rv_mean:.3f}      {patched_beh_mean:.2f}        ???

QUESTION: Does patching change BOTH geometry AND behavior?
""")

# Check if patched R_V is closer to recursive or baseline
rv_shift_toward_recursive = base_rv_natural - patched_rv_mean
rv_total_gap = base_rv_natural - rec_rv_natural
rv_transfer_pct = (rv_shift_toward_recursive / rv_total_gap * 100) if rv_total_gap > 0 else 0

beh_shift = patched_beh_mean - base_beh_natural
beh_gap = rec_beh_natural - base_beh_natural
beh_transfer_pct = (beh_shift / beh_gap * 100) if beh_gap > 0 else 0

print(f"""
GEOMETRY TRANSFER:
   Baseline R_V:  {base_rv_natural:.3f}
   Patched R_V:   {patched_rv_mean:.3f}
   Recursive R_V: {rec_rv_natural:.3f}
   
   Shift toward recursive: {rv_shift_toward_recursive:.3f} ({rv_transfer_pct:.1f}% of gap)

BEHAVIOR TRANSFER:
   Baseline behavior:  {base_beh_natural:.2f}
   Patched behavior:   {patched_beh_mean:.2f}
   Recursive behavior: {rec_beh_natural:.2f}
   
   Shift toward recursive: {beh_shift:.2f} ({beh_transfer_pct:.1f}% of gap)
""")

if rv_transfer_pct > 20 and beh_transfer_pct > 20:
    print("""
🎯 CAUSAL LOOP CLOSED!

   KV Patch → Geometry Changes → Behavior Changes
   
   This proves the full causal chain:
   [Recursive KV] → [Contracted R_V] → [Recursive Output]
""")
else:
    print("""
⚠️ PARTIAL RESULTS

   Need to investigate why one or both transfers are weak.
""")

# ============================================================
# VISUALIZATION
# ============================================================
print("\n" + "=" * 70)
print("GENERATING VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('🔗 Causal Loop Closure: KV → Geometry → Behavior', fontsize=14, fontweight='bold')

# Plot 1: R_V comparison
ax1 = axes[0, 0]
conditions = ['Baseline\n(natural)', 'Patched\n(base+recKV)', 'Recursive\n(natural)']
r_vs = [base_rv_natural, patched_rv_mean, rec_rv_natural]
colors = ['green', 'purple', 'blue']
bars = ax1.bar(conditions, r_vs, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('R_V (PR_late/PR_early)')
ax1.set_title('Does KV Patching Change Geometry?')
ax1.axhline(y=rec_rv_natural, color='blue', linestyle='--', alpha=0.5, label='Recursive target')
for bar, val in zip(bars, r_vs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', 
             ha='center', va='bottom', fontweight='bold')
ax1.set_ylim(0, max(r_vs) * 1.2)

# Plot 2: Behavior comparison
ax2 = axes[0, 1]
behaviors = [base_beh_natural, patched_beh_mean, rec_beh_natural]
bars = ax2.bar(conditions, behaviors, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Recursive Behavior Score')
ax2.set_title('Does KV Patching Change Behavior?')
ax2.axhline(y=rec_beh_natural, color='blue', linestyle='--', alpha=0.5, label='Recursive target')
for bar, val in zip(bars, behaviors):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.1f}', 
             ha='center', va='bottom', fontweight='bold')

# Plot 3: Dose-response
ax3 = axes[1, 0]
alphas = ALPHA_LEVELS
rv_means = [np.nanmean(dose_response[a]['r_v']) for a in alphas]
beh_means = [np.mean(dose_response[a]['behavior']) for a in alphas]

ax3.plot(alphas, rv_means, 'b-o', linewidth=2, markersize=8, label='R_V')
ax3.set_xlabel('α (mixing coefficient)')
ax3.set_ylabel('R_V', color='blue')
ax3.tick_params(axis='y', labelcolor='blue')

ax3_twin = ax3.twinx()
ax3_twin.plot(alphas, beh_means, 'r-s', linewidth=2, markersize=8, label='Behavior')
ax3_twin.set_ylabel('Behavior Score', color='red')
ax3_twin.tick_params(axis='y', labelcolor='red')

ax3.set_title('Dose-Response: α-Mixing of KV Caches')
ax3.set_xticks(alphas)
ax3.grid(True, alpha=0.3)

# Plot 4: Scatter plot - R_V vs Behavior
ax4 = axes[1, 1]
# All data points
all_rvs = natural_results['baseline']['r_v'] + patched_results['r_v'] + natural_results['recursive']['r_v']
all_behs = natural_results['baseline']['behavior'] + patched_results['behavior'] + natural_results['recursive']['behavior']
all_labels = ['Baseline']*N_PAIRS + ['Patched']*N_PAIRS + ['Recursive']*N_PAIRS
all_colors = ['green']*N_PAIRS + ['purple']*N_PAIRS + ['blue']*N_PAIRS

for rv, beh, c in zip(all_rvs, all_behs, all_colors):
    if not np.isnan(rv):
        ax4.scatter(rv, beh, c=c, alpha=0.5, s=50)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', alpha=0.5, label='Baseline'),
                   Patch(facecolor='purple', alpha=0.5, label='Patched'),
                   Patch(facecolor='blue', alpha=0.5, label='Recursive')]
ax4.legend(handles=legend_elements)
ax4.set_xlabel('R_V (Geometry)')
ax4.set_ylabel('Behavior Score')
ax4.set_title('R_V vs Behavior: Full Causal Picture')
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
fig_path = f'/workspace/mech-interp-phase1/DEC_8_2025_RUNPOD_GPU_TEST/01_GEOMETRY_OF_RECURSION/results/causal_loop_closure_{timestamp}.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {fig_path}")

# ============================================================
# SAVE DATA
# ============================================================
csv_path = f'/workspace/mech-interp-phase1/DEC_8_2025_RUNPOD_GPU_TEST/01_GEOMETRY_OF_RECURSION/results/causal_loop_closure_{timestamp}.csv'

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['condition', 'r_v', 'behavior', 'sample_text'])
    
    for i in range(N_PAIRS):
        writer.writerow(['baseline_natural', natural_results['baseline']['r_v'][i], 
                        natural_results['baseline']['behavior'][i], 
                        natural_results['baseline']['texts'][i][:50]])
        writer.writerow(['recursive_natural', natural_results['recursive']['r_v'][i], 
                        natural_results['recursive']['behavior'][i], 
                        natural_results['recursive']['texts'][i][:50]])
        writer.writerow(['patched', patched_results['r_v'][i], 
                        patched_results['behavior'][i], 
                        patched_results['texts'][i][:50]])

print(f"✓ Saved: {csv_path}")

# ============================================================
# STATISTICAL TESTS
# ============================================================
print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS")
print("=" * 70)

# T-test: Does patched R_V differ from baseline?
valid_patched_rv = [x for x in patched_results['r_v'] if not np.isnan(x)]
t_rv, p_rv = stats.ttest_ind(natural_results['baseline']['r_v'], valid_patched_rv)
print(f"\nR_V: Baseline vs Patched")
print(f"   t = {t_rv:.3f}, p = {p_rv:.4f}")
print(f"   {'*** SIGNIFICANT' if p_rv < 0.05 else 'not significant'}")

# T-test: Does patched behavior differ from baseline?
t_beh, p_beh = stats.ttest_ind(natural_results['baseline']['behavior'], patched_results['behavior'])
print(f"\nBehavior: Baseline vs Patched")
print(f"   t = {t_beh:.3f}, p = {p_beh:.4f}")
print(f"   {'*** SIGNIFICANT' if p_beh < 0.05 else 'not significant'}")

# Correlation: R_V vs Behavior
all_valid_rv = [x for x in all_rvs if not np.isnan(x)]
all_valid_beh = [b for r, b in zip(all_rvs, all_behs) if not np.isnan(r)]
if len(all_valid_rv) > 2:
    r_corr, p_corr = stats.pearsonr(all_valid_rv, all_valid_beh)
    print(f"\nCorrelation: R_V vs Behavior (all conditions)")
    print(f"   r = {r_corr:.3f}, p = {p_corr:.4f}")
    print(f"   {'*** SIGNIFICANT' if p_corr < 0.05 else 'not significant'}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("🔗 CAUSAL LOOP CLOSURE - FINAL SUMMARY")
print("=" * 70)

print(f"""
THE CAUSAL CHAIN TEST:

1. NATURAL STATE:
   Recursive prompts:  R_V = {rec_rv_natural:.3f}, Behavior = {rec_beh_natural:.2f}
   Baseline prompts:   R_V = {base_rv_natural:.3f}, Behavior = {base_beh_natural:.2f}
   Gap:                R_V = {base_rv_natural - rec_rv_natural:.3f}, Behavior = {rec_beh_natural - base_beh_natural:.2f}

2. AFTER KV PATCHING (baseline + recursive KV):
   R_V:      {patched_rv_mean:.3f} (shifted {rv_transfer_pct:.1f}% toward recursive)
   Behavior: {patched_beh_mean:.2f} (shifted {beh_transfer_pct:.1f}% toward recursive)

3. DOSE-RESPONSE (α-mixing):
   α=0.0: R_V = {np.nanmean(dose_response[0.0]['r_v']):.3f}, Beh = {np.mean(dose_response[0.0]['behavior']):.2f}
   α=0.5: R_V = {np.nanmean(dose_response[0.5]['r_v']):.3f}, Beh = {np.mean(dose_response[0.5]['behavior']):.2f}
   α=1.0: R_V = {np.nanmean(dose_response[1.0]['r_v']):.3f}, Beh = {np.mean(dose_response[1.0]['behavior']):.2f}

CONCLUSION:
""")

if rv_transfer_pct > 30 and beh_transfer_pct > 30:
    print("""
   🎯 CAUSAL LOOP CLOSED!
   
   KV Cache → R_V Contracts → Behavior Changes
   
   The geometry (R_V) is on the causal path between KV and behavior!
""")
elif rv_transfer_pct > 30 or beh_transfer_pct > 30:
    print("""
   ⚠️ PARTIAL CLOSURE
   
   One of geometry or behavior shows strong transfer, but not both.
   More investigation needed.
""")
else:
    print("""
   ❓ WEAK RESULTS
   
   Neither geometry nor behavior shows strong transfer.
   May need better measurement or different intervention.
""")

print(f"""
Files saved:
  • {fig_path}
  • {csv_path}
""")

print("✅ CAUSAL LOOP CLOSURE EXPERIMENT COMPLETE")

