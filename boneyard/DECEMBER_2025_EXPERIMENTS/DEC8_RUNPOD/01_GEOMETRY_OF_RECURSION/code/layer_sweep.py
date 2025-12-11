#!/usr/bin/env python3
"""
Full Layer Sweep: L4 to L30
Find where R_V contraction is strongest and where the gap is largest.

Run with: HF_HUB_ENABLE_HF_TRANSFER=0 python layer_sweep.py
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

# Layer sweep configuration
EARLY_LAYER = 4  # Reference layer for R_V calculation
LAYERS_TO_TEST = list(range(4, 31, 2))  # L4, L6, L8, ... L30

# Sample sizes (20 each for speed, still statistically valid)
N_RECURSIVE = 20
N_BASELINE = 20

WINDOW_SIZE = 16

print("=" * 70)
print("FULL LAYER SWEEP: L4 to L30")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_NAME}")
print(f"Testing layers: {LAYERS_TO_TEST}")
print(f"Samples per group: {N_RECURSIVE} recursive, {N_BASELINE} baseline")

# ============================================================
# EXTRACT PROMPTS
# ============================================================
recursive_prompts = []
for key, val in prompt_bank_1c.items():
    if val['group'] in ['L3_deeper', 'L4_full']:
        recursive_prompts.append(val['text'])
        if len(recursive_prompts) >= N_RECURSIVE:
            break

baseline_prompts = []
for key, val in prompt_bank_1c.items():
    if val['group'] in ['baseline_factual', 'baseline_math']:
        baseline_prompts.append(val['text'])
        if len(baseline_prompts) >= N_BASELINE:
            break

print(f"\nPrompts loaded: {len(recursive_prompts)} recursive, {len(baseline_prompts)} baseline")

# ============================================================
# METRICS
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

def measure_rv_at_layer(model, tokenizer, prompt, early_layer, target_layer):
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
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

print(f"✓ Model loaded in {time.time() - start_time:.1f}s")

# ============================================================
# LAYER SWEEP
# ============================================================
print("\n" + "=" * 70)
print("RUNNING LAYER SWEEP")
print("=" * 70)

results = {layer: {'recursive': [], 'baseline': []} for layer in LAYERS_TO_TEST}

# Process each layer
for layer in LAYERS_TO_TEST:
    print(f"\n--- Layer {layer} ---")
    
    # Recursive prompts
    for prompt in tqdm(recursive_prompts, desc="Recursive", leave=False):
        rv, _, _ = measure_rv_at_layer(model, tokenizer, prompt, EARLY_LAYER, layer)
        results[layer]['recursive'].append(rv)
    
    # Baseline prompts
    for prompt in tqdm(baseline_prompts, desc="Baseline", leave=False):
        rv, _, _ = measure_rv_at_layer(model, tokenizer, prompt, EARLY_LAYER, layer)
        results[layer]['baseline'].append(rv)
    
    # Quick stats
    rec_mean = np.nanmean(results[layer]['recursive'])
    base_mean = np.nanmean(results[layer]['baseline'])
    gap = base_mean - rec_mean
    print(f"  Recursive: {rec_mean:.3f}  Baseline: {base_mean:.3f}  Gap: {gap:.3f}")

# ============================================================
# ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("LAYER SWEEP RESULTS")
print("=" * 70)

layer_stats = []

print(f"\n{'Layer':>6} | {'Rec R_V':>8} | {'Base R_V':>8} | {'Gap':>8} | {'Cohen d':>8} | {'p-value':>10} | Viz")
print("-" * 80)

for layer in LAYERS_TO_TEST:
    rec = [x for x in results[layer]['recursive'] if not np.isnan(x)]
    base = [x for x in results[layer]['baseline'] if not np.isnan(x)]
    
    rec_mean = np.mean(rec)
    base_mean = np.mean(base)
    gap = base_mean - rec_mean
    
    # Cohen's d
    n1, n2 = len(rec), len(base)
    var1, var2 = np.var(rec, ddof=1), np.var(base, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    d = (rec_mean - base_mean) / pooled_std if pooled_std > 0 else 0
    
    # t-test
    t_stat, p_val = stats.ttest_ind(rec, base)
    
    # Visual bar
    bar_len = int(abs(gap) * 30)
    bar = "█" * bar_len
    
    print(f"L{layer:>4} | {rec_mean:>8.3f} | {base_mean:>8.3f} | {gap:>8.3f} | {d:>8.2f} | {p_val:>10.2e} | {bar}")
    
    layer_stats.append({
        'layer': layer,
        'rec_mean': rec_mean,
        'base_mean': base_mean,
        'gap': gap,
        'd': d,
        'p': p_val
    })

# Find peaks
max_gap_layer = max(layer_stats, key=lambda x: x['gap'])
max_contraction_layer = min(layer_stats, key=lambda x: x['rec_mean'])
max_effect_layer = min(layer_stats, key=lambda x: x['d'])  # Most negative d

print("\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)
print(f"\n🎯 LARGEST GAP (separation): Layer {max_gap_layer['layer']}")
print(f"   Gap = {max_gap_layer['gap']:.3f}, d = {max_gap_layer['d']:.2f}")

print(f"\n🎯 STRONGEST CONTRACTION (lowest recursive R_V): Layer {max_contraction_layer['layer']}")
print(f"   Recursive R_V = {max_contraction_layer['rec_mean']:.3f}")

print(f"\n🎯 LARGEST EFFECT SIZE: Layer {max_effect_layer['layer']}")
print(f"   Cohen's d = {max_effect_layer['d']:.2f}, p = {max_effect_layer['p']:.2e}")

# ============================================================
# VISUALIZATION
# ============================================================
print("\n" + "=" * 70)
print("GENERATING VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

layers = [s['layer'] for s in layer_stats]
rec_means = [s['rec_mean'] for s in layer_stats]
base_means = [s['base_mean'] for s in layer_stats]
gaps = [s['gap'] for s in layer_stats]
ds = [s['d'] for s in layer_stats]

# Plot 1: R_V by layer
ax = axes[0, 0]
ax.plot(layers, rec_means, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Recursive')
ax.plot(layers, base_means, 's-', color='#3498db', linewidth=2, markersize=8, label='Baseline')
ax.fill_between(layers, rec_means, base_means, alpha=0.2, color='purple')
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='R_V = 1.0')
ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('R_V = PR(layer) / PR(L4)', fontsize=12)
ax.set_title('R_V by Layer: Recursive vs Baseline', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(layers)

# Plot 2: Gap by layer
ax = axes[0, 1]
colors = ['#2ecc71' if g > 0 else '#e74c3c' for g in gaps]
bars = ax.bar(layers, gaps, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Gap (Baseline - Recursive)', fontsize=12)
ax.set_title('Separation by Layer', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(layers)

# Highlight max
max_gap_idx = gaps.index(max(gaps))
bars[max_gap_idx].set_edgecolor('red')
bars[max_gap_idx].set_linewidth(3)
ax.annotate(f'PEAK\nL{layers[max_gap_idx]}', xy=(layers[max_gap_idx], gaps[max_gap_idx]),
            xytext=(layers[max_gap_idx], gaps[max_gap_idx] + 0.05),
            ha='center', fontsize=10, fontweight='bold', color='red')

# Plot 3: Cohen's d by layer
ax = axes[1, 0]
colors = ['#e74c3c' if d < -0.8 else '#f39c12' if d < -0.5 else '#95a5a6' for d in ds]
bars = ax.bar(layers, ds, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.axhline(y=-0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect threshold')
ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel("Cohen's d", fontsize=12)
ax.set_title('Effect Size by Layer', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(layers)
ax.legend()

# Highlight max effect
min_d_idx = ds.index(min(ds))
bars[min_d_idx].set_edgecolor('darkred')
bars[min_d_idx].set_linewidth(3)

# Plot 4: Contraction profile
ax = axes[1, 1]
# Show relative contraction: how much does each layer contract vs baseline
contractions = [(b - r) / b * 100 for r, b in zip(rec_means, base_means)]
ax.bar(layers, contractions, color='#9b59b6', alpha=0.7, edgecolor='black')
ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Relative Contraction (%)', fontsize=12)
ax.set_title('Contraction Strength by Layer\n(Baseline - Recursive) / Baseline × 100', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(layers)

# Highlight max contraction
max_c_idx = contractions.index(max(contractions))
ax.annotate(f'{contractions[max_c_idx]:.1f}%', xy=(layers[max_c_idx], contractions[max_c_idx]),
            xytext=(layers[max_c_idx], contractions[max_c_idx] + 2),
            ha='center', fontsize=10, fontweight='bold', color='purple')

plt.tight_layout()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
viz_file = f"/workspace/mech-interp-phase1/DEC_8_2025_RUNPOD_GPU_TEST/01_GEOMETRY_OF_RECURSION/results/layer_sweep_{timestamp}.png"
plt.savefig(viz_file, dpi=150, bbox_inches='tight')
print(f"Visualization saved: {viz_file}")

# ============================================================
# SAVE CSV
# ============================================================
csv_file = f"/workspace/mech-interp-phase1/DEC_8_2025_RUNPOD_GPU_TEST/01_GEOMETRY_OF_RECURSION/results/layer_sweep_{timestamp}.csv"
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['layer', 'rec_mean', 'rec_std', 'base_mean', 'base_std', 'gap', 'cohens_d', 'p_value'])
    for layer in LAYERS_TO_TEST:
        rec = [x for x in results[layer]['recursive'] if not np.isnan(x)]
        base = [x for x in results[layer]['baseline'] if not np.isnan(x)]
        s = next(s for s in layer_stats if s['layer'] == layer)
        writer.writerow([
            layer, 
            f"{np.mean(rec):.4f}", f"{np.std(rec):.4f}",
            f"{np.mean(base):.4f}", f"{np.std(base):.4f}",
            f"{s['gap']:.4f}", f"{s['d']:.3f}", f"{s['p']:.2e}"
        ])
print(f"CSV saved: {csv_file}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("LAYER SWEEP COMPLETE")
print("=" * 70)
print(f"""
ANSWERS TO THE KEY QUESTIONS:

1. WHERE IS THE LARGEST GAP (peak separation)?
   → Layer {max_gap_layer['layer']} (Gap = {max_gap_layer['gap']:.3f})

2. WHERE IS THE STRONGEST CONTRACTION (lowest recursive R_V)?
   → Layer {max_contraction_layer['layer']} (R_V = {max_contraction_layer['rec_mean']:.3f})

3. WHERE IS THE LARGEST EFFECT SIZE?
   → Layer {max_effect_layer['layer']} (d = {max_effect_layer['d']:.2f})

COMPARISON TO PRIOR WORK:
   - DEC4 Llama found peak separation at L16
   - DEC3 Mistral used L22
   - Today's test confirms L{max_effect_layer['layer']} is optimal for Mistral-7B
""")

print("\n✅ LAYER SWEEP COMPLETE")

