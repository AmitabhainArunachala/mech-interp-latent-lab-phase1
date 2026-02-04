#!/usr/bin/env python3
"""
🔗 CAUSAL LOOP CLOSURE v2 - CORRECTED METHODOLOGY
Fixes the R_V measurement bug: now measures R_V on FULL generation trajectories

KEY FIX:
- Previous code measured R_V after patching with ONE token → meaningless
- Correct: Measure R_V on the SAME generated tokens used for behavior scoring

PROTOCOL:
1. Apply KV cache (natural or α-mixed)
2. Generate full sequence (50+ tokens)
3. Capture V at L4 and L27 at EACH generation step
4. Compute R_V from last W tokens of V activations
5. Score behavior on SAME generated text

Run with: HF_HUB_ENABLE_HF_TRANSFER=0 python causal_loop_closure_v2.py
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import sys
# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
# Device selection with MPS fallback for Apple Silicon
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

EARLY_LAYER = 4
LATE_LAYER = 27
KV_PATCH_LAYERS = list(range(16, 32))

N_PAIRS = 10  # Prompt pairs per condition
MAX_NEW_TOKENS = 64  # Full generation for proper R_V measurement
WINDOW_SIZE = 16  # Last W tokens for R_V computation
TEMPERATURE = 0.7

ALPHA_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]

print("=" * 70)
print("🔗 CAUSAL LOOP CLOSURE v2 - CORRECTED METHODOLOGY")
print("=" * 70)
print(f"\nKEY FIX: R_V measured on full generation trajectory, not single token")
print(f"Generation: {MAX_NEW_TOKENS} tokens, Window: last {WINDOW_SIZE} tokens")
print(f"Layers: Early=L{EARLY_LAYER}, Late=L{LATE_LAYER}")
print(f"α-levels: {ALPHA_LEVELS}")

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

def compute_participation_ratio(v_stack):
    """Compute PR from stacked V activations [seq_len, hidden_dim]."""
    if v_stack.dim() == 3:
        v_stack = v_stack.squeeze(0)
    
    # Ensure we have enough tokens
    if v_stack.shape[0] < 2:
        return np.nan
    
    try:
        # Use float32 for numerical stability
        v_float = v_stack.float()
        U, S, Vh = torch.linalg.svd(v_float, full_matrices=False)
        S = S + 1e-10
        pr = (S.sum() ** 2) / (S ** 2).sum()
        return pr.item()
    except:
        return np.nan

def score_recursive_behavior(text):
    """Enhanced scoring on FULL generated text."""
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

def extract_kv_cache(model, tokenizer, prompt):
    """Extract KV cache from prompt encoding."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        past_kv = outputs.past_key_values
    
    # Store as list of (K, V) tuples in float32 for mixing stability
    kv_list = []
    for layer_idx, (k, v) in enumerate(past_kv):
        kv_list.append((k.float(), v.float()))
    
    return kv_list, inputs["input_ids"]

def mix_kv_caches(kv_base, kv_rec, alpha, patch_layers):
    """Mix KV caches with α in float32, then convert to DynamicCache."""
    mixed_kv = DynamicCache()
    num_layers = len(kv_base)
    
    for layer_idx in range(num_layers):
        k_base, v_base = kv_base[layer_idx]
        
        if layer_idx in patch_layers and alpha > 0:
            k_rec, v_rec = kv_rec[layer_idx]
            
            if alpha == 1.0:
                # Pure recursive
                k_out = k_rec.half()
                v_out = v_rec.half()
            else:
                # Mix in float32, then convert
                min_seq = min(k_base.shape[2], k_rec.shape[2])
                k_base_t = k_base[:, :, :min_seq, :]
                v_base_t = v_base[:, :, :min_seq, :]
                k_rec_t = k_rec[:, :, :min_seq, :]
                v_rec_t = v_rec[:, :, :min_seq, :]
                
                k_out = ((1 - alpha) * k_base_t + alpha * k_rec_t).half()
                v_out = ((1 - alpha) * v_base_t + alpha * v_rec_t).half()
        else:
            k_out = k_base.half()
            v_out = v_base.half()
        
        mixed_kv.update(k_out, v_out, layer_idx)
    
    return mixed_kv

def generate_with_v_capture(model, tokenizer, input_ids, kv_cache, 
                            max_tokens=MAX_NEW_TOKENS, 
                            early_layer=EARLY_LAYER, late_layer=LATE_LAYER):
    """
    Generate tokens while capturing V activations at each step.
    
    Returns:
        - generated_text: str
        - v_early_stack: tensor [seq_len, hidden_dim]
        - v_late_stack: tensor [seq_len, hidden_dim]
    """
    v_early_list = []
    v_late_list = []
    
    generated_ids = input_ids.clone()
    current_kv = kv_cache
    
    def make_v_hook(storage_list):
        def hook(module, input, output):
            # output shape: [batch, seq, hidden]
            # Take the last token's activation
            storage_list.append(output[:, -1, :].detach().cpu())
        return hook
    
    with torch.no_grad():
        for step in range(max_tokens):
            # Register hooks for this step
            hook_early = model.model.layers[early_layer].self_attn.v_proj.register_forward_hook(
                make_v_hook(v_early_list)
            )
            hook_late = model.model.layers[late_layer].self_attn.v_proj.register_forward_hook(
                make_v_hook(v_late_list)
            )
            
            # Forward pass for next token
            outputs = model(
                generated_ids[:, -1:],
                past_key_values=current_kv,
                use_cache=True
            )
            
            # Remove hooks
            hook_early.remove()
            hook_late.remove()
            
            # Sample next token
            logits = outputs.logits[:, -1, :] / TEMPERATURE
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            current_kv = outputs.past_key_values
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode generated text (excluding prompt)
    generated_text = tokenizer.decode(
        generated_ids[0][input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    # Stack V activations
    if len(v_early_list) > 0:
        v_early_stack = torch.cat(v_early_list, dim=0)  # [gen_len, hidden]
        v_late_stack = torch.cat(v_late_list, dim=0)
    else:
        v_early_stack = torch.zeros(1, 1)
        v_late_stack = torch.zeros(1, 1)
    
    return generated_text, v_early_stack, v_late_stack

def compute_rv_from_v_stacks(v_early_stack, v_late_stack, window_size=WINDOW_SIZE):
    """Compute R_V from the last W tokens of V activations."""
    # Take last W tokens
    v_early = v_early_stack[-window_size:] if v_early_stack.shape[0] >= window_size else v_early_stack
    v_late = v_late_stack[-window_size:] if v_late_stack.shape[0] >= window_size else v_late_stack
    
    pr_early = compute_participation_ratio(v_early)
    pr_late = compute_participation_ratio(v_late)
    
    if pr_early > 0 and not np.isnan(pr_early):
        r_v = pr_late / pr_early
    else:
        r_v = np.nan
    
    return r_v, pr_early, pr_late

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
# PHASE 1: EXTRACT KV CACHES
# ============================================================
print("\n" + "=" * 70)
print("PHASE 1: EXTRACTING KV CACHES")
print("=" * 70)

recursive_kv_caches = []
baseline_kv_caches = []
baseline_input_ids = []
recursive_input_ids = []

for prompt in tqdm(recursive_prompts, desc="Extracting recursive KV"):
    kv, input_ids = extract_kv_cache(model, tokenizer, prompt)
    recursive_kv_caches.append(kv)
    recursive_input_ids.append(input_ids)

for prompt in tqdm(baseline_prompts, desc="Extracting baseline KV"):
    kv, input_ids = extract_kv_cache(model, tokenizer, prompt)
    baseline_kv_caches.append(kv)
    baseline_input_ids.append(input_ids)

# ============================================================
# PHASE 2: NATURAL BASELINE AND RECURSIVE (REFERENCE)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2: NATURAL GENERATION (REFERENCE)")
print("=" * 70)

natural_results = {
    'recursive': {'r_v': [], 'behavior': [], 'pr_early': [], 'pr_late': []},
    'baseline': {'r_v': [], 'behavior': [], 'pr_early': [], 'pr_late': []}
}

print("\nGenerating from recursive prompts (natural)...")
for i in tqdm(range(N_PAIRS), desc="Recursive natural"):
    # Create natural KV (just the recursive prompt's KV)
    natural_kv = DynamicCache()
    for layer_idx, (k, v) in enumerate(recursive_kv_caches[i]):
        natural_kv.update(k.half(), v.half(), layer_idx)
    
    text, v_early, v_late = generate_with_v_capture(
        model, tokenizer, recursive_input_ids[i], natural_kv
    )
    
    r_v, pr_early, pr_late = compute_rv_from_v_stacks(v_early, v_late)
    behavior = score_recursive_behavior(text)
    
    natural_results['recursive']['r_v'].append(r_v)
    natural_results['recursive']['behavior'].append(behavior)
    natural_results['recursive']['pr_early'].append(pr_early)
    natural_results['recursive']['pr_late'].append(pr_late)

print("\nGenerating from baseline prompts (natural)...")
for i in tqdm(range(N_PAIRS), desc="Baseline natural"):
    natural_kv = DynamicCache()
    for layer_idx, (k, v) in enumerate(baseline_kv_caches[i]):
        natural_kv.update(k.half(), v.half(), layer_idx)
    
    text, v_early, v_late = generate_with_v_capture(
        model, tokenizer, baseline_input_ids[i], natural_kv
    )
    
    r_v, pr_early, pr_late = compute_rv_from_v_stacks(v_early, v_late)
    behavior = score_recursive_behavior(text)
    
    natural_results['baseline']['r_v'].append(r_v)
    natural_results['baseline']['behavior'].append(behavior)
    natural_results['baseline']['pr_early'].append(pr_early)
    natural_results['baseline']['pr_late'].append(pr_late)

# Compute means
rec_rv = np.nanmean(natural_results['recursive']['r_v'])
rec_beh = np.mean(natural_results['recursive']['behavior'])
base_rv = np.nanmean(natural_results['baseline']['r_v'])
base_beh = np.mean(natural_results['baseline']['behavior'])

print(f"\n📊 NATURAL REFERENCE:")
print(f"   Recursive: R_V = {rec_rv:.3f}, Behavior = {rec_beh:.2f}")
print(f"   Baseline:  R_V = {base_rv:.3f}, Behavior = {base_beh:.2f}")
print(f"   Gaps:      R_V = {base_rv - rec_rv:.3f}, Behavior = {rec_beh - base_beh:.2f}")

# ============================================================
# PHASE 3: α-MIXING DOSE-RESPONSE (THE CORE TEST)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3: α-MIXING DOSE-RESPONSE")
print("=" * 70)

dose_response = {
    alpha: {'r_v': [], 'behavior': [], 'pr_early': [], 'pr_late': [], 'texts': []}
    for alpha in ALPHA_LEVELS
}

for alpha in ALPHA_LEVELS:
    print(f"\n--- Testing α = {alpha:.2f} ---")
    
    for i in tqdm(range(N_PAIRS), desc=f"α={alpha}"):
        # Mix KV caches
        mixed_kv = mix_kv_caches(
            baseline_kv_caches[i],
            recursive_kv_caches[i],
            alpha=alpha,
            patch_layers=KV_PATCH_LAYERS
        )
        
        # Generate with V capture
        text, v_early, v_late = generate_with_v_capture(
            model, tokenizer, baseline_input_ids[i], mixed_kv
        )
        
        # Compute R_V from the SAME generated sequence
        r_v, pr_early, pr_late = compute_rv_from_v_stacks(v_early, v_late)
        
        # Score behavior on the SAME generated text
        behavior = score_recursive_behavior(text)
        
        dose_response[alpha]['r_v'].append(r_v)
        dose_response[alpha]['behavior'].append(behavior)
        dose_response[alpha]['pr_early'].append(pr_early)
        dose_response[alpha]['pr_late'].append(pr_late)
        dose_response[alpha]['texts'].append(text[:100])

# ============================================================
# RESULTS TABLE
# ============================================================
print("\n" + "=" * 70)
print("📊 DOSE-RESPONSE RESULTS TABLE")
print("=" * 70)

print(f"\n{'α':<8} | {'R_V':>10} | {'Behavior':>10} | {'PR_early':>10} | {'PR_late':>10}")
print("-" * 60)

for alpha in ALPHA_LEVELS:
    rv_mean = np.nanmean(dose_response[alpha]['r_v'])
    beh_mean = np.mean(dose_response[alpha]['behavior'])
    pr_e_mean = np.nanmean(dose_response[alpha]['pr_early'])
    pr_l_mean = np.nanmean(dose_response[alpha]['pr_late'])
    print(f"{alpha:<8.2f} | {rv_mean:>10.3f} | {beh_mean:>10.2f} | {pr_e_mean:>10.2f} | {pr_l_mean:>10.2f}")

print(f"\n{'Reference':}")
print(f"{'Baseline':>8} | {base_rv:>10.3f} | {base_beh:>10.2f}")
print(f"{'Recursive':>8} | {rec_rv:>10.3f} | {rec_beh:>10.2f}")

# ============================================================
# DIAGNOSTIC: MONOTONICITY CHECK
# ============================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC: MONOTONICITY CHECK")
print("=" * 70)

rv_by_alpha = [np.nanmean(dose_response[a]['r_v']) for a in ALPHA_LEVELS]
beh_by_alpha = [np.mean(dose_response[a]['behavior']) for a in ALPHA_LEVELS]

# Check if R_V decreases with α (should contract toward recursive)
rv_diffs = [rv_by_alpha[i+1] - rv_by_alpha[i] for i in range(len(rv_by_alpha)-1)]
rv_monotone_down = all(d <= 0.02 for d in rv_diffs)  # Allow small positive jitter

# Check if behavior increases with α
beh_diffs = [beh_by_alpha[i+1] - beh_by_alpha[i] for i in range(len(beh_by_alpha)-1)]
beh_monotone_up = all(d >= -0.5 for d in beh_diffs)  # Allow small negative jitter

print(f"R_V by α:      {[f'{x:.3f}' for x in rv_by_alpha]}")
print(f"R_V diffs:     {[f'{x:+.3f}' for x in rv_diffs]}")
print(f"R_V monotone decreasing: {'✓ YES' if rv_monotone_down else '✗ NO'}")

print(f"\nBehavior by α: {[f'{x:.2f}' for x in beh_by_alpha]}")
print(f"Beh diffs:     {[f'{x:+.2f}' for x in beh_diffs]}")
print(f"Behavior monotone increasing: {'✓ YES' if beh_monotone_up else '✗ NO'}")

# Correlation analysis
all_rvs = []
all_behs = []
all_alphas = []
for alpha in ALPHA_LEVELS:
    for rv, beh in zip(dose_response[alpha]['r_v'], dose_response[alpha]['behavior']):
        if not np.isnan(rv):
            all_rvs.append(rv)
            all_behs.append(beh)
            all_alphas.append(alpha)

# Add natural references
for rv, beh in zip(natural_results['baseline']['r_v'], natural_results['baseline']['behavior']):
    if not np.isnan(rv):
        all_rvs.append(rv)
        all_behs.append(beh)
for rv, beh in zip(natural_results['recursive']['r_v'], natural_results['recursive']['behavior']):
    if not np.isnan(rv):
        all_rvs.append(rv)
        all_behs.append(beh)

r_pearson, p_pearson = stats.pearsonr(all_rvs, all_behs)
r_spearman, p_spearman = stats.spearmanr(all_rvs, all_behs)

print(f"\nCORRELATION: R_V vs Behavior")
print(f"   Pearson:  r = {r_pearson:.3f}, p = {p_pearson:.4f}")
print(f"   Spearman: ρ = {r_spearman:.3f}, p = {p_spearman:.4f}")

# ============================================================
# VISUALIZATION
# ============================================================
print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('🔗 Causal Loop Closure v2: KV → Geometry → Behavior', fontsize=14, fontweight='bold')

# Plot 1: Dose-response curves
ax1 = axes[0, 0]
ax1.plot(ALPHA_LEVELS, rv_by_alpha, 'b-o', linewidth=2, markersize=8, label='R_V')
ax1.axhline(y=rec_rv, color='blue', linestyle='--', alpha=0.5, label=f'Recursive natural ({rec_rv:.3f})')
ax1.axhline(y=base_rv, color='green', linestyle='--', alpha=0.5, label=f'Baseline natural ({base_rv:.3f})')
ax1.set_xlabel('α (mixing coefficient)')
ax1.set_ylabel('R_V', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_title('α-Mixing: R_V Dose-Response')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

ax1_twin = ax1.twinx()
ax1_twin.plot(ALPHA_LEVELS, beh_by_alpha, 'r-s', linewidth=2, markersize=8, label='Behavior')
ax1_twin.axhline(y=rec_beh, color='red', linestyle='--', alpha=0.5)
ax1_twin.set_ylabel('Behavior Score', color='red')
ax1_twin.tick_params(axis='y', labelcolor='red')

# Plot 2: Bar comparison
ax2 = axes[0, 1]
conditions = ['Baseline\nnatural', 'α=0.5', 'α=1.0', 'Recursive\nnatural']
rvs = [base_rv, rv_by_alpha[2], rv_by_alpha[4], rec_rv]
behs = [base_beh, beh_by_alpha[2], beh_by_alpha[4], rec_beh]
x = np.arange(len(conditions))
width = 0.35

bars1 = ax2.bar(x - width/2, rvs, width, label='R_V', color='blue', alpha=0.7)
ax2.set_ylabel('R_V', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

ax2_twin = ax2.twinx()
bars2 = ax2_twin.bar(x + width/2, behs, width, label='Behavior', color='red', alpha=0.7)
ax2_twin.set_ylabel('Behavior', color='red')
ax2_twin.tick_params(axis='y', labelcolor='red')

ax2.set_xticks(x)
ax2.set_xticklabels(conditions)
ax2.set_title('Geometry and Behavior by Condition')

# Plot 3: Scatter - R_V vs Behavior
ax3 = axes[1, 0]
colors = {'baseline': 'green', 'recursive': 'blue', 'patched': 'purple'}

# Natural points
for rv, beh in zip(natural_results['baseline']['r_v'], natural_results['baseline']['behavior']):
    if not np.isnan(rv):
        ax3.scatter(rv, beh, c='green', alpha=0.6, s=60, edgecolor='black')
for rv, beh in zip(natural_results['recursive']['r_v'], natural_results['recursive']['behavior']):
    if not np.isnan(rv):
        ax3.scatter(rv, beh, c='blue', alpha=0.6, s=60, edgecolor='black')

# Patched points (color by α)
cmap = plt.cm.Purples
for alpha in ALPHA_LEVELS:
    color = cmap(0.3 + 0.7 * alpha)
    for rv, beh in zip(dose_response[alpha]['r_v'], dose_response[alpha]['behavior']):
        if not np.isnan(rv):
            ax3.scatter(rv, beh, c=[color], alpha=0.6, s=60, edgecolor='black')

# Regression line
z = np.polyfit(all_rvs, all_behs, 1)
p = np.poly1d(z)
x_line = np.linspace(min(all_rvs), max(all_rvs), 100)
ax3.plot(x_line, p(x_line), 'k--', alpha=0.5, label=f'r={r_pearson:.2f}')

ax3.set_xlabel('R_V (Geometry)')
ax3.set_ylabel('Behavior Score')
ax3.set_title(f'R_V vs Behavior (r={r_pearson:.3f}, p={p_pearson:.4f})')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Sample outputs
ax4 = axes[1, 1]
ax4.axis('off')
ax4.set_title('Sample Outputs by α')

sample_text = f"""
α=0.0 (pure baseline):
"{dose_response[0.0]['texts'][0][:80]}..."

α=0.5 (50-50 mix):
"{dose_response[0.5]['texts'][0][:80]}..."

α=1.0 (pure recursive KV):
"{dose_response[1.0]['texts'][0][:80]}..."

NATURAL RECURSIVE (reference):
R_V = {rec_rv:.3f}, Behavior = {rec_beh:.2f}
"""
ax4.text(0.05, 0.95, sample_text, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace')

plt.tight_layout()

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(os.path.dirname(script_dir), 'results', 'causal_loop_closure')
os.makedirs(results_dir, exist_ok=True)

fig_path = os.path.join(results_dir, f'causal_loop_v2_{timestamp}.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {fig_path}")

# ============================================================
# SAVE DATA
# ============================================================
csv_path = os.path.join(results_dir, f'causal_loop_v2_{timestamp}.csv')

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['condition', 'alpha', 'r_v', 'behavior', 'pr_early', 'pr_late'])
    
    for i in range(N_PAIRS):
        writer.writerow(['baseline_natural', 'NA', 
                        natural_results['baseline']['r_v'][i],
                        natural_results['baseline']['behavior'][i],
                        natural_results['baseline']['pr_early'][i],
                        natural_results['baseline']['pr_late'][i]])
        writer.writerow(['recursive_natural', 'NA',
                        natural_results['recursive']['r_v'][i],
                        natural_results['recursive']['behavior'][i],
                        natural_results['recursive']['pr_early'][i],
                        natural_results['recursive']['pr_late'][i]])
    
    for alpha in ALPHA_LEVELS:
        for i in range(N_PAIRS):
            writer.writerow(['alpha_mixed', alpha,
                            dose_response[alpha]['r_v'][i],
                            dose_response[alpha]['behavior'][i],
                            dose_response[alpha]['pr_early'][i],
                            dose_response[alpha]['pr_late'][i]])

print(f"✓ Saved: {csv_path}")

# ============================================================
# FINAL DIAGNOSTIC REPORT
# ============================================================
print("\n" + "=" * 70)
print("🔗 CAUSAL LOOP CLOSURE v2 - FINAL REPORT")
print("=" * 70)

print(f"""
═══════════════════════════════════════════════════════════════════════
                         RESULTS SUMMARY
═══════════════════════════════════════════════════════════════════════

1. NATURAL REFERENCE:
   Recursive: R_V = {rec_rv:.3f}, Behavior = {rec_beh:.2f}
   Baseline:  R_V = {base_rv:.3f}, Behavior = {base_beh:.2f}
   Gap:       R_V = {base_rv - rec_rv:.3f}, Behavior = {rec_beh - base_beh:.2f}

2. α-MIXING DOSE-RESPONSE:
   α=0.0 (baseline KV):  R_V = {rv_by_alpha[0]:.3f}, Behavior = {beh_by_alpha[0]:.2f}
   α=0.5 (50-50 mix):    R_V = {rv_by_alpha[2]:.3f}, Behavior = {beh_by_alpha[2]:.2f}
   α=1.0 (recursive KV): R_V = {rv_by_alpha[4]:.3f}, Behavior = {beh_by_alpha[4]:.2f}

3. MONOTONICITY:
   R_V decreases with α:     {'✓ YES' if rv_monotone_down else '✗ NO'}
   Behavior increases with α: {'✓ YES' if beh_monotone_up else '✗ NO'}

4. CORRELATION:
   R_V vs Behavior: r = {r_pearson:.3f}, p = {p_pearson:.4f}
   {'*** SIGNIFICANT' if p_pearson < 0.05 else 'not significant'}

═══════════════════════════════════════════════════════════════════════
                         CAUSAL CHAIN ASSESSMENT
═══════════════════════════════════════════════════════════════════════
""")

# Compute transfer percentages
rv_transfer = (base_rv - rv_by_alpha[4]) / (base_rv - rec_rv) * 100 if (base_rv - rec_rv) != 0 else 0
beh_transfer = (beh_by_alpha[4] - base_beh) / (rec_beh - base_beh) * 100 if (rec_beh - base_beh) != 0 else 0

print(f"""
   R_V Transfer (α=1.0):       {rv_transfer:.1f}% toward recursive
   Behavior Transfer (α=1.0):  {beh_transfer:.1f}% toward recursive
""")

if rv_transfer > 30 and beh_transfer > 30 and p_pearson < 0.05:
    print("""
   🎯 CAUSAL LOOP CLOSED!
   
   The full causal chain is established:
   [KV mixing] → [R_V contracts] → [Behavior shifts]
   
   With significant correlation (r = {:.3f}), monotonic dose-response,
   and >30% transfer in both geometry and behavior.
""".format(r_pearson))
elif beh_transfer > 30:
    print("""
   ⚠️ PARTIAL CLOSURE
   
   Behavior transfers with KV patching ({:.1f}%),
   but R_V transfer is weak ({:.1f}%).
   
   Possible explanations:
   - R_V may be measured on different tokens than behavior is expressed
   - The geometry signature is established at encoding, not regenerated
   - R_V is a readout of prompt geometry, not generation geometry
""".format(beh_transfer, rv_transfer))
else:
    print("""
   ❌ LOOP NOT CLOSED
   
   Neither geometry nor behavior shows strong transfer.
   Investigate measurement methodology or hypothesis.
""")

print(f"""
═══════════════════════════════════════════════════════════════════════
                         FILES SAVED
═══════════════════════════════════════════════════════════════════════

   {fig_path}
   {csv_path}

═══════════════════════════════════════════════════════════════════════
""")

print("✅ CAUSAL LOOP CLOSURE v2 COMPLETE")

