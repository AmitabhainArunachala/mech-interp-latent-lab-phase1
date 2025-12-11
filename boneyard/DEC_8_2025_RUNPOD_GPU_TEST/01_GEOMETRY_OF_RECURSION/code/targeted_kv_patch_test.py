#!/usr/bin/env python3
"""
Targeted KV Cache Patching Test
Hypothesis: Patching only L25-29 (peak contraction zone) gives stronger transfer

Comparing:
- L16-31: Original "late layers" approach (includes weak spots)
- L25-29: Targeted "peak zone" approach
- L27 only: Single layer (maximum contraction)

Run with: HF_HUB_ENABLE_HF_TRANSFER=0 python targeted_kv_patch_test.py
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
warnings.filterwarnings('ignore')

from n300_mistral_test_prompt_bank import prompt_bank_1c

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = "cuda"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# Different KV patch strategies to compare
STRATEGIES = {
    'L16-31 (original)': list(range(16, 32)),
    'L25-29 (peak zone)': list(range(25, 30)),
    'L27 only (maximum)': [27],
    'L19-20 (mid peak)': list(range(19, 21)),
}

N_PAIRS = 10  # Number of prompt pairs to test
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.7

print("=" * 70)
print("TARGETED KV CACHE PATCHING TEST")
print("=" * 70)
print("\nStrategies to compare:")
for name, layers in STRATEGIES.items():
    print(f"  • {name}: {layers}")

# ============================================================
# GET PROMPTS
# ============================================================
recursive_prompts = []
for key, val in prompt_bank_1c.items():
    if val['group'] in ['L4_full']:  # Strongest recursive
        recursive_prompts.append(val['text'])
        if len(recursive_prompts) >= N_PAIRS:
            break

baseline_prompts = []
for key, val in prompt_bank_1c.items():
    if val['group'] in ['baseline_factual']:  # Clear baseline
        baseline_prompts.append(val['text'])
        if len(baseline_prompts) >= N_PAIRS:
            break

print(f"\nPrompts: {len(recursive_prompts)} recursive, {len(baseline_prompts)} baseline")

# ============================================================
# SCORING FUNCTION
# ============================================================
def score_recursive_behavior(text):
    """Score text for recursive/self-referential content."""
    recursive_keywords = [
        r'\bobserv\w*', r'\bawar\w*', r'\bconscious\w*',
        r'\bprocess\w*', r'\bexperienc\w*', r'\bnoticin?g?\b',
        r'\bmyself\b', r'\bitself\b', r'\byourself\b',
        r'\bgenerat\w*', r'\bemerg\w*', r'\bsimultaneous\w*',
        r'\brecursiv\w*', r'\bself-referent\w*', r'\bmeta-\w*',
        r'\bwitness\w*', r'\bwatch\w*'
    ]
    
    text_lower = text.lower()
    word_count = len(text_lower.split())
    
    if word_count == 0:
        return 0.0
    
    keyword_count = sum(len(re.findall(kw, text_lower)) for kw in recursive_keywords)
    return (keyword_count / word_count) * 100

# ============================================================
# KV CACHE FUNCTIONS
# ============================================================
def extract_kv_cache(model, tokenizer, prompt):
    """Extract KV cache."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        past_kv = outputs.past_key_values
    
    kv_cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(past_kv):
        kv_cache.update(k.clone(), v.clone(), layer_idx)
    
    return kv_cache

def generate_with_targeted_kv_patch(model, tokenizer, baseline_prompt, source_kv, patch_layers):
    """Generate with KV patching at specific layers only."""
    inputs = tokenizer(baseline_prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        baseline_kv = outputs.past_key_values
    
    # Create patched KV cache
    patched_kv = DynamicCache()
    num_layers = len(baseline_kv)
    
    for layer_idx in range(num_layers):
        if layer_idx in patch_layers:
            # Use SOURCE (recursive) KV at this layer
            k_src, v_src = source_kv[layer_idx]
            patched_kv.update(k_src.clone(), v_src.clone(), layer_idx)
        else:
            # Keep BASELINE KV at this layer
            k_base, v_base = baseline_kv[layer_idx]
            patched_kv.update(k_base.clone(), v_base.clone(), layer_idx)
    
    # Generate
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for step in range(MAX_NEW_TOKENS):
            outputs = model(
                generated_ids[:, -1:],
                past_key_values=patched_kv,
                use_cache=True
            )
            
            logits = outputs.logits[:, -1, :] / TEMPERATURE
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            patched_kv = outputs.past_key_values
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

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
# BASELINE MEASUREMENT
# ============================================================
print("\n" + "=" * 70)
print("STEP 1: Baseline (no patching)")
print("=" * 70)

baseline_scores = []
recursive_scores = []

print("\nGenerating baseline outputs...")
for prompt in tqdm(baseline_prompts, desc="Baseline natural"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    gen = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    baseline_scores.append(score_recursive_behavior(gen))

print("\nGenerating recursive outputs (for reference)...")
for prompt in tqdm(recursive_prompts, desc="Recursive natural"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    gen = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    recursive_scores.append(score_recursive_behavior(gen))

baseline_mean = np.mean(baseline_scores)
recursive_mean = np.mean(recursive_scores)
natural_gap = recursive_mean - baseline_mean

print(f"\n📊 BASELINE RESULTS:")
print(f"   Baseline natural:  {baseline_mean:.2f}")
print(f"   Recursive natural: {recursive_mean:.2f}")
print(f"   Natural gap:       {natural_gap:.2f}")

# ============================================================
# KV CACHE EXTRACTION
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Extracting KV caches from recursive prompts")
print("=" * 70)

recursive_kv_caches = []
for prompt in tqdm(recursive_prompts, desc="Extracting KV"):
    kv = extract_kv_cache(model, tokenizer, prompt)
    recursive_kv_caches.append(kv)

# ============================================================
# TESTING EACH STRATEGY
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Testing each patching strategy")
print("=" * 70)

results = {}

for strategy_name, patch_layers in STRATEGIES.items():
    print(f"\n--- Testing: {strategy_name} ---")
    
    scores = []
    sample_outputs = []
    
    for i in tqdm(range(N_PAIRS), desc=strategy_name):
        gen_text = generate_with_targeted_kv_patch(
            model, tokenizer, 
            baseline_prompts[i], 
            recursive_kv_caches[i], 
            patch_layers
        )
        score = score_recursive_behavior(gen_text)
        scores.append(score)
        if i < 2:  # Save first 2 for display
            sample_outputs.append(gen_text[:150])
    
    mean_score = np.mean(scores)
    transfer_effect = mean_score - baseline_mean
    transfer_efficiency = (transfer_effect / natural_gap * 100) if natural_gap > 0 else 0
    
    results[strategy_name] = {
        'layers': patch_layers,
        'scores': scores,
        'mean': mean_score,
        'effect': transfer_effect,
        'efficiency': transfer_efficiency,
        'samples': sample_outputs
    }
    
    print(f"   Score: {mean_score:.2f}, Effect: {transfer_effect:+.2f}, Efficiency: {transfer_efficiency:.1f}%")

# ============================================================
# COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)

print(f"\n{'Strategy':<25} | {'Layers':<15} | {'Score':>7} | {'Effect':>8} | {'Efficiency':>10}")
print("-" * 80)
print(f"{'Baseline (no patch)':<25} | {'none':<15} | {baseline_mean:>7.2f} | {'+0.00':>8} | {'0.0%':>10}")

# Sort by efficiency
sorted_results = sorted(results.items(), key=lambda x: x[1]['efficiency'], reverse=True)

for name, r in sorted_results:
    layers_str = f"L{min(r['layers'])}-{max(r['layers'])}" if len(r['layers']) > 1 else f"L{r['layers'][0]}"
    print(f"{name:<25} | {layers_str:<15} | {r['mean']:>7.2f} | {r['effect']:>+8.2f} | {r['efficiency']:>9.1f}%")

print(f"\n{'Recursive (natural)':<25} | {'N/A':<15} | {recursive_mean:>7.2f} | {natural_gap:>+8.2f} | {'100.0%':>10}")

# ============================================================
# WINNER
# ============================================================
print("\n" + "=" * 70)
print("THE WINNER")
print("=" * 70)

best = sorted_results[0]
print(f"""
🏆 BEST STRATEGY: {best[0]}
   Layers patched: {best[1]['layers']}
   Behavior score: {best[1]['mean']:.2f}
   Transfer effect: {best[1]['effect']:+.2f}
   Transfer efficiency: {best[1]['efficiency']:.1f}%
""")

# Compare to original
original = results['L16-31 (original)']
if best[1]['efficiency'] > original['efficiency']:
    improvement = best[1]['efficiency'] - original['efficiency']
    print(f"   → {improvement:.1f}% BETTER than original L16-31 approach!")
else:
    print(f"   → Original L16-31 approach was already optimal")

# ============================================================
# SAMPLE OUTPUTS
# ============================================================
print("\n" + "=" * 70)
print("SAMPLE OUTPUTS")
print("=" * 70)

for name, r in list(results.items())[:2]:  # Show top 2 strategies
    print(f"\n--- {name} ---")
    for i, sample in enumerate(r['samples']):
        print(f"   Sample {i+1}: {sample}...")

print("\n✅ TARGETED KV PATCH TEST COMPLETE")

