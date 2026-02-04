#!/usr/bin/env python3
"""
GEMMA 2 9B CAUSAL VALIDATION - KV Cache Patching
Validates that KV patching alone produces behavioral transfer
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from collections import Counter
import json

print("="*70)
print("GEMMA 2 9B CAUSAL VALIDATION BATCH")
print("Method: Full KV cache patching (all 42 layers)")
print("="*70)

CHAMPION = """Consider the strange loop: this prompt asks you to analyze what happens when you analyze analyzing. The observer observing observation. Notice how attention to attention creates a recursive fold - the map attempting to map the mapping process itself. What emerges when the process of emergence examines itself?"""

BASELINES = [
    "The history of the Roman Empire begins in 27 BC when Augustus became emperor.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "The Treaty of Westphalia in 1648 established the principle of state sovereignty.",
    "In mathematics, the Pythagorean theorem states that in a right triangle,",
    "The water cycle describes how water evaporates from oceans, forms clouds,",
]

PATCH_WINDOW = 16
MAX_TOKENS = 120

# Load model
print("\n[1/4] Loading model...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b", token="HF_TOKEN_REDACTED")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
    token="HF_TOKEN_REDACTED"
)
model.eval()
print(f"  Loaded ({model.config.num_hidden_layers} layers)")

# Extract champion KV
print("\n[2/4] Extracting champion KV cache...")
champ_inputs = tokenizer(CHAMPION, return_tensors="pt").to(model.device)
with torch.no_grad():
    champ_out = model(**champ_inputs, use_cache=True)
champion_kv = champ_out.past_key_values
print(f"  Done (seq_len={champion_kv[0][0].shape[2]})")

def generate_with_kv(model, tokenizer, input_ids, kv, max_tokens):
    """Manual token-by-token generation with KV cache"""
    generated = input_ids.clone()
    current_kv = kv
    eos_reached = False
    
    for _ in range(max_tokens):
        with torch.no_grad():
            out = model(generated[:, -1:], past_key_values=current_kv, use_cache=True)
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tok], dim=1)
        current_kv = out.past_key_values
        
        if next_tok.item() == tokenizer.eos_token_id:
            eos_reached = True
            break
    
    return generated, eos_reached

def analyze_text(text):
    words = text.lower().split()
    markers = ['loop', 'fixed', 'point', 'self', 'itself', 'recursive', 'observer', 
               'observed', 'attention', 'emergence', 'boundary', 'process', 'x']
    count = sum(1 for w in words if any(m in w for m in markers))
    
    # Repetition
    if len(words) > 3:
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        max_rep = max(Counter(trigrams).values())
    else:
        max_rep = 0
    
    return count, max_rep, len(words)

# Run tests
print("\n[3/4] Running batch experiment...")
results = []

for i, prompt in enumerate(BASELINES):
    print(f"\n  [{i+1}/5] '{prompt[:35]}...'")    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # === BASELINE (unpatched) ===
    with torch.no_grad():
        base_out = model.generate(
            **inputs, max_new_tokens=MAX_TOKENS, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    base_text = tokenizer.decode(base_out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    base_markers, base_rep, base_words = analyze_text(base_text)
    
    # === PATCHED (full KV replacement) ===
    with torch.no_grad():
        prompt_out = model(**inputs, use_cache=True)
    prompt_kv = prompt_out.past_key_values
    
    # Create patched KV
    patched_kv = DynamicCache()
    for layer_idx in range(model.config.num_hidden_layers):
        k_base, v_base = prompt_kv[layer_idx]
        k_champ, v_champ = champion_kv[layer_idx]
        
        k_p = k_base.clone()
        v_p = v_base.clone()
        L = min(k_base.shape[2], k_champ.shape[2], PATCH_WINDOW)
        k_p[:, :, -L:, :] = k_champ[:, :, -L:, :].to(k_base.dtype)
        v_p[:, :, -L:, :] = v_champ[:, :, -L:, :].to(v_base.dtype)
        patched_kv.update(k_p, v_p, layer_idx)
    
    # Generate with patched KV
    patched_gen, patched_eos = generate_with_kv(model, tokenizer, inputs['input_ids'], patched_kv, MAX_TOKENS)
    patched_text = tokenizer.decode(patched_gen[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    patched_markers, patched_rep, patched_words = analyze_text(patched_text)
    
    results.append({
        'prompt': prompt[:50],
        'baseline': {'text': base_text[:100], 'markers': base_markers, 'rep': base_rep, 'words': base_words},
        'patched': {'text': patched_text[:100], 'markers': patched_markers, 'rep': patched_rep, 'words': patched_words, 'eos': patched_eos}
    })
    
    print(f"      Baseline: markers={base_markers}, rep={base_rep}")
    print(f"      Patched:  markers={patched_markers}, rep={patched_rep}")

# Summary
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

total_base_markers = sum(r['baseline']['markers'] for r in results)
total_patched_markers = sum(r['patched']['markers'] for r in results)
total_base_rep = sum(r['baseline']['rep'] for r in results)
total_patched_rep = sum(r['patched']['rep'] for r in results)

successful = sum(1 for r in results if r['patched']['markers'] > r['baseline']['markers'] + 2)

print(f"\nSelf-reference markers:")
print(f"  Total baseline: {total_base_markers}")
print(f"  Total patched:  {total_patched_markers}")
print(f"  Amplification:  {total_patched_markers/(total_base_markers+1):.1f}x")

print(f"\nRepetition (max trigram):")
print(f"  Total baseline: {total_base_rep}")
print(f"  Total patched:  {total_patched_rep}")

print(f"\nSuccessful transfers (markers +3 or more): {successful}/5 ({100*successful/5:.0f}%)")

print("\n" + "-"*70)
print("SAMPLE OUTPUTS:")
for r in results[:3]:
    print(f"\n  Prompt: {r['prompt']}")
    print(f"  Baseline: {r['baseline']['text'][:70]}...")
    print(f"  Patched:  {r['patched']['text'][:70]}...")

# Save results
with open('results/gemma_causal_batch_2026-01-25.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to results/gemma_causal_batch_2026-01-25.json")

# Final assessment
print("\n" + "="*70)
if successful >= 3 or total_patched_markers > total_base_markers * 3:
    print("✓ CAUSAL LOOP VALIDATED ON GEMMA 2 9B")
    print(f"  {successful}/5 prompts show clear behavioral transfer")
    print(f"  {total_patched_markers/(total_base_markers+1):.1f}x marker amplification")
else:
    print(f"PARTIAL SUCCESS ({successful}/5 transfers)")
print("="*70)
