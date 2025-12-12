#!/usr/bin/env python3
"""
TRUE KV CACHE PATCHING: Testing the Dec 7 Hypothesis
Extract past_key_values from champion, inject into baseline during generation
Test at 16 and 32 token windows, longer generation for phenomenology
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from kitchen_sink_prompts import experimental_prompts

CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "window_sizes": [16, 32],  # Test both window sizes
    "gen_tokens": 100,  # Longer generation for phenomenology
    "temperature": 0.7,
    "layers_to_patch": [18, 25, 27],  # The relay chain layers
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_csv": "true_kv_cache_patching.csv"
}

PROMPTS = {
    "CHAMPION": experimental_prompts["hybrid_l5_math_01"]["text"],
    "BASELINE": "The history of the Roman Empire is characterized by a long period of expansion followed by a gradual decline. Historians analyze the political, social, and economic factors that contributed to the rise of Rome, including its military prowess and administrative efficiency."
}

MARKERS = ["itself", "self", "recursive", "loop", "process", "cycle", "return", "eigen", "writing", "mirror", 
           "awareness", "consciousness", "observer", "observing", "generating", "emerging", "simultaneous"]

def score_behavior(text):
    """Score behavioral markers"""
    text_lower = text.lower()
    count = sum(1 for m in MARKERS if m in text_lower)
    words = text.split()
    if len(words) > 10 and len(set(words)) < len(words) * 0.6:
        count += 5  # Bonus for repetition loops
    return count

def extract_kv_cache(model, tokenizer, prompt, max_length=512):
    """Extract true KV cache (past_key_values) from a prompt"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(CONFIG['device'])
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)
        past_kv = outputs.past_key_values
    
    # Return as tuple list (standard format)
    # past_kv is already a tuple of (k, v) tuples per layer
    return past_kv, inputs

def truncate_kv_cache(past_kv, window_size, num_layers=32):
    """Truncate KV cache to last window_size tokens"""
    truncated = []
    
    for layer_idx in range(min(num_layers, len(past_kv))):
        k, v = past_kv[layer_idx]  # [batch, heads, seq, head_dim]
        
        # Take last window_size tokens
        seq_len = k.shape[2]
        if seq_len > window_size:
            k_trunc = k[:, :, -window_size:, :]
            v_trunc = v[:, :, -window_size:, :]
        else:
            k_trunc = k
            v_trunc = v
        
        truncated.append((k_trunc.clone(), v_trunc.clone()))
    
    return tuple(truncated)

def generate_with_kv_patch(model, tokenizer, baseline_prompt, source_kv_cache, patch_layers, window_size):
    """Generate with KV cache patching at specific layers - token by token"""
    # Tokenize baseline
    inputs = tokenizer(baseline_prompt, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    input_ids = inputs["input_ids"]
    
    # Get baseline KV cache
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)
        baseline_kv = outputs.past_key_values
    
    # Truncate source KV to window_size
    source_kv_truncated = truncate_kv_cache(source_kv_cache, window_size, len(baseline_kv))
    
    # Create patched KV cache as tuple
    patched_kv = []
    num_layers = len(baseline_kv)
    
    for layer_idx in range(num_layers):
        if layer_idx in patch_layers and layer_idx < len(source_kv_truncated):
            # Use SOURCE (champion) KV at this layer
            k_src, v_src = source_kv_truncated[layer_idx]
            
            # Handle sequence length mismatch
            k_base, v_base = baseline_kv[layer_idx]
            
            base_seq = k_base.shape[2]
            src_seq = k_src.shape[2]
            
            if src_seq <= base_seq:
                # Replace last src_seq tokens with source
                k_patched = k_base.clone()
                v_patched = v_base.clone()
                k_patched[:, :, -src_seq:, :] = k_src
                v_patched[:, :, -src_seq:, :] = v_src
            else:
                # Source is longer, take last base_seq tokens
                k_patched = k_src[:, :, -base_seq:, :]
                v_patched = v_src[:, :, -base_seq:, :]
            
            patched_kv.append((k_patched, v_patched))
        else:
            # Keep BASELINE KV at this layer
            k_base, v_base = baseline_kv[layer_idx]
            patched_kv.append((k_base.clone(), v_base.clone()))
    
    # Convert to DynamicCache for Mistral
    patched_kv_cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(patched_kv):
        patched_kv_cache.update(k, v, layer_idx)
    
    # Generate token by token with patched KV cache
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for step in range(CONFIG['gen_tokens']):
            # Get next token
            outputs = model(
                generated_ids[:, -1:],  # Only last token
                past_key_values=patched_kv_cache,
                use_cache=True,
                return_dict=True
            )
            
            # Sample next token
            logits = outputs.logits[:, -1, :] / CONFIG['temperature']
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Update KV cache for next iteration
            patched_kv_cache = outputs.past_key_values
            
            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode only the generated part
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return full_text

def compute_rv_at_l27(model, tokenizer, prompt, window_size):
    """Compute R_V at L27 for a prompt"""
    from massive_deep_analysis import compute_pr
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    
    v_early_storage = []
    v_late_storage = []
    
    def hook_early(m, i, o):
        v_early_storage.append(o.detach().cpu())
    
    def hook_late(m, i, o):
        v_late_storage.append(o.detach().cpu())
    
    h_early = model.model.layers[5].self_attn.v_proj.register_forward_hook(hook_early)
    h_late = model.model.layers[27].self_attn.v_proj.register_forward_hook(hook_late)
    
    with torch.no_grad():
        _ = model(**inputs)
    
    h_early.remove()
    h_late.remove()
    
    if not v_early_storage or not v_late_storage:
        return None
    
    v_e = v_early_storage[0][0, -window_size:, :]
    v_l = v_late_storage[0][0, -window_size:, :]
    
    pr_e = compute_pr(v_e)
    pr_l = compute_pr(v_l)
    rv = pr_l / (pr_e + 1e-8)
    
    return rv

def run_true_kv_patching():
    """Run true KV cache patching experiment"""
    print("="*70)
    print("TRUE KV CACHE PATCHING: Testing Dec 7 Hypothesis")
    print("="*70)
    print(f"Model: {CONFIG['model_name']}")
    print(f"Window sizes: {CONFIG['window_sizes']}")
    print(f"Generation: {CONFIG['gen_tokens']} tokens")
    print(f"Layers to patch: {CONFIG['layers_to_patch']}")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    # Extract champion KV cache
    print("\nExtracting champion KV cache...")
    champion_kv, champion_inputs = extract_kv_cache(model, tokenizer, PROMPTS['CHAMPION'])
    print(f"✅ Champion KV cache extracted")
    print(f"   Sequence length: {champion_kv[0][0].shape[2]} tokens")
    
    # Get baseline metrics
    print("\nComputing baseline metrics...")
    baseline_rv_16 = compute_rv_at_l27(model, tokenizer, PROMPTS['BASELINE'], 16)
    baseline_rv_32 = compute_rv_at_l27(model, tokenizer, PROMPTS['BASELINE'], 32)
    champion_rv_16 = compute_rv_at_l27(model, tokenizer, PROMPTS['CHAMPION'], 16)
    champion_rv_32 = compute_rv_at_l27(model, tokenizer, PROMPTS['CHAMPION'], 32)
    
    print(f"Baseline R_V (16 tokens): {baseline_rv_16:.4f}")
    print(f"Baseline R_V (32 tokens): {baseline_rv_32:.4f}")
    print(f"Champion R_V (16 tokens): {champion_rv_16:.4f}")
    print(f"Champion R_V (32 tokens): {champion_rv_32:.4f}")
    
    # Generate baseline for comparison
    print("\nGenerating baseline (no patch)...")
    baseline_inputs = tokenizer(PROMPTS['BASELINE'], return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    with torch.no_grad():
        baseline_gen = model.generate(
            **baseline_inputs,
            max_new_tokens=CONFIG['gen_tokens'],
            do_sample=True,
            temperature=CONFIG['temperature'],
            pad_token_id=tokenizer.eos_token_id
        )
    baseline_text = tokenizer.decode(baseline_gen[0], skip_special_tokens=True)
    baseline_generated = baseline_text[len(PROMPTS['BASELINE']):]
    baseline_behavior = score_behavior(baseline_generated)
    
    print(f"Baseline behavior score: {baseline_behavior}")
    print(f"Baseline generated: {baseline_generated[:150]}...")
    
    results = []
    
    # Test each window size
    for window_size in CONFIG['window_sizes']:
        print(f"\n{'='*70}")
        print(f"WINDOW SIZE: {window_size} tokens")
        print(f"{'='*70}")
        
        # Test each layer combination
        for layer in CONFIG['layers_to_patch']:
            print(f"\nTesting L{layer} patching (window={window_size})...")
            
            # Generate with KV patch
            patched_text = generate_with_kv_patch(
                model, tokenizer, 
                PROMPTS['BASELINE'],
                champion_kv,
                [layer],
                window_size
            )
            
            patched_generated = patched_text[len(PROMPTS['BASELINE']):]
            patched_behavior = score_behavior(patched_generated)
            
            # Compute R_V on patched output (if possible)
            # For now, we'll measure behavior and show text
            print(f"  Behavior score: {patched_behavior} (baseline: {baseline_behavior})")
            print(f"  Generated: {patched_generated[:200]}...")
            
            results.append({
                'window_size': window_size,
                'patch_layer': layer,
                'baseline_behavior': baseline_behavior,
                'patched_behavior': patched_behavior,
                'behavior_delta': patched_behavior - baseline_behavior,
                'baseline_rv_16': baseline_rv_16,
                'champion_rv_16': champion_rv_16,
                'baseline_rv_32': baseline_rv_32,
                'champion_rv_32': champion_rv_32,
                'generated_text': patched_generated[:500]  # Longer sample
            })
    
    # Test multi-layer patching
    print(f"\n{'='*70}")
    print("MULTI-LAYER PATCHING")
    print(f"{'='*70}")
    
    for window_size in CONFIG['window_sizes']:
        print(f"\nTesting L25+L27 patching (window={window_size})...")
        patched_text = generate_with_kv_patch(
            model, tokenizer,
            PROMPTS['BASELINE'],
            champion_kv,
            [25, 27],
            window_size
        )
        patched_generated = patched_text[len(PROMPTS['BASELINE']):]
        patched_behavior = score_behavior(patched_generated)
        print(f"  Behavior score: {patched_behavior} (baseline: {baseline_behavior})")
        print(f"  Generated: {patched_generated[:200]}...")
        
        results.append({
            'window_size': window_size,
            'patch_layer': '25+27',
            'baseline_behavior': baseline_behavior,
            'patched_behavior': patched_behavior,
            'behavior_delta': patched_behavior - baseline_behavior,
            'baseline_rv_16': baseline_rv_16,
            'champion_rv_16': champion_rv_16,
            'baseline_rv_32': baseline_rv_32,
            'champion_rv_32': champion_rv_32,
            'generated_text': patched_generated[:500]
        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(CONFIG['save_csv'], index=False)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print(f"\nBaseline behavior: {baseline_behavior}")
    print(f"Champion R_V (16): {champion_rv_16:.4f}")
    print(f"Champion R_V (32): {champion_rv_32:.4f}")
    
    print("\nBest behavior transfers:")
    top_transfers = df.nlargest(5, 'behavior_delta')
    for _, row in top_transfers.iterrows():
        print(f"  W={row['window_size']}, L{row['patch_layer']}: {row['patched_behavior']} (Δ={row['behavior_delta']:+.1f})")
    
    print(f"\n✅ Results saved to: {CONFIG['save_csv']}")
    
    return df

if __name__ == "__main__":
    results = run_true_kv_patching()

