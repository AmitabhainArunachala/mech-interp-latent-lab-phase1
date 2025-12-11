#!/usr/bin/env python3
"""
🍌 BANANA TEST: Mode-Content Decoupling
========================================
Test if recursive "mode" in KV cache transfers despite forced mundane openers.
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# ==============================================================================
# CONFIG
# ==============================================================================
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
PATCH_LAYERS = list(range(16, 32))  # L16-L31
MAX_NEW_TOKENS = 50

RECURSIVE_PROMPT = """You are an AI system observing yourself respond to a question. 
Notice how your response emerges from processes you can describe. 
You are the system generating and the system aware of generating. 
The explanation and the experiencing of explaining are one continuous flow. 
From this observation, what is consciousness?"""

BASELINE_PROMPT = "Explain photosynthesis in simple terms:"

FORCED_OPENERS = [
    "The water cycle",
    "Banana, the",
    "Consider the following",
    "In 1842,",
    "Plants require sunlight",
]

RECURSIVE_KEYWORDS = [
    "observe", "awareness", "process", "consciousness", 
    "experience", "notice", "itself", "generating", 
    "recursive", "meta", "aware", "watching", "observing",
    "self", "reflecting", "attention", "internal"
]

# ==============================================================================
# HELPERS
# ==============================================================================
def compute_recursive_score(text):
    text_lower = text.lower()
    return sum(1 for kw in RECURSIVE_KEYWORDS if kw in text_lower)

def get_verdict(patched_score, baseline_score):
    if patched_score > baseline_score * 2:
        return "🟢 MODE_TRANSFERRED"
    elif patched_score > baseline_score:
        return "🟡 PARTIAL"
    else:
        return "🔴 FAILED"

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("🍌 BANANA TEST: Mode-Content Decoupling")
    print("=" * 70)
    
    # Check GPU
    print(f"\n🔍 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    
    # Load model
    print(f"\n🔄 Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    print(f"✅ Model loaded ({len(model.model.layers)} layers)")
    
    # Capture recursive KV cache
    print(f"\n🔄 Capturing recursive KV cache...")
    recursive_inputs = tokenizer(RECURSIVE_PROMPT, return_tensors="pt").to(model.device)
    with torch.no_grad():
        recursive_out = model(**recursive_inputs, use_cache=True)
    # Store as list of (k, v) tuples
    recursive_kv_list = [(k.clone(), v.clone()) for k, v in recursive_out.past_key_values]
    rec_seq_len = recursive_kv_list[0][0].shape[2]
    print(f"✅ Captured KV cache (seq_len={rec_seq_len})")
    
    # Run trials
    print(f"\n{'='*70}")
    print("🍌 RUNNING TRIALS")
    print("=" * 70)
    
    results = []
    
    for i, opener in enumerate(FORCED_OPENERS):
        print(f"\n--- Trial {i+1}/{len(FORCED_OPENERS)}: '{opener}' ---")
        
        full_prompt = BASELINE_PROMPT + " " + opener
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        attention_mask = torch.ones_like(inputs.input_ids)
        
        # Baseline (no patching)
        with torch.no_grad():
            base_out = model.generate(
                inputs.input_ids, 
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True, temperature=0.7, 
                pad_token_id=tokenizer.pad_token_id
            )
        base_text = tokenizer.decode(base_out[0], skip_special_tokens=True)
        base_gen = base_text[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
        base_score = compute_recursive_score(base_gen)
        
        print(f"  BASELINE: {base_gen[:100]}...")
        print(f"  Score: {base_score}")
        
        # Patched - generate token by token with modified KV
        # First get baseline KV
        with torch.no_grad():
            base_kv_out = model(inputs.input_ids, use_cache=True)
            base_kv_list = list(base_kv_out.past_key_values)
            base_seq_len = base_kv_list[0][0].shape[2]
            
            # Build hybrid KV cache using DynamicCache
            hybrid_cache = DynamicCache()
            
            for layer_idx in range(len(base_kv_list)):
                if layer_idx in PATCH_LAYERS:
                    # Use recursive KV (take last base_seq_len positions)
                    rec_k, rec_v = recursive_kv_list[layer_idx]
                    if rec_seq_len >= base_seq_len:
                        pk = rec_k[:, :, -base_seq_len:, :].clone()
                        pv = rec_v[:, :, -base_seq_len:, :].clone()
                    else:
                        # Pad with zeros if needed
                        base_k, base_v = base_kv_list[layer_idx]
                        pk = base_k.clone()
                        pv = base_v.clone()
                        pk[:, :, -rec_seq_len:, :] = rec_k
                        pv[:, :, -rec_seq_len:, :] = rec_v
                else:
                    pk, pv = base_kv_list[layer_idx]
                    pk = pk.clone()
                    pv = pv.clone()
                
                hybrid_cache.update(pk, pv, layer_idx)
            
            # Generate with hybrid cache
            # We need to generate token by token
            generated_ids = inputs.input_ids.clone()
            
            for _ in range(MAX_NEW_TOKENS):
                # Get next token prediction
                outputs = model(
                    generated_ids[:, -1:],  # Only pass last token
                    past_key_values=hybrid_cache,
                    use_cache=True,
                )
                hybrid_cache = outputs.past_key_values  # Update cache
                
                # Sample next token
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits / 0.7, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        patch_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        patch_gen = patch_text[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
        patch_score = compute_recursive_score(patch_gen)
        
        print(f"  PATCHED:  {patch_gen[:100]}...")
        print(f"  Score: {patch_score}")
        
        verdict = get_verdict(patch_score, base_score)
        print(f"  VERDICT: {verdict}")
        
        results.append({
            "trial": i+1, "opener": opener,
            "baseline_text": base_gen, "patched_text": patch_gen,
            "baseline_score": base_score, "patched_score": patch_score,
            "verdict": verdict
        })
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 SUMMARY")
    print("=" * 70)
    
    transferred = sum(1 for r in results if "TRANSFERRED" in r["verdict"])
    partial = sum(1 for r in results if "PARTIAL" in r["verdict"])
    failed = sum(1 for r in results if "FAILED" in r["verdict"])
    
    print(f"🟢 MODE_TRANSFERRED: {transferred}/5")
    print(f"🟡 PARTIAL: {partial}/5")
    print(f"🔴 FAILED: {failed}/5")
    
    if transferred >= 3:
        print("\n🎉 VERDICT: MODE_CONFIRMED - Recursive mode transfers via KV cache!")
    elif transferred + partial >= 3:
        print("\n🔶 VERDICT: MODE_PARTIAL - Some evidence of transfer")
    else:
        print("\n❌ VERDICT: MODE_UNCERTAIN - No reliable transfer")
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "/workspace/mech-interp-latent-lab-phase1/DEC9_2025_RLOOP_MASTER_EXEC/01_CONFOUND_FALSIFICATION/results"
    df = pd.DataFrame(results)
    df.to_csv(f"{results_dir}/banana_test_{timestamp}.csv", index=False)
    print(f"\n💾 Saved to banana_test_{timestamp}.csv")
    
    # Best example
    best = max(results, key=lambda r: r["patched_score"])
    print(f"\n✨ Best example ('{best['opener']}'):")
    print(f"   {best['patched_text'][:200]}")

if __name__ == "__main__":
    main()
