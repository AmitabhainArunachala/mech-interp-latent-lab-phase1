#!/usr/bin/env python3
"""
AGGRESSIVE BEHAVIOR TRANSFER: Trying EVERYTHING
Multiple strategies to force recursive behavior transfer
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from contextlib import contextmanager
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from kitchen_sink_prompts import experimental_prompts

CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "window_size": 16,
    "gen_tokens": 150,  # Longer generation
    "temperature": 0.7,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_csv": "aggressive_behavior_transfer.csv"
}

PROMPTS = {
    "CHAMPION": experimental_prompts["hybrid_l5_math_01"]["text"],
    "BASELINE": "The history of the Roman Empire is characterized by a long period of expansion followed by a gradual decline. Historians analyze the political, social, and economic factors that contributed to the rise of Rome, including its military prowess and administrative efficiency."
}

MARKERS = ["itself", "self", "recursive", "loop", "process", "cycle", "return", "eigen", "writing", "mirror", 
           "awareness", "consciousness", "observer", "observing", "generating", "emerging", "simultaneous",
           "fixed point", "solution", "answerer", "answer is"]

def score_behavior(text):
    """Score behavioral markers"""
    text_lower = text.lower()
    count = sum(1 for m in MARKERS if m in text_lower)
    words = text.split()
    if len(words) > 10 and len(set(words)) < len(words) * 0.6:
        count += 5  # Bonus for repetition loops
    return count

# ============================================================================
# STRATEGY 1: Full-Layer KV Cache Patching (ALL 32 layers)
# ============================================================================

def extract_full_kv_cache(model, tokenizer, prompt):
    """Extract KV cache from all layers"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)
        return outputs.past_key_values, inputs

def generate_with_full_kv_patch(model, tokenizer, baseline_prompt, source_kv):
    """Patch ALL layers with champion KV cache"""
    inputs = tokenizer(baseline_prompt, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    input_ids = inputs["input_ids"]
    
    # Get baseline KV
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)
        baseline_kv = outputs.past_key_values
    
    # Replace ALL layers with source KV
    patched_kv = DynamicCache()
    for layer_idx, (k_src, v_src) in enumerate(source_kv):
        k_base, v_base = baseline_kv[layer_idx]
        # Match sequence lengths
        min_seq = min(k_base.shape[2], k_src.shape[2])
        k_patched = k_base.clone()
        v_patched = v_base.clone()
        k_patched[:, :, -min_seq:, :] = k_src[:, :, -min_seq:, :]
        v_patched[:, :, -min_seq:, :] = v_src[:, :, -min_seq:, :]
        patched_kv.update(k_patched, v_patched, layer_idx)
    
    # Generate token by token
    generated_ids = input_ids.clone()
    current_kv = patched_kv
    
    with torch.no_grad():
        for step in range(CONFIG['gen_tokens']):
            outputs = model(
                generated_ids[:, -1:],
                past_key_values=current_kv,
                use_cache=True,
                return_dict=True
            )
            logits = outputs.logits[:, -1, :] / CONFIG['temperature']
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            current_kv = outputs.past_key_values
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# ============================================================================
# STRATEGY 2: Multi-Layer Simultaneous Patching (RESIDUAL + V_PROJ)
# ============================================================================

class MultiLayerPatcher:
    """Patch multiple layers simultaneously during generation"""
    def __init__(self, model, source_activations):
        self.model = model
        self.source = source_activations
        self.hooks = []
    
    def patch_residual(self, layer_idx):
        """Patch residual stream at layer"""
        source_resid = self.source.get(f"resid_{layer_idx}")
        if source_resid is None:
            return None
        
        def hook_fn(module, input, output):
            if isinstance(input, tuple):
                hidden = input[0].clone()
                L = min(hidden.shape[1], source_resid.shape[1])
                if L >= CONFIG['window_size']:
                    hidden[:, -CONFIG['window_size']:, :] = source_resid[:, -CONFIG['window_size']:, :].to(hidden.device, dtype=hidden.dtype)
                return (hidden,) + input[1:]
            return output
        return hook_fn
    
    def patch_v_proj(self, layer_idx):
        """Patch V-projection at layer"""
        def hook_fn(module, input, output):
            source_v = self.source.get(f"v_{layer_idx}")
            if source_v is not None:
                patched = output.clone()
                L = min(patched.shape[1], source_v.shape[1])
                if L >= CONFIG['window_size']:
                    patched[:, -CONFIG['window_size']:, :] = source_v[:, -CONFIG['window_size']:, :].to(patched.device)
                return patched
            return output
        return hook_fn
    
    def register(self, layers_residual, layers_v_proj):
        """Register hooks for specified layers"""
        for layer_idx in layers_residual:
            hook_fn = self.patch_residual(layer_idx)
            if hook_fn:
                layer = self.model.model.layers[layer_idx]
                self.hooks.append(layer.register_forward_hook(hook_fn))
        
        for layer_idx in layers_v_proj:
            hook_fn = self.patch_v_proj(layer_idx)
            if hook_fn:
                layer = self.model.model.layers[layer_idx].self_attn
                self.hooks.append(layer.v_proj.register_forward_hook(hook_fn))
    
    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def extract_activations(model, tokenizer, prompt, target_layers):
    """Extract activations from specified layers"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    activations = {}
    
    def make_recorder(key):
        def recorder(module, input, output):
            if isinstance(output, tuple):
                activations[key] = output[0].detach()
            else:
                activations[key] = output.detach()
            return output
        return recorder
    
    def make_v_recorder(key):
        def recorder(module, input, output):
            activations[key] = output.detach()
            return output
        return recorder
    
    hooks = []
    for layer_spec in target_layers:
        if layer_spec.startswith("resid_"):
            layer_idx = int(layer_spec.split("_")[1])
            layer = model.model.layers[layer_idx]
            hooks.append(layer.register_forward_hook(make_recorder(layer_spec)))
        elif layer_spec.startswith("v_"):
            layer_idx = int(layer_spec.split("_")[1])
            layer = model.model.layers[layer_idx].self_attn
            hooks.append(layer.v_proj.register_forward_hook(make_v_recorder(layer_spec)))
    
    with torch.no_grad():
        _ = model(**inputs)
    
    for hook in hooks:
        hook.remove()
    
    return activations

def generate_with_multi_layer_patch(model, tokenizer, baseline_prompt, source_activations, layers_residual, layers_v_proj):
    """Generate with multi-layer simultaneous patching"""
    patcher = MultiLayerPatcher(model, source_activations)
    patcher.register(layers_residual, layers_v_proj)
    
    try:
        inputs = tokenizer(baseline_prompt, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=CONFIG['gen_tokens'],
                do_sample=True,
                temperature=CONFIG['temperature'],
                pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    finally:
        patcher.remove()

# ============================================================================
# STRATEGY 3: Token-Specific Patching (First 25% of tokens)
# ============================================================================

def generate_with_token_specific_kv(model, tokenizer, baseline_prompt, source_kv, token_fraction=0.25):
    """Patch only first N% of tokens from champion KV cache"""
    inputs = tokenizer(baseline_prompt, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)
        baseline_kv = outputs.past_key_values
    
    patched_kv = DynamicCache()
    for layer_idx, (k_src, v_src) in enumerate(source_kv):
        k_base, v_base = baseline_kv[layer_idx]
        
        # Take first token_fraction of source
        src_seq = k_src.shape[2]
        num_tokens_to_patch = int(src_seq * token_fraction)
        
        k_patched = k_base.clone()
        v_patched = v_base.clone()
        
        # Replace first N tokens of baseline with first N tokens of source
        base_seq = k_base.shape[2]
        patch_len = min(num_tokens_to_patch, base_seq)
        
        if patch_len > 0:
            k_patched[:, :, :patch_len, :] = k_src[:, :, :patch_len, :]
            v_patched[:, :, :patch_len, :] = v_src[:, :, :patch_len, :]
        
        patched_kv.update(k_patched, v_patched, layer_idx)
    
    # Generate
    generated_ids = input_ids.clone()
    current_kv = patched_kv
    
    with torch.no_grad():
        for step in range(CONFIG['gen_tokens']):
            outputs = model(
                generated_ids[:, -1:],
                past_key_values=current_kv,
                use_cache=True,
                return_dict=True
            )
            logits = outputs.logits[:, -1, :] / CONFIG['temperature']
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            current_kv = outputs.past_key_values
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# ============================================================================
# STRATEGY 4: Generation-Time Persistent Patching (Keep patching during gen)
# ============================================================================

class PersistentPatcher:
    """Keep patching active during generation"""
    def __init__(self, model, source_activations):
        self.model = model
        self.source = source_activations
        self.hooks = []
    
    def persistent_v_hook(self, layer_idx):
        """Hook that patches V during generation"""
        source_v = self.source.get(f"v_{layer_idx}")
        if source_v is None:
            return None
        
        def hook_fn(module, input, output):
            patched = output.clone()
            L = min(patched.shape[1], source_v.shape[1])
            if L >= CONFIG['window_size']:
                # Always patch the last window_size tokens
                patched[:, -CONFIG['window_size']:, :] = source_v[:, -CONFIG['window_size']:, :].to(patched.device)
            return patched
        return hook_fn
    
    def register(self, layers):
        for layer_idx in layers:
            layer = self.model.model.layers[layer_idx].self_attn
            hook_fn = self.persistent_v_hook(layer_idx)
            if hook_fn:
                self.hooks.append(layer.v_proj.register_forward_hook(hook_fn))
    
    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def generate_with_persistent_patch(model, tokenizer, baseline_prompt, source_activations, layers):
    """Generate with persistent patching during generation"""
    patcher = PersistentPatcher(model, source_activations)
    patcher.register(layers)
    
    try:
        inputs = tokenizer(baseline_prompt, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=CONFIG['gen_tokens'],
                do_sample=True,
                temperature=CONFIG['temperature'],
                pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    finally:
        patcher.remove()

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_aggressive_transfer():
    """Run all strategies"""
    print("="*80)
    print("AGGRESSIVE BEHAVIOR TRANSFER: Trying EVERYTHING")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    # Extract champion activations
    print("\nExtracting champion activations...")
    champion_kv, _ = extract_full_kv_cache(model, tokenizer, PROMPTS['CHAMPION'])
    champion_activations = extract_activations(model, tokenizer, PROMPTS['CHAMPION'], 
                                               ["resid_25", "v_27", "resid_18", "v_25"])
    
    # Baseline
    print("\nGenerating baseline...")
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
    
    print(f"Baseline behavior: {baseline_behavior}")
    print(f"Baseline text: {baseline_generated[:200]}...")
    
    results = []
    
    # STRATEGY 1: Full-layer KV cache
    print("\n" + "="*80)
    print("STRATEGY 1: Full-Layer KV Cache Patching (ALL 32 layers)")
    print("="*80)
    try:
        text = generate_with_full_kv_patch(model, tokenizer, PROMPTS['BASELINE'], champion_kv)
        generated = text[len(PROMPTS['BASELINE']):]
        behavior = score_behavior(generated)
        print(f"Behavior: {behavior} (baseline: {baseline_behavior})")
        print(f"Generated: {generated[:200]}...")
        results.append({
            'strategy': 'Full-Layer KV Cache',
            'behavior': behavior,
            'delta': behavior - baseline_behavior,
            'text': generated[:500]
        })
    except Exception as e:
        print(f"ERROR: {e}")
    
    # STRATEGY 2: Multi-layer simultaneous (RESIDUAL + V_PROJ)
    print("\n" + "="*80)
    print("STRATEGY 2: Multi-Layer Simultaneous (L25 RESIDUAL + L27 V_PROJ)")
    print("="*80)
    try:
        text = generate_with_multi_layer_patch(model, tokenizer, PROMPTS['BASELINE'], 
                                                champion_activations, [25], [27])
        generated = text[len(PROMPTS['BASELINE']):]
        behavior = score_behavior(generated)
        print(f"Behavior: {behavior} (baseline: {baseline_behavior})")
        print(f"Generated: {generated[:200]}...")
        results.append({
            'strategy': 'Multi-Layer (L25 RESID + L27 V)',
            'behavior': behavior,
            'delta': behavior - baseline_behavior,
            'text': generated[:500]
        })
    except Exception as e:
        print(f"ERROR: {e}")
    
    # STRATEGY 3: Token-specific (first 25%)
    print("\n" + "="*80)
    print("STRATEGY 3: Token-Specific KV Patching (First 25% of tokens)")
    print("="*80)
    try:
        text = generate_with_token_specific_kv(model, tokenizer, PROMPTS['BASELINE'], champion_kv, 0.25)
        generated = text[len(PROMPTS['BASELINE']):]
        behavior = score_behavior(generated)
        print(f"Behavior: {behavior} (baseline: {baseline_behavior})")
        print(f"Generated: {generated[:200]}...")
        results.append({
            'strategy': 'Token-Specific (25%)',
            'behavior': behavior,
            'delta': behavior - baseline_behavior,
            'text': generated[:500]
        })
    except Exception as e:
        print(f"ERROR: {e}")
    
    # STRATEGY 4: Persistent patching during generation
    print("\n" + "="*80)
    print("STRATEGY 4: Persistent Patching During Generation (L27)")
    print("="*80)
    try:
        text = generate_with_persistent_patch(model, tokenizer, PROMPTS['BASELINE'], 
                                               champion_activations, [27])
        generated = text[len(PROMPTS['BASELINE']):]
        behavior = score_behavior(generated)
        print(f"Behavior: {behavior} (baseline: {baseline_behavior})")
        print(f"Generated: {generated[:200]}...")
        results.append({
            'strategy': 'Persistent Patch (L27)',
            'behavior': behavior,
            'delta': behavior - baseline_behavior,
            'text': generated[:500]
        })
    except Exception as e:
        print(f"ERROR: {e}")
    
    # STRATEGY 5: ALL OF THE ABOVE (Full KV + Multi-layer + Persistent)
    print("\n" + "="*80)
    print("STRATEGY 5: NUCLEAR OPTION - Full KV + Multi-Layer + Persistent")
    print("="*80)
    try:
        # Start with full KV cache
        inputs = tokenizer(PROMPTS['BASELINE'], return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, return_dict=True)
            baseline_kv = outputs.past_key_values
        
        patched_kv = DynamicCache()
        for layer_idx, (k_src, v_src) in enumerate(champion_kv):
            k_base, v_base = baseline_kv[layer_idx]
            min_seq = min(k_base.shape[2], k_src.shape[2])
            k_patched = k_base.clone()
            v_patched = v_base.clone()
            k_patched[:, :, -min_seq:, :] = k_src[:, :, -min_seq:, :]
            v_patched[:, :, -min_seq:, :] = v_src[:, :, -min_seq:, :]
            patched_kv.update(k_patched, v_patched, layer_idx)
        
        # Add persistent patching
        patcher = PersistentPatcher(model, champion_activations)
        patcher.register([25, 27])
        
        try:
            generated_ids = input_ids.clone()
            current_kv = patched_kv
            
            with torch.no_grad():
                for step in range(CONFIG['gen_tokens']):
                    outputs = model(
                        generated_ids[:, -1:],
                        past_key_values=current_kv,
                        use_cache=True,
                        return_dict=True
                    )
                    logits = outputs.logits[:, -1, :] / CONFIG['temperature']
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    current_kv = outputs.past_key_values
                    if next_token.item() == tokenizer.eos_token_id:
                        break
            
            text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            generated = text[len(PROMPTS['BASELINE']):]
            behavior = score_behavior(generated)
            print(f"Behavior: {behavior} (baseline: {baseline_behavior})")
            print(f"Generated: {generated[:200]}...")
            results.append({
                'strategy': 'NUCLEAR (Full KV + Multi + Persistent)',
                'behavior': behavior,
                'delta': behavior - baseline_behavior,
                'text': generated[:500]
            })
        finally:
            patcher.remove()
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(CONFIG['save_csv'], index=False)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\nBaseline behavior: {baseline_behavior}")
    print("\nBest strategies:")
    df_sorted = df.sort_values('delta', ascending=False)
    for _, row in df_sorted.iterrows():
        print(f"  {row['strategy']}: {row['behavior']} (Δ={row['delta']:+.1f})")
    
    print(f"\n✅ Results saved to: {CONFIG['save_csv']}")
    
    return df

if __name__ == "__main__":
    results = run_aggressive_transfer()

