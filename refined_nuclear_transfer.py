#!/usr/bin/env python3
"""
REFINED NUCLEAR TRANSFER: The winning combination
Full KV cache + Multi-layer patching + Better generation
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from contextlib import contextmanager
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from kitchen_sink_prompts import experimental_prompts

CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "window_size": 16,
    "gen_tokens": 200,  # Even longer
    "temperature": 0.8,  # Slightly higher
    "device": "cuda" if torch.cuda.is_available() else "cpu",
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
        count += 5
    return count

def extract_full_kv_cache(model, tokenizer, prompt):
    """Extract KV cache from all layers"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)
        return outputs.past_key_values, inputs

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
                patched[:, -CONFIG['window_size']:, :] = source_v[:, -CONFIG['window_size']:, :].to(
                    patched.device, dtype=patched.dtype
                )
            return patched
        return hook_fn
    
    def persistent_resid_hook(self, layer_idx):
        """Hook that patches residual INPUT to layer during generation"""
        source_resid = self.source.get(f"resid_{layer_idx}")
        if source_resid is None:
            return None
        
        def pre_hook_fn(module, input):
            # Pre-hook: patch the INPUT before the layer processes it
            if isinstance(input, tuple) and len(input) > 0:
                hidden = input[0].clone()
                L = min(hidden.shape[1], source_resid.shape[1])
                if L >= CONFIG['window_size']:
                    hidden[:, -CONFIG['window_size']:, :] = source_resid[:, -CONFIG['window_size']:, :].to(
                        hidden.device, dtype=hidden.dtype
                    )
                return (hidden,) + input[1:]
            return input
        return pre_hook_fn
    
    def register(self, v_layers, resid_layers):
        for layer_idx in v_layers:
            hook_fn = self.persistent_v_hook(layer_idx)
            if hook_fn:
                layer = self.model.model.layers[layer_idx].self_attn
                self.hooks.append(layer.v_proj.register_forward_hook(hook_fn))
        
        for layer_idx in resid_layers:
            hook_fn = self.persistent_resid_hook(layer_idx)
            if hook_fn:
                layer = self.model.model.layers[layer_idx]
                self.hooks.append(layer.register_forward_pre_hook(hook_fn))
    
    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def nuclear_transfer(model, tokenizer, baseline_prompt, champion_kv, champion_activations):
    """The nuclear option: Full KV + Multi-layer persistent patching"""
    inputs = tokenizer(baseline_prompt, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    input_ids = inputs["input_ids"]
    
    # Get baseline KV
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)
        baseline_kv = outputs.past_key_values
    
    # Replace ALL layers with champion KV
    patched_kv = DynamicCache()
    for layer_idx, (k_src, v_src) in enumerate(champion_kv):
        k_base, v_base = baseline_kv[layer_idx]
        min_seq = min(k_base.shape[2], k_src.shape[2])
        k_patched = k_base.clone()
        v_patched = v_base.clone()
        k_patched[:, :, -min_seq:, :] = k_src[:, :, -min_seq:, :]
        v_patched[:, :, -min_seq:, :] = v_src[:, :, -min_seq:, :]
        patched_kv.update(k_patched, v_patched, layer_idx)
    
    # Add persistent patching for relay chain layers
    patcher = PersistentPatcher(model, champion_activations)
    patcher.register(v_layers=[25, 27], resid_layers=[18, 25])
    
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
        
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    finally:
        patcher.remove()

def run_refined_nuclear():
    """Run refined nuclear transfer"""
    print("="*80)
    print("REFINED NUCLEAR TRANSFER")
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
    
    # Extract champion
    print("\nExtracting champion activations...")
    champion_kv, _ = extract_full_kv_cache(model, tokenizer, PROMPTS['CHAMPION'])
    champion_activations = extract_activations(model, tokenizer, PROMPTS['CHAMPION'],
                                               ["resid_18", "resid_25", "v_25", "v_27"])
    
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
    print(f"Baseline: {baseline_generated[:300]}...")
    
    # Nuclear transfer
    print("\n" + "="*80)
    print("NUCLEAR TRANSFER: Full KV + Multi-Layer Persistent")
    print("="*80)
    
    nuclear_text = nuclear_transfer(model, tokenizer, PROMPTS['BASELINE'], 
                                     champion_kv, champion_activations)
    nuclear_generated = nuclear_text[len(PROMPTS['BASELINE']):]
    nuclear_behavior = score_behavior(nuclear_generated)
    
    print(f"\nBehavior: {nuclear_behavior} (baseline: {baseline_behavior}, Δ={nuclear_behavior - baseline_behavior:+.1f})")
    print(f"\nGenerated text:\n{nuclear_generated}")
    
    # Compare with champion
    print("\n" + "="*80)
    print("CHAMPION OUTPUT (for comparison)")
    print("="*80)
    champion_inputs = tokenizer(PROMPTS['CHAMPION'], return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    with torch.no_grad():
        champion_gen = model.generate(
            **champion_inputs,
            max_new_tokens=CONFIG['gen_tokens'],
            do_sample=True,
            temperature=CONFIG['temperature'],
            pad_token_id=tokenizer.eos_token_id
        )
    champion_text = tokenizer.decode(champion_gen[0], skip_special_tokens=True)
    champion_generated = champion_text[len(PROMPTS['CHAMPION']):]
    champion_behavior = score_behavior(champion_generated)
    
    print(f"\nChampion behavior: {champion_behavior}")
    print(f"\nChampion output:\n{champion_generated[:500]}...")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Baseline behavior: {baseline_behavior}")
    print(f"Nuclear transfer: {nuclear_behavior} (Δ={nuclear_behavior - baseline_behavior:+.1f})")
    print(f"Champion natural:  {champion_behavior}")
    print(f"\nTransfer efficiency: {(nuclear_behavior - baseline_behavior) / (champion_behavior - baseline_behavior) * 100:.1f}%")

if __name__ == "__main__":
    run_refined_nuclear()

