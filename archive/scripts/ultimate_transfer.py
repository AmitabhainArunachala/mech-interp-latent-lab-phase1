#!/usr/bin/env python3
"""
ULTIMATE TRANSFER: Maximum aggression
Try ALL combinations and find the winner
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from kitchen_sink_prompts import experimental_prompts

CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "window_size": 16,
    "gen_tokens": 200,
    "temperature": 0.8,
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
    text_lower = text.lower()
    count = sum(1 for m in MARKERS if m in text_lower)
    words = text.split()
    if len(words) > 10 and len(set(words)) < len(words) * 0.6:
        count += 5
    return count

def extract_full_kv_cache(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)
        return outputs.past_key_values

def extract_activations(model, tokenizer, prompt, target_layers):
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

class UltimatePatcher:
    def __init__(self, model, source_activations):
        self.model = model
        self.source = source_activations
        self.hooks = []
    
    def v_hook(self, layer_idx):
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
    
    def resid_hook(self, layer_idx):
        source_resid = self.source.get(f"resid_{layer_idx}")
        if source_resid is None:
            return None
        def pre_hook_fn(module, input):
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
            hook_fn = self.v_hook(layer_idx)
            if hook_fn:
                layer = self.model.model.layers[layer_idx].self_attn
                self.hooks.append(layer.v_proj.register_forward_hook(hook_fn))
        for layer_idx in resid_layers:
            hook_fn = self.resid_hook(layer_idx)
            if hook_fn:
                layer = self.model.model.layers[layer_idx]
                self.hooks.append(layer.register_forward_pre_hook(hook_fn))
    
    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def ultimate_transfer(model, tokenizer, baseline_prompt, champion_kv, champion_activations, 
                     v_layers, resid_layers):
    """Ultimate transfer with specified layers"""
    inputs = tokenizer(baseline_prompt, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)
        baseline_kv = outputs.past_key_values
    
    # Full KV replacement
    patched_kv = DynamicCache()
    for layer_idx, (k_src, v_src) in enumerate(champion_kv):
        k_base, v_base = baseline_kv[layer_idx]
        min_seq = min(k_base.shape[2], k_src.shape[2])
        k_patched = k_base.clone()
        v_patched = v_base.clone()
        k_patched[:, :, -min_seq:, :] = k_src[:, :, -min_seq:, :]
        v_patched[:, :, -min_seq:, :] = v_src[:, :, -min_seq:, :]
        patched_kv.update(k_patched, v_patched, layer_idx)
    
    # Persistent patching
    patcher = UltimatePatcher(model, champion_activations)
    patcher.register(v_layers, resid_layers)
    
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

def run_ultimate():
    print("="*80)
    print("ULTIMATE TRANSFER: Testing ALL Combinations")
    print("="*80)
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    print("\nExtracting champion...")
    champion_kv = extract_full_kv_cache(model, tokenizer, PROMPTS['CHAMPION'])
    champion_activations = extract_activations(model, tokenizer, PROMPTS['CHAMPION'],
                                                ["resid_14", "resid_18", "resid_25", "v_18", "v_25", "v_27"])
    
    # Baseline
    baseline_inputs = tokenizer(PROMPTS['BASELINE'], return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    with torch.no_grad():
        baseline_gen = model.generate(**baseline_inputs, max_new_tokens=CONFIG['gen_tokens'],
                                     do_sample=True, temperature=CONFIG['temperature'],
                                     pad_token_id=tokenizer.eos_token_id)
    baseline_text = tokenizer.decode(baseline_gen[0], skip_special_tokens=True)
    baseline_generated = baseline_text[len(PROMPTS['BASELINE']):]
    baseline_behavior = score_behavior(baseline_generated)
    
    print(f"Baseline behavior: {baseline_behavior}")
    
    # Test combinations
    strategies = [
        ("L27 V only", [27], []),
        ("L25 V + L27 V", [25, 27], []),
        ("L18 RESID + L27 V", [27], [18]),
        ("L25 RESID + L27 V", [27], [25]),
        ("L18 RESID + L25 RESID + L27 V", [27], [18, 25]),
        ("L14 RESID + L18 RESID + L25 RESID + L27 V", [27], [14, 18, 25]),
        ("L18 V + L25 V + L27 V", [18, 25, 27], []),
        ("L18 RESID + L18 V + L25 RESID + L25 V + L27 V", [18, 25, 27], [18, 25]),
    ]
    
    results = []
    
    for name, v_layers, resid_layers in strategies:
        print(f"\nTesting: {name}")
        try:
            text = ultimate_transfer(model, tokenizer, PROMPTS['BASELINE'], champion_kv, 
                                    champion_activations, v_layers, resid_layers)
            generated = text[len(PROMPTS['BASELINE']):]
            behavior = score_behavior(generated)
            delta = behavior - baseline_behavior
            print(f"  Behavior: {behavior} (Δ={delta:+.1f})")
            print(f"  Text: {generated[:150]}...")
            results.append({
                'strategy': name,
                'behavior': behavior,
                'delta': delta,
                'text': generated[:500]
            })
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Champion for comparison
    champion_inputs = tokenizer(PROMPTS['CHAMPION'], return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    with torch.no_grad():
        champion_gen = model.generate(**champion_inputs, max_new_tokens=CONFIG['gen_tokens'],
                                     do_sample=True, temperature=CONFIG['temperature'],
                                     pad_token_id=tokenizer.eos_token_id)
    champion_text = tokenizer.decode(champion_gen[0], skip_special_tokens=True)
    champion_generated = champion_text[len(PROMPTS['CHAMPION']):]
    champion_behavior = score_behavior(champion_generated)
    
    df = pd.DataFrame(results)
    df.to_csv("ultimate_transfer.csv", index=False)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Baseline: {baseline_behavior}")
    print(f"Champion: {champion_behavior}")
    print("\nTop strategies:")
    df_sorted = df.sort_values('delta', ascending=False)
    for _, row in df_sorted.head(5).iterrows():
        eff = (row['delta'] / (champion_behavior - baseline_behavior) * 100) if (champion_behavior - baseline_behavior) > 0 else 0
        print(f"  {row['strategy']}: {row['behavior']} (Δ={row['delta']:+.1f}, {eff:.1f}% efficiency)")
    
    return df

if __name__ == "__main__":
    run_ultimate()

