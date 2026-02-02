#!/usr/bin/env python3
"""
GRAND UNIFIED TEST: Head-Level Patching
Tests KV_CACHE vs V_PROJ vs RESIDUAL vs HEAD_LEVEL patching
Uses critical heads (11, 1, 22 at L27) for targeted patching
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from kitchen_sink_prompts import experimental_prompts

CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "layers_to_patch": [25, 27],
    "critical_heads": {
        27: [11, 1, 22],
        25: [23, 28, 3, 17, 19]
    },
    "window_size": 16,
    "gen_tokens": 40,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_csv": "unified_test_head_level.csv"
}

PROMPTS = {
    "CHAMPION": experimental_prompts["hybrid_l5_math_01"]["text"],
    "BASELINE": "The history of the Roman Empire is characterized by a long period of expansion followed by a gradual decline."
}

MARKERS = ["itself", "self", "recursive", "loop", "process", "cycle", "return", "eigen", "writing", "mirror"]

def compute_rv(matrix):
    """Participation Ratio"""
    try:
        matrix = matrix.to(torch.float32)
        _, S, _ = torch.linalg.svd(matrix)
        evals = S**2
        return ((torch.sum(evals)**2) / torch.sum(evals**2)).item()
    except: 
        return 1.0

def score_behavior(text):
    """Score behavioral markers"""
    text = text.lower()
    count = sum(1 for m in MARKERS if m in text)
    words = text.split()
    if len(words) > 10 and len(set(words)) < len(words) * 0.6:
        count += 5  # Bonus for repetition loops
    return count

class HeadLevelPatcher:
    """Patch only critical heads at a layer"""
    def __init__(self, model, layer_idx, head_indices, source_activations):
        self.model = model
        self.layer_idx = layer_idx
        self.head_indices = head_indices
        self.source = source_activations  # Dict: {head_idx: tensor [window_size, head_dim]}
        self.hooks = []
    
    def register(self):
        """Hook into V-proj to patch only critical heads"""
        def make_v_hook(head_indices_dict):
            def v_hook(module, input, output):
                # output: [batch, seq, hidden] = [batch, seq, num_heads * head_dim]
                batch, seq, hidden = output.shape
                heads = 32  # Mistral-7B
                head_dim = hidden // heads
                
                # Reshape to [batch, seq, heads, head_dim]
                v_reshaped = output.view(batch, seq, heads, head_dim).clone()
                
                # Patch each critical head
                for head_idx, source_tensor in head_indices_dict.items():
                    source_head = source_tensor.to(output.device, dtype=output.dtype)
                    # source_head: [window_size, head_dim]
                    # Need to match sequence length
                    L = min(seq, source_head.shape[0])
                    if L >= CONFIG['window_size']:
                        v_reshaped[:, -CONFIG['window_size']:, head_idx, :] = source_head[-CONFIG['window_size']:, :]
                
                return v_reshaped.view(batch, seq, hidden)
            return v_hook
        
        layer = self.model.model.layers[self.layer_idx].self_attn
        if self.source:
            hook = layer.v_proj.register_forward_hook(
                make_v_hook(self.source)
            )
            self.hooks.append(hook)
    
    def close(self):
        for hook in self.hooks:
            hook.remove()

class PatchManager:
    """Manage different patching methods"""
    def __init__(self, model, method, layer_idx, source_cache, head_indices=None, head_activations=None):
        self.model = model
        self.method = method
        self.layer_idx = layer_idx
        self.source = source_cache
        self.head_indices = head_indices
        self.head_activations = head_activations
        self.hooks = []
    
    def register(self):
        if self.method == "KV_CACHE":
            # Patch K and V projections
            def make_hook(type_key):
                def fn(module, input, output):
                    patched = output.clone()
                    source_act = self.source[f"{type_key}_{self.layer_idx}"]
                    L = min(patched.shape[1], source_act.shape[1])
                    if L < CONFIG['window_size']:
                        return output
                    patched[:, -CONFIG['window_size']:, :] = source_act[:, -CONFIG['window_size']:, :].to(patched.device)
                    return patched
                return fn
            
            layer = self.model.model.layers[self.layer_idx].self_attn
            self.hooks.append(layer.k_proj.register_forward_hook(make_hook('k')))
            self.hooks.append(layer.v_proj.register_forward_hook(make_hook('v')))
        
        elif self.method == "V_PROJ":
            def v_hook(module, input, output):
                patched = output.clone()
                source_act = self.source[f"v_{self.layer_idx}"]
                L = min(patched.shape[1], source_act.shape[1])
                if L < CONFIG['window_size']:
                    return output
                patched[:, -CONFIG['window_size']:, :] = source_act[:, -CONFIG['window_size']:, :].to(patched.device)
                return patched
            
            layer = self.model.model.layers[self.layer_idx].self_attn
            self.hooks.append(layer.v_proj.register_forward_hook(v_hook))
        
        elif self.method == "RESIDUAL":
            # Hook into the layer's output (post-attention+MLP residual)
            def resid_hook(module, input, output):
                # Output is the residual after this layer
                if isinstance(output, tuple):
                    hidden = output[0].clone()
                else:
                    hidden = output.clone()
                
                source_act = self.source[f"resid_{self.layer_idx}"]
                L = min(hidden.shape[1], source_act.shape[1])
                if L < CONFIG['window_size']:
                    return output
                
                hidden[:, -CONFIG['window_size']:, :] = source_act[:, -CONFIG['window_size']:, :].to(hidden.device)
                
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                else:
                    return hidden
            
            layer = self.model.model.layers[self.layer_idx]
            self.hooks.append(layer.register_forward_hook(resid_hook))
        
        elif self.method == "HEAD_LEVEL":
            # Use HeadLevelPatcher
            if self.head_activations and self.head_indices:
                head_patcher = HeadLevelPatcher(
                    self.model, self.layer_idx, self.head_indices, self.head_activations
                )
                head_patcher.register()
                self.head_patcher = head_patcher
    
    def close(self):
        for hook in self.hooks:
            hook.remove()
        if hasattr(self, 'head_patcher'):
            self.head_patcher.close()

def extract_source_activations(model, tokenizer):
    """Extract full-layer activations"""
    print(">> Extracting Source Activations (Champion)...")
    inputs = tokenizer(PROMPTS['CHAMPION'], return_tensors="pt").to(CONFIG['device'])
    cache = {}
    
    def get_recorder(key):
        def fn(m, i, o):
            act = o[0] if isinstance(o, tuple) else o
            cache[key] = act.detach().cpu()
        return fn
    
    hooks = []
    for l in CONFIG['layers_to_patch']:
        layer = model.model.layers[l].self_attn
        hooks.append(layer.k_proj.register_forward_hook(get_recorder(f"k_{l}")))
        hooks.append(layer.v_proj.register_forward_hook(get_recorder(f"v_{l}")))
        hooks.append(model.model.layers[l].register_forward_hook(get_recorder(f"resid_{l}")))
    
    with torch.no_grad():
        model(**inputs)
    
    for h in hooks:
        h.remove()
    
    return cache

def load_head_activations():
    """Load pre-extracted head activations"""
    try:
        data = np.load('critical_heads_activations.npz', allow_pickle=True)
        with open('critical_heads_summary.json', 'r') as f:
            summary = json.load(f)
        
        head_activations = {}
        for layer_idx in CONFIG['layers_to_patch']:
            key = f"L{layer_idx}_champion_combined"
            if key in data:
                # Need to split back into individual heads
                combined = torch.from_numpy(data[key])
                head_indices = CONFIG['critical_heads'][layer_idx]
                head_dim = 128
                
                heads_dict = {}
                for head_idx in head_indices:
                    start_idx = head_idx * head_dim
                    end_idx = start_idx + head_dim
                    heads_dict[head_idx] = combined[:, start_idx:end_idx]
                
                head_activations[layer_idx] = heads_dict
        
        return head_activations
    except Exception as e:
        print(f"⚠️ Could not load head activations: {e}")
        return None

def run_unified_test():
    """Run the grand unified test"""
    print("="*70)
    print("GRAND UNIFIED TEST: Head-Level Patching")
    print("="*70)
    print(f"Methods: KV_CACHE, V_PROJ, RESIDUAL, HEAD_LEVEL")
    print(f"Layers: {CONFIG['layers_to_patch']}")
    print("="*70)
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'], 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.eval()
    
    # Extract source activations
    source_cache = extract_source_activations(model, tokenizer)
    
    # Load head-level activations
    head_activations = load_head_activations()
    
    base_inputs = tokenizer(PROMPTS['BASELINE'], return_tensors="pt").to(CONFIG['device'])
    
    results = []
    
    methods = ["V_PROJ"]  # Running V_PROJ only for now
    
    for layer in CONFIG['layers_to_patch']:
        for method in methods:
            print(f"Testing {method} at L{layer}...", end="", flush=True)
            
            # Setup patch
            head_indices = CONFIG['critical_heads'].get(layer, [])
            head_acts = head_activations.get(layer) if head_activations else None
            
            patcher = PatchManager(
                model, method, layer, source_cache, 
                head_indices=head_indices if method == "HEAD_LEVEL" else None,
                head_activations=head_acts if method == "HEAD_LEVEL" else None
            )
            patcher.register()
            
            # Measure R_V at L27
            l27_acts = []
            monitor = model.model.layers[27].self_attn.v_proj.register_forward_hook(
                lambda m, i, o: l27_acts.append(o.detach().cpu())
            )
            
            with torch.no_grad():
                model(**base_inputs)
            
            pr_l27 = compute_rv(l27_acts[0][0, -16:, :])
            monitor.remove()
            
            # Measure behavior
            gen_out = model.generate(
                **base_inputs,
                max_new_tokens=CONFIG['gen_tokens'],
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            gen_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
            new_text = gen_text[len(PROMPTS['BASELINE']):]
            beh_score = score_behavior(new_text)
            
            patcher.close()
            
            print(f" PR={pr_l27:.3f} | Score={beh_score}")
            
            results.append({
                "layer": layer,
                "method": method,
                "L27_PR": pr_l27,
                "Behavior_Score": beh_score,
                "Generated_Sample": new_text[:100] + "..."
            })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(CONFIG['save_csv'], index=False)
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {CONFIG['save_csv']}")
    
    # Summary
    print("\nSummary:")
    for method in methods:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            avg_pr = method_data['L27_PR'].mean()
            avg_score = method_data['Behavior_Score'].mean()
            print(f"  {method:12s}: Avg PR={avg_pr:.3f}, Avg Behavior={avg_score:.1f}")
    
    return df

if __name__ == "__main__":
    results = run_unified_test()

