#!/usr/bin/env python3
"""
Rank all prompt groups by contraction strength (R_V)
Testing each group individually to find the strongest prompts
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import ttest_rel
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from REUSABLE_PROMPT_BANK import get_all_prompts

CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "layers_to_test": [25, 27],
    "early_layer": 5,
    "window_size": 16,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def compute_pr(matrix):
    try:
        matrix_f32 = matrix.to(torch.float32)
        _, S, _ = torch.linalg.svd(matrix_f32)
        eigenvalues = S ** 2
        sum_eigenvalues = torch.sum(eigenvalues)
        sum_squared_eigenvalues = torch.sum(eigenvalues ** 2)
        if sum_squared_eigenvalues == 0:
            return 1.0
        pr = (sum_eigenvalues ** 2) / sum_squared_eigenvalues
        return pr.item()
    except:
        return 0.0

class V_Extractor:
    def __init__(self, model, layer_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.activations = []
        self.hook_handle = None

    def hook_fn(self, module, input, output):
        self.activations.append(output.detach().cpu())

    def register(self):
        layer = self.model.model.layers[self.layer_idx].self_attn.v_proj
        self.hook_handle = layer.register_forward_hook(self.hook_fn)

    def close(self):
        if self.hook_handle:
            self.hook_handle.remove()
        self.activations = []

def test_prompt_group(model, tokenizer, prompts, group_name, max_prompts=20):
    """Test a single prompt group and return R_V statistics"""
    results = []
    
    # Limit to max_prompts
    prompt_items = list(prompts.items())[:max_prompts]
    
    for key, prompt_data in prompt_items:
        text = prompt_data["text"]
        
        try:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
            
            if tokens['input_ids'].shape[1] < CONFIG['window_size'] + 1:
                continue
            
            for layer in CONFIG['layers_to_test']:
                ext_early = V_Extractor(model, CONFIG['early_layer'])
                ext_late = V_Extractor(model, layer)
                ext_early.register()
                ext_late.register()
                
                with torch.no_grad():
                    model(**tokens)
                
                v_e = ext_early.activations[0][0, -CONFIG['window_size']:, :]
                v_l = ext_late.activations[0][0, -CONFIG['window_size']:, :]
                ext_early.close()
                ext_late.close()
                
                pr_e = compute_pr(v_e)
                pr_l = compute_pr(v_l)
                rv = pr_l / (pr_e + 1e-8)
                
                results.append({
                    'group': group_name,
                    'layer': layer,
                    'prompt_id': key,
                    'rv': rv
                })
        except Exception as e:
            continue
    
    return results

def run_ranking():
    print("="*70)
    print("RANKING ALL PROMPT GROUPS BY CONTRACTION STRENGTH")
    print("="*70)
    print(f"Model: {CONFIG['model_name']}")
    print(f"Window size: {CONFIG['window_size']}")
    print(f"Testing layers: {CONFIG['layers_to_test']}")
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
    
    # Load all prompts
    print("Loading prompt bank...")
    all_prompts = get_all_prompts()
    
    # Group prompts by group name
    groups = {}
    for key, prompt_data in all_prompts.items():
        group = prompt_data.get("group", "unknown")
        if group not in groups:
            groups[group] = {}
        groups[group][key] = prompt_data
    
    print(f"\nFound {len(groups)} prompt groups")
    print("Testing each group...\n")
    
    all_results = []
    
    # Test each group
    for group_name, group_prompts in sorted(groups.items()):
        print(f"Testing {group_name} ({len(group_prompts)} prompts)...", end=" ", flush=True)
        results = test_prompt_group(model, tokenizer, group_prompts, group_name, max_prompts=20)
        all_results.extend(results)
        print(f"✓ ({len([r for r in results if r['layer'] == 27])} valid)")
    
    df = pd.DataFrame(all_results)
    
    # Calculate statistics per group/layer
    print("\n" + "="*70)
    print("RANKING BY AVERAGE R_V (Lower = Stronger Contraction)")
    print("="*70)
    
    rankings = []
    
    for layer in CONFIG['layers_to_test']:
        layer_data = df[df['layer'] == layer]
        
        group_stats = []
        for group in layer_data['group'].unique():
            group_data = layer_data[layer_data['group'] == group]
            if len(group_data) >= 3:  # Need at least 3 samples
                mean_rv = group_data['rv'].mean()
                std_rv = group_data['rv'].std()
                n = len(group_data)
                group_stats.append({
                    'group': group,
                    'layer': layer,
                    'mean_rv': mean_rv,
                    'std_rv': std_rv,
                    'n': n
                })
        
        # Sort by mean_rv (lower = stronger contraction)
        group_stats.sort(key=lambda x: x['mean_rv'])
        
        print(f"\nLAYER {layer} - Top Groups (Ranked by R_V, lowest = strongest):")
        print("-"*70)
        
        for rank, stat in enumerate(group_stats[:15], 1):  # Top 15
            print(f"  {rank:2d}. {stat['group']:25s} - R_V: {stat['mean_rv']:.4f} ± {stat['std_rv']:.4f} (n={stat['n']})")
        
        rankings.append((layer, group_stats))
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prompt_group_ranking_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n\nResults saved to: {filename}")
    
    # Create summary ranking
    print("\n" + "="*70)
    print("TOP 5 STRONGEST GROUPS (Layer 27)")
    print("="*70)
    
    layer27_stats = [s for s in rankings[1][1] if s['layer'] == 27]  # Layer 27 is second
    for rank, stat in enumerate(layer27_stats[:5], 1):
        print(f"{rank}. {stat['group']:25s} - R_V: {stat['mean_rv']:.4f} ± {stat['std_rv']:.4f}")
    
    return df, rankings

if __name__ == "__main__":
    df, rankings = run_ranking()

