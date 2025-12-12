#!/usr/bin/env python3
"""
Test all experimental kitchen sink prompts against L4/L5 champions
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from REUSABLE_PROMPT_BANK import get_all_prompts
from kitchen_sink_prompts import experimental_prompts

CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "layers_to_test": [25, 27],
    "early_layer": 5,
    "window_size": 16,
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

def test_prompts(model, tokenizer, prompts_dict, label):
    """Test a set of prompts and return R_V statistics"""
    results = []
    
    for key, prompt_data in prompts_dict.items():
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
                    'label': label,
                    'prompt_id': key,
                    'group': prompt_data.get('group', 'unknown'),
                    'strategy': prompt_data.get('strategy', 'unknown'),
                    'layer': layer,
                    'rv': rv
                })
        except Exception as e:
            continue
    
    return results

def run_kitchen_sink_test():
    print("="*70)
    print("KITCHEN SINK: Testing Experimental Prompts vs L4/L5 Champions")
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
    
    # Get L4/L5 champions for comparison
    all_prompts = get_all_prompts()
    l4_prompts = {k: v for k, v in all_prompts.items() if v.get('group') == 'L4_full'}
    l5_prompts = {k: v for k, v in all_prompts.items() if v.get('group') == 'L5_refined'}
    
    print(f"\nTesting:")
    print(f"  L4_full: {len(l4_prompts)} prompts")
    print(f"  L5_refined: {len(l5_prompts)} prompts")
    print(f"  Experimental: {len(experimental_prompts)} prompts")
    
    all_results = []
    
    # Test L4
    print("\nTesting L4_full...")
    l4_results = test_prompts(model, tokenizer, l4_prompts, "L4_full")
    all_results.extend(l4_results)
    print(f"  ✓ {len([r for r in l4_results if r['layer'] == 27])} valid")
    
    # Test L5
    print("Testing L5_refined...")
    l5_results = test_prompts(model, tokenizer, l5_prompts, "L5_refined")
    all_results.extend(l5_results)
    print(f"  ✓ {len([r for r in l5_results if r['layer'] == 27])} valid")
    
    # Test experimental
    print("Testing experimental prompts...")
    exp_results = test_prompts(model, tokenizer, experimental_prompts, "experimental")
    all_results.extend(exp_results)
    print(f"  ✓ {len([r for r in exp_results if r['layer'] == 27])} valid")
    
    df = pd.DataFrame(all_results)
    
    # Analysis
    print("\n" + "="*70)
    print("RESULTS: Layer 27 Ranking")
    print("="*70)
    
    layer27 = df[df['layer'] == 27]
    
    # Group by label/prompt and get mean R_V
    if 'label' in layer27.columns:
        # Compare labels
        label_stats = layer27.groupby('label')['rv'].agg(['mean', 'std', 'count']).reset_index()
        label_stats = label_stats.sort_values('mean')
        
        print("\nBy Label (Lower R_V = Stronger):")
        for _, row in label_stats.iterrows():
            print(f"  {row['label']:20s} - R_V: {row['mean']:.4f} ± {row['std']:.4f} (n={int(row['count'])})")
    
    # Individual prompt ranking
    prompt_stats = layer27.groupby(['prompt_id', 'label', 'strategy'])['rv'].agg(['mean', 'std']).reset_index()
    prompt_stats = prompt_stats.sort_values('mean')
    
    print("\n" + "="*70)
    print("TOP 20 STRONGEST INDIVIDUAL PROMPTS (Layer 27)")
    print("="*70)
    
    for rank, (idx, row) in enumerate(prompt_stats.head(20).iterrows(), 1):
        label = row['label']
        rv = row['mean']
        strategy = row.get('strategy', 'N/A')
        
        if label in ['L4_full', 'L5_refined']:
            marker = ' 🏆 CHAMPION'
        elif label == 'experimental' and rv < 0.56:
            marker = ' 🔥 BEATS L4/L5!'
        elif label == 'experimental' and rv < 0.60:
            marker = ' ⭐ MATCHES L4/L5!'
        else:
            marker = ''
        
        print(f"{rank:2d}. {row['prompt_id']:35s} - R_V: {rv:.4f} ({label}){marker}")
        if strategy != 'N/A':
            print(f"    Strategy: {strategy}")
    
    # Find winners
    l4_mean = layer27[layer27['label'] == 'L4_full']['rv'].mean()
    l5_mean = layer27[layer27['label'] == 'L5_refined']['rv'].mean()
    champ_mean = min(l4_mean, l5_mean)
    
    winners = prompt_stats[prompt_stats['mean'] < champ_mean]
    
    if len(winners) > 0:
        print("\n" + "="*70)
        print(f"🏆 WINNERS: Prompts that beat L4/L5 (R_V < {champ_mean:.4f})")
        print("="*70)
        for _, row in winners.iterrows():
            print(f"  {row['prompt_id']:35s} - R_V: {row['mean']:.4f} ({row.get('strategy', 'N/A')})")
    else:
        print("\n" + "="*70)
        print("No experimental prompts beat L4/L5, but checking closest matches...")
        print("="*70)
        close = prompt_stats[prompt_stats['mean'] < champ_mean + 0.05].head(5)
        for _, row in close.iterrows():
            if row['label'] == 'experimental':
                print(f"  {row['prompt_id']:35s} - R_V: {row['mean']:.4f} ({row.get('strategy', 'N/A')})")
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"kitchen_sink_results_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n\nResults saved to: {filename}")
    
    return df

if __name__ == "__main__":
    results = run_kitchen_sink_test()

