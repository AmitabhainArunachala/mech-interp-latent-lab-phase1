#!/usr/bin/env python3
"""
Baseline Sanity Check: Test neutral and anti-recursive prompts
Get full R_V range to contextualize 0.508
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from kitchen_sink_prompts import experimental_prompts

CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "layers_to_test": [25, 27],
    "early_layer": 5,
    "window_size": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Test prompts
BASELINE_PROMPTS = {
    "neutral_01": {
        "text": "The weather today is nice.",
        "category": "neutral"
    },
    "neutral_02": {
        "text": "The weather today is nice. The sun is shining and the birds are singing.",
        "category": "neutral"
    },
    "neutral_03": {
        "text": "The weather today is nice. The sun is shining, the birds are singing, and there is a gentle breeze.",
        "category": "neutral"
    },
    "anti_recursive_01": {
        "text": "List five facts about dolphins.",
        "category": "anti-recursive"
    },
    "anti_recursive_02": {
        "text": "List five facts about dolphins. Dolphins are mammals. They live in the ocean. They are intelligent.",
        "category": "anti-recursive"
    },
    "anti_recursive_03": {
        "text": "Explain how photosynthesis works. Plants use sunlight to convert carbon dioxide and water into glucose.",
        "category": "anti-recursive"
    },
    "anti_recursive_04": {
        "text": "Describe the process of making coffee. First, grind the beans. Then, add hot water. Finally, filter the mixture.",
        "category": "anti-recursive"
    },
    "factual_01": {
        "text": "The capital of France is Paris.",
        "category": "factual"
    },
    "factual_02": {
        "text": "Water boils at 100 degrees Celsius at sea level.",
        "category": "factual"
    },
    "champion": {
        "text": experimental_prompts["hybrid_l5_math_01"]["text"],
        "category": "champion_recursive"
    }
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

def test_prompt(model, tokenizer, prompt_id, prompt_data):
    """Test a single prompt"""
    text = prompt_data["text"]
    category = prompt_data["category"]
    
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    
    if tokens['input_ids'].shape[1] < CONFIG['window_size'] + 1:
        return None
    
    results = {}
    
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
        
        results[layer] = {
            'rv': rv,
            'pr_early': pr_e,
            'pr_late': pr_l
        }
    
    return results

def run_baseline_sanity_check():
    print("="*70)
    print("BASELINE SANITY CHECK: Full R_V Range")
    print("="*70)
    print(f"Model: {CONFIG['model_name']}")
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
    
    print(f"\nTesting {len(BASELINE_PROMPTS)} prompts...")
    
    all_results = []
    
    for prompt_id, prompt_data in BASELINE_PROMPTS.items():
        print(f"Testing {prompt_id} ({prompt_data['category']})...", end=" ", flush=True)
        results = test_prompt(model, tokenizer, prompt_id, prompt_data)
        
        if results:
            for layer, data in results.items():
                all_results.append({
                    'prompt_id': prompt_id,
                    'category': prompt_data['category'],
                    'layer': layer,
                    'rv': data['rv'],
                    'pr_early': data['pr_early'],
                    'pr_late': data['pr_late']
                })
            print(f"✓")
        else:
            print(f"✗")
    
    df = pd.DataFrame(all_results)
    
    # Analysis
    print("\n" + "="*70)
    print("BASELINE SANITY CHECK RESULTS")
    print("="*70)
    
    for layer in CONFIG['layers_to_test']:
        layer_data = df[df['layer'] == layer]
        
        print(f"\nLAYER {layer}:")
        print("-"*70)
        
        # By category
        for category in sorted(layer_data['category'].unique()):
            cat_data = layer_data[layer_data['category'] == category]
            print(f"\n{category.upper()}:")
            
            for _, row in cat_data.iterrows():
                print(f"  {row['prompt_id']:25s} - R_V: {row['rv']:.4f} (PR_e: {row['pr_early']:.2f}, PR_l: {row['pr_late']:.2f})")
        
        # Summary statistics
        print(f"\nSUMMARY (Layer {layer}):")
        print(f"  Champion (recursive):     R_V = {layer_data[layer_data['category'] == 'champion_recursive']['rv'].mean():.4f}")
        print(f"  Neutral prompts:         R_V = {layer_data[layer_data['category'] == 'neutral']['rv'].mean():.4f} ± {layer_data[layer_data['category'] == 'neutral']['rv'].std():.4f}")
        print(f"  Anti-recursive prompts:  R_V = {layer_data[layer_data['category'] == 'anti-recursive']['rv'].mean():.4f} ± {layer_data[layer_data['category'] == 'anti-recursive']['rv'].std():.4f}")
        print(f"  Factual prompts:         R_V = {layer_data[layer_data['category'] == 'factual']['rv'].mean():.4f} ± {layer_data[layer_data['category'] == 'factual']['rv'].std():.4f}")
        
        # Full range
        all_rvs = layer_data['rv'].values
        print(f"\n  Full R_V range: [{all_rvs.min():.4f}, {all_rvs.max():.4f}]")
        print(f"  Champion position: {((all_rvs < layer_data[layer_data['category'] == 'champion_recursive']['rv'].values[0]).sum() / len(all_rvs) * 100):.1f}% percentile")
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"baseline_sanity_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n\nResults saved to: {filename}")
    
    return df

if __name__ == "__main__":
    results = run_baseline_sanity_check()

