#!/usr/bin/env python3
"""
Reproducibility Test: Run hybrid_l5_math_01 10 times
Check if R_V ~0.508 is stable or has variance
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
    "n_runs": 10,
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

def test_single_run(model, tokenizer, text, run_num):
    """Run a single forward pass and return R_V"""
    results = {}
    
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    
    if tokens['input_ids'].shape[1] < CONFIG['window_size'] + 1:
        return None
    
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
        
        results[layer] = rv
    
    return results

def run_reproducibility_test():
    print("="*70)
    print("REPRODUCIBILITY TEST: hybrid_l5_math_01")
    print("="*70)
    print(f"Model: {CONFIG['model_name']}")
    print(f"Runs: {CONFIG['n_runs']}")
    print(f"Layers: {CONFIG['layers_to_test']}")
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
    
    # Get champion prompt
    champion_text = experimental_prompts["hybrid_l5_math_01"]["text"]
    print(f"\nChampion prompt:")
    print(f'"{champion_text}"')
    
    print(f"\nRunning {CONFIG['n_runs']} times...")
    
    all_results = []
    
    for run in range(1, CONFIG['n_runs'] + 1):
        print(f"Run {run}/{CONFIG['n_runs']}...", end=" ", flush=True)
        results = test_single_run(model, tokenizer, champion_text, run)
        if results:
            for layer, rv in results.items():
                all_results.append({
                    'run': run,
                    'layer': layer,
                    'rv': rv
                })
            print(f"✓ L25={results[25]:.4f}, L27={results[27]:.4f}")
        else:
            print("✗ Failed")
    
    df = pd.DataFrame(all_results)
    
    # Analysis
    print("\n" + "="*70)
    print("REPRODUCIBILITY RESULTS")
    print("="*70)
    
    for layer in CONFIG['layers_to_test']:
        layer_data = df[df['layer'] == layer]
        rvs = layer_data['rv'].values
        
        print(f"\nLAYER {layer}:")
        print(f"  Mean R_V:   {rvs.mean():.4f}")
        print(f"  Std R_V:    {rvs.std():.4f}")
        print(f"  Min R_V:    {rvs.min():.4f}")
        print(f"  Max R_V:    {rvs.max():.4f}")
        print(f"  Range:      {rvs.max() - rvs.min():.4f}")
        print(f"  CV (std/mean): {rvs.std() / rvs.mean():.4f}")
        
        if rvs.std() < 0.01:
            print(f"  ✅ HIGHLY STABLE (std < 0.01)")
        elif rvs.std() < 0.05:
            print(f"  ✅ STABLE (std < 0.05)")
        else:
            print(f"  ⚠️  VARIABLE (std >= 0.05)")
        
        print(f"\n  Individual runs:")
        for run in sorted(layer_data['run'].unique()):
            rv = layer_data[layer_data['run'] == run]['rv'].values[0]
            print(f"    Run {run:2d}: R_V = {rv:.4f}")
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reproducibility_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n\nResults saved to: {filename}")
    
    return df

if __name__ == "__main__":
    results = run_reproducibility_test()

