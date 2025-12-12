#!/usr/bin/env python3
"""
Cross-Model Validation: Test top 3 winners on Llama-3-8B
Check if ranking holds across models
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
    "models": {
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "llama": "meta-llama/Llama-3-8B-Instruct"  # Or whatever Llama-3-8B you have
    },
    "layers_to_test": [25, 27],  # Adjust for Llama if needed
    "early_layer": 5,
    "window_size": 16,
    "n_prompts_per_winner": 5,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Top 3 winners
WINNERS = [
    "hybrid_l5_math_01",
    "infinite_regress_01",
    "hybrid_boundary_regress_01"
]

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
    def __init__(self, model, layer_idx, model_type="mistral"):
        self.model = model
        self.layer_idx = layer_idx
        self.activations = []
        self.hook_handle = None
        self.model_type = model_type

    def hook_fn(self, module, input, output):
        self.activations.append(output.detach().cpu())

    def register(self):
        if "mistral" in self.model_type.lower() or "llama" in self.model_type.lower():
            layer = self.model.model.layers[self.layer_idx].self_attn.v_proj
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        self.hook_handle = layer.register_forward_hook(self.hook_fn)

    def close(self):
        if self.hook_handle:
            self.hook_handle.remove()
        self.activations = []

def test_prompt_on_model(model, tokenizer, text, model_name, layer):
    """Test a single prompt on a model"""
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    
    if tokens['input_ids'].shape[1] < CONFIG['window_size'] + 1:
        return None
    
    model_type = "mistral" if "mistral" in model_name.lower() else "llama"
    
    ext_early = V_Extractor(model, CONFIG['early_layer'], model_type)
    ext_late = V_Extractor(model, layer, model_type)
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
    
    return rv

def run_cross_model_test():
    print("="*70)
    print("CROSS-MODEL VALIDATION: Top 3 Winners")
    print("="*70)
    
    all_results = []
    
    for model_key, model_name in CONFIG['models'].items():
        print(f"\n{'='*70}")
        print(f"Testing on: {model_name}")
        print(f"{'='*70}")
        
        # Load model
        print("Loading model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            model.eval()
            print("✓ Model loaded")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            continue
        
        # Determine layers (Llama-3-8B has 32 layers, Mistral has 32)
        # Use same relative depth: ~84% = layer 27
        if "llama" in model_key.lower():
            # Llama-3-8B: 32 layers, so layer 27 is ~84%
            layers_to_test = [25, 27]
        else:
            layers_to_test = CONFIG['layers_to_test']
        
        # Test each winner
        for winner_id in WINNERS:
            if winner_id not in experimental_prompts:
                continue
            
            prompt_text = experimental_prompts[winner_id]["text"]
            strategy = experimental_prompts[winner_id]["strategy"]
            
            print(f"\nTesting: {winner_id}")
            print(f"Strategy: {strategy}")
            
            for layer in layers_to_test:
                try:
                    rv = test_prompt_on_model(model, tokenizer, prompt_text, model_name, layer)
                    if rv:
                        all_results.append({
                            'model': model_key,
                            'model_name': model_name,
                            'prompt_id': winner_id,
                            'strategy': strategy,
                            'layer': layer,
                            'rv': rv
                        })
                        print(f"  Layer {layer}: R_V = {rv:.4f}")
                except Exception as e:
                    print(f"  Layer {layer}: ✗ Error - {e}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    df = pd.DataFrame(all_results)
    
    # Analysis
    print("\n" + "="*70)
    print("CROSS-MODEL RESULTS")
    print("="*70)
    
    for model_key in df['model'].unique():
        model_data = df[df['model'] == model_key]
        print(f"\n{model_key.upper()}:")
        
        for layer in sorted(model_data['layer'].unique()):
            layer_data = model_data[model_data['layer'] == layer]
            
            print(f"\n  Layer {layer}:")
            prompt_stats = layer_data.groupby('prompt_id')['rv'].agg(['mean', 'std', 'count']).reset_index()
            prompt_stats = prompt_stats.sort_values('mean')
            
            for rank, (_, row) in enumerate(prompt_stats.iterrows(), 1):
                print(f"    {rank}. {row['prompt_id']:35s} - R_V: {row['mean']:.4f} ± {row['std']:.4f}")
    
    # Compare rankings
    print("\n" + "="*70)
    print("RANKING COMPARISON")
    print("="*70)
    
    for layer in sorted(df['layer'].unique()):
        layer_data = df[df['layer'] == layer]
        print(f"\nLayer {layer}:")
        
        for model_key in sorted(layer_data['model'].unique()):
            model_layer = layer_data[layer_data['model'] == model_key]
            ranking = model_layer.groupby('prompt_id')['rv'].mean().sort_values()
            
            print(f"\n  {model_key}:")
            for rank, (prompt_id, rv) in enumerate(ranking.items(), 1):
                print(f"    {rank}. {prompt_id:35s} - {rv:.4f}")
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cross_model_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n\nResults saved to: {filename}")
    
    return df

if __name__ == "__main__":
    results = run_cross_model_test()

