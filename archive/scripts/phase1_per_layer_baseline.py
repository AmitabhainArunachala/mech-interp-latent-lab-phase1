#!/usr/bin/env python3
"""
Phase 1: Per-Layer Baseline Sweep
Sweep all 32 layers on L4, factual, neutral, hybrid to normalize landscape
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from kitchen_sink_prompts import experimental_prompts
from REUSABLE_PROMPT_BANK import get_all_prompts

CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "num_layers": 32,  # Mistral-7B has 32 layers
    "early_layer": 5,
    "window_size": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Baseline prompts
BASELINE_PROMPTS = {
    "champion_hybrid": {
        "text": experimental_prompts["hybrid_l5_math_01"]["text"],
        "category": "champion"
    },
    "l4_sample": {
        "text": get_all_prompts()["L4_full_01"]["text"],
        "category": "l4"
    },
    "factual": {
        "text": "Water boils at 100 degrees Celsius at sea level. This is a fundamental property of water.",
        "category": "factual"
    },
    "neutral": {
        "text": "The weather today is nice. The sun is shining, the birds are singing, and there is a gentle breeze.",
        "category": "neutral"
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

def compute_effective_rank(matrix):
    """Effective rank via log-det of Gram matrix"""
    try:
        matrix_f32 = matrix.to(torch.float32)
        gram = torch.mm(matrix_f32.T, matrix_f32)
        gram_reg = gram + torch.eye(gram.shape[0], device=gram.device) * 1e-6
        logdet = torch.logdet(gram_reg)
        eff_rank = logdet.item() / np.log(2)
        return max(1.0, eff_rank)
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

def test_prompt_at_layer(model, tokenizer, prompt_id, prompt_data, layer):
    """Test a prompt at a specific layer"""
    text = prompt_data["text"]
    
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    
    if tokens['input_ids'].shape[1] < CONFIG['window_size'] + 1:
        return None
    
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
    
    eff_rank_e = compute_effective_rank(v_e)
    eff_rank_l = compute_effective_rank(v_l)
    
    return {
        'rv': rv,
        'pr_early': pr_e,
        'pr_late': pr_l,
        'eff_rank_early': eff_rank_e,
        'eff_rank_late': eff_rank_l
    }

def run_per_layer_baseline():
    print("="*70)
    print("PHASE 1: PER-LAYER BASELINE SWEEP")
    print("="*70)
    print(f"Model: {CONFIG['model_name']}")
    print(f"Layers: 0-{CONFIG['num_layers']-1} (all {CONFIG['num_layers']} layers)")
    print(f"Prompts: {len(BASELINE_PROMPTS)}")
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
    
    print(f"\nSweeping all {CONFIG['num_layers']} layers...")
    
    all_results = []
    
    for prompt_id, prompt_data in BASELINE_PROMPTS.items():
        print(f"\nTesting {prompt_id} ({prompt_data['category']})...")
        print("Progress: ", end="", flush=True)
        
        for layer in range(CONFIG['num_layers']):
            try:
                results = test_prompt_at_layer(model, tokenizer, prompt_id, prompt_data, layer)
                if results:
                    all_results.append({
                        'prompt_id': prompt_id,
                        'category': prompt_data['category'],
                        'layer': layer,
                        'depth_pct': (layer / CONFIG['num_layers']) * 100,
                        'rv': results['rv'],
                        'pr_early': results['pr_early'],
                        'pr_late': results['pr_late'],
                        'eff_rank_early': results['eff_rank_early'],
                        'eff_rank_late': results['eff_rank_late']
                    })
                
                if layer % 5 == 0:
                    print(".", end="", flush=True)
            except Exception as e:
                print(f"\nError at layer {layer}: {e}")
                continue
        
        print()  # New line
    
    df = pd.DataFrame(all_results)
    
    # Analysis
    print("\n" + "="*70)
    print("PER-LAYER BASELINE RESULTS")
    print("="*70)
    
    # Find key layers
    champion_data = df[df['category'] == 'champion']
    
    print("\nChampion (hybrid_l5_math_01) Key Layers:")
    key_layers = [14, 18, 25, 27]
    for layer in key_layers:
        layer_data = champion_data[champion_data['layer'] == layer]
        if len(layer_data) > 0:
            rv = layer_data['rv'].values[0]
            print(f"  Layer {layer:2d} ({layer/CONFIG['num_layers']*100:5.1f}%): R_V = {rv:.4f}")
    
    # Delta analysis (champion vs baseline mean)
    print("\n" + "="*70)
    print("DELTA ANALYSIS: Champion vs Baseline Mean")
    print("="*70)
    
    baseline_categories = ['l4', 'factual', 'neutral']
    baseline_data = df[df['category'].isin(baseline_categories)]
    
    for layer in range(CONFIG['num_layers']):
        champ_layer = champion_data[champion_data['layer'] == layer]
        base_layer = baseline_data[baseline_data['layer'] == layer]
        
        if len(champ_layer) > 0 and len(base_layer) > 0:
            champ_rv = champ_layer['rv'].values[0]
            base_rv = base_layer['rv'].mean()
            delta = champ_rv - base_rv
            delta_pct = (delta / base_rv) * 100
            
            if abs(delta_pct) > 15:  # Significant difference
                print(f"  Layer {layer:2d}: ΔR_V = {delta:+.4f} ({delta_pct:+.1f}%) - Champion: {champ_rv:.4f}, Baseline: {base_rv:.4f}")
    
    # Peak detection
    print("\n" + "="*70)
    print("PEAK DETECTION")
    print("="*70)
    
    champ_rvs = champion_data.groupby('layer')['rv'].mean()
    base_rvs = baseline_data.groupby('layer')['rv'].mean()
    
    # Find layers where champion has largest delta
    deltas = champ_rvs - base_rvs
    top_peaks = deltas.nsmallest(5)  # Most negative (strongest contraction)
    
    print("\nTop 5 Layers with Strongest Champion Contraction:")
    for layer, delta in top_peaks.items():
        champ_rv = champ_rvs[layer]
        base_rv = base_rvs[layer]
        print(f"  Layer {layer:2d}: ΔR_V = {delta:.4f} (Champion: {champ_rv:.4f}, Baseline: {base_rv:.4f})")
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"per_layer_baseline_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n\nResults saved to: {filename}")
    
    # Create summary plot data
    summary = df.groupby(['category', 'layer']).agg({
        'rv': 'mean',
        'eff_rank_late': 'mean'
    }).reset_index()
    
    summary_filename = f"per_layer_baseline_summary_{timestamp}.csv"
    summary.to_csv(summary_filename, index=False)
    print(f"Summary saved to: {summary_filename}")
    
    return df

if __name__ == "__main__":
    results = run_per_layer_baseline()

