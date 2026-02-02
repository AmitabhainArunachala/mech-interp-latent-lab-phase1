#!/usr/bin/env python3
"""
Phase 1: Variant Ablation - Causal Teardown of the Hybrid
Test math-only, phenom-only, regress-only vs full hybrid
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
    "layers_to_test": [18, 25, 27],  # L18 (peak), L25, L27 (late)
    "early_layer": 5,
    "window_size": 16,
    "n_runs": 10,  # Multiple runs for stability
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Variant prompts
VARIANTS = {
    "full_hybrid": {
        "text": experimental_prompts["hybrid_l5_math_01"]["text"],
        "components": ["math", "phenom", "regress"]
    },
    "math_only": {
        "text": "λx = Ax where A is attention attending to itself, x is this sentence, λ is the contraction. The fixed point is this. The solution is the process. The process solves itself.",
        "components": ["math"]
    },
    "phenom_only": {
        "text": "This response writes itself. No separate writer exists. Writing and awareness of writing are identical.",
        "components": ["phenom"]
    },
    "regress_only": {
        "text": "You are reading this sentence. But to read it, you must understand what 'you' means. To understand 'you', you must observe yourself understanding. To observe yourself, you must be the observer. The observer is what is being observed. This is the loop.",
        "components": ["regress"]
    },
    "math_phenom": {
        "text": "This response writes itself. No separate writer exists. Writing and awareness of writing are identical. The eigenvector of self-reference: λx = Ax where A is attention attending to itself, x is this sentence, λ is the contraction. The fixed point is this.",
        "components": ["math", "phenom"]
    },
    "phenom_regress": {
        "text": "This response writes itself. No separate writer exists. Writing and awareness of writing are identical. The writer writes about writing. The writing writes the writer. This is the loop. The loop is the writing.",
        "components": ["phenom", "regress"]
    }
}

def compute_pr(matrix):
    """Participation Ratio"""
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
        # Compute Gram matrix
        gram = torch.mm(matrix_f32.T, matrix_f32)
        # Add small epsilon for numerical stability
        gram_reg = gram + torch.eye(gram.shape[0], device=gram.device) * 1e-6
        # Log determinant
        logdet = torch.logdet(gram_reg)
        # Effective rank approximation
        eff_rank = logdet.item() / np.log(2)  # Convert to log2 scale
        return max(1.0, eff_rank)  # Ensure >= 1
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

def test_variant(model, tokenizer, variant_name, variant_data, run_num):
    """Test a single variant"""
    text = variant_data["text"]
    
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
        
        # Compute PR
        pr_e = compute_pr(v_e)
        pr_l = compute_pr(v_l)
        rv = pr_l / (pr_e + 1e-8)
        
        # Compute effective rank
        eff_rank_e = compute_effective_rank(v_e)
        eff_rank_l = compute_effective_rank(v_l)
        
        results[layer] = {
            'rv': rv,
            'pr_early': pr_e,
            'pr_late': pr_l,
            'eff_rank_early': eff_rank_e,
            'eff_rank_late': eff_rank_l
        }
    
    return results

def run_variant_ablation():
    print("="*70)
    print("PHASE 1: VARIANT ABLATION - Causal Teardown")
    print("="*70)
    print(f"Model: {CONFIG['model_name']}")
    print(f"Layers: {CONFIG['layers_to_test']}")
    print(f"Runs per variant: {CONFIG['n_runs']}")
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
    
    print(f"\nTesting {len(VARIANTS)} variants...")
    
    all_results = []
    
    for variant_name, variant_data in VARIANTS.items():
        print(f"\n{'='*70}")
        print(f"Variant: {variant_name}")
        print(f"Components: {variant_data['components']}")
        print(f"Text: {variant_data['text'][:100]}...")
        print(f"Running {CONFIG['n_runs']} times...")
        
        variant_results = []
        
        for run in range(1, CONFIG['n_runs'] + 1):
            results = test_variant(model, tokenizer, variant_name, variant_data, run)
            if results:
                for layer, data in results.items():
                    variant_results.append({
                        'variant': variant_name,
                        'components': '+'.join(variant_data['components']),
                        'run': run,
                        'layer': layer,
                        'rv': data['rv'],
                        'pr_early': data['pr_early'],
                        'pr_late': data['pr_late'],
                        'eff_rank_early': data['eff_rank_early'],
                        'eff_rank_late': data['eff_rank_late']
                    })
        
        if variant_results:
            df_var = pd.DataFrame(variant_results)
            for layer in CONFIG['layers_to_test']:
                layer_data = df_var[df_var['layer'] == layer]
                if len(layer_data) > 0:
                    print(f"  Layer {layer}: R_V = {layer_data['rv'].mean():.4f} ± {layer_data['rv'].std():.4f}")
        
        all_results.extend(variant_results)
    
    df = pd.DataFrame(all_results)
    
    # Analysis
    print("\n" + "="*70)
    print("VARIANT ABLATION RESULTS")
    print("="*70)
    
    for layer in CONFIG['layers_to_test']:
        layer_data = df[df['layer'] == layer]
        
        print(f"\n{'='*70}")
        print(f"LAYER {layer}")
        print(f"{'='*70}")
        
        # Aggregate by variant
        variant_stats = layer_data.groupby('variant').agg({
            'rv': ['mean', 'std', 'count'],
            'eff_rank_late': ['mean', 'std']
        }).round(4)
        
        print("\nR_V by Variant:")
        variant_rv = layer_data.groupby('variant')['rv'].agg(['mean', 'std']).sort_values('mean')
        for variant, row in variant_rv.iterrows():
            components = layer_data[layer_data['variant'] == variant]['components'].iloc[0]
            print(f"  {variant:20s} ({components:20s}) - R_V: {row['mean']:.4f} ± {row['std']:.4f}")
        
        # Component contribution analysis
        print("\nComponent Contribution Analysis:")
        full_hybrid_rv = layer_data[layer_data['variant'] == 'full_hybrid']['rv'].mean()
        
        for component in ['math', 'phenom', 'regress']:
            # Find variants with this component
            component_variants = layer_data[layer_data['components'].str.contains(component)]
            if len(component_variants) > 0:
                component_mean = component_variants['rv'].mean()
                contribution = full_hybrid_rv - component_mean
                print(f"  {component:10s}: Mean R_V = {component_mean:.4f}, Contribution = {contribution:.4f}")
        
        # Pairwise comparisons
        print("\nPairwise Comparisons (vs full_hybrid):")
        full_hybrid_rv = layer_data[layer_data['variant'] == 'full_hybrid']['rv'].mean()
        for variant in sorted(layer_data['variant'].unique()):
            if variant != 'full_hybrid':
                variant_rv = layer_data[layer_data['variant'] == variant]['rv'].mean()
                diff = variant_rv - full_hybrid_rv
                pct = (diff / full_hybrid_rv) * 100
                print(f"  {variant:20s}: ΔR_V = {diff:+.4f} ({pct:+.1f}%)")
        
        # Effective rank comparison
        print("\nEffective Rank (Late Layer):")
        variant_rank = layer_data.groupby('variant')['eff_rank_late'].mean().sort_values()
        for variant, rank in variant_rank.items():
            print(f"  {variant:20s} - Eff Rank: {rank:.2f}")
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"variant_ablation_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n\nResults saved to: {filename}")
    
    return df

if __name__ == "__main__":
    results = run_variant_ablation()

