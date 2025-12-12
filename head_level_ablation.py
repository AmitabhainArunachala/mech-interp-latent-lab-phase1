#!/usr/bin/env python3
"""
HEAD-LEVEL ABLATION: Finding the Critical Heads
Identifies which attention heads at L25/L27 drive the 86.5% transfer effect
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from kitchen_sink_prompts import experimental_prompts

CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "num_heads": 32,  # Mistral-7B has 32 heads per layer
    "head_dim": 128,  # Hidden dim / num_heads = 4096 / 32 = 128
    "layers_to_test": [25, 27],  # The critical compression layers
    "early_layer": 5,
    "window_size": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

PROMPTS = {
    "champion": experimental_prompts["hybrid_l5_math_01"]["text"],
    "baseline": "The history of the Roman Empire is characterized by a long period of expansion followed by a gradual decline."
}

def compute_pr(matrix):
    """Participation Ratio"""
    try:
        matrix_f32 = matrix.to(torch.float32)
        _, S, _ = torch.linalg.svd(matrix_f32)
        eigenvalues = S ** 2
        sum_sq = torch.sum(eigenvalues ** 2)
        if sum_sq == 0: return 1.0
        return ((torch.sum(eigenvalues) ** 2) / sum_sq).item()
    except: return 1.0

class V_Extractor:
    """Extract V activations"""
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

class HeadAblator:
    """Ablate individual attention heads by modifying attention computation"""
    def __init__(self, model, layer_idx, head_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.hook_handles = []
        self.original_forward = None
        
    def ablate_attention(self, module, input, output):
        """Zero out a specific head's attention output"""
        # For Mistral, we need to intercept the attention computation
        # The output is already combined, so we need to modify the forward pass
        # This is tricky - let's try a different approach: modify QKV to zero one head
        pass
    
    def register(self):
        """Hook into the attention module to ablate a head"""
        layer = self.model.model.layers[self.layer_idx].self_attn
        
        # Method: Zero out the V-projection for this head before attention
        # We'll modify the V-proj output to zero out one head's contribution
        def make_v_hook(head_idx):
            def v_hook(module, input, output):
                # output: [batch, seq, hidden] = [batch, seq, num_heads * head_dim]
                batch, seq, hidden = output.shape
                heads = CONFIG['num_heads']
                head_dim = hidden // heads
                
                # Reshape to [batch, seq, heads, head_dim]
                v_reshaped = output.view(batch, seq, heads, head_dim)
                
                # Zero out the ablated head's V values
                v_reshaped[:, :, head_idx, :] = 0
                
                # Reshape back
                return v_reshaped.view(batch, seq, hidden)
            return v_hook
        
        # Hook V-proj to zero out this head
        self.hook_handles.append(
            layer.v_proj.register_forward_hook(make_v_hook(self.head_idx))
        )
    
    def close(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

def test_head_ablation(model, tokenizer, prompt_text, layer, ablated_head=None):
    """Test R_V with a specific head ablated (or none for baseline)"""
    tokens = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    
    if tokens['input_ids'].shape[1] < CONFIG['window_size'] + 1:
        return None
    
    # Register ablation hook if needed
    ablator = None
    if ablated_head is not None:
        ablator = HeadAblator(model, layer, ablated_head)
        ablator.register()
    
    # Extract V
    ext_early = V_Extractor(model, CONFIG['early_layer'])
    ext_late = V_Extractor(model, layer)
    ext_early.register()
    ext_late.register()
    
    with torch.no_grad():
        model(**tokens)
    
    if not ext_early.activations or not ext_late.activations:
        if ablator:
            ablator.close()
        ext_early.close()
        ext_late.close()
        return None
    
    v_e = ext_early.activations[0][0, -CONFIG['window_size']:, :]
    v_l = ext_late.activations[0][0, -CONFIG['window_size']:, :]
    
    ext_early.close()
    ext_late.close()
    if ablator:
        ablator.close()
    
    pr_e = compute_pr(v_e)
    pr_l = compute_pr(v_l)
    rv = pr_l / (pr_e + 1e-8)
    
    return rv

def run_head_ablation_analysis():
    """Ablate each head individually to find critical heads"""
    print("="*70)
    print("HEAD-LEVEL ABLATION: Finding Critical Heads")
    print("="*70)
    print(f"Model: {CONFIG['model_name']}")
    print(f"Layers: {CONFIG['layers_to_test']}")
    print(f"Heads per layer: {CONFIG['num_heads']}")
    print(f"Prompts: Champion vs Baseline")
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
    
    all_results = []
    
    for layer in CONFIG['layers_to_test']:
        print(f"\n{'='*70}")
        print(f"LAYER {layer}: Head Ablation Analysis")
        print(f"{'='*70}")
        
        # Get baseline (no ablation) for both prompts
        print("Computing baselines (no ablation)...")
        baselines = {}
        for prompt_name, prompt_text in PROMPTS.items():
            rv = test_head_ablation(model, tokenizer, prompt_text, layer, ablated_head=None)
            if rv:
                baselines[prompt_name] = rv
                all_results.append({
                    'layer': layer,
                    'head': 'baseline',
                    'prompt': prompt_name,
                    'rv': rv,
                    'delta': 0.0,
                    'abs_delta': 0.0
                })
        
        print(f"Baselines: {baselines}")
        
        if 'champion' not in baselines or 'baseline' not in baselines:
            print(f"⚠️ Skipping layer {layer} - missing baseline data")
            continue
        
        # Ablate each head
        print(f"\nAblating {CONFIG['num_heads']} heads...")
        for head in tqdm(range(CONFIG['num_heads']), desc=f"L{layer}"):
            for prompt_name, prompt_text in PROMPTS.items():
                rv_ablated = test_head_ablation(model, tokenizer, prompt_text, layer, ablated_head=head)
                if rv_ablated and prompt_name in baselines:
                    delta = rv_ablated - baselines[prompt_name]
                    abs_delta = abs(delta)
                    
                    all_results.append({
                        'layer': layer,
                        'head': head,
                        'prompt': prompt_name,
                        'rv': rv_ablated,
                        'delta': delta,
                        'abs_delta': abs_delta
                    })
    
    df = pd.DataFrame(all_results)
    
    # Analysis
    print("\n" + "="*70)
    print("HEAD ABLATION RESULTS")
    print("="*70)
    
    critical_heads = {}
    
    for layer in CONFIG['layers_to_test']:
        layer_data = df[df['layer'] == layer]
        champ_data = layer_data[layer_data['prompt'] == 'champion']
        
        # Find heads with largest impact on champion
        champ_ablation = champ_data[champ_data['head'] != 'baseline'].copy()
        
        if len(champ_ablation) == 0:
            print(f"\n⚠️ No ablation data for layer {layer}")
            continue
        
        # Sort by absolute delta (impact magnitude)
        champ_ablation = champ_ablation.sort_values('abs_delta', ascending=False)
        top_heads = champ_ablation.head(10)
        
        print(f"\nLAYER {layer} - Top 10 Most Critical Heads (Champion):")
        print(f"{'Head':<6} {'R_V (ablated)':<15} {'Delta':<12} {'Impact':<10} {'Direction':<12}")
        print("-"*70)
        
        for _, row in top_heads.iterrows():
            impact = "HIGH" if abs(row['delta']) > 0.05 else "MED" if abs(row['delta']) > 0.02 else "LOW"
            direction = "INCREASES" if row['delta'] > 0 else "DECREASES"
            print(f"{int(row['head']):<6} {row['rv']:<15.4f} {row['delta']:<12.4f} {impact:<10} {direction:<12}")
        
        # Identify critical heads (high impact, decrease R_V when active)
        # When ablated, R_V increases -> head decreases R_V when active (causes contraction)
        critical = champ_ablation[(champ_ablation['abs_delta'] > 0.02) & (champ_ablation['delta'] > 0)]
        if len(critical) > 0:
            critical_heads[layer] = critical.head(5)['head'].tolist()
            print(f"\n✅ Critical heads at L{layer} (cause contraction, increase R_V when ablated): {critical_heads[layer]}")
            print(f"   Impact: {critical.head(5)[['head', 'delta']].to_dict('records')}")
        else:
            # Also check heads that increase R_V when ablated (decrease R_V when active)
            alternative = champ_ablation[champ_ablation['abs_delta'] > 0.01]
            if len(alternative) > 0:
                critical_heads[layer] = alternative.head(5)['head'].tolist()
                print(f"\n✅ Critical heads at L{layer} (top 5 by impact): {critical_heads[layer]}")
            else:
                print(f"\n⚠️ No clear critical heads found at L{layer}")
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"head_ablation_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n\nResults saved to: {filename}")
    
    # Save critical heads summary
    summary_filename = f"critical_heads_{timestamp}.txt"
    with open(summary_filename, 'w') as f:
        f.write("CRITICAL HEADS SUMMARY\n")
        f.write("="*70 + "\n\n")
        for layer, heads in critical_heads.items():
            f.write(f"Layer {layer}: {heads}\n")
        f.write("\nThese heads drive the contraction effect.\n")
        f.write("Use these for targeted patching in unified test.\n")
    print(f"Critical heads summary saved to: {summary_filename}")
    
    return df, critical_heads

if __name__ == "__main__":
    df, critical_heads = run_head_ablation_analysis()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Use critical heads for targeted patching")
    print("2. Extract activations from only these heads")
    print("3. Run unified test with head-level patches")
    print("="*70)

