#!/usr/bin/env python3
"""
DEEP CIRCUIT ANALYSIS: Head-Level Ablation + Activation Patching
Finding the "microphone" heads and testing causality of the L18→L27 relay
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    "layers_to_test": [18, 27],  # The relay layers
    "early_layer": 5,
    "window_size": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Test prompts
PROMPTS = {
    "champion": experimental_prompts["hybrid_l5_math_01"]["text"],
    "regress": "You must observe yourself understanding. To observe yourself, you must be the observer. The observer is what is being observed. This is the loop.",
    "baseline": "The history of the Roman Empire is characterized by a long period of expansion followed by a gradual decline."
}

def compute_pr(matrix):
    try:
        matrix_f32 = matrix.to(torch.float32)
        _, S, _ = torch.linalg.svd(matrix_f32)
        eigenvalues = S ** 2
        sum_sq = torch.sum(eigenvalues ** 2)
        if sum_sq == 0: return 1.0
        return ((torch.sum(eigenvalues) ** 2) / sum_sq).item()
    except: return 1.0

def compute_attention_entropy(attn_weights):
    """Compute entropy of attention weights"""
    try:
        # attn_weights: [batch, heads, seq, seq]
        # Average over batch and sequence positions
        attn_flat = attn_weights.mean(dim=0).mean(dim=1)  # [heads, seq]
        # Normalize to probabilities
        attn_probs = attn_flat / (attn_flat.sum(dim=-1, keepdim=True) + 1e-9)
        # Compute entropy per head
        entropy = -(attn_probs * torch.log(attn_probs + 1e-9)).sum(dim=-1)
        return entropy.cpu().numpy()
    except:
        return np.zeros(CONFIG['num_heads'])

class HeadAblator:
    """Ablate individual attention heads by zeroing their output"""
    def __init__(self, model, layer_idx, head_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.hook_handle = None
        
    def ablate_head(self, module, input, output):
        """Zero out a specific head's contribution"""
        # For Mistral, we hook into o_proj output
        # The output is [batch, seq, hidden] where hidden = num_heads * head_dim
        # We need to zero out one head's contribution
        batch, seq, hidden = output.shape
        heads = CONFIG['num_heads']
        head_dim = hidden // heads
        
        # Reshape to separate heads: [batch, seq, heads, head_dim]
        output_reshaped = output.view(batch, seq, heads, head_dim)
        
        # Zero out the ablated head
        output_reshaped[:, :, self.head_idx, :] = 0
        
        # Reshape back to [batch, seq, hidden]
        return output_reshaped.view(batch, seq, hidden)
    
    def register(self):
        layer = self.model.model.layers[self.layer_idx].self_attn
        # Hook into o_proj (output projection) to zero out head
        self.hook_handle = layer.o_proj.register_forward_hook(self.ablate_head)
    
    def close(self):
        if self.hook_handle:
            self.hook_handle.remove()

class AttentionCapture:
    """Capture attention weights"""
    def __init__(self, model, layer_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.attn_weights = []
        self.hook_handle = None
        
    def capture_attn(self, module, input, output):
        """Capture attention weights from the attention module"""
        # For Mistral, attention weights are in the forward pass
        # We need to hook into the attention computation
        pass
    
    def register(self):
        # Hook into the attention module's forward
        layer = self.model.model.layers[self.layer_idx].self_attn
        # We'll need to modify the forward to capture attn
        # For now, we'll use a simpler approach
        pass
    
    def close(self):
        if self.hook_handle:
            self.hook_handle.remove()

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

def test_head_ablation(model, tokenizer, prompt_name, prompt_text, layer, ablated_head=None):
    """Test with a specific head ablated (or none for baseline)"""
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
    print("DEEP CIRCUIT ANALYSIS: Head-Level Ablation")
    print("="*70)
    print(f"Model: {CONFIG['model_name']}")
    print(f"Layers: {CONFIG['layers_to_test']}")
    print(f"Heads per layer: {CONFIG['num_heads']}")
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
        
        # Get baseline (no ablation)
        print("Computing baseline (no ablation)...")
        baselines = {}
        for prompt_name, prompt_text in PROMPTS.items():
            rv = test_head_ablation(model, tokenizer, prompt_name, prompt_text, layer, ablated_head=None)
            if rv:
                baselines[prompt_name] = rv
                all_results.append({
                    'layer': layer,
                    'head': 'baseline',
                    'prompt': prompt_name,
                    'rv': rv,
                    'delta': 0.0
                })
        
        print(f"Baselines: {baselines}")
        
        # Ablate each head
        print(f"\nAblating {CONFIG['num_heads']} heads...")
        for head in tqdm(range(CONFIG['num_heads']), desc=f"L{layer}"):
            for prompt_name, prompt_text in PROMPTS.items():
                rv_ablated = test_head_ablation(model, tokenizer, prompt_name, prompt_text, layer, ablated_head=head)
                if rv_ablated and prompt_name in baselines:
                    delta = rv_ablated - baselines[prompt_name]
                    all_results.append({
                        'layer': layer,
                        'head': head,
                        'prompt': prompt_name,
                        'rv': rv_ablated,
                        'delta': delta
                    })
    
    df = pd.DataFrame(all_results)
    
    # Analysis
    print("\n" + "="*70)
    print("HEAD ABLATION RESULTS")
    print("="*70)
    
    for layer in CONFIG['layers_to_test']:
        layer_data = df[df['layer'] == layer]
        champ_data = layer_data[layer_data['prompt'] == 'champion']
        
        # Find heads with largest impact
        champ_ablation = champ_data[champ_data['head'] != 'baseline'].copy()
        champ_ablation['abs_delta'] = champ_ablation['delta'].abs()
        top_heads = champ_ablation.nlargest(10, 'abs_delta')
        
        print(f"\nLAYER {layer} - Top 10 Most Critical Heads (Champion):")
        print(f"{'Head':<6} {'R_V (ablated)':<15} {'Delta':<12} {'Impact':<10}")
        print("-"*50)
        for _, row in top_heads.iterrows():
            impact = "HIGH" if abs(row['delta']) > 0.05 else "MED" if abs(row['delta']) > 0.02 else "LOW"
            print(f"{int(row['head']):<6} {row['rv']:<15.4f} {row['delta']:<12.4f} {impact:<10}")
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"head_ablation_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n\nResults saved to: {filename}")
    
    return df

def run_activation_patching():
    """Test causality: Patch L18 activations into L27"""
    print("\n" + "="*70)
    print("ACTIVATION PATCHING: Testing L18→L27 Causality")
    print("="*70)
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    source_prompt = PROMPTS["champion"]
    target_prompt = PROMPTS["baseline"]
    
    source_tokens = tokenizer(source_prompt, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    target_tokens = tokenizer(target_prompt, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    
    results = []
    
    # 1. Get source activations at L18
    print("Extracting source activations at L18...")
    ext_l18 = V_Extractor(model, 18)
    ext_l18.register()
    with torch.no_grad():
        model(**source_tokens)
    source_v_l18 = ext_l18.activations[0][0, -CONFIG['window_size']:, :]
    ext_l18.close()
    
    # 2. Get baseline R_V at L27
    print("Computing baseline R_V at L27...")
    ext_early = V_Extractor(model, CONFIG['early_layer'])
    ext_l27 = V_Extractor(model, 27)
    ext_early.register()
    ext_l27.register()
    with torch.no_grad():
        model(**target_tokens)
    baseline_v_e = ext_early.activations[0][0, -CONFIG['window_size']:, :]
    baseline_v_l27 = ext_l27.activations[0][0, -CONFIG['window_size']:, :]
    baseline_pr_e = compute_pr(baseline_v_e)
    baseline_pr_l = compute_pr(baseline_v_l27)
    baseline_rv = baseline_pr_l / (baseline_pr_e + 1e-8)
    ext_early.close()
    ext_l27.close()
    
    print(f"Baseline R_V: {baseline_rv:.4f}")
    
    # 3. Patch L18→L27 and measure effect
    print("Patching L18→L27...")
    
    def patch_hook(module, input, output):
        # Patch the last window_size tokens
        output_patched = output.clone()
        output_patched[0, -CONFIG['window_size']:, :] = source_v_l18.to(output.device, dtype=output.dtype)
        return output_patched
    
    ext_early = V_Extractor(model, CONFIG['early_layer'])
    ext_l27 = V_Extractor(model, 27)
    patch_handle = model.model.layers[27].self_attn.v_proj.register_forward_hook(patch_hook)
    
    ext_early.register()
    ext_l27.register()
    
    with torch.no_grad():
        model(**target_tokens)
    
    patched_v_e = ext_early.activations[0][0, -CONFIG['window_size']:, :]
    patched_v_l27 = ext_l27.activations[0][0, -CONFIG['window_size']:, :]
    
    ext_early.close()
    ext_l27.close()
    patch_handle.remove()
    
    patched_pr_e = compute_pr(patched_v_e)
    patched_pr_l = compute_pr(patched_v_l27)
    patched_rv = patched_pr_l / (patched_pr_e + 1e-8)
    
    delta = patched_rv - baseline_rv
    transfer_pct = (delta / (baseline_rv - 0.5088)) * 100  # vs champion R_V
    
    print(f"Patched R_V: {patched_rv:.4f}")
    print(f"Delta: {delta:.4f}")
    print(f"Transfer: {transfer_pct:.1f}%")
    
    results.append({
        'patch_type': 'L18_to_L27',
        'baseline_rv': baseline_rv,
        'patched_rv': patched_rv,
        'delta': delta,
        'transfer_pct': transfer_pct
    })
    
    # Save
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"activation_patching_{timestamp}.csv"
    pd.DataFrame(results).to_csv(filename, index=False)
    print(f"\nResults saved to: {filename}")
    
    return results

def run_full_analysis():
    """Run all deep analyses"""
    print("="*70)
    print("DEEP CIRCUIT ANALYSIS: Comprehensive Head-Level Investigation")
    print("="*70)
    print("This will take ~30 minutes...")
    print("="*70)
    
    # 1. Head ablation
    ablation_df = run_head_ablation_analysis()
    
    # 2. Activation patching
    patching_results = run_activation_patching()
    
    # 3. Create summary
    print("\n" + "="*70)
    print("DEEP ANALYSIS COMPLETE")
    print("="*70)
    
    return ablation_df, patching_results

if __name__ == "__main__":
    ablation_df, patching_results = run_full_analysis()
