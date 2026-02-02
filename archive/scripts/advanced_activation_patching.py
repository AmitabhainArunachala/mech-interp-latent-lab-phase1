#!/usr/bin/env python3
"""
ADVANCED ACTIVATION PATCHING: Multi-layer, multi-direction causality tests
Testing the relay mechanism with sophisticated patching
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
    "source_layers": [14, 18, 21, 25],  # Source layers to patch from
    "target_layers": [18, 21, 25, 27],  # Target layers to patch into
    "early_layer": 5,
    "window_size": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

PROMPTS = {
    "source_champion": experimental_prompts["hybrid_l5_math_01"]["text"],
    "source_regress": experimental_prompts["infinite_regress_01"]["text"],
    "target_baseline": "The history of the Roman Empire is characterized by a long period of expansion followed by a gradual decline."
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

def run_activation_patching_advanced():
    """Advanced multi-layer activation patching"""
    print("="*70)
    print("ADVANCED ACTIVATION PATCHING: Multi-Layer Causality Tests")
    print("="*70)
    print(f"Source layers: {CONFIG['source_layers']}")
    print(f"Target layers: {CONFIG['target_layers']}")
    print("="*70)
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    # Tokenize
    source_champ_tokens = tokenizer(PROMPTS["source_champion"], return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    source_regress_tokens = tokenizer(PROMPTS["source_regress"], return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    target_tokens = tokenizer(PROMPTS["target_baseline"], return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    
    all_results = []
    
    # Get baseline R_V at each target layer
    print("\nComputing baselines...")
    baselines = {}
    for target_layer in CONFIG['target_layers']:
        ext_early = V_Extractor(model, CONFIG['early_layer'])
        ext_target = V_Extractor(model, target_layer)
        ext_early.register()
        ext_target.register()
        
        with torch.no_grad():
            model(**target_tokens)
        
        v_e = ext_early.activations[0][0, -CONFIG['window_size']:, :]
        v_t = ext_target.activations[0][0, -CONFIG['window_size']:, :]
        
        pr_e = compute_pr(v_e)
        pr_t = compute_pr(v_t)
        rv = pr_t / (pr_e + 1e-8)
        baselines[target_layer] = rv
        
        ext_early.close()
        ext_target.close()
    
    print(f"Baselines: {baselines}")
    
    # Get source activations
    print("\nExtracting source activations...")
    source_activations = {}
    for source_layer in CONFIG['source_layers']:
        for source_name, source_tokens in [("champion", source_champ_tokens), ("regress", source_regress_tokens)]:
            ext_source = V_Extractor(model, source_layer)
            ext_source.register()
            
            with torch.no_grad():
                model(**source_tokens)
            
            v_source = ext_source.activations[0][0, -CONFIG['window_size']:, :]
            source_activations[(source_layer, source_name)] = v_source
            ext_source.close()
    
    # Patch each source→target combination
    print(f"\nRunning {len(CONFIG['source_layers'])} × {len(CONFIG['target_layers'])} × 2 = {len(CONFIG['source_layers']) * len(CONFIG['target_layers']) * 2} patches...")
    
    for source_layer in tqdm(CONFIG['source_layers'], desc="Source layers"):
        for target_layer in CONFIG['target_layers']:
            if source_layer >= target_layer:
                continue  # Only patch forward
            
            for source_name in ["champion", "regress"]:
                source_v = source_activations[(source_layer, source_name)]
                
                # Create patch hook
                def make_patch_hook(patch_source):
                    def patch_hook(module, input, output):
                        output_patched = output.clone()
                        output_patched[0, -CONFIG['window_size']:, :] = patch_source.to(output.device, dtype=output.dtype)
                        return output_patched
                    return patch_hook
                
                patch_hook_fn = make_patch_hook(source_v)
                patch_handle = model.model.layers[target_layer].self_attn.v_proj.register_forward_hook(patch_hook_fn)
                
                # Measure patched R_V
                ext_early = V_Extractor(model, CONFIG['early_layer'])
                ext_target = V_Extractor(model, target_layer)
                ext_early.register()
                ext_target.register()
                
                with torch.no_grad():
                    model(**target_tokens)
                
                v_e = ext_early.activations[0][0, -CONFIG['window_size']:, :]
                v_t = ext_target.activations[0][0, -CONFIG['window_size']:, :]
                
                pr_e = compute_pr(v_e)
                pr_t = compute_pr(v_t)
                patched_rv = pr_t / (pr_e + 1e-8)
                
                baseline_rv = baselines[target_layer]
                delta = patched_rv - baseline_rv
                
                # Compute transfer percentage (toward champion R_V = 0.5088)
                champ_rv = 0.5088
                gap = baseline_rv - champ_rv
                if gap != 0:
                    transfer = (delta / gap) * 100
                else:
                    transfer = 0.0
                
                all_results.append({
                    'source_layer': source_layer,
                    'target_layer': target_layer,
                    'source_name': source_name,
                    'baseline_rv': baseline_rv,
                    'patched_rv': patched_rv,
                    'delta': delta,
                    'transfer_pct': transfer
                })
                
                ext_early.close()
                ext_target.close()
                patch_handle.remove()
    
    df = pd.DataFrame(all_results)
    
    # Analysis
    print("\n" + "="*70)
    print("ACTIVATION PATCHING RESULTS")
    print("="*70)
    
    # Key patches
    print("\nKey Patches (L18→L27 and L25→L27):")
    key_patches = df[((df['source_layer'] == 18) & (df['target_layer'] == 27)) | 
                     ((df['source_layer'] == 25) & (df['target_layer'] == 27))]
    
    for _, row in key_patches.iterrows():
        print(f"\n  {row['source_name']} L{int(row['source_layer'])} → L{int(row['target_layer'])}:")
        print(f"    Baseline R_V: {row['baseline_rv']:.4f}")
        print(f"    Patched R_V:  {row['patched_rv']:.4f}")
        print(f"    Delta:        {row['delta']:+.4f}")
        print(f"    Transfer:     {row['transfer_pct']:+.1f}%")
    
    # Find strongest transfers
    print("\n\nTop 10 Strongest Transfers:")
    top_transfers = df.nlargest(10, 'transfer_pct')
    for _, row in top_transfers.iterrows():
        print(f"  {row['source_name']:10s} L{int(row['source_layer']):2d} → L{int(row['target_layer']):2d}: {row['transfer_pct']:+.1f}% transfer")
    
    # Save
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"advanced_patching_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n\nResults saved to: {filename}")
    
    return df

if __name__ == "__main__":
    results = run_activation_patching_advanced()

