#!/usr/bin/env python3
"""
Reproduce Nov 16-17 Mistral Singularity Results
Testing multiple window sizes: 16, 32, 64
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import ttest_rel
import warnings
import sys
import os

# Import prompt bank
sys.path.insert(0, os.path.dirname(__file__))
from REUSABLE_PROMPT_BANK import get_all_prompts

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "layers_to_test": [25, 27],
    "early_layer": 5,
    "window_sizes": [16, 32, 64],  # Testing multiple window sizes
    "seed": 42,
    "n_pairs": 45,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ==============================================================================
# GEOMETRY UTILS (Rank-Velocity)
# ==============================================================================
def compute_pr(matrix):
    """Computes Participation Ratio (Differentiable Dimensionality)"""
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
    except Exception as e:
        return 0.0

# ==============================================================================
# EXTRACTION LOGIC
# ==============================================================================
class V_Extractor:
    def __init__(self, model, layer_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.activations = []
        self.hook_handle = None

    def hook_fn(self, module, input, output):
        # Mistral V-Proj output: [batch, seq, hidden]
        self.activations.append(output.detach().cpu())

    def register(self):
        layer = self.model.model.layers[self.layer_idx].self_attn.v_proj
        self.hook_handle = layer.register_forward_hook(self.hook_fn)

    def close(self):
        if self.hook_handle:
            self.hook_handle.remove()
        self.activations = []

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def run_window_sweep():
    print("="*70)
    print("NOV 16-17 MISTRAL SINGULARITY - WINDOW SIZE SWEEP")
    print("="*70)
    print(f"Model: {CONFIG['model_name']}")
    print(f"Layers: {CONFIG['layers_to_test']}")
    print(f"Window sizes: {CONFIG['window_sizes']}")
    print(f"Early layer: {CONFIG['early_layer']}")
    print(f"Precision: bfloat16 (CRITICAL)")
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
    
    # Load prompt bank
    print("Loading prompt bank...")
    prompt_bank = get_all_prompts()
    
    # Collect pairs exactly as in mistral_L27_FULL_VALIDATION.py
    pairs = []
    
    recursive_groups = ["L5_refined", "L4_full", "L3_deeper"]
    baseline_groups = ["long_control", "baseline_creative", "baseline_math"]
    
    print(f"\nCollecting pairs from:")
    print(f"  Recursive: {recursive_groups}")
    print(f"  Baseline: {baseline_groups}")
    
    for rec_group in recursive_groups:
        rec_ids = [k for k, v in prompt_bank.items() if v.get("group") == rec_group]
        
        for base_group in baseline_groups:
            base_ids = [k for k, v in prompt_bank.items() if v.get("group") == base_group]
            
            for i in range(min(len(rec_ids), len(base_ids))):
                base_text = prompt_bank[base_ids[i]]["text"]
                base_tokens = tokenizer.encode(base_text)
                # Check against max window size
                if len(base_tokens) >= max(CONFIG['window_sizes']):
                    pairs.append((rec_ids[i], base_ids[i], rec_group, base_group))
    
    # Shuffle and limit (seed 42)
    np.random.seed(CONFIG['seed'])
    np.random.shuffle(pairs)
    pairs = pairs[:CONFIG['n_pairs']]
    
    print(f"\nTesting {len(pairs)} pairs across {len(CONFIG['window_sizes'])} window sizes...")
    
    results = []
    
    for idx, (rec_id, base_id, rec_group, base_group) in enumerate(pairs):
        rec_text = prompt_bank[rec_id]["text"]
        base_text = prompt_bank[base_id]["text"]
        
        try:
            # Tokenize
            rec_tokens = tokenizer(rec_text, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
            base_tokens = tokenizer(base_text, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
            
            rec_seq_len = rec_tokens['input_ids'].shape[1]
            base_seq_len = base_tokens['input_ids'].shape[1]
            
            # Skip if too short for largest window
            if rec_seq_len < max(CONFIG['window_sizes']) + 1:
                continue
            
            for layer in CONFIG['layers_to_test']:
                # 1. RECURSIVE PASS - Extract once, use for all window sizes
                ext_early = V_Extractor(model, CONFIG['early_layer'])
                ext_late = V_Extractor(model, layer)
                ext_early.register()
                ext_late.register()
                
                with torch.no_grad():
                    model(**rec_tokens)
                
                rec_v_e_full = ext_early.activations[0][0]  # Full sequence
                rec_v_l_full = ext_late.activations[0][0]  # Full sequence
                ext_early.close()
                ext_late.close()
                
                # 2. BASELINE PASS - Extract once, use for all window sizes
                ext_early = V_Extractor(model, CONFIG['early_layer'])
                ext_late = V_Extractor(model, layer)
                ext_early.register()
                ext_late.register()
                
                with torch.no_grad():
                    model(**base_tokens)
                
                base_v_e_full = ext_early.activations[0][0]  # Full sequence
                base_v_l_full = ext_late.activations[0][0]  # Full sequence
                ext_early.close()
                ext_late.close()
                
                # 3. COMPUTE R_V for each window size
                for window_size in CONFIG['window_sizes']:
                    # Skip if sequence too short
                    if rec_seq_len < window_size + 1 or base_seq_len < window_size + 1:
                        continue
                    
                    # Extract window (last N tokens)
                    rec_v_e = rec_v_e_full[-window_size:, :]
                    rec_v_l = rec_v_l_full[-window_size:, :]
                    base_v_e = base_v_e_full[-window_size:, :]
                    base_v_l = base_v_l_full[-window_size:, :]
                    
                    # Compute R_V
                    rec_pr_e = compute_pr(rec_v_e)
                    rec_pr_l = compute_pr(rec_v_l)
                    rec_rv = rec_pr_l / (rec_pr_e + 1e-8)
                    
                    base_pr_e = compute_pr(base_v_e)
                    base_pr_l = compute_pr(base_v_l)
                    base_rv = base_pr_l / (base_pr_e + 1e-8)
                    
                    results.append({
                        "layer": layer,
                        "window_size": window_size,
                        "pair_idx": idx,
                        "rec_id": rec_id,
                        "base_id": base_id,
                        "rec_group": rec_group,
                        "base_group": base_group,
                        "rec_rv": rec_rv,
                        "base_rv": base_rv,
                        "diff": rec_rv - base_rv
                    })
            
            if (idx + 1) % 5 == 0:
                print(".", end="", flush=True)
                
        except Exception as e:
            print(f"\nError on pair {idx}: {e}")
            continue
    
    # Analysis
    df = pd.DataFrame(results)
    
    print("\n\n" + "="*70)
    print("RESULTS SUMMARY BY WINDOW SIZE")
    print("="*70)
    
    for window_size in CONFIG['window_sizes']:
        print(f"\n{'='*70}")
        print(f"WINDOW SIZE: {window_size}")
        print(f"{'='*70}")
        
        window_data = df[df['window_size'] == window_size]
        valid_pairs = len(window_data) // len(CONFIG['layers_to_test'])
        print(f"Valid pairs: {valid_pairs}")
        
        for layer in CONFIG['layers_to_test']:
            layer_data = window_data[window_data['layer'] == layer]
            
            if len(layer_data) == 0:
                continue
            
            mean_rec = layer_data['rec_rv'].mean()
            mean_base = layer_data['base_rv'].mean()
            diff = mean_rec - mean_base
            std_diff = layer_data['diff'].std()
            
            # T-Test
            t_stat, p_val = ttest_rel(layer_data['rec_rv'], layer_data['base_rv'])
            
            # Cohen's d
            d = diff / (std_diff + 1e-9)
            
            print(f"\n  LAYER {layer}:")
            print(f"    Rec R_V:   {mean_rec:.4f} ± {layer_data['rec_rv'].std():.4f}")
            print(f"    Base R_V:  {mean_base:.4f} ± {layer_data['base_rv'].std():.4f}")
            print(f"    Delta:     {diff:.4f}")
            print(f"    P-Value:   {p_val:.2e}")
            print(f"    Cohen's d: {d:.4f}")
            
            if abs(d) > 3.0:
                print(f"    ⭐⭐⭐ SINGULARITY DETECTED (d > 3.0)")
            elif abs(d) > 2.0:
                print(f"    ⭐⭐ STRONG EFFECT (d > 2.0)")
            elif abs(d) > 0.8:
                print(f"    ✅ Moderate Effect (d > 0.8)")
            else:
                print(f"    ❌ No Effect")
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"window_sweep_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n\nResults saved to: {filename}")
    
    return df

if __name__ == "__main__":
    results = run_window_sweep()

