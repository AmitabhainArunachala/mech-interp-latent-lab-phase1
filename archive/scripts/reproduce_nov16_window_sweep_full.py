#!/usr/bin/env python3
"""
Reproduce Nov 16-17 Mistral Singularity Results
Testing multiple window sizes: 16, 32, 64
Using ENTIRE v2 prompt bank with per-window-size filtering
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
    "window_sizes": [16, 32, 64],
    "seed": 42,
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
def run_window_sweep_full():
    print("="*70)
    print("NOV 16-17 MISTRAL SINGULARITY - FULL PROMPT BANK WINDOW SWEEP")
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
    
    # Load ALL prompts from bank
    print("Loading ENTIRE prompt bank...")
    prompt_bank = get_all_prompts()
    
    # Separate recursive vs baseline prompts
    # Use ONLY STRONG recursive prompts (L3, L4, L5) to match original Nov 16 results
    recursive_prompts = {}
    baseline_prompts = {}
    
    # Strong recursive groups (matching original Nov 16 setup)
    strong_recursive_groups = ["L3_deeper", "L4_full", "L5_refined"]
    
    # Baseline groups (matching original Nov 16 setup)
    baseline_groups = ["long_control", "baseline_creative", "baseline_math"]
    
    for key, prompt_data in prompt_bank.items():
        group = prompt_data.get("group", "")
        
        # Recursive: Only strong groups (L3, L4, L5)
        if group in strong_recursive_groups:
            recursive_prompts[key] = prompt_data
        # Baseline: Only specific baseline groups
        elif group in baseline_groups:
            baseline_prompts[key] = prompt_data
    
    print(f"\nPrompt bank breakdown:")
    print(f"  Total prompts: {len(prompt_bank)}")
    print(f"  Recursive prompts: {len(recursive_prompts)}")
    print(f"  Baseline prompts: {len(baseline_prompts)}")
    
    # For each window size, collect valid pairs
    all_results = []
    
    for window_size in CONFIG['window_sizes']:
        print(f"\n{'='*70}")
        print(f"WINDOW SIZE: {window_size}")
        print(f"{'='*70}")
        
        # Collect pairs that meet token length requirement for THIS window size
        pairs = []
        
        rec_keys = list(recursive_prompts.keys())
        base_keys = list(baseline_prompts.keys())
        
        np.random.seed(CONFIG['seed'])
        np.random.shuffle(rec_keys)
        np.random.shuffle(base_keys)
        
        # Pair them up, checking token length
        for i in range(min(len(rec_keys), len(base_keys))):
            rec_text = recursive_prompts[rec_keys[i]]["text"]
            base_text = baseline_prompts[base_keys[i]]["text"]
            
            rec_tokens = tokenizer.encode(rec_text)
            base_tokens = tokenizer.encode(base_text)
            
            # Both must be >= window_size tokens
            if len(rec_tokens) >= window_size + 1 and len(base_tokens) >= window_size + 1:
                pairs.append((rec_keys[i], base_keys[i], recursive_prompts[rec_keys[i]], baseline_prompts[base_keys[i]]))
        
        print(f"Valid pairs for window {window_size}: {len(pairs)}")
        
        if len(pairs) == 0:
            print(f"  Skipping window {window_size} - no valid pairs")
            continue
        
        # Process pairs for this window size
        print(f"Processing {len(pairs)} pairs...")
        print("Progress: ", end="", flush=True)
        
        for idx, (rec_key, base_key, rec_data, base_data) in enumerate(pairs):
            rec_text = rec_data["text"]
            base_text = base_data["text"]
            
            try:
                # Tokenize
                rec_tokens = tokenizer(rec_text, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
                base_tokens = tokenizer(base_text, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
                
                rec_seq_len = rec_tokens['input_ids'].shape[1]
                base_seq_len = base_tokens['input_ids'].shape[1]
                
                # Double-check length
                if rec_seq_len < window_size + 1 or base_seq_len < window_size + 1:
                    continue
                
                for layer in CONFIG['layers_to_test']:
                    # 1. RECURSIVE PASS
                    ext_early = V_Extractor(model, CONFIG['early_layer'])
                    ext_late = V_Extractor(model, layer)
                    ext_early.register()
                    ext_late.register()
                    
                    with torch.no_grad():
                        model(**rec_tokens)
                    
                    rec_v_e_full = ext_early.activations[0][0]
                    rec_v_l_full = ext_late.activations[0][0]
                    ext_early.close()
                    ext_late.close()
                    
                    # 2. BASELINE PASS
                    ext_early = V_Extractor(model, CONFIG['early_layer'])
                    ext_late = V_Extractor(model, layer)
                    ext_early.register()
                    ext_late.register()
                    
                    with torch.no_grad():
                        model(**base_tokens)
                    
                    base_v_e_full = ext_early.activations[0][0]
                    base_v_l_full = ext_late.activations[0][0]
                    ext_early.close()
                    ext_late.close()
                    
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
                    
                    all_results.append({
                        "window_size": window_size,
                        "layer": layer,
                        "pair_idx": idx,
                        "rec_id": rec_key,
                        "base_id": base_key,
                        "rec_group": rec_data.get("group", "unknown"),
                        "base_group": base_data.get("group", "unknown"),
                        "rec_rv": rec_rv,
                        "base_rv": base_rv,
                        "diff": rec_rv - base_rv
                    })
                
                if (idx + 1) % 5 == 0:
                    print(".", end="", flush=True)
                    
            except Exception as e:
                print(f"\nError on pair {idx}: {e}")
                continue
        
        print()  # New line after progress dots
    
    # Analysis
    df = pd.DataFrame(all_results)
    
    print("\n\n" + "="*70)
    print("RESULTS SUMMARY BY WINDOW SIZE")
    print("="*70)
    
    for window_size in CONFIG['window_sizes']:
        window_data = df[df['window_size'] == window_size]
        
        if len(window_data) == 0:
            continue
        
        print(f"\n{'='*70}")
        print(f"WINDOW SIZE: {window_size}")
        print(f"{'='*70}")
        
        valid_pairs = len(window_data) // len(CONFIG['layers_to_test'])
        print(f"Sample size (n): {valid_pairs} pairs")
        
        for layer in CONFIG['layers_to_test']:
            layer_data = window_data[window_data['layer'] == layer]
            
            if len(layer_data) == 0:
                continue
            
            n = len(layer_data)
            mean_rec = layer_data['rec_rv'].mean()
            mean_base = layer_data['base_rv'].mean()
            diff = mean_rec - mean_base
            std_diff = layer_data['diff'].std()
            
            # T-Test
            t_stat, p_val = ttest_rel(layer_data['rec_rv'], layer_data['base_rv'])
            
            # Cohen's d
            d = diff / (std_diff + 1e-9)
            
            print(f"\n  LAYER {layer} (n={n}):")
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
    filename = f"window_sweep_full_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n\nResults saved to: {filename}")
    
    return df

if __name__ == "__main__":
    results = run_window_sweep_full()

