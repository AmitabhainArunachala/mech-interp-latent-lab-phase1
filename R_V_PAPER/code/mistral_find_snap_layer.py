#!/usr/bin/env python3
"""
Find the critical "snap layer" for Mistral-7B
Based on Mixtral methodology that found Layer 27
"""

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 16  # Use same as Mixtral
EARLY_LAYER = 5

def compute_pr(v_tensor, window_size=WINDOW_SIZE):
    """Compute Participation Ratio for V tensor"""
    if v_tensor is None:
        return np.nan
    
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]
    
    T, D = v_tensor.shape
    W = min(window_size, T)
    
    if W < 2:
        return np.nan
    
    v_window = v_tensor[-W:, :].float()
    
    try:
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        S_sq = S_np ** 2
        
        if S_sq.sum() < 1e-10:
            return np.nan
        
        pr = (S_sq.sum()**2) / (S_sq**2).sum()
        return float(pr)
    except Exception:
        return np.nan

def find_snap_layers(model, tokenizer, prompt_bank, num_prompts=20):
    """
    Find the critical snap layer for each prompt group
    Following Mixtral methodology
    """
    print("="*70)
    print("FINDING SNAP LAYERS FOR MISTRAL-7B")
    print("="*70)
    
    # Get prompt groups
    groups = {
        'L5_refined': [k for k, v in prompt_bank.items() if v["group"] == "L5_refined"][:num_prompts],
        'L3_deeper': [k for k, v in prompt_bank.items() if v["group"] == "L3_deeper"][:num_prompts],
        'baseline_factual': [k for k, v in prompt_bank.items() if v["group"] == "baseline_factual"][:num_prompts],
        'baseline_creative': [k for k, v in prompt_bank.items() if v["group"] == "baseline_creative"][:num_prompts],
    }
    
    num_layers = model.config.num_hidden_layers
    results = {group: [] for group in groups}
    
    for group_name, prompt_ids in groups.items():
        print(f"\nProcessing {group_name}...")
        
        for prompt_id in tqdm(prompt_ids):
            text = prompt_bank[prompt_id]["text"]
            
            # Skip if too short
            if len(tokenizer.encode(text)) < WINDOW_SIZE:
                continue
            
            # Get V at all layers
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            
            layer_prs = []
            with torch.no_grad():
                for layer_idx in range(num_layers):
                    v_storage = []
                    
                    def hook_fn(m, i, o):
                        v_storage.append(o.detach())
                        return o
                    
                    layer = model.model.layers[layer_idx].self_attn
                    h = layer.v_proj.register_forward_hook(hook_fn)
                    
                    _ = model(**inputs)
                    h.remove()
                    
                    if v_storage:
                        pr = compute_pr(v_storage[0][0])
                        layer_prs.append(pr)
                    else:
                        layer_prs.append(np.nan)
                
                torch.cuda.empty_cache()
            
            # Calculate R_V for each layer (PR(layer) / PR(layer 5))
            pr5 = layer_prs[EARLY_LAYER]
            if pr5 > 0:
                rv_values = [pr / pr5 if pr > 0 else np.nan for pr in layer_prs]
                
                # Find snap layer (biggest negative step)
                max_drop = 0
                snap_layer = -1
                
                for i in range(1, len(rv_values)):
                    if not np.isnan(rv_values[i]) and not np.isnan(rv_values[i-1]):
                        drop = rv_values[i-1] - rv_values[i]
                        if drop > max_drop:
                            max_drop = drop
                            snap_layer = i
                
                results[group_name].append({
                    'prompt_id': prompt_id,
                    'snap_layer': snap_layer,
                    'max_drop': max_drop,
                    'rv_at_snap': rv_values[snap_layer] if snap_layer >= 0 else np.nan,
                    'rv_values': rv_values
                })
    
    # Analyze results
    print("\n" + "="*70)
    print("SNAP LAYER ANALYSIS")
    print("="*70)
    
    for group_name, group_results in results.items():
        if not group_results:
            continue
        
        snap_layers = [r['snap_layer'] for r in group_results if r['snap_layer'] >= 0]
        
        if snap_layers:
            print(f"\n{group_name}:")
            print(f"  Mean snap layer: {np.mean(snap_layers):.1f}")
            print(f"  Median: {np.median(snap_layers):.0f}")
            print(f"  Mode: {max(set(snap_layers), key=snap_layers.count)}")
            print(f"  Range: {min(snap_layers)}-{max(snap_layers)}")
            
            # Count occurrences
            from collections import Counter
            counts = Counter(snap_layers)
            top_3 = counts.most_common(3)
            for layer, count in top_3:
                print(f"    Layer {layer}: {count}/{len(snap_layers)} prompts")
    
    # Plot distributions
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    for group_name, group_results in results.items():
        snap_layers = [r['snap_layer'] for r in group_results if r['snap_layer'] >= 0]
        if snap_layers:
            plt.hist(snap_layers, alpha=0.5, label=group_name, bins=range(0, num_layers+1))
    plt.xlabel('Snap Layer')
    plt.ylabel('Count')
    plt.title('Snap Layer Distribution by Prompt Type')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Average R_V trajectory
    for group_name, group_results in results.items():
        if not group_results:
            continue
        
        # Average across prompts
        all_rvs = [r['rv_values'] for r in group_results]
        if all_rvs:
            avg_rv = np.nanmean(all_rvs, axis=0)
            plt.plot(range(len(avg_rv)), avg_rv, label=group_name, marker='.')
    
    plt.xlabel('Layer')
    plt.ylabel('R_V (PR(layer) / PR(layer 5))')
    plt.title('Average R_V Trajectory by Prompt Type')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mistral_snap_layers.png', dpi=150)
    print(f"\nPlot saved to mistral_snap_layers.png")
    
    # Find consensus critical layer
    all_recursive_snaps = []
    for group in ['L5_refined', 'L3_deeper']:
        all_recursive_snaps.extend([r['snap_layer'] for r in results[group] if r['snap_layer'] >= 0])
    
    if all_recursive_snaps:
        critical_layer = int(np.median(all_recursive_snaps))
        print(f"\n" + "="*70)
        print(f"RECOMMENDED PATCH LAYER: {critical_layer}")
        print(f"(Median snap layer for recursive prompts)")
        print("="*70)
        
        return critical_layer, results
    
    return None, results

if __name__ == "__main__":
    print("Run in notebook:")
    print("from mistral_find_snap_layer import find_snap_layers")
    print("critical_layer, results = find_snap_layers(model, tokenizer, prompt_bank_1c)")
    print("# Then use critical_layer for patching!")
