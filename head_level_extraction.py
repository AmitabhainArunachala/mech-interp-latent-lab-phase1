#!/usr/bin/env python3
"""
HEAD-LEVEL ACTIVATION EXTRACTION: Extract activations from critical heads only
Prepares targeted patch vectors for unified test
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
    "num_heads": 32,
    "head_dim": 128,
    "critical_heads": {
        27: [11, 1, 22],  # L27 critical heads
        25: [23, 28, 3, 17, 19]  # L25 top heads (distributed)
    },
    "window_size": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

PROMPTS = {
    "champion": experimental_prompts["hybrid_l5_math_01"]["text"],
    "baseline": "The history of the Roman Empire is characterized by a long period of expansion followed by a gradual decline."
}

def extract_head_activations(model, tokenizer, prompt_text, layer_idx, head_indices):
    """
    Extract activations from specific heads at a layer
    
    Returns: Dictionary with head activations
    """
    tokens = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    
    if tokens['input_ids'].shape[1] < CONFIG['window_size'] + 1:
        return None
    
    head_activations = {}
    
    # Hook into V-projection to extract head-specific activations
    def make_v_hook(head_idx):
        activations = []
        def v_hook(module, input, output):
            # output: [batch, seq, hidden] = [batch, seq, num_heads * head_dim]
            batch, seq, hidden = output.shape
            heads = CONFIG['num_heads']
            head_dim = hidden // heads
            
            # Reshape to [batch, seq, heads, head_dim]
            v_reshaped = output.view(batch, seq, heads, head_dim)
            
            # Extract only this head's activations
            head_v = v_reshaped[:, :, head_idx, :].clone()  # [batch, seq, head_dim]
            activations.append(head_v.detach().cpu())
        return v_hook, activations
    
    hooks = []
    activations_dict = {}
    
    for head_idx in head_indices:
        hook_fn, acts = make_v_hook(head_idx)
        activations_dict[head_idx] = acts
        hooks.append(
            model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(hook_fn)
        )
    
    with torch.no_grad():
        model(**tokens)
    
    # Extract the last window for each head
    for head_idx in head_indices:
        if activations_dict[head_idx]:
            head_v = activations_dict[head_idx][-1]  # [batch, seq, head_dim]
            # Take last window_size tokens
            head_v_window = head_v[0, -CONFIG['window_size']:, :]  # [window_size, head_dim]
            head_activations[head_idx] = head_v_window
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    return head_activations

def combine_head_activations(head_activations_dict, head_indices):
    """
    Combine multiple head activations into a single tensor
    
    head_activations_dict: {head_idx: tensor [window_size, head_dim]}
    Returns: Combined tensor [window_size, num_heads * head_dim] with zeros for non-critical heads
    """
    if not head_activations_dict:
        return None
    
    window_size = list(head_activations_dict.values())[0].shape[0]
    head_dim = list(head_activations_dict.values())[0].shape[1]
    
    # Create full tensor with zeros
    combined = torch.zeros(window_size, CONFIG['num_heads'] * head_dim)
    
    # Fill in critical heads
    for head_idx in head_indices:
        if head_idx in head_activations_dict:
            start_idx = head_idx * head_dim
            end_idx = start_idx + head_dim
            combined[:, start_idx:end_idx] = head_activations_dict[head_idx]
    
    return combined

def run_extraction():
    """Extract activations from critical heads"""
    print("="*70)
    print("HEAD-LEVEL ACTIVATION EXTRACTION")
    print("="*70)
    print(f"Model: {CONFIG['model_name']}")
    print(f"Critical heads: {CONFIG['critical_heads']}")
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
    
    extracted = {}
    
    for layer_idx, head_indices in CONFIG['critical_heads'].items():
        print(f"\nExtracting from Layer {layer_idx}, Heads {head_indices}...")
        
        # Extract from champion prompt
        champ_heads = extract_head_activations(
            model, tokenizer, PROMPTS["champion"], layer_idx, head_indices
        )
        
        if champ_heads:
            # Combine into single tensor
            champ_combined = combine_head_activations(champ_heads, head_indices)
            extracted[f"L{layer_idx}_champion"] = {
                'individual': champ_heads,
                'combined': champ_combined,
                'head_indices': head_indices
            }
            print(f"  ✅ Extracted {len(champ_heads)} heads, combined shape: {champ_combined.shape}")
        
        # Extract from baseline prompt (for comparison)
        base_heads = extract_head_activations(
            model, tokenizer, PROMPTS["baseline"], layer_idx, head_indices
        )
        
        if base_heads:
            base_combined = combine_head_activations(base_heads, head_indices)
            extracted[f"L{layer_idx}_baseline"] = {
                'individual': base_heads,
                'combined': base_combined,
                'head_indices': head_indices
            }
            print(f"  ✅ Extracted baseline for comparison")
    
    # Save extracted activations
    print("\n" + "="*70)
    print("SAVING EXTRACTED ACTIVATIONS")
    print("="*70)
    
    save_dict = {}
    for key, value in extracted.items():
        if value['combined'] is not None:
            save_dict[f"{key}_combined"] = value['combined'].numpy()
            save_dict[f"{key}_head_indices"] = value['head_indices']
    
    # Save as numpy file
    np.savez('critical_heads_activations.npz', **save_dict)
    print("✅ Saved to: critical_heads_activations.npz")
    
    # Create summary
    summary = {
        'layers': list(CONFIG['critical_heads'].keys()),
        'heads': CONFIG['critical_heads'],
        'window_size': CONFIG['window_size'],
        'head_dim': CONFIG['head_dim']
    }
    
    import json
    with open('critical_heads_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("✅ Summary saved to: critical_heads_summary.json")
    
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print("\nNext: Use these activations for targeted head-level patching")
    print("Files:")
    print("  - critical_heads_activations.npz (activation tensors)")
    print("  - critical_heads_summary.json (metadata)")
    
    return extracted

if __name__ == "__main__":
    extracted = run_extraction()

