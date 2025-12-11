"""
phase2_layer_ablation_sweep.py

Task: Locate the "Speaker" layer by ablating ENTIRE layers (all heads).
Hypothesis: At some layer L (24-31), ablating all heads will destroy the recursive behavior while maintaining (or not affecting) the geometric contraction.

Method:
1. Load Recursive Prompts (L4/L5).
2. For each layer L in [24, 25, ..., 31]:
   - Zero out ALL attention heads at Layer L.
   - Generate response.
   - Score behavior.
3. Find the layer with maximal behavioral drop.

This is a coarse-grain search. Once we find the layer, we can find the heads.
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from contextlib import contextmanager

sys.path.insert(0, os.path.abspath('.'))

from src.core.models import load_model, set_seed
from src.core.utils import behavior_score
from prompts.loader import PromptLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS_TO_TEST = list(range(24, 32))

@contextmanager
def ablate_layer(model, layer_idx):
    """
    Context manager to zero-out the ENTIRE attention output at a specific layer.
    """
    # Hook the self_attn module
    layer = model.model.layers[layer_idx].self_attn
    
    def hook_fn(module, inputs, outputs):
        # outputs[0] is the attention output tensor
        # We zero it out completely.
        # This simulates removing the attention mechanism's contribution to the residual stream for this layer.
        # Note: The residual connection (x + attn(x)) still adds x. So we are removing the update.
        
        attn_output = outputs[0]
        zeros = torch.zeros_like(attn_output)
        return (zeros,) + outputs[1:]

    handle = layer.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()

def run_sweep():
    print("Initializing Phase 2: Layer-Wise Ablation Sweep...")
    set_seed(42)
    
    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    loader = PromptLoader()
    
    # Use subset of recursive prompts
    prompts = loader.get_by_group("L4_full", limit=10, seed=42)
    print(f"Testing on {len(prompts)} recursive prompts.")
    
    # 1. Baseline
    print("Computing Baseline...")
    base_scores = []
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        base_scores.append(behavior_score(text))
    
    mean_base = np.mean(base_scores)
    print(f"Baseline Behavior Score: {mean_base:.2f}")
    
    results = []
    
    # 2. Sweep
    print("\nStarting Layer Sweep...")
    for layer_idx in LAYERS_TO_TEST:
        layer_scores = []
        texts = []
        
        for p in prompts:
            with ablate_layer(model, layer_idx):
                inputs = tokenizer(p, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
                text = tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                layer_scores.append(behavior_score(text))
                texts.append(text)
        
        mean_score = np.mean(layer_scores)
        drop = mean_base - mean_score
        pct_drop = (drop / mean_base) * 100 if mean_base > 0 else 0
        
        print(f"Layer {layer_idx}: Score {mean_score:.2f} (Drop: {pct_drop:.1f}%)")
        print(f"  Sample: {texts[0][:80]}...")
        
        results.append({
            "layer": layer_idx,
            "mean_score": mean_score,
            "drop": drop,
            "pct_drop": pct_drop,
            "sample_text": texts[0]
        })
        
    # Save
    os.makedirs("results/dec11_evening", exist_ok=True)
    pd.DataFrame(results).to_csv("results/dec11_evening/layer_ablation_sweep.csv", index=False)
    
    # Find max drop
    best = max(results, key=lambda x: x["drop"])
    print(f"\nMax Impact Layer: {best['layer']} (Drop: {best['pct_drop']:.1f}%)")
    
    # Log
    with open("logs/dec11_evening/layer_ablation_log.txt", "w") as f:
        f.write("# Layer Ablation Sweep\n\n")
        f.write(f"Baseline Score: {mean_base:.2f}\n\n")
        for r in results:
            f.write(f"Layer {r['layer']}: Score {r['mean_score']:.2f} (-{r['pct_drop']:.1f}%)\n")
            f.write(f"  Sample: {r['sample_text']}\n")

if __name__ == "__main__":
    run_sweep()

