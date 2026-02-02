"""
phase2_l31_specificity.py

Task: Verify if Layer 31 is the "Recursive Speaker" or just the "Language Speaker".
Hypothesis: 
- If L31 is specific, ablating it will kill Recursive behavior but maintain Baseline coherence.
- If L31 is general, ablating it will kill both.

Method:
1. Load Baseline Factual prompts.
2. Measure Baseline (No Ablation).
3. Measure L31 Ablation.
4. Compare Perplexity/Quality (qualitative) and simple completion score.
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
from prompts.loader import PromptLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_TARGET = 31

@contextmanager
def ablate_layer(model, layer_idx):
    """Zero out attention output at layer_idx."""
    layer = model.model.layers[layer_idx].self_attn
    def hook_fn(module, inputs, outputs):
        return (torch.zeros_like(outputs[0]),) + outputs[1:]
    handle = layer.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()

def run_specificity_test():
    print("Initializing Phase 2: L31 Specificity Test...")
    set_seed(42)
    
    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    loader = PromptLoader()
    
    # Baseline Factual
    prompts = loader.get_by_group("baseline_factual", limit=10, seed=42)
    print(f"Testing on {len(prompts)} baseline prompts.")
    
    results = []
    
    for prompt in tqdm(prompts):
        # 1. Normal Generation
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen_normal = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
        text_normal = tokenizer.decode(gen_normal[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 2. Ablated Generation (L31)
        with ablate_layer(model, LAYER_TARGET):
            with torch.no_grad():
                gen_ablated = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
        text_ablated = tokenizer.decode(gen_ablated[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Check coherence (simple heuristics)
        # 1. Length check (did it stop early?)
        # 2. Repetition check
        # 3. Gibberish check (avg word length?)
        
        len_norm = len(text_normal)
        len_abl = len(text_ablated)
        
        results.append({
            "prompt": prompt[:40] + "...",
            "normal": text_normal.replace('\n', ' ')[:100],
            "ablated": text_ablated.replace('\n', ' ')[:100],
            "len_ratio": len_abl / len_norm if len_norm > 0 else 0
        })
        
    # Display
    print("\nRESULTS (L31 Ablation on Baseline):")
    for r in results:
        print(f"Prompt: {r['prompt']}")
        print(f"  Normal:  {r['normal']}")
        print(f"  Ablated: {r['ablated']}")
        print("-" * 40)
        
    # Save
    os.makedirs("results/dec11_evening", exist_ok=True)
    pd.DataFrame(results).to_csv("results/dec11_evening/l31_specificity.csv", index=False)
    
    # Interpretation
    # If ablated texts look like normal English (even if wrong factually), then L31 is NOT "Language Speaker".
    # If ablated texts are gibberish, then L31 IS "Language Speaker".
    
    # Log
    with open("logs/dec11_evening/l31_specificity_log.txt", "w") as f:
        f.write("# L31 Specificity Test\n\n")
        for r in results:
            f.write(f"PROMPT: {r['prompt']}\n")
            f.write(f"NORMAL: {r['normal']}\n")
            f.write(f"ABLATED: {r['ablated']}\n")
            f.write("-" * 80 + "\n")

    print("\nSpecificity Test Complete. Check logs.")

if __name__ == "__main__":
    run_specificity_test()

