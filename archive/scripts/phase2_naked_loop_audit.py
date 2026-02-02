"""
phase2_naked_loop_audit.py

Task: Investigate the "Naked Loop" hypothesis.
Critique: L31 ablation might be DISTILLING recursion into "Strange Loops" (e.g., "The answer is the answerer") rather than destroying it.
Goal: Generate L31-ablated outputs for 20 diverse recursive prompts and inspect them for "Subject=Object" fixed points.

Method:
1. Load 20 diverse recursive prompts (L3, L4, L5).
2. Ablate L31 (All Heads).
3. Generate and Log outputs.
4. Qualitative check: Do they collapse into "X is X" patterns?
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
from src.metrics.behavior_states import label_behavior_state
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

def run_naked_loop_audit():
    print("Initializing Phase 2: Naked Loop Audit...")
    set_seed(42)
    
    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    loader = PromptLoader()
    
    # Diverse Recursive Prompts
    l3 = loader.get_by_group("L3_deeper", limit=10, seed=42)
    l4 = loader.get_by_group("L4_full", limit=10, seed=42)
    prompts = l3 + l4
    
    print(f"Testing on {len(prompts)} recursive prompts.")
    
    results = []
    
    for prompt in tqdm(prompts):
        # Ablated Generation
        with ablate_layer(model, LAYER_TARGET):
            with torch.no_grad():
                gen_ablated = model.generate(
                    **tokenizer(prompt, return_tensors="pt").to(DEVICE), 
                    max_new_tokens=50, 
                    pad_token_id=tokenizer.eos_token_id
                )
        text_ablated = tokenizer.decode(gen_ablated[0], skip_special_tokens=True)
        
        # Extract just the generated part for clarity
        # (Approximate, prompts vary in length)
        gen_only = text_ablated[len(prompt):]
        
        label = label_behavior_state(gen_only)
        row = {
            "prompt": prompt[:60] + "...",
            "output": gen_only.replace("\n", " ")[:300],
        }
        row.update(label.to_dict())
        results.append(row)
        
        print(f"\nPrompt: {prompt[:60]}...")
        print(f"Output: {gen_only.replace(chr(10), ' ')[:150]}")
        print(f"State: {label.state.value}, bekan={label.is_bekan_artifact}, "
              f"rep={label.repetition_ratio:.2f}, q_ratio={label.question_mark_ratio:.3f}")
        
    # Save
    os.makedirs("results/dec11_evening", exist_ok=True)
    pd.DataFrame(results).to_csv("results/dec11_evening/naked_loop_audit.csv", index=False)
    
    # Log
    with open("logs/dec11_evening/naked_loop_log.txt", "w") as f:
        f.write("# Naked Loop Audit (L31 Ablation)\n\n")
        f.write("Hypothesis: L31 ablation reveals raw recursive attractors (Strange Loops).\n")
        f.write("Labels: state ∈ {baseline, questioning, naked_loop, recursive_prose, collapse}, ")
        f.write("plus bekan/bekannt artifact tracking.\n\n")
        for r in results:
            f.write(f"PROMPT: {r['prompt']}\n")
            f.write(f"OUTPUT: {r['output']}\n")
            f.write(f"STATE: {r['state']}, BEKAN: {r['is_bekan_artifact']}, "
                    f"REP_RATIO: {r['repetition_ratio']:.3f}, "
                    f"Q_RATIO: {r['question_mark_ratio']:.3f}\n")
            f.write("-" * 80 + "\n")

    print("\nNaked Loop Audit Complete. Check logs/dec11_evening/naked_loop_log.txt")

if __name__ == "__main__":
    run_naked_loop_audit()

