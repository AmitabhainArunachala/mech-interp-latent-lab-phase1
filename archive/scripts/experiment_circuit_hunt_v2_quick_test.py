"""
QUICK TEST: Verify the experiment code works before running full version.
Tests just a few samples to ensure everything is set up correctly.
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, List
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))

from src.core.models import load_model, set_seed
from src.metrics.rv import participation_ratio
from prompts.loader import PromptLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
WINDOW = 16
EARLY_LAYER = 5
MEASURE_LAYER = 27
NUM_HEADS = 32
SEED = 42

def compute_rv(v_early: torch.Tensor, v_late: torch.Tensor, window: int = 16) -> float:
    pr_early = participation_ratio(v_early, window_size=window)
    pr_late = participation_ratio(v_late, window_size=window)
    if pr_early == 0 or np.isnan(pr_early) or np.isnan(pr_late):
        return float('nan')
    return float(pr_late / pr_early)

@contextmanager
def capture_v_projections(model, layers: List[int]):
    storage = {l: None for l in layers}
    handles = []
    
    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            storage[layer_idx] = out.detach()
            return out
        return hook_fn
    
    for l in layers:
        h = model.model.layers[l].self_attn.v_proj.register_forward_hook(make_hook(l))
        handles.append(h)
    
    try:
        yield storage
    finally:
        for h in handles:
            h.remove()

@contextmanager
def zero_heads_at_layer(model, layer_idx: int, head_indices: List[int]):
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            attn_output = out[0]
        else:
            attn_output = out
            
        batch, seq, hidden = attn_output.shape
        reshaped = attn_output.clone().view(batch, seq, num_heads, head_dim)
        for h in head_indices:
            reshaped[:, :, h, :] = 0
        modified = reshaped.view(batch, seq, hidden)
        
        if isinstance(out, tuple):
            return (modified,) + out[1:]
        return modified
    
    handle = model.model.layers[layer_idx].self_attn.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()

def run_and_measure_rv(model, tokenizer, prompt: str, intervention_fn=None) -> float:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    with capture_v_projections(model, [EARLY_LAYER, MEASURE_LAYER]) as storage:
        if intervention_fn is not None:
            with intervention_fn(model):
                with torch.no_grad():
                    model(**enc)
        else:
            with torch.no_grad():
                model(**enc)
    
    v_early = storage[EARLY_LAYER]
    v_late = storage[MEASURE_LAYER]
    
    if v_early is None or v_late is None:
        return float('nan')
    
    return compute_rv(v_early[0], v_late[0], WINDOW)

def main():
    print("="*60)
    print("QUICK TEST: Circuit Hunt V2")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Model: {MODEL_NAME}")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: No CUDA available. This will be VERY slow on CPU.")
        print("   Consider running on RunPod or a GPU machine.")
        print("   Continuing anyway for testing...")
    
    set_seed(SEED)
    
    print("\nLoading model...")
    model, tokenizer = load_model(MODEL_NAME, device=DEVICE)
    model.eval()
    
    print("Loading prompts...")
    loader = PromptLoader()
    recursive_prompts = loader.get_by_pillar("dose_response", limit=5)
    baseline_prompts = loader.get_by_pillar("baselines", limit=5)
    
    print(f"Loaded {len(recursive_prompts)} recursive, {len(baseline_prompts)} baseline prompts")
    
    # Quick test: Just test a few heads at one layer
    print("\n" + "="*60)
    print("QUICK TEST: Testing 3 heads at L15")
    print("="*60)
    
    test_layer = 15
    test_heads = [0, 10, 20]  # Sample heads
    test_prompts = recursive_prompts[:2] + baseline_prompts[:2]
    
    results = []
    
    for prompt in tqdm(test_prompts, desc="Prompts"):
        rv_baseline = run_and_measure_rv(model, tokenizer, prompt)
        
        for head in test_heads:
            rv_zero = run_and_measure_rv(
                model, tokenizer, prompt,
                intervention_fn=lambda m, l=test_layer, h=head: zero_heads_at_layer(m, l, [h])
            )
            
            results.append({
                "layer": test_layer,
                "head": head,
                "prompt_type": "recursive" if prompt in recursive_prompts else "baseline",
                "rv_baseline": rv_baseline,
                "rv_zero": rv_zero,
                "delta": rv_zero - rv_baseline,
            })
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for r in results:
        print(f"L{r['layer']} H{r['head']} ({r['prompt_type']}): "
              f"RV_baseline={r['rv_baseline']:.4f}, "
              f"RV_zero={r['rv_zero']:.4f}, "
              f"Δ={r['delta']:+.4f}")
    
    # Save results
    output_dir = Path("results/circuit_hunt_v2_focused")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(output_dir / f"quick_test_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Quick test complete! Results saved to: {output_dir}")
    print("\nIf this works, run the full experiment:")
    print("  python3 experiment_circuit_hunt_v2_focused.py")

if __name__ == "__main__":
    main()

