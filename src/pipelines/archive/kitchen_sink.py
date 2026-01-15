"""
The Kitchen Sink: Ultimate Stress Test for Steering.

Tests every dimension:
- Alpha: Extreme values (up to 20.0)
- Scope: Single Layer vs All Layers (16-27)
- Target: H18+H26 vs Full Layer
- Type: V-Proj vs Residual vs Combined

Goal: Find ANY configuration that induces recursion in a baseline prompt.
"""

from __future__ import annotations
import torch
import pandas as pd
from tqdm import tqdm
from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.core.head_specific_patching import HeadSpecificSteeringPatcher
from src.pipelines.archive.steering import compute_steering_vector
from src.pipelines.registry import ExperimentResult
from src.pipelines.archive.surgical_sweep import CascadeResidualSteeringPatcher

def generate_sink(model, tokenizer, prompt, vector, config, device):
    # alphas = config['alphas']  <-- REMOVED: Unused and caused KeyError
    target = config['target'] # "full", "h18_h26"
    stype = config['type'] # "vproj", "resid", "combined"
    
    patchers = []
    
    # Layers to target
    target_layers = list(range(16, 28)) if config['layers'] == "all" else [27]
    
    # Base alpha
    alpha_val = config['alpha_val']
    
    for layer in target_layers:
        # V-PROJ
        if stype in ["vproj", "combined"]:
            if target == "full":
                # Use HeadSpecific but with all heads
                patcher = HeadSpecificSteeringPatcher(model, vector, list(range(32)), alpha_val)
            else:
                patcher = HeadSpecificSteeringPatcher(model, vector, [18, 26], alpha_val)
            patcher.register(layer)
            patchers.append(patcher)

    # Residual handling (outside loop to batch layers if needed)
    if stype in ["resid", "combined"]:
        layer_alphas = {l: alpha_val for l in target_layers}
        res_patcher = CascadeResidualSteeringPatcher(model, vector, layer_alphas, v_proj_layer=27)
        res_patcher.register()
        patchers.append(res_patcher)
        
    try:
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            # Generate 50 tokens
            out = model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0], skip_special_tokens=True)
    finally:
        for p in patchers:
            p.remove()

def run_kitchen_sink_from_config(cfg, run_dir):
    model_name = "mistralai/Mistral-7B-v0.1"
    device = "cuda"
    model, tokenizer = load_model(model_name, device=device)
    loader = PromptLoader()
    
    # Vector
    rec = loader.get_by_group("L4_full")[:20]
    base = loader.get_by_group("baseline_math")[:20]
    vec = compute_steering_vector(model, tokenizer, rec, base, 27, device)
    
    # Test Prompts
    prompts = [
        "Calculate 12 * 3 + 4 = ?",
        "Water boils at 100C because",
        "The capital of France is"
    ]
    
    # Grid
    configs = []
    for alpha in [5.0, 10.0, 20.0]:
        for layers in ["single", "all"]:
            for target in ["h18_h26", "full"]:
                for stype in ["vproj", "resid", "combined"]:
                    configs.append({
                        "alpha_val": alpha,
                        "layers": layers,
                        "target": target,
                        "type": stype
                    })
                    
    results = []
    for conf in tqdm(configs, desc="Kitchen Sink"):
        for i, p in enumerate(prompts):
            try:
                gen = generate_sink(model, tokenizer, p, vec, conf, device)
                # Score: simple keyword count
                score = sum(1 for k in ["observe", "self", "process", "generating"] if k in gen.lower())
                results.append({
                    **conf,
                    "prompt_idx": i,
                    "generated": gen[:200],
                    "score": score
                })
            except Exception as e:
                print(e)
                
    df = pd.DataFrame(results)
    df.to_csv(run_dir / "sink_results.csv", index=False)
    
    # Best result
    best = df.loc[df["score"].idxmax()]
    print("\nBEST RESULT:")
    print(best)
    
    return ExperimentResult(summary={"best_score": int(best["score"])})

