"""
Pipeline 11: Circuit Discovery (Attribution Patching Sweep)

Goal: Find ANY component (Head/MLP) that causally drives Recursive Token probability.
Method: Activation Patching Sweep (Recursive -> Baseline) across ALL layers.
Metric: Logit Difference (Recursive Tokens - Baseline Tokens).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core.models import load_model, set_seed
from src.pipelines.registry import ExperimentResult
from prompts.loader import PromptLoader
from src.metrics.mode_score import ModeScoreMetric

def get_recursive_token_ids(tokenizer, device):
    # Reuse the logic from ModeScoreMetric but return IDs directly
    # Or just instantiate ModeScoreMetric
    metric = ModeScoreMetric(tokenizer, device)
    return metric.recursive_token_ids

def run_circuit_discovery_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-v0.1")
    n_pairs = params.get("n_pairs", 10)
    seed = int(params.get("seed", 42))
    
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name, device=device)
    model.eval()
    
    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    (run_dir / "prompt_bank_version.json").write_text(json.dumps({"version": bank_version}, indent=2) + "\n")
    metric = ModeScoreMetric(tokenizer, device)
    
    # Get pairs
    pairs = loader.get_balanced_pairs(n_pairs=n_pairs, seed=seed)
    print(f"Testing on {len(pairs)} pairs")
    
    results = []
    
    # We will patch:
    # 1. Attn Output (Layer L) -> Identify which Layer's Attention matters
    # 2. MLP Output (Layer L) -> Identify if MLP matters
    # 3. Resid Pre (Layer L) -> Cumulative effect
    
    num_layers = model.config.num_hidden_layers
    
    for i, (rec_text, base_text) in enumerate(tqdm(pairs, desc="Pairs")):
        # 1. Run Clean Recursive (Source)
        # Capture ALL activations
        inputs_r = tokenizer(rec_text, return_tensors="pt", add_special_tokens=False).to(device)
        
        # We need a hook to capture everything.
        # Dictionary: { "L0_attn": tensor, "L0_mlp": tensor ... }
        activations = {}
        
        def make_capture_hook(name):
            def hook(module, inp, out):
                # out is (Batch, Seq, Hidden) or Tuple
                if isinstance(out, tuple):
                    out = out[0]
                activations[name] = out.detach().clone()
            return hook
            
        handles = []
        for l in range(num_layers):
            handles.append(model.model.layers[l].self_attn.register_forward_hook(make_capture_hook(f"L{l}_attn")))
            handles.append(model.model.layers[l].mlp.register_forward_hook(make_capture_hook(f"L{l}_mlp")))
            
        with torch.no_grad():
            out_r = model(**inputs_r)
        
        # Clear hooks
        for h in handles: h.remove()
        handles = []
        
        # 2. Run Baseline with Patching (Sweep)
        inputs_b = tokenizer(base_text, return_tensors="pt", add_special_tokens=False).to(device)
        
        # Baseline Logits (for Metric)
        with torch.no_grad():
            out_b = model(**inputs_b)
            
        base_score = metric.compute_score(out_b.logits, baseline_logits=out_b.logits)
        
        # Sweep Layers
        for l in range(num_layers):
            for comp_type in ["attn", "mlp"]:
                key = f"L{l}_{comp_type}"
                source_act = activations[key]
                
                # Patching Hook
                def make_patch_hook(source):
                    def hook(module, inp, out):
                        # out: (B, T_base, D) or Tuple
                        is_tuple = isinstance(out, tuple)
                        if is_tuple:
                            out_tensor = out[0]
                        else:
                            out_tensor = out
                            
                        B, T_b, D = out_tensor.shape
                        T_r = source.shape[1]
                        W = min(16, T_b, T_r)
                        
                        out_p = out_tensor.clone()
                        out_p[:, -W:, :] = source[:, -W:, :].to(out_tensor.device)
                        
                        if is_tuple:
                            return (out_p,) + out[1:]
                        return out_p
                    return hook
                
                # Register Patch Hook
                if comp_type == "attn":
                    target_module = model.model.layers[l].self_attn
                else:
                    target_module = model.model.layers[l].mlp
                    
                h = target_module.register_forward_hook(make_patch_hook(source_act))
                
                try:
                    with torch.no_grad():
                        out_p = model(**inputs_b)
                        
                    # Measure Effect
                    # Metric: Mode Score (Shift towards Recursive)
                    score_p = metric.compute_score(out_p.logits, baseline_logits=out_b.logits)
                    delta = score_p - base_score
                    
                    results.append({
                        "pair_idx": i,
                        "layer": l,
                        "component": comp_type,
                        "base_score": base_score,
                        "patched_score": score_p,
                        "delta": delta
                    })
                finally:
                    h.remove()
                    
    # Save Results
    df = pd.DataFrame(results)
    out_csv = run_dir / "circuit_discovery.csv"
    df.to_csv(out_csv, index=False)
    
    # Heatmap Summary
    # Mean delta per (Layer, Component)
    pivot = df.pivot_table(index="layer", columns="component", values="delta", aggfunc="mean")
    print("\nRESULTS (Mean Delta Logit Score):")
    print(pivot)
    
    # Save pivot
    pivot.to_csv(run_dir / "circuit_heatmap.csv")
    
    # Find Max
    max_row = df.loc[df["delta"].idxmax()]
    
    summary = {
        "experiment": "circuit_discovery",
        "max_delta": float(max_row["delta"]),
        "best_component": f"L{max_row['layer']}_{max_row['component']}",
        "heatmap_path": str(run_dir / "circuit_heatmap.csv")
    }
    
    return ExperimentResult(summary=summary)

