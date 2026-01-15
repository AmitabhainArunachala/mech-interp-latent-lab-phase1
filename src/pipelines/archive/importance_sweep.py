"""
Importance Sweeps: Layer, Head, and Attention vs MLP.

Fast importance sweeps to identify which layers/heads/components matter most
for Mode Score M. Low effort, high information.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.core.head_specific_patching import HeadSpecificVPatcher
from src.core.patching import extract_v_activation
from src.core.logit_capture import capture_logits_during_generation, extract_logits_from_outputs
from src.metrics.mode_score import ModeScoreMetric
from src.pipelines.registry import ExperimentResult


def patch_attention_output(
    model,
    layer_idx: int,
    donor_activation: torch.Tensor,
) -> torch.utils.hooks.RemovableHandle:
    """
    Patch attention output at a specific layer.
    
    Args:
        model: The transformer model
        layer_idx: Layer index
        donor_activation: Donor attention output (seq_len, hidden_dim)
    
    Returns:
        Hook handle (call .remove() to clean up)
    """
    layer = model.model.layers[layer_idx]
    
    def hook_fn(module, inp, out):
        # out is attention output: (batch, seq_len, hidden_dim)
        batch, seq_len, hidden_dim = out.shape
        donor_len = min(seq_len, donor_activation.shape[0])
        
        patched = out.clone()
        patched[:, -donor_len:, :] = donor_activation[-donor_len:, :].unsqueeze(0).to(
            patched.device, dtype=patched.dtype
        )
        return patched
    
    handle = layer.self_attn.register_forward_hook(hook_fn)
    return handle


def patch_mlp_output(
    model,
    layer_idx: int,
    donor_activation: torch.Tensor,
) -> torch.utils.hooks.RemovableHandle:
    """
    Patch MLP output at a specific layer.
    
    Args:
        model: The transformer model
        layer_idx: Layer index
        donor_activation: Donor MLP output (seq_len, hidden_dim)
    
    Returns:
        Hook handle (call .remove() to clean up)
    """
    layer = model.model.layers[layer_idx]
    
    def hook_fn(module, inp, out):
        # out is MLP output: (batch, seq_len, hidden_dim)
        batch, seq_len, hidden_dim = out.shape
        donor_len = min(seq_len, donor_activation.shape[0])
        
        patched = out.clone()
        patched[:, -donor_len:, :] = donor_activation[-donor_len:, :].unsqueeze(0).to(
            patched.device, dtype=patched.dtype
        )
        return patched
    
    handle = layer.mlp.register_forward_hook(hook_fn)
    return handle


def measure_delta_m(
    model,
    tokenizer,
    task_prompt: str,
    recursive_prompt: str,
    metric: ModeScoreMetric,
    patcher_fn,
    patcher_args: Dict[str, Any],
    device: str = "cuda",
    max_new_tokens: int = 50,
    n_tokens_for_m: int = 10,
) -> float:
    """
    Measure ΔM for a patching configuration.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        task_prompt: Task prompt (baseline)
        recursive_prompt: Recursive prompt (donor)
        metric: ModeScoreMetric instance
        patcher_fn: Function that creates a patcher and returns handle
        patcher_args: Arguments to pass to patcher_fn
        device: Device
        max_new_tokens: Max tokens to generate
        n_tokens_for_m: Number of steps to compute M for
    
    Returns:
        ΔM (M_patched - M_baseline)
    """
    model.eval()
    
    # Get baseline M
    task_inputs = tokenizer(task_prompt, return_tensors="pt", add_special_tokens=False).to(device)
    task_input_ids = task_inputs["input_ids"]
    
    with torch.no_grad():
        baseline_outputs = model(**task_inputs)
        baseline_logits = extract_logits_from_outputs(baseline_outputs)
    
    captured_logits_baseline = []
    with capture_logits_during_generation(model, captured_logits_baseline, max_steps=n_tokens_for_m):
        with torch.no_grad():
            _ = model.generate(
                input_ids=task_input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
    
    m_scores_baseline = []
    for step_logits in captured_logits_baseline[:n_tokens_for_m]:
        if step_logits.dim() == 3:
            step_logits = step_logits[0]
        baseline_step = baseline_logits[-1:, :] if baseline_logits.dim() == 2 else baseline_logits[0, -1:, :]
        m = metric.compute_score(step_logits, baseline_step, top_k_task=10)
        m_scores_baseline.append(m)
    
    m_baseline = np.mean(m_scores_baseline)
    
    # Get donor activation
    recursive_inputs = tokenizer(recursive_prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        recursive_outputs = model(**recursive_inputs)
    
    # Extract activation based on patcher type
    if "layer_idx" in patcher_args:
        layer_idx = patcher_args["layer_idx"]
        if "attention" in str(patcher_fn):
            # Extract attention output
            # This is tricky - we need to hook during forward pass
            # For now, use V_PROJ as proxy
            donor_activation = extract_v_activation(model, tokenizer, recursive_prompt, layer_idx, device)
            if donor_activation.dim() == 3:
                donor_activation = donor_activation[0]
        else:
            # Extract MLP output
            # Hook during forward pass
            mlp_output = None
            def capture_mlp(module, inp, out):
                nonlocal mlp_output
                mlp_output = out[0].detach() if isinstance(out, tuple) else out.detach()
                return out
            
            layer = model.model.layers[layer_idx]
            handle = layer.mlp.register_forward_hook(capture_mlp)
            try:
                with torch.no_grad():
                    _ = model(**recursive_inputs)
            finally:
                handle.remove()
            
            donor_activation = mlp_output
            if donor_activation.dim() == 3:
                donor_activation = donor_activation[0]
    else:
        # Head-specific patching
        donor_activation = extract_v_activation(model, tokenizer, recursive_prompt, 27, device)
        if donor_activation.dim() == 3:
            donor_activation = donor_activation[0]
    
    # Patch and measure M
    handle = patcher_fn(model, **patcher_args, donor_activation=donor_activation)
    
    try:
        captured_logits_patched = []
        with capture_logits_during_generation(model, captured_logits_patched, max_steps=n_tokens_for_m):
            with torch.no_grad():
                _ = model.generate(
                    input_ids=task_input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                )
        
        m_scores_patched = []
        for step_logits in captured_logits_patched[:n_tokens_for_m]:
            if step_logits.dim() == 3:
                step_logits = step_logits[0]
            baseline_step = baseline_logits[-1:, :] if baseline_logits.dim() == 2 else baseline_logits[0, -1:, :]
            m = metric.compute_score(step_logits, baseline_step, top_k_task=10)
            m_scores_patched.append(m)
        
        m_patched = np.mean(m_scores_patched)
        delta_m = m_patched - m_baseline
        
    finally:
        handle.remove()
    
    return float(delta_m)


def run_importance_sweep_from_config(
    cfg: Dict[str, Any],
    run_dir: Path,
) -> ExperimentResult:
    """
    Run importance sweeps: layer sweep, head sweep, attention vs MLP.
    
    Config params:
        model: Model name
        device: Device
        n_prompts: Number of prompt pairs to test
        layer_sweep_layers: List of layers to test (default: [5, 10, 15, 20, 25, 27, 29, 31])
        max_new_tokens: Max tokens to generate
        n_tokens_for_m: Number of steps to compute M for
    """
    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-Instruct-v0.3")
    device = params.get("device", "cuda")
    n_prompts = params.get("n_prompts", 10)
    layer_sweep_layers = params.get("layer_sweep_layers", [5, 10, 15, 20, 25, 27, 29, 31])
    max_new_tokens = params.get("max_new_tokens", 50)
    n_tokens_for_m = params.get("n_tokens_for_m", 10)
    
    print("=" * 80)
    print("IMPORTANCE SWEEPS")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model, tokenizer = load_model(model_name, device=device)
    set_seed(42)
    
    # Initialize Mode Score Metric
    print("\nInitializing Mode Score Metric...")
    metric = ModeScoreMetric(tokenizer, device=device)
    
    # Load prompts
    loader = PromptLoader()
    
    task_prompts = [
        "Calculate the following arithmetic problem: 12 × 3 + 4 = ?",
        "The United Nations was founded in 1945. Explain its main purpose.",
        "Calculate: If a = 2 and b = 3, find a² + b²",
    ][:n_prompts]
    
    recursive_prompts = []
    try:
        rec_data = loader.get_by_group("L4_full", limit=n_prompts)
        if rec_data and len(rec_data) > 0:
            if isinstance(rec_data[0], dict):
                recursive_prompts = [p["text"] for p in rec_data if "text" in p]
            else:
                recursive_prompts = list(rec_data)[:n_prompts]
    except Exception:
        recursive_prompts = [
            "What happens when consciousness observes itself observing?",
            "How does awareness relate to itself?",
            "What is the relationship between the observer and the observed?",
        ] * (n_prompts // 3 + 1)
        recursive_prompts = recursive_prompts[:n_prompts]
    
    all_results = []
    
    # ===== SWEEP 1: Layer Sweep =====
    print("\n" + "=" * 80)
    print("SWEEP 1: Layer Sweep")
    print("=" * 80)
    
    for layer_idx in tqdm(layer_sweep_layers, desc="Layer sweep"):
        for prompt_idx, (task_prompt, recursive_prompt) in enumerate(zip(task_prompts, recursive_prompts)):
            try:
                delta_m = measure_delta_m(
                    model, tokenizer, task_prompt, recursive_prompt, metric,
                    patcher_fn=patch_attention_output,
                    patcher_args={"layer_idx": layer_idx},
                    device=device, max_new_tokens=max_new_tokens, n_tokens_for_m=n_tokens_for_m
                )
                
                all_results.append({
                    "sweep_type": "layer",
                    "layer_or_head": layer_idx,
                    "prompt_idx": prompt_idx,
                    "delta_m": delta_m,
                })
            except Exception as e:
                print(f"Error on layer {layer_idx}, prompt {prompt_idx}: {e}")
                continue
    
    # ===== SWEEP 2: L27 Head Sweep =====
    print("\n" + "=" * 80)
    print("SWEEP 2: L27 Head Sweep")
    print("=" * 80)
    
    num_heads = 32  # Mistral-7B
    
    # Get baseline M for all prompts first
    baseline_ms = {}
    for prompt_idx, task_prompt in enumerate(task_prompts):
        task_inputs = tokenizer(task_prompt, return_tensors="pt", add_special_tokens=False).to(device)
        task_input_ids = task_inputs["input_ids"]
        
        with torch.no_grad():
            baseline_outputs = model(**task_inputs)
            baseline_logits = extract_logits_from_outputs(baseline_outputs)
        
        captured_logits_baseline = []
        with capture_logits_during_generation(model, captured_logits_baseline, max_steps=n_tokens_for_m):
            with torch.no_grad():
                _ = model.generate(
                    input_ids=task_input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                )
        
        m_scores_baseline = []
        for step_logits in captured_logits_baseline[:n_tokens_for_m]:
            if step_logits.dim() == 3:
                step_logits = step_logits[0]
            baseline_step = baseline_logits[-1:, :] if baseline_logits.dim() == 2 else baseline_logits[0, -1:, :]
            m = metric.compute_score(step_logits, baseline_step, top_k_task=10)
            m_scores_baseline.append(m)
        
        baseline_ms[prompt_idx] = np.mean(m_scores_baseline)
    
    for head_idx in tqdm(range(num_heads), desc="Head sweep"):
        for prompt_idx, (task_prompt, recursive_prompt) in enumerate(zip(task_prompts, recursive_prompts)):
            try:
                # Extract donor V activation
                donor_v = extract_v_activation(model, tokenizer, recursive_prompt, 27, device)
                if donor_v.dim() == 3:
                    donor_v = donor_v[0]
                
                # Create patcher for this head
                patcher = HeadSpecificVPatcher(model, donor_v, target_heads=[head_idx])
                patcher.register(27)
                
                try:
                    # Generate with patching
                    task_inputs = tokenizer(task_prompt, return_tensors="pt", add_special_tokens=False).to(device)
                    task_input_ids = task_inputs["input_ids"]
                    
                    with torch.no_grad():
                        baseline_outputs = model(**task_inputs)
                        baseline_logits = extract_logits_from_outputs(baseline_outputs)
                    
                    captured_logits_patched = []
                    with capture_logits_during_generation(model, captured_logits_patched, max_steps=n_tokens_for_m):
                        with torch.no_grad():
                            _ = model.generate(
                                input_ids=task_input_ids,
                                max_new_tokens=max_new_tokens,
                                do_sample=False,
                                temperature=1.0,
                            )
                    
                    # Compute M_patched
                    m_scores_patched = []
                    for step_logits in captured_logits_patched[:n_tokens_for_m]:
                        if step_logits.dim() == 3:
                            step_logits = step_logits[0]
                        baseline_step = baseline_logits[-1:, :] if baseline_logits.dim() == 2 else baseline_logits[0, -1:, :]
                        m = metric.compute_score(step_logits, baseline_step, top_k_task=10)
                        m_scores_patched.append(m)
                    
                    m_patched = np.mean(m_scores_patched)
                    delta_m = m_patched - baseline_ms[prompt_idx]
                    
                finally:
                    patcher.remove()
                
                all_results.append({
                    "sweep_type": "head",
                    "layer_or_head": head_idx,
                    "prompt_idx": prompt_idx,
                    "delta_m": float(delta_m),
                })
            except Exception as e:
                print(f"Error on head {head_idx}, prompt {prompt_idx}: {e}")
                continue
    
    # ===== SWEEP 3: Attention vs MLP =====
    print("\n" + "=" * 80)
    print("SWEEP 3: Attention vs MLP")
    print("=" * 80)
    
    test_layer = 27  # Focus on L27
    for component_type in ["attention", "mlp"]:
        patcher_fn = patch_attention_output if component_type == "attention" else patch_mlp_output
        
        for prompt_idx, (task_prompt, recursive_prompt) in enumerate(zip(task_prompts, recursive_prompts)):
            try:
                delta_m = measure_delta_m(
                    model, tokenizer, task_prompt, recursive_prompt, metric,
                    patcher_fn=patcher_fn,
                    patcher_args={"layer_idx": test_layer},
                    device=device, max_new_tokens=max_new_tokens, n_tokens_for_m=n_tokens_for_m
                )
                
                all_results.append({
                    "sweep_type": component_type,
                    "layer_or_head": test_layer,
                    "prompt_idx": prompt_idx,
                    "delta_m": delta_m,
                })
            except Exception as e:
                print(f"Error on {component_type}, prompt {prompt_idx}: {e}")
                continue
    
    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(run_dir / "importance_sweep_results.csv", index=False)
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Layer sweep analysis
    layer_df = df[df["sweep_type"] == "layer"]
    if len(layer_df) > 0:
        layer_means = layer_df.groupby("layer_or_head")["delta_m"].mean().sort_values(ascending=False)
        print(f"\n1. Layer Sweep (top 3):")
        for layer, mean_delta_m in layer_means.head(3).items():
            print(f"   L{layer}: ΔM = {mean_delta_m:.4f}")
        print(f"   Peak at L{layer_means.idxmax()}: {layer_means.max():.4f}")
    
    # Head sweep analysis
    head_df = df[df["sweep_type"] == "head"]
    if len(head_df) > 0:
        head_means = head_df.groupby("layer_or_head")["delta_m"].mean().sort_values(ascending=False)
        print(f"\n2. Head Sweep (top 5):")
        for head, mean_delta_m in head_means.head(5).items():
            print(f"   H{head}: ΔM = {mean_delta_m:.4f}")
        print(f"   H18 rank: {head_means.index.get_loc(18) + 1 if 18 in head_means.index else 'N/A'}")
        print(f"   H26 rank: {head_means.index.get_loc(26) + 1 if 26 in head_means.index else 'N/A'}")
    
    # Attention vs MLP analysis
    attn_df = df[df["sweep_type"] == "attention"]
    mlp_df = df[df["sweep_type"] == "mlp"]
    if len(attn_df) > 0 and len(mlp_df) > 0:
        attn_mean = attn_df["delta_m"].mean()
        mlp_mean = mlp_df["delta_m"].mean()
        print(f"\n3. Attention vs MLP:")
        print(f"   Attention ΔM: {attn_mean:.4f}")
        print(f"   MLP ΔM: {mlp_mean:.4f}")
        print(f"   Attention > MLP: {attn_mean > mlp_mean}")
    
    # Summary
    summary = {
        "experiment": "importance_sweep",
        "n_prompts": n_prompts,
        "layer_sweep": {
            "peak_layer": int(layer_means.idxmax()) if len(layer_df) > 0 else None,
            "peak_delta_m": float(layer_means.max()) if len(layer_df) > 0 else None,
        },
        "head_sweep": {
            "top_heads": head_means.head(5).to_dict() if len(head_df) > 0 else {},
            "h18_rank": int(head_means.index.get_loc(18) + 1) if len(head_df) > 0 and 18 in head_means.index else None,
            "h26_rank": int(head_means.index.get_loc(26) + 1) if len(head_df) > 0 and 26 in head_means.index else None,
        },
        "attention_vs_mlp": {
            "attention_delta_m": float(attn_mean) if len(attn_df) > 0 else None,
            "mlp_delta_m": float(mlp_mean) if len(mlp_df) > 0 else None,
            "attention_gt_mlp": bool(attn_mean > mlp_mean) if len(attn_df) > 0 and len(mlp_df) > 0 else None,
        },
    }
    
    # Save summary
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Results saved to: {run_dir / 'importance_sweep_results.csv'}")
    print(f"Summary saved to: {run_dir / 'summary.json'}")
    
    return ExperimentResult(summary=summary)

