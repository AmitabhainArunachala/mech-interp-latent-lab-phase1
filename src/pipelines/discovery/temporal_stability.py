"""
Temporal Stability Pipeline (Gold Standard Pipeline 6)

Tests if R_V contraction persists across autoregressive generation (multi-token dynamics).
Implements the "Dynamic R_V" contract from docs/MEASUREMENT_CONTRACT.md.

Based on experiment_multi_token_generation.py, now hardened for the registry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import entropy as scipy_entropy

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.core.hooks import capture_v_projection, capture_attention_patterns
from src.metrics.rv import participation_ratio
from src.pipelines.registry import ExperimentResult


def compute_h31_entropy(attn_weights: torch.Tensor, head_idx: int = 31) -> float:
    """Compute H31 entropy from attention weights."""
    if attn_weights is None:
        return float('nan')
    
    # attn_weights shape: (batch, num_heads, seq_len, seq_len)
    # We take batch 0
    head_attn = attn_weights[0, head_idx, :, :].cpu().numpy()
    entropies = []
    for i in range(head_attn.shape[0]):
        row = head_attn[i] + 1e-10
        row = row / row.sum()
        entropies.append(scipy_entropy(row))
    return float(np.mean(entropies))


def compute_rv_at_step(
    model,
    input_ids: torch.Tensor,
    past_key_values: Optional[Tuple],
    early_layer: int,
    late_layer: int,
    window: int,
) -> Tuple[float, Optional[float]]:
    """
    Compute R_V for current sequence state.
    
    Note: We must perform a full forward pass (or at least capture V) to get R_V.
    Optimizing this with past_key_values is tricky because we need the *full* V-matrix
    history to compute SVD over the window, not just the last token's V-vector.
    
    For correctness/simplicity in this pipeline, we run a full forward pass on the
    windowed input_ids if past_key_values are not sufficient for V-retrieval.
    HOWEVER, `capture_v_projection` only captures the *current* forward pass.
    If we use past_key_values, the model only computes the *new* token's V.
    
    CRITICAL FIX: To measure R_V over a window W=16, we need the V-vectors for the last 16 tokens.
    If we use KV-caching, we don't recompute V for old tokens.
    Therefore, we must run a full forward pass on the last W tokens to get their V-vectors.
    """
    
    # CRITICAL FIX: We must run the FULL sequence to preserve attention context.
    # If we only run the window, the tokens lose their history and attention patterns change.
    # We slice the V-tensor *after* the forward pass.
    
    with torch.no_grad():
        with capture_v_projection(model, early_layer) as v_early_storage:
            with capture_v_projection(model, late_layer) as v_late_storage:
                with capture_attention_patterns(model, late_layer) as attn_storage:
                    # Full forward pass on the ENTIRE sequence
                    # Expensive but necessary for correct attention context
                    model(input_ids, use_cache=False, output_attentions=True)
        
        v_early = v_early_storage.get("v")
        v_late = v_late_storage.get("v")
        attn_weights = attn_storage.get("attn_weights")
        
        if v_early is None or v_late is None:
            return float('nan'), None
            
        # Normalize dimensions (batch, seq, dim) -> (seq, dim)
        if v_early.dim() == 3:
            v_early = v_early[0]
        if v_late.dim() == 3:
            v_late = v_late[0]
            
        # CRITICAL: Slice the last W tokens for measurement
        pr_early = participation_ratio(v_early, window_size=window)
        pr_late = participation_ratio(v_late, window_size=window)
        
        if pr_early == 0 or np.isnan(pr_early) or np.isnan(pr_late):
            rv = float('nan')
        else:
            rv = float(pr_late / pr_early)
            
        h31_entropy = compute_h31_entropy(attn_weights) if attn_weights is not None else None
        
        return rv, h31_entropy


def generate_with_metrics(
    model,
    tokenizer,
    prompt: str,
    temperature: float,
    max_steps: int,
    early_layer: int,
    late_layer: int,
    window: int,
    device: str,
) -> Dict:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # Step 0 measurement
    rv_0, h31_0 = compute_rv_at_step(model, input_ids, None, early_layer, late_layer, window)
    
    results = {
        "step": [0],
        "rv": [rv_0],
        "h31_entropy": [h31_0 if h31_0 is not None else float('nan')],
        "generated_tokens": [""],
        "cumulative_text": [prompt],
    }
    
    current_ids = input_ids
    # Initial KV cache for generation
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        past_key_values = outputs.past_key_values

    for step in range(1, max_steps + 1):
        with torch.no_grad():
            # 1. Generate next token (using cache)
            outputs = model(
                input_ids=current_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = outputs.logits[:, -1, :]
            
            if temperature == 0.0:
                next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
                
            # Handle shape quirks
            if next_token_id.dim() == 0:
                next_token_id = next_token_id.unsqueeze(0).unsqueeze(0)
            elif next_token_id.dim() == 1:
                next_token_id = next_token_id.unsqueeze(0)
                
            current_ids = torch.cat([current_ids, next_token_id], dim=1)
            past_key_values = outputs.past_key_values
            
            new_token = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            cumulative_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
            
            # 2. Measure metrics (Full pass on window)
            rv, h31 = compute_rv_at_step(
                model, current_ids, None, early_layer, late_layer, window
            )
            
            results["step"].append(step)
            results["rv"].append(rv)
            results["h31_entropy"].append(h31 if h31 is not None else float('nan'))
            results["generated_tokens"].append(new_token)
            results["cumulative_text"].append(cumulative_text)
            
    return results


def run_temporal_stability_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """
    Run temporal stability pipeline.
    """
    model_cfg = cfg.get("model") or {}
    params = cfg.get("params") or {}
    
    seed = int(cfg.get("seed") or 42)
    device = str(model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = str(model_cfg.get("name") or "mistralai/Mistral-7B-v0.1")
    
    early_layer = int(params.get("early_layer") or 5)
    late_layer = int(params.get("late_layer") or 27)
    window = int(params.get("window") or 16)
    max_steps = int(params.get("max_steps") or 20)
    temperatures = params.get("temperatures") or [0.0, 0.7]
    n_prompts = int(params.get("n_prompts") or 10)
    
    # Load model
    set_seed(seed)
    model, tokenizer = load_model(model_name, device=device, attn_implementation="eager")
    
    # Load prompts
    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    (run_dir / "prompt_bank_version.json").write_text(
        json.dumps({"version": bank_version}, indent=2) + "\n"
    )
    
    recursive_prompts = loader.get_by_pillar("dose_response", limit=n_prompts, seed=seed)
    # Baseline fallback strategy
    baseline_prompts = loader.get_by_pillar("baselines", limit=n_prompts, seed=seed)
    if not baseline_prompts:
        baseline_prompts = loader.get_by_type("baseline", limit=n_prompts, seed=seed)
        
    all_results = []
    
    for prompt_type, prompts in [("recursive", recursive_prompts), ("baseline", baseline_prompts)]:
        for temp in temperatures:
            for i, prompt in enumerate(tqdm(prompts, desc=f"{prompt_type} T={temp}")):
                try:
                    metrics = generate_with_metrics(
                        model, tokenizer, prompt, temp, max_steps,
                        early_layer, late_layer, window, device
                    )
                    
                    for step_idx in range(len(metrics["step"])):
                        all_results.append({
                            "prompt_type": prompt_type,
                            "temperature": temp,
                            "prompt_idx": i,
                            "step": metrics["step"][step_idx],
                            "rv": metrics["rv"][step_idx],
                            "h31_entropy": metrics["h31_entropy"][step_idx],
                            "token": metrics["generated_tokens"][step_idx],
                        })
                except Exception as e:
                    print(f"Error on {prompt_type} prompt {i}: {e}")
                    continue

    df = pd.DataFrame(all_results)
    out_csv = run_dir / "temporal_stability.csv"
    df.to_csv(out_csv, index=False)
    
    # Compute summary stats
    summary = {
        "experiment": "temporal_stability",
        "model_name": model_name,
        "params": params,
        "n_recursive": len(recursive_prompts),
        "n_baseline": len(baseline_prompts),
    }
    
    # Pass/Fail check: Does R_V stay low for recursive?
    if not df.empty:
        rec_t0 = df[(df["prompt_type"] == "recursive") & (df["temperature"] == 0.0)]
        mean_rv_rec = rec_t0["rv"].mean()
        # Persistence: Fraction of steps where R_V < 0.8 (Contract threshold)
        persistence = (rec_t0["rv"] < 0.8).mean()
        
        summary["stats"] = {
            "recursive_mean_rv": float(mean_rv_rec),
            "recursive_persistence": float(persistence)
        }
        summary["prompt_bank_version"] = bank_version
        
    return ExperimentResult(summary=summary)

