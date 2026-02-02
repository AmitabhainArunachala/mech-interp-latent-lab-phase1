#!/usr/bin/env python3
"""
EXPERIMENT 1: Multi-Token Generation Dynamics
=============================================

Tests if R_V contraction persists across autoregressive generation.

This directly addresses reviewer question: "Does contraction persist across
multi-token generation, or only at the input step?"

Measures:
- R_V at each generation step (0-20 tokens)
- H31 entropy at each step (optional)
- State persistence metrics (threshold crossings)

Conditions:
- Recursive vs baseline prompts
- Fixed decoding (temperature=0) vs sampling (temperature=0.7)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import entropy as scipy_entropy

sys.path.insert(0, str(Path(__file__).parent))

from src.core.models import load_model, set_seed
from src.core.hooks import capture_v_projection, capture_attention_patterns
from src.metrics.rv import participation_ratio
from prompts.loader import PromptLoader

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
EARLY_LAYER = 5
LATE_LAYER = 27
WINDOW = 16
SEED = 42
MAX_GENERATION_STEPS = 20
TEMPERATURES = [0.0, 0.7]  # Fixed decoding and sampling
N_PROMPTS = 10  # Per type

# Thresholds for state persistence
RV_CONTRACTION_THRESHOLD = 0.8  # R_V < 0.8 = contracted state
H31_ENTROPY_THRESHOLD = 0.5  # Entropy < 0.5 = focused state

# =============================================================================
# UTILITIES
# =============================================================================

def compute_h31_entropy(attn_weights: torch.Tensor, head_idx: int = 31) -> float:
    """Compute H31 entropy from attention weights."""
    if attn_weights is None:
        return float('nan')
    
    head_attn = attn_weights[0, head_idx, :, :].cpu().numpy()
    entropies = []
    for i in range(head_attn.shape[0]):
        row = head_attn[i] + 1e-10
        row = row / row.sum()
        entropies.append(scipy_entropy(row))
    return float(np.mean(entropies))


def compute_rv_at_step(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    past_key_values: Optional[Tuple] = None,
    device: str = "cuda",
) -> Tuple[float, Optional[float]]:
    """
    Compute R_V for current sequence state.
    
    Returns:
        (rv_value, h31_entropy)
    """
    with torch.no_grad():
        # Capture V-projections
        with capture_v_projection(model, EARLY_LAYER) as v_early_storage:
            with capture_v_projection(model, LATE_LAYER) as v_late_storage:
                with capture_attention_patterns(model, LATE_LAYER) as attn_storage:
                    outputs = model(
                        input_ids=input_ids,
                        past_key_values=past_key_values,
                        output_attentions=True,
                        use_cache=True,
                    )
        
        v_early = v_early_storage.get("v")
        v_late = v_late_storage.get("v")
        attn_weights = attn_storage.get("attn_weights")
        
        # Compute R_V
        # Handle tensor shape: V-projection might be (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
        if v_early is None or v_late is None:
            rv = float('nan')
        else:
            # Normalize to 2D: participation_ratio expects (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
            # and handles 3D by taking [0]. So convert both to 2D to avoid dimension mismatches.
            if v_early.dim() == 3:
                v_early = v_early[0]  # (batch, seq, hidden) -> (seq, hidden)
            if v_late.dim() == 3:
                v_late = v_late[0]  # (batch, seq, hidden) -> (seq, hidden)
            
            # Now both should be 2D: (seq_len, hidden_dim)
            # participation_ratio will handle 2D directly
            try:
                pr_early = participation_ratio(v_early, window_size=WINDOW)
                pr_late = participation_ratio(v_late, window_size=WINDOW)
            except Exception as e:
                # If still fails, return NaN
                rv = float('nan')
                h31_entropy = compute_h31_entropy(attn_weights) if attn_weights is not None else None
                return rv, h31_entropy
            
            if pr_early == 0 or np.isnan(pr_early) or np.isnan(pr_late):
                rv = float('nan')
            else:
                rv = float(pr_late / pr_early)
        
        # Compute H31 entropy
        h31_entropy = compute_h31_entropy(attn_weights) if attn_weights is not None else None
        
        return rv, h31_entropy


def generate_with_metrics(
    model,
    tokenizer,
    prompt: str,
    temperature: float = 0.0,
    max_steps: int = 20,
    device: str = "cuda",
) -> Dict:
    """
    Generate tokens while measuring R_V and H31 entropy at each step.
    
    Returns:
        Dictionary with step-by-step metrics
    """
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # Initial forward pass (prompt encoding)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=True)
        past_key_values = outputs.past_key_values
    
    # Measure at step 0 (after prompt encoding)
    rv_0, h31_0 = compute_rv_at_step(
        model, tokenizer, input_ids, past_key_values, device
    )
    
    results = {
        "step": [0],
        "rv": [rv_0],
        "h31_entropy": [h31_0 if h31_0 is not None else float('nan')],
        "generated_tokens": [""],
        "cumulative_text": [prompt],
    }
    
    # Generate tokens step by step
    current_ids = input_ids
    for step in range(1, max_steps + 1):
        with torch.no_grad():
            # Get next token logits
            outputs = model(
                input_ids=current_ids[:, -1:],  # Only last token
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            
            # Sample next token
            if temperature == 0.0:
                next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            # Ensure next_token_id has correct shape: (1, 1) for batch=1, seq_len=1
            if next_token_id.dim() == 0:
                next_token_id = next_token_id.unsqueeze(0).unsqueeze(0)  # (1, 1)
            elif next_token_id.dim() == 1:
                next_token_id = next_token_id.unsqueeze(0)  # (1, seq_len) -> should be (1, 1)
            # current_ids is (1, seq_len), next_token_id should be (1, 1)
            current_ids = torch.cat([current_ids, next_token_id], dim=1)
            past_key_values = outputs.past_key_values
            
            # Decode new token
            new_token = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            cumulative_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
        
        # Measure metrics at this step
        # Use full sequence (current_ids) but with past_key_values for efficiency
        # The model will use past_key_values for the prefix and only compute new tokens
        rv, h31_entropy = compute_rv_at_step(
            model, tokenizer, current_ids, None, device  # Don't use past_key_values for R_V computation - need full forward pass
        )
        
        results["step"].append(step)
        results["rv"].append(rv)
        results["h31_entropy"].append(h31_entropy if h31_entropy is not None else float('nan'))
        results["generated_tokens"].append(new_token)
        results["cumulative_text"].append(cumulative_text)
    
    return results


def compute_state_persistence(metrics: Dict, threshold: float, metric_name: str = "rv") -> Dict:
    """
    Compute state persistence metrics.
    
    Returns:
        - persistence_ratio: Fraction of steps below threshold
        - crossings: Number of times trajectory crosses threshold
        - mean_below_threshold: Mean value when below threshold
        - mean_above_threshold: Mean value when above threshold
    """
    values = metrics[metric_name]
    below_threshold = [v < threshold for v in values if not np.isnan(v)]
    
    if not below_threshold:
        return {
            "persistence_ratio": float('nan'),
            "crossings": 0,
            "mean_below_threshold": float('nan'),
            "mean_above_threshold": float('nan'),
        }
    
    persistence_ratio = sum(below_threshold) / len(below_threshold)
    
    # Count threshold crossings
    crossings = 0
    was_below = below_threshold[0] if below_threshold else None
    for is_below in below_threshold[1:]:
        if was_below is not None and is_below != was_below:
            crossings += 1
        was_below = is_below
    
    # Mean values
    valid_values = [v for v in values if not np.isnan(v)]
    below_values = [v for i, v in enumerate(valid_values) if i < len(below_threshold) and below_threshold[i]]
    above_values = [v for i, v in enumerate(valid_values) if i < len(below_threshold) and not below_threshold[i]]
    
    return {
        "persistence_ratio": persistence_ratio,
        "crossings": crossings,
        "mean_below_threshold": float(np.mean(below_values)) if below_values else float('nan'),
        "mean_above_threshold": float(np.mean(above_values)) if above_values else float('nan'),
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 80)
    print("EXPERIMENT 1: MULTI-TOKEN GENERATION DYNAMICS")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Max steps: {MAX_GENERATION_STEPS}")
    print(f"Temperatures: {TEMPERATURES}")
    print("=" * 80)
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/path_b_validation/runs/{timestamp}_multi_token_generation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    set_seed(SEED)
    model, tokenizer = load_model(MODEL_NAME, device=DEVICE, attn_implementation="eager")
    model.eval()
    
    # Load prompts
    loader = PromptLoader()
    recursive_prompts = loader.get_by_pillar("dose_response", limit=N_PROMPTS, seed=SEED)
    # Try multiple baseline pillar names
    baseline_prompts = []
    for pillar_name in ["baseline", "baselines", "control"]:
        baseline_prompts = loader.get_by_pillar(pillar_name, limit=N_PROMPTS, seed=SEED)
        if len(baseline_prompts) > 0:
            break
    # If still empty, use get_by_type
    if len(baseline_prompts) == 0:
        baseline_prompts = loader.get_by_type("baseline", limit=N_PROMPTS, seed=SEED)
    
    print(f"\nLoaded {len(recursive_prompts)} recursive and {len(baseline_prompts)} baseline prompts")
    
    all_results = []
    
    # Test each prompt type × temperature combination
    for prompt_type, prompts in [("recursive", recursive_prompts), ("baseline", baseline_prompts)]:
        for temp in TEMPERATURES:
            print(f"\n{'='*80}")
            print(f"Testing {prompt_type} prompts at temperature={temp}")
            print(f"{'='*80}")
            
            for prompt_idx, prompt in enumerate(tqdm(prompts, desc=f"{prompt_type} T={temp}")):
                try:
                    metrics = generate_with_metrics(
                        model, tokenizer, prompt, temperature=temp,
                        max_steps=MAX_GENERATION_STEPS, device=DEVICE
                    )
                    
                    # Check if we got valid results
                    if len(metrics["step"]) == 0:
                        print(f"  ⚠️  No steps collected for {prompt_type} prompt {prompt_idx}")
                        continue
                    
                    # Compute state persistence
                    rv_persistence = compute_state_persistence(
                        metrics, RV_CONTRACTION_THRESHOLD, "rv"
                    )
                    h31_persistence = compute_state_persistence(
                        metrics, H31_ENTROPY_THRESHOLD, "h31_entropy"
                    )
                    
                    # Store results
                    for step_idx in range(len(metrics["step"])):
                        all_results.append({
                            "prompt_type": prompt_type,
                            "temperature": temp,
                            "prompt_id": prompt_idx,
                            "prompt": prompt[:100],  # Truncate for CSV
                            "step": metrics["step"][step_idx],
                            "rv": metrics["rv"][step_idx],
                            "h31_entropy": metrics["h31_entropy"][step_idx],
                            "generated_token": metrics["generated_tokens"][step_idx],
                            "is_contracted": metrics["rv"][step_idx] < RV_CONTRACTION_THRESHOLD if not np.isnan(metrics["rv"][step_idx]) else False,
                            "is_focused": metrics["h31_entropy"][step_idx] < H31_ENTROPY_THRESHOLD if not np.isnan(metrics["h31_entropy"][step_idx]) else False,
                        })
                    
                    # Save individual prompt trajectory
                    prompt_df = pd.DataFrame(metrics)
                    prompt_df.to_csv(
                        output_dir / f"trajectory_{prompt_type}_T{temp}_P{prompt_idx}.csv",
                        index=False
                    )
                    
                except Exception as e:
                    import traceback
                    error_msg = f"Error processing {prompt_type} prompt {prompt_idx}: {str(e)}\n{traceback.format_exc()}"
                    print(f"\n{error_msg}")
                    # Save first error for debugging
                    if prompt_idx == 0 and len(all_results) == 0:
                        with open(output_dir / "first_error.txt", "w") as f:
                            f.write(error_msg)
                    continue
                    
                    # Store persistence metrics (once per prompt)
                    persistence_results = {
                        "prompt_type": prompt_type,
                        "temperature": temp,
                        "prompt_id": prompt_idx,
                        "prompt": prompt[:100],
                        "rv_persistence_ratio": rv_persistence["persistence_ratio"],
                        "rv_crossings": rv_persistence["crossings"],
                        "rv_mean_below_threshold": rv_persistence["mean_below_threshold"],
                        "rv_mean_above_threshold": rv_persistence["mean_above_threshold"],
                        "h31_persistence_ratio": h31_persistence["persistence_ratio"],
                        "h31_crossings": h31_persistence["crossings"],
                        "h31_mean_below_threshold": h31_persistence["mean_below_threshold"],
                        "h31_mean_above_threshold": h31_persistence["mean_above_threshold"],
                    }
                    
                    # Save individual prompt trajectory
                    prompt_df = pd.DataFrame(metrics)
                    prompt_df.to_csv(
                        output_dir / f"trajectory_{prompt_type}_T{temp}_P{prompt_idx}.csv",
                        index=False
                    )
                    
                except Exception as e:
                    import traceback
                    print(f"\nError processing {prompt_type} prompt {prompt_idx}: {e}")
                    if prompt_idx == 0:
                        print(traceback.format_exc())
                    continue
    
    # Save all results
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / "all_trajectories.csv", index=False)
    
    # Aggregate persistence metrics (only if we have data)
    if len(df) > 0:
        persistence_df = df.groupby(["prompt_type", "temperature", "prompt_id"]).agg({
            "is_contracted": lambda x: x.sum() / len(x) if len(x) > 0 else 0,
            "is_focused": lambda x: x.sum() / len(x) if len(x) > 0 else 0,
            "rv": ["mean", "std", "min", "max"],
            "h31_entropy": ["mean", "std", "min", "max"],
        }).reset_index()
        
        persistence_df.to_csv(output_dir / "persistence_summary.csv", index=False)
    else:
        print("\n⚠️  WARNING: No valid results collected. Check errors above.")
        persistence_df = pd.DataFrame()
    
    # Summary statistics (only if we have data)
    if len(df) > 0:
        recursive_t0_df = df[(df["prompt_type"] == "recursive") & (df["temperature"] == 0.0)]
        baseline_t0_df = df[(df["prompt_type"] == "baseline") & (df["temperature"] == 0.0)]
        
        summary = {
            "experiment": "multi_token_generation",
            "model": MODEL_NAME,
            "n_recursive": len(recursive_prompts),
            "n_baseline": len(baseline_prompts),
            "max_steps": MAX_GENERATION_STEPS,
            "temperatures": TEMPERATURES,
            "thresholds": {
                "rv_contraction": RV_CONTRACTION_THRESHOLD,
                "h31_entropy": H31_ENTROPY_THRESHOLD,
            },
            "results": {
                "recursive_t0": {
                    "mean_rv": float(recursive_t0_df["rv"].mean()) if len(recursive_t0_df) > 0 else float('nan'),
                    "std_rv": float(recursive_t0_df["rv"].std()) if len(recursive_t0_df) > 0 else float('nan'),
                    "mean_h31": float(recursive_t0_df["h31_entropy"].mean()) if len(recursive_t0_df) > 0 else float('nan'),
                    "persistence_ratio": float(recursive_t0_df["is_contracted"].mean()) if len(recursive_t0_df) > 0 else float('nan'),
                },
                "baseline_t0": {
                    "mean_rv": float(baseline_t0_df["rv"].mean()) if len(baseline_t0_df) > 0 else float('nan'),
                    "std_rv": float(baseline_t0_df["rv"].std()) if len(baseline_t0_df) > 0 else float('nan'),
                    "mean_h31": float(baseline_t0_df["h31_entropy"].mean()) if len(baseline_t0_df) > 0 else float('nan'),
                    "persistence_ratio": float(baseline_t0_df["is_contracted"].mean()) if len(baseline_t0_df) > 0 else float('nan'),
                },
            },
        }
    else:
        summary = {
            "experiment": "multi_token_generation",
            "model": MODEL_NAME,
            "error": "No valid results collected",
        }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    if "results" in summary:
        print(f"\nRecursive (T=0.0):")
        rec_t0 = summary['results']['recursive_t0']
        if not np.isnan(rec_t0['mean_rv']):
            print(f"  Mean R_V: {rec_t0['mean_rv']:.4f} ± {rec_t0['std_rv']:.4f}")
            print(f"  Persistence ratio: {rec_t0['persistence_ratio']:.2%}")
        else:
            print(f"  No valid data")
        print(f"\nBaseline (T=0.0):")
        base_t0 = summary['results']['baseline_t0']
        if not np.isnan(base_t0['mean_rv']):
            print(f"  Mean R_V: {base_t0['mean_rv']:.4f} ± {base_t0['std_rv']:.4f}")
            print(f"  Persistence ratio: {base_t0['persistence_ratio']:.2%}")
        else:
            print(f"  No valid data")
    else:
        print(f"\n⚠️  No results collected - check errors above")
    print(f"\n✅ Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

