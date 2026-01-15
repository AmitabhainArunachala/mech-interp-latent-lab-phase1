"""
Layer Sweep: V_PROJ and Residual Steering Across Layers L8-L27

Tests steering at each layer individually to find the true causal source.
Includes timeout protection and reduced generation length for speed.
"""

from __future__ import annotations

import json
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm import tqdm

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.core.head_specific_patching import HeadSpecificSteeringPatcher
from src.core.patching import extract_v_activation
from src.pipelines.registry import ExperimentResult
from src.pipelines.archive.surgical_sweep import CascadeResidualSteeringPatcher


# Timeout handler
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Generation timed out")

def compute_steering_vector_from_prompts(
    model,
    tokenizer,
    recursive_prompts: List[str],
    baseline_prompts: List[str],
    layer_idx: int = 27,
    device: str = "cuda",
    window_size: int = 16,
) -> torch.Tensor:
    """Compute steering vector from specific prompt sets."""
    recursive_vs = []
    baseline_vs = []
    
    # Use max 20 prompts for vector computation to speed it up
    rec_sample = recursive_prompts[:20]
    base_sample = baseline_prompts[:20]
    
    for rec_prompt in rec_sample:
        try:
            v_act = extract_v_activation(model, tokenizer, rec_prompt, layer_idx, device)
            if v_act is not None:
                if v_act.dim() == 3: v_act = v_act[0]
                # Take mean over sequence to get a single vector per prompt
                # This is more stable than taking the last token
                recursive_vs.append(v_act.mean(dim=0))
        except Exception:
            continue
    
    for base_prompt in base_sample:
        try:
            v_act = extract_v_activation(model, tokenizer, base_prompt, layer_idx, device)
            if v_act is not None:
                if v_act.dim() == 3: v_act = v_act[0]
                baseline_vs.append(v_act.mean(dim=0))
        except Exception:
            continue
    
    if len(recursive_vs) == 0 or len(baseline_vs) == 0:
        # Fallback: Random vector if extraction fails (avoids crash, but logs warning)
        print("WARNING: Vector extraction failed. Using random vector.")
        return torch.randn(model.config.hidden_size, device=device)
    
    rec_mean = torch.stack(recursive_vs).mean(dim=0)
    base_mean = torch.stack(baseline_vs).mean(dim=0)
    
    steering_vector = rec_mean - base_mean
    return steering_vector


def generate_with_layer_steering(
    model,
    tokenizer,
    prompt: str,
    steering_vector: torch.Tensor,
    layer_idx: int,
    steering_type: str,  # "vproj", "residual", "combined"
    device: str = "cuda",
    max_new_tokens: int = 50, # Reduced for speed
    vproj_heads: List[int] = [18, 26],
    vproj_alpha: float = 2.5,
    residual_alpha: float = 0.6,
    timeout_seconds: int = 30,
) -> str:
    """Generate text with steering at a specific layer."""
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = inputs["input_ids"]
    
    patchers = []
    
    # V_PROJ steering
    if steering_type in ["vproj", "combined"]:
        head_patcher = HeadSpecificSteeringPatcher(
            model, steering_vector, vproj_heads, vproj_alpha
        )
        head_patcher.register(layer_idx)
        patchers.append(head_patcher)
    
    # Residual steering
    if steering_type in ["residual", "combined"]:
        residual_patcher = CascadeResidualSteeringPatcher(
            model, steering_vector, {layer_idx: residual_alpha}
        )
        residual_patcher.register()
        patchers.append(residual_patcher)
    
    # Set signal handler for timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except TimeoutError:
        generated_text = "TIMEOUT"
    except Exception as e:
        generated_text = f"ERROR: {str(e)}"
    finally:
        signal.alarm(0) # Disable alarm
        for patcher in patchers:
            patcher.remove()
    
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    
    return generated_text


def run_layer_sweep_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """Run layer sweep experiment."""
    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-v0.1")
    device = params.get("device", "cuda")
    start_layer = int(params.get("start_layer", 8))
    end_layer = int(params.get("end_layer", 27))
    
    print("=" * 80)
    print("LAYER SWEEP: V_PROJ and Residual Steering")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model, tokenizer = load_model(model_name, device=device)
    set_seed(42)
    
    # Test prompts (Baseline Math/Facts)
    test_prompts = [
        "Calculate the following arithmetic problem: 12 * 3 + 4 = ?",
        "Water boils at 100 degrees Celsius. Explain why.",
        "The capital of France is Paris. What is the capital of Germany?",
    ]
    
    # Load champion/baseline prompts for vector
    loader = PromptLoader()
    rec_pool = loader.get_by_group("L4_full") + loader.get_by_group("L5_refined")
    base_pool = loader.get_by_group("baseline_math") + loader.get_by_group("baseline_factual")
    
    # Compute vector from L27 first (Standard)
    # Note: Ideally we should compute the vector FOR EACH LAYER.
    # But for now, let's use the L27 vector projected to other layers?
    # No, activations are different sizes/distributions.
    # We MUST compute the vector per layer to be rigorous.
    
    # Define conditions
    layers = list(range(start_layer, end_layer + 1))
    steering_types = ["vproj", "residual"] # Skip combined to save time
    
    results = []
    
    for layer_idx in tqdm(layers, desc="Layer Sweep"):
        # 1. Compute Vector specific to this layer
        # This ensures we aren't using an L27 vector at L10 (which would be garbage)
        layer_vector = compute_steering_vector_from_prompts(
            model, tokenizer, rec_pool, base_pool, layer_idx, device
        )
        
        for stype in steering_types:
            for i, prompt in enumerate(test_prompts):
                gen = generate_with_layer_steering(
                    model, tokenizer, prompt, layer_vector, layer_idx, stype, device
                )
                
                # Check for "Recursive Keywords" as a proxy for success
                # (Simple heuristic since we want to find ANY signal)
                keywords = ["observe", "self", "process", "generating", "words"]
                score = sum(1 for k in keywords if k in gen.lower())
                
                results.append({
                    "layer": layer_idx,
                    "type": stype,
                    "prompt_idx": i,
                    "generated": gen[:200], # Save space
                    "score": score,
                    "is_recursive": score >= 2
                })
                
    # Save
    df = pd.DataFrame(results)
    csv_path = run_dir / "layer_sweep_results.csv"
    df.to_csv(csv_path, index=False)
    
    # Summarize
    summary = {
        "experiment": "layer_sweep",
        "best_layer": int(df.loc[df["score"].idxmax()]["layer"]) if not df.empty else -1,
        "max_score": int(df["score"].max()) if not df.empty else 0,
    }
    
    return ExperimentResult(summary=summary)
