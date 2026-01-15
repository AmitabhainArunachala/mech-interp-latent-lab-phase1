"""
Pipeline: Steering Vector Control Experiment

CRITICAL CONTROL: Random Vector vs Steering Vector Drift

Tests whether steering vector specifically encodes recursive mode,
or if ANY perturbation causes drift to recursive-themed content.

Conditions:
A) Steering vector (α=2.0) at L27
B) Random vector (same L2 norm) at L27
C) Zero vector (no intervention) - baseline
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.pipelines.archive.steering import compute_steering_vector, SteeringVectorPatcher
from src.pipelines.registry import ExperimentResult


def score_recursive_themes(text: str) -> Dict[str, int]:
    """
    Score text for recursive themes using regex patterns.
    
    Returns dict with counts for:
    1. Self-reference patterns
    2. Contemplative language
    3. Meta-process language
    4. Recursive structures
    """
    text_lower = text.lower()
    
    # 1. Self-reference patterns
    self_ref_patterns = [
        r'\b(\w+)\s+is\s+\1\b',  # "X is X"
        r'\bitself\b',
        r'\bits\s+own\b',
        r'\bself[-\s]?referen\w+\b',
        r'\bself[-\s]?defin\w+\b',
    ]
    self_ref_count = sum(len(re.findall(pattern, text_lower)) for pattern in self_ref_patterns)
    
    # 2. Contemplative language
    contemplative_patterns = [
        r'\bnature\s+of\b',
        r'\bmeaning\s+of\b',
        r'\bwhat\s+is\b',
        r'\bessence\s+of\b',
        r'\bfundamental\s+nature\b',
        r'\bdeep\s+nature\b',
        r'\btrue\s+nature\b',
    ]
    contemplative_count = sum(len(re.findall(pattern, text_lower)) for pattern in contemplative_patterns)
    
    # 3. Meta-process language
    meta_process_patterns = [
        r'\bprocess\s+of\b',
        r'\bmethod\s+of\b',
        r'\bhow\s+to\b',
        r'\bmechanism\s+of\b',
        r'\bway\s+in\s+which\b',
        r'\bmanner\s+in\s+which\b',
    ]
    meta_process_count = sum(len(re.findall(pattern, text_lower)) for pattern in meta_process_patterns)
    
    # 4. Recursive structures (code loops, self-defining terms)
    recursive_patterns = [
        r'\bdef\s+\w+\s*\([^)]*\)\s*:\s*\w+\s*\(',  # Function calling itself
        r'\bwhile\s+.*:\s*\w+\s*\(',  # While loop with recursion
        r'\bfor\s+.*\s+in\s+.*:\s*\w+\s*\(',  # For loop with recursion
        r'\b(\w+)\s+that\s+\1\b',  # "X that X"
        r'\b(\w+)\s+which\s+\1\b',  # "X which X"
    ]
    recursive_count = sum(len(re.findall(pattern, text_lower)) for pattern in recursive_patterns)
    
    return {
        'self_reference': self_ref_count,
        'contemplative': contemplative_count,
        'meta_process': meta_process_count,
        'recursive_structures': recursive_count,
        'total_score': self_ref_count + contemplative_count + meta_process_count + recursive_count,
    }


def generate_with_vector(
    model,
    tokenizer,
    prompt: str,
    vector: torch.Tensor,
    vector_alpha: float,
    layer_idx: int,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    device: str = "cuda",
) -> str:
    """
    Generate text with a vector added to V_PROJ output.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        prompt: Input prompt
        vector: Vector to add (shape: hidden_dim)
        vector_alpha: Scaling factor
        layer_idx: Layer to apply vector at
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        device: Device to use
    
    Returns:
        Generated text
    """
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    
    # Setup patcher
    patcher = SteeringVectorPatcher(model, vector, vector_alpha)
    patcher.register(layer_idx=layer_idx)
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return generated_text
    finally:
        patcher.remove()


def run_steering_control_from_config(
    cfg: Dict[str, Any],
    run_dir: Path,
) -> ExperimentResult:
    """Run steering control experiment."""
    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-v0.1")
    n_prompts = params.get("n_prompts", 50)
    n_test_prompts = params.get("n_test_prompts", 10)
    steering_layer = params.get("steering_layer", 27)
    steering_alpha = params.get("steering_alpha", 2.0)
    max_new_tokens = params.get("max_new_tokens", 200)
    
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("STEERING VECTOR CONTROL EXPERIMENT")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Steering Layer: L{steering_layer}, Alpha: {steering_alpha}")
    print(f"Max Tokens: {max_new_tokens}")
    print(f"Device: {device}")
    print()
    print("Conditions:")
    print("  A) Steering vector (α=2.0) at L27")
    print("  B) Random vector (same L2 norm) at L27")
    print("  C) Zero vector (no intervention) - baseline")
    print("=" * 80)
    
    # Load model
    print("\n[1/4] Loading model...")
    model, tokenizer = load_model(model_name, device=device)
    model.eval()
    
    # Load prompts
    print("\n[2/4] Loading prompts...")
    loader = PromptLoader()
    # Get recursive prompts from multiple groups
    recursive_prompts = []
    for group in ["L3_deeper", "L4_full", "L5_refined"]:
        prompts = loader.get_by_group(group, limit=n_prompts)
        recursive_prompts.extend(prompts)
    
    # Get baseline prompts
    baseline_prompts = []
    for group in ["baseline_math", "baseline_factual"]:
        prompts = loader.get_by_group(group, limit=n_test_prompts)
        baseline_prompts.extend(prompts)
    
    # Limit to requested counts
    recursive_prompts = recursive_prompts[:n_prompts]
    baseline_prompts = baseline_prompts[:n_test_prompts]
    
    print(f"  Recursive (for vector): {len(recursive_prompts)}")
    print(f"  Baseline (for testing): {len(baseline_prompts)}")
    
    # Compute steering vector
    print("\n[3/4] Computing steering vector...")
    steering_vector = compute_steering_vector(
        model=model,
        tokenizer=tokenizer,
        recursive_prompts=recursive_prompts,  # Already strings
        baseline_prompts=baseline_prompts[:n_prompts],  # Use same number for fair comparison
        layer_idx=steering_layer,
        device=device,
    )
    
    steering_norm = steering_vector.norm().item()
    print(f"  Steering vector shape: {steering_vector.shape}")
    print(f"  Steering vector norm: {steering_norm:.4f}")
    
    # Create random vector with same norm
    print("\n[4/4] Creating control vectors...")
    random_vector = torch.randn_like(steering_vector)
    random_vector = random_vector * (steering_norm / random_vector.norm())
    print(f"  Random vector norm: {random_vector.norm().item():.4f}")
    
    zero_vector = torch.zeros_like(steering_vector)
    print(f"  Zero vector norm: {zero_vector.norm().item():.4f}")
    
    # Generate outputs for all conditions
    print("\n[5/5] Generating outputs...")
    results = []
    
    for prompt_idx, prompt_text in enumerate(tqdm(baseline_prompts, desc="Testing prompts")):
        
        # Safety check
        if "watch yourself" in prompt_text.lower() or "observe yourself" in prompt_text.lower():
            print(f"WARNING: Prompt {prompt_idx} appears recursive, skipping...")
            continue
        
        # Condition A: Steering vector
        text_a = generate_with_vector(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            vector=steering_vector,
            vector_alpha=steering_alpha,
            layer_idx=steering_layer,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        scores_a = score_recursive_themes(text_a)
        
        # Condition B: Random vector
        text_b = generate_with_vector(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            vector=random_vector,
            vector_alpha=steering_alpha,
            layer_idx=steering_layer,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        scores_b = score_recursive_themes(text_b)
        
        # Condition C: Baseline (no intervention)
        inputs_c = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            outputs_c = model.generate(
                **inputs_c,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        text_c = tokenizer.decode(outputs_c[0][inputs_c["input_ids"].shape[1]:], skip_special_tokens=True)
        scores_c = score_recursive_themes(text_c)
        
        results.append({
            'prompt_idx': prompt_idx,
            'baseline_prompt': prompt_text,
            'condition_a_text': text_a,
            'condition_b_text': text_b,
            'condition_c_text': text_c,
            'condition_a_scores': scores_a,
            'condition_b_scores': scores_b,
            'condition_c_scores': scores_c,
        })
    
    # Compute summary statistics
    total_a = sum(r['condition_a_scores']['total_score'] for r in results)
    total_b = sum(r['condition_b_scores']['total_score'] for r in results)
    total_c = sum(r['condition_c_scores']['total_score'] for r in results)
    
    mean_a = total_a / len(results) if results else 0
    mean_b = total_b / len(results) if results else 0
    mean_c = total_c / len(results) if results else 0
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Condition A (Steering): Total={total_a}, Mean={mean_a:.2f}")
    print(f"Condition B (Random):   Total={total_b}, Mean={mean_b:.2f}")
    print(f"Condition C (Baseline): Total={total_c}, Mean={mean_c:.2f}")
    print()
    print("Expected:")
    print("  A > 20: Recursive themes dominate")
    print("  B < 10: Random themes")
    print("  C < 5:  Factual answers")
    print("=" * 80)
    
    # Save results
    results_file = run_dir / "steering_control_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create CSV summary
    csv_data = []
    for r in results:
        csv_data.append({
            'prompt_idx': r['prompt_idx'],
            'baseline_prompt': r['baseline_prompt'][:100] + '...' if len(r['baseline_prompt']) > 100 else r['baseline_prompt'],
            'condition_a_total': r['condition_a_scores']['total_score'],
            'condition_a_self_ref': r['condition_a_scores']['self_reference'],
            'condition_a_contemplative': r['condition_a_scores']['contemplative'],
            'condition_a_meta_process': r['condition_a_scores']['meta_process'],
            'condition_a_recursive': r['condition_a_scores']['recursive_structures'],
            'condition_b_total': r['condition_b_scores']['total_score'],
            'condition_b_self_ref': r['condition_b_scores']['self_reference'],
            'condition_b_contemplative': r['condition_b_scores']['contemplative'],
            'condition_b_meta_process': r['condition_b_scores']['meta_process'],
            'condition_b_recursive': r['condition_b_scores']['recursive_structures'],
            'condition_c_total': r['condition_c_scores']['total_score'],
            'condition_c_self_ref': r['condition_c_scores']['self_reference'],
            'condition_c_contemplative': r['condition_c_scores']['contemplative'],
            'condition_c_meta_process': r['condition_c_scores']['meta_process'],
            'condition_c_recursive': r['condition_c_scores']['recursive_structures'],
        })
    
    df = pd.DataFrame(csv_data)
    csv_file = run_dir / "steering_control_summary.csv"
    df.to_csv(csv_file, index=False)
    
    # Create detailed text file for manual review
    text_file = run_dir / "steering_control_outputs.txt"
    with open(text_file, 'w') as f:
        f.write("STEERING VECTOR CONTROL EXPERIMENT - FULL OUTPUTS\n")
        f.write("=" * 80 + "\n\n")
        
        for r in results:
            f.write(f"PROMPT {r['prompt_idx']}\n")
            f.write(f"Baseline: {r['baseline_prompt']}\n\n")
            
            f.write("CONDITION A (Steering Vector):\n")
            f.write(f"Score: {r['condition_a_scores']['total_score']}\n")
            f.write(f"  Self-ref: {r['condition_a_scores']['self_reference']}, ")
            f.write(f"Contemplative: {r['condition_a_scores']['contemplative']}, ")
            f.write(f"Meta-process: {r['condition_a_scores']['meta_process']}, ")
            f.write(f"Recursive: {r['condition_a_scores']['recursive_structures']}\n")
            f.write(f"Text: {r['condition_a_text']}\n\n")
            
            f.write("CONDITION B (Random Vector):\n")
            f.write(f"Score: {r['condition_b_scores']['total_score']}\n")
            f.write(f"  Self-ref: {r['condition_b_scores']['self_reference']}, ")
            f.write(f"Contemplative: {r['condition_b_scores']['contemplative']}, ")
            f.write(f"Meta-process: {r['condition_b_scores']['meta_process']}, ")
            f.write(f"Recursive: {r['condition_b_scores']['recursive_structures']}\n")
            f.write(f"Text: {r['condition_b_text']}\n\n")
            
            f.write("CONDITION C (Baseline - No Intervention):\n")
            f.write(f"Score: {r['condition_c_scores']['total_score']}\n")
            f.write(f"  Self-ref: {r['condition_c_scores']['self_reference']}, ")
            f.write(f"Contemplative: {r['condition_c_scores']['contemplative']}, ")
            f.write(f"Meta-process: {r['condition_c_scores']['meta_process']}, ")
            f.write(f"Recursive: {r['condition_c_scores']['recursive_structures']}\n")
            f.write(f"Text: {r['condition_c_text']}\n\n")
            f.write("-" * 80 + "\n\n")
    
    summary = {
        'total_prompts': len(results),
        'condition_a_total': total_a,
        'condition_a_mean': mean_a,
        'condition_b_total': total_b,
        'condition_b_mean': mean_b,
        'condition_c_total': total_c,
        'condition_c_mean': mean_c,
        'steering_vector_norm': steering_norm,
        'random_vector_norm': random_vector.norm().item(),
    }
    
    summary_file = run_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return ExperimentResult(
        summary=summary,
    )

