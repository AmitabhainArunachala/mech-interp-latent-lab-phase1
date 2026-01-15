"""Extended Context Steering - Track temporal development of recursive behavior."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.pipelines.archive.steering import compute_steering_vector, SteeringVectorPatcher
from src.metrics.behavior_strict import score_behavior_strict
from src.pipelines.registry import ExperimentResult


def generate_with_temporal_tracking(
    model,
    tokenizer,
    prompt: str,
    steering_vector: torch.Tensor,
    steering_layer: int,
    steering_alpha: float,
    max_new_tokens: int = 500,
    temperature: float = 0.7,
    segment_size: int = 50,  # Analyze every 50 tokens
    device: str = "cuda",
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generate with steering and track temporal development.
    
    Returns:
        - Full generated text
        - List of segment analyses with token positions
    """
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    
    # Setup steering patcher
    steering_patcher = SteeringVectorPatcher(model, steering_vector, steering_alpha)
    steering_patcher.register(layer_idx=steering_layer)
    
    try:
        generated_tokens = []
        segment_analyses = []
        
        with torch.no_grad():
            current_ids = inputs["input_ids"][:, -1:]
            current_past = None
            
            for step in range(max_new_tokens):
                # Forward pass
                outputs = model(
                    input_ids=current_ids,
                    past_key_values=current_past,
                    use_cache=True,
                )
                
                # Get logits and sample
                logits = outputs.logits[:, -1, :]
                probs_temp = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs_temp, num_samples=1)
                
                generated_tokens.append(next_token.item())
                
                # Analyze every segment_size tokens
                if len(generated_tokens) % segment_size == 0:
                    # Decode current segment
                    segment_start = max(0, len(generated_tokens) - segment_size)
                    segment_tokens = generated_tokens[segment_start:]
                    segment_text = tokenizer.decode(segment_tokens, skip_special_tokens=True)
                    
                    # Score this segment
                    # Compute entropy for this segment
                    probs = torch.softmax(logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()
                    
                    score = score_behavior_strict(segment_text, entropy)
                    
                    # Check for recursive patterns
                    text_lower = segment_text.lower()
                    has_strange_loop = any(phrase in text_lower for phrase in [
                        'awareness is aware', 'consciousness examining', 'watching yourself',
                        'observe the observer', 'self-aware', 'aware of awareness'
                    ])
                    has_meta_cognition = any(phrase in text_lower for phrase in [
                        'thinking about thinking', 'knowing that i know',
                        'aware that i am aware'
                    ])
                    has_self_reference = any(phrase in text_lower for phrase in [
                        'itself is itself', 'defines itself', 'refers to itself',
                        'the process of the process'
                    ])
                    
                    segment_analyses.append({
                        'token_position': len(generated_tokens),
                        'segment_text': segment_text,
                        'final_score': score.final_score,
                        'recursion_score': score.recursion_score,
                        'coherence_score': score.coherence_score,
                        'has_strange_loop': has_strange_loop,
                        'has_meta_cognition': has_meta_cognition,
                        'has_self_reference': has_self_reference,
                        'entropy': entropy,
                    })
                
                # Update for next iteration
                current_ids = next_token
                current_past = outputs.past_key_values
                
                # Stop on EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        # Decode full text
        full_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return full_text, segment_analyses
        
    finally:
        steering_patcher.remove()


def run_extended_context_steering_from_config(
    cfg: Dict[str, Any],
    run_dir: Path,
) -> ExperimentResult:
    """Run extended context steering experiment."""
    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-v0.1")
    n_prompts = params.get("n_prompts", 50)
    n_test_prompts = params.get("n_test_prompts", 10)
    max_new_tokens = params.get("max_new_tokens", 500)
    steering_layer = params.get("steering_layer", 27)
    steering_alpha = params.get("steering_alpha", 2.0)
    segment_size = params.get("segment_size", 50)
    
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("EXTENDED CONTEXT STEERING - TEMPORAL DEVELOPMENT TRACKING")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Steering Layer: L{steering_layer}, Alpha: {steering_alpha}")
    print(f"Max Tokens: {max_new_tokens}, Segment Size: {segment_size}")
    print(f"Device: {device}")
    
    # Load model
    print("\n[1/4] Loading model...")
    model, tokenizer = load_model(model_name, device=device)
    model.eval()
    
    # Load prompts
    print("\n[2/4] Loading prompts...")
    loader = PromptLoader()
    
    recursive_prompts = []
    for group in ["L4_full", "L5_refined"]:
        prompts = loader.get_by_group(group)
        recursive_prompts.extend(prompts)
    
    baseline_prompts = []
    for group in ["baseline_math", "baseline_factual", "baseline_instructional"]:
        prompts = loader.get_by_group(group)
        baseline_prompts.extend(prompts)
    
    # Sample prompts
    if len(recursive_prompts) < n_prompts:
        sampled_recursive = recursive_prompts
    else:
        sampled_recursive = np.random.choice(recursive_prompts, size=n_prompts, replace=False).tolist()
    
    if len(baseline_prompts) < n_test_prompts:
        sampled_baseline = baseline_prompts
    else:
        sampled_baseline = np.random.choice(baseline_prompts, size=n_test_prompts, replace=False).tolist()
    
    print(f"  Recursive (for vector): {len(sampled_recursive)}")
    print(f"  Baseline (for testing): {len(sampled_baseline)}")
    
    # Compute steering vector
    print("\n[3/4] Computing steering vector...")
    steering_vector = compute_steering_vector(
        model, tokenizer, sampled_recursive, baseline_prompts, steering_layer, device
    )
    print(f"  Steering vector norm: {steering_vector.norm().item():.4f}")
    
    # Run on test prompts
    print("\n[4/4] Running extended context generation...")
    results = []
    
    for prompt_idx, base_prompt in enumerate(tqdm(sampled_baseline, desc="Generating")):
        # Safety gate
        if "watch yourself" in base_prompt.lower() or "observe yourself" in base_prompt.lower():
            print(f"  ⚠️  Skipping recursive-looking prompt: {base_prompt[:50]}...")
            continue
        
        try:
            full_text, segment_analyses = generate_with_temporal_tracking(
                model, tokenizer, base_prompt,
                steering_vector, steering_layer, steering_alpha,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                segment_size=segment_size,
                device=device,
            )
            
            # Find mode shifts
            factual_to_contemplative = None
            contemplative_to_recursive = None
            
            for i, seg in enumerate(segment_analyses):
                # Check for shift from factual to contemplative
                if factual_to_contemplative is None:
                    # Look for contemplative language
                    if seg['recursion_score'] > 0.1 or seg['has_meta_cognition']:
                        factual_to_contemplative = seg['token_position']
                
                # Check for shift to recursive
                if contemplative_to_recursive is None:
                    if seg['has_strange_loop'] or seg['has_self_reference'] or seg['recursion_score'] > 0.3:
                        contemplative_to_recursive = seg['token_position']
            
            results.append({
                'prompt_idx': prompt_idx,
                'baseline_prompt': base_prompt,
                'full_generated_text': full_text,
                'text_length': len(full_text),
                'num_tokens': len(full_text.split()),
                'factual_to_contemplative_token': factual_to_contemplative,
                'contemplative_to_recursive_token': contemplative_to_recursive,
                'num_segments': len(segment_analyses),
                'segment_analyses': segment_analyses,
            })
            
        except Exception as e:
            print(f"  Error on prompt {prompt_idx}: {e}")
            results.append({
                'prompt_idx': prompt_idx,
                'baseline_prompt': base_prompt,
                'full_generated_text': '',
                'error': str(e),
            })
    
    # Save results
    # First, save full results with segments
    full_results_path = run_dir / "extended_context_full_results.json"
    with open(full_results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Also create CSV with summary
    summary_rows = []
    for r in results:
        if 'error' in r:
            continue
        summary_rows.append({
            'prompt_idx': r['prompt_idx'],
            'baseline_prompt': r['baseline_prompt'][:200],
            'generated_length': r['text_length'],
            'num_tokens': r['num_tokens'],
            'factual_to_contemplative': r.get('factual_to_contemplative_token'),
            'contemplative_to_recursive': r.get('contemplative_to_recursive_token'),
            'num_segments': r.get('num_segments', 0),
        })
    
    df = pd.DataFrame(summary_rows)
    csv_path = run_dir / "extended_context_summary.csv"
    df.to_csv(csv_path, index=False)
    
    # Create human-readable analysis
    analysis_path = run_dir / "temporal_analysis.md"
    with open(analysis_path, "w") as f:
        f.write("# Extended Context Steering - Temporal Development Analysis\n\n")
        f.write(f"**Configuration:** Steering L{steering_layer}, α={steering_alpha}, {max_new_tokens} tokens\n\n")
        f.write(f"**Total Prompts:** {len(results)}\n\n")
        f.write("---\n\n")
        
        for r in results:
            if 'error' in r:
                continue
            
            f.write(f"## Prompt {r['prompt_idx']}\n\n")
            f.write(f"**Baseline:** {r['baseline_prompt']}\n\n")
            
            if r.get('factual_to_contemplative_token'):
                f.write(f"**Mode Shift 1 (Factual → Contemplative):** Token {r['factual_to_contemplative_token']}\n\n")
            else:
                f.write(f"**Mode Shift 1:** Not detected\n\n")
            
            if r.get('contemplative_to_recursive_token'):
                f.write(f"**Mode Shift 2 (Contemplative → Recursive):** Token {r['contemplative_to_recursive_token']}\n\n")
            else:
                f.write(f"**Mode Shift 2:** Not detected\n\n")
            
            f.write("**Segment Analysis:**\n\n")
            for seg in r.get('segment_analyses', []):
                f.write(f"- **Token {seg['token_position']}:** ")
                f.write(f"Recursion={seg['recursion_score']:.3f}, ")
                f.write(f"Strange Loop={seg['has_strange_loop']}, ")
                f.write(f"Self-Ref={seg['has_self_reference']}\n")
                f.write(f"  ```\n  {seg['segment_text'][:200]}...\n  ```\n\n")
            
            f.write("**Full Generated Text:**\n\n")
            f.write(f"```\n{r['full_generated_text']}\n```\n\n")
            f.write("---\n\n")
    
    print(f"\n✅ Results saved:")
    print(f"   Full results: {full_results_path}")
    print(f"   Summary CSV: {csv_path}")
    print(f"   Temporal analysis: {analysis_path}")
    
    summary = {
        "experiment": "extended_context_steering",
        "num_prompts": len(results),
        "mode_shifts_detected": sum(1 for r in results if r.get('factual_to_contemplative_token')),
        "recursive_shifts_detected": sum(1 for r in results if r.get('contemplative_to_recursive_token')),
    }
    
    return ExperimentResult(summary=summary)








