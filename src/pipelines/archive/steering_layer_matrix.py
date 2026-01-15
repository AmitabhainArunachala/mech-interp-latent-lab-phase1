"""Pipeline for Layer Matrix experiment - Extract/Apply steering vectors across layers."""
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.pipelines.archive.steering import (
    compute_steering_vector,
    SteeringVectorPatcher,
    _generate_with_steering,
)
from src.metrics.behavior_strict import score_behavior_strict
from src.pipelines.registry import ExperimentResult


def run_layer_matrix_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """Run Layer Matrix experiment: Extract from layers, apply to layers."""
    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-v0.1")
    layer_idx = params.get("layer", 27)  # Default, but extract_layers overrides
    n_prompts = params.get("n_prompts", 50)
    n_test_pairs = params.get("n_test_pairs", 20)
    max_new_tokens = params.get("max_new_tokens", 100)
    temperature = params.get("temperature", 0.7)
    extract_layers = params.get("extract_layers", [20, 24, 25, 26, 27, 28])
    apply_layers = params.get("apply_layers", [20, 24, 25, 26, 27, 28])
    alpha = params.get("alpha", 1.0)
    recursive_groups = params.get("recursive_groups", ["L4_full", "L5_refined"])
    baseline_groups = params.get("baseline_groups", ["baseline_math", "baseline_factual", "baseline_instructional"])
    seed = int(params.get("seed", 42))
    
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("LAYER MATRIX EXPERIMENT")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Extract layers: {extract_layers}")
    print(f"Apply layers: {apply_layers}")
    print(f"Alpha: {alpha}")
    print(f"Test pairs: {n_test_pairs}")
    print(f"Device: {device}")
    
    # Load model
    print("\n[1/3] Loading model...")
    model, tokenizer = load_model(model_name, device=device)
    model.eval()
    
    # Load prompts
    print("\n[2/3] Loading prompts...")
    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    (run_dir / "prompt_bank_version.json").write_text(json.dumps({"version": bank_version}, indent=2) + "\n")
    
    recursive_prompts = []
    for group in recursive_groups:
        prompts = loader.get_by_group(group)
        recursive_prompts.extend(prompts)
    
    baseline_prompts = []
    for group in baseline_groups:
        prompts = loader.get_by_group(group)
        baseline_prompts.extend(prompts)
    
    # Sample prompts
    if len(recursive_prompts) < n_prompts:
        sampled_recursive = recursive_prompts
    else:
        sampled_recursive = np.random.choice(recursive_prompts, size=n_prompts, replace=False).tolist()
    
    if len(baseline_prompts) < n_prompts:
        sampled_baseline = baseline_prompts
    else:
        sampled_baseline = np.random.choice(baseline_prompts, size=n_prompts, replace=False).tolist()
    
    print(f"  Recursive: {len(sampled_recursive)}")
    print(f"  Baseline: {len(sampled_baseline)}")
    
    # Get test pairs
    test_pairs = loader.get_balanced_pairs(n_pairs=n_test_pairs, seed=seed)
    print(f"  Test pairs: {len(test_pairs)}")
    
    # Extract vectors from each layer
    print("\n[3/3] Extracting steering vectors from each layer...")
    extract_vectors = {}
    for layer_idx in extract_layers:
        print(f"  Extracting from L{layer_idx}...")
        vec = compute_steering_vector(
            model, tokenizer, sampled_recursive, sampled_baseline, layer_idx, device
        )
        extract_vectors[layer_idx] = vec
        torch.save(vec.cpu(), run_dir / f"steering_vector_L{layer_idx}.pt")
        print(f"    Vector norm: {vec.norm().item():.4f}")
    
    # Test all combinations
    print("\n" + "=" * 80)
    print("TESTING ALL COMBINATIONS")
    print("=" * 80)
    results = []
    total_combos = len(extract_layers) * len(apply_layers) * len(test_pairs)
    
    with tqdm(total=total_combos, desc="Testing combinations") as pbar:
        for extract_layer in extract_layers:
            vec = extract_vectors[extract_layer]
            
            for apply_layer in apply_layers:
                # Test on all pairs
                scores = []
                passes = []
                
                # CRITICAL FIX: get_balanced_pairs returns (recursive, baseline)
                for rec_text, base_text in test_pairs:
                    # Safety gate: Verify base_text is actually a baseline prompt
                    if "watch yourself" in base_text.lower() or "observe yourself" in base_text.lower() or "consciousness examining" in base_text.lower():
                        raise ValueError(f"CRITICAL ERROR: Baseline prompt appears recursive! Text: {base_text[:100]}")
                    
                    patcher = SteeringVectorPatcher(model, vec, alpha)
                    patcher.register(layer_idx=apply_layer)
                    
                    try:
                        text, entropy = _generate_with_steering(
                            model, tokenizer, base_text, max_new_tokens, temperature, device
                        )
                        score = score_behavior_strict(text, entropy)
                        scores.append(score.final_score)
                        passes.append(score.passed_gates)
                    except Exception as e:
                        print(f"  Error: {e}")
                        scores.append(0.0)
                        passes.append(False)
                    finally:
                        patcher.remove()
                    
                    pbar.update(1)
                
                transfer_rate = sum(s > 0.3 for s in scores) / len(scores)
                pass_rate = sum(passes) / len(passes)
                mean_score = np.mean(scores)
                collapse_rate = 1.0 - pass_rate
                
                results.append({
                    "extract_layer": extract_layer,
                    "apply_layer": apply_layer,
                    "transfer_rate": transfer_rate,
                    "pass_rate": pass_rate,
                    "collapse_rate": collapse_rate,
                    "mean_score": mean_score,
                })
                
                print(f"  Extract L{extract_layer} → Apply L{apply_layer}: Transfer={transfer_rate*100:.1f}%, Collapse={collapse_rate*100:.1f}%, Score={mean_score:.4f}")
    
    # Save results
    df = pd.DataFrame(results)
    csv_path = run_dir / "layer_matrix_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved results to {csv_path}")
    
    # Create pivot table for easy viewing
    pivot = df.pivot(index='extract_layer', columns='apply_layer', values='transfer_rate')
    pivot_path = run_dir / "layer_matrix_pivot.csv"
    pivot.to_csv(pivot_path)
    print(f"✅ Saved pivot table to {pivot_path}")
    
    # Find best combinations
    best_row = df.loc[df['transfer_rate'].idxmax()]
    best_extract = best_row['extract_layer']
    best_apply = best_row['apply_layer']
    best_transfer = best_row['transfer_rate']
    
    # Find earliest working layers
    df_above_20 = df[df['transfer_rate'] >= 0.20]
    if len(df_above_20) > 0:
        earliest_extract = df_above_20['extract_layer'].min()
        earliest_apply = df_above_20['apply_layer'].min()
    else:
        earliest_extract = None
        earliest_apply = None
    
    summary = {
        "experiment": "layer_matrix",
        "model_name": model_name,
        "extract_layers": extract_layers,
        "apply_layers": apply_layers,
        "alpha": alpha,
        "n_test_pairs": n_test_pairs,
        "best_extract_layer": int(best_extract),
        "best_apply_layer": int(best_apply),
        "best_transfer_rate": float(best_transfer),
        "earliest_extract_layer": int(earliest_extract) if earliest_extract is not None else None,
        "earliest_apply_layer": int(earliest_apply) if earliest_apply is not None else None,
        "results": results,
        "finding": f"Best: Extract L{best_extract} → Apply L{best_apply} ({best_transfer*100:.1f}% transfer)"
    }
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Best combination: Extract L{best_extract} → Apply L{best_apply}")
    print(f"  Transfer Rate: {best_transfer*100:.1f}%")
    print(f"  Mean Score: {best_row['mean_score']:.4f}")
    if earliest_extract is not None:
        print(f"\nEarliest working extract layer: L{earliest_extract}")
        print(f"Earliest working apply layer: L{earliest_apply}")
    
    return ExperimentResult(summary=summary)

