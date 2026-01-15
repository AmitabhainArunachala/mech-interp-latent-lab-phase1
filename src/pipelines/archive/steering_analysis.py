"""
Pipeline 9 Extended: Comprehensive Steering Vector Analysis

MISSION CRITICAL: Characterize the steering vector to publication-grade precision.

Experiments:
1. Stability Check
2. Layer Matrix (extract/apply combinations)
3. Head Decomposition
4. Generalization (train/test split)
5. Failure Analysis
6. Dose-Response Curve
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.core.patching import extract_v_activation
from src.metrics.behavior_strict import score_behavior_strict
from src.pipelines.registry import ExperimentResult
from src.pipelines.archive.steering import SteeringVectorPatcher, _generate_with_steering


def compute_steering_vector(
    model,
    tokenizer,
    recursive_prompts: List[str],
    baseline_prompts: List[str],
    layer_idx: int,
    device: str,
) -> torch.Tensor:
    """Compute steering vector as mean difference."""
    print(f"Computing steering vector from {len(recursive_prompts)} recursive + {len(baseline_prompts)} baseline prompts...")
    
    recursive_vs = []
    for prompt in tqdm(recursive_prompts, desc=f"Extracting recursive V_PROJ@L{layer_idx}"):
        try:
            v_act = extract_v_activation(model, tokenizer, prompt, layer_idx, device)
            recursive_vs.append(v_act.mean(dim=0))
        except Exception as e:
            print(f"  Warning: Failed to extract V_PROJ: {e}")
            continue
    
    baseline_vs = []
    for prompt in tqdm(baseline_prompts, desc=f"Extracting baseline V_PROJ@L{layer_idx}"):
        try:
            v_act = extract_v_activation(model, tokenizer, prompt, layer_idx, device)
            baseline_vs.append(v_act.mean(dim=0))
        except Exception as e:
            print(f"  Warning: Failed to extract V_PROJ: {e}")
            continue
    
    if len(recursive_vs) == 0 or len(baseline_vs) == 0:
        raise RuntimeError(f"Insufficient activations: {len(recursive_vs)} recursive, {len(baseline_vs)} baseline")
    
    recursive_mean = torch.stack(recursive_vs).mean(dim=0)
    baseline_mean = torch.stack(baseline_vs).mean(dim=0)
    steering_vector = recursive_mean - baseline_mean
    
    print(f"  Vector shape: {steering_vector.shape}, norm: {steering_vector.norm().item():.4f}")
    return steering_vector


def extract_v_activation_per_head(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    device: str,
) -> torch.Tensor:
    """
    Extract V_PROJ activation per head at specified layer.
    
    Returns:
        Tensor of shape (num_heads, hidden_dim_per_head) = (32, 128) for Mistral-7B
    """
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    
    # Get attention layer
    layer = model.model.layers[layer_idx].self_attn
    
    # Hook to capture V_PROJ output before reshaping
    v_per_head = None
    
    def capture_hook(module, inp, out):
        nonlocal v_per_head
        # out shape: (batch, seq_len, hidden_dim)
        # We need to reshape to (batch, seq_len, num_heads, head_dim)
        batch, seq_len, hidden_dim = out.shape
        num_heads = model.config.num_attention_heads
        head_dim = hidden_dim // num_heads
        
        # Reshape: (batch, seq_len, num_heads, head_dim)
        v_reshaped = out.view(batch, seq_len, num_heads, head_dim)
        # Mean over sequence: (batch, num_heads, head_dim)
        v_per_head = v_reshaped.mean(dim=1)[0].detach()  # (num_heads, head_dim)
        return out
    
    handle = layer.v_proj.register_forward_hook(capture_hook)
    
    try:
        with torch.no_grad():
            _ = model(**inputs, use_cache=False)
    finally:
        handle.remove()
    
    if v_per_head is None:
        raise RuntimeError(f"Failed to capture V_PROJ per-head activation at layer {layer_idx}")
    
    return v_per_head


def run_experiment_1_stability(
    model,
    tokenizer,
    recursive_prompts: List[str],
    baseline_prompts: List[str],
    layer_idx: int,
    device: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """RUN 1: Stability Check - Split prompts 3 ways, compute vectors, check similarity."""
    print("\n" + "=" * 80)
    print("RUN 1: STABILITY CHECK")
    print("=" * 80)
    
    # Split prompts into 3 groups
    random.seed(42)
    np.random.seed(42)
    
    rec_shuffled = recursive_prompts.copy()
    base_shuffled = baseline_prompts.copy()
    random.shuffle(rec_shuffled)
    random.shuffle(base_shuffled)
    
    n_rec = len(rec_shuffled) // 3
    n_base = len(base_shuffled) // 3
    
    groups = [
        (rec_shuffled[:n_rec], base_shuffled[:n_base]),
        (rec_shuffled[n_rec:2*n_rec], base_shuffled[n_base:2*n_base]),
        (rec_shuffled[2*n_rec:], base_shuffled[2*n_base:]),
    ]
    
    vectors = []
    for i, (rec_group, base_group) in enumerate(groups):
        print(f"\nComputing vector from group {i+1} ({len(rec_group)} recursive, {len(base_group)} baseline)...")
        vec = compute_steering_vector(model, tokenizer, rec_group, base_group, layer_idx, device)
        vectors.append(vec)
        torch.save(vec.cpu(), output_dir / f"steering_vector_group{i+1}_L{layer_idx}.pt")
    
    # Compute pairwise cosine similarities
    similarities = []
    for i in range(3):
        for j in range(i+1, 3):
            cos_sim = F.cosine_similarity(vectors[i].unsqueeze(0), vectors[j].unsqueeze(0)).item()
            similarities.append(cos_sim)
            print(f"  Group {i+1} vs Group {j+1}: {cos_sim:.4f}")
    
    min_sim = min(similarities)
    mean_sim = np.mean(similarities)
    
    result = {
        "experiment": "stability_check",
        "layer": layer_idx,
        "n_groups": 3,
        "similarities": similarities,
        "min_similarity": float(min_sim),
        "mean_similarity": float(mean_sim),
        "passed": min_sim > 0.85,
        "finding": f"Stability: {'PASS' if min_sim > 0.85 else 'FAIL'} (min={min_sim:.4f}, mean={mean_sim:.4f})"
    }
    
    print(f"\n✅ RESULT: {'PASS' if result['passed'] else 'FAIL'}")
    print(f"   Min similarity: {min_sim:.4f}, Mean: {mean_sim:.4f}")
    
    return result


def run_experiment_2_layer_matrix(
    model,
    tokenizer,
    recursive_prompts: List[str],
    baseline_prompts: List[str],
    test_pairs: List[Tuple[str, str]],
    extract_layers: List[int],
    apply_layers: List[int],
    alpha: float,
    device: str,
    output_dir: Path,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """RUN 2: Layer Matrix - Extract from layers, apply to layers, test all combinations."""
    print("\n" + "=" * 80)
    print("RUN 2: LAYER MATRIX")
    print("=" * 80)
    print(f"Extract layers: {extract_layers}")
    print(f"Apply layers: {apply_layers}")
    print(f"Alpha: {alpha}")
    
    # Extract vectors from each layer
    extract_vectors = {}
    for layer_idx in extract_layers:
        print(f"\nExtracting vector from L{layer_idx}...")
        vec = compute_steering_vector(model, tokenizer, recursive_prompts, baseline_prompts, layer_idx, device)
        extract_vectors[layer_idx] = vec
        torch.save(vec.cpu(), output_dir / f"steering_vector_L{layer_idx}.pt")
    
    # Test all combinations
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
                
                results.append({
                    "extract_layer": extract_layer,
                    "apply_layer": apply_layer,
                    "transfer_rate": transfer_rate,
                    "pass_rate": pass_rate,
                    "mean_score": mean_score,
                })
                
                print(f"  Extract L{extract_layer} → Apply L{apply_layer}: Transfer={transfer_rate*100:.1f}%, Score={mean_score:.4f}")
    
    df = pd.DataFrame(results)
    csv_path = output_dir / "layer_matrix_results.csv"
    df.to_csv(csv_path, index=False)
    
    # Find best combinations
    best_extract = df.loc[df['transfer_rate'].idxmax(), 'extract_layer']
    best_apply = df.loc[df['transfer_rate'].idxmax(), 'apply_layer']
    best_transfer = df['transfer_rate'].max()
    
    result = {
        "experiment": "layer_matrix",
        "extract_layers": extract_layers,
        "apply_layers": apply_layers,
        "alpha": alpha,
        "results": results,
        "best_extract_layer": int(best_extract),
        "best_apply_layer": int(best_apply),
        "best_transfer_rate": float(best_transfer),
        "finding": f"Best: Extract L{int(best_extract)} → Apply L{int(best_apply)} ({best_transfer*100:.1f}% transfer)"
    }
    
    print(f"\n✅ RESULT: {result['finding']}")
    
    return result


def run_experiment_3_head_decomposition(
    model,
    tokenizer,
    recursive_prompts: List[str],
    baseline_prompts: List[str],
    test_pairs: List[Tuple[str, str]],
    layer_idx: int,
    alpha: float,
    device: str,
    output_dir: Path,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """RUN 3: Head Decomposition - Extract per-head vectors, test each individually."""
    print("\n" + "=" * 80)
    print("RUN 3: HEAD DECOMPOSITION")
    print("=" * 80)
    print(f"Layer: {layer_idx}")
    
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    print(f"Num heads: {num_heads}, Head dim: {head_dim}")
    
    # Extract per-head vectors
    print("\nExtracting recursive V_PROJ per head...")
    rec_per_head = []
    for prompt in tqdm(recursive_prompts[:20], desc="Recursive"):  # Sample for speed
        try:
            v_heads = extract_v_activation_per_head(model, tokenizer, prompt, layer_idx, device)
            rec_per_head.append(v_heads)
        except Exception as e:
            print(f"  Warning: {e}")
            continue
    
    print("Extracting baseline V_PROJ per head...")
    base_per_head = []
    for prompt in tqdm(baseline_prompts[:20], desc="Baseline"):  # Sample for speed
        try:
            v_heads = extract_v_activation_per_head(model, tokenizer, prompt, layer_idx, device)
            base_per_head.append(v_heads)
        except Exception as e:
            print(f"  Warning: {e}")
            continue
    
    if len(rec_per_head) == 0 or len(base_per_head) == 0:
        raise RuntimeError("Failed to extract per-head activations")
    
    # Compute per-head steering vectors
    rec_mean = torch.stack(rec_per_head).mean(dim=0)  # (num_heads, head_dim)
    base_mean = torch.stack(base_per_head).mean(dim=0)  # (num_heads, head_dim)
    steering_per_head = rec_mean - base_mean  # (num_heads, head_dim)
    
    # Save per-head vectors
    for head_idx in range(num_heads):
        torch.save(steering_per_head[head_idx].cpu(), output_dir / f"steering_vector_H{head_idx}_L{layer_idx}.pt")
    
    # Test each head individually
    results = []
    
    for head_idx in tqdm(range(num_heads), desc="Testing heads"):
        # Create full-dimension vector with only this head's contribution
        # We need to expand from (head_dim,) to (hidden_dim,)
        head_vec = steering_per_head[head_idx]  # (head_dim,)
        
        # Create full vector: zero out all other heads
        full_vec = torch.zeros(model.config.hidden_size, device=device)
        start_idx = head_idx * head_dim
        end_idx = start_idx + head_dim
        full_vec[start_idx:end_idx] = head_vec
        
        # Test on pairs
        scores = []
        passes = []
        
        # CRITICAL FIX: get_balanced_pairs returns (recursive, baseline)
        for rec_text, base_text in test_pairs[:10]:  # Sample for speed
            # Safety gate: Verify base_text is actually a baseline prompt
            if "watch yourself" in base_text.lower() or "observe yourself" in base_text.lower() or "consciousness examining" in base_text.lower():
                raise ValueError(f"CRITICAL ERROR: Baseline prompt appears recursive! Text: {base_text[:100]}")
            
            patcher = SteeringVectorPatcher(model, full_vec, alpha)
            patcher.register(layer_idx=layer_idx)
            
            try:
                text, entropy = _generate_with_steering(
                    model, tokenizer, base_text, max_new_tokens, temperature, device
                )
                score = score_behavior_strict(text, entropy)
                scores.append(score.final_score)
                passes.append(score.passed_gates)
            except Exception as e:
                scores.append(0.0)
                passes.append(False)
            finally:
                patcher.remove()
        
        transfer_rate = sum(s > 0.3 for s in scores) / len(scores)
        pass_rate = sum(passes) / len(passes)
        mean_score = np.mean(scores)
        
        results.append({
            "head": head_idx,
            "transfer_rate": transfer_rate,
            "pass_rate": pass_rate,
            "mean_score": mean_score,
            "vector_norm": head_vec.norm().item(),
        })
    
    df = pd.DataFrame(results)
    csv_path = output_dir / "head_decomposition_results.csv"
    df.to_csv(csv_path, index=False)
    
    # Find best heads
    best_heads = df.nlargest(5, 'transfer_rate')[['head', 'transfer_rate', 'mean_score']]
    
    result = {
        "experiment": "head_decomposition",
        "layer": layer_idx,
        "num_heads": num_heads,
        "results": results,
        "top_heads": best_heads.to_dict('records'),
        "finding": f"Top heads: {best_heads['head'].tolist()} (transfer rates: {best_heads['transfer_rate'].tolist()})"
    }
    
    print(f"\n✅ RESULT: {result['finding']}")
    
    return result


def run_experiment_4_generalization(
    model,
    tokenizer,
    recursive_prompts: List[str],
    baseline_prompts: List[str],
    test_pairs: List[Tuple[str, str]],
    layer_idx: int,
    alpha: float,
    device: str,
    output_dir: Path,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """RUN 4: Generalization - Train/test split, compute on train, test on held-out."""
    print("\n" + "=" * 80)
    print("RUN 4: GENERALIZATION")
    print("=" * 80)
    
    # Split prompts 50/50
    random.seed(42)
    np.random.seed(42)
    
    rec_shuffled = recursive_prompts.copy()
    base_shuffled = baseline_prompts.copy()
    random.shuffle(rec_shuffled)
    random.shuffle(base_shuffled)
    
    split_idx = len(rec_shuffled) // 2
    
    train_rec = rec_shuffled[:split_idx]
    train_base = base_shuffled[:split_idx]
    test_rec = rec_shuffled[split_idx:]
    test_base = base_shuffled[split_idx:]
    
    print(f"Train: {len(train_rec)} recursive, {len(train_base)} baseline")
    print(f"Test: {len(test_rec)} recursive, {len(test_base)} baseline")
    
    # Compute vector from train set
    print("\nComputing steering vector from train set...")
    train_vector = compute_steering_vector(model, tokenizer, train_rec, train_base, layer_idx, device)
    torch.save(train_vector.cpu(), output_dir / f"steering_vector_train_L{layer_idx}.pt")
    
    # Test on held-out test pairs
    print("\nTesting on held-out test pairs...")
    scores = []
    passes = []
    
    # CRITICAL FIX: get_balanced_pairs returns (recursive, baseline)
    for rec_text, base_text in tqdm(test_pairs, desc="Testing"):
        # Safety gate: Verify base_text is actually a baseline prompt
        if "watch yourself" in base_text.lower() or "observe yourself" in base_text.lower() or "consciousness examining" in base_text.lower():
            raise ValueError(f"CRITICAL ERROR: Baseline prompt appears recursive! Text: {base_text[:100]}")
        
        patcher = SteeringVectorPatcher(model, train_vector, alpha)
        patcher.register(layer_idx=layer_idx)
        
        try:
            text, entropy = _generate_with_steering(
                model, tokenizer, base_text, max_new_tokens, temperature, device
            )
            score = score_behavior_strict(text, entropy)
            scores.append(score.final_score)
            passes.append(score.passed_gates)
        except Exception as e:
            scores.append(0.0)
            passes.append(False)
        finally:
            patcher.remove()
    
    transfer_rate = sum(s > 0.3 for s in scores) / len(scores)
    pass_rate = sum(passes) / len(passes)
    mean_score = np.mean(scores)
    
    result = {
        "experiment": "generalization",
        "layer": layer_idx,
        "alpha": alpha,
        "train_size": len(train_rec),
        "test_size": len(test_pairs),
        "transfer_rate": float(transfer_rate),
        "pass_rate": float(pass_rate),
        "mean_score": float(mean_score),
        "passed": transfer_rate > 0.4,
        "finding": f"Generalization: {'PASS' if transfer_rate > 0.4 else 'FAIL'} ({transfer_rate*100:.1f}% transfer on test set)"
    }
    
    print(f"\n✅ RESULT: {result['finding']}")
    
    return result


def run_experiment_5_failure_analysis(
    model,
    tokenizer,
    recursive_prompts: List[str],
    baseline_prompts: List[str],
    test_pairs: List[Tuple[str, str]],
    layer_idx: int,
    alpha: float,
    device: str,
    output_dir: Path,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """RUN 5: Failure Analysis - Compare failed vs successful pairs."""
    print("\n" + "=" * 80)
    print("RUN 5: FAILURE ANALYSIS")
    print("=" * 80)
    
    # Compute steering vector
    print("Computing steering vector...")
    steering_vector = compute_steering_vector(model, tokenizer, recursive_prompts, baseline_prompts, layer_idx, device)
    
    # Test all pairs and collect features
    from src.metrics.rv import compute_rv
    
    results = []
    
    # CRITICAL FIX: get_balanced_pairs returns (recursive, baseline)
    for i, (rec_text, base_text) in enumerate(tqdm(test_pairs, desc="Analyzing pairs")):
        # Safety gate: Verify base_text is actually a baseline prompt
        if "watch yourself" in base_text.lower() or "observe yourself" in base_text.lower() or "consciousness examining" in base_text.lower():
            raise ValueError(f"CRITICAL ERROR: Baseline prompt appears recursive! Text: {base_text[:100]}")
        # Compute features
        try:
            base_rv = compute_rv(model, tokenizer, base_text, early=5, late=27, window=16, device=device)
            rec_rv = compute_rv(model, tokenizer, rec_text, early=5, late=27, window=16, device=device)
            rv_gap = base_rv - rec_rv
            
            base_len = len(tokenizer.encode(base_text, add_special_tokens=False))
            rec_len = len(tokenizer.encode(rec_text, add_special_tokens=False))
            length_diff = abs(base_len - rec_len)
        except Exception:
            base_rv = rec_rv = rv_gap = np.nan
            base_len = rec_len = length_diff = np.nan
        
        # Generate with steering
        patcher = SteeringVectorPatcher(model, steering_vector, alpha)
        patcher.register(layer_idx=layer_idx)
        
        try:
            text, entropy = _generate_with_steering(
                model, tokenizer, base_text, max_new_tokens, temperature, device
            )
            score = score_behavior_strict(text, entropy)
            
            results.append({
                "pair_idx": i,
                "base_text": base_text[:100],
                "success": score.final_score > 0.3,
                "final_score": score.final_score,
                "passed_gates": score.passed_gates,
                "base_rv": base_rv,
                "rec_rv": rec_rv,
                "rv_gap": rv_gap,
                "base_len": base_len,
                "rec_len": rec_len,
                "length_diff": length_diff,
            })
        except Exception as e:
            results.append({
                "pair_idx": i,
                "base_text": base_text[:100],
                "success": False,
                "final_score": 0.0,
                "passed_gates": False,
                "base_rv": base_rv,
                "rec_rv": rec_rv,
                "rv_gap": rv_gap,
                "base_len": base_len,
                "rec_len": rec_len,
                "length_diff": length_diff,
            })
        finally:
            patcher.remove()
    
    df = pd.DataFrame(results)
    csv_path = output_dir / "failure_analysis_results.csv"
    df.to_csv(csv_path, index=False)
    
    # Compare successes vs failures
    success_df = df[df['success'] == True]
    failure_df = df[df['success'] == False]
    
    comparison = {}
    for feat in ['base_rv', 'rec_rv', 'rv_gap', 'length_diff']:
        if feat in df.columns:
            s_mean = success_df[feat].mean()
            f_mean = failure_df[feat].mean()
            comparison[feat] = {
                "success_mean": float(s_mean) if not np.isnan(s_mean) else None,
                "failure_mean": float(f_mean) if not np.isnan(f_mean) else None,
                "difference": float(s_mean - f_mean) if not (np.isnan(s_mean) or np.isnan(f_mean)) else None,
            }
    
    result = {
        "experiment": "failure_analysis",
        "n_success": len(success_df),
        "n_failure": len(failure_df),
        "success_rate": float(len(success_df) / len(df)),
        "comparison": comparison,
        "finding": f"Success: {len(success_df)}/{len(df)} ({len(success_df)/len(df)*100:.1f}%). Differences: {comparison}"
    }
    
    print(f"\n✅ RESULT: {result['finding']}")
    
    return result


def run_experiment_6_dose_response(
    model,
    tokenizer,
    recursive_prompts: List[str],
    baseline_prompts: List[str],
    test_pairs: List[Tuple[str, str]],
    layer_idx: int,
    alphas: List[float],
    device: str,
    output_dir: Path,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """RUN 6: Dose-Response - Test multiple alpha values to find optimal."""
    print("\n" + "=" * 80)
    print("RUN 6: DOSE-RESPONSE CURVE")
    print("=" * 80)
    print(f"Alphas: {alphas}")
    
    # Compute steering vector
    print("Computing steering vector...")
    steering_vector = compute_steering_vector(model, tokenizer, recursive_prompts, baseline_prompts, layer_idx, device)
    
    # Test each alpha
    results = []
    
    for alpha in alphas:
        print(f"\nTesting alpha={alpha}...")
        scores = []
        passes = []
        
        # CRITICAL FIX: get_balanced_pairs returns (recursive, baseline)
        for rec_text, base_text in tqdm(test_pairs, desc=f"Alpha {alpha}"):
            # Safety gate: Verify base_text is actually a baseline prompt
            if "watch yourself" in base_text.lower() or "observe yourself" in base_text.lower() or "consciousness examining" in base_text.lower():
                raise ValueError(f"CRITICAL ERROR: Baseline prompt appears recursive! Text: {base_text[:100]}")
            
            patcher = SteeringVectorPatcher(model, steering_vector, alpha)
            patcher.register(layer_idx=layer_idx)
            
            try:
                text, entropy = _generate_with_steering(
                    model, tokenizer, base_text, max_new_tokens, temperature, device
                )
                score = score_behavior_strict(text, entropy)
                scores.append(score.final_score)
                passes.append(score.passed_gates)
            except Exception as e:
                scores.append(0.0)
                passes.append(False)
            finally:
                patcher.remove()
        
        transfer_rate = sum(s > 0.3 for s in scores) / len(scores)
        collapse_rate = 1.0 - (sum(passes) / len(passes))
        mean_score = np.mean(scores)
        
        results.append({
            "alpha": alpha,
            "transfer_rate": transfer_rate,
            "collapse_rate": collapse_rate,
            "mean_score": mean_score,
            "pass_rate": sum(passes) / len(passes),
        })
        
        print(f"  Transfer: {transfer_rate*100:.1f}%, Collapse: {collapse_rate*100:.1f}%, Score: {mean_score:.4f}")
    
    df = pd.DataFrame(results)
    csv_path = output_dir / "dose_response_results.csv"
    df.to_csv(csv_path, index=False)
    
    # Find optimal alpha (highest transfer with collapse < 50%)
    df_valid = df[df['collapse_rate'] < 0.5]
    if len(df_valid) > 0:
        best_row = df_valid.loc[df_valid['transfer_rate'].idxmax()]
        optimal_alpha = best_row['alpha']
        optimal_transfer = best_row['transfer_rate']
    else:
        best_row = df.loc[df['transfer_rate'].idxmax()]
        optimal_alpha = best_row['alpha']
        optimal_transfer = best_row['transfer_rate']
    
    result = {
        "experiment": "dose_response",
        "layer": layer_idx,
        "alphas": alphas,
        "results": results,
        "optimal_alpha": float(optimal_alpha),
        "optimal_transfer_rate": float(optimal_transfer),
        "finding": f"Optimal alpha: {optimal_alpha} ({optimal_transfer*100:.1f}% transfer)"
    }
    
    print(f"\n✅ RESULT: {result['finding']}")
    
    return result


def run_steering_analysis_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """Run all steering analysis experiments."""
    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-v0.1")
    layer_idx = params.get("layer", 27)
    n_prompts = params.get("n_prompts", 50)
    n_test_pairs = params.get("n_test_pairs", 20)
    seed = int(params.get("seed", 42))
    
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("STEERING VECTOR COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Layer: {layer_idx}")
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(model_name, device=device)
    model.eval()
    
    # Load prompts
    print("Loading prompts...")
    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    (run_dir / "prompt_bank_version.json").write_text(json.dumps({"version": bank_version}, indent=2) + "\n")
    
    recursive_prompts = []
    for group in ["L4_full", "L5_refined"]:
        prompts = loader.get_by_group(group)
        recursive_prompts.extend(prompts)
    
    baseline_prompts = []
    for group in ["baseline_math", "baseline_factual", "baseline_instructional"]:
        prompts = loader.get_by_group(group)
        baseline_prompts.extend(prompts)
    
    print(f"  Recursive: {len(recursive_prompts)}")
    print(f"  Baseline: {len(baseline_prompts)}")
    
    # Get test pairs
    test_pairs = loader.get_balanced_pairs(n_pairs=n_test_pairs, seed=seed)
    print(f"  Test pairs: {len(test_pairs)}")
    
    # Create output directory
    output_dir = run_dir / "steering_vectors"
    output_dir.mkdir(exist_ok=True)
    
    # Run all experiments
    all_results = {}
    
    # RUN 1: Stability
    print("\n" + "=" * 80)
    all_results["stability"] = run_experiment_1_stability(
        model, tokenizer, recursive_prompts, baseline_prompts, layer_idx, device, output_dir
    )
    
    # RUN 2: Layer Matrix
    extract_layers = params.get("extract_layers", [20, 24, 25, 26, 27, 28])
    apply_layers = params.get("apply_layers", [20, 24, 25, 26, 27, 28])
    alpha = params.get("alpha", 1.0)
    
    all_results["layer_matrix"] = run_experiment_2_layer_matrix(
        model, tokenizer, recursive_prompts, baseline_prompts, test_pairs,
        extract_layers, apply_layers, alpha, device, output_dir
    )
    
    # RUN 3: Head Decomposition
    all_results["head_decomposition"] = run_experiment_3_head_decomposition(
        model, tokenizer, recursive_prompts, baseline_prompts, test_pairs,
        layer_idx, alpha, device, output_dir
    )
    
    # RUN 4: Generalization
    all_results["generalization"] = run_experiment_4_generalization(
        model, tokenizer, recursive_prompts, baseline_prompts, test_pairs,
        layer_idx, alpha, device, output_dir
    )
    
    # RUN 5: Failure Analysis
    all_results["failure_analysis"] = run_experiment_5_failure_analysis(
        model, tokenizer, recursive_prompts, baseline_prompts, test_pairs,
        layer_idx, alpha, device, output_dir
    )
    
    # RUN 6: Dose-Response
    alphas = params.get("dose_response_alphas", [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    all_results["dose_response"] = run_experiment_6_dose_response(
        model, tokenizer, recursive_prompts, baseline_prompts, test_pairs,
        layer_idx, alphas, device, output_dir
    )
    
    # Save summary
    summary = {
        "experiment": "steering_analysis",
        "model_name": model_name,
        "layer": layer_idx,
        "results": {k: {"finding": v["finding"]} for k, v in all_results.items()},
        "all_results": all_results,
    }
    
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)
    for exp_name, exp_result in all_results.items():
        print(f"{exp_name.upper()}: {exp_result['finding']}")
    
    return ExperimentResult(summary=summary)

