"""
Random Direction Control Test: Verify steering direction specificity.

Goal: Prove that L2 MLP steering effect is specific to our computed direction,
not just "any perturbation at L2 causes R_V expansion."
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from scipy import stats as scipy_stats

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.metrics.behavior_strict import score_behavior_strict
from src.metrics.mode_score import ModeScoreMetric
from src.metrics.rv import compute_rv
from src.pipelines.archive.mlp_steering_sweep import (
    MLPSteeringPatcher,
    compute_steering_vector,
    extract_mlp_outputs,
)
from src.pipelines.registry import ExperimentResult


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled standard deviation."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    n1, n2 = len(a), len(b)
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _ci_95(arr: np.ndarray) -> tuple:
    """95% confidence interval for mean."""
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return (float("nan"), float("nan"))
    sem = scipy_stats.sem(arr)
    ci = scipy_stats.t.interval(0.95, len(arr) - 1, loc=np.mean(arr), scale=sem)
    return (float(ci[0]), float(ci[1]))


def generate_random_vectors(n: int, dim: int, device: str, dtype: torch.dtype = None) -> List[torch.Tensor]:
    """Generate n random unit vectors."""
    random_vecs = []
    for i in range(n):
        random_vec = torch.randn(dim, device=device, dtype=dtype)
        random_vec = random_vec / random_vec.norm()
        random_vecs.append(random_vec)
    return random_vecs


def generate_orthogonal_vector(steering_vec: torch.Tensor, device: str) -> torch.Tensor:
    """Generate a vector orthogonal to steering_vec."""
    dtype = steering_vec.dtype
    # Start with random vector
    ortho_vec = torch.randn(steering_vec.shape[0], device=device, dtype=dtype)
    
    # Remove component parallel to steering_vec
    parallel_component = (ortho_vec @ steering_vec) * steering_vec
    ortho_vec = ortho_vec - parallel_component
    
    # Normalize
    norm = ortho_vec.norm()
    if norm < 1e-8:
        # If orthogonal vector is too small, try again
        return generate_orthogonal_vector(steering_vec, device)
    ortho_vec = ortho_vec / norm
    
    # Verify orthogonality (allow for floating point precision)
    dot_product = (ortho_vec @ steering_vec).item()
    if abs(dot_product) > 0.01:
        # If not orthogonal enough, try again (up to 10 times)
        for attempt in range(10):
            ortho_vec = torch.randn(steering_vec.shape[0], device=device, dtype=dtype)
            parallel_component = (ortho_vec @ steering_vec) * steering_vec
            ortho_vec = ortho_vec - parallel_component
            norm = ortho_vec.norm()
            if norm > 1e-8:
                ortho_vec = ortho_vec / norm
                dot_product = (ortho_vec @ steering_vec).item()
                if abs(dot_product) <= 0.01:
                    break
        if abs(dot_product) > 0.01:
            print(f"  Warning: Orthogonal vector dot product: {dot_product:.6e} (target: <0.01)")
    
    return ortho_vec


def run_random_direction_control_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """Run random direction control experiment."""
    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-v0.1")
    layer_idx = params.get("layer", 2)
    alpha_param = params.get("alpha", 2.0)
    # Support both single alpha and list of alphas
    if isinstance(alpha_param, list):
        alphas = alpha_param
    else:
        alphas = [alpha_param]
    n_random = params.get("n_random", 5)
    n_pairs = params.get("n_pairs", 10)
    include_orthogonal = params.get("include_orthogonal", True)
    window_size = params.get("window_size", 16)
    max_new_tokens = params.get("max_new_tokens", 200)
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
    mode_metric = ModeScoreMetric(tokenizer, device)
    
    # Get prompt pairs
    pairs = loader.get_balanced_pairs(n_pairs=n_pairs, seed=seed)
    recursive_prompts = [rec for rec, _ in pairs]
    baseline_prompts = [base for _, base in pairs]
    
    print(f"Testing {len(pairs)} pairs at layer {layer_idx} with alphas={alphas}")
    
    # Extract MLP outputs and compute TRUE steering vector
    print(f"\nExtracting MLP outputs at L{layer_idx}...")
    recursive_mlps = extract_mlp_outputs(model, tokenizer, recursive_prompts, layer_idx, device, window_size)
    baseline_mlps = extract_mlp_outputs(model, tokenizer, baseline_prompts, layer_idx, device, window_size)
    
    if recursive_mlps is None or baseline_mlps is None:
        raise RuntimeError(f"Could not extract MLP outputs at L{layer_idx}")
    
    true_steering_vec = compute_steering_vector(recursive_mlps, baseline_mlps)
    steering_dim = true_steering_vec.shape[0]
    steering_norm = true_steering_vec.norm().item()
    steering_dtype = true_steering_vec.dtype
    
    print(f"  TRUE steering vector norm: {steering_norm:.6f}")
    print(f"  Steering vector dimension: {steering_dim}")
    print(f"  Steering vector dtype: {steering_dtype}")
    
    # Generate control vectors
    print(f"\nGenerating {n_random} random control vectors...")
    random_vecs = generate_random_vectors(n_random, steering_dim, device, dtype=steering_dtype)
    for i, rv in enumerate(random_vecs):
        print(f"  Random {i+1} norm: {rv.norm().item():.6f}")
    
    # Generate orthogonal vector
    control_vectors = {"true_steering": true_steering_vec}
    
    if include_orthogonal:
        print(f"\nGenerating orthogonal control vector...")
        ortho_vec = generate_orthogonal_vector(true_steering_vec, device)
        control_vectors["orthogonal"] = ortho_vec
        dot_check = (ortho_vec @ true_steering_vec).item()
        print(f"  Orthogonal vector norm: {ortho_vec.norm().item():.6f}")
        print(f"  Dot product with steering (should be ~0): {dot_check:.6e}")
    
    for i, rv in enumerate(random_vecs):
        control_vectors[f"random_{i+1}"] = rv
    
    # Test each control vector with each alpha
    results = []
    
    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"ALPHA = {alpha}")
        print(f"{'='*60}")
        
        for condition_name, steering_vec in control_vectors.items():
            print(f"\n{'='*60}")
            print(f"Testing condition: {condition_name} (α={alpha})")
            print(f"{'='*60}")
            
            condition_results = []
            
            for pair_idx, (rec_text, base_text) in enumerate(tqdm(pairs, desc=f"{condition_name} α={alpha}")):
                # Baseline metrics (no steering)
                inputs_base = tokenizer(base_text, return_tensors="pt", add_special_tokens=False).to(device)
                
                with torch.no_grad():
                    out_base = model(**inputs_base)
                
                # Compute baseline R_V
                rv_base = compute_rv(model, tokenizer, base_text, early=5, late=27, window=window_size, device=device)
                
                # Compute baseline mode score
                mode_base = mode_metric.compute_score(out_base.logits, baseline_logits=out_base.logits)
                
                # Generate with steering
                patcher = MLPSteeringPatcher(model, steering_vec, alpha=alpha)
                patcher.register(layer_idx)
                
                try:
                    with torch.no_grad():
                        inputs_gen = tokenizer(base_text, return_tensors="pt", add_special_tokens=False).to(device)
                        try:
                            outputs_steered = model.generate(
                                **inputs_gen,
                                max_new_tokens=max_new_tokens,
                                temperature=0.7,
                                do_sample=True,
                                pad_token_id=tokenizer.eos_token_id
                            )
                        except RuntimeError as e:
                            print(f"  Error during generation at pair {pair_idx}: {e}")
                            continue
                    
                    generated_text = tokenizer.decode(outputs_steered[0], skip_special_tokens=True)
                    
                    # Compute steered R_V
                    try:
                        rv_steered = compute_rv(model, tokenizer, base_text, early=5, late=27, window=window_size, device=device)
                    except Exception as e:
                        print(f"  Error computing R_V: {e}")
                        rv_steered = float("nan")
                    
                    # Compute mode score
                    try:
                        with torch.no_grad():
                            max_len = 512
                            inputs_gen_full = tokenizer(
                                generated_text[:max_len*4], 
                                return_tensors="pt", 
                                add_special_tokens=False, 
                                max_length=max_len, 
                                truncation=True
                            ).to(device)
                            out_steered = model(**inputs_gen_full)
                        
                        mode_steered = mode_metric.compute_score(out_steered.logits, baseline_logits=out_base.logits)
                    except Exception as e:
                        print(f"  Error computing mode score: {e}")
                        mode_steered = float("nan")
                    
                    # Compute coherence
                    try:
                        behavior_score = score_behavior_strict(generated_text)
                        coherence = behavior_score.coherence_score
                    except Exception as e:
                        print(f"  Error computing coherence: {e}")
                        coherence = 0.0
                    
                finally:
                    patcher.remove()
                
                condition_results.append({
                    "condition": condition_name,
                    "pair_idx": pair_idx,
                    "alpha": alpha,
                    "layer": layer_idx,
                "recursive_text": rec_text,
                "baseline_text": base_text,
                "generated_text": generated_text,
                "rv_baseline": rv_base,
                "rv_steered": rv_steered,
                "rv_delta": rv_steered - rv_base,
                "mode_baseline": mode_base,
                "mode_steered": mode_steered,
                "mode_delta": mode_steered - mode_base if not np.isnan(mode_steered) else float("nan"),
                    "coherence": coherence,
                })
            
            # Extend results with this condition's data
            results.extend(condition_results)
            print(f"  Added {len(condition_results)} results for {condition_name} (α={alpha}). Total results: {len(results)}")
    
    # Save results
    print(f"\n{'='*60}")
    print(f"SAVING RESULTS: Total {len(results)} rows")
    print(f"{'='*60}")
    df = pd.DataFrame(results)
    csv_path = run_dir / "random_direction_control.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved {len(df)} rows to CSV")
    print(f"   Conditions: {sorted(df['condition'].unique())}")
    print(f"   Alphas: {sorted(df['alpha'].unique())}")
    
    # Summary statistics by condition and alpha
    summary = df.groupby(["condition", "alpha"]).agg({
        "rv_delta": ["mean", "std"],
        "mode_delta": ["mean", "std"],
        "coherence": ["mean", "std"]
    }).round(4)
    
    print(f"\n{'='*60}")
    print("SUMMARY BY CONDITION")
    print(f"{'='*60}")
    print(summary)
    
    # Create comparison table
    comparison = []
    for alpha_val in sorted(df["alpha"].unique()):
        alpha_df = df[df["alpha"] == alpha_val]
        for condition in alpha_df["condition"].unique():
            cond_df = alpha_df[alpha_df["condition"] == condition]
            comparison.append({
                "condition": condition,
                "alpha": alpha_val,
                "layer": layer_idx,
                "rv_delta_mean": float(cond_df["rv_delta"].mean()),
                "rv_delta_std": float(cond_df["rv_delta"].std()),
                "mode_delta_mean": float(cond_df["mode_delta"].mean()) if cond_df["mode_delta"].notna().any() else None,
                "mode_delta_std": float(cond_df["mode_delta"].std()) if cond_df["mode_delta"].notna().any() else None,
                "coherence_mean": float(cond_df["coherence"].mean()),
                "coherence_std": float(cond_df["coherence"].std()),
            })
    
    comparison_df = pd.DataFrame(comparison)
    comparison_path = run_dir / "comparison_table.csv"
    comparison_df.to_csv(comparison_path, index=False)
    
    # Analysis by alpha
    analysis_by_alpha = {}
    for alpha_val in sorted(df["alpha"].unique()):
        alpha_df = df[df["alpha"] == alpha_val]
        true_df = alpha_df[alpha_df["condition"] == "true_steering"]
        random_dfs = [alpha_df[alpha_df["condition"] == f"random_{i+1}"] for i in range(n_random)]

        true_rv_vals = true_df["rv_delta"].dropna().values
        true_rv_mean = np.mean(true_rv_vals) if len(true_rv_vals) > 0 else float("nan")
        random_rv_means = [rdf["rv_delta"].mean() for rdf in random_dfs]
        random_rv_mean = np.mean(random_rv_means)

        # Combine all random condition values for comparison
        all_random_rv_vals = np.concatenate([rdf["rv_delta"].dropna().values for rdf in random_dfs if len(rdf["rv_delta"].dropna()) > 0]) if random_dfs else np.array([])

        true_mode_mean = true_df["mode_delta"].mean() if true_df["mode_delta"].notna().any() else None
        random_mode_means = [rdf["mode_delta"].mean() if rdf["mode_delta"].notna().any() else None for rdf in random_dfs]
        random_mode_mean = np.mean([m for m in random_mode_means if m is not None]) if any(m is not None for m in random_mode_means) else None

        # Compute t-test between true steering and pooled random
        ttest_result = {"t": float("nan"), "p": float("nan")}
        if len(true_rv_vals) >= 2 and len(all_random_rv_vals) >= 2:
            t_stat, p_val = scipy_stats.ttest_ind(true_rv_vals, all_random_rv_vals, equal_var=False)
            ttest_result = {"t": float(t_stat), "p": float(p_val)}

        # Compute Cohen's d between true steering and pooled random
        cohens_d_val = _cohens_d(true_rv_vals, all_random_rv_vals) if len(true_rv_vals) >= 2 and len(all_random_rv_vals) >= 2 else float("nan")

        # 95% CI for true steering rv_delta
        rv_ci_95 = _ci_95(true_rv_vals) if len(true_rv_vals) >= 2 else (float("nan"), float("nan"))

        analysis_by_alpha[str(alpha_val)] = {
            "true_steering_rv_delta": float(true_rv_mean),
            "true_steering_rv_ci_95": rv_ci_95,
            "random_avg_rv_delta": float(random_rv_mean),
            "rv_ratio": float(true_rv_mean / random_rv_mean) if random_rv_mean != 0 else None,
            "rv_cohens_d": cohens_d_val,
            "rv_ttest": ttest_result,
            "true_steering_mode_delta": float(true_mode_mean) if true_mode_mean is not None else None,
            "random_avg_mode_delta": float(random_mode_mean) if random_mode_mean is not None else None,
            "mode_ratio": float(true_mode_mean / random_mode_mean) if (true_mode_mean is not None and random_mode_mean is not None and random_mode_mean != 0) else None,
            "verdict": "REAL" if (true_rv_mean > 3 * abs(random_rv_mean)) else "ARTIFACT",
        }
    
    analysis = {
        "experiment": "random_direction_control",
        "n_pairs": len(pairs),
        "layer": layer_idx,
        "alphas": alphas,
        "n_random": n_random,
        "include_orthogonal": include_orthogonal,
        "steering_vector_norm": float(steering_norm),
        "steering_vector_dim": int(steering_dim),
        "analysis_by_alpha": analysis_by_alpha,
    }
    
    summary_json = {
        **analysis,
        "comparison_table": comparison_df.to_dict("records"),
    }
    
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    
    print(f"\n{'='*60}")
    print("ANALYSIS BY ALPHA")
    print(f"{'='*60}")
    for alpha_val in sorted(df["alpha"].unique()):
        alpha_analysis = analysis_by_alpha[str(alpha_val)]
        print(f"\nAlpha {alpha_val}:")
        print(f"  TRUE steering R_V delta: {alpha_analysis['true_steering_rv_delta']:.4f}")
        print(f"  Random average R_V delta: {alpha_analysis['random_avg_rv_delta']:.4f}")
        if alpha_analysis['rv_ratio'] is not None:
            print(f"  Ratio: {alpha_analysis['rv_ratio']:.2f}x")
        print(f"  VERDICT: {alpha_analysis['verdict']}")
    print(f"\n✅ Results saved to: {csv_path}")
    print(f"✅ Comparison table saved to: {comparison_path}")
    print(f"✅ Summary saved to: {summary_path}")
    
    return ExperimentResult(summary=summary_json)

