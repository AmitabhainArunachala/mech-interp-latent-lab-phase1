"""
MLP Steering Sweep: Test MLP output steering at multiple layers.

Goal: Find the optimal layer for MLP steering to transfer recursive mode.

Method:
1. Extract MLP outputs from recursive vs baseline prompts
2. Compute steering vector: mean(recursive_MLP) - mean(baseline_MLP)
3. Apply steering at layers [0, 6, 12, 18, 24]
4. Measure: R_V (geometry), Mode Score (behavior), Coherence (quality)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.metrics.behavior_strict import score_behavior_strict
from src.metrics.mode_score import ModeScoreMetric
from src.metrics.rv import compute_rv
from src.pipelines.registry import ExperimentResult


class MLPSteeringPatcher:
    """Patches MLP output by adding a steering vector."""
    
    def __init__(self, model, steering_vector: torch.Tensor, alpha: float):
        self.model = model
        self.steering_vector = steering_vector.detach().to(model.device)
        self.alpha = alpha
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.layer_idx: Optional[int] = None
    
    def register(self, layer_idx: int):
        """Register forward hook at specified layer's MLP output."""
        if self.handle is not None:
            raise RuntimeError("Patcher already registered. Call remove() first.")
        
        self.layer_idx = layer_idx
        mlp = self.model.model.layers[layer_idx].mlp
        
        def hook_fn(module, inp, out):
            """Add steering vector to MLP output."""
            # Handle tuple output (some models return tuple)
            if isinstance(out, tuple):
                out_tensor = out[0]
            else:
                out_tensor = out
            
            # out_tensor shape: (batch, seq_len, hidden_dim)
            # steering_vector shape: (hidden_dim,)
            batch, seq_len, hidden_dim = out_tensor.shape
            
            # Ensure steering vector matches hidden_dim
            if self.steering_vector.shape[0] != hidden_dim:
                # This shouldn't happen, but handle gracefully
                return out
            
            # Broadcast steering vector: (hidden_dim,) -> (1, 1, hidden_dim) -> (batch, seq_len, hidden_dim)
            steering_broadcast = self.steering_vector.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1)
            steered = out_tensor + self.alpha * steering_broadcast
            
            # Return in same format as input
            if isinstance(out, tuple):
                return (steered,) + out[1:]
            return steered
        
        self.handle = mlp.register_forward_hook(hook_fn)
    
    def remove(self):
        """Remove the forward hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
            self.layer_idx = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


def extract_mlp_outputs(model, tokenizer, prompts: List[str], layer_idx: int, device: str, window_size: int = 16) -> torch.Tensor:
    """
    Extract MLP outputs from prompts at specified layer.
    
    Returns:
        Stacked MLP outputs: (n_prompts, window_size, hidden_dim)
    """
    mlp_outputs = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        seq_len = inputs["input_ids"].shape[1]
        
        if seq_len < window_size:
            continue
        
        mlp_out = None
        
        def hook_fn(module, inp, out):
            nonlocal mlp_out
            # out shape: (batch, seq_len, hidden_dim)
            if isinstance(out, tuple):
                out = out[0]
            mlp_out = out.detach().clone()
        
        handle = model.model.layers[layer_idx].mlp.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                _ = model(**inputs)
            
            # Extract last window_size tokens
            if mlp_out is not None:
                actual_seq_len = mlp_out.shape[1]
                if actual_seq_len >= window_size:
                    mlp_window = mlp_out[0, -window_size:, :]  # (window_size, hidden_dim)
                    mlp_outputs.append(mlp_window)
                # Skip prompts shorter than window_size
        finally:
            handle.remove()
    
    if not mlp_outputs:
        return None
    
    return torch.stack(mlp_outputs)  # (n_prompts, window_size, hidden_dim)


def compute_steering_vector(recursive_mlps: torch.Tensor, baseline_mlps: torch.Tensor) -> torch.Tensor:
    """
    Compute steering vector as mean difference.
    
    Args:
        recursive_mlps: (n_rec, window_size, hidden_dim)
        baseline_mlps: (n_base, window_size, hidden_dim)
    
    Returns:
        Steering vector: (hidden_dim,)
    """
    # Mean over prompts and window: (hidden_dim,)
    recursive_mean = recursive_mlps.mean(dim=(0, 1))  # (hidden_dim,)
    baseline_mean = baseline_mlps.mean(dim=(0, 1))  # (hidden_dim,)
    
    steering_vec = recursive_mean - baseline_mean
    # Normalize to unit vector
    norm = steering_vec.norm()
    if norm > 1e-8:
        steering_vec = steering_vec / norm
    
    return steering_vec


def run_mlp_steering_sweep_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """Run MLP steering sweep experiment."""
    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-v0.1")
    n_pairs = params.get("n_pairs", 80)  # Protocol minimum: 80 pairs for statistical power
    layers = params.get("layers", [0, 6, 12, 18, 24])
    # Support both "alpha" (single) and "alphas" (list)
    alpha_param = params.get("alphas", params.get("alpha", 2.0))
    if isinstance(alpha_param, list):
        alphas = alpha_param
    else:
        alphas = [alpha_param]
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
    
    print(f"Testing {len(pairs)} pairs at layers {layers}")
    
    results = []
    
    # Extract MLP outputs and compute steering vectors for each layer
    steering_vectors = {}
    
    for layer_idx in layers:
        print(f"\nExtracting MLP outputs at L{layer_idx}...")
        recursive_mlps = extract_mlp_outputs(model, tokenizer, recursive_prompts, layer_idx, device, window_size)
        baseline_mlps = extract_mlp_outputs(model, tokenizer, baseline_prompts, layer_idx, device, window_size)
        
        if recursive_mlps is None or baseline_mlps is None:
            print(f"  Warning: Could not extract MLP outputs at L{layer_idx}")
            continue
        
        steering_vec = compute_steering_vector(recursive_mlps, baseline_mlps)
        steering_vectors[layer_idx] = steering_vec
        print(f"  Steering vector norm: {steering_vec.norm().item():.4f}")
    
    # Test steering at each layer and alpha combination
    for layer_idx in layers:
        if layer_idx not in steering_vectors:
            continue
        
        steering_vec = steering_vectors[layer_idx]
        
        for alpha in alphas:
            print(f"\nTesting steering at L{layer_idx} with alpha={alpha}...")
            
            for pair_idx, (rec_text, base_text) in enumerate(tqdm(pairs, desc=f"L{layer_idx} α={alpha}")):
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
                                temperature=0.0,  # Deterministic generation
                                do_sample=False,  # Deterministic generation
                                pad_token_id=tokenizer.eos_token_id
                            )
                        except RuntimeError as e:
                            print(f"  Error during generation at pair {pair_idx}: {e}")
                            continue
                    
                    generated_text = tokenizer.decode(outputs_steered[0], skip_special_tokens=True)
                    
                    # Compute steered R_V on FULL generated text (prompt + generated tokens)
                    # This measures geometry of the actual steered output
                    try:
                        # Use full generated text, but ensure it's long enough for window_size
                        if len(outputs_steered[0]) >= window_size:
                            rv_steered = compute_rv(model, tokenizer, generated_text, early=5, late=27, window=window_size, device=device)
                        else:
                            # If too short, fall back to measuring on prompt portion
                            rv_steered = compute_rv(model, tokenizer, base_text, early=5, late=27, window=window_size, device=device)
                    except Exception as e:
                        print(f"  Error computing R_V: {e}")
                        rv_steered = float("nan")
                    
                    # Compute mode score from generation - use a forward pass on generated text
                    try:
                        with torch.no_grad():
                            # Truncate generated text if too long to avoid memory issues
                            max_len = 512
                            # Ensure we have enough tokens for comparison
                            inputs_gen_full = tokenizer(
                                generated_text[:max_len*4], 
                                return_tensors="pt", 
                                add_special_tokens=False, 
                                max_length=max_len, 
                                truncation=True
                            ).to(device)
                            
                            # Ensure baseline and steered have same sequence length for mode score
                            base_seq_len = inputs_base.input_ids.shape[1]
                            gen_seq_len = inputs_gen_full.input_ids.shape[1]
                            
                            if gen_seq_len < base_seq_len:
                                # Pad generated to match baseline length
                                pad_length = base_seq_len - gen_seq_len
                                pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
                                pad_tokens = torch.full((1, pad_length), pad_token_id, device=device)
                                inputs_gen_full.input_ids = torch.cat([inputs_gen_full.input_ids, pad_tokens], dim=1)
                            
                            out_steered = model(**inputs_gen_full)
                            
                            # Truncate both to same length for comparison
                            min_len = min(out_base.logits.shape[1], out_steered.logits.shape[1])
                            mode_steered = mode_metric.compute_score(
                                out_steered.logits[:, :min_len, :], 
                                baseline_logits=out_base.logits[:, :min_len, :]
                            )
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
                
                results.append({
                    "layer": layer_idx,
                    "pair_idx": pair_idx,
                    "alpha": alpha,
                    "recursive_text": rec_text,
                    "baseline_text": base_text,
                    "generated_text": generated_text,
                    "rv_baseline": rv_base,
                    "rv_steered": rv_steered,
                    "rv_delta": rv_steered - rv_base,
                    "mode_baseline": mode_base,
                    "mode_steered": mode_steered,
                    "mode_delta": mode_steered - mode_base,
                    "coherence": coherence
                })
    
    # Save results
    df = pd.DataFrame(results)
    csv_path = run_dir / "mlp_steering_sweep.csv"
    df.to_csv(csv_path, index=False)
    
    # Summary statistics with statistical testing
    summary_json = {
        "experiment": "mlp_steering_sweep",
        "n_pairs": len(pairs),
        "layers": layers,
        "alphas": alphas,
        "summary_by_layer_alpha": {},
        "statistical_tests": {}
    }
    
    # Statistical testing: t-tests with Bonferroni correction
    n_comparisons = len(layers) * len(alphas)
    bonferroni_alpha = 0.01 / n_comparisons if n_comparisons > 0 else 0.01
    summary_json["bonferroni_alpha"] = bonferroni_alpha
    summary_json["n_comparisons"] = n_comparisons
    
    # Create summary by layer and alpha with statistical tests
    for layer in layers:
        if layer not in df["layer"].values:
            continue
        layer_df = df[df["layer"] == layer]
        summary_json["summary_by_layer_alpha"][str(layer)] = {}
        
        for alpha in alphas:
            alpha_df = layer_df[layer_df["alpha"] == alpha]
            if len(alpha_df) > 0:
                # Extract valid (non-NaN) deltas for statistical testing
                rv_deltas = alpha_df["rv_delta"].dropna().values
                mode_deltas = alpha_df["mode_delta"].dropna().values
                
                # Statistical tests: one-sample t-test against zero (null: no effect)
                rv_stat = None
                rv_pvalue = None
                rv_significant = None
                if len(rv_deltas) >= 3:  # Minimum for t-test
                    t_stat, p_val = stats.ttest_1samp(rv_deltas, 0.0)
                    rv_stat = float(t_stat)
                    rv_pvalue = float(p_val)
                    rv_significant = p_val < bonferroni_alpha
                
                mode_stat = None
                mode_pvalue = None
                mode_significant = None
                if len(mode_deltas) >= 3:
                    t_stat, p_val = stats.ttest_1samp(mode_deltas, 0.0)
                    mode_stat = float(t_stat)
                    mode_pvalue = float(p_val)
                    mode_significant = p_val < bonferroni_alpha
                
                # Effect size: Cohen's d
                rv_cohens_d = None
                if len(rv_deltas) >= 2:
                    rv_cohens_d = float(np.mean(rv_deltas) / np.std(rv_deltas)) if np.std(rv_deltas) > 0 else None
                
                mode_cohens_d = None
                if len(mode_deltas) >= 2:
                    mode_cohens_d = float(np.mean(mode_deltas) / np.std(mode_deltas)) if np.std(mode_deltas) > 0 else None
                
                summary_json["summary_by_layer_alpha"][str(layer)][str(alpha)] = {
                    "n_samples": len(alpha_df),
                    "rv_delta_mean": float(alpha_df["rv_delta"].mean()),
                    "rv_delta_std": float(alpha_df["rv_delta"].std()),
                    "rv_t_statistic": rv_stat,
                    "rv_pvalue": rv_pvalue,
                    "rv_significant": rv_significant,
                    "rv_cohens_d": rv_cohens_d,
                    "mode_delta_mean": float(alpha_df["mode_delta"].mean()) if alpha_df["mode_delta"].notna().any() else None,
                    "mode_delta_std": float(alpha_df["mode_delta"].std()) if alpha_df["mode_delta"].notna().any() else None,
                    "mode_t_statistic": mode_stat,
                    "mode_pvalue": mode_pvalue,
                    "mode_significant": mode_significant,
                    "mode_cohens_d": mode_cohens_d,
                    "coherence_mean": float(alpha_df["coherence"].mean()),
                    "coherence_std": float(alpha_df["coherence"].std()),
                }
                
                # Store in statistical_tests section for easy lookup
                test_key = f"L{layer}_alpha{alpha}"
                summary_json["statistical_tests"][test_key] = {
                    "rv_significant": rv_significant,
                    "rv_pvalue": rv_pvalue,
                    "rv_cohens_d": rv_cohens_d,
                    "mode_significant": mode_significant,
                    "mode_pvalue": mode_pvalue,
                    "mode_cohens_d": mode_cohens_d,
                }
    
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    
    print(f"\n✅ Results saved to: {csv_path}")
    print(f"✅ Summary saved to: {summary_path}")
    
    return ExperimentResult(summary=summary_json)

