"""
MLP Ablation Necessity Test: Zero out L0 MLP to test causal necessity.

Goal: Determine if L0 MLP is NECESSARY for recursive behavior.

Method:
1. Zero out L0 MLP output on recursive prompts
2. Measure R_V contraction (does it disappear?)
3. Measure recursive behavior (mode score, does it stop?)
4. Compare to baseline (no ablation)

Expected Results:
- If L0 MLP is NECESSARY:
  - R_V contraction disappears (R_V → 1.0)
  - Recursive behavior stops (mode score → baseline)
- If L0 MLP is NOT necessary:
  - R_V contraction persists
  - Recursive behavior continues
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
from src.utils.run_metadata import get_run_metadata, append_to_run_index, save_metadata


class MLPAblationHook:
    """Zero out MLP output at specified layer."""
    
    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
    
    def register(self):
        """Register forward hook to zero MLP output."""
        if self.handle is not None:
            raise RuntimeError("Hook already registered. Call remove() first.")
        
        mlp = self.model.model.layers[self.layer_idx].mlp
        
        def hook_fn(module, inp, out):
            """Zero out MLP output."""
            # Return zeros with same shape as output
            if isinstance(out, tuple):
                out_tensor = out[0]
                zeros = torch.zeros_like(out_tensor)
                return (zeros,) + out[1:]
            else:
                return torch.zeros_like(out)
        
        self.handle = mlp.register_forward_hook(hook_fn)
    
    def remove(self):
        """Remove the forward hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
    
    def __enter__(self):
        self.register()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


def run_mlp_ablation_necessity_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """Run MLP ablation necessity test."""
    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-v0.1")
    layer_idx = params.get("layer", 0)  # Default to L0
    n_pairs = params.get("n_pairs", 80)  # Protocol minimum
    window_size = params.get("window_size", 16)
    max_new_tokens = params.get("max_new_tokens", 200)
    seed = int(params.get("seed", 42))
    
    set_seed(seed)
    
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name, device=device)
    model.eval()
    
    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    (run_dir / "prompt_bank_version.json").write_text(json.dumps({"version": bank_version}, indent=2) + "\n")
    mode_metric = ModeScoreMetric(tokenizer, device)
    
    # Get prompt pairs WITH IDs (Industry-grade reproducibility)
    pairs_with_ids = loader.get_balanced_pairs_with_ids(n_pairs=n_pairs, seed=seed)
    pairs = [(rec_text, base_text) for _, _, rec_text, base_text in pairs_with_ids]
    recursive_prompts = [rec_text for _, _, rec_text, _ in pairs_with_ids]
    baseline_prompts = [base_text for _, _, _, base_text in pairs_with_ids]
    
    print(f"\n{'='*60}")
    print(f"MLP ABLATION NECESSITY TEST")
    print(f"{'='*60}")
    print(f"Layer: L{layer_idx}")
    print(f"Pairs: {len(pairs)}")
    print(f"Test: Zero out L{layer_idx} MLP on recursive prompts")
    print(f"{'='*60}\n")
    
    results = []
    
    for pair_idx, (rec_id, base_id, rec_text, base_text) in enumerate(tqdm(pairs_with_ids, desc="Testing pairs")):
        # Tokenize both prompts
        inputs_base = tokenizer(base_text, return_tensors="pt", add_special_tokens=False).to(device)
        inputs_rec = tokenizer(rec_text, return_tensors="pt", add_special_tokens=False).to(device)
        
        # BASELINE: Run model on both prompts to get logits for mode score
        with torch.no_grad():
            out_base = model(**inputs_base)
            out_rec_baseline = model(**inputs_rec)  # Recursive prompt WITHOUT ablation
        
        # BASELINE: Generate text without ablation
        with torch.no_grad():
            inputs_gen_baseline = tokenizer(rec_text, return_tensors="pt", add_special_tokens=False).to(device)
            outputs_baseline = model.generate(
                **inputs_gen_baseline,
                max_new_tokens=max_new_tokens,
                temperature=0.0,  # Deterministic
                do_sample=False,  # Deterministic
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text_baseline = tokenizer.decode(outputs_baseline[0], skip_special_tokens=True)
        
        # Compute baseline R_V on generated text
        rv_rec_baseline = compute_rv(model, tokenizer, generated_text_baseline, early=5, late=27, window=window_size, device=device)
        
        # Compute baseline mode score (compare recursive vs baseline logits)
        # Handle sequence length mismatch by truncating to shorter length
        try:
            rec_seq_len = out_rec_baseline.logits.shape[-2] if out_rec_baseline.logits.dim() == 2 else out_rec_baseline.logits.shape[1]
            base_seq_len = out_base.logits.shape[-2] if out_base.logits.dim() == 2 else out_base.logits.shape[1]
            min_len = min(rec_seq_len, base_seq_len)
            
            # Truncate both to same length
            rec_logits_trunc = out_rec_baseline.logits[:min_len] if out_rec_baseline.logits.dim() == 2 else out_rec_baseline.logits[:, :min_len]
            base_logits_trunc = out_base.logits[:min_len] if out_base.logits.dim() == 2 else out_base.logits[:, :min_len]
            
            mode_rec_baseline = mode_metric.compute_score(rec_logits_trunc, baseline_logits=base_logits_trunc)
        except Exception as e:
            print(f"  Error computing baseline mode score: {e}")
            mode_rec_baseline = float("nan")
        
        # ABLATION: Zero out L0 MLP on recursive prompt
        ablation_hook = MLPAblationHook(model, layer_idx)
        
        try:
            # Generate text with ablation and compute metrics INSIDE hook context
            with ablation_hook:
                with torch.no_grad():
                    # Generate text with ablation
                    inputs_gen = tokenizer(rec_text, return_tensors="pt", add_special_tokens=False).to(device)
                    outputs_ablated = model.generate(
                        **inputs_gen,
                        max_new_tokens=max_new_tokens,
                        temperature=0.0,  # Deterministic
                        do_sample=False,  # Deterministic
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generated_text = tokenizer.decode(outputs_ablated[0], skip_special_tokens=True)
                
                # Compute R_V with ablation (INSIDE hook context - compute_rv does forward pass)
                rv_rec_ablated = compute_rv(model, tokenizer, generated_text, early=5, late=27, window=window_size, device=device)
                
                # Compute mode score with ablation (forward pass for logits) - INSIDE hook context
                # Mode score needs ablated logits, so compute while hook is active
                with torch.no_grad():
                    out_rec_ablated = model(**inputs_rec)
                
                # Compute mode score INSIDE hook context (needs ablated logits)
                # Handle sequence length mismatch by truncating to shorter length
                try:
                    ablated_seq_len = out_rec_ablated.logits.shape[-2] if out_rec_ablated.logits.dim() == 2 else out_rec_ablated.logits.shape[1]
                    base_seq_len = out_base.logits.shape[-2] if out_base.logits.dim() == 2 else out_base.logits.shape[1]
                    min_len = min(ablated_seq_len, base_seq_len)
                    
                    # Truncate both to same length
                    ablated_logits_trunc = out_rec_ablated.logits[:min_len] if out_rec_ablated.logits.dim() == 2 else out_rec_ablated.logits[:, :min_len]
                    base_logits_trunc = out_base.logits[:min_len] if out_base.logits.dim() == 2 else out_base.logits[:, :min_len]
                    
                    mode_rec_ablated = mode_metric.compute_score(ablated_logits_trunc, baseline_logits=base_logits_trunc)
                except Exception as e:
                    print(f"  Error computing ablated mode score: {e}")
                    mode_rec_ablated = float("nan")
            
            # Compute coherence
            try:
                behavior_score = score_behavior_strict(generated_text)
                coherence = behavior_score.coherence_score
                recursion_score = behavior_score.recursion_score
            except Exception as e:
                print(f"  Error computing behavior score: {e}")
                coherence = 0.0
                recursion_score = 0.0
        
        except Exception as e:
            print(f"  Error during ablation at pair {pair_idx}: {e}")
            rv_rec_ablated = float("nan")
            mode_rec_ablated = float("nan")
            generated_text = ""
            coherence = 0.0
            recursion_score = 0.0
            # Still need to compute baseline for comparison (in error case)
            try:
                with ablation_hook:
                    with torch.no_grad():
                        out_rec_ablated = model(**inputs_rec)
                    # Handle sequence length mismatch
                    ablated_seq_len = out_rec_ablated.logits.shape[-2] if out_rec_ablated.logits.dim() == 2 else out_rec_ablated.logits.shape[1]
                    base_seq_len = out_base.logits.shape[-2] if out_base.logits.dim() == 2 else out_base.logits.shape[1]
                    min_len = min(ablated_seq_len, base_seq_len)
                    ablated_logits_trunc = out_rec_ablated.logits[:min_len] if out_rec_ablated.logits.dim() == 2 else out_rec_ablated.logits[:, :min_len]
                    base_logits_trunc = out_base.logits[:min_len] if out_base.logits.dim() == 2 else out_base.logits[:, :min_len]
                    mode_rec_ablated = mode_metric.compute_score(ablated_logits_trunc, baseline_logits=base_logits_trunc)
            except:
                pass
        
        # Compute deltas
        rv_delta = rv_rec_ablated - rv_rec_baseline
        mode_delta = mode_rec_ablated - mode_rec_baseline if not np.isnan(mode_rec_ablated) else float("nan")
        
        results.append({
            "pair_idx": pair_idx,
            "recursive_prompt_id": rec_id,
            "baseline_prompt_id": base_id,
            "recursive_text": rec_text,
            "baseline_text": base_text,
            "generated_text_baseline": generated_text_baseline,
            "generated_text_ablated": generated_text,
            "layer": layer_idx,
            "rv_baseline": rv_rec_baseline,
            "rv_ablated": rv_rec_ablated,
            "rv_delta": rv_delta,
            "mode_baseline": mode_rec_baseline,
            "mode_ablated": mode_rec_ablated,
            "mode_delta": mode_delta,
            "coherence": coherence,
            "recursion_score": recursion_score,
        })
    
    # Save results
    df = pd.DataFrame(results)
    csv_path = run_dir / "mlp_ablation_necessity.csv"
    df.to_csv(csv_path, index=False)
    
    # Statistical analysis
    rv_deltas = df["rv_delta"].dropna().values
    mode_deltas = df["mode_delta"].dropna().values
    
    # Helper: 95% CI for mean
    def compute_ci_95(arr):
        if len(arr) < 2:
            return (float("nan"), float("nan"))
        sem = stats.sem(arr)
        ci = stats.t.interval(0.95, len(arr) - 1, loc=np.mean(arr), scale=sem)
        return (float(ci[0]), float(ci[1]))

    # Helper: Cohen's d with pooled std
    def compute_cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return float("nan")
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std < 1e-10:
            return 0.0
        return float((np.mean(group1) - np.mean(group2)) / pooled_std)

    rv_baselines = df["rv_baseline"].dropna().values
    rv_ablateds = df["rv_ablated"].dropna().values

    # One-sample t-test: Is delta significantly different from zero?
    rv_stat = None
    rv_pvalue = None
    rv_significant = None
    rv_ci_95 = (float("nan"), float("nan"))
    if len(rv_deltas) >= 3:
        t_stat, p_val = stats.ttest_1samp(rv_deltas, 0.0)
        rv_stat = float(t_stat)
        rv_pvalue = float(p_val)
        rv_significant = bool(p_val < 0.01)  # Bonferroni correction if multiple tests
        rv_ci_95 = compute_ci_95(rv_deltas)

    mode_stat = None
    mode_pvalue = None
    mode_significant = None
    mode_ci_95 = (float("nan"), float("nan"))
    if len(mode_deltas) >= 3:
        t_stat, p_val = stats.ttest_1samp(mode_deltas, 0.0)
        mode_stat = float(t_stat)
        mode_pvalue = float(p_val)
        mode_significant = bool(p_val < 0.01)
        mode_ci_95 = compute_ci_95(mode_deltas)

    # Effect sizes - use proper Cohen's d with pooled std
    rv_cohens_d = compute_cohens_d(rv_baselines, rv_ablateds) if len(rv_baselines) >= 2 and len(rv_ablateds) >= 2 else None
    mode_baselines = df["mode_baseline"].dropna().values
    mode_ablateds = df["mode_ablated"].dropna().values
    mode_cohens_d = compute_cohens_d(mode_baselines, mode_ablateds) if len(mode_baselines) >= 2 and len(mode_ablateds) >= 2 else None
    
    # Get standardized metadata
    metadata = get_run_metadata(
        cfg,
        prompt_ids=pairs_with_ids,
        eval_window=window_size,
        intervention_scope="all_tokens",  # MLP ablation affects all tokens
        behavior_metric="mode_score_m",
    )
    
    # Summary statistics (Industry-grade metric contract)
    summary = {
        "experiment": "mlp_ablation_necessity",
        "layer": layer_idx,
        "n_pairs": len(pairs),
        # PRIMARY: Mode Score M (renamed from mode_baseline_mean)
        "mode_score_m": float(df["mode_baseline"].mean()) if df["mode_baseline"].notna().any() else None,
        "mode_score_m_delta": float(df["mode_delta"].mean()) if df["mode_delta"].notna().any() else None,
        "mode_score_m_ablated": float(df["mode_ablated"].mean()) if df["mode_ablated"].notna().any() else None,
        "mode_t_statistic": mode_stat,
        "mode_pvalue": mode_pvalue,
        "mode_significant": mode_significant,
        "mode_cohens_d": mode_cohens_d,
        "mode_ci_95": mode_ci_95,
        # SECONDARY: R_V signature
        "rv": float(df["rv_baseline"].mean()),
        "rv_baseline_mean": float(df["rv_baseline"].mean()),
        "rv_baseline_std": float(df["rv_baseline"].std()),
        "rv_ablated_mean": float(df["rv_ablated"].mean()),
        "rv_ablated_std": float(df["rv_ablated"].std()),
        "rv_delta_mean": float(df["rv_delta"].mean()),
        "rv_delta_std": float(df["rv_delta"].std()),
        "rv_t_statistic": rv_stat,
        "rv_pvalue": rv_pvalue,
        "rv_significant": rv_significant,
        "rv_cohens_d": rv_cohens_d,
        "rv_delta_ci_95": rv_ci_95,
        # Standardized metadata
        "eval_window": window_size,
        "intervention_scope": "all_tokens",
        "behavior_metric": "mode_score_m",
        # Legacy metrics (secondary/non-comparable)
        "coherence_mean": float(df["coherence"].mean()),
        "recursion_score_mean": float(df["recursion_score"].mean()),
        # Merge metadata
        **metadata,
    }
    
    # Verdict
    # CORRECT LOGIC:
    # - If delta > 0.1: ablation REMOVES contraction (rv_ablated > rv_baseline) → Layer IS NECESSARY
    # - If delta < -0.1: ablation INCREASES contraction (rv_ablated < rv_baseline) → Layer is NOT necessary
    # - If delta ≈ 0: ablation has no effect → Layer is NOT necessary
    rv_delta_mean = summary["rv_delta_mean"]
    if rv_significant and rv_delta_mean > 0.1:
        verdict = f"L{layer_idx} MLP IS NECESSARY - ablation removes contraction (delta: +{rv_delta_mean:.3f})"
    elif rv_significant and rv_delta_mean < -0.1:
        verdict = f"L{layer_idx} MLP is NOT necessary - ablation increases contraction (delta: {rv_delta_mean:.3f})"
    else:
        verdict = f"L{layer_idx} MLP has minimal effect - inconclusive (delta: {rv_delta_mean:.3f})"
    
    summary["verdict"] = verdict
    
    # Save metadata separately
    save_metadata(run_dir, metadata)
    
    # Save summary
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Append to run index
    append_to_run_index(run_dir, summary)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"R_V baseline: {summary['rv_baseline_mean']:.4f} ± {summary['rv_baseline_std']:.4f}")
    print(f"R_V ablated:  {summary['rv_ablated_mean']:.4f} ± {summary['rv_ablated_std']:.4f}")
    print(f"R_V delta:    {summary['rv_delta_mean']:.4f} ± {summary['rv_delta_std']:.4f}")
    if rv_significant:
        print(f"  p-value: {rv_pvalue:.4f} (SIGNIFICANT)")
    else:
        print(f"  p-value: {rv_pvalue:.4f} (not significant)")
    print(f"\nVERDICT: {verdict}")
    print(f"\n✅ Results saved to: {csv_path}")
    print(f"✅ Summary saved to: {summary_path}")
    
    return ExperimentResult(summary=summary)
