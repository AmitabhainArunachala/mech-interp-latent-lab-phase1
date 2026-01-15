"""
L0 MLP Sufficiency Test: Patch L0 from recursive into baseline.

Goal: Test if L0 MLP alone is SUFFICIENT to induce contraction,
not just necessary.

Method:
1. Run model on BASELINE prompt (clean)
2. Patch ONLY L0 MLP output from RECURSIVE prompt activations
3. Measure: Does R_V contract? Does behavior shift toward recursive?

This answers: "Is L0 sufficient, not just necessary?"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

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


class MLPPatchingHook:
    """Patch MLP output at specified layer with source activations."""
    
    def __init__(self, model, layer_idx: int, source_activation: torch.Tensor):
        """
        Args:
            model: The model
            layer_idx: Layer to patch
            source_activation: Activation tensor to patch in (from recursive run)
                              Shape: (batch, seq_len, hidden_dim)
        """
        self.model = model
        self.layer_idx = layer_idx
        self.source_activation = source_activation.detach()
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
    
    def register(self):
        """Register forward hook to patch MLP output."""
        if self.handle is not None:
            raise RuntimeError("Hook already registered. Call remove() first.")
        
        mlp = self.model.model.layers[self.layer_idx].mlp
        
        def hook_fn(module, inp, out):
            """Patch MLP output with source activation."""
            if isinstance(out, tuple):
                out_tensor = out[0]
            else:
                out_tensor = out
            
            # Match sequence lengths
            batch_size, target_seq_len, hidden_dim = out_tensor.shape
            source_seq_len = self.source_activation.shape[1]
            
            # Use last W tokens from source (matching window size)
            W = min(16, source_seq_len, target_seq_len)
            source_patch = self.source_activation[:, -W:, :].to(out_tensor.device)
            
            # Patch last W tokens
            out_patched = out_tensor.clone()
            out_patched[:, -W:, :] = source_patch[:, :W, :]
            
            if isinstance(out, tuple):
                return (out_patched,) + out[1:]
            return out_patched
        
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


def run_mlp_sufficiency_test_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """Run L0 MLP sufficiency test."""
    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-v0.1")
    layer_idx = params.get("layer", 0)
    n_pairs = params.get("n_pairs", 30)
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
    
    # Get prompt pairs WITH IDs (Industry-grade reproducibility)
    pairs_with_ids = loader.get_balanced_pairs_with_ids(n_pairs=n_pairs, seed=seed)
    pairs = [(rec_text, base_text) for _, _, rec_text, base_text in pairs_with_ids]
    
    print(f"\n{'='*60}")
    print(f"L{layer_idx} MLP SUFFICIENCY TEST")
    print(f"{'='*60}")
    print(f"Layer: L{layer_idx}")
    print(f"Pairs: {len(pairs)}")
    print(f"Test: Patch L{layer_idx} MLP from recursive into baseline")
    print(f"{'='*60}\n")
    
    results = []
    
    for pair_idx, (rec_id, base_id, rec_text, base_text) in enumerate(tqdm(pairs_with_ids, desc="Testing pairs")):
        # BASELINE: Run on baseline prompt (clean)
        inputs_base = tokenizer(base_text, return_tensors="pt", add_special_tokens=False).to(device)
        
        with torch.no_grad():
            out_base = model(**inputs_base)
        
        # Compute baseline R_V
        rv_base = compute_rv(model, tokenizer, base_text, early=5, late=27, window=window_size, device=device)
        
        # Compute baseline mode score
        try:
            mode_base = mode_metric.compute_score(out_base.logits, baseline_logits=out_base.logits)
        except Exception as e:
            print(f"  Error computing baseline mode score: {e}")
            mode_base = float("nan")
        
        # RECURSIVE: Run on recursive prompt to capture L0 MLP activation
        inputs_rec = tokenizer(rec_text, return_tensors="pt", add_special_tokens=False).to(device)
        
        # Capture L0 MLP output from recursive run
        l0_mlp_recursive = None
        
        def capture_hook(module, inp, out):
            nonlocal l0_mlp_recursive
            if isinstance(out, tuple):
                l0_mlp_recursive = out[0].detach().clone()
            else:
                l0_mlp_recursive = out.detach().clone()
            return out
        
        mlp = model.model.layers[layer_idx].mlp
        capture_handle = mlp.register_forward_hook(capture_hook)
        
        try:
            with torch.no_grad():
                out_rec = model(**inputs_rec)
        finally:
            capture_handle.remove()
        
        if l0_mlp_recursive is None:
            print(f"  Warning: Failed to capture L{layer_idx} MLP activation")
            continue
        
        # PATCH: Run baseline prompt with L0 MLP patched from recursive
        patching_hook = MLPPatchingHook(model, layer_idx, l0_mlp_recursive)
        
        try:
            # Compute R_V with patching (on base_text, not generated)
            with patching_hook:
                with torch.no_grad():
                    rv_patched = compute_rv(model, tokenizer, base_text, early=5, late=27, window=window_size, device=device)
                    out_patched = model(**inputs_base)
            
            # Generate text with patching (for behavior analysis)
            with patching_hook:
                with torch.no_grad():
                    inputs_gen = tokenizer(base_text, return_tensors="pt", add_special_tokens=False).to(device)
                    outputs_patched = model.generate(
                        **inputs_gen,
                        max_new_tokens=max_new_tokens,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        max_length=inputs_gen["input_ids"].shape[1] + max_new_tokens  # Add explicit max_length
                    )
            
            generated_text = tokenizer.decode(outputs_patched[0], skip_special_tokens=True)
            
            try:
                mode_patched = mode_metric.compute_score(out_patched.logits, baseline_logits=out_base.logits)
            except Exception as e:
                print(f"  Error computing patched mode score: {e}")
                mode_patched = float("nan")
            
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
            print(f"  Error during patching at pair {pair_idx}: {e}")
            rv_patched = float("nan")
            mode_patched = float("nan")
            generated_text = ""
            coherence = 0.0
            recursion_score = 0.0
        
        # Compute deltas (how much did patching restore toward recursive?)
        # For comparison, get recursive R_V
        rv_rec = compute_rv(model, tokenizer, rec_text, early=5, late=27, window=window_size, device=device)
        
        # Restoration metrics
        rv_gap = rv_base - rv_rec  # Gap between baseline and recursive
        rv_restored = rv_base - rv_patched  # How much patching restored
        rv_restoration_pct = (rv_restored / rv_gap * 100.0) if rv_gap > 1e-6 else 0.0
        
        mode_delta = mode_patched - mode_base if not np.isnan(mode_patched) else float("nan")
        
        results.append({
            "pair_idx": pair_idx,
            "recursive_prompt_id": rec_id,
            "baseline_prompt_id": base_id,
            "recursive_text": rec_text,
            "baseline_text": base_text,
            "generated_text": generated_text,
            "layer": layer_idx,
            "rv_baseline": rv_base,
            "rv_recursive": rv_rec,
            "rv_patched": rv_patched,
            "rv_gap": rv_gap,
            "rv_restored": rv_restored,
            "rv_restoration_pct": rv_restoration_pct,
            "mode_baseline": mode_base,
            "mode_patched": mode_patched,
            "mode_delta": mode_delta,
            "coherence": coherence,
            "recursion_score": recursion_score,
        })
    
    # Save results
    df = pd.DataFrame(results)
    csv_path = run_dir / "mlp_sufficiency_test.csv"
    df.to_csv(csv_path, index=False)
    
    # Statistical analysis
    rv_restoration_pcts = df["rv_restoration_pct"].dropna().values
    mode_deltas = df["mode_delta"].dropna().values
    rv_bases = df["rv_baseline"].dropna().values
    rv_patcheds = df["rv_patched"].dropna().values

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

    # One-sample t-test: Is restoration significantly > 0?
    rv_stat = None
    rv_pvalue = None
    rv_significant = None
    rv_cohens_d = None
    rv_ci_95 = (float("nan"), float("nan"))
    if len(rv_restoration_pcts) >= 3:
        t_stat, p_val = stats.ttest_1samp(rv_restoration_pcts, 0.0)
        rv_stat = float(t_stat)
        rv_pvalue = float(p_val)
        rv_significant = p_val < 0.01 and np.mean(rv_restoration_pcts) > 0
        rv_ci_95 = compute_ci_95(rv_restoration_pcts)
        # Cohen's d for baseline vs patched
        if len(rv_bases) >= 2 and len(rv_patcheds) >= 2:
            rv_cohens_d = compute_cohens_d(rv_bases, rv_patcheds)

    mode_stat = None
    mode_pvalue = None
    mode_significant = None
    mode_ci_95 = (float("nan"), float("nan"))
    if len(mode_deltas) >= 3:
        t_stat, p_val = stats.ttest_1samp(mode_deltas, 0.0)
        mode_stat = float(t_stat)
        mode_pvalue = float(p_val)
        mode_significant = p_val < 0.01
        mode_ci_95 = compute_ci_95(mode_deltas)
    
    # Get standardized metadata
    metadata = get_run_metadata(
        cfg,
        prompt_ids=pairs_with_ids,
        eval_window=window_size,
        intervention_scope="last_16",  # Patching last 16 tokens (matches window)
        behavior_metric="mode_score_m",
    )
    
    # Summary (Industry-grade metric contract)
    summary = {
        "experiment": "mlp_sufficiency_test",
        "layer": layer_idx,
        "n_pairs": len(pairs),
        # PRIMARY: Mode Score M
        "mode_score_m": float(df["mode_baseline"].mean()) if df["mode_baseline"].notna().any() else None,
        "mode_score_m_delta": float(df["mode_delta"].mean()) if df["mode_delta"].notna().any() else None,
        "mode_t_statistic": mode_stat,
        "mode_pvalue": mode_pvalue,
        "mode_significant": mode_significant,
        "mode_ci_95": mode_ci_95,
        # SECONDARY: R_V signature
        "rv": float(df["rv_baseline"].mean()),
        "rv_baseline_mean": float(df["rv_baseline"].mean()),
        "rv_recursive_mean": float(df["rv_recursive"].mean()),
        "rv_patched_mean": float(df["rv_patched"].mean()),
        "rv_restoration_pct": float(df["rv_restoration_pct"].mean()),
        "rv_restoration_pct_mean": float(df["rv_restoration_pct"].mean()),
        "rv_restoration_pct_std": float(df["rv_restoration_pct"].std()),
        "rv_t_statistic": rv_stat,
        "rv_pvalue": rv_pvalue,
        "rv_significant": rv_significant,
        "rv_cohens_d": rv_cohens_d,
        "rv_restoration_ci_95": rv_ci_95,
        # Standardized metadata
        "eval_window": window_size,
        "intervention_scope": "last_16",
        "behavior_metric": "mode_score_m",
        # Merge metadata
        **metadata,
    }
    
    # Verdict
    if rv_significant and summary["rv_restoration_pct_mean"] > 50.0:
        verdict = f"L{layer_idx} MLP is SUFFICIENT - Patching restores {summary['rv_restoration_pct_mean']:.1f}% of contraction"
    elif rv_significant and summary["rv_restoration_pct_mean"] > 0:
        verdict = f"L{layer_idx} MLP is PARTIALLY SUFFICIENT - Patching restores {summary['rv_restoration_pct_mean']:.1f}% of contraction"
    else:
        verdict = f"L{layer_idx} MLP is NOT SUFFICIENT - Patching does not restore contraction"
    
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
    print(f"R_V baseline: {summary['rv_baseline_mean']:.4f}")
    print(f"R_V recursive: {summary['rv_recursive_mean']:.4f}")
    print(f"R_V patched: {summary['rv_patched_mean']:.4f}")
    print(f"Restoration: {summary['rv_restoration_pct_mean']:.1f}% ± {summary['rv_restoration_pct_std']:.1f}%")
    if rv_significant:
        print(f"  p-value: {rv_pvalue:.4e} (SIGNIFICANT)")
    print(f"\nVERDICT: {verdict}")
    print(f"\n✅ Results saved to: {csv_path}")
    print(f"✅ Summary saved to: {summary_path}")
    
    return ExperimentResult(summary=summary)

