"""
Combined MLP Sufficiency Test: Patch multiple MLP layers together.

Tests if L0+L1 (or L0+L1+L3) together are SUFFICIENT to induce contraction.

Method:
1. Run model on BASELINE prompt (clean)
2. Capture L0, L1 (and optionally L3) MLP outputs from RECURSIVE prompt
3. Patch ALL captured layers into baseline run
4. Measure: Does R_V contract? Does behavior shift?
5. Log activation norms to detect norm inflation artifacts

This answers: "Are multiple MLP layers together sufficient?"
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


class MultiMLPPatchingHook:
    """Patch multiple MLP outputs at specified layers with source activations."""
    
    def __init__(self, model, layer_activations: Dict[int, torch.Tensor]):
        """
        Args:
            model: The model
            layer_activations: Dict mapping layer_idx -> source activation tensor
                            Shape: (batch, seq_len, hidden_dim)
        """
        self.model = model
        self.layer_activations = {k: v.detach() for k, v in layer_activations.items()}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.norm_logs: Dict[int, Dict[str, float]] = {}  # layer -> {before_norm, after_norm}
    
    def register(self):
        """Register forward hooks to patch MLP outputs."""
        if self.handles:
            raise RuntimeError("Hooks already registered. Call remove() first.")
        
        for layer_idx, source_activation in self.layer_activations.items():
            mlp = self.model.model.layers[layer_idx].mlp
            
            def make_hook(layer, source):
                def hook_fn(module, inp, out):
                    """Patch MLP output with source activation."""
                    if isinstance(out, tuple):
                        out_tensor = out[0]
                    else:
                        out_tensor = out
                    
                    # Log norm before patching
                    norm_before = out_tensor.norm().item()
                    
                    # Match sequence lengths
                    batch_size, target_seq_len, hidden_dim = out_tensor.shape
                    source_seq_len = source.shape[1]
                    
                    # Use last W tokens from source (matching window size)
                    W = min(16, source_seq_len, target_seq_len)
                    source_patch = source[:, -W:, :].to(out_tensor.device)
                    
                    # Patch last W tokens
                    out_patched = out_tensor.clone()
                    out_patched[:, -W:, :] = source_patch[:, :W, :]
                    
                    # Log norm after patching
                    norm_after = out_patched.norm().item()
                    
                    # Store norm logs
                    if layer not in self.norm_logs:
                        self.norm_logs[layer] = {}
                    self.norm_logs[layer]['before'] = norm_before
                    self.norm_logs[layer]['after'] = norm_after
                    self.norm_logs[layer]['delta'] = norm_after - norm_before
                    
                    if isinstance(out, tuple):
                        return (out_patched,) + out[1:]
                    return out_patched
                
                return hook_fn
            
            handle = mlp.register_forward_hook(make_hook(layer_idx, source_activation))
            self.handles.append(handle)
    
    def remove(self):
        """Remove all forward hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def get_norm_logs(self) -> Dict[int, Dict[str, float]]:
        """Get norm logs for all patched layers."""
        return self.norm_logs.copy()
    
    def __enter__(self):
        self.register()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


def capture_residual_norm(model, tokenizer, text: str, layer_idx: int, device: str) -> float:
    """Capture residual stream norm at specified layer."""
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    
    residual_norm = None
    
    def capture_hook(module, inp):
        nonlocal residual_norm
        # Residual stream is typically the input to the layer
        if isinstance(inp, tuple):
            residual = inp[0]
        else:
            residual = inp
        residual_norm = residual.norm().item()
    
    # Hook at the layer's input (residual stream)
    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_pre_hook(capture_hook)
    
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    
    return residual_norm if residual_norm is not None else float("nan")


def run_combined_mlp_sufficiency_test_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """Run combined MLP sufficiency test."""
    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-v0.1")
    layers = params.get("layers", [0, 1])  # Default: L0+L1
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
    
    layers_str = "+".join([f"L{l}" for l in layers])
    print(f"\n{'='*60}")
    print(f"{layers_str} MLP COMBINED SUFFICIENCY TEST")
    print(f"{'='*60}")
    print(f"Layers: {layers}")
    print(f"Pairs: {len(pairs)}")
    print(f"Test: Patch {layers_str} MLP from recursive into baseline")
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
        
        # Capture residual norm at layer 2 (after L0+L1)
        residual_norm_baseline = capture_residual_norm(model, tokenizer, base_text, 2, device)
        
        # RECURSIVE: Run on recursive prompt to capture MLP activations
        inputs_rec = tokenizer(rec_text, return_tensors="pt", add_special_tokens=False).to(device)
        
        # Capture MLP outputs from recursive run
        mlp_activations = {}
        
        def make_capture_hook(layer_idx):
            def capture_hook(module, inp, out):
                if isinstance(out, tuple):
                    mlp_activations[layer_idx] = out[0].detach().clone()
                else:
                    mlp_activations[layer_idx] = out.detach().clone()
                return out
            return capture_hook
        
        capture_handles = []
        for layer_idx in layers:
            mlp = model.model.layers[layer_idx].mlp
            handle = mlp.register_forward_hook(make_capture_hook(layer_idx))
            capture_handles.append(handle)
        
        try:
            with torch.no_grad():
                out_rec = model(**inputs_rec)
        finally:
            for handle in capture_handles:
                handle.remove()
        
        # Check if all activations captured
        if not all(layer_idx in mlp_activations for layer_idx in layers):
            print(f"  Warning: Failed to capture all MLP activations for pair {pair_idx}")
            continue
        
        # Compute recursive mode score (clean target for restore_norm)
        # This is M_clean in restore_norm formula
        try:
            mode_rec = mode_metric.compute_score(out_rec.logits, baseline_logits=out_base.logits)
        except Exception as e:
            print(f"  Error computing recursive mode score: {e}")
            mode_rec = float("nan")
        
        # PATCH: Run baseline prompt with MLP layers patched from recursive
        patching_hook = MultiMLPPatchingHook(model, mlp_activations)
        
        # Capture residual norm during patched forward pass
        residual_norm_patched = None
        
        def capture_residual_hook(module, inp):
            nonlocal residual_norm_patched
            if isinstance(inp, tuple):
                residual = inp[0]
            else:
                residual = inp
            residual_norm_patched = residual.norm().item()
            return None  # Pre-hook doesn't return anything
        
        layer2 = model.model.layers[2]
        residual_handle = layer2.register_forward_pre_hook(capture_residual_hook)
        
        try:
            # Generate text with patching
            with patching_hook:
                with torch.no_grad():
                    inputs_gen = tokenizer(base_text, return_tensors="pt", add_special_tokens=False).to(device)
                    outputs_patched = model.generate(
                        **inputs_gen,
                        max_new_tokens=max_new_tokens,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
            
            generated_text = tokenizer.decode(outputs_patched[0], skip_special_tokens=True)
            
            # Get norm logs
            norm_logs = patching_hook.get_norm_logs()
            
            # Residual norm was captured during patched forward pass
            if residual_norm_patched is None:
                residual_norm_patched = float("nan")
            
            # Compute R_V with patching
            rv_patched = compute_rv(model, tokenizer, generated_text, early=5, late=27, window=window_size, device=device)
            
            # Compute mode score with patching (need fresh forward pass)
            with patching_hook:
                with torch.no_grad():
                    out_patched = model(**inputs_base)
            
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
            norm_logs = {}
            residual_norm_patched = float("nan")
        finally:
            residual_handle.remove()
        
        # For comparison, get recursive R_V
        rv_rec = compute_rv(model, tokenizer, rec_text, early=5, late=27, window=window_size, device=device)
        
        # Restoration metrics
        rv_gap = rv_base - rv_rec  # Gap between baseline and recursive
        rv_restored = rv_base - rv_patched  # How much patching restored
        rv_restoration_pct = (rv_restored / rv_gap * 100.0) if rv_gap > 1e-6 else 0.0
        
        mode_delta = mode_patched - mode_base if not np.isnan(mode_patched) else float("nan")
        
        # GAP-003 Fix: Compute mode_restore_norm(M) for sufficiency tests
        # restore_norm(M) = (M_patched - M_corrupt) / (M_clean - M_corrupt)
        # Where: M_clean = mode_rec (recursive), M_corrupt = mode_base (baseline), M_patched = mode_patched
        mode_clean = mode_rec  # Already computed above
        mode_corrupt = mode_base
        mode_restore_norm = float("nan")
        if not (np.isnan(mode_clean) or np.isnan(mode_corrupt) or np.isnan(mode_patched)):
            denominator = mode_clean - mode_corrupt
            if abs(denominator) > 1e-6:
                mode_restore_norm = (mode_patched - mode_corrupt) / denominator
        
        # Build norm log entry
        norm_entry = {}
        for layer_idx in layers:
            if layer_idx in norm_logs:
                norm_entry[f"l{layer_idx}_mlp_norm_before"] = norm_logs[layer_idx].get("before", float("nan"))
                norm_entry[f"l{layer_idx}_mlp_norm_after"] = norm_logs[layer_idx].get("after", float("nan"))
                norm_entry[f"l{layer_idx}_mlp_norm_delta"] = norm_logs[layer_idx].get("delta", float("nan"))
        
        norm_entry["residual_l2_norm_baseline"] = residual_norm_baseline
        norm_entry["residual_l2_norm_patched"] = residual_norm_patched
        norm_entry["residual_l2_norm_delta"] = residual_norm_patched - residual_norm_baseline
        
        results.append({
            "pair_idx": pair_idx,
            "recursive_prompt_id": rec_id,
            "baseline_prompt_id": base_id,
            "recursive_text": rec_text,
            "baseline_text": base_text,
            "generated_text": generated_text,
            "layers": str(layers),
            "rv_baseline": rv_base,
            "rv_recursive": rv_rec,
            "rv_patched": rv_patched,
            "rv_gap": rv_gap,
            "rv_restored": rv_restored,
            "rv_restoration_pct": rv_restoration_pct,
            "mode_baseline": mode_base,
            "mode_recursive": mode_rec,  # Clean target for restore_norm
            "mode_patched": mode_patched,
            "mode_delta": mode_delta,
            "mode_restore_norm": mode_restore_norm,  # GAP-003: restore_norm(M) computed
            "coherence": coherence,
            "recursion_score": recursion_score,
            **norm_entry,
        })
    
    # Save results
    df = pd.DataFrame(results)
    csv_path = run_dir / "combined_mlp_sufficiency_test.csv"
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
        rv_significant = bool(p_val < 0.01 and np.mean(rv_restoration_pcts) > 0)
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
        mode_significant = bool(p_val < 0.01)
        mode_ci_95 = compute_ci_95(mode_deltas)
    
    # Norm analysis
    norm_columns = [col for col in df.columns if "norm" in col]
    norm_summary = {}
    for col in norm_columns:
        if df[col].notna().any():
            norm_summary[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
            }
    
    # Get standardized metadata
    metadata = get_run_metadata(
        cfg,
        prompt_ids=pairs_with_ids,
        eval_window=window_size,
        intervention_scope="last_16",  # Patching last 16 tokens (matches window)
        behavior_metric="mode_score_m",
    )
    
    # Add layers_patched to metadata (required for combined sufficiency tests)
    metadata["layers_patched"] = layers
    
    # Add generation_params to metadata (GAP-006 fix)
    metadata["generation_params"] = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    # Summary (Industry-grade metric contract)
    summary = {
        "experiment": "combined_mlp_sufficiency_test",
        "layers": layers,
        "n_pairs": len(pairs),
        # PRIMARY: Mode Score M
        "mode_score_m": float(df["mode_baseline"].mean()) if df["mode_baseline"].notna().any() else None,
        "mode_score_m_delta": float(df["mode_delta"].mean()) if df["mode_delta"].notna().any() else None,
        "mode_restore_norm": float(df["mode_restore_norm"].mean()) if df["mode_restore_norm"].notna().any() else None,  # GAP-003
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
        # Norm logging (diagnostic)
        "norm_summary": norm_summary,
        # Merge metadata
        **metadata,
    }
    
    # Verdict
    if rv_significant and summary["rv_restoration_pct_mean"] > 50.0:
        verdict = f"{layers_str} MLP is SUFFICIENT - Patching restores {summary['rv_restoration_pct_mean']:.1f}% of contraction"
    elif rv_significant and summary["rv_restoration_pct_mean"] > 0:
        verdict = f"{layers_str} MLP is PARTIALLY SUFFICIENT - Patching restores {summary['rv_restoration_pct_mean']:.1f}% of contraction"
    else:
        verdict = f"{layers_str} MLP is NOT SUFFICIENT - Patching does not restore contraction"
    
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
    
    print(f"\nNorm Changes:")
    for layer_idx in layers:
        before_col = f"l{layer_idx}_mlp_norm_before"
        after_col = f"l{layer_idx}_mlp_norm_after"
        if before_col in norm_summary and after_col in norm_summary:
            before = norm_summary[before_col]["mean"]
            after = norm_summary[after_col]["mean"]
            delta = norm_summary.get(f"l{layer_idx}_mlp_norm_delta", {}).get("mean", after - before)
            print(f"  L{layer_idx} MLP: {before:.2f} → {after:.2f} (Δ={delta:+.2f})")
    
    if "residual_l2_norm_baseline" in norm_summary:
        baseline = norm_summary["residual_l2_norm_baseline"]["mean"]
        patched = norm_summary["residual_l2_norm_patched"]["mean"]
        delta = norm_summary.get("residual_l2_norm_delta", {}).get("mean", patched - baseline)
        print(f"  Residual L2: {baseline:.2f} → {patched:.2f} (Δ={delta:+.2f})")
    
    print(f"\nVERDICT: {verdict}")
    print(f"\n✅ Results saved to: {csv_path}")
    print(f"✅ Summary saved to: {summary_path}")
    
    return ExperimentResult(summary=summary)

