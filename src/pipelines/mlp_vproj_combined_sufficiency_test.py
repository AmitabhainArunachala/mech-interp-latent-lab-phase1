"""
Combined MLP + V_proj Sufficiency Test.

Tests if MLP layers (gate + amplifier) + L27 V_proj (readout) together are SUFFICIENT.

This tests the complete circuit:
- L0+L1 MLP: Gate
- L18-L20 MLP: Amplifier  
- L27 V_proj: Readout

Method:
1. Run model on BASELINE prompt (clean)
2. Capture MLP outputs from RECURSIVE prompt (L0, L1, L18, L19, L20)
3. Capture V_proj from RECURSIVE prompt (L27)
4. Patch ALL (MLP + V_proj) into baseline run
5. Measure: Does R_V contract? Does behavior shift?
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
from src.core.patching import PersistentVPatcher, extract_v_activation
from src.metrics.behavior_strict import score_behavior_strict
from src.metrics.mode_score import ModeScoreMetric
from src.metrics.rv import compute_rv
from src.pipelines.registry import ExperimentResult
from src.utils.run_metadata import get_run_metadata, save_metadata, append_to_run_index


class MultiMLPPatchingHook:
    """Hook to patch multiple MLP layers simultaneously."""
    
    def __init__(self, model, mlp_activations: Dict[int, torch.Tensor]):
        self.model = model
        self.mlp_activations = mlp_activations
        self.handles = []
        self.norm_logs = {}  # Track norm changes
        
        # Register hooks for each layer
        for layer_idx, mlp_act in mlp_activations.items():
            mlp = model.model.layers[layer_idx].mlp
            
            def make_hook(layer_idx, source_act):
                def hook(module, inp, out):
                    # Log norm before patching
                    if isinstance(out, tuple):
                        out_tensor = out[0]
                    else:
                        out_tensor = out
                    
                    if layer_idx not in self.norm_logs:
                        self.norm_logs[layer_idx] = {"before": [], "after": []}
                    
                    norm_before = out_tensor.norm().item()
                    self.norm_logs[layer_idx]["before"].append(norm_before)
                    
                    # Patch: replace with source activation
                    # Handle sequence length matching
                    B, T, D = out_tensor.shape
                    T_src = source_act.shape[1]
                    W = min(16, T, T_src)  # Window size
                    
                    out_patched = out_tensor.clone()
                    out_patched[:, -W:, :] = source_act[:, -W:, :].to(out_tensor.device)
                    
                    norm_after = out_patched.norm().item()
                    self.norm_logs[layer_idx]["after"].append(norm_after)
                    
                    if isinstance(out, tuple):
                        return (out_patched,) + out[1:]
                    return out_patched
                
                return hook
            
            handle = mlp.register_forward_hook(make_hook(layer_idx, mlp_act))
            self.handles.append(handle)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def get_norm_logs(self) -> Dict[int, Dict[str, float]]:
        """Get average norm changes."""
        result = {}
        for layer_idx, logs in self.norm_logs.items():
            if logs["before"] and logs["after"]:
                result[layer_idx] = {
                    "before": np.mean(logs["before"]),
                    "after": np.mean(logs["after"]),
                    "delta": np.mean(logs["after"]) - np.mean(logs["before"]),
                }
        return result


def capture_residual_norm(model, tokenizer, text: str, layer_idx: int, device: str) -> float:
    """Capture residual stream norm at specified layer."""
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    
    residual_norm = None
    
    def capture_hook(module, inp):
        nonlocal residual_norm
        if isinstance(inp, tuple):
            residual = inp[0]
        else:
            residual = inp
        residual_norm = residual.norm().item()
        return inp
    
    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_pre_hook(capture_hook)
    
    try:
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        handle.remove()
    
    return residual_norm if residual_norm is not None else float("nan")


def run_mlp_vproj_combined_sufficiency_test_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """Run combined MLP + V_proj sufficiency test."""
    
    model_cfg = cfg.get("model") or {}
    params = cfg.get("params") or {}
    
    seed = int(cfg.get("seed") or 42)
    device = str(model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = str(model_cfg.get("name") or "mistralai/Mistral-7B-v0.1")
    
    n_pairs = int(params.get("n_pairs") or 30)
    mlp_layers = params.get("mlp_layers", [0, 1, 18, 19, 20])
    vproj_layer = int(params.get("vproj_layer", 27))
    window_size = int(params.get("window_size", 16))
    max_new_tokens = int(params.get("max_new_tokens", 200))
    
    set_seed(seed)
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name, device=device)
    model.eval()
    
    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    
    pairs_with_ids = loader.get_balanced_pairs_with_ids(n_pairs=n_pairs, seed=seed)
    
    mode_metric = ModeScoreMetric(tokenizer, device=device)
    
    results = []
    
    for pair_idx, (rec_id, base_id, rec_text, base_text) in enumerate(tqdm(pairs_with_ids, desc="MLP+V_proj patching")):
        
        # BASELINE: Run model on baseline prompt
        inputs_base = tokenizer(base_text, return_tensors="pt", add_special_tokens=False).to(device)
        
        with torch.no_grad():
            out_base = model(**inputs_base)
        
        # Compute baseline metrics
        rv_base = compute_rv(model, tokenizer, base_text, early=5, late=27, window=window_size, device=device)
        
        try:
            mode_base = mode_metric.compute_score(out_base.logits, baseline_logits=out_base.logits)
        except Exception as e:
            print(f"  Error computing baseline mode score: {e}")
            mode_base = float("nan")
        
        # Capture residual norm at layer 2
        residual_norm_baseline = capture_residual_norm(model, tokenizer, base_text, 2, device)
        
        # RECURSIVE: Capture MLP activations and V_proj
        inputs_rec = tokenizer(rec_text, return_tensors="pt", add_special_tokens=False).to(device)
        
        # Capture MLP outputs
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
        for layer_idx in mlp_layers:
            mlp = model.model.layers[layer_idx].mlp
            handle = mlp.register_forward_hook(make_capture_hook(layer_idx))
            capture_handles.append(handle)
        
        try:
            with torch.no_grad():
                out_rec = model(**inputs_rec)
        finally:
            for handle in capture_handles:
                handle.remove()
        
        # Check if all MLP activations captured
        if not all(layer_idx in mlp_activations for layer_idx in mlp_layers):
            print(f"  Warning: Failed to capture all MLP activations for pair {pair_idx}")
            continue
        
        # Capture V_proj from recursive run
        v_activation = extract_v_activation(model, tokenizer, rec_text, layer_idx=vproj_layer, device=device)
        
        # Compute recursive mode score
        try:
            mode_rec = mode_metric.compute_score(out_rec.logits, baseline_logits=out_base.logits)
        except Exception as e:
            print(f"  Error computing recursive mode score: {e}")
            mode_rec = float("nan")
        
        # PATCH: Run baseline prompt with MLP + V_proj patched from recursive
        mlp_patching_hook = MultiMLPPatchingHook(model, mlp_activations)
        vproj_patcher = PersistentVPatcher(model, v_activation)
        
        # Capture residual norm during patched forward pass
        residual_norm_patched = None
        
        def capture_residual_hook(module, inp):
            nonlocal residual_norm_patched
            if isinstance(inp, tuple):
                residual = inp[0]
            else:
                residual = inp
            residual_norm_patched = residual.norm().item()
            return None
        
        layer2 = model.model.layers[2]
        residual_handle = layer2.register_forward_pre_hook(capture_residual_hook)
        
        try:
            # Register V_proj patcher
            vproj_patcher.register(layer_idx=vproj_layer)
            
            # Generate text with patching
            with mlp_patching_hook:
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
            norm_logs = mlp_patching_hook.get_norm_logs()
            
            # Residual norm was captured during patched forward pass
            if residual_norm_patched is None:
                residual_norm_patched = float("nan")
            
            # Compute R_V with patching
            rv_patched = compute_rv(model, tokenizer, generated_text, early=5, late=27, window=window_size, device=device)
            
            # Compute mode score with patching (need fresh forward pass)
            inputs_base_fresh = tokenizer(base_text, return_tensors="pt", add_special_tokens=False).to(device)
            with mlp_patching_hook:
                with torch.no_grad():
                    out_patched = model(**inputs_base_fresh)
            
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
            vproj_patcher.remove()
            residual_handle.remove()
        
        # For comparison, get recursive R_V
        rv_rec = compute_rv(model, tokenizer, rec_text, early=5, late=27, window=window_size, device=device)
        
        # Restoration metrics
        rv_gap = rv_base - rv_rec  # Gap between baseline and recursive
        rv_restored = rv_base - rv_patched  # How much patching restored
        rv_restoration_pct = (rv_restored / rv_gap * 100.0) if rv_gap > 1e-6 else 0.0
        
        mode_delta = mode_patched - mode_base if not np.isnan(mode_patched) else float("nan")
        
        # Compute mode_restore_norm(M)
        mode_clean = mode_rec
        mode_corrupt = mode_base
        mode_restore_norm = float("nan")
        if not (np.isnan(mode_clean) or np.isnan(mode_corrupt) or np.isnan(mode_patched)):
            denominator = mode_clean - mode_corrupt
            if abs(denominator) > 1e-6:
                mode_restore_norm = (mode_patched - mode_corrupt) / denominator
        
        # Compile results
        row = {
            "recursive_prompt_id": rec_id,
            "baseline_prompt_id": base_id,
            
            # R_V metrics
            "rv_baseline": rv_base,
            "rv_recursive": rv_rec,
            "rv_patched": rv_patched,
            "rv_restoration_pct": rv_restoration_pct,
            
            # Mode score metrics
            "mode_baseline": mode_base,
            "mode_recursive": mode_rec,
            "mode_patched": mode_patched,
            "mode_delta": mode_delta,
            "mode_restore_norm": mode_restore_norm,
            
            # Behavior metrics
            "coherence": coherence,
            "recursion_score": recursion_score,
            
            # Norm logs
            **{f"mlp_L{layer}_norm_before": norm_logs.get(layer, {}).get("before", float("nan"))
               for layer in mlp_layers},
            **{f"mlp_L{layer}_norm_after": norm_logs.get(layer, {}).get("after", float("nan"))
               for layer in mlp_layers},
            **{f"mlp_L{layer}_norm_delta": norm_logs.get(layer, {}).get("delta", float("nan"))
               for layer in mlp_layers},
            "residual_L2_norm_baseline": residual_norm_baseline,
            "residual_L2_norm_patched": residual_norm_patched,
            "residual_L2_norm_delta": residual_norm_patched - residual_norm_baseline if not np.isnan(residual_norm_patched) else float("nan"),
        }
        
        results.append(row)
        
        # Clear cache periodically
        if (pair_idx + 1) % 5 == 0:
            torch.cuda.empty_cache()
    
    # Save CSV
    df = pd.DataFrame(results)
    csv_path = run_dir / "mlp_vproj_combined_sufficiency_test.csv"
    df.to_csv(csv_path, index=False)
    
    # Compute summary statistics
    rv_restoration_pct_mean = float(df["rv_restoration_pct"].mean())
    rv_restoration_pct_std = float(df["rv_restoration_pct"].std())
    
    # Statistical test
    rv_t_stat, rv_pvalue = stats.ttest_1samp(df["rv_restoration_pct"], 0.0)
    rv_significant = rv_pvalue < 0.01
    
    # Mode score stats
    mode_restore_norm_mean = float(df["mode_restore_norm"].dropna().mean()) if df["mode_restore_norm"].notna().any() else None
    mode_t_stat = None
    mode_pvalue = None
    mode_significant = None
    
    if df["mode_restore_norm"].notna().any():
        mode_t_stat, mode_pvalue = stats.ttest_1samp(df["mode_restore_norm"].dropna(), 0.0)
        mode_significant = mode_pvalue < 0.01
    
    # Verdict
    if rv_significant and rv_restoration_pct_mean > 50.0:
        verdict = f"MLP+V_proj IS SUFFICIENT - Restoration: {rv_restoration_pct_mean:.1f}%"
    elif rv_significant and rv_restoration_pct_mean > 0.0:
        verdict = f"MLP+V_proj is PARTIALLY SUFFICIENT - Restoration: {rv_restoration_pct_mean:.1f}%"
    elif rv_significant and rv_restoration_pct_mean < -100.0:
        verdict = f"MLP+V_proj is ANTI-SUFFICIENT - Restoration: {rv_restoration_pct_mean:.1f}%"
    else:
        verdict = f"MLP+V_proj has minimal effect - Restoration: {rv_restoration_pct_mean:.1f}%"
    
    summary = {
        "experiment": "mlp_vproj_combined_sufficiency_test",
        "mlp_layers": mlp_layers,
        "vproj_layer": vproj_layer,
        "n_pairs": len(df),
        
        # R_V metrics
        "rv_baseline_mean": float(df["rv_baseline"].mean()),
        "rv_recursive_mean": float(df["rv_recursive"].mean()),
        "rv_patched_mean": float(df["rv_patched"].mean()),
        "rv_restoration_pct": rv_restoration_pct_mean,
        "rv_restoration_pct_std": rv_restoration_pct_std,
        "rv_t_statistic": float(rv_t_stat),
        "rv_pvalue": float(rv_pvalue),
        "rv_significant": bool(rv_significant),
        
        # Mode score metrics
        "mode_score_m": float(df["mode_patched"].mean()),
        "mode_score_m_delta": float(df["mode_delta"].mean()),
        "mode_restore_norm": mode_restore_norm_mean,
        "mode_t_statistic": float(mode_t_stat) if mode_t_stat is not None else None,
        "mode_pvalue": float(mode_pvalue) if mode_pvalue is not None else None,
        "mode_significant": bool(mode_significant) if mode_significant is not None else None,
        
        # Verdict
        "verdict": verdict,
        
        "prompt_bank_version": bank_version,
        "artifacts": {"csv": str(csv_path)},
    }
    
    # Save summary
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save metadata
    metadata = get_run_metadata(
        cfg,
        prompt_ids=pairs_with_ids,
        eval_window=window_size,
        intervention_scope=f"mlp_{mlp_layers}_vproj_L{vproj_layer}",
        behavior_metric="mode_score_m",
    )
    save_metadata(run_dir, metadata)
    
    # Append to run index
    append_to_run_index(run_dir, summary)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"R_V baseline: {summary['rv_baseline_mean']:.4f}")
    print(f"R_V recursive: {summary['rv_recursive_mean']:.4f}")
    print(f"R_V patched: {summary['rv_patched_mean']:.4f}")
    print(f"Restoration: {summary['rv_restoration_pct']:.1f}% ± {summary['rv_restoration_pct_std']:.1f}%")
    print(f"\nVERDICT: {verdict}")
    print(f"\n✅ Results saved to: {csv_path}")
    print(f"✅ Summary saved to: {summary_path}")
    
    return ExperimentResult(summary=summary)
