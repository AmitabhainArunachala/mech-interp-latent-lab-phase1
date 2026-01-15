"""
Head Ablation Validation Pipeline (Gold Standard Pipeline 4)

Tests whether specific KV-head groups drive R_V contraction at L27.
Includes proper controls: different KV-head, wrong layer.

Based on validate_h18_h26_gold_standard.py, now integrated into the pipeline system.
"""

from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.metrics.rv import participation_ratio
from src.pipelines.registry import ExperimentResult


# Mistral-7B GQA parameters
NUM_KV_HEADS = 8
HEAD_DIM = 128


def _tok_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


@contextmanager
def ablate_kv_head(model, layer_idx: int, kv_head_idx: int):
    """Zero out a specific KV-head in V-projection at given layer."""
    handle = None

    def hook_fn(module, inp, out):
        batch, seq, _ = out.shape
        out_view = out.view(batch, seq, NUM_KV_HEADS, HEAD_DIM)
        out_view[:, :, kv_head_idx, :] = 0.0
        return out_view.view(batch, seq, -1)

    layer = model.model.layers[layer_idx]
    handle = layer.self_attn.v_proj.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        if handle:
            handle.remove()


class VProjectionCapture:
    """Capture V-projection activations at multiple layers."""
    
    def __init__(self, model, layer_indices: List[int]):
        self.model = model
        self.layer_indices = layer_indices
        self.activations: Dict[int, Optional[torch.Tensor]] = {}
        self.handles = []
    
    def __enter__(self):
        for idx in self.layer_indices:
            self.activations[idx] = None
            
            def make_hook(layer_idx):
                def hook_fn(module, inp, out):
                    self.activations[layer_idx] = out.detach()[0]  # Remove batch dim
                    return out
                return hook_fn
            
            layer = self.model.model.layers[idx]
            handle = layer.self_attn.v_proj.register_forward_hook(make_hook(idx))
            self.handles.append(handle)
        return self
    
    def __exit__(self, *args):
        for h in self.handles:
            h.remove()


def compute_rv_with_ablation(
    model,
    tokenizer,
    text: str,
    early_layer: int,
    late_layer: int,
    window: int,
    ablate_layer: Optional[int] = None,
    ablate_kv_head_idx: Optional[int] = None,
    max_length: int = 512,
) -> Tuple[float, int]:
    """Compute R_V with optional KV-head ablation."""
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = toks["input_ids"].to(model.device)
    tlen = int(input_ids.shape[1])
    
    if tlen < window + 1:
        return float("nan"), tlen
    
    with torch.no_grad():
        with VProjectionCapture(model, [early_layer, late_layer]) as cap:
            if ablate_layer is not None and ablate_kv_head_idx is not None:
                with ablate_kv_head(model, ablate_layer, ablate_kv_head_idx):
                    model(input_ids=input_ids)
            else:
                model(input_ids=input_ids)
        
        v_early = cap.activations[early_layer]
        v_late = cap.activations[late_layer]
        
        if v_early is None or v_late is None:
            return float("nan"), tlen
        
        pr_early = participation_ratio(v_early, window_size=window)
        pr_late = participation_ratio(v_late, window_size=window)
        
        if pr_early == 0 or np.isnan(pr_early) or np.isnan(pr_late):
            return float("nan"), tlen
        
        return float(pr_late / pr_early), tlen


def _mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
        "n": int(arr.size),
    }


def _compute_ci(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval."""
    try:
        from scipy import stats
        arr = np.asarray([v for v in data if not np.isnan(v)])
        n = len(arr)
        if n < 2:
            return float("nan"), float("nan")
        mean = np.mean(arr)
        se = stats.sem(arr)
        h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        return float(mean - h), float(mean + h)
    except:
        return float("nan"), float("nan")


def run_head_ablation_validation_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """Run head ablation validation experiment."""
    model_cfg = cfg.get("model") or {}
    params = cfg.get("params") or {}
    
    seed = int(cfg.get("seed") or 42)
    device = str(model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = str(model_cfg.get("name") or "mistralai/Mistral-7B-v0.1")
    
    early_layer = int(params.get("early_layer") or 5)
    target_layer = int(params.get("target_layer") or 27)
    control_layer = int(params.get("control_layer") or 21)
    window = int(params.get("window") or 16)
    
    target_kv_head = int(params.get("target_kv_head") or 2)
    control_kv_head = int(params.get("control_kv_head") or 0)
    
    n_recursive = int(params.get("n_recursive") or 50)
    n_baseline = int(params.get("n_baseline") or 50)
    max_length = int(params.get("max_length") or 512)
    
    recursive_groups = params.get("recursive_groups") or ["champions", "L5_refined", "L4_full", "L3_deeper"]
    baseline_groups = params.get("baseline_groups") or ["baseline_math", "baseline_factual", "baseline_creative"]
    
    # Load model and prompts
    set_seed(seed)
    model, tokenizer = load_model(model_name, device=device)
    
    loader = PromptLoader()
    bank_version = loader.version
    
    # Log prompt bank version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    (run_dir / "prompt_bank_version.json").write_text(
        json.dumps({"version": bank_version}, indent=2) + "\n"
    )
    
    # Gather prompts
    recursive_prompts = []
    for group in recursive_groups:
        recursive_prompts.extend(loader.get_by_group(group))
    
    baseline_prompts = []
    for group in baseline_groups:
        baseline_prompts.extend(loader.get_by_group(group))
    
    # Filter for valid length and sample
    rng = np.random.default_rng(seed)
    
    # get_by_group returns List[str], not List[dict]
    recursive_prompts = [p for p in recursive_prompts if _tok_len(tokenizer, p) >= window]
    baseline_prompts = [p for p in baseline_prompts if _tok_len(tokenizer, p) >= window]
    
    if len(recursive_prompts) > n_recursive:
        recursive_prompts = list(rng.choice(recursive_prompts, n_recursive, replace=False))
    if len(baseline_prompts) > n_baseline:
        baseline_prompts = list(rng.choice(baseline_prompts, n_baseline, replace=False))
    
    # Conditions to test
    conditions = [
        ("no_ablation", None, None),
        ("target_at_target_layer", target_layer, target_kv_head),
        ("control_head_at_target_layer", target_layer, control_kv_head),
        ("target_at_control_layer", control_layer, target_kv_head),
    ]
    
    rows: List[Dict[str, Any]] = []
    
    for prompt_type, prompts in [("recursive", recursive_prompts), ("baseline", baseline_prompts)]:
        for i, p in enumerate(prompts):
            # p is a string (from get_by_group), not a dict
            text = p
            prompt_id = f"{prompt_type}_{i}"
            
            row = {"prompt_type": prompt_type, "prompt_idx": i, "prompt_id": prompt_id}
            
            for cond_name, ablate_layer, ablate_head in conditions:
                rv, tlen = compute_rv_with_ablation(
                    model, tokenizer, text,
                    early_layer, target_layer, window,
                    ablate_layer, ablate_head, max_length
                )
                row[f"rv_{cond_name}"] = rv
                row["token_len"] = tlen
            
            # Compute deltas (if baseline is valid)
            if not np.isnan(row["rv_no_ablation"]):
                row["delta_target_at_target_layer"] = row["rv_target_at_target_layer"] - row["rv_no_ablation"]
                row["delta_control_head_at_target_layer"] = row["rv_control_head_at_target_layer"] - row["rv_no_ablation"]
                row["delta_target_at_control_layer"] = row["rv_target_at_control_layer"] - row["rv_no_ablation"]
                rows.append(row)
    
    # Save CSV
    out_csv = run_dir / "head_ablation_results.csv"
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    
    # Statistical analysis
    rec_rows = [r for r in rows if r["prompt_type"] == "recursive"]
    bas_rows = [r for r in rows if r["prompt_type"] == "baseline"]
    
    analysis = {}
    
    def analyze_deltas(rows_subset: List[Dict], prefix: str):
        if not rows_subset:
            return {}
        
        result = {}
        for cond in ["target_at_target_layer", "control_head_at_target_layer", "target_at_control_layer"]:
            deltas = [r[f"delta_{cond}"] for r in rows_subset if not np.isnan(r.get(f"delta_{cond}", np.nan))]
            if not deltas:
                continue
            
            stats_dict = _mean_std(deltas)
            ci_low, ci_high = _compute_ci(deltas)
            
            # t-test vs 0
            try:
                from scipy import stats as sp_stats
                t_stat, p_val = sp_stats.ttest_1samp(deltas, 0)
                stats_dict["t_stat"] = float(t_stat)
                stats_dict["p_value"] = float(p_val)
            except:
                stats_dict["t_stat"] = float("nan")
                stats_dict["p_value"] = float("nan")
            
            stats_dict["ci_95_low"] = ci_low
            stats_dict["ci_95_high"] = ci_high
            
            # Cohen's d
            if stats_dict["std"] > 0:
                stats_dict["cohens_d"] = float(stats_dict["mean"] / stats_dict["std"])
            else:
                stats_dict["cohens_d"] = 0.0
            
            result[cond] = stats_dict
        
        return result
    
    analysis["recursive"] = analyze_deltas(rec_rows, "recursive")
    analysis["baseline"] = analyze_deltas(bas_rows, "baseline")
    
    # Key comparisons
    comparisons = {}
    
    try:
        from scipy import stats as sp_stats
        
        # 1. Target vs Control head
        for prompt_type, rows_subset in [("recursive", rec_rows), ("baseline", bas_rows)]:
            target_deltas = [r["delta_target_at_target_layer"] for r in rows_subset if not np.isnan(r.get("delta_target_at_target_layer", np.nan))]
            control_deltas = [r["delta_control_head_at_target_layer"] for r in rows_subset if not np.isnan(r.get("delta_control_head_at_target_layer", np.nan))]
            
            if target_deltas and control_deltas:
                min_len = min(len(target_deltas), len(control_deltas))
                t_stat, p_val = sp_stats.ttest_rel(target_deltas[:min_len], control_deltas[:min_len])
                comparisons[f"{prompt_type}_target_vs_control_head"] = {
                    "target_mean": float(np.mean(target_deltas)),
                    "control_mean": float(np.mean(control_deltas)),
                    "t_stat": float(t_stat),
                    "p_value": float(p_val),
                }
        
        # 2. Target layer vs Control layer
        for prompt_type, rows_subset in [("recursive", rec_rows), ("baseline", bas_rows)]:
            target_layer_deltas = [r["delta_target_at_target_layer"] for r in rows_subset if not np.isnan(r.get("delta_target_at_target_layer", np.nan))]
            control_layer_deltas = [r["delta_target_at_control_layer"] for r in rows_subset if not np.isnan(r.get("delta_target_at_control_layer", np.nan))]
            
            if target_layer_deltas and control_layer_deltas:
                min_len = min(len(target_layer_deltas), len(control_layer_deltas))
                t_stat, p_val = sp_stats.ttest_rel(target_layer_deltas[:min_len], control_layer_deltas[:min_len])
                comparisons[f"{prompt_type}_L{target_layer}_vs_L{control_layer}"] = {
                    f"L{target_layer}_mean": float(np.mean(target_layer_deltas)),
                    f"L{control_layer}_mean": float(np.mean(control_layer_deltas)),
                    "t_stat": float(t_stat),
                    "p_value": float(p_val),
                }
    except Exception:
        pass
    
    # Determine pass/fail
    passes = []
    
    # Check 1: Target ablation significantly increases R_V
    rec_target = analysis.get("recursive", {}).get("target_at_target_layer", {})
    if rec_target.get("p_value", 1.0) < 0.001 and rec_target.get("mean", 0) > 0:
        passes.append(("target_effect_significant", True, f"p={rec_target.get('p_value', 1.0):.2e}"))
    else:
        passes.append(("target_effect_significant", False, f"p={rec_target.get('p_value', 1.0):.2e}"))
    
    # Check 2: Target > control head
    rec_vs_ctrl = comparisons.get("recursive_target_vs_control_head", {})
    if rec_vs_ctrl.get("target_mean", 0) > rec_vs_ctrl.get("control_mean", 0):
        passes.append(("target_gt_control_head", True, f"{rec_vs_ctrl.get('target_mean', 0):.4f} > {rec_vs_ctrl.get('control_mean', 0):.4f}"))
    else:
        passes.append(("target_gt_control_head", False, f"{rec_vs_ctrl.get('target_mean', 0):.4f} <= {rec_vs_ctrl.get('control_mean', 0):.4f}"))
    
    # Check 3: L27 > L21
    rec_layer = comparisons.get(f"recursive_L{target_layer}_vs_L{control_layer}", {})
    if rec_layer.get(f"L{target_layer}_mean", 0) > rec_layer.get(f"L{control_layer}_mean", 0):
        passes.append(("target_layer_gt_control_layer", True, f"L{target_layer} > L{control_layer}"))
    else:
        passes.append(("target_layer_gt_control_layer", False, f"L{target_layer} <= L{control_layer}"))
    
    all_pass = all(p[1] for p in passes)
    
    summary = {
        "experiment": "head_ablation_validation",
        "model_name": model_name,
        "device": device,
        "seed": seed,
        "prompt_bank_version": bank_version,
        "params": {
            "early_layer": early_layer,
            "target_layer": target_layer,
            "control_layer": control_layer,
            "window": window,
            "target_kv_head": target_kv_head,
            "control_kv_head": control_kv_head,
            "n_recursive": n_recursive,
            "n_baseline": n_baseline,
        },
        "n_recursive_actual": len(rec_rows),
        "n_baseline_actual": len(bas_rows),
        "analysis": analysis,
        "comparisons": comparisons,
        "pass_checks": [{"check": p[0], "passed": p[1], "detail": p[2]} for p in passes],
        "all_passed": all_pass,
        "artifacts": {"results_csv": str(out_csv)},
        "notes": {
            "gqa_aliasing": f"KV-head {target_kv_head} is shared by Q-heads {target_kv_head}, {target_kv_head+8}, {target_kv_head+16}, {target_kv_head+24} due to GQA",
        },
    }
    
    # Write VERDICT.md
    verdict_path = run_dir / "VERDICT.md"
    verdict_lines = [
        "# Head Ablation Validation - VERDICT",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Model:** {model_name}",
        "",
        "## Pass/Fail",
        "",
    ]
    for check, passed, detail in passes:
        emoji = "✅" if passed else "❌"
        verdict_lines.append(f"- {emoji} **{check}**: {detail}")
    
    verdict_lines.extend([
        "",
        f"## Overall: {'✅ ALL PASSED' if all_pass else '❌ SOME FAILED'}",
        "",
        "## Note on GQA Aliasing",
        "",
        f"In Mistral-7B with GQA, KV-head {target_kv_head} serves Q-heads {target_kv_head}, {target_kv_head+8}, {target_kv_head+16}, {target_kv_head+24}.",
        "Claims should reference 'KV-head group' not individual Q-heads.",
    ])
    
    verdict_path.write_text("\n".join(verdict_lines))
    
    return ExperimentResult(summary=summary)


__all__ = ["run_head_ablation_validation_from_config"]

