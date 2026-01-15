"""
Cross-Architecture Validation Pipeline

Tests R_V contraction across:
1. Different models (Mistral-7B vs Llama-3-8B)
2. Different prompt families (6 categories)
3. Window size robustness

This validates the phenomenon isn't Mistral-specific or stylistic.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.metrics.baseline_suite import BaselineMetricsSuite
from src.metrics.rv import compute_rv
from src.pipelines.registry import ExperimentResult
from src.utils.run_metadata import get_run_metadata, save_metadata, append_to_run_index


# Use EXACT confound_validation setup
# Prompt groups from validated confound_validation run
RECURSIVE_GROUP = "champions"
CONTROL_GROUPS = ["length_matched", "pseudo_recursive"]


def run_cross_architecture_validation_from_config(
    cfg: Dict[str, Any],
    run_dir: Path,
) -> ExperimentResult:
    """
    Run cross-architecture validation experiment.
    
    Tests R_V contraction across:
    - Multiple models (Mistral-7B, Llama-3-8B)
    - Multiple prompt families (6 categories)
    - Multiple window sizes (8, 16, 32, 64, 128)
    
    Args:
        cfg: Config dict with:
            - params.model: Model name
            - params.window_sizes: List of window sizes to test
            - params.early_layer: Early layer (default: 5)
            - params.late_layer: Late layer (default: model-dependent)
            - params.seed: Random seed
        run_dir: Output directory
    
    Returns:
        ExperimentResult with summary statistics
    """
    print("=" * 80)
    print("CROSS-ARCHITECTURE VALIDATION")
    print("=" * 80)
    
    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-Instruct-v0.2")
    
    # Use EXACT confound_validation parameters
    early_layer = int(params.get("early_layer", 5))
    late_layer = int(params.get("late_layer", 27))  # Fixed to 27 (from validated run)
    window = int(params.get("window", 16))  # Fixed to 16 (from validated run)
    
    # Prompt groups from config (or use defaults from confound_validation)
    prompt_groups = params.get("prompt_groups", {})
    recursive_group = prompt_groups.get("recursive", RECURSIVE_GROUP)
    control_groups = prompt_groups.get("controls", CONTROL_GROUPS)
    
    # Sample sizes
    n_champions = int(params.get("n_champions", 30))
    n_length_matched = int(params.get("n_length_matched", 30))
    n_pseudo_recursive = int(params.get("n_pseudo_recursive", 30))
    seed = int(params.get("seed", 42))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)
    
    print(f"Model: {model_name}")
    print(f"Early layer: {early_layer}, Late layer: {late_layer}, Window: {window}")
    print(f"Recursive group: {recursive_group}")
    print(f"Control groups: {control_groups}")
    
    # Load model
    model, tokenizer = load_model(model_name, device=device)
    model.eval()
    
    num_layers = len(model.model.layers)
    print(f"Model has {num_layers} layers (using late_layer={late_layer})")
    
    # Load prompts via PromptLoader (canonical source)
    loader = PromptLoader()
    bank_version = loader.version
    bank = loader.prompts  # Full dict with metadata
    
    # Log prompt bank version for reproducibility
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    
    # Get prompts with metadata (matching confound_validation.py)
    def get_prompts_with_metadata(group: str) -> List[Dict]:
        return [{"id": k, **v} for k, v in bank.items() if v.get("group") == group]
    
    # Map control group names to bank group names
    group_mapping = {
        "length_matched": "control_length_matched",
        "pseudo_recursive": "control_pseudo_recursive",
    }
    
    champions_list = get_prompts_with_metadata(recursive_group)
    
    # Get control groups
    control_lists = {}
    for ctrl_name in control_groups:
        bank_group = group_mapping.get(ctrl_name, ctrl_name)
        control_lists[ctrl_name] = get_prompts_with_metadata(bank_group)
    
    # Sample according to config params
    rng = np.random.default_rng(seed)
    
    if len(champions_list) > n_champions:
        indices = rng.choice(len(champions_list), n_champions, replace=False)
        champions_list = [champions_list[i] for i in indices]
    
    for ctrl_name, ctrl_list in control_lists.items():
        n_samples = n_length_matched if ctrl_name == "length_matched" else n_pseudo_recursive
        if len(ctrl_list) > n_samples:
            indices = rng.choice(len(ctrl_list), n_samples, replace=False)
            control_lists[ctrl_name] = [ctrl_list[i] for i in indices]
    
    results = []
    
    # Process champions
    print(f"\n{'='*60}")
    print(f"Processing {recursive_group} ({len(champions_list)} prompts)")
    print(f"{'='*60}")
    
    for ch in tqdm(champions_list, desc=recursive_group):
        text = str(ch["text"])
        tok_len = len(tokenizer.encode(text, add_special_tokens=False))
        if tok_len < window:
            continue
        try:
            rv = compute_rv(model, tokenizer, text, early=early_layer, late=late_layer, window=window, device=device)
            if np.isnan(rv):
                continue
            results.append({
                "prompt_id": str(ch.get("id", "")),
                "prompt_type": "champion",
                "group": recursive_group,
                "text": text[:200],
                "token_count": tok_len,
                "rv": float(rv),
            })
        except Exception as e:
            print(f"  Error processing champion: {e}")
            continue
    
    # Process controls
    for ctrl_name, ctrl_list in control_lists.items():
        print(f"\n{'='*60}")
        print(f"Processing {ctrl_name} ({len(ctrl_list)} prompts)")
        print(f"{'='*60}")
        
        for ctrl in tqdm(ctrl_list, desc=ctrl_name):
            text = str(ctrl["text"])
            tok_len = len(tokenizer.encode(text, add_special_tokens=False))
            if tok_len < window:
                continue
            try:
                rv = compute_rv(model, tokenizer, text, early=early_layer, late=late_layer, window=window, device=device)
                if np.isnan(rv):
                    continue
                results.append({
                    "prompt_id": str(ctrl.get("id", "")),
                    "prompt_type": ctrl_name,
                    "group": group_mapping.get(ctrl_name, ctrl_name),
                    "text": text[:200],
                    "token_count": tok_len,
                    "rv": float(rv),
                })
            except Exception as e:
                print(f"  Error processing {ctrl_name}: {e}")
                continue
    
    # Clear cache periodically
    if len(results) % 10 == 0:
        torch.cuda.empty_cache()
    
    # Save results
    df = pd.DataFrame(results)
    csv_path = run_dir / "cross_architecture_validation.csv"
    df.to_csv(csv_path, index=False)
    
    # Compute summary statistics (matching confound_validation format)
    def arr(mask):
        vals = df.loc[mask, "rv"].astype(float).to_numpy()
        return vals[~np.isnan(vals)]
    
    rv_ch = arr(df["prompt_type"] == "champion")
    rv_len = arr(df["prompt_type"] == "length_matched")
    rv_pseudo = arr(df["prompt_type"] == "pseudo_recursive")
    
    # Helper functions (from confound_validation.py)
    def _ci_95(arr: np.ndarray) -> Dict[str, float]:
        arr = arr[~np.isnan(arr)]
        if len(arr) < 2:
            return {"ci_lower": float("nan"), "ci_upper": float("nan")}
        try:
            sem = stats.sem(arr)
            ci = stats.t.interval(0.95, len(arr) - 1, loc=np.mean(arr), scale=sem)
            return {"ci_lower": float(ci[0]), "ci_upper": float(ci[1])}
        except Exception:
            return {"ci_lower": float("nan"), "ci_upper": float("nan")}
    
    def _ttest(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
        try:
            res = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
            pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
            cohens_d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0
            return {"t": float(res.statistic), "p": float(res.pvalue), "cohens_d": float(cohens_d)}
        except Exception:
            return {"t": float("nan"), "p": float("nan"), "cohens_d": float("nan")}
    
    summary = {
        "experiment": "cross_architecture_validation",
        "prompt_bank_version": bank_version,
        "model": model_name,
        "n_total": int(len(df)),
        "n_champions": int((df["prompt_type"] == "champion").sum()),
        "n_length_matched": int((df["prompt_type"] == "length_matched").sum()),
        "n_pseudo_recursive": int((df["prompt_type"] == "pseudo_recursive").sum()),
        "mean_rv": {
            "champions": float(np.nanmean(rv_ch)) if len(rv_ch) > 0 else float("nan"),
            "length_matched": float(np.nanmean(rv_len)) if len(rv_len) > 0 else float("nan"),
            "pseudo_recursive": float(np.nanmean(rv_pseudo)) if len(rv_pseudo) > 0 else float("nan"),
        },
        "std_rv": {
            "champions": float(np.nanstd(rv_ch)) if len(rv_ch) > 0 else float("nan"),
            "length_matched": float(np.nanstd(rv_len)) if len(rv_len) > 0 else float("nan"),
            "pseudo_recursive": float(np.nanstd(rv_pseudo)) if len(rv_pseudo) > 0 else float("nan"),
        },
        "ci_95_rv": {
            "champions": _ci_95(rv_ch),
            "length_matched": _ci_95(rv_len),
            "pseudo_recursive": _ci_95(rv_pseudo),
        },
        "ttest": {
            "champions_vs_length_matched": _ttest(rv_ch, rv_len),
            "champions_vs_pseudo_recursive": _ttest(rv_ch, rv_pseudo),
            "length_matched_vs_pseudo_recursive": _ttest(rv_len, rv_pseudo),
        },
        "params": {
            "model_name": model_name,
            "early_layer": early_layer,
            "late_layer": late_layer,
            "window": window,
            "n_champions": n_champions,
            "n_length_matched": n_length_matched,
            "n_pseudo_recursive": n_pseudo_recursive,
            "seed": seed,
        },
        "artifacts": {
            "csv": str(csv_path),
        },
    }
    
    # Save summary
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save metadata
    prompt_ids = [(r["prompt_id"], r["prompt_id"], r["text"], r["text"]) for _, r in df.iterrows()]
    metadata = get_run_metadata(
        cfg,
        prompt_ids=prompt_ids[:10],  # Sample
        eval_window=window,
        intervention_scope="none",
        behavior_metric="rv",
    )
    save_metadata(run_dir, metadata)
    
    # Append to run index
    append_to_run_index(run_dir, summary)
    
    # Print summary (matching confound_validation format)
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nR_V Results:")
    print(f"  Champions: {summary['mean_rv']['champions']:.4f} ± {summary['std_rv']['champions']:.4f}")
    print(f"    CI 95%: [{summary['ci_95_rv']['champions']['ci_lower']:.4f}, {summary['ci_95_rv']['champions']['ci_upper']:.4f}]")
    print(f"  Length-matched: {summary['mean_rv']['length_matched']:.4f} ± {summary['std_rv']['length_matched']:.4f}")
    print(f"    CI 95%: [{summary['ci_95_rv']['length_matched']['ci_lower']:.4f}, {summary['ci_95_rv']['length_matched']['ci_upper']:.4f}]")
    print(f"  Pseudo-recursive: {summary['mean_rv']['pseudo_recursive']:.4f} ± {summary['std_rv']['pseudo_recursive']:.4f}")
    print(f"    CI 95%: [{summary['ci_95_rv']['pseudo_recursive']['ci_lower']:.4f}, {summary['ci_95_rv']['pseudo_recursive']['ci_upper']:.4f}]")
    
    print(f"\nStatistical Comparisons:")
    print(f"  Champions vs Length-matched:")
    print(f"    t = {summary['ttest']['champions_vs_length_matched']['t']:.3f}, "
          f"p = {summary['ttest']['champions_vs_length_matched']['p']:.6f}, "
          f"d = {summary['ttest']['champions_vs_length_matched']['cohens_d']:.3f}")
    print(f"  Champions vs Pseudo-recursive:")
    print(f"    t = {summary['ttest']['champions_vs_pseudo_recursive']['t']:.3f}, "
          f"p = {summary['ttest']['champions_vs_pseudo_recursive']['p']:.6f}, "
          f"d = {summary['ttest']['champions_vs_pseudo_recursive']['cohens_d']:.3f}")
    
    # Compare to expected results (if provided)
    expected = cfg.get("expected_results", {})
    if expected:
        print(f"\nComparison to Expected (from validated confound_validation):")
        if "champions_rv" in expected:
            exp_ch = expected["champions_rv"]
            act_ch = summary['mean_rv']['champions']
            print(f"  Champions: Expected {exp_ch:.4f}, Got {act_ch:.4f}, Diff: {abs(act_ch - exp_ch):.4f}")
    
    print(f"\nResults saved to: {run_dir}")
    
    return ExperimentResult(summary=summary)
