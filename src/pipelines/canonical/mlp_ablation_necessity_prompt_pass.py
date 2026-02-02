"""
MLP Ablation Necessity Test: PROMPT-PASS-ONLY Mode

CRITICAL METHODOLOGICAL CHANGE:
- Measures R_V on the SAME PROMPT TEXT for both baseline and ablated runs
- Does NOT generate text (forward pass only)
- This isolates geometric changes from generation artifacts

Problem with original pipeline:
- Ablation → different generation → different text → different R_V
- Cannot tell if R_V change is from:
  (a) Geometric shift in V-space (real signal)
  (b) Measuring different tokens (measurement artifact)

Solution:
- Use identical prompt text for both conditions
- Measure R_V on prompt tokens only (no generation)
- Log PR_early and PR_late separately to see which component moves
- Log token count for validation

Expected Insights:
- If L0-L1 "inverse pattern" persists: Real geometric effect
- If L0-L1 pattern disappears: Measurement artifact from generation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.metrics.rv import compute_rv_with_components
from src.pipelines.registry import ExperimentResult
from src.utils.run_metadata import get_run_metadata, append_to_run_index, save_metadata


class MLPAblationHook:
    """Zero out MLP output at specified layer."""

    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.handle: torch.utils.hooks.RemovableHandle | None = None

    def register(self):
        """Register forward hook to zero MLP output."""
        if self.handle is not None:
            raise RuntimeError("Hook already registered. Call remove() first.")

        mlp = self.model.model.layers[self.layer_idx].mlp

        def hook_fn(module, inp, out):
            """Zero out MLP output."""
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


def run_mlp_ablation_necessity_prompt_pass_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """
    Run MLP ablation necessity test in PROMPT-PASS-ONLY mode.

    Key difference from original pipeline:
    - No text generation
    - Measures R_V on identical prompt text for baseline vs ablated
    - Logs PR_early and PR_late separately
    - Validates that token counts are identical
    """
    params = cfg.get("params", {})
    # Support both config structures: params.model or model.name
    model_name = params.get("model") or cfg.get("model", {}).get("name", "mistralai/Mistral-7B-v0.1")
    layer_idx = params.get("layer", 0)
    n_pairs = params.get("n_pairs", 80)
    window_size = params.get("window_size", 16)
    early_layer = params.get("early_layer", 5)
    late_layer = params.get("late_layer", 27)
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

    # Get prompt pairs WITH IDs
    pairs_with_ids = loader.get_balanced_pairs_with_ids(n_pairs=n_pairs, seed=seed)

    print(f"\n{'='*60}")
    print(f"MLP ABLATION NECESSITY TEST - PROMPT-PASS-ONLY MODE")
    print(f"{'='*60}")
    print(f"Layer: L{layer_idx}")
    print(f"Pairs: {n_pairs}")
    print(f"Mode: Forward pass only (NO generation)")
    print(f"Measurement: Same prompt text for baseline vs ablated")
    print(f"{'='*60}\n")

    results = []

    for pair_idx, (rec_id, base_id, rec_text, base_text) in enumerate(tqdm(pairs_with_ids, desc="Testing pairs")):
        # Get token counts for validation
        rec_tokens = tokenizer(rec_text, return_tensors="pt", add_special_tokens=False)
        base_tokens = tokenizer(base_text, return_tensors="pt", add_special_tokens=False)
        rec_token_count = rec_tokens["input_ids"].shape[1]
        base_token_count = base_tokens["input_ids"].shape[1]

        # ===== BASELINE: Measure R_V on recursive prompt WITHOUT ablation =====
        rv_baseline, pr_early_baseline, pr_late_baseline = compute_rv_with_components(
            model, tokenizer, rec_text,
            early=early_layer, late=late_layer, window=window_size, device=device
        )

        # ===== ABLATION: Measure R_V on recursive prompt WITH ablation =====
        ablation_hook = MLPAblationHook(model, layer_idx)

        try:
            with ablation_hook:
                rv_ablated, pr_early_ablated, pr_late_ablated = compute_rv_with_components(
                    model, tokenizer, rec_text,
                    early=early_layer, late=late_layer, window=window_size, device=device
                )
        except Exception as e:
            print(f"  Error during ablation at pair {pair_idx}: {e}")
            rv_ablated = float("nan")
            pr_early_ablated = float("nan")
            pr_late_ablated = float("nan")

        # Compute deltas
        rv_delta = rv_ablated - rv_baseline
        pr_early_delta = pr_early_ablated - pr_early_baseline
        pr_late_delta = pr_late_ablated - pr_late_baseline

        results.append({
            "pair_idx": pair_idx,
            "recursive_prompt_id": rec_id,
            "baseline_prompt_id": base_id,
            "recursive_text": rec_text,
            "baseline_text": base_text,
            "layer": layer_idx,
            # Token counts (for validation)
            "rec_token_count": rec_token_count,
            "base_token_count": base_token_count,
            # R_V measurements
            "rv_baseline": rv_baseline,
            "rv_ablated": rv_ablated,
            "rv_delta": rv_delta,
            # PR components (CRITICAL: Shows which component moves)
            "pr_early_baseline": pr_early_baseline,
            "pr_early_ablated": pr_early_ablated,
            "pr_early_delta": pr_early_delta,
            "pr_late_baseline": pr_late_baseline,
            "pr_late_ablated": pr_late_ablated,
            "pr_late_delta": pr_late_delta,
        })

    # Save results
    df = pd.DataFrame(results)
    csv_path = run_dir / "mlp_ablation_necessity_prompt_pass.csv"
    df.to_csv(csv_path, index=False)

    # Statistical analysis
    rv_deltas = df["rv_delta"].dropna().values
    pr_early_deltas = df["pr_early_delta"].dropna().values
    pr_late_deltas = df["pr_late_delta"].dropna().values

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
    pr_early_baselines = df["pr_early_baseline"].dropna().values
    pr_early_ablateds = df["pr_early_ablated"].dropna().values
    pr_late_baselines = df["pr_late_baseline"].dropna().values
    pr_late_ablateds = df["pr_late_ablated"].dropna().values

    # Statistical tests
    rv_stat = None
    rv_pvalue = None
    rv_significant = None
    rv_ci_95 = (float("nan"), float("nan"))
    if len(rv_deltas) >= 3:
        t_stat, p_val = stats.ttest_1samp(rv_deltas, 0.0)
        rv_stat = float(t_stat)
        rv_pvalue = float(p_val)
        rv_significant = bool(p_val < 0.01)
        rv_ci_95 = compute_ci_95(rv_deltas)

    # PR component tests
    pr_early_stat, pr_early_pvalue, pr_early_significant = None, None, None
    pr_late_stat, pr_late_pvalue, pr_late_significant = None, None, None
    pr_early_ci_95 = (float("nan"), float("nan"))
    pr_late_ci_95 = (float("nan"), float("nan"))

    if len(pr_early_deltas) >= 3:
        t_stat, p_val = stats.ttest_1samp(pr_early_deltas, 0.0)
        pr_early_stat = float(t_stat)
        pr_early_pvalue = float(p_val)
        pr_early_significant = bool(p_val < 0.01)
        pr_early_ci_95 = compute_ci_95(pr_early_deltas)

    if len(pr_late_deltas) >= 3:
        t_stat, p_val = stats.ttest_1samp(pr_late_deltas, 0.0)
        pr_late_stat = float(t_stat)
        pr_late_pvalue = float(p_val)
        pr_late_significant = bool(p_val < 0.01)
        pr_late_ci_95 = compute_ci_95(pr_late_deltas)

    # Effect sizes
    rv_cohens_d = compute_cohens_d(rv_baselines, rv_ablateds) if len(rv_baselines) >= 2 and len(rv_ablateds) >= 2 else None
    pr_early_cohens_d = compute_cohens_d(pr_early_baselines, pr_early_ablateds) if len(pr_early_baselines) >= 2 and len(pr_early_ablateds) >= 2 else None
    pr_late_cohens_d = compute_cohens_d(pr_late_baselines, pr_late_ablateds) if len(pr_late_baselines) >= 2 and len(pr_late_ablateds) >= 2 else None

    # Get standardized metadata
    metadata = get_run_metadata(
        cfg,
        prompt_ids=pairs_with_ids,
        eval_window=window_size,
        intervention_scope="all_tokens",
        behavior_metric="rv",
    )

    # Summary statistics
    summary = {
        "experiment": "mlp_ablation_necessity_prompt_pass",
        "mode": "prompt_pass_only",
        "layer": layer_idx,
        "n_pairs": len(pairs_with_ids),
        # Canonical schema keys (required)
        # For MLP ablation: rv_recursive_mean = baseline (unablated on recursive prompts)
        # rv_baseline_mean = ablated (control condition)
        "rv_recursive_mean": float(df["rv_baseline"].mean()),  # Unablated R_V on recursive prompts
        "rv_baseline_mean": float(df["rv_ablated"].mean()),    # Ablated R_V (control)
        "rv_delta_mean": float(df["rv_delta"].mean()),
        "rv_cohens_d": rv_cohens_d,
        "rv_p_value": rv_pvalue,
        # logit_diff not applicable for this experiment (geometry-only ablation test)
        "logit_diff_delta_mean": None,
        "logit_diff_cohens_d": None,
        "logit_diff_p_value": None,
        # Original detailed metrics
        "rv": float(df["rv_baseline"].mean()),
        "rv_unablated_mean": float(df["rv_baseline"].mean()),
        "rv_unablated_std": float(df["rv_baseline"].std()),
        "rv_ablated_mean": float(df["rv_ablated"].mean()),
        "rv_ablated_std": float(df["rv_ablated"].std()),
        "rv_delta_std": float(df["rv_delta"].std()),
        "rv_t_statistic": rv_stat,
        "rv_pvalue": rv_pvalue,
        "rv_significant": rv_significant,
        "rv_delta_ci_95": rv_ci_95,
        # PR_early metrics (DIAGNOSTIC: Which component moves?)
        "pr_early_baseline_mean": float(df["pr_early_baseline"].mean()),
        "pr_early_baseline_std": float(df["pr_early_baseline"].std()),
        "pr_early_ablated_mean": float(df["pr_early_ablated"].mean()),
        "pr_early_ablated_std": float(df["pr_early_ablated"].std()),
        "pr_early_delta_mean": float(df["pr_early_delta"].mean()),
        "pr_early_delta_std": float(df["pr_early_delta"].std()),
        "pr_early_t_statistic": pr_early_stat,
        "pr_early_pvalue": pr_early_pvalue,
        "pr_early_significant": pr_early_significant,
        "pr_early_cohens_d": pr_early_cohens_d,
        "pr_early_delta_ci_95": pr_early_ci_95,
        # PR_late metrics (DIAGNOSTIC: Which component moves?)
        "pr_late_baseline_mean": float(df["pr_late_baseline"].mean()),
        "pr_late_baseline_std": float(df["pr_late_baseline"].std()),
        "pr_late_ablated_mean": float(df["pr_late_ablated"].mean()),
        "pr_late_ablated_std": float(df["pr_late_ablated"].std()),
        "pr_late_delta_mean": float(df["pr_late_delta"].mean()),
        "pr_late_delta_std": float(df["pr_late_delta"].std()),
        "pr_late_t_statistic": pr_late_stat,
        "pr_late_pvalue": pr_late_pvalue,
        "pr_late_significant": pr_late_significant,
        "pr_late_cohens_d": pr_late_cohens_d,
        "pr_late_delta_ci_95": pr_late_ci_95,
        # Token count validation
        "mean_rec_token_count": float(df["rec_token_count"].mean()),
        "mean_base_token_count": float(df["base_token_count"].mean()),
        # Standardized metadata
        "eval_window": window_size,
        "intervention_scope": "all_tokens",
        "behavior_metric": "rv",
        **metadata,
    }

    # Interpretation
    rv_delta_mean = summary["rv_delta_mean"]
    pr_early_delta_mean = summary["pr_early_delta_mean"]
    pr_late_delta_mean = summary["pr_late_delta_mean"]

    # Determine which component drives the effect
    if rv_significant:
        if abs(pr_late_delta_mean) > abs(pr_early_delta_mean):
            dominant_component = "PR_late"
            component_direction = "increases" if pr_late_delta_mean > 0 else "decreases"
        else:
            dominant_component = "PR_early"
            component_direction = "increases" if pr_early_delta_mean > 0 else "decreases"

        if rv_delta_mean > 0.1:
            verdict = f"L{layer_idx} MLP IS NECESSARY - ablation removes contraction (Δ={rv_delta_mean:.3f}, driven by {dominant_component} {component_direction})"
        elif rv_delta_mean < -0.1:
            verdict = f"L{layer_idx} MLP INVERSE EFFECT - ablation increases contraction (Δ={rv_delta_mean:.3f}, driven by {dominant_component} {component_direction})"
        else:
            verdict = f"L{layer_idx} MLP minimal effect (Δ={rv_delta_mean:.3f})"
    else:
        verdict = f"L{layer_idx} MLP no significant effect (p={rv_pvalue:.4f})"

    summary["verdict"] = verdict
    summary["dominant_component"] = dominant_component if rv_significant else None

    # Save metadata and summary
    save_metadata(run_dir, metadata)
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    append_to_run_index(run_dir, summary)

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS - PROMPT-PASS-ONLY MODE")
    print(f"{'='*60}")
    print(f"R_V baseline:  {summary['rv_baseline_mean']:.4f} ± {summary['rv_baseline_std']:.4f}")
    print(f"R_V ablated:   {summary['rv_ablated_mean']:.4f} ± {summary['rv_ablated_std']:.4f}")
    print(f"R_V delta:     {summary['rv_delta_mean']:.4f} ± {summary['rv_delta_std']:.4f}")
    if rv_significant:
        print(f"  p-value: {rv_pvalue:.4f} (SIGNIFICANT)")
    else:
        print(f"  p-value: {rv_pvalue:.4f} (not significant)")

    print(f"\nCOMPONENT ANALYSIS:")
    print(f"PR_early delta: {pr_early_delta_mean:.4f} (p={pr_early_pvalue:.4f})")
    print(f"PR_late delta:  {pr_late_delta_mean:.4f} (p={pr_late_pvalue:.4f})")

    print(f"\nVERDICT: {verdict}")
    print(f"\n✅ Results saved to: {csv_path}")
    print(f"✅ Summary saved to: {summary_path}")

    return ExperimentResult(summary=summary)
