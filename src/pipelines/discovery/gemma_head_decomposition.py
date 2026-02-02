"""
Gemma 2 9B Head-wise Decomposition at L3 (Source Layer)

Tests each of the 8 KV-heads at L3 to identify which heads drive the R_V effect.
Implements proper controls:
1. Ablate each KV-head at L3 (source) vs L5 (non-source control)
2. Compare recursive vs baseline prompts
3. Identify head(s) with strongest R_V disruption
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
from scipy import stats
from tqdm import tqdm

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.metrics.rv import participation_ratio
from src.pipelines.registry import ExperimentResult


@contextmanager
def ablate_kv_head(model, layer_idx: int, kv_head_idx: int, num_kv_heads: int, head_dim: int):
    """Zero out a specific KV-head in V-projection at given layer."""
    handle = None

    def hook_fn(module, inp, out):
        batch, seq, _ = out.shape
        out_view = out.view(batch, seq, num_kv_heads, head_dim)
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


def compute_rv_with_head_ablation(
    model,
    tokenizer,
    text: str,
    early_layer: int,
    late_layer: int,
    window: int,
    ablate_layer: Optional[int] = None,
    ablate_kv_head_idx: Optional[int] = None,
    num_kv_heads: int = 8,
    head_dim: int = 256,
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
                with ablate_kv_head(model, ablate_layer, ablate_kv_head_idx, num_kv_heads, head_dim):
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


def run_gemma_head_decomposition_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """Run head-wise decomposition for Gemma 2 9B at L3."""
    params = cfg.get("params", {})
    model_name = cfg.get("model", {}).get("name", "google/gemma-2-9b")

    # Parameters
    source_layer = params.get("source_layer", 3)  # L3 is our validated source
    control_layer = params.get("control_layer", 5)  # L5 as non-source control
    early_layer = params.get("early_layer", 0)  # Use L0 for PR_early
    late_layer = params.get("late_layer", 35)  # Optimal late layer from circuit analysis
    window = params.get("window", 16)
    n_prompts = params.get("n_prompts", 30)
    seed = int(cfg.get("seed", 42))

    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name, device=device)
    model.eval()

    # Get architecture params
    config = model.config
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

    print(f"Architecture: {num_kv_heads} KV-heads, {head_dim} head_dim")

    # Load prompts
    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)

    recursive_prompts = loader.get_by_group("L5_refined")[:n_prompts]
    baseline_prompts = loader.get_by_group("baseline_factual")[:n_prompts]

    print(f"\n{'='*70}")
    print("GEMMA 2 9B HEAD-WISE DECOMPOSITION AT L3")
    print(f"{'='*70}")
    print(f"Source layer: L{source_layer}")
    print(f"Control layer: L{control_layer}")
    print(f"KV-heads to test: 0-{num_kv_heads-1}")
    print(f"Prompts: {len(recursive_prompts)} recursive, {len(baseline_prompts)} baseline")
    print(f"{'='*70}\n")

    all_results = []
    head_summaries = []

    # For each KV-head, test ablation at source vs control layer
    for kv_head in tqdm(range(num_kv_heads), desc="KV-head sweep"):
        print(f"\n--- Testing KV-head {kv_head} ---")

        # Conditions:
        # 1. No ablation (baseline)
        # 2. Ablate this head at SOURCE layer (L3)
        # 3. Ablate this head at CONTROL layer (L5)

        rec_baseline_rvs = []
        rec_source_ablate_rvs = []
        rec_control_ablate_rvs = []

        base_baseline_rvs = []
        base_source_ablate_rvs = []
        base_control_ablate_rvs = []

        # Test on recursive prompts
        for i, text in enumerate(recursive_prompts):
            # No ablation
            rv_base, tlen = compute_rv_with_head_ablation(
                model, tokenizer, text, early_layer, late_layer, window,
                ablate_layer=None, ablate_kv_head_idx=None,
                num_kv_heads=num_kv_heads, head_dim=head_dim
            )

            # Ablate at source layer (L3)
            rv_source, _ = compute_rv_with_head_ablation(
                model, tokenizer, text, early_layer, late_layer, window,
                ablate_layer=source_layer, ablate_kv_head_idx=kv_head,
                num_kv_heads=num_kv_heads, head_dim=head_dim
            )

            # Ablate at control layer (L5)
            rv_control, _ = compute_rv_with_head_ablation(
                model, tokenizer, text, early_layer, late_layer, window,
                ablate_layer=control_layer, ablate_kv_head_idx=kv_head,
                num_kv_heads=num_kv_heads, head_dim=head_dim
            )

            if not np.isnan(rv_base):
                rec_baseline_rvs.append(rv_base)
                rec_source_ablate_rvs.append(rv_source)
                rec_control_ablate_rvs.append(rv_control)

                all_results.append({
                    "kv_head": kv_head,
                    "prompt_type": "recursive",
                    "prompt_idx": i,
                    "rv_baseline": rv_base,
                    f"rv_ablate_L{source_layer}": rv_source,
                    f"rv_ablate_L{control_layer}": rv_control,
                    f"delta_L{source_layer}": rv_source - rv_base,
                    f"delta_L{control_layer}": rv_control - rv_base,
                    "token_len": tlen,
                })

        # Test on baseline prompts
        for i, text in enumerate(baseline_prompts):
            rv_base, tlen = compute_rv_with_head_ablation(
                model, tokenizer, text, early_layer, late_layer, window,
                ablate_layer=None, ablate_kv_head_idx=None,
                num_kv_heads=num_kv_heads, head_dim=head_dim
            )

            rv_source, _ = compute_rv_with_head_ablation(
                model, tokenizer, text, early_layer, late_layer, window,
                ablate_layer=source_layer, ablate_kv_head_idx=kv_head,
                num_kv_heads=num_kv_heads, head_dim=head_dim
            )

            rv_control, _ = compute_rv_with_head_ablation(
                model, tokenizer, text, early_layer, late_layer, window,
                ablate_layer=control_layer, ablate_kv_head_idx=kv_head,
                num_kv_heads=num_kv_heads, head_dim=head_dim
            )

            if not np.isnan(rv_base):
                base_baseline_rvs.append(rv_base)
                base_source_ablate_rvs.append(rv_source)
                base_control_ablate_rvs.append(rv_control)

                all_results.append({
                    "kv_head": kv_head,
                    "prompt_type": "baseline",
                    "prompt_idx": i,
                    "rv_baseline": rv_base,
                    f"rv_ablate_L{source_layer}": rv_source,
                    f"rv_ablate_L{control_layer}": rv_control,
                    f"delta_L{source_layer}": rv_source - rv_base,
                    f"delta_L{control_layer}": rv_control - rv_base,
                    "token_len": tlen,
                })

        # Compute statistics for this head
        rec_source_deltas = [s - b for s, b in zip(rec_source_ablate_rvs, rec_baseline_rvs)]
        rec_control_deltas = [c - b for c, b in zip(rec_control_ablate_rvs, rec_baseline_rvs)]

        base_source_deltas = [s - b for s, b in zip(base_source_ablate_rvs, base_baseline_rvs)]
        base_control_deltas = [c - b for c, b in zip(base_control_ablate_rvs, base_baseline_rvs)]

        # T-test: Does ablating this head at source layer significantly change R_V?
        if len(rec_source_deltas) >= 3:
            t_rec_source, p_rec_source = stats.ttest_1samp(rec_source_deltas, 0)
            t_rec_control, p_rec_control = stats.ttest_1samp(rec_control_deltas, 0)

            # Key test: Is source layer effect stronger than control layer?
            t_source_vs_ctrl, p_source_vs_ctrl = stats.ttest_rel(rec_source_deltas, rec_control_deltas)
        else:
            t_rec_source, p_rec_source = 0, 1
            t_rec_control, p_rec_control = 0, 1
            t_source_vs_ctrl, p_source_vs_ctrl = 0, 1

        head_summary = {
            "kv_head": kv_head,
            "n_recursive": len(rec_source_deltas),
            "n_baseline": len(base_source_deltas),
            # Recursive prompts
            "rec_delta_source_mean": float(np.mean(rec_source_deltas)) if rec_source_deltas else float("nan"),
            "rec_delta_source_std": float(np.std(rec_source_deltas)) if rec_source_deltas else float("nan"),
            "rec_delta_control_mean": float(np.mean(rec_control_deltas)) if rec_control_deltas else float("nan"),
            "rec_delta_control_std": float(np.std(rec_control_deltas)) if rec_control_deltas else float("nan"),
            "rec_t_source": float(t_rec_source),
            "rec_p_source": float(p_rec_source),
            "rec_t_control": float(t_rec_control),
            "rec_p_control": float(p_rec_control),
            "rec_t_source_vs_control": float(t_source_vs_ctrl),
            "rec_p_source_vs_control": float(p_source_vs_ctrl),
            # Baseline prompts
            "base_delta_source_mean": float(np.mean(base_source_deltas)) if base_source_deltas else float("nan"),
            "base_delta_source_std": float(np.std(base_source_deltas)) if base_source_deltas else float("nan"),
            "base_delta_control_mean": float(np.mean(base_control_deltas)) if base_control_deltas else float("nan"),
            "base_delta_control_std": float(np.std(base_control_deltas)) if base_control_deltas else float("nan"),
            # Interpretation flags
            "source_significant": float(p_rec_source) < 0.05,
            "source_stronger_than_control": float(p_source_vs_ctrl) < 0.05 and float(np.mean(rec_source_deltas)) > float(np.mean(rec_control_deltas)),
        }

        head_summaries.append(head_summary)

        print(f"  KV-head {kv_head}: L3 delta = {head_summary['rec_delta_source_mean']:.4f} ± {head_summary['rec_delta_source_std']:.4f}")
        print(f"             L5 delta = {head_summary['rec_delta_control_mean']:.4f} ± {head_summary['rec_delta_control_std']:.4f}")
        print(f"             Source sig: {head_summary['source_significant']}, Source>Control: {head_summary['source_stronger_than_control']}")

    # Save raw results
    results_df = run_dir / "head_ablation_raw.csv"
    if all_results:
        import pandas as pd
        pd.DataFrame(all_results).to_csv(results_df, index=False)

    # Save head summaries
    summary_df = run_dir / "head_summaries.csv"
    import pandas as pd
    pd.DataFrame(head_summaries).to_csv(summary_df, index=False)

    # Identify driver heads
    driver_heads = [h for h in head_summaries if h["source_significant"] and h["source_stronger_than_control"]]

    print(f"\n{'='*70}")
    print("HEAD-WISE DECOMPOSITION RESULTS")
    print(f"{'='*70}")

    if driver_heads:
        print(f"\nDRIVER HEADS (significant L3 effect, stronger than L5):")
        for h in sorted(driver_heads, key=lambda x: abs(x["rec_delta_source_mean"]), reverse=True):
            print(f"  KV-head {h['kv_head']}: delta = {h['rec_delta_source_mean']:+.4f} (p = {h['rec_p_source']:.2e})")
    else:
        print("\nNo single KV-head identified as primary driver.")
        print("Effect may be distributed across heads or interaction-based.")

    # Create summary
    summary = {
        "experiment": "gemma_head_decomposition",
        "model": model_name,
        "source_layer": source_layer,
        "control_layer": control_layer,
        "early_layer": early_layer,
        "late_layer": late_layer,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "n_prompts": n_prompts,
        "n_driver_heads": len(driver_heads),
        "driver_heads": [h["kv_head"] for h in driver_heads],
        "head_summaries": head_summaries,
        "bank_version": bank_version,
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Write VERDICT.md
    verdict = [
        "# Gemma 2 9B Head-wise Decomposition at L3",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Configuration",
        f"- Source layer: L{source_layer}",
        f"- Control layer: L{control_layer}",
        f"- KV-heads tested: {num_kv_heads}",
        f"- Prompts per type: {n_prompts}",
        "",
        "## Results by KV-head",
        "",
        "| KV-head | Δ_L3 (mean±std) | Δ_L5 (mean±std) | p(L3≠0) | Sig? | L3>L5? |",
        "|---------|-----------------|-----------------|---------|------|--------|",
    ]

    for h in head_summaries:
        sig = "✓" if h["source_significant"] else ""
        stronger = "✓" if h["source_stronger_than_control"] else ""
        verdict.append(
            f"| {h['kv_head']} | {h['rec_delta_source_mean']:+.4f}±{h['rec_delta_source_std']:.4f} | "
            f"{h['rec_delta_control_mean']:+.4f}±{h['rec_delta_control_std']:.4f} | "
            f"{h['rec_p_source']:.2e} | {sig} | {stronger} |"
        )

    verdict.extend([
        "",
        "## Driver Heads",
        "",
    ])

    if driver_heads:
        verdict.append("The following KV-heads show significant effects specifically at L3:")
        for h in sorted(driver_heads, key=lambda x: abs(x["rec_delta_source_mean"]), reverse=True):
            verdict.append(f"- **KV-head {h['kv_head']}**: Δ = {h['rec_delta_source_mean']:+.4f} (p = {h['rec_p_source']:.2e})")
    else:
        verdict.append("No single head identified as primary driver. Effect may be:")
        verdict.append("1. Distributed across multiple heads")
        verdict.append("2. Arising from head interactions")
        verdict.append("3. Requiring MLP interaction (not just attention)")

    (run_dir / "VERDICT.md").write_text("\n".join(verdict))

    print(f"\nResults saved to: {run_dir}")

    return ExperimentResult(summary=summary)


__all__ = ["run_gemma_head_decomposition_from_config"]
