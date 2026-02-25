"""
Multi-Token Bridge Experiment: R_V to Behavioral Correlation

Links R_V (measured during prompt processing) to behavioral markers (in generated text).

KEY MEASUREMENT CONTRACT:
- R_V measured on PROMPT TOKENS ONLY (window=16)
- Behavioral metrics extracted from GENERATED TEXT (200 tokens)
- Tests: Does prompt-time geometry predict generation-time behavior?

Hypotheses:
- H1: R_V is negatively correlated with word count (lower R_V → shorter outputs)
- H2: L4 prompts have lower R_V than L3 prompts
- H3: L4 marker presence correlates with lower R_V
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
from src.metrics.rv import compute_rv_with_components
from src.metrics.behavioral_bridge import extract_bridge_metrics, compute_l4_score
from src.pipelines.registry import ExperimentResult
from src.utils.run_metadata import get_run_metadata, append_to_run_index, save_metadata


@dataclass
class GenerationResult:
    """Result from text generation with truncation tracking."""
    text: str
    token_count: int
    eos_reached: bool
    truncated: bool


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    do_sample: bool = True,
    device: str = "cuda",
) -> GenerationResult:
    """
    Generate text continuation from prompt with truncation tracking.

    Args:
        model: Transformer model.
        tokenizer: Tokenizer.
        prompt: Input prompt.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature (0.0 = deterministic).
        do_sample: Whether to sample (False = greedy).
        device: Target device.

    Returns:
        GenerationResult with text, token_count, and truncation flags.
    """
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    prompt_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        if temperature == 0.0:
            outputs = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            outputs = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

    # Extract generated portion
    generated_ids = outputs[0, prompt_len:]
    token_count = len(generated_ids)

    # Check if EOS was reached (not truncated at max_new_tokens)
    eos_id = tokenizer.eos_token_id
    eos_reached = (generated_ids[-1].item() == eos_id) if len(generated_ids) > 0 else False
    truncated = (token_count >= max_new_tokens) and not eos_reached

    # Decode
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return GenerationResult(
        text=generated_text,
        token_count=token_count,
        eos_reached=eos_reached,
        truncated=truncated,
    )


def run_multi_token_bridge_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """
    Run multi-token bridge experiment: R_V to behavioral correlation.

    Config parameters:
        model.name: Model identifier (e.g., "google/gemma-2-9b")
        params.n_prompts: Number of prompts per group (default: 20)
        params.early_layer: Early layer for R_V (default: 5)
        params.late_layer: Late layer for R_V (default: 38 for Gemma)
        params.window: Window size for R_V (default: 16)
        params.max_new_tokens: Generation length (default: 200)
        params.temperatures: List of temperatures (default: [0.0, 0.7])
        params.seed: Random seed (default: 42)
        params.recursive_groups: List of recursive prompt groups (default: ["champions", "L4_full", "L3_deeper"])
        params.baseline_groups: List of baseline prompt groups (default: ["baseline_factual", "baseline_math", "baseline_creative"])
    """
    params = cfg.get("params", {})
    model_name = params.get("model") or cfg.get("model", {}).get("name", "google/gemma-2-9b")
    n_prompts = params.get("n_prompts", 20)
    early_layer = params.get("early_layer", 5)
    late_layer = params.get("late_layer", 38)
    window = params.get("window", 16)
    max_new_tokens = params.get("max_new_tokens", 200)
    temperatures = params.get("temperatures", [0.0, 0.7])
    seed = int(params.get("seed", 42))

    # Config-driven prompt groups (matching Mistral methodology)
    recursive_groups = params.get("recursive_groups", ["champions", "L4_full", "L3_deeper"])
    baseline_groups = params.get("baseline_groups", ["baseline_factual", "baseline_math", "baseline_creative"])

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

    # Load prompts
    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)

    # Get prompts by group - using config-driven groups
    all_prompts: List[tuple[str, str]] = []

    # Recursive groups (champions are the strongest R_V inducers)
    for group in recursive_groups:
        prompts = loader.get_by_group(group, limit=n_prompts, seed=seed)
        for p in prompts:
            all_prompts.append((p, group))

    # Baseline groups (multiple domains per GPT feedback)
    for group in baseline_groups:
        prompts = loader.get_by_group(group, limit=n_prompts, seed=seed)
        for p in prompts:
            all_prompts.append((p, group))

    # Count by type for summary
    recursive_count = sum(1 for _, g in all_prompts if g in recursive_groups)
    baseline_count = sum(1 for _, g in all_prompts if g in baseline_groups)

    print(f"\n{'='*60}")
    print(f"MULTI-TOKEN BRIDGE EXPERIMENT (v2 - GPT Audit Fixes)")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Prompts: {len(all_prompts)} total")
    print(f"  Recursive: {recursive_count} ({recursive_groups})")
    print(f"  Baseline: {baseline_count} ({baseline_groups})")
    print(f"R_V layers: early={early_layer}, late={late_layer}, window={window}")
    print(f"Generation: max_tokens={max_new_tokens}")
    print(f"Temperatures: {temperatures}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")

    all_results = []

    for temp in temperatures:
        print(f"\n--- Temperature: {temp} ---")

        for prompt_text, group in tqdm(all_prompts, desc=f"T={temp}"):
            # Step 1: Measure R_V on prompt tokens
            rv, pr_early, pr_late = compute_rv_with_components(
                model, tokenizer, prompt_text,
                early=early_layer, late=late_layer, window=window, device=device
            )

            # Step 2: Generate text (with truncation tracking)
            gen_result = generate_text(
                model, tokenizer, prompt_text,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                do_sample=(temp > 0.0),
                device=device,
            )

            # Step 3: Extract behavioral metrics
            metrics = extract_bridge_metrics(gen_result.text)
            l4_score = compute_l4_score(gen_result.text)

            # Classify group type for analysis
            is_recursive = group in recursive_groups
            group_type = "recursive" if is_recursive else "baseline"

            # Step 4: Record result (with truncation tracking per GPT feedback)
            result = {
                "group": group,
                "group_type": group_type,
                "temperature": temp,
                "rv": rv,
                "pr_early": pr_early,
                "pr_late": pr_late,
                # Truncation tracking (GPT fix #1)
                "generated_token_count": gen_result.token_count,
                "eos_reached": gen_result.eos_reached,
                "truncated": gen_result.truncated,
                # Behavioral metrics
                "word_count": metrics.word_count,
                "l4_count": metrics.l4_count,
                "l3_count": metrics.l3_count,
                "l4_density": metrics.l4_density,
                "l3_density": metrics.l3_density,
                "has_l4": metrics.has_l4,
                "has_l3": metrics.has_l3,
                "l4_score": l4_score,
                "unique_word_ratio": metrics.unique_word_ratio,
                "l4_markers": ",".join(metrics.l4_markers),
                "l3_markers": ",".join(metrics.l3_markers),
                "prompt_preview": prompt_text[:100],
                "generated_preview": gen_result.text[:200],
            }
            all_results.append(result)

    # Save raw data
    df = pd.DataFrame(all_results)
    df.to_csv(run_dir / "rv_behavioral_correlation.csv", index=False)

    # Statistical Analysis
    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*60}")

    analysis = {}

    for temp in temperatures:
        temp_df = df[df["temperature"] == temp]
        temp_key = f"temp_{temp:.1f}"

        # Filter valid R_V values
        valid_df = temp_df[~temp_df["rv"].isna()]

        # Truncation stats (GPT fix #1: filter by non-truncated)
        n_truncated = int(temp_df["truncated"].sum())
        n_eos_reached = int(temp_df["eos_reached"].sum())
        pct_truncated = 100 * n_truncated / len(temp_df) if len(temp_df) > 0 else 0

        # Filter to non-truncated outputs for cleaner correlation
        non_trunc_df = valid_df[~valid_df["truncated"]]

        # H1: R_V vs word_count (Spearman) - on NON-TRUNCATED only
        if len(non_trunc_df) > 5:
            r_word, p_word = stats.spearmanr(non_trunc_df["rv"], non_trunc_df["word_count"])
        else:
            # Fall back to all data if too few non-truncated
            r_word, p_word = stats.spearmanr(valid_df["rv"], valid_df["word_count"])

        # H2: Recursive vs Baseline R_V (t-test using group_type)
        recursive_rv = valid_df[valid_df["group_type"] == "recursive"]["rv"]
        baseline_rv = valid_df[valid_df["group_type"] == "baseline"]["rv"]

        if len(recursive_rv) > 1 and len(baseline_rv) > 1:
            t_rec_base, p_rec_base = stats.ttest_ind(recursive_rv, baseline_rv)
            d_rec_base = (baseline_rv.mean() - recursive_rv.mean()) / np.sqrt(
                ((len(baseline_rv) - 1) * baseline_rv.std() ** 2 + (len(recursive_rv) - 1) * recursive_rv.std() ** 2)
                / (len(baseline_rv) + len(recursive_rv) - 2)
            )
        else:
            t_rec_base, p_rec_base, d_rec_base = np.nan, np.nan, np.nan

        # H3: has_l4 point-biserial correlation
        r_l4_marker, p_l4_marker = stats.pointbiserialr(
            valid_df["has_l4"].astype(int), valid_df["rv"]
        )

        # Per-group R_V means
        group_rv_means = {}
        group_word_means = {}
        for grp in valid_df["group"].unique():
            grp_data = valid_df[valid_df["group"] == grp]
            group_rv_means[grp] = float(grp_data["rv"].mean())
            group_word_means[grp] = float(grp_data["word_count"].mean())

        # Store results
        analysis[temp_key] = {
            "n_valid": len(valid_df),
            "n_total": len(temp_df),
            "n_non_truncated": len(non_trunc_df),
            "n_truncated": int(n_truncated),
            "n_eos_reached": int(n_eos_reached),
            "pct_truncated": float(pct_truncated),
            # H1: R_V vs word_count
            "h1_spearman_r": float(r_word),
            "h1_spearman_p": float(p_word),
            "h1_significant": bool(p_word < 0.01),
            # H2: Recursive vs Baseline (new: uses group_type)
            "h2_recursive_rv_mean": float(recursive_rv.mean()) if len(recursive_rv) > 0 else None,
            "h2_baseline_rv_mean": float(baseline_rv.mean()) if len(baseline_rv) > 0 else None,
            "h2_t_stat": float(t_rec_base) if not np.isnan(t_rec_base) else None,
            "h2_p_value": float(p_rec_base) if not np.isnan(p_rec_base) else None,
            "h2_cohens_d": float(d_rec_base) if not np.isnan(d_rec_base) else None,
            "h2_significant": bool(p_rec_base < 0.01) if not bool(np.isnan(p_rec_base)) else False,
            # H3: L4 marker correlation
            "h3_point_biserial_r": float(r_l4_marker),
            "h3_point_biserial_p": float(p_l4_marker),
            "h3_significant": bool(p_l4_marker < 0.01),
            # Per-group stats
            "group_rv_means": group_rv_means,
            "group_word_means": group_word_means,
        }

        print(f"\n=== Temperature {temp} ===")
        print(f"Truncation: {n_truncated}/{len(temp_df)} ({pct_truncated:.1f}%) truncated, {n_eos_reached} hit EOS")
        print(f"H1 (R_V vs word_count): r={r_word:.3f}, p={p_word:.2e} {'*' if p_word < 0.01 else ''}")
        print(f"H2 (Recursive vs Baseline R_V): t={t_rec_base:.2f}, p={p_rec_base:.2e}, d={d_rec_base:.2f} {'*' if p_rec_base < 0.01 else ''}")
        print(f"H3 (L4 marker → R_V): r={r_l4_marker:.3f}, p={p_l4_marker:.2e} {'*' if p_l4_marker < 0.01 else ''}")
        print(f"R_V means: Recursive={recursive_rv.mean():.3f}, Baseline={baseline_rv.mean():.3f}")

    # Generate VERDICT
    verdict_lines = ["# Multi-Token Bridge Experiment Results (v2 - GPT Audit Fixes)\n\n"]
    verdict_lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    verdict_lines.append(f"**Model**: {model_name}\n")
    verdict_lines.append(f"**N prompts**: {len(all_prompts)} total\n")
    verdict_lines.append(f"**Recursive groups**: {recursive_groups}\n")
    verdict_lines.append(f"**Baseline groups**: {baseline_groups}\n")
    verdict_lines.append(f"**Seed**: {seed}\n\n")

    for temp in temperatures:
        temp_key = f"temp_{temp:.1f}"
        a = analysis[temp_key]
        verdict_lines.append(f"## Temperature {temp}\n\n")

        # Truncation stats
        verdict_lines.append(f"**Truncation**: {a['n_truncated']}/{a['n_total']} ({a['pct_truncated']:.1f}%) truncated, {a['n_eos_reached']} hit EOS\n\n")

        verdict_lines.append("| Hypothesis | Statistic | p-value | Significant |\n")
        verdict_lines.append("|------------|-----------|---------|-------------|\n")
        verdict_lines.append(f"| H1: R_V vs word_count | r={a['h1_spearman_r']:.3f} | {a['h1_spearman_p']:.2e} | {'Yes' if a['h1_significant'] else 'No'} |\n")
        d_val = a['h2_cohens_d'] if a['h2_cohens_d'] is not None else 0
        p_val = a['h2_p_value'] if a['h2_p_value'] is not None else 1
        verdict_lines.append(f"| H2: Recursive vs Baseline R_V | d={d_val:.2f} | {p_val:.2e} | {'Yes' if a['h2_significant'] else 'No'} |\n")
        verdict_lines.append(f"| H3: L4 markers | r={a['h3_point_biserial_r']:.3f} | {a['h3_point_biserial_p']:.2e} | {'Yes' if a['h3_significant'] else 'No'} |\n\n")

        rec_rv = a['h2_recursive_rv_mean'] if a['h2_recursive_rv_mean'] is not None else 0
        base_rv = a['h2_baseline_rv_mean'] if a['h2_baseline_rv_mean'] is not None else 0
        verdict_lines.append(f"**R_V means**: Recursive={rec_rv:.3f}, Baseline={base_rv:.3f}\n\n")

        # Per-group breakdown
        verdict_lines.append("**Per-group R_V means**:\n")
        for grp, rv_mean in sorted(a.get("group_rv_means", {}).items()):
            verdict_lines.append(f"- {grp}: R_V={rv_mean:.3f}\n")
        verdict_lines.append("\n")

    # Overall verdict
    temp_0_key = "temp_0.0"
    if temp_0_key in analysis:
        a = analysis[temp_0_key]
        h1_pass = a["h1_spearman_r"] < -0.25 and a["h1_significant"]
        h2_pass = a["h2_cohens_d"] is not None and a["h2_cohens_d"] > 0.5 and a["h2_significant"]
        h3_pass = a["h3_point_biserial_r"] < -0.15 and a["h3_significant"]

        if h1_pass and h2_pass:
            verdict = "STRONG CORRELATION - Proceed to sufficiency tests"
        elif h1_pass or h2_pass:
            verdict = "PARTIAL CORRELATION - Investigate confounds"
        else:
            verdict = "NO CORRELATION - R_V does not predict behavior"

        verdict_lines.append(f"## Overall Verdict\n\n**{verdict}**\n")

    (run_dir / "VERDICT.md").write_text("".join(verdict_lines))

    # Summary JSON
    primary_temp = f"temp_{temperatures[0]:.1f}"
    primary_analysis = analysis.get(primary_temp, {})
    rv_recursive_mean = primary_analysis.get("h2_recursive_rv_mean")
    rv_baseline_mean = primary_analysis.get("h2_baseline_rv_mean")
    rv_delta_mean = (
        (rv_baseline_mean - rv_recursive_mean)
        if rv_baseline_mean is not None and rv_recursive_mean is not None
        else None
    )
    summary = {
        "experiment": "multi_token_bridge",
        "version": "v2_gpt_audit_fixes",
        "model": model_name,
        "schema_version": "metrics_summary_v1",
        "n_prompts_per_group": n_prompts,
        "n_total_prompts": len(all_prompts),
        # Top-level fields for validator compatibility
        "n_pairs": len(all_prompts),
        "rv_cohens_d": primary_analysis.get("h2_cohens_d"),
        "rv_p_value": primary_analysis.get("h2_p_value"),
        "cohens_d": primary_analysis.get("h2_cohens_d"),
        "p_value": primary_analysis.get("h2_p_value"),
        "rv_recursive_mean": rv_recursive_mean,
        "rv_baseline_mean": rv_baseline_mean,
        "rv_delta_mean": rv_delta_mean,
        "recursive_groups": recursive_groups,
        "baseline_groups": baseline_groups,
        "temperatures": temperatures,
        "early_layer": early_layer,
        "late_layer": late_layer,
        "window": window,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "prompt_bank_version": bank_version,
        "analysis": analysis,
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    return ExperimentResult(summary=summary)
