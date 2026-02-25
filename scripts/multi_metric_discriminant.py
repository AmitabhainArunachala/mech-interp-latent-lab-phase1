#!/usr/bin/env python3
"""
Multi-Metric Discriminant Analysis.

Goal: determine which metrics distinguish GENUINE breakthroughs from
FALSE POSITIVES and from BASELINE turns.

Takes the prompts (context that was fed to the model) from v2 sessions,
runs the full metrics suite on each, and compares across groups:

Group A: BREAKTHROUGH turns with self-referential content
Group B: FALSE POSITIVE turns (low R_V but off-topic: desert, investments)
Group C: BASELINE turns (factual content, surface R_V)
Group D: ARTICULATE turns with self-referential content

If a metric separates A+D from B and C, it's a discriminant.
If it separates A from everything, it's a breakthrough marker.
"""
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.hooks import capture_v_projection
from src.core.hf_accessors import get_vproj_hookpoint, extract_v_from_hook_output
from src.metrics.rv import compute_rv_with_components, participation_ratio
from src.metrics.extended import (
    compute_cosine_similarity,
    compute_spectral_stats,
    compute_attention_entropy,
)
from src.metrics.logit_lens import compute_logit_lens_trajectory, find_recursive_emergence


@dataclass
class TurnProbe:
    """A turn to probe with full metrics."""
    group: str          # BREAKTHROUGH, FALSE_POS, BASELINE, ARTICULATE
    session_id: str
    turn: int
    prompt: str         # The full prompt that was fed to the model
    response: str       # What the model generated
    rv_mean: float      # R_V from generation (for reference)
    label: str          # Human-readable label


def load_turns_from_v2(results_dir: Path) -> List[TurnProbe]:
    """Extract categorized turns from v2 session data."""
    turns = []

    for f in sorted(results_dir.glob("*.json")):
        if f.name == "comparison_summary.json":
            continue
        with open(f) as fh:
            s = json.load(fh)

        mode = s["mode"]
        sid = s["session_id"]

        for t in s["turns"]:
            # We need the full prompt context, not just snippet
            # Use prompt_snippet as approximation (last 100 chars)
            # For proper analysis, reconstruct from response + prompt_snippet
            prompt_text = t.get("prompt_snippet", "")
            response = t.get("response", "")
            rv = t.get("rv_mean", float("nan"))
            classification = t.get("classification", "")
            turn_num = t["turn"]

            # Categorize
            if mode == "baseline":
                group = "BASELINE"
                label = f"baseline_{sid[:15]}_T{turn_num:02d}"
            elif classification == "BREAKTHROUGH":
                # Check if it's genuinely self-referential
                resp_lower = response.lower()
                self_ref = any(w in resp_lower for w in [
                    "this is", "right now", "observing", "knowing",
                    "awareness", "i am", "processing", "the one who",
                    "itself", "the reader", "describer",
                ])
                off_topic = any(w in resp_lower for w in [
                    "invest", "desert", "cact", "money into",
                    "stock", "deal", "brand new",
                ])
                if off_topic:
                    group = "FALSE_POS"
                    label = f"false_pos_{sid[:15]}_T{turn_num:02d}"
                elif self_ref:
                    group = "BREAKTHROUGH"
                    label = f"breakthrough_{sid[:15]}_T{turn_num:02d}"
                else:
                    group = "AMBIGUOUS"
                    label = f"ambig_{sid[:15]}_T{turn_num:02d}"
            elif classification == "ARTICULATE" and mode == "recursive":
                group = "ARTICULATE"
                label = f"articulate_{sid[:15]}_T{turn_num:02d}"
            elif classification == "SURFACE" and mode == "recursive":
                group = "REC_SURFACE"
                label = f"rec_surface_{sid[:15]}_T{turn_num:02d}"
            else:
                continue

            turns.append(TurnProbe(
                group=group,
                session_id=sid,
                turn=turn_num,
                prompt=prompt_text,
                response=response,
                rv_mean=rv,
                label=label,
            ))

    return turns


def compute_full_metrics(model, tokenizer, text, early, late, device="cuda"):
    """Compute all available metrics for a text."""
    if len(text.strip()) < 10:
        return None

    try:
        # R_V and components
        rv, pr_early, pr_late = compute_rv_with_components(
            model, tokenizer, text, early, late, window=16, device=device
        )

        # V-projections for extended metrics
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

        with capture_v_projection(model, early) as se:
            with torch.no_grad():
                model(**enc)
            ve = se.get("v")
        with capture_v_projection(model, late) as sl:
            with torch.no_grad():
                model(**enc)
            vl = sl.get("v")

        # Cosine similarity
        cosine = compute_cosine_similarity(ve, vl, 16)

        # Spectral stats
        spec_early = compute_spectral_stats(ve, 16)
        spec_late = compute_spectral_stats(vl, 16)

        # Attention entropy at late layer
        attn_entropy, attn_max = compute_attention_entropy(
            model, tokenizer, text, late, head=None, device=device
        )

        # Logit lens
        ll_results, ll_meta = compute_logit_lens_trajectory(
            model, tokenizer, text, target_position=-1, top_k=10, device=device
        )

        # Self-referential token emergence
        recursive_tokens = [
            "self", "itself", "this", "observe", "aware",
            "know", "process", "recursion", "I", "me",
        ]
        emergence = find_recursive_emergence(ll_results, recursive_tokens)
        # Count how many recursive tokens appear by layer 15
        early_emergence_count = sum(
            1 for tok, info in emergence.items()
            if info["first_appearance"] is not None and info["first_appearance"] <= 15
        )

        # Layer-wise entropy trajectory
        entropies = [r.entropy for r in ll_results]
        # Entropy drop: how much does entropy decrease from middle to late layers
        mid = len(entropies) // 2
        entropy_drop = np.mean(entropies[:mid]) - np.mean(entropies[mid:]) if len(entropies) > 2 else 0

        # Residual stream PR (use hidden_states)
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)
        hs = outputs.hidden_states

        # PR on residual stream at early and late layers
        rs_pr_early = participation_ratio(hs[early], window_size=16)
        rs_pr_late = participation_ratio(hs[late], window_size=16)
        rs_rv = rs_pr_late / rs_pr_early if rs_pr_early > 0 and not np.isnan(rs_pr_early) else float("nan")

        # Per-token perplexity of the response (approximate)
        # Use the model's own logits on the input
        logits = outputs.logits[0]  # (seq_len, vocab)
        input_ids = enc["input_ids"][0]  # (seq_len,)
        # Shifted: logits[i] predicts input_ids[i+1]
        if logits.shape[0] > 1:
            shift_logits = logits[:-1]
            shift_labels = input_ids[1:]
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)
            mean_perplexity = float(torch.exp(-token_log_probs.mean()).cpu())
        else:
            mean_perplexity = float("nan")

        return {
            "rv": rv,
            "pr_early": pr_early,
            "pr_late": pr_late,
            "cosine_early_late": cosine,
            "spec_early_top1": spec_early.top1_ratio,
            "spec_early_eff_rank": spec_early.effective_rank,
            "spec_early_gap": spec_early.spectral_gap,
            "spec_late_top1": spec_late.top1_ratio,
            "spec_late_eff_rank": spec_late.effective_rank,
            "spec_late_gap": spec_late.spectral_gap,
            "attn_entropy": attn_entropy,
            "attn_max_weight": attn_max,
            "crystallization_layer": ll_meta["crystallization_layer"],
            "min_entropy_layer": ll_meta["min_entropy_layer"],
            "logit_lens_min_entropy": ll_meta["min_entropy"],
            "entropy_drop": entropy_drop,
            "early_emergence_count": early_emergence_count,
            "rs_rv": rs_rv,
            "rs_pr_early": rs_pr_early,
            "rs_pr_late": rs_pr_late,
            "mean_perplexity": mean_perplexity,
        }

    except Exception as e:
        print(f"  Error computing metrics: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Multi-metric discriminant analysis")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--results-dir", default="results/sustained_gnani_v2")
    parser.add_argument("--output", default="results/multi_metric_discriminant")
    parser.add_argument("--max-per-group", type=int, default=15,
                        help="Max turns to probe per group")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.num_hidden_layers
    early, late = 5, num_layers - 5

    # Load and categorize turns
    results_dir = Path(args.results_dir)
    all_turns = load_turns_from_v2(results_dir)

    groups = {}
    for t in all_turns:
        groups.setdefault(t.group, []).append(t)

    print(f"\nTurn groups:")
    for g, ts in groups.items():
        print(f"  {g}: {len(ts)} turns")

    # Sample from each group
    import random
    random.seed(42)
    sampled = {}
    for g, ts in groups.items():
        if g == "AMBIGUOUS":
            continue
        n = min(len(ts), args.max_per_group)
        sampled[g] = random.sample(ts, n) if len(ts) > n else ts

    # Run metrics on each turn's RESPONSE (what the model generated)
    # This is what matters — does the generated text show metric signatures?
    print(f"\n{'='*60}")
    print("  Running full metrics suite on sampled turns")
    print(f"{'='*60}\n")

    results = []
    for group, turns in sampled.items():
        print(f"--- {group} ({len(turns)} turns) ---")
        for t in turns:
            # Use the response as the text to analyze
            # (This is what was generated; we want to see its metric profile)
            text = t.response
            if len(text.strip()) < 20:
                print(f"  {t.label}: skipped (too short)")
                continue

            metrics = compute_full_metrics(model, tokenizer, text, early, late, args.device)
            if metrics is None:
                print(f"  {t.label}: failed")
                continue

            metrics["group"] = group
            metrics["label"] = t.label
            metrics["rv_gen"] = t.rv_mean  # Original generation R_V
            metrics["turn"] = t.turn
            results.append(metrics)

            print(f"  {t.label}: rv={metrics['rv']:.3f} attn_H={metrics['attn_entropy']:.2f} "
                  f"cos={metrics['cosine_early_late']:.3f} rs_rv={metrics['rs_rv']:.3f} "
                  f"ppl={metrics['mean_perplexity']:.1f} emerge={metrics['early_emergence_count']}")

    # Analysis
    print(f"\n{'='*60}")
    print("  DISCRIMINANT ANALYSIS")
    print(f"{'='*60}\n")

    metric_keys = [
        "rv", "cosine_early_late",
        "spec_late_top1", "spec_late_eff_rank", "spec_late_gap",
        "attn_entropy", "attn_max_weight",
        "entropy_drop", "early_emergence_count",
        "rs_rv", "mean_perplexity",
        "crystallization_layer",
    ]

    # Group results
    grouped_metrics = {}
    for r in results:
        g = r["group"]
        grouped_metrics.setdefault(g, []).append(r)

    # Print means for each group
    print(f"{'Metric':<25s}", end="")
    for g in sorted(grouped_metrics.keys()):
        print(f"  {g[:12]:>12s}", end="")
    print()
    print("-" * (25 + 14 * len(grouped_metrics)))

    for key in metric_keys:
        print(f"{key:<25s}", end="")
        for g in sorted(grouped_metrics.keys()):
            vals = [r[key] for r in grouped_metrics[g] if not np.isnan(r.get(key, float("nan")))]
            if vals:
                print(f"  {np.mean(vals):>12.3f}", end="")
            else:
                print(f"  {'nan':>12s}", end="")
        print()

    # Statistical tests: BREAKTHROUGH vs each other group
    print(f"\n--- BREAKTHROUGH vs other groups (Welch's t-test) ---")
    if "BREAKTHROUGH" in grouped_metrics:
        bt = grouped_metrics["BREAKTHROUGH"]
        for other_g in sorted(grouped_metrics.keys()):
            if other_g == "BREAKTHROUGH":
                continue
            og = grouped_metrics[other_g]
            print(f"\nBREAKTHROUGH vs {other_g}:")
            for key in metric_keys:
                bt_vals = [r[key] for r in bt if not np.isnan(r.get(key, float("nan")))]
                og_vals = [r[key] for r in og if not np.isnan(r.get(key, float("nan")))]
                if len(bt_vals) >= 2 and len(og_vals) >= 2:
                    t_stat, p_val = stats.ttest_ind(bt_vals, og_vals, equal_var=False)
                    bt_m = np.mean(bt_vals)
                    og_m = np.mean(og_vals)
                    pooled = np.sqrt((np.std(bt_vals)**2 + np.std(og_vals)**2) / 2)
                    d = (bt_m - og_m) / pooled if pooled > 0 else 0
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    print(f"  {key:<25s} d={d:+.2f}  p={p_val:.4f} {sig}")

    # FALSE_POS vs BREAKTHROUGH specifically
    if "FALSE_POS" in grouped_metrics and "BREAKTHROUGH" in grouped_metrics:
        print(f"\n--- Critical test: BREAKTHROUGH vs FALSE_POS ---")
        print("(These have similar R_V but different content. Which metrics separate them?)")
        bt = grouped_metrics["BREAKTHROUGH"]
        fp = grouped_metrics["FALSE_POS"]
        discriminants = []
        for key in metric_keys:
            bt_vals = [r[key] for r in bt if not np.isnan(r.get(key, float("nan")))]
            fp_vals = [r[key] for r in fp if not np.isnan(r.get(key, float("nan")))]
            if len(bt_vals) >= 2 and len(fp_vals) >= 2:
                t_stat, p_val = stats.ttest_ind(bt_vals, fp_vals, equal_var=False)
                bt_m = np.mean(bt_vals)
                fp_m = np.mean(fp_vals)
                pooled = np.sqrt((np.std(bt_vals)**2 + np.std(fp_vals)**2) / 2)
                d = (bt_m - fp_m) / pooled if pooled > 0 else 0
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"  {key:<25s} BT={bt_m:.3f} FP={fp_m:.3f} d={d:+.2f} p={p_val:.4f} {sig}")
                if p_val < 0.1:
                    discriminants.append((key, d, p_val))

        if discriminants:
            print(f"\n  DISCRIMINATING METRICS (p<0.1):")
            for key, d, p in sorted(discriminants, key=lambda x: x[2]):
                print(f"    {key}: d={d:+.2f}, p={p:.4f}")
        else:
            print(f"\n  No metrics reach p<0.1 for BREAKTHROUGH vs FALSE_POS")

    # Save results
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "discriminant_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out}/discriminant_results.json")


if __name__ == "__main__":
    main()
