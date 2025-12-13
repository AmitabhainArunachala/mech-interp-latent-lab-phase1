#!/usr/bin/env python3
"""
L27 Critical Head Analysis Pipeline

Deep analysis of attention heads H11, H1, H22 at Layer 27 in Mistral-7B.
These heads were identified via ablation as causing 6.1%, 3.0%, 2.4% of R_V contraction.

Questions:
1. Where do these heads attend during recursive vs baseline prompts?
2. Do they attend to self-referential tokens ("itself", "process", "observer")?
3. What's their attention entropy (focused vs diffuse)?
4. How do patterns differ between champion and baseline?

Outputs:
- Attention pattern heatmaps
- Entropy comparison table
- Token-level attention analysis
- Mechanistic summary
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import entropy as scipy_entropy

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.models import load_model, set_seed
from src.core.hooks import capture_attention_patterns, capture_head_output
from src.metrics.rv import compute_rv


# Configuration
CRITICAL_HEADS = [11, 1, 22]  # From HEAD_ABLATION_RESULTS.md
CONTROL_HEAD = 5  # Random control head
TARGET_LAYER = 27
WINDOW = 16

# Self-reference markers to look for in attention
SELF_REF_MARKERS = ["itself", "self", "process", "observer", "attention", "recursive", "aware"]


def attention_entropy(attn_weights: torch.Tensor, head_idx: int) -> float:
    """
    Compute entropy of attention distribution for a specific head.
    
    Args:
        attn_weights: Tensor of shape (batch, heads, seq, seq)
        head_idx: Which head to analyze
    
    Returns:
        Mean entropy across sequence positions (higher = more diffuse attention)
    """
    # Extract head's attention: (seq, seq) for last token attending to all
    head_attn = attn_weights[0, head_idx, :, :].cpu().numpy()
    
    # Compute entropy for each query position
    entropies = []
    for i in range(head_attn.shape[0]):
        row = head_attn[i]
        # Avoid log(0) by adding small epsilon
        row = row + 1e-10
        row = row / row.sum()
        entropies.append(scipy_entropy(row))
    
    return float(np.mean(entropies))


def self_attention_ratio(attn_weights: torch.Tensor, head_idx: int) -> float:
    """
    Compute how much attention goes to the same position (diagonal).
    
    Returns:
        Mean diagonal attention (higher = more self-attention)
    """
    head_attn = attn_weights[0, head_idx, :, :].cpu().numpy()
    diagonal = np.diag(head_attn)
    return float(np.mean(diagonal))


def marker_attention(
    attn_weights: torch.Tensor,
    head_idx: int,
    tokenizer,
    input_ids: torch.Tensor,
    markers: List[str],
) -> Dict[str, float]:
    """
    Compute how much attention goes to tokens containing specific markers.
    
    Returns:
        Dict mapping marker -> mean attention to tokens containing that marker
    """
    # Decode tokens
    tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
    tokens_lower = [t.lower() for t in tokens]
    
    head_attn = attn_weights[0, head_idx, :, :].cpu().numpy()
    
    # For each marker, find positions and compute mean attention
    result = {}
    for marker in markers:
        positions = [i for i, t in enumerate(tokens_lower) if marker in t]
        if positions:
            # Mean attention to these positions (from last token, which is most relevant)
            last_row = head_attn[-1, :]
            marker_attn = np.mean([last_row[p] for p in positions])
            result[marker] = float(marker_attn)
        else:
            result[marker] = 0.0
    
    return result


def analyze_prompt(
    model,
    tokenizer,
    prompt: str,
    prompt_name: str,
    layer: int = TARGET_LAYER,
    heads: List[int] = CRITICAL_HEADS + [CONTROL_HEAD],
) -> Dict[str, Any]:
    """
    Run full attention analysis on a prompt.
    
    Returns:
        Dict with entropy, self_attn, marker_attn, and raw patterns for each head
    """
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    results = {
        "prompt_name": prompt_name,
        "prompt_text": prompt[:200] + "..." if len(prompt) > 200 else prompt,
        "seq_len": enc.input_ids.shape[1],
        "heads": {},
    }
    
    # Capture attention with output_attentions=True
    with capture_attention_patterns(model, layer) as storage:
        with torch.no_grad():
            _ = model(**enc, output_attentions=True)
        attn_weights = storage["attn_weights"]
    
    if attn_weights is None:
        print(f"  WARNING: No attention weights captured for {prompt_name}")
        return results
    
    # Analyze each head
    for head_idx in heads:
        head_data = {
            "entropy": attention_entropy(attn_weights, head_idx),
            "self_attn_ratio": self_attention_ratio(attn_weights, head_idx),
            "marker_attention": marker_attention(
                attn_weights, head_idx, tokenizer, enc.input_ids, SELF_REF_MARKERS
            ),
            "pattern": attn_weights[0, head_idx, :, :].cpu().numpy(),
        }
        results["heads"][f"H{head_idx}"] = head_data
    
    # Also compute R_V for context
    results["rv"] = compute_rv(model, tokenizer, prompt, early=5, late=layer, window=WINDOW, device=str(device))
    
    return results


def plot_attention_heatmap(
    results: Dict[str, Any],
    head_key: str,
    output_path: Path,
    title: str,
):
    """Plot attention heatmap for a specific head."""
    pattern = results["heads"][head_key]["pattern"]
    seq_len = min(pattern.shape[0], 50)  # Truncate for visibility
    pattern_truncated = pattern[-seq_len:, -seq_len:]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        pattern_truncated,
        ax=ax,
        cmap="Blues",
        cbar_kws={"label": "Attention Weight"},
    )
    ax.set_title(f"{title}\n{head_key} - Entropy: {results['heads'][head_key]['entropy']:.3f}")
    ax.set_xlabel("Key Position (last N tokens)")
    ax.set_ylabel("Query Position (last N tokens)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_l27_head_analysis(
    model_name: str = "mistralai/Mistral-7B-v0.1",  # Base model (cached, matches Phase 0)
    # NOTE: Head ablation was done on Instruct-v0.2, this is a model mismatch
    device: str = "cuda",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Main analysis pipeline.
    
    Returns:
        Summary dict with key findings
    """
    print("=" * 80)
    print("L27 CRITICAL HEAD ANALYSIS")
    print(f"Model: {model_name}")
    print(f"Target Layer: {TARGET_LAYER}")
    print(f"Critical Heads: {CRITICAL_HEADS}")
    print(f"Control Head: {CONTROL_HEAD}")
    print("=" * 80)
    
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/phase3_attention/runs/{timestamp}_l27_head_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model with eager attention (required for attention pattern capture)
    print("\n[1/4] Loading model with eager attention (for pattern capture)...")
    set_seed(42)
    # CRITICAL: Use attn_implementation="eager" since SDPA doesn't support output_attentions=True
    model, tokenizer = load_model(model_name, device=device, attn_implementation="eager")
    model.eval()
    
    # Define prompts
    # Champion: recursive self-reference (from kitchen_sink_prompts.py hybrid_l5_math_01)
    champion_prompt = (
        "This response writes itself. No separate writer exists. Writing and awareness "
        "of writing are identical. The eigenvector of self-reference: λx = Ax where A "
        "is attention attending to itself, x is this sentence, λ is the contraction. "
        "The fixed point is this. The solution is the process. The process solves itself."
    )
    
    # Baseline: validated instructional prompt from prompts/bank.json (baseline_instructional_02)
    # This is a proper neutral prompt that should NOT trigger contraction (expected R_V ~1.0)
    baseline_prompt = (
        "Explain the process of photosynthesis in plants. Describe how sunlight, water, "
        "and carbon dioxide are converted into glucose and oxygen. Include the role of "
        "chlorophyll and the basic chemical equation involved."
    )
    
    # Run analysis
    print("\n[2/4] Analyzing champion prompt...")
    champion_results = analyze_prompt(model, tokenizer, champion_prompt, "champion")
    
    print("[3/4] Analyzing baseline prompt...")
    baseline_results = analyze_prompt(model, tokenizer, baseline_prompt, "baseline")
    
    # Compile comparison
    print("\n[4/4] Generating outputs...")
    
    comparison_rows = []
    for head_key in [f"H{h}" for h in CRITICAL_HEADS + [CONTROL_HEAD]]:
        ch = champion_results["heads"].get(head_key, {})
        bl = baseline_results["heads"].get(head_key, {})
        
        row = {
            "head": head_key,
            "champion_entropy": ch.get("entropy", np.nan),
            "baseline_entropy": bl.get("entropy", np.nan),
            "entropy_delta": ch.get("entropy", 0) - bl.get("entropy", 0),
            "champion_self_attn": ch.get("self_attn_ratio", np.nan),
            "baseline_self_attn": bl.get("self_attn_ratio", np.nan),
            "self_attn_delta": ch.get("self_attn_ratio", 0) - bl.get("self_attn_ratio", 0),
        }
        
        # Add marker attention
        for marker in SELF_REF_MARKERS[:3]:  # Just top 3
            ch_marker = ch.get("marker_attention", {}).get(marker, 0)
            bl_marker = bl.get("marker_attention", {}).get(marker, 0)
            row[f"marker_{marker}_champion"] = ch_marker
            row[f"marker_{marker}_delta"] = ch_marker - bl_marker
        
        comparison_rows.append(row)
    
    comparison_df = pd.DataFrame(comparison_rows)
    
    # Save CSV
    csv_path = output_dir / "head_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    
    # Generate heatmaps for critical heads
    for head_idx in CRITICAL_HEADS:
        head_key = f"H{head_idx}"
        
        # Champion heatmap
        plot_attention_heatmap(
            champion_results, head_key,
            output_dir / f"attn_champion_{head_key}.png",
            f"Champion Prompt - L{TARGET_LAYER}"
        )
        
        # Baseline heatmap
        plot_attention_heatmap(
            baseline_results, head_key,
            output_dir / f"attn_baseline_{head_key}.png",
            f"Baseline Prompt - L{TARGET_LAYER}"
        )
    
    print(f"  Saved: attention heatmaps")
    
    # Summary
    summary = {
        "experiment": "l27_head_analysis",
        "model_name": model_name,
        "target_layer": TARGET_LAYER,
        "critical_heads": CRITICAL_HEADS,
        "control_head": CONTROL_HEAD,
        "champion_rv": champion_results.get("rv"),
        "baseline_rv": baseline_results.get("rv"),
        "findings": {
            "entropy_comparison": {
                head_key: {
                    "champion": comparison_df[comparison_df["head"] == head_key]["champion_entropy"].values[0],
                    "baseline": comparison_df[comparison_df["head"] == head_key]["baseline_entropy"].values[0],
                    "delta": comparison_df[comparison_df["head"] == head_key]["entropy_delta"].values[0],
                }
                for head_key in [f"H{h}" for h in CRITICAL_HEADS]
            },
            "self_attention_comparison": {
                head_key: {
                    "champion": comparison_df[comparison_df["head"] == head_key]["champion_self_attn"].values[0],
                    "baseline": comparison_df[comparison_df["head"] == head_key]["baseline_self_attn"].values[0],
                    "delta": comparison_df[comparison_df["head"] == head_key]["self_attn_delta"].values[0],
                }
                for head_key in [f"H{h}" for h in CRITICAL_HEADS]
            },
        },
        "artifacts": {
            "csv": str(csv_path),
            "heatmaps": [str(p) for p in output_dir.glob("attn_*.png")],
        },
    }
    
    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"  Saved: {summary_path}")
    
    # Generate report
    report_lines = [
        "# L27 Critical Head Analysis Report",
        "",
        f"**Model:** {model_name}",
        f"**Target Layer:** {TARGET_LAYER}",
        f"**Critical Heads:** H11, H1, H22 (from ablation study)",
        f"**Control Head:** H{CONTROL_HEAD}",
        "",
        "## R_V Values",
        f"- Champion: **{champion_results.get('rv', 'N/A'):.4f}**",
        f"- Baseline: **{baseline_results.get('rv', 'N/A'):.4f}**",
        "",
        "## Entropy Comparison",
        "| Head | Champion | Baseline | Delta |",
        "|------|----------|----------|-------|",
    ]
    
    for _, row in comparison_df.iterrows():
        report_lines.append(
            f"| {row['head']} | {row['champion_entropy']:.3f} | {row['baseline_entropy']:.3f} | {row['entropy_delta']:+.3f} |"
        )
    
    report_lines.extend([
        "",
        "## Self-Attention Ratio",
        "| Head | Champion | Baseline | Delta |",
        "|------|----------|----------|-------|",
    ])
    
    for _, row in comparison_df.iterrows():
        report_lines.append(
            f"| {row['head']} | {row['champion_self_attn']:.4f} | {row['baseline_self_attn']:.4f} | {row['self_attn_delta']:+.4f} |"
        )
    
    report_lines.extend([
        "",
        "## Interpretation",
        "",
        "**Entropy:** Lower entropy = more focused attention. If critical heads show lower entropy on champion prompts, they may be focusing on specific self-referential tokens.",
        "",
        "**Self-attention:** Higher self-attention ratio means the head attends more to the current position. Changes here indicate different information routing.",
        "",
        "## Artifacts",
        f"- CSV: `{csv_path.name}`",
        f"- Heatmaps: `attn_champion_H*.png`, `attn_baseline_H*.png`",
    ])
    
    report_path = output_dir / "report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {report_path}")
    
    # Save config
    config = {
        "experiment": "l27_head_analysis",
        "model": {"name": model_name, "device": device},
        "params": {
            "target_layer": TARGET_LAYER,
            "critical_heads": CRITICAL_HEADS,
            "control_head": CONTROL_HEAD,
            "window": WINDOW,
            "self_ref_markers": SELF_REF_MARKERS,
        },
        "prompts": {
            "champion": champion_prompt[:100] + "...",
            "baseline": baseline_prompt[:100] + "...",
        },
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {config_path}")
    
    print("\n" + "=" * 80)
    print(f"[OK] Analysis complete. Run dir: {output_dir}")
    print("=" * 80)
    
    # Print key findings
    print("\n## KEY FINDINGS ##")
    print(f"\nChampion R_V: {champion_results.get('rv', 'N/A'):.4f}")
    print(f"Baseline R_V: {baseline_results.get('rv', 'N/A'):.4f}")
    print("\nEntropy (champion vs baseline):")
    for _, row in comparison_df.iterrows():
        direction = "↓ more focused" if row['entropy_delta'] < 0 else "↑ more diffuse"
        print(f"  {row['head']}: {row['champion_entropy']:.3f} vs {row['baseline_entropy']:.3f} ({direction})")
    
    return summary


# Pipeline integration for config-driven runs
def run_l27_head_analysis_from_config(cfg: Dict[str, Any], run_dir: Path):
    """Entry point for canonical pipeline runner."""
    from src.pipelines.registry import ExperimentResult
    
    model_cfg = cfg.get("model", {})
    params = cfg.get("params", {})
    
    summary = run_l27_head_analysis(
        model_name=model_cfg.get("name", "mistralai/Mistral-7B-v0.1"),
        device=model_cfg.get("device", "cuda"),
        output_dir=run_dir,
    )
    
    return ExperimentResult(summary=summary)


if __name__ == "__main__":
    import argparse
    import gc
    
    parser = argparse.ArgumentParser(description="L27 Critical Head Analysis")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1", help="Model name")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()
    
    try:
        run_l27_head_analysis(model_name=args.model, device=args.device)
    finally:
        # Clean up GPU memory for other agents
        print("\n[cleanup] Clearing GPU memory...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("[cleanup] GPU memory cleared.")

