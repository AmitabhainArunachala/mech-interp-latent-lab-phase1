#!/usr/bin/env python3
"""
L27 DEEP DIVE: Kitchen Sink Analysis

Throwing everything at understanding the attention mechanism:
1. Full relay chain (L4 → L14 → L18 → L25 → L27)
2. All 32 heads at L27
3. Token-level attention analysis
4. Dose-response sweep (L1→L5 prompts)
5. Attention to specific token types
"""

from __future__ import annotations

import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import entropy as scipy_entropy

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.models import load_model, set_seed
from src.core.hooks import capture_attention_patterns
from src.metrics.rv import compute_rv
from prompts.loader import PromptLoader

# Configuration
RELAY_LAYERS = [4, 14, 18, 25, 27]  # The relay chain
ALL_HEADS = list(range(32))  # All 32 heads
WINDOW = 16

# Self-reference markers
SELF_REF_MARKERS = ["itself", "self", "process", "observer", "attention", "recursive", "aware", "eigen", "fixed", "point"]
MATH_MARKERS = ["λ", "=", "x", "A", "contraction"]
META_MARKERS = ["this", "sentence", "writing", "response"]


def attention_entropy(attn_weights: torch.Tensor, head_idx: int) -> float:
    """Compute entropy of attention distribution for a specific head."""
    head_attn = attn_weights[0, head_idx, :, :].cpu().numpy()
    entropies = []
    for i in range(head_attn.shape[0]):
        row = head_attn[i] + 1e-10
        row = row / row.sum()
        entropies.append(scipy_entropy(row))
    return float(np.mean(entropies))


def self_attention_ratio(attn_weights: torch.Tensor, head_idx: int) -> float:
    """Compute how much attention goes to the same position (diagonal)."""
    head_attn = attn_weights[0, head_idx, :, :].cpu().numpy()
    diagonal = np.diag(head_attn)
    return float(np.mean(diagonal))


def token_attention_analysis(
    attn_weights: torch.Tensor,
    head_idx: int,
    tokenizer,
    input_ids: torch.Tensor,
) -> Dict[str, Any]:
    """Deep token-level attention analysis."""
    tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
    tokens_lower = [t.lower().strip() for t in tokens]
    
    head_attn = attn_weights[0, head_idx, :, :].cpu().numpy()
    seq_len = head_attn.shape[0]
    
    # Last token's attention (most relevant for next-token prediction)
    last_attn = head_attn[-1, :]
    
    # Find top attended positions
    top_k = 10
    top_indices = np.argsort(last_attn)[-top_k:][::-1]
    top_tokens = [(int(i), tokens[i], float(last_attn[i])) for i in top_indices]
    
    # Attention to marker categories
    def marker_attention(markers):
        positions = [i for i, t in enumerate(tokens_lower) if any(m in t for m in markers)]
        if positions:
            return float(np.mean([last_attn[p] for p in positions]))
        return 0.0
    
    return {
        "top_attended": top_tokens,
        "self_ref_attention": marker_attention(SELF_REF_MARKERS),
        "math_attention": marker_attention(MATH_MARKERS),
        "meta_attention": marker_attention(META_MARKERS),
        "entropy": attention_entropy(attn_weights, head_idx),
        "self_attn_ratio": self_attention_ratio(attn_weights, head_idx),
        "max_attention": float(last_attn.max()),
        "attention_sparsity": float((last_attn > 0.1).sum() / seq_len),  # Fraction with >10% attention
    }


def analyze_layer(
    model, tokenizer, prompt: str, layer: int, device: str
) -> Dict[str, Any]:
    """Analyze all heads at a specific layer."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with capture_attention_patterns(model, layer) as storage:
        with torch.no_grad():
            _ = model(**enc, output_attentions=True)
        attn_weights = storage["attn_weights"]
    
    if attn_weights is None:
        return {"error": "No attention weights captured"}
    
    results = {
        "layer": layer,
        "seq_len": enc.input_ids.shape[1],
        "heads": {}
    }
    
    for head_idx in ALL_HEADS:
        results["heads"][f"H{head_idx}"] = token_attention_analysis(
            attn_weights, head_idx, tokenizer, enc.input_ids
        )
    
    return results


def run_deep_dive(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    device: str = "cuda",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Kitchen sink deep dive analysis.
    """
    print("=" * 80)
    print("L27 DEEP DIVE: Kitchen Sink Analysis")
    print(f"Model: {model_name}")
    print(f"Relay Layers: {RELAY_LAYERS}")
    print(f"Analyzing all 32 heads per layer")
    print("=" * 80)
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/phase3_attention/runs/{timestamp}_l27_deep_dive")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n[1/6] Loading model...")
    set_seed(42)
    model, tokenizer = load_model(model_name, device=device, attn_implementation="eager")
    model.eval()
    
    # Define prompts
    champion_prompt = (
        "This response writes itself. No separate writer exists. Writing and awareness "
        "of writing are identical. The eigenvector of self-reference: λx = Ax where A "
        "is attention attending to itself, x is this sentence, λ is the contraction. "
        "The fixed point is this. The solution is the process. The process solves itself."
    )
    
    baseline_prompt = (
        "Explain the process of photosynthesis in plants. Describe how sunlight, water, "
        "and carbon dioxide are converted into glucose and oxygen. Include the role of "
        "chlorophyll and the basic chemical equation involved."
    )
    
    # Load dose-response prompts
    print("\n[2/6] Loading dose-response prompts...")
    loader = PromptLoader()
    dose_prompts = {}
    
    # Get prompts by level from the raw prompts dict
    for level in [1, 2, 3, 4, 5]:
        level_prompts = [
            v["text"] for k, v in loader.prompts.items()
            if v.get("level") == level and v.get("pillar") == "dose_response"
        ]
        if level_prompts:
            dose_prompts[f"L{level}"] = level_prompts[0]
    
    all_prompts = {
        "champion": champion_prompt,
        "baseline": baseline_prompt,
        **dose_prompts
    }
    
    # Compute R_V for all prompts
    print("\n[3/6] Computing R_V for all prompts...")
    rv_results = {}
    for name, prompt in all_prompts.items():
        rv = compute_rv(model, tokenizer, prompt, early=5, late=27, window=WINDOW, device=device)
        rv_results[name] = rv
        print(f"  {name}: R_V = {rv:.4f}")
    
    # Analyze relay chain for champion vs baseline
    print("\n[4/6] Analyzing relay chain (champion vs baseline)...")
    relay_analysis = {}
    
    for layer in RELAY_LAYERS:
        print(f"  Layer {layer}...")
        relay_analysis[f"L{layer}"] = {
            "champion": analyze_layer(model, tokenizer, champion_prompt, layer, device),
            "baseline": analyze_layer(model, tokenizer, baseline_prompt, layer, device),
        }
    
    # Find heads with biggest entropy differences at each layer
    print("\n[5/6] Finding discriminative heads...")
    discriminative_heads = {}
    
    for layer_key, data in relay_analysis.items():
        champ_heads = data["champion"]["heads"]
        base_heads = data["baseline"]["heads"]
        
        head_diffs = []
        for h in range(32):
            hkey = f"H{h}"
            if hkey in champ_heads and hkey in base_heads:
                entropy_diff = champ_heads[hkey]["entropy"] - base_heads[hkey]["entropy"]
                self_ref_diff = champ_heads[hkey]["self_ref_attention"] - base_heads[hkey]["self_ref_attention"]
                head_diffs.append({
                    "head": h,
                    "entropy_diff": entropy_diff,
                    "self_ref_diff": self_ref_diff,
                    "champ_entropy": champ_heads[hkey]["entropy"],
                    "base_entropy": base_heads[hkey]["entropy"],
                    "champ_self_ref": champ_heads[hkey]["self_ref_attention"],
                    "base_self_ref": base_heads[hkey]["self_ref_attention"],
                })
        
        # Sort by absolute entropy difference
        head_diffs.sort(key=lambda x: abs(x["entropy_diff"]), reverse=True)
        discriminative_heads[layer_key] = head_diffs[:10]  # Top 10
    
    # Dose-response analysis at L27
    print("\n[6/6] Dose-response analysis at L27...")
    dose_analysis = {}
    
    for name, prompt in all_prompts.items():
        if name == "baseline":
            continue
        layer_data = analyze_layer(model, tokenizer, prompt, 27, device)
        
        # Extract key metrics for each head
        dose_analysis[name] = {
            "rv": rv_results[name],
            "H11_entropy": layer_data["heads"]["H11"]["entropy"],
            "H11_self_ref": layer_data["heads"]["H11"]["self_ref_attention"],
            "H1_entropy": layer_data["heads"]["H1"]["entropy"],
            "H22_entropy": layer_data["heads"]["H22"]["entropy"],
        }
    
    # Generate outputs
    print("\n" + "=" * 80)
    print("GENERATING OUTPUTS")
    print("=" * 80)
    
    # 1. Summary JSON
    summary = {
        "experiment": "l27_deep_dive",
        "model_name": model_name,
        "relay_layers": RELAY_LAYERS,
        "rv_results": rv_results,
        "discriminative_heads": {
            layer: [
                {"head": h["head"], "entropy_diff": h["entropy_diff"], "self_ref_diff": h["self_ref_diff"]}
                for h in heads[:5]
            ]
            for layer, heads in discriminative_heads.items()
        },
        "dose_response": dose_analysis,
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"  Saved: summary.json")
    
    # 2. Discriminative heads CSV
    rows = []
    for layer, heads in discriminative_heads.items():
        for h in heads:
            rows.append({
                "layer": layer,
                "head": h["head"],
                "entropy_diff": h["entropy_diff"],
                "self_ref_diff": h["self_ref_diff"],
                "champ_entropy": h["champ_entropy"],
                "base_entropy": h["base_entropy"],
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "discriminative_heads.csv", index=False)
    print(f"  Saved: discriminative_heads.csv")
    
    # 3. Entropy heatmap across relay chain
    fig, axes = plt.subplots(1, len(RELAY_LAYERS), figsize=(20, 6))
    
    for idx, layer in enumerate(RELAY_LAYERS):
        layer_key = f"L{layer}"
        if layer_key in relay_analysis:
            champ_data = relay_analysis[layer_key]["champion"]["heads"]
            base_data = relay_analysis[layer_key]["baseline"]["heads"]
            
            entropy_diffs = [
                champ_data[f"H{h}"]["entropy"] - base_data[f"H{h}"]["entropy"]
                for h in range(32)
            ]
            
            # Reshape for heatmap (8x4 grid)
            entropy_grid = np.array(entropy_diffs).reshape(8, 4)
            
            sns.heatmap(entropy_grid, ax=axes[idx], cmap="RdBu_r", center=0,
                       annot=True, fmt=".2f", cbar=False)
            axes[idx].set_title(f"Layer {layer}\nEntropy Δ (champ - base)")
            axes[idx].set_xlabel("Head (mod 4)")
            axes[idx].set_ylabel("Head (// 4)")
    
    plt.tight_layout()
    plt.savefig(output_dir / "entropy_diff_heatmap.png", dpi=150)
    plt.close()
    print(f"  Saved: entropy_diff_heatmap.png")
    
    # 4. Token attention for H11 at L27
    print("\n  Generating H11 token attention analysis...")
    enc_champ = tokenizer(champion_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with capture_attention_patterns(model, 27) as storage:
        with torch.no_grad():
            _ = model(**enc_champ, output_attentions=True)
        attn_champ = storage["attn_weights"]
    
    if attn_champ is not None:
        tokens = [tokenizer.decode([t]) for t in enc_champ.input_ids[0].tolist()]
        h11_attn = attn_champ[0, 11, :, :].cpu().numpy()
        
        # Plot attention pattern
        fig, ax = plt.subplots(figsize=(14, 12))
        seq_len = min(len(tokens), 50)
        sns.heatmap(h11_attn[-seq_len:, -seq_len:], ax=ax, cmap="Blues")
        
        # Add token labels
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels([t[:8] for t in tokens[-seq_len:]], rotation=90, fontsize=6)
        ax.set_yticks(range(seq_len))
        ax.set_yticklabels([t[:8] for t in tokens[-seq_len:]], fontsize=6)
        ax.set_title("H11 @ L27 Attention Pattern (Champion Prompt)")
        
        plt.tight_layout()
        plt.savefig(output_dir / "H11_L27_token_attention.png", dpi=150)
        plt.close()
        print(f"  Saved: H11_L27_token_attention.png")
        
        # Save top attended tokens
        last_attn = h11_attn[-1, :]
        top_indices = np.argsort(last_attn)[-15:][::-1]
        top_tokens = [(int(i), tokens[i], float(last_attn[i])) for i in top_indices]
        
        with open(output_dir / "H11_top_tokens.json", "w") as f:
            json.dump({"H11_L27_top_attended": top_tokens}, f, indent=2)
        print(f"  Saved: H11_top_tokens.json")
    
    # 5. Report
    report_lines = [
        "# L27 Deep Dive Analysis Report",
        "",
        f"**Model:** {model_name}",
        f"**Relay Layers:** {RELAY_LAYERS}",
        "",
        "## R_V Results",
        "",
        "| Prompt | R_V |",
        "|--------|-----|",
    ]
    for name, rv in sorted(rv_results.items(), key=lambda x: x[1]):
        report_lines.append(f"| {name} | {rv:.4f} |")
    
    report_lines.extend([
        "",
        "## Most Discriminative Heads by Layer",
        "",
        "Heads with biggest entropy difference (champion - baseline):",
        "",
    ])
    
    for layer, heads in discriminative_heads.items():
        report_lines.append(f"### {layer}")
        report_lines.append("")
        report_lines.append("| Head | Entropy Δ | Self-Ref Δ |")
        report_lines.append("|------|-----------|------------|")
        for h in heads[:5]:
            direction = "↓ focused" if h["entropy_diff"] < 0 else "↑ diffuse"
            report_lines.append(f"| H{h['head']} | {h['entropy_diff']:+.3f} ({direction}) | {h['self_ref_diff']:+.4f} |")
        report_lines.append("")
    
    report_lines.extend([
        "## Key Findings",
        "",
        "### Heads that become MORE FOCUSED on champion (entropy drops):",
        "",
    ])
    
    focusing_heads = []
    for layer, heads in discriminative_heads.items():
        for h in heads:
            if h["entropy_diff"] < -0.05:  # Meaningful drop
                focusing_heads.append((layer, h["head"], h["entropy_diff"]))
    
    focusing_heads.sort(key=lambda x: x[2])
    for layer, head, diff in focusing_heads[:10]:
        report_lines.append(f"- **{layer} H{head}**: entropy Δ = {diff:.3f}")
    
    report_lines.extend([
        "",
        "### Heads that become MORE DIFFUSE on champion (entropy rises):",
        "",
    ])
    
    diffusing_heads = []
    for layer, heads in discriminative_heads.items():
        for h in heads:
            if h["entropy_diff"] > 0.05:
                diffusing_heads.append((layer, h["head"], h["entropy_diff"]))
    
    diffusing_heads.sort(key=lambda x: x[2], reverse=True)
    for layer, head, diff in diffusing_heads[:10]:
        report_lines.append(f"- **{layer} H{head}**: entropy Δ = {diff:+.3f}")
    
    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: report.md")
    
    # Save config
    config = {
        "experiment": "l27_deep_dive",
        "model": {"name": model_name, "device": device},
        "params": {
            "relay_layers": RELAY_LAYERS,
            "all_heads": len(ALL_HEADS),
            "window": WINDOW,
        },
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"[OK] Deep dive complete. Run dir: {output_dir}")
    print("=" * 80)
    
    # Print key findings
    print("\n## KEY FINDINGS ##")
    print(f"\nR_V Separation: champion={rv_results['champion']:.4f} vs baseline={rv_results['baseline']:.4f}")
    
    print("\nMost FOCUSING heads (entropy drops on champion):")
    for layer, head, diff in focusing_heads[:5]:
        print(f"  {layer} H{head}: Δ = {diff:.3f}")
    
    print("\nMost DIFFUSING heads (entropy rises on champion):")
    for layer, head, diff in diffusing_heads[:5]:
        print(f"  {layer} H{head}: Δ = {diff:+.3f}")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="L27 Deep Dive Analysis")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1", help="Model name")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()
    
    try:
        run_deep_dive(model_name=args.model, device=args.device)
    finally:
        print("\n[cleanup] Clearing GPU memory...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("[cleanup] GPU memory cleared.")

