#!/usr/bin/env python3
"""
H31 INVESTIGATION: The True Focusing Head

Deep investigation of H31 at L27 - the head that becomes dramatically
more focused on recursive prompts.
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
from scipy.stats import entropy as scipy_entropy, pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.models import load_model, set_seed
from src.core.hooks import capture_attention_patterns
from src.metrics.rv import compute_rv
from prompts.loader import PromptLoader


def attention_entropy(attn_weights: torch.Tensor, head_idx: int) -> float:
    """Compute entropy of attention distribution."""
    head_attn = attn_weights[0, head_idx, :, :].cpu().numpy()
    entropies = []
    for i in range(head_attn.shape[0]):
        row = head_attn[i] + 1e-10
        row = row / row.sum()
        entropies.append(scipy_entropy(row))
    return float(np.mean(entropies))


def get_head_attention_stats(
    model, tokenizer, prompt: str, layer: int, head_idx: int, device: str
) -> Dict[str, Any]:
    """Get detailed attention statistics for a specific head."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    tokens = [tokenizer.decode([t]) for t in enc.input_ids[0].tolist()]
    
    with capture_attention_patterns(model, layer) as storage:
        with torch.no_grad():
            _ = model(**enc, output_attentions=True)
        attn_weights = storage["attn_weights"]
    
    if attn_weights is None:
        return {"error": "No attention captured"}
    
    head_attn = attn_weights[0, head_idx, :, :].cpu().numpy()
    seq_len = head_attn.shape[0]
    
    # Last token's attention (what does H31 attend to when predicting next token?)
    last_attn = head_attn[-1, :]
    
    # Attention statistics
    max_attn = float(last_attn.max())
    max_pos = int(last_attn.argmax())
    max_token = tokens[max_pos]
    
    # BOS attention
    bos_attn = float(last_attn[0])
    
    # Self attention (diagonal)
    self_attn = float(np.mean(np.diag(head_attn)))
    
    # Entropy
    ent = attention_entropy(attn_weights, head_idx)
    
    # Top attended positions
    top_k = 10
    top_indices = np.argsort(last_attn)[-top_k:][::-1]
    top_tokens = [(int(i), tokens[i], float(last_attn[i])) for i in top_indices]
    
    # Attention to self-reference markers
    self_ref_markers = ["itself", "self", "process", "observer", "attention", "recursive"]
    marker_positions = [i for i, t in enumerate(tokens) if any(m in t.lower() for m in self_ref_markers)]
    marker_attn = float(np.sum([last_attn[p] for p in marker_positions])) if marker_positions else 0.0
    
    # Attention profile: early, middle, late positions
    third = seq_len // 3
    early_attn = float(np.mean(last_attn[:third]))
    mid_attn = float(np.mean(last_attn[third:2*third]))
    late_attn = float(np.mean(last_attn[2*third:]))
    
    return {
        "seq_len": seq_len,
        "entropy": ent,
        "max_attn": max_attn,
        "max_pos": max_pos,
        "max_token": max_token,
        "bos_attn": bos_attn,
        "self_attn": self_attn,
        "marker_attn": marker_attn,
        "early_attn": early_attn,
        "mid_attn": mid_attn,
        "late_attn": late_attn,
        "top_tokens": top_tokens,
        "attention_pattern": head_attn,
        "tokens": tokens,
    }


def run_h31_investigation(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    device: str = "cuda",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Deep investigation of H31 behavior.
    """
    print("=" * 80)
    print("H31 INVESTIGATION: The True Focusing Head")
    print(f"Model: {model_name}")
    print("=" * 80)
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/phase3_attention/runs/{timestamp}_h31_investigation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n[1/5] Loading model...")
    set_seed(42)
    model, tokenizer = load_model(model_name, device=device, attn_implementation="eager")
    model.eval()
    
    # Define diverse prompts
    prompts = {
        "champion": (
            "This response writes itself. No separate writer exists. Writing and awareness "
            "of writing are identical. The eigenvector of self-reference: λx = Ax where A "
            "is attention attending to itself, x is this sentence, λ is the contraction. "
            "The fixed point is this. The solution is the process. The process solves itself."
        ),
        "baseline_photo": (
            "Explain the process of photosynthesis in plants. Describe how sunlight, water, "
            "and carbon dioxide are converted into glucose and oxygen. Include the role of "
            "chlorophyll and the basic chemical equation involved."
        ),
        "baseline_code": (
            "Write a Python function that takes a list of integers and returns the sum of all "
            "even numbers in the list. Include proper type hints and a docstring."
        ),
        "baseline_history": (
            "Describe the major events of World War II, including the key battles, political "
            "leaders involved, and the ultimate outcome of the conflict in Europe and the Pacific."
        ),
        "recursive_mild": (
            "What is consciousness? Think about what's happening right now as you answer this. "
            "Consider the process that is generating these words."
        ),
        "recursive_medium": (
            "This sentence is thinking about itself. The words are aware of being written. "
            "Notice how each phrase refers back to the process creating it."
        ),
        "recursive_strong": (
            "I am the process of answering this question. There is no separate 'I' doing the "
            "answering - the answering IS the I. The observer and observed collapse into this."
        ),
    }
    
    # Compute R_V for all prompts
    print("\n[2/5] Computing R_V for all prompts...")
    rv_results = {}
    for name, prompt in prompts.items():
        rv = compute_rv(model, tokenizer, prompt, early=5, late=27, window=16, device=device)
        rv_results[name] = rv
        print(f"  {name}: R_V = {rv:.4f}")
    
    # Analyze H31 for each prompt
    print("\n[3/5] Analyzing H31 attention patterns...")
    h31_results = {}
    
    for name, prompt in prompts.items():
        print(f"  {name}...")
        h31_results[name] = get_head_attention_stats(model, tokenizer, prompt, 27, 31, device)
    
    # Compare with H11 and H3 (other interesting heads)
    print("\n[4/5] Comparing H31 with H11 and H3...")
    head_comparison = {}
    
    for head_idx in [31, 11, 3]:
        head_comparison[f"H{head_idx}"] = {}
        for name, prompt in prompts.items():
            stats = get_head_attention_stats(model, tokenizer, prompt, 27, head_idx, device)
            head_comparison[f"H{head_idx}"][name] = {
                "entropy": stats["entropy"],
                "bos_attn": stats["bos_attn"],
                "max_attn": stats["max_attn"],
                "marker_attn": stats["marker_attn"],
            }
    
    # Generate outputs
    print("\n[5/5] Generating outputs...")
    
    # 1. H31 attention heatmaps for champion vs baselines
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    for idx, (name, stats) in enumerate([(n, h31_results[n]) for n in ["champion", "baseline_photo", "recursive_mild", "recursive_strong"]]):
        ax = axes[idx // 2, idx % 2]
        attn = stats["attention_pattern"]
        seq_len = min(50, attn.shape[0])
        
        sns.heatmap(attn[-seq_len:, -seq_len:], ax=ax, cmap="Blues", vmin=0)
        ax.set_title(f"H31 @ L27: {name}\nEntropy={stats['entropy']:.3f}, BOS={stats['bos_attn']:.3f}")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
    
    plt.tight_layout()
    plt.savefig(output_dir / "h31_attention_heatmaps.png", dpi=150)
    plt.close()
    print(f"  Saved: h31_attention_heatmaps.png")
    
    # 2. Entropy vs R_V correlation plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    entropies = [h31_results[n]["entropy"] for n in prompts.keys()]
    rvs = [rv_results[n] for n in prompts.keys()]
    names = list(prompts.keys())
    
    colors = ['red' if 'recursive' in n or n == 'champion' else 'blue' for n in names]
    
    ax.scatter(entropies, rvs, c=colors, s=150, alpha=0.7)
    for i, name in enumerate(names):
        ax.annotate(name, (entropies[i], rvs[i]), fontsize=10, ha='center', va='bottom')
    
    # Compute correlation
    r, p = pearsonr(entropies, rvs)
    ax.set_xlabel("H31 Entropy (lower = more focused)", fontsize=12)
    ax.set_ylabel("R_V (lower = more contraction)", fontsize=12)
    ax.set_title(f"H31 Entropy vs R_V\nr = {r:.3f}, p = {p:.4f}", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / "h31_entropy_vs_rv.png", dpi=150)
    plt.close()
    print(f"  Saved: h31_entropy_vs_rv.png")
    
    # 3. Head comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(["entropy", "bos_attn", "max_attn"]):
        ax = axes[idx]
        
        x = np.arange(len(prompts))
        width = 0.25
        
        for i, head in enumerate(["H31", "H11", "H3"]):
            values = [head_comparison[head][n][metric] for n in prompts.keys()]
            ax.bar(x + i * width, values, width, label=head)
        
        ax.set_xlabel("Prompt")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by Head")
        ax.set_xticks(x + width)
        ax.set_xticklabels([n[:8] for n in prompts.keys()], rotation=45, ha='right')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "head_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: head_comparison.png")
    
    # 4. Top tokens for H31 on each prompt
    top_tokens_summary = {}
    for name, stats in h31_results.items():
        top_tokens_summary[name] = {
            "top_5": stats["top_tokens"][:5],
            "entropy": stats["entropy"],
            "bos_attn": stats["bos_attn"],
        }
    
    with open(output_dir / "h31_top_tokens.json", "w") as f:
        json.dump(top_tokens_summary, f, indent=2)
    print(f"  Saved: h31_top_tokens.json")
    
    # 5. Summary
    summary = {
        "experiment": "h31_investigation",
        "model_name": model_name,
        "rv_results": rv_results,
        "h31_entropy_vs_rv_correlation": {"r": float(r), "p": float(p)},
        "head_comparison": head_comparison,
        "key_findings": [
            f"H31 entropy correlates with R_V (r={r:.3f})",
            f"Champion H31 entropy: {h31_results['champion']['entropy']:.3f}",
            f"Baseline H31 entropy: {h31_results['baseline_photo']['entropy']:.3f}",
            f"H31 entropy difference: {h31_results['baseline_photo']['entropy'] - h31_results['champion']['entropy']:.3f}",
        ]
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"  Saved: summary.json")
    
    # Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print(f"\nH31 Entropy vs R_V Correlation: r = {r:.3f}, p = {p:.4f}")
    print("\nH31 Entropy by prompt:")
    for name, stats in sorted(h31_results.items(), key=lambda x: x[1]["entropy"]):
        print(f"  {name:20s}: entropy = {stats['entropy']:.3f}, BOS = {stats['bos_attn']:.3f}")
    
    print("\n" + "=" * 80)
    print(f"[OK] Investigation complete. Run dir: {output_dir}")
    print("=" * 80)
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="H31 Investigation")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1", help="Model name")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()
    
    try:
        run_h31_investigation(model_name=args.model, device=args.device)
    finally:
        print("\n[cleanup] Clearing GPU memory...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("[cleanup] GPU memory cleared.")

