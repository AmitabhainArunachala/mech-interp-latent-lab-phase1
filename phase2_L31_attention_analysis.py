"""
phase2_L31_attention_analysis.py

Task: Inspect what L31 attends to for recursive vs baseline prompts.

For a small set of prompts:
- 10 recursive (L4_full),
- 10 baseline factual (baseline_factual),

we:
- Run the model with output_attentions=True,
- Extract L31 attention (layer index 31),
- Compute, per prompt type:
  - Mean attention entropy per head,
  - Fraction of attention directed to self-referential tokens:
    ["observe", "watch", "self", "aware", "consciousness", "response", "words"].

Outputs:
- CSV: results/dec11_evening/L31_attention_summary.csv
- Log: logs/dec11_evening/L31_attention_summary.txt
"""

import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))

from src.core.models import load_model, set_seed
from prompts.loader import PromptLoader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
L31 = 31
N_RECURSIVE = 10
N_BASELINE = 10

SELF_REF_TOKENS = ["observe", "observing", "watch", "watching", "self", "aware", "consciousness", "response", "words"]


def attention_entropy(attn: torch.Tensor) -> float:
    """
    Compute mean entropy of attention distribution for a single head.

    attn: (seq, seq) or (batch, seq, seq)
    """
    if attn.dim() == 3:
        # (batch, seq, seq)
        probs = attn
    else:
        probs = attn.unsqueeze(0)
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    probs = probs + eps
    ent = -(probs * probs.log()).sum(dim=-1)  # (batch, seq)
    return float(ent.mean().item())


def get_self_ref_mask(tokenizer, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Build a boolean mask over positions that correspond to self-referential tokens.

    input_ids: (batch, seq)
    """
    batch, seq = input_ids.shape
    mask = torch.zeros((batch, seq), dtype=torch.bool, device=input_ids.device)
    vocab = {tok: tokenizer.encode(tok, add_special_tokens=False) for tok in SELF_REF_TOKENS}
    for tok, ids in vocab.items():
        if not ids:
            continue
        tid = ids[0]
        mask |= (input_ids == tid)
    return mask


def analyze_prompts(prompts: List[str], label: str, model, tokenizer) -> List[Dict]:
    rows: List[Dict] = []

    for prompt in tqdm(prompts, desc=f"L31 attention ({label})"):
        enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**enc, output_attentions=True, use_cache=False)

        # attentions is a tuple of length num_layers: each (batch, heads, seq, seq)
        attn_L31 = out.attentions[L31]  # (1, heads, seq, seq)
        batch, n_heads, seq, _ = attn_L31.shape

        # Self-ref mask over keys (destination positions)
        self_mask = get_self_ref_mask(tokenizer, enc.input_ids)  # (1, seq)

        for h in range(n_heads):
            head_attn = attn_L31[0, h]  # (seq, seq)
            ent = attention_entropy(head_attn)

            # For each query position, sum attention onto self-ref tokens
            # Then average over all query positions.
            # head_attn[q, k]: attention from q -> k
            self_mass_per_q = head_attn[:, self_mask[0]].sum(dim=-1)  # (seq,)
            mean_self_mass = float(self_mass_per_q.mean().item())

            rows.append(
                {
                    "prompt_label": label,
                    "prompt_snip": prompt[:80] + ("..." if len(prompt) > 80 else ""),
                    "head": h,
                    "entropy": ent,
                    "self_ref_mass": mean_self_mass,
                }
            )

    return rows


def run_L31_attention_analysis():
    print("Initializing Phase 2: L31 attention analysis...")
    set_seed(42)

    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    try:
        # Switch to eager attention to enable output_attentions
        model.set_attn_implementation("eager")
    except Exception:
        pass
    loader = PromptLoader()

    rec_prompts = loader.get_by_group("L4_full", limit=N_RECURSIVE, seed=42)
    base_prompts = loader.get_by_group("baseline_factual", limit=N_BASELINE, seed=42)
    print(f"Using {len(rec_prompts)} recursive and {len(base_prompts)} baseline prompts.")

    rows: List[Dict] = []
    rows.extend(analyze_prompts(rec_prompts, "recursive", model, tokenizer))
    rows.extend(analyze_prompts(base_prompts, "baseline", model, tokenizer))

    os.makedirs("results/dec11_evening", exist_ok=True)
    out_csv = "results/dec11_evening/L31_attention_summary.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # Aggregate summary
    summary = (
        df.groupby(["prompt_label", "head"])
        .agg(
            mean_entropy=("entropy", "mean"),
            mean_self_ref_mass=("self_ref_mass", "mean"),
        )
        .reset_index()
    )

    out_log = "logs/dec11_evening/L31_attention_summary.txt"
    os.makedirs("logs/dec11_evening", exist_ok=True)
    with open(out_log, "w") as f:
        f.write("# L31 Attention Analysis Summary\n\n")
        for label in ["recursive", "baseline"]:
            sub = summary[summary["prompt_label"] == label]
            f.write(f"== {label.upper()} ==\n")
            f.write(
                f"Global mean entropy: {sub['mean_entropy'].mean():.4f}, "
                f"global mean self_ref_mass: {sub['mean_self_ref_mass'].mean():.6f}\n"
            )
            # Top-3 heads by self_ref_mass
            top_heads = (
                sub.sort_values("mean_self_ref_mass", ascending=False)
                .head(3)[["head", "mean_entropy", "mean_self_ref_mass"]]
            )
            f.write("Top heads by self-ref attention:\n")
            for _, row in top_heads.iterrows():
                f.write(
                    f"  Head {int(row['head'])}: entropy={row['mean_entropy']:.4f}, "
                    f"self_ref_mass={row['mean_self_ref_mass']:.6f}\n"
                )
            f.write("\n")

    print(f"L31 attention analysis complete. CSV: {out_csv}, log: {out_log}")


if __name__ == "__main__":
    run_L31_attention_analysis()

