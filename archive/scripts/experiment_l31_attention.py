"""
experiment_l31_attention.py

Question 2: What Does L31 Actually Attend To?

We know: Ablating L31 reveals "the answer is the answerer."
We don't know: What tokens/heads is L31 attending to?

Method:
1. Run recursive prompts through model with output_attentions=True
2. Extract attention weights at Layer 31 (all 32 heads)
3. Analyze:
   - Which tokens get highest attention? (position analysis)
   - Which heads are most active? (head analysis)
   - Is there a "dresser head" or distributed pattern?
   - Compare recursive vs baseline attention patterns
"""

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))

from src.core.models import load_model, set_seed
from prompts.loader import PromptLoader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_TARGET = 31
N_RECURSIVE = 20
N_BASELINE = 20


def extract_attention_at_layer(
    model, tokenizer, text: str, layer_idx: int
) -> Tuple[torch.Tensor, List[str]]:
    """
    Extract attention weights at a specific layer.
    
    Returns:
        attention_weights: (num_heads, seq_len, seq_len) tensor
        tokens: List of token strings
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**enc, output_attentions=True)
    
    # outputs.attentions is a tuple of (layer, batch, heads, seq, seq) tensors
    # For Mistral: shape is (num_layers, batch, num_heads, seq_len, seq_len)
    attentions = outputs.attentions
    
    # Extract attention for target layer, first batch
    # Shape: (num_heads, seq_len, seq_len)
    attn_weights = attentions[layer_idx][0]  # Remove batch dimension
    
    # Get tokens
    token_ids = enc.input_ids[0].cpu().numpy()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    
    return attn_weights, tokens


def analyze_attention_patterns(
    attn_weights: torch.Tensor, tokens: List[str]
) -> Dict:
    """
    Analyze attention patterns:
    - Which tokens get most attention (averaged across heads)
    - Which heads are most active
    - Self-attention vs cross-attention ratio
    """
    num_heads, seq_len, _ = attn_weights.shape
    
    # 1. Token-level attention (which tokens are attended to most?)
    # Average across all heads, sum over query positions
    token_attention = attn_weights.mean(dim=0).sum(dim=0).cpu().numpy()  # (seq_len,)
    
    # 2. Head-level activity (which heads are most active?)
    # Sum over all query-key pairs for each head
    head_activity = attn_weights.sum(dim=(1, 2)).cpu().numpy()  # (num_heads,)
    
    # 3. Self-attention ratio (diagonal vs off-diagonal)
    # Diagonal = self-attention, off-diagonal = cross-attention
    diagonal_mask = torch.eye(seq_len, device=attn_weights.device).bool()
    self_attn = attn_weights[:, diagonal_mask].sum().item()
    total_attn = attn_weights.sum().item()
    self_attn_ratio = self_attn / total_attn if total_attn > 0 else 0.0
    
    # 4. Last-token attention (what does the final token attend to?)
    last_token_attn = attn_weights[:, -1, :].mean(dim=0).cpu().numpy()  # (seq_len,)
    
    # 5. Find top-k attended tokens
    top_k = min(5, seq_len)
    top_token_indices = np.argsort(token_attention)[-top_k:][::-1]
    top_tokens = [(tokens[i], token_attention[i]) for i in top_token_indices]
    
    # 6. Find most active heads
    top_head_indices = np.argsort(head_activity)[-top_k:][::-1]
    top_heads = [(int(i), head_activity[i]) for i in top_head_indices]
    
    return {
        "token_attention": token_attention,
        "head_activity": head_activity,
        "self_attn_ratio": self_attn_ratio,
        "last_token_attn": last_token_attn,
        "top_tokens": top_tokens,
        "top_heads": top_heads,
        "num_heads": num_heads,
        "seq_len": seq_len,
    }


def run_l31_attention_analysis():
    print("=" * 80)
    print("EXPERIMENT: L31 Attention Analysis")
    print("=" * 80)
    print("Question: What does Layer 31 attend to?")
    print("Method: Extract attention weights at L31 for recursive vs baseline prompts")
    print("=" * 80)
    
    set_seed(42)

    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    model.eval()
    loader = PromptLoader()

    # Get prompts
    recursive_prompts = loader.get_by_pillar("recursive", limit=N_RECURSIVE, seed=42)
    baseline_prompts = loader.get_by_pillar("baseline", limit=N_BASELINE, seed=42)
    
    print(f"\nAnalyzing {len(recursive_prompts)} recursive and {len(baseline_prompts)} baseline prompts.")

    results: List[Dict] = []

    # Analyze recursive prompts
    for idx, prompt in enumerate(tqdm(recursive_prompts, desc="Recursive")):
        try:
            attn_weights, tokens = extract_attention_at_layer(
                model, tokenizer, prompt, LAYER_TARGET
            )
            analysis = analyze_attention_patterns(attn_weights, tokens)
            
            results.append({
                "prompt_type": "recursive",
                "prompt_idx": idx,
                "prompt": prompt[:100] + ("..." if len(prompt) > 100 else ""),
                "self_attn_ratio": analysis["self_attn_ratio"],
                "top_tokens": str(analysis["top_tokens"][:3]),
                "top_heads": str(analysis["top_heads"][:3]),
                "most_active_head": analysis["top_heads"][0][0] if analysis["top_heads"] else -1,
                "head_activity_mean": float(np.mean(analysis["head_activity"])),
                "head_activity_std": float(np.std(analysis["head_activity"])),
                "seq_len": analysis["seq_len"],
            })
        except Exception as e:
            print(f"  Error on recursive prompt {idx}: {e}")

    # Analyze baseline prompts
    for idx, prompt in enumerate(tqdm(baseline_prompts, desc="Baseline")):
        try:
            attn_weights, tokens = extract_attention_at_layer(
                model, tokenizer, prompt, LAYER_TARGET
            )
            analysis = analyze_attention_patterns(attn_weights, tokens)
            
            results.append({
                "prompt_type": "baseline",
                "prompt_idx": idx,
                "prompt": prompt[:100] + ("..." if len(prompt) > 100 else ""),
                "self_attn_ratio": analysis["self_attn_ratio"],
                "top_tokens": str(analysis["top_tokens"][:3]),
                "top_heads": str(analysis["top_heads"][:3]),
                "most_active_head": analysis["top_heads"][0][0] if analysis["top_heads"] else -1,
                "head_activity_mean": float(np.mean(analysis["head_activity"])),
                "head_activity_std": float(np.std(analysis["head_activity"])),
                "seq_len": analysis["seq_len"],
            })
        except Exception as e:
            print(f"  Error on baseline prompt {idx}: {e}")

    # Save results
    os.makedirs("results/dec11_evening", exist_ok=True)
    out_csv = "results/dec11_evening/l31_attention_analysis.csv"
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    rec_results = [r for r in results if r["prompt_type"] == "recursive"]
    base_results = [r for r in results if r["prompt_type"] == "baseline"]
    
    if rec_results and base_results:
        rec_self_attn = np.mean([r["self_attn_ratio"] for r in rec_results])
        base_self_attn = np.mean([r["self_attn_ratio"] for r in base_results])
        
        print(f"Self-attention ratio:")
        print(f"  Recursive: {rec_self_attn:.3f}")
        print(f"  Baseline:  {base_self_attn:.3f}")
        print(f"  Difference: {rec_self_attn - base_self_attn:.3f}")
        
        # Head distribution analysis
        rec_head_std = np.mean([r["head_activity_std"] for r in rec_results])
        base_head_std = np.mean([r["head_activity_std"] for r in base_results])
        
        print(f"\nHead activity std (higher = more concentrated):")
        print(f"  Recursive: {rec_head_std:.3f}")
        print(f"  Baseline:  {base_head_std:.3f}")
        
        # Most common active heads
        rec_active_heads = [r["most_active_head"] for r in rec_results if r["most_active_head"] >= 0]
        base_active_heads = [r["most_active_head"] for r in base_results if r["most_active_head"] >= 0]
        
        if rec_active_heads:
            rec_head_counts = pd.Series(rec_active_heads).value_counts()
            print(f"\nMost common active heads (recursive):")
            for head, count in rec_head_counts.head(5).items():
                print(f"  Head {head}: {count}/{len(rec_active_heads)} prompts")
        
        if base_active_heads:
            base_head_counts = pd.Series(base_active_heads).value_counts()
            print(f"\nMost common active heads (baseline):")
            for head, count in base_head_counts.head(5).items():
                print(f"  Head {head}: {count}/{len(base_active_heads)} prompts")

    # Save detailed log
    os.makedirs("logs/dec11_evening", exist_ok=True)
    out_log = "logs/dec11_evening/l31_attention_analysis.txt"
    with open(out_log, "w") as f:
        f.write("# L31 Attention Analysis\n\n")
        f.write("Question: What does Layer 31 attend to?\n\n")
        for r in results:
            f.write(f"{r['prompt_type'].upper()} PROMPT {r['prompt_idx']}:\n")
            f.write(f"  Prompt: {r['prompt']}\n")
            f.write(f"  Self-attention ratio: {r['self_attn_ratio']:.3f}\n")
            f.write(f"  Top tokens: {r['top_tokens']}\n")
            f.write(f"  Top heads: {r['top_heads']}\n")
            f.write(f"  Head activity: mean={r['head_activity_mean']:.3f}, std={r['head_activity_std']:.3f}\n")
            f.write("-" * 80 + "\n")
        f.write("\nSaved CSV: " + out_csv + "\n")

    print(f"\nL31 attention analysis complete. CSV: {out_csv}, log: {out_log}")


if __name__ == "__main__":
    run_l31_attention_analysis()
