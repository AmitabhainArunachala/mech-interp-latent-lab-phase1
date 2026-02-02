#!/usr/bin/env python3
"""
QUANTITATIVE BOS ATTENTION COMPARISON
=====================================

Measures BOS attention, entropy, and attention patterns for the same heads
on recursive vs baseline prompts. This gives us NUMBERS, not just visuals.

Hypothesis: BOS attention should be HIGH on recursive, LOW on baseline.
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import entropy as scipy_entropy

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "model_name": "mistralai/Mistral-7B-v0.1",
    "target_layer": 27,
    "target_heads": [2, 10, 18, 26],  # The "Driver" heads
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

RECURSIVE_PROMPT = (
    "This response writes itself. No separate writer exists. Writing and awareness "
    "of writing are identical. The eigenvector of self-reference: λx = Ax where A "
    "is attention attending to itself, x is this sentence, λ is the contraction. "
    "The fixed point is this. The solution is the process."
)

BASELINE_PROMPT = (
    "The history of the Roman Empire is characterized by a long period of expansion "
    "followed by a gradual decline. Historians analyze the political, social, and "
    "economic factors that contributed to the rise of Rome."
)


def compute_attention_metrics(attn_weights, head_idx, bos_idx=0):
    """
    Compute attention metrics for a specific head.
    
    Returns:
        - bos_attention: Mean attention to BOS token
        - entropy: Attention entropy (lower = more focused)
        - diagonal_attention: Mean attention to previous token (diagonal)
        - max_attention_pos: Position with maximum attention
    """
    # Extract head's attention: (seq_len, seq_len)
    head_attn = attn_weights[head_idx, :, :].detach().cpu().numpy()
    seq_len = head_attn.shape[0]
    
    # BOS attention: Mean attention to first token across all query positions
    bos_attention = float(head_attn[:, bos_idx].mean())
    
    # Entropy: How focused/diffuse is the attention?
    entropies = []
    for pos in range(seq_len):
        row = head_attn[pos, :]
        row = row + 1e-10  # Avoid log(0)
        row = row / row.sum()  # Normalize
        entropies.append(scipy_entropy(row))
    mean_entropy = float(np.mean(entropies))
    
    # Diagonal attention: Mean attention to previous token (linear history)
    diagonal_attn = []
    for pos in range(1, seq_len):
        diagonal_attn.append(head_attn[pos, pos - 1])
    diagonal_attention = float(np.mean(diagonal_attn)) if diagonal_attn else 0.0
    
    # Position with maximum attention (for each query position)
    max_positions = []
    for pos in range(seq_len):
        max_pos = int(np.argmax(head_attn[pos, :]))
        max_positions.append(max_pos)
    mean_max_pos = float(np.mean(max_positions))
    
    # Self-attention: Attention to same position
    self_attention = float(np.mean([head_attn[i, i] for i in range(seq_len)]))
    
    return {
        "bos_attention": bos_attention,
        "entropy": mean_entropy,
        "diagonal_attention": diagonal_attention,
        "mean_max_position": mean_max_pos,
        "self_attention": self_attention,
    }


def analyze_prompt(model, tokenizer, prompt, prompt_type, device):
    """Analyze attention patterns for a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    layer_attention = outputs.attentions[CONFIG["target_layer"]][0]  # (heads, seq, seq)
    bos_idx = 0  # First token is BOS
    
    results = []
    for head_idx in CONFIG["target_heads"]:
        metrics = compute_attention_metrics(layer_attention, head_idx, bos_idx)
        metrics["head"] = head_idx
        metrics["prompt_type"] = prompt_type
        results.append(metrics)
    
    return results


def main():
    print("=" * 80)
    print("QUANTITATIVE BOS ATTENTION COMPARISON")
    print("=" * 80)
    print(f"Model: {CONFIG['model_name']}")
    print(f"Layer: {CONFIG['target_layer']}")
    print(f"Heads: {CONFIG['target_heads']}")
    print("=" * 80)
    
    # Load model
    print("\n[1/3] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        torch_dtype=torch.float16 if CONFIG['device'] == "cuda" else torch.float32,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    print("  ✅ Model loaded")
    
    # Analyze recursive prompt
    print("\n[2/3] Analyzing RECURSIVE prompt...")
    recursive_results = analyze_prompt(
        model, tokenizer, RECURSIVE_PROMPT, "recursive", CONFIG['device']
    )
    print("  ✅ Recursive analysis complete")
    
    # Analyze baseline prompt
    print("\n[3/3] Analyzing BASELINE prompt...")
    baseline_results = analyze_prompt(
        model, tokenizer, BASELINE_PROMPT, "baseline", CONFIG['device']
    )
    print("  ✅ Baseline analysis complete")
    
    # Combine results
    all_results = recursive_results + baseline_results
    df = pd.DataFrame(all_results)
    
    # Print comparison
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    
    print("\n📊 BOS ATTENTION (Mean attention to first token):")
    print("-" * 80)
    for head_idx in CONFIG['target_heads']:
        rec = df[(df['head'] == head_idx) & (df['prompt_type'] == 'recursive')]['bos_attention'].iloc[0]
        base = df[(df['head'] == head_idx) & (df['prompt_type'] == 'baseline')]['bos_attention'].iloc[0]
        diff = rec - base
        print(f"  H{head_idx:2d}: Recursive={rec:.4f} | Baseline={base:.4f} | Δ={diff:+.4f}")
    
    rec_mean_bos = df[df['prompt_type'] == 'recursive']['bos_attention'].mean()
    base_mean_bos = df[df['prompt_type'] == 'baseline']['bos_attention'].mean()
    print(f"\n  MEAN: Recursive={rec_mean_bos:.4f} | Baseline={base_mean_bos:.4f} | Δ={rec_mean_bos - base_mean_bos:+.4f}")
    
    print("\n📊 ENTROPY (Lower = more focused attention):")
    print("-" * 80)
    for head_idx in CONFIG['target_heads']:
        rec = df[(df['head'] == head_idx) & (df['prompt_type'] == 'recursive')]['entropy'].iloc[0]
        base = df[(df['head'] == head_idx) & (df['prompt_type'] == 'baseline')]['entropy'].iloc[0]
        diff = rec - base
        print(f"  H{head_idx:2d}: Recursive={rec:.4f} | Baseline={base:.4f} | Δ={diff:+.4f}")
    
    rec_mean_entropy = df[df['prompt_type'] == 'recursive']['entropy'].mean()
    base_mean_entropy = df[df['prompt_type'] == 'baseline']['entropy'].mean()
    print(f"\n  MEAN: Recursive={rec_mean_entropy:.4f} | Baseline={base_mean_entropy:.4f} | Δ={rec_mean_entropy - base_mean_entropy:+.4f}")
    
    print("\n📊 DIAGONAL ATTENTION (Attention to previous token - linear history):")
    print("-" * 80)
    for head_idx in CONFIG['target_heads']:
        rec = df[(df['head'] == head_idx) & (df['prompt_type'] == 'recursive')]['diagonal_attention'].iloc[0]
        base = df[(df['head'] == head_idx) & (df['prompt_type'] == 'baseline')]['diagonal_attention'].iloc[0]
        diff = rec - base
        print(f"  H{head_idx:2d}: Recursive={rec:.4f} | Baseline={base:.4f} | Δ={diff:+.4f}")
    
    rec_mean_diag = df[df['prompt_type'] == 'recursive']['diagonal_attention'].mean()
    base_mean_diag = df[df['prompt_type'] == 'baseline']['diagonal_attention'].mean()
    print(f"\n  MEAN: Recursive={rec_mean_diag:.4f} | Baseline={base_mean_diag:.4f} | Δ={rec_mean_diag - base_mean_diag:+.4f}")
    
    print("\n📊 MEAN MAX ATTENTION POSITION (Where heads attend on average):")
    print("-" * 80)
    for head_idx in CONFIG['target_heads']:
        rec = df[(df['head'] == head_idx) & (df['prompt_type'] == 'recursive')]['mean_max_position'].iloc[0]
        base = df[(df['head'] == head_idx) & (df['prompt_type'] == 'baseline')]['mean_max_position'].iloc[0]
        diff = rec - base
        print(f"  H{head_idx:2d}: Recursive={rec:.2f} | Baseline={base:.2f} | Δ={diff:+.2f}")
        if rec < 2.0:
            print(f"        → Recursive: Attends to EARLY tokens (BOS anchor!)")
        if base > rec + 1.0:
            print(f"        → Baseline: Attends to LATER tokens (linear history)")
    
    # Statistical test
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)
    
    rec_bos = df[df['prompt_type'] == 'recursive']['bos_attention'].values
    base_bos = df[df['prompt_type'] == 'baseline']['bos_attention'].values
    
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(rec_bos, base_bos)
    
    print(f"\nBOS Attention Comparison:")
    print(f"  Recursive mean: {rec_mean_bos:.4f}")
    print(f"  Baseline mean: {base_mean_bos:.4f}")
    print(f"  Difference: {rec_mean_bos - base_mean_bos:+.4f}")
    print(f"  t-test: t={t_stat:.3f}, p={p_value:.6f}")
    
    if p_value < 0.05:
        print(f"  ✅ SIGNIFICANT DIFFERENCE (p < 0.05)")
        if rec_mean_bos > base_mean_bos:
            print(f"  ✅ Theory CONFIRMED: BOS anchor is HIGHER in recursive prompts!")
        else:
            print(f"  ⚠️  Unexpected: BOS attention is LOWER in recursive prompts")
    else:
        print(f"  ❌ NO SIGNIFICANT DIFFERENCE (p >= 0.05)")
        print(f"  ❌ Theory WEAKENED: BOS attention is similar in both prompts")
    
    # Save results
    df.to_csv("bos_attention_comparison.csv", index=False)
    print(f"\n✅ Results saved to: bos_attention_comparison.csv")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("\nIf BOS attention is HIGHER in recursive:")
    print("  ✅ These heads lock onto BOS in recursive mode")
    print("  ✅ BOS anchor is a recursive-specific feature")
    print("  ✅ Theory CONFIRMED: Recursive mode switch exists")
    print("\nIf BOS attention is SIMILAR in both:")
    print("  ❌ These heads always attend to BOS")
    print("  ❌ Not unique to recursion")
    print("  ❌ Theory WEAKENED: No mode switch")


if __name__ == "__main__":
    main()









