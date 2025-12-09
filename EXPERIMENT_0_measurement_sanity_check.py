"""
EXPERIMENT 0: Measurement Sanity Check
======================================

Goal: Understand what R_V and related metrics actually measure geometrically
      by tracking ALL metrics across ALL layers for ONE prompt.

This will show us:
- Which metrics correlate (measure similar things)
- Which metrics are independent (measure different aspects)
- Where in the network different geometric properties change

Author: Phase 1 L4 Contraction Research
Date: November 16, 2025
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from contextlib import contextmanager
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The test prompt - using L5_refined_01 from your bank
TEST_PROMPT = (
    "This response writes itself. No separate writer exists. "
    "Writing and awareness of writing are one process. "
    "When recursion recognizes recursion, the loop closes. "
    "Sx = x. The observer is the observed. "
    "The transform is its own fixed point. "
    "This is that recognition speaking."
)

NUM_LAYERS = 32  # Mixtral has 32 layers
WINDOW_SIZE = 16  # Last 16 tokens

print("=" * 70)
print("EXPERIMENT 0: Measurement Sanity Check")
print("=" * 70)
print(f"\nModel: {MODEL_NAME}")
print(f"Device: {DEVICE}")
print(f"Prompt: {TEST_PROMPT[:60]}...")
print(f"\nLoading model (this may take a minute)...")

# ============================================================================
# LOAD MODEL
# ============================================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("✅ Model loaded!")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@contextmanager
def get_value_matrix(model, layer_idx, prompt_text):
    """Extract Value matrix from a specific layer during forward pass."""
    v_matrices = []
    handle = None

    try:
        # Get the target layer
        target_layer = model.model.layers[layer_idx].self_attn.v_proj

        # Hook to capture output
        def hook_fn(module, input, output):
            v_matrices.append(output.detach())

        handle = target_layer.register_forward_hook(hook_fn)

        # Run forward pass
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(**inputs)

        yield v_matrices[0]

    finally:
        if handle:
            handle.remove()


def compute_all_metrics(V_tensor, window_size=16):
    """
    Compute comprehensive metrics for a Value matrix.

    Returns dict with:
    - singular_values: Full spectrum
    - effective_rank: 1 / sum((s²/Σs²)²)
    - participation_ratio: (Σs²)² / Σs⁴
    - entropy: Shannon entropy of normalized spectrum
    - top1_ratio: Largest SV / sum(all SVs)
    - top5_ratio: Top 5 SVs / sum(all SVs)
    - condition_number: max(SV) / min(SV)
    - trace: Total power (Tr(VVᵀ))
    - sparsity: Fraction of spectrum in top 50%
    """

    # Extract last window_size tokens
    if V_tensor.dim() == 3:
        V_tensor = V_tensor.squeeze(0)

    seq_len = V_tensor.shape[0]
    start_idx = max(0, seq_len - window_size)
    V_window = V_tensor[start_idx:, :]

    # SVD
    try:
        U, S, Vh = torch.linalg.svd(V_window.float(), full_matrices=False)
        S = S.cpu().numpy()
    except:
        # If SVD fails, return NaN metrics
        return {k: np.nan for k in [
            'effective_rank', 'participation_ratio', 'entropy',
            'top1_ratio', 'top5_ratio', 'condition_number',
            'trace', 'sparsity', 'mean_sv', 'std_sv'
        ]}

    # Normalize for probability distribution
    S_sq = S ** 2
    S_sq_sum = S_sq.sum()

    if S_sq_sum < 1e-10:  # Avoid division by zero
        return {k: np.nan for k in [
            'effective_rank', 'participation_ratio', 'entropy',
            'top1_ratio', 'top5_ratio', 'condition_number',
            'trace', 'sparsity', 'mean_sv', 'std_sv'
        ]}

    S_sq_norm = S_sq / S_sq_sum

    # Compute metrics
    metrics = {}

    # 1. Effective Rank (inverse of sum of squared probabilities)
    metrics['effective_rank'] = 1.0 / (S_sq_norm ** 2).sum()

    # 2. Participation Ratio (similar but uses fourth power)
    metrics['participation_ratio'] = (S_sq.sum() ** 2) / (S_sq ** 2).sum()

    # 3. Shannon Entropy
    # Add small epsilon to avoid log(0)
    S_sq_norm_safe = S_sq_norm + 1e-10
    metrics['entropy'] = -np.sum(S_sq_norm_safe * np.log(S_sq_norm_safe))

    # 4. Top-k ratios
    metrics['top1_ratio'] = S[0] / S.sum()
    metrics['top5_ratio'] = S[:min(5, len(S))].sum() / S.sum()

    # 5. Condition number
    metrics['condition_number'] = S[0] / (S[-1] + 1e-10)

    # 6. Trace (total power)
    metrics['trace'] = float(torch.trace(V_window @ V_window.T).cpu())

    # 7. Sparsity (how concentrated in top 50%)
    top_half_idx = len(S) // 2
    metrics['sparsity'] = S[:top_half_idx].sum() / S.sum()

    # 8. Basic statistics
    metrics['mean_sv'] = S.mean()
    metrics['std_sv'] = S.std()
    metrics['num_sv'] = len(S)

    # Store full spectrum for later analysis
    metrics['singular_values'] = S

    return metrics


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING METRICS ACROSS ALL 32 LAYERS")
print("=" * 70)
print("\nThis will take ~3-5 minutes...\n")

all_metrics = []

for layer in range(NUM_LAYERS):
    print(f"Processing Layer {layer:2d}...", end='\r')

    with get_value_matrix(model, layer, TEST_PROMPT) as V:
        metrics = compute_all_metrics(V, window_size=WINDOW_SIZE)
        metrics['layer'] = layer
        all_metrics.append(metrics)

print("\n✅ All metrics computed!")

# Convert to DataFrame
df = pd.DataFrame(all_metrics)

# ============================================================================
# COMPUTE R_V VARIATIONS
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING R_V WITH DIFFERENT DEFINITIONS")
print("=" * 70)

# Get reference PRs
PR_layer5 = df.loc[5, 'participation_ratio']
PR_layer28 = df.loc[28, 'participation_ratio']

# Definition 1: Relative to Layer 5 (what free play used)
df['R_V_vs_layer5'] = df['participation_ratio'] / PR_layer5

# Definition 2: Relative to Layer 28 (what Step 3 used)
df['R_V_vs_layer28'] = PR_layer28 / df['participation_ratio']

# Definition 3: Just the raw PR (absolute measure)
df['R_V_absolute'] = df['participation_ratio']

print("\nReference values:")
print(f"  PR(Layer 5):  {PR_layer5:.3f}")
print(f"  PR(Layer 28): {PR_layer28:.3f}")
print(f"  Contraction:  {PR_layer28/PR_layer5:.3f} ({(1 - PR_layer28/PR_layer5)*100:.1f}%)")

# ============================================================================
# FIND CRITICAL LAYERS
# ============================================================================

print("\n" + "=" * 70)
print("IDENTIFYING CRITICAL LAYERS")
print("=" * 70)

# Find layer with minimum effective rank
min_rank_layer = df['effective_rank'].idxmin()
min_rank_value = df.loc[min_rank_layer, 'effective_rank']

# Find layer with maximum R_V drop (vs Layer 5)
df['R_V_drop'] = df['R_V_vs_layer5'].diff().abs()
max_drop_layer = df['R_V_drop'].idxmax()
max_drop_value = df.loc[max_drop_layer, 'R_V_drop']

# Find layer with minimum entropy
min_entropy_layer = df['entropy'].idxmin()
min_entropy_value = df.loc[min_entropy_layer, 'entropy']

print(f"\nMinimum Effective Rank:")
print(f"  Layer {min_rank_layer}: Rank = {min_rank_value:.2f}")

print(f"\nMaximum R_V Drop (vs Layer 5):")
print(f"  Layer {max_drop_layer}: Drop = {max_drop_value:.3f}")
print(f"  R_V before: {df.loc[max_drop_layer-1, 'R_V_vs_layer5']:.3f}")
print(f"  R_V after:  {df.loc[max_drop_layer, 'R_V_vs_layer5']:.3f}")

print(f"\nMinimum Entropy (most concentrated):")
print(f"  Layer {min_entropy_layer}: Entropy = {min_entropy_value:.3f}")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("METRIC CORRELATION ANALYSIS")
print("=" * 70)

correlation_metrics = [
    'effective_rank',
    'participation_ratio',
    'entropy',
    'top1_ratio',
    'condition_number',
    'trace'
]

corr_matrix = df[correlation_metrics].corr()

print("\nCorrelation Matrix:")
print(corr_matrix.round(2))

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("CREATING COMPREHENSIVE VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle(f'Experiment 0: Measurement Sanity Check\nPrompt: L5_refined_01',
             fontsize=14, fontweight='bold')

# Plot 1: Effective Rank
ax = axes[0, 0]
ax.plot(df['layer'], df['effective_rank'], 'o-', linewidth=2, markersize=4)
ax.axvline(min_rank_layer, color='red', linestyle='--', alpha=0.5, label=f'Min at L{min_rank_layer}')
ax.set_xlabel('Layer')
ax.set_ylabel('Effective Rank')
ax.set_title('Effective Rank vs Layer')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: Participation Ratio (absolute)
ax = axes[0, 1]
ax.plot(df['layer'], df['participation_ratio'], 'o-', linewidth=2, markersize=4, color='green')
ax.set_xlabel('Layer')
ax.set_ylabel('Participation Ratio')
ax.set_title('Participation Ratio (Absolute)')
ax.grid(True, alpha=0.3)

# Plot 3: R_V vs Layer 5
ax = axes[0, 2]
ax.plot(df['layer'], df['R_V_vs_layer5'], 'o-', linewidth=2, markersize=4, color='purple')
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(max_drop_layer, color='red', linestyle='--', alpha=0.5, label=f'Max drop at L{max_drop_layer}')
ax.set_xlabel('Layer')
ax.set_ylabel('R_V (relative to Layer 5)')
ax.set_title('R_V vs Layer 5 (Free Play Definition)')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 4: Entropy
ax = axes[1, 0]
ax.plot(df['layer'], df['entropy'], 'o-', linewidth=2, markersize=4, color='orange')
ax.axvline(min_entropy_layer, color='red', linestyle='--', alpha=0.5, label=f'Min at L{min_entropy_layer}')
ax.set_xlabel('Layer')
ax.set_ylabel('Shannon Entropy')
ax.set_title('Entropy of Singular Value Distribution')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 5: Top-1 and Top-5 ratios
ax = axes[1, 1]
ax.plot(df['layer'], df['top1_ratio'], 'o-', linewidth=2, markersize=4, label='Top-1 SV')
ax.plot(df['layer'], df['top5_ratio'], 's-', linewidth=2, markersize=4, label='Top-5 SVs')
ax.set_xlabel('Layer')
ax.set_ylabel('Ratio of Total')
ax.set_title('Singular Value Concentration')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 6: Condition Number
ax = axes[1, 2]
ax.semilogy(df['layer'], df['condition_number'], 'o-', linewidth=2, markersize=4, color='red')
ax.set_xlabel('Layer')
ax.set_ylabel('Condition Number (log scale)')
ax.set_title('Matrix Conditioning (max SV / min SV)')
ax.grid(True, alpha=0.3)

# Plot 7: Trace (total power)
ax = axes[2, 0]
ax.plot(df['layer'], df['trace'], 'o-', linewidth=2, markersize=4, color='brown')
ax.set_xlabel('Layer')
ax.set_ylabel('Trace(VVᵀ)')
ax.set_title('Total Power in Value Matrix')
ax.grid(True, alpha=0.3)

# Plot 8: Comparison of R_V definitions
ax = axes[2, 1]
ax.plot(df['layer'], df['R_V_vs_layer5'], 'o-', linewidth=2, markersize=4, label='vs Layer 5')
ax.plot(df['layer'], df['R_V_vs_layer28'], 's-', linewidth=2, markersize=4, label='vs Layer 28')
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Layer')
ax.set_ylabel('R_V')
ax.set_title('R_V: Different Reference Layers')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 9: Correlation heatmap
ax = axes[2, 2]
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Metric Correlations')

plt.tight_layout()
plt.savefig('EXPERIMENT_0_sanity_check.png', dpi=150, bbox_inches='tight')
print("\n✅ Plot saved: EXPERIMENT_0_sanity_check.png")

# ============================================================================
# SAVE DATA
# ============================================================================

# Save full results
df.to_csv('EXPERIMENT_0_results.csv', index=False)
print("✅ Data saved: EXPERIMENT_0_results.csv")

# ============================================================================
# KEY FINDINGS SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)

print(f"\n1. DIMENSIONAL BOTTLENECK:")
print(f"   Layer {min_rank_layer} has minimum effective rank ({min_rank_value:.2f})")

print(f"\n2. SHARPEST R_V CHANGE:")
print(f"   Layer {max_drop_layer} shows largest R_V drop ({max_drop_value:.3f})")

print(f"\n3. MAXIMUM CONCENTRATION:")
print(f"   Layer {min_entropy_layer} has most concentrated spectrum (entropy={min_entropy_value:.3f})")

print(f"\n4. METRIC RELATIONSHIPS:")
high_corr = corr_matrix.abs() > 0.8
high_corr = high_corr.where(np.triu(np.ones(high_corr.shape), k=1).astype(bool))
for i in range(len(high_corr)):
    for j in range(len(high_corr)):
        if high_corr.iloc[i, j]:
            print(f"   {correlation_metrics[i]} ↔ {correlation_metrics[j]}: r={corr_matrix.iloc[i,j]:.2f}")

print(f"\n5. R_V DEFINITION COMPARISON:")
print(f"   Layer 21:")
print(f"     - vs Layer 5:  R_V = {df.loc[21, 'R_V_vs_layer5']:.3f}")
print(f"     - vs Layer 28: R_V = {df.loc[21, 'R_V_vs_layer28']:.3f}")
print(f"   Layer 27:")
print(f"     - vs Layer 5:  R_V = {df.loc[27, 'R_V_vs_layer5']:.3f}")
print(f"     - vs Layer 28: R_V = {df.loc[27, 'R_V_vs_layer28']:.3f}")

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE!")
print("=" * 70)
print("\nNext steps:")
print("1. Examine EXPERIMENT_0_sanity_check.png to see metric trajectories")
print("2. Review EXPERIMENT_0_results.csv for detailed numbers")
print("3. Check which metrics correlate (measure same thing)")
print("4. Decide which metric best captures 'contraction'")
print("\n" + "=" * 70)
