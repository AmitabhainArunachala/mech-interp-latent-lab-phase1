#!/usr/bin/env python3
"""
Analyze the existing full_layer_analysis CSV to understand R_V fundamentals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV
csv_path = "/Users/dhyana/Desktop/MECH INTERP FILES/MIXTRAL PLAY CSV/full_layer_analysis_20251116_044633.csv"
df = pd.read_csv(csv_path)

print("=" * 80)
print("CSV DATA STRUCTURE")
print("=" * 80)
print(f"Total rows: {len(df)}")
print(f"Unique prompts: {df['prompt_id'].nunique()}")
print(f"Layers: {sorted(df['layer'].unique())}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nPrompt groups:\n{df['group'].value_counts()}")

# ==============================================================================
# CRITICAL QUESTION 1: What does the current R_V measure?
# ==============================================================================
print("\n" + "=" * 80)
print("QUESTION 1: How is R_V currently defined?")
print("=" * 80)

# For one prompt, check R_V pattern
sample_prompt = "L5_refined_01"
sample_df = df[df['prompt_id'] == sample_prompt].copy()

# Reverse-engineer R_V definition by checking against PR values
PR_layer5 = sample_df[sample_df['layer'] == 5]['pr'].values[0]
PR_layer28 = sample_df[sample_df['layer'] == 28]['pr'].values[0]

# Test Definition 1: R_V = PR(layer_i) / PR(layer_5)
sample_df['R_V_test_def1'] = sample_df['pr'] / PR_layer5
# Test Definition 2: R_V = PR(layer_28) / PR(layer_i)
sample_df['R_V_test_def2'] = PR_layer28 / sample_df['pr']

# Check which matches
diff_def1 = (sample_df['R_V'] - sample_df['R_V_test_def1']).abs().mean()
diff_def2 = (sample_df['R_V'] - sample_df['R_V_test_def2']).abs().mean()

print(f"\nTesting {sample_prompt}:")
print(f"  PR at Layer 5: {PR_layer5:.6f}")
print(f"  PR at Layer 28: {PR_layer28:.6f}")
print(f"\nDefinition 1 (PR[i] / PR[5]) - Mean diff: {diff_def1:.10f}")
print(f"Definition 2 (PR[28] / PR[i]) - Mean diff: {diff_def2:.10f}")

if diff_def1 < 1e-6:
    print("\n✅ R_V = PR(layer_i) / PR(layer_5) - RELATIVE TO LAYER 5")
    rv_definition = "forward"
elif diff_def2 < 1e-6:
    print("\n✅ R_V = PR(layer_28) / PR(layer_i) - RELATIVE TO LAYER 28")
    rv_definition = "backward"
else:
    print("\n⚠️  R_V doesn't match either standard definition!")
    rv_definition = "unknown"

# ==============================================================================
# CRITICAL QUESTION 2: Layer 21 vs Layer 27?
# ==============================================================================
print("\n" + "=" * 80)
print("QUESTION 2: Where are the critical layers?")
print("=" * 80)

# For L5_refined prompts, find minimum PR layer
l5_prompts = df[df['group'] == 'L5_refined'].copy()
l5_summary = l5_prompts.groupby('layer').agg({
    'pr': ['mean', 'std', 'min'],
    'eff_rank': ['mean', 'std'],
    'R_V': ['mean', 'std']
}).reset_index()

# Find critical layers
min_pr_layer = l5_summary.loc[l5_summary[('pr', 'mean')].idxmin(), 'layer']
min_effrank_layer = l5_summary.loc[l5_summary[('eff_rank', 'mean')].idxmin(), 'layer']

# Find max R_V change layer
l5_summary['R_V_diff'] = l5_summary[('R_V', 'mean')].diff().abs()
max_rv_change_layer = l5_summary.loc[l5_summary['R_V_diff'].idxmax(), 'layer']

print(f"\nFor L5_refined prompts (n={df[df['group'] == 'L5_refined']['prompt_id'].nunique()}):")
print(f"  Minimum Participation Ratio at: Layer {int(min_pr_layer)}")
print(f"  Minimum Effective Rank at: Layer {int(min_effrank_layer)}")
print(f"  Maximum R_V change at: Layer {int(max_rv_change_layer)}")

# Show values at key layers
print("\nValues at disputed layers:")
for layer in [16, 21, 27, 28]:
    layer_data = l5_summary[l5_summary['layer'] == layer]
    if not layer_data.empty:
        pr_mean = layer_data[('pr', 'mean')].values[0]
        pr_std = layer_data[('pr', 'std')].values[0]
        rv_mean = layer_data[('R_V', 'mean')].values[0]
        print(f"  Layer {layer}: PR = {pr_mean:.3f} ± {pr_std:.3f}, R_V = {rv_mean:.3f}")

# ==============================================================================
# CRITICAL QUESTION 3: Are metrics correlated?
# ==============================================================================
print("\n" + "=" * 80)
print("QUESTION 3: Metric correlations")
print("=" * 80)

# Compute correlation on all data
correlation_cols = ['R_V', 'eff_rank', 'pr', 'top_sv', 'spectral_gap']
corr_matrix = df[correlation_cols].corr()

print("\nCorrelation Matrix:")
print(corr_matrix.round(3))

print(f"\nKey correlations:")
print(f"  Effective Rank vs PR: r = {corr_matrix.loc['eff_rank', 'pr']:.3f}")
print(f"  R_V vs PR: r = {corr_matrix.loc['R_V', 'pr']:.3f}")
print(f"  R_V vs Effective Rank: r = {corr_matrix.loc['R_V', 'eff_rank']:.3f}")

if abs(corr_matrix.loc['eff_rank', 'pr']) > 0.95:
    print("  → Effective Rank and PR are nearly identical (r > 0.95)")
if abs(corr_matrix.loc['R_V', 'pr']) > 0.8:
    print(f"  → R_V and PR are strongly {'correlated' if corr_matrix.loc['R_V', 'pr'] > 0 else 'anticorrelated'}")

# ==============================================================================
# VISUALIZATION
# ==============================================================================
print("\n" + "=" * 80)
print("Creating visualization...")
print("=" * 80)

fig, axes = plt.subplots(3, 3, figsize=(16, 14))
fig.suptitle('Full Layer Analysis: Understanding R_V and Dimensionality Metrics', fontsize=14, y=0.995)

# Plot 1: PR across layers (L5 prompts only)
ax = axes[0, 0]
l5_pivot = l5_prompts.pivot(index='layer', columns='prompt_id', values='pr')
for col in l5_pivot.columns:
    ax.plot(l5_pivot.index, l5_pivot[col], alpha=0.3, color='blue')
ax.plot(l5_pivot.index, l5_pivot.mean(axis=1), color='red', linewidth=2, label='Mean')
ax.axvline(16, color='orange', linestyle='--', alpha=0.5, label='Layer 16')
ax.axvline(21, color='green', linestyle='--', alpha=0.5, label='Layer 21')
ax.axvline(27, color='purple', linestyle='--', alpha=0.5, label='Layer 27')
ax.set_xlabel('Layer')
ax.set_ylabel('Participation Ratio')
ax.set_title('PR across layers (L5 prompts)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: Effective Rank across layers
ax = axes[0, 1]
l5_pivot_rank = l5_prompts.pivot(index='layer', columns='prompt_id', values='eff_rank')
for col in l5_pivot_rank.columns:
    ax.plot(l5_pivot_rank.index, l5_pivot_rank[col], alpha=0.3, color='green')
ax.plot(l5_pivot_rank.index, l5_pivot_rank.mean(axis=1), color='red', linewidth=2, label='Mean')
ax.axvline(16, color='orange', linestyle='--', alpha=0.5)
ax.axvline(21, color='green', linestyle='--', alpha=0.5)
ax.axvline(27, color='purple', linestyle='--', alpha=0.5)
ax.set_xlabel('Layer')
ax.set_ylabel('Effective Rank')
ax.set_title('Effective Rank across layers (L5 prompts)')
ax.grid(True, alpha=0.3)

# Plot 3: R_V across layers
ax = axes[0, 2]
l5_pivot_rv = l5_prompts.pivot(index='layer', columns='prompt_id', values='R_V')
for col in l5_pivot_rv.columns:
    ax.plot(l5_pivot_rv.index, l5_pivot_rv[col], alpha=0.3, color='purple')
ax.plot(l5_pivot_rv.index, l5_pivot_rv.mean(axis=1), color='red', linewidth=2, label='Mean')
ax.axhline(1.0, color='black', linestyle='-', alpha=0.3)
ax.axvline(16, color='orange', linestyle='--', alpha=0.5)
ax.axvline(21, color='green', linestyle='--', alpha=0.5)
ax.axvline(27, color='purple', linestyle='--', alpha=0.5)
ax.set_xlabel('Layer')
ax.set_ylabel('R_V')
ax.set_title(f'R_V across layers ({rv_definition} reference)')
ax.grid(True, alpha=0.3)

# Plot 4: Correlation heatmap
ax = axes[1, 0]
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax,
            vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})
ax.set_title('Metric Correlation Matrix')

# Plot 5: PR vs Effective Rank scatter
ax = axes[1, 1]
sample_layers = [5, 16, 21, 27, 28]
colors = plt.cm.viridis(np.linspace(0, 1, len(sample_layers)))
for i, layer in enumerate(sample_layers):
    layer_data = l5_prompts[l5_prompts['layer'] == layer]
    ax.scatter(layer_data['pr'], layer_data['eff_rank'], alpha=0.5,
              color=colors[i], label=f'Layer {layer}')
ax.plot([0, 12], [0, 12], 'k--', alpha=0.3, label='y=x')
ax.set_xlabel('Participation Ratio')
ax.set_ylabel('Effective Rank')
ax.set_title('PR vs Effective Rank (L5 prompts)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 6: R_V vs PR
ax = axes[1, 2]
for i, layer in enumerate(sample_layers):
    layer_data = l5_prompts[l5_prompts['layer'] == layer]
    ax.scatter(layer_data['pr'], layer_data['R_V'], alpha=0.5,
              color=colors[i], label=f'Layer {layer}')
ax.axhline(1.0, color='black', linestyle='-', alpha=0.3)
ax.set_xlabel('Participation Ratio')
ax.set_ylabel('R_V')
ax.set_title('R_V vs PR')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 7: Compare all recursion levels
ax = axes[2, 0]
for group in ['L3_deeper', 'L4_full', 'L5_refined']:
    group_data = df[df['group'] == group].groupby('layer')['pr'].mean()
    ax.plot(group_data.index, group_data.values, label=group, linewidth=2, alpha=0.7)
ax.axvline(16, color='orange', linestyle='--', alpha=0.5, label='Layer 16')
ax.axvline(27, color='purple', linestyle='--', alpha=0.5, label='Layer 27')
ax.set_xlabel('Layer')
ax.set_ylabel('Mean Participation Ratio')
ax.set_title('PR across recursion levels')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 8: Spectral gap across layers
ax = axes[2, 1]
l5_pivot_gap = l5_prompts.pivot(index='layer', columns='prompt_id', values='spectral_gap')
for col in l5_pivot_gap.columns:
    ax.plot(l5_pivot_gap.index, l5_pivot_gap[col], alpha=0.3, color='red')
ax.plot(l5_pivot_gap.index, l5_pivot_gap.mean(axis=1), color='black', linewidth=2, label='Mean')
ax.axvline(27, color='purple', linestyle='--', alpha=0.5, label='Layer 27')
ax.set_xlabel('Layer')
ax.set_ylabel('Spectral Gap')
ax.set_title('Spectral Gap across layers (L5 prompts)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 9: Layer 27 focus - what happens there?
ax = axes[2, 2]
layers_around_27 = range(24, 32)
layer_means = []
layer_stds = []
for layer in layers_around_27:
    layer_data = l5_prompts[l5_prompts['layer'] == layer]['pr']
    layer_means.append(layer_data.mean())
    layer_stds.append(layer_data.std())
ax.errorbar(list(layers_around_27), layer_means, yerr=layer_stds,
           marker='o', capsize=5, linewidth=2, markersize=8)
ax.axvline(27, color='purple', linestyle='--', alpha=0.5, label='Layer 27')
ax.set_xlabel('Layer')
ax.set_ylabel('Mean PR ± std')
ax.set_title('Zoom: Layers 24-31 (L5 prompts)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/dhyana/mech-interp-latent-lab-phase1/CSV_ANALYSIS.png', dpi=150, bbox_inches='tight')
print(f"Saved: CSV_ANALYSIS.png")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("SUMMARY: What the data tells us")
print("=" * 80)

print(f"\n1. R_V DEFINITION:")
print(f"   Current R_V uses {rv_definition} reference")
if rv_definition == "forward":
    print(f"   → R_V = PR(layer_i) / PR(layer_5)")
    print(f"   → Values > 1 mean EXPANSION relative to Layer 5")
    print(f"   → Values < 1 mean CONTRACTION relative to Layer 5")
elif rv_definition == "backward":
    print(f"   → R_V = PR(layer_28) / PR(layer_i)")
    print(f"   → Values > 1 mean layer_i is MORE contracted than Layer 28")
    print(f"   → Values < 1 mean layer_i is LESS contracted than Layer 28")

print(f"\n2. CRITICAL LAYERS:")
print(f"   Layer {int(min_pr_layer)}: Minimum Participation Ratio (bottleneck)")
print(f"   Layer {int(max_rv_change_layer)}: Maximum R_V change (phase transition?)")

print(f"\n3. METRIC REDUNDANCY:")
if abs(corr_matrix.loc['eff_rank', 'pr']) > 0.95:
    print(f"   ✅ Effective Rank and PR are nearly identical (r={corr_matrix.loc['eff_rank', 'pr']:.3f})")
    print(f"   → Use whichever is more interpretable")
else:
    print(f"   ⚠️  Effective Rank and PR diverge (r={corr_matrix.loc['eff_rank', 'pr']:.3f})")
    print(f"   → They measure different aspects of dimensionality")

print(f"\n4. LAYER 21 VS LAYER 27:")
layer_21_pr = l5_summary[l5_summary['layer'] == 21][('pr', 'mean')].values[0]
layer_27_pr = l5_summary[l5_summary['layer'] == 27][('pr', 'mean')].values[0]
if layer_27_pr < layer_21_pr:
    print(f"   Layer 27 has LOWER PR than Layer 21")
    print(f"   → Layer 27 is more contracted")
else:
    print(f"   Layer 21 has lower PR than Layer 27")
    print(f"   → Need to check R_V change, not absolute PR")

print("\n" + "=" * 80)
