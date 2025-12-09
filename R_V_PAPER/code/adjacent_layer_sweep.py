#!/usr/bin/env python3
"""
ADJACENT LAYER SWEEP (L24-L30)
==============================

Critical validation: Confirm Layer 27 is the peak effect layer, not arbitrary.

Tests activation patching at layers 24, 25, 26, 27, 28, 29, 30
with same methodology (patch at layer X, measure at layer X).

EXPECTED PATTERN:
- Gradual build-up toward L27
- Peak effect at L27
- Maintenance or decay L28-L30

GROK'S CRITIQUE: "Adjacent Layer Gradient: Peak Confirmed"
"The table shows a sharp peak at L27 (-0.248, p<0.001) with build-up
(L26 -0.089) and maintenance (L28-29 ~ -0.2) - rules out smooth gradient."

This experiment tests that claim rigorously with n=30 pairs across all layers.

RUNTIME: ~30-45 minutes
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import validated functions
from VALIDATED_mistral7b_layer27_activation_patching import (
    compute_metrics_fast,
    EARLY_LAYER,
    WINDOW_SIZE
)

from n300_mistral_test_prompt_bank import prompt_bank_1c


def run_single_forward_get_V_at_layer(prompt_text, model, tokenizer, target_layer, device=None):
    """
    Run model and capture V at specified target layer and early layer.

    Returns:
        (v_early, v_target) tuple
    """
    if device is None:
        device = model.device

    v_early_list, v_target_list = [], []

    def capture_early(module, inp, out):
        v_early_list.append(out.detach()[0])
        return out

    def capture_target(module, inp, out):
        v_target_list.append(out.detach()[0])
        return out

    # Register hooks
    h_early = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(capture_early)
    h_target = model.model.layers[target_layer].self_attn.v_proj.register_forward_hook(capture_target)

    # Forward pass
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        _ = model(**inputs)

    # Clean up
    h_early.remove()
    h_target.remove()

    v_early = v_early_list[0] if v_early_list else None
    v_target = v_target_list[0] if v_target_list else None

    return v_early, v_target


def run_patched_forward_at_layer(baseline_text, rec_v_source, model, tokenizer,
                                   patch_layer, device=None):
    """
    Patch at specified layer, measure at same layer.

    Args:
        baseline_text: Baseline prompt
        rec_v_source: Recursive V activations to inject
        model: Transformer model
        tokenizer: Tokenizer
        patch_layer: Which layer to patch at
        device: Device

    Returns:
        (v_early, v_target_patched) tuple
    """
    if device is None:
        device = model.device

    v_early_list, v_target_list = [], []

    def patch_and_capture(module, inp, out):
        """Patch activations during forward pass"""
        out = out.clone()
        B, T, D = out.shape

        # Prepare source
        src = rec_v_source.to(out.device, dtype=out.dtype)
        T_src = src.shape[0]

        # Patch last WINDOW_SIZE positions
        k = min(WINDOW_SIZE, T, T_src)
        if k > 0:
            out[:, -k:, :] = src[-k:, :].unsqueeze(0)

        v_target_list.append(out.detach()[0])
        return out

    def capture_early(module, inp, out):
        v_early_list.append(out.detach()[0])
        return out

    # Register hooks
    layer = model.model.layers[patch_layer].self_attn
    h_patch = layer.v_proj.register_forward_hook(patch_and_capture)
    h_early = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(capture_early)

    # Forward pass
    inputs = tokenizer(
        baseline_text,
        return_tensors='pt',
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        _ = model(**inputs)

    # Clean up
    h_patch.remove()
    h_early.remove()

    v_early = v_early_list[0] if v_early_list else None
    v_target = v_target_list[0] if v_target_list else None

    return v_early, v_target


def run_adjacent_layer_sweep(model, tokenizer, n200_results_csv, num_pairs=30, device=None):
    """
    Test patching at layers 24-30 on a subset of successful pairs.

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        n200_results_csv: Path to n=151 results CSV
        num_pairs: Number of pairs to test (default 30)
        device: Device

    Returns:
        DataFrame with results for each layer
    """
    if device is None:
        device = model.device

    print("=" * 80)
    print("ADJACENT LAYER SWEEP (L24-L30)")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load n=151 results to select pairs
    n200_df = pd.read_csv(n200_results_csv)

    # Select pairs with strong main effect (for cleaner signal)
    # Sort by delta_main (most negative first)
    n200_df_sorted = n200_df.sort_values('delta_main')

    # Take top num_pairs with strongest negative delta
    selected_pairs = n200_df_sorted.head(num_pairs)

    print(f"Selected {len(selected_pairs)} pairs with strongest main effect")
    print(f"Delta range: {selected_pairs['delta_main'].min():.3f} to {selected_pairs['delta_main'].max():.3f}")
    print()

    # Layers to test
    layers_to_test = list(range(24, 31))  # 24, 25, 26, 27, 28, 29, 30

    print(f"Testing layers: {layers_to_test}")
    print()

    results = []

    for idx, row in selected_pairs.iterrows():
        rec_id = row['rec_id']
        base_id = row['base_id']

        print(f"[{len(results)//len(layers_to_test) + 1}/{len(selected_pairs)}] {rec_id} → {base_id}")

        # Get texts
        if rec_id not in prompt_bank_1c or base_id not in prompt_bank_1c:
            print(f"  ⚠️  Prompt not found, skipping")
            continue

        rec_text = prompt_bank_1c[rec_id]["text"]
        base_text = prompt_bank_1c[base_id]["text"]

        # Test each layer
        for layer_idx in layers_to_test:
            try:
                # 1. Get recursive activations at this layer
                v5_r, vL_r = run_single_forward_get_V_at_layer(rec_text, model, tokenizer,
                                                                layer_idx, device)

                # 2. Get baseline unpatched at this layer
                v5_b, vL_b = run_single_forward_get_V_at_layer(base_text, model, tokenizer,
                                                                layer_idx, device)

                # Compute R_V for baseline (unpatched)
                _, pr5_b = compute_metrics_fast(v5_b, WINDOW_SIZE)
                _, prL_b = compute_metrics_fast(vL_b, WINDOW_SIZE)

                if pr5_b is None or pr5_b == 0 or pd.isna(pr5_b):
                    continue

                rv_base = prL_b / pr5_b

                # 3. Patch baseline with recursive at this layer
                v5_p, vL_p = run_patched_forward_at_layer(base_text, vL_r, model, tokenizer,
                                                           layer_idx, device)

                # Compute R_V for patched
                _, pr5_p = compute_metrics_fast(v5_p, WINDOW_SIZE)
                _, prL_p = compute_metrics_fast(vL_p, WINDOW_SIZE)

                if pr5_p is None or pr5_p == 0 or pd.isna(pr5_p):
                    continue

                rv_patch = prL_p / pr5_p

                delta = rv_patch - rv_base

                results.append({
                    'pair_id': f"{rec_id}→{base_id}",
                    'rec_id': rec_id,
                    'base_id': base_id,
                    'layer': layer_idx,
                    'rv_base': rv_base,
                    'rv_patch': rv_patch,
                    'delta': delta
                })

            except Exception as e:
                print(f"  ❌ Error at layer {layer_idx}: {e}")
                continue

        print(f"  Completed all layers")

    df = pd.DataFrame(results)

    print()
    print("=" * 80)
    print("LAYER SWEEP COMPLETE")
    print("=" * 80)
    print(f"Total measurements: {len(df)}")
    print()

    # Analyze by layer
    print("RESULTS BY LAYER:")
    print("-" * 80)

    layer_summary = df.groupby('layer')['delta'].agg(['mean', 'std', 'count'])

    for layer in layers_to_test:
        layer_data = layer_summary.loc[layer] if layer in layer_summary.index else None
        if layer_data is not None:
            mean_delta = layer_data['mean']
            std_delta = layer_data['std']
            n = int(layer_data['count'])

            # t-test against zero
            try:
                from scipy import stats
                layer_deltas = df[df['layer'] == layer]['delta']
                t_stat, p_value = stats.ttest_1samp(layer_deltas, 0)

                marker = "★" if layer == 27 else " "
                print(f"{marker} Layer {layer}: Δ={mean_delta:+.3f}±{std_delta:.3f} (n={n}, t={t_stat:.2f}, p={p_value:.4f})")

            except:
                print(f"  Layer {layer}: Δ={mean_delta:+.3f}±{std_delta:.3f} (n={n})")

    print()

    # Find peak layer
    peak_layer = layer_summary['mean'].idxmin()
    peak_delta = layer_summary.loc[peak_layer, 'mean']

    print(f"PEAK LAYER: {peak_layer} (Δ={peak_delta:.3f})")

    if peak_layer == 27:
        print("✅ L27 confirmed as critical layer!")
    else:
        print(f"⚠️  Peak is at L{peak_layer}, not L27!")

    print()

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'adjacent_layer_sweep_{timestamp}.csv'
    df.to_csv(output_file, index=False)

    print(f"✅ Saved: {output_file}")

    # Visualization
    create_layer_sweep_plot(df, layer_summary, output_file.replace('.csv', '.png'))

    return df, layer_summary


def create_layer_sweep_plot(df, layer_summary, output_file):
    """
    Create publication-quality visualization of layer sweep.
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean delta by layer with error bars
    ax = axes[0]

    layers = sorted(layer_summary.index)
    means = [layer_summary.loc[l, 'mean'] for l in layers]
    stds = [layer_summary.loc[l, 'std'] for l in layers]

    ax.errorbar(layers, means, yerr=stds, marker='o', markersize=10,
                linewidth=2, capsize=5, capthick=2)

    # Highlight L27
    l27_mean = layer_summary.loc[27, 'mean']
    ax.scatter([27], [l27_mean], color='red', s=200, zorder=10, marker='*',
               label='L27 (validated)')

    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Δ (R_V)', fontsize=12)
    ax.set_title('Layer Sweep: Effect Size by Layer', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Individual pairs as spaghetti plot
    ax = axes[1]

    for pair_id in df['pair_id'].unique()[:20]:  # Plot first 20 pairs for clarity
        pair_df = df[df['pair_id'] == pair_id]
        ax.plot(pair_df['layer'], pair_df['delta'], alpha=0.3, linewidth=1)

    # Overlay mean
    ax.plot(layers, means, color='red', linewidth=3, label='Mean', zorder=10)

    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.axvline(27, color='red', linestyle=':', alpha=0.5, label='L27')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Δ (R_V)', fontsize=12)
    ax.set_title('Individual Pair Trajectories', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plot: {output_file}")


def main():
    print("=" * 80)
    print("LOADING MODEL...")
    print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ Model loaded")
    print()

    # Run sweep
    df, summary = run_adjacent_layer_sweep(
        model=model,
        tokenizer=tokenizer,
        n200_results_csv='mistral7b_n200_BULLETPROOF.csv',
        num_pairs=30,
        device=model.device
    )

    print()
    print("=" * 80)
    print("✅ LAYER SWEEP COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
