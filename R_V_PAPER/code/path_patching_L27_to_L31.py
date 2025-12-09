#!/usr/bin/env python3
"""
PATH PATCHING: L27 → L31 Causal Trace
======================================

Traces how Layer 27 intervention propagates through downstream layers.

METHODOLOGY:
1. Patch at L27 (inject recursive activations)
2. Measure R_V at L27, L28, L29, L30, L31
3. Compare to baseline (no patch) measurements at same layers
4. Quantify: How much does the effect persist/amplify/decay downstream?

CRITICAL QUESTION:
Does the L27 intervention affect ONLY L27, or does it cascade through
the residual stream to later layers?

EXPECTED OUTCOMES:
- Immediate effect at L27 (by design)
- Propagation to L28-L31 via residual connections
- Possible amplification (if later layers process the injected geometry)
- Or decay (if later layers "correct" the intervention)

RUNTIME: ~30 minutes for n=30 pairs
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


def run_path_patching_experiment(model, tokenizer, n200_results_csv, num_pairs=30, device=None):
    """
    Trace causal path from L27 intervention through downstream layers.

    For each pair:
    1. Get recursive activations at L27
    2. Inject at L27, measure at L27, L28, L29, L30, L31
    3. Compare to baseline (unpatched) at same layers

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        n200_results_csv: Path to n=151 results
        num_pairs: Number of pairs to test
        device: Device

    Returns:
        DataFrame with path measurements
    """
    if device is None:
        device = model.device

    print("=" * 80)
    print("PATH PATCHING: L27 → L31 CAUSAL TRACE")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load n=151 results to select pairs
    n200_df = pd.read_csv(n200_results_csv)

    # Select pairs with strong main effect
    n200_df_sorted = n200_df.sort_values('delta')
    selected_pairs = n200_df_sorted.head(num_pairs)

    print(f"Selected {len(selected_pairs)} pairs with strongest effects")
    print()

    # Layers to measure (patch at L27, measure downstream)
    patch_layer = 27
    measure_layers = [27, 28, 29, 30, 31]

    print(f"Patch layer: {patch_layer}")
    print(f"Measure layers: {measure_layers}")
    print()

    results = []

    for idx, row in selected_pairs.iterrows():
        rec_id = row['rec_id']
        base_id = row['base_id']

        print(f"[{len(results)//(len(measure_layers)*2) + 1}/{len(selected_pairs)}] {rec_id} → {base_id}")

        # Get texts
        if rec_id not in prompt_bank_1c or base_id not in prompt_bank_1c:
            print(f"  ⚠️  Prompt not found, skipping")
            continue

        rec_text = prompt_bank_1c[rec_id]["text"]
        base_text = prompt_bank_1c[base_id]["text"]

        try:
            # STEP 1: Get recursive activations at L27
            v27_rec = get_v_at_layer(rec_text, model, tokenizer, patch_layer, device)

            if v27_rec is None:
                print(f"  ❌ Failed to get recursive activations")
                continue

            # STEP 2: Baseline (no patch) - measure at all downstream layers
            for measure_layer in measure_layers:
                v5_base, vL_base = get_v_at_early_and_target(base_text, model, tokenizer,
                                                               measure_layer, device)

                _, pr5_base = compute_metrics_fast(v5_base, WINDOW_SIZE)
                _, prL_base = compute_metrics_fast(vL_base, WINDOW_SIZE)

                if pr5_base is None or pr5_base == 0 or pd.isna(pr5_base):
                    continue

                rv_base = prL_base / pr5_base

                results.append({
                    'pair_id': f"{rec_id}→{base_id}",
                    'rec_id': rec_id,
                    'base_id': base_id,
                    'condition': 'baseline',
                    'measure_layer': measure_layer,
                    'rv': rv_base
                })

            # STEP 3: Patched (inject at L27) - measure at all downstream layers
            for measure_layer in measure_layers:
                v5_patch, vL_patch = patch_at_L27_measure_at_layer(
                    base_text, v27_rec, model, tokenizer,
                    patch_layer, measure_layer, device
                )

                _, pr5_patch = compute_metrics_fast(v5_patch, WINDOW_SIZE)
                _, prL_patch = compute_metrics_fast(vL_patch, WINDOW_SIZE)

                if pr5_patch is None or pr5_patch == 0 or pd.isna(pr5_patch):
                    continue

                rv_patch = prL_patch / pr5_patch

                results.append({
                    'pair_id': f"{rec_id}→{base_id}",
                    'rec_id': rec_id,
                    'base_id': base_id,
                    'condition': 'patched',
                    'measure_layer': measure_layer,
                    'rv': rv_patch
                })

            print(f"  ✅ Completed path trace")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    df = pd.DataFrame(results)

    print()
    print("=" * 80)
    print("PATH PATCHING COMPLETE")
    print("=" * 80)
    print(f"Total measurements: {len(df)}")
    print()

    # Analyze propagation
    analyze_path_propagation(df, measure_layers)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'path_patching_L27_to_L31_{timestamp}.csv'
    df.to_csv(output_file, index=False)

    print(f"✅ Saved: {output_file}")

    # Visualization
    create_path_patching_plot(df, measure_layers, output_file.replace('.csv', '.png'))

    return df


def get_v_at_layer(prompt_text, model, tokenizer, layer_idx, device):
    """
    Get V activations at specified layer.

    Returns:
        Tensor [seq_len, hidden]
    """
    v_list = []

    def capture(module, inp, out):
        v_list.append(out.detach()[0])
        return out

    h = model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(capture)

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        _ = model(**inputs)

    h.remove()

    return v_list[0] if v_list else None


def get_v_at_early_and_target(prompt_text, model, tokenizer, target_layer, device):
    """
    Get V at early layer (L5) and target layer.

    Returns:
        (v_early, v_target) tuple
    """
    v_early_list, v_target_list = [], []

    def capture_early(module, inp, out):
        v_early_list.append(out.detach()[0])
        return out

    def capture_target(module, inp, out):
        v_target_list.append(out.detach()[0])
        return out

    h_early = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(capture_early)
    h_target = model.model.layers[target_layer].self_attn.v_proj.register_forward_hook(capture_target)

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        _ = model(**inputs)

    h_early.remove()
    h_target.remove()

    v_early = v_early_list[0] if v_early_list else None
    v_target = v_target_list[0] if v_target_list else None

    return v_early, v_target


def patch_at_L27_measure_at_layer(baseline_text, rec_v27, model, tokenizer,
                                    patch_layer, measure_layer, device):
    """
    Patch at L27, measure at specified downstream layer.

    Args:
        baseline_text: Baseline prompt
        rec_v27: Recursive V activations at L27
        model: Model
        tokenizer: Tokenizer
        patch_layer: Layer to patch (should be 27)
        measure_layer: Layer to measure (27-31)
        device: Device

    Returns:
        (v_early, v_measured) tuple
    """
    v_early_list, v_measure_list = [], []

    def patch_hook(module, inp, out):
        """Patch at L27"""
        out = out.clone()
        B, T, D = out.shape

        src = rec_v27.to(out.device, dtype=out.dtype)
        T_src = src.shape[0]

        k = min(WINDOW_SIZE, T, T_src)
        if k > 0:
            out[:, -k:, :] = src[-k:, :].unsqueeze(0)

        return out

    def capture_early(module, inp, out):
        v_early_list.append(out.detach()[0])
        return out

    def capture_measure(module, inp, out):
        v_measure_list.append(out.detach()[0])
        return out

    # Register hooks
    h_patch = model.model.layers[patch_layer].self_attn.v_proj.register_forward_hook(patch_hook)
    h_early = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(capture_early)
    h_measure = model.model.layers[measure_layer].self_attn.v_proj.register_forward_hook(capture_measure)

    inputs = tokenizer(baseline_text, return_tensors='pt', truncation=True, max_length=512).to(device)

    with torch.no_grad():
        _ = model(**inputs)

    # Clean up
    h_patch.remove()
    h_early.remove()
    h_measure.remove()

    v_early = v_early_list[0] if v_early_list else None
    v_measure = v_measure_list[0] if v_measure_list else None

    return v_early, v_measure


def analyze_path_propagation(df, measure_layers):
    """
    Analyze how the intervention propagates through layers.
    """
    print("=" * 80)
    print("PATH PROPAGATION ANALYSIS")
    print("=" * 80)

    # Compute deltas for each layer
    for layer in measure_layers:
        layer_data = df[df['measure_layer'] == layer]

        baseline_rvs = layer_data[layer_data['condition'] == 'baseline']['rv']
        patched_rvs = layer_data[layer_data['condition'] == 'patched']['rv']

        # Match pairs
        baseline_df = layer_data[layer_data['condition'] == 'baseline'].set_index('pair_id')
        patched_df = layer_data[layer_data['condition'] == 'patched'].set_index('pair_id')

        common_pairs = baseline_df.index.intersection(patched_df.index)

        if len(common_pairs) == 0:
            continue

        deltas = (patched_df.loc[common_pairs, 'rv'] - baseline_df.loc[common_pairs, 'rv']).values

        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)

        # Statistical test
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_1samp(deltas, 0)
            cohens_d = mean_delta / std_delta if std_delta > 0 else 0

            print(f"\nLayer {layer} (n={len(common_pairs)}):")
            print(f"  Mean Δ:      {mean_delta:+.3f} ± {std_delta:.3f}")
            print(f"  Cohen's d:   {cohens_d:.2f}")
            print(f"  t={t_stat:.2f}, p={p_value:.2e}")

            if layer == 27:
                print(f"  → Immediate effect (injection layer)")
            elif abs(mean_delta) > 0.05:
                print(f"  → Strong propagation downstream")
            else:
                print(f"  → Weak propagation")

        except:
            print(f"\nLayer {layer}: Δ={mean_delta:+.3f} ± {std_delta:.3f}")

    print()
    print("=" * 80)


def create_path_patching_plot(df, measure_layers, output_file):
    """
    Visualize path propagation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean R_V by layer and condition
    ax = axes[0]

    baseline_means = []
    patched_means = []
    baseline_stds = []
    patched_stds = []

    for layer in measure_layers:
        layer_data = df[df['measure_layer'] == layer]

        baseline_rvs = layer_data[layer_data['condition'] == 'baseline']['rv']
        patched_rvs = layer_data[layer_data['condition'] == 'patched']['rv']

        baseline_means.append(baseline_rvs.mean())
        patched_means.append(patched_rvs.mean())
        baseline_stds.append(baseline_rvs.std())
        patched_stds.append(patched_rvs.std())

    ax.errorbar(measure_layers, baseline_means, yerr=baseline_stds,
                marker='o', label='Baseline', linewidth=2, capsize=5)
    ax.errorbar(measure_layers, patched_means, yerr=patched_stds,
                marker='s', label='Patched (L27)', linewidth=2, capsize=5)

    ax.axvline(27, color='red', linestyle='--', alpha=0.5, label='Injection point')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean R_V', fontsize=12)
    ax.set_title('Path Propagation: Baseline vs Patched', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Delta by layer
    ax = axes[1]

    deltas_by_layer = []
    for layer in measure_layers:
        layer_data = df[df['measure_layer'] == layer]

        baseline_df = layer_data[layer_data['condition'] == 'baseline'].set_index('pair_id')
        patched_df = layer_data[layer_data['condition'] == 'patched'].set_index('pair_id')

        common_pairs = baseline_df.index.intersection(patched_df.index)

        if len(common_pairs) > 0:
            deltas = (patched_df.loc[common_pairs, 'rv'] - baseline_df.loc[common_pairs, 'rv']).values
            deltas_by_layer.append(deltas)
        else:
            deltas_by_layer.append([])

    positions = measure_layers
    ax.violinplot(deltas_by_layer, positions=positions, widths=0.7, showmeans=True, showmedians=True)

    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.axvline(27, color='red', linestyle='--', alpha=0.5, label='Injection point')
    ax.set_xlabel('Measurement Layer', fontsize=12)
    ax.set_ylabel('Δ (R_V)', fontsize=12)
    ax.set_title('Effect Propagation Downstream', fontsize=14, fontweight='bold')
    ax.set_xticks(positions)
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

    # Run path patching
    # Use n=15 results (adjust to available file)
    df = run_path_patching_experiment(
        model=model,
        tokenizer=tokenizer,
        n200_results_csv='mistral7b_L27_patching_n15_results_20251116_211154.csv',
        num_pairs=15,  # Use all 15 available pairs
        device=model.device
    )

    print()
    print("=" * 80)
    print("✅ PATH PATCHING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
