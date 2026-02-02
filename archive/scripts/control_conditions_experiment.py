#!/usr/bin/env python3
"""
CONTROL CONDITIONS EXPERIMENT
==============================

Tests control conditions to verify the Layer 27 effect is specific to recursive
activations, not just any intervention.

VALIDATED EFFECT (n=15):
------------------------
Real L27 patch: Δ = -0.264, p<0.0001, 100% consistency

CONTROL CONDITIONS:
------------------
1. Random Noise Patch: Norm-matched random vectors
   - H₀: Random noise should NOT cause transfer
   - Expected: Δ ≈ 0 or positive

2. Shuffled Activation Patch: Permuted token positions
   - H₀: Destroying token order should break the effect
   - Expected: Δ ≈ 0 or positive

3. Wrong Layer Patch: Layer 15 (47% depth) instead of Layer 27 (84% depth)
   - H₀: Early layers don't carry recursive geometry
   - Expected: Δ ≈ 0 or weaker than Layer 27

EXPECTED RUNTIME: ~15-20 minutes on GPU
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from datetime import datetime

# Import validated functions
from VALIDATED_mistral7b_layer27_activation_patching import (
    capture_v_at_layer,
    compute_metrics_fast,
    run_single_forward_get_V,
    TARGET_LAYER,
    EARLY_LAYER,
    WINDOW_SIZE
)

# Import prompt bank
from n300_mistral_test_prompt_bank import prompt_bank_1c


def run_control_patched_forward(baseline_text, control_v_source, model, tokenizer,
                                  patch_layer=TARGET_LAYER, device=None):
    """
    Run baseline with CONTROL patching (same interface as validated function).

    Args:
        baseline_text: Baseline prompt
        control_v_source: Control activations to inject (random/shuffled/wrong-layer)
        model: Transformer model
        tokenizer: Tokenizer
        patch_layer: Layer to patch at
        device: Device

    Returns:
        (v_early, v_target_patched) tuple
    """
    if device is None:
        device = model.device

    v_early_list, v_target_list = [], []

    def patch_and_capture(module, inp, out):
        """Hook function that patches activations during forward pass"""
        out = out.clone()
        B, T, D = out.shape

        # Prepare source
        src = control_v_source.to(out.device, dtype=out.dtype)
        T_src = src.shape[0]

        # Patch last WINDOW_SIZE positions
        k = min(WINDOW_SIZE, T, T_src)
        if k > 0:
            out[:, -k:, :] = src[-k:, :].unsqueeze(0)

        v_target_list.append(out.detach()[0])
        return out

    # Register hooks
    layer = model.model.layers[patch_layer].self_attn
    h_patch = layer.v_proj.register_forward_hook(patch_and_capture)

    def capture_early(module, inp, out):
        v_early_list.append(out.detach()[0])
        return out

    h_early = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(
        capture_early
    )

    # Run forward pass
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


def create_random_control(v_real, device):
    """
    Create norm-matched random noise with same shape as real activations.

    Args:
        v_real: Real activation tensor [seq_len, hidden]
        device: Device to create on

    Returns:
        Random tensor with same shape and frobenius norm
    """
    # Create random tensor
    v_random = torch.randn_like(v_real, device=device)

    # Match Frobenius norm
    real_norm = torch.linalg.norm(v_real)
    random_norm = torch.linalg.norm(v_random)

    v_random = v_random * (real_norm / random_norm)

    return v_random


def create_shuffled_control(v_real):
    """
    Shuffle token positions to destroy sequential structure.

    Args:
        v_real: Real activation tensor [seq_len, hidden]

    Returns:
        Shuffled tensor (same tokens, random order)
    """
    # Get random permutation of token indices
    perm = torch.randperm(v_real.shape[0])

    # Shuffle along sequence dimension
    v_shuffled = v_real[perm, :]

    return v_shuffled


def run_control_experiment(model, tokenizer, prompt_bank, num_pairs=5, device=None):
    """
    Run all three control conditions on num_pairs.

    Returns:
        DataFrame with results for all conditions
    """
    if device is None:
        device = model.device

    print("=" * 80)
    print("CONTROL CONDITIONS EXPERIMENT")
    print("=" * 80)
    print(f"Testing {num_pairs} pairs across 3 control conditions")
    print()

    # Build pairs
    pairs = []
    for i in range(1, num_pairs + 1):
        rec_id = f"L5_refined_{i:02d}"
        base_id = f"long_new_{i:02d}"
        if rec_id in prompt_bank and base_id in prompt_bank:
            pairs.append((rec_id, base_id))

    print(f"Pairs: {len(pairs)}")
    print()

    rows = []

    for pair_idx, (rec_id, base_id) in enumerate(pairs, 1):
        rec_text = prompt_bank[rec_id]["text"]
        base_text = prompt_bank[base_id]["text"]

        # Check lengths
        base_tokens = tokenizer(base_text, return_tensors='pt')
        base_len = base_tokens['input_ids'].shape[1]

        if base_len < WINDOW_SIZE:
            print(f"⚠️  Skipping pair {pair_idx}: baseline too short")
            continue

        print(f"Pair {pair_idx}/{len(pairs)}: {rec_id} → {base_id}")

        # Get real recursive activations at Layer 27
        v5_r, v27_r = run_single_forward_get_V(rec_text, model, tokenizer,
                                                TARGET_LAYER, device)

        # Baseline (unpatched)
        v5_b, v27_b = run_single_forward_get_V(base_text, model, tokenizer,
                                                TARGET_LAYER, device)
        _, pr5_b = compute_metrics_fast(v5_b, WINDOW_SIZE)
        _, pr27_b = compute_metrics_fast(v27_b, WINDOW_SIZE)
        rv_base = pr27_b / pr5_b if pr5_b > 0 else np.nan

        # CONTROL 1: Random Noise Patch
        v_random = create_random_control(v27_r, device)
        v5_c1, v27_c1 = run_control_patched_forward(base_text, v_random, model,
                                                      tokenizer, TARGET_LAYER, device)
        _, pr5_c1 = compute_metrics_fast(v5_c1, WINDOW_SIZE)
        _, pr27_c1 = compute_metrics_fast(v27_c1, WINDOW_SIZE)
        rv_c1 = pr27_c1 / pr5_c1 if pr5_c1 > 0 else np.nan
        delta_c1 = rv_c1 - rv_base

        # CONTROL 2: Shuffled Activation Patch
        v_shuffled = create_shuffled_control(v27_r)
        v5_c2, v27_c2 = run_control_patched_forward(base_text, v_shuffled, model,
                                                      tokenizer, TARGET_LAYER, device)
        _, pr5_c2 = compute_metrics_fast(v5_c2, WINDOW_SIZE)
        _, pr27_c2 = compute_metrics_fast(v27_c2, WINDOW_SIZE)
        rv_c2 = pr27_c2 / pr5_c2 if pr5_c2 > 0 else np.nan
        delta_c2 = rv_c2 - rv_base

        # CONTROL 3: Wrong Layer Patch (Layer 15 instead of 27)
        v5_r_l15, v15_r = run_single_forward_get_V(rec_text, model, tokenizer,
                                                     15, device)
        v5_c3, v27_c3 = run_control_patched_forward(base_text, v15_r, model,
                                                      tokenizer, TARGET_LAYER, device)
        _, pr5_c3 = compute_metrics_fast(v5_c3, WINDOW_SIZE)
        _, pr27_c3 = compute_metrics_fast(v27_c3, WINDOW_SIZE)
        rv_c3 = pr27_c3 / pr5_c3 if pr5_c3 > 0 else np.nan
        delta_c3 = rv_c3 - rv_base

        rows.append({
            'pair': f"{rec_id}→{base_id}",
            'rv_baseline': rv_base,
            'rv_random': rv_c1,
            'rv_shuffled': rv_c2,
            'rv_wrong_layer': rv_c3,
            'delta_random': delta_c1,
            'delta_shuffled': delta_c2,
            'delta_wrong_layer': delta_c3
        })

        print(f"  Baseline:     {rv_base:.3f}")
        print(f"  Random:       {rv_c1:.3f} (Δ={delta_c1:+.3f})")
        print(f"  Shuffled:     {rv_c2:.3f} (Δ={delta_c2:+.3f})")
        print(f"  Wrong Layer:  {rv_c3:.3f} (Δ={delta_c3:+.3f})")
        print()

    df = pd.DataFrame(rows)

    # Summary
    print("=" * 80)
    print("CONTROL SUMMARY")
    print("=" * 80)

    for control_name, delta_col in [
        ('Random Noise', 'delta_random'),
        ('Shuffled Tokens', 'delta_shuffled'),
        ('Wrong Layer (L15)', 'delta_wrong_layer')
    ]:
        mean_delta = df[delta_col].mean()
        std_delta = df[delta_col].std()
        negative_count = (df[delta_col] < 0).sum()
        consistency = (negative_count / len(df)) * 100

        print(f"\n{control_name}:")
        print(f"  Mean Δ: {mean_delta:+.3f} ± {std_delta:.3f}")
        print(f"  Negative: {negative_count}/{len(df)} ({consistency:.1f}%)")

        # Statistical test
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_1samp(df[delta_col], 0)
            print(f"  t-test: t={t_stat:.3f}, p={p_value:.4f}")

            if abs(mean_delta) < 0.05 and p_value > 0.05:
                print(f"  ✅ No significant effect (as expected)")
            elif mean_delta < -0.1 and p_value < 0.05:
                print(f"  ⚠️  Unexpected negative effect!")
            else:
                print(f"  ℹ️  Weak/mixed effect")
        except:
            pass

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("\nExpected outcomes:")
    print("  Random Noise:    Δ ≈ 0 (no structured information)")
    print("  Shuffled Tokens: Δ ≈ 0 (sequential structure destroyed)")
    print("  Wrong Layer:     Δ ≈ 0 or weaker than L27")
    print()
    print("If controls show NO effect but real L27 shows strong effect:")
    print("  → Effect is SPECIFIC to Layer 27 recursive geometry")
    print("=" * 80)

    return df


def main():
    print("=" * 80)
    print("CONTROL CONDITIONS EXPERIMENT")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load model
    print("Loading Mistral-7B-Instruct-v0.2...")
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

    # Run controls on first 5 pairs
    results = run_control_experiment(
        model=model,
        tokenizer=tokenizer,
        prompt_bank=prompt_bank_1c,
        num_pairs=5,
        device=model.device
    )

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'mistral7b_L27_controls_n{len(results)}_{timestamp}.csv'
    results.to_csv(output_file, index=False)

    print(f"\n✅ Results saved: {output_file}")


if __name__ == "__main__":
    main()
