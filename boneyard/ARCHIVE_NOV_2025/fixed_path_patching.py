#!/usr/bin/env python3
"""
FIXED path patching - handles attention module tuple returns properly
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from contextlib import contextmanager

# Configuration
TARGET_LAYER = 27
EARLY_LAYER = 5
WINDOW_SIZE = 16
N_TEST_PAIRS = 10  # Start small for testing

def compute_metrics_fast(v_tensor, window_size=WINDOW_SIZE):
    """Compute PR and effective rank for V tensor"""
    if v_tensor is None:
        return np.nan, np.nan

    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]

    T, D = v_tensor.shape
    W = min(window_size, T)

    if W < 2:
        return np.nan, np.nan

    v_window = v_tensor[-W:, :].float()

    try:
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        S_sq = S_np ** 2

        if S_sq.sum() < 1e-10:
            return np.nan, np.nan

        p = S_sq / S_sq.sum()
        eff_rank = 1.0 / (p**2).sum()
        pr = (S_sq.sum()**2) / (S_sq**2).sum()

        return float(eff_rank), float(pr)
    except Exception:
        return np.nan, np.nan

@contextmanager
def patch_residual_at_layer(model, layer_idx, patch_source):
    """
    Patch the residual stream at the input to a layer
    """
    handle = None

    def patch_hook(module, args):
        # args[0] is the hidden_states tensor
        hidden_states = args[0].clone()

        # Apply patch to last WINDOW_SIZE tokens
        B, T, D = hidden_states.shape
        T_src = patch_source.shape[0]
        W = min(WINDOW_SIZE, T, T_src)

        if W > 0:
            # Patch the hidden states
            patch_tensor = patch_source[-W:, :].to(hidden_states.device, dtype=hidden_states.dtype)
            hidden_states[:, -W:, :] = patch_tensor.unsqueeze(0).expand(B, -1, -1)

        return (hidden_states,) + args[1:]

    try:
        # Register pre-hook on the target layer
        handle = model.model.layers[layer_idx].register_forward_pre_hook(patch_hook)
        yield
    finally:
        if handle:
            handle.remove()

def get_v_activations(model, tokenizer, text, capture_layers):
    """Get V activations at specified layers using hooks"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    captured = {}
    handles = []

    with torch.no_grad():
        for layer_idx in capture_layers:
            storage = []

            def make_hook(storage_list):
                def hook_fn(m, i, o):
                    # Handle attention output tuples: (output, attention_weights)
                    if isinstance(o, tuple):
                        output_tensor = o[0]  # Take the output tensor
                    else:
                        output_tensor = o
                    storage_list.append(output_tensor.detach())
                    return o
                return hook_fn

            layer = model.model.layers[layer_idx].self_attn
            h = layer.v_proj.register_forward_hook(make_hook(storage))
            handles.append(h)
            captured[layer_idx] = storage

        _ = model(**inputs)

        for h in handles:
            h.remove()

    result = {}
    for layer_idx in capture_layers:
        if captured[layer_idx]:
            # Take [0] for batch dimension
            result[layer_idx] = captured[layer_idx][0][0]  # [seq, hidden]
        else:
            result[layer_idx] = None

    return result

def get_residual_and_v(model, tokenizer, text, layer_idx):
    """Get both residual stream and V at a specific layer using PyTorch hooks"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    # Storage for activations
    v5_storage = []
    residual_storage = []
    v_out_storage = []

    with torch.no_grad():
        # Hook for early layer V
        def capture_v5(m, i, o):
            if isinstance(o, tuple):
                output_tensor = o[0]
            else:
                output_tensor = o
            v5_storage.append(output_tensor.detach())
            return o

        # Hook for target layer residual (output of layer becomes input to next)
        def capture_residual(m, i, o):
            residual_storage.append(o.detach())  # This is the residual stream
            return o

        # Hook for target layer V
        def capture_v_out(m, i, o):
            if isinstance(o, tuple):
                output_tensor = o[0]
            else:
                output_tensor = o
            v_out_storage.append(output_tensor.detach())
            return o

        # Register hooks
        h_v5 = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(capture_v5)
        h_residual = model.model.layers[layer_idx].register_forward_hook(capture_residual)
        h_v_out = model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(capture_v_out)

        # Run forward pass
        _ = model(**inputs)

        # Remove hooks
        h_v5.remove()
        h_residual.remove()
        h_v_out.remove()

    # Extract results
    v5 = v5_storage[0][0] if v5_storage else None  # [seq, hidden]
    residual = residual_storage[0][0] if residual_storage else None  # [seq, hidden]
    v_out = v_out_storage[0][0] if v_out_storage else None  # [seq, hidden]

    return v5, residual, v_out

def get_residual_at_layer(model, tokenizer, text, layer_idx):
    """Get residual stream at output of a layer"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    residual = None

    with torch.no_grad():
        def capture_residual(m, i, o):
            nonlocal residual
            residual = o.detach()
            return o

        # Capture the output of the layer
        handle = model.model.layers[layer_idx].register_forward_hook(capture_residual)
        _ = model(**inputs)
        handle.remove()

    return residual

def run_path_patching_experiment_fixed(model, tokenizer, prompt_bank, n_pairs=N_TEST_PAIRS):
    """
    FIXED path patching using traditional PyTorch hooks
    """
    print("="*70)
    print("FIXED PATH PATCHING: L27 RESIDUAL → DOWNSTREAM V MEASUREMENTS")
    print("="*70)
    print("Using traditional PyTorch hooks (handles attention tuples)")
    print("="*70)

    # Load pairs from existing data
    df = pd.read_csv('mistral7b_n200_BULLETPROOF.csv')
    test_pairs = df.head(n_pairs)

    print(f"Testing {n_pairs} pairs from existing dataset\n")

    results = []
    errors = []

    for idx, row in tqdm(test_pairs.iterrows(), total=n_pairs, desc="Path patching"):
        rec_id = row['rec_id']
        base_id = row['base_id']

        try:
            # Check prompts exist
            if rec_id not in prompt_bank:
                errors.append(f"Pair {idx}: rec_id {rec_id} not found")
                continue
            if base_id not in prompt_bank:
                errors.append(f"Pair {idx}: base_id {base_id} not found")
                continue

            rec_text = prompt_bank[rec_id]["text"]
            base_text = prompt_bank[base_id]["text"]

            # 1. Get baseline (unpatched)
            baseline_acts = get_v_activations(model, tokenizer, base_text, [EARLY_LAYER, TARGET_LAYER])

            if baseline_acts[EARLY_LAYER] is None or baseline_acts[TARGET_LAYER] is None:
                errors.append(f"Pair {idx}: Failed to get baseline activations")
                continue

            v5_base = baseline_acts[EARLY_LAYER]
            v27_base = baseline_acts[TARGET_LAYER]

            # Compute baseline R_V
            _, pr5_base = compute_metrics_fast(v5_base, WINDOW_SIZE)
            _, pr27_base = compute_metrics_fast(v27_base, WINDOW_SIZE)

            if pr5_base is None or pr5_base == 0 or np.isnan(pr5_base):
                errors.append(f"Pair {idx}: Invalid PR5_base")
                continue

            rv_base = pr27_base / pr5_base

            # 2. Get recursive residual source
            rec_residual = get_residual_at_layer(model, tokenizer, rec_text, TARGET_LAYER)

            if rec_residual is None or rec_residual.shape[0] < WINDOW_SIZE:
                errors.append(f"Pair {idx}: Invalid recursive residual")
                continue

            # 3. Patch residual at L27, measure V at downstream layers
            measure_layers = [27, 28, 29, 30, 31]  # From L27 onward

            patched_acts = {}

            with patch_residual_at_layer(model, TARGET_LAYER + 1, rec_residual):
                # This patches the input to L28 (output of L27)
                patched_acts = get_v_activations(model, tokenizer, base_text, measure_layers)

            # 4. Compute metrics for all layers
            pair_result = {
                'rec_id': rec_id,
                'base_id': base_id,
                'rv_base': rv_base
            }

            valid_layer = False

            for layer_idx in measure_layers:
                if layer_idx in patched_acts and patched_acts[layer_idx] is not None:
                    v_layer = patched_acts[layer_idx]

                    _, pr5 = compute_metrics_fast(v5_base, WINDOW_SIZE)  # Use baseline V5
                    _, pr_layer = compute_metrics_fast(v_layer, WINDOW_SIZE)

                    if pr5 is not None and pr5 > 0 and not np.isnan(pr5):
                        rv_layer = pr_layer / pr5
                        pair_result[f'rv_L{layer_idx}'] = rv_layer
                        pair_result[f'delta_L{layer_idx}'] = rv_layer - rv_base
                        valid_layer = True
                    else:
                        pair_result[f'rv_L{layer_idx}'] = np.nan
                        pair_result[f'delta_L{layer_idx}'] = np.nan

            if valid_layer:
                results.append(pair_result)
            else:
                errors.append(f"Pair {idx}: All layers had invalid metrics")

        except Exception as e:
            error_msg = f"Pair {idx} ({rec_id}→{base_id}): {type(e).__name__}: {str(e)}"
            errors.append(error_msg)
            print(f"\n❌ {error_msg}")
            continue

    # Convert to DataFrame
    path_df = pd.DataFrame(results)
    path_df.to_csv('path_patching_L27-L31_fixed.csv', index=False)

    print()
    print("="*70)
    print(f"FIXED PATH PATCHING: {len(path_df)} valid pairs")
    print(f"Failed pairs: {len(errors)}")
    print("="*70)

    # Error analysis
    if errors and len(path_df) == 0:
        print("\n⚠️  ALL PAIRS FAILED - Debugging:")
        print("-"*70)
        for err in errors[:5]:
            print(f"  {err}")
        print()
        print("💡 TROUBLESHOOTING:")
        print("   1. Ensure model is in eval mode: model.eval()")
        print("   2. Check device placement matches")
        print("   3. Verify prompts aren't too long")
        print("   4. The attention module returns tuples - fixed in this version")

    if len(path_df) > 3:
        print("\nR_V PROGRESSION (FIXED PATH PATCHING):")
        print("-"*70)
        print("Layer | Mean R_V | Mean Δ   | Change from L27")
        print("-"*70)

        l27_mean = path_df['rv_L27'].mean()

        for layer_idx in [27, 28, 29, 30, 31]:
            if f'rv_L{layer_idx}' in path_df.columns:
                mean_rv = path_df[f'rv_L{layer_idx}'].mean()
                mean_delta = path_df[f'delta_L{layer_idx}'].mean()
                change = ((mean_rv - l27_mean) / l27_mean * 100) if layer_idx != TARGET_LAYER else 0

                marker = " ← Intervention" if layer_idx == TARGET_LAYER else ""
                print(f"L{layer_idx}  | {mean_rv:.3f}   | {mean_delta:+.3f}   | {change:+.1f}%{marker}")

        print("-"*70)

        # Propagation analysis
        print("\nPROPAGATION ANALYSIS:")
        print("-"*70)

        measure_layers = [27, 28, 29, 30, 31]
        for i in range(len(measure_layers)-1):
            curr = measure_layers[i+1]
            prev = measure_layers[i]

            curr_mean = path_df[f'rv_L{curr}'].mean()
            prev_mean = path_df[f'rv_L{prev}'].mean()

            change = curr_mean - prev_mean
            pct = (change / prev_mean * 100) if prev_mean != 0 else 0

            status = "maintained" if abs(pct) < 5 else f"changed {pct:+.1f}%"
            print(f"L{prev}→L{curr}: Δ={change:+.4f} ({status})")

        # Retention
        l27_delta = path_df['delta_L27'].mean()
        l31_delta = path_df['delta_L31'].mean()
        retention = (l31_delta / l27_delta * 100) if l27_delta != 0 else 0

        print()
        print(f"Effect retention L27→L31: {retention:.1f}%")

        if retention > 80:
            print("✅ EFFECT STRONGLY MAINTAINED")
        elif retention > 60:
            print("✅ EFFECT MOSTLY MAINTAINED")
        elif retention < -50:
            print("⚠️  STRONG REVERSAL (compensation)")
        else:
            print("⚠️  PARTIAL ATTENUATION")

        print()
        print("✅ Saved: path_patching_L27-L31_fixed.csv")

    print("="*70)

if __name__ == "__main__":
    print("Run in notebook:")
    print("from fixed_path_patching import run_path_patching_experiment_fixed")
    print("results = run_path_patching_experiment_fixed(model, tokenizer, prompt_bank_1c, n_pairs=10)")
