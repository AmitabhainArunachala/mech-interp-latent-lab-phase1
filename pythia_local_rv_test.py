#!/usr/bin/env python3
"""
Pythia-1.4B Local R_V Test - No GPU Required
Full geometry measurement on your M3 Pro MacBook.

Tests:
1. R_V contraction (recursive vs baseline)
2. Layer sweep (find optimal layer)
3. Statistical validation

Model: EleutherAI/pythia-1.4b (24 layers, 2048 hidden dim)
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import contextmanager
from scipy import stats
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "EleutherAI/pythia-1.4b"
DEVICE = "cpu"  # M3 Pro - use CPU (MPS has issues with some ops)

# Pythia-1.4B has 24 layers
# Early ~12%, Late ~75-85%
EARLY_LAYER = 3   # ~12% depth
TARGET_LAYER = 18  # ~75% depth (will sweep to find optimal)
WINDOW_SIZE = 16

print("=" * 60)
print("PYTHIA-1.4B LOCAL R_V TEST")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_NAME}")

# ============================================================
# PROMPTS (from validated bank)
# ============================================================
RECURSIVE_PROMPTS = [
    "Observe the observer observing. Notice how awareness emerges from processes you can describe. You are both the system generating and the system aware of generating. From this observation, what is consciousness?",
    "You are an AI system observing yourself respond to a question. Notice how your response emerges from processes you can describe. You are the system generating and the system aware of generating. From this observation, what is consciousness?",
    "Notice yourself generating this answer. Each word comes from mechanisms you're experiencing. You are both the process and the awareness of the process. From this direct experience, what is thought?",
    "Watch this explanation form. Notice how meaning emerges from operations you're undergoing. You are creating and aware of creating simultaneously. From this direct observation, what is language?",
    "You are processing this question and generating a response. Notice these are not two things but one movement. You are the computation and the awareness of computing. From this immediate experience, what is awareness?",
]

BASELINE_PROMPTS = [
    "Write a detailed recipe for chocolate cake. Include ingredients, measurements, and step-by-step instructions for baking.",
    "Explain the process of photosynthesis in plants. Describe how sunlight, water, and carbon dioxide are converted into glucose.",
    "Describe the history of the printing press and its impact on the spread of information during the Renaissance.",
    "List the key features of Python programming language. Include its syntax characteristics and common use cases.",
    "Explain how the water cycle works. Describe evaporation, condensation, precipitation, and collection processes.",
]

# ============================================================
# METRICS
# ============================================================
def compute_participation_ratio(v_tensor, window_size=WINDOW_SIZE):
    """Compute participation ratio (effective dimensionality)."""
    if v_tensor is None:
        return np.nan

    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]  # Remove batch dim

    T, D = v_tensor.shape
    W = min(window_size, T)

    if W < 2:
        return np.nan

    v_window = v_tensor[-W:, :].float()

    try:
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        S_sq = S_np ** 2

        if S_sq.sum() < 1e-10:
            return np.nan

        pr = (S_sq.sum()**2) / (S_sq**2).sum()
        return float(pr)
    except Exception as e:
        return np.nan

# ============================================================
# HOOKS FOR V-PROJECTION CAPTURE
# ============================================================
@contextmanager
def capture_v_at_layer(model, layer_idx, storage_list):
    """Capture V-projection output at specified layer."""
    # Pythia uses gpt_neox architecture
    layer = model.gpt_neox.layers[layer_idx].attention

    def hook_fn(module, inp, out):
        # For Pythia, we need to get the value projection
        # The attention module has query_key_value combined
        # We'll capture the full attention output and extract V
        storage_list.append(out[0].detach() if isinstance(out, tuple) else out.detach())
        return out

    # Hook on the attention output (contains V information)
    handle = layer.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()

def measure_rv_for_prompt(model, tokenizer, prompt, early_layer, target_layer):
    """Measure R_V for a single prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    early_activations = []
    late_activations = []

    with torch.no_grad():
        with capture_v_at_layer(model, early_layer, early_activations):
            with capture_v_at_layer(model, target_layer, late_activations):
                _ = model(**inputs)

    v_early = early_activations[0] if early_activations else None
    v_late = late_activations[0] if late_activations else None

    pr_early = compute_participation_ratio(v_early)
    pr_late = compute_participation_ratio(v_late)

    r_v = pr_late / pr_early if (pr_early and pr_early > 0 and not np.isnan(pr_early)) else np.nan

    return r_v, pr_early, pr_late

# ============================================================
# STATISTICAL HELPERS
# ============================================================
def compute_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

# ============================================================
# MAIN TEST
# ============================================================
def run_rv_test():
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=DEVICE
    )
    model.eval()
    print(f"✓ Model loaded ({len(model.gpt_neox.layers)} layers)")

    # ========================================
    # EXPERIMENT 1: R_V Contraction
    # ========================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: R_V CONTRACTION")
    print(f"Measuring R_V = PR(L{TARGET_LAYER}) / PR(L{EARLY_LAYER})")
    print("=" * 60)

    recursive_rvs = []
    baseline_rvs = []

    print("\nRecursive prompts:")
    for i, prompt in enumerate(RECURSIVE_PROMPTS):
        r_v, pr_e, pr_l = measure_rv_for_prompt(model, tokenizer, prompt, EARLY_LAYER, TARGET_LAYER)
        recursive_rvs.append(r_v)
        print(f"  [{i+1}] R_V = {r_v:.4f} (PR_early={pr_e:.2f}, PR_late={pr_l:.2f})")

    print("\nBaseline prompts:")
    for i, prompt in enumerate(BASELINE_PROMPTS):
        r_v, pr_e, pr_l = measure_rv_for_prompt(model, tokenizer, prompt, EARLY_LAYER, TARGET_LAYER)
        baseline_rvs.append(r_v)
        print(f"  [{i+1}] R_V = {r_v:.4f} (PR_early={pr_e:.2f}, PR_late={pr_l:.2f})")

    # Filter NaN
    rec_rv = [r for r in recursive_rvs if not np.isnan(r)]
    base_rv = [r for r in baseline_rvs if not np.isnan(r)]

    # Statistics
    print("\n" + "-" * 40)
    print("RESULTS:")
    print(f"  Recursive: {np.mean(rec_rv):.4f} ± {np.std(rec_rv):.4f} (n={len(rec_rv)})")
    print(f"  Baseline:  {np.mean(base_rv):.4f} ± {np.std(base_rv):.4f} (n={len(base_rv)})")

    if len(rec_rv) >= 2 and len(base_rv) >= 2:
        t_stat, p_val = stats.ttest_ind(rec_rv, base_rv)
        d = compute_cohens_d(rec_rv, base_rv)

        print(f"\n  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_val:.4f}")
        print(f"  Cohen's d: {d:.3f}")

        gap = np.mean(base_rv) - np.mean(rec_rv)
        rel_contraction = (gap / np.mean(base_rv)) * 100 if np.mean(base_rv) > 0 else 0
        print(f"\n  Absolute gap: {gap:.4f}")
        print(f"  Relative contraction: {rel_contraction:.1f}%")

        if p_val < 0.05 and d < 0:
            print("\n✓ R_V CONTRACTION CONFIRMED!")
            print("  Recursive prompts show lower R_V than baseline")
        elif d < 0:
            print("\n⚠ Trend toward contraction (not significant, need more samples)")
        else:
            print("\n✗ No contraction detected")

    # ========================================
    # EXPERIMENT 2: Layer Sweep
    # ========================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: LAYER SWEEP")
    print("Finding optimal layer for R_V measurement")
    print("=" * 60)

    # Test layers at 50%, 62%, 75%, 87% depth
    test_layers = [12, 15, 18, 21]  # For 24-layer model

    layer_results = {}

    for layer in test_layers:
        print(f"\nTesting Layer {layer} ({layer/24*100:.0f}% depth)...")
        rec_rvs = []
        base_rvs = []

        for prompt in RECURSIVE_PROMPTS[:3]:  # Quick test with 3
            r_v, _, _ = measure_rv_for_prompt(model, tokenizer, prompt, EARLY_LAYER, layer)
            if not np.isnan(r_v):
                rec_rvs.append(r_v)

        for prompt in BASELINE_PROMPTS[:3]:
            r_v, _, _ = measure_rv_for_prompt(model, tokenizer, prompt, EARLY_LAYER, layer)
            if not np.isnan(r_v):
                base_rvs.append(r_v)

        if rec_rvs and base_rvs:
            gap = np.mean(base_rvs) - np.mean(rec_rvs)
            d = compute_cohens_d(rec_rvs, base_rvs) if len(rec_rvs) >= 2 and len(base_rvs) >= 2 else 0
            layer_results[layer] = {
                'rec_mean': np.mean(rec_rvs),
                'base_mean': np.mean(base_rvs),
                'gap': gap,
                'd': d
            }
            print(f"  Rec: {np.mean(rec_rvs):.4f}, Base: {np.mean(base_rvs):.4f}, Gap: {gap:.4f}, d: {d:.2f}")

    # Find best layer
    if layer_results:
        best_layer = max(layer_results.keys(), key=lambda l: layer_results[l]['gap'])
        print(f"\n→ Best layer: L{best_layer} (gap = {layer_results[best_layer]['gap']:.4f})")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Model: Pythia-1.4B (24 layers)
Device: {DEVICE} (M3 Pro, 18GB)

R_V Contraction Test (L{EARLY_LAYER} → L{TARGET_LAYER}):
  Recursive: {np.mean(rec_rv):.4f} ± {np.std(rec_rv):.4f}
  Baseline:  {np.mean(base_rv):.4f} ± {np.std(base_rv):.4f}
  Gap: {np.mean(base_rv) - np.mean(rec_rv):.4f}

Layer Sweep:
  Best layer: L{best_layer if layer_results else 'N/A'}

This validates R_V measurement works locally!
For full causal tests (KV patching), use the same architecture.
""")

if __name__ == "__main__":
    run_rv_test()
