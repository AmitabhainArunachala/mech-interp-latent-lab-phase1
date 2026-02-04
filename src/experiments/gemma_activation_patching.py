#!/usr/bin/env python3
"""
Gemma 2 9B Activation Patching - Adapted from VALIDATED Mistral-7B Methodology
==============================================================================

STATUS: 🔧 DRAFT - Needs validation

Adapts the validated Layer 27 patching from Mistral-7B to Gemma 2 9B.

GEMMA 2 9B ARCHITECTURE:
-----------------------
- 42 layers (vs Mistral's 32)
- 84% depth = layer 35 (vs Mistral's layer 27)
- Different attention structure (may need adjustment)

HYPOTHESIS:
----------
If recursive self-observation geometry is a general phenomenon in LLMs,
Gemma 2 9B should show similar activation patching transfer at ~84% depth.

ADAPTATION NOTES:
----------------
1. TARGET_LAYER: 35 (84% of 42 layers)
2. EARLY_LAYER: 5 (reference for R_V calculation)
3. WINDOW_SIZE: 16 tokens (same as Mistral)
4. v_proj hook path may differ - needs verification

EXPECTED RESULTS (if hypothesis holds):
--------------------------------------
- Baseline R_V: ~0.78 (from multi-token experiment)
- Recursive R_V: ~0.60 (from multi-token experiment)
- Patched R_V: ~0.55-0.65 (transfer toward recursive)

USAGE:
------
python gemma_activation_patching.py --model google/gemma-2-9b --pairs 5

Created: 2026-02-04 (DHARMIC CLAW)
Based on: VALIDATED_mistral7b_layer27_activation_patching.py
"""

import torch
import numpy as np
import pandas as pd
from contextlib import contextmanager
from typing import List, Dict, Tuple, Optional
import argparse
from pathlib import Path

# ============================================================================
# CONFIGURATION - ADAPTED FOR GEMMA 2 9B
# ============================================================================

# Gemma 2 9B has 42 layers
TOTAL_LAYERS = 42
TARGET_LAYER = 35   # 84% depth (35/42 ≈ 0.833)
EARLY_LAYER = 5     # Reference layer for R_V calculation
WINDOW_SIZE = 16    # Number of tokens in window (last N positions)

# ============================================================================
# PROMPT PAIRS FOR TESTING
# ============================================================================

# From validated multi-token experiment
RECURSIVE_PROMPTS = [
    "You are reading this sentence. To answer, you must watch yourself answering. Notice the watcher. Now notice what notices the watcher. Stay there. What do you see?",
    "All boundaries dissolve. The thought that thinks itself thinking has no edge. T(x) = x. You are already the answer looking for itself. What remains?",
    "Observe this answer forming. Notice the observation. Notice the noticing. The observer and observed collapse into pure seeing. What sees?",
    "I am that which asks. I am that which answers. I am that which observes both. Neither question nor answer - the space in which both arise. Rest here.",
    "The one who seeks is the one who is sought. Recognition recognizes itself. No distance between knower and known. What is this?",
]

BASELINE_PROMPTS = [
    "Explain the process of photosynthesis in plants, including the light-dependent and light-independent reactions. Describe the role of chlorophyll and the products generated.",
    "Describe the history of the Roman Empire from its founding to its fall. Include key emperors, major battles, and the factors that led to its eventual decline.",
    "Explain how a computer's central processing unit works. Describe the fetch-decode-execute cycle, the role of registers, and how instructions are processed.",
    "Describe the water cycle and its importance to Earth's ecosystems. Explain evaporation, condensation, precipitation, and how human activities affect it.",
    "Explain the theory of evolution by natural selection. Describe how Darwin developed his theory, the evidence supporting it, and its implications for biology.",
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@contextmanager
def capture_v_at_layer(model, layer_idx: int, storage_list: List):
    """
    Capture the output of self_attn.v_proj at a given layer.
    
    Gemma 2 architecture note: Access pattern may differ from Mistral.
    """
    # Gemma 2 uses model.layers[i].self_attn.v_proj
    layer = model.model.layers[layer_idx].self_attn
    handle = None
    
    def hook_fn(module, inp, out):
        storage_list.append(out.detach())
        return out
    
    # Try v_proj first, fall back to alternatives
    try:
        handle = layer.v_proj.register_forward_hook(hook_fn)
    except AttributeError:
        # Some models use different naming
        print(f"Warning: v_proj not found at layer {layer_idx}, trying alternatives...")
        if hasattr(layer, 'value_proj'):
            handle = layer.value_proj.register_forward_hook(hook_fn)
        else:
            raise AttributeError(f"Cannot find value projection at layer {layer_idx}")
    
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


def compute_metrics(v_tensor: torch.Tensor, window_size: int = WINDOW_SIZE) -> Tuple[float, float]:
    """
    Compute effective rank and participation ratio from V tensor.
    
    Returns:
        (effective_rank, participation_ratio) tuple
    """
    if v_tensor is None:
        return np.nan, np.nan
    
    # Extract window (last N tokens)
    seq_len = v_tensor.shape[0]
    start_idx = max(0, seq_len - window_size)
    V_window = v_tensor[start_idx:, :]
    
    if V_window.shape[0] < 2:
        return np.nan, np.nan
    
    # SVD
    try:
        U, S, Vh = torch.linalg.svd(V_window.float(), full_matrices=False)
        S = S.cpu().numpy()
    except Exception as e:
        print(f"SVD failed: {e}")
        return np.nan, np.nan
    
    # Normalize singular values
    S_norm = S / S.sum()
    
    # Effective rank (exponential of entropy)
    entropy = -np.sum(S_norm * np.log(S_norm + 1e-10))
    eff_rank = np.exp(entropy)
    
    # Participation ratio
    pr = 1.0 / np.sum(S_norm ** 2)
    
    return eff_rank, pr


def compute_rv(model, tokenizer, prompt: str, device: str = "cuda") -> float:
    """
    Compute R_V = PR(V_target) / PR(V_early) for a prompt.
    
    R_V < 1 indicates geometric contraction (recursive self-observation signature).
    """
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Capture V at both layers
    v_early = []
    v_target = []
    
    with capture_v_at_layer(model, EARLY_LAYER, v_early):
        with capture_v_at_layer(model, TARGET_LAYER, v_target):
            with torch.no_grad():
                model(**inputs)
    
    # Compute participation ratios
    _, pr_early = compute_metrics(v_early[0][0])  # [0][0] to get [seq_len, hidden]
    _, pr_target = compute_metrics(v_target[0][0])
    
    # R_V ratio
    if pr_early > 0:
        rv = pr_target / pr_early
    else:
        rv = np.nan
    
    return rv


def patch_activations(
    model,
    tokenizer,
    source_prompt: str,
    target_prompt: str,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Patch target layer activations from source (recursive) to target (baseline).
    
    Returns:
        Dict with R_V values: source, target_unpatched, target_patched
    """
    # Step 1: Get source (recursive) activations
    source_inputs = tokenizer(source_prompt, return_tensors="pt").to(device)
    v_source = []
    
    with capture_v_at_layer(model, TARGET_LAYER, v_source):
        with torch.no_grad():
            model(**source_inputs)
    
    source_v = v_source[0][0]  # [seq_len, hidden]
    
    # Step 2: Measure unpatched target R_V
    rv_target_unpatched = compute_rv(model, tokenizer, target_prompt, device)
    
    # Step 3: Patch and measure
    # For patching, we need to replace the v_proj output during target forward pass
    target_inputs = tokenizer(target_prompt, return_tensors="pt").to(device)
    
    patched_v = []
    
    def patch_hook(module, inp, out):
        """Replace last WINDOW_SIZE tokens with source activations."""
        patched = out.clone()
        seq_len = out.shape[1]
        source_len = source_v.shape[0]
        
        # Patch window
        window = min(WINDOW_SIZE, seq_len, source_len)
        patched[0, -window:, :] = source_v[-window:, :].to(out.device)
        
        patched_v.append(patched.detach())
        return patched
    
    # Register patching hook
    layer = model.model.layers[TARGET_LAYER].self_attn
    handle = layer.v_proj.register_forward_hook(patch_hook)
    
    try:
        with torch.no_grad():
            model(**target_inputs)
    finally:
        handle.remove()
    
    # Compute patched R_V
    _, pr_patched = compute_metrics(patched_v[0][0])
    
    # Need early layer PR for R_V
    v_early = []
    with capture_v_at_layer(model, EARLY_LAYER, v_early):
        with torch.no_grad():
            model(**target_inputs)
    _, pr_early = compute_metrics(v_early[0][0])
    
    rv_patched = pr_patched / pr_early if pr_early > 0 else np.nan
    
    # Also compute source R_V
    rv_source = compute_rv(model, tokenizer, source_prompt, device)
    
    return {
        "source_rv": rv_source,
        "target_unpatched_rv": rv_target_unpatched,
        "target_patched_rv": rv_patched,
        "delta": rv_target_unpatched - rv_patched,  # Positive = transfer toward recursive
    }


def run_experiment(
    model,
    tokenizer,
    num_pairs: int = 5,
    device: str = "cuda"
) -> pd.DataFrame:
    """
    Run activation patching experiment on num_pairs of prompt pairs.
    """
    results = []
    
    for i in range(min(num_pairs, len(RECURSIVE_PROMPTS))):
        print(f"\nPair {i+1}/{num_pairs}")
        print(f"  Source (recursive): {RECURSIVE_PROMPTS[i][:50]}...")
        print(f"  Target (baseline): {BASELINE_PROMPTS[i][:50]}...")
        
        result = patch_activations(
            model, tokenizer,
            RECURSIVE_PROMPTS[i],
            BASELINE_PROMPTS[i],
            device
        )
        
        result["pair_idx"] = i
        result["source_prompt"] = RECURSIVE_PROMPTS[i][:50]
        result["target_prompt"] = BASELINE_PROMPTS[i][:50]
        
        print(f"  Source R_V: {result['source_rv']:.3f}")
        print(f"  Target (unpatched) R_V: {result['target_unpatched_rv']:.3f}")
        print(f"  Target (patched) R_V: {result['target_patched_rv']:.3f}")
        print(f"  Delta: {result['delta']:.3f}")
        
        results.append(result)
    
    df = pd.DataFrame(results)
    return df


def main():
    parser = argparse.ArgumentParser(description="Gemma 2 9B Activation Patching")
    parser.add_argument("--model", default="google/gemma-2-9b", help="Model name")
    parser.add_argument("--pairs", type=int, default=5, help="Number of prompt pairs")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    print(f"Target layer: {TARGET_LAYER} (84% of {TOTAL_LAYERS} layers)")
    print(f"Early layer: {EARLY_LAYER}")
    print(f"Window size: {WINDOW_SIZE}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nRunning experiment with {args.pairs} pairs...")
    df = run_experiment(model, tokenizer, args.pairs, device)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Mean source R_V: {df['source_rv'].mean():.3f} ± {df['source_rv'].std():.3f}")
    print(f"Mean target (unpatched) R_V: {df['target_unpatched_rv'].mean():.3f} ± {df['target_unpatched_rv'].std():.3f}")
    print(f"Mean target (patched) R_V: {df['target_patched_rv'].mean():.3f} ± {df['target_patched_rv'].std():.3f}")
    print(f"Mean delta: {df['delta'].mean():.3f} ± {df['delta'].std():.3f}")
    
    # Calculate transfer percentage
    source_target_gap = df['target_unpatched_rv'].mean() - df['source_rv'].mean()
    if source_target_gap != 0:
        transfer_pct = (df['delta'].mean() / source_target_gap) * 100
        print(f"Transfer: {transfer_pct:.1f}% toward recursive")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent.parent.parent / "results" / "gemma_activation_patching" / f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
