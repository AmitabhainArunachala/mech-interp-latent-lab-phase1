"""
experiment_l8_ablation.py

Task: L8 Ablation Test
Purpose: Test if ablating L8 breaks coherence differently for recursive vs baseline prompts

For recursive and baseline prompts:
- Normal generation (no ablation)
- Generation with L8 ablated (zeroed out)

We measure:
- BehaviorState (baseline/questioning/naked_loop/recursive_prose/collapse)
- Actual generated text
- R_V (if possible)

Key Questions:
1. Does L8 ablation break recursive prompts differently than baseline?
2. If yes → L8 is important for recursion specifically
3. If no → L8 is just important for general coherence
"""

import os
import sys
from contextlib import contextmanager
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))

from src.core.models import load_model, set_seed
from src.metrics.rv import compute_rv
from src.metrics.behavior_states import label_behavior_state
from prompts.loader import PromptLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_RECURSIVE = None  # Use all available (105 total)
N_BASELINE = None   # Use all available (100 total)


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 60) -> str:
    """Generate text from a prompt."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    text = tokenizer.decode(gen[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    return text


@contextmanager
def ablate_layer(model, layer_idx: int):
    """
    Ablate (zero out) a layer's attention output.
    
    This zeros the attention contribution to the residual stream.
    Note: The residual connection still adds the input, so we're removing the update.
    This matches the pattern from phase2_layer_ablation_sweep.py
    """
    handle = None

    def ablate_hook(module, inputs, outputs):
        # Zero out attention output
        # outputs[0] is the attention output tensor
        zeros = torch.zeros_like(outputs[0])
        return (zeros,) + outputs[1:]

    handle = model.model.layers[layer_idx].self_attn.register_forward_hook(ablate_hook)
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


@contextmanager
def ablate_layer_residual(model, layer_idx: int):
    """
    Alternative: Ablate by zeroing the residual stream input to the layer.
    
    This is more aggressive - it removes ALL information from previous layers
    at this point, not just this layer's contribution.
    """
    handle = None

    def ablate_hook(module, args):
        # Zero out the input hidden states
        hidden_states = args[0]
        zeroed = torch.zeros_like(hidden_states)
        return (zeroed,) + args[1:]

    handle = model.model.layers[layer_idx].register_forward_pre_hook(ablate_hook)
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


def run_l8_ablation_test():
    """Run L8 ablation test on recursive and baseline prompts."""
    print("=" * 70)
    print("L8 Gap-Filling Experiment 2: L8 Ablation Test")
    print("=" * 70)
    print(f"Testing L8 ablation on {N_RECURSIVE} recursive + {N_BASELINE} baseline prompts")
    print()

    set_seed(42)

    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    loader = PromptLoader()

    # Get ALL available prompts from REUSABLE_PROMPT_BANK
    # Recursive prompts from all dose-response groups
    recursive_prompts = []
    recursive_groups = ["L1_hint", "L2_simple", "L3_deeper", "L4_full", "L5_refined"]
    for group in recursive_groups:
        try:
            group_prompts = loader.get_by_group(group, limit=None, seed=123)
            recursive_prompts.extend(group_prompts)
            print(f"  Loaded {len(group_prompts)} prompts from {group}")
        except Exception as e:
            print(f"  Warning: Could not load {group}: {e}")
    
    # Baseline prompts from all baseline groups
    baseline_prompts = []
    baseline_groups = ["baseline_math", "baseline_factual", "baseline_creative", "baseline_impossible", "baseline_personal"]
    for group in baseline_groups:
        try:
            group_prompts = loader.get_by_group(group, limit=None, seed=456)
            baseline_prompts.extend(group_prompts)
            print(f"  Loaded {len(group_prompts)} prompts from {group}")
        except Exception as e:
            print(f"  Warning: Could not load {group}: {e}")
    
    # Shuffle for reproducibility
    import random
    random.Random(123).shuffle(recursive_prompts)
    random.Random(456).shuffle(baseline_prompts)

    print(f"\nLoaded {len(recursive_prompts)} recursive prompts")
    print(f"Loaded {len(baseline_prompts)} baseline prompts")
    print(f"Total: {len(recursive_prompts) + len(baseline_prompts)} prompts")
    print()

    rows: List[Dict] = []

    # Test recursive prompts
    for idx, prompt in enumerate(tqdm(recursive_prompts, desc="Recursive prompts")):
        # Normal generation
        try:
            text_normal = generate_text(model, tokenizer, prompt)
            label_normal = label_behavior_state(text_normal)
        except Exception as e:
            print(f"Warning: Failed normal generation for recursive {idx}: {e}")
            text_normal = ""
            label_normal = label_behavior_state("")

        try:
            rv_normal = compute_rv(model, tokenizer, prompt, device=DEVICE)
        except Exception as e:
            print(f"Warning: Failed R_V computation for recursive {idx}: {e}")
            rv_normal = float("nan")

        # Ablated generation (zero layer output)
        try:
            with ablate_layer(model, layer_idx=8):
                text_ablated = generate_text(model, tokenizer, prompt)
                label_ablated = label_behavior_state(text_ablated)
        except Exception as e:
            print(f"Warning: Failed ablated generation for recursive {idx}: {e}")
            text_ablated = ""
            label_ablated = label_behavior_state("")

        try:
            with ablate_layer(model, layer_idx=8):
                rv_ablated = compute_rv(model, tokenizer, prompt, device=DEVICE)
        except Exception as e:
            rv_ablated = float("nan")

        row: Dict = {
            "prompt_type": "recursive",
            "prompt_idx": idx,
            "prompt": prompt[:150] + ("..." if len(prompt) > 150 else ""),
            "normal_output": text_normal[:300],
            "ablated_output": text_ablated[:300],
            "normal_state": label_normal.state.value,
            "ablated_state": label_ablated.state.value,
            "normal_rv": rv_normal,
            "ablated_rv": rv_ablated,
            "rv_delta": rv_ablated - rv_normal,
            "normal_repetition": label_normal.repetition_ratio,
            "ablated_repetition": label_ablated.repetition_ratio,
            "normal_question_ratio": label_normal.question_mark_ratio,
            "ablated_question_ratio": label_ablated.question_mark_ratio,
            "state_changed": label_normal.state != label_ablated.state,
        }
        rows.append(row)

        print(
            f"\n[Recursive {idx}] "
            f"Normal: {label_normal.state.value} | "
            f"Ablated: {label_ablated.state.value} | "
            f"RV: {rv_normal:.3f} → {rv_ablated:.3f} (Δ={rv_ablated-rv_normal:+.3f})"
        )

    # Test baseline prompts
    for idx, prompt in enumerate(tqdm(baseline_prompts, desc="Baseline prompts")):
        # Normal generation
        try:
            text_normal = generate_text(model, tokenizer, prompt)
            label_normal = label_behavior_state(text_normal)
        except Exception as e:
            print(f"Warning: Failed normal generation for baseline {idx}: {e}")
            text_normal = ""
            label_normal = label_behavior_state("")

        try:
            rv_normal = compute_rv(model, tokenizer, prompt, device=DEVICE)
        except Exception as e:
            rv_normal = float("nan")

        # Ablated generation
        try:
            with ablate_layer(model, layer_idx=8):
                text_ablated = generate_text(model, tokenizer, prompt)
                label_ablated = label_behavior_state(text_ablated)
        except Exception as e:
            print(f"Warning: Failed ablated generation for baseline {idx}: {e}")
            text_ablated = ""
            label_ablated = label_behavior_state("")

        try:
            with ablate_layer(model, layer_idx=8):
                rv_ablated = compute_rv(model, tokenizer, prompt, device=DEVICE)
        except Exception as e:
            rv_ablated = float("nan")

        row: Dict = {
            "prompt_type": "baseline",
            "prompt_idx": idx,
            "prompt": prompt[:150] + ("..." if len(prompt) > 150 else ""),
            "normal_output": text_normal[:300],
            "ablated_output": text_ablated[:300],
            "normal_state": label_normal.state.value,
            "ablated_state": label_ablated.state.value,
            "normal_rv": rv_normal,
            "ablated_rv": rv_ablated,
            "rv_delta": rv_ablated - rv_normal,
            "normal_repetition": label_normal.repetition_ratio,
            "ablated_repetition": label_ablated.repetition_ratio,
            "normal_question_ratio": label_normal.question_mark_ratio,
            "ablated_question_ratio": label_ablated.question_mark_ratio,
            "state_changed": label_normal.state != label_ablated.state,
        }
        rows.append(row)

        print(
            f"\n[Baseline {idx}] "
            f"Normal: {label_normal.state.value} | "
            f"Ablated: {label_ablated.state.value} | "
            f"RV: {rv_normal:.3f} → {rv_ablated:.3f} (Δ={rv_ablated-rv_normal:+.3f})"
        )

    # Save results
    os.makedirs("results/dec11_evening", exist_ok=True)
    out_csv = "results/dec11_evening/l8_ablation_test.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    rec_data = df[df["prompt_type"] == "recursive"]
    base_data = df[df["prompt_type"] == "baseline"]

    print(f"\nRecursive Prompts ({len(rec_data)}):")
    print(f"  State changes: {rec_data['state_changed'].sum()}/{len(rec_data)}")
    print(f"  Normal states: {rec_data['normal_state'].value_counts().to_dict()}")
    print(f"  Ablated states: {rec_data['ablated_state'].value_counts().to_dict()}")
    print(f"  Mean RV delta: {rec_data['rv_delta'].mean():+.3f}")

    print(f"\nBaseline Prompts ({len(base_data)}):")
    print(f"  State changes: {base_data['state_changed'].sum()}/{len(base_data)}")
    print(f"  Normal states: {base_data['normal_state'].value_counts().to_dict()}")
    print(f"  Ablated states: {base_data['ablated_state'].value_counts().to_dict()}")
    print(f"  Mean RV delta: {base_data['rv_delta'].mean():+.3f}")

    # Compare
    print(f"\nComparison:")
    print(f"  State change rate - Recursive: {rec_data['state_changed'].mean():.1%}")
    print(f"  State change rate - Baseline: {base_data['state_changed'].mean():.1%}")
    print(f"  RV delta difference: {rec_data['rv_delta'].mean() - base_data['rv_delta'].mean():+.3f}")

    print(f"\nResults saved to: {out_csv}")
    print("=" * 70)


if __name__ == "__main__":
    run_l8_ablation_test()
