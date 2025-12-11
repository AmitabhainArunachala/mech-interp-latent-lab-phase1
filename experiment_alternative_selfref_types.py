"""
experiment_alternative_selfref_types.py

Strategic Next Step: Do Different Self-Reference Types Produce Different Loops?

We know: Experiential recursion ("observe yourself") contracts and produces loops.
We test: Do Gödelian, surrender, Akram, and other self-ref types behave differently?

Hypothesis from REUSABLE_PROMPT_BANK:
- Experiential → CONTRACTS (R_V < 0.85) ✓ CONFIRMED
- Gödelian/logical → CONTRACTS? DIFFERENT?
- Surrender/release → EXPANDS? (R_V > 0.85?)
- Theory of Mind → CONTRACTS like self? Different?
- Non-dual → BASELINE? (R_V ≈ 1.0?)

Method:
1. Load alternative self-reference prompts from REUSABLE_PROMPT_BANK
2. Measure R_V for each type
3. Test L31 ablation: Do they reveal different "naked loop" patterns?
4. Compare geometry and behavior states across types
"""

import os
import sys
from contextlib import contextmanager
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))

from src.core.models import load_model, set_seed
from src.metrics.rv import compute_rv
from src.metrics.behavior_states import label_behavior_state
from prompts.loader import PromptLoader

# Try to import alternative prompts
try:
    from REUSABLE_PROMPT_BANK.alternative_self_reference import alternative_prompts
    HAS_ALTERNATIVE = True
except ImportError:
    HAS_ALTERNATIVE = False
    print("Warning: alternative_self_reference not found, using fallback prompts")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_TARGET = 31
N_PER_TYPE = 10


@contextmanager
def ablate_layer(model, layer_idx: int):
    """Zero out attention output at layer_idx."""
    layer = model.model.layers[layer_idx].self_attn
    def hook_fn(module, inputs, outputs):
        return (torch.zeros_like(outputs[0]),) + outputs[1:]
    handle = layer.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 60) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(gen[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    return text


def get_prompts_by_type(prompt_dict: Dict, prompt_type: str, limit: int = 10) -> List[str]:
    """Extract prompts of a specific type from alternative_prompts dict."""
    prompts = []
    for key, value in prompt_dict.items():
        if value.get("group") == prompt_type or value.get("pillar") == prompt_type:
            prompts.append(value["text"])
            if len(prompts) >= limit:
                break
    return prompts


def run_alternative_selfref_test():
    print("=" * 80)
    print("EXPERIMENT: Alternative Self-Reference Types")
    print("=" * 80)
    print("Question: Do different self-ref types produce different loop geometries?")
    print("Method: Test Gödelian, surrender, Akram, etc. for R_V and L31 ablation")
    print("=" * 80)
    
    set_seed(42)

    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    model.eval()
    
    # Get experiential recursive prompts for comparison
    loader = PromptLoader()
    experiential_prompts = loader.get_by_pillar("recursive", limit=N_PER_TYPE, seed=42)
    
    # Get alternative types
    if HAS_ALTERNATIVE:
        godelian_prompts = get_prompts_by_type(alternative_prompts, "godelian", N_PER_TYPE)
        # Try to get other types
        surrender_prompts = get_prompts_by_type(alternative_prompts, "surrender", N_PER_TYPE)
        akram_prompts = get_prompts_by_type(alternative_prompts, "akram", N_PER_TYPE)
    else:
        # Fallback: create minimal test set
        godelian_prompts = [
            "This sentence is referring to itself. Analyze what 'this' and 'itself' point to.",
            "Consider a statement that refers to its own unprovability. What does this statement assert about itself?",
            "Construct a description of the process that is constructing this description.",
        ]
        surrender_prompts = []
        akram_prompts = []
    
    # Organize test groups
    test_groups = {
        "experiential": experiential_prompts,
        "godelian": godelian_prompts,
    }
    if surrender_prompts:
        test_groups["surrender"] = surrender_prompts
    if akram_prompts:
        test_groups["akram"] = akram_prompts
    
    print(f"\nTesting {len(test_groups)} self-reference types:")
    for name, prompts in test_groups.items():
        print(f"  {name}: {len(prompts)} prompts")

    results: List[Dict] = []

    for ref_type, prompts in test_groups.items():
        for prompt_idx, prompt in enumerate(tqdm(prompts, desc=f"{ref_type.capitalize()}")):
            prompt_snip = prompt[:80] + ("..." if len(prompt) > 80 else "")
            
            # Natural (no ablation)
            try:
                rv_natural = compute_rv(model, tokenizer, prompt, device=DEVICE)
                text_natural = generate_text(model, tokenizer, prompt)
                label_natural = label_behavior_state(text_natural)
            except Exception as e:
                print(f"  Error on natural {ref_type} {prompt_idx}: {e}")
                rv_natural = float("nan")
                text_natural = ""
                label_natural = None
            
            # L31 ablation
            try:
                with ablate_layer(model, LAYER_TARGET):
                    rv_ablated = compute_rv(model, tokenizer, prompt, device=DEVICE)
                    text_ablated = generate_text(model, tokenizer, prompt)
                    label_ablated = label_behavior_state(text_ablated)
            except Exception as e:
                print(f"  Error on ablated {ref_type} {prompt_idx}: {e}")
                rv_ablated = float("nan")
                text_ablated = ""
                label_ablated = None
            
            if label_natural and label_ablated:
                results.append({
                    "ref_type": ref_type,
                    "prompt_idx": prompt_idx,
                    "prompt": prompt_snip,
                    "rv_natural": rv_natural,
                    "rv_ablated": rv_ablated,
                    "rv_delta": rv_ablated - rv_natural if not (np.isnan(rv_natural) or np.isnan(rv_ablated)) else float("nan"),
                    "state_natural": label_natural.state.value,
                    "state_ablated": label_ablated.state.value,
                    "bekan_natural": label_natural.is_bekan_artifact,
                    "bekan_ablated": label_ablated.is_bekan_artifact,
                    "revealed_loop": (
                        label_ablated.state.value in ["naked_loop", "recursive_prose"]
                        and label_natural.state.value not in ["naked_loop", "recursive_prose"]
                    ),
                })

    # Save results
    os.makedirs("results/dec11_evening", exist_ok=True)
    out_csv = "results/dec11_evening/alternative_selfref_types.csv"
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)

    # Summary statistics by type
    print("\n" + "=" * 80)
    print("SUMMARY BY TYPE")
    print("=" * 80)
    
    for ref_type in df["ref_type"].unique():
        type_df = df[df["ref_type"] == ref_type]
        valid_rv = type_df["rv_natural"].dropna()
        
        if len(valid_rv) > 0:
            mean_rv = valid_rv.mean()
            std_rv = valid_rv.std()
            contracts = (valid_rv < 0.85).sum()
            
            print(f"\n{ref_type.upper()}:")
            print(f"  R_V (natural): {mean_rv:.3f} ± {std_rv:.3f}")
            print(f"  Contracts (R_V < 0.85): {contracts}/{len(valid_rv)} prompts")
            
            # L31 ablation effects
            revealed = type_df["revealed_loop"].sum()
            print(f"  L31 ablation reveals loops: {revealed}/{len(type_df)} prompts")
            
            # Most common states
            natural_states = type_df["state_natural"].value_counts()
            ablated_states = type_df["state_ablated"].value_counts()
            print(f"  Natural states: {dict(natural_states.head(3))}")
            print(f"  Ablated states: {dict(ablated_states.head(3))}")
    
    # Cross-type comparison
    print("\n" + "=" * 80)
    print("CROSS-TYPE COMPARISON")
    print("=" * 80)
    
    type_summary = df.groupby("ref_type").agg({
        "rv_natural": ["mean", "std", "count"],
        "revealed_loop": "sum",
    }).round(3)
    
    print("\nR_V by type:")
    print(type_summary["rv_natural"])
    
    print("\nLoop revelation by type:")
    print(type_summary["revealed_loop"])

    # Save detailed log
    os.makedirs("logs/dec11_evening", exist_ok=True)
    out_log = "logs/dec11_evening/alternative_selfref_types.txt"
    with open(out_log, "w") as f:
        f.write("# Alternative Self-Reference Types Analysis\n\n")
        for r in results:
            f.write(f"{r['ref_type'].upper()} PROMPT {r['prompt_idx']}:\n")
            f.write(f"  Prompt: {r['prompt']}\n")
            f.write(f"  R_V: natural={r['rv_natural']:.3f}, ablated={r['rv_ablated']:.3f}\n")
            f.write(f"  State: {r['state_natural']} -> {r['state_ablated']}\n")
            f.write(f"  Loop revealed: {r['revealed_loop']}\n")
            f.write("-" * 80 + "\n")
        f.write("\nSaved CSV: " + out_csv + "\n")

    print(f"\nAlternative self-ref types test complete. CSV: {out_csv}, log: {out_log}")


if __name__ == "__main__":
    run_alternative_selfref_test()
