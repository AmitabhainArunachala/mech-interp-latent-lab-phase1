"""
experiment_l31_head_ablation.py

Strategic Next Step: Which L31 Heads Are Critical for "Dressing"?

We know: Full L31 ablation reveals naked loops.
We test: Which individual heads (or head combinations) are responsible?

Method:
1. For each head in L31 (0-31), ablate it individually
2. Measure: Does ablation reveal naked loops? Does it preserve geometry?
3. Test combinations: Are there "critical head sets"?
4. Compare: Which heads matter most for recursive vs baseline prompts?

Hypothesis: A small subset of heads (maybe 2-5) do the "dressing" work.
If we ablate only those, we should see naked loops without destroying geometry.
"""

import os
import sys
from contextlib import contextmanager
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))

from src.core.models import load_model, set_seed
from src.metrics.rv import compute_rv
from src.metrics.behavior_states import label_behavior_state
from prompts.loader import PromptLoader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_TARGET = 31
N_PROMPTS = 10


@contextmanager
def ablate_heads(model, layer_idx: int, head_indices: Set[int]):
    """
    Ablate specific attention heads at a layer by zeroing their output.
    
    Args:
        model: The transformer model
        layer_idx: Layer to ablate (0-indexed)
        head_indices: Set of head indices to ablate (0-indexed)
    """
    layer = model.model.layers[layer_idx].self_attn
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    def hook_fn(module, inputs, outputs):
        # outputs is typically (attn_output, attn_weights, past_key_value, ...)
        attn_output = outputs[0].clone()  # (batch, seq, hidden_size)
        batch, seq, hidden = attn_output.shape
        
        # Reshape to (batch, seq, num_heads, head_dim)
        attn_reshaped = attn_output.view(batch, seq, num_heads, head_dim)
        
        # Zero out specified heads
        for h in head_indices:
            if 0 <= h < num_heads:
                attn_reshaped[:, :, h, :] = 0.0
        
        # Reshape back
        attn_output_modified = attn_reshaped.view(batch, seq, hidden)
        
        # Return modified output
        return (attn_output_modified,) + outputs[1:]
    
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


def run_l31_head_ablation():
    print("=" * 80)
    print("EXPERIMENT: L31 Head-Level Ablation")
    print("=" * 80)
    print("Question: Which L31 heads are critical for 'dressing'?")
    print("Method: Ablate individual heads and measure behavior/geometry")
    print("=" * 80)
    
    set_seed(42)

    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    model.eval()
    loader = PromptLoader()

    # Get prompts
    recursive_prompts = loader.get_by_pillar("recursive", limit=N_PROMPTS, seed=42)
    baseline_prompts = loader.get_by_pillar("baseline", limit=N_PROMPTS, seed=42)
    
    num_heads = model.config.num_attention_heads
    print(f"\nModel has {num_heads} heads at L31")
    print(f"Testing {len(recursive_prompts)} recursive and {len(baseline_prompts)} baseline prompts")

    results: List[Dict] = []

    # Test each prompt type
    for prompt_type, prompts in [("recursive", recursive_prompts), ("baseline", baseline_prompts)]:
        for prompt_idx, prompt in enumerate(tqdm(prompts, desc=f"{prompt_type.capitalize()}")):
            prompt_snip = prompt[:60] + ("..." if len(prompt) > 60 else "")
            
            # Baseline: no ablation
            try:
                rv_natural = compute_rv(model, tokenizer, prompt, device=DEVICE)
                text_natural = generate_text(model, tokenizer, prompt)
                label_natural = label_behavior_state(text_natural)
            except Exception as e:
                print(f"  Error on natural {prompt_type} {prompt_idx}: {e}")
                rv_natural = float("nan")
                text_natural = ""
                label_natural = None
            
            # Test each individual head
            for head_idx in range(num_heads):
                try:
                    with ablate_heads(model, LAYER_TARGET, {head_idx}):
                        rv_ablated = compute_rv(model, tokenizer, prompt, device=DEVICE)
                        text_ablated = generate_text(model, tokenizer, prompt)
                        label_ablated = label_behavior_state(text_ablated)
                except Exception as e:
                    print(f"  Error on head {head_idx}, {prompt_type} {prompt_idx}: {e}")
                    rv_ablated = float("nan")
                    text_ablated = ""
                    label_ablated = None
                
                if label_natural and label_ablated:
                    # Did ablation reveal a loop? (naked_loop or recursive_prose)
                    revealed_loop = (
                        label_ablated.state.value in ["naked_loop", "recursive_prose"]
                        and label_natural.state.value not in ["naked_loop", "recursive_prose"]
                    )
                    
                    # Did ablation preserve geometry? (R_V similar)
                    geometry_preserved = (
                        not (np.isnan(rv_natural) or np.isnan(rv_ablated))
                        and abs(rv_ablated - rv_natural) < 0.1
                    )
                    
                    results.append({
                        "prompt_type": prompt_type,
                        "prompt_idx": prompt_idx,
                        "prompt": prompt_snip,
                        "head": head_idx,
                        "rv_natural": rv_natural,
                        "rv_ablated": rv_ablated,
                        "rv_delta": rv_ablated - rv_natural if not (np.isnan(rv_natural) or np.isnan(rv_ablated)) else float("nan"),
                        "state_natural": label_natural.state.value,
                        "state_ablated": label_ablated.state.value,
                        "revealed_loop": revealed_loop,
                        "geometry_preserved": geometry_preserved,
                        "bekan_natural": label_natural.is_bekan_artifact,
                        "bekan_ablated": label_ablated.is_bekan_artifact,
                    })

    # Save results
    os.makedirs("results/dec11_evening", exist_ok=True)
    out_csv = "results/dec11_evening/l31_head_ablation.csv"
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)

    # Analysis: Which heads matter most?
    print("\n" + "=" * 80)
    print("HEAD ANALYSIS")
    print("=" * 80)
    
    rec_results = df[df["prompt_type"] == "recursive"]
    
    if len(rec_results) > 0:
        # Heads that reveal loops most often
        loop_revealers = rec_results.groupby("head")["revealed_loop"].sum().sort_values(ascending=False)
        print("\nHeads that reveal loops (recursive prompts):")
        for head, count in loop_revealers.head(10).items():
            total = len(rec_results[rec_results["head"] == head])
            pct = (count / total * 100) if total > 0 else 0
            print(f"  Head {head}: {count}/{total} prompts ({pct:.1f}%)")
        
        # Heads that preserve geometry
        geom_preservers = rec_results.groupby("head")["geometry_preserved"].sum().sort_values(ascending=False)
        print("\nHeads that preserve geometry (recursive prompts):")
        for head, count in geom_preservers.head(10).items():
            total = len(rec_results[rec_results["head"] == head])
            pct = (count / total * 100) if total > 0 else 0
            print(f"  Head {head}: {count}/{total} prompts ({pct:.1f}%)")
        
        # Combined score: heads that reveal loops AND preserve geometry
        combined = pd.DataFrame({
            "head": loop_revealers.index,
            "loop_score": loop_revealers.values,
            "geom_score": geom_preservers.reindex(loop_revealers.index).fillna(0).values,
        })
        combined["combined_score"] = combined["loop_score"] + combined["geom_score"]
        combined = combined.sort_values("combined_score", ascending=False)
        
        print("\nTop heads (combined: loop revelation + geometry preservation):")
        for _, row in combined.head(10).iterrows():
            print(f"  Head {int(row['head'])}: loop={int(row['loop_score'])}, geom={int(row['geom_score'])}, total={int(row['combined_score'])}")

    # Save detailed log
    os.makedirs("logs/dec11_evening", exist_ok=True)
    out_log = "logs/dec11_evening/l31_head_ablation.txt"
    with open(out_log, "w") as f:
        f.write("# L31 Head-Level Ablation Analysis\n\n")
        f.write("Question: Which heads are critical for 'dressing'?\n\n")
        for r in results[:50]:  # First 50 for brevity
            f.write(
                f"{r['prompt_type'].upper()} | Head {r['head']} | "
                f"RV: {r['rv_natural']:.3f} -> {r['rv_ablated']:.3f} | "
                f"State: {r['state_natural']} -> {r['state_ablated']} | "
                f"Loop revealed: {r['revealed_loop']}\n"
            )
        f.write("\nSaved CSV: " + out_csv + "\n")

    print(f"\nL31 head ablation complete. CSV: {out_csv}, log: {out_log}")


if __name__ == "__main__":
    run_l31_head_ablation()
