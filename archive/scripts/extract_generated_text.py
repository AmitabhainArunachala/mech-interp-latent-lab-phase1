#!/usr/bin/env python3
"""
Extract actual generated text for key pairs.

If text is not in CSV, regenerate with text capture enabled.
"""

import json
import pandas as pd
import torch
from pathlib import Path
from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.core.patching import PersistentVPatcher, extract_v_activation
from src.metrics.rv import compute_rv
from src.metrics.behavior_strict import score_behavior_strict
from transformers import DynamicCache

SEED = 42
N_PAIRS = 20
WINDOW = 16
EARLY_LAYER = 5
LATE_LAYER = 27
TARGET_LAYER_V = 27
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _generate_with_kv(model, tokenizer, prompt_ids, past_key_values, max_new_tokens, temperature):
    """Generate text using provided KV cache."""
    current_ids = prompt_ids[:, -1:]
    current_kv = past_key_values
    
    generated_tokens = []
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(current_ids, past_key_values=current_kv, use_cache=True)
            logits = out.logits[:, -1, :]
            
            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            else:
                probs_temp = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs_temp, num_samples=1)
                
            generated_tokens.append(next_token.item())
            current_ids = next_token
            current_kv = out.past_key_values
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return text

def reproduce_pairs():
    """Reproduce exact pairs used in pipeline."""
    set_seed(SEED)
    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    
    loader = PromptLoader()
    raw_pairs = loader.get_balanced_pairs(n_pairs=N_PAIRS*5, seed=SEED)
    
    pairs = []
    filtered_pairs = []
    
    for rec_text, base_text in raw_pairs:
        r_ids = tokenizer.encode(rec_text, add_special_tokens=False)
        b_ids = tokenizer.encode(base_text, add_special_tokens=False)
        common_len = min(len(r_ids), len(b_ids))
        if common_len < WINDOW:
            continue
        
        try:
            rv_rec = compute_rv(model, tokenizer, rec_text, early=EARLY_LAYER, late=LATE_LAYER, window=WINDOW, device=DEVICE)
            if rv_rec < 0.9:
                filtered_pairs.append((rec_text, base_text, r_ids[:common_len], b_ids[:common_len], rv_rec))
        except Exception:
            continue
        
        if len(filtered_pairs) >= N_PAIRS * 2:
            break
    
    for rec_text, base_text, r_ids, b_ids, rv_rec in filtered_pairs[:N_PAIRS]:
        rec_ids = torch.tensor([r_ids], device=DEVICE)
        base_ids = torch.tensor([b_ids], device=DEVICE)
        pairs.append((rec_text, base_text, rec_ids, base_ids, rv_rec))
    
    return model, tokenizer, pairs

def generate_text_for_pair(model, tokenizer, pair_idx, rec_text, base_text, rec_ids, base_ids):
    """Generate text for all conditions for a single pair."""
    results = []
    
    # Extract KV and V_PROJ
    with torch.no_grad():
        out_rec = model(rec_ids, use_cache=True)
        rec_kv = out_rec.past_key_values
        
        out_base = model(base_ids, use_cache=True)
        base_kv = out_base.past_key_values
    
    rec_v_l27 = extract_v_activation(model, tokenizer, rec_text, layer_idx=TARGET_LAYER_V, device=DEVICE)
    
    # 1. Baseline Control
    baseline_text = _generate_with_kv(model, tokenizer, base_ids, base_kv, MAX_NEW_TOKENS, TEMPERATURE)
    baseline_score = score_behavior_strict(baseline_text, None)
    results.append({
        "pair_idx": pair_idx,
        "condition": "Baseline_Control",
        "prompt": base_text,
        "generated_text": baseline_text,
        "final_score": baseline_score.final_score,
        "passed_gates": baseline_score.passed_gates,
        "failure_reason": baseline_score.failure_reason,
    })
    
    # 2. Recursive Control
    v_patcher_rec = PersistentVPatcher(model, rec_v_l27)
    v_patcher_rec.register(layer_idx=TARGET_LAYER_V)
    try:
        recursive_text = _generate_with_kv(model, tokenizer, rec_ids, rec_kv, MAX_NEW_TOKENS, TEMPERATURE)
        recursive_score = score_behavior_strict(recursive_text, None)
    finally:
        v_patcher_rec.remove()
    results.append({
        "pair_idx": pair_idx,
        "condition": "Recursive_Control",
        "prompt": rec_text,
        "generated_text": recursive_text,
        "final_score": recursive_score.final_score,
        "passed_gates": recursive_score.passed_gates,
        "failure_reason": recursive_score.failure_reason,
    })
    
    # 3. Transfer (baseline prompt + recursive KV + V_PROJ patching)
    v_patcher_transfer = PersistentVPatcher(model, rec_v_l27)
    v_patcher_transfer.register(layer_idx=TARGET_LAYER_V)
    try:
        transfer_text = _generate_with_kv(model, tokenizer, base_ids, rec_kv, MAX_NEW_TOKENS, TEMPERATURE)
        transfer_score = score_behavior_strict(transfer_text, None)
    finally:
        v_patcher_transfer.remove()
    results.append({
        "pair_idx": pair_idx,
        "condition": "Transfer",
        "prompt": base_text,
        "generated_text": transfer_text,
        "final_score": transfer_score.final_score,
        "passed_gates": transfer_score.passed_gates,
        "failure_reason": transfer_score.failure_reason,
    })
    
    return results

def main():
    print("=" * 80)
    print("EXTRACTING GENERATED TEXT FOR KEY PAIRS")
    print("=" * 80)
    
    # Load existing results to identify key pairs
    df = pd.read_csv("results/runs/20251216_130512_behavior_strict/behavior_strict_results.csv")
    
    transfer = df[df["condition"] == "Transfer"]
    perfect_pairs = []
    for pair_idx in transfer["pair_idx"].unique():
        t_row = transfer[transfer["pair_idx"] == pair_idx].iloc[0]
        r_row = df[(df["pair_idx"] == pair_idx) & (df["condition"] == "Recursive_Control")].iloc[0]
        if abs(t_row["final_score"] - r_row["final_score"]) < 0.01 and t_row["final_score"] > 0.5:
            perfect_pairs.append(int(pair_idx))
    
    gate_failures = transfer[~transfer["passed_gates"]]["pair_idx"].unique()[:3].tolist()
    passed_zero = transfer[(transfer["passed_gates"] == True) & (transfer["final_score"] == 0.0)]["pair_idx"].unique().tolist()
    print(f"  Gate failures: {gate_failures}")
    print(f"  Passed gates, zero score: {passed_zero}")
    
    key_pairs = sorted(set(perfect_pairs + gate_failures + passed_zero))
    
    print(f"\nKey pairs to extract:")
    print(f"  Perfect matches: {perfect_pairs}")
    print(f"  Gate failures: {gate_failures}")
    print(f"  Passed gates, zero score: {passed_zero}")
    print(f"  Total: {key_pairs}")
    
    # Reproduce pairs
    print("\nReproducing pairs...")
    model, tokenizer, pairs = reproduce_pairs()
    
    # Generate text for key pairs
    all_results = []
    for pair_idx in key_pairs:
        if pair_idx < len(pairs):
            print(f"\nGenerating text for pair {pair_idx}...")
            rec_text, base_text, rec_ids, base_ids, rv_rec = pairs[pair_idx]
            results = generate_text_for_pair(model, tokenizer, pair_idx, rec_text, base_text, rec_ids, base_ids)
            all_results.extend(results)
    
    # Save CSV
    df_text = pd.DataFrame(all_results)
    df_text.to_csv("generated_text_comparison.csv", index=False)
    print(f"\n✅ Saved generated_text_comparison.csv ({len(df_text)} rows)")
    
    # Create markdown comparison
    with open("text_samples.md", "w") as f:
        f.write("# Generated Text Comparison\n\n")
        f.write("Side-by-side comparison of generated text for key pairs.\n\n")
        
        for pair_idx in key_pairs:
            pair_results = [r for r in all_results if r["pair_idx"] == pair_idx]
            if not pair_results:
                continue
            
            rec_prompt = next((r["prompt"] for r in pair_results if r["condition"] == "Recursive_Control"), "N/A")
            base_prompt = next((r["prompt"] for r in pair_results if r["condition"] == "Baseline_Control"), "N/A")
            
            f.write(f"## Pair {pair_idx}\n\n")
            f.write(f"**Recursive Prompt:** {rec_prompt[:200]}...\n\n")
            f.write(f"**Baseline Prompt:** {base_prompt[:200]}...\n\n")
            
            for condition in ["Baseline_Control", "Transfer", "Recursive_Control"]:
                result = next((r for r in pair_results if r["condition"] == condition), None)
                if result:
                    f.write(f"### {condition}\n\n")
                    f.write(f"**Score:** {result['final_score']:.4f}  ")
                    f.write(f"**Passed Gates:** {result['passed_gates']}  ")
                    if result['failure_reason']:
                        f.write(f"**Failure:** {result['failure_reason']}\n\n")
                    else:
                        f.write("\n\n")
                    f.write(f"{result['generated_text']}\n\n")
                    f.write("---\n\n")
    
    print("✅ Saved text_samples.md")
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

