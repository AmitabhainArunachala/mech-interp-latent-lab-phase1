#!/usr/bin/env python3
"""
PHASE 1B: K vs V Separation in KV Cache

Test if Keys or Values matter more for behavior transfer.
"""

import json
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.core.patching import PersistentVPatcher, extract_v_activation
from src.metrics.rv import compute_rv
from src.metrics.behavior_strict import score_behavior_strict
from transformers import DynamicCache
from tqdm import tqdm

SEED = 42
N_PAIRS = 20
WINDOW = 16
EARLY_LAYER = 5
LATE_LAYER = 27
TARGET_LAYER_V = 27
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def split_kv(past_kv):
    """Split KV cache into separate K and V lists."""
    if hasattr(past_kv, "to_legacy_cache"):
        legacy = past_kv.to_legacy_cache()
    else:
        legacy = past_kv
    
    keys = [layer_kv[0] for layer_kv in legacy]
    values = [layer_kv[1] for layer_kv in legacy]
    return keys, values

def combine_kv(keys, values):
    """Combine K and V lists back into KV cache format."""
    return DynamicCache.from_legacy_cache(tuple((k, v) for k, v in zip(keys, values)))

def _generate_with_kv(model, tokenizer, prompt_ids, past_key_values, max_new_tokens, temperature):
    """Generate text using provided KV cache."""
    current_ids = prompt_ids[:, -1:]
    current_kv = past_key_values
    
    generated_tokens = []
    entropies = []
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(current_ids, past_key_values=current_kv, use_cache=True)
            logits = out.logits[:, -1, :]
            
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()
            entropies.append(entropy)
            
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
    mean_entropy = float(torch.tensor(entropies).mean()) if entropies else 0.0
    return text, mean_entropy

def run_kv_separation_experiment():
    """Run K vs V separation experiment."""
    set_seed(SEED)
    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    
    loader = PromptLoader()
    bank_version = loader.version
    
    # Reproduce pairs
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
    
    print(f"Loaded {len(pairs)} pairs.")
    
    results = []
    
    for i, (rec_text, base_text, rec_ids, base_ids, rv_rec) in enumerate(tqdm(pairs)):
        try:
            # Extract KV caches
            with torch.no_grad():
                out_rec = model(rec_ids, use_cache=True)
                rec_kv = out_rec.past_key_values
                
                out_base = model(base_ids, use_cache=True)
                base_kv = out_base.past_key_values
            
            # Split K and V
            rec_keys, rec_values = split_kv(rec_kv)
            base_keys, base_values = split_kv(base_kv)
            
            # Create hybrid KV caches
            k_only_kv = combine_kv(rec_keys, base_values)  # Recursive K, Baseline V
            v_only_kv = combine_kv(base_keys, rec_values)  # Baseline K, Recursive V
            
            # Extract V_PROJ
            rec_v_l27 = extract_v_activation(model, tokenizer, rec_text, layer_idx=TARGET_LAYER_V, device=DEVICE)
            
            # Conditions
            conditions = [
                ("Baseline_Control", base_ids, base_kv, False),
                ("K_Only", base_ids, k_only_kv, True),  # Recursive K, Baseline V, V_PROJ patch
                ("V_Only", base_ids, v_only_kv, True),  # Baseline K, Recursive V, V_PROJ patch
                ("Full_KV", base_ids, rec_kv, True),  # Full recursive KV, V_PROJ patch
            ]
            
            for cond_name, prompt_ids, kv_cache, use_v_patching in conditions:
                v_patcher = None
                if use_v_patching:
                    v_patcher = PersistentVPatcher(model, rec_v_l27)
                    v_patcher.register(layer_idx=TARGET_LAYER_V)
                
                try:
                    text, entropy = _generate_with_kv(
                        model, tokenizer, prompt_ids, kv_cache, MAX_NEW_TOKENS, TEMPERATURE
                    )
                    
                    score = score_behavior_strict(text, entropy)
                    
                    results.append({
                        "pair_idx": i,
                        "condition": cond_name,
                        "prompt": base_text,
                        "generated_text": text,
                        "text_len": len(text),
                        "entropy": entropy,
                        "used_v_patching": use_v_patching,
                        **score.to_dict()
                    })
                finally:
                    if v_patcher is not None:
                        v_patcher.remove()
        
        except Exception as e:
            print(f"Error on pair {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"results/runs/{timestamp}_kv_separation")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(run_dir / "kv_separation_results.csv", index=False)
    
    # Summary
    summary = {
        "experiment": "kv_separation",
        "model_name": "mistralai/Mistral-7B-v0.1",
        "n_pairs": len(pairs),
        "conditions": {}
    }
    
    for cond in df["condition"].unique():
        sub = df[df["condition"] == cond]
        summary["conditions"][cond] = {
            "mean_score": float(sub["final_score"].mean()),
            "pass_rate": float(sub["passed_gates"].mean()),
            "samples_above_zero": int((sub["final_score"] > 0).sum()),
            "samples_above_0_3": int((sub["final_score"] > 0.3).sum()),
        }
    
    summary["prompt_bank_version"] = bank_version
    
    with open(run_dir / "kv_separation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Results saved to {run_dir}")
    print("\n=== SUMMARY ===")
    for cond, stats in summary["conditions"].items():
        print(f"{cond}:")
        print(f"  Mean Score: {stats['mean_score']:.4f}")
        print(f"  Pass Rate: {stats['pass_rate']*100:.1f}%")
        print(f"  Samples > 0: {stats['samples_above_zero']}/{len(pairs)}")
        print(f"  Samples > 0.3: {stats['samples_above_0_3']}/{len(pairs)}")
    
    return run_dir, summary

if __name__ == "__main__":
    run_kv_separation_experiment()









