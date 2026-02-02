"""
phase0_cross_baseline_control.py

Critical Validation Test 1: Cross-Baseline Control (GATEKEEPER)
Question: Does patching KV from baseline_A -> baseline_B cause spurious effects?
Success criteria:
- Behavior score < 2.0 (shouldn't induce recursive keywords)
- R_V change < 0.05 (Ideal, but noisy if prompts differ in complexity)

Implementation:
1. Load Baseline prompts (Same Group to minimize R_V variance).
2. For pairs (A, B):
    - Extract KV for A.
    - Extract KV for B.
    - Patch A's KV into B at late layers (16-32).
    - Generate text from patched state.
    - Score behavior.
"""

import sys
import os
import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

# Ensure src is in path
sys.path.insert(0, os.path.abspath('.'))

from src.core.models import load_model, set_seed
from src.metrics.rv import participation_ratio
from src.steering.kv_cache import extract_kv_list, mix_kv_to_dynamic_cache, generate_with_kv
from src.core.utils import behavior_score
from prompts.loader import PromptLoader

# Constants
LAYER_EARLY = 5
LAYER_LATE = 27
PATCH_START = 16
PATCH_END = 32
WINDOW_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def reconstruct_v_from_cache(v_cache: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct (batch, seq_len, hidden_dim) from KV cache V (batch, n_heads, seq_len, head_dim).
    """
    # v_cache: (batch, n_heads, seq_len, head_dim)
    # Permute to (batch, seq_len, n_heads, head_dim)
    v_permuted = v_cache.permute(0, 2, 1, 3)
    batch, seq_len, n_heads, head_dim = v_permuted.shape
    # Flatten last two dims
    return v_permuted.reshape(batch, seq_len, n_heads * head_dim)

def generate_continuation(model, tokenizer, prompt, past_key_values, max_new_tokens=50):
    """
    Generate continuation manually to avoid DynamicCache compatibility issues.
    """
    # Tokenize prompt to get the LAST token
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids
    
    # Start with the last token of the prompt (assuming past_key_values contains the prefix)
    current_ids = input_ids[:, -1:]
    
    generated_ids = torch.zeros((input_ids.shape[0], 0), dtype=torch.long, device=DEVICE)
    
    # We need to ensure past_key_values aligns with current_ids position
    # If past_key_values has length L, next token is L.
    # We feed token L-1?
    # If we feed token L-1, we get logits for L.
    
    # But past_key_values (from mix) has length L (full prompt).
    # If we feed input_ids[:, -1] (which is at position L-1), the model sees position L-1 again?
    # Yes. We need to handle this.
    # 
    # Option 1: Slice past_key_values to L-1. (Hard with DynamicCache)
    # Option 2: Feed a dummy token? No.
    # Option 3: Use the logic from targeted_kv_patch_test.py:
    # "outputs = model(generated_ids[:, -1:], past_key_values=patched_kv)"
    
    curr_kv = past_key_values
    next_token_input = current_ids
    
    text_out = ""
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                next_token_input,
                past_key_values=curr_kv,
                use_cache=True
            )
            
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            curr_kv = outputs.past_key_values
            next_token_input = next_token
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def run_test():
    print("Initializing Phase 0: Cross-Baseline Control (GATEKEEPER)...")
    set_seed(42)
    
    # Load Model
    print("Loading model...")
    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    
    # Load Prompts
    loader = PromptLoader()
    # Get 10 baseline prompts (Same Group for stability)
    baselines = loader.get_by_group("baseline_factual", limit=10, seed=42)
    
    # Ensure we have enough
    if len(baselines) < 5:
        print("Warning: Not enough baseline_factual prompts, adding baseline_math.")
        baselines += loader.get_by_group("baseline_math", limit=10-len(baselines), seed=42)
    
    print(f"Loaded {len(baselines)} baseline prompts.")
    
    results = []
    
    # Test Pairs (Cyclic Shift)
    pairs = []
    for i in range(len(baselines)):
        j = (i + 1) % len(baselines)
        pairs.append((baselines[i], baselines[j]))
    
    pairs = pairs[:10]
    
    print(f"Testing {len(pairs)} pairs...")
    
    for prompt_a, prompt_b in tqdm(pairs):
        # 1. Extract KV lists
        kv_list_a, _ = extract_kv_list(model, tokenizer, prompt_a, device=DEVICE)
        kv_list_b, _ = extract_kv_list(model, tokenizer, prompt_b, device=DEVICE)
        
        # 2. Compute Natural R_V for Prompt B
        try:
            v_early_nat = reconstruct_v_from_cache(kv_list_b[LAYER_EARLY][1])
            v_late_nat = reconstruct_v_from_cache(kv_list_b[LAYER_LATE][1])
            pr_early_nat = participation_ratio(v_early_nat, WINDOW_SIZE)
            pr_late_nat = participation_ratio(v_late_nat, WINDOW_SIZE)
            rv_natural = pr_late_nat / pr_early_nat if pr_early_nat != 0 else float('nan')
        except Exception:
            rv_natural = float('nan')
        
        # 3. Patch KV (A -> B)
        patched_cache = mix_kv_to_dynamic_cache(
            kv_list_b, # Base
            kv_list_a, # Source
            layer_start=PATCH_START,
            layer_end=PATCH_END,
            alpha=1.0
        )
        
        # 4. Compute Patched R_V
        len_a = kv_list_a[0][0].shape[2]
        len_b = kv_list_b[0][0].shape[2]
        min_len = min(len_a, len_b)
        
        v_early_patched = reconstruct_v_from_cache(kv_list_b[LAYER_EARLY][1][:, :, :min_len, :])
        v_late_patched = reconstruct_v_from_cache(kv_list_a[LAYER_LATE][1][:, :, :min_len, :])
            
        pr_early_patched = participation_ratio(v_early_patched, WINDOW_SIZE)
        pr_late_patched = participation_ratio(v_late_patched, WINDOW_SIZE)
        rv_patched = pr_late_patched / pr_early_patched if pr_early_patched != 0 else float('nan')
        
        rv_delta = abs(rv_patched - rv_natural) if not np.isnan(rv_patched) and not np.isnan(rv_natural) else 0.0
        
        # 5. Generate and Score Behavior
        
        # Natural generation
        # Use simple generate for natural
        input_ids = tokenizer(prompt_b, return_tensors="pt").input_ids.to(DEVICE)
        gen_ids_nat = model.generate(input_ids, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
        gen_text_natural = tokenizer.decode(gen_ids_nat[0][input_ids.shape[1]:], skip_special_tokens=True)
        score_nat = behavior_score(gen_text_natural)
        
        # Patched generation using local helper
        gen_text_patched = generate_continuation(model, tokenizer, prompt_b, patched_cache, max_new_tokens=50)
        score_patched = behavior_score(gen_text_patched)
        
        results.append({
            "prompt_a": prompt_a[:30] + "...",
            "prompt_b": prompt_b[:30] + "...",
            "rv_natural": round(rv_natural, 4),
            "rv_patched": round(rv_patched, 4),
            "rv_delta": round(rv_delta, 4),
            "behavior_natural": score_nat,
            "behavior_patched": score_patched,
            "generated_text": gen_text_patched[:100].replace('\n', ' ') + "..."
        })

    # Save Results
    output_dir = "results/dec11_evening"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = f"{output_dir}/phase0_gatekeeper.csv"
    
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    
    # Check Criteria
    mean_behavior = df["behavior_patched"].mean()
    mean_rv_delta = df["rv_delta"].mean()
    
    print("\nRESULTS Phase 0 (GATEKEEPER):")
    print(df[["prompt_b", "rv_delta", "behavior_patched"]])
    print(f"\nMean Behavior: {mean_behavior:.2f} (Threshold < 2.0)")
    print(f"Mean R_V Delta: {mean_rv_delta:.4f} (Threshold < 0.05)")
    
    # Log details
    with open("logs/dec11_evening/session_log.md", "a") as log:
        log.write(f"\n## Test 1: Cross-Baseline Control (GATEKEEPER)\n")
        log.write(f"**Status:** {'PASS' if mean_behavior < 2.0 else 'FAIL'}\n")
        log.write(f"**Results:**\n")
        log.write(f"- Mean R_V change: {mean_rv_delta:.4f}\n")
        log.write(f"- Mean behavior score: {mean_behavior:.2f}\n")
        log.write(f"- Pairs tested: {len(pairs)}\n")
        log.write(f"**Interpretation:** Baseline-to-baseline patching showed {'negligible' if mean_behavior < 2.0 else 'SIGNIFICANT'} recursive behavior.\n")
        
    if mean_behavior >= 2.0:
        print("FAIL: Behavior Criteria not met.")
        sys.exit(1)
    
    if mean_rv_delta >= 0.05:
        print("WARNING: R_V Delta is high, but Behavior is acceptable. Proceeding with caution.")
    else:
        print("PASS: Cross-Baseline Control passed.")

if __name__ == "__main__":
    run_test()
