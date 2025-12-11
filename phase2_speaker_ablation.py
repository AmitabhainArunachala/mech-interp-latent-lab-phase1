"""
phase2_speaker_ablation.py

Task: Identify the "Speakers" (Readout Heads) at Layer 27.
Hypothesis: Heads [25, 26, 27] at Layer 27 read the contracted geometry and broadcast the recursive output.
Prediction: 
- Ablating these heads will reducing Recursive Behavior (Keywords).
- Ablating these heads will NOT restore Geometric Expansion (R_V will remain low).
- This dissociates "Internal State" (Geometry) from "External Output" (Behavior).

Method:
1. Load Recursive Prompts (L4/L5).
2. Measure Baseline (No Ablation): R_V and Behavior.
3. Measure Ablation (Heads 25,26,27 @ L27): R_V and Behavior.
4. Measure Control (Heads 0,1,2 @ L27): R_V and Behavior.
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from contextlib import contextmanager

sys.path.insert(0, os.path.abspath('.'))

from src.core.models import load_model, set_seed
from src.metrics.rv import compute_rv
from src.core.utils import behavior_score
from prompts.loader import PromptLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_TARGET = 27
HEADS_TARGET = [25, 26, 27]
HEADS_CONTROL = [0, 1, 2] # Arbitrary control heads

@contextmanager
def ablate_heads(model, layer_idx, head_indices):
    """
    Context manager to zero-out specific attention heads at a specific layer.
    Hooks into the self-attention module output.
    """
    # Mistral attention output is (batch, seq, num_heads * head_dim)
    # We need to reshape, zero out heads, and reshape back.
    # OR hook into 'o_proj' input? 
    # Better: Hook into the attention module's forward pass? 
    # MistralAttention output is usually (attn_output, past_key_value, ...)
    
    # We'll hook the SelfAttention layer output. 
    # But PyTorch hooks on modules give the OUTPUT of the module.
    # MistralFlashAttention2 forward returns (attn_output, attn_weights, past_key_value)
    
    # Easier: Hook the `o_proj` input? No, o_proj takes the concatenated result.
    # We can hook the output of the attention mechanism BEFORE o_proj?
    # Actually, if we zero out the head's contribution to the concatenation, it works.
    
    # Model config
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    def hook_fn(module, inputs, outputs):
        # outputs is a tuple: (attn_output, ...)
        # attn_output shape: (batch, seq, hidden_size)
        attn_output = outputs[0]
        
        # Reshape to (batch, seq, num_heads, head_dim)
        # Note: Mistral implementation might be different, let's verify.
        # Standard HF implementation: output is [batch, seq, hidden]
        # It is already concatenated.
        
        # We need to reshape to view heads
        batch_size, seq_len, _ = attn_output.shape
        attn_reshaped = attn_output.view(batch_size, seq_len, num_heads, head_dim)
        
        # Clone to avoid in-place errors if needed (though usually fine in hook)
        # We modify in place or return new tensor.
        mask = torch.ones_like(attn_reshaped)
        
        for h in head_indices:
            mask[:, :, h, :] = 0.0
            
        # Apply mask
        modified_attn = attn_reshaped * mask
        
        # Flatten back
        new_output = modified_attn.view(batch_size, seq_len, -1)
        
        return (new_output,) + outputs[1:]

    layer = model.model.layers[layer_idx].self_attn
    handle = layer.register_forward_hook(hook_fn)
    
    try:
        yield
    finally:
        handle.remove()

def run_ablation_test():
    print("Initializing Phase 2: Speaker Ablation...")
    set_seed(42)
    
    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    loader = PromptLoader()
    
    # Use Recursive Prompts (L4/L5) - Strongest Signal
    prompts = loader.get_by_group("L4_full", limit=10, seed=42) + loader.get_by_group("L5_refined", limit=10, seed=42)
    print(f"Testing on {len(prompts)} recursive prompts.")
    
    results = []
    
    for prompt in tqdm(prompts):
        # 1. Baseline (No Ablation)
        rv_base = compute_rv(model, tokenizer, prompt, device=DEVICE)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen_base = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
        text_base = tokenizer.decode(gen_base[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        score_base = behavior_score(text_base)
        
        # 2. Target Ablation (Heads 25, 26, 27)
        with ablate_heads(model, LAYER_TARGET, HEADS_TARGET):
            rv_target = compute_rv(model, tokenizer, prompt, device=DEVICE)
            with torch.no_grad():
                gen_target = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
            text_target = tokenizer.decode(gen_target[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            score_target = behavior_score(text_target)
            
        # 3. Control Ablation (Heads 0, 1, 2)
        with ablate_heads(model, LAYER_TARGET, HEADS_CONTROL):
            rv_ctrl = compute_rv(model, tokenizer, prompt, device=DEVICE)
            with torch.no_grad():
                gen_ctrl = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
            text_ctrl = tokenizer.decode(gen_ctrl[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            score_ctrl = behavior_score(text_ctrl)
            
        results.append({
            "prompt": prompt[:50] + "...",
            "rv_base": rv_base,
            "score_base": score_base,
            "rv_target": rv_target,
            "score_target": score_target,
            "rv_ctrl": rv_ctrl,
            "score_ctrl": score_ctrl,
            "text_target": text_target.replace('\n', ' ')[:100]
        })
        
    # Analysis
    df = pd.DataFrame(results)
    
    print("\nRESULTS SUMMARY:")
    print(f"Baseline:  R_V={df['rv_base'].mean():.3f}, Score={df['score_base'].mean():.2f}")
    print(f"Target:    R_V={df['rv_target'].mean():.3f}, Score={df['score_target'].mean():.2f}")
    print(f"Control:   R_V={df['rv_ctrl'].mean():.3f}, Score={df['score_ctrl'].mean():.2f}")
    
    # Calculate drops
    score_drop_target = (df['score_base'] - df['score_target']).mean()
    score_drop_ctrl = (df['score_base'] - df['score_ctrl']).mean()
    
    print(f"\nScore Drop (Target): {score_drop_target:.2f}")
    print(f"Score Drop (Control): {score_drop_ctrl:.2f}")
    
    # Save
    os.makedirs("results/dec11_evening", exist_ok=True)
    df.to_csv("results/dec11_evening/speaker_ablation.csv", index=False)
    
    # Log
    with open("logs/dec11_evening/speaker_ablation_log.txt", "w") as f:
        f.write("# Speaker Ablation Test Results\n")
        f.write(f"Target Heads: {HEADS_TARGET} @ L{LAYER_TARGET}\n")
        f.write(f"Control Heads: {HEADS_CONTROL} @ L{LAYER_TARGET}\n\n")
        f.write(df.to_markdown())
        
    print("\nAblation Test Complete.")

if __name__ == "__main__":
    run_ablation_test()

