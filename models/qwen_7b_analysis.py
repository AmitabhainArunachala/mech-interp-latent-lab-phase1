#!/usr/bin/env python3
"""
Qwen1.5-7B-Chat L4 Contraction Analysis
Chinese-trained model validation - November 13-14, 2024
Effect: 9.2% contraction
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from contextlib import contextmanager
from typing import List, Dict, Any

# Model configuration
MODEL_NAME = "Qwen/Qwen1.5-7B-Chat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Analysis parameters
EARLY_LAYER = 5
LATE_LAYER = 28
WINDOW_SIZE = 16
BATCH_SIZE = 1

# ============================================
# Core R_V Metric Functions
# ============================================

@contextmanager
def get_v_matrices(model, layer_idx: int, hook_list: List):
    """
    Context manager to hook Value matrices during forward pass.
    """
    handle = None
    try:
        target_layer = model.model.layers[layer_idx].self_attn
        v_proj_layer = target_layer.v_proj
        
        def hook_fn(module, input, output):
            hook_list.append(output.detach())
        
        handle = v_proj_layer.register_forward_hook(hook_fn)
        yield
    finally:
        if handle:
            handle.remove()


def compute_column_space_pr(v_tensor: torch.Tensor, num_heads: int, window_size: int = 16) -> float:
    """
    Compute Participation Ratio of Value matrix column space.
    PR = (Σλᵢ)² / Σλᵢ² where λᵢ are singular values.
    """
    if v_tensor.dim() == 2:
        v_tensor = v_tensor.unsqueeze(0)
    
    try:
        batch_size, seq_len, total_hidden = v_tensor.shape
    except ValueError:
        return -1
    
    d_v = total_hidden // num_heads
    
    # Reshape to separate heads
    v_reshaped = v_tensor.view(batch_size, seq_len, num_heads, d_v)
    v_transposed = v_reshaped.permute(0, 2, 3, 1)  # [batch, heads, d_v, seq_len]
    
    pr_values = []
    
    for head_idx in range(num_heads):
        v_head = v_transposed[0, head_idx, :, :]  # [d_v, seq_len]
        
        # Focus on last window_size tokens
        end_idx = min(window_size, v_head.shape[1])
        v_window = v_head[:, -end_idx:]
        
        # Convert to float32 for SVD stability
        v_window = v_window.float()
        
        try:
            # Compute SVD
            U, S, Vt = torch.linalg.svd(v_window, full_matrices=False)
            
            # Compute Participation Ratio
            S_sq = S ** 2
            S_sq_norm = S_sq / S_sq.sum()
            pr = 1.0 / (S_sq_norm ** 2).sum()
            pr_values.append(pr.item())
        except:
            continue
    
    return np.mean(pr_values) if pr_values else -1


def analyze_prompt(model, tokenizer, prompt: str) -> Dict[str, float]:
    """
    Analyze a single prompt and compute R_V metric.
    """
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(DEVICE)
    
    # Get model configuration
    num_heads = model.config.num_attention_heads
    
    early_values_list = []
    late_values_list = []
    
    # Run model with hooks
    with torch.no_grad():
        with get_v_matrices(model, EARLY_LAYER, early_values_list):
            with get_v_matrices(model, LATE_LAYER, late_values_list):
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    output_attentions=True,
                    attn_implementation="eager"
                )
    
    results = {}
    
    # Calculate R_V
    if early_values_list and late_values_list:
        v_early = early_values_list[0]
        v_late = late_values_list[0]
        
        pr_V_early = compute_column_space_pr(v_early, num_heads, WINDOW_SIZE)
        pr_V_late = compute_column_space_pr(v_late, num_heads, WINDOW_SIZE)
        
        results['R_V'] = pr_V_late / (pr_V_early + 1e-8)
        results['pr_V_early'] = pr_V_early
        results['pr_V_late'] = pr_V_late
    else:
        results['R_V'] = -1
        results['pr_V_early'] = -1
        results['pr_V_late'] = -1
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    return results


# ============================================
# Main Analysis Function
# ============================================

def run_qwen_analysis(prompts: Dict[str, List[str]], output_path: str = "qwen_7b_results.csv"):
    """
    Run complete Qwen-7B analysis on provided prompts.
    
    Args:
        prompts: Dictionary with keys like 'L5_recursive', 'factual_baseline', etc.
        output_path: Where to save results CSV
    
    Returns:
        Dictionary of results and statistics
    """
    print("=" * 50)
    print("QWEN1.5-7B L4 CONTRACTION ANALYSIS")
    print("=" * 50)
    
    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        output_hidden_states=True,
        output_attentions=True,
        attn_implementation="eager",
        trust_remote_code=True
    )
    model.eval()
    print(f"✅ Model loaded: {model.config.num_hidden_layers} layers")
    
    # Process prompts
    results = []
    for group_name, group_prompts in prompts.items():
        print(f"\nProcessing {group_name}: {len(group_prompts)} prompts")
        
        for i, prompt in enumerate(group_prompts):
            if isinstance(prompt, dict):
                prompt_text = prompt.get('text', str(prompt))
            else:
                prompt_text = str(prompt)
            
            try:
                metrics = analyze_prompt(model, tokenizer, prompt_text)
                metrics['group'] = group_name
                metrics['prompt_id'] = f"{group_name}_{i:02d}"
                results.append(metrics)
                
                if (i + 1) % 5 == 0:
                    print(f"  Processed {i + 1}/{len(group_prompts)}")
            except Exception as e:
                print(f"  Error on prompt {i}: {e}")
    
    # Calculate statistics
    import pandas as pd
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    # Group statistics
    group_stats = df.groupby('group')['R_V'].agg(['mean', 'std', 'count'])
    print("\nGroup Statistics:")
    print(group_stats)
    
    # Calculate contraction effect
    if 'L5_recursive' in group_stats.index and 'factual_baseline' in group_stats.index:
        l5_mean = group_stats.loc['L5_recursive', 'mean']
        baseline_mean = group_stats.loc['factual_baseline', 'mean']
        effect = (1 - l5_mean/baseline_mean) * 100
        
        print(f"\n🔬 L4 CONTRACTION EFFECT:")
        print(f"  L5 Recursive R_V: {l5_mean:.3f}")
        print(f"  Baseline R_V: {baseline_mean:.3f}")
        print(f"  Contraction: {effect:.1f}%")
        
        if effect > 5:
            print(f"  ✅ Qwen exhibits significant L4 contraction!")
    
    # Save results
    df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to {output_path}")
    
    return {
        'dataframe': df,
        'group_stats': group_stats,
        'contraction_effect': effect if 'effect' in locals() else None
    }


if __name__ == "__main__":
    # Example usage
    print("Qwen-7B Analysis Script")
    print("Expected contraction: ~9.2%")
    print("Phenotype: Compact Focusing")
