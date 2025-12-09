#!/usr/bin/env python3
"""
Pythia-2.8B R_V Measurement - EXACT Mistral Methodology

This script replicates the EXACT measurement protocol used for Mistral-7B:
- Same layer indices (early=5, late=28)
- Same window size (16 tokens)
- Same value extraction method (v_proj output)
- Same participation ratio computation
- Same R_V formula (PR_late / PR_early)

Adapted from: models/mistral_7b_analysis.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from contextlib import contextmanager
from typing import List, Dict, Any
import pandas as pd
import json
from pathlib import Path

# Import original prompt bank
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from n300_mistral_test_prompt_bank import prompt_bank_1c

# ============================================
# EXACT MISTRAL CONFIGURATION
# ============================================

# Model configuration - CHANGE THIS FOR PYTHIA
MODEL_NAME = "EleutherAI/pythia-2.8b"  # Change to pythia-2.8b-deduped for checkpoint 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# EXACT parameters from Mistral measurement
EARLY_LAYER = 5
LATE_LAYER = 28  # For 32-layer model (84% depth)
WINDOW_SIZE = 16
BATCH_SIZE = 1

# ============================================
# EXACT MISTRAL VALUE EXTRACTION METHOD
# ============================================

@contextmanager
def get_v_matrices(model, layer_idx: int, hook_list: List):
    """
    EXACT Mistral method: Hook v_proj output during forward pass.
    
    For Mistral: model.model.layers[layer_idx].self_attn.v_proj
    For Pythia:  model.gpt_neox.layers[layer_idx].attention.v_proj
    """
    handle = None
    try:
        # Architecture detection
        if hasattr(model, 'gpt_neox'):  # Pythia (GPT-NeoX)
            target_layer = model.gpt_neox.layers[layer_idx].attention.v_proj
        elif hasattr(model, 'model'):  # Mistral/Qwen/Llama
            target_layer = model.model.layers[layer_idx].self_attn.v_proj
        else:
            raise ValueError(f"Unknown architecture: {type(model)}")
        
        def hook_fn(module, input, output):
            hook_list.append(output.detach())
        
        handle = target_layer.register_forward_hook(hook_fn)
        yield
    finally:
        if handle:
            handle.remove()


# ============================================
# EXACT MISTRAL PARTICIPATION RATIO COMPUTATION
# ============================================

def compute_column_space_pr(v_tensor: torch.Tensor, num_heads: int, window_size: int = 16) -> float:
    """
    EXACT Mistral method: Compute Participation Ratio of Value matrix column space.
    
    Formula: PR = (Σλᵢ)² / Σλᵢ² where λᵢ are singular values.
    
    Process:
    1. Reshape to separate heads: [batch, seq_len, num_heads, d_v]
    2. Transpose to [batch, heads, d_v, seq_len]
    3. For each head, take last window_size tokens
    4. Compute SVD on [d_v, window_size] matrix
    5. Average PR across heads
    """
    if v_tensor.dim() == 2:
        v_tensor = v_tensor.unsqueeze(0)
    
    try:
        batch_size, seq_len, total_hidden = v_tensor.shape
    except ValueError:
        return -1
    
    d_v = total_hidden // num_heads
    
    # Reshape to separate heads - EXACT Mistral method
    v_reshaped = v_tensor.view(batch_size, seq_len, num_heads, d_v)
    v_transposed = v_reshaped.permute(0, 2, 3, 1)  # [batch, heads, d_v, seq_len]
    
    pr_values = []
    
    for head_idx in range(num_heads):
        v_head = v_transposed[0, head_idx, :, :]  # [d_v, seq_len]
        
        # Focus on last window_size tokens - EXACT Mistral method
        end_idx = min(window_size, v_head.shape[1])
        v_window = v_head[:, -end_idx:]
        
        # Convert to float32 for SVD stability
        v_window = v_window.float()
        
        try:
            # Compute SVD - EXACT Mistral method
            U, S, Vt = torch.linalg.svd(v_window, full_matrices=False)
            
            # Compute Participation Ratio - EXACT Mistral formula
            S_sq = S ** 2
            S_sq_norm = S_sq / S_sq.sum()
            pr = 1.0 / (S_sq_norm ** 2).sum()
            pr_values.append(pr.item())
        except:
            continue
    
    return np.mean(pr_values) if pr_values else -1


# ============================================
# EXACT MISTRAL R_V COMPUTATION
# ============================================

def analyze_prompt(model, tokenizer, prompt: str) -> Dict[str, float]:
    """
    EXACT Mistral method: Analyze a single prompt and compute R_V metric.
    
    Process:
    1. Tokenize prompt (max_length=512, padding=True, truncation=True)
    2. Hook v_proj at EARLY_LAYER and LATE_LAYER
    3. Run forward pass
    4. Compute PR_early and PR_late
    5. R_V = PR_late / PR_early
    """
    # Tokenize - EXACT Mistral method
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
    
    # Run model with hooks - EXACT Mistral method
    with torch.no_grad():
        with get_v_matrices(model, EARLY_LAYER, early_values_list):
            with get_v_matrices(model, LATE_LAYER, late_values_list):
                outputs = model(
                    **inputs,
                    output_hidden_states=True
                )
    
    results = {}
    
    # Calculate R_V - EXACT Mistral formula
    if early_values_list and late_values_list:
        v_early = early_values_list[0]
        v_late = late_values_list[0]
        
        pr_V_early = compute_column_space_pr(v_early, num_heads, WINDOW_SIZE)
        pr_V_late = compute_column_space_pr(v_late, num_heads, WINDOW_SIZE)
        
        # EXACT Mistral formula: R_V = PR_late / PR_early
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
# MAIN ANALYSIS FUNCTION
# ============================================

def run_pythia_analysis(prompts: Dict[str, List[str]], output_path: str = "pythia_2.8b_results.csv", 
                        checkpoint_name: str = None):
    """
    Run complete Pythia-2.8B analysis using EXACT Mistral methodology.
    
    Args:
        prompts: Dictionary with keys like 'L5_recursive', 'factual_baseline', etc.
        output_path: Where to save results CSV
        checkpoint_name: Optional checkpoint name (e.g., "pythia-2.8b-deduped" for checkpoint 0)
    
    Returns:
        Dictionary of results and statistics
    """
    print("=" * 70)
    print("PYTHIA-2.8B R_V ANALYSIS - EXACT MISTRAL METHODOLOGY")
    print("=" * 70)
    
    # Load model - EXACT Mistral configuration
    model_name = checkpoint_name if checkpoint_name else MODEL_NAME
    print(f"\nLoading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        output_hidden_states=True,
        attn_implementation="eager"  # EXACT Mistral setting
    )
    model.eval()
    
    # Verify layer count
    if hasattr(model, 'gpt_neox'):
        num_layers = len(model.gpt_neox.layers)
    elif hasattr(model, 'model'):
        num_layers = len(model.model.layers)
    else:
        num_layers = 32
    
    print(f"✅ Model loaded: {num_layers} layers")
    print(f"   Early layer: {EARLY_LAYER}")
    print(f"   Late layer: {LATE_LAYER} ({LATE_LAYER/num_layers*100:.1f}% depth)")
    print(f"   Window size: {WINDOW_SIZE} tokens")
    
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
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    # Group statistics
    group_stats = df.groupby('group')['R_V'].agg(['mean', 'std', 'count'])
    print("\nGroup Statistics:")
    print(group_stats)
    
    # Calculate contraction effect
    if 'L5_recursive' in group_stats.index and 'factual_baseline' in group_stats.index:
        l5_mean = group_stats.loc['L5_recursive', 'mean']
        baseline_mean = group_stats.loc['factual_baseline', 'mean']
        effect = (1 - l5_mean/baseline_mean) * 100
        
        print(f"\n🔬 R_V COMPARISON:")
        print(f"  L5 Recursive R_V: {l5_mean:.3f}")
        print(f"  Baseline R_V: {baseline_mean:.3f}")
        print(f"  Difference: {effect:+.1f}%")
        
        if effect < -5:
            print(f"  ⚠️  EXPANSION detected (opposite of Mistral contraction)")
        elif effect > 5:
            print(f"  ✅ CONTRACTION detected (matches Mistral)")
        else:
            print(f"  ➡️  NEUTRAL (no significant effect)")
    
    # Save results
    df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to {output_path}")
    
    return {
        'dataframe': df,
        'group_stats': group_stats,
        'contraction_effect': effect if 'effect' in locals() else None
    }


# ============================================
# CHECKPOINT SWEEP FUNCTION
# ============================================

def run_checkpoint_sweep(checkpoints: List[int], prompts: Dict[str, List[str]], 
                         base_model: str = "EleutherAI/pythia-2.8b"):
    """
    Run analysis across multiple Pythia checkpoints.
    
    Args:
        checkpoints: List of checkpoint indices (e.g., [0, 5000, 10000, 143000])
        prompts: Prompt dictionary
        base_model: Base model name
    """
    all_results = []
    
    for checkpoint in checkpoints:
        print(f"\n{'='*70}")
        print(f"CHECKPOINT {checkpoint}")
        print(f"{'='*70}")
        
        if checkpoint == 0:
            checkpoint_name = f"{base_model}-deduped"
        elif checkpoint == 143000:  # Final checkpoint
            checkpoint_name = base_model
        else:
            checkpoint_name = f"{base_model}-step{checkpoint}"
        
        try:
            results = run_pythia_analysis(
                prompts=prompts,
                output_path=f"pythia_checkpoint_{checkpoint}_results.csv",
                checkpoint_name=checkpoint_name
            )
            
            # Add checkpoint info
            results['checkpoint'] = checkpoint
            all_results.append(results)
            
        except Exception as e:
            print(f"Error processing checkpoint {checkpoint}: {e}")
            continue
    
    return all_results


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Load prompts from original bank
    print("Loading prompts from n300_mistral_test_prompt_bank.py...")
    
    # Extract L5_recursive and factual_baseline prompts
    l5_prompts = [v['text'] for k, v in prompt_bank_1c.items() 
                  if v['group'] == 'L5_refined'][:20]  # First 20
    
    factual_prompts = [v['text'] for k, v in prompt_bank_1c.items() 
                      if 'factual' in v['group']][:20]  # First 20
    
    test_prompts = {
        'L5_recursive': l5_prompts,
        'factual_baseline': factual_prompts
    }
    
    print(f"Loaded {len(l5_prompts)} L5 prompts and {len(factual_prompts)} factual prompts")
    
    # Run single checkpoint test
    print("\n" + "="*70)
    print("TESTING FINAL CHECKPOINT (143k)")
    print("="*70)
    
    results = run_pythia_analysis(
        prompts=test_prompts,
        output_path="pythia_2.8b_final_checkpoint.csv"
    )
    
    print("\n✅ Pythia-2.8B analysis complete!")
    print("\nNext: Run checkpoint sweep to test emergence hypothesis")

