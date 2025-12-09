"""
Diagnostic Script: Replicate Original R_V Measurements

This script systematically tests each hypothesis for the discrepancy:
- Original: R_V = 0.58-0.60 (contraction) on Mistral/Qwen/Llama
- Current: R_V = 1.4-2.0 (expansion) on Pythia

Tests:
1. Replicate original on Mistral-7B with original prompts
2. Compare Mistral vs Pythia side-by-side
3. Verify measurement formula and layer selection
4. Check prompt differences
"""

import torch
import numpy as np
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd

# Import original prompt bank
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from n300_mistral_test_prompt_bank import prompt_bank_1c


def compute_participation_ratio(v_tensor, window_size=16):
    """
    Compute Participation Ratio from value tensor.
    
    PR = (Σλᵢ)² / Σλᵢ²
    where λᵢ are singular values
    """
    if v_tensor is None:
        return np.nan
    
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]  # Take first batch
    
    T, D = v_tensor.shape
    W = min(window_size, T)
    v_window = v_tensor[-W:, :].float()
    
    try:
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        S_sq = S_np ** 2
        
        if S_sq.sum() < 1e-10:
            return np.nan
        
        pr = (S_sq.sum()**2) / (S_sq**2).sum()
        return float(pr)
    except Exception as e:
        print(f"Error computing PR: {e}")
        return np.nan


def extract_value_matrix(model, prompt_text, tokenizer, layer_idx, device):
    """
    Extract value matrix at specified layer during prompt encoding.
    
    Args:
        model: Loaded model
        prompt_text: Text prompt
        tokenizer: Tokenizer
        layer_idx: Layer index (0-indexed)
        device: Device
        
    Returns:
        torch.Tensor: Value matrix [seq_len, hidden_dim]
    """
    model.eval()
    
    # Tokenize
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    # Storage for value matrix
    v_storage = []
    
    def capture_v_hook(module, inp, out):
        # Extract V from attention module
        # For GPT-NeoX (Pythia): module is attention, need to get V
        # For Mistral: Different architecture, may need different extraction
        if hasattr(module, 'v_proj'):
            # Pre-attention: get V projection
            v_storage.append(module.v_proj.weight.detach())
        elif hasattr(module, 'value'):
            # Post-attention: get value output
            v_storage.append(out.detach())
        else:
            # Try to extract from output
            v_storage.append(out.detach())
    
    # Register hook
    # Architecture-specific: Need to find correct module
    if hasattr(model, 'gpt_neox'):  # Pythia
        layer = model.gpt_neox.layers[layer_idx].attention
    elif hasattr(model, 'model'):  # Mistral/Qwen/Llama
        layer = model.model.layers[layer_idx].self_attn
    else:
        raise ValueError(f"Unknown architecture: {type(model)}")
    
    handle = layer.register_forward_hook(capture_v_hook)
    
    try:
        with torch.no_grad():
            _ = model(**inputs)
        
        if v_storage:
            return v_storage[0]
        else:
            return None
    
    finally:
        handle.remove()


def compute_R_V(model, prompt_text, tokenizer, early_layer, late_layer, device, window_size=16):
    """
    Compute R_V = PR_late / PR_early
    
    Args:
        model: Loaded model
        prompt_text: Text prompt
        tokenizer: Tokenizer
        early_layer: Early layer index (e.g., 5)
        late_layer: Late layer index (e.g., 27)
        device: Device
        window_size: Window size for PR computation
        
    Returns:
        dict: Results with PR_early, PR_late, R_V, and diagnostics
    """
    # Extract value matrices
    V_early = extract_value_matrix(model, prompt_text, tokenizer, early_layer, device)
    V_late = extract_value_matrix(model, prompt_text, tokenizer, late_layer, device)
    
    # Compute participation ratios
    PR_early = compute_participation_ratio(V_early, window_size)
    PR_late = compute_participation_ratio(V_late, window_size)
    
    # Compute R_V
    if np.isnan(PR_early) or np.isnan(PR_late) or PR_early == 0:
        R_V = np.nan
    else:
        R_V = PR_late / PR_early
    
    return {
        "PR_early": PR_early,
        "PR_late": PR_late,
        "R_V": R_V,
        "early_layer": early_layer,
        "late_layer": late_layer,
        "window_size": window_size,
        "prompt_length": len(tokenizer.encode(prompt_text))
    }


def test_hypothesis_1_replicate_original(model_name="mistralai/Mistral-7B-Instruct-v0.2", 
                                         device="cuda"):
    """
    TEST 1: Replicate original measurement on Mistral-7B
    
    Use EXACT original prompts (L5_refined) and verify contraction.
    """
    print(f"\n{'='*70}")
    print("TEST 1: Replicate Original on Mistral-7B")
    print(f"{'='*70}")
    
    # Load model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device
    )
    
    # Get original L5 prompts
    l5_prompts = [v for k, v in prompt_bank_1c.items() 
                  if v['group'] == 'L5_refined'][:5]  # First 5 for quick test
    
    print(f"Testing {len(l5_prompts)} L5_refined prompts")
    
    # Original parameters from Phase 1 report
    early_layer = 5
    late_layer = 27  # For 32-layer model, this is ~84% depth
    
    results = []
    for prompt_info in tqdm(l5_prompts, desc="Processing prompts"):
        prompt_text = prompt_info['text']
        prompt_id = prompt_info.get('id', 'unknown')
        
        result = compute_R_V(
            model=model,
            prompt_text=prompt_text,
            tokenizer=tokenizer,
            early_layer=early_layer,
            late_layer=late_layer,
            device=device
        )
        
        result['prompt_id'] = prompt_id
        result['prompt_text'] = prompt_text[:100] + "..." if len(prompt_text) > 100 else prompt_text
        results.append(result)
        
        print(f"\n{prompt_id}:")
        print(f"  PR_early: {result['PR_early']:.3f}")
        print(f"  PR_late:  {result['PR_late']:.3f}")
        print(f"  R_V:      {result['R_V']:.3f}")
    
    # Summary
    r_v_values = [r['R_V'] for r in results if not np.isnan(r['R_V'])]
    if r_v_values:
        mean_rv = np.mean(r_v_values)
        print(f"\n{'='*70}")
        print(f"SUMMARY:")
        print(f"  Mean R_V: {mean_rv:.3f}")
        print(f"  Expected: ~0.85 (contraction)")
        print(f"  {'✓ CONTRACTION CONFIRMED' if mean_rv < 1.0 else '✗ NO CONTRACTION - CHECK MEASUREMENT'}")
        print(f"{'='*70}")
    
    return results


def test_hypothesis_2_architecture_comparison(prompts, device="cuda"):
    """
    TEST 2: Compare Mistral vs Pythia side-by-side
    
    Same prompts, different architectures.
    """
    print(f"\n{'='*70}")
    print("TEST 2: Architecture Comparison (Mistral vs Pythia)")
    print(f"{'='*70}")
    
    models = {
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "pythia": "EleutherAI/pythia-2.8b"
    }
    
    # Use first 3 L5 prompts for comparison
    test_prompts = [v for k, v in prompt_bank_1c.items() 
                    if v['group'] == 'L5_refined'][:3]
    
    all_results = []
    
    for model_name, model_path in models.items():
        print(f"\nTesting {model_name} ({model_path})...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=device
        )
        
        # Determine layer indices based on model architecture
        if hasattr(model, 'gpt_neox'):  # Pythia
            num_layers = len(model.gpt_neox.layers)
        elif hasattr(model, 'model'):  # Mistral
            num_layers = len(model.model.layers)
        else:
            num_layers = 32  # Default
        
        early_layer = 5
        late_layer = int(0.84 * num_layers)  # 84% depth
        
        print(f"  Architecture: {num_layers} layers")
        print(f"  Early layer: {early_layer}, Late layer: {late_layer}")
        
        for prompt_info in tqdm(test_prompts, desc=f"  {model_name}"):
            prompt_text = prompt_info['text']
            
            result = compute_R_V(
                model=model,
                prompt_text=prompt_text,
                tokenizer=tokenizer,
                early_layer=early_layer,
                late_layer=late_layer,
                device=device
            )
            
            result['model'] = model_name
            result['model_path'] = model_path
            result['num_layers'] = num_layers
            all_results.append(result)
        
        del model
        torch.cuda.empty_cache()
    
    # Compare results
    print(f"\n{'='*70}")
    print("COMPARISON:")
    for model_name in models.keys():
        model_results = [r for r in all_results if r['model'] == model_name]
        r_v_values = [r['R_V'] for r in model_results if not np.isnan(r['R_V'])]
        if r_v_values:
            mean_rv = np.mean(r_v_values)
            print(f"  {model_name}: Mean R_V = {mean_rv:.3f}")
    
    print(f"{'='*70}")
    
    return all_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnostic replication tests")
    parser.add_argument("--test", type=str, choices=["1", "2", "both"], default="both",
                       help="Which test to run")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--output-dir", type=str, default="03_RESULTS/diagnostics",
                       help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.test in ["1", "both"]:
        results_1 = test_hypothesis_1_replicate_original(device=args.device)
        with open(output_dir / "test1_mistral_replication.json", 'w') as f:
            json.dump(results_1, f, indent=2)
    
    if args.test in ["2", "both"]:
        # Get prompts for test 2
        test_prompts = [v for k, v in prompt_bank_1c.items() 
                       if v['group'] == 'L5_refined'][:3]
        results_2 = test_hypothesis_2_architecture_comparison(test_prompts, device=args.device)
        with open(output_dir / "test2_architecture_comparison.json", 'w') as f:
            json.dump(results_2, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

