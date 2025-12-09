"""
Enhanced Measurement Pipeline for 2D Subsystem Mapping

Measures R_V (Participation Ratio) and Attention Entropy for prompt batches.
Designed for Phase 1: Creative, Planning, Uncertainty subsystems.
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# TODO: Copy validated functions from previous session
# - compute_participation_ratio()
# - compute_attention_entropy()
# - analyze_prompt_enhanced()

def compute_participation_ratio(v_tensor, window_size=16):
    """
    Compute R_V (Participation Ratio) from value tensor.
    
    Args:
        v_tensor: Value activations [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
        window_size: Window size for computation
        
    Returns:
        float: Participation ratio
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
        print(f"Error computing participation ratio: {e}")
        return np.nan


def compute_attention_entropy(attn_weights, layer_idx=None):
    """
    Compute attention entropy from attention weights.
    
    Args:
        attn_weights: Attention weights [batch, heads, seq_len, seq_len]
        layer_idx: Optional layer index for logging
        
    Returns:
        float: Normalized attention entropy
    """
    if attn_weights is None:
        return np.nan
    
    # Average over heads and batch
    if attn_weights.dim() == 4:
        attn_avg = attn_weights.mean(dim=(0, 1))  # [seq_len, seq_len]
    else:
        attn_avg = attn_weights
    
    # Normalize to probabilities
    attn_probs = attn_avg / (attn_avg.sum(dim=-1, keepdim=True) + 1e-10)
    
    # Compute entropy per position, then average
    entropy_per_pos = -torch.sum(attn_probs * torch.log(attn_probs + 1e-10), dim=-1)
    avg_entropy = entropy_per_pos.mean().item()
    
    # Normalize by log(seq_len) for comparison across lengths
    seq_len = attn_probs.shape[-1]
    normalized_entropy = avg_entropy / np.log(seq_len) if seq_len > 1 else avg_entropy
    
    return float(normalized_entropy)


def analyze_prompt_enhanced(model, tokenizer, prompt_text, device, layer_idx=27):
    """
    Analyze a single prompt and return R_V and Attention Entropy.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompt_text: Text prompt
        device: Device to run on
        layer_idx: Layer to measure at
        
    Returns:
        dict: Results with r_v, attention_entropy, and metadata
    """
    model.eval()
    
    # Tokenize
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    # Storage for activations
    v_storage = []
    attn_storage = []
    
    # Hook to capture activations
    def capture_v_hook(module, inp, out):
        # Extract v from attention output or intermediate
        v_storage.append(out.detach() if isinstance(out, torch.Tensor) else None)
    
    def capture_attn_hook(module, inp, out):
        # Extract attention weights
        if hasattr(module, 'attn_weights'):
            attn_storage.append(module.attn_weights.detach())
    
    # Register hooks
    layer = model.gpt_neox.layers[layer_idx].attention
    v_handle = layer.register_forward_hook(capture_v_hook)
    attn_handle = None  # TODO: Register attention hook appropriately
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Compute metrics
        v_tensor = v_storage[0] if v_storage else None
        r_v = compute_participation_ratio(v_tensor)
        
        attn_weights = attn_storage[0] if attn_storage else None
        attention_entropy = compute_attention_entropy(attn_weights)
        
        return {
            "r_v": r_v,
            "attention_entropy": attention_entropy,
            "prompt_length": inputs.input_ids.shape[1],
            "layer": layer_idx
        }
    
    except Exception as e:
        print(f"Error analyzing prompt: {e}")
        return {
            "r_v": np.nan,
            "attention_entropy": np.nan,
            "error": str(e)
        }
    
    finally:
        v_handle.remove()
        if attn_handle:
            attn_handle.remove()


def run_subsystem_batch(prompt_file, model_name="EleutherAI/pythia-2.8b", 
                        output_dir=None, device="cuda", layer_idx=27):
    """
    Run measurement pipeline on a subsystem's prompt bank.
    
    Args:
        prompt_file: Path to JSON prompt bank file
        model_name: Model to load
        output_dir: Directory to save results
        device: Device to run on
        layer_idx: Layer to measure at
    """
    # Load prompts
    with open(prompt_file, 'r') as f:
        prompt_data = json.load(f)
    
    subsystem = prompt_data['subsystem']
    prompts = prompt_data['prompts']
    
    print(f"\n{'='*60}")
    print(f"Processing subsystem: {subsystem}")
    print(f"Prompts: {len(prompts)}")
    print(f"Model: {model_name}")
    print(f"Layer: {layer_idx}")
    print(f"{'='*60}\n")
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device
    )
    
    # Process each prompt
    results = []
    for prompt_info in tqdm(prompts, desc=f"Processing {subsystem}"):
        prompt_text = prompt_info['text']
        prompt_id = prompt_info['id']
        
        result = analyze_prompt_enhanced(model, tokenizer, prompt_text, device, layer_idx)
        result.update({
            "prompt_id": prompt_id,
            "subsystem": subsystem,
            "prompt_text": prompt_text[:100] + "..." if len(prompt_text) > 100 else prompt_text
        })
        
        results.append(result)
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = output_dir / f"{subsystem}_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV summary
        df = pd.DataFrame(results)
        csv_path = output_dir / f"{subsystem}_summary.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"\nResults saved:")
        print(f"  JSON: {json_path}")
        print(f"  CSV:  {csv_path}")
    
    # Print summary
    r_v_mean = np.nanmean([r['r_v'] for r in results])
    ent_mean = np.nanmean([r['attention_entropy'] for r in results])
    
    print(f"\nSummary for {subsystem}:")
    print(f"  Mean R_V: {r_v_mean:.3f}")
    print(f"  Mean Attention Entropy: {ent_mean:.3f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run enhanced measurement pipeline")
    parser.add_argument("--prompt-file", type=str, required=True,
                       help="Path to prompt bank JSON file")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-2.8b",
                       help="Model name")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for results")
    parser.add_argument("--layer", type=int, default=27,
                       help="Layer index to measure at")
    
    args = parser.parse_args()
    
    run_subsystem_batch(
        prompt_file=args.prompt_file,
        model_name=args.model,
        output_dir=args.output_dir,
        layer_idx=args.layer
    )

