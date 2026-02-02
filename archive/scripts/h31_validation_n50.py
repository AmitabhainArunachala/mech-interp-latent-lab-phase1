#!/usr/bin/env python3
"""
H31 Validation: Scale up to n=50-100 prompts

Measures H31 BOS attention, H31 entropy, and R_V at L27 for a larger sample
of prompts to validate the entropy separation claim (0.28 vs 0.81).

Uses PromptLoader to get diverse prompts from the bank.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List
import csv

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))

from src.core.models import load_model, set_seed
from src.metrics.rv import compute_rv
from prompts.loader import PromptLoader


def get_head_attention_stats(model, tokenizer, prompt: str, layer_idx: int, head_idx: int, device: str) -> Dict[str, float]:
    """
    Get H31 attention statistics at a specific layer.
    Returns: entropy, bos_attn, max_attn, marker_attn
    """
    from contextlib import contextmanager
    
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    # Move to device
    if device.startswith("cuda") or device == "mps":
        enc = {k: v.to(device) for k, v in enc.items()}
    # CPU: keep on CPU
    
    attention_weights = None
    
    def attention_hook(module, input, output):
        nonlocal attention_weights
        # output is tuple: (hidden_states, attention_weights, ...)
        if len(output) > 1 and output[1] is not None:
            # attention_weights shape: (batch, num_heads, seq_len, seq_len)
            attention_weights = output[1].detach()
    
    # Register hook on attention layer
    layer = model.model.layers[layer_idx].self_attn
    handle = layer.register_forward_hook(attention_hook)
    
    try:
        with torch.no_grad():
            _ = model(**enc, output_attentions=True)
    finally:
        handle.remove()
    
    if attention_weights is None:
        return {
            "entropy": np.nan,
            "bos_attn": np.nan,
            "max_attn": np.nan,
            "marker_attn": np.nan,
        }
    
    # Extract head attention (last token's attention pattern)
    # attention_weights: (batch, num_heads, seq_len, seq_len)
    head_attn = attention_weights[0, head_idx, -1, :].cpu().numpy()  # Last token attends to all
    
    # Add small epsilon to avoid log(0)
    head_attn = head_attn + 1e-10
    head_attn = head_attn / head_attn.sum()
    
    # Entropy
    entropy = float(-np.sum(head_attn * np.log(head_attn)))
    
    # BOS attention (token 0)
    bos_attn = float(head_attn[0])
    
    # Max attention
    max_attn = float(head_attn.max())
    
    # Marker attention (self-reference keywords)
    marker_tokens = ["itself", "observer", "process", "self", "aware"]
    marker_attn = 0.0
    for i in range(min(len(marker_tokens), len(head_attn))):
        # Simple approximation: check if token contains marker
        try:
            token_text = tokenizer.decode([enc["input_ids"][0, i].item()]).lower()
            if any(m in token_text for m in marker_tokens):
                marker_attn += head_attn[i]
        except:
            pass
    
    return {
        "entropy": entropy,
        "bos_attn": bos_attn,
        "max_attn": max_attn,
        "marker_attn": marker_attn,
    }


def determine_device():
    """Determine best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def main():
    print("=" * 80)
    print("H31 VALIDATION: n=50-100 Prompt Analysis")
    print("=" * 80)
    
    # Determine device
    device = determine_device()
    print(f"\nDevice: {device}")
    
    if device == "cpu":
        print("⚠️  WARNING: CPU inference will be VERY slow (hours for 50 prompts)")
        print("   Consider using RunPod with GPU instead.")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            return
    
    # Load model with appropriate settings
    print("\n[1/4] Loading model...")
    set_seed(42)
    
    # Load model - use load_model function but override attn_implementation
    if device == "cuda":
        # Load with eager attention for hook access
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=False)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
        )
        model.eval()
    elif device == "mps":
        # MPS: Try float16 first
        print("   Loading with float16 on MPS...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=False)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            torch_dtype=torch.float16,
            device_map="cpu",  # Load to CPU first
            attn_implementation="eager",
        )
        model = model.to("mps")
        model.eval()
        print("   Model moved to MPS")
    else:
        # CPU: Use float32
        print("   Loading on CPU (this will be slow)...")
        model, tokenizer = load_model(
            "mistralai/Mistral-7B-v0.1",
            device="cpu",
            torch_dtype=torch.float32,
        )
    
    model.eval()
    print(f"   Model loaded on {device}")
    
    # Load prompts
    print("\n[2/4] Loading prompts from PromptLoader...")
    loader = PromptLoader()
    
    # Get diverse prompts
    recursive_prompts = []
    baseline_prompts = []
    
    # Recursive prompts (target: 50)
    for group in ["L3_deeper", "L4_full", "L5_refined"]:
        prompts = loader.get_by_group(group, limit=20, seed=42)
        recursive_prompts.extend(prompts)
    
    # Baseline prompts (target: 50)
    for group in ["baseline_math", "baseline_creative", "long_control"]:
        prompts = loader.get_by_group(group, limit=20, seed=42)
        baseline_prompts.extend(prompts)
    
    # Trim to 50 each if we got more
    recursive_prompts = recursive_prompts[:50]
    baseline_prompts = baseline_prompts[:50]
    
    print(f"   Loaded {len(recursive_prompts)} recursive prompts")
    print(f"   Loaded {len(baseline_prompts)} baseline prompts")
    print(f"   Total: {len(recursive_prompts) + len(baseline_prompts)} prompts")
    
    # Prepare output
    output_dir = Path("results/h31_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "h31_validation_n50.csv"
    
    # Process prompts
    print("\n[3/4] Processing prompts...")
    rows = []
    
    all_prompts = [
        ("recursive", p) for p in recursive_prompts
    ] + [
        ("baseline", p) for p in baseline_prompts
    ]
    
    for prompt_type, prompt_text in tqdm(all_prompts, desc="Prompts"):
        try:
            # Compute R_V
            rv = compute_rv(model, tokenizer, prompt_text, early=5, late=27, window=16, device=device)
            
            # Get H31 stats
            h31_stats = get_head_attention_stats(model, tokenizer, prompt_text, layer_idx=27, head_idx=31, device=device)
            
            rows.append({
                "prompt_type": prompt_type,
                "prompt_text": prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text,
                "rv": rv,
                "h31_entropy": h31_stats["entropy"],
                "h31_bos_attn": h31_stats["bos_attn"],
                "h31_max_attn": h31_stats["max_attn"],
                "h31_marker_attn": h31_stats["marker_attn"],
            })
        except Exception as e:
            print(f"\n   Error processing prompt: {e}")
            continue
    
    # Save CSV
    print("\n[4/4] Saving results...")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_type", "prompt_text", "rv", "h31_entropy", "h31_bos_attn", "h31_max_attn", "h31_marker_attn"])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n✅ Results saved to: {csv_path}")
    
    # Quick summary
    recursive_rows = [r for r in rows if r["prompt_type"] == "recursive"]
    baseline_rows = [r for r in rows if r["prompt_type"] == "baseline"]
    
    if recursive_rows and baseline_rows:
        rec_entropy = [r["h31_entropy"] for r in recursive_rows if not np.isnan(r["h31_entropy"])]
        base_entropy = [r["h31_entropy"] for r in baseline_rows if not np.isnan(r["h31_entropy"])]
        
        if rec_entropy and base_entropy:
            print("\n" + "=" * 80)
            print("QUICK SUMMARY")
            print("=" * 80)
            print(f"Recursive prompts (n={len(rec_entropy)}):")
            print(f"  H31 Entropy: {np.mean(rec_entropy):.3f} ± {np.std(rec_entropy):.3f}")
            print(f"  Range: {np.min(rec_entropy):.3f} - {np.max(rec_entropy):.3f}")
            print(f"\nBaseline prompts (n={len(base_entropy)}):")
            print(f"  H31 Entropy: {np.mean(base_entropy):.3f} ± {np.std(base_entropy):.3f}")
            print(f"  Range: {np.min(base_entropy):.3f} - {np.max(base_entropy):.3f}")
            print(f"\nSeparation: {np.mean(base_entropy) - np.mean(rec_entropy):.3f}")
            print("=" * 80)


if __name__ == "__main__":
    main()

