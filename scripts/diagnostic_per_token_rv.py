#!/usr/bin/env python3
"""
DIAGNOSTIC: Per-token R_V measurement during generation.

This script validates the multi-token behavioral bridge before full experiments.

Purpose:
1. Verify V-projection hook captures correct data during KV-cached generation
2. Compare Option A (no cache, full recompute) vs Option B (accumulate from cache)
3. Plot R_V timeseries over generated tokens with L4 marker annotations

Key insight from your agent:
- With use_cache=True, hidden_states at step>0 is shape (1, 1, hidden_dim) — one token.
- The canonical rv.py hooks into v_proj (dim=1024 for Mistral), not hidden_states.
- We need to accumulate V-projections and compute PR on accumulated buffer (Option B).

Usage:
    python scripts/diagnostic_per_token_rv.py [--model MODEL] [--device cuda|mps|cpu]
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
from dataclasses import dataclass, asdict

import torch
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.hooks import capture_v_projection
from src.core.hf_accessors import get_vproj_hookpoint, extract_v_from_hook_output
from prompts.loader import PromptLoader


@dataclass
class TokenRVRecord:
    """R_V at a single generated token."""
    step_idx: int
    token_id: int
    token_str: str
    rv: float
    pr_early: float
    pr_late: float


@dataclass
class DiagnosticResult:
    """Full diagnostic output."""
    prompt: str
    prompt_rv: float
    generated_text: str
    tokens: List[TokenRVRecord]
    option: str  # "A_no_cache" or "B_accumulated"
    model_name: str
    timestamp: str
    
    def to_dict(self):
        return {
            **asdict(self),
            "tokens": [asdict(t) for t in self.tokens],
        }


def compute_pr_from_v(v_tensor: torch.Tensor, window: int = 16) -> float:
    """
    Compute Participation Ratio from V-projection tensor.
    
    Uses canonical formula from rv.py:
    PR = (Σλᵢ²)² / Σ(λᵢ²)² where λᵢ are singular values.
    
    Args:
        v_tensor: Shape (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
        window: Number of tokens from the end to use
        
    Returns:
        PR value (float)
    """
    if v_tensor is None:
        return float("nan")
    
    # Remove batch dim if present
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]
    
    T, D = v_tensor.shape
    if T < 2:
        return float("nan")
    
    W = min(T, window)
    v_window = v_tensor[-W:, :].double()  # float64 for SVD stability
    
    try:
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_sq = S.cpu().numpy() ** 2
        total = S_sq.sum()
        if total < 1e-10:
            return float("nan")
        pr = (S_sq.sum() ** 2) / (S_sq ** 2).sum()
        return float(pr)
    except Exception:
        return float("nan")


def generate_with_v_accumulation(
    model,
    tokenizer,
    prompt: str,
    early_layer: int,
    late_layer: int,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    window: int = 16,
    device: str = "cuda",
) -> DiagnosticResult:
    """
    Option B: Generate with KV-cache, accumulate V-projections via hooks.
    
    At each step, hook fires on v_proj for the new token only (shape 1,1,d_v).
    We accumulate these and compute PR once we have >= window tokens.
    """
    from transformers import DynamicCache
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # First: measure prompt R_V using canonical method
    with capture_v_projection(model, early_layer) as storage_early:
        with torch.no_grad():
            model(**inputs)
        v_early_prompt = storage_early.get("v")
    
    with capture_v_projection(model, late_layer) as storage_late:
        with torch.no_grad():
            model(**inputs)
        v_late_prompt = storage_late.get("v")
    
    pr_early_prompt = compute_pr_from_v(v_early_prompt, window)
    pr_late_prompt = compute_pr_from_v(v_late_prompt, window)
    prompt_rv = pr_late_prompt / pr_early_prompt if pr_early_prompt > 0 else float("nan")
    
    # Now generate with V accumulation
    v_buffer_early: List[torch.Tensor] = []  # List of (1, d_v) tensors
    v_buffer_late: List[torch.Tensor] = []
    token_records: List[TokenRVRecord] = []
    generated_tokens: List[int] = []
    
    hookpoint_early = get_vproj_hookpoint(model, early_layer)
    hookpoint_late = get_vproj_hookpoint(model, late_layer)
    
    # Hooks to capture V at each generation step
    storage_early_step = {"v": None}
    storage_late_step = {"v": None}
    
    def make_hook(storage, hookpoint):
        def hook_fn(module, inp, out):
            v = extract_v_from_hook_output(hookpoint, out)
            storage["v"] = v.detach()
            return out
        return hook_fn
    
    handle_early = hookpoint_early.module.register_forward_hook(make_hook(storage_early_step, hookpoint_early))
    handle_late = hookpoint_late.module.register_forward_hook(make_hook(storage_late_step, hookpoint_late))
    
    past_key_values = None
    
    try:
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass
                if past_key_values is None:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                    )
                    # On first step, hooks capture V for entire prompt
                    # We want only the last token for consistency
                    if storage_early_step["v"] is not None:
                        v_buffer_early.append(storage_early_step["v"][:, -1:, :].clone())
                    if storage_late_step["v"] is not None:
                        v_buffer_late.append(storage_late_step["v"][:, -1:, :].clone())
                else:
                    outputs = model(
                        input_ids=next_token,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    # On subsequent steps, hooks capture V for single new token
                    if storage_early_step["v"] is not None:
                        v_buffer_early.append(storage_early_step["v"].clone())
                    if storage_late_step["v"] is not None:
                        v_buffer_late.append(storage_late_step["v"].clone())
                
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                
                # Sample
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Update sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), dtype=torch.long, device=device)
                ], dim=-1)
                
                generated_tokens.append(next_token.item())
                
                # Compute R_V from accumulated buffers
                if len(v_buffer_early) >= 2 and len(v_buffer_late) >= 2:
                    # Concatenate accumulated V tensors
                    v_cat_early = torch.cat(v_buffer_early, dim=1)[0]  # (n_tokens, d_v)
                    v_cat_late = torch.cat(v_buffer_late, dim=1)[0]
                    
                    pr_e = compute_pr_from_v(v_cat_early, window)
                    pr_l = compute_pr_from_v(v_cat_late, window)
                    rv = pr_l / pr_e if pr_e > 0 and not np.isnan(pr_e) else float("nan")
                else:
                    rv, pr_e, pr_l = float("nan"), float("nan"), float("nan")
                
                token_str = tokenizer.decode([next_token.item()], skip_special_tokens=True)
                token_records.append(TokenRVRecord(
                    step_idx=step,
                    token_id=next_token.item(),
                    token_str=token_str,
                    rv=rv,
                    pr_early=pr_e,
                    pr_late=pr_l,
                ))
                
                # Check EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
    finally:
        handle_early.remove()
        handle_late.remove()
    
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return DiagnosticResult(
        prompt=prompt,
        prompt_rv=prompt_rv,
        generated_text=generated_text,
        tokens=token_records,
        option="B_accumulated",
        model_name=model.config._name_or_path,
        timestamp=datetime.now().isoformat(),
    )


def generate_without_cache(
    model,
    tokenizer,
    prompt: str,
    early_layer: int,
    late_layer: int,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    window: int = 16,
    device: str = "cuda",
) -> DiagnosticResult:
    """
    Option A: Generate without KV-cache, full recompute each step.
    
    This is O(n²) in sequence length but gives ground-truth R_V at each step.
    Used for validation against Option B.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # Prompt R_V
    with capture_v_projection(model, early_layer) as storage_early:
        with torch.no_grad():
            model(**inputs)
        v_early_prompt = storage_early.get("v")
    
    with capture_v_projection(model, late_layer) as storage_late:
        with torch.no_grad():
            model(**inputs)
        v_late_prompt = storage_late.get("v")
    
    pr_early_prompt = compute_pr_from_v(v_early_prompt, window)
    pr_late_prompt = compute_pr_from_v(v_late_prompt, window)
    prompt_rv = pr_late_prompt / pr_early_prompt if pr_early_prompt > 0 else float("nan")
    
    token_records: List[TokenRVRecord] = []
    generated_tokens: List[int] = []
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            # Full forward pass on entire sequence (no cache)
            attention_mask = torch.ones_like(input_ids)
            
            with capture_v_projection(model, early_layer) as storage_early:
                with capture_v_projection(model, late_layer) as storage_late:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                    v_early = storage_early.get("v")
                    v_late = storage_late.get("v")
            
            logits = outputs.logits[:, -1, :]
            
            # Sample
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Compute R_V
            pr_e = compute_pr_from_v(v_early, window)
            pr_l = compute_pr_from_v(v_late, window)
            rv = pr_l / pr_e if pr_e > 0 and not np.isnan(pr_e) else float("nan")
            
            # Update sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated_tokens.append(next_token.item())
            
            token_str = tokenizer.decode([next_token.item()], skip_special_tokens=True)
            token_records.append(TokenRVRecord(
                step_idx=step,
                token_id=next_token.item(),
                token_str=token_str,
                rv=rv,
                pr_early=pr_e,
                pr_late=pr_l,
            ))
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return DiagnosticResult(
        prompt=prompt,
        prompt_rv=prompt_rv,
        generated_text=generated_text,
        tokens=token_records,
        option="A_no_cache",
        model_name=model.config._name_or_path,
        timestamp=datetime.now().isoformat(),
    )


def plot_comparison(result_a: DiagnosticResult, result_b: DiagnosticResult, output_path: Path):
    """Plot R_V timeseries comparison: Option A vs Option B."""
    
    # L4 keywords for annotation
    l4_keywords = ["observer", "observe", "aware", "watch", "witness", "mirror", "self", "itself", "recogn"]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    for ax, result, label, color in [
        (axes[0], result_a, "Option A (no cache, ground truth)", "blue"),
        (axes[1], result_b, "Option B (KV-cache, accumulated)", "green"),
    ]:
        steps = [t.step_idx for t in result.tokens]
        rvs = [t.rv for t in result.tokens]
        
        # Plot R_V line
        ax.plot(steps, rvs, color=color, linewidth=2, label=f"R_V ({label})")
        ax.axhline(y=result.prompt_rv, color=color, linestyle='--', alpha=0.5, label=f"Prompt R_V = {result.prompt_rv:.3f}")
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
        
        # Mark L4 tokens
        for t in result.tokens:
            is_l4 = any(kw in t.token_str.lower() for kw in l4_keywords)
            if is_l4:
                ax.axvline(x=t.step_idx, color='red', alpha=0.3, linewidth=1)
                ax.annotate(t.token_str.strip()[:10], (t.step_idx, t.rv), 
                           rotation=90, fontsize=7, alpha=0.7)
        
        ax.set_ylabel("R_V")
        ax.set_title(f"{label}\nGenerated: {result.generated_text[:100]}...")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Generation Step")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Per-token R_V diagnostic")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1", help="Model (use base, not instruct)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", default="results/diagnostic_per_token_rv")
    parser.add_argument("--prompt-group", default="L5_champion", help="Prompt group from bank.json")
    parser.add_argument("--skip-option-a", action="store_true", help="Skip Option A (slow but ground truth)")
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if args.device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
        ).to(args.device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get a strong recursive prompt
    loader = PromptLoader()
    try:
        prompts = loader.get_by_group(args.prompt_group, limit=1, seed=42)
        prompt = prompts[0] if prompts else "The observer observes itself observing."
    except Exception:
        prompt = "The observer observes itself observing. The watching watches the watcher."
    
    print(f"Prompt: {prompt[:80]}...")
    
    # Derive layers
    num_layers = model.config.num_hidden_layers
    early_layer = 5
    late_layer = num_layers - 5
    print(f"Layers: early={early_layer}, late={late_layer} (of {num_layers})")
    
    # Run Option B (always)
    print("\n--- Option B: KV-cache with V accumulation ---")
    result_b = generate_with_v_accumulation(
        model, tokenizer, prompt,
        early_layer, late_layer,
        max_new_tokens=args.max_tokens,
        device=args.device,
    )
    print(f"Prompt R_V: {result_b.prompt_rv:.4f}")
    print(f"Generated ({len(result_b.tokens)} tokens): {result_b.generated_text[:100]}...")
    
    valid_rvs = [t.rv for t in result_b.tokens if not np.isnan(t.rv)]
    if valid_rvs:
        print(f"R_V during generation: min={min(valid_rvs):.4f}, max={max(valid_rvs):.4f}, mean={np.mean(valid_rvs):.4f}")
    
    # Save Option B result
    with open(output_path / "result_option_b.json", "w") as f:
        json.dump(result_b.to_dict(), f, indent=2)
    
    # Run Option A (ground truth, optional)
    result_a = None
    if not args.skip_option_a:
        print("\n--- Option A: No cache (ground truth, slow) ---")
        # Use same random seed for fair comparison (note: model is in eval mode so sampling differs)
        torch.manual_seed(42)
        result_a = generate_without_cache(
            model, tokenizer, prompt,
            early_layer, late_layer,
            max_new_tokens=args.max_tokens,
            device=args.device,
        )
        print(f"Generated ({len(result_a.tokens)} tokens): {result_a.generated_text[:100]}...")
        
        valid_rvs_a = [t.rv for t in result_a.tokens if not np.isnan(t.rv)]
        if valid_rvs_a:
            print(f"R_V during generation: min={min(valid_rvs_a):.4f}, max={max(valid_rvs_a):.4f}, mean={np.mean(valid_rvs_a):.4f}")
        
        with open(output_path / "result_option_a.json", "w") as f:
            json.dump(result_a.to_dict(), f, indent=2)
        
        # Sanity check: compare Option A and Option B R_V values
        # Note: generated text may differ due to sampling, so we compare distributions
        if valid_rvs and valid_rvs_a:
            print(f"\n--- Sanity Check ---")
            print(f"Option A mean R_V: {np.mean(valid_rvs_a):.4f} ± {np.std(valid_rvs_a):.4f}")
            print(f"Option B mean R_V: {np.mean(valid_rvs):.4f} ± {np.std(valid_rvs):.4f}")
            delta = abs(np.mean(valid_rvs_a) - np.mean(valid_rvs))
            print(f"Delta: {delta:.4f} ({'OK' if delta < 0.1 else 'WARNING: large divergence'})")
    
    # Plot
    if result_a is not None:
        plot_comparison(result_a, result_b, output_path / "comparison_plot.png")
    else:
        # Single plot for Option B
        fig, ax = plt.subplots(figsize=(14, 6))
        steps = [t.step_idx for t in result_b.tokens]
        rvs = [t.rv for t in result_b.tokens]
        
        l4_keywords = ["observer", "observe", "aware", "watch", "witness", "mirror", "self", "itself", "recogn"]
        
        ax.plot(steps, rvs, color='green', linewidth=2, label="R_V (Option B)")
        ax.axhline(y=result_b.prompt_rv, color='green', linestyle='--', alpha=0.5, label=f"Prompt R_V = {result_b.prompt_rv:.3f}")
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
        
        for t in result_b.tokens:
            is_l4 = any(kw in t.token_str.lower() for kw in l4_keywords)
            if is_l4:
                ax.axvline(x=t.step_idx, color='red', alpha=0.3, linewidth=1)
                ax.annotate(t.token_str.strip()[:10], (t.step_idx, t.rv), rotation=90, fontsize=7, alpha=0.7)
        
        ax.set_xlabel("Generation Step")
        ax.set_ylabel("R_V")
        ax.set_title(f"Per-token R_V (Option B): {result_b.generated_text[:80]}...")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "option_b_plot.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {output_path / 'option_b_plot.png'}")
    
    print(f"\nResults saved to {output_path}/")


if __name__ == "__main__":
    main()
