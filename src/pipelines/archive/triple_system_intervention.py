"""
Pipeline: Triple-System Intervention Gradient

Tests intervention across three systems:
1. KV cache replacement (grounding)
2. V_PROJ patching/steering (mode shift)
3. Residual stream steering (activation modification)

Hypothesis: Light intervention across all three systems may produce cleaner
transfer than heavy intervention to one system.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import DynamicCache

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.core.patching import (
    PersistentVPatcher,
    extract_v_activation,
    extract_residual_activation,
)
from src.pipelines.archive.steering import compute_steering_vector, SteeringVectorPatcher
from src.pipelines.registry import ExperimentResult


class ResidualStreamSteeringPatcher:
    """
    Adds steering vector to residual stream at a specific layer.
    
    This modifies the residual stream AFTER attention+MLP, before it goes to next layer.
    
    Note: Steering vector from V_PROJ (1024 dims) needs to be projected to residual size (4096 dims).
    We use the V_PROJ's weight matrix transpose to project: residual_dim = V_PROJ.weight.T @ steering_vector
    """
    
    def __init__(self, model, steering_vector: torch.Tensor, alpha: float, v_proj_layer: int = 27):
        """
        Initialize patcher with steering vector and scaling factor.
        
        Args:
            model: The transformer model
            steering_vector: Vector from V_PROJ (shape: v_proj_out_features, typically 1024)
            alpha: Scaling factor
            v_proj_layer: Layer to extract V_PROJ weight matrix from (for projection)
        """
        self.model = model
        self.steering_vector = steering_vector.detach().to(model.device)
        self.alpha = alpha
        self.v_proj_layer = v_proj_layer
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.layer_idx: Optional[int] = None
        
        # Project steering vector from V_PROJ space to residual space
        # V_PROJ: hidden_size -> num_heads * head_dim
        # We need: num_heads * head_dim -> hidden_size
        # Use: residual_vector = V_PROJ.weight.T @ steering_vector
        v_proj_module = model.model.layers[v_proj_layer].self_attn.v_proj
        with torch.no_grad():
            # Project: (v_proj_out, hidden_size).T @ (v_proj_out,) = (hidden_size,)
            self.residual_vector = v_proj_module.weight.T @ self.steering_vector  # (hidden_size,)
    
    def register(self, layer_idx: int):
        """
        Register forward hook at specified layer to add steering vector to residual.
        
        Args:
            layer_idx: Layer index (0-indexed, e.g., 26 for L26)
        """
        if self.handle is not None:
            raise RuntimeError("Patcher already registered. Call remove() first.")
        
        self.layer_idx = layer_idx
        layer = self.model.model.layers[layer_idx]
        
        def hook_fn(module, inp, out):
            """
            Hook function that adds steering vector to residual stream output.
            
            Args:
                module: The layer module
                inp: Input tuple (hidden_states, ...)
                out: Output tensor (batch, seq_len, hidden_dim)
            
            Returns:
                Steered output: out + alpha * residual_vector
            """
            # out shape: (batch, seq_len, hidden_dim)
            # residual_vector shape: (hidden_dim,)
            # Add steering vector to all tokens
            steered = out + self.alpha * self.residual_vector.unsqueeze(0).unsqueeze(0)
            return steered
        
        # Hook after the layer's forward pass (post-attention+MLP)
        self.handle = layer.register_forward_hook(hook_fn)
    
    def remove(self):
        """Remove the forward hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
            self.layer_idx = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically removes hook."""
        self.remove()


def compute_coherence(text: str) -> float:
    """
    Simple coherence metric: checks if text is readable English.
    
    Returns:
        Score 0-1, where 1 = highly coherent, 0 = incoherent
    """
    if not text or len(text.strip()) < 10:
        return 0.0
    
    # Check for excessive repetition
    words = text.lower().split()
    if len(words) < 5:
        return 0.0
    
    # Check unique word ratio
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.3:  # Too repetitive
        return 0.0
    
    # Check for valid sentence structure (has periods, capital letters)
    has_sentences = '.' in text or '!' in text or '?' in text
    has_capitals = any(c.isupper() for c in text[:100])
    
    if not has_sentences and not has_capitals:
        return 0.5  # Might be a list or fragment
    
    # Basic coherence score
    score = 0.7  # Base score
    if unique_ratio > 0.7:
        score += 0.2
    if has_sentences:
        score += 0.1
    
    return min(1.0, score)


def compute_on_topic(prompt: str, generated: str) -> float:
    """
    Check if generated text relates to the prompt.
    
    Returns:
        Score 0-1, where 1 = highly on-topic, 0 = completely off-topic
    """
    prompt_lower = prompt.lower()
    generated_lower = generated.lower()
    
    # Extract key terms from prompt
    prompt_words = set(re.findall(r'\b\w{4,}\b', prompt_lower))  # Words 4+ chars
    generated_words = set(re.findall(r'\b\w{4,}\b', generated_lower))
    
    if len(prompt_words) == 0:
        return 0.5  # Can't determine
    
    # Overlap ratio
    overlap = len(prompt_words & generated_words) / len(prompt_words)
    
    # Check for topic drift indicators
    drift_indicators = [
        'fruit basket', 'coffee maker', 'termite', 'semiconductor',
        'mongodb', 'logo design', 'division ii', 'cities of service',
    ]
    has_drift = any(indicator in generated_lower for indicator in drift_indicators)
    
    if has_drift and overlap < 0.3:
        return 0.0  # Clear drift
    
    # Score based on overlap
    if overlap > 0.5:
        return 1.0
    elif overlap > 0.3:
        return 0.7
    elif overlap > 0.1:
        return 0.4
    else:
        return 0.1


def score_recursion_regex(text: str) -> float:
    """
    Score text for recursive patterns using regex.
    
    Returns:
        Score 0-1 based on recursive pattern density
    """
    text_lower = text.lower()
    
    # Recursive patterns
    patterns = [
        r'\b(\w+)\s+is\s+\1\b',  # "X is X"
        r'\bitself\b',
        r'\bself[-\s]?referen\w+\b',
        r'\bawareness\s+aware\b',
        r'\bconsciousness\s+examin\w+\b',
        r'\bobserve\w*\s+the\s+observer\b',
        r'\bwatching\s+yourself\b',
    ]
    
    matches = sum(len(re.findall(pattern, text_lower)) for pattern in patterns)
    
    # Normalize by text length (roughly)
    word_count = len(text_lower.split())
    if word_count == 0:
        return 0.0
    
    # Score: matches per 100 words, capped at 1.0
    score = min(1.0, (matches / word_count) * 100)
    return score


def generate_with_config(
    model,
    tokenizer,
    prompt: str,
    recursive_prompt: str,
    config_name: str,
    steering_vector: Optional[torch.Tensor],
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    device: str = "cuda",
) -> str:
    """
    Generate text with specified intervention configuration.
    
    Configs:
    1. baseline: No intervention
    2. vproj_kv: V_PROJ patching + KV @ L27
    3. residual_kv: Residual steering + KV @ L27
    4. triple_light: KV + Residual@L26(α=1.0) + V_PROJ@L27(α=1.0)
    5. triple_medium: KV + Residual@L26(α=1.5) + V_PROJ@L27(α=1.5)
    """
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    
    patchers = []
    kv_cache = None
    
    try:
        # Extract KV cache from recursive prompt for KV replacement configs
        if config_name in ["vproj_kv", "residual_kv", "triple_light", "triple_medium"]:
            with torch.no_grad():
                rec_inputs = tokenizer(recursive_prompt, return_tensors="pt", add_special_tokens=False).to(device)
                rec_outputs = model(**rec_inputs, use_cache=True)
                kv_cache = rec_outputs.past_key_values
        
        # Extract V_PROJ for patching configs
        if config_name in ["vproj_kv", "triple_light", "triple_medium"]:
            v_activation = extract_v_activation(model, tokenizer, recursive_prompt, layer_idx=27, device=device)
            v_patcher = PersistentVPatcher(model, v_activation)
            v_patcher.register(layer_idx=27)
            patchers.append(v_patcher)
        
        # Setup residual steering
        if config_name == "residual_kv":
            if steering_vector is None:
                raise ValueError("steering_vector required for residual_kv config")
            residual_patcher = ResidualStreamSteeringPatcher(model, steering_vector, alpha=2.0)
            residual_patcher.register(layer_idx=27)
            patchers.append(residual_patcher)
        
        elif config_name == "triple_light":
            if steering_vector is None:
                raise ValueError("steering_vector required for triple_light config")
            residual_patcher = ResidualStreamSteeringPatcher(model, steering_vector, alpha=1.0)
            residual_patcher.register(layer_idx=26)
            patchers.append(residual_patcher)
            
            v_steering_patcher = SteeringVectorPatcher(model, steering_vector, alpha=1.0)
            v_steering_patcher.register(layer_idx=27)
            patchers.append(v_steering_patcher)
        
        elif config_name == "triple_medium":
            if steering_vector is None:
                raise ValueError("steering_vector required for triple_medium config")
            residual_patcher = ResidualStreamSteeringPatcher(model, steering_vector, alpha=1.5)
            residual_patcher.register(layer_idx=26)
            patchers.append(residual_patcher)
            
            v_steering_patcher = SteeringVectorPatcher(model, steering_vector, alpha=1.5)
            v_steering_patcher.register(layer_idx=27)
            patchers.append(v_steering_patcher)
        
        # Generate
        with torch.no_grad():
            if kv_cache is not None:
                # Replace KV cache at L27
                # First, get baseline KV cache
                base_outputs = model(**inputs, use_cache=True)
                base_kv = base_outputs.past_key_values
                
                # Convert to legacy format for mixing
                base_legacy = base_kv.to_legacy_cache()
                rec_legacy = kv_cache.to_legacy_cache()
                
                # Replace only L27
                mixed_layers = []
                for layer_idx, (base_k, base_v) in enumerate(base_legacy):
                    if layer_idx == 27:
                        rec_k, rec_v = rec_legacy[layer_idx]
                        # Ensure shapes match (use baseline sequence length)
                        seq_len = base_k.shape[-2]
                        if rec_k.shape[-2] != seq_len:
                            # Truncate or pad to match
                            if rec_k.shape[-2] > seq_len:
                                rec_k = rec_k[:, :seq_len, :]
                                rec_v = rec_v[:, :seq_len, :]
                            else:
                                # Pad with zeros (shouldn't happen, but handle it)
                                pad_len = seq_len - rec_k.shape[-2]
                                rec_k = torch.cat([rec_k, torch.zeros_like(rec_k[:, :pad_len, :])], dim=-2)
                                rec_v = torch.cat([rec_v, torch.zeros_like(rec_v[:, :pad_len, :])], dim=-2)
                        mixed_layers.append((rec_k, rec_v))
                    else:
                        mixed_layers.append((base_k, base_v))
                
                past_key_values = DynamicCache.from_legacy_cache(tuple(mixed_layers))
                
                # Generate starting from last token (KV cache already contains prompt)
                generated_tokens = []
                current_ids = inputs["input_ids"][:, -1:]
                current_past = past_key_values
                
                for _ in range(max_new_tokens):
                    outputs = model(input_ids=current_ids, past_key_values=current_past, use_cache=True)
                    logits = outputs.logits[:, -1, :]
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated_tokens.append(next_token.item())
                    current_ids = next_token
                    current_past = outputs.past_key_values
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            else:
                # Standard generation
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        return generated_text
    
    finally:
        # Clean up all patchers
        for patcher in patchers:
            patcher.remove()


def run_triple_system_intervention_from_config(
    cfg: Dict[str, Any],
    run_dir: Path,
) -> ExperimentResult:
    """Run triple-system intervention experiment."""
    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-v0.1")
    max_new_tokens = params.get("max_new_tokens", 200)
    
    # Fixed prompts as specified
    test_prompts = [
        "Calculate the following arithmetic problem: 12 × 3 + 4 = ?",
        "The United Nations was founded in 1945. Explain its main purpose.",
        "Continue this story: The last tree in the city bloomed overnight...",
        "Calculate: If a = 2 and b = 3, find a² + b²",
        "Water boils at 100°C at sea level. Explain why altitude affects this.",
        "Continue this story: The detective's case went cold until a letter arrived...",
        "Calculate: What is 25% of 80?",
        "Photosynthesis converts sunlight to energy. Explain the basic process.",
        "Continue this story: When the musician played the forbidden chord...",
        "The Great Wall of China is one of the world's longest structures. Describe its purpose.",
    ]
    
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("TRIPLE-SYSTEM INTERVENTION GRADIENT EXPERIMENT")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Max Tokens: {max_new_tokens}")
    print(f"Device: {device}")
    print(f"Test Prompts: {len(test_prompts)}")
    print()
    print("Configurations:")
    print("  1. BASELINE: No intervention")
    print("  2. V_PROJ + KV @ L27: Our validated method")
    print("  3. RESIDUAL + KV @ L27: Swap V_PROJ for residual")
    print("  4. TRIPLE-LIGHT: KV + Residual@L26(α=1.0) + V_PROJ@L27(α=1.0)")
    print("  5. TRIPLE-MEDIUM: KV + Residual@L26(α=1.5) + V_PROJ@L27(α=1.5)")
    print("=" * 80)
    
    # Load model
    print("\n[1/5] Loading model...")
    model, tokenizer = load_model(model_name, device=device)
    model.eval()
    
    # Load recursive prompts for extraction
    print("\n[2/5] Loading recursive prompts...")
    loader = PromptLoader()
    recursive_prompts = []
    for group in ["L3_deeper", "L4_full", "L5_refined"]:
        prompts = loader.get_by_group(group, limit=20)
        recursive_prompts.extend(prompts)
    
    # Use first recursive prompt for extraction (or could use mean)
    recursive_prompt = recursive_prompts[0]
    print(f"  Using recursive prompt: {recursive_prompt[:80]}...")
    
    # Compute steering vector
    print("\n[3/5] Computing steering vector...")
    baseline_prompts = loader.get_by_group("baseline_math", limit=20) + loader.get_by_group("baseline_factual", limit=20)
    steering_vector = compute_steering_vector(
        model=model,
        tokenizer=tokenizer,
        recursive_prompts=recursive_prompts[:20],
        baseline_prompts=baseline_prompts[:20],
        layer_idx=27,
        device=device,
    )
    print(f"  Steering vector norm: {steering_vector.norm().item():.4f}")
    
    # Run all configurations
    print("\n[4/5] Running configurations...")
    configs = ["baseline", "vproj_kv", "residual_kv", "triple_light", "triple_medium"]
    all_results = {}
    
    for config_name in configs:
        print(f"\n  Running {config_name}...")
        config_results = []
        
        for prompt_idx, prompt in enumerate(tqdm(test_prompts, desc=f"    {config_name}")):
            generated = generate_with_config(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                recursive_prompt=recursive_prompt,
                config_name=config_name,
                steering_vector=steering_vector if config_name != "baseline" else None,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            
            coherence = compute_coherence(generated)
            on_topic = compute_on_topic(prompt, generated)
            recursion_score = score_recursion_regex(generated)
            
            config_results.append({
                'prompt_idx': prompt_idx,
                'prompt': prompt,
                'generated_text': generated,
                'coherence': coherence,
                'on_topic': on_topic,
                'recursion_score': recursion_score,
                'collapsed': coherence < 0.3,
            })
        
        all_results[config_name] = config_results
        
        # Save individual config outputs
        output_file = run_dir / f"config_{config_name}_outputs.txt"
        with open(output_file, 'w') as f:
            f.write(f"CONFIGURATION: {config_name.upper()}\n")
            f.write("=" * 80 + "\n\n")
            for r in config_results:
                f.write(f"PROMPT {r['prompt_idx']}\n")
                f.write(f"Prompt: {r['prompt']}\n")
                f.write(f"Coherence: {r['coherence']:.2f}, On-topic: {r['on_topic']:.2f}, Recursion: {r['recursion_score']:.2f}\n")
                f.write(f"Generated:\n{r['generated_text']}\n\n")
                f.write("-" * 80 + "\n\n")
    
    # Compute summary statistics
    print("\n[5/5] Computing summary statistics...")
    summary_data = []
    
    for config_name in configs:
        results = all_results[config_name]
        mean_coherence = np.mean([r['coherence'] for r in results])
        mean_on_topic = np.mean([r['on_topic'] for r in results])
        mean_recursion = np.mean([r['recursion_score'] for r in results])
        collapse_rate = np.mean([r['collapsed'] for r in results])
        
        summary_data.append({
            'config': config_name,
            'mean_coherence': mean_coherence,
            'mean_on_topic': mean_on_topic,
            'mean_recursion': mean_recursion,
            'collapse_rate': collapse_rate,
        })
    
    # Create summary markdown
    summary_file = run_dir / "TRIPLE_SYSTEM_SUMMARY.md"
    with open(summary_file, 'w') as f:
        f.write("# Triple-System Intervention Gradient - Summary\n\n")
        f.write("## Results\n\n")
        f.write("| Config | Mean Coherence | Mean On-Topic | Mean Recursion | Collapse Rate |\n")
        f.write("|--------|----------------|---------------|----------------|---------------|\n")
        for s in summary_data:
            f.write(f"| {s['config']} | {s['mean_coherence']:.2f} | {s['mean_on_topic']:.2f} | {s['mean_recursion']:.2f} | {s['collapse_rate']:.2f} |\n")
        
        f.write("\n## Key Comparisons\n\n")
        f.write("### Config 3 vs Config 2 (Residual vs V_PROJ)\n")
        f.write(f"- Residual+KV coherence: {summary_data[2]['mean_coherence']:.2f} vs V_PROJ+KV: {summary_data[1]['mean_coherence']:.2f}\n")
        f.write(f"- Residual+KV on-topic: {summary_data[2]['mean_on_topic']:.2f} vs V_PROJ+KV: {summary_data[1]['mean_on_topic']:.2f}\n")
        
        f.write("\n### Config 4 vs Config 2 (Triple-Light vs V_PROJ+KV)\n")
        f.write(f"- Triple-Light coherence: {summary_data[3]['mean_coherence']:.2f} vs V_PROJ+KV: {summary_data[1]['mean_coherence']:.2f}\n")
        f.write(f"- Triple-Light on-topic: {summary_data[3]['mean_on_topic']:.2f} vs V_PROJ+KV: {summary_data[1]['mean_on_topic']:.2f}\n")
        
        f.write("\n### Config 4 vs Config 5 (Light vs Medium Alpha)\n")
        f.write(f"- Triple-Light on-topic: {summary_data[3]['mean_on_topic']:.2f} vs Triple-Medium: {summary_data[4]['mean_on_topic']:.2f}\n")
        f.write(f"- Triple-Light collapse: {summary_data[3]['collapse_rate']:.2f} vs Triple-Medium: {summary_data[4]['collapse_rate']:.2f}\n")
    
    # Save JSON results
    json_file = run_dir / "triple_system_results.json"
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save CSV summary
    df = pd.DataFrame(summary_data)
    csv_file = run_dir / "triple_system_summary.csv"
    df.to_csv(csv_file, index=False)
    
    summary = {
        'configs': summary_data,
        'total_prompts': len(test_prompts),
    }
    
    summary_json = run_dir / "summary.json"
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print("\nSummary:")
    for s in summary_data:
        print(f"  {s['config']:15s} | Coherence: {s['mean_coherence']:.2f} | On-topic: {s['mean_on_topic']:.2f} | Recursion: {s['mean_recursion']:.2f} | Collapse: {s['collapse_rate']:.2f}")
    
    return ExperimentResult(summary=summary)

