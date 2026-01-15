"""
Pipeline: Surgical-Causal Configuration + Parameter Sweep

Tests head-specific steering, split-brain KV cache, and cascade residual steering
based on theoretical consensus recommendations.
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
from src.core.head_specific_patching import HeadSpecificSteeringPatcher
from src.core.patching import extract_v_activation, extract_residual_activation
from src.pipelines.archive.steering import compute_steering_vector, SteeringVectorPatcher
from src.pipelines.registry import ExperimentResult


class CascadeResidualSteeringPatcher:
    """
    Applies steering vector to residual stream at multiple layers with different alphas.
    
    Usage:
        patcher = CascadeResidualSteeringPatcher(model, steering_vector, {24: 0.3, 26: 0.6})
        patcher.register()
        # Generate...
        patcher.remove()
    """
    
    def __init__(
        self,
        model,
        steering_vector: torch.Tensor,
        layer_alphas: Dict[int, float],
        v_proj_layer: int = 27,
    ):
        """
        Initialize cascade patcher.
        
        Args:
            model: The transformer model
            steering_vector: Vector from V_PROJ (shape: v_proj_out_features, typically 1024)
            layer_alphas: Dict mapping layer_idx -> alpha value
            v_proj_layer: Layer to extract V_PROJ weight matrix from (for projection)
        """
        self.model = model
        self.steering_vector = steering_vector.detach().to(model.device)
        self.layer_alphas = layer_alphas
        self.v_proj_layer = v_proj_layer
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        
        # Project steering vector from V_PROJ space to residual space
        v_proj_module = model.model.layers[v_proj_layer].self_attn.v_proj
        with torch.no_grad():
            self.residual_vector = v_proj_module.weight.T @ self.steering_vector  # (hidden_size,)
    
    def register(self):
        """Register forward hooks for all specified layers."""
        if self.handles:
            raise RuntimeError("Patcher already registered. Call remove() first.")
        
        for layer_idx, alpha in self.layer_alphas.items():
            layer = self.model.model.layers[layer_idx]
            
            def make_hook(alpha_val):
                def hook_fn(module, inp, out):
                    steered = out + alpha_val * self.residual_vector.unsqueeze(0).unsqueeze(0)
                    return steered
                return hook_fn
            
            handle = layer.register_forward_hook(make_hook(alpha))
            self.handles.append(handle)
    
    def remove(self):
        """Remove all forward hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.remove()


class SplitBrainKVPatcher:
    """
    Modifies attention to use different KV cache for different heads.
    
    For causal heads (H18, H26): Use recursive KV (blended with baseline)
    For content heads (all others): Use baseline KV
    
    This requires intercepting the attention forward pass.
    """
    
    def __init__(
        self,
        model,
        baseline_kv: DynamicCache,
        recursive_kv: DynamicCache,
        causal_heads: List[int] = [18, 26],
        blend_ratio: float = 0.8,
        target_layer: int = 27,
    ):
        """
        Initialize split-brain KV patcher.
        
        Args:
            model: The transformer model
            baseline_kv: KV cache from baseline prompt
            recursive_kv: KV cache from recursive prompt
            causal_heads: List of head indices that should see recursive KV
            blend_ratio: Ratio of recursive KV for causal heads (0.8 = 80% recursive, 20% baseline)
            target_layer: Layer to apply split-brain KV at
        """
        self.model = model
        self.baseline_kv = baseline_kv
        self.recursive_kv = recursive_kv
        self.causal_heads = causal_heads
        self.blend_ratio = blend_ratio
        self.target_layer = target_layer
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        
        # Convert to legacy format for easier manipulation
        self.baseline_legacy = baseline_kv.to_legacy_cache()
        self.recursive_legacy = recursive_kv.to_legacy_cache()
    
    def register(self):
        """Register forward hook to intercept attention and modify KV."""
        if self.handle is not None:
            raise RuntimeError("Patcher already registered. Call remove() first.")
        
        layer = self.model.model.layers[self.target_layer].self_attn
        
        def hook_fn(module, inp, out):
            """
            Intercept attention output and modify KV cache used.
            
            Note: This is a simplified approach. In practice, we need to modify
            the attention forward pass itself, which is more complex.
            
            For now, we'll modify the past_key_values that get passed to next steps.
            """
            # This hook runs after attention, so we can't modify KV here directly.
            # Instead, we'll need to modify the model's past_key_values during generation.
            # This is a limitation - we'll handle it in the generation function.
            return out
        
        # Actually, we need to hook into the attention forward pass itself.
        # Let's use a pre-hook to modify the inputs.
        def pre_hook_fn(module, inp):
            """
            Pre-hook to modify KV cache before attention computation.
            
            inp[0] = hidden_states
            inp[1] = attention_mask (optional)
            inp[2] = position_ids (optional)
            inp[3] = past_key_value (optional)
            inp[4] = output_attentions (optional)
            inp[5] = use_cache (optional)
            """
            # We can't easily modify past_key_value here because it's passed
            # through the model's forward pass. We'll handle this differently.
            return inp
        
        # For now, we'll mark that split-brain is active and handle it in generation
        self.handle = layer.register_forward_pre_hook(pre_hook_fn)
    
    def remove(self):
        """Remove the forward hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.remove()
    
    def create_split_brain_kv(self, base_kv: DynamicCache) -> DynamicCache:
        """
        Create split-brain KV cache by blending recursive and baseline KV per-head.
        
        Args:
            base_kv: Baseline KV cache (will be modified)
        
        Returns:
            Modified KV cache with split-brain configuration
        """
        base_legacy = base_kv.to_legacy_cache()
        rec_legacy = self.recursive_legacy
        
        # For the target layer, blend KV per-head
        target_k, target_v = base_legacy[self.target_layer]
        rec_k, rec_v = rec_legacy[self.target_layer]
        
        # KV shape: (batch, num_heads, seq_len, head_dim)
        batch, num_heads, seq_len_base, head_dim = target_k.shape
        _, _, seq_len_rec, _ = rec_k.shape
        
        # Ensure sequence lengths match (take minimum)
        min_seq_len = min(seq_len_base, seq_len_rec)
        
        if min_seq_len == 0:
            raise ValueError(f"Empty KV cache: base_seq_len={seq_len_base}, rec_seq_len={seq_len_rec}")
        
        # Create blended KV
        blended_k = target_k[:, :, :min_seq_len, :].clone()
        blended_v = target_v[:, :, :min_seq_len, :].clone()
        rec_k_slice = rec_k[:, :, :min_seq_len, :]
        rec_v_slice = rec_v[:, :, :min_seq_len, :]
        
        for head_idx in range(num_heads):
            if head_idx in self.causal_heads:
                # Causal heads: blend recursive and baseline
                blended_k[:, head_idx, :, :] = (
                    self.blend_ratio * rec_k_slice[:, head_idx, :, :] +
                    (1 - self.blend_ratio) * blended_k[:, head_idx, :, :]
                )
                blended_v[:, head_idx, :, :] = (
                    self.blend_ratio * rec_v_slice[:, head_idx, :, :] +
                    (1 - self.blend_ratio) * blended_v[:, head_idx, :, :]
                )
            # else: keep baseline KV (already cloned)
        
        # Reconstruct legacy cache with modified layer
        mixed_layers = []
        for layer_idx, (k, v) in enumerate(base_legacy):
            if layer_idx == self.target_layer:
                mixed_layers.append((blended_k, blended_v))
            else:
                # For other layers, also trim to min_seq_len if needed
                if k.shape[2] > min_seq_len:
                    mixed_layers.append((k[:, :, :min_seq_len, :], v[:, :, :min_seq_len, :]))
                else:
                    mixed_layers.append((k, v))
        
        return DynamicCache.from_legacy_cache(tuple(mixed_layers))


def compute_coherence(text: str) -> float:
    """Simple coherence metric."""
    if not text or len(text.strip()) < 10:
        return 0.0
    
    words = text.lower().split()
    if len(words) < 5:
        return 0.0
    
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.3:
        return 0.0
    
    has_sentences = '.' in text or '!' in text or '?' in text
    has_capitals = any(c.isupper() for c in text[:100])
    
    if not has_sentences and not has_capitals:
        return 0.5
    
    score = 0.7
    if unique_ratio > 0.7:
        score += 0.2
    if has_sentences:
        score += 0.1
    
    return min(1.0, score)


def compute_on_topic(prompt: str, generated: str) -> float:
    """Check if generated text relates to the prompt."""
    prompt_lower = prompt.lower()
    generated_lower = generated.lower()
    
    prompt_words = set(re.findall(r'\b\w{4,}\b', prompt_lower))
    generated_words = set(re.findall(r'\b\w{4,}\b', generated_lower))
    
    if len(prompt_words) == 0:
        return 0.5
    
    overlap = len(prompt_words & generated_words) / len(prompt_words)
    
    drift_indicators = [
        'fruit basket', 'coffee maker', 'termite', 'semiconductor',
        'mongodb', 'logo design', 'division ii', 'cities of service',
    ]
    has_drift = any(indicator in generated_lower for indicator in drift_indicators)
    
    if has_drift and overlap < 0.3:
        return 0.0
    
    if overlap > 0.5:
        return 1.0
    elif overlap > 0.3:
        return 0.7
    elif overlap > 0.1:
        return 0.4
    else:
        return 0.1


def score_recursion_regex(text: str) -> float:
    """Score text for recursive patterns using regex."""
    text_lower = text.lower()
    
    patterns = [
        r'\b(\w+)\s+is\s+\1\b',
        r'\bitself\b',
        r'\bself[-\s]?referen\w+\b',
        r'\bawareness\s+aware\b',
        r'\bconsciousness\s+examin\w+\b',
        r'\bobserve\w*\s+the\s+observer\b',
        r'\bwatching\s+yourself\b',
    ]
    
    matches = sum(len(re.findall(pattern, text_lower)) for pattern in patterns)
    word_count = len(text_lower.split())
    
    if word_count == 0:
        return 0.0
    
    score = min(1.0, (matches / word_count) * 100)
    return score


def generate_with_config(
    model,
    tokenizer,
    prompt: str,
    recursive_prompt: str,
    config: Dict[str, Any],
    steering_vector: Optional[torch.Tensor],
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    device: str = "cuda",
) -> str:
    """
    Generate text with specified surgical configuration.
    
    Config keys:
    - head_target: "full", "h18", "h26", "h18_h26"
    - kv_strategy: "none", "full", "split_brain", "interpolated"
    - residual_alphas: Dict[int, float] or None
    - vproj_alpha: float
    """
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    
    patchers = []
    split_brain_kv_patcher = None
    
    try:
        # Extract KV caches
        baseline_kv = None
        recursive_kv = None
        if config["kv_strategy"] != "none":
            with torch.no_grad():
                rec_inputs = tokenizer(recursive_prompt, return_tensors="pt", add_special_tokens=False).to(device)
                rec_outputs = model(**rec_inputs, use_cache=True)
                recursive_kv = rec_outputs.past_key_values
                
                base_outputs = model(**inputs, use_cache=True)
                baseline_kv = base_outputs.past_key_values
        
        # Setup residual steering (cascade)
        if config.get("residual_alphas"):
            cascade_patcher = CascadeResidualSteeringPatcher(
                model, steering_vector, config["residual_alphas"]
            )
            cascade_patcher.register()
            patchers.append(cascade_patcher)
        
        # Setup V_PROJ steering (head-specific or full)
        if config.get("vproj_alpha", 0) > 0:
            head_target = config.get("head_target", "full")
            
            if head_target == "full":
                v_steering_patcher = SteeringVectorPatcher(
                    model, steering_vector, config["vproj_alpha"]
                )
                v_steering_patcher.register(layer_idx=27)
                patchers.append(v_steering_patcher)
            else:
                # Head-specific steering
                target_heads = []
                if "h18" in head_target:
                    target_heads.append(18)
                if "h26" in head_target:
                    target_heads.append(26)
                
                if target_heads:
                    v_steering_patcher = HeadSpecificSteeringPatcher(
                        model, steering_vector, target_heads, config["vproj_alpha"]
                    )
                    v_steering_patcher.register(layer_idx=27)
                    patchers.append(v_steering_patcher)
        
        # Setup KV cache replacement
        kv_cache_to_use = None
        if config["kv_strategy"] == "full":
            kv_cache_to_use = recursive_kv
        elif config["kv_strategy"] == "split_brain":
            # Create split-brain KV
            # Check sequence lengths match first
            base_legacy = baseline_kv.to_legacy_cache()
            rec_legacy = recursive_kv.to_legacy_cache()
            base_seq_len = base_legacy[27][0].shape[2]
            rec_seq_len = rec_legacy[27][0].shape[2]
            
            if base_seq_len != rec_seq_len:
                # Use baseline KV only if lengths don't match (fallback)
                print(f"  WARNING: Sequence length mismatch (base={base_seq_len}, rec={rec_seq_len}). Using baseline KV only.")
                kv_cache_to_use = baseline_kv
            else:
                split_brain_patcher = SplitBrainKVPatcher(
                    model, baseline_kv, recursive_kv,
                    causal_heads=config.get("causal_heads", [18, 26]),
                    blend_ratio=config.get("blend_ratio", 0.8),
                    target_layer=27,
                )
                kv_cache_to_use = split_brain_patcher.create_split_brain_kv(baseline_kv)
        elif config["kv_strategy"] == "interpolated":
            # Interpolate all heads
            base_legacy = baseline_kv.to_legacy_cache()
            rec_legacy = recursive_kv.to_legacy_cache()
            blend_ratio = config.get("blend_ratio", 0.7)
            
            mixed_layers = []
            for layer_idx, (base_k, base_v) in enumerate(base_legacy):
                if layer_idx == 27:
                    rec_k, rec_v = rec_legacy[layer_idx]
                    mixed_k = blend_ratio * rec_k + (1 - blend_ratio) * base_k
                    mixed_v = blend_ratio * rec_v + (1 - blend_ratio) * base_v
                    mixed_layers.append((mixed_k, mixed_v))
                else:
                    mixed_layers.append((base_k, base_v))
            
            kv_cache_to_use = DynamicCache.from_legacy_cache(tuple(mixed_layers))
        
        # Generate
        with torch.no_grad():
            if kv_cache_to_use is not None:
                # When using past_key_values, we need to generate token-by-token
                # because the KV cache already contains the prompt
                current_ids = inputs["input_ids"][:, -1:]  # Only last token
                current_past = kv_cache_to_use
                generated_ids = inputs["input_ids"].clone()
                
                for _ in range(max_new_tokens):
                    outputs = model(input_ids=current_ids, past_key_values=current_past, use_cache=True)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.multinomial(
                        torch.softmax(next_token_logits / temperature, dim=-1), 1
                    )
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                    
                    current_ids = next_token
                    current_past = outputs.past_key_values
                
                outputs = generated_ids
            else:
                # No KV replacement - normal generation
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    finally:
        # Clean up all patchers
        for patcher in patchers:
            patcher.remove()


def run_surgical_sweep_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """
    Run surgical-causal configuration sweep.
    
    Config params:
    - model: Model name
    - n_baseline_prompts: Number of baseline prompts to test
    - n_recursive_prompts: Number of recursive prompts for steering vector
    - max_new_tokens: Tokens to generate
    - temperature: Generation temperature
    - run_priority_1_only: If True, only run Priority 1 configs (7 configs)
    """
    print("=" * 80)
    print("SURGICAL-CAUSAL CONFIGURATION SWEEP")
    print("=" * 80)
    
    model_name = cfg["params"]["model"]
    n_baseline = cfg["params"]["n_baseline_prompts"]
    n_recursive = cfg["params"]["n_recursive_prompts"]
    max_new_tokens = cfg["params"]["max_new_tokens"]
    temperature = cfg["params"]["temperature"]
    run_priority_1_only = cfg["params"].get("run_priority_1_only", False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model, tokenizer = load_model(model_name, device=device)
    model.eval()
    
    # Load prompts
    loader = PromptLoader()
    
    # Use specific prompts if provided, otherwise load from loader
    if "prompts" in cfg["params"]:
        baseline_prompts = cfg["params"]["prompts"]
    else:
        baseline_prompts = loader.get_by_group("baseline", limit=n_baseline)
    
    # Always load recursive prompts from loader (needed for steering vector)
    recursive_prompts = []
    for group in ["L3_deeper", "L4_full", "L5_refined"]:
        prompts = loader.get_by_group(group, limit=n_recursive // 3 + 1)
        recursive_prompts.extend(prompts)
    recursive_prompts = recursive_prompts[:n_recursive]
    
    print(f"Loaded {len(baseline_prompts)} baseline prompts")
    print(f"Loaded {len(recursive_prompts)} recursive prompts")
    
    # Compute steering vector
    print("\nComputing steering vector...")
    steering_vector = compute_steering_vector(
        model, tokenizer, recursive_prompts, baseline_prompts[:n_recursive],
        layer_idx=27, device=device
    )
    print(f"Steering vector shape: {steering_vector.shape}")
    print(f"Steering vector norm: {steering_vector.norm().item():.4f}")
    
    # Define configurations
    configs = {}
    
    # PRIORITY 1 - Core Hypothesis Tests
    configs["A1"] = {
        "name": "A1_SplitBrain_Surgical",
        "head_target": "h18_h26",
        "kv_strategy": "split_brain",
        "residual_alphas": {24: 0.3, 26: 0.6},
        "vproj_alpha": 2.5,
        "causal_heads": [18, 26],
        "blend_ratio": 0.8,
    }
    
    configs["B1"] = {
        "name": "B1_Full_4096_SplitBrain",
        "head_target": "full",
        "kv_strategy": "split_brain",
        "residual_alphas": {26: 0.6},
        "vproj_alpha": 1.5,
        "causal_heads": [18, 26],
        "blend_ratio": 0.8,
    }
    
    configs["B2"] = {
        "name": "B2_H18_Only_SplitBrain",
        "head_target": "h18",
        "kv_strategy": "split_brain",
        "residual_alphas": {26: 0.6},
        "vproj_alpha": 2.5,
        "causal_heads": [18, 26],
        "blend_ratio": 0.8,
    }
    
    configs["B3"] = {
        "name": "B3_H26_Only_SplitBrain",
        "head_target": "h26",
        "kv_strategy": "split_brain",
        "residual_alphas": {26: 0.6},
        "vproj_alpha": 2.5,
        "causal_heads": [18, 26],
        "blend_ratio": 0.8,
    }
    
    configs["C1"] = {
        "name": "C1_H18_H26_NoKV",
        "head_target": "h18_h26",
        "kv_strategy": "none",
        "residual_alphas": {26: 0.6},
        "vproj_alpha": 2.5,
    }
    
    configs["C2"] = {
        "name": "C2_H18_H26_FullKV",
        "head_target": "h18_h26",
        "kv_strategy": "full",
        "residual_alphas": {26: 0.6},
        "vproj_alpha": 2.5,
    }
    
    configs["C4"] = {
        "name": "C4_H18_H26_InterpolatedKV",
        "head_target": "h18_h26",
        "kv_strategy": "interpolated",
        "residual_alphas": {26: 0.6},
        "vproj_alpha": 2.5,
        "blend_ratio": 0.7,
    }
    
    # PRIORITY 2 - Refinement (if time permits)
    if not run_priority_1_only:
        configs["D1"] = {
            "name": "D1_Symmetric",
            "head_target": "h18_h26",
            "kv_strategy": "split_brain",
            "residual_alphas": {26: 1.0},
            "vproj_alpha": 1.0,
            "causal_heads": [18, 26],
            "blend_ratio": 0.8,
        }
        
        configs["D2"] = {
            "name": "D2_ResHeavy",
            "head_target": "h18_h26",
            "kv_strategy": "split_brain",
            "residual_alphas": {26: 1.5},
            "vproj_alpha": 0.8,
            "causal_heads": [18, 26],
            "blend_ratio": 0.8,
        }
        
        configs["D3"] = {
            "name": "D3_VprojHeavy",
            "head_target": "h18_h26",
            "kv_strategy": "split_brain",
            "residual_alphas": {26: 0.6},
            "vproj_alpha": 2.0,
            "causal_heads": [18, 26],
            "blend_ratio": 0.8,
        }
        
        configs["E1"] = {
            "name": "E1_EarlySeed_L24",
            "head_target": "h18_h26",
            "kv_strategy": "split_brain",
            "residual_alphas": {24: 0.3},
            "vproj_alpha": 2.5,
            "causal_heads": [18, 26],
            "blend_ratio": 0.8,
        }
        
        configs["E2"] = {
            "name": "E2_FullCascade",
            "head_target": "h18_h26",
            "kv_strategy": "split_brain",
            "residual_alphas": {24: 0.3, 25: 0.5, 26: 0.6},
            "vproj_alpha": 2.5,
            "causal_heads": [18, 26],
            "blend_ratio": 0.8,
        }
    
    # Run experiments
    results = []
    
    for config_id, config in tqdm(configs.items(), desc="Configurations"):
        print(f"\n{'='*80}")
        print(f"Running: {config['name']}")
        print(f"{'='*80}")
        
        for prompt_idx, prompt in enumerate(tqdm(baseline_prompts, desc=f"  {config_id}")):
            try:
                # Use first recursive prompt for KV extraction
                rec_prompt = recursive_prompts[0] if recursive_prompts else baseline_prompts[0]
                
                generated = generate_with_config(
                    model, tokenizer, prompt, rec_prompt, config,
                    steering_vector, max_new_tokens, temperature, device
                )
                
                # Compute metrics
                coherence = compute_coherence(generated)
                on_topic = compute_on_topic(prompt, generated)
                recursion_score = score_recursion_regex(generated)
                collapsed = coherence < 0.3
                
                results.append({
                    "config_id": config_id,
                    "config_name": config["name"],
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "generated_text": generated,
                    "coherence": coherence,
                    "on_topic": on_topic,
                    "recursion_score": recursion_score,
                    "collapsed": collapsed,
                })
            
            except Exception as e:
                print(f"  ERROR on prompt {prompt_idx}: {e}")
                results.append({
                    "config_id": config_id,
                    "config_name": config["name"],
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "generated_text": f"ERROR: {e}",
                    "coherence": 0.0,
                    "on_topic": 0.0,
                    "recursion_score": 0.0,
                    "collapsed": True,
                })
    
    # Save results
    df = pd.DataFrame(results)
    results_csv = run_dir / "surgical_sweep_results.csv"
    df.to_csv(results_csv, index=False)
    print(f"\nSaved results to: {results_csv}")
    
    # Save full text outputs
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    for config_id in configs.keys():
        config_outputs = df[df["config_id"] == config_id]
        output_file = outputs_dir / f"{config_id}_{configs[config_id]['name']}_outputs.txt"
        
        with open(output_file, "w") as f:
            f.write(f"CONFIGURATION: {configs[config_id]['name']}\n")
            f.write("=" * 80 + "\n\n")
            
            for _, row in config_outputs.iterrows():
                f.write(f"PROMPT {row['prompt_idx']}:\n")
                f.write(f"{row['prompt']}\n\n")
                f.write(f"GENERATED:\n{row['generated_text']}\n\n")
                f.write(f"Metrics: coherence={row['coherence']:.2f}, "
                       f"on_topic={row['on_topic']:.2f}, "
                       f"recursion={row['recursion_score']:.2f}, "
                       f"collapsed={row['collapsed']}\n")
                f.write("-" * 80 + "\n\n")
    
    # Compute summary statistics
    summary = {}
    for config_id in configs.keys():
        config_results = df[df["config_id"] == config_id]
        summary[config_id] = {
            "name": configs[config_id]["name"],
            "mean_coherence": config_results["coherence"].mean(),
            "mean_on_topic": config_results["on_topic"].mean(),
            "mean_recursion": config_results["recursion_score"].mean(),
            "collapse_rate": config_results["collapsed"].mean(),
            "n_prompts": len(config_results),
        }
    
    # Save summary
    summary_json = run_dir / "surgical_sweep_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Config':<20} {'Coherence':<12} {'On-Topic':<12} {'Recursion':<12} {'Collapse':<12}")
    print("-" * 80)
    
    for config_id in sorted(configs.keys()):
        stats = summary[config_id]
        print(f"{stats['name']:<20} {stats['mean_coherence']:<12.2f} "
              f"{stats['mean_on_topic']:<12.2f} {stats['mean_recursion']:<12.2f} "
              f"{stats['collapse_rate']:<12.2f}")
    
    summary = {
        "configs": summary,
        "total_prompts": len(baseline_prompts),
        "results_file": str(results_csv),
        "summary_file": str(summary_json),
    }
    
    return ExperimentResult(summary=summary)

