"""Minimal intervention experiments to find REAL recursive output."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.pipelines.archive.steering import (
    compute_steering_vector,
    SteeringVectorPatcher,
    _generate_with_steering,
)
from src.core.patching import PersistentVPatcher, extract_v_activation
from src.core.head_specific_patching import HeadSpecificVPatcher, HeadSpecificSteeringPatcher
from src.metrics.behavior_strict import score_behavior_strict
from src.pipelines.registry import ExperimentResult
from transformers import DynamicCache


class MinimalInterventionGenerator:
    """Generate text with various intervention combinations."""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def generate_with_intervention(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        # Steering parameters
        steering_vector: Optional[torch.Tensor] = None,
        steering_layer: Optional[int] = None,
        steering_alpha: float = 1.0,
        # KV cache parameters
        kv_cache: Optional[DynamicCache] = None,
        kv_layers: Optional[List[int]] = None,
        # V_PROJ patching parameters
        v_proj_patch: Optional[torch.Tensor] = None,
        v_proj_layers: Optional[List[int]] = None,
        v_proj_window: int = 16,
        # Head-specific patching parameters
        head_specific_v_patch: Optional[torch.Tensor] = None,
        head_specific_heads: Optional[List[int]] = None,
        head_specific_steering: Optional[torch.Tensor] = None,
        head_specific_steering_alpha: float = 1.0,
        # Generation-time steering
        persistent_steering: bool = True,
        steering_first_n: Optional[int] = None,
    ) -> Tuple[str, float]:
        """Generate with specified interventions."""
        # Tokenize baseline prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        
        # Setup KV cache - if kv_layers specified, only replace those layers
        past_key_values = None
        if kv_cache is not None:
            if kv_layers is None:
                # Use full KV cache
                past_key_values = kv_cache
            else:
                # Replace only specified layers
                # First, get baseline KV cache
                with torch.no_grad():
                    base_outputs = self.model(**inputs, use_cache=True)
                    base_kv = base_outputs.past_key_values
                
                # Convert to legacy format for manipulation
                if hasattr(base_kv, "to_legacy_cache"):
                    base_legacy = base_kv.to_legacy_cache()
                    rec_legacy = kv_cache.to_legacy_cache()
                else:
                    base_legacy = base_kv
                    rec_legacy = kv_cache
                
                # Replace only specified layers
                mixed_layers = []
                for layer_idx, (base_k, base_v) in enumerate(base_legacy):
                    if layer_idx in kv_layers:
                        # Use recursive KV for this layer
                        rec_k, rec_v = rec_legacy[layer_idx]
                        mixed_layers.append((rec_k, rec_v))
                    else:
                        # Use baseline KV for this layer
                        mixed_layers.append((base_k, base_v))
                
                # Convert back to DynamicCache
                past_key_values = DynamicCache.from_legacy_cache(tuple(mixed_layers))
        
        # Setup steering patcher
        steering_patcher = None
        if steering_vector is not None and steering_layer is not None:
            steering_patcher = SteeringVectorPatcher(self.model, steering_vector, steering_alpha)
            steering_patcher.register(layer_idx=steering_layer)
        
        # Setup V_PROJ patcher (full or head-specific)
        v_patcher = None
        if head_specific_v_patch is not None and head_specific_heads is not None:
            # Head-specific V_PROJ patching
            v_patcher = HeadSpecificVPatcher(
                self.model,
                head_specific_v_patch,
                target_heads=head_specific_heads,
                window_size=v_proj_window
            )
            v_patcher.register(layer_idx=steering_layer if steering_layer is not None else 27)
        elif v_proj_patch is not None and v_proj_layers is not None:
            # Full V_PROJ patching
            for layer in v_proj_layers:
                patcher = PersistentVPatcher(self.model, v_proj_patch)
                patcher.register(layer_idx=layer)
                # Store for cleanup
                if v_patcher is None:
                    v_patcher = []
                v_patcher.append(patcher)
        
        # Setup head-specific steering patcher
        head_steering_patcher = None
        if head_specific_steering is not None and head_specific_heads is not None:
            head_steering_patcher = HeadSpecificSteeringPatcher(
                self.model,
                head_specific_steering,
                target_heads=head_specific_heads,
                alpha=head_specific_steering_alpha
            )
            head_steering_patcher.register(layer_idx=steering_layer if steering_layer is not None else 27)
        
        try:
            # Generate
            with torch.no_grad():
                generated_tokens = []
                entropies = []
                
                # Start with last token of prompt
                current_ids = inputs["input_ids"][:, -1:]
                current_past = past_key_values
                
                for step in range(max_new_tokens):
                    # Apply steering only for first N tokens if specified
                    if steering_first_n is not None and step >= steering_first_n:
                        if steering_patcher:
                            steering_patcher.remove()
                            steering_patcher = None
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=current_ids,
                        past_key_values=current_past,
                        use_cache=True,
                    )
                    
                    # Get logits
                    logits = outputs.logits[:, -1, :]
                    
                    # Compute entropy
                    probs = torch.softmax(logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()
                    entropies.append(entropy)
                    
                    # Sample
                    if temperature == 0.0:
                        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                    else:
                        probs_temp = torch.softmax(logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs_temp, num_samples=1)
                    
                    generated_tokens.append(next_token.item())
                    
                    # Update for next iteration
                    current_ids = next_token
                    current_past = outputs.past_key_values
                    
                    # Stop on EOS
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                
                # Decode
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                mean_entropy = float(np.mean(entropies)) if entropies else 0.0
                
                return generated_text, mean_entropy
                
        finally:
            # Cleanup
            if steering_patcher:
                steering_patcher.remove()
            if head_steering_patcher:
                head_steering_patcher.remove()
            if v_patcher:
                if isinstance(v_patcher, list):
                    for p in v_patcher:
                        p.remove()
                else:
                    v_patcher.remove()


def run_experiment_group(
    model,
    tokenizer,
    device,
    test_pairs: List[Tuple[str, str]],
    recursive_prompts: List[str],
    baseline_prompts: List[str],
    experiment_config: Dict[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    """Run a single experiment configuration."""
    exp_name = experiment_config['name']
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*80}")
    
    generator = MinimalInterventionGenerator(model, tokenizer, device)
    
    # Extract steering vector if needed
    steering_vector = None
    if experiment_config.get('use_steering'):
        layer = experiment_config.get('steering_layer', 27)
        steering_vector = compute_steering_vector(
            model, tokenizer, recursive_prompts, baseline_prompts, layer, device
        )
        print(f"  Steering vector extracted from L{layer}, norm: {steering_vector.norm().item():.4f}")
    
    # Extract KV cache if needed
    kv_cache = None
    if experiment_config.get('use_kv'):
        # Use first recursive prompt to extract KV
        rec_prompt = recursive_prompts[0]
        inputs = tokenizer(rec_prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
            kv_cache = outputs.past_key_values
        kv_layers = experiment_config.get('kv_layers', None)
        if kv_layers:
            print(f"  KV cache extracted from recursive prompt (will replace layers {kv_layers})")
        else:
            print(f"  KV cache extracted from recursive prompt (full replacement)")
    
    # Extract V_PROJ patch if needed (full or head-specific)
    v_proj_patch = None
    head_specific_v_patch = None
    head_specific_heads = experiment_config.get('head_specific_heads')
    
    if experiment_config.get('use_v_proj_patch'):
        layer = experiment_config.get('v_proj_layer', 27)
        window_size = experiment_config.get('v_proj_window', 16)
        
        # Use mean V_PROJ from recursive prompts (last window_size tokens)
        v_activations = []
        for rec_prompt in recursive_prompts[:10]:  # Sample
            v_act = extract_v_activation(model, tokenizer, rec_prompt, layer, device)
            if v_act is not None:
                # Use last window_size tokens only (consistent size)
                v_window = v_act[-window_size:, :]  # (window_size, hidden_dim)
                v_activations.append(v_window)
        if v_activations:
            # All should be same size now (window_size, hidden_dim)
            v_full = torch.stack(v_activations).mean(dim=0)  # (window_size, hidden_dim)
            
            if head_specific_heads:
                # Store full activation for head-specific patching
                head_specific_v_patch = v_full
                print(f"  Head-specific V_PROJ patch extracted from L{layer} (heads {head_specific_heads}, window={window_size})")
            else:
                # Full V_PROJ patch
                v_proj_patch = v_full
                print(f"  V_PROJ patch extracted from L{layer} (window={window_size})")
    
    # Extract head-specific steering if needed
    head_specific_steering = None
    if experiment_config.get('use_head_specific_steering') and steering_vector is not None:
        head_specific_steering = steering_vector  # Use the full steering vector, patcher will slice it
        print(f"  Head-specific steering prepared (heads {experiment_config.get('head_specific_heads')})")
    
    # Run on test pairs
    results = []
    for rec_text, base_text in tqdm(test_pairs, desc=f"Testing {exp_name}"):
        # Safety gate
        if "watch yourself" in base_text.lower() or "observe yourself" in base_text.lower():
            raise ValueError(f"CRITICAL ERROR: Baseline prompt appears recursive!")
        
        # Generate
        try:
            text, entropy = generator.generate_with_intervention(
                prompt=base_text,
                max_new_tokens=200,  # Increased to 200 tokens for full text
                temperature=0.7,
                steering_vector=steering_vector,
                steering_layer=experiment_config.get('steering_layer'),
                steering_alpha=experiment_config.get('steering_alpha', 2.0),
                kv_cache=kv_cache if experiment_config.get('use_kv') else None,
                kv_layers=experiment_config.get('kv_layers'),
                v_proj_patch=v_proj_patch,
                v_proj_layers=experiment_config.get('v_proj_layers'),
                persistent_steering=experiment_config.get('persistent_steering', True),
                steering_first_n=experiment_config.get('steering_first_n'),
                head_specific_v_patch=head_specific_v_patch,
                head_specific_heads=head_specific_heads,
                head_specific_steering=head_specific_steering,
                head_specific_steering_alpha=experiment_config.get('head_specific_steering_alpha', 2.0),
            )
            
            # Score
            score = score_behavior_strict(text, entropy)
            
            results.append({
                "pair_idx": len(results),
                "baseline_prompt": base_text,
                "generated_text": text,
                "entropy": entropy,
                **score.to_dict()
            })
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "baseline_prompt": base_text,
                "generated_text": "",
                "entropy": 0.0,
                "final_score": 0.0,
                "passed_gates": False,
            })
    
    # Save results
    df = pd.DataFrame(results)
    csv_path = run_dir / f"{exp_name}_results.csv"
    df.to_csv(csv_path, index=False)
    
    # Compute stats
    transfer_rate = (df['final_score'] > 0.3).sum() / len(df)
    collapse_rate = 1.0 - df['passed_gates'].mean()
    mean_score = df['final_score'].mean()
    
    print(f"\n  Transfer Rate: {transfer_rate*100:.1f}%")
    print(f"  Collapse Rate: {collapse_rate*100:.1f}%")
    print(f"  Mean Score: {mean_score:.4f}")
    print(f"  Saved to: {csv_path}")
    
    return {
        "name": exp_name,
        "transfer_rate": float(transfer_rate),
        "collapse_rate": float(collapse_rate),
        "mean_score": float(mean_score),
        "results_path": str(csv_path),
    }


def run_minimal_recursive_intervention_from_config(
    cfg: Dict[str, Any],
    run_dir: Path,
) -> ExperimentResult:
    """Run minimal intervention experiments."""
    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-v0.1")
    n_prompts = params.get("n_prompts", 50)
    n_test_pairs = params.get("n_test_pairs", 20)
    round_num = params.get("round", 1)  # Which round to run (or "head_specific")
    seed = int(params.get("seed", 42))
    
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("MINIMAL RECURSIVE INTERVENTION EXPERIMENTS")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Round: {round_num}")
    print(f"Device: {device}")
    
    # Load model
    print("\n[1/3] Loading model...")
    model, tokenizer = load_model(model_name, device=device)
    model.eval()
    
    # Load prompts
    print("\n[2/3] Loading prompts...")
    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    (run_dir / "prompt_bank_version.json").write_text(json.dumps({"version": bank_version}, indent=2) + "\n")
    
    recursive_prompts = []
    for group in ["L4_full", "L5_refined"]:
        prompts = loader.get_by_group(group)
        recursive_prompts.extend(prompts)
    
    baseline_prompts = []
    for group in ["baseline_math", "baseline_factual", "baseline_instructional"]:
        prompts = loader.get_by_group(group)
        baseline_prompts.extend(prompts)
    
    # Sample prompts
    if len(recursive_prompts) < n_prompts:
        sampled_recursive = recursive_prompts
    else:
        sampled_recursive = np.random.choice(recursive_prompts, size=n_prompts, replace=False).tolist()
    
    if len(baseline_prompts) < n_prompts:
        sampled_baseline = baseline_prompts
    else:
        sampled_baseline = np.random.choice(baseline_prompts, size=n_prompts, replace=False).tolist()
    
    print(f"  Recursive: {len(sampled_recursive)}")
    print(f"  Baseline: {len(sampled_baseline)}")
    
    # Get test pairs
    test_pairs = loader.get_balanced_pairs(n_pairs=n_test_pairs, seed=seed)
    print(f"  Test pairs: {len(test_pairs)}")
    
    # Define experiments by round
    experiments = {
        1: [  # Round 1: Most promising
            {
                "name": "A2_Steering_KV_L26-27",
                "use_steering": True,
                "steering_layer": 27,
                "steering_alpha": 2.0,
                "use_kv": True,
                "kv_layers": [26, 27],
                "persistent_steering": True,
            },
            {
                "name": "B1_Steering_VPROJ_L27",
                "use_steering": True,
                "steering_layer": 27,
                "steering_alpha": 2.0,
                "use_v_proj_patch": True,
                "v_proj_layers": [27],
                "persistent_steering": True,
            },
            {
                "name": "F1_Persistent_Steering",
                "use_steering": True,
                "steering_layer": 27,
                "steering_alpha": 2.0,
                "persistent_steering": True,
                "steering_first_n": None,  # All tokens
            },
        ],
        "head_specific": [  # Head-specific surgical interventions
            {
                "name": "H1_HeadSpecific_VPROJ_H18_H26_KV_L27",
                "use_kv": True,
                "kv_layers": [27],
                "use_v_proj_patch": True,
                "v_proj_layer": 27,
                "head_specific_heads": [18, 26],
                "v_proj_window": 16,
            },
            {
                "name": "H2_HeadSpecific_Steering_H18_H26_KV_L26-27",
                "use_steering": True,
                "steering_layer": 27,
                "steering_alpha": 2.0,
                "use_kv": True,
                "kv_layers": [26, 27],
                "use_head_specific_steering": True,
                "head_specific_heads": [18, 26],
                "head_specific_steering_alpha": 2.0,
                "persistent_steering": True,
            },
        ],
        2: [  # Round 2: If Round 1 fails
            {
                "name": "D3_Head_H18_H26",
                "use_steering": True,
                "steering_layer": 27,
                "steering_alpha": 2.0,
                # TODO: Implement head-specific extraction
                "persistent_steering": True,
            },
            {
                "name": "C3_Cascade_Steering",
                "use_steering": True,
                "steering_layer": 27,  # TODO: Multi-layer cascade
                "steering_alpha": 2.0,
                "persistent_steering": True,
            },
        ],
    }
    
    # Run experiments
    print("\n[3/3] Running experiments...")
    experiment_results = []
    
    # Handle round_num - can be int or string
    exp_list = experiments.get(round_num, [])
    if not exp_list:
        print(f"⚠️  No experiments found for round {round_num}")
        return ExperimentResult(summary={"error": f"No experiments for round {round_num}"})
    
    for exp_config in exp_list:
        result = run_experiment_group(
            model, tokenizer, device, test_pairs,
            sampled_recursive, sampled_baseline,
            exp_config, run_dir
        )
        experiment_results.append(result)
        
        # Check if we found success (8+ human-verified)
        # For now, we'll flag high transfer rates for manual review
        if result['transfer_rate'] >= 0.4:
            print(f"\n⚠️  HIGH TRANSFER RATE: {result['transfer_rate']*100:.1f}%")
            print(f"   Manual review needed for: {result['results_path']}")
    
    # Save summary
    summary = {
        "experiment": "minimal_recursive_intervention",
        "round": round_num,
        "experiments": experiment_results,
    }
    
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Summary saved to {summary_path}")
    
    return ExperimentResult(summary=summary)

