"""
Pipeline 9: Causal Steering (DAS / Mean Difference)

Goal: Find the "Surgical Needle" (Steering Vector) that induces Recursive Behavior
without the blunt trauma of a full KV swap.

Method: Compute mean difference vector between Recursive and Baseline V_PROJ activations
at Layer 27, then steer by adding alpha * vec to V_PROJ output during generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import DynamicCache

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.core.patching import extract_v_activation
from src.metrics.behavior_strict import score_behavior_strict
from src.metrics.rv import compute_rv
from src.pipelines.registry import ExperimentResult


class SteeringVectorPatcher:
    """
    Patches V_PROJ output by adding a steering vector scaled by alpha.
    
    This implements Direct Activation Steering (DAS) via mean difference:
    output = output + alpha * (mean_recursive - mean_baseline)
    """
    
    def __init__(self, model, steering_vector: torch.Tensor, alpha: float):
        """
        Initialize patcher with steering vector and scaling factor.
        
        Args:
            model: The transformer model
            steering_vector: Difference vector (mean_recursive - mean_baseline)
                           Shape: (hidden_dim,)
            alpha: Scaling factor for steering strength
        """
        self.model = model
        self.steering_vector = steering_vector.detach().to(model.device)
        self.alpha = alpha
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.layer_idx: Optional[int] = None
    
    def register(self, layer_idx: int):
        """
        Register forward hook at specified layer to add steering vector.
        
        Args:
            layer_idx: Layer index (0-indexed, e.g., 27 for L27)
        """
        if self.handle is not None:
            raise RuntimeError("Patcher already registered. Call remove() first.")
        
        self.layer_idx = layer_idx
        layer = self.model.model.layers[layer_idx].self_attn
        
        def hook_fn(module, inp, out):
            """
            Hook function that adds steering vector to V_PROJ output.
            
            Args:
                module: The v_proj module
                inp: Input to v_proj (hidden states)
                out: Output from v_proj (batch, seq_len, hidden_dim)
            
            Returns:
                Steered output: out + alpha * steering_vector
            """
            # out shape: (batch, seq_len, hidden_dim)
            # steering_vector shape: (hidden_dim,)
            # Add steering vector to all tokens in sequence
            steered = out + self.alpha * self.steering_vector.unsqueeze(0).unsqueeze(0)
            return steered
        
        self.handle = layer.v_proj.register_forward_hook(hook_fn)
    
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


def _generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: str,
) -> Tuple[str, float]:
    """
    Generate text with steering vector applied.
    Returns: (generated_text, mean_step_entropy)
    """
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    
    generated_tokens = []
    entropies = []
    
    with torch.no_grad():
        # First pass: get KV cache for prompt
        out = model(**inputs, use_cache=True)
        current_ids = inputs["input_ids"][:, -1:]
        current_kv = out.past_key_values
        
        for _ in range(max_new_tokens):
            out = model(current_ids, past_key_values=current_kv, use_cache=True)
            
            logits = out.logits[:, -1, :]
            
            # Compute entropy of this step
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()
            entropies.append(entropy)
            
            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            else:
                probs_temp = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs_temp, num_samples=1)
                
            generated_tokens.append(next_token.item())
            
            # Update
            current_ids = next_token
            current_kv = out.past_key_values
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    mean_entropy = float(np.mean(entropies)) if entropies else 0.0
    return text, mean_entropy


def compute_steering_vector(
    model,
    tokenizer,
    recursive_prompts: List[str],
    baseline_prompts: List[str],
    layer_idx: int,
    device: str,
) -> torch.Tensor:
    """
    Compute steering vector as mean difference: mean(recursive) - mean(baseline).
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        recursive_prompts: List of recursive prompt texts
        baseline_prompts: List of baseline prompt texts
        layer_idx: Layer index for V_PROJ extraction
        device: Device to run on
    
    Returns:
        Steering vector: (hidden_dim,) tensor
    """
    print(f"Computing steering vector from {len(recursive_prompts)} recursive + {len(baseline_prompts)} baseline prompts...")
    
    # Collect V_PROJ activations from recursive prompts
    recursive_vs = []
    for prompt in tqdm(recursive_prompts, desc="Extracting recursive V_PROJ"):
        try:
            v_act = extract_v_activation(model, tokenizer, prompt, layer_idx, device)
            # Use mean over sequence dimension to get single vector per prompt
            recursive_vs.append(v_act.mean(dim=0))  # (hidden_dim,)
        except Exception as e:
            print(f"  Warning: Failed to extract V_PROJ for recursive prompt: {e}")
            continue
    
    # Collect V_PROJ activations from baseline prompts
    baseline_vs = []
    for prompt in tqdm(baseline_prompts, desc="Extracting baseline V_PROJ"):
        try:
            v_act = extract_v_activation(model, tokenizer, prompt, layer_idx, device)
            # Use mean over sequence dimension to get single vector per prompt
            baseline_vs.append(v_act.mean(dim=0))  # (hidden_dim,)
        except Exception as e:
            print(f"  Warning: Failed to extract V_PROJ for baseline prompt: {e}")
            continue
    
    if len(recursive_vs) == 0 or len(baseline_vs) == 0:
        raise RuntimeError(f"Insufficient activations: {len(recursive_vs)} recursive, {len(baseline_vs)} baseline")
    
    # Stack and compute means
    recursive_mean = torch.stack(recursive_vs).mean(dim=0)  # (hidden_dim,)
    baseline_mean = torch.stack(baseline_vs).mean(dim=0)    # (hidden_dim,)
    
    # Steering vector = difference
    steering_vector = recursive_mean - baseline_mean
    
    print(f"  Steering vector shape: {steering_vector.shape}")
    print(f"  Steering vector norm: {steering_vector.norm().item():.4f}")
    print(f"  Recursive mean norm: {recursive_mean.norm().item():.4f}")
    print(f"  Baseline mean norm: {baseline_mean.norm().item():.4f}")
    
    return steering_vector


def run_steering_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """
    Run Pipeline 9: Causal Steering (Mean Difference DAS).
    
    Config params:
        model: Model name (default: "mistralai/Mistral-7B-v0.1")
        layer: Layer index for V_PROJ (default: 27)
        n_prompts: Number of prompts per class for vector computation (default: 50)
        n_test_pairs: Number of test pairs for evaluation (default: 20)
        alphas: List of alpha values to test (default: [0.5, 1.0, 2.0, 5.0])
        max_new_tokens: Max tokens to generate (default: 100)
        temperature: Sampling temperature (default: 0.7)
    """
    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-v0.1")
    layer_idx = params.get("layer", 27)
    n_prompts = params.get("n_prompts", 50)
    n_test_pairs = params.get("n_test_pairs", 20)
    alphas = params.get("alphas", [0.5, 1.0, 2.0, 5.0])
    max_new_tokens = params.get("max_new_tokens", 100)
    temperature = params.get("temperature", 0.7)
    seed = int(params.get("seed", 42))
    
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("PIPELINE 9: CAUSAL STEERING (MEAN DIFFERENCE DAS)")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Layer: {layer_idx}")
    print(f"Prompts per class: {n_prompts}")
    print(f"Test pairs: {n_test_pairs}")
    print(f"Alphas: {alphas}")
    print("=" * 80)
    
    # Load model
    print("\n[1/4] Loading model...")
    model, tokenizer = load_model(model_name, device=device)
    model.eval()
    
    # Load prompts
    print("\n[2/4] Loading prompts...")
    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    (run_dir / "prompt_bank_version.json").write_text(json.dumps({"version": bank_version}, indent=2) + "\n")
    
    # Get recursive prompts (L4/L5)
    recursive_prompts = []
    for group in ["L4_full", "L5_refined"]:
        prompts = loader.get_by_group(group)
        recursive_prompts.extend(prompts)  # get_by_group returns List[str]
    
    # Get baseline prompts
    baseline_prompts = []
    for group in ["baseline_math", "baseline_factual", "baseline_instructional"]:
        prompts = loader.get_by_group(group)
        baseline_prompts.extend(prompts)  # get_by_group returns List[str]
    
    print(f"  Found {len(recursive_prompts)} recursive prompts")
    print(f"  Found {len(baseline_prompts)} baseline prompts")
    
    # Sample prompts for vector computation
    if len(recursive_prompts) < n_prompts:
        print(f"  Warning: Only {len(recursive_prompts)} recursive prompts available, using all")
        sampled_recursive = recursive_prompts
    else:
        sampled_recursive = np.random.choice(recursive_prompts, size=n_prompts, replace=False).tolist()
    
    if len(baseline_prompts) < n_prompts:
        print(f"  Warning: Only {len(baseline_prompts)} baseline prompts available, using all")
        sampled_baseline = baseline_prompts
    else:
        sampled_baseline = np.random.choice(baseline_prompts, size=n_prompts, replace=False).tolist()
    
    # Compute steering vector
    print("\n[3/4] Computing steering vector...")
    steering_vector = compute_steering_vector(
        model, tokenizer, sampled_recursive, sampled_baseline, layer_idx, device
    )
    
    # Save steering vector
    steering_dir = run_dir.parent / "steering"
    steering_dir.mkdir(exist_ok=True)
    steering_path = steering_dir / "recursive_vector.pt"
    torch.save(steering_vector.cpu(), steering_path)
    print(f"  Saved steering vector to {steering_path}")
    
    # Get test pairs (balanced pairs)
    print("\n[4/4] Testing steering with different alpha values...")
    pairs = loader.get_balanced_pairs(n_pairs=n_test_pairs, seed=seed)
    print(f"  Testing on {len(pairs)} pairs")
    
    results = []
    
    # get_balanced_pairs returns (recursive_prompt, baseline_prompt) tuple
    # CRITICAL BUG FIX: We were testing on RECURSIVE prompts instead of BASELINE prompts!
    # The function returns (recursive, baseline) - unpack correctly!
    for i, (rec_text, base_text) in enumerate(tqdm(pairs, desc="Testing pairs")):
        # Safety gate: Verify base_text is actually a baseline prompt
        if "watch yourself" in base_text.lower() or "observe yourself" in base_text.lower() or "consciousness examining" in base_text.lower():
            raise ValueError(f"CRITICAL ERROR: Baseline prompt appears recursive! Text: {base_text[:100]}")
        
        for alpha in alphas:
            # Create steering patcher
            patcher = SteeringVectorPatcher(model, steering_vector, alpha)
            patcher.register(layer_idx=layer_idx)
            
            try:
                # Generate with steering on the BASELINE prompt (not recursive!)
                text, entropy = _generate_with_steering(
                    model, tokenizer, base_text, max_new_tokens, temperature, device
                )
                
                # Score behavior
                score = score_behavior_strict(text, entropy)
                
                results.append({
                    "pair_idx": i,
                    "alpha": alpha,
                    "baseline_prompt": base_text[:100],  # Truncate for CSV
                    "generated_text": text[:500],  # Truncate for CSV
                    "text_len": len(text),
                    "entropy": entropy,
                    **score.to_dict()
                })
            except Exception as e:
                print(f"  Error on pair {i}, alpha {alpha}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                patcher.remove()
    
    # Save results
    df = pd.DataFrame(results)
    csv_path = run_dir / "steering_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")
    
    # Compute summary statistics
    summary = {
        "experiment": "steering",
        "model_name": model_name,
        "layer": layer_idx,
        "n_prompts": n_prompts,
        "n_test_pairs": len(pairs),
        "alphas": alphas,
        "steering_vector_path": str(steering_path),
        "conditions": {}
    }
    
    for alpha in alphas:
        alpha_results = df[df["alpha"] == alpha]
        if len(alpha_results) > 0:
            summary["conditions"][f"alpha_{alpha}"] = {
                "mean_score": float(alpha_results["final_score"].mean()),
                "pass_rate": float(alpha_results["passed_gates"].mean()),
                "samples_above_zero": int((alpha_results["final_score"] > 0.0).sum()),
                "samples_above_0_3": int((alpha_results["final_score"] > 0.3).sum()),
                "collapse_rate": float(1.0 - alpha_results["passed_gates"].mean()),
                "mean_diversity": float(alpha_results["diversity_score"].mean()),
            }
    
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    for alpha in alphas:
        cond_key = f"alpha_{alpha}"
        if cond_key in summary["conditions"]:
            stats = summary["conditions"][cond_key]
            print(f"\nAlpha {alpha}:")
            print(f"  Mean Score: {stats['mean_score']:.4f}")
            print(f"  Pass Rate: {stats['pass_rate']*100:.1f}%")
            print(f"  Transfer Rate (>0.3): {stats['samples_above_0_3']}/{len(pairs)} ({stats['samples_above_0_3']/len(pairs)*100:.1f}%)")
            print(f"  Collapse Rate: {stats['collapse_rate']*100:.1f}%")
    
    return ExperimentResult(summary=summary)

