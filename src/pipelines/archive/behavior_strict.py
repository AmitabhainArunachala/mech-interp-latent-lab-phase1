"""
Behavior Strict Pipeline (Gold Standard Pipeline 5)

Tests if Geometric Contraction ($R_V < 1.0$) causes measurable "Recursive Behavior"
while strictly filtering out model collapse/repetition.

Intervention: KV Cache Swap (from Pipeline 8).
Measurement: StrictBehaviorScore (Gates + Features).
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
from src.core.patching import (
    PersistentVPatcher,
    PersistentResidualPatcher,
    extract_v_activation,
    extract_residual_activation,
)
from src.metrics.behavior_strict import score_behavior_strict
from src.metrics.rv import compute_rv
from src.pipelines.registry import ExperimentResult


def _extract_kv(model, tokenizer, prompt: str, device: str) -> DynamicCache:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
        return out.past_key_values


def _generate_with_kv(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    past_key_values: DynamicCache,
    max_new_tokens: int,
    temperature: float,
) -> Tuple[str, float]:
    """
    Generate text using provided KV cache.
    Returns: (generated_text, mean_step_entropy)
    """
    current_ids = prompt_ids[:, -1:]
    current_kv = past_key_values
    
    generated_tokens = []
    entropies = []
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
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


def _shuffle_kv(kv: DynamicCache, device: str) -> DynamicCache:
    """
    Shuffle KV cache along the sequence dimension.
    Preserves token features, destroys order/structure.
    """
    # Legacy tuple format: (batch, heads, seq, dim)
    if hasattr(kv, "to_legacy_cache"):
        legacy = kv.to_legacy_cache()
    else:
        legacy = kv
        
    shuffled_layers = []
    for k, v in legacy:
        # k: (B, H, S, D)
        seq_len = k.shape[2]
        perm = torch.randperm(seq_len, device=device)
        
        k_shuf = k[:, :, perm, :]
        v_shuf = v[:, :, perm, :]
        shuffled_layers.append((k_shuf, v_shuf))
        
    return DynamicCache.from_legacy_cache(tuple(shuffled_layers))


def _random_kv(ref_kv: DynamicCache, device: str) -> DynamicCache:
    """
    Generate Gaussian noise KV with same shape/stats as reference.
    """
    if hasattr(ref_kv, "to_legacy_cache"):
        legacy = ref_kv.to_legacy_cache()
    else:
        legacy = ref_kv
        
    rand_layers = []
    for k, v in legacy:
        k_noise = torch.randn_like(k) * k.std() + k.mean()
        v_noise = torch.randn_like(v) * v.std() + v.mean()
        rand_layers.append((k_noise, v_noise))
        
    return DynamicCache.from_legacy_cache(tuple(rand_layers))


def run_behavior_strict_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """Run strict behavior validation pipeline."""
    model_cfg = cfg.get("model") or {}
    params = cfg.get("params") or {}
    
    seed = int(cfg.get("seed") or 42)
    device = str(model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = str(model_cfg.get("name") or "mistralai/Mistral-7B-v0.1")
    
    n_pairs = int(params.get("n_pairs") or 20)
    max_new_tokens = int(params.get("max_new_tokens") or 100)
    temperature = float(params.get("temperature") or 0.7)
    
    set_seed(seed)
    model, tokenizer = load_model(model_name, device=device)
    
    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    (run_dir / "prompt_bank_version.json").write_text(json.dumps({"version": bank_version}))
    
    # Load Balanced Pairs with R_V filtering (V3: Improve consistency)
    raw_pairs = loader.get_balanced_pairs(n_pairs=n_pairs*5, seed=seed)  # Get more to filter
    pairs = []
    
    window = 16 # Minimum context for validity
    early_layer = 5
    late_layer = 27
    
    # V3: Pre-filter by geometric signature strength
    print("Pre-filtering pairs by R_V signature strength...")
    filtered_pairs = []
    for rec_text, base_text in raw_pairs:
        r_ids = tokenizer.encode(rec_text, add_special_tokens=False)
        b_ids = tokenizer.encode(base_text, add_special_tokens=False)
        common_len = min(len(r_ids), len(b_ids))
        if common_len < window: continue
        
        # Check if recursive prompt has strong geometric signature (R_V < 0.9)
        try:
            rv_rec = compute_rv(model, tokenizer, rec_text, early=early_layer, late=late_layer, window=window, device=device)
            if rv_rec < 0.9:  # Strong contraction signal
                filtered_pairs.append((rec_text, base_text, r_ids[:common_len], b_ids[:common_len]))
        except Exception as e:
            # Skip if R_V computation fails
            continue
        
        if len(filtered_pairs) >= n_pairs * 2: break
    
    print(f"Filtered to {len(filtered_pairs)} pairs with strong R_V signature.")
    
    # Convert to tensors
    for rec_text, base_text, r_ids, b_ids in filtered_pairs[:n_pairs]:
        rec_ids = torch.tensor([r_ids], device=device)
        base_ids = torch.tensor([b_ids], device=device)
        pairs.append((rec_text, base_text, rec_ids, base_ids))
        
    print(f"Loaded {len(pairs)} length-matched pairs with strong geometric signatures.")
    
    results = []
    
    for i, (rec_text, base_text, rec_ids, base_ids) in enumerate(tqdm(pairs)):
        try:
            # 1. Extract KV Caches
            with torch.no_grad():
                out_rec = model(rec_ids, use_cache=True)
                rec_kv = out_rec.past_key_values
            
            # 2. Extract activations from recursive prompt (V3: Multi-layer patching)
            # V3: Extract both L18 RESIDUAL and L27 V_PROJ (Dec 12 showed L18+L27 works best)
            target_layer_v = 27  # L27 V_PROJ (contraction phase)
            target_layer_r = 18  # L18 RESIDUAL (expansion phase)
            
            try:
                rec_v_l27 = extract_v_activation(
                    model, tokenizer, rec_text, layer_idx=target_layer_v, device=device
                )
                rec_r_l18 = extract_residual_activation(
                    model, tokenizer, rec_text, layer_idx=target_layer_r, device=device
                )
            except Exception as e:
                print(f"  Warning: Failed to extract activations for pair {i}: {e}")
                continue
            
            # 3. Controls
            shuffled_kv = _shuffle_kv(rec_kv, device)
            random_kv = _random_kv(rec_kv, device)
            
            # 4. Generate & Score Conditions (V3: Multi-layer patching)
            conditions = [
                ("Recursive_Control", rec_ids, rec_kv, True, True),      # High: KV + L18+L27 patching
                ("Baseline_Control", base_ids, None, False, False),        # Low: No patching
                ("Transfer", base_ids, rec_kv, True, True),               # Test: KV + L18+L27 patching
                ("Transfer_L27_Only", base_ids, rec_kv, True, False),    # Ablation: KV + L27 only
                ("Shuffled_Control", base_ids, shuffled_kv, False, False), # Structure Check
                ("Random_Control", base_ids, random_kv, False, False)     # Damage Check
            ]
            
            for cond_name, prompt_ids, kv_cache, use_v_patching, use_r_patching in conditions:
                # If kv_cache is None, generate normally (Baseline)
                if kv_cache is None:
                    # We need to re-run the baseline prompt to get its natural KV
                    with torch.no_grad():
                        out = model(prompt_ids, use_cache=True)
                        kv_to_use = out.past_key_values
                else:
                    kv_to_use = kv_cache
                    
                # V3: Create and register multi-layer patchers
                v_patcher = None
                r_patcher = None
                
                if use_v_patching:
                    try:
                        v_patcher = PersistentVPatcher(model, rec_v_l27)
                        v_patcher.register(layer_idx=target_layer_v)
                    except Exception as e:
                        print(f"  Warning: Failed to register V patcher for {cond_name}: {e}")
                        continue
                
                if use_r_patching:
                    try:
                        r_patcher = PersistentResidualPatcher(model, rec_r_l18)
                        r_patcher.register(layer_idx=target_layer_r)
                    except Exception as e:
                        print(f"  Warning: Failed to register R patcher for {cond_name}: {e}")
                        # Continue without R patcher if V patcher is registered
                        pass
                
                try:
                    text, entropy = _generate_with_kv(
                        model, tokenizer, prompt_ids, kv_to_use, max_new_tokens, temperature
                    )
                    
                    score = score_behavior_strict(text, entropy)
                    
                    results.append({
                        "pair_idx": i,
                        "condition": cond_name,
                        "text_len": len(text),
                        "entropy": entropy,
                        "used_v_patching": use_v_patching,  # Track whether patching was used
                        "used_r_patching": use_r_patching,  # V3: Track residual patching
                        **score.to_dict()
                    })
                except Exception as e:
                    print(f"  Error generating {cond_name} for pair {i}: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # Always remove patchers if they were created
                    if v_patcher is not None:
                        try:
                            v_patcher.remove()
                        except Exception:
                            pass
                    if r_patcher is not None:
                        try:
                            r_patcher.remove()
                        except Exception:
                            pass
                
        except Exception as e:
            print(f"Error on pair {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    # Save Results
    df = pd.DataFrame(results)
    out_csv = run_dir / "behavior_strict_results.csv"
    df.to_csv(out_csv, index=False)
    
    # Stats Summary
    summary = {
        "experiment": "behavior_strict",
        "model_name": model_name,
        "n_pairs": len(pairs),
        "conditions": {}
    }
    
    for cond in df["condition"].unique():
        sub = df[df["condition"] == cond]
        summary["conditions"][cond] = {
            "mean_score": float(sub["final_score"].mean()),
            "pass_rate": float(sub["passed_gates"].mean()),
            "diversity": float(sub["diversity_score"].mean())
        }
        
    summary["prompt_bank_version"] = bank_version
    summary["artifacts"] = {"csv": str(out_csv)}
    
    return ExperimentResult(summary=summary)

