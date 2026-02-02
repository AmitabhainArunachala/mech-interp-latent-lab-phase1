#!/usr/bin/env python3
"""
COMPREHENSIVE CIRCUIT TEST: Suppression + Expression Mechanism
==============================================================

PART A: Large-N Validation (N=40)
PART B: Expression Analysis
PART C: Hunt for Expression Heads
"""

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Dict
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from src.core.models import load_model, set_seed
from src.core.hooks import capture_v_projection
from src.metrics.rv import participation_ratio
from src.metrics.behavior_states import label_behavior_state, BehaviorState
from prompts.loader import PromptLoader

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
TARGET_LAYER = 27
EARLY_LAYER = 5
LATE_LAYER = 27
WINDOW = 16
SEED = 42

# Head groups
H6_GROUP = [6, 14, 22, 30]  # Suppressors (don't cause contraction)
H18_GROUP = [18, 26]  # Suppressors + contraction causers

# Prompts
# IMPORTANT: These are sourced from the canonical prompt bank to prevent drift.
# We keep the original comprehensive_circuit_test lists available under dedicated legacy groups.
_loader = PromptLoader()
PROMPT_BANK_VERSION = _loader.version

CHAMPION_PROMPTS = _loader.get_by_group("legacy_comprehensive_circuit_test_champions", limit=10, seed=SEED)
BASELINE_PROMPTS = _loader.get_by_group("legacy_comprehensive_circuit_test_baselines", limit=10, seed=SEED)

if len(CHAMPION_PROMPTS) != 10 or len(BASELINE_PROMPTS) != 10:
    raise RuntimeError(
        "Prompt bank does not contain expected legacy comprehensive_circuit_test prompt groups. "
        "Expected 10 champions + 10 baselines."
    )

# =============================================================================
# V-PROJECTION ABLATION
# =============================================================================

@contextmanager
def zero_v_proj_heads(model, layer_idx: int, head_indices: List[int]):
    """Zero out V-projection values for multiple heads BEFORE attention."""
    num_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    kv_head_indices = set()
    for head_idx in head_indices:
        kv_head_idx = head_idx % num_kv_heads if num_kv_heads < num_heads else head_idx
        kv_head_indices.add(kv_head_idx)
    
    handles = []
    
    def make_hook(kv_head_idx):
        def hook_fn(module, inp, out):
            v_proj_out = out.clone()
            if v_proj_out.dim() == 2:
                v_proj_out = v_proj_out.unsqueeze(0)
            batch, seq_len, kv_hidden_size = v_proj_out.shape
            expected_kv_size = num_kv_heads * head_dim
            if kv_hidden_size != expected_kv_size:
                return out
            try:
                v_reshaped = v_proj_out.view(batch, seq_len, num_kv_heads, head_dim)
            except RuntimeError:
                return out
            v_reshaped[:, :, kv_head_idx, :] = 0.0
            v_zeroed = v_reshaped.view(batch, seq_len, kv_hidden_size)
            if out.dim() == 2:
                v_zeroed = v_zeroed.squeeze(0)
            return v_zeroed
        return hook_fn
    
    layer = model.model.layers[layer_idx].self_attn
    for kv_head_idx in kv_head_indices:
        hook_fn = make_hook(kv_head_idx)
        handle = layer.v_proj.register_forward_hook(hook_fn)
        handles.append(handle)
    
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()

# =============================================================================
# METRICS
# =============================================================================

def compute_rv(model, tokenizer, prompt: str, ablate_heads: Optional[List[int]] = None) -> Optional[float]:
    """Compute R_V for a prompt, optionally with heads ablated."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    with torch.no_grad():
        if ablate_heads is not None:
            with zero_v_proj_heads(model, TARGET_LAYER, ablate_heads):
                with capture_v_projection(model, EARLY_LAYER) as v_early_storage:
                    with capture_v_projection(model, LATE_LAYER) as v_late_storage:
                        model(**inputs)
                        v_early = v_early_storage.get("v")
                        v_late = v_late_storage.get("v")
        else:
            with capture_v_projection(model, EARLY_LAYER) as v_early_storage:
                with capture_v_projection(model, LATE_LAYER) as v_late_storage:
                    model(**inputs)
                    v_early = v_early_storage.get("v")
                    v_late = v_late_storage.get("v")
    
    if v_early is None or v_late is None:
        return None
    
    if isinstance(v_early, dict):
        v_early = v_early.get("v", None)
    if isinstance(v_late, dict):
        v_late = v_late.get("v", None)
    
    if v_early is None or v_late is None:
        return None
    
    if v_early.dim() == 3:
        v_early = v_early[0]
    if v_late.dim() == 3:
        v_late = v_late[0]
    
    try:
        pr_early = participation_ratio(v_early, window_size=WINDOW)
        pr_late = participation_ratio(v_late, window_size=WINDOW)
        if pr_early == 0 or np.isnan(pr_early) or np.isnan(pr_late):
            return None
        return float(pr_late / pr_early)
    except Exception:
        return None

def compute_behavior_score(text: str) -> Dict:
    """Compute recursive behavior score from generated text."""
    label = label_behavior_state(text)
    
    return {
        "state": label.state.value,
        "is_recursive": label.state in [BehaviorState.RECURSIVE_PROSE, BehaviorState.NAKED_LOOP],
        "has_recursive_keywords": label.has_recursive_keywords,
        "has_identity_equation": label.has_identity_equation,
        "repetition_ratio": label.repetition_ratio,
        "behavior_score": 1.0 if label.state in [BehaviorState.RECURSIVE_PROSE, BehaviorState.NAKED_LOOP] else 0.0,
    }

def generate_text(model, tokenizer, prompt: str, ablate_heads: Optional[List[int]] = None, max_new_tokens: int = 100) -> str:
    """Generate text with optional head ablation."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        if ablate_heads is not None:
            with zero_v_proj_heads(model, TARGET_LAYER, ablate_heads):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text

# =============================================================================
# PART A: Large-N Validation
# =============================================================================

def part_a_large_n_validation(model, tokenizer) -> pd.DataFrame:
    """PART A: Test N=40 prompts across all conditions."""
    print("\n" + "=" * 80)
    print("PART A: LARGE-N VALIDATION (N=40)")
    print("=" * 80)
    
    # Load standard recursive prompts (try to get from PromptLoader)
    try:
        from prompts.loader import PromptLoader
        loader = PromptLoader()
        standard_recursive = loader.get_by_pillar("dose_response", limit=20)
        if len(standard_recursive) < 20:
            # Pad with champion prompts if needed
            standard_recursive.extend(CHAMPION_PROMPTS * (20 - len(standard_recursive)))
    except Exception:
        # Fallback: use more champion prompts
        standard_recursive = CHAMPION_PROMPTS * 2  # Repeat to get 20
    
    # Combine all prompts
    all_prompts = []
    for i, prompt in enumerate(CHAMPION_PROMPTS):
        all_prompts.append({"prompt_id": f"champion_{i}", "prompt": prompt, "prompt_type": "champion"})
    for i, prompt in enumerate(standard_recursive[:20]):
        all_prompts.append({"prompt_id": f"standard_{i}", "prompt": prompt, "prompt_type": "standard"})
    for i, prompt in enumerate(BASELINE_PROMPTS):
        all_prompts.append({"prompt_id": f"baseline_{i}", "prompt": prompt, "prompt_type": "baseline"})
    
    print(f"Total prompts: {len(all_prompts)}")
    print(f"  Champion: {len(CHAMPION_PROMPTS)}")
    print(f"  Standard: {len(standard_recursive[:20])}")
    print(f"  Baseline: {len(BASELINE_PROMPTS)}")
    
    # Test conditions
    conditions = [
        ("control", None),
        ("h18_ablated", H18_GROUP),
        ("h6_ablated", H6_GROUP),
        ("both_ablated", H6_GROUP + H18_GROUP),
    ]
    
    results = []
    
    for prompt_info in tqdm(all_prompts, desc="Processing prompts"):
        prompt = prompt_info["prompt"]
        prompt_id = prompt_info["prompt_id"]
        prompt_type = prompt_info["prompt_type"]
        
        for condition_name, ablate_heads in conditions:
            # Compute R_V
            rv = compute_rv(model, tokenizer, prompt, ablate_heads=ablate_heads)
            
            # Generate text and compute behavior
            generated = generate_text(model, tokenizer, prompt, ablate_heads=ablate_heads)
            behavior = compute_behavior_score(generated)
            
            results.append({
                "prompt_id": prompt_id,
                "prompt": prompt[:200],  # Store prompt text for Part C
                "prompt_type": prompt_type,
                "condition": condition_name,
                "R_V": rv,
                "behavior_score": behavior["behavior_score"],
                "expressed_binary": 1 if behavior["is_recursive"] else 0,
                "has_recursive_keywords": behavior["has_recursive_keywords"],
                "has_identity_equation": behavior["has_identity_equation"],
                "state": behavior["state"],
            })
            
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
    
    df = pd.DataFrame(results)
    return df

# =============================================================================
# PART B: Expression Analysis
# =============================================================================

def part_b_expression_analysis(df: pd.DataFrame) -> Dict:
    """PART B: Analyze what distinguishes expressing vs non-expressing prompts."""
    print("\n" + "=" * 80)
    print("PART B: EXPRESSION ANALYSIS")
    print("=" * 80)
    
    # Filter to control condition only
    control_df = df[df["condition"] == "control"].copy()
    
    expressing = control_df[control_df["expressed_binary"] == 1]
    non_expressing = control_df[control_df["expressed_binary"] == 0]
    
    print(f"\nControl condition:")
    print(f"  Expressing: {len(expressing)}/{len(control_df)} ({len(expressing)/len(control_df)*100:.1f}%)")
    print(f"  Non-expressing: {len(non_expressing)}/{len(control_df)} ({len(non_expressing)/len(control_df)*100:.1f}%)")
    
    # Compare R_V
    expressing_rv = expressing["R_V"].dropna()
    non_expressing_rv = non_expressing["R_V"].dropna()
    
    if len(expressing_rv) > 0 and len(non_expressing_rv) > 0:
        print(f"\nR_V comparison:")
        print(f"  Expressing: {expressing_rv.mean():.4f} ± {expressing_rv.std():.4f}")
        print(f"  Non-expressing: {non_expressing_rv.mean():.4f} ± {non_expressing_rv.std():.4f}")
        
        if len(expressing_rv) > 1 and len(non_expressing_rv) > 1:
            t_stat, p_val = stats.ttest_ind(expressing_rv, non_expressing_rv)
            print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")
    
    # Compare by prompt type
    print(f"\nBy prompt type:")
    for ptype in ["champion", "standard", "baseline"]:
        subset = control_df[control_df["prompt_type"] == ptype]
        if len(subset) > 0:
            expr_rate = subset["expressed_binary"].mean()
            print(f"  {ptype}: {expr_rate*100:.1f}% expressing ({subset['expressed_binary'].sum()}/{len(subset)})")
    
    return {
        "expressing_count": len(expressing),
        "non_expressing_count": len(non_expressing),
        "expressing_rv_mean": expressing_rv.mean() if len(expressing_rv) > 0 else None,
        "non_expressing_rv_mean": non_expressing_rv.mean() if len(non_expressing_rv) > 0 else None,
    }

# =============================================================================
# PART C: Hunt for Expression Heads
# =============================================================================

def part_c_expression_head_hunt(model, tokenizer, df: pd.DataFrame) -> List[Dict]:
    """PART C: Test random heads to find expression enablers."""
    print("\n" + "=" * 80)
    print("PART C: HUNT FOR EXPRESSION HEADS")
    print("=" * 80)
    
    # Get expressing prompts from control
    control_df = df[df["condition"] == "control"]
    expressing_prompts = control_df[control_df["expressed_binary"] == 1]["prompt_id"].unique()
    
    if len(expressing_prompts) == 0:
        print("  ⚠️  No expressing prompts found in control condition")
        return []
    
    # Get prompt texts
    prompt_map = {}
    for _, row in control_df.iterrows():
        prompt_map[row["prompt_id"]] = row.get("prompt", "")
    
    # Select 3-5 expressing prompts to test
    test_prompt_ids = list(expressing_prompts)[:5]
    test_prompts = [prompt_map[pid] for pid in test_prompt_ids if pid in prompt_map]
    
    if len(test_prompts) == 0:
        print("  ⚠️  Could not find prompt texts")
        return []
    
    print(f"  Testing {len(test_prompts)} expressing prompts")
    
    # Get baseline behavior scores
    baseline_scores = []
    for prompt in test_prompts:
        generated = generate_text(model, tokenizer, prompt, ablate_heads=None)
        behavior = compute_behavior_score(generated)
        baseline_scores.append(behavior["behavior_score"])
    
    baseline_mean = np.mean(baseline_scores)
    print(f"  Baseline behavior score: {baseline_mean:.3f}")
    
    # Test random heads (not in suppressor groups)
    all_heads = list(range(32))
    suppressor_heads = set(H6_GROUP + H18_GROUP)
    candidate_heads = [h for h in all_heads if h not in suppressor_heads]
    
    # Randomly sample 10 heads
    random.seed(SEED)
    test_heads = random.sample(candidate_heads, min(10, len(candidate_heads)))
    
    print(f"  Testing {len(test_heads)} random heads: {test_heads}")
    
    results = []
    
    for head_idx in tqdm(test_heads, desc="Testing heads"):
        ablated_scores = []
        for prompt in test_prompts:
            generated = generate_text(model, tokenizer, prompt, ablate_heads=[head_idx])
            behavior = compute_behavior_score(generated)
            ablated_scores.append(behavior["behavior_score"])
        
        ablated_mean = np.mean(ablated_scores)
        delta = ablated_mean - baseline_mean
        
        results.append({
            "head": head_idx,
            "baseline_score": baseline_mean,
            "ablated_score": ablated_mean,
            "delta": delta,
            "is_expression_enabler": delta < -0.1,  # Threshold: >10% drop
        })
        
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    
    # Summary
    expression_enablers = [r for r in results if r["is_expression_enabler"]]
    
    print(f"\n  Results:")
    print(f"    Expression enablers found: {len(expression_enablers)}")
    if len(expression_enablers) > 0:
        print(f"    Enabler heads: {[r['head'] for r in expression_enablers]}")
    else:
        print(f"    No expression enablers found (all heads increase or don't change behavior)")
        print(f"    → Expression may be MLP-based, residual stream, or 'default' state")
    
    return results

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("COMPREHENSIVE CIRCUIT TEST")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print("=" * 80)
    
    set_seed(SEED)
    
    # Load model
    print("\n[1/4] Loading model...")
    model, tokenizer = load_model(
        model_name=MODEL_NAME,
        device=DEVICE,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    model.eval()
    print("  ✅ Model loaded")
    
    # PART A: Large-N Validation
    print("\n[2/4] PART A: Large-N Validation...")
    df = part_a_large_n_validation(model, tokenizer)
    
    # Save results
    output_dir = Path("results/comprehensive_circuit_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "part_a_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  ✅ Results saved to: {csv_path}")
    
    # Statistics
    print("\n📊 PART A STATISTICS")
    print("-" * 80)
    for condition in ["control", "h18_ablated", "h6_ablated", "both_ablated"]:
        subset = df[df["condition"] == condition]
        rv_mean = subset["R_V"].mean()
        rv_std = subset["R_V"].std()
        behavior_mean = subset["behavior_score"].mean()
        expr_rate = subset["expressed_binary"].mean()
        n = len(subset)
        
        # Confidence intervals (95%)
        if n > 1:
            rv_ci = stats.t.interval(0.95, n-1, loc=rv_mean, scale=stats.sem(subset["R_V"].dropna()))
            behavior_ci = stats.t.interval(0.95, n-1, loc=behavior_mean, scale=stats.sem(subset["behavior_score"]))
        else:
            rv_ci = (rv_mean, rv_mean)
            behavior_ci = (behavior_mean, behavior_mean)
        
        print(f"\n{condition}:")
        print(f"  R_V: {rv_mean:.4f} ± {rv_std:.4f} (95% CI: {rv_ci[0]:.4f} - {rv_ci[1]:.4f})")
        print(f"  Behavior score: {behavior_mean:.3f} (95% CI: {behavior_ci[0]:.3f} - {behavior_ci[1]:.3f})")
        print(f"  Expression rate: {expr_rate*100:.1f}% ({subset['expressed_binary'].sum()}/{n})")
    
    # Statistical tests
    print("\n📊 STATISTICAL TESTS")
    print("-" * 80)
    control_rv = df[df["condition"] == "control"]["R_V"].dropna()
    h18_rv = df[df["condition"] == "h18_ablated"]["R_V"].dropna()
    h6_rv = df[df["condition"] == "h6_ablated"]["R_V"].dropna()
    both_rv = df[df["condition"] == "both_ablated"]["R_V"].dropna()
    
    if len(control_rv) > 1 and len(h18_rv) > 1:
        t, p = stats.ttest_ind(control_rv, h18_rv)
        print(f"Control vs H18-ablated R_V: t={t:.3f}, p={p:.4f}")
    
    if len(control_rv) > 1 and len(h6_rv) > 1:
        t, p = stats.ttest_ind(control_rv, h6_rv)
        print(f"Control vs H6-ablated R_V: t={t:.3f}, p={p:.4f}")
    
    # PART B: Expression Analysis
    print("\n[3/4] PART B: Expression Analysis...")
    part_b_results = part_b_expression_analysis(df)
    
    # PART C: Expression Head Hunt
    print("\n[4/4] PART C: Expression Head Hunt...")
    part_c_results = part_c_expression_head_hunt(model, tokenizer, df)
    
    # Save summary
    summary = {
        "part_a": {
            "n_prompts": len(df["prompt_id"].unique()),
            "n_conditions": len(df["condition"].unique()),
            "csv_path": str(csv_path),
        },
        "part_b": part_b_results,
        "part_c": {
            "heads_tested": len(part_c_results),
            "expression_enablers": [r["head"] for r in part_c_results if r["is_expression_enabler"]],
        },
    }
    
    import json
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✅ Summary saved to: {summary_path}")
    
    print("\n" + "=" * 80)
    print("✅ COMPREHENSIVE TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

