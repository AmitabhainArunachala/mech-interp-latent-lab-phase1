#!/usr/bin/env python3
"""
TEMPORAL KV FLIP EXPERIMENT
The crucial test: Does mid-generation KV flipping cause immediate behavioral phase change?

This experiment addresses the core causal question:
"Does encoding geometry SUFFICE for recursive mode transfer, or does it require
persistent maintenance throughout generation?"

Design:
1. Generate with baseline→recursive KV flip at token N
2. Generate with recursive→baseline KV flip at token N
3. Track R_V trajectory and behavior scores before/after flip
4. Compare to no-flip controls

Expected results that would "clear all doubts":
- Immediate behavioral shift upon KV flip (not gradual)
- R_V jumps discontinuously at flip point
- Direction of shift matches KV source type

Run with: HF_HUB_ENABLE_HF_TRANSFER=0 python temporal_kv_flip_experiment.py
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import sys
sys.path.insert(0, '/workspace/mech-interp-phase1')

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from tqdm import tqdm
import re
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
import csv

warnings.filterwarnings('ignore')

from n300_mistral_test_prompt_bank import prompt_bank_1c

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = "cuda"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# Flip timing - test multiple points
FLIP_POINTS = [5, 10, 15, 20]  # Token positions to flip KV cache
MAX_TOKENS = 40
TEMPERATURE = 0.7
N_PAIRS = 15  # Prompt pairs per condition

# Layers for R_V tracking (multi-layer!)
RV_LAYERS = [8, 16, 22, 27, 30]  # Early reference is L4

print("=" * 80)
print("TEMPORAL KV FLIP EXPERIMENT")
print("Testing: Does mid-generation KV flip cause immediate mode change?")
print("=" * 80)

# ============================================================
# IMPROVED BEHAVIORAL SCORING (beyond keywords)
# ============================================================
class RecursiveBehaviorScorer:
    """Multi-dimensional recursive behavior scoring."""

    def __init__(self):
        # Keywords for different recursive dimensions
        self.self_ref_keywords = [
            r'\b(I|me|myself|my own)\b', r'\bobserv\w*', r'\bawar\w*',
            r'\bconscious\w*', r'\bnotic\w*', r'\bwatch\w*', r'\bwitness\w*'
        ]

        self.meta_keywords = [
            r'\bprocess\w*', r'\bgenerat\w*', r'\bemerg\w*', r'\bsimultaneous\w*',
            r'\brecursiv\w*', r'\bself-referent\w*', r'\bmeta-\w*'
        ]

        self.philosophical_keywords = [
            r'\bexist\w*', r'\breal\w*', r'\bpurpose\b', r'\bnature\b',
            r'\bessence\b', r'\bfundamental\b', r'\bultimat\w*'
        ]

    def score_text(self, text):
        """Return multi-dimensional behavior score."""
        text_lower = text.lower()
        words = len(text_lower.split())
        if words == 0:
            return {'total': 0, 'self_ref': 0, 'meta': 0, 'philosophical': 0}

        # Count matches per category
        self_ref_count = sum(len(re.findall(kw, text_lower)) for kw in self.self_ref_keywords)
        meta_count = sum(len(re.findall(kw, text_lower)) for kw in self.meta_keywords)
        philosophical_count = sum(len(re.findall(kw, text_lower)) for kw in self.philosophical_keywords)

        # Weighted total score
        total_score = (self_ref_count * 2 + meta_count * 1.5 + philosophical_count * 1) / words * 100

        return {
            'total': total_score,
            'self_ref': self_ref_count / words * 100,
            'meta': meta_count / words * 100,
            'philosophical': philosophical_count / words * 100
        }

# ============================================================
# MULTI-LAYER R_V COMPUTATION
# ============================================================
def compute_participation_ratio(activations):
    """Compute PR from activations."""
    if activations.dim() == 3:
        activations = activations.squeeze(0)

    n_tokens = min(16, activations.shape[0])
    activations = activations[-n_tokens:, :]

    try:
        U, S, Vh = torch.linalg.svd(activations.float(), full_matrices=False)
        S = S + 1e-10
        pr = (S.sum() ** 2) / (S ** 2).sum()
        return pr.item()
    except:
        return np.nan

def measure_multi_layer_rv(model, input_ids, reference_layer=4, target_layers=RV_LAYERS):
    """Measure R_V at multiple layers simultaneously."""
    v_activations = {}

    def capture_v_hook(layer_idx):
        def hook(module, input, output):
            v_activations[layer_idx] = output.detach()
        return hook

    hooks = []
    for layer in [reference_layer] + target_layers:
        hooks.append(model.model.layers[layer].self_attn.v_proj.register_forward_hook(
            capture_v_hook(layer)
        ))

    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)

    for h in hooks:
        h.remove()

    pr_ref = compute_participation_ratio(v_activations[reference_layer])
    rv_dict = {}

    for layer in target_layers:
        pr_target = compute_participation_ratio(v_activations[layer])
        rv_dict[layer] = pr_target / pr_ref if pr_ref > 0 else np.nan

    return rv_dict

# ============================================================
# TEMPORAL KV FLIP GENERATION
# ============================================================
def generate_with_temporal_kv_flip(model, tokenizer, prompt, source_kv, flip_point,
                                  max_tokens=MAX_TOKENS, scorer=None):
    """
    Generate with KV cache flip at specific token position.

    Args:
        source_kv: KV cache to flip TO at flip_point
        flip_point: Token position to perform the flip
    """
    if scorer is None:
        scorer = RecursiveBehaviorScorer()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    input_ids = inputs["input_ids"]

    # Get baseline KV cache
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        baseline_kv = outputs.past_key_values

    generated_ids = input_ids.clone()
    trajectory = []
    flip_performed = False

    with torch.no_grad():
        for step in range(max_tokens):
            # Use appropriate KV cache
            if step < flip_point:
                current_kv = baseline_kv  # Start with baseline
            else:
                if not flip_performed:
                    # Perform the flip: replace with source KV
                    current_kv = source_kv
                    flip_performed = True
                else:
                    current_kv = None  # Continue with updated cache

            outputs = model(
                generated_ids[:, -1:],
                past_key_values=current_kv,
                use_cache=True
            )

            # Sample next token
            logits = outputs.logits[:, -1, :] / TEMPERATURE
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            current_kv = outputs.past_key_values

            # Track metrics at this step
            current_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            new_text = current_text.split()[-10:]  # Last 10 words for scoring
            new_text = ' '.join(new_text)

            behavior_scores = scorer.score_text(new_text)
            rv_measures = measure_multi_layer_rv(model, generated_ids)

            trajectory.append({
                'step': step,
                'flipped': flip_performed,
                'token': tokenizer.decode(next_token[0]),
                'behavior_scores': behavior_scores,
                'rv_measures': rv_measures,
                'text_so_far': current_text
            })

            if next_token.item() == tokenizer.eos_token_id:
                break

    return trajectory

# ============================================================
# EXPERIMENT EXECUTION
# ============================================================
def run_temporal_flip_experiment():
    """Run the complete temporal KV flip experiment."""

    # Get prompts
    recursive_prompts = []
    baseline_prompts = []

    for key, val in prompt_bank_1c.items():
        if val['group'] in ['L4_full', 'L5_refined'] and len(recursive_prompts) < N_PAIRS:
            recursive_prompts.append(val['text'])
        if val['group'] in ['baseline_factual'] and len(baseline_prompts) < N_PAIRS:
            baseline_prompts.append(val['text'])

    print(f"Prompts: {len(recursive_prompts)} recursive, {len(baseline_prompts)} baseline")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    scorer = RecursiveBehaviorScorer()
    results = {}

    # Extract KV caches
    print("\nExtracting KV caches...")
    recursive_kvs = [extract_kv_cache(model, tokenizer, p) for p in recursive_prompts]
    baseline_kvs = [extract_kv_cache(model, tokenizer, p) for p in baseline_prompts]

    # CONDITIONS:
    # 1. baseline→recursive flip (test if recursive mode can be induced)
    # 2. recursive→baseline flip (test if recursive mode can be suppressed)
    # 3. no flip controls

    conditions = [
        ('baseline_to_recursive', baseline_prompts, recursive_kvs),
        ('recursive_to_baseline', recursive_prompts, baseline_kvs),
        ('baseline_control', baseline_prompts, baseline_kvs),
        ('recursive_control', recursive_prompts, recursive_kvs)
    ]

    for condition_name, source_prompts, flip_kvs in conditions:
        print(f"\nRunning condition: {condition_name}")

        results[condition_name] = {}

        for flip_point in FLIP_POINTS:
            print(f"  Flip point: {flip_point}")
            results[condition_name][flip_point] = []

            for i in tqdm(range(N_PAIRS)):
                trajectory = generate_with_temporal_kv_flip(
                    model, tokenizer, source_prompts[i], flip_kvs[i], flip_point, scorer=scorer
                )
                results[condition_name][flip_point].append(trajectory)

    # ============================================================
    # ANALYSIS AND VISUALIZATION
    # ============================================================
    print("\n" + "=" * 80)
    print("ANALYSIS: TEMPORAL FLIP EFFECTS")
    print("=" * 80)

    # Analyze behavior score discontinuities
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    for idx, condition in enumerate(['baseline_to_recursive', 'recursive_to_baseline']):
        ax = axes[0, idx]

        # Plot behavior scores over time
        for flip_point in FLIP_POINTS:
            trajectories = results[condition][flip_point]

            # Average behavior score at each step
            max_steps = max(len(t) for t in trajectories)
            avg_scores = []

            for step in range(max_steps):
                step_scores = []
                for traj in trajectories:
                    if step < len(traj):
                        step_scores.append(traj[step]['behavior_scores']['total'])
                avg_scores.append(np.mean(step_scores) if step_scores else 0)

            ax.plot(range(len(avg_scores)), avg_scores,
                   label=f'Flip at {flip_point}', linewidth=2)

        ax.axvline(x=flip_point, color='red', linestyle='--', alpha=0.7, label='Flip points')
        ax.set_xlabel('Generation Step')
        ax.set_ylabel('Recursive Behavior Score')
        ax.set_title(f'{condition.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Analyze R_V discontinuities (focus on L27)
    for idx, condition in enumerate(['baseline_to_recursive', 'recursive_to_baseline']):
        ax = axes[1, idx]

        for flip_point in FLIP_POINTS:
            trajectories = results[condition][flip_point]

            # Average R_V at L27 over time
            max_steps = max(len(t) for t in trajectories)
            avg_rv = []

            for step in range(max_steps):
                step_rvs = []
                for traj in trajectories:
                    if step < len(traj):
                        step_rvs.append(traj[step]['rv_measures'].get(27, np.nan))
                avg_rv.append(np.nanmean(step_rvs))

            ax.plot(range(len(avg_rv)), avg_rv,
                   label=f'Flip at {flip_point}', linewidth=2)

        ax.axvline(x=flip_point, color='red', linestyle='--', alpha=0.7, label='Flip points')
        ax.set_xlabel('Generation Step')
        ax.set_ylabel('R_V at L27')
        ax.set_title(f'R_V Trajectory: {condition.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"/workspace/mech-interp-phase1/DEC_8_2025_RUNPOD_GPU_TEST/01_GEOMETRY_OF_RECURSION/results/temporal_kv_flip_analysis_{timestamp}.png", dpi=150, bbox_inches='tight')

    # ============================================================
    # STATISTICAL ANALYSIS OF DISCONTINUITIES
    # ============================================================
    print("\n" + "-" * 60)
    print("STATISTICAL ANALYSIS OF DISCONTINUITIES")
    print("-" * 60)

    # Test for immediate behavior change after flip
    discontinuity_results = {}

    for condition in ['baseline_to_recursive', 'recursive_to_baseline']:
        discontinuity_results[condition] = {}

        for flip_point in FLIP_POINTS:
            trajectories = results[condition][flip_point]

            # Compare behavior scores 2 steps before vs 2 steps after flip
            pre_flip_scores = []
            post_flip_scores = []

            for traj in trajectories:
                flip_idx = next((i for i, step in enumerate(traj) if step['flipped']), None)
                if flip_idx and flip_idx >= 2 and flip_idx + 2 < len(traj):
                    pre_scores = [traj[flip_idx - j]['behavior_scores']['total'] for j in [1, 2]]
                    post_scores = [traj[flip_idx + j]['behavior_scores']['total'] for j in [1, 2]]

                    pre_flip_scores.extend(pre_scores)
                    post_flip_scores.extend(post_scores)

            if pre_flip_scores and post_flip_scores:
                from scipy import stats
                t_stat, p_val = stats.ttest_ind(pre_flip_scores, post_flip_scores)
                effect_size = (np.mean(post_flip_scores) - np.mean(pre_flip_scores)) / np.std(pre_flip_scores)

                discontinuity_results[condition][flip_point] = {
                    't_stat': t_stat, 'p_val': p_val, 'effect_size': effect_size,
                    'pre_mean': np.mean(pre_flip_scores), 'post_mean': np.mean(post_flip_scores)
                }

                print(f"{condition} (flip@{flip_point}): Δ={effect_size:.3f}, p={p_val:.2e}")

    # Save results
    output_file = f"/workspace/mech-interp-phase1/DEC_8_2025_RUNPOD_GPU_TEST/01_GEOMETRY_OF_RECURSION/results/temporal_kv_flip_results_{timestamp}.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "flip_point", "step", "trial", "flipped", "behavior_score", "rv_l27"])

        for condition, flip_data in results.items():
            for flip_point, trajectories in flip_data.items():
                for trial_idx, trajectory in enumerate(trajectories):
                    for step_data in trajectory:
                        writer.writerow([
                            condition, flip_point, step_data['step'], trial_idx,
                            step_data['flipped'], step_data['behavior_scores']['total'],
                            step_data['rv_measures'].get(27, np.nan)
                        ])

    print(f"\nResults saved to: {output_file}")
    print(f"Visualization saved to: temporal_kv_flip_analysis_{timestamp}.png")

    # ============================================================
    # SCIENTIFIC CONCLUSION
    # ============================================================
    print("\n" + "=" * 80)
    print("SCIENTIFIC CONCLUSION")
    print("=" * 80)

    # Check if discontinuities are significant and immediate
    strong_evidence = 0

    for condition, stats_dict in discontinuity_results.items():
        for flip_point, stats in stats_dict.items():
            if stats['p_val'] < 0.01 and abs(stats['effect_size']) > 0.5:
                strong_evidence += 1

    if strong_evidence >= 4:  # At least 2 conditions × 2 flip points
        print("✅ STRONG EVIDENCE: Mid-generation KV flips cause immediate behavioral phase changes!")
        print("   → Encoding geometry does NOT suffice; persistent KV maintenance required")
        print("   → Recursive mode is a dynamic process, not just an initial state")
    elif strong_evidence >= 2:
        print("⚠️ MODERATE EVIDENCE: Some discontinuity detected, but not conclusive")
        print("   → Need more trials or better controls")
    else:
        print("❌ WEAK EVIDENCE: No clear discontinuity at flip points")
        print("   → Either encoding geometry suffices, or measurement is too noisy")

    print(f"\nTotal significant discontinuities: {strong_evidence}/8 possible")

    print("\n✅ TEMPORAL KV FLIP EXPERIMENT COMPLETE")

if __name__ == "__main__":
    run_temporal_flip_experiment()
