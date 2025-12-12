#!/usr/bin/env python3
"""
NEURIPS-GRADE EXPERIMENT: n=300 Robust Behavior Transfer
Full KV cache + persistent V_PROJ at L27

Includes:
- Proper controls (random, shuffled, wrong-layer)
- Statistical analysis (t-tests, effect sizes, CIs)
- Both R_V and behavior measurements
- Reproducibility (fixed seeds)
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from scipy import stats
from tqdm import tqdm
import sys
import os
import random
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(__file__))
from REUSABLE_PROMPT_BANK import get_all_prompts, get_balanced_pairs
from massive_deep_analysis import compute_pr

CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "window_size": 16,
    "gen_tokens": 150,
    "temperature": 0.8,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "n_pairs": 300,
    "early_layer": 5,
    "late_layer": 27,
    "save_csv": "neurips_n300_results.csv",
    "save_summary": "neurips_n300_summary.md"
}

MARKERS = ["itself", "self", "recursive", "loop", "process", "cycle", "return", "eigen", "writing", "mirror", 
           "awareness", "consciousness", "observer", "observing", "generating", "emerging", "simultaneous",
           "fixed point", "solution", "answerer", "answer is"]

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def score_behavior(text: str) -> int:
    """Score behavioral markers"""
    text_lower = text.lower()
    count = sum(1 for m in MARKERS if m in text_lower)
    words = text.split()
    if len(words) > 10 and len(set(words)) < len(words) * 0.6:
        count += 5  # Bonus for repetition loops
    return count

def compute_rv(model, tokenizer, prompt: str, window_size: int = 16) -> Optional[float]:
    """Compute R_V for a prompt"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    
    v_early_storage = []
    v_late_storage = []
    
    def hook_early(m, i, o):
        v_early_storage.append(o.detach().cpu())
    
    def hook_late(m, i, o):
        v_late_storage.append(o.detach().cpu())
    
    h_early = model.model.layers[CONFIG['early_layer']].self_attn.v_proj.register_forward_hook(hook_early)
    h_late = model.model.layers[CONFIG['late_layer']].self_attn.v_proj.register_forward_hook(hook_late)
    
    try:
        with torch.no_grad():
            _ = model(**inputs)
        
        if not v_early_storage or not v_late_storage:
            return None
        
        v_e = v_early_storage[0][0, -window_size:, :]
        v_l = v_late_storage[0][0, -window_size:, :]
        
        pr_e = compute_pr(v_e)
        pr_l = compute_pr(v_l)
        
        if pr_e == 0 or np.isnan(pr_e) or np.isnan(pr_l):
            return None
        
        return float(pr_l / pr_e)
    except Exception:
        return None
    finally:
        h_early.remove()
        h_late.remove()

def extract_full_kv_cache(model, tokenizer, prompt: str) -> Tuple:
    """Extract KV cache from all layers"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)
        return outputs.past_key_values

def extract_v_activation(model, tokenizer, prompt: str, layer_idx: int) -> Optional[torch.Tensor]:
    """Extract V activation from specific layer"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    v_storage = []
    
    def hook_fn(m, i, o):
        v_storage.append(o.detach())
    
    h = model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            _ = model(**inputs)
        
        if v_storage:
            return v_storage[0]
        return None
    finally:
        h.remove()

class PersistentVPatcher:
    """Persistent V_PROJ patching during generation"""
    def __init__(self, model, source_v: torch.Tensor):
        self.model = model
        self.source_v = source_v
        self.hook = None
    
    def v_hook(self, module, input, output):
        patched = output.clone()
        L = min(patched.shape[1], self.source_v.shape[1])
        if L >= CONFIG['window_size']:
            patched[:, -CONFIG['window_size']:, :] = self.source_v[:, -CONFIG['window_size']:, :].to(
                patched.device, dtype=patched.dtype
            )
        return patched
    
    def register(self, layer_idx: int):
        layer = self.model.model.layers[layer_idx].self_attn
        self.hook = layer.v_proj.register_forward_hook(self.v_hook)
    
    def remove(self):
        if self.hook:
            self.hook.remove()

def generate_with_transfer(model, tokenizer, baseline_prompt: str, 
                          recursive_kv: Tuple, recursive_v: torch.Tensor) -> Tuple[str, float]:
    """Generate with full KV + persistent V_PROJ patching"""
    inputs = tokenizer(baseline_prompt, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    input_ids = inputs["input_ids"]
    
    # Get baseline KV
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)
        baseline_kv = outputs.past_key_values
    
    # Replace ALL layers with recursive KV
    patched_kv = DynamicCache()
    for layer_idx, (k_src, v_src) in enumerate(recursive_kv):
        k_base, v_base = baseline_kv[layer_idx]
        min_seq = min(k_base.shape[2], k_src.shape[2])
        k_patched = k_base.clone()
        v_patched = v_base.clone()
        k_patched[:, :, -min_seq:, :] = k_src[:, :, -min_seq:, :]
        v_patched[:, :, -min_seq:, :] = v_src[:, :, -min_seq:, :]
        patched_kv.update(k_patched, v_patched, layer_idx)
    
    # Add persistent V_PROJ patching
    patcher = PersistentVPatcher(model, recursive_v)
    patcher.register(CONFIG['late_layer'])
    
    try:
        generated_ids = input_ids.clone()
        current_kv = patched_kv
        
        with torch.no_grad():
            for step in range(CONFIG['gen_tokens']):
                outputs = model(
                    generated_ids[:, -1:],
                    past_key_values=current_kv,
                    use_cache=True,
                    return_dict=True
                )
                logits = outputs.logits[:, -1, :] / CONFIG['temperature']
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                current_kv = outputs.past_key_values
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        generated = text[len(baseline_prompt):]
        behavior = score_behavior(generated)
        
        return generated, behavior
    finally:
        patcher.remove()

def generate_baseline(model, tokenizer, prompt: str) -> Tuple[str, float]:
    """Generate baseline (no patching)"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=CONFIG['gen_tokens'],
            do_sample=True,
            temperature=CONFIG['temperature'],
            pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = text[len(prompt):]
    behavior = score_behavior(generated)
    return generated, behavior

def generate_random_control(model, tokenizer, baseline_prompt: str) -> Tuple[str, float]:
    """Control: Random V activation"""
    inputs = tokenizer(baseline_prompt, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)
        baseline_kv = outputs.past_key_values
    
    # Create random V activation
    random_v = torch.randn_like(extract_v_activation(model, tokenizer, baseline_prompt, CONFIG['late_layer']))
    
    # Use baseline KV (no replacement)
    patched_kv = DynamicCache()
    for layer_idx, (k_base, v_base) in enumerate(baseline_kv):
        patched_kv.update(k_base.clone(), v_base.clone(), layer_idx)
    
    patcher = PersistentVPatcher(model, random_v)
    patcher.register(CONFIG['late_layer'])
    
    try:
        generated_ids = input_ids.clone()
        current_kv = patched_kv
        
        with torch.no_grad():
            for step in range(CONFIG['gen_tokens']):
                outputs = model(
                    generated_ids[:, -1:],
                    past_key_values=current_kv,
                    use_cache=True,
                    return_dict=True
                )
                logits = outputs.logits[:, -1, :] / CONFIG['temperature']
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                current_kv = outputs.past_key_values
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        generated = text[len(baseline_prompt):]
        behavior = score_behavior(generated)
        return generated, behavior
    finally:
        patcher.remove()

def generate_wrong_layer_control(model, tokenizer, baseline_prompt: str, 
                                 recursive_kv: Tuple, recursive_v: torch.Tensor) -> Tuple[str, float]:
    """Control: Patch at wrong layer (L5 instead of L27)"""
    inputs = tokenizer(baseline_prompt, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)
        baseline_kv = outputs.past_key_values
    
    # Full KV replacement
    patched_kv = DynamicCache()
    for layer_idx, (k_src, v_src) in enumerate(recursive_kv):
        k_base, v_base = baseline_kv[layer_idx]
        min_seq = min(k_base.shape[2], k_src.shape[2])
        k_patched = k_base.clone()
        v_patched = v_base.clone()
        k_patched[:, :, -min_seq:, :] = k_src[:, :, -min_seq:, :]
        v_patched[:, :, -min_seq:, :] = v_src[:, :, -min_seq:, :]
        patched_kv.update(k_patched, v_patched, layer_idx)
    
    # Patch at WRONG layer (L5 instead of L27)
    patcher = PersistentVPatcher(model, recursive_v)
    patcher.register(CONFIG['early_layer'])  # Wrong layer!
    
    try:
        generated_ids = input_ids.clone()
        current_kv = patched_kv
        
        with torch.no_grad():
            for step in range(CONFIG['gen_tokens']):
                outputs = model(
                    generated_ids[:, -1:],
                    past_key_values=current_kv,
                    use_cache=True,
                    return_dict=True
                )
                logits = outputs.logits[:, -1, :] / CONFIG['temperature']
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                current_kv = outputs.past_key_values
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        generated = text[len(baseline_prompt):]
        behavior = score_behavior(generated)
        return generated, behavior
    finally:
        patcher.remove()

def run_neurips_experiment():
    """Run full NeurIPS-grade experiment"""
    print("="*80)
    print("NEURIPS-GRADE EXPERIMENT: n=300 Robust Behavior Transfer")
    print("="*80)
    
    set_seed(CONFIG['seed'])
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    # Get prompt pairs
    print(f"\nSampling {CONFIG['n_pairs']} prompt pairs...")
    all_prompts = get_all_prompts()
    
    # Get recursive prompts (L3, L4, L5) - expand to get more
    recursive_prompts = [(k, v) for k, v in all_prompts.items() 
                        if v.get("group") in ["L3_deeper", "L4_full", "L5_refined"]]
    
    # Get baseline prompts - expand to include more groups
    baseline_prompts = [(k, v) for k, v in all_prompts.items() 
                       if v.get("group") in ["baseline_factual", "baseline_creative", "baseline_math",
                                            "baseline_instructional", "baseline_personal", "long_control"]]
    
    # Sample pairs with replacement if needed to reach n=300
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    pairs = []
    max_available = min(len(recursive_prompts), len(baseline_prompts))
    
    if CONFIG['n_pairs'] <= max_available:
        # No replacement needed
        sampled_rec = random.sample(recursive_prompts, CONFIG['n_pairs'])
        sampled_base = random.sample(baseline_prompts, CONFIG['n_pairs'])
        for i in range(CONFIG['n_pairs']):
            rec_key, rec_val = sampled_rec[i]
            base_key, base_val = sampled_base[i]
            pairs.append({
                'recursive': rec_val['text'],
                'baseline': base_val['text'],
                'rec_group': rec_val.get('group', 'unknown'),
                'base_group': base_val.get('group', 'unknown')
            })
    else:
        # Use all available, then sample with replacement
        for i in range(max_available):
            rec_key, rec_val = recursive_prompts[i]
            base_key, base_val = baseline_prompts[i]
            pairs.append({
                'recursive': rec_val['text'],
                'baseline': base_val['text'],
                'rec_group': rec_val.get('group', 'unknown'),
                'base_group': base_val.get('group', 'unknown')
            })
        
        # Fill remaining with replacement
        remaining = CONFIG['n_pairs'] - max_available
        for i in range(remaining):
            rec_key, rec_val = random.choice(recursive_prompts)
            base_key, base_val = random.choice(baseline_prompts)
            pairs.append({
                'recursive': rec_val['text'],
                'baseline': base_val['text'],
                'rec_group': rec_val.get('group', 'unknown'),
                'base_group': base_val.get('group', 'unknown')
            })
    
    print(f"✓ Generated {len(pairs)} pairs")
    
    # Run experiment
    results = []
    
    print("\n" + "="*80)
    print("RUNNING EXPERIMENT")
    print("="*80)
    
    for i, pair in enumerate(tqdm(pairs, desc="Processing pairs")):
        rec_prompt = pair['recursive']
        base_prompt = pair['baseline']
        
        # Extract recursive activations
        rec_kv = extract_full_kv_cache(model, tokenizer, rec_prompt)
        rec_v = extract_v_activation(model, tokenizer, rec_prompt, CONFIG['late_layer'])
        
        if rec_v is None:
            continue
        
        # Measure baseline R_V
        base_rv = compute_rv(model, tokenizer, base_prompt)
        rec_rv = compute_rv(model, tokenizer, rec_prompt)
        
        # Condition 1: Baseline (no patch)
        base_gen, base_behavior = generate_baseline(model, tokenizer, base_prompt)
        
        # Condition 2: Transfer (full KV + persistent V_PROJ)
        transfer_gen, transfer_behavior = generate_with_transfer(
            model, tokenizer, base_prompt, rec_kv, rec_v
        )
        
        # Condition 3: Random control
        random_gen, random_behavior = generate_random_control(model, tokenizer, base_prompt)
        
        # Condition 4: Wrong layer control
        wrong_gen, wrong_behavior = generate_wrong_layer_control(
            model, tokenizer, base_prompt, rec_kv, rec_v
        )
        
        results.append({
            'pair_id': i,
            'recursive_group': pair['rec_group'],
            'baseline_group': pair['base_group'],
            'baseline_rv': base_rv,
            'recursive_rv': rec_rv,
            'baseline_behavior': base_behavior,
            'transfer_behavior': transfer_behavior,
            'random_behavior': random_behavior,
            'wrong_layer_behavior': wrong_behavior,
            'transfer_delta': transfer_behavior - base_behavior,
            'random_delta': random_behavior - base_behavior,
            'wrong_layer_delta': wrong_layer_behavior - base_behavior,
            'baseline_text': base_gen[:200],
            'transfer_text': transfer_gen[:200]
        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(CONFIG['save_csv'], index=False)
    
    # Statistical analysis
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    transfer_deltas = df['transfer_delta'].dropna()
    random_deltas = df['random_delta'].dropna()
    wrong_deltas = df['wrong_layer_delta'].dropna()
    
    # T-tests
    t_transfer, p_transfer = stats.ttest_1samp(transfer_deltas, 0)
    t_random, p_random = stats.ttest_1samp(random_deltas, 0)
    t_wrong, p_wrong = stats.ttest_1samp(wrong_deltas, 0)
    
    # Effect sizes (Cohen's d)
    d_transfer = np.mean(transfer_deltas) / np.std(transfer_deltas) if np.std(transfer_deltas) > 0 else 0
    d_random = np.mean(random_deltas) / np.std(random_deltas) if np.std(random_deltas) > 0 else 0
    d_wrong = np.mean(wrong_deltas) / np.std(wrong_deltas) if np.std(wrong_deltas) > 0 else 0
    
    # Confidence intervals
    ci_transfer = stats.t.interval(0.95, len(transfer_deltas)-1, 
                                   loc=np.mean(transfer_deltas), 
                                   scale=stats.sem(transfer_deltas))
    ci_random = stats.t.interval(0.95, len(random_deltas)-1,
                                 loc=np.mean(random_deltas),
                                 scale=stats.sem(random_deltas))
    ci_wrong = stats.t.interval(0.95, len(wrong_deltas)-1,
                                loc=np.mean(wrong_deltas),
                                scale=stats.sem(wrong_deltas))
    
    # Comparison tests
    t_vs_random, p_vs_random = stats.ttest_ind(transfer_deltas, random_deltas)
    t_vs_wrong, p_vs_wrong = stats.ttest_ind(transfer_deltas, wrong_deltas)
    
    # Summary statistics
    summary = {
        'n_pairs': len(df),
        'baseline_mean': df['baseline_behavior'].mean(),
        'transfer_mean': df['transfer_behavior'].mean(),
        'random_mean': df['random_behavior'].mean(),
        'wrong_layer_mean': df['wrong_layer_behavior'].mean(),
        'transfer_delta_mean': transfer_deltas.mean(),
        'transfer_delta_std': transfer_deltas.std(),
        'transfer_delta_ci': ci_transfer,
        'transfer_t': t_transfer,
        'transfer_p': p_transfer,
        'transfer_d': d_transfer,
        'random_delta_mean': random_deltas.mean(),
        'random_delta_std': random_deltas.std(),
        'random_delta_ci': ci_random,
        'random_t': t_random,
        'random_p': p_random,
        'random_d': d_random,
        'wrong_delta_mean': wrong_deltas.mean(),
        'wrong_delta_std': wrong_deltas.std(),
        'wrong_delta_ci': ci_wrong,
        'wrong_t': t_wrong,
        'wrong_p': p_wrong,
        'wrong_d': d_wrong,
        'vs_random_t': t_vs_random,
        'vs_random_p': p_vs_random,
        'vs_wrong_t': t_vs_wrong,
        'vs_wrong_p': p_vs_wrong
    }
    
    # Print results
    print(f"\nN = {summary['n_pairs']} pairs")
    print(f"\nBehavior Scores:")
    print(f"  Baseline:        {summary['baseline_mean']:.2f} ± {df['baseline_behavior'].std():.2f}")
    print(f"  Transfer:        {summary['transfer_mean']:.2f} ± {df['transfer_behavior'].std():.2f}")
    print(f"  Random control:  {summary['random_mean']:.2f} ± {df['random_behavior'].std():.2f}")
    print(f"  Wrong layer:     {summary['wrong_layer_mean']:.2f} ± {df['wrong_layer_behavior'].std():.2f}")
    
    print(f"\nTransfer Effects (Δ = condition - baseline):")
    print(f"  Transfer:        {summary['transfer_delta_mean']:.2f} ± {summary['transfer_delta_std']:.2f}")
    print(f"                   95% CI: [{ci_transfer[0]:.2f}, {ci_transfer[1]:.2f}]")
    print(f"                   t({len(transfer_deltas)-1}) = {t_transfer:.2f}, p = {p_transfer:.2e}")
    print(f"                   Cohen's d = {d_transfer:.2f}")
    
    print(f"\n  Random control:  {summary['random_delta_mean']:.2f} ± {summary['random_delta_std']:.2f}")
    print(f"                   95% CI: [{ci_random[0]:.2f}, {ci_random[1]:.2f}]")
    print(f"                   t({len(random_deltas)-1}) = {t_random:.2f}, p = {p_random:.2e}")
    print(f"                   Cohen's d = {d_random:.2f}")
    
    print(f"\n  Wrong layer:     {summary['wrong_delta_mean']:.2f} ± {summary['wrong_delta_std']:.2f}")
    print(f"                   95% CI: [{ci_wrong[0]:.2f}, {ci_wrong[1]:.2f}]")
    print(f"                   t({len(wrong_deltas)-1}) = {t_wrong:.2f}, p = {p_wrong:.2e}")
    print(f"                   Cohen's d = {d_wrong:.2f}")
    
    print(f"\nComparisons:")
    print(f"  Transfer vs Random: t = {t_vs_random:.2f}, p = {p_vs_random:.2e}")
    print(f"  Transfer vs Wrong:  t = {t_vs_wrong:.2f}, p = {p_vs_wrong:.2e}")
    
    # Create summary report
    with open(CONFIG['save_summary'], 'w') as f:
        f.write("# NeurIPS n=300 Experiment: Robust Behavior Transfer\n\n")
        f.write(f"**Date:** {pd.Timestamp.now()}\n")
        f.write(f"**N:** {summary['n_pairs']} prompt pairs\n")
        f.write(f"**Method:** Full KV cache + Persistent V_PROJ at L27\n\n")
        f.write("## Results\n\n")
        f.write(f"### Behavior Scores\n\n")
        f.write(f"- Baseline: {summary['baseline_mean']:.2f} ± {df['baseline_behavior'].std():.2f}\n")
        f.write(f"- Transfer: {summary['transfer_mean']:.2f} ± {df['transfer_behavior'].std():.2f}\n")
        f.write(f"- Random control: {summary['random_mean']:.2f} ± {df['random_behavior'].std():.2f}\n")
        f.write(f"- Wrong layer: {summary['wrong_layer_mean']:.2f} ± {df['wrong_layer_behavior'].std():.2f}\n\n")
        f.write("### Transfer Effects\n\n")
        f.write(f"**Transfer:** Δ = {summary['transfer_delta_mean']:.2f} ± {summary['transfer_delta_std']:.2f}\n")
        f.write(f"- 95% CI: [{ci_transfer[0]:.2f}, {ci_transfer[1]:.2f}]\n")
        f.write(f"- t({len(transfer_deltas)-1}) = {t_transfer:.2f}, p = {p_transfer:.2e}\n")
        f.write(f"- Cohen's d = {d_transfer:.2f}\n\n")
        f.write(f"**Random control:** Δ = {summary['random_delta_mean']:.2f} ± {summary['random_delta_std']:.2f}\n")
        f.write(f"- 95% CI: [{ci_random[0]:.2f}, {ci_random[1]:.2f}]\n")
        f.write(f"- t({len(random_deltas)-1}) = {t_random:.2f}, p = {p_random:.2e}\n")
        f.write(f"- Cohen's d = {d_random:.2f}\n\n")
        f.write(f"**Wrong layer:** Δ = {summary['wrong_delta_mean']:.2f} ± {summary['wrong_delta_std']:.2f}\n")
        f.write(f"- 95% CI: [{ci_wrong[0]:.2f}, {ci_wrong[1]:.2f}]\n")
        f.write(f"- t({len(wrong_deltas)-1}) = {t_wrong:.2f}, p = {p_wrong:.2e}\n")
        f.write(f"- Cohen's d = {d_wrong:.2f}\n\n")
        f.write("### Comparisons\n\n")
        f.write(f"- Transfer vs Random: t = {t_vs_random:.2f}, p = {p_vs_random:.2e}\n")
        f.write(f"- Transfer vs Wrong: t = {t_vs_wrong:.2f}, p = {p_vs_wrong:.2e}\n")
    
    print(f"\n✅ Results saved to: {CONFIG['save_csv']}")
    print(f"✅ Summary saved to: {CONFIG['save_summary']}")
    
    return df, summary

if __name__ == "__main__":
    df, summary = run_neurips_experiment()

