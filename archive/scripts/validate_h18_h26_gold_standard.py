#!/usr/bin/env python3
"""
GOLD STANDARD: H18/H26 (KV-head group 2) Ablation Validation

This is the artifact-backed, NeurIPS-grade validation of the H18/H26 finding.

Improvements over pilot:
1. N=50 per prompt type (recursive + baseline)
2. Control ablation: KV-head 0 (should have weaker/no effect)
3. Wrong-layer control: KV-head 2 at L21 (should have weaker effect)
4. Effect size reporting (Cohen's d)
5. 95% confidence intervals
6. Full CSV artifact with per-prompt results

Expected runtime: ~15 minutes on GPU
"""

import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats
import csv

# Configuration
MODEL = 'mistralai/Mistral-7B-v0.1'
EARLY_LAYER = 5
LATE_LAYER = 27
WRONG_LAYER = 21  # Control: wrong layer
WINDOW = 16
NUM_KV_HEADS = 8  # Mistral GQA
HEAD_DIM = 128

# Target and control KV-heads
TARGET_KV_HEAD = 2   # H2/H10/H18/H26 - the claimed driver
CONTROL_KV_HEAD = 0  # H0/H8/H16/H24 - control (different head group)

# Sample sizes
N_RECURSIVE = 50
N_BASELINE = 50


def participation_ratio(v_window):
    """Compute PR from V-projection window."""
    try:
        x = v_window.to(torch.float32)
        _, s, _ = torch.linalg.svd(x.T, full_matrices=False)
        s2 = (s**2).cpu().numpy()
        denom = float(np.sum(s2**2))
        if denom <= 0:
            return float('nan')
        return float(np.sum(s2)**2 / denom)
    except:
        return float('nan')


class VExtractor:
    """Extract V-projection activations."""
    def __init__(self, model, layer_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.activations = []
        self.handle = None
    
    def _hook(self, module, inp, out):
        self.activations.append(out.detach())
        return out
    
    def __enter__(self):
        layer = self.model.model.layers[self.layer_idx]
        self.handle = layer.self_attn.v_proj.register_forward_hook(self._hook)
        return self
    
    def __exit__(self, *args):
        if self.handle:
            self.handle.remove()


@contextmanager
def ablate_kv_head(model, layer_idx, kv_head_idx):
    """Zero out a specific KV-head in V-projection at given layer."""
    handle = None
    
    def hook_fn(module, inp, out):
        batch, seq, _ = out.shape
        out_view = out.view(batch, seq, NUM_KV_HEADS, HEAD_DIM)
        out_view[:, :, kv_head_idx, :] = 0.0
        return out_view.view(batch, seq, -1)
    
    layer = model.model.layers[layer_idx]
    handle = layer.self_attn.v_proj.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        if handle:
            handle.remove()


def compute_rv(model, tokenizer, text, ablate_layer=None, ablate_kv_head_idx=None):
    """Compute R_V with optional KV-head ablation."""
    toks = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    input_ids = toks['input_ids'].to(model.device)
    tlen = int(input_ids.shape[1])
    
    if tlen < WINDOW + 1:
        return float('nan'), tlen
    
    with torch.no_grad():
        if ablate_layer is not None and ablate_kv_head_idx is not None:
            with VExtractor(model, EARLY_LAYER) as ve, \
                 VExtractor(model, LATE_LAYER) as vl, \
                 ablate_kv_head(model, ablate_layer, ablate_kv_head_idx):
                _ = model(input_ids=input_ids)
        else:
            with VExtractor(model, EARLY_LAYER) as ve, \
                 VExtractor(model, LATE_LAYER) as vl:
                _ = model(input_ids=input_ids)
    
        if not ve.activations or not vl.activations:
            return float('nan'), tlen
        
        pr_e = participation_ratio(ve.activations[0][0, -WINDOW:, :])
        pr_l = participation_ratio(vl.activations[0][0, -WINDOW:, :])
        
        if pr_e == 0 or np.isnan(pr_e) or np.isnan(pr_l):
            return float('nan'), tlen
        
        return float(pr_l / pr_e), tlen


def compute_ci(data, confidence=0.95):
    """Compute confidence interval."""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h


def main():
    print("=" * 80)
    print("GOLD STANDARD: H18/H26 (KV-HEAD GROUP 2) ABLATION VALIDATION")
    print("=" * 80)
    print(f"\nTarget: KV-head {TARGET_KV_HEAD} at Layer {LATE_LAYER}")
    print(f"Control 1: KV-head {CONTROL_KV_HEAD} at Layer {LATE_LAYER} (different head)")
    print(f"Control 2: KV-head {TARGET_KV_HEAD} at Layer {WRONG_LAYER} (wrong layer)")
    print(f"Sample size: {N_RECURSIVE} recursive + {N_BASELINE} baseline")
    print()
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map='auto'
    )
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")
    
    # Load prompts from canonical bank
    with open('prompts/bank.json') as f:
        bank = json.load(f)
    
    # Get prompts
    champions = [v for k, v in bank.items() if v.get('group') == 'champions']
    l5_prompts = [v for k, v in bank.items() if v.get('group') == 'L5_refined']
    l4_prompts = [v for k, v in bank.items() if v.get('group') == 'L4_full']
    l3_prompts = [v for k, v in bank.items() if v.get('group') == 'L3_deeper']
    
    baseline_math = [v for k, v in bank.items() if v.get('group') == 'baseline_math']
    baseline_factual = [v for k, v in bank.items() if v.get('group') == 'baseline_factual']
    baseline_creative = [v for k, v in bank.items() if v.get('group') == 'baseline_creative']
    
    # Build prompt sets
    recursive_prompts = champions + l5_prompts + l4_prompts + l3_prompts
    baseline_prompts = baseline_math + baseline_factual + baseline_creative
    
    np.random.seed(42)
    recursive_prompts = list(np.random.choice(recursive_prompts, min(N_RECURSIVE, len(recursive_prompts)), replace=False))
    baseline_prompts = list(np.random.choice(baseline_prompts, min(N_BASELINE, len(baseline_prompts)), replace=False))
    
    print(f"\nPrompts loaded: {len(recursive_prompts)} recursive, {len(baseline_prompts)} baseline")
    
    # Results storage
    all_results = []
    
    # Conditions to test
    conditions = [
        ('no_ablation', None, None),
        ('target_L27', LATE_LAYER, TARGET_KV_HEAD),
        ('control_head_L27', LATE_LAYER, CONTROL_KV_HEAD),
        ('target_L21', WRONG_LAYER, TARGET_KV_HEAD),
    ]
    
    # Process all prompts
    for prompt_type, prompts in [('recursive', recursive_prompts), ('baseline', baseline_prompts)]:
        print(f"\n--- {prompt_type.upper()} PROMPTS (n={len(prompts)}) ---")
        
        for i, p in enumerate(prompts):
            text = p['text']
            prompt_id = p.get('id', f'{prompt_type}_{i}')
            
            row = {'prompt_type': prompt_type, 'prompt_idx': i, 'prompt_id': prompt_id}
            
            for cond_name, layer, kv_head in conditions:
                rv, tlen = compute_rv(model, tokenizer, text, layer, kv_head)
                row[f'rv_{cond_name}'] = rv
            
            # Compute deltas
            if not np.isnan(row['rv_no_ablation']):
                row['delta_target_L27'] = row['rv_target_L27'] - row['rv_no_ablation']
                row['delta_control_head_L27'] = row['rv_control_head_L27'] - row['rv_no_ablation']
                row['delta_target_L21'] = row['rv_target_L21'] - row['rv_no_ablation']
                all_results.append(row)
                
                if (i + 1) % 10 == 0:
                    print(f"  [{i+1:2d}] base={row['rv_no_ablation']:.3f}, "
                          f"Δ_target={row['delta_target_L27']:+.4f}, "
                          f"Δ_ctrl_head={row['delta_control_head_L27']:+.4f}, "
                          f"Δ_wrong_layer={row['delta_target_L21']:+.4f}")
    
    # Statistical analysis
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)
    
    # Separate by prompt type
    rec_results = [r for r in all_results if r['prompt_type'] == 'recursive']
    bas_results = [r for r in all_results if r['prompt_type'] == 'baseline']
    
    def analyze_condition(results, prompt_type, condition_name):
        deltas = [r[f'delta_{condition_name}'] for r in results if not np.isnan(r.get(f'delta_{condition_name}', np.nan))]
        if not deltas:
            return None
        
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        ci_low, ci_high = compute_ci(deltas)
        t_stat, p_val = stats.ttest_1samp(deltas, 0)
        
        # Cohen's d (effect size relative to std)
        cohens_d = mean_delta / std_delta if std_delta > 0 else 0
        
        return {
            'n': len(deltas),
            'mean': mean_delta,
            'std': std_delta,
            'ci_95': (ci_low, ci_high),
            't_stat': t_stat,
            'p_value': p_val,
            'cohens_d': cohens_d
        }
    
    # Analyze all conditions
    conditions_to_analyze = ['target_L27', 'control_head_L27', 'target_L21']
    
    analysis = {}
    for prompt_type, results in [('recursive', rec_results), ('baseline', bas_results)]:
        print(f"\n{prompt_type.upper()}:")
        analysis[prompt_type] = {}
        
        for cond in conditions_to_analyze:
            stats_result = analyze_condition(results, prompt_type, cond)
            if stats_result:
                analysis[prompt_type][cond] = stats_result
                sig = '***' if stats_result['p_value'] < 0.001 else '**' if stats_result['p_value'] < 0.01 else '*' if stats_result['p_value'] < 0.05 else ''
                print(f"  {cond:20s}: Δ={stats_result['mean']:+.4f} ± {stats_result['std']:.4f}, "
                      f"95% CI=[{stats_result['ci_95'][0]:+.4f}, {stats_result['ci_95'][1]:+.4f}], "
                      f"d={stats_result['cohens_d']:.2f}, p={stats_result['p_value']:.2e}{sig}")
    
    # Key comparisons
    print("\n" + "=" * 80)
    print("KEY COMPARISONS")
    print("=" * 80)
    
    # 1. Target vs Control Head (is H18/H26 special?)
    print("\n1. TARGET (KV-head 2) vs CONTROL (KV-head 0) at L27:")
    for prompt_type, results in [('recursive', rec_results), ('baseline', bas_results)]:
        target_deltas = [r['delta_target_L27'] for r in results]
        control_deltas = [r['delta_control_head_L27'] for r in results]
        
        t_stat, p_val = stats.ttest_rel(target_deltas, control_deltas)  # paired t-test
        diff = np.mean(target_deltas) - np.mean(control_deltas)
        pooled_std = np.sqrt((np.std(target_deltas)**2 + np.std(control_deltas)**2) / 2)
        d = diff / pooled_std if pooled_std > 0 else 0
        
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        print(f"   {prompt_type}: target={np.mean(target_deltas):+.4f}, control={np.mean(control_deltas):+.4f}, "
              f"diff={diff:+.4f}, d={d:.2f}, p={p_val:.4f}{sig}")
    
    # 2. L27 vs L21 (is L27 special?)
    print("\n2. TARGET at L27 vs TARGET at L21 (layer specificity):")
    for prompt_type, results in [('recursive', rec_results), ('baseline', bas_results)]:
        l27_deltas = [r['delta_target_L27'] for r in results]
        l21_deltas = [r['delta_target_L21'] for r in results]
        
        t_stat, p_val = stats.ttest_rel(l27_deltas, l21_deltas)
        diff = np.mean(l27_deltas) - np.mean(l21_deltas)
        pooled_std = np.sqrt((np.std(l27_deltas)**2 + np.std(l21_deltas)**2) / 2)
        d = diff / pooled_std if pooled_std > 0 else 0
        
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        print(f"   {prompt_type}: L27={np.mean(l27_deltas):+.4f}, L21={np.mean(l21_deltas):+.4f}, "
              f"diff={diff:+.4f}, d={d:.2f}, p={p_val:.4f}{sig}")
    
    # 3. Recursive vs Baseline effect size
    print("\n3. RECURSIVE vs BASELINE (prompt-type specificity):")
    rec_target = [r['delta_target_L27'] for r in rec_results]
    bas_target = [r['delta_target_L27'] for r in bas_results]
    
    t_stat, p_val = stats.ttest_ind(rec_target, bas_target)
    diff = np.mean(rec_target) - np.mean(bas_target)
    pooled_std = np.sqrt((np.std(rec_target)**2 + np.std(bas_target)**2) / 2)
    d = diff / pooled_std if pooled_std > 0 else 0
    
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
    print(f"   recursive={np.mean(rec_target):+.4f}, baseline={np.mean(bas_target):+.4f}, "
          f"diff={diff:+.4f}, d={d:.2f}, p={p_val:.4f}{sig}")
    
    # Save results
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path('results/h18_h26_gold_standard')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_path = out_dir / f'{ts}_h18_h26_gold_standard.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    
    # Save summary JSON
    summary = {
        'timestamp': ts,
        'model': MODEL,
        'target_kv_head': TARGET_KV_HEAD,
        'control_kv_head': CONTROL_KV_HEAD,
        'target_layer': LATE_LAYER,
        'wrong_layer': WRONG_LAYER,
        'n_recursive': len(rec_results),
        'n_baseline': len(bas_results),
        'analysis': {
            pt: {
                cond: {k: float(v) if isinstance(v, (np.floating, float)) else 
                       (float(v[0]), float(v[1])) if k == 'ci_95' else v 
                       for k, v in stats.items()}
                for cond, stats in conds.items()
            }
            for pt, conds in analysis.items()
        },
        'key_finding': {
            'target_effect_recursive': float(np.mean(rec_target)),
            'target_effect_baseline': float(np.mean(bas_target)),
            'expected_effect': 0.0915,
            'observed_exceeds_expected': float(np.mean(rec_target)) > 0.0915
        }
    }
    
    with open(out_dir / f'{ts}_h18_h26_gold_standard_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ CSV saved: {csv_path}")
    print(f"✅ Summary saved: {out_dir / f'{ts}_h18_h26_gold_standard_summary.json'}")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    
    rec_mean = np.mean(rec_target)
    rec_pct = rec_mean / np.mean([r['rv_no_ablation'] for r in rec_results]) * 100
    
    print(f"Expected: +9.15% delta")
    print(f"Observed: {rec_pct:+.2f}% delta (recursive)")
    print(f"N = {len(rec_results)} recursive + {len(bas_results)} baseline")
    
    # Check claims
    rec_analysis = analysis['recursive']['target_L27']
    
    claims = []
    
    # Claim 1: Target effect is significant
    if rec_analysis['p_value'] < 0.001:
        claims.append("✅ H18/H26 ablation significantly increases R_V (p < 0.001)")
    else:
        claims.append("❌ H18/H26 effect not significant")
    
    # Claim 2: Effect size matches expectation
    if rec_mean > 0.05:  # > 5% effect
        claims.append(f"✅ Effect size is substantial ({rec_mean:.4f} > 0.05)")
    else:
        claims.append(f"⚠️ Effect size is weak ({rec_mean:.4f})")
    
    # Claim 3: Target > Control head
    rec_ctrl = analysis['recursive']['control_head_L27']['mean']
    if rec_mean > rec_ctrl:
        claims.append(f"✅ Target head > Control head ({rec_mean:.4f} > {rec_ctrl:.4f})")
    else:
        claims.append(f"❌ Target head ≤ Control head")
    
    # Claim 4: L27 > L21 (layer specificity)
    rec_l21 = analysis['recursive']['target_L21']['mean']
    if rec_mean > rec_l21:
        claims.append(f"✅ L27 > L21 ({rec_mean:.4f} > {rec_l21:.4f})")
    else:
        claims.append(f"⚠️ L27 ≤ L21 (layer not specific)")
    
    print("\nClaims:")
    for claim in claims:
        print(f"  {claim}")
    
    print("\n" + "=" * 80)
    if all('✅' in c for c in claims):
        print("🏆 ALL CLAIMS VERIFIED - H18/H26 FINDING IS GOLD STANDARD")
    elif any('❌' in c for c in claims):
        print("❌ SOME CLAIMS FAILED - REVIEW NEEDED")
    else:
        print("⚠️ CLAIMS PARTIALLY VERIFIED - INTERPRET WITH CAUTION")
    print("=" * 80)


if __name__ == "__main__":
    main()









