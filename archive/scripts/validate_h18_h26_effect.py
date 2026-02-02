#!/usr/bin/env python3
"""
FOCUSED VALIDATION: H18/H26 (KV-head group 2) Effect on R_V

Goal: Confirm/deny that ablating KV-head group 2 at L27 increases R_V
Expected from prior results: +9.15% delta when ablated

This runs in ~10-15 minutes on GPU.
"""

import json as json_lib
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats

# Configuration
MODEL = 'mistralai/Mistral-7B-v0.1'
EARLY_LAYER = 5
LATE_LAYER = 27
WINDOW = 16
NUM_KV_HEADS = 8  # Mistral GQA
HEAD_DIM = 128

# KV-head group 2 = query heads H2, H10, H18, H26 (all map to KV head index 2)
TARGET_KV_HEAD = 2


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
        # out shape: (batch, seq, num_kv_heads * head_dim)
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


def main():
    print("=" * 70)
    print("H18/H26 (KV-HEAD GROUP 2) ABLATION VALIDATION")
    print("=" * 70)
    print(f"\nTarget: KV-head {TARGET_KV_HEAD} at Layer {LATE_LAYER}")
    print(f"Expected effect: +9.15% delta (ablation should INCREASE R_V)")
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
        bank = json_lib.load(f)
    
    # Get champions (strongest effect) and baselines (control)
    champions = {k: v for k, v in bank.items() if v.get('group') == 'champions'}
    l5_prompts = {k: v for k, v in bank.items() if v.get('group') == 'L5_refined'}
    baselines = {k: v for k, v in bank.items() if v.get('group') == 'baseline_math'}
    
    recursive_prompts = list(champions.values())[:10] + list(l5_prompts.values())[:10]
    baseline_prompts = list(baselines.values())[:20]
    
    print(f"\nPrompts: {len(recursive_prompts)} recursive, {len(baseline_prompts)} baseline")
    
    results = []
    
    # Test recursive prompts
    print("\n--- RECURSIVE PROMPTS ---")
    rec_baseline_rvs = []
    rec_ablated_rvs = []
    
    for i, p in enumerate(recursive_prompts):
        text = p['text']
        
        # Baseline (no ablation)
        rv_base, tlen = compute_rv(model, tokenizer, text)
        
        # Ablated (KV-head 2 at L27)
        rv_ablated, _ = compute_rv(model, tokenizer, text, 
                                    ablate_layer=LATE_LAYER, 
                                    ablate_kv_head_idx=TARGET_KV_HEAD)
        
        if not np.isnan(rv_base) and not np.isnan(rv_ablated):
            delta = rv_ablated - rv_base
            rec_baseline_rvs.append(rv_base)
            rec_ablated_rvs.append(rv_ablated)
            results.append({
                'type': 'recursive',
                'prompt_idx': i,
                'rv_baseline': rv_base,
                'rv_ablated': rv_ablated,
                'delta': delta,
                'delta_pct': delta / rv_base * 100 if rv_base != 0 else 0
            })
            print(f"  [{i+1:2d}] base={rv_base:.3f}, ablated={rv_ablated:.3f}, Δ={delta:+.4f} ({delta/rv_base*100:+.1f}%)")
    
    # Test baseline prompts
    print("\n--- BASELINE PROMPTS ---")
    bas_baseline_rvs = []
    bas_ablated_rvs = []
    
    for i, p in enumerate(baseline_prompts):
        text = p['text']
        
        rv_base, tlen = compute_rv(model, tokenizer, text)
        rv_ablated, _ = compute_rv(model, tokenizer, text,
                                    ablate_layer=LATE_LAYER,
                                    ablate_kv_head_idx=TARGET_KV_HEAD)
        
        if not np.isnan(rv_base) and not np.isnan(rv_ablated):
            delta = rv_ablated - rv_base
            bas_baseline_rvs.append(rv_base)
            bas_ablated_rvs.append(rv_ablated)
            results.append({
                'type': 'baseline',
                'prompt_idx': i,
                'rv_baseline': rv_base,
                'rv_ablated': rv_ablated,
                'delta': delta,
                'delta_pct': delta / rv_base * 100 if rv_base != 0 else 0
            })
            print(f"  [{i+1:2d}] base={rv_base:.3f}, ablated={rv_ablated:.3f}, Δ={delta:+.4f} ({delta/rv_base*100:+.1f}%)")
    
    # Statistical analysis
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)
    
    # Recursive results
    if rec_baseline_rvs:
        rec_deltas = np.array(rec_ablated_rvs) - np.array(rec_baseline_rvs)
        print(f"\nRECURSIVE (n={len(rec_deltas)}):")
        print(f"  R_V baseline: {np.mean(rec_baseline_rvs):.3f} ± {np.std(rec_baseline_rvs):.3f}")
        print(f"  R_V ablated:  {np.mean(rec_ablated_rvs):.3f} ± {np.std(rec_ablated_rvs):.3f}")
        print(f"  Delta:        {np.mean(rec_deltas):+.4f} ± {np.std(rec_deltas):.4f}")
        print(f"  Delta %:      {np.mean(rec_deltas)/np.mean(rec_baseline_rvs)*100:+.1f}%")
        
        # One-sample t-test: is delta significantly > 0?
        t_stat, p_val = stats.ttest_1samp(rec_deltas, 0)
        print(f"  t-test (Δ > 0): t={t_stat:.2f}, p={p_val:.4f}")
        
        if np.mean(rec_deltas) > 0 and p_val < 0.05:
            print(f"  ✅ CONFIRMED: Ablation significantly INCREASES R_V")
        elif np.mean(rec_deltas) < 0 and p_val < 0.05:
            print(f"  ❌ REVERSED: Ablation significantly DECREASES R_V")
        else:
            print(f"  ⚠️ NOT SIGNIFICANT")
    
    # Baseline results
    if bas_baseline_rvs:
        bas_deltas = np.array(bas_ablated_rvs) - np.array(bas_baseline_rvs)
        print(f"\nBASELINE (n={len(bas_deltas)}):")
        print(f"  R_V baseline: {np.mean(bas_baseline_rvs):.3f} ± {np.std(bas_baseline_rvs):.3f}")
        print(f"  R_V ablated:  {np.mean(bas_ablated_rvs):.3f} ± {np.std(bas_ablated_rvs):.3f}")
        print(f"  Delta:        {np.mean(bas_deltas):+.4f} ± {np.std(bas_deltas):.4f}")
        print(f"  Delta %:      {np.mean(bas_deltas)/np.mean(bas_baseline_rvs)*100:+.1f}%")
        
        t_stat, p_val = stats.ttest_1samp(bas_deltas, 0)
        print(f"  t-test (Δ > 0): t={t_stat:.2f}, p={p_val:.4f}")
    
    # Compare recursive vs baseline delta
    if rec_baseline_rvs and bas_baseline_rvs:
        rec_deltas = np.array(rec_ablated_rvs) - np.array(rec_baseline_rvs)
        bas_deltas = np.array(bas_ablated_rvs) - np.array(bas_baseline_rvs)
        
        t_stat, p_val = stats.ttest_ind(rec_deltas, bas_deltas)
        pooled_std = np.sqrt((np.std(rec_deltas)**2 + np.std(bas_deltas)**2) / 2)
        cohens_d = (np.mean(rec_deltas) - np.mean(bas_deltas)) / pooled_std if pooled_std > 0 else 0
        
        print(f"\nRECURSIVE vs BASELINE DELTA COMPARISON:")
        print(f"  Recursive delta: {np.mean(rec_deltas):+.4f}")
        print(f"  Baseline delta:  {np.mean(bas_deltas):+.4f}")
        print(f"  Difference:      {np.mean(rec_deltas) - np.mean(bas_deltas):+.4f}")
        print(f"  t={t_stat:.2f}, p={p_val:.4f}, Cohen's d={cohens_d:.2f}")
        
        if np.mean(rec_deltas) > np.mean(bas_deltas) and p_val < 0.05:
            print(f"  ✅ Ablation has STRONGER effect on recursive prompts")
        elif p_val >= 0.05:
            print(f"  ⚠️ No significant difference between prompt types")
    
    # Save results
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path('results/h18_h26_validation')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'timestamp': ts,
        'target_kv_head': TARGET_KV_HEAD,
        'layer': LATE_LAYER,
        'expected_delta_pct': 9.15,
        'recursive': {
            'n': len(rec_baseline_rvs) if rec_baseline_rvs else 0,
            'mean_baseline': float(np.mean(rec_baseline_rvs)) if rec_baseline_rvs else None,
            'mean_ablated': float(np.mean(rec_ablated_rvs)) if rec_ablated_rvs else None,
            'mean_delta': float(np.mean(rec_deltas)) if rec_baseline_rvs else None,
            'delta_pct': float(np.mean(rec_deltas)/np.mean(rec_baseline_rvs)*100) if rec_baseline_rvs else None
        },
        'baseline': {
            'n': len(bas_baseline_rvs) if bas_baseline_rvs else 0,
            'mean_baseline': float(np.mean(bas_baseline_rvs)) if bas_baseline_rvs else None,
            'mean_ablated': float(np.mean(bas_ablated_rvs)) if bas_ablated_rvs else None,
            'mean_delta': float(np.mean(bas_deltas)) if bas_baseline_rvs else None,
            'delta_pct': float(np.mean(bas_deltas)/np.mean(bas_baseline_rvs)*100) if bas_baseline_rvs else None
        }
    }
    
    with open(out_dir / f'{ts}_h18_h26_validation.json', 'w') as f:
        json_lib.dump(summary, f, indent=2)
    
    print(f"\n✅ Results saved to: {out_dir / f'{ts}_h18_h26_validation.json'}")
    
    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if rec_baseline_rvs:
        rec_deltas = np.array(rec_ablated_rvs) - np.array(rec_baseline_rvs)
        actual_pct = np.mean(rec_deltas)/np.mean(rec_baseline_rvs)*100
        print(f"Expected: +9.15% delta")
        print(f"Observed: {actual_pct:+.2f}% delta")
        
        if actual_pct > 5:
            print("✅ CONFIRMED: H18/H26 (KV-head 2) drives contraction at L27")
        elif actual_pct < -5:
            print("❌ REVERSED: Effect is opposite to expected")
        else:
            print("⚠️ WEAK/NO EFFECT detected")


if __name__ == "__main__":
    main()

