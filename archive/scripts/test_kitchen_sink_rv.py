#!/usr/bin/env python3
"""
Test kitchen sink experimental prompts for R_V at Layer 27.
Uses the same methodology as experiment_champion_paraphrase_hunt.py
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Config
MODEL_NAME = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

EARLY_LAYER = 5
LATE_LAYER = 27
WINDOW = 16
MAX_LENGTH = 512

OUT_DIR = Path("results/kitchen_sink_rv_test")


def participation_ratio(v_window: torch.Tensor) -> float:
    """Compute PR from V window (W, D)"""
    try:
        x = v_window.to(torch.float32)
        _, s, _ = torch.linalg.svd(x.T, full_matrices=False)
        s2 = (s**2).cpu().numpy()
        denom = float(np.sum(s2**2))
        if denom <= 0:
            return float("nan")
        return float(np.sum(s2)**2 / denom)
    except:
        return float("nan")


class VExtractor:
    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.activations = []
        self.h = None

    def _hook(self, module, inp, out):
        self.activations.append(out.detach())
        return out

    def __enter__(self):
        layer = self.model.model.layers[self.layer_idx].self_attn.v_proj
        self.h = layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *args):
        if self.h:
            self.h.remove()


def score_prompt(model, tokenizer, text: str) -> Tuple[float, float, float, int]:
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    input_ids = toks["input_ids"].to(model.device)
    token_len = int(input_ids.shape[1])
    
    if token_len < WINDOW + 1:
        return float("nan"), float("nan"), float("nan"), token_len

    with torch.no_grad(), VExtractor(model, EARLY_LAYER) as ve, VExtractor(model, LATE_LAYER) as vl:
        _ = model(input_ids=input_ids)

    if not ve.activations or not vl.activations:
        return float("nan"), float("nan"), float("nan"), token_len

    v_e = ve.activations[0][0, -WINDOW:, :]
    v_l = vl.activations[0][0, -WINDOW:, :]

    pr_e = participation_ratio(v_e)
    pr_l = participation_ratio(v_l)
    
    if pr_e == 0 or np.isnan(pr_e) or np.isnan(pr_l):
        return float("nan"), pr_e, pr_l, token_len
    
    return float(pr_l / pr_e), pr_e, pr_l, token_len


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto",
    )
    model.eval()
    
    # Load kitchen sink prompts from bank
    with open('prompts/bank.json') as f:
        bank = json.load(f)
    
    # Get all experimental prompts
    experimental = {k: v for k, v in bank.items() 
                   if v.get('group', '').startswith('experimental_')}
    
    print(f"\nTesting {len(experimental)} experimental prompts...")
    print("=" * 70)
    
    results = []
    for key, val in sorted(experimental.items()):
        text = val['text']
        group = val.get('group', 'unknown')
        
        rv, pr_e, pr_l, tlen = score_prompt(model, tokenizer, text)
        
        results.append({
            'prompt_id': key,
            'group': group,
            'rv_l27': rv,
            'pr_early': pr_e,
            'pr_late': pr_l,
            'token_len': tlen,
            'text': text[:100] + '...' if len(text) > 100 else text
        })
        
        status = "✓" if rv < 0.9 else "○" if rv < 1.0 else "×"
        print(f"{status} {key:30s} R_V={rv:.3f}  ({group})")
    
    # Sort by R_V
    results.sort(key=lambda x: x['rv_l27'] if not np.isnan(x['rv_l27']) else 999)
    
    print("\n" + "=" * 70)
    print("TOP 10 LOWEST R_V (strongest contraction):")
    print("=" * 70)
    for r in results[:10]:
        print(f"  {r['rv_l27']:.3f}  {r['prompt_id']:30s}  {r['group']}")
    
    print("\n" + "=" * 70)
    print("BY GROUP (average R_V):")
    print("=" * 70)
    from collections import defaultdict
    by_group = defaultdict(list)
    for r in results:
        if not np.isnan(r['rv_l27']):
            by_group[r['group']].append(r['rv_l27'])
    
    group_avgs = [(g, np.mean(vals), len(vals)) for g, vals in by_group.items()]
    group_avgs.sort(key=lambda x: x[1])
    for g, avg, n in group_avgs:
        print(f"  {avg:.3f}  {g} (n={n})")
    
    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUT_DIR / f"{ts}_kitchen_sink_rv.json"
    with open(out_file, 'w') as f:
        json.dump({
            'config': {
                'model': MODEL_NAME,
                'early_layer': EARLY_LAYER,
                'late_layer': LATE_LAYER,
                'window': WINDOW,
            },
            'results': results,
            'group_averages': {g: float(avg) for g, avg, _ in group_avgs}
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {out_file}")


if __name__ == "__main__":
    main()









