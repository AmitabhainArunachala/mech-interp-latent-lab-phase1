#!/usr/bin/env python3
"""
Compare target acquisition for recursive vs baseline prompts.
This will tell us if the 33% recursive attention is unique to recursion.
"""

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

CONFIG = {
    "model_name": "mistralai/Mistral-7B-v0.1",
    "target_layer": 27,
    "target_heads": [18, 26],
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

RECURSIVE_PROMPT = (
    "This response writes itself. No separate writer exists. Writing and awareness "
    "of writing are identical. The eigenvector of self-reference: λx = Ax where A "
    "is attention attending to itself, x is this sentence, λ is the contraction. "
    "The fixed point is this. The solution is the process."
)

BASELINE_PROMPT = (
    "The history of the Roman Empire is characterized by a long period of expansion "
    "followed by a gradual decline. Historians analyze the political, social, and "
    "economic factors that contributed to the rise of Rome."
)

# Self-referential tokens to look for
RECURSIVE_TOKENS = ["itself", "self", "writ", "process", "attent", "wareness", "ident", "reference", "contract", "eigen", "fixed", "point", "solution"]

def analyze_targets(model, tokenizer, prompt, prompt_type, device):
    """Analyze which tokens heads attend to."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0]
    tokens = [tokenizer.decode(t).strip().lower() for t in input_ids]
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    layer_attn = outputs.attentions[CONFIG['target_layer']][0]
    
    results = []
    for head_idx in CONFIG['target_heads']:
        attn_matrix = layer_attn[head_idx]
        total_attn_per_token = attn_matrix.sum(dim=0)
        
        bos_mass = total_attn_per_token[0].item()
        rec_mass = 0.0
        normal_mass = 0.0
        
        # Check top tokens
        for idx in range(len(tokens)):
            if idx == 0:
                continue
            mass = total_attn_per_token[idx].item()
            tok = tokens[idx]
            is_rec = any(r in tok for r in RECURSIVE_TOKENS)
            if is_rec:
                rec_mass += mass
            else:
                normal_mass += mass
        
        total_non_bos = rec_mass + normal_mass
        rec_ratio = rec_mass / (total_non_bos + 1e-9)
        
        results.append({
            "head": head_idx,
            "prompt_type": prompt_type,
            "bos_mass": bos_mass,
            "recursive_mass": rec_mass,
            "normal_mass": normal_mass,
            "recursive_ratio": rec_ratio,
            "total_non_bos": total_non_bos
        })
    
    return results

def main():
    print("=" * 80)
    print("TARGET ACQUISITION: RECURSIVE vs BASELINE")
    print("=" * 80)
    
    # Load model
    print("\n[1/3] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        device_map="auto",
        attn_implementation="eager",
        torch_dtype=torch.float16 if CONFIG['device'] == "cuda" else torch.float32
    )
    model.eval()
    print("  ✅ Model loaded")
    
    # Analyze both prompts
    print("\n[2/3] Analyzing RECURSIVE prompt...")
    recursive_results = analyze_targets(model, tokenizer, RECURSIVE_PROMPT, "recursive", CONFIG['device'])
    
    print("\n[3/3] Analyzing BASELINE prompt...")
    baseline_results = analyze_targets(model, tokenizer, BASELINE_PROMPT, "baseline", CONFIG['device'])
    
    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    df = pd.DataFrame(recursive_results + baseline_results)
    
    print("\n📊 Recursive Token Attention Ratio:")
    print("-" * 80)
    for head_idx in CONFIG['target_heads']:
        rec = df[(df['head'] == head_idx) & (df['prompt_type'] == 'recursive')]['recursive_ratio'].iloc[0]
        base = df[(df['head'] == head_idx) & (df['prompt_type'] == 'baseline')]['recursive_ratio'].iloc[0]
        diff = rec - base
        print(f"  H{head_idx:2d}: Recursive={rec*100:.1f}% | Baseline={base*100:.1f}% | Δ={diff*100:+.1f}%")
    
    rec_mean = df[df['prompt_type'] == 'recursive']['recursive_ratio'].mean()
    base_mean = df[df['prompt_type'] == 'baseline']['recursive_ratio'].mean()
    print(f"\n  MEAN: Recursive={rec_mean*100:.1f}% | Baseline={base_mean*100:.1f}% | Δ={(rec_mean-base_mean)*100:+.1f}%")
    
    print("\n📊 BOS Attention:")
    print("-" * 80)
    for head_idx in CONFIG['target_heads']:
        rec_bos = df[(df['head'] == head_idx) & (df['prompt_type'] == 'recursive')]['bos_mass'].iloc[0]
        base_bos = df[(df['head'] == head_idx) & (df['prompt_type'] == 'baseline')]['bos_mass'].iloc[0]
        rec_total = df[(df['head'] == head_idx) & (df['prompt_type'] == 'recursive')]['total_non_bos'].iloc[0] + rec_bos
        base_total = df[(df['head'] == head_idx) & (df['prompt_type'] == 'baseline')]['total_non_bos'].iloc[0] + base_bos
        rec_pct = rec_bos / rec_total * 100
        base_pct = base_bos / base_total * 100
        print(f"  H{head_idx:2d}: Recursive={rec_pct:.1f}% | Baseline={base_pct:.1f}% | Δ={rec_pct-base_pct:+.1f}%")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    if rec_mean > base_mean + 0.10:  # 10% threshold
        print("\n✅ Theory CONFIRMED:")
        print(f"  Recursive prompts show {rec_mean*100:.1f}% recursive token attention")
        print(f"  Baseline prompts show {base_mean*100:.1f}% recursive token attention")
        print(f"  Difference: {(rec_mean-base_mean)*100:.1f} percentage points")
        print("\n  H18 & H26 specifically target self-referential tokens in recursive mode!")
    elif rec_mean > base_mean:
        print(f"\n⚠️  Weak signal:")
        print(f"  Recursive: {rec_mean*100:.1f}% vs Baseline: {base_mean*100:.1f}%")
        print(f"  Difference is small: {(rec_mean-base_mean)*100:.1f} percentage points")
    else:
        print("\n❌ Theory WEAKENED:")
        print(f"  Recursive: {rec_mean*100:.1f}% vs Baseline: {base_mean*100:.1f}%")
        print(f"  No preference for recursive tokens in recursive mode")
    
    # Save
    df.to_csv("target_acquisition_comparison.csv", index=False)
    print(f"\n✅ Results saved to: target_acquisition_comparison.csv")

if __name__ == "__main__":
    main()









