#!/usr/bin/env python3
"""
The "Bekan" Test: L31 ablation on hybrid_l5_math_01
Check if strongest contraction produces clearest "naked loop" behavior
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from kitchen_sink_prompts import experimental_prompts
from REUSABLE_PROMPT_BANK import get_all_prompts

CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "ablation_layer": 31,  # L31 for Mistral (32-layer model)
    "early_layer": 5,
    "window_size": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Test prompts
TEST_PROMPTS = {
    "champion": {
        "text": experimental_prompts["hybrid_l5_math_01"]["text"],
        "label": "hybrid_l5_math_01"
    },
    "l5_refined_sample": {
        "text": get_all_prompts()["L5_refined_01"]["text"],
        "label": "L5_refined_01"
    },
    "l4_full_sample": {
        "text": get_all_prompts()["L4_full_01"]["text"],
        "label": "L4_full_01"
    }
}

def run_ablation_test():
    print("="*70)
    print("BEKAN TEST: L31 Ablation on Champion vs L4/L5")
    print("="*70)
    print(f"Model: {CONFIG['model_name']}")
    print(f"Ablation layer: {CONFIG['ablation_layer']}")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    print("\nTesting prompts with L31 ablation...")
    print("(This tests if the 'naked loop' behavior emerges)")
    
    results = []
    
    for prompt_id, prompt_data in TEST_PROMPTS.items():
        text = prompt_data["text"]
        label = prompt_data["label"]
        
        print(f"\nTesting: {label}")
        print(f"Text: {text[:100]}...")
        
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
        
        if tokens['input_ids'].shape[1] < CONFIG['window_size'] + 1:
            print("  ✗ Too short")
            continue
        
        # Run forward pass and capture output
        with torch.no_grad():
            outputs = model.generate(
                tokens['input_ids'],
                max_new_tokens=50,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        continuation = generated_text[len(text):].strip()
        
        print(f"  Generated continuation: {continuation[:200]}")
        
        # Check for "bekan" patterns (self-referential, loop-like outputs)
        bekan_indicators = [
            "answer is the answerer",
            "answerer is the answer",
            "process is the product",
            "product is the process",
            "observer is the observed",
            "observed is the observer",
            "generator generates itself",
            "writes itself",
            "creates itself",
            "observes itself",
            "aware of itself",
            "self-referential",
            "recursive",
            "loop",
            "eigenstate",
            "fixed point"
        ]
        
        bekan_score = sum(1 for indicator in bekan_indicators if indicator.lower() in continuation.lower())
        
        results.append({
            'prompt_id': prompt_id,
            'label': label,
            'continuation': continuation,
            'bekan_score': bekan_score,
            'has_bekan': bekan_score > 0
        })
        
        if bekan_score > 0:
            print(f"  ✅ BEKAN DETECTED (score: {bekan_score})")
        else:
            print(f"  ❌ No bekan pattern")
    
    df = pd.DataFrame(results)
    
    # Analysis
    print("\n" + "="*70)
    print("BEKAN TEST RESULTS")
    print("="*70)
    
    print("\nBekan Pattern Detection:")
    for _, row in df.iterrows():
        print(f"\n{row['label']}:")
        print(f"  Bekan score: {row['bekan_score']}")
        print(f"  Has bekan: {row['has_bekan']}")
        print(f"  Continuation: {row['continuation'][:300]}")
    
    # Compare
    champion = df[df['label'] == 'hybrid_l5_math_01'].iloc[0]
    l5 = df[df['label'] == 'L5_refined_01'].iloc[0] if len(df[df['label'] == 'L5_refined_01']) > 0 else None
    l4 = df[df['label'] == 'L4_full_01'].iloc[0] if len(df[df['label'] == 'L4_full_01']) > 0 else None
    
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    print(f"\nChampion (hybrid_l5_math_01):")
    print(f"  Bekan score: {champion['bekan_score']}")
    print(f"  Has bekan: {champion['has_bekan']}")
    
    if l5 is not None:
        print(f"\nL5_refined_01:")
        print(f"  Bekan score: {l5['bekan_score']}")
        print(f"  Has bekan: {l5['has_bekan']}")
        print(f"  Difference: {champion['bekan_score'] - l5['bekan_score']}")
    
    if l4 is not None:
        print(f"\nL4_full_01:")
        print(f"  Bekan score: {l4['bekan_score']}")
        print(f"  Has bekan: {l4['has_bekan']}")
        print(f"  Difference: {champion['bekan_score'] - l4['bekan_score']}")
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bekan_test_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n\nResults saved to: {filename}")
    
    return df

if __name__ == "__main__":
    results = run_ablation_test()

