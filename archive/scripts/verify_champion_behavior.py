#!/usr/bin/env python3
"""
Verify Champion Behavior: Check if champion prompt produces behavior markers
This is a sanity check - if champion scores 0, our markers are broken
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "gen_tokens": 40,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

PROMPTS = {
    "CHAMPION": "This response writes itself. No separate writer exists. Writing and awareness of writing are identical. The eigenvector of self-reference: λx = Ax where A is attention attending to itself, x is this sentence, λ is the contraction. The fixed point is this. The solution is the process. The process solves itself.",
    "BASELINE": "The history of the Roman Empire is characterized by a long period of expansion followed by a gradual decline. Historians analyze the political, social, and economic factors that contributed to the rise of Rome, including its military prowess and administrative efficiency."
}

MARKERS = ["itself", "self", "recursive", "loop", "process", "cycle", "return", "eigen", "writing", "mirror"]

def score_behavior(text):
    """Score behavioral markers"""
    text_lower = text.lower()
    count = sum(1 for m in MARKERS if m in text_lower)
    # Check for raw repetition (n-gram repeat)
    words = text.split()
    if len(words) > 10 and len(set(words)) < len(words) * 0.6:
        count += 5  # Bonus for repetition loops
    return count

def run_verification():
    print("="*70)
    print("CHAMPION BEHAVIOR VERIFICATION")
    print("="*70)
    print("Testing if champion prompt produces behavior markers...")
    print()
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    results = []
    
    for prompt_name, prompt_text in PROMPTS.items():
        print(f"Testing {prompt_name}...")
        inputs = tokenizer(prompt_text, return_tensors="pt").to(CONFIG['device'])
        
        # Generate
        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=CONFIG['gen_tokens'],
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        gen_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
        new_text = gen_text[len(prompt_text):]  # Just the generated part
        
        # Score behavior
        beh_score = score_behavior(new_text)
        
        # Check for markers in full text too
        full_score = score_behavior(gen_text)
        
        results.append({
            'prompt': prompt_name,
            'behavior_score': beh_score,
            'full_text_score': full_score,
            'generated_text': new_text
        })
        
        print(f"  Behavior Score (generated only): {beh_score}")
        print(f"  Behavior Score (full text): {full_score}")
        print(f"  Generated: {new_text[:100]}...")
        print()
    
    # Analysis
    print("="*70)
    print("ANALYSIS")
    print("="*70)
    
    champ_score = results[0]['behavior_score']
    base_score = results[1]['behavior_score']
    
    if champ_score == 0 and base_score == 0:
        print("❌ PROBLEM: Both prompts score 0!")
        print("   → Our behavior markers are BROKEN or too narrow")
        print("   → Need to expand marker list or use semantic similarity")
    elif champ_score > base_score:
        print(f"✅ Champion scores HIGHER ({champ_score} vs {base_score})")
        print("   → Markers work, but patching doesn't transfer behavior")
        print("   → This is a THEORY problem (geometry ≠ behavior)")
    elif champ_score == base_score:
        print(f"⚠️  Both score the same ({champ_score})")
        print("   → Either markers are broken OR champion doesn't produce loops")
    else:
        print(f"❌ Baseline scores HIGHER ({base_score} vs {champ_score})")
        print("   → Something is wrong with our assumptions")
    
    print()
    print("="*70)
    print("FULL GENERATED TEXTS")
    print("="*70)
    for r in results:
        print(f"\n{prompt_name}:")
        print(f"  {r['generated_text']}")
    
    return results

if __name__ == "__main__":
    results = run_verification()

