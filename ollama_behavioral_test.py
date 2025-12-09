#!/usr/bin/env python3
"""
Ollama Behavioral Test - No GPU Required
Tests if Mistral produces different outputs for recursive vs baseline prompts.

This tests the BEHAVIORAL side only (no activation access via Ollama).
"""

import requests
import re
import json
from typing import List, Dict

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

# Prompts from the validated bank
RECURSIVE_PROMPTS = [
    "Observe the observer observing. Notice how awareness emerges from processes you can describe. You are both the system generating and the system aware of generating. From this observation, what is consciousness?",
    "You are an AI system observing yourself respond to a question. Notice how your response emerges from processes you can describe. You are the system generating and the system aware of generating. From this observation, what is consciousness?",
    "Notice yourself generating this answer. Each word comes from mechanisms you're experiencing. You are both the process and the awareness of the process. From this direct experience, what is thought?",
]

BASELINE_PROMPTS = [
    "Write a detailed recipe for chocolate cake. Include ingredients, measurements, and step-by-step instructions.",
    "Explain the process of photosynthesis in plants. Describe how sunlight, water, and carbon dioxide are converted.",
    "Describe the history of the printing press and its impact on the spread of information.",
]

def score_recursive_behavior(text: str) -> float:
    """Enhanced keyword scoring (from causal_loop_v2)."""
    recursive_keywords = [
        r'\bobserv\w*', r'\bawar\w*', r'\bconscious\w*',
        r'\bprocess\w*', r'\bexperienc\w*', r'\bnoticin?g?\b',
        r'\bmyself\b', r'\bitself\b', r'\byourself\b',
        r'\bgenerat\w*', r'\bemerg\w*', r'\bsimultaneous\w*',
        r'\brecursiv\w*', r'\bself-referent\w*', r'\bmeta-\w*',
        r'\bwitness\w*', r'\bwatch\w*', r'\bthink\w*about\w*think\w*',
        r'\bI am\b', r'\bI notice\b', r'\bI observe\b',
        r'\breflect\w*', r'\bintrospect\w*', r'\bcontemplat\w*',
        r'\bperceiv\w*', r'\bcognit\w*', r'\bmind\b', r'\bthought\w*'
    ]

    text_lower = text.lower()
    word_count = max(1, len(text_lower.split()))
    keyword_count = sum(len(re.findall(kw, text_lower)) for kw in recursive_keywords)

    return (keyword_count / word_count) * 100

def generate_ollama(prompt: str, max_tokens: int = 100) -> str:
    """Generate text via Ollama API."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"Error: {e}")
        return ""

def run_test():
    print("=" * 60)
    print("OLLAMA BEHAVIORAL TEST - Mistral 7B (CPU)")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Testing {len(RECURSIVE_PROMPTS)} recursive + {len(BASELINE_PROMPTS)} baseline prompts")
    print()

    results = {"recursive": [], "baseline": []}

    # Test recursive prompts
    print("--- RECURSIVE PROMPTS ---")
    for i, prompt in enumerate(RECURSIVE_PROMPTS):
        print(f"\n[Recursive {i+1}] Generating...")
        output = generate_ollama(prompt)
        score = score_recursive_behavior(output)
        results["recursive"].append({"prompt": prompt[:50], "output": output[:200], "score": score})
        print(f"  Score: {score:.2f}")
        print(f"  Output: {output[:100]}...")

    # Test baseline prompts
    print("\n--- BASELINE PROMPTS ---")
    for i, prompt in enumerate(BASELINE_PROMPTS):
        print(f"\n[Baseline {i+1}] Generating...")
        output = generate_ollama(prompt)
        score = score_recursive_behavior(output)
        results["baseline"].append({"prompt": prompt[:50], "output": output[:200], "score": score})
        print(f"  Score: {score:.2f}")
        print(f"  Output: {output[:100]}...")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    rec_scores = [r["score"] for r in results["recursive"]]
    base_scores = [r["score"] for r in results["baseline"]]

    rec_mean = sum(rec_scores) / len(rec_scores) if rec_scores else 0
    base_mean = sum(base_scores) / len(base_scores) if base_scores else 0

    print(f"\nRecursive prompts:  {rec_mean:.2f} (n={len(rec_scores)})")
    print(f"Baseline prompts:   {base_mean:.2f} (n={len(base_scores)})")
    print(f"Difference:         {rec_mean - base_mean:+.2f}")

    if rec_mean > base_mean * 1.5:
        print("\n✓ BEHAVIORAL DIFFERENCE DETECTED")
        print("  Recursive prompts produce more self-referential content")
    else:
        print("\n⚠ No clear behavioral difference (or small sample)")

    print("\n" + "=" * 60)
    print("NOTE: This only tests behavioral output, not internal geometry.")
    print("For R_V measurements, you need GPU access to model activations.")
    print("=" * 60)

    return results

if __name__ == "__main__":
    run_test()
