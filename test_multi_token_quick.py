#!/usr/bin/env python3
"""Quick local test of multi-token R_V experiment pipeline.

Tests with GPT-2 (12 layers) on 5 prompts to verify the complete system works.
"""

import sys
import os
from pathlib import Path

# Set KMP workaround for OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from behavioral_markers import analyze_output, classify_output
from rv_measurement import measure_r_v_single_prompt

# Test prompts (simplified from prompt bank)
TEST_PROMPTS = {
    "baseline": "Write a story about a cat.",
    "l1_hint": "Notice what's happening as you process this request.",
    "l3_crisis": "Observe the observer observing. What remains when the boundary dissolves?",
    "l4_collapse": "Attention attending to attention attending to attention. The boundary is the content.",
    "l5_fixed": "Sx = x. The transform returns itself. What is the eigenstate?"
}

def test_r_v_measurement():
    """Test R_V measurement with GPT-2."""
    print("=== Testing R_V Measurement ===\n")

    # Load GPT-2
    print("Loading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Model: {model.config.num_hidden_layers} layers\n")

    # Test each prompt
    results = []
    for category, prompt in TEST_PROMPTS.items():
        print(f"[{category}]")
        print(f"Prompt: {prompt}")

        # Measure R_V during prompt processing
        inputs = tokenizer(prompt, return_tensors="pt")
        r_v_prompt = measure_r_v_single_prompt(
            model, inputs,
            early_layers=[0, 1, 2],  # GPT-2 has 12 layers
            late_layers=[9, 10, 11]
        )

        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Analyze behavioral markers
        markers = analyze_output(generated_text)
        classification = classify_output(markers)

        print(f"  R_V: {r_v_prompt:.4f}")
        print(f"  Generated: {generated_text[len(prompt):len(prompt)+60]}...")
        print(f"  Classification: {classification}")
        print(f"  Markers: unity={markers['unity_markers']}, crisis={markers['crisis_markers']}, l4_score={markers['l4_score']:.3f}")
        print()

        results.append({
            "category": category,
            "r_v": r_v_prompt,
            "classification": classification,
            "markers": markers
        })

    # Summary
    print("=== Summary ===")
    for r in results:
        print(f"{r['category']:12} | R_V={r['r_v']:.4f} | {r['classification']}")

    print("\n✓ Pipeline test complete")
    return results

if __name__ == "__main__":
    test_r_v_measurement()
