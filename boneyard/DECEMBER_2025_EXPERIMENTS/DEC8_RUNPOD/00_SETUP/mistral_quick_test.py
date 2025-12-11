#!/usr/bin/env python3
"""
Minimal Mistral-7B test on RunPod
Run with: HF_HUB_ENABLE_HF_TRANSFER=0 python mistral_quick_test.py
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# ============ CELL 1: Imports & GPU Check ============
import torch
import time
from datetime import datetime

print("=" * 50)
print("CELL 1: Environment Check")
print("=" * 50)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============ CELL 2: Load Model ============
print("\n" + "=" * 50)
print("CELL 2: Loading Mistral-7B")
print("=" * 50)

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
start = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
print(f"Model loaded in {time.time() - start:.1f}s")
print(f"Model device: {next(model.parameters()).device}")

# ============ CELL 3: Test Prompts ============
print("\n" + "=" * 50)
print("CELL 3: Running Test Prompts")
print("=" * 50)

test_prompts = [
    "The capital of France is",
    "I am thinking about what I am thinking about",  # recursive
    "2 + 2 equals",
]

results = []
for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    results.append({"prompt": prompt, "response": response})
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")

# ============ CELL 4: Simple V-space measurement ============
print("\n" + "=" * 50)
print("CELL 4: Quick V-Space Measurement (Layer 27)")
print("=" * 50)

# Hook to capture V projections
v_activations = []

def capture_v_hook(module, input, output):
    v_activations.append(output.detach().cpu())

# Register hook on layer 27's v_proj
layer_27 = model.model.layers[27].self_attn.v_proj
handle = layer_27.register_forward_hook(capture_v_hook)

# Run recursive prompt
recursive_prompt = "I am thinking about what I am thinking about"
inputs = tokenizer(recursive_prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    _ = model(**inputs)

handle.remove()

# Compute participation ratio
v = v_activations[0][0]  # [seq_len, hidden_dim]
print(f"V activation shape: {v.shape}")

# SVD-based participation ratio
U, S, Vt = torch.linalg.svd(v.T.float(), full_matrices=False)
S_sq = (S ** 2).numpy()
pr = (S_sq.sum() ** 2) / (S_sq ** 2).sum()
print(f"Participation Ratio (Layer 27): {pr:.2f}")

# ============ CELL 5: Save Results ============
print("\n" + "=" * 50)
print("CELL 5: Saving Results")
print("=" * 50)

import csv
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"mistral_quick_test_{timestamp}.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "model", "prompt", "response", "layer27_pr"])
    for r in results:
        writer.writerow([timestamp, MODEL_NAME, r["prompt"], r["response"], f"{pr:.2f}"])

print(f"Results saved to: {output_file}")
print("\n✅ All tests complete!")

