#!/usr/bin/env python3
"""
Simple test to debug head discovery pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.core.models import load_model, set_seed
from prompts.loader import PromptLoader

# Test basic functionality
print("Testing basic setup...")
set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load model
print("\n[1] Loading model...")
model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=device, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
print("✅ Model loaded")

# Load prompts
print("\n[2] Loading prompts...")
loader = PromptLoader()
recursive = loader.get_by_pillar("recursive", limit=2, seed=42)
baseline = loader.get_by_pillar("baseline", limit=2, seed=42)
print(f"✅ Recursive: {len(recursive)} prompts")
print(f"✅ Baseline: {len(baseline)} prompts")
if recursive:
    print(f"   Example: {recursive[0][:60]}...")
if baseline:
    print(f"   Example: {baseline[0][:60]}...")

# Test attention capture
print("\n[3] Testing attention capture...")
prompt = recursive[0] if recursive else "Test prompt"
enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

with torch.no_grad():
    outputs = model(**enc, output_attentions=True)
    print(f"✅ Forward pass complete")
    print(f"   Output type: {type(outputs)}")
    if hasattr(outputs, 'attentions'):
        print(f"   Attentions: {len(outputs.attentions) if outputs.attentions else 0} layers")
        if outputs.attentions and len(outputs.attentions) > 0:
            attn = outputs.attentions[27]  # L27
            print(f"   L27 attention shape: {attn.shape}")
            print(f"   L27 attention type: {type(attn)}")
    else:
        print("   ⚠️ No attentions in output")

print("\n✅ Basic test complete!")

