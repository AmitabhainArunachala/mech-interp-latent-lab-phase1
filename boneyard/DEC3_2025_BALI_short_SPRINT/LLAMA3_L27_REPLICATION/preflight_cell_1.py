# ============================================================================
# LLAMA-3-8B L27 REPLICATION - CELL 1: SETUP & PRE-FLIGHT
# ============================================================================
# Date: December 3, 2025
# Purpose: Replicate Mistral-7B L27 causal patching on Llama-3-8B
# ============================================================================

import torch
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 70)
print("LLAMA-3-8B L27 REPLICATION - PRE-FLIGHT CHECKS")
print("=" * 70)
print(f"Timestamp: {datetime.now()}")
print()

# Check GPU
print("1. GPU CHECK")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

# Model config
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DTYPE = torch.float16

print("2. LOADING MODEL")
print(f"   Model: {MODEL_NAME}")
print(f"   Dtype: {DTYPE}")
print("   Loading... (this may take 2-3 minutes)")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=DTYPE,
    attn_implementation="eager"
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("   ✓ Model loaded")
print()

# Architecture verification
print("3. ARCHITECTURE VERIFICATION")
num_layers = len(model.model.layers)
print(f"   Layer count: {num_layers} (expected: 32)")

has_v_proj = hasattr(model.model.layers[27].self_attn, 'v_proj')
print(f"   Layer 27 v_proj exists: {has_v_proj}")

# Verify hook works
print("   Testing hook registration...")
test_storage = []
def test_hook(m, i, o):
    test_storage.append(o.shape)
    return o

h = model.model.layers[27].self_attn.v_proj.register_forward_hook(test_hook)
test_input = tokenizer("Test prompt", return_tensors="pt").to(model.device)
with torch.no_grad():
    _ = model(**test_input)
h.remove()

print(f"   Hook captured tensor shape: {test_storage[0]}")
print()

# Final verdict
print("4. PRE-FLIGHT VERDICT")
checks_passed = (
    torch.cuda.is_available() and 
    num_layers == 32 and 
    has_v_proj and 
    len(test_storage) > 0
)

if checks_passed:
    print("   ✓ ALL CHECKS PASSED - Ready to proceed")
else:
    print("   ✗ CHECKS FAILED - Do not proceed")
    if num_layers != 32:
        print(f"     - Layer count mismatch: {num_layers} != 32")
    if not has_v_proj:
        print("     - Layer 27 missing v_proj")

print()
print("=" * 70)
print("PRE-FLIGHT COMPLETE")
print("=" * 70)

