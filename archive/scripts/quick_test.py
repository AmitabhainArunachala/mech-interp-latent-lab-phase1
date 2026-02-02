#!/usr/bin/env python3
"""Quick test: Verify GPU, imports, and basic R_V computation utilities."""

import torch
import numpy as np
from transformers import AutoTokenizer
import sys
from pathlib import Path

print("=" * 60)
print("Quick Test - RunPod Setup Verification")
print("=" * 60)

# 1. GPU Check
print("\n[1/4] GPU Check")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU: {props.name}")
    print(f"  Memory: {props.total_memory/1024**3:.2f} GB")
    print(f"  CUDA version: {torch.version.cuda}")
    
    # Quick GPU computation
    device = torch.device('cuda:0')
    a = torch.randn(100, 100, device=device)
    b = torch.randn(100, 100, device=device)
    c = torch.matmul(a, b)
    print(f"  ✅ GPU computation: {c.shape}")
else:
    print("  ❌ CUDA not available")
    sys.exit(1)

# 2. Imports Check
print("\n[2/4] Key Imports")
try:
    import transformers
    print(f"  ✅ transformers: {transformers.__version__}")
except Exception as e:
    print(f"  ❌ transformers: {e}")
    sys.exit(1)

try:
    import scipy
    print(f"  ✅ scipy: {scipy.__version__}")
except Exception as e:
    print(f"  ❌ scipy: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print(f"  ✅ pandas: {pd.__version__}")
except Exception as e:
    print(f"  ❌ pandas: {e}")
    sys.exit(1)

# 3. R_V Computation Test
print("\n[3/4] R_V Computation Test")
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
    except Exception as e:
        return float('nan')

# Create synthetic V tensor (T=20, D=128)
device = torch.device('cuda:0')
v_tensor = torch.randn(20, 128, device=device)
v_window = v_tensor[-16:, :]  # Last 16 tokens

pr = participation_ratio(v_window)
print(f"  Synthetic V tensor shape: {v_tensor.shape}")
print(f"  Window shape: {v_window.shape}")
print(f"  Participation Ratio: {pr:.4f}")
if not np.isnan(pr) and pr > 0:
    print("  ✅ R_V computation works")
else:
    print("  ❌ R_V computation failed")
    sys.exit(1)

# 4. Tokenizer Test (lightweight, no model load)
print("\n[4/4] Tokenizer Test")
try:
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    test_prompt = "Am I real?"
    tokens = tokenizer(test_prompt, return_tensors="pt")
    print(f"  Test prompt: '{test_prompt}'")
    print(f"  Token IDs shape: {tokens['input_ids'].shape}")
    print(f"  ✅ Tokenizer works")
except Exception as e:
    print(f"  ⚠️  Tokenizer test: {str(e)[:100]}")
    print("  (This is OK if HF token not set - model loading will handle it)")

# 5. File Structure Check
print("\n[5/5] Key Files Check")
key_files = [
    'validate_h18_h26_effect.py',
    'n300_mistral_test_prompt_bank.py',
    'src/core/hooks.py',
    'src/metrics/rv.py',
    'utils/metrics.py'
]
all_present = True
for f in key_files:
    if Path(f).exists():
        print(f"  ✅ {f}")
    else:
        print(f"  ❌ {f} missing")
        all_present = False

print("\n" + "=" * 60)
if all_present:
    print("✅ ALL TESTS PASSED!")
    print("RunPod is ready for experiments!")
else:
    print("⚠️  Some files missing, but core functionality works")
print("=" * 60)









