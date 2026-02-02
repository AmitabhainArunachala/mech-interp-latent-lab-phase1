#!/usr/bin/env python3
"""Quick test: Can we actually load Mistral-7B on this machine?"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Testing model load...")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")

device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
print(f"Using device: {device}")

print("\nAttempting to load Mistral-7B-v0.1...")
print("(This will take a few minutes and use ~14GB RAM)")

try:
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device == "cpu" else None,
        attn_implementation="eager",
    )
    
    if device != "cpu" and device != "cuda":
        # Move to MPS manually
        model = model.to(device)
    
    model.eval()
    
    print("\n✅ Model loaded successfully!")
    print(f"   Device: {next(model.parameters()).device}")
    print(f"   Dtype: {next(model.parameters()).dtype}")
    
    # Quick test inference
    print("\nTesting inference...")
    test_prompt = "Hello, how are you?"
    enc = tokenizer(test_prompt, return_tensors="pt")
    if device != "cpu":
        enc = {k: v.to(device) for k, v in enc.items()}
    
    with torch.no_grad():
        output = model(**enc, output_attentions=True, max_new_tokens=5)
    
    print("✅ Inference works!")
    print(f"   Output shape: {output.logits.shape}")
    print(f"   Attention available: {output.attentions is not None}")
    
    print("\n🎉 Ready to run h31_validation_n50.py!")
    
except Exception as e:
    print(f"\n❌ Failed to load model: {e}")
    print("\nRecommendation: Wait for RunPod with GPU")
    import traceback
    traceback.print_exc()










