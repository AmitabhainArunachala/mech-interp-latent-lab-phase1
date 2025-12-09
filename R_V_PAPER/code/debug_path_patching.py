#!/usr/bin/env python3
"""
DEBUG version of path patching to identify the issue
"""

import torch

def debug_single_pair(model, tokenizer, prompt_bank, pair_idx=0):
    """Debug a single pair to see what's happening"""
    print("="*70)
    print("DEBUGGING SINGLE PAIR")
    print("="*70)

    # Get pair from CSV
    import pandas as pd
    df = pd.read_csv('mistral7b_n200_BULLETPROOF.csv')
    row = df.iloc[pair_idx]
    rec_id = row['rec_id']
    base_id = row['base_id']

    print(f"Testing pair: {rec_id} → {base_id}")

    # Check prompts exist
    if rec_id not in prompt_bank:
        print(f"❌ rec_id {rec_id} not in prompt_bank")
        return
    if base_id not in prompt_bank:
        print(f"❌ base_id {base_id} not in prompt_bank")
        return

    rec_text = prompt_bank[rec_id]["text"]
    base_text = prompt_bank[base_id]["text"]

    print(f"Rec text length: {len(rec_text)} chars")
    print(f"Base text length: {len(base_text)} chars")

    # Check tokenization
    rec_tokens = tokenizer.encode(rec_text)
    base_tokens = tokenizer.encode(base_text)

    print(f"Rec tokens: {len(rec_tokens)}")
    print(f"Base tokens: {len(base_tokens)}")

    if len(rec_tokens) < 16 or len(base_tokens) < 16:
        print("❌ Prompts too short (<16 tokens)")
        return

    # Test individual hook functions
    print("\nTesting hooks individually...")

    # Test V5 hook
    print("1. Testing V5 hook...")
    inputs = tokenizer(rec_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    v5_storage = []
    def capture_v5(m, i, o):
        print(f"   V5 hook fired! Output shape: {o.shape}")
        v5_storage.append(o.detach())
        return o

    with torch.no_grad():
        h = model.model.layers[5].self_attn.v_proj.register_forward_hook(capture_v5)
        _ = model(**inputs)
        h.remove()

    print(f"   V5 storage length: {len(v5_storage)}")
    if v5_storage:
        print(f"   V5 shape: {v5_storage[0].shape}")

    # Test residual hook
    print("\n2. Testing residual hook...")
    residual_storage = []
    def capture_residual(m, i, o):
        print(f"   Residual hook fired! Output shape: {o.shape}")
        residual_storage.append(o.detach())
        return o

    with torch.no_grad():
        h = model.model.layers[27].register_forward_hook(capture_residual)
        _ = model(**inputs)
        h.remove()

    print(f"   Residual storage length: {len(residual_storage)}")
    if residual_storage:
        print(f"   Residual shape: {residual_storage[0].shape}")

    # Test V27 hook
    print("\n3. Testing V27 hook...")
    v27_storage = []
    def capture_v27(m, i, o):
        print(f"   V27 hook fired! Output shape: {o.shape}")
        v27_storage.append(o.detach())
        return o

    with torch.no_grad():
        h = model.model.layers[27].self_attn.v_proj.register_forward_hook(capture_v27)
        _ = model(**inputs)
        h.remove()

    print(f"   V27 storage length: {len(v27_storage)}")
    if v27_storage:
        print(f"   V27 shape: {v27_storage[0].shape}")

    # Test combined function
    print("\n4. Testing combined function...")
    try:
        v5, residual, v27 = get_residual_and_v(model, tokenizer, rec_text, 27)
        print(f"   Combined function returned: v5={v5.shape if v5 is not None else None}, residual={residual.shape if residual is not None else None}, v27={v27.shape if v27 is not None else None}")
    except Exception as e:
        print(f"   ❌ Combined function failed: {e}")

def get_residual_and_v(model, tokenizer, text, layer_idx):
    """Get both residual stream and V at a specific layer using PyTorch hooks"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    # Storage for activations
    v5_storage = []
    residual_storage = []
    v_out_storage = []

    with torch.no_grad():
        # Hook for early layer V
        def capture_v5(m, i, o):
            v5_storage.append(o.detach())
            return o

        # Hook for target layer residual (output of layer becomes input to next)
        def capture_residual(m, i, o):
            residual_storage.append(o.detach())  # This is the residual stream
            return o

        # Hook for target layer V
        def capture_v_out(m, i, o):
            v_out_storage.append(o.detach())
            return o

        # Register hooks
        h_v5 = model.model.layers[5].self_attn.v_proj.register_forward_hook(capture_v5)
        h_residual = model.model.layers[layer_idx].register_forward_hook(capture_residual)
        h_v_out = model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(capture_v_out)

        # Run forward pass
        _ = model(**inputs)

        # Remove hooks
        h_v5.remove()
        h_residual.remove()
        h_v_out.remove()

    # Extract results
    v5 = v5_storage[0][0] if v5_storage else None  # [seq, hidden]
    residual = residual_storage[0][0] if residual_storage else None  # [seq, hidden]
    v_out = v_out_storage[0][0] if v_out_storage else None  # [seq, hidden]

    return v5, residual, v_out

# Run debug
if __name__ == "__main__":
    print("Run this in your notebook:")
    print("from debug_path_patching import debug_single_pair")
    print("debug_single_pair(model, tokenizer, prompt_bank_1c, pair_idx=0)")
