#!/usr/bin/env python3
"""
NECESSITY TEST: H18 & H26
==========================

Test if ablating H18 & H26 breaks recursive behavior.

Method:
1. Generate text from recursive prompts WITH H18 & H26 ablated
2. Generate text from recursive prompts WITHOUT ablation (control)
3. Measure behavioral markers (recursive keywords, identity equations, etc.)
4. Compare: Does ablation break recursive behavior?

If YES: H18 & H26 are NECESSARY for recursive behavior
If NO: They're not necessary (or other heads compensate)
"""

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.core.models import load_model, set_seed
from src.metrics.behavior_states import label_behavior_state, BehaviorState
from prompts.loader import PromptLoader

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
TARGET_LAYER = 27
TARGET_HEADS = [18, 26]  # The mode-switching heads
SEED = 42
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
DO_SAMPLE = True

# Test prompts (recursive) - sourced from prompt bank to prevent drift
_loader = PromptLoader()
PROMPT_BANK_VERSION = _loader.version
TEST_PROMPTS = _loader.get_by_group("legacy_comprehensive_circuit_test_champions", limit=5, seed=SEED)

# =============================================================================
# V-PROJECTION ABLATION
# =============================================================================

@contextmanager
def zero_v_proj_heads(model, layer_idx: int, head_indices: List[int]):
    """
    Zero out V-projection values for multiple heads BEFORE attention.
    """
    num_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    handles = []
    
    def make_hook(head_idx):
        kv_head_idx = head_idx % num_kv_heads if num_kv_heads < num_heads else head_idx
        
        def hook_fn(module, inp, out):
            v_proj_out = out.clone()
            
            if v_proj_out.dim() == 2:
                v_proj_out = v_proj_out.unsqueeze(0)
            
            batch, seq_len, kv_hidden_size = v_proj_out.shape
            expected_kv_size = num_kv_heads * head_dim
            
            if kv_hidden_size != expected_kv_size:
                return out
            
            # Reshape to (batch, seq, num_kv_heads, head_dim)
            v_reshaped = v_proj_out.view(batch, seq_len, num_kv_heads, head_dim)
            
            # Zero out the KV head corresponding to this query head
            v_reshaped[:, :, kv_head_idx, :] = 0.0
            
            # Reshape back
            v_zeroed = v_reshaped.view(batch, seq_len, kv_hidden_size)
            
            return v_zeroed
        
        return hook_fn
    
    # Register hooks for all target heads
    layer = model.model.layers[layer_idx].self_attn
    for head_idx in head_indices:
        hook_fn = make_hook(head_idx)
        handle = layer.v_proj.register_forward_hook(hook_fn)
        handles.append(handle)
    
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()

# =============================================================================
# GENERATION & ANALYSIS
# =============================================================================

def generate_with_ablation(model, tokenizer, prompt: str, ablate: bool) -> str:
    """Generate text with or without ablation."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        if ablate:
            with zero_v_proj_heads(model, TARGET_LAYER, TARGET_HEADS):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    do_sample=DO_SAMPLE,
                    pad_token_id=tokenizer.eos_token_id,
                )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=DO_SAMPLE,
                pad_token_id=tokenizer.eos_token_id,
            )
    
    # Decode only the new tokens
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text

def analyze_behavior(text: str) -> dict:
    """Analyze generated text for recursive behavior markers."""
    label = label_behavior_state(text)
    
    return {
        "state": label.state.value,
        "has_recursive_keywords": label.has_recursive_keywords,
        "has_identity_equation": label.has_identity_equation,
        "repetition_ratio": label.repetition_ratio,
        "question_mark_ratio": label.question_mark_ratio,
        "is_recursive": label.state in [BehaviorState.RECURSIVE_PROSE, BehaviorState.NAKED_LOOP],
        "text": text[:200],  # First 200 chars for inspection
    }

# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    print("=" * 80)
    print("NECESSITY TEST: H18 & H26")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Layer: {TARGET_LAYER}, Heads: {TARGET_HEADS}")
    print(f"Prompts: {len(TEST_PROMPTS)}")
    print("=" * 80)
    
    set_seed(SEED)
    
    # Load model
    print("\n[1/3] Loading model...")
    model, tokenizer = load_model(
        model_name=MODEL_NAME,
        device=DEVICE,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    model.eval()
    print("  ✅ Model loaded")
    
    # Generate with and without ablation
    print("\n[2/3] Generating text...")
    results = []
    
    for i, prompt in enumerate(tqdm(TEST_PROMPTS, desc="Processing prompts")):
        # Control (no ablation)
        control_text = generate_with_ablation(model, tokenizer, prompt, ablate=False)
        control_analysis = analyze_behavior(control_text)
        
        # Ablated
        ablated_text = generate_with_ablation(model, tokenizer, prompt, ablate=True)
        ablated_analysis = analyze_behavior(ablated_text)
        
        results.append({
            "prompt_idx": i,
            "prompt": prompt[:100],
            "control_state": control_analysis["state"],
            "control_is_recursive": control_analysis["is_recursive"],
            "control_has_recursive_keywords": control_analysis["has_recursive_keywords"],
            "control_has_identity_equation": control_analysis["has_identity_equation"],
            "control_text": control_analysis["text"],
            "ablated_state": ablated_analysis["state"],
            "ablated_is_recursive": ablated_analysis["is_recursive"],
            "ablated_has_recursive_keywords": ablated_analysis["has_recursive_keywords"],
            "ablated_has_identity_equation": ablated_analysis["has_identity_equation"],
            "ablated_text": ablated_analysis["text"],
        })
    
    # Analyze results
    print("\n[3/3] Analyzing results...")
    print("=" * 80)
    
    control_recursive_count = sum(1 for r in results if r["control_is_recursive"])
    ablated_recursive_count = sum(1 for r in results if r["ablated_is_recursive"])
    
    control_recursive_keywords = sum(1 for r in results if r["control_has_recursive_keywords"])
    ablated_recursive_keywords = sum(1 for r in results if r["ablated_has_recursive_keywords"])
    
    control_identity_eq = sum(1 for r in results if r["control_has_identity_equation"])
    ablated_identity_eq = sum(1 for r in results if r["ablated_has_identity_equation"])
    
    print("\n📊 RESULTS SUMMARY")
    print("-" * 80)
    print(f"Control (no ablation):")
    print(f"  Recursive states: {control_recursive_count}/{len(results)} ({control_recursive_count/len(results)*100:.1f}%)")
    print(f"  Has recursive keywords: {control_recursive_keywords}/{len(results)} ({control_recursive_keywords/len(results)*100:.1f}%)")
    print(f"  Has identity equations: {control_identity_eq}/{len(results)} ({control_identity_eq/len(results)*100:.1f}%)")
    
    print(f"\nAblated (H18 & H26 zeroed):")
    print(f"  Recursive states: {ablated_recursive_count}/{len(results)} ({ablated_recursive_count/len(results)*100:.1f}%)")
    print(f"  Has recursive keywords: {ablated_recursive_keywords}/{len(results)} ({ablated_recursive_keywords/len(results)*100:.1f}%)")
    print(f"  Has identity equations: {ablated_identity_eq}/{len(results)} ({ablated_identity_eq/len(results)*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    recursive_drop = control_recursive_count - ablated_recursive_count
    keywords_drop = control_recursive_keywords - ablated_recursive_keywords
    
    if recursive_drop > 0 or keywords_drop > 0:
        print("\n✅ NECESSITY CONFIRMED:")
        print(f"  Ablating H18 & H26 REDUCES recursive behavior")
        print(f"  Recursive states dropped by: {recursive_drop} ({recursive_drop/len(results)*100:.1f}%)")
        print(f"  Recursive keywords dropped by: {keywords_drop} ({keywords_drop/len(results)*100:.1f}%)")
        print("\n  H18 & H26 are NECESSARY for recursive behavior!")
    elif ablated_recursive_count == control_recursive_count:
        print("\n❌ NECESSITY NOT CONFIRMED:")
        print(f"  Ablating H18 & H26 does NOT reduce recursive behavior")
        print(f"  Recursive states: {control_recursive_count} → {ablated_recursive_count}")
        print("\n  H18 & H26 may not be necessary (or other heads compensate)")
    else:
        print("\n⚠️  MIXED RESULTS:")
        print(f"  Some reduction but not clear-cut")
    
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    for i, r in enumerate(results):
        print(f"\nPrompt {i+1}: {r['prompt'][:60]}...")
        print(f"  Control: {r['control_state']} | Recursive: {r['control_is_recursive']}")
        print(f"  Ablated: {r['ablated_state']} | Recursive: {r['ablated_is_recursive']}")
        print(f"  Control text: {r['control_text'][:80]}...")
        print(f"  Ablated text: {r['ablated_text'][:80]}...")
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

