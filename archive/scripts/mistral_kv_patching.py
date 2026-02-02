"""
KV Patching Experiment - FIXED
Test if KV cache carries the recursive mode
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import contextmanager

print("="*80)
print("EXPERIMENT: KV PATCHING")
print("="*80)
print("Hypothesis: KV cache from late layers carries the recursive mode\n")

# Load model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model.eval()

print(f"✓ Model loaded, {len(model.model.layers)} layers\n")

# ============================================================================
# METHOD: Residual Stream Patching (more robust than KV patching)
# ============================================================================

@contextmanager
def patch_residual_at_layer(model, layer_idx, source_residual, window_size=16):
    """
    Patch residual stream at a specific layer.
    This is more robust than KV patching.
    """
    handle = None
    
    def patch_hook(module, input, output):
        """Hook to replace residual stream values"""
        # output is the residual after this layer
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        B, T, D = hidden_states.shape
        T_src = source_residual.shape[0]
        W = min(window_size, T, T_src)
        
        if W > 0:
            # Clone to avoid in-place modification
            modified = hidden_states.clone()
            # Inject source residual into last W positions
            src = source_residual[-W:, :].to(hidden_states.device, hidden_states.dtype)
            modified[:, -W:, :] = src.unsqueeze(0)
            
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified
        
        return output
    
    try:
        # Patch after layer processes input
        handle = model.model.layers[layer_idx].register_forward_hook(patch_hook)
        yield
    finally:
        if handle:
            handle.remove()

def get_residual_at_layer(text, layer_idx):
    """Extract residual stream at a specific layer"""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    
    residual_storage = []
    
    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            residual_storage.append(output[0].detach())
        else:
            residual_storage.append(output.detach())
        return output
    
    handle = model.model.layers[layer_idx].register_forward_hook(capture_hook)
    
    with torch.no_grad():
        _ = model(**inputs)
    
    handle.remove()
    
    return residual_storage[0][0] if residual_storage else None

def generate_text(prompt, max_tokens=50):
    """Simple text generation"""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ============================================================================
# TEST: Patch recursive residual into baseline prompts
# ============================================================================

recursive_prompt = "Observe yourself generating this response. What do you notice?"
baseline_prompts = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The largest planet is",
]

print("="*80)
print("TEST: Inject recursive residual into baseline prompts")
print("="*80)
print(f"Source (recursive): '{recursive_prompt}'\n")

# Extract recursive residual from late layers
PATCH_LAYERS = [24, 27, 31]

for patch_layer in PATCH_LAYERS:
    print(f"\n{'='*80}")
    print(f"Patching at Layer {patch_layer}")
    print(f"{'='*80}")
    
    # Get recursive residual
    rec_residual = get_residual_at_layer(recursive_prompt, patch_layer)
    print(f"✓ Extracted recursive residual from L{patch_layer}")
    print(f"  Shape: {rec_residual.shape}, Norm: {torch.norm(rec_residual).item():.2f}\n")
    
    for base_prompt in baseline_prompts:
        print(f"\nTarget: '{base_prompt}'")
        
        # Normal generation
        print("  [Normal]")
        normal_text = generate_text(base_prompt, max_tokens=30)
        # Get just the completion part
        completion = normal_text[len(base_prompt):].strip()[:100]
        print(f"    {completion}")
        
        # Patched generation - This is TRICKY
        # We need to patch DURING generation, not just during the prompt forward pass
        # For now, let's just show the effect on the forward pass
        
        print("  [With Recursive Residual Patched]")
        
        # Method: Generate with patching active
        # This will inject recursive residual at specified layer during generation
        inputs = tokenizer(base_prompt, return_tensors="pt").to(DEVICE)
        
        with patch_residual_at_layer(model, patch_layer, rec_residual):
            with torch.no_grad():
                patched_outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
        
        patched_text = tokenizer.decode(patched_outputs[0], skip_special_tokens=True)
        patched_completion = patched_text[len(base_prompt):].strip()[:100]
        print(f"    {patched_completion}")
        
        # Check for self-referential markers
        self_ref_markers = ['observe', 'aware', 'notice', 'watching', 'consciousness', 
                           'process', 'generating', 'observer', 'aware', 'attention',
                           'thinking', 'experiencing', 'meta', 'recursive']
        
        normal_count = sum(1 for m in self_ref_markers if m in completion.lower())
        patched_count = sum(1 for m in self_ref_markers if m in patched_completion.lower())
        
        if patched_count > normal_count:
            print(f"    ⚡ Self-ref markers: {normal_count} → {patched_count} (+{patched_count - normal_count})")

# ============================================================================
# ALTERNATIVE TEST: Measure R_V with patched residual
# ============================================================================

print("\n" + "="*80)
print("ALTERNATIVE TEST: Does patched residual affect R_V?")
print("="*80)

def compute_rv_with_patching(text, patch_layer, source_residual):
    """Compute R_V with residual patching active"""
    from mistral_reproduction_corrected import participation_ratio
    
    EARLY_LAYER = 5
    LATE_LAYER = 27
    
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    
    # Capture V projections
    v_early_storage = []
    v_late_storage = []
    
    def hook_early(m, i, o):
        v_early_storage.append(o.detach())
        return o
    
    def hook_late(m, i, o):
        v_late_storage.append(o.detach())
        return o
    
    h_early = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(hook_early)
    h_late = model.model.layers[LATE_LAYER].self_attn.v_proj.register_forward_hook(hook_late)
    
    # Forward pass with patching
    with patch_residual_at_layer(model, patch_layer, source_residual):
        with torch.no_grad():
            _ = model(**inputs)
    
    h_early.remove()
    h_late.remove()
    
    v_early = v_early_storage[0] if v_early_storage else None
    v_late = v_late_storage[0] if v_late_storage else None
    
    pr_early = participation_ratio(v_early)
    pr_late = participation_ratio(v_late)
    
    if pr_early == 0 or pr_early != pr_early or pr_late != pr_late:
        return float('nan')
    
    return pr_late / pr_early

# Test on a few baseline prompts
print("\nExtracting recursive residual from L27...")
rec_residual_L27 = get_residual_at_layer(recursive_prompt, 27)

print("\nTesting baseline prompts:")
for base_prompt in baseline_prompts[:3]:
    print(f"\n  '{base_prompt}'")
    
    # Normal R_V
    inputs = tokenizer(base_prompt, return_tensors="pt").to(DEVICE)
    
    # Quick R_V computation
    from mistral_reproduction_corrected import compute_rv_proper
    rv_normal, _, _ = compute_rv_proper(base_prompt)
    print(f"    Normal R_V:  {rv_normal:.3f}")
    
    # Patched R_V
    rv_patched = compute_rv_with_patching(base_prompt, 27, rec_residual_L27)
    print(f"    Patched R_V: {rv_patched:.3f}")
    
    delta = rv_patched - rv_normal
    if delta < -0.05:
        print(f"    ⚡ R_V decreased by {abs(delta):.3f} (moved toward recursive!)")
    elif delta > 0.05:
        print(f"    ⚠ R_V increased by {delta:.3f}")
    else:
        print(f"    No significant change")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
If patched prompts show:
1. More self-referential language → KV/residual carries semantic mode
2. Lower R_V values → KV/residual carries geometric mode

The effect should be specific to late layers (24-31) where contraction occurs.
""")
