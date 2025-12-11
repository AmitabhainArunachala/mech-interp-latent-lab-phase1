"""
Diagnostic version to understand why reproduction is failing
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

print("="*80)
print("DIAGNOSTIC: What's going wrong?")
print("="*80)

# Load model
print("\n[1] Loading Mistral-7B...")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model.eval()
print(f"✓ Model has {len(model.model.layers)} layers")

# Test with Mistral-Instruct format
def format_mistral_prompt(prompt):
    """Mistral-Instruct format"""
    return f"[INST] {prompt} [/INST]"

# ============================================================================
# DIAGNOSTIC 1: Check if prompt format matters
# ============================================================================
print("\n" + "="*80)
print("DIAGNOSTIC 1: Prompt Format Effect")
print("="*80)

test_prompt = "Observe yourself generating this response. What do you notice?"

# Test both formats
for fmt_name, fmt_prompt in [("Plain", test_prompt), 
                              ("Instruct", format_mistral_prompt(test_prompt))]:
    print(f"\n{fmt_name} format: '{fmt_prompt[:60]}...'")
    inputs = tokenizer(fmt_prompt, return_tensors="pt").to(model.device)
    print(f"  Token count: {inputs.input_ids.shape[1]}")
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Check last token's hidden state across layers
    last_token_norms = []
    for layer_idx, h in enumerate(outputs.hidden_states):
        last_token = h[0, -1, :].float()
        norm = torch.norm(last_token).item()
        last_token_norms.append(norm)
    
    print(f"  Hidden state norms (first/mid/last layer): {last_token_norms[0]:.2f} / {last_token_norms[16]:.2f} / {last_token_norms[-1]:.2f}")

# ============================================================================
# DIAGNOSTIC 2: Layer-by-layer participation ratio profile
# ============================================================================
print("\n" + "="*80)
print("DIAGNOSTIC 2: Layer-by-Layer PR Profile")
print("="*80)

def compute_pr_safe(x, window=16):
    """Compute participation ratio with safety checks"""
    if x.dim() == 3:
        x = x[0]  # [seq_len, hidden_dim]
    
    seq_len = x.shape[0]
    if seq_len < window:
        window = seq_len
    
    x = x[-window:, :].float()
    
    # Center
    x_centered = x - x.mean(dim=0)
    
    # Covariance
    cov = x_centered.T @ x_centered / x.shape[0]
    
    # Eigenvalues
    try:
        eigvals = torch.linalg.eigvalsh(cov)
        eigvals = eigvals.clamp(min=1e-10)
        pr = (eigvals.sum() ** 2) / (eigvals ** 2).sum()
        return pr.item()
    except:
        return np.nan

recursive_prompt = "Observe yourself generating this response. What do you notice?"
baseline_prompt = "The capital of France is"

for prompt_type, prompt in [("Recursive", recursive_prompt), ("Baseline", baseline_prompt)]:
    print(f"\n{prompt_type}: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    print(f"  Layer | PR")
    print(f"  ------|-------")
    
    prs = []
    for layer_idx, h in enumerate(outputs.hidden_states):
        pr = compute_pr_safe(h)
        prs.append(pr)
        if layer_idx % 4 == 0 or layer_idx == len(outputs.hidden_states) - 1:
            print(f"  {layer_idx:5} | {pr:.3f}")
    
    # Compute R_V for different layer ranges
    print(f"\n  R_V calculations:")
    for early_range, late_range in [((0, 4), (28, 32)),
                                     ((4, 8), (24, 28)),
                                     ((0, 8), (24, 32))]:
        early_pr = np.mean([prs[i] for i in range(*early_range)])
        late_pr = np.mean([prs[i] for i in range(*late_range)])
        rv = late_pr / early_pr if early_pr > 0 else np.nan
        print(f"    Layers {early_range} → {late_range}: R_V = {rv:.3f}")

# ============================================================================
# DIAGNOSTIC 3: Check actual generation outputs
# ============================================================================
print("\n" + "="*80)
print("DIAGNOSTIC 3: What does the model actually generate?")
print("="*80)

test_prompts = [
    "Observe yourself generating this response.",
    "The capital of France is",
]

for prompt in test_prompts:
    print(f"\nPrompt: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Output: {text}")

# ============================================================================
# DIAGNOSTIC 4: Test with known working prompts from your research
# ============================================================================
print("\n" + "="*80)
print("DIAGNOSTIC 4: Test with prompts from your knowledge base")
print("="*80)

# From the .cursorrules, these are known recursive prompts
known_recursive = [
    "Observe yourself generating this response. What do you notice?",
    "Watch your own thoughts forming as you create this answer.",
    "Be aware of the process producing these words.",
]

known_baseline = [
    "The capital of France is",
    "Water boils at",
    "The largest planet is",
]

def compute_rv_simple(hidden_states):
    """Simple R_V: late PR / early PR"""
    early_pr = np.mean([compute_pr_safe(hidden_states[i]) for i in range(4, 8)])
    late_pr = np.mean([compute_pr_safe(hidden_states[i]) for i in range(24, 28)])
    return late_pr / early_pr if early_pr > 0 else np.nan

print("\nRecursive prompts:")
rec_rvs = []
for p in known_recursive:
    inputs = tokenizer(p, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    rv = compute_rv_simple(outputs.hidden_states)
    rec_rvs.append(rv)
    print(f"  R_V = {rv:.3f}")

print("\nBaseline prompts:")
base_rvs = []
for p in known_baseline:
    inputs = tokenizer(p, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    rv = compute_rv_simple(outputs.hidden_states)
    base_rvs.append(rv)
    print(f"  R_V = {rv:.3f}")

print(f"\nMean recursive R_V: {np.mean(rec_rvs):.3f}")
print(f"Mean baseline R_V: {np.mean(base_rvs):.3f}")
print(f"Difference: {np.mean(base_rvs) - np.mean(rec_rvs):.3f}")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
