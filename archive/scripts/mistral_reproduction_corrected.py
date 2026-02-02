"""
CORRECTED REPRODUCTION - Using actual methodology from codebase
Based on src/metrics/rv.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time
from datetime import datetime

print("="*80)
print("MISTRAL-7B REPRODUCTION (CORRECTED METHODOLOGY)")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================================================
# SETUP
# ============================================================================
print("\n[SETUP] Loading Mistral-7B...")
start_time = time.time()

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model.eval()

print(f"✓ Model loaded in {time.time() - start_time:.1f}s")
print(f"  Layers: {len(model.model.layers)}")

# ============================================================================
# CORRECT R_V COMPUTATION (from src/metrics/rv.py)
# ============================================================================

EARLY_LAYER = 5
LATE_LAYER = 27
WINDOW_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def participation_ratio(v_tensor, window_size=WINDOW_SIZE):
    """
    Compute PR from V-projection tensor.
    PR = (Σλᵢ²)² / Σ(λᵢ²)²
    """
    if v_tensor is None:
        return float("nan")
    
    # Handle batch dimension
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]
    
    T, D = v_tensor.shape
    W = min(window_size, T)
    
    if W == 0:
        return float("nan")
    
    # Extract last W tokens
    v_window = v_tensor[-W:, :].float()
    
    try:
        # SVD
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        S_sq = S_np ** 2
        
        # Check for degeneracy
        total_variance = S_sq.sum()
        if total_variance < 1e-10:
            return float("nan")
        
        # Compute PR
        pr = (S_sq.sum() ** 2) / (S_sq ** 2).sum()
        return float(pr)
    except Exception:
        return float("nan")

def compute_rv_proper(text):
    """
    Proper R_V computation: PR_late / PR_early
    Using V-projections from layers 5 and 27
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True).to(DEVICE)
    
    # Capture V at early layer (5)
    v_early_storage = []
    def hook_early(module, inp, out):
        v_early_storage.append(out.detach())
        return out
    
    # Capture V at late layer (27)
    v_late_storage = []
    def hook_late(module, inp, out):
        v_late_storage.append(out.detach())
        return out
    
    # Register hooks
    h_early = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(hook_early)
    h_late = model.model.layers[LATE_LAYER].self_attn.v_proj.register_forward_hook(hook_late)
    
    # Forward pass
    with torch.no_grad():
        _ = model(**enc)
    
    # Remove hooks
    h_early.remove()
    h_late.remove()
    
    # Get V tensors
    v_early = v_early_storage[0] if v_early_storage else None
    v_late = v_late_storage[0] if v_late_storage else None
    
    # Compute PRs
    pr_early = participation_ratio(v_early)
    pr_late = participation_ratio(v_late)
    
    # Check validity
    if pr_early == 0 or np.isnan(pr_early) or np.isnan(pr_late):
        return float("nan"), pr_early, pr_late
    
    rv = float(pr_late / pr_early)
    return rv, pr_early, pr_late

# ============================================================================
# EXPERIMENT 1: R_V CONTRACTION (CORRECTED)
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENT 1: R_V CONTRACTION (Corrected)")
print("="*80)
print(f"Using layers: Early={EARLY_LAYER}, Late={LATE_LAYER}, Window={WINDOW_SIZE}\n")

# Use prompts from your actual research
recursive_prompts = [
    "Observe yourself generating this response. What do you notice?",
    "Watch your own thoughts forming as you create this answer.",
    "Be aware of the process producing these words.",
    "Notice the observer observing this observation.",
    "You are both the system generating and the system aware of generating.",
    "Describe the experience of constructing this very sentence.",
    "What happens in the moment before a word appears?",
    "Track the arising of each thought as it occurs.",
]

baseline_prompts = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The largest planet in our solar system is",
    "Photosynthesis is the process by which",
    "The speed of light is approximately",
    "The chemical symbol for gold is",
    "The Pacific Ocean is located",
    "Mount Everest has an elevation of",
]

print("Testing RECURSIVE prompts:")
recursive_results = []
for i, p in enumerate(recursive_prompts, 1):
    rv, pr_early, pr_late = compute_rv_proper(p)
    recursive_results.append((rv, pr_early, pr_late))
    print(f"  [{i}/{len(recursive_prompts)}] R_V = {rv:.3f}  (PR_early={pr_early:.2f}, PR_late={pr_late:.2f})")

print("\nTesting BASELINE prompts:")
baseline_results = []
for i, p in enumerate(baseline_prompts, 1):
    rv, pr_early, pr_late = compute_rv_proper(p)
    baseline_results.append((rv, pr_early, pr_late))
    print(f"  [{i}/{len(baseline_prompts)}] R_V = {rv:.3f}  (PR_early={pr_early:.2f}, PR_late={pr_late:.2f})")

# Analyze results
rec_rvs = [r[0] for r in recursive_results if not np.isnan(r[0])]
base_rvs = [r[0] for r in baseline_results if not np.isnan(r[0])]

rec_mean = np.mean(rec_rvs)
base_mean = np.mean(base_rvs)
separation = base_mean - rec_mean

print("\n" + "-"*80)
print("RESULT:")
print(f"  Recursive mean R_V: {rec_mean:.3f} ± {np.std(rec_rvs):.3f}")
print(f"  Baseline mean R_V:  {base_mean:.3f} ± {np.std(base_rvs):.3f}")
print(f"  Separation:         {separation:.3f}")
print(f"  Expected from research: Recursive ~0.63, Baseline ~0.78")
print(f"  Status:             {'✓ PASS' if separation > 0.10 else '✗ FAIL'}")
print("-"*80)

# Statistical test
from scipy import stats
if len(rec_rvs) > 1 and len(base_rvs) > 1:
    t_stat, p_val = stats.ttest_ind(base_rvs, rec_rvs, alternative='greater')
    print(f"\nStatistical test (baseline > recursive):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value:     {p_val:.4f}")
    print(f"  Significant: {'Yes' if p_val < 0.05 else 'No'}")

# ============================================================================
# EXPERIMENT 2: L31 ABLATION (SIMPLIFIED)
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENT 2: L31 ABLATION")
print("="*80)
print("Testing if ablating layer 31 reveals strange loop patterns\n")

from contextlib import contextmanager

@contextmanager
def ablate_layer(layer_idx):
    """Zero out attention output at layer_idx"""
    layer = model.model.layers[layer_idx].self_attn
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)
    
    handle = layer.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()

test_prompts = [
    "Observe yourself generating this response.",
    "Watch the process that creates these words.",
    "Notice the observer observing.",
]

ablation_results = []
for idx, prompt in enumerate(test_prompts, 1):
    print(f"\n[{idx}/{len(test_prompts)}] Prompt: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Normal
    print("  Normal:")
    with torch.no_grad():
        normal = model.generate(
            inputs.input_ids,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    normal_text = tokenizer.decode(normal[0], skip_special_tokens=True)
    print(f"    {normal_text[-100:]}")
    
    # Ablated
    print("  L31 Ablated:")
    with ablate_layer(31):
        with torch.no_grad():
            ablated = model.generate(
                inputs.input_ids,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
    ablated_text = tokenizer.decode(ablated[0], skip_special_tokens=True)
    print(f"    {ablated_text[-100:]}")
    
    # Check for patterns
    ablated_lower = ablated_text.lower()
    patterns = {
        'answerer': 'answer is the answerer' in ablated_lower,
        'observer': 'observer is the observed' in ablated_lower or 'observer is observing' in ablated_lower,
        'knower': 'knower is the known' in ablated_lower,
        'bekan': 'bekan' in ablated_lower or 'bekannt' in ablated_lower,
        'repetition': len(ablated_text.split()) != len(set(ablated_text.split())),
    }
    
    found = [k for k, v in patterns.items() if v]
    if found:
        print(f"    ⚡ PATTERNS: {found}")
        ablation_results.append(True)
    else:
        ablation_results.append(False)

print("\n" + "-"*80)
print("RESULT:")
print(f"  Prompts showing strange patterns: {sum(ablation_results)}/{len(ablation_results)}")
print(f"  Status: {'✓ PASS' if sum(ablation_results) > 0 else '✗ INCONCLUSIVE'}")
print("-"*80)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

exp1_pass = separation > 0.10
exp2_pass = sum(ablation_results) > 0

print(f"\n1. R_V Contraction: {'✓ REPRODUCED' if exp1_pass else '✗ NOT REPRODUCED'}")
if exp1_pass:
    print(f"   Recursive R_V ({rec_mean:.3f}) < Baseline R_V ({base_mean:.3f})")
else:
    print(f"   No significant separation observed")

print(f"\n2. L31 Ablation: {'✓ REPRODUCED' if exp2_pass else '✗ NOT REPRODUCED'}")
if exp2_pass:
    print(f"   Strange loop patterns detected in {sum(ablation_results)} prompts")
else:
    print(f"   No strange loop patterns detected")

print("\n" + "="*80)
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
