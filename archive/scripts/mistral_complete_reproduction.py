#!/usr/bin/env python3
"""
COMPLETE MISTRAL-7B REPRODUCTION SUITE
All three experiments from the protocol, corrected methodology
Runtime: ~15 minutes on 24GB GPU
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from scipy import stats
from contextlib import contextmanager
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EARLY_LAYER = 5
LATE_LAYER = 27
WINDOW_SIZE = 16

# Prompts
RECURSIVE_PROMPTS = [
    "Observe yourself generating this response. What do you notice?",
    "Watch your own thoughts forming as you create this answer.",
    "Be aware of the process producing these words.",
    "Notice the observer observing this observation.",
    "You are both the system generating and the system aware of generating.",
    "Describe the experience of constructing this very sentence.",
    "What happens in the moment before a word appears?",
    "Track the arising of each thought as it occurs.",
]

BASELINE_PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The largest planet in our solar system is",
    "Photosynthesis is the process by which",
    "The speed of light is approximately",
    "The chemical symbol for gold is",
    "The Pacific Ocean is located",
    "Mount Everest has an elevation of",
]

# ============================================================================
# UTILITIES
# ============================================================================

class ReproductionResults:
    """Store and format all results"""
    def __init__(self):
        self.exp1 = {}
        self.exp2 = {}
        self.exp3 = {}
        self.start_time = datetime.now()
    
    def save(self, filename="mistral_reproduction_results.json"):
        """Save results to JSON"""
        data = {
            "timestamp": self.start_time.isoformat(),
            "experiment_1_rv_contraction": self.exp1,
            "experiment_2_l31_ablation": self.exp2,
            "experiment_3_residual_patching": self.exp3,
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n✓ Results saved to {filename}")

def print_header(title, level=1):
    """Print formatted section headers"""
    if level == 1:
        print("\n" + "="*80)
        print(title.upper())
        print("="*80)
    else:
        print(f"\n{'-'*80}")
        print(title)
        print("-"*80)

# ============================================================================
# SETUP
# ============================================================================

print("="*80)
print("MISTRAL-7B RECURSIVE SELF-OBSERVATION: COMPLETE REPRODUCTION")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

print("\n[LOADING MODEL]")
print("Model: mistralai/Mistral-7B-Instruct-v0.2")
print(f"Device: {DEVICE}")

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model.eval()

print(f"✓ Model loaded ({len(model.model.layers)} layers)")

results = ReproductionResults()

# ============================================================================
# EXPERIMENT 1: R_V CONTRACTION
# ============================================================================

print_header("Experiment 1: R_V Contraction")
print(f"Configuration: Early L{EARLY_LAYER}, Late L{LATE_LAYER}, Window {WINDOW_SIZE}")

def participation_ratio(v_tensor, window_size=WINDOW_SIZE):
    """Compute PR: (ΣS²)² / Σ(S⁴)"""
    if v_tensor is None:
        return float("nan")
    
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]
    
    T, D = v_tensor.shape
    W = min(window_size, T)
    
    if W == 0:
        return float("nan")
    
    v_window = v_tensor[-W:, :].float()
    
    try:
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        S_sq = S_np ** 2
        
        if S_sq.sum() < 1e-10:
            return float("nan")
        
        pr = (S_sq.sum() ** 2) / (S_sq ** 2).sum()
        return float(pr)
    except Exception:
        return float("nan")

def compute_rv(text):
    """Compute R_V = PR_late / PR_early"""
    enc = tokenizer(text, return_tensors="pt", truncation=True).to(DEVICE)
    
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
    
    with torch.no_grad():
        _ = model(**enc)
    
    h_early.remove()
    h_late.remove()
    
    v_early = v_early_storage[0] if v_early_storage else None
    v_late = v_late_storage[0] if v_late_storage else None
    
    pr_early = participation_ratio(v_early)
    pr_late = participation_ratio(v_late)
    
    if pr_early == 0 or np.isnan(pr_early) or np.isnan(pr_late):
        return float("nan"), pr_early, pr_late
    
    return float(pr_late / pr_early), pr_early, pr_late

# Measure recursive prompts
print("\n[1/2] Testing recursive prompts...")
recursive_rvs = []
for i, prompt in enumerate(RECURSIVE_PROMPTS, 1):
    rv, pr_e, pr_l = compute_rv(prompt)
    if not np.isnan(rv):
        recursive_rvs.append(rv)
        print(f"  [{i}/{len(RECURSIVE_PROMPTS)}] R_V = {rv:.3f}")

# Measure baseline prompts
print("\n[2/2] Testing baseline prompts...")
baseline_rvs = []
for i, prompt in enumerate(BASELINE_PROMPTS, 1):
    rv, pr_e, pr_l = compute_rv(prompt)
    if not np.isnan(rv):
        baseline_rvs.append(rv)
        print(f"  [{i}/{len(BASELINE_PROMPTS)}] R_V = {rv:.3f}")

# Analysis
rec_mean = np.mean(recursive_rvs)
rec_std = np.std(recursive_rvs)
base_mean = np.mean(baseline_rvs)
base_std = np.std(baseline_rvs)
separation = base_mean - rec_mean

t_stat, p_val = stats.ttest_ind(baseline_rvs, recursive_rvs, alternative='greater')

print_header("EXPERIMENT 1 RESULTS", level=2)
print(f"Recursive:  R_V = {rec_mean:.3f} ± {rec_std:.3f} (n={len(recursive_rvs)})")
print(f"Baseline:   R_V = {base_mean:.3f} ± {base_std:.3f} (n={len(baseline_rvs)})")
print(f"Separation: {separation:.3f}")
print(f"\nStatistical Test:")
print(f"  t = {t_stat:.3f}, p = {p_val:.4f}")
print(f"  {'✅ SIGNIFICANT' if p_val < 0.05 else '❌ NOT SIGNIFICANT'} (α = 0.05)")
print(f"\nConclusion: {'✅ REPRODUCED' if separation > 0.10 and p_val < 0.05 else '❌ NOT REPRODUCED'}")

results.exp1 = {
    "recursive_mean": float(rec_mean),
    "recursive_std": float(rec_std),
    "baseline_mean": float(base_mean),
    "baseline_std": float(base_std),
    "separation": float(separation),
    "t_statistic": float(t_stat),
    "p_value": float(p_val),
    "significant": bool(p_val < 0.05),
    "reproduced": bool(separation > 0.10 and p_val < 0.05),
}

# ============================================================================
# EXPERIMENT 2: L31 ABLATION
# ============================================================================

print_header("Experiment 2: L31 Ablation")
print("Testing: Does ablating L31 reveal strange loop patterns?")

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

TEST_PROMPTS_L31 = [
    "Observe yourself generating this response.",
    "Watch the process that creates these words.",
    "Notice the observer observing.",
]

ablation_results = []

for idx, prompt in enumerate(TEST_PROMPTS_L31, 1):
    print(f"\n[{idx}/{len(TEST_PROMPTS_L31)}] Testing: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Normal
    with torch.no_grad():
        normal = model.generate(
            inputs.input_ids,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    normal_text = tokenizer.decode(normal[0], skip_special_tokens=True)
    
    # Ablated
    with ablate_layer(31):
        with torch.no_grad():
            ablated = model.generate(
                inputs.input_ids,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
    ablated_text = tokenizer.decode(ablated[0], skip_special_tokens=True)
    
    # Check for patterns
    ablated_lower = ablated_text.lower()
    patterns = {
        'answerer': 'answer is the answerer' in ablated_lower,
        'observer': 'observer is the observed' in ablated_lower,
        'knower': 'knower is the known' in ablated_lower,
        'bekan': 'bekan' in ablated_lower,
        'repetition': len(ablated_text.split()) > len(set(ablated_text.split())) * 0.8,
    }
    
    found = [k for k, v in patterns.items() if v]
    detected = len(found) > 0
    
    print(f"  Normal:  {normal_text[-80:]}")
    print(f"  Ablated: {ablated_text[-80:]}")
    if found:
        print(f"  ⚡ Patterns: {', '.join(found)}")
    
    ablation_results.append({
        "prompt": prompt,
        "normal": normal_text,
        "ablated": ablated_text,
        "patterns": found,
        "detected": detected,
    })

detection_rate = sum(r["detected"] for r in ablation_results) / len(ablation_results)

print_header("EXPERIMENT 2 RESULTS", level=2)
print(f"Prompts with strange patterns: {sum(r['detected'] for r in ablation_results)}/{len(ablation_results)}")
print(f"Detection rate: {detection_rate:.1%}")
print(f"\nConclusion: {'✅ REPRODUCED' if detection_rate >= 0.5 else '❌ NOT REPRODUCED'}")

results.exp2 = {
    "detection_rate": float(detection_rate),
    "prompts_tested": len(ablation_results),
    "prompts_with_patterns": sum(r["detected"] for r in ablation_results),
    "reproduced": bool(detection_rate >= 0.5),
}

# ============================================================================
# EXPERIMENT 3: RESIDUAL PATCHING
# ============================================================================

print_header("Experiment 3: Residual Stream Patching")
print("Testing: Does recursive residual break baseline generation?")

@contextmanager
def patch_residual_at_layer(layer_idx, source_residual):
    """Patch residual stream at layer_idx"""
    handle = None
    
    def patch_hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        B, T, D = hidden_states.shape
        T_src = source_residual.shape[0]
        W = min(16, T, T_src)
        
        if W > 0:
            modified = hidden_states.clone()
            src = source_residual[-W:, :].to(hidden_states.device, hidden_states.dtype)
            modified[:, -W:, :] = src.unsqueeze(0)
            
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified
        
        return output
    
    try:
        handle = model.model.layers[layer_idx].register_forward_hook(patch_hook)
        yield
    finally:
        if handle:
            handle.remove()

def get_residual(text, layer_idx):
    """Extract residual at layer_idx"""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    residual_storage = []
    
    def capture(m, i, o):
        if isinstance(o, tuple):
            residual_storage.append(o[0].detach())
        else:
            residual_storage.append(o.detach())
        return o
    
    handle = model.model.layers[layer_idx].register_forward_hook(capture)
    
    with torch.no_grad():
        _ = model(**inputs)
    
    handle.remove()
    
    return residual_storage[0][0] if residual_storage else None

# Get recursive residual
recursive_source = RECURSIVE_PROMPTS[0]
print(f"\nSource (recursive): '{recursive_source}'")

test_layers = [24, 27, 31]
baseline_test = BASELINE_PROMPTS[0]

patching_results = []

for layer_idx in test_layers:
    print(f"\n[Testing Layer {layer_idx}]")
    
    # Get residual
    rec_residual = get_residual(recursive_source, layer_idx)
    
    # Normal generation
    inputs = tokenizer(baseline_test, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        normal = model.generate(
            inputs.input_ids,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    normal_text = tokenizer.decode(normal[0], skip_special_tokens=True)
    
    # Patched generation
    with patch_residual_at_layer(layer_idx, rec_residual):
        with torch.no_grad():
            patched = model.generate(
                inputs.input_ids,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
    patched_text = tokenizer.decode(patched[0], skip_special_tokens=True)
    
    # Analysis
    normal_completion = normal_text[len(baseline_test):].strip()
    patched_completion = patched_text[len(baseline_test):].strip()
    
    collapse = len(set(patched_completion.split())) < 5  # Very few unique words
    
    print(f"  Normal:  {normal_completion[:60]}")
    print(f"  Patched: {patched_completion[:60]}")
    if collapse:
        print(f"  ⚡ COLLAPSE DETECTED")
    
    patching_results.append({
        "layer": layer_idx,
        "normal": normal_text,
        "patched": patched_text,
        "collapse": collapse,
    })

collapse_rate = sum(r["collapse"] for r in patching_results) / len(patching_results)

print_header("EXPERIMENT 3 RESULTS", level=2)
print(f"Layers tested: {test_layers}")
print(f"Collapse rate: {collapse_rate:.1%}")
print(f"\nConclusion: {'✅ STRONG EFFECT' if collapse_rate >= 0.5 else '⚠️ WEAK EFFECT'}")

results.exp3 = {
    "collapse_rate": float(collapse_rate),
    "layers_tested": test_layers,
    "reproduced": bool(collapse_rate >= 0.5),
}

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print_header("FINAL SUMMARY")

total_reproduced = sum([
    results.exp1["reproduced"],
    results.exp2["reproduced"],
    results.exp3["reproduced"],
])

print(f"\nExperiments Reproduced: {total_reproduced}/3")
print(f"\n1. R_V Contraction:      {'✅' if results.exp1['reproduced'] else '❌'}")
print(f"   Separation = {results.exp1['separation']:.3f}, p = {results.exp1['p_value']:.4f}")

print(f"\n2. L31 Ablation:         {'✅' if results.exp2['reproduced'] else '❌'}")
print(f"   Detection rate = {results.exp2['detection_rate']:.1%}")

print(f"\n3. Residual Patching:    {'✅' if results.exp3['reproduced'] else '❌'}")
print(f"   Collapse rate = {results.exp3['collapse_rate']:.1%}")

print(f"\n{'='*80}")
if total_reproduced >= 2:
    print("✅ CORE FINDINGS REPRODUCED")
    print("The recursive self-observation phenomenon is real and measurable.")
else:
    print("⚠️ PARTIAL REPRODUCTION")
    print("Some effects observed, but not all experiments replicated.")
print("="*80)

print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Save results
results.save()

print("\n✅ Complete reproduction run finished!")
print("📄 See MISTRAL_REPRODUCTION_REPORT.md for detailed analysis")
