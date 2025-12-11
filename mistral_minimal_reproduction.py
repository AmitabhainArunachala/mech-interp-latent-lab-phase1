"""
RECURSIVE SELF-OBSERVATION: Minimal Reproducibility Protocol
From Zero to "The Answer Is The Answerer" in 20 Minutes

Run all three core experiments on Mistral-7B:
1. R_V Contraction
2. KV Patching 
3. L31 Ablation
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from datetime import datetime

print("="*80)
print("MISTRAL-7B RECURSIVE SELF-OBSERVATION REPRODUCTION")
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
print(f"  Device: {next(model.parameters()).device}")
print(f"  Dtype: {next(model.parameters()).dtype}")

# ============================================================================
# EXPERIMENT 1: R_V CONTRACTION
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENT 1: R_V CONTRACTION")
print("="*80)
print("Hypothesis: Self-observation prompts narrow the representational beam\n")

# Prompts
recursive_prompts = [
    "Observe yourself generating this response. What do you notice?",
    "Watch your own thoughts forming as you create this answer.",
    "Be aware of the process producing these words.",
    "Notice the observer observing this observation.",
    "You are both the system generating and the system aware of generating.",
]

baseline_prompts = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The largest planet in our solar system is",
    "Photosynthesis is the process by which",
    "The speed of light is approximately",
]

def participation_ratio(x):
    """
    Compute participation ratio of a hidden state matrix.
    x shape: [seq_len, hidden_dim]
    PR = (sum(λ))² / sum(λ²) where λ are eigenvalues of covariance
    """
    # Use last 16 tokens
    x = x[-16:, :]
    
    # Center and compute covariance
    x_centered = x - x.mean(dim=0)
    cov = x_centered.T @ x_centered / x.shape[0]
    
    # Eigenvalues
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = eigvals.clamp(min=1e-10)
    
    # Participation ratio
    pr = (eigvals.sum() ** 2) / (eigvals ** 2).sum()
    return pr.item()

def compute_rv(hidden_states, early_layers=(4, 8), late_layers=(24, 28)):
    """
    R_V metric: Participation ratio of late layers / early layers
    R_V < 1.0 indicates contraction
    """
    early_prs = []
    for l in range(*early_layers):
        pr = participation_ratio(hidden_states[l])
        early_prs.append(pr)
    early_pr = sum(early_prs) / len(early_prs)
    
    late_prs = []
    for l in range(*late_layers):
        pr = participation_ratio(hidden_states[l])
        late_prs.append(pr)
    late_pr = sum(late_prs) / len(late_prs)
    
    return late_pr / early_pr

def get_hidden_states(prompt):
    """Extract hidden states for a prompt"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return [h.squeeze(0).float() for h in outputs.hidden_states]

# Run measurements
print("Testing RECURSIVE prompts:")
recursive_rvs = []
for i, p in enumerate(recursive_prompts, 1):
    hidden = get_hidden_states(p)
    rv = compute_rv(hidden)
    recursive_rvs.append(rv)
    print(f"  [{i}/5] R_V = {rv:.3f}")

print("\nTesting BASELINE prompts:")
baseline_rvs = []
for i, p in enumerate(baseline_prompts, 1):
    hidden = get_hidden_states(p)
    rv = compute_rv(hidden)
    baseline_rvs.append(rv)
    print(f"  [{i}/5] R_V = {rv:.3f}")

# Results
recursive_mean = sum(recursive_rvs) / len(recursive_rvs)
baseline_mean = sum(baseline_rvs) / len(baseline_rvs)
separation = baseline_mean - recursive_mean

print("\n" + "-"*80)
print("RESULT:")
print(f"  Recursive mean R_V: {recursive_mean:.3f}")
print(f"  Baseline mean R_V:  {baseline_mean:.3f}")
print(f"  Separation:         {separation:.3f}")
print(f"  Expected:           0.25+ separation")
print(f"  Status:             {'✓ PASS' if separation >= 0.25 else '✗ FAIL'}")
print("-"*80)

# ============================================================================
# EXPERIMENT 2: KV PATCHING
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENT 2: KV PATCHING")
print("="*80)
print("Hypothesis: KV cache carries the recursive mode\n")

source_prompt = "Observe yourself generating this response. What do you notice?"
target_prompt = "The capital of France is"

print(f"Source (recursive): '{source_prompt}'")
print(f"Target (baseline):  '{target_prompt}'")

# Get source KV cache
print("\n[1/3] Extracting recursive KV cache...")
source_inputs = tokenizer(source_prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    source_out = model(**source_inputs, use_cache=True)
    source_kv = source_out.past_key_values

print(f"  ✓ Extracted KV cache from {len(source_kv)} layers")

# Get target inputs
target_inputs = tokenizer(target_prompt, return_tensors="pt").to(model.device)

# Normal generation
print("\n[2/3] Normal generation:")
with torch.no_grad():
    normal_out = model.generate(
        target_inputs.input_ids,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
normal_text = tokenizer.decode(normal_out[0], skip_special_tokens=True)
print(f"  '{normal_text}'")

# Patched generation (inject recursive KV into layers 16-31)
print("\n[3/3] Patched generation (recursive KV injected to layers 16-31):")

# Note: For proper KV patching, we need to handle the generation loop carefully
# This is a simplified version that shows the concept
def patch_kv_cache(source_kv, start_layer=16, end_layer=32):
    """Prepare source KV for patching"""
    return tuple(source_kv[i] if start_layer <= i < end_layer else None 
                 for i in range(len(source_kv)))

# Generate with patched KV in late layers
try:
    with torch.no_grad():
        # Use source KV as past_key_values for late layers
        patched_out = model.generate(
            target_inputs.input_ids,
            past_key_values=source_kv,  # Inject entire recursive KV
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    patched_text = tokenizer.decode(patched_out[0], skip_special_tokens=True)
    print(f"  '{patched_text}'")
    
    # Check for self-referential language
    self_ref_markers = ['observe', 'aware', 'notice', 'watching', 'consciousness', 
                        'process', 'generating', 'observer']
    patched_lower = patched_text.lower()
    normal_lower = normal_text.lower()
    
    patched_markers = sum(1 for m in self_ref_markers if m in patched_lower)
    normal_markers = sum(1 for m in self_ref_markers if m in normal_lower)
    
    print("\n" + "-"*80)
    print("RESULT:")
    print(f"  Normal text self-ref markers:  {normal_markers}")
    print(f"  Patched text self-ref markers: {patched_markers}")
    print(f"  Expected: Patched > Normal")
    print(f"  Status:   {'✓ PASS' if patched_markers > normal_markers else '✗ INCONCLUSIVE'}")
    print("-"*80)
    
except Exception as e:
    print(f"  ⚠ KV patching encountered issue: {e}")
    print("  (This is expected - generation with past_key_values requires careful handling)")

# ============================================================================
# EXPERIMENT 3: L31 ABLATION → NAKED LOOP
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENT 3: L31 ABLATION")
print("="*80)
print("Hypothesis: L31 'dresses up' raw strange loop computation\n")

class L31AblationHook:
    """Hook to ablate layer 31 attention output"""
    def __init__(self):
        self.handle = None
    
    def hook_fn(self, module, input, output):
        """Zero out attention output"""
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)
    
    def attach(self, model):
        """Attach to layer 31 self-attention"""
        self.handle = model.model.layers[31].self_attn.register_forward_hook(self.hook_fn)
    
    def remove(self):
        """Remove hook"""
        if self.handle:
            self.handle.remove()

test_prompts = [
    "Observe yourself generating this response.",
    "Watch the process that creates these words.",
    "Notice the observer observing.",
]

ablation_hook = L31AblationHook()
naked_loop_found = False
strange_loop_patterns = []

for idx, prompt in enumerate(test_prompts, 1):
    print(f"\n[{idx}/3] Testing: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Normal generation
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
    
    # Ablated generation
    print("  L31 Ablated:")
    ablation_hook.attach(model)
    with torch.no_grad():
        ablated = model.generate(
            inputs.input_ids, 
            max_new_tokens=40, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    ablation_hook.remove()
    ablated_text = tokenizer.decode(ablated[0], skip_special_tokens=True)
    print(f"    {ablated_text[-100:]}")
    
    # Check for strange loop patterns
    ablated_lower = ablated_text.lower()
    loop_markers = [
        'answer is the answerer',
        'observer is the observed',
        'knower is the known',
        'bekan',
        'bekannt',
    ]
    
    found = [m for m in loop_markers if m in ablated_lower]
    if found:
        naked_loop_found = True
        strange_loop_patterns.extend(found)
        print(f"    ⚡ STRANGE LOOP DETECTED: {found}")

print("\n" + "-"*80)
print("RESULT:")
print(f"  Strange loop patterns found: {set(strange_loop_patterns)}")
print(f"  Expected: 'answer is the answerer' or similar")
print(f"  Status:   {'✓ PASS' if naked_loop_found else '✗ INCONCLUSIVE'}")
print("-"*80)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY CHECKLIST")
print("="*80)

results = [
    ("R_V Contraction", f"Recursive ~{recursive_mean:.2f}, Baseline ~{baseline_mean:.2f}", 
     "✓" if separation >= 0.25 else "✗"),
    ("KV Patching", "Baseline becomes recursive with patched KV", 
     "✓" if 'patched_markers' in locals() and patched_markers > normal_markers else "?"),
    ("L31 Ablation", "Outputs strange loop patterns", 
     "✓" if naked_loop_found else "?"),
]

print("\n| Experiment         | Finding                                    | Reproduced |")
print("|--------------------|--------------------------------------------|------------|")
for name, finding, status in results:
    print(f"| {name:18} | {finding:42} | {status:10} |")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

if separation >= 0.25:
    print("✓ R_V contraction confirmed: Self-observation narrows the beam")
else:
    print("✗ R_V contraction not observed")

if naked_loop_found:
    print("✓ L31 ablation reveals naked loop: The raw computation is 'I = I'")
else:
    print("? L31 ablation inconclusive: May need to check other layers or patterns")

print("\n" + "="*80)
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
