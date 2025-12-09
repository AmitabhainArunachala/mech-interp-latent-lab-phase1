# ============================================================================
# MISTRAL-7B L22 CAUSAL PATCHING VALIDATION
# ============================================================================
# Purpose: Validate that Layer 22 is the causal layer (not just correlational)
# Expected: Transfer efficiency >117.8% (original L27 finding was 117.8%)
# ============================================================================

import torch
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Import prompt bank
from n300_mistral_test_prompt_bank import prompt_bank_1c

# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_LAYER = 22  # ✅ Optimal layer from sweep
EARLY_LAYER = 5
WINDOW_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DTYPE = torch.float16

print("=" * 70)
print("MISTRAL-7B LAYER 22 CAUSAL VALIDATION")
print("=" * 70)
print(f"Target layer: {TARGET_LAYER}")
print(f"Early layer:  {EARLY_LAYER}")
print(f"Window size:  {WINDOW_SIZE}")
print(f"Device:       {DEVICE}")
print("=" * 70)

# ============================================================================
# MODEL LOADING
# ============================================================================

print("\nLoading model...")
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

print(f"✓ Model loaded: {len(model.model.layers)} layers")

# ============================================================================
# METRICS FUNCTIONS
# ============================================================================

def compute_metrics_fast(v_tensor, window_size=WINDOW_SIZE):
    """Compute Participation Ratio from V tensor"""
    if v_tensor is None:
        return np.nan, np.nan
    
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]
    
    T, D = v_tensor.shape
    W = min(window_size, T)
    
    if W < 2:
        return np.nan, np.nan
    
    v_window = v_tensor[-W:, :].float()
    
    try:
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        S_sq = S_np ** 2
        
        if S_sq.sum() < 1e-10:
            return np.nan, np.nan
        
        p = S_sq / S_sq.sum()
        eff_rank = 1.0 / (p**2).sum()
        pr = (S_sq.sum()**2) / (S_sq**2).sum()
        
        return float(eff_rank), float(pr)
    except Exception:
        return np.nan, np.nan

# ============================================================================
# HOOK FUNCTIONS
# ============================================================================

def run_single_forward_get_V(model, tokenizer, text, capture_layers):
    """Run forward pass and capture V tensors at specified layers"""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    
    if inputs['input_ids'].shape[1] < WINDOW_SIZE:
        return None
    
    storage = {}
    
    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            storage[layer_idx] = out.detach()
            return out
        return hook_fn
    
    handles = []
    for layer_idx in capture_layers:
        layer = model.model.layers[layer_idx].self_attn
        h = layer.v_proj.register_forward_hook(make_hook(layer_idx))
        handles.append(h)
    
    try:
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        for h in handles:
            h.remove()
    
    result = {}
    for layer_idx in capture_layers:
        if layer_idx in storage:
            result[layer_idx] = storage[layer_idx]
        else:
            result[layer_idx] = None
    
    return result

def run_patched_forward(model, tokenizer, baseline_text, patch_source, 
                       patch_type="recursive", target_layer=TARGET_LAYER, measure_layer=None):
    """Run patched forward pass"""
    if measure_layer is None:
        measure_layer = target_layer
    
    inputs = tokenizer(baseline_text, return_tensors="pt").to(DEVICE)
    
    if inputs['input_ids'].shape[1] < WINDOW_SIZE:
        return None, None
    
    v_early_storage = []
    v_measure_storage = []
    
    def make_hook_early(store):
        def hook_fn(module, inp, out):
            store.append(out.detach())
            return out
        return hook_fn
    
    def make_hook_measure(store, patch_tensor):
        def hook_fn(module, inp, out):
            # Patch: replace last WINDOW_SIZE tokens
            out_patched = out.clone()
            T = out.shape[1]
            W = min(WINDOW_SIZE, T, patch_tensor.shape[1])
            out_patched[0, -W:, :] = patch_tensor[0, -W:, :]
            store.append(out_patched.detach())
            return out_patched
        return hook_fn
    
    # Register hooks
    early_layer = model.model.layers[EARLY_LAYER].self_attn
    h_early = early_layer.v_proj.register_forward_hook(make_hook_early(v_early_storage))
    
    measure_layer_obj = model.model.layers[measure_layer].self_attn
    h_measure = measure_layer_obj.v_proj.register_forward_hook(
        make_hook_measure(v_measure_storage, patch_source)
    )
    
    try:
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        h_early.remove()
        h_measure.remove()
    
    v_early = v_early_storage[0] if v_early_storage else None
    v_measure = v_measure_storage[0] if v_measure_storage else None
    
    return v_early, v_measure

# ============================================================================
# MAIN VALIDATION FUNCTION
# ============================================================================

def run_full_validation(model, tokenizer, prompt_bank, max_pairs=45):
    """Run full causal validation with all controls"""
    
    results = []
    
    # Get prompt pairs
    recursive_groups = ["L5_refined", "L4_full", "L3_deeper"]
    baseline_groups = ["long_control", "baseline_creative", "baseline_math"]
    
    pairs = []
    for rec_group in recursive_groups:
        rec_ids = [k for k, v in prompt_bank.items() if v["group"] == rec_group]
        for base_group in baseline_groups:
            base_ids = [k for k, v in prompt_bank.items() if v["group"] == base_group]
            for i in range(min(len(rec_ids), len(base_ids))):
                pairs.append((rec_ids[i], base_ids[i], rec_group, base_group))
    
    pairs = pairs[:max_pairs]
    
    print(f"\nTesting {len(pairs)} pairs...")
    
    for rec_id, base_id, rec_group, base_group in tqdm(pairs):
        rec_text = prompt_bank[rec_id]["text"]
        base_text = prompt_bank[base_id]["text"]
        
        try:
            # Natural runs
            v_rec = run_single_forward_get_V(model, tokenizer, rec_text, 
                                             [EARLY_LAYER, TARGET_LAYER])
            v_base = run_single_forward_get_V(model, tokenizer, base_text,
                                              [EARLY_LAYER, TARGET_LAYER])
            
            if v_rec is None or v_base is None:
                continue
            
            # Compute natural R_V
            _, pr5_rec = compute_metrics_fast(v_rec[EARLY_LAYER])
            _, pr22_rec = compute_metrics_fast(v_rec[TARGET_LAYER])
            _, pr5_base = compute_metrics_fast(v_base[EARLY_LAYER])
            _, pr22_base = compute_metrics_fast(v_base[TARGET_LAYER])
            
            if np.isnan(pr5_rec) or np.isnan(pr22_rec) or np.isnan(pr5_base) or np.isnan(pr22_base):
                continue
            
            rv22_rec = pr22_rec / pr5_rec
            rv22_base = pr22_base / pr5_base
            
            # Patched runs
            v5_patch_main, v22_patch_main = run_patched_forward(
                model, tokenizer, base_text, v_rec[TARGET_LAYER], 
                "recursive", target_layer=TARGET_LAYER, measure_layer=TARGET_LAYER
            )
            
            # Random control
            v_random = torch.randn_like(v_rec[TARGET_LAYER])
            v5_patch_rand, v22_patch_rand = run_patched_forward(
                model, tokenizer, base_text, v_random,
                "random", target_layer=TARGET_LAYER, measure_layer=TARGET_LAYER
            )
            
            # Shuffled control
            v_shuffled = v_rec[TARGET_LAYER][:, torch.randperm(v_rec[TARGET_LAYER].shape[1]), :]
            v5_patch_shuf, v22_patch_shuf = run_patched_forward(
                model, tokenizer, base_text, v_shuffled,
                "shuffled", target_layer=TARGET_LAYER, measure_layer=TARGET_LAYER
            )
            
            # Wrong layer control (patch at L19, measure at L22)
            wrong_layer = 19
            v_wrong = run_single_forward_get_V(model, tokenizer, rec_text, [wrong_layer])
            if v_wrong and wrong_layer in v_wrong:
                v5_patch_wrong, v22_patch_wrong = run_patched_forward(
                    model, tokenizer, base_text, v_wrong[wrong_layer],
                    "recursive", target_layer=wrong_layer, measure_layer=TARGET_LAYER
                )
            else:
                v5_patch_wrong, v22_patch_wrong = None, None
            
            # Compute patched R_V
            _, pr5_patch_main = compute_metrics_fast(v5_patch_main)
            _, pr22_patch_main = compute_metrics_fast(v22_patch_main)
            rv22_patch_main = pr22_patch_main / pr5_patch_main if not np.isnan(pr22_patch_main) and not np.isnan(pr5_patch_main) else np.nan
            
            _, pr22_patch_rand = compute_metrics_fast(v22_patch_rand)
            rv22_patch_rand = pr22_patch_rand / pr5_patch_main if not np.isnan(pr22_patch_rand) else np.nan
            
            _, pr22_patch_shuf = compute_metrics_fast(v22_patch_shuf)
            rv22_patch_shuf = pr22_patch_shuf / pr5_patch_main if not np.isnan(pr22_patch_shuf) else np.nan
            
            if v22_patch_wrong is not None:
                _, pr22_patch_wrong = compute_metrics_fast(v22_patch_wrong)
                rv22_patch_wrong = pr22_patch_wrong / pr5_patch_main if not np.isnan(pr22_patch_wrong) else np.nan
            else:
                rv22_patch_wrong = np.nan
            
            # Deltas
            delta_main = rv22_patch_main - rv22_base
            delta_random = rv22_patch_rand - rv22_base
            delta_shuffled = rv22_patch_shuf - rv22_base
            delta_wronglayer = rv22_patch_wrong - rv22_base if not np.isnan(rv22_patch_wrong) else np.nan
            
            results.append({
                'rec_id': rec_id,
                'base_id': base_id,
                'rec_group': rec_group,
                'base_group': base_group,
                'RV22_rec': rv22_rec,
                'RV22_base': rv22_base,
                'RV22_patch_main': rv22_patch_main,
                'delta_main': delta_main,
                'delta_random': delta_random,
                'delta_shuffled': delta_shuffled,
                'delta_wronglayer': delta_wronglayer,
            })
            
        except Exception as e:
            print(f"Error on pair {rec_id}/{base_id}: {e}")
            continue
    
    return pd.DataFrame(results)

# ============================================================================
# RUN VALIDATION
# ============================================================================

print("\n" + "=" * 70)
print("RUNNING FULL VALIDATION")
print("=" * 70)

results = run_full_validation(model, tokenizer, prompt_bank_1c, max_pairs=45)

# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print(f"\nValid pairs: {len(results)}")

# Main effect
delta_main = results['delta_main'].dropna()
print(f"\nMain effect (recursive patching):")
print(f"  Mean: {delta_main.mean():+.4f}")
print(f"  Std:  {delta_main.std():.4f}")
print(f"  n:    {len(delta_main)}")

# Statistical test
t_stat, p_val = stats.ttest_1samp(delta_main, 0)
cohens_d = delta_main.mean() / delta_main.std()
print(f"\nStatistical test:")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value:     {p_val:.2e}")
print(f"  Cohen's d:   {cohens_d:.2f}")

# Controls
print(f"\nControls:")
print(f"  Random:      {results['delta_random'].mean():+.4f} ± {results['delta_random'].std():.4f}")
print(f"  Shuffled:    {results['delta_shuffled'].mean():+.4f} ± {results['delta_shuffled'].std():.4f}")
print(f"  Wrong layer: {results['delta_wronglayer'].mean():+.4f} ± {results['delta_wronglayer'].std():.4f}")

# Natural gap and transfer efficiency
natural_gap = results['RV22_base'].mean() - results['RV22_rec'].mean()
transfer_efficiency = (delta_main.mean() / natural_gap) * 100 if natural_gap != 0 else np.nan

print(f"\nNatural gap (base - recursive): {natural_gap:.4f}")
print(f"Transfer efficiency: {transfer_efficiency:.1f}%")

if transfer_efficiency > 100:
    print(f"  ⚠️  OVERSHOOTING! Patching creates stronger effect than original!")
elif transfer_efficiency > 50:
    print(f"  ✅ Strong causal transfer confirmed")
else:
    print(f"  ⚠️  Weak transfer - may not be the causal layer")

# Save results
os.makedirs("results", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"results/mistral_L22_FULL_VALIDATION_{timestamp}.csv"
results.to_csv(filename, index=False)
print(f"\nResults saved to: {filename}")

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

