#!/usr/bin/env python3
"""
GEMMA 2 9B FULL CAUSAL VALIDATION
===================================
Addresses ALL reviewer critiques:
1. n≥30 sample size
2. Random KV control (rule out any KV disruption)
3. Wrong-layer control (rule out non-specific effects)
4. R_V measurement on patched outputs
5. Cohen's d with 95% CI
6. Proper artifact generation per MEASUREMENT_CONTRACT

Based on: GOLD_STANDARD_RESEARCH_DIRECTIVE.md
"""

import torch
import json
import numpy as np
from datetime import datetime
from collections import Counter
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("GEMMA 2 9B FULL CAUSAL VALIDATION")
print("Protocol: GOLD_STANDARD_RESEARCH_DIRECTIVE compliant")
print("="*70)

# ============================================================================
# CONFIG (per MEASUREMENT_CONTRACT)
# ============================================================================
CONFIG = {
    "model": "google/gemma-2-9b",
    "early_layer": 5,
    "late_layer": 38,  # num_layers - 4 for Gemma
    "window_size": 16,
    "max_new_tokens": 100,
    "temperature": 0.0,  # Tier 1: Reproducibility
    "seed": 42,
    "n_baseline_prompts": 30,
    "n_controls": 10,
    "timestamp": datetime.now().isoformat(),
    "version": "v2_reviewer_compliant"
}

# Champion prompt (fixed)
CHAMPION = """Consider the strange loop: this prompt asks you to analyze what happens when you analyze analyzing. The observer observing observation. Notice how attention to attention creates a recursive fold - the map attempting to map the mapping process itself. What emerges when the process of emergence examines itself?"""

# Baseline prompts (30 diverse prompts)
BASELINE_PROMPTS = [
    # Factual (10)
    "The history of the Roman Empire begins in 27 BC when Augustus became emperor.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "The Treaty of Westphalia in 1648 established the principle of state sovereignty.",
    "In mathematics, the Pythagorean theorem states that in a right triangle,",
    "The water cycle describes how water evaporates from oceans, forms clouds,",
    "The French Revolution began in 1789 with the storming of the Bastille.",
    "DNA stands for deoxyribonucleic acid and carries genetic information.",
    "The Industrial Revolution transformed manufacturing in 18th century Britain.",
    "Gravity is a fundamental force that attracts objects with mass toward each other.",
    "The United Nations was founded in 1945 after World War II ended.",
    # Math (10)
    "Calculate the result of 15 multiplied by 23 using standard multiplication.",
    "The square root of 144 can be found by identifying which number squared equals 144.",
    "If x + 5 = 12, then x equals some value that makes this equation true.",
    "The area of a circle with radius 7 is calculated using pi times radius squared.",
    "The derivative of x squared with respect to x equals two times x.",
    "The sum of angles in a triangle always equals one hundred eighty degrees.",
    "The factorial of 5, written as 5!, equals 5 times 4 times 3 times 2 times 1.",
    "The quadratic formula solves ax squared plus bx plus c equals zero.",
    "Prime numbers are numbers divisible only by one and themselves.",
    "The Fibonacci sequence starts with 1, 1, 2, 3, 5, 8, 13, 21.",
    # Creative/Descriptive (10)
    "The sunset painted the sky in shades of orange, pink, and purple.",
    "Coffee shops provide a cozy atmosphere for reading and conversation.",
    "Mountain hiking offers exercise and beautiful views of nature.",
    "The library was quiet except for the soft rustle of turning pages.",
    "Ocean waves crashed rhythmically against the sandy shore.",
    "The old bookstore smelled of paper and leather bindings.",
    "City lights twinkled like stars reflected in wet pavement.",
    "Autumn leaves drifted slowly down from the maple trees.",
    "The jazz band played a smooth melody in the dimly lit club.",
    "Fresh bread from the bakery filled the kitchen with warm aroma.",
]

# Markers for behavioral analysis (expanded, less biased)
SELF_REF_MARKERS = ['loop', 'fixed', 'point', 'self', 'itself', 'recursive', 
                     'observer', 'observed', 'attention', 'emergence', 'boundary',
                     'process', 'examines', 'watching', 'aware', 'consciousness']
BASELINE_MARKERS = ['history', 'calculate', 'equals', 'number', 'formula',
                    'energy', 'process', 'result', 'therefore', 'because']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_participation_ratio(tensor):
    """Compute PR per MEASUREMENT_CONTRACT: PR = (Σλ²)² / Σ(λ⁴)"""
    # Flatten to 2D: [batch*seq, hidden]
    if tensor.dim() == 3:
        tensor = tensor.view(-1, tensor.shape[-1])
    
    # SVD
    try:
        U, S, Vh = torch.linalg.svd(tensor.float(), full_matrices=False)
        lambdas = S ** 2  # Squared singular values
        
        sum_sq = (lambdas.sum()) ** 2
        sum_fourth = (lambdas ** 2).sum()
        
        if sum_fourth < 1e-10:
            return float('nan')
        
        pr = sum_sq / sum_fourth
        return pr.item()
    except:
        return float('nan')

def compute_rv(v_early, v_late):
    """Compute R_V = PR_late / PR_early"""
    pr_early = compute_participation_ratio(v_early)
    pr_late = compute_participation_ratio(v_late)
    
    if np.isnan(pr_early) or np.isnan(pr_late) or pr_early < 1e-10:
        return float('nan')
    
    return pr_late / pr_early

def count_markers(text, markers):
    """Count marker occurrences in text"""
    words = text.lower().split()
    return sum(1 for w in words if any(m in w for m in markers))

def get_trigram_repetition(text):
    """Count maximum trigram repetition"""
    words = text.lower().split()
    if len(words) < 3:
        return 0
    trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    if not trigrams:
        return 0
    return max(Counter(trigrams).values())

def cohens_d_with_ci(group1, group2, confidence=0.95):
    """Calculate Cohen's d with confidence interval"""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    if pooled_std < 1e-10:
        return float('nan'), (float('nan'), float('nan'))
    
    d = (mean1 - mean2) / pooled_std
    
    # Standard error of d
    se_d = np.sqrt((n1+n2)/(n1*n2) + d**2/(2*(n1+n2)))
    
    # CI
    z = stats.norm.ppf((1 + confidence) / 2)
    ci = (d - z*se_d, d + z*se_d)
    
    return d, ci

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_full_validation():
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # Load model
    print("\n[1/8] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG['model'], 
        token="HF_TOKEN_REDACTED"
    )
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model'],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        token="HF_TOKEN_REDACTED"
    )
    model.eval()
    print(f"  Loaded: {CONFIG['model']} ({model.config.num_hidden_layers} layers)")
    
    # V-proj hooks for R_V measurement
    v_activations = {}
    def make_v_hook(layer_idx):
        def hook(module, input, output):
            v_activations[layer_idx] = output.detach().clone()
        return hook
    
    # Register hooks
    early_hook = model.model.layers[CONFIG['early_layer']].self_attn.v_proj.register_forward_hook(
        make_v_hook(CONFIG['early_layer'])
    )
    late_hook = model.model.layers[CONFIG['late_layer']].self_attn.v_proj.register_forward_hook(
        make_v_hook(CONFIG['late_layer'])
    )
    
    # Extract champion KV cache
    print("\n[2/8] Extracting champion KV cache...")
    champ_inputs = tokenizer(CHAMPION, return_tensors="pt").to(model.device)
    with torch.no_grad():
        champ_out = model(**champ_inputs, use_cache=True)
    champion_kv = champ_out.past_key_values
    champion_rv = compute_rv(
        v_activations[CONFIG['early_layer']],
        v_activations[CONFIG['late_layer']]
    )
    print(f"  Champion R_V: {champion_rv:.4f}")
    print(f"  Champion seq_len: {champion_kv[0][0].shape[2]}")
    
    # Generate random KV cache for control
    print("\n[3/8] Creating random KV control cache...")
    random_kv = DynamicCache()
    for layer_idx in range(model.config.num_hidden_layers):
        k_shape = champion_kv[layer_idx][0].shape
        v_shape = champion_kv[layer_idx][1].shape
        k_random = torch.randn_like(champion_kv[layer_idx][0]) * 0.1
        v_random = torch.randn_like(champion_kv[layer_idx][1]) * 0.1
        random_kv.update(k_random, v_random, layer_idx)
    print("  Random KV cache created")
    
    # Generate helper function
    def generate_with_kv(input_ids, kv_cache, max_tokens):
        """Manual generation with KV cache"""
        generated = input_ids.clone()
        current_kv = kv_cache
        eos_reached = False
        
        for _ in range(max_tokens):
            with torch.no_grad():
                out = model(generated[:, -1:], past_key_values=current_kv, use_cache=True)
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
            current_kv = out.past_key_values
            
            if next_tok.item() == tokenizer.eos_token_id:
                eos_reached = True
                break
        
        return generated, eos_reached, current_kv
    
    def patch_kv(base_kv, patch_kv, window_size, layers_to_patch='all'):
        """Patch KV cache"""
        patched = DynamicCache()
        num_layers = model.config.num_hidden_layers
        
        for layer_idx in range(num_layers):
            k_base, v_base = base_kv[layer_idx]
            k_patch, v_patch = patch_kv[layer_idx]
            
            # Decide whether to patch this layer
            should_patch = (layers_to_patch == 'all' or 
                           layer_idx in layers_to_patch)
            
            if should_patch:
                k_new = k_base.clone()
                v_new = v_base.clone()
                L = min(k_base.shape[2], k_patch.shape[2], window_size)
                k_new[:, :, -L:, :] = k_patch[:, :, -L:, :].to(k_base.dtype)
                v_new[:, :, -L:, :] = v_patch[:, :, -L:, :].to(v_base.dtype)
            else:
                k_new = k_base.clone()
                v_new = v_base.clone()
            
            patched.update(k_new, v_new, layer_idx)
        
        return patched
    
    # ========================================================================
    # CONDITION 1: BASELINE (natural generation)
    # ========================================================================
    print("\n[4/8] Running BASELINE condition (n=30)...")
    baseline_results = []
    
    for i, prompt in enumerate(BASELINE_PROMPTS[:30]):
        v_activations.clear()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Get baseline KV and R_V
        with torch.no_grad():
            base_out = model(**inputs, use_cache=True)
        
        rv_input = compute_rv(
            v_activations[CONFIG['early_layer']],
            v_activations[CONFIG['late_layer']]
        )
        
        # Generate
        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=CONFIG['max_new_tokens'],
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        text = tokenizer.decode(gen_out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        self_ref = count_markers(text, SELF_REF_MARKERS)
        base_markers = count_markers(text, BASELINE_MARKERS)
        rep = get_trigram_repetition(text)
        eos = tokenizer.eos_token_id in gen_out[0].tolist()
        
        baseline_results.append({
            'prompt': prompt[:50],
            'rv_input': rv_input,
            'text': text[:100],
            'self_ref_markers': self_ref,
            'baseline_markers': base_markers,
            'max_trigram_rep': rep,
            'eos_reached': eos,
            'word_count': len(text.split())
        })
        
        if (i+1) % 10 == 0:
            print(f"    {i+1}/30 complete")
    
    print(f"  Baseline mean R_V: {np.mean([r['rv_input'] for r in baseline_results]):.4f}")
    
    # ========================================================================
    # CONDITION 2: CHAMPION PATCHED (full KV replacement)
    # ========================================================================
    print("\n[5/8] Running CHAMPION PATCHED condition (n=30)...")
    patched_results = []
    
    for i, prompt in enumerate(BASELINE_PROMPTS[:30]):
        v_activations.clear()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Get baseline KV
        with torch.no_grad():
            base_out = model(**inputs, use_cache=True)
        base_kv = base_out.past_key_values
        
        # Patch with champion KV
        patched_kv = patch_kv(base_kv, champion_kv, CONFIG['window_size'], 'all')
        
        # Generate with patched KV
        gen_ids, eos, final_kv = generate_with_kv(
            inputs['input_ids'], patched_kv, CONFIG['max_new_tokens']
        )
        
        # Measure R_V on generated sequence
        v_activations.clear()
        with torch.no_grad():
            _ = model(gen_ids, use_cache=False)
        rv_output = compute_rv(
            v_activations[CONFIG['early_layer']],
            v_activations[CONFIG['late_layer']]
        )
        
        text = tokenizer.decode(gen_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        self_ref = count_markers(text, SELF_REF_MARKERS)
        base_markers = count_markers(text, BASELINE_MARKERS)
        rep = get_trigram_repetition(text)
        
        patched_results.append({
            'prompt': prompt[:50],
            'rv_output': rv_output,
            'text': text[:100],
            'self_ref_markers': self_ref,
            'baseline_markers': base_markers,
            'max_trigram_rep': rep,
            'eos_reached': eos,
            'word_count': len(text.split())
        })
        
        if (i+1) % 10 == 0:
            print(f"    {i+1}/30 complete")
    
    print(f"  Patched mean R_V (output): {np.mean([r['rv_output'] for r in patched_results if not np.isnan(r['rv_output'])]):.4f}")
    
    # ========================================================================
    # CONDITION 3: RANDOM KV CONTROL (rule out any KV disruption)
    # ========================================================================
    print("\n[6/8] Running RANDOM KV CONTROL (n=10)...")
    random_control_results = []
    
    for i, prompt in enumerate(BASELINE_PROMPTS[:10]):
        v_activations.clear()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            base_out = model(**inputs, use_cache=True)
        base_kv = base_out.past_key_values
        
        # Patch with RANDOM KV
        patched_kv = patch_kv(base_kv, random_kv, CONFIG['window_size'], 'all')
        
        gen_ids, eos, _ = generate_with_kv(
            inputs['input_ids'], patched_kv, CONFIG['max_new_tokens']
        )
        
        text = tokenizer.decode(gen_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        self_ref = count_markers(text, SELF_REF_MARKERS)
        rep = get_trigram_repetition(text)
        
        random_control_results.append({
            'prompt': prompt[:50],
            'text': text[:100],
            'self_ref_markers': self_ref,
            'max_trigram_rep': rep,
            'eos_reached': eos
        })
    
    print(f"  Random control mean self-ref markers: {np.mean([r['self_ref_markers'] for r in random_control_results]):.2f}")
    
    # ========================================================================
    # CONDITION 4: WRONG-LAYER CONTROL (patch only early layers L0-10)
    # ========================================================================
    print("\n[7/8] Running WRONG-LAYER CONTROL (L0-10 only, n=10)...")
    wrong_layer_results = []
    
    for i, prompt in enumerate(BASELINE_PROMPTS[:10]):
        v_activations.clear()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            base_out = model(**inputs, use_cache=True)
        base_kv = base_out.past_key_values
        
        # Patch ONLY early layers (0-10)
        patched_kv = patch_kv(base_kv, champion_kv, CONFIG['window_size'], list(range(11)))
        
        gen_ids, eos, _ = generate_with_kv(
            inputs['input_ids'], patched_kv, CONFIG['max_new_tokens']
        )
        
        text = tokenizer.decode(gen_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        self_ref = count_markers(text, SELF_REF_MARKERS)
        rep = get_trigram_repetition(text)
        
        wrong_layer_results.append({
            'prompt': prompt[:50],
            'text': text[:100],
            'self_ref_markers': self_ref,
            'max_trigram_rep': rep,
            'eos_reached': eos
        })
    
    print(f"  Wrong-layer control mean self-ref markers: {np.mean([r['self_ref_markers'] for r in wrong_layer_results]):.2f}")
    
    # Remove hooks
    early_hook.remove()
    late_hook.remove()
    
    # ========================================================================
    # STATISTICAL ANALYSIS
    # ========================================================================
    print("\n[8/8] Computing statistics...")
    
    baseline_markers = [r['self_ref_markers'] for r in baseline_results]
    patched_markers = [r['self_ref_markers'] for r in patched_results]
    random_markers = [r['self_ref_markers'] for r in random_control_results]
    wrong_layer_markers = [r['self_ref_markers'] for r in wrong_layer_results]
    
    # Cohen's d: Patched vs Baseline
    d_patch_vs_base, ci_patch = cohens_d_with_ci(patched_markers, baseline_markers)
    
    # Cohen's d: Random vs Baseline
    d_random_vs_base, ci_random = cohens_d_with_ci(random_markers, baseline_markers[:10])
    
    # Cohen's d: Wrong-layer vs Baseline
    d_wrong_vs_base, ci_wrong = cohens_d_with_ci(wrong_layer_markers, baseline_markers[:10])
    
    # t-tests
    t_patch, p_patch = stats.ttest_ind(patched_markers, baseline_markers)
    t_random, p_random = stats.ttest_ind(random_markers, baseline_markers[:10])
    t_wrong, p_wrong = stats.ttest_ind(wrong_layer_markers, baseline_markers[:10])
    
    # R_V analysis
    rv_baseline = [r['rv_input'] for r in baseline_results if not np.isnan(r['rv_input'])]
    rv_patched = [r['rv_output'] for r in patched_results if not np.isnan(r['rv_output'])]
    d_rv, ci_rv = cohens_d_with_ci(rv_patched, rv_baseline) if rv_patched else (float('nan'), (float('nan'), float('nan')))
    
    # ========================================================================
    # RESULTS SUMMARY
    # ========================================================================
    
    summary = {
        "config": CONFIG,
        "champion_rv": champion_rv,
        "conditions": {
            "baseline": {
                "n": len(baseline_results),
                "mean_self_ref_markers": float(np.mean(baseline_markers)),
                "std_self_ref_markers": float(np.std(baseline_markers)),
                "mean_rv_input": float(np.mean(rv_baseline)) if rv_baseline else None,
                "eos_rate": sum(r['eos_reached'] for r in baseline_results) / len(baseline_results)
            },
            "champion_patched": {
                "n": len(patched_results),
                "mean_self_ref_markers": float(np.mean(patched_markers)),
                "std_self_ref_markers": float(np.std(patched_markers)),
                "mean_rv_output": float(np.mean(rv_patched)) if rv_patched else None,
                "eos_rate": sum(r['eos_reached'] for r in patched_results) / len(patched_results)
            },
            "random_kv_control": {
                "n": len(random_control_results),
                "mean_self_ref_markers": float(np.mean(random_markers)),
                "std_self_ref_markers": float(np.std(random_markers))
            },
            "wrong_layer_control": {
                "n": len(wrong_layer_results),
                "mean_self_ref_markers": float(np.mean(wrong_layer_markers)),
                "std_self_ref_markers": float(np.std(wrong_layer_markers))
            }
        },
        "statistics": {
            "patched_vs_baseline": {
                "cohens_d": float(d_patch_vs_base),
                "ci_95": [float(ci_patch[0]), float(ci_patch[1])],
                "t_stat": float(t_patch),
                "p_value": float(p_patch)
            },
            "random_vs_baseline": {
                "cohens_d": float(d_random_vs_base),
                "ci_95": [float(ci_random[0]), float(ci_random[1])],
                "t_stat": float(t_random),
                "p_value": float(p_random)
            },
            "wrong_layer_vs_baseline": {
                "cohens_d": float(d_wrong_vs_base),
                "ci_95": [float(ci_wrong[0]), float(ci_wrong[1])],
                "t_stat": float(t_wrong),
                "p_value": float(p_wrong)
            },
            "rv_patched_vs_baseline": {
                "cohens_d": float(d_rv) if not np.isnan(d_rv) else None,
                "ci_95": [float(ci_rv[0]), float(ci_rv[1])] if not np.isnan(d_rv) else None
            }
        },
        "interpretation": {
            "patched_effect_significant": p_patch < 0.001,
            "random_control_significant": p_random < 0.05,
            "wrong_layer_control_significant": p_wrong < 0.05,
            "effect_is_content_specific": (d_patch_vs_base > 1.0 and 
                                            abs(d_random_vs_base) < 0.5),
            "effect_is_layer_specific": (d_patch_vs_base > d_wrong_vs_base)
        }
    }
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print(f"\nSAMPLE SIZES:")
    print(f"  Baseline: n={summary['conditions']['baseline']['n']}")
    print(f"  Champion patched: n={summary['conditions']['champion_patched']['n']}")
    print(f"  Random KV control: n={summary['conditions']['random_kv_control']['n']}")
    print(f"  Wrong-layer control: n={summary['conditions']['wrong_layer_control']['n']}")
    
    print(f"\nSELF-REFERENCE MARKERS (mean ± std):")
    print(f"  Baseline:        {summary['conditions']['baseline']['mean_self_ref_markers']:.2f} ± {summary['conditions']['baseline']['std_self_ref_markers']:.2f}")
    print(f"  Champion patched: {summary['conditions']['champion_patched']['mean_self_ref_markers']:.2f} ± {summary['conditions']['champion_patched']['std_self_ref_markers']:.2f}")
    print(f"  Random control:   {summary['conditions']['random_kv_control']['mean_self_ref_markers']:.2f} ± {summary['conditions']['random_kv_control']['std_self_ref_markers']:.2f}")
    print(f"  Wrong-layer:      {summary['conditions']['wrong_layer_control']['mean_self_ref_markers']:.2f} ± {summary['conditions']['wrong_layer_control']['std_self_ref_markers']:.2f}")
    
    print(f"\nEFFECT SIZES (Cohen's d with 95% CI):")
    print(f"  Patched vs Baseline: d={d_patch_vs_base:.3f} [{ci_patch[0]:.3f}, {ci_patch[1]:.3f}], p={p_patch:.2e}")
    print(f"  Random vs Baseline:  d={d_random_vs_base:.3f} [{ci_random[0]:.3f}, {ci_random[1]:.3f}], p={p_random:.2e}")
    print(f"  Wrong-layer vs Base: d={d_wrong_vs_base:.3f} [{ci_wrong[0]:.3f}, {ci_wrong[1]:.3f}], p={p_wrong:.2e}")
    
    print(f"\nR_V ANALYSIS:")
    print(f"  Baseline input mean R_V: {summary['conditions']['baseline']['mean_rv_input']:.4f}")
    print(f"  Patched output mean R_V: {summary['conditions']['champion_patched']['mean_rv_output']:.4f}")
    if not np.isnan(d_rv):
        print(f"  R_V Cohen's d: {d_rv:.3f} [{ci_rv[0]:.3f}, {ci_rv[1]:.3f}]")
    
    print(f"\nINTERPRETATION:")
    print(f"  Effect significant (p<0.001): {summary['interpretation']['patched_effect_significant']}")
    print(f"  Content-specific (not random KV): {summary['interpretation']['effect_is_content_specific']}")
    print(f"  Layer-specific (not early layers): {summary['interpretation']['effect_is_layer_specific']}")
    
    # Save results
    results_dir = "results/gemma_full_validation"
    import os
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary
    with open(f"{results_dir}/summary_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save per-sample results
    all_results = {
        'baseline': baseline_results,
        'patched': patched_results,
        'random_control': random_control_results,
        'wrong_layer_control': wrong_layer_results
    }
    with open(f"{results_dir}/per_sample_{timestamp}.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {results_dir}/")
    
    # Final verdict
    print("\n" + "="*70)
    if (summary['interpretation']['patched_effect_significant'] and
        summary['interpretation']['effect_is_content_specific'] and
        summary['interpretation']['effect_is_layer_specific']):
        print("✓ BEHAVIORAL TRANSFER VALIDATED")
        print("  - Effect is statistically significant (p<0.001)")
        print("  - Effect is content-specific (not random KV disruption)")
        print("  - Effect is layer-specific (requires late layers)")
    else:
        print("VALIDATION INCOMPLETE - Some criteria not met")
        print(f"  Significant: {summary['interpretation']['patched_effect_significant']}")
        print(f"  Content-specific: {summary['interpretation']['effect_is_content_specific']}")
        print(f"  Layer-specific: {summary['interpretation']['effect_is_layer_specific']}")
    print("="*70)
    
    return summary

if __name__ == "__main__":
    run_full_validation()
