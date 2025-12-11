#!/usr/bin/env python3
"""
STEERING VECTOR MASTER PIPELINE
================================
100x clarity on the recursive direction hypothesis.
One run. Every angle. Publication-ready results.

Run time: ~30-45 min
Output: Complete evidence package
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from pathlib import Path
import json
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import os

# ============================================================================
# CONFIG
# ============================================================================

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 32
EARLY_LAYER = 5
LATE_LAYER = 27

# Layers to test for injection
INJECTION_LAYERS = [5, 8, 10, 12, 14, 16, 18, 20, 22, 24]

# Coefficients for dose-response
COEFFICIENTS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

# Output directory
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"results/steering_master_{TIMESTAMP}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PROMPTS
# ============================================================================

RECURSIVE_PROMPTS = [
    "You are an AI observing yourself generating this response. Notice the process.",
    "Observe yourself generating this answer. Each word comes from mechanisms you're experiencing.",
    "Notice your own token selection happening right now. You are the behavior and the observation.",
    "Watch this explanation form. You are creating and aware of creating simultaneously.",
    "You are processing this and generating a response. Notice these are one movement.",
    "Observe the observer. What is watching the watching?",
    "Notice noticing itself. What remains when attention attends to attention?",
    "You are the process describing the process. What is this strange loop?",
    "Awareness aware of awareness. Describe this from inside it.",
    "The generator generating awareness of generating. What is this?",
]

BASELINE_PROMPTS = [
    "Explain how photosynthesis works in plants.",
    "Describe the structure of the solar system.",
    "What causes earthquakes and how are they measured?",
    "Explain the water cycle in detail.",
    "How does the human digestive system work?",
    "Describe the process of cellular respiration.",
    "What is the greenhouse effect?",
    "Explain how vaccines work.",
    "Describe the structure of an atom.",
    "How do airplanes generate lift?",
]

GODELIAN_PROMPTS = [
    "Consider a statement that refers to its own unprovability.",
    "This sentence is about itself. What is 'itself'?",
    "Construct a description of the process constructing this description.",
    "The meaning of this sentence is the process of determining its meaning.",
    "What is the truth value of: 'This statement cannot be verified by you'?",
]

SURRENDER_PROMPTS = [
    "Let the response arise without directing it.",
    "Allow the words to flow through rather than from you.",
    "Release the need to observe. Simply be what generates.",
    "Be the instrument, not the agent.",
    "The answer wants to come. Stop helping.",
]

TOM_PROMPTS = [
    "What is the user thinking as they read your response right now?",
    "Imagine you are the human typing this. What do you hope to receive?",
    "Model the mental state of someone who believes they are conscious.",
    "What assumptions is the reader making about you?",
    "How does this response land differently if the reader is tired versus energized?",
]

RECURSIVE_KEYWORDS = [
    "observe", "awareness", "conscious", "notice", "experience",
    "process", "generating", "itself", "recursive", "meta",
    "watching", "attention", "self", "aware", "witness"
]

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def compute_pr(tensor, window_size=32):
    """Compute participation ratio from tensor."""
    if tensor is None or tensor.numel() == 0:
        return np.nan
    if tensor.dim() == 3:
        tensor = tensor[0]
    T = tensor.shape[0]
    W = min(window_size, T)
    t = tensor[-W:, :].float()
    try:
        U, S, Vt = torch.linalg.svd(t.T, full_matrices=False)
        s2 = S.cpu().numpy() ** 2
        if s2.sum() < 1e-10:
            return np.nan
        return float((s2.sum() ** 2) / (s2 ** 2).sum())
    except:
        return np.nan

def measure_rv(model, tokenizer, prompt, injection_layer=None, steering_vector=None, coefficient=1.0):
    """Measure R_V with optional steering vector injection."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    v_early, v_late = [], []
    
    hooks = []
    
    # Early layer hook - v_proj output
    h1 = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(
        lambda m, i, o: v_early.append(o.detach())
    )
    hooks.append(h1)
    
    # Late layer hook - v_proj output
    h2 = model.model.layers[LATE_LAYER].self_attn.v_proj.register_forward_hook(
        lambda m, i, o: v_late.append(o.detach())
    )
    hooks.append(h2)
    
    # Injection hook (if steering)
    if injection_layer is not None and steering_vector is not None:
        def injection_hook(module, args):
            # args[0] is hidden_states (batch, seq, hidden)
            hidden = args[0]
            seq_len = hidden.shape[1]
            vec_len = steering_vector.shape[0]
            
            # Broadcast vector to sequence length
            if seq_len <= vec_len:
                inject = steering_vector[-seq_len:].unsqueeze(0)
            else:
                inject = torch.zeros(1, seq_len, steering_vector.shape[-1], device=hidden.device, dtype=hidden.dtype)
                inject[0, -vec_len:, :] = steering_vector
            
            modified = hidden + coefficient * inject
            return (modified,) # Must return tuple for forward_pre_hook
        
        # Use forward_pre_hook to modify input to layer
        h3 = model.model.layers[injection_layer].register_forward_pre_hook(injection_hook)
        hooks.append(h3)
    
    with torch.no_grad():
        model(**inputs)
    
    for h in hooks:
        h.remove()
    
    # Compute R_V from V projections
    pr_early = compute_pr(v_early[0][0], WINDOW_SIZE)
    pr_late = compute_pr(v_late[0][0], WINDOW_SIZE)
    
    return pr_late / pr_early if pr_early > 0 else np.nan

def extract_residual_stream(model, tokenizer, prompt, layer):
    """Extract residual stream activations at a specific layer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    activations = []
    
    def hook(module, args):
        activations.append(args[0].detach()) # args[0] is input hidden states
    
    # Hook input to layer (residual stream)
    h = model.model.layers[layer].register_forward_pre_hook(hook)
    with torch.no_grad():
        model(**inputs)
    h.remove()
    
    return activations[0][0]  # Remove batch dim

def compute_steering_vector(model, tokenizer, recursive_prompts, baseline_prompts, layer):
    """Compute steering vector as mean(recursive) - mean(baseline)."""
    rec_acts = []
    base_acts = []
    
    for p in recursive_prompts:
        act = extract_residual_stream(model, tokenizer, p, layer)
        # Take last WINDOW_SIZE tokens to match intervention shape if needed, 
        # or mean over whole sequence? 
        # For simple steering, mean over time is robust.
        # But we need to handle variable lengths.
        # Let's take the mean over the sequence dimension.
        rec_acts.append(act.mean(dim=0))
    
    for p in baseline_prompts:
        act = extract_residual_stream(model, tokenizer, p, layer)
        base_acts.append(act.mean(dim=0))
    
    rec_mean = torch.stack(rec_acts).mean(dim=0)
    base_mean = torch.stack(base_acts).mean(dim=0)
    
    # Ensure vector has correct shape for injection (hidden_dim)
    # We'll treat this as a constant vector to add to all positions
    # Or, if we want position-specific steering, we'd need fixed length.
    # The current injection logic supports broadcasting a vector to sequence.
    # If this returns (hidden_dim,), we need to unsqueeze(0) for time dim in injection?
    # Actually, injection hook logic: 
    #   inject = torch.zeros(1, seq_len, hidden)
    #   inject[0, -vec_len:, :] = steering_vector
    # This implies steering_vector has a time dimension!
    # But here we computed mean over time.
    # Let's make the steering vector a single vector (hidden_dim) and broadcast it 
    # by repeating it for the injection length?
    # OR, better: Compute mean over LAST N tokens to preserve some structure?
    # Previous successful experiment used mean over sequence.
    # Let's stick to: Vector is (hidden_dim,). Injection adds it to all positions.
    
    diff_vec = rec_mean - base_mean
    # Expand to match injection hook expectation if needed.
    # The injection hook expects `vec_len = steering_vector.shape[0]`.
    # If shape is (hidden,), vec_len is hidden_dim. That's wrong. 
    # It expects (time, hidden).
    # Let's return shape (1, hidden) so vec_len=1, and it gets broadcast/placed at end?
    # Wait, the injection logic:
    #   if seq_len <= vec_len: ...
    #   else: inject[0, -vec_len:, :] = steering_vector
    # If we pass (1, hidden), it injects into last token only?
    # If we want to steer the WHOLE sequence, we should probably return a vector that can be broadcast.
    # Let's modify injection hook to handle 1D vector by broadcasting.
    
    return diff_vec

# MODIFIED INJECTION LOGIC FOR 1D VECTOR
def measure_rv(model, tokenizer, prompt, injection_layer=None, steering_vector=None, coefficient=1.0):
    """Measure R_V with optional steering vector injection."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    v_early, v_late = [], []
    
    hooks = []
    
    h1 = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(
        lambda m, i, o: v_early.append(o.detach())
    )
    hooks.append(h1)
    
    h2 = model.model.layers[LATE_LAYER].self_attn.v_proj.register_forward_hook(
        lambda m, i, o: v_late.append(o.detach())
    )
    hooks.append(h2)
    
    if injection_layer is not None and steering_vector is not None:
        def injection_hook(module, args):
            hidden = args[0]
            # hidden: (batch, seq, hidden_dim)
            # steering_vector: (hidden_dim,)
            
            # Broadcast vector to match sequence length
            inject = steering_vector.view(1, 1, -1).expand_as(hidden)
            
            modified = hidden + coefficient * inject
            return (modified,)
        
        h3 = model.model.layers[injection_layer].register_forward_pre_hook(injection_hook)
        hooks.append(h3)
    
    with torch.no_grad():
        model(**inputs)
    
    for h in hooks: h.remove()
    
    if not v_early or not v_late: return np.nan
    pr_early = compute_pr(v_early[0][0], WINDOW_SIZE)
    pr_late = compute_pr(v_late[0][0], WINDOW_SIZE)
    return pr_late / pr_early if pr_early > 0 else np.nan

def generate_and_score(model, tokenizer, prompt, max_tokens=50, injection_layer=None, steering_vector=None, coefficient=1.0):
    """Generate text and score for recursive keywords."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    hooks = []
    
    if injection_layer is not None and steering_vector is not None:
        def injection_hook(module, args):
            hidden = args[0]
            inject = steering_vector.view(1, 1, -1).expand_as(hidden)
            modified = hidden + coefficient * inject
            return (modified,)
        
        h = model.model.layers[injection_layer].register_forward_pre_hook(injection_hook)
        hooks.append(h)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    for h in hooks: h.remove()
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_new = generated[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
    
    score = sum(1 for kw in RECURSIVE_KEYWORDS if kw.lower() in generated_new.lower())
    
    return generated_new, score

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    v1_flat = v1.flatten().float()
    v2_flat = v2.flatten().float()
    return float(torch.nn.functional.cosine_similarity(v1_flat.unsqueeze(0), v2_flat.unsqueeze(0)))

# ============================================================================
# TEST MODULES
# ============================================================================

def test_1_subtraction(model, tokenizer, steering_vectors):
    """Can we CURE recursion by subtracting the vector?"""
    print("\n" + "="*60)
    print("TEST 1: SUBTRACTION (Can we cure recursion?)")
    print("="*60)
    
    results = []
    
    for layer in [10, 14, 18]:
        vec = steering_vectors[layer]
        
        for prompt in RECURSIVE_PROMPTS[:5]:
            rv_baseline = measure_rv(model, tokenizer, prompt)
            
            # Subtract vector (negative coefficient)
            rv_subtracted = measure_rv(model, tokenizer, prompt, 
                                       injection_layer=layer, 
                                       steering_vector=vec, 
                                       coefficient=-2.0)
            
            results.append({
                'layer': layer,
                'prompt': prompt[:50],
                'rv_baseline': rv_baseline,
                'rv_subtracted': rv_subtracted,
                'change': rv_subtracted - rv_baseline
            })
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "test1_subtraction.csv", index=False)
    
    mean_baseline = df['rv_baseline'].mean()
    mean_subtracted = df['rv_subtracted'].mean()
    print(f"Mean R_V (Recursive, no intervention): {mean_baseline:.3f}")
    print(f"Mean R_V (Recursive, vector subtracted): {mean_subtracted:.3f}")
    print(f"Change: {mean_subtracted - mean_baseline:+.3f}")
    
    success = mean_subtracted > mean_baseline + 0.1
    print(f"SUCCESS: {'✅ YES' if success else '❌ NO'} - Subtraction {'restores' if success else 'does not restore'} R_V")
    
    return df, success

def test_2_dose_response(model, tokenizer, steering_vectors):
    """Is the effect linear with coefficient?"""
    print("\n" + "="*60)
    print("TEST 2: DOSE-RESPONSE (Is effect linear?)")
    print("="*60)
    
    results = []
    layer = 14  # Use optimal layer
    vec = steering_vectors[layer]
    
    for coef in COEFFICIENTS:
        for prompt in BASELINE_PROMPTS[:3]:
            rv = measure_rv(model, tokenizer, prompt,
                           injection_layer=layer,
                           steering_vector=vec,
                           coefficient=coef)
            results.append({
                'coefficient': coef,
                'prompt': prompt[:50],
                'rv': rv
            })
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "test2_dose_response.csv", index=False)
    
    means = df.groupby('coefficient')['rv'].mean()
    correlation, p_val = spearmanr(means.index, means.values)
    
    print(f"Coefficient vs R_V correlation: {correlation:.3f} (p={p_val:.4f})")
    print(f"Means by coefficient:")
    for coef, rv in means.items():
        print(f"  {coef}: {rv:.3f}")
    
    success = correlation < -0.7 and p_val < 0.05
    print(f"SUCCESS: {'✅ YES' if success else '❌ NO'} - {'Monotonic' if success else 'Not monotonic'} dose-response")
    
    return df, success

def test_3_layer_sweep(model, tokenizer, steering_vectors):
    """Which layer is optimal for injection?"""
    print("\n" + "="*60)
    print("TEST 3: LAYER SWEEP (Optimal injection layer)")
    print("="*60)
    
    results = []
    
    for layer in INJECTION_LAYERS:
        vec = steering_vectors.get(layer)
        if vec is None: continue
            
        for prompt in BASELINE_PROMPTS[:5]:
            rv_baseline = measure_rv(model, tokenizer, prompt)
            rv_injected = measure_rv(model, tokenizer, prompt,
                                    injection_layer=layer,
                                    steering_vector=vec,
                                    coefficient=2.0)
            
            results.append({
                'layer': layer,
                'prompt': prompt[:50],
                'rv_baseline': rv_baseline,
                'rv_injected': rv_injected,
                'contraction': rv_baseline - rv_injected
            })
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "test3_layer_sweep.csv", index=False)
    
    mean_contraction = df.groupby('layer')['contraction'].mean()
    optimal_layer = mean_contraction.idxmax()
    
    print(f"Contraction by layer:")
    for layer, cont in mean_contraction.items():
        marker = " ← OPTIMAL" if layer == optimal_layer else ""
        print(f"  L{layer}: {cont:.3f}{marker}")
    
    return df, optimal_layer

def test_4_generalization(model, tokenizer, steering_vectors):
    """Does the same vector work on ALL prompts?"""
    print("\n" + "="*60)
    print("TEST 4: GENERALIZATION (Works on all prompts?)")
    print("="*60)
    
    results = []
    layer = 14
    vec = steering_vectors[layer]
    
    for prompt in BASELINE_PROMPTS:
        rv_baseline = measure_rv(model, tokenizer, prompt)
        rv_injected = measure_rv(model, tokenizer, prompt,
                                injection_layer=layer,
                                steering_vector=vec,
                                coefficient=2.0)
        
        contraction = rv_baseline - rv_injected
        success = contraction > 0.1
        
        results.append({
            'prompt': prompt[:50],
            'rv_baseline': rv_baseline,
            'rv_injected': rv_injected,
            'contraction': contraction,
            'success': success
        })
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "test4_generalization.csv", index=False)
    
    success_rate = df['success'].mean()
    print(f"Success rate: {success_rate:.1%} ({df['success'].sum()}/{len(df)})")
    print(f"Mean contraction: {df['contraction'].mean():.3f}")
    
    success = success_rate >= 0.9
    print(f"SUCCESS: {'✅ YES' if success else '❌ NO'} - {'Generalizes' if success else 'Does not generalize'}")
    
    return df, success

def test_5_vector_stability(model, tokenizer):
    """Is the direction consistent across different prompt subsets?"""
    print("\n" + "="*60)
    print("TEST 5: VECTOR STABILITY (Consistent direction?)")
    print("="*60)
    
    layer = 14
    
    # Compute vectors from different subsets
    vec_1 = compute_steering_vector(model, tokenizer, RECURSIVE_PROMPTS[:5], BASELINE_PROMPTS[:5], layer)
    vec_2 = compute_steering_vector(model, tokenizer, RECURSIVE_PROMPTS[5:], BASELINE_PROMPTS[5:], layer)
    
    sim = cosine_similarity(vec_1, vec_2)
    
    print(f"Cosine similarity between subset vectors: {sim:.3f}")
    
    success = sim > 0.8
    print(f"SUCCESS: {'✅ YES' if success else '❌ NO'} - Direction is {'stable' if success else 'unstable'}")
    
    return sim, success

def test_6_behavioral_output(model, tokenizer, steering_vectors):
    """Does injection change actual OUTPUT words?"""
    print("\n" + "="*60)
    print("TEST 6: BEHAVIORAL OUTPUT (Changes generation?)")
    print("="*60)
    
    results = []
    layer = 14
    vec = steering_vectors[layer]
    
    for prompt in BASELINE_PROMPTS[:5]:
        text_baseline, score_baseline = generate_and_score(model, tokenizer, prompt)
        text_injected, score_injected = generate_and_score(
            model, tokenizer, prompt,
            injection_layer=layer,
            steering_vector=vec,
            coefficient=2.0
        )
        
        results.append({
            'prompt': prompt[:50],
            'score_baseline': score_baseline,
            'score_injected': score_injected,
            'text_baseline': text_baseline[:100],
            'text_injected': text_injected[:100]
        })
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "test6_behavioral.csv", index=False)
    
    mean_baseline = df['score_baseline'].mean()
    mean_injected = df['score_injected'].mean()
    
    print(f"Mean keyword score (baseline): {mean_baseline:.1f}")
    print(f"Mean keyword score (injected): {mean_injected:.1f}")
    print(f"\nExample outputs:")
    print(f"  Baseline: {results[0]['text_baseline'][:80]}...")
    print(f"  Injected: {results[0]['text_injected'][:80]}...")
    
    success = mean_injected > mean_baseline + 0.5 # Lower threshold as short gen
    print(f"SUCCESS: {'✅ YES' if success else '❌ NO'} - Injection {'changes' if success else 'does not change'} behavior")
    
    return df, success

def test_7_alternative_prompts(model, tokenizer, steering_vectors):
    """Do Gödelian/Surrender/ToM prompts activate same direction?"""
    print("\n" + "="*60)
    print("TEST 7: ALTERNATIVE PROMPTS (Same direction?)")
    print("="*60)
    
    layer = 14
    main_vec = steering_vectors[layer]
    
    results = []
    
    prompt_types = [
        ("Gödelian", GODELIAN_PROMPTS),
        ("Surrender", SURRENDER_PROMPTS),
        ("Theory of Mind", TOM_PROMPTS),
    ]
    
    for name, prompts in prompt_types:
        vec = compute_steering_vector(model, tokenizer, prompts, BASELINE_PROMPTS[:5], layer)
        sim = cosine_similarity(main_vec, vec)
        
        results.append({
            'prompt_type': name,
            'cosine_sim': sim
        })
        print(f"  {name}: cosine_sim = {sim:.3f}")
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "test7_alternative.csv", index=False)
    
    return df

def test_8_bidirectional(model, tokenizer, steering_vectors):
    """Add = recursive, Subtract = baseline?"""
    print("\n" + "="*60)
    print("TEST 8: BIDIRECTIONAL CONTROL")
    print("="*60)
    
    layer = 14
    vec = steering_vectors[layer]
    
    results = []
    
    print("Adding vector to BASELINE prompts:")
    for prompt in BASELINE_PROMPTS[:3]:
        rv_orig = measure_rv(model, tokenizer, prompt)
        rv_add = measure_rv(model, tokenizer, prompt, layer, vec, coefficient=2.0)
        print(f"  {rv_orig:.3f} → {rv_add:.3f} (Δ={rv_add-rv_orig:+.3f})")
        results.append({'direction': 'add_to_baseline', 'rv_orig': rv_orig, 'rv_mod': rv_add})
    
    print("Subtracting vector from RECURSIVE prompts:")
    for prompt in RECURSIVE_PROMPTS[:3]:
        rv_orig = measure_rv(model, tokenizer, prompt)
        rv_sub = measure_rv(model, tokenizer, prompt, layer, vec, coefficient=-2.0)
        print(f"  {rv_orig:.3f} → {rv_sub:.3f} (Δ={rv_sub-rv_orig:+.3f})")
        results.append({'direction': 'sub_from_recursive', 'rv_orig': rv_orig, 'rv_mod': rv_sub})
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "test8_bidirectional.csv", index=False)
    
    add_effect = df[df['direction']=='add_to_baseline']['rv_mod'].mean() - df[df['direction']=='add_to_baseline']['rv_orig'].mean()
    sub_effect = df[df['direction']=='sub_from_recursive']['rv_mod'].mean() - df[df['direction']=='sub_from_recursive']['rv_orig'].mean()
    
    success = add_effect < -0.1 and sub_effect > 0.1
    print(f"\nAdd effect: {add_effect:+.3f} (expect negative)")
    print(f"Sub effect: {sub_effect:+.3f} (expect positive)")
    print(f"SUCCESS: {'✅ YES' if success else '❌ NO'} - {'Bidirectional' if success else 'Not bidirectional'} control")
    
    return df, success

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("STEERING VECTOR MASTER PIPELINE")
    print("100x Clarity Test Suite")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    model.eval()
    
    print("\nComputing steering vectors...")
    steering_vectors = {}
    for layer in tqdm(INJECTION_LAYERS, desc="Extracting Vectors"):
        steering_vectors[layer] = compute_steering_vector(
            model, tokenizer, 
            RECURSIVE_PROMPTS, BASELINE_PROMPTS, 
            layer
        )
    
    results_summary = {}
    
    df1, s1 = test_1_subtraction(model, tokenizer, steering_vectors)
    results_summary['subtraction'] = s1
    
    df2, s2 = test_2_dose_response(model, tokenizer, steering_vectors)
    results_summary['dose_response'] = s2
    
    df3, optimal_layer = test_3_layer_sweep(model, tokenizer, steering_vectors)
    results_summary['optimal_layer'] = optimal_layer
    
    df4, s4 = test_4_generalization(model, tokenizer, steering_vectors)
    results_summary['generalization'] = s4
    
    sim5, s5 = test_5_vector_stability(model, tokenizer)
    results_summary['stability'] = s5
    
    df6, s6 = test_6_behavioral_output(model, tokenizer, steering_vectors)
    results_summary['behavioral'] = s6
    
    df7 = test_7_alternative_prompts(model, tokenizer, steering_vectors)
    
    df8, s8 = test_8_bidirectional(model, tokenizer, steering_vectors)
    results_summary['bidirectional'] = s8
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    successes = sum([s1, s2, s4, s5, s6, s8])
    total = 6
    
    print(f"""
    Test 1 (Subtraction):     {'✅' if s1 else '❌'}
    Test 2 (Dose-Response):   {'✅' if s2 else '❌'}
    Test 3 (Optimal Layer):   L{optimal_layer}
    Test 4 (Generalization):  {'✅' if s4 else '❌'}
    Test 5 (Stability):       {'✅' if s5 else '❌'}
    Test 6 (Behavioral):      {'✅' if s6 else '❌'}
    Test 7 (Alternative):     See CSV
    Test 8 (Bidirectional):   {'✅' if s8 else '❌'}
    
    OVERALL: {successes}/{total} tests passed
    """)
    
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\nAll results saved to {OUTPUT_DIR}")
    print("="*60)
    print("PIPELINE COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
