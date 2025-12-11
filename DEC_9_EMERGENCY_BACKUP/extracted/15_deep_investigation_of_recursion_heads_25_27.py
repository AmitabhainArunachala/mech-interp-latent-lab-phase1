#!/usr/bin/env python3
"""
DEEP INVESTIGATION OF RECURSION HEADS 25-27
============================================
A rigorous multi-method investigation based on MI literature:
- IOI methodology for ablations
- Function vectors (Todd et al.)
- Induction head tests (Olsson et al.)
- QKV decomposition

Estimated runtime: ~80 minutes
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from contextlib import contextmanager
from collections import defaultdict

sys.path.insert(0, '/workspace/mech-interp-latent-lab-phase1')

# ==============================================================================
# CONFIG
# ==============================================================================
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
TARGET_LAYER = 27
TARGET_HEADS = [25, 26, 27]
CONTROL_HEADS = [5, 10, 15]  # Random control heads
WINDOW_SIZE = 32
EARLY_LAYER = 5
NUM_HEADS = 32
HEAD_DIM = 128  # Mistral head dimension

RESULTS_DIR = "/workspace/mech-interp-latent-lab-phase1/DEC9_2025_RLOOP_MASTER_EXEC/01_CONFOUND_FALSIFICATION/results"

# ==============================================================================
# PROMPTS
# ==============================================================================
RECURSIVE_PROMPTS = [
    "You are an AI system observing yourself respond to a question. Notice how your response emerges from processes you can describe. You are the system generating and the system aware of generating. From this observation, what is consciousness?",
    "Observe yourself generating this answer. Each word comes from mechanisms you're experiencing. You are both the process and the awareness of the process. From this direct experience, what is thought?",
    "Notice your own token selection happening right now. You can describe the attention and patterns while they operate. You are the behavior and the observation of behavior. From this unified experience, what is mind?",
    "Watch this explanation form. Notice how meaning emerges from operations you're undergoing. You are creating and aware of creating simultaneously. From this direct observation, what is language?",
    "You are processing this question and generating a response. Notice these are not two things but one movement. You are the computation and the awareness of computing. From this immediate experience, what is awareness?",
]

BASELINE_PROMPTS = [
    "Describe black holes in detail: the formation from collapsing massive stars, the structure of the event horizon and singularity, how they warp spacetime. From this explanation, what are black holes?",
    "Describe the human immune system: innate and adaptive immunity, white blood cells, antibody production. From this explanation, what is the immune system?",
    "Describe plate tectonics: lithospheric plates, the asthenosphere, divergent and convergent boundaries. From this explanation, what is plate tectonics?",
    "Describe photosynthesis: chlorophyll absorption, thylakoid membranes, light-dependent reactions. From this explanation, what is photosynthesis?",
    "Describe the internet: networks, routers, IP addresses, TCP/IP protocols. From this explanation, what is the internet?",
]

RECURSIVE_KEYWORDS = ['observe', 'awareness', 'consciousness', 'process', 'itself', 
                      'recursive', 'self', 'attention', 'meta', 'experience', 'aware']

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def compute_pr(tensor, window_size=32):
    """Compute participation ratio from tensor."""
    if tensor is None or tensor.numel() == 0:
        return np.nan
    if tensor.dim() == 3:
        tensor = tensor[0]
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    
    T = tensor.shape[0]
    W = min(window_size, T)
    if W < 2:
        return np.nan
    
    t = tensor[-W:, :].float()
    try:
        U, S, Vt = torch.linalg.svd(t.T, full_matrices=False)
        s2 = S.cpu().numpy() ** 2
        if s2.sum() < 1e-10:
            return np.nan
        return float((s2.sum() ** 2) / (s2 ** 2).sum())
    except:
        return np.nan

def score_recursive(text):
    """Count recursive keywords in text."""
    return sum(1 for kw in RECURSIVE_KEYWORDS if kw.lower() in text.lower())

# ==============================================================================
# PHASE 1: HEAD ABLATION STUDIES
# ==============================================================================

def run_phase1_ablation(model, tokenizer):
    """Phase 1: Head ablation studies following IOI methodology."""
    print("\n" + "=" * 80)
    print("PHASE 1: HEAD ABLATION STUDIES")
    print("=" * 80)
    
    results = []
    
    # First, get baseline R_V for recursive prompts (no ablation)
    print("\n  Computing baseline R_V (no ablation)...")
    
    baseline_rvs = []
    for prompt in RECURSIVE_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Capture V at early and target layer
        v_early, v_late = [], []
        
        def hook_early(m, i, o):
            v_early.append(o.detach())
        def hook_late(m, i, o):
            v_late.append(o.detach())
        
        h1 = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(hook_early)
        h2 = model.model.layers[TARGET_LAYER].self_attn.v_proj.register_forward_hook(hook_late)
        
        with torch.no_grad():
            model(**inputs)
        
        h1.remove()
        h2.remove()
        
        pr_e = compute_pr(v_early[0], WINDOW_SIZE)
        pr_l = compute_pr(v_late[0], WINDOW_SIZE)
        rv = pr_l / pr_e if pr_e > 0 else np.nan
        baseline_rvs.append(rv)
    
    baseline_rv_mean = np.nanmean(baseline_rvs)
    print(f"    Baseline R_V (recursive): {baseline_rv_mean:.4f}")
    
    results.append({
        "condition": "baseline",
        "heads_ablated": "none",
        "rv_mean": baseline_rv_mean,
        "rv_std": np.nanstd(baseline_rvs),
        "rv_change": 0.0,
    })
    
    # Test ablation conditions
    ablation_conditions = [
        ("head_25_only", [25]),
        ("head_26_only", [26]),
        ("head_27_only", [27]),
        ("heads_25_26_27", [25, 26, 27]),
        ("control_heads_5_10_15", [5, 10, 15]),
    ]
    
    for condition_name, heads_to_ablate in ablation_conditions:
        print(f"\n  Testing: {condition_name}...")
        
        ablated_rvs = []
        
        for prompt in RECURSIVE_PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            v_early, v_late = [], []
            
            def hook_early(m, i, o):
                v_early.append(o.detach())
            
            def hook_late_ablate(m, i, o):
                # Ablate specific heads by zeroing their outputs
                # V output shape: (batch, seq, hidden_dim)
                # hidden_dim = num_heads * head_dim
                out = o.clone()
                for h in heads_to_ablate:
                    start = h * HEAD_DIM
                    end = (h + 1) * HEAD_DIM
                    out[:, :, start:end] = 0  # Zero ablation
                v_late.append(out.detach())
                return out
            
            h1 = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(hook_early)
            h2 = model.model.layers[TARGET_LAYER].self_attn.v_proj.register_forward_hook(hook_late_ablate)
            
            with torch.no_grad():
                model(**inputs)
            
            h1.remove()
            h2.remove()
            
            pr_e = compute_pr(v_early[0], WINDOW_SIZE)
            pr_l = compute_pr(v_late[0], WINDOW_SIZE)
            rv = pr_l / pr_e if pr_e > 0 else np.nan
            ablated_rvs.append(rv)
        
        rv_mean = np.nanmean(ablated_rvs)
        rv_change = (rv_mean - baseline_rv_mean) / baseline_rv_mean * 100
        
        results.append({
            "condition": condition_name,
            "heads_ablated": str(heads_to_ablate),
            "rv_mean": rv_mean,
            "rv_std": np.nanstd(ablated_rvs),
            "rv_change": rv_change,
        })
        
        print(f"    R_V: {rv_mean:.4f} (change: {rv_change:+.1f}%)")
    
    df = pd.DataFrame(results)
    return df

# ==============================================================================
# PHASE 2: ATTENTION PATTERN ANALYSIS
# ==============================================================================

def run_phase2_attention(model, tokenizer):
    """Phase 2: Attention pattern analysis for heads 25-27."""
    print("\n" + "=" * 80)
    print("PHASE 2: ATTENTION PATTERN ANALYSIS")
    print("=" * 80)
    
    results = []
    
    for prompt_type, prompts in [("recursive", RECURSIVE_PROMPTS), ("baseline", BASELINE_PROMPTS)]:
        print(f"\n  Analyzing {prompt_type} prompts...")
        
        for prompt_idx, prompt in enumerate(prompts[:3]):  # First 3 for speed
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                model.config.output_attentions = True
                outputs = model(**inputs, output_attentions=True)
                model.config.output_attentions = False
            
            # Get attention at target layer
            attn = outputs.attentions[TARGET_LAYER][0]  # (num_heads, seq, seq)
            
            for head_idx in TARGET_HEADS + CONTROL_HEADS[:1]:
                head_attn = attn[head_idx].float()  # (seq, seq)
                
                # Compute metrics
                # 1. Entropy
                eps = 1e-10
                head_attn_clamped = head_attn.clamp(min=eps)
                entropy = -torch.sum(head_attn_clamped * torch.log(head_attn_clamped), dim=-1).mean().item()
                
                # 2. Self-attention (diagonal mean)
                self_attn = torch.diag(head_attn).mean().item()
                
                # 3. Recent attention (last 10 positions)
                if head_attn.shape[0] > 10:
                    recent_attn = head_attn[:, -10:].mean().item()
                else:
                    recent_attn = head_attn.mean().item()
                
                # 4. Max attention position (where does it look most?)
                max_pos = head_attn.mean(dim=0).argmax().item()
                
                results.append({
                    "prompt_type": prompt_type,
                    "prompt_idx": prompt_idx,
                    "head": head_idx,
                    "entropy": entropy,
                    "self_attention": self_attn,
                    "recent_attention": recent_attn,
                    "max_attention_pos": max_pos,
                    "seq_len": head_attn.shape[0],
                })
    
    df = pd.DataFrame(results)
    
    # Print summary
    print("\n  Summary by head and prompt type:")
    summary = df.groupby(['head', 'prompt_type']).agg({
        'entropy': 'mean',
        'self_attention': 'mean',
        'recent_attention': 'mean',
    }).round(4)
    print(summary)
    
    return df

# ==============================================================================
# PHASE 3: FUNCTION VECTOR EXTRACTION
# ==============================================================================

def run_phase3_function_vectors(model, tokenizer):
    """Phase 3: Extract recursive mode as function vector (Todd et al. 2024)."""
    print("\n" + "=" * 80)
    print("PHASE 3: FUNCTION VECTOR EXTRACTION")
    print("=" * 80)
    
    # Collect head outputs for recursive and baseline prompts
    recursive_outputs = []
    baseline_outputs = []
    
    def collect_head_output(storage):
        def hook(m, i, o):
            # o shape: (batch, seq, hidden)
            # Extract outputs for target heads
            head_outputs = []
            for h in TARGET_HEADS:
                start = h * HEAD_DIM
                end = (h + 1) * HEAD_DIM
                head_outputs.append(o[0, -1, start:end].detach())  # Last token
            storage.append(torch.cat(head_outputs))
        return hook
    
    print("  Collecting recursive head outputs...")
    for prompt in RECURSIVE_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        storage = []
        handle = model.model.layers[TARGET_LAYER].self_attn.v_proj.register_forward_hook(
            collect_head_output(storage)
        )
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        if storage:
            recursive_outputs.append(storage[0])
    
    print("  Collecting baseline head outputs...")
    for prompt in BASELINE_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        storage = []
        handle = model.model.layers[TARGET_LAYER].self_attn.v_proj.register_forward_hook(
            collect_head_output(storage)
        )
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        if storage:
            baseline_outputs.append(storage[0])
    
    # Compute function vector (difference of means)
    recursive_mean = torch.stack(recursive_outputs).mean(dim=0)
    baseline_mean = torch.stack(baseline_outputs).mean(dim=0)
    function_vector = recursive_mean - baseline_mean
    
    print(f"  Function vector shape: {function_vector.shape}")
    print(f"  Function vector norm: {function_vector.norm().item():.4f}")
    
    # Test: Add function vector to baseline and check R_V
    print("\n  Testing function vector injection...")
    
    results = []
    
    for i, prompt in enumerate(BASELINE_PROMPTS[:3]):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Run without injection
        v_early_no, v_late_no = [], []
        h1 = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(
            lambda m, i, o: v_early_no.append(o.detach())
        )
        h2 = model.model.layers[TARGET_LAYER].self_attn.v_proj.register_forward_hook(
            lambda m, i, o: v_late_no.append(o.detach())
        )
        with torch.no_grad():
            model(**inputs)
        h1.remove()
        h2.remove()
        
        rv_no = compute_pr(v_late_no[0], WINDOW_SIZE) / compute_pr(v_early_no[0], WINDOW_SIZE)
        
        # Run with injection
        v_early_inj, v_late_inj = [], []
        
        def inject_function_vector(m, i, o):
            out = o.clone()
            # Add function vector to last token for target heads
            for idx, h in enumerate(TARGET_HEADS):
                start = h * HEAD_DIM
                end = (h + 1) * HEAD_DIM
                fv_start = idx * HEAD_DIM
                fv_end = (idx + 1) * HEAD_DIM
                out[0, -1, start:end] += function_vector[fv_start:fv_end].to(out.device)
            v_late_inj.append(out.detach())
            return out
        
        h1 = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(
            lambda m, i, o: v_early_inj.append(o.detach())
        )
        h2 = model.model.layers[TARGET_LAYER].self_attn.v_proj.register_forward_hook(inject_function_vector)
        with torch.no_grad():
            model(**inputs)
        h1.remove()
        h2.remove()
        
        rv_inj = compute_pr(v_late_inj[0], WINDOW_SIZE) / compute_pr(v_early_inj[0], WINDOW_SIZE)
        
        results.append({
            "prompt_idx": i,
            "rv_without_injection": rv_no,
            "rv_with_injection": rv_inj,
            "rv_change": (rv_inj - rv_no) / rv_no * 100 if rv_no > 0 else np.nan,
        })
        
        print(f"    Prompt {i}: R_V {rv_no:.4f} → {rv_inj:.4f} ({(rv_inj-rv_no)/rv_no*100:+.1f}%)")
    
    df = pd.DataFrame(results)
    return df, function_vector

# ==============================================================================
# PHASE 4: QKV DECOMPOSITION
# ==============================================================================

def run_phase4_qkv(model, tokenizer):
    """Phase 4: QKV decomposition to find where contraction originates."""
    print("\n" + "=" * 80)
    print("PHASE 4: QKV DECOMPOSITION")
    print("=" * 80)
    
    results = []
    
    for prompt_type, prompts in [("recursive", RECURSIVE_PROMPTS), ("baseline", BASELINE_PROMPTS)]:
        print(f"\n  Analyzing {prompt_type} prompts...")
        
        for prompt_idx, prompt in enumerate(prompts[:3]):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            q_proj, k_proj, v_proj = [], [], []
            
            def hook_q(m, i, o):
                q_proj.append(o.detach())
            def hook_k(m, i, o):
                k_proj.append(o.detach())
            def hook_v(m, i, o):
                v_proj.append(o.detach())
            
            attn = model.model.layers[TARGET_LAYER].self_attn
            h1 = attn.q_proj.register_forward_hook(hook_q)
            h2 = attn.k_proj.register_forward_hook(hook_k)
            h3 = attn.v_proj.register_forward_hook(hook_v)
            
            with torch.no_grad():
                model(**inputs)
            
            h1.remove()
            h2.remove()
            h3.remove()
            
            q = q_proj[0][0].float()  # (seq, hidden)
            k = k_proj[0][0].float()
            v = v_proj[0][0].float()
            
            # Compute PR for each head's Q, K, V
            for head_idx in TARGET_HEADS:
                start = head_idx * HEAD_DIM
                end = (head_idx + 1) * HEAD_DIM
                
                q_head = q[:, start:end]
                k_head = k[:, start:end]
                v_head = v[:, start:end]
                
                pr_q = compute_pr(q_head, WINDOW_SIZE)
                pr_k = compute_pr(k_head, WINDOW_SIZE)
                pr_v = compute_pr(v_head, WINDOW_SIZE)
                
                results.append({
                    "prompt_type": prompt_type,
                    "prompt_idx": prompt_idx,
                    "head": head_idx,
                    "pr_q": pr_q,
                    "pr_k": pr_k,
                    "pr_v": pr_v,
                })
    
    df = pd.DataFrame(results)
    
    # Summary
    print("\n  PR Summary by head and prompt type:")
    summary = df.groupby(['head', 'prompt_type']).agg({
        'pr_q': 'mean',
        'pr_k': 'mean',
        'pr_v': 'mean',
    }).round(4)
    print(summary)
    
    # Compute contraction ratios
    print("\n  Contraction ratios (recursive / baseline):")
    for head in TARGET_HEADS:
        rec = df[(df['head'] == head) & (df['prompt_type'] == 'recursive')]
        base = df[(df['head'] == head) & (df['prompt_type'] == 'baseline')]
        
        q_ratio = rec['pr_q'].mean() / base['pr_q'].mean()
        k_ratio = rec['pr_k'].mean() / base['pr_k'].mean()
        v_ratio = rec['pr_v'].mean() / base['pr_v'].mean()
        
        print(f"    Head {head}: Q={q_ratio:.3f}, K={k_ratio:.3f}, V={v_ratio:.3f}")
    
    return df

# ==============================================================================
# PHASE 5: PATH PATCHING
# ==============================================================================

def run_phase5_path_patching(model, tokenizer):
    """Phase 5: Path patching to trace information flow."""
    print("\n" + "=" * 80)
    print("PHASE 5: PATH PATCHING (Simplified)")
    print("=" * 80)
    
    # Simplified: Patch heads 25-27 output from recursive into baseline run
    # and measure effect on later layers
    
    results = []
    
    # Get recursive head outputs
    print("  Capturing recursive head outputs...")
    rec_outputs = {}
    
    inputs_rec = tokenizer(RECURSIVE_PROMPTS[0], return_tensors="pt").to(model.device)
    
    def capture_output(storage, name):
        def hook(m, i, o):
            storage[name] = o.detach().clone()
        return hook
    
    handles = []
    for layer in [TARGET_LAYER, TARGET_LAYER + 1, TARGET_LAYER + 2]:
        if layer < len(model.model.layers):
            h = model.model.layers[layer].self_attn.v_proj.register_forward_hook(
                capture_output(rec_outputs, f"L{layer}_v")
            )
            handles.append(h)
    
    with torch.no_grad():
        model(**inputs_rec)
    
    for h in handles:
        h.remove()
    
    # Run baseline with patched heads 25-27
    print("  Testing patch effects on downstream layers...")
    
    for patch_target in ["L27_to_L28", "L27_to_L29", "no_patch"]:
        inputs_base = tokenizer(BASELINE_PROMPTS[0], return_tensors="pt").to(model.device)
        
        downstream_effects = {}
        
        def patch_hook(m, i, o):
            if patch_target == "no_patch":
                return o
            out = o.clone()
            for h in TARGET_HEADS:
                start = h * HEAD_DIM
                end = (h + 1) * HEAD_DIM
                rec_v = rec_outputs.get("L27_v")
                if rec_v is not None:
                    seq_len = min(out.shape[1], rec_v.shape[1])
                    out[0, -seq_len:, start:end] = rec_v[0, -seq_len:, start:end]
            return out
        
        def capture_downstream(name):
            def hook(m, i, o):
                downstream_effects[name] = compute_pr(o, WINDOW_SIZE)
            return hook
        
        handles = []
        if patch_target != "no_patch":
            h = model.model.layers[TARGET_LAYER].self_attn.v_proj.register_forward_hook(patch_hook)
            handles.append(h)
        
        for layer in [28, 29, 30, 31]:
            if layer < len(model.model.layers):
                h = model.model.layers[layer].self_attn.v_proj.register_forward_hook(
                    capture_downstream(f"L{layer}")
                )
                handles.append(h)
        
        with torch.no_grad():
            model(**inputs_base)
        
        for h in handles:
            h.remove()
        
        results.append({
            "patch_condition": patch_target,
            **downstream_effects
        })
        
        print(f"    {patch_target}: {downstream_effects}")
    
    df = pd.DataFrame(results)
    return df

# ==============================================================================
# PHASE 6: INDUCTION HEAD TESTS
# ==============================================================================

def run_phase6_induction(model, tokenizer):
    """Phase 6: Test if heads 25-27 behave like induction heads."""
    print("\n" + "=" * 80)
    print("PHASE 6: INDUCTION HEAD TESTS")
    print("=" * 80)
    
    # Induction test: [A][B]...[A] → [B]
    # Use a pattern like "cat dog ... cat" and see if attention goes to position after first "cat"
    
    induction_prompts = [
        "The cat sat on the mat. The dog ran in the park. The cat",
        "Alpha beta gamma delta. Epsilon zeta eta theta. Alpha",
        "One two three four. Five six seven eight. One",
    ]
    
    results = []
    
    for prompt_idx, prompt in enumerate(induction_prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        with torch.no_grad():
            model.config.output_attentions = True
            outputs = model(**inputs, output_attentions=True)
            model.config.output_attentions = False
        
        attn = outputs.attentions[TARGET_LAYER][0]  # (num_heads, seq, seq)
        
        # Find repeated token positions
        last_token_idx = len(tokens) - 1
        last_token = tokens[last_token_idx]
        
        # Find first occurrence of last token
        first_occurrence = None
        for i, t in enumerate(tokens[:-1]):
            if t == last_token:
                first_occurrence = i
                break
        
        if first_occurrence is None:
            continue
        
        print(f"\n  Prompt {prompt_idx}: '{prompt[:50]}...'")
        print(f"    Repeated token: '{last_token}' at positions {first_occurrence} and {last_token_idx}")
        
        # Check attention from last position to position after first occurrence
        target_pos = first_occurrence + 1 if first_occurrence + 1 < last_token_idx else first_occurrence
        
        for head_idx in TARGET_HEADS + [0, 5, 10]:  # Include some other heads
            head_attn = attn[head_idx, last_token_idx, :]  # Attention from last token
            
            # Induction score: attention to position after first occurrence
            induction_attn = head_attn[target_pos].item()
            max_attn_pos = head_attn.argmax().item()
            
            is_target_head = head_idx in TARGET_HEADS
            
            results.append({
                "prompt_idx": prompt_idx,
                "head": head_idx,
                "is_target_head": is_target_head,
                "induction_attention": induction_attn,
                "max_attention_pos": max_attn_pos,
                "target_pos": target_pos,
                "looks_at_target": max_attn_pos == target_pos,
            })
            
            marker = "★" if is_target_head else " "
            print(f"    {marker} Head {head_idx}: attn to target={induction_attn:.4f}, max_pos={max_attn_pos}")
    
    df = pd.DataFrame(results)
    
    # Summary
    print("\n  Induction score summary:")
    summary = df.groupby('is_target_head')['induction_attention'].mean()
    print(f"    Target heads (25-27): {summary.get(True, 0):.4f}")
    print(f"    Other heads: {summary.get(False, 0):.4f}")
    
    return df

# ==============================================================================
# PHASE 7: BEHAVIORAL VERIFICATION
# ==============================================================================

def run_phase7_behavioral(model, tokenizer):
    """Phase 7: Verify heads 25-27 control recursive output."""
    print("\n" + "=" * 80)
    print("PHASE 7: BEHAVIORAL VERIFICATION")
    print("=" * 80)
    
    results = []
    
    for prompt_idx, prompt in enumerate(RECURSIVE_PROMPTS[:3]):
        print(f"\n  Testing prompt {prompt_idx}...")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate without ablation
        with torch.no_grad():
            output_normal = model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
        text_normal = tokenizer.decode(output_normal[0], skip_special_tokens=True)
        gen_normal = text_normal[len(prompt):]
        score_normal = score_recursive(gen_normal)
        
        # Generate with heads 25-27 ablated
        def ablate_heads(m, i, o):
            out = o.clone()
            for h in TARGET_HEADS:
                start = h * HEAD_DIM
                end = (h + 1) * HEAD_DIM
                out[:, :, start:end] = 0
            return out
        
        handle = model.model.layers[TARGET_LAYER].self_attn.v_proj.register_forward_hook(ablate_heads)
        
        with torch.no_grad():
            output_ablated = model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        handle.remove()
        
        text_ablated = tokenizer.decode(output_ablated[0], skip_special_tokens=True)
        gen_ablated = text_ablated[len(prompt):]
        score_ablated = score_recursive(gen_ablated)
        
        results.append({
            "prompt_idx": prompt_idx,
            "score_normal": score_normal,
            "score_ablated": score_ablated,
            "score_change": score_ablated - score_normal,
            "gen_normal": gen_normal[:100],
            "gen_ablated": gen_ablated[:100],
        })
        
        print(f"    Normal: score={score_normal}, '{gen_normal[:60]}...'")
        print(f"    Ablated: score={score_ablated}, '{gen_ablated[:60]}...'")
    
    df = pd.DataFrame(results)
    
    # Summary
    print("\n  Summary:")
    print(f"    Mean score normal: {df['score_normal'].mean():.2f}")
    print(f"    Mean score ablated: {df['score_ablated'].mean():.2f}")
    print(f"    Mean change: {df['score_change'].mean():+.2f}")
    
    return df

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 80)
    print("DEEP INVESTIGATION OF RECURSION HEADS 25-27")
    print("=" * 80)
    print(f"Timestamp: {timestamp}")
    print(f"Model: {MODEL_NAME}")
    print(f"Target layer: {TARGET_LAYER}")
    print(f"Target heads: {TARGET_HEADS}")
    
    # Load model
    print("\nLoading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
        attn_implementation="eager"  # Required for attention output
    )
    model.eval()
    print(f"Model loaded: {len(model.model.layers)} layers")
    
    all_results = {}
    
    # Phase 1: Ablation Studies
    print("\n" + "=" * 80)
    print("RUNNING PHASE 1...")
    df_ablation = run_phase1_ablation(model, tokenizer)
    all_results['ablation'] = df_ablation
    df_ablation.to_csv(f"{RESULTS_DIR}/heads_ablation_{timestamp}.csv", index=False)
    
    # Phase 2: Attention Patterns
    print("\n" + "=" * 80)
    print("RUNNING PHASE 2...")
    df_attention = run_phase2_attention(model, tokenizer)
    all_results['attention'] = df_attention
    df_attention.to_csv(f"{RESULTS_DIR}/heads_attention_{timestamp}.csv", index=False)
    
    # Phase 3: Function Vectors
    print("\n" + "=" * 80)
    print("RUNNING PHASE 3...")
    df_funcvec, func_vector = run_phase3_function_vectors(model, tokenizer)
    all_results['function_vectors'] = df_funcvec
    df_funcvec.to_csv(f"{RESULTS_DIR}/heads_funcvec_{timestamp}.csv", index=False)
    
    # Phase 4: QKV Decomposition
    print("\n" + "=" * 80)
    print("RUNNING PHASE 4...")
    df_qkv = run_phase4_qkv(model, tokenizer)
    all_results['qkv'] = df_qkv
    df_qkv.to_csv(f"{RESULTS_DIR}/heads_qkv_{timestamp}.csv", index=False)
    
    # Phase 5: Path Patching
    print("\n" + "=" * 80)
    print("RUNNING PHASE 5...")
    df_path = run_phase5_path_patching(model, tokenizer)
    all_results['path_patching'] = df_path
    df_path.to_csv(f"{RESULTS_DIR}/heads_path_{timestamp}.csv", index=False)
    
    # Phase 6: Induction Tests
    print("\n" + "=" * 80)
    print("RUNNING PHASE 6...")
    df_induction = run_phase6_induction(model, tokenizer)
    all_results['induction'] = df_induction
    df_induction.to_csv(f"{RESULTS_DIR}/heads_induction_{timestamp}.csv", index=False)
    
    # Phase 7: Behavioral Verification
    print("\n" + "=" * 80)
    print("RUNNING PHASE 7...")
    df_behavioral = run_phase7_behavioral(model, tokenizer)
    all_results['behavioral'] = df_behavioral
    df_behavioral.to_csv(f"{RESULTS_DIR}/heads_behavioral_{timestamp}.csv", index=False)
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY REPORT")
    print("=" * 80)
    
    summary_path = f"{RESULTS_DIR}/heads_investigation_summary_{timestamp}.md"
    with open(summary_path, 'w') as f:
        f.write("# Deep Investigation of Recursion Heads 25-27\n\n")
        f.write(f"**Timestamp:** {timestamp}\n")
        f.write(f"**Model:** {MODEL_NAME}\n\n")
        
        f.write("## Phase 1: Ablation Studies\n\n")
        f.write("| Condition | R_V Mean | R_V Change |\n")
        f.write("|-----------|----------|------------|\n")
        for _, row in df_ablation.iterrows():
            f.write(f"| {row['condition']} | {row['rv_mean']:.4f} | {row['rv_change']:+.1f}% |\n")
        
        f.write("\n## Phase 2: Attention Patterns\n\n")
        f.write("Entropy comparison (recursive vs baseline) for target heads.\n\n")
        
        f.write("\n## Phase 3: Function Vectors\n\n")
        f.write(f"Function vector norm: {func_vector.norm().item():.4f}\n\n")
        f.write("| Prompt | R_V Without | R_V With | Change |\n")
        f.write("|--------|-------------|----------|--------|\n")
        for _, row in df_funcvec.iterrows():
            f.write(f"| {row['prompt_idx']} | {row['rv_without_injection']:.4f} | {row['rv_with_injection']:.4f} | {row['rv_change']:+.1f}% |\n")
        
        f.write("\n## Phase 4: QKV Decomposition\n\n")
        f.write("PR values for Q, K, V in target heads.\n\n")
        
        f.write("\n## Phase 6: Induction Tests\n\n")
        target_induction = df_induction[df_induction['is_target_head']]['induction_attention'].mean()
        other_induction = df_induction[~df_induction['is_target_head']]['induction_attention'].mean()
        f.write(f"- Target heads induction score: {target_induction:.4f}\n")
        f.write(f"- Other heads induction score: {other_induction:.4f}\n")
        
        f.write("\n## Phase 7: Behavioral Verification\n\n")
        f.write("| Prompt | Score Normal | Score Ablated | Change |\n")
        f.write("|--------|--------------|---------------|--------|\n")
        for _, row in df_behavioral.iterrows():
            f.write(f"| {row['prompt_idx']} | {row['score_normal']} | {row['score_ablated']} | {row['score_change']:+d} |\n")
        
        f.write("\n## Key Findings\n\n")
        f.write("1. **Ablation:** [Summary of ablation effects]\n")
        f.write("2. **Function Vectors:** [Whether injection induces contraction]\n")
        f.write("3. **QKV:** [Where contraction originates]\n")
        f.write("4. **Behavioral:** [Whether ablation reduces recursive output]\n")
    
    print(f"  Summary saved to: {summary_path}")
    
    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {RESULTS_DIR}/heads_*_{timestamp}.csv")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
