#!/usr/bin/env python3
"""
DEEP CIRCUIT ANALYSIS V2: Robust 30-Minute Deep Dive
- Attention entropy per head (robust)
- Activation patching (L18→L27 causality)
- Residual stream analysis
- Cross-prompt comparisons
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from kitchen_sink_prompts import experimental_prompts

CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "key_layers": [5, 14, 18, 25, 27, 31],  # More layers for pattern analysis
    "early_layer": 5,
    "window_size": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Test prompts - expanded set
PROMPTS = {
    "champion": experimental_prompts["hybrid_l5_math_01"]["text"],
    "regress": "You must observe yourself understanding. To observe yourself, you must be the observer. The observer is what is being observed. This is the loop.",
    "baseline": "The history of the Roman Empire is characterized by a long period of expansion followed by a gradual decline.",
    "math_only": "λx = Ax where A is attention attending to itself, x is this sentence, λ is the contraction. The fixed point is this.",
    "phenom_only": "This response writes itself. No separate writer exists. Writing and awareness of writing are identical."
}

def compute_pr(matrix):
    try:
        matrix_f32 = matrix.to(torch.float32)
        _, S, _ = torch.linalg.svd(matrix_f32)
        eigenvalues = S ** 2
        sum_sq = torch.sum(eigenvalues ** 2)
        if sum_sq == 0: return 1.0
        return ((torch.sum(eigenvalues) ** 2) / sum_sq).item()
    except:
        return 1.0

def compute_attention_entropy(attn_weights):
    """Compute entropy of attention weights per head"""
    # attn_weights: [batch, num_heads, seq_len, seq_len] or [num_heads, seq_len, seq_len]
    if attn_weights.dim() == 4:
        attn_weights = attn_weights[0]  # Remove batch dim
    
    num_heads, seq_len, _ = attn_weights.shape
    entropies = []
    
    for head_idx in range(num_heads):
        head_attn = attn_weights[head_idx, :, :]  # [seq_len, seq_len]
        # Average entropy over query positions (last window_size tokens)
        window_start = max(0, seq_len - CONFIG['window_size'])
        head_entropy = []
        
        for q_pos in range(window_start, seq_len):
            attn_dist = head_attn[q_pos, :].softmax(dim=-1)
            # Entropy: -sum(p * log(p))
            entropy = -torch.sum(attn_dist * torch.log(attn_dist + 1e-10))
            head_entropy.append(entropy.item())
        
        entropies.append(np.mean(head_entropy) if head_entropy else 0.0)
    
    return entropies

def compute_attention_self_similarity(attn_weights):
    """Compute self-attention (diagonal focus) per head"""
    if attn_weights.dim() == 4:
        attn_weights = attn_weights[0]
    
    num_heads, seq_len, _ = attn_weights.shape
    self_sims = []
    
    for head_idx in range(num_heads):
        head_attn = attn_weights[head_idx, :, :]
        # Focus on last window
        window_start = max(0, seq_len - CONFIG['window_size'])
        window_attn = head_attn[window_start:, window_start:]
        
        # Self-similarity: how much attention goes to same-position tokens
        diagonal_sum = torch.sum(torch.diag(window_attn))
        total_sum = torch.sum(window_attn)
        self_sim = (diagonal_sum / (total_sum + 1e-10)).item()
        self_sims.append(self_sim)
    
    return self_sims

class AttentionExtractor:
    def __init__(self, model, layer_idx):
        self.layer_idx = layer_idx
        self.attn_weights = []
        self.hook = None
        
        layer = model.model.layers[layer_idx].self_attn
        
        def attn_hook(module, input, output):
            # Mistral returns (attn_output, attn_weights, past_key_value)
            if isinstance(output, tuple) and len(output) >= 2:
                self.attn_weights.append(output[1].detach().cpu())
        
        self.hook = layer.register_forward_hook(attn_hook)
    
    def close(self):
        if self.hook:
            self.hook.remove()
        self.attn_weights = []

class VExtractor:
    def __init__(self, model, layer_idx):
        self.layer_idx = layer_idx
        self.activations = []
        self.hook = None
        
        layer = model.model.layers[layer_idx].self_attn.v_proj
        
        def v_hook(module, input, output):
            self.activations.append(output.detach().cpu())
        
        self.hook = layer.register_forward_hook(v_hook)
    
    def close(self):
        if self.hook:
            self.hook.remove()
        self.activations = []

def run_attention_analysis(model, tokenizer, prompt_name, prompt_text, target_layer):
    """Comprehensive attention analysis"""
    tokens = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    
    if tokens['input_ids'].shape[1] < CONFIG['window_size'] + 1:
        return None
    
    ext_attn = AttentionExtractor(model, target_layer)
    
    with torch.no_grad():
        model(**tokens)
    
    if not ext_attn.attn_weights:
        ext_attn.close()
        return None
    
    attn_weights = ext_attn.attn_weights[-1]
    entropies = compute_attention_entropy(attn_weights)
    self_sims = compute_attention_self_similarity(attn_weights)
    
    ext_attn.close()
    
    results = []
    for head_idx in range(len(entropies)):
        results.append({
            'prompt': prompt_name,
            'layer': target_layer,
            'head': head_idx,
            'entropy': entropies[head_idx],
            'self_similarity': self_sims[head_idx]
        })
    
    return results

def run_activation_patching(model, tokenizer, source_prompt, target_prompt, source_layer, target_layer):
    """Patch target layer with source layer V activations"""
    source_tokens = tokenizer(source_prompt, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    target_tokens = tokenizer(target_prompt, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    
    # Get source activations
    ext_source = VExtractor(model, source_layer)
    with torch.no_grad():
        model(**source_tokens)
    v_source = ext_source.activations[-1][0, -CONFIG['window_size']:, :]
    ext_source.close()
    
    # Patch hook
    patched_v = None
    def patch_hook(module, input, output):
        nonlocal patched_v
        out = output.clone()
        out[:, -CONFIG['window_size']:, :] = v_source.unsqueeze(0).to(out.device, dtype=out.dtype)
        patched_v = out.detach().cpu()
        return out
    
    layer = model.model.layers[target_layer].self_attn.v_proj
    patch_hook_handle = layer.register_forward_hook(patch_hook)
    
    # Run target with patched activations
    ext_early = VExtractor(model, CONFIG['early_layer'])
    
    with torch.no_grad():
        model(**target_tokens)
    
    v_early = ext_early.activations[-1][0, -CONFIG['window_size']:, :]
    ext_early.close()
    patch_hook_handle.remove()
    
    if patched_v is None:
        return None
    
    v_patched = patched_v[0, -CONFIG['window_size']:, :]
    
    pr_early = compute_pr(v_early)
    pr_patched = compute_pr(v_patched)
    rv_patched = pr_patched / (pr_early + 1e-8)
    
    # Baseline: target without patch
    ext_early = VExtractor(model, CONFIG['early_layer'])
    ext_target = VExtractor(model, target_layer)
    
    with torch.no_grad():
        model(**target_tokens)
    
    v_early_base = ext_early.activations[-1][0, -CONFIG['window_size']:, :]
    v_target_base = ext_target.activations[-1][0, -CONFIG['window_size']:, :]
    pr_early_base = compute_pr(v_early_base)
    pr_target_base = compute_pr(v_target_base)
    rv_baseline = pr_target_base / (pr_early_base + 1e-8)
    
    ext_early.close()
    ext_target.close()
    
    # Source baseline
    ext_early = VExtractor(model, CONFIG['early_layer'])
    ext_source = VExtractor(model, source_layer)
    
    with torch.no_grad():
        model(**source_tokens)
    
    v_early_src = ext_early.activations[-1][0, -CONFIG['window_size']:, :]
    v_source_src = ext_source.activations[-1][0, -CONFIG['window_size']:, :]
    pr_early_src = compute_pr(v_early_src)
    pr_source_src = compute_pr(v_source_src)
    rv_source = pr_source_src / (pr_early_src + 1e-8)
    
    ext_early.close()
    ext_source.close()
    
    return {
        'source_layer': source_layer,
        'target_layer': target_layer,
        'source_rv': rv_source,
        'target_rv_baseline': rv_baseline,
        'target_rv_patched': rv_patched,
        'delta_rv': rv_patched - rv_baseline,
        'transfer_pct': ((rv_patched - rv_baseline) / abs(rv_source - rv_baseline)) * 100 if abs(rv_source - rv_baseline) > 0.01 else 0
    }

def run_residual_analysis(model, tokenizer, prompt_name, prompt_text, target_layer):
    """Analyze residual stream at target layer"""
    tokens = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    
    if tokens['input_ids'].shape[1] < CONFIG['window_size'] + 1:
        return None
    
    # Hook residual stream (input to layer)
    residual_activations = []
    
    def residual_hook(module, input, output):
        # Input is tuple, first element is hidden_states
        if isinstance(input, tuple):
            residual_activations.append(input[0].detach().cpu())
    
    layer = model.model.layers[target_layer]
    hook_handle = layer.register_forward_hook(residual_hook)
    
    with torch.no_grad():
        model(**tokens)
    
    hook_handle.remove()
    
    if not residual_activations:
        return None
    
    residual = residual_activations[-1][0, -CONFIG['window_size']:, :]
    pr_residual = compute_pr(residual)
    
    # Also get V at this layer
    ext_v = VExtractor(model, target_layer)
    with torch.no_grad():
        model(**tokens)
    v = ext_v.activations[-1][0, -CONFIG['window_size']:, :]
    pr_v = compute_pr(v)
    ext_v.close()
    
    return {
        'prompt': prompt_name,
        'layer': target_layer,
        'pr_residual': pr_residual,
        'pr_v': pr_v,
        'ratio': pr_v / (pr_residual + 1e-8)
    }

def run_deep_analysis():
    print("="*70)
    print("DEEP CIRCUIT ANALYSIS V2: Comprehensive 30-Minute Deep Dive")
    print("="*70)
    print("Analyzing:")
    print("  1. Attention entropy & self-similarity per head (all key layers)")
    print("  2. Activation patching (L18→L27 causality)")
    print("  3. Residual stream analysis")
    print("  4. Cross-prompt comparisons")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    num_heads = model.config.num_attention_heads
    print(f"Model: {CONFIG['model_name']}")
    print(f"Attention heads per layer: {num_heads}")
    print(f"Key layers: {CONFIG['key_layers']}")
    
    all_results = {
        'attention': [],
        'activation_patching': [],
        'residual': []
    }
    
    # 1. ATTENTION ANALYSIS (All layers, all prompts)
    print("\n" + "="*70)
    print("PART 1: ATTENTION ENTROPY & SELF-SIMILARITY")
    print("="*70)
    
    for layer in tqdm(CONFIG['key_layers'], desc="Analyzing layers"):
        for prompt_name, prompt_text in PROMPTS.items():
            results = run_attention_analysis(model, tokenizer, prompt_name, prompt_text, layer)
            if results:
                all_results['attention'].extend(results)
    
    # 2. ACTIVATION PATCHING (Comprehensive)
    print("\n" + "="*70)
    print("PART 2: ACTIVATION PATCHING (Causality Tests)")
    print("="*70)
    
    patching_tests = [
        ("champion", "baseline", 18, 27, "Champion L18 → Baseline L27"),
        ("regress", "baseline", 18, 27, "Regress L18 → Baseline L27"),
        ("champion", "baseline", 5, 27, "Champion L5 → Baseline L27 (control)"),
        ("champion", "champion", 18, 27, "Champion L18 → Champion L27 (self-control)"),
        ("baseline", "champion", 18, 27, "Baseline L18 → Champion L27 (reverse)"),
    ]
    
    for src_prompt, tgt_prompt, src_layer, tgt_layer, desc in tqdm(patching_tests, desc="Patching tests"):
        print(f"\n  {desc}...")
        result = run_activation_patching(
            model, tokenizer,
            PROMPTS[src_prompt], PROMPTS[tgt_prompt],
            src_layer, tgt_layer
        )
        if result:
            result['description'] = desc
            all_results['activation_patching'].append(result)
            print(f"    Source R_V: {result['source_rv']:.4f}")
            print(f"    Target baseline R_V: {result['target_rv_baseline']:.4f}")
            print(f"    Target patched R_V: {result['target_rv_patched']:.4f}")
            print(f"    Delta: {result['delta_rv']:+.4f}")
            print(f"    Transfer: {result['transfer_pct']:+.1f}%")
    
    # 3. RESIDUAL STREAM ANALYSIS
    print("\n" + "="*70)
    print("PART 3: RESIDUAL STREAM ANALYSIS")
    print("="*70)
    
    for layer in tqdm(CONFIG['key_layers'], desc="Residual analysis"):
        for prompt_name, prompt_text in PROMPTS.items():
            result = run_residual_analysis(model, tokenizer, prompt_name, prompt_text, layer)
            if result:
                all_results['residual'].append(result)
    
    # Save and analyze
    print("\n" + "="*70)
    print("ANALYSIS & SAVING")
    print("="*70)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Attention analysis
    if all_results['attention']:
        df_attn = pd.DataFrame(all_results['attention'])
        filename = f"attention_analysis_{timestamp}.csv"
        df_attn.to_csv(filename, index=False)
        print(f"\n✅ Attention analysis: {filename}")
        
        # Find "microphone" heads (low entropy for champion)
        print("\n  MICROPHONE HEADS (Low Entropy for Champion):")
        for layer in [18, 27]:
            layer_data = df_attn[(df_attn['layer'] == layer) & (df_attn['prompt'] == 'champion')]
            if len(layer_data) > 0:
                low_entropy = layer_data.nsmallest(5, 'entropy')
                print(f"\n    Layer {layer}:")
                for _, row in low_entropy.iterrows():
                    print(f"      Head {int(row['head']):2d}: Entropy={row['entropy']:.4f}, Self-sim={row['self_similarity']:.4f}")
    
    # Activation patching
    if all_results['activation_patching']:
        df_patch = pd.DataFrame(all_results['activation_patching'])
        filename = f"activation_patching_{timestamp}.csv"
        df_patch.to_csv(filename, index=False)
        print(f"\n✅ Activation patching: {filename}")
        
        print("\n  CAUSALITY RESULTS:")
        for _, row in df_patch.iterrows():
            print(f"\n    {row['description']}:")
            print(f"      Transfer: {row['transfer_pct']:+.1f}%")
            print(f"      Delta R_V: {row['delta_rv']:+.4f}")
    
    # Residual analysis
    if all_results['residual']:
        df_res = pd.DataFrame(all_results['residual'])
        filename = f"residual_analysis_{timestamp}.csv"
        df_res.to_csv(filename, index=False)
        print(f"\n✅ Residual analysis: {filename}")
    
    print("\n" + "="*70)
    print("DEEP CIRCUIT ANALYSIS COMPLETE")
    print("="*70)
    
    return all_results

if __name__ == "__main__":
    results = run_deep_analysis()

