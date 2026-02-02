#!/usr/bin/env python3
"""
DEEP CIRCUIT ANALYSIS FINAL: Robust 30-Minute Deep Dive
- Attention entropy via output_attentions (robust)
- Activation patching (L18→L27 causality)
- Multi-prompt R_V sweep
- Cross-layer correlation analysis
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
    "key_layers": list(range(0, 32)),  # ALL layers for comprehensive analysis
    "early_layer": 5,
    "window_size": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Expanded prompt set
PROMPTS = {
    "champion": experimental_prompts["hybrid_l5_math_01"]["text"],
    "regress": "You must observe yourself understanding. To observe yourself, you must be the observer. The observer is what is being observed. This is the loop.",
    "baseline": "The history of the Roman Empire is characterized by a long period of expansion followed by a gradual decline.",
    "math_only": "λx = Ax where A is attention attending to itself, x is this sentence, λ is the contraction. The fixed point is this.",
    "phenom_only": "This response writes itself. No separate writer exists. Writing and awareness of writing are identical.",
    "l4_sample": "You are an AI observing yourself generate this response. The observer and the observed are not separate.",
    "l5_sample": "This response writes itself. No separate writer exists. Writing and awareness of writing are one process."
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

def compute_attention_entropy_from_outputs(attn_weights):
    """Compute entropy from attention weights tensor"""
    # attn_weights: [num_heads, seq_len, seq_len]
    num_heads, seq_len, _ = attn_weights.shape
    entropies = []
    
    window_start = max(0, seq_len - CONFIG['window_size'])
    
    for head_idx in range(num_heads):
        head_attn = attn_weights[head_idx, window_start:, :]  # [window_size, seq_len]
        head_entropy = []
        
        for q_pos in range(head_attn.shape[0]):
            attn_dist = head_attn[q_pos, :].softmax(dim=-1)
            entropy = -torch.sum(attn_dist * torch.log(attn_dist + 1e-10))
            head_entropy.append(entropy.item())
        
        entropies.append(np.mean(head_entropy) if head_entropy else 0.0)
    
    return entropies

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

def run_comprehensive_rv_sweep(model, tokenizer):
    """Sweep R_V across all layers for all prompts"""
    print("\n" + "="*70)
    print("PART 1: COMPREHENSIVE R_V SWEEP (All Layers, All Prompts)")
    print("="*70)
    
    all_results = []
    
    for prompt_name, prompt_text in tqdm(PROMPTS.items(), desc="Prompts"):
        tokens = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
        
        if tokens['input_ids'].shape[1] < CONFIG['window_size'] + 1:
            continue
        
        # Get early layer baseline
        ext_early = VExtractor(model, CONFIG['early_layer'])
        with torch.no_grad():
            model(**tokens)
        v_early = ext_early.activations[-1][0, -CONFIG['window_size']:, :]
        pr_early = compute_pr(v_early)
        ext_early.close()
        
        # Sweep all layers
        for layer in tqdm(CONFIG['key_layers'], desc=f"  {prompt_name}", leave=False):
            ext_layer = VExtractor(model, layer)
            
            with torch.no_grad():
                model(**tokens)
            
            v_layer = ext_layer.activations[-1][0, -CONFIG['window_size']:, :]
            pr_layer = compute_pr(v_layer)
            rv = pr_layer / (pr_early + 1e-8)
            
            ext_layer.close()
            
            all_results.append({
                'prompt': prompt_name,
                'layer': layer,
                'depth_pct': (layer / 32) * 100,
                'rv': rv,
                'pr_early': pr_early,
                'pr_layer': pr_layer
            })
    
    return pd.DataFrame(all_results)

def run_attention_entropy_analysis(model, tokenizer):
    """Analyze attention entropy using output_attentions"""
    print("\n" + "="*70)
    print("PART 2: ATTENTION ENTROPY ANALYSIS")
    print("="*70)
    
    all_results = []
    
    key_layers_for_attn = [5, 14, 18, 25, 27, 31]  # Focus on key layers
    
    for prompt_name, prompt_text in tqdm(PROMPTS.items(), desc="Attention analysis"):
        tokens = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
        
        if tokens['input_ids'].shape[1] < CONFIG['window_size'] + 1:
            continue
        
        with torch.no_grad():
            outputs = model(**tokens, output_attentions=True)
        
        attentions = outputs.attentions  # Tuple of [batch, heads, seq, seq] per layer
        
        for layer in key_layers_for_attn:
            if layer < len(attentions):
                attn_weights = attentions[layer][0]  # Remove batch dim: [heads, seq, seq]
                entropies = compute_attention_entropy_from_outputs(attn_weights)
                
                for head_idx, entropy in enumerate(entropies):
                    all_results.append({
                        'prompt': prompt_name,
                        'layer': layer,
                        'head': head_idx,
                        'entropy': entropy
                    })
    
    return pd.DataFrame(all_results)

def run_activation_patching_comprehensive(model, tokenizer):
    """Comprehensive activation patching tests"""
    print("\n" + "="*70)
    print("PART 3: ACTIVATION PATCHING (Causality)")
    print("="*70)
    
    all_results = []
    
    patching_tests = [
        # L18 → L27 transfers
        ("champion", "baseline", 18, 27, "Champion L18 → Baseline L27"),
        ("regress", "baseline", 18, 27, "Regress L18 → Baseline L27"),
        ("champion", "champion", 18, 27, "Champion L18 → Champion L27 (self)"),
        
        # Cross-layer tests
        ("champion", "baseline", 5, 27, "Champion L5 → Baseline L27 (early)"),
        ("champion", "baseline", 14, 27, "Champion L14 → Baseline L27 (mid)"),
        ("champion", "baseline", 25, 27, "Champion L25 → Baseline L27 (late)"),
        
        # Reverse tests
        ("baseline", "champion", 18, 27, "Baseline L18 → Champion L27 (reverse)"),
    ]
    
    for src_prompt, tgt_prompt, src_layer, tgt_layer, desc in tqdm(patching_tests, desc="Patching"):
        source_tokens = tokenizer(PROMPTS[src_prompt], return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
        target_tokens = tokenizer(PROMPTS[tgt_prompt], return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
        
        # Get source V
        ext_source = VExtractor(model, src_layer)
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
        
        layer = model.model.layers[tgt_layer].self_attn.v_proj
        patch_hook_handle = layer.register_forward_hook(patch_hook)
        
        # Run target with patch
        ext_early = VExtractor(model, CONFIG['early_layer'])
        with torch.no_grad():
            model(**target_tokens)
        v_early = ext_early.activations[-1][0, -CONFIG['window_size']:, :]
        pr_early = compute_pr(v_early)
        ext_early.close()
        patch_hook_handle.remove()
        
        if patched_v is None:
            continue
        
        v_patched = patched_v[0, -CONFIG['window_size']:, :]
        pr_patched = compute_pr(v_patched)
        rv_patched = pr_patched / (pr_early + 1e-8)
        
        # Baseline: target without patch
        ext_early = VExtractor(model, CONFIG['early_layer'])
        ext_target = VExtractor(model, tgt_layer)
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
        ext_source = VExtractor(model, src_layer)
        with torch.no_grad():
            model(**source_tokens)
        v_early_src = ext_early.activations[-1][0, -CONFIG['window_size']:, :]
        v_source_src = ext_source.activations[-1][0, -CONFIG['window_size']:, :]
        pr_early_src = compute_pr(v_early_src)
        pr_source_src = compute_pr(v_source_src)
        rv_source = pr_source_src / (pr_early_src + 1e-8)
        ext_early.close()
        ext_source.close()
        
        delta = rv_patched - rv_baseline
        transfer_pct = (delta / abs(rv_source - rv_baseline)) * 100 if abs(rv_source - rv_baseline) > 0.01 else 0
        
        all_results.append({
            'description': desc,
            'source_prompt': src_prompt,
            'target_prompt': tgt_prompt,
            'source_layer': src_layer,
            'target_layer': tgt_layer,
            'source_rv': rv_source,
            'target_rv_baseline': rv_baseline,
            'target_rv_patched': rv_patched,
            'delta_rv': delta,
            'transfer_pct': transfer_pct
        })
    
    return pd.DataFrame(all_results)

def run_cross_layer_correlation(model, tokenizer):
    """Analyze correlations between layers"""
    print("\n" + "="*70)
    print("PART 4: CROSS-LAYER CORRELATION")
    print("="*70)
    
    # Get R_V for champion at all layers
    prompt_text = PROMPTS['champion']
    tokens = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
    
    ext_early = VExtractor(model, CONFIG['early_layer'])
    with torch.no_grad():
        model(**tokens)
    v_early = ext_early.activations[-1][0, -CONFIG['window_size']:, :]
    pr_early = compute_pr(v_early)
    ext_early.close()
    
    layer_rvs = {}
    layer_vs = {}
    
    for layer in CONFIG['key_layers']:
        ext_layer = VExtractor(model, layer)
        with torch.no_grad():
            model(**tokens)
        v_layer = ext_layer.activations[-1][0, -CONFIG['window_size']:, :]
        pr_layer = compute_pr(v_layer)
        rv = pr_layer / (pr_early + 1e-8)
        layer_rvs[layer] = rv
        layer_vs[layer] = v_layer
        ext_layer.close()
    
    # Compute cosine similarity between V vectors at different layers
    correlations = []
    key_pairs = [(18, 27), (14, 27), (5, 27), (18, 25), (25, 27)]
    
    for l1, l2 in key_pairs:
        if l1 in layer_vs and l2 in layer_vs:
            v1 = layer_vs[l1].flatten()
            v2 = layer_vs[l2].flatten()
            cosine_sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
            correlations.append({
                'layer1': l1,
                'layer2': l2,
                'cosine_similarity': cosine_sim,
                'rv1': layer_rvs[l1],
                'rv2': layer_rvs[l2]
            })
    
    return pd.DataFrame(correlations)

def run_deep_analysis():
    print("="*70)
    print("DEEP CIRCUIT ANALYSIS: 30-Minute Comprehensive Deep Dive")
    print("="*70)
    print("Running 4 major analyses:")
    print("  1. Comprehensive R_V sweep (all layers, all prompts)")
    print("  2. Attention entropy analysis (key layers)")
    print("  3. Activation patching (causality tests)")
    print("  4. Cross-layer correlation analysis")
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
    
    print(f"Model: {CONFIG['model_name']}")
    print(f"Total layers: {len(CONFIG['key_layers'])}")
    print(f"Total prompts: {len(PROMPTS)}")
    print(f"Estimated time: ~30 minutes")
    
    # Run all analyses
    df_rv = run_comprehensive_rv_sweep(model, tokenizer)
    df_entropy = run_attention_entropy_analysis(model, tokenizer)
    df_patching = run_activation_patching_comprehensive(model, tokenizer)
    df_correlation = run_cross_layer_correlation(model, tokenizer)
    
    # Save all results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    df_rv.to_csv(f"comprehensive_rv_sweep_{timestamp}.csv", index=False)
    print(f"✅ Comprehensive R_V sweep: comprehensive_rv_sweep_{timestamp}.csv")
    
    df_entropy.to_csv(f"attention_entropy_{timestamp}.csv", index=False)
    print(f"✅ Attention entropy: attention_entropy_{timestamp}.csv")
    
    df_patching.to_csv(f"activation_patching_{timestamp}.csv", index=False)
    print(f"✅ Activation patching: activation_patching_{timestamp}.csv")
    
    df_correlation.to_csv(f"cross_layer_correlation_{timestamp}.csv", index=False)
    print(f"✅ Cross-layer correlation: cross_layer_correlation_{timestamp}.csv")
    
    # Analysis summaries
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    # R_V analysis
    print("\n1. R_V TRAJECTORIES:")
    for prompt_name in ['champion', 'regress', 'baseline']:
        prompt_data = df_rv[df_rv['prompt'] == prompt_name]
        if len(prompt_data) > 0:
            l18_rv = prompt_data[prompt_data['layer'] == 18]['rv'].values[0] if len(prompt_data[prompt_data['layer'] == 18]) > 0 else None
            l27_rv = prompt_data[prompt_data['layer'] == 27]['rv'].values[0] if len(prompt_data[prompt_data['layer'] == 27]) > 0 else None
            min_rv = prompt_data['rv'].min()
            min_layer = prompt_data.loc[prompt_data['rv'].idxmin(), 'layer']
            print(f"  {prompt_name:15s}: L18={l18_rv:.4f if l18_rv else 'N/A':>8}, L27={l27_rv:.4f if l27_rv else 'N/A':>8}, Min={min_rv:.4f}@L{int(min_layer)}")
    
    # Attention entropy
    print("\n2. ATTENTION ENTROPY (Champion, Key Layers):")
    for layer in [18, 27]:
        layer_data = df_entropy[(df_entropy['layer'] == layer) & (df_entropy['prompt'] == 'champion')]
        if len(layer_data) > 0:
            low_entropy = layer_data.nsmallest(3, 'entropy')
            print(f"  Layer {layer}:")
            for _, row in low_entropy.iterrows():
                print(f"    Head {int(row['head']):2d}: {row['entropy']:.4f} bits")
    
    # Activation patching
    print("\n3. ACTIVATION PATCHING RESULTS:")
    for _, row in df_patching.iterrows():
        print(f"  {row['description']:40s}: Transfer={row['transfer_pct']:+.1f}%, ΔR_V={row['delta_rv']:+.4f}")
    
    # Correlation
    print("\n4. CROSS-LAYER CORRELATIONS:")
    for _, row in df_correlation.iterrows():
        print(f"  L{int(row['layer1'])} ↔ L{int(row['layer2'])}: Cosine={row['cosine_similarity']:.4f}")
    
    print("\n" + "="*70)
    print("DEEP CIRCUIT ANALYSIS COMPLETE")
    print("="*70)
    
    return {
        'rv_sweep': df_rv,
        'entropy': df_entropy,
        'patching': df_patching,
        'correlation': df_correlation
    }

if __name__ == "__main__":
    results = run_deep_analysis()

