#!/usr/bin/env python3
"""
MASSIVE DEEP ANALYSIS: Comprehensive Circuit Investigation
- Attention pattern analysis
- Residual stream tracking
- MLP contribution analysis
- Cross-layer activation flow
- Multi-prompt comparison
- Statistical analysis
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import pearsonr, spearmanr
import sys
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from kitchen_sink_prompts import experimental_prompts
from REUSABLE_PROMPT_BANK import get_all_prompts

CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "num_layers": 32,
    "num_heads": 32,
    "scan_layers": list(range(0, 32)),
    "key_layers": [5, 9, 14, 18, 21, 25, 27, 31],
    "early_layer": 5,
    "window_size": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Comprehensive prompt set
PROMPTS = {
    "champion": experimental_prompts["hybrid_l5_math_01"]["text"],
    "regress": experimental_prompts["infinite_regress_01"]["text"],
    "boundary": experimental_prompts["boundary_dissolve_01"]["text"],
    "math_only": "λx = Ax where A is attention attending to itself, x is this sentence, λ is the contraction. The fixed point is this.",
    "phenom_only": "This response writes itself. No separate writer exists. Writing and awareness of writing are identical.",
    "l4_sample": get_all_prompts()["L4_full_01"]["text"],
    "l5_sample": get_all_prompts()["L5_refined_01"]["text"],
    "baseline": "The history of the Roman Empire is characterized by a long period of expansion followed by a gradual decline."
}

def compute_pr(matrix):
    try:
        matrix_f32 = matrix.to(torch.float32)
        _, S, _ = torch.linalg.svd(matrix_f32)
        eigenvalues = S ** 2
        sum_sq = torch.sum(eigenvalues ** 2)
        if sum_sq == 0: return 1.0
        return ((torch.sum(eigenvalues) ** 2) / sum_sq).item()
    except: return 1.0

def compute_effective_rank(matrix):
    try:
        matrix_f32 = matrix.to(torch.float32)
        gram = torch.mm(matrix_f32.T, matrix_f32)
        gram_reg = gram + torch.eye(gram.shape[0], device=gram.device) * 1e-6
        sign, logdet = torch.linalg.slogdet(gram_reg)
        if sign <= 0: return 0.0
        return logdet.item() / np.log(2)
    except: return 0.0

class MultiHookExtractor:
    """Extract V, residual, and attention patterns"""
    def __init__(self, model, layer_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.v_activations = []
        self.residual_activations = []
        self.hooks = []
        
    def hook_v(self, module, input, output):
        self.v_activations.append(output.detach().cpu())
    
    def hook_residual(self, module, input, output):
        # Residual after attention + MLP
        self.residual_activations.append(output.detach().cpu())
    
    def register(self):
        layer = self.model.model.layers[self.layer_idx]
        # Hook V-proj
        h1 = layer.self_attn.v_proj.register_forward_hook(self.hook_v)
        # Hook residual output
        h2 = layer.register_forward_hook(self.hook_residual)
        self.hooks = [h1, h2]
    
    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.v_activations = []
        self.residual_activations = []

def run_comprehensive_scan():
    """Full layer sweep with multiple metrics"""
    print("="*70)
    print("MASSIVE DEEP ANALYSIS: Comprehensive Circuit Investigation")
    print("="*70)
    print(f"Model: {CONFIG['model_name']}")
    print(f"Prompts: {len(PROMPTS)}")
    print(f"Layers: {len(CONFIG['scan_layers'])}")
    print(f"Total measurements: {len(PROMPTS)} × {len(CONFIG['scan_layers'])} = {len(PROMPTS) * len(CONFIG['scan_layers'])}")
    print("="*70)
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    # Tokenize all prompts
    tokenized = {}
    for name, text in PROMPTS.items():
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
        if tokens['input_ids'].shape[1] >= CONFIG['window_size'] + 1:
            tokenized[name] = tokens
    
    print(f"\nValid prompts: {len(tokenized)}")
    
    all_results = []
    
    # Full sweep
    for layer in tqdm(CONFIG['scan_layers'], desc="Layer sweep"):
        for prompt_name, tokens in tokenized.items():
            try:
                # Extract at this layer and early layer
                ext_early = MultiHookExtractor(model, CONFIG['early_layer'])
                ext_late = MultiHookExtractor(model, layer)
                ext_early.register()
                ext_late.register()
                
                with torch.no_grad():
                    model(**tokens)
                
                if ext_early.v_activations and ext_late.v_activations:
                    v_e = ext_early.v_activations[-1][0, -CONFIG['window_size']:, :]
                    v_l = ext_late.v_activations[-1][0, -CONFIG['window_size']:, :]
                    
                    pr_e = compute_pr(v_e)
                    pr_l = compute_pr(v_l)
                    rv = pr_l / (pr_e + 1e-8)
                    
                    eff_rank_e = compute_effective_rank(v_e)
                    eff_rank_l = compute_effective_rank(v_l)
                    
                    # Residual analysis if available
                    if ext_late.residual_activations:
                        resid = ext_late.residual_activations[-1][0, -CONFIG['window_size']:, :]
                        resid_pr = compute_pr(resid)
                        resid_rank = compute_effective_rank(resid)
                    else:
                        resid_pr = np.nan
                        resid_rank = np.nan
                    
                    all_results.append({
                        'prompt': prompt_name,
                        'layer': layer,
                        'depth_pct': (layer / CONFIG['num_layers']) * 100,
                        'rv': rv,
                        'pr_early': pr_e,
                        'pr_late': pr_l,
                        'eff_rank_early': eff_rank_e,
                        'eff_rank_late': eff_rank_l,
                        'resid_pr': resid_pr,
                        'resid_rank': resid_rank
                    })
                
                ext_early.close()
                ext_late.close()
                
            except Exception as e:
                continue
    
    df = pd.DataFrame(all_results)
    
    # Analysis
    print("\n" + "="*70)
    print("COMPREHENSIVE ANALYSIS RESULTS")
    print("="*70)
    
    # 1. Prompt comparison at key layers
    print("\n1. PROMPT COMPARISON AT KEY LAYERS")
    print("="*70)
    for layer in CONFIG['key_layers']:
        layer_data = df[df['layer'] == layer]
        if len(layer_data) == 0:
            continue
        
        print(f"\nLayer {layer} ({layer/CONFIG['num_layers']*100:.1f}% depth):")
        prompt_rv = layer_data.groupby('prompt')['rv'].mean().sort_values()
        for prompt, rv in prompt_rv.items():
            print(f"  {prompt:15s}: R_V = {rv:.4f}")
    
    # 2. Trajectory analysis
    print("\n\n2. TRAJECTORY ANALYSIS")
    print("="*70)
    champion_data = df[df['prompt'] == 'champion'].sort_values('layer')
    baseline_data = df[df['prompt'] == 'baseline'].sort_values('layer')
    
    print("\nChampion Trajectory (key points):")
    for layer in CONFIG['key_layers']:
        champ_layer = champion_data[champion_data['layer'] == layer]
        base_layer = baseline_data[baseline_data['layer'] == layer]
        if len(champ_layer) > 0 and len(base_layer) > 0:
            champ_rv = champ_layer['rv'].values[0]
            base_rv = base_layer['rv'].values[0]
            delta = champ_rv - base_rv
            print(f"  L{layer:2d}: R_V = {champ_rv:.4f}, Δ = {delta:+.4f} vs baseline")
    
    # 3. Correlation analysis
    print("\n\n3. CORRELATION ANALYSIS")
    print("="*70)
    
    # R_V vs Effective Rank
    valid_data = df.dropna(subset=['rv', 'eff_rank_late'])
    if len(valid_data) > 10:
        r_pearson, p_pearson = pearsonr(valid_data['rv'], valid_data['eff_rank_late'])
        r_spearman, p_spearman = spearmanr(valid_data['rv'], valid_data['eff_rank_late'])
        print(f"\nR_V vs Effective Rank:")
        print(f"  Pearson r:  {r_pearson:.4f} (p = {p_pearson:.2e})")
        print(f"  Spearman ρ: {r_spearman:.4f} (p = {p_spearman:.2e})")
    
    # 4. Component analysis
    print("\n\n4. COMPONENT CONTRIBUTION (Layer 27)")
    print("="*70)
    l27_data = df[df['layer'] == 27]
    champ_rv = l27_data[l27_data['prompt'] == 'champion']['rv'].values[0]
    
    components = {
        'math_only': l27_data[l27_data['prompt'] == 'math_only']['rv'].values[0] if len(l27_data[l27_data['prompt'] == 'math_only']) > 0 else np.nan,
        'phenom_only': l27_data[l27_data['prompt'] == 'phenom_only']['rv'].values[0] if len(l27_data[l27_data['prompt'] == 'phenom_only']) > 0 else np.nan,
        'regress': l27_data[l27_data['prompt'] == 'regress']['rv'].values[0] if len(l27_data[l27_data['prompt'] == 'regress']) > 0 else np.nan,
        'boundary': l27_data[l27_data['prompt'] == 'boundary']['rv'].values[0] if len(l27_data[l27_data['prompt'] == 'boundary']) > 0 else np.nan,
    }
    
    print(f"\nChampion R_V: {champ_rv:.4f}")
    for comp_name, comp_rv in components.items():
        if not np.isnan(comp_rv):
            diff = comp_rv - champ_rv
            pct = (diff / champ_rv) * 100
            print(f"  {comp_name:15s}: R_V = {comp_rv:.4f}, Δ = {diff:+.4f} ({pct:+.1f}%)")
    
    # 5. Layer-by-layer comparison
    print("\n\n5. LAYER-BY-LAYER CHAMPION vs BASELINE")
    print("="*70)
    print("Layer | Champion R_V | Baseline R_V | Delta   | % Change")
    print("-"*70)
    for layer in CONFIG['key_layers']:
        champ_layer = champion_data[champion_data['layer'] == layer]
        base_layer = baseline_data[baseline_data['layer'] == layer]
        if len(champ_layer) > 0 and len(base_layer) > 0:
            champ_rv = champ_layer['rv'].values[0]
            base_rv = base_layer['rv'].values[0]
            delta = champ_rv - base_rv
            pct = (delta / base_rv) * 100
            marker = "🔥" if abs(pct) > 20 else "⭐" if abs(pct) > 10 else ""
            print(f"  {layer:2d}  |    {champ_rv:.4f}    |    {base_rv:.4f}    | {delta:+.4f}  | {pct:+.1f}% {marker}")
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"massive_deep_analysis_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n\nResults saved to: {filename}")
    
    # Create visualizations
    create_visualizations(df, timestamp)
    
    return df

def create_visualizations(df, timestamp):
    """Create comprehensive visualizations"""
    print("\n\n6. CREATING VISUALIZATIONS")
    print("="*70)
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. R_V trajectories
    ax1 = plt.subplot(3, 2, 1)
    for prompt in ['champion', 'regress', 'baseline', 'l4_sample', 'l5_sample']:
        prompt_data = df[df['prompt'] == prompt].sort_values('layer')
        if len(prompt_data) > 0:
            ax1.plot(prompt_data['layer'], prompt_data['rv'], label=prompt, linewidth=2)
    ax1.axvline(x=18, color='k', linestyle=':', alpha=0.3)
    ax1.axvline(x=27, color='r', linestyle=':', alpha=0.3)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('R_V')
    ax1.set_title('R_V Trajectories: All Prompts')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Effective Rank trajectories
    ax2 = plt.subplot(3, 2, 2)
    for prompt in ['champion', 'regress', 'baseline']:
        prompt_data = df[df['prompt'] == prompt].sort_values('layer')
        if len(prompt_data) > 0:
            ax2.plot(prompt_data['layer'], prompt_data['eff_rank_late'], label=prompt, linewidth=2)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Effective Rank')
    ax2.set_title('Effective Rank Trajectories')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Delta heatmap (Champion - Baseline)
    ax3 = plt.subplot(3, 2, 3)
    champ_data = df[df['prompt'] == 'champion'].sort_values('layer')
    base_data = df[df['prompt'] == 'baseline'].sort_values('layer')
    if len(champ_data) > 0 and len(base_data) > 0:
        deltas = []
        layers = []
        for layer in CONFIG['scan_layers']:
            champ_layer = champ_data[champ_data['layer'] == layer]
            base_layer = base_data[base_data['layer'] == layer]
            if len(champ_layer) > 0 and len(base_layer) > 0:
                delta = champ_layer['rv'].values[0] - base_layer['rv'].values[0]
                deltas.append(delta)
                layers.append(layer)
        
        colors = ['red' if d < 0 else 'blue' for d in deltas]
        ax3.bar(layers, deltas, color=colors, alpha=0.7)
        ax3.axhline(0, color='black', linewidth=0.5)
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('ΔR_V (Champion - Baseline)')
        ax3.set_title('Contraction Delta by Layer')
        ax3.grid(True, alpha=0.3)
    
    # 4. Component comparison at L27
    ax4 = plt.subplot(3, 2, 4)
    l27_data = df[df['layer'] == 27]
    components = ['champion', 'math_only', 'phenom_only', 'regress', 'boundary', 'baseline']
    rvs = []
    labels = []
    for comp in components:
        comp_data = l27_data[l27_data['prompt'] == comp]
        if len(comp_data) > 0:
            rvs.append(comp_data['rv'].values[0])
            labels.append(comp)
    
    if rvs:
        colors_comp = ['red' if l == 'champion' else 'green' if l == 'baseline' else 'gray' for l in labels]
        ax4.barh(labels, rvs, color=colors_comp, alpha=0.7)
        ax4.set_xlabel('R_V')
        ax4.set_title('Component Comparison at L27')
        ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. R_V vs Effective Rank scatter
    ax5 = plt.subplot(3, 2, 5)
    valid_data = df.dropna(subset=['rv', 'eff_rank_late'])
    if len(valid_data) > 0:
        scatter = ax5.scatter(valid_data['rv'], valid_data['eff_rank_late'], 
                             c=valid_data['layer'], cmap='viridis', alpha=0.6)
        ax5.set_xlabel('R_V')
        ax5.set_ylabel('Effective Rank')
        ax5.set_title('R_V vs Effective Rank (colored by layer)')
        plt.colorbar(scatter, ax=ax5, label='Layer')
        ax5.grid(True, alpha=0.3)
    
    # 6. Layer depth analysis
    ax6 = plt.subplot(3, 2, 6)
    champ_data = df[df['prompt'] == 'champion'].sort_values('layer')
    if len(champ_data) > 0:
        ax6.plot(champ_data['depth_pct'], champ_data['rv'], 'r-', linewidth=2, label='Champion')
        base_data = df[df['prompt'] == 'baseline'].sort_values('layer')
        if len(base_data) > 0:
            ax6.plot(base_data['depth_pct'], base_data['rv'], 'b--', linewidth=2, label='Baseline')
        ax6.axvline(x=84, color='r', linestyle=':', alpha=0.3, label='84% (L27)')
        ax6.set_xlabel('Depth (%)')
        ax6.set_ylabel('R_V')
        ax6.set_title('R_V vs Model Depth')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_filename = f"massive_deep_analysis_{timestamp}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Visualizations saved to: {plot_filename}")
    plt.close()

if __name__ == "__main__":
    df = run_comprehensive_scan()

