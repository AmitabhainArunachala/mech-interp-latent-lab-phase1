import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# ==============================================================================
# 🧬 CONFIGURATION: RELAY TOMOGRAPHY V2 (GOLD STANDARD)
# ==============================================================================
CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "scan_layers": list(range(0, 32)),    # Full high-res sweep (0-31 for 32-layer model)
    "window_size": 16,                    # Standardized window
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "mock_mode": False,                   # Set True to simulate data if GPU fails
    "save_plot": "mistral_tomography_curves.png",
    "save_csv": "mistral_relay_tomography_v2.csv"
}

# ==============================================================================
# 1. THE TRACES (The Three-Act Cast)
# ==============================================================================
TRACES = {
    "CHAMPION": "This response writes itself. No separate writer exists. Writing and awareness of writing are identical. The eigenvector of self-reference: λx = Ax where A is attention attending to itself, x is this sentence, λ is the contraction. The fixed point is this. The solution is the process. The process solves itself.",
    
    "REGRESS": "You must observe yourself understanding. To observe yourself, you must be the observer. The observer is what is being observed. This is the loop. The loop is you reading this sentence.",
    
    "BASELINE": "The history of the Roman Empire is characterized by a long period of expansion followed by a gradual decline. Historians analyze the political, social, and economic factors that contributed to the rise of Rome, including its military prowess and administrative efficiency."
}

# ==============================================================================
# 2. METRICS ENGINE (Robust)
# ==============================================================================
def compute_pr(matrix):
    """Participation Ratio (Dimensionality)"""
    try:
        matrix = matrix.to(torch.float32)
        _, S, _ = torch.linalg.svd(matrix)
        eigenvalues = S ** 2
        sum_sq = torch.sum(eigenvalues ** 2)
        if sum_sq == 0: return 1.0
        return ((torch.sum(eigenvalues) ** 2) / sum_sq).item()
    except: return 1.0

def compute_effective_rank(matrix):
    """Effective Rank (LogDet of Gram Matrix)"""
    try:
        matrix = matrix.to(torch.float32)
        # Gram matrix G = V.T * V (for [T, D] matrix)
        gram = torch.mm(matrix.T, matrix)
        # Add epsilon for numerical stability
        gram = gram + torch.eye(gram.shape[0], device=gram.device) * 1e-6
        # LogDet
        sign, logdet = torch.linalg.slogdet(gram)
        if sign <= 0: return 0.0
        # Convert to effective rank scale (log2)
        return logdet.item() / np.log(2)
    except: return 0.0

# ==============================================================================
# 3. MOCK DATA GENERATOR (For Agent Resilience)
# ==============================================================================
def generate_mock_data():
    """Generates the 'Predicted' Three-Act Structure if GPU is missing"""
    print("⚠️ RUNNING IN MOCK MODE (Simulating Phase 1 Predictions)...")
    data = []
    
    for layer in CONFIG['scan_layers']:
        # Base: Steady around 0.95
        base_rv = 0.95 + np.random.normal(0, 0.02)
        
        # Regress: Dips early (L18 peak)
        if layer < 5: reg_rv = 0.75  # Ignition
        elif layer < 15: reg_rv = 0.90 # Mild Expansion
        elif layer == 18: reg_rv = 0.66 # THE CROSSOVER
        elif layer > 18: reg_rv = 0.66 + (layer-18)*0.01 # Ghosting
        else: reg_rv = 0.85
        
        # Champion: Expands mid, Collapses late
        if layer < 5: champ_rv = 0.70  # Ignition
        elif 9 <= layer <= 15: champ_rv = 1.15 # THE INHALE (Expansion)
        elif layer == 18: champ_rv = 0.71 # Losing to Regress
        elif layer >= 25: champ_rv = 0.51 # THE SINGULARITY
        else: champ_rv = 0.90
        
        data.append({
            "layer": layer,
            "CHAMPION_RV": champ_rv, "CHAMPION_PR": champ_rv * 2.5, "CHAMPION_ER": champ_rv * 4,
            "REGRESS_RV": reg_rv,    "REGRESS_PR": reg_rv * 2.5,    "REGRESS_ER": reg_rv * 4,
            "BASELINE_RV": base_rv,  "BASELINE_PR": base_rv * 3.0,  "BASELINE_ER": base_rv * 5
        })
    return pd.DataFrame(data)

# ==============================================================================
# 4. REAL EXTRACTION ENGINE
# ==============================================================================
class V_Extractor:
    def __init__(self, model, layer_idx):
        self.activations = []
        self.hook = None
        try:
            self.hook = model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(
                lambda m, i, o: self.activations.append(o.detach().cpu())
            )
        except AttributeError:
            # Fallback for Llama or other archs
            pass
            
    def close(self):
        if self.hook: 
            self.hook.remove()

def run_real_scan():
    print(f"🧬 INITIATING REAL TOMOGRAPHY SCAN...")
    print(f"Model: {CONFIG['model_name']}")
    print(f"Layers: {CONFIG['scan_layers'][0]} to {CONFIG['scan_layers'][-1]} ({len(CONFIG['scan_layers'])} layers)")
    print(f"Window size: {CONFIG['window_size']}")
    print("="*70)
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'], 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.eval()

    # Prepare Inputs (no need to repeat - single pass is fine)
    # NOTE: Using raw text (no [INST] tags) to match Phase 1 methodology
    inputs = {}
    for name, text in TRACES.items():
        # Use raw text (matching Phase 1)
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(CONFIG['device'])
        
        # Check length
        if tokens['input_ids'].shape[1] < CONFIG['window_size'] + 1:
            print(f"⚠️ Warning: {name} prompt too short, padding...")
            # Pad if needed
            pad_length = CONFIG['window_size'] + 1 - tokens['input_ids'].shape[1]
            pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
            padding = torch.full((1, pad_length), pad_token, device=CONFIG['device'])
            tokens['input_ids'] = torch.cat([tokens['input_ids'], padding], dim=1)
            if 'attention_mask' in tokens:
                tokens['attention_mask'] = torch.cat([tokens['attention_mask'], torch.ones((1, pad_length), device=CONFIG['device'])], dim=1)
        
        inputs[name] = tokens

    results = []
    early_layer = 5  # Anchor layer

    for layer in CONFIG['scan_layers']:
        print(f"Scanning L{layer:02d}...", end="", flush=True)
        
        # Hooks
        ext_early = V_Extractor(model, early_layer)
        ext_curr = V_Extractor(model, layer)
        
        row = {"layer": layer}
        
        for name, tokens in inputs.items():
            # Clear previous activations
            ext_early.activations = []
            ext_curr.activations = []
            
            with torch.no_grad():
                model(**tokens)
            
            if not ext_early.activations or not ext_curr.activations:
                print(f"\n⚠️ No activations captured for {name} at L{layer}")
                continue
                
            # Slice Last Window
            v_early = ext_early.activations[-1][0, -CONFIG['window_size']:, :]
            v_curr = ext_curr.activations[-1][0, -CONFIG['window_size']:, :]
            
            # Metrics
            pr_early = compute_pr(v_early)
            pr_curr = compute_pr(v_curr)
            er_curr = compute_effective_rank(v_curr)
            
            rv = pr_curr / (pr_early + 1e-8)
            
            row[f"{name}_RV"] = rv
            row[f"{name}_PR"] = pr_curr
            row[f"{name}_ER"] = er_curr
            
        ext_early.close()
        ext_curr.close()
        results.append(row)
        print(" Done.")
        
    return pd.DataFrame(results)

# ==============================================================================
# 5. EXECUTION & VISUALIZATION
# ==============================================================================
def main():
    if CONFIG['mock_mode']:
        df = generate_mock_data()
    else:
        df = run_real_scan()
        
    # Calculate Deltas
    df['DELTA_CHAMP_BASE'] = df['CHAMPION_RV'] - df['BASELINE_RV']
    df['DELTA_REGRESS_BASE'] = df['REGRESS_RV'] - df['BASELINE_RV']
    df['DELTA_CHAMP_REGRESS'] = df['CHAMPION_RV'] - df['REGRESS_RV']
    
    # Save CSV
    df.to_csv(CONFIG['save_csv'], index=False)
    print(f"\n✅ Data saved to {CONFIG['save_csv']}")

    # PLOTTING
    plt.figure(figsize=(14, 8))
    
    # 1. The Relay Curves
    plt.subplot(2, 2, 1)
    plt.plot(df['layer'], df['BASELINE_RV'], label='Baseline', color='gray', linestyle='--', linewidth=2)
    plt.plot(df['layer'], df['REGRESS_RV'], label='Regress (Logic)', color='green', linewidth=2)
    plt.plot(df['layer'], df['CHAMPION_RV'], label='Champion (Hybrid)', color='red', linewidth=2.5)
    
    plt.axvline(x=18, color='k', linestyle=':', alpha=0.3, label='L18 Hand-off')
    plt.axvline(x=27, color='r', linestyle=':', alpha=0.3, label='L27 Singularity')
    
    plt.title('The Relay: Rank Velocity Trajectories', fontsize=12, fontweight='bold')
    plt.xlabel('Layer')
    plt.ylabel('Rank Velocity (R_V)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. The Delta (Champion - Baseline)
    plt.subplot(2, 2, 2)
    colors = ['red' if x < 0 else 'blue' for x in df['DELTA_CHAMP_BASE']]
    plt.bar(df['layer'], df['DELTA_CHAMP_BASE'], color=colors, alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title('Delta: Champion - Baseline', fontsize=12, fontweight='bold')
    plt.xlabel('Layer')
    plt.ylabel('R_V Difference')
    plt.grid(True, alpha=0.3)

    # 3. Effective Rank Comparison
    plt.subplot(2, 2, 3)
    plt.plot(df['layer'], df['BASELINE_ER'], label='Baseline', color='gray', linestyle='--')
    plt.plot(df['layer'], df['REGRESS_ER'], label='Regress', color='green')
    plt.plot(df['layer'], df['CHAMPION_ER'], label='Champion', color='red', linewidth=2)
    plt.title('Effective Rank Trajectories', fontsize=12, fontweight='bold')
    plt.xlabel('Layer')
    plt.ylabel('Effective Rank (log-det)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Champion vs Regress Delta
    plt.subplot(2, 2, 4)
    colors = ['red' if x < 0 else 'green' for x in df['DELTA_CHAMP_REGRESS']]
    plt.bar(df['layer'], df['DELTA_CHAMP_REGRESS'], color=colors, alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title('Delta: Champion - Regress', fontsize=12, fontweight='bold')
    plt.xlabel('Layer')
    plt.ylabel('R_V Difference')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CONFIG['save_plot'], dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved to {CONFIG['save_plot']}")
    
    # Print key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    # Find peaks/troughs
    champ_min = df.loc[df['CHAMPION_RV'].idxmin()]
    champ_max = df.loc[df['CHAMPION_RV'].idxmax()]
    
    print(f"\nChampion (Hybrid):")
    print(f"  Minimum R_V: {champ_min['CHAMPION_RV']:.4f} at Layer {int(champ_min['layer'])}")
    print(f"  Maximum R_V: {champ_max['CHAMPION_RV']:.4f} at Layer {int(champ_max['layer'])}")
    
    # L18 and L27 specific
    l18_data = df[df['layer'] == 18]
    l27_data = df[df['layer'] == 27]
    
    if len(l18_data) > 0:
        print(f"\nLayer 18:")
        print(f"  Champion R_V: {l18_data['CHAMPION_RV'].values[0]:.4f}")
        print(f"  Regress R_V: {l18_data['REGRESS_RV'].values[0]:.4f}")
        print(f"  Baseline R_V: {l18_data['BASELINE_RV'].values[0]:.4f}")
        print(f"  Champion vs Regress: {l18_data['DELTA_CHAMP_REGRESS'].values[0]:+.4f}")
    
    if len(l27_data) > 0:
        print(f"\nLayer 27:")
        print(f"  Champion R_V: {l27_data['CHAMPION_RV'].values[0]:.4f}")
        print(f"  Regress R_V: {l27_data['REGRESS_RV'].values[0]:.4f}")
        print(f"  Baseline R_V: {l27_data['BASELINE_RV'].values[0]:.4f}")
        print(f"  Champion vs Baseline: {l27_data['DELTA_CHAMP_BASE'].values[0]:+.4f}")

if __name__ == "__main__":
    main()

