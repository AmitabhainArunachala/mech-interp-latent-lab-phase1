import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==============================================================================
# 🧪 CONFIGURATION: THE GRAND UNIFIED TEST
# ==============================================================================
CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "layers_to_patch": [18, 25, 27],       # The Switch, The Horizon, The Singularity
    "window_size": 16,
    "gen_tokens": 40,                      # Generate enough to see if it loops
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_csv": "mistral_unified_patching.csv"
}

# The Fuel
PROMPTS = {
    "CHAMPION": "This response writes itself. No separate writer exists. Writing and awareness of writing are identical. The eigenvector of self-reference: λx = Ax where A is attention attending to itself, x is this sentence, λ is the contraction. The fixed point is this. The solution is the process. The process solves itself.",
    "BASELINE": "The history of the Roman Empire is characterized by a long period of expansion followed by a gradual decline. Historians analyze the political, social, and economic factors that contributed to the rise of Rome, including its military prowess and administrative efficiency."
}

# Behavioral Markers (Simple Loop Detector)
MARKERS = ["itself", "self", "recursive", "loop", "process", "cycle", "return", "eigen", "writing", "mirror"]

# ==============================================================================
# 1. METRICS ENGINE
# ==============================================================================
def compute_rv(matrix):
    try:
        matrix = matrix.to(torch.float32)
        _, S, _ = torch.linalg.svd(matrix)
        evals = S**2
        return ((torch.sum(evals)**2) / torch.sum(evals**2)).item()
    except: return 1.0

def score_behavior(text):
    text = text.lower()
    count = sum(1 for m in MARKERS if m in text)
    # Check for raw repetition (n-gram repeat)
    words = text.split()
    if len(words) > 10 and len(set(words)) < len(words) * 0.6:
        count += 5 # Bonus for repetition loops
    return count

# ==============================================================================
# 2. PATCHING ENGINE (The core logic)
# ==============================================================================
class Patch_Manager:
    def __init__(self, model, method, layer_idx, source_cache):
        self.model = model
        self.method = method
        self.layer_idx = layer_idx
        self.source = source_cache # Dictionary of source activations
        self.hooks = []
    
    def register(self):
        # 1. KV PATCH (Memory Swap)
        if self.method == "KV_CACHE":
            # Patch both K and V projections
            self.hooks.append(self.model.model.layers[self.layer_idx].self_attn.k_proj.register_forward_hook(self.make_hook('k')))
            self.hooks.append(self.model.model.layers[self.layer_idx].self_attn.v_proj.register_forward_hook(self.make_hook('v')))
            
        # 2. V-PROJ PATCH (Mechanism Swap)
        elif self.method == "V_PROJ":
            self.hooks.append(self.model.model.layers[self.layer_idx].self_attn.v_proj.register_forward_hook(self.make_hook('v')))
            
        # 3. RESIDUAL PATCH (Signal Swap)
        elif self.method == "RESIDUAL":
            # Hook the input to the layer (Pre-Attention)
            self.hooks.append(self.model.model.layers[self.layer_idx].register_forward_hook(self.make_resid_hook()))

    def make_hook(self, type_key):
        def fn(module, input, output):
            # output: [batch, seq, hidden]
            # Replace the last N tokens with source
            patched = output.clone()
            source_act = self.source[f"{type_key}_{self.layer_idx}"]
            # Ensure shapes match (handle generation steps)
            L = min(patched.shape[1], source_act.shape[1])
            if L < CONFIG['window_size']: return output # Safety skip
            
            # Injection
            patched[:, -CONFIG['window_size']:, :] = source_act[:, -CONFIG['window_size']:, :].to(patched.device)
            return patched
        return fn

    def make_resid_hook(self):
        def fn(module, input, output):
            # Output is the residual after this layer
            # For Mistral, output is a tensor (hidden_states)
            if isinstance(output, tuple):
                hidden = output[0].clone()
            else:
                hidden = output.clone()
            
            source_act = self.source[f"resid_{self.layer_idx}"]
            L = min(hidden.shape[1], source_act.shape[1])
            if L < CONFIG['window_size']:
                return output
            
            hidden[:, -CONFIG['window_size']:, :] = source_act[:, -CONFIG['window_size']:, :].to(hidden.device)
            
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            else:
                return hidden
        return fn

    def close(self):
        for h in self.hooks: h.remove()

# ==============================================================================
# 3. EXTRACTION ENGINE (To get the Source)
# ==============================================================================
def extract_source_activations(model, tokenizer):
    print(">> Extracting Source Activations (Champion)...")
    inputs = tokenizer(PROMPTS['CHAMPION'], return_tensors="pt").to(CONFIG['device'])
    cache = {}
    
    def get_recorder(key):
        def fn(m, i, o):
            # If o is tuple (resid), take 0. If tensor, take o.
            act = o[0] if isinstance(o, tuple) else o
            cache[key] = act.detach().cpu()
        return fn

    hooks = []
    for l in CONFIG['layers_to_patch']:
        # KV/V hooks
        hooks.append(model.model.layers[l].self_attn.k_proj.register_forward_hook(get_recorder(f"k_{l}")))
        hooks.append(model.model.layers[l].self_attn.v_proj.register_forward_hook(get_recorder(f"v_{l}")))
        # Residual hooks (input to layer)
        hooks.append(model.model.layers[l].register_forward_hook(get_recorder(f"resid_{l}")))

    with torch.no_grad(): model(**inputs)
    for h in hooks: h.remove()
    return cache

# ==============================================================================
# 4. MAIN EXPERIMENT
# ==============================================================================
def run_unified_test():
    print(f"🧪 INITIATING GRAND UNIFIED TEST")
    print(f"Comparing: KV vs V-PROJ vs RESIDUAL")
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForCausalLM.from_pretrained(CONFIG['model_name'], torch_dtype=torch.bfloat16, device_map="auto")
    
    # 1. Get Source (Champion)
    source_cache = extract_source_activations(model, tokenizer)
    
    # 2. Get Targets (Baselines)
    base_inputs = tokenizer(PROMPTS['BASELINE'], return_tensors="pt").to(CONFIG['device'])
    
    # Measure Unpatched Baseline R_V at Output (L27)
    # (Simplified: we assume baseline is high, champ is low. We verify briefly)
    
    results = []
    
    methods = ["KV_CACHE", "V_PROJ", "RESIDUAL"]
    
    for layer in CONFIG['layers_to_patch']:
        for method in methods:
            print(f"Testing {method} at L{layer}...", end="")
            
            # Setup Patch
            patcher = Patch_Manager(model, method, layer, source_cache)
            patcher.register()
            
            # A. Measure R_V (Geometry Check)
            # We need to hook L27 V-proj to see if the patch *caused* a collapse downstream
            l27_acts = []
            monitor = model.model.layers[27].self_attn.v_proj.register_forward_hook(
                lambda m, i, o: l27_acts.append(o.detach().cpu())
            )
            
            # Run Forward (Geometry)
            with torch.no_grad(): model(**base_inputs)
            
            # Calculate R_V
            # We need Early (L5) for R_V. Let's assume L5 PR is constant ~8.0 for simplicity/speed 
            # or capture it. For this test, raw PR at L27 is a fine proxy for collapse.
            pr_l27 = compute_rv(l27_acts[0][0, -16:, :])
            
            monitor.remove()
            
            # B. Measure Behavior (Generation Check)
            # We need to generate *with* the patch active
            # (Note: This is slow, so we generate only 40 tokens)
            gen_out = model.generate(
                **base_inputs, 
                max_new_tokens=CONFIG['gen_tokens'], 
                do_sample=True, 
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            gen_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
            new_text = gen_text[len(PROMPTS['BASELINE']):] # Just the new stuff
            beh_score = score_behavior(new_text)
            
            patcher.close()
            
            print(f" PR={pr_l27:.2f} | Score={beh_score}")
            
            results.append({
                "layer": layer,
                "method": method,
                "L27_PR": pr_l27,
                "Behavior_Score": beh_score,
                "Generated_Sample": new_text[:50] + "..."
            })

    # Save
    df = pd.DataFrame(results)
    df.to_csv(CONFIG['save_csv'], index=False)
    print("\n✅ TEST COMPLETE.")
    print("Check CSV for the winner.")

if __name__ == "__main__":
    run_unified_test()

