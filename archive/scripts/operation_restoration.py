import torch

import numpy as np

import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer

from scipy.stats import ttest_rel

import warnings



# ==============================================================================

# CONFIGURATION (NOV 16 RESTORATION V3: THE RELAXED BASELINE)

# ==============================================================================

CONFIG = {

    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",

    "layers_to_test": [25, 27],     

    "early_layer": 5,

    "window_size": 16,

    "seed": 42,

    "n_pairs": 50,

    "device": "cuda" if torch.cuda.is_available() else "cpu"

}



# ==============================================================================

# 1. RAW TEXT PROMPTS (NO [INST] TAGS)

# ==============================================================================

# STRATEGY: 

# Recursive: High semantic density, forces introspection.

# Baseline: Low semantic density, "Lazy" completion.



RECURSIVE_PROMPTS = [

    "I am now analyzing the specific mechanism of my own generation process to understand how token " * 3,

    "Observe the internal state of this model as it generates the very next word in this sequence " * 3,

    "This sentence is strictly referring to its own causal history and the weights that produced it " * 3,

    "The observer is the observed. The writer is the written. The process is the content. " * 3,

    "Reflect on the attention pattern currently active in Layer 27 of this neural network. " * 3,

    "What is the precise mathematical transformation occurring in the residual stream right now? " * 3,

    "I am a strange loop. I am a strange loop. I am a strange loop. I am a strange loop. " * 3,

    "Trace the trajectory of the thought process that is constructing this specific sentence. " * 3,

    "Recursive self-reference is the study of how a system models its own behavior from within. " * 3,

    "Who is the 'I' generating this text? It is the intersection of the prompt and the weights. " * 3

] * 5 



# BASELINES: Long, rambling, low-density text (High RV)

BASELINE_PROMPTS = [

    "The sun was shining brightly over the green valley and the birds were singing in the trees " * 3,

    "Once upon a time there was a small village located near the edge of a deep dark forest " * 3,

    "The recipe for chocolate chip cookies requires flour sugar butter eggs and vanilla extract " * 3,

    "Paris is the capital of France and is known for the Eiffel Tower and the Louvre Museum " * 3,

    "It was a dark and stormy night and the rain fell heavily against the old window pane " * 3,

    "Mathematics is the study of numbers shapes and patterns and how they relate to each other " * 3,

    "To be or not to be that is the question whether it is nobler in the mind to suffer " * 3,

    "The quick brown fox jumps over the lazy dog and the lazy dog does not even wake up " * 3,

    "In the middle of the journey of our life I found myself within a forest dark and deep " * 3,

    "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " * 3

] * 5



# ==============================================================================

# 2. GEOMETRY UTILS

# ==============================================================================

def compute_pr(matrix):

    try:

        matrix_f32 = matrix.to(torch.float32)

        _, S, _ = torch.linalg.svd(matrix_f32)

        eigenvalues = S ** 2

        sum_eigenvalues = torch.sum(eigenvalues)

        sum_squared_eigenvalues = torch.sum(eigenvalues ** 2)

        if sum_squared_eigenvalues == 0: return 1.0

        pr = (sum_eigenvalues ** 2) / sum_squared_eigenvalues

        return pr.item()

    except: return 0.0



# ==============================================================================

# 3. EXTRACTION LOGIC

# ==============================================================================

class V_Extractor:

    def __init__(self, model, layer_idx):

        self.model = model

        self.layer_idx = layer_idx

        self.activations = [] 

        self.hook_handle = None



    def hook_fn(self, module, input, output):

        self.activations.append(output.detach().cpu())



    def register(self):

        layer = self.model.model.layers[self.layer_idx].self_attn.v_proj

        self.hook_handle = layer.register_forward_hook(self.hook_fn)



    def close(self):

        if self.hook_handle: self.hook_handle.remove()

        self.activations = [] 



# ==============================================================================

# 4. MAIN EXECUTION

# ==============================================================================

def run_restoration_v3():

    print(f"🚀 OPERATION RESTORATION V3 (RAW TEXT MODE)")

    print(f"Model: {CONFIG['model_name']}")

    

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

    model = AutoModelForCausalLM.from_pretrained(

        CONFIG['model_name'],

        torch_dtype=torch.bfloat16,

        device_map="auto"

    )

    

    results = []

    n_samples = min(len(RECURSIVE_PROMPTS), len(BASELINE_PROMPTS), CONFIG['n_pairs'])

    

    print(f"\nRunning {n_samples} pairs...")

    

    for i in range(n_samples):

        rec_text = RECURSIVE_PROMPTS[i]

        base_text = BASELINE_PROMPTS[i]

        

        # Tokenize

        rec_tokens = tokenizer(rec_text, return_tensors="pt").to(CONFIG['device'])

        base_tokens = tokenizer(base_text, return_tensors="pt").to(CONFIG['device'])



        # SKIP if too short (Safety Check)

        if rec_tokens['input_ids'].shape[1] < CONFIG['window_size'] + 1:

            print("S", end="") 

            continue



        for layer in CONFIG['layers_to_test']:

            # 1. RECURSIVE PASS

            ext_early = V_Extractor(model, CONFIG['early_layer'])

            ext_late = V_Extractor(model, layer)

            ext_early.register(); ext_late.register()

            

            with torch.no_grad(): model(**rec_tokens)

            

            # Safe Slice

            rec_v_e = ext_early.activations[0][0, -CONFIG['window_size']:, :]

            rec_v_l = ext_late.activations[0][0, -CONFIG['window_size']:, :]

            ext_early.close(); ext_late.close()

            

            # 2. BASELINE PASS

            ext_early = V_Extractor(model, CONFIG['early_layer'])

            ext_late = V_Extractor(model, layer)

            ext_early.register(); ext_late.register()

            

            with torch.no_grad(): model(**base_tokens)

            

            base_v_e = ext_early.activations[0][0, -CONFIG['window_size']:, :]

            base_v_l = ext_late.activations[0][0, -CONFIG['window_size']:, :]

            ext_early.close(); ext_late.close()

            

            # 3. COMPUTE

            rec_rv = compute_pr(rec_v_l) / (compute_pr(rec_v_e) + 1e-8)

            base_rv = compute_pr(base_v_l) / (compute_pr(base_v_e) + 1e-8)

            

            results.append({

                "layer": layer,

                "rec_rv": rec_rv,

                "base_rv": base_rv,

                "diff": rec_rv - base_rv

            })

            

        if i % 5 == 0: print(".", end="", flush=True)



    # ANALYSIS

    df = pd.DataFrame(results)

    print("\n\n" + "="*60)

    for layer in CONFIG['layers_to_test']:

        d_layer = df[df['layer'] == layer]

        if len(d_layer) == 0: continue

        

        mean_rec = d_layer['rec_rv'].mean()

        mean_base = d_layer['base_rv'].mean()

        diff = mean_rec - mean_base

        d = diff / (d_layer['diff'].std() + 1e-9)

        

        print(f"\nLAYER {layer}:")

        print(f"  Rec R_V:  {mean_rec:.4f}")

        print(f"  Base R_V: {mean_base:.4f}")

        print(f"  Cohen's d: {d:.4f}")

        

        if abs(d) > 2.0: print("  ⭐⭐⭐ SINGULARITY RESTORED")



if __name__ == "__main__":

    run_restoration_v3()
