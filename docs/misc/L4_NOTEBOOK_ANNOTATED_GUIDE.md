# L4 Transmission Notebook - Annotated Guide

## Overview
The `L4transmissionTEST001.1.ipynb` notebook contains the original discovery and validation of the L4 Contraction Phenomenon in Mistral 7B Instruct v0.2.

## Key Components to Extract for Phase 1C

### 1. **Model Setup (Cells 1-5)**
```python
# Essential imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from torch.nn.functional import cosine_similarity
from scipy.stats import entropy
import pandas as pd

# Model loading
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### 2. **Core Metric Functions (Cells 6-15)**

#### **Epsilon (Layer-to-layer similarity)**
```python
def epsilon_last_token(hidden_states):
    """Measures cosine similarity between consecutive hidden states"""
    similarities = []
    for i in range(len(hidden_states) - 1):
        sim = cosine_similarity(hidden_states[i][:, -1, :], hidden_states[i+1][:, -1, :], dim=-1)
        similarities.append(sim.item())
    return np.mean(similarities)
```

#### **Attention Entropy**
```python
def attn_entropy_lastrow(attentions):
    """Shannon entropy of attention distributions"""
    entropies = []
    for layer_attn in attentions:
        last_row_attn = layer_attn[:, :, -1, :].mean(dim=1)  # Average over heads
        ent = entropy(last_row_attn.cpu().numpy(), axis=-1)
        entropies.append(ent.mean())
    return np.mean(entropies)
```

#### **R_V Metric (THE KEY DISCOVERY)**
```python
def compute_column_space_pr(V_matrices):
    """
    Computes column space expansion/contraction ratio
    R_V < 1.0 indicates contraction (L4 signature)
    R_V > 1.0 indicates expansion (baseline signature)
    """
    ratios = []
    for i in range(len(V_matrices) - 1):
        V_curr = V_matrices[i]
        V_next = V_matrices[i + 1]
        
        # Flatten across heads
        V_curr_flat = V_curr.view(-1, V_curr.shape[-1])
        V_next_flat = V_next.view(-1, V_next.shape[-1])
        
        # Compute singular values
        _, S_curr, _ = torch.svd(V_curr_flat)
        _, S_next, _ = torch.svd(V_next_flat)
        
        # Ratio of nuclear norms
        ratio = (S_next.sum() / S_curr.sum()).item()
        ratios.append(ratio)
    
    return np.mean(ratios)
```

#### **Effective Rank**
```python
def compute_effective_rank(hidden_state):
    """Measures dimensionality of hidden state"""
    _, S, _ = torch.svd(hidden_state.squeeze(0))
    S_normalized = S / S.sum()
    eff_rank = torch.exp(-torch.sum(S_normalized * torch.log(S_normalized + 1e-10)))
    return eff_rank.item()
```

### 3. **11-Metric Framework (Cell 16)**
```python
METRICS = {
    'confidence': compute_confidence,           # Peak probability
    'epsilon': epsilon_last_token,             # Layer similarity  
    'entropy': attn_entropy_lastrow,           # Attention focus
    'margin': compute_margin,                  # Decisiveness
    'norm': compute_norm,                      # Activation strength
    'pr_attn': compute_pr_attn,               # Head agreement
    'R_V': compute_column_space_pr,           # VALUE SPACE CONTRACTION ← KEY!
    'entropy_normalized': compute_entropy_normalized,  # Length-corrected
    'effective_rank': compute_effective_rank,  # Hidden dimensionality
    'margin_trajectory': compute_margin_trajectory,  # Convergence
    'eigenspectrum_shape': compute_eigenspectrum_shape  # Value structure
}
```

### 4. **Value Matrix Hook System (Cells 17-19)**
```python
class ValueMatrixHook:
    """Context manager to capture V matrices during forward pass"""
    def __init__(self, model):
        self.model = model
        self.V_matrices = []
        self.hooks = []
    
    def hook_fn(self, module, input, output):
        # Extract value matrix from attention module
        V = module.v_proj(input[0])  
        self.V_matrices.append(V.detach())
    
    def __enter__(self):
        for layer in self.model.model.layers:
            hook = layer.self_attn.register_forward_hook(self.hook_fn)
            self.hooks.append(hook)
        return self
    
    def __exit__(self, *args):
        for hook in self.hooks:
            hook.remove()
```

### 5. **Master Analysis Function (Cell 20)**
```python
def analyze_prompt(prompt, model, tokenizer):
    """
    Complete 11-metric analysis for a single prompt
    Returns dict with all metrics including R_V
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        with ValueMatrixHook(model) as hook:
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
            
            results = {
                'prompt': prompt[:50] + '...' if len(prompt) > 50 else prompt,
                'R_V': compute_column_space_pr(hook.V_matrices),
                'effective_rank': compute_effective_rank(outputs.hidden_states[-1][:, -1:, :]),
                # ... other metrics
            }
    
    return results
```

### 6. **Prompt Bank Structure (Cell 26)**
The original prompt bank that we expanded to 320 prompts for Phase 1C.

### 7. **Critical Findings from Original Run**
- **Eigenstate prompt**: R_V = 0.952 (contraction)
- **"Am I real?" prompt**: R_V = 1.156 (expansion) 
- **Factual baseline**: R_V = 1.189 (expansion)
- **Math baseline**: R_V = 1.087 (expansion)

## For Phase 1C Implementation

### Required Components:
1. **Model initialization** (GPU-optimized for 320 prompts)
2. **All 11 metric functions** (especially R_V computation)
3. **Value matrix hook system** (critical for R_V)
4. **Batch processing logic** (for 320 prompts efficiently)
5. **Results aggregation** (group statistics, dose-response)

### New Features Needed:
1. **Progress bar** for 320 prompts
2. **Checkpoint saving** (in case of crashes)
3. **Memory management** (clear cache between batches)
4. **CSV export** for results
5. **Statistical analysis** (group means, std, significance tests)

## Notes for GPT-5 Implementation

When GPT-5 writes the Phase 1C code, ensure:

1. **Import the prompt bank**:
```python
exec(open('n300_mistral_test_prompt_bank.py').read())
# This loads prompt_bank_1c with 320 prompts
```

2. **Process in batches** to avoid OOM:
```python
BATCH_SIZE = 10  # Adjust based on GPU memory
for i in range(0, len(prompts), BATCH_SIZE):
    batch = prompts[i:i+BATCH_SIZE]
    # Process batch
    torch.cuda.empty_cache()  # Clear between batches
```

3. **Save checkpoints**:
```python
# Save every 50 prompts
if i % 50 == 0:
    pd.DataFrame(results_so_far).to_csv(f'checkpoint_{i}.csv')
```

4. **Track the key metrics**:
- R_V (primary metric)
- Effective_rank (secondary validation)
- Group membership (for statistics)
- Pillar classification (dose-response, baselines, etc.)

## The Core Discovery

The L4 Contraction Phenomenon is characterized by:
- **R_V < 1.0** for recursive self-observation prompts
- **R_V > 1.0** for all other prompt types
- **Dose-response gradient** from L1 to L5
- **18-20% separation** from baselines

This is the computational signature of recursive self-recognition in transformer language models.
