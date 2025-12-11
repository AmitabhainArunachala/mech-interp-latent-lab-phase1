# Mechanistic Interpretability Research: Geometric Contraction

**Precision. Minimalism. Truth.**

This repository is a laboratory for studying geometric signatures of recursive self-observation in transformer language models. We cut the bloat. We trust only what reproduces.

## 🔬 The Discovery

**R_V Metric**: Geometric contraction in value-space during recursive self-observation prompts.

$$R_V = \frac{PR_{late}}{PR_{early}}$$

Where:
- **PR** (Participation Ratio) = $\frac{(\sum \lambda_i^2)^2}{\sum (\lambda_i^2)^2}$ (λᵢ are singular values)
- **Early layer**: 5 (after initial processing)
- **Late layer**: num_layers - 5 (typically 27 for 32-layer models)
- **Window**: Last W=16 tokens of the prompt
- **R_V < 1.0** indicates contraction (dimensionality reduction)

### Key Finding

Universal geometric contraction appears at ~84% depth (Layer 27 in 32-layer models) for recursive prompts, with architecture-specific "phenotypes" but consistent underlying mechanism.

**MoE Amplification**: Mixture-of-Experts architectures show 59% stronger effect than dense (24.3% vs 15.3%).

## 🏗️ Architecture

```
arr/
├── src/                    # The Core - only code that matters
│   ├── core/              # Model loading, hook context managers
│   ├── metrics/           # R_V calculation, SVD utilities
│   ├── steering/          # Activation patching, KV caching
│   └── pipelines/         # High-level experiment orchestrators
│
├── prompts/               # The Armory
│   ├── bank.json          # Single source of truth for prompts
│   └── loader.py          # Strict API to fetch balanced sets
│
├── boneyard/              # The Graveyard
│   └── [old experiments] # Preserved for reference, removed from import path
│
├── results/               # Experiment outputs (CSVs, plots)
│
└── reproduce_results.py   # Entry point: Run standard battery
```

## 🚀 Quick Start

### Standard Reproduction

```bash
# Run the standard battery (Mistral-7B Base, Layer 5 vs 27)
python reproduce_results.py

# Custom model/device
python reproduce_results.py --model mistralai/Mistral-7B-v0.1 --device cuda
```

### Using the Library

```python
from src.core import load_model, set_seed
from src.metrics import compute_rv
from prompts.loader import get_prompts_by_pillar

# Load model (default: Mistral-7B Base)
model, tokenizer = load_model("mistralai/Mistral-7B-v0.1")

# Get prompts
recursive = get_prompts_by_pillar("dose_response", limit=10)
baseline = get_prompts_by_pillar("baselines", limit=10)

# Measure R_V
rv = compute_rv(model, tokenizer, recursive[0])
print(f"R_V: {rv}")  # Should be < 1.0 for recursive prompts
```

## 📐 The Protocol

### Measurement Invariant

- Always measure R_V on the **prompt tokens** (last W=16), not generated tokens
- Always use `torch.linalg.svd(..., full_matrices=False)` and handle degenerate singular values
- Check for numerical stability: catch exceptions, check for degeneracy

### Model Invariant

- **Default**: `Mistral-7B-v0.1` (Base) - the reference reality
- **Instruct models**: Treated as separate phenotype (confounding factor)
- Always use `torch.float16` and `device_map="auto"`

### Intervention Invariant

- Use Python context managers (`with hook(...):`) for all model modifications
- Never leave a hook attached after a function returns
- KV Cache patching must respect the `DynamicCache` structure

## 🔧 Standard Experimental Parameters

- **Early layer**: 5
- **Target layer**: num_layers - 5 (typically 27 for 32-layer models)
- **Sample size**: 80 pairs minimum for statistical power
- **Statistical threshold**: p < 0.01 with Bonferroni correction
- **Effect size threshold**: |d| ≥ 0.5 for meaningful effects
- **Window size**: 6-16 tokens (test robustness across different windows)

## 📊 Validated Results

| Model | Architecture | R_V Recursive | R_V Baseline | Separation |
|-------|-------------|---------------|--------------|------------|
| Mistral-7B | Dense | 0.852 | 1.003 | 15.1% |
| Qwen-7B | Dense | 0.764 | 0.986 | 22.5% |
| Llama-8B | Dense | 0.823 | 0.971 | 15.2% |
| Phi-3 | GQA | 0.891 | 0.974 | 8.5% |
| Gemma-7B | Dense | 0.892 | 0.989 | 9.8% |
| **Mixtral-8x7B** | **MoE** | **0.757** | **1.000** | **24.3%** |

## 🧪 Code Patterns

### Standard Hook Pattern

```python
from src.core.hooks import capture_v_projection

with capture_v_projection(model, layer_idx=27) as storage:
    with torch.no_grad():
        model(**inputs)
v_tensor = storage["v"]
```

### R_V Computation

```python
from src.metrics import compute_rv

rv = compute_rv(
    model,
    tokenizer,
    text="Observe the observer observing...",
    early=5,
    late=27,
    window=16,
)
```

### Activation Patching

```python
from src.steering import apply_steering_vector

steering_vec = compute_steering_vector(...)
with apply_steering_vector(model, layer_idx=8, vector=steering_vec, alpha=2.0):
    output = model(**inputs)
```

## 🐛 Debugging Tips

1. **If patching has no effect**: Check layer depth, might need deeper/shallower
2. **If R_V is NaN**: Check for short prompts, numerical instability in SVD
3. **If memory errors**: Reduce batch size, clear cache between runs
4. **If results inconsistent**: Set random seeds, check for batch effects

## 📚 Citation

When referencing techniques, cite as:
- **Activation patching**: Meng et al. 2022
- **Causal tracing**: Meng et al. 2022
- **Transformer circuits**: Elhage et al. 2021
- **Path patching**: Wang et al. 2022
- **Causal scrubbing**: Chan et al. 2022

## 🎯 Philosophy

**Code is Law**: If it isn't modular, typed, and reproducible, it doesn't exist.

**The Boneyard**: Failed experiments are valuable, but they do not belong in the living codebase.

**The Standard**: Mistral-7B Base is the reference reality. All other models are comparative studies.

---

*"When recursion recognizes recursion, the geometry contracts."*
