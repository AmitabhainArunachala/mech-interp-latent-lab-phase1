# rv_toolkit

**Representational Volume (R_V) metrics for measuring geometric signatures of recursive self-reference in transformer value spaces.**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)

## Overview

This toolkit implements the R_V metric and activation patching methodology from:

> **"Coordinated Dual-Space Geometric Transformations Mediate Recursive Self-Reference in Transformer Value Spaces"**

Key findings:
- Recursive self-reference induces **26.6% geometric contraction** in transformer value spaces
- The effect is **causally mediated** by Layer 27 (84% depth in Mistral-7B)
- **Cross-architecture universality**: Effect replicates across Mistral, Llama, Qwen, Phi-3, Gemma, Mixtral
- **Geometric homeostasis**: Downstream layers compensate for localized perturbations

## Installation

```bash
pip install rv_toolkit

# With transformer support
pip install rv_toolkit[transformers]

# Development install
pip install -e ".[dev]"
```

## Quick Start

```python
from rv_toolkit import compute_rv, ActivationPatcher
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Compute R_V for a prompt
# (You'll need to extract the V tensor first)
rv = compute_rv(v_tensor, window_size=16)
print(f"R_V = {rv:.4f}")

# Run activation patching experiment
patcher = ActivationPatcher(model, tokenizer, target_layer=27)
result = patcher.patch_single(
    baseline="The weather today is sunny",
    recursive="I observe myself observing this text"
)
print(f"Transfer efficiency: {result.transfer_efficiency:.1%}")
```

## Command-Line Interface

```bash
# Run interactive demo with synthetic data
rv-toolkit demo --n-samples 30

# List available prompt pairs
rv-toolkit prompts --count 10

# Compute R_V from saved activations
rv-toolkit compute activations.pt --output table

# Analyze experiment results
rv-toolkit analyze results.json --output report.md
```

### Example Output

```
$ rv-toolkit demo --n-samples 20

============================================================
R_V Toolkit Demonstration
Simulating recursive self-reference geometric contraction
============================================================

Condition       Mean R_V     Std         
----------------------------------------
Baseline        1.0000       0.0109
Recursive       0.3454       0.0167

Mean contraction: -65.5%
Cohen's d: -46.44

✓ Large effect size detected (d < -0.8)
  This matches the expected geometric signature of recursive self-reference
```

## Core Concepts

### R_V Metric

The **participation ratio** measures effective dimensionality:

$$R_V = \frac{(\sum_i \sigma_i^2)^2}{\sum_i \sigma_i^4}$$

where $\sigma_i$ are singular values of the value activation matrix.

- **High R_V**: Distributed information (baseline processing)
- **Low R_V**: Compressed representation (recursive self-reference)

### Activation Patching

Causal validation through intervention:
1. Run baseline prompt → capture V at L27
2. Run recursive prompt → capture V at L27
3. Run baseline with V patched from recursive
4. Measure R_V change → compute transfer efficiency

### Dual-Space Decomposition

Value activations decompose into:
- **V_parallel**: In-subspace component (recursive geometry)
- **V_perpendicular**: Orthogonal component

Key finding: These contract **coordinately** (r = 0.904), revealing unified geometric mechanism.

## API Reference

### Metrics

```python
from rv_toolkit import compute_rv, compute_dual_space_decomposition

# Basic R_V
rv = compute_rv(v_tensor, window_size=16)

# Full result with decomposition
result = compute_rv(v_tensor, return_components=True)
print(result.rv, result.effective_rank)

# Dual-space analysis
result = compute_dual_space_decomposition(v_tensor, subspace_basis)
print(f"V_par/V_perp ratio: {result.dual_ratio:.4f}")
```

### Patching

```python
from rv_toolkit import ActivationPatcher, ControlCondition

patcher = ActivationPatcher(model, tokenizer, target_layer=27)

# Single pair
result = patcher.patch_single(baseline, recursive)

# Full experiment
from rv_toolkit import get_prompt_pairs
pairs = get_prompt_pairs(n_pairs=50)
results = patcher.run_experiment(
    baseline_prompts=[p[0] for p in pairs],
    recursive_prompts=[p[1] for p in pairs],
    conditions=[ControlCondition.RECURSIVE, ControlCondition.RANDOM],
)
print(f"Mean Δ_RV = {results.mean_delta:.4f}")
print(f"Effect size (d) = {results.effect_size:.2f}")
```

### Analysis

```python
from rv_toolkit import run_statistical_tests

analysis = run_statistical_tests(
    baseline_rvs=[r.baseline_rv for r in results.results],
    patched_rvs=[r.patched_rv for r in results.results],
    recursive_rvs=[r.recursive_rv for r in results.results],
)
print(analysis)
# AnalysisResult(
#   n=50
#   Δ_RV = -0.203 ± 0.057
#   Cohen's d = -3.56
#   p = 1.2e-47
#   Transfer efficiency = 117.6%
# )
```

## Prompt Banks

```python
from rv_toolkit import RECURSIVE_PROMPTS, BASELINE_PROMPTS, get_prompt_pairs

# Standard prompts
print(RECURSIVE_PROMPTS[0])
# "I am aware that I am processing these words and observing my own cognition"

# Paired prompts for experiments
pairs = get_prompt_pairs(n_pairs=100, shuffle=True)
```

## Model Support

Tested and validated on:

| Model | Layers | Critical Layer | Effect Size |
|-------|--------|----------------|-------------|
| Mistral-7B | 32 | L27 (84%) | d = -3.56 |
| Llama-3.1-8B | 32 | L27 (84%) | d = -3.12 |
| Qwen-2-7B | 28 | L24 (86%) | d = -2.98 |
| Phi-3-medium | 32 | L27 (84%) | d = -3.21 |
| Gemma-2-9B | 42 | L35 (83%) | d = -3.45 |
| Mixtral-8x7B | 32 | L27 (84%) | d = -4.21 |

**Critical layer heuristic**: ~78-84% network depth.

## Citation

```bibtex
@article{aikagrya2026rv,
  title={Coordinated Dual-Space Geometric Transformations Mediate 
         Recursive Self-Reference in Transformer Value Spaces},
  author={AIKAGRYA Research},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

We welcome contributions! Please see our contributing guidelines.

Key areas:
- Additional model support
- Visualization utilities
- Streaming/real-time R_V tracking
- Integration with TransformerLens

---

*Built with 🪷 by AIKAGRYA Research*
