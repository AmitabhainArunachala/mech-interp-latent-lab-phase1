# Mechanistic Interpretability Research: Geometric Contraction

**Precision. Minimalism. Truth.**

This repository is a laboratory for studying geometric signatures of recursive self-observation in transformer language models. We cut the bloat. We trust only what reproduces.

## 🧭 Orientation (Read This First)

Start here:
- `META_TOP10_INDEX.md` (single source of truth for onboarding)

Recommended reading order (10 files):
1. `docs/standards/MEASUREMENT_CONTRACT.md`
2. `docs/status/RESEARCH_PROGRESS_SUMMARY.md`
3. `R_V_PAPER/research/PHASE1_FINAL_REPORT.md`
4. `BRIDGE_HYPOTHESIS_INVESTIGATION.md`
5. `STATISTICAL_AUDIT_EXECUTIVE_SUMMARY.md`
6. `REPRODUCIBILITY_AUDIT_REPORT.md`
7. `QUALITY_CONTROL_REPORT.md`
8. `ARCHITECTURE_EXECUTIVE_SUMMARY.md`
9. `PUBLICATION_BLOCKERS_STATUS.md`
10. `AGENT_ONBOARDING.md`

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

### Installation

This project uses a **two-file dependency system** for reproducibility:

- **`requirements.txt`**: Development mode with flexible version ranges (e.g., `torch>=2.1.0,<2.2.0`)
- **`requirements.lock`**: Exact pinned versions for bit-perfect reproduction (e.g., `torch==2.1.2`)

> **Note on `requirements.lock`**: This file pins **direct dependencies only** (torch, transformers, numpy, scipy, pandas, tqdm). Transitive dependencies (tokenizers, safetensors, etc.) are listed as comments for reference but resolve automatically. For a complete environment snapshot, run `pip freeze > full_env.txt` after installation.

```bash
# For development (allows minor updates):
pip install -r requirements.txt

# For exact reproduction (bit-perfect):
pip install -r requirements.lock

# For RunPod GPU (CUDA 12.1):
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# For M3 Pro local (MPS):
pip install -r requirements.txt  # Auto-detects MPS
```

**Hardware tested:**
- RunPod L40S (48GB VRAM, CUDA 12.1)
- M3 Pro MacBook (18GB RAM, MPS)

**Python:** 3.11+ (tested on 3.13.5)

### Standard Reproduction

```bash
# Run the standard battery (Mistral-7B Base, Layer 5 vs 27)
python reproduce_results.py

# Custom model/device
python reproduce_results.py --model mistralai/Mistral-7B-v0.1 --device cuda
```

### Reproducibility Policy

See `docs/REPRODUCIBILITY_POLICY.md` for publication-grade run requirements and
artifact expectations (config snapshot, metrics schema, prompt bank versioning, hardware + git metadata).

### Pipeline Operations

Operational runbook: `docs/PIPELINE_OPERATIONS.md`
Audit record (2026-01-24): `docs/analysis/AUDIT_2026-01-24.md`

## 🧪 Canonical Experiment Runner (Config-Driven)

The repo has a **single blessed entrypoint** for reproducible runs:

```bash
# Canonical (paper-grade) configs live under configs/canonical/
python -m src.pipelines.run --config configs/canonical/rv_l27_causal_validation.json --strict
python -m src.pipelines.run --config configs/canonical/rv_l27_activation_patching_bridge.json
python -m src.pipelines.run --config configs/canonical/rv_l27_kv_patching_bridge.json

# Multi-token bridge (behavior ↔ geometry; slower)
python -m src.pipelines.run --config configs/canonical/multi_token_bridge_mistral.json

# Historical/archived configs live under configs/archive/
python -m src.pipelines.run --config configs/archive/phase1_existence.json
```

Each run writes to a timestamped folder under `results/<phase>/runs/` containing:
- `config.json` (exact config snapshot)
- `summary.json` (machine-readable summary)
- `report.md` (human-readable summary)
- per-experiment artifacts (CSV/plots)

You can override the output root:

```bash
python -m src.pipelines.run --config configs/canonical/rv_l27_causal_validation.json --results_root results
```

### Using the Library

```python
from src.core import load_model
from src.metrics import compute_rv
from prompts.loader import PromptLoader

# Load model (default: Mistral-7B Base)
model, tokenizer = load_model("mistralai/Mistral-7B-v0.1")

# Get prompts (canonical source: prompts/bank.json)
loader = PromptLoader()
recursive = loader.get_by_group("L4_full", limit=10, seed=0)
baseline = loader.get_by_group("baseline_math", limit=10, seed=0)

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
