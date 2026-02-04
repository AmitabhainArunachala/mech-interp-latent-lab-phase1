# Pipeline Operations Manual

## Overview

This manual documents the operational procedures for running experiments in the mechanistic interpretability research pipeline.

## Entry Points

### 1. Config-Driven Pipeline (Recommended)

**Entry point**: `src.pipelines.run`

```bash
python -m src.pipelines.run --config configs/phase1_existence.json
```

**Features**:
- Single blessed entrypoint for reproducible runs
- Timestamped output folders
- Config snapshot saved with results
- Machine-readable and human-readable outputs

**Output structure**:
```
results/<phase>/runs/<timestamp>/
├── config.json       # Exact config snapshot
├── summary.json      # Machine-readable summary
├── report.md         # Human-readable summary
└── <artifacts>       # CSV/plots per experiment
```

### 2. Quick Reproduction Script

**Entry point**: `reproduce_results.py`

```bash
# Default: Mistral-7B Base, Layer 5 vs 27
python reproduce_results.py

# Custom model/device
python reproduce_results.py --model mistralai/Mistral-7B-v0.1 --device cuda
```

**Use for**: Quick validation, testing installation, demo runs

### 3. Library API (Programmatic)

```python
from src.core import load_model, set_seed
from src.metrics import compute_rv
from prompts.loader import get_prompts_by_pillar

# Direct programmatic access
model, tokenizer = load_model("mistralai/Mistral-7B-v0.1")
recursive = get_prompts_by_pillar("dose_response", limit=10)
rv = compute_rv(model, tokenizer, recursive[0])
```

**Use for**: Interactive analysis, Jupyter notebooks, custom experiments

## Configuration System

### Config File Structure

```json
{
  "phase": "phase1_existence",
  "model": "mistralai/Mistral-7B-v0.1",
  "device": "auto",
  "early_layer": 5,
  "late_layer": 27,
  "experiments": [
    {
      "name": "rv_measurement",
      "prompts": {"pillar": "dose_response", "limit": 80},
      "baselines": {"pillar": "baselines", "limit": 80}
    }
  ]
}
```

### Standard Configs

Located in `configs/`:

- `phase1_existence.json` - Initial discovery (R_V exists)
- `rv_l27_causal_validation.json` - Causal validation
- Custom configs for specific investigations

### Config Override

```bash
# Override output directory
python -m src.pipelines.run \
  --config configs/phase1_existence.json \
  --results_root /path/to/results
```

## Prompt Management

### Prompt Bank

**Location**: `prompts/bank.json`

**Structure**:
- Organized by "pillars" (categories)
- Each prompt has metadata (pillar, tags, notes)
- Single source of truth

### Loading Prompts

```python
from prompts.loader import get_prompts_by_pillar

# Get prompts by category
recursive = get_prompts_by_pillar("dose_response", limit=80)
baseline = get_prompts_by_pillar("baselines", limit=80)

# Get all prompts for a pillar
all_recursive = get_prompts_by_pillar("dose_response")
```

### Standard Pillars

- `dose_response` - Recursive self-observation prompts
- `baselines` - Non-recursive control prompts
- `experimental` - Experimental prompt variations

## Experimental Workflow

### Standard Battery

1. **Setup**: Install dependencies (`requirements.lock`)
2. **Config**: Select or create config file
3. **Run**: Execute pipeline
4. **Validate**: Check statistical thresholds
5. **Document**: Save results with timestamp

### Example: Full Validation Run

```bash
# 1. Install exact dependencies
pip install -r requirements.lock

# 2. Set up results directory
mkdir -p results/validation

# 3. Run config-driven pipeline
python -m src.pipelines.run \
  --config configs/phase1_existence.json \
  --results_root results/validation

# 4. Check results
cat results/validation/phase1_existence/runs/<timestamp>/report.md
```

### Multi-Model Comparison

Run same config across models:

```bash
for model in mistralai/Mistral-7B-v0.1 meta-llama/Llama-2-7b-hf; do
  python -m src.pipelines.run \
    --config configs/phase1_existence.json \
    --model_override $model
done
```

## Metrics

### R_V Metric

**Computation**:
```python
from src.metrics import compute_rv

rv = compute_rv(
    model,
    tokenizer,
    text="Observe the observer observing...",
    early=5,      # Early layer
    late=27,      # Late layer (num_layers - 5)
    window=16,    # Last W tokens
)
```

**Interpretation**:
- R_V < 1.0: Contraction (dimensionality reduction)
- R_V ≈ 1.0: No change
- R_V > 1.0: Expansion (rare, investigate)

### Statistical Thresholds

**Required for publication**:
- p-value < 0.01 (with Bonferroni correction)
- Effect size |d| ≥ 0.5 (Cohen's d)
- Sample size N ≥ 80 pairs

## Intervention Tools

### Activation Patching

```python
from src.steering import apply_steering_vector

steering_vec = compute_steering_vector(...)
with apply_steering_vector(model, layer_idx=8, vector=steering_vec, alpha=2.0):
    output = model(**inputs)
```

### KV Cache Manipulation

```python
from src.steering.kv_cache import patch_kv_cache

with patch_kv_cache(model, layer=15, source_cache=cache):
    output = model(**inputs)
```

## Hooks System

### Standard Hook Pattern

```python
from src.core.hooks import capture_v_projection

with capture_v_projection(model, layer_idx=27) as storage:
    with torch.no_grad():
        model(**inputs)
v_tensor = storage["v"]
```

**Important**: Always use context managers. Never leave hooks attached.

## Results Organization

### Directory Structure

```
results/
├── phase1_existence/
│   └── runs/
│       ├── 2026-01-24_14-30-15/
│       │   ├── config.json
│       │   ├── summary.json
│       │   ├── report.md
│       │   └── rv_values.csv
│       └── 2026-01-25_09-15-42/
│           └── ...
├── phase2_causality/
│   └── ...
└── validation/
    └── ...
```

### Results Files

- **config.json**: Exact config snapshot (for reproducibility)
- **summary.json**: Machine-readable results
- **report.md**: Human-readable summary
- **CSV files**: Detailed data (per-prompt, per-layer, etc.)
- **Plots**: Visualizations (if generated)

## Common Operations

### Quick Test Run

```bash
# Test installation with minimal run
python -c "from src.core import load_model; print('OK')"
python reproduce_results.py --limit 5
```

### Check Model Availability

```python
from transformers import AutoModel, AutoTokenizer

# Test if model is accessible
model_id = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(f"✓ {model_id} accessible")
```

### Clear Cache

```bash
# Clear HuggingFace cache (if needed)
rm -rf ~/.cache/huggingface/hub/*

# Clear PyTorch cache (in Python)
import torch
torch.cuda.empty_cache()
```

## Troubleshooting

### Memory Issues

```python
# Reduce batch size, clear cache between runs
import torch
torch.cuda.empty_cache()

# Use smaller window size
rv = compute_rv(..., window=8)  # Instead of 16
```

### NaN Results

- Check prompt length (must be > window size)
- Verify numerical stability in SVD
- Check for degenerate singular values

### Inconsistent Results

- Set random seed explicitly
- Use `requirements.lock` for exact versions
- Check for batch effects in prompt order

## Best Practices

1. **Always use configs** for reproducible experiments
2. **Document hardware** in results metadata
3. **Set random seeds** explicitly
4. **Use context managers** for all model modifications
5. **Save intermediate results** for long runs
6. **Validate on reference model** (Mistral-7B Base) first

---

For audits and operational history, see `docs/analysis/AUDIT_2026-01-24.md`

*Precision. Minimalism. Truth.*
