# Cross-Architecture R_V Discovery Pipeline

**Version**: 2026-01-15
**Status**: Production-ready
**Prerequisites**: GPU with ≥24GB VRAM, HuggingFace access

---

## Overview

This document is the **single source of truth** for running R_V circuit discovery on new model architectures. Follow it exactly. Do not improvise.

The goal: Identify whether the R_V contraction circuit (discovered on Mistral-7B) generalizes to other architectures, and map any differences.

---

## Reference Circuit (Mistral-7B Validated)

```
L0 MLP (source) → L3-L4 MLP (transfer) → L27 Attention H18+H26 (readout)
                                              ↓
                                         R_V < 1.0

Key metrics:
- Champions R_V: 0.52 ± 0.05
- Controls R_V: 1.01 ± 0.06
- Cohen's d: -3.56
- p-value: < 10⁻⁶
```

Your job is to find the equivalent circuit (or document differences) on the target model.

---

## Critical Invariants (DO NOT CHANGE)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Prompt bank | `prompts/` directory | Versioned, validated |
| Early layer | 5 | After initial embedding processing |
| Window size | 16 | Last 16 tokens of prompt |
| Late layer | 84% of depth | L27 for 32-layer, L21 for 25-layer, etc. |
| Seed | 42 | Reproducibility |
| Min n per group | 40 | Statistical power |

**These are NOT suggestions. They are requirements.**

---

## Phase 0: Environment Setup

### 0.1 Verify Repository State

```bash
cd /path/to/mech-interp-latent-lab-phase1
git status  # Should be clean or on a feature branch
```

### 0.2 Verify Prompt Bank

```bash
python -c "from prompts.loader import PromptLoader; l = PromptLoader(); print(f'Version: {l.version}, Groups: {list(l.groups.keys())}')"
```

Expected output includes: `champions`, `L5_refined`, `L4_full`, `L3_deeper`, `baseline_*`, `length_matched`, `pseudo_recursive`

### 0.3 Verify Metrics

```bash
python -c "from src.metrics import compute_rv, compute_extended_metrics; print('Metrics OK')"
```

---

## Phase 1: Generate Model Configs

### 1.1 Auto-Generate Configs

```bash
python scripts/generate_model_configs.py \
  --model [HF_MODEL_NAME] \
  --output-dir configs/discovery
```

**Example:**
```bash
python scripts/generate_model_configs.py --model meta-llama/Meta-Llama-3-8B
python scripts/generate_model_configs.py --model mistralai/Mixtral-8x7B-v0.1
python scripts/generate_model_configs.py --model Qwen/Qwen2-7B
```

### 1.2 Verify Generated Configs

```bash
ls configs/discovery/[MODEL_SHORT]/
# Should see: 01_baseline_rv.json, 02_source_hunt_*.json, 03_transfer_hunt_*.json, etc.
```

### 1.3 Check Registry Compatibility

```bash
python -c "
from src.utils import validate_registry_compatibility
import json
from pathlib import Path

configs = [json.load(open(f)) for f in Path('configs/discovery/[MODEL_SHORT]').glob('*.json')]
missing = validate_registry_compatibility(configs)
print('Missing experiments:', missing if missing else 'NONE - ALL REGISTERED')
"
```

**STOP if any experiments are missing from registry.**

---

## Phase 2: Baseline R_V Separation

This is the **gate phase**. If this fails, the circuit does not generalize to this model.

### 2.1 Run Baseline

```bash
python -m src.pipelines.run \
  --config configs/discovery/[MODEL_SHORT]/01_baseline_rv.json
```

### 2.2 Success Criteria (ALL must pass)

| Criterion | Threshold | Why |
|-----------|-----------|-----|
| Champions R_V | < 0.75 | Must show contraction |
| Controls R_V | > 0.85 | Must NOT show contraction |
| Cohen's d | > 1.0 (absolute) | Meaningful effect size |
| p-value | < 0.01 | Statistical significance |
| n (champions) | ≥ 40 | Power |
| n (controls) | ≥ 40 | Power |

### 2.3 Check Results

```bash
cat results/phase2_generalization/[MODEL_SHORT]/*/summary.json | python -m json.tool
```

Look for:
```json
{
  "mean_rv": {
    "champions": 0.XX,  // Must be < 0.75
    "length_matched": 0.XX  // Must be > 0.85
  },
  "ttest": {
    "champions_vs_length_matched": {
      "cohens_d": -X.XX,  // Must be < -1.0
      "p": 0.00XXX  // Must be < 0.01
    }
  }
}
```

### 2.4 Decision Gate

| Result | Action |
|--------|--------|
| ALL criteria pass | Proceed to Phase 3 |
| Weak effect (0.5 < d < 1.0) | Proceed with caution, note in report |
| No effect (d < 0.5) | **STOP** - Circuit does not generalize |
| Wrong direction (champions > controls) | **STOP** - Investigate data quality |

---

## Phase 3: Source Layer Hunt (MLP Ablation)

Goal: Find which early-layer MLP is **necessary** for contraction.

### 3.1 Run Ablation Sweep

```bash
for layer in 0 1 2 3 4 5 6 7 8; do
  echo "=== Ablating Layer $layer ==="
  python -m src.pipelines.run \
    --config configs/discovery/[MODEL_SHORT]/02_source_hunt_mlp_ablation_l${layer}.json
done
```

### 3.2 Success Criteria

A layer is the **SOURCE** if ablating it:
- Increases R_V by > 0.3 (removes contraction)
- Shows statistical significance (p < 0.01)

### 3.3 Expected Pattern

```
Layer | R_V (ablated) | Delta | Role
------|---------------|-------|-----
L0    | ~1.5-2.0      | +0.8+ | PRIMARY SOURCE (expected)
L1    | ~1.2-1.5      | +0.5  | Secondary
L2-L4 | ~0.9-1.1      | +0.2  | Minor
L5+   | ~0.7-0.8      | ~0    | No effect
```

### 3.4 Record Findings

Note which layer(s) show the largest ablation effect. This is your SOURCE.

---

## Phase 4: Transfer Layer Hunt (MLP Steering)

Goal: Find which layer's MLP can **induce** contraction via steering.

### 4.1 Run Steering Sweep

```bash
for layer in 0 1 2 3 4 5 6 7 8 9 10; do
  echo "=== Steering Layer $layer ==="
  python -m src.pipelines.run \
    --config configs/discovery/[MODEL_SHORT]/03_transfer_hunt_mlp_steer_l${layer}.json
done
```

### 4.2 Success Criteria

A layer is the **TRANSFER POINT** if steering it:
- Decreases R_V (induces contraction on baseline prompts)
- Random direction control shows NO effect
- Effect is statistically significant

### 4.3 Expected Pattern

```
Layer | R_V (steered) | Delta | Effect
------|---------------|-------|-------
L0-L2 | variable      | ??    | May cause artifacts
L3-L4 | ~0.7-0.8      | -0.2  | TRANSFER POINT (expected)
L5+   | ~0.95         | ~0    | No effect
```

### 4.4 Handle Anomalies

If steering causes EXPANSION (R_V increases):
1. This is a real finding - document it
2. Try V-projection steering instead of residual stream
3. Check if model uses different gating (SiLU vs GELU)

---

## Phase 5: Readout Validation

Goal: Confirm the late-layer readout mechanism.

### 5.1 Run Validation

```bash
python -m src.pipelines.run \
  --config configs/discovery/[MODEL_SHORT]/04_readout_validation.json
```

### 5.2 Four-Way Control

The pipeline runs four conditions:
1. **Target patch** (L27 → L27): Should transfer R_V
2. **Wrong layer patch** (L21 → L27): Should NOT transfer
3. **Random patch**: Should NOT transfer
4. **Shuffled patch**: Should NOT transfer

### 5.3 Success Criteria

- Target patch transfer efficiency > 50%
- All controls < 20% transfer
- Statistical separation between target and controls

---

## Phase 6: Extended Metrics (Publication-Grade)

For EVERY successful run, compute extended metrics.

### 6.1 Compute Extended Metrics

```python
from src.metrics import compute_extended_metrics
from src.core.models import load_model

model, tokenizer = load_model("[HF_MODEL_NAME]")

# For each prompt type
for prompt in prompts:
    ext = compute_extended_metrics(
        model, tokenizer, prompt,
        early_layer=5, late_layer=27, window_size=16
    )

    # Record these values:
    # - cosine_early_late (directional alignment)
    # - spectral_late_top1_ratio (dominance of first SV)
    # - spectral_late_spectral_gap (σ₁ - σ₂)
    # - attention_entropy (focus at readout)
```

### 6.2 Extended Metrics Hypotheses

| Metric | Champions (expected) | Controls (expected) |
|--------|---------------------|---------------------|
| cosine_early_late | > 0.8 (converging) | < 0.6 (diverging) |
| spectral_late_top1_ratio | > 0.5 (one dominant direction) | < 0.3 (distributed) |
| attention_entropy | Lower (focused) | Higher (diffuse) |

---

## Output Requirements

### Required Artifacts

Create `results/phase2_generalization/[MODEL_SHORT]/CIRCUIT_MAP.md` with:

```markdown
# [MODEL_NAME] R_V Circuit Map

**Date:** [DATE]
**Model:** [HF_MODEL_NAME]
**Status:** [VALIDATED / PARTIAL / NOT FOUND]

## Model Architecture
- Layers: [N]
- Heads: [N]
- Hidden dim: [N]
- GQA: [Yes/No]
- KV Heads: [N if GQA]

## Phase 2: Baseline R_V

| Group | R_V | Std | n | CI 95% |
|-------|-----|-----|---|--------|
| Champions | X.XX | X.XX | XX | [X.XX, X.XX] |
| Controls | X.XX | X.XX | XX | [X.XX, X.XX] |

**Separation:** d = X.XX, p = X.XXe-XX

## Phase 3: Source Layer

| Layer | R_V (ablated) | Delta | p-value | Role |
|-------|---------------|-------|---------|------|
| L0 | X.XX | +X.XX | X.XX | [PRIMARY/SECONDARY/NONE] |
| ... | ... | ... | ... | ... |

**Source Layer:** L[X]

## Phase 4: Transfer Layer

| Layer | R_V (steered) | Delta | p-value | Effect |
|-------|---------------|-------|---------|--------|
| L0 | X.XX | X.XX | X.XX | [TRANSFER/NONE/EXPANSION] |
| ... | ... | ... | ... | ... |

**Transfer Layer:** L[X] (or NOT FOUND)

## Phase 5: Readout Validation

| Condition | Transfer % | p-value |
|-----------|------------|---------|
| Target (L27) | XX% | X.XX |
| Wrong layer | XX% | X.XX |
| Random | XX% | X.XX |
| Shuffled | XX% | X.XX |

## Extended Metrics

| Metric | Champions | Controls | Delta |
|--------|-----------|----------|-------|
| cosine_early_late | X.XX | X.XX | X.XX |
| spectral_top1_ratio | X.XX | X.XX | X.XX |
| attention_entropy | X.XX | X.XX | X.XX |

## Circuit Comparison

| Component | Mistral-7B | [THIS MODEL] | Match? |
|-----------|------------|--------------|--------|
| Source | L0 MLP | L[X] MLP | [YES/NO] |
| Transfer | L3-L4 MLP | L[X] MLP | [YES/NO] |
| Readout | L27 | L[X] | [YES/NO] |

## Conclusions

[Summary of findings]

## Raw Data Paths

- Phase 2: `results/phase2_generalization/[MODEL_SHORT]/[TIMESTAMP]_cross_architecture_validation/`
- Phase 3: `results/phase2_generalization/[MODEL_SHORT]/[TIMESTAMP]_mlp_ablation_necessity/`
- Phase 4: `results/phase2_generalization/[MODEL_SHORT]/[TIMESTAMP]_combined_mlp_sufficiency/`
- Phase 5: `results/phase2_generalization/[MODEL_SHORT]/[TIMESTAMP]_rv_l27_causal_validation/`
```

---

## DO NOT

| Do Not | Why |
|--------|-----|
| Change the prompt bank | Breaks comparability |
| Skip phases | Incomplete validation |
| Report n < 30 | Underpowered |
| Use different layer indices | Breaks protocol |
| Make up numbers | Scientific fraud |
| Proceed after Phase 2 failure | Wasted compute |
| Ignore anomalies | Miss real findings |

---

## Troubleshooting

### "Model won't load"
- Check VRAM (need ≥24GB for 7B models with hooks)
- Try `device_map="auto"` for larger models
- Use `torch_dtype=torch.float16`

### "Low sample sizes"
- Check for tokenization errors (prompts too long)
- Check for OOM during batch processing
- Run with smaller batches: add `--batch-size 8` if supported

### "Registry experiment not found"
- Run `validate_registry_compatibility()` first
- Check experiment name matches registry exactly
- May need to add pipeline to registry

### "Steering causes expansion"
- This may be a real finding (architecture difference)
- Try V-projection steering instead
- Check if using correct intervention (MLP output, not input)

### "Pseudo-recursive also contracts"
- This is the "Llama Anomaly" - document it
- May indicate different training data distribution
- Still valid finding - note the lack of separation

---

## Quick Reference: One-Liner Commands

```bash
# Generate configs for new model
python scripts/generate_model_configs.py --model [HF_MODEL]

# Run Phase 2 (baseline)
python -m src.pipelines.run --config configs/discovery/[SHORT]/01_baseline_rv.json

# Run Phase 3 (ablation sweep)
for i in {0..8}; do python -m src.pipelines.run --config configs/discovery/[SHORT]/02_source_hunt_mlp_ablation_l${i}.json; done

# Run Phase 4 (steering sweep)
for i in {0..10}; do python -m src.pipelines.run --config configs/discovery/[SHORT]/03_transfer_hunt_mlp_steer_l${i}.json; done

# Run Phase 5 (readout validation)
python -m src.pipelines.run --config configs/discovery/[SHORT]/04_readout_validation.json
```

---

## Checklist Before Declaring "Done"

- [ ] Phase 2 passed gate criteria
- [ ] Phase 3 identified source layer
- [ ] Phase 4 identified transfer layer (or documented anomaly)
- [ ] Phase 5 validated readout
- [ ] Extended metrics computed
- [ ] CIRCUIT_MAP.md created with all tables filled
- [ ] All raw data paths documented
- [ ] Comparison to Mistral reference included
- [ ] Any anomalies explicitly documented

---

*This protocol is designed for reproducibility. Follow it exactly.*
