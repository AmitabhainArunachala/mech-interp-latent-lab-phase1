# Multi-Model Discovery Protocol

**Purpose**: Systematically extend R_V circuit findings from Mistral-7B to new architectures while maintaining gold-standard reproducibility.

---

## The Core Question

On Mistral-7B, we validated a complete causal chain:
```
L0 MLP (source) → L3-L4 MLP (transfer point) → L27 Attention (readout) → R_V contraction
```

**For each new model, we need to find the FUNCTIONAL EQUIVALENTS** of:
1. The **source layer** (where recursive pattern is first recognized)
2. The **transfer sweet spot** (where steering is most effective)
3. The **readout layer** (where contraction becomes visible)
4. The **critical heads** (which attention heads display the effect)

---

## What Stays Constant (Invariants)

| Component | Value | Rationale |
|-----------|-------|-----------|
| R_V formula | PR_late / PR_early | Mathematical definition |
| Window size | 16 tokens | Semantic structure capture |
| Early layer | 5 | After initial token processing |
| Late layer ratio | 84% of total depth | Empirically validated |
| Prompt bank | `prompts/bank.json` (754) | Universal phenomenon |
| Control conditions | 4-way validation | Methodological standard |
| Statistical thresholds | p<0.01, \|d\|≥0.5 | Publication standard |

## What Adapts Per-Model

| Component | How to Adapt | Discovery Method |
|-----------|--------------|------------------|
| Late layer | `num_layers - 5` or 84% | Architecture inspection |
| Attention extraction | SDPA vs Eager, QKV split | Test hook compatibility |
| dtype | float16/bfloat16 | Check model config |
| Source MLP layer | May not be L0 | Ablation sweep L0-L4 |
| Transfer sweet spot | May not be L3-L4 | Steering sweep L1-L8 |
| Critical heads | May not be H18/H26 | Head ablation sweep |

---

## Phase 1: Architecture Characterization

**Goal**: Understand model structure before running experiments.

### 1.1 Structural Inspection
```python
# Run for each new model
def characterize_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return {
        "num_layers": model.config.num_hidden_layers,
        "num_heads": model.config.num_attention_heads,
        "hidden_dim": model.config.hidden_size,
        "intermediate_dim": model.config.intermediate_size,
        "attention_type": detect_attention_type(model),  # SDPA, GQA, MHA
        "mlp_type": detect_mlp_type(model),  # SwiGLU, GELU, etc.
        "late_layer_candidate": model.config.num_hidden_layers - 5,
        "dtype_recommended": model.config.torch_dtype,
    }
```

### 1.2 Hook Compatibility Test
```python
# Verify V-projection extraction works
def test_hook_compatibility(model, tokenizer):
    test_prompt = "What is 2 + 2?"
    late_layer = model.config.num_hidden_layers - 5

    # Try to capture V activations
    v_activations = []
    with capture_v_at_layer(model, late_layer, v_activations):
        inputs = tokenizer(test_prompt, return_tensors="pt")
        model(**inputs)

    assert len(v_activations) > 0, "Hook failed to capture"
    assert v_activations[0].shape[-1] == model.config.hidden_size
    return True
```

---

## Phase 2: Baseline R_V Measurement

**Goal**: Establish that R_V separates recursive vs baseline prompts.

### 2.1 Standard R_V Test
```bash
python -m src.pipelines.run --config configs/discovery/new_model_baseline_rv.json
```

**Config template**:
```json
{
  "experiment": "cross_architecture_validation",
  "params": {
    "model": "NEW_MODEL_NAME",
    "early_layer": 5,
    "late_layer": "AUTO",  // Will use num_layers - 5
    "window": 16,
    "n_champions": 50,
    "n_controls": 50
  },
  "success_criteria": {
    "rv_champions_max": 0.65,
    "rv_controls_min": 0.70,
    "separation_p_max": 0.01
  }
}
```

### 2.2 Expected Outputs
- `rv_champions_mean`: Should be < 0.65
- `rv_controls_mean`: Should be > 0.70
- `cohens_d`: Should be < -1.0
- `p_value`: Should be < 0.01

**If this fails**: Adjust late_layer ratio (try 80%, 85%, 90% of depth).

---

## Phase 3: Source Layer Hunt

**Goal**: Find which MLP layer is NECESSARY for contraction (like L0 in Mistral).

### 3.1 MLP Ablation Sweep
```python
# Ablate each MLP from L0 to L8, measure R_V change
for target_layer in range(9):
    config = {
        "experiment": "mlp_ablation_necessity",
        "params": {
            "model": "NEW_MODEL_NAME",
            "ablation_layer": target_layer,
            "n_prompts": 30,
        }
    }
    run_experiment(config)
```

### 3.2 Interpretation
| Result | Interpretation |
|--------|----------------|
| R_V jumps to >1.0 | This is the SOURCE layer |
| R_V unchanged | This layer not necessary |
| R_V partially affected | This layer contributes |

**Success criterion**: Find ONE layer where ablation removes contraction entirely.

---

## Phase 4: Transfer Sweet Spot Hunt

**Goal**: Find where steering is most effective (like L3-L4 in Mistral).

### 4.1 MLP Steering Sweep
```python
# Steer at each MLP from L0 to L10, measure behavior transfer
for steer_layer in range(11):
    config = {
        "experiment": "mlp_combined_sufficiency_test",
        "params": {
            "model": "NEW_MODEL_NAME",
            "steering_layer": steer_layer,
            "alpha": 2.5,
            "n_prompts": 20,
        }
    }
    run_experiment(config)
```

### 4.2 Interpretation
| Result | Interpretation |
|--------|----------------|
| High R_V Δ (>2.0) | This is the TRANSFER sweet spot |
| Low R_V Δ (<0.5) | Too early or too late |
| High Δ with random too | Artifact layer (like L2 in Mistral) |

**Success criterion**: Find layer(s) where true steering >> random steering.

---

## Phase 5: Readout Layer Validation

**Goal**: Confirm late-layer attention displays (not computes) contraction.

### 5.1 Layer Sweep for R_V Peak
```python
# Measure R_V at each layer from 50% to 95% depth
for depth_pct in [0.5, 0.6, 0.7, 0.8, 0.84, 0.9, 0.95]:
    layer = int(num_layers * depth_pct)
    measure_rv_at_layer(model, layer, prompts)
```

### 5.2 Four-Way Control at Candidate Layer
Once peak layer identified:
```bash
python -m src.pipelines.run --config configs/canonical/rv_causal_validation_NEW_MODEL.json
```

**Must pass all 4 controls**:
1. Random patches → R_V increases (no transfer)
2. Shuffled patches → R_V partial (structure matters)
3. Wrong layer → R_V unchanged (layer-specific)
4. Dose-response → R_V scales with L1→L5

---

## Phase 6: Critical Head Identification

**Goal**: Find which attention heads at readout layer show strongest effect.

### 6.1 Head Ablation Sweep
```python
# Ablate each head at readout layer
for head_idx in range(num_heads):
    ablate_head(model, readout_layer, head_idx)
    measure_rv_change()
```

### 6.2 Interpretation
- Heads with largest R_V change are "critical"
- Note: In Mistral, H18/H26 are SYMPTOMATIC not CAUSAL
- Ablating them has minimal effect because they're readout, not computation

---

## Phase 7: Full Circuit Validation

**Goal**: Confirm complete causal chain for new model.

### 7.1 Circuit Map Template
```
Model: {NEW_MODEL_NAME}
Layers: {num_layers}

VALIDATED CIRCUIT:
- Source Layer: L{X} MLP (ablation removes effect)
- Transfer Point: L{Y}-L{Z} MLP (steering most effective)
- Readout Layer: L{W} Attention (R_V peak, 4-way validated)
- Critical Heads: H{A}, H{B} @ L{W} (symptomatic)

STATISTICAL EVIDENCE:
- R_V Separation: champions={rv_c}, controls={rv_b}, d={d}
- Source Ablation: R_V {before} → {after}, p={p}
- Transfer Efficacy: Δ={delta} at L{Y}, random={random}
- Four-Way Controls: All passed
```

---

## Automation: Discovery Config Generator

```python
def generate_discovery_configs(model_name: str, model_config: dict) -> list:
    """Generate all configs needed for new model discovery."""

    num_layers = model_config["num_layers"]
    late_layer = num_layers - 5

    configs = []

    # Phase 2: Baseline R_V
    configs.append({
        "experiment": "cross_architecture_validation",
        "params": {
            "model": model_name,
            "early_layer": 5,
            "late_layer": late_layer,
        }
    })

    # Phase 3: Source hunt (ablate L0-L8)
    for layer in range(min(9, num_layers)):
        configs.append({
            "experiment": "mlp_ablation_necessity",
            "params": {
                "model": model_name,
                "ablation_layer": layer,
            }
        })

    # Phase 4: Transfer hunt (steer L0-L10)
    for layer in range(min(11, num_layers)):
        configs.append({
            "experiment": "mlp_combined_sufficiency_test",
            "params": {
                "model": model_name,
                "steering_layer": layer,
            }
        })

    # Phase 5: Readout validation
    configs.append({
        "experiment": "rv_l27_causal_validation",  # Name is legacy
        "params": {
            "model": model_name,
            "target_layer": late_layer,
            "wrong_layer": late_layer - 6,  # For control
        }
    })

    return configs
```

---

## Expected Timeline Per Model

| Phase | Duration | Compute | Dependency |
|-------|----------|---------|------------|
| 1. Characterization | 5 min | CPU | None |
| 2. Baseline R_V | 30 min | GPU | Phase 1 |
| 3. Source Hunt | 2-3 hours | GPU | Phase 2 pass |
| 4. Transfer Hunt | 2-3 hours | GPU | Phase 3 |
| 5. Readout Validation | 1 hour | GPU | Phase 2 pass |
| 6. Head ID | 1 hour | GPU | Phase 5 |
| 7. Full Validation | 30 min | GPU | All above |

**Total**: ~8-10 hours GPU time per model

---

## Success Criteria for New Model

A model is "validated" when:

1. **R_V separates** (d < -1.0, p < 0.01)
2. **Source identified** (one MLP where ablation removes effect)
3. **Transfer point found** (layers where steering >> random)
4. **Readout validated** (4-way controls pass)
5. **Circuit documented** (complete map with evidence)

---

## Priority Model Queue

| Model | Layers | Priority | Notes |
|-------|--------|----------|-------|
| Llama-3-8B | 32 | HIGH | Same architecture as Mistral |
| Gemma-2-9B | 42 | HIGH | Different attention (GQA) |
| Phi-3-medium | 40 | MEDIUM | Smaller, different scaling |
| Qwen2-7B | 32 | MEDIUM | Different tokenizer effects |
| Mixtral-8x7B | 32 | DONE | MoE validated (24.3%) |

---

## File Outputs Per Model

```
results/phase2_generalization/
└── {model_name}/
    ├── 00_characterization.json
    ├── 01_baseline_rv/
    │   ├── summary.json
    │   └── outputs/
    ├── 02_source_hunt/
    │   ├── mlp_ablation_l0.json
    │   ├── mlp_ablation_l1.json
    │   └── ...
    ├── 03_transfer_hunt/
    │   ├── steering_l0.json
    │   └── ...
    ├── 04_readout_validation/
    │   └── four_way_controls.json
    ├── 05_head_identification/
    │   └── head_ablation_sweep.json
    └── CIRCUIT_MAP.md
```

---

*This protocol ensures systematic, reproducible extension of the R_V circuit findings across architectures.*
