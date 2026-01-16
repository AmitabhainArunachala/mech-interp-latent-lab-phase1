# Cross-Architecture R_V Circuit Discovery: Step-by-Step Execution Plan

**Date:** January 15, 2026
**Protocol Version:** 2026-01-15
**Status:** Ready for Advisor Approval
**Reference:** `docs/CROSS_ARCHITECTURE_DISCOVERY_PIPELINE_ENTRYGATE.md`

---

## Research Question

**Does the R_V contraction circuit (discovered on Mistral-7B) generalize to other transformer architectures?**

The R_V metric measures geometric contraction in Value matrix column space during recursive self-observation prompts. We have validated this effect on Mistral-7B (Cohen's d = -3.56, p < 10⁻⁶) and partially validated on Llama-3-8B (d = -1.34, p = 0.0087). This protocol systematically tests new architectures to map circuit similarities and differences.

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

---

## Critical Invariants (Fixed Parameters)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Prompt bank | `prompts/` directory (754 prompts) | Versioned, validated |
| Early layer | 5 | After initial embedding processing |
| Window size | 16 tokens | Last 16 tokens of prompt |
| Late layer | 84% of depth | L27 for 32-layer, L21 for 25-layer, etc. |
| Seed | 42 | Reproducibility |
| Min n per group | 40 | Statistical power (≥30 for publication) |

**These parameters are fixed across all experiments to ensure comparability.**

---

## Phase 0: Pre-Flight Checks

### Step 0.1: Verify Repository State
- **Action:** Check git status, ensure clean working directory
- **Command:** `git status`
- **Success:** Clean or on feature branch
- **Time:** 1 minute

### Step 0.2: Verify Prompt Bank
- **Action:** Confirm prompt loader works and has required groups
- **Command:** `python -c "from prompts.loader import PromptLoader; l = PromptLoader(); print(f'Version: {l.version}')"`
- **Success:** Version hash printed, no errors
- **Required groups:** `champions`, `L4_full`, `L3_deeper`, `baseline_*`, `length_matched`, `pseudo_recursive`
- **Time:** 1 minute

### Step 0.3: Verify Metrics
- **Action:** Confirm R_V and extended metrics import correctly
- **Command:** `python -c "from src.metrics import compute_rv, compute_extended_metrics; print('Metrics OK')"`
- **Success:** No import errors
- **Time:** 1 minute

**Total Phase 0 Time:** ~3 minutes

---

## Phase 1: Generate Model Configs

### Step 1.1: Select Target Model
- **Action:** Choose HuggingFace model identifier
- **Examples:**
  - `mistralai/Mixtral-8x7B-v0.1` (MoE - expected stronger effect)
  - `Qwen/Qwen2-7B` (Different architecture family)
  - `google/gemma-2-9b` (Another variant)
- **Decision:** Advisor approval on model selection

### Step 1.2: Auto-Generate Configs
- **Action:** Generate standardized config files for all phases
- **Command:** `python scripts/generate_model_configs.py --model [HF_MODEL_NAME] --output-dir configs/discovery`
- **What it does:**
  - Auto-detects model architecture (layers, heads, hidden dim)
  - Calculates late layer (84% depth)
  - Generates configs for Phases 2-6
  - Creates directory: `configs/discovery/[MODEL_SHORT]/`
- **Output:** 23 JSON config files
- **Time:** 2-5 minutes (depends on HF API speed)

### Step 1.3: Verify Generated Configs
- **Action:** Check that all required configs exist
- **Command:** `ls configs/discovery/[MODEL_SHORT]/`
- **Required files:**
  - `01_baseline_rv.json` (Phase 2)
  - `02_source_hunt_mlp_ablation_l*.json` (Phase 3, layers 0-8)
  - `03_transfer_hunt_mlp_steer_l*.json` (Phase 4, layers 0-10)
  - `04_readout_validation.json` (Phase 5)
  - `05_head_identification.json` (Phase 6)
- **Time:** 1 minute

### Step 1.4: Validate Registry Compatibility
- **Action:** Ensure all experiments are registered in pipeline registry
- **Command:**
```bash
python -c "
from src.utils import validate_registry_compatibility
import json
from pathlib import Path
configs = [json.load(open(f)) for f in Path('configs/discovery/[MODEL_SHORT]').glob('*.json')]
missing = validate_registry_compatibility(configs)
print('Missing:', missing if missing else 'NONE')
"
```
- **Success:** "Missing: NONE" or empty list
- **If missing:** STOP - must add experiments to registry first
- **Time:** 1 minute

**Total Phase 1 Time:** ~5-10 minutes

---

## Phase 2: Baseline R_V Separation (GATE PHASE)

**This is the critical decision gate. If Phase 2 fails, the circuit does not generalize.**

### Step 2.1: Run Baseline R_V Measurement
- **Action:** Measure R_V for champions vs controls
- **Command:** `python -m src.pipelines.run --config configs/discovery/[MODEL_SHORT]/01_baseline_rv.json`
- **What it does:**
  - Loads model (requires ≥24GB VRAM for 7B models)
  - Processes ~100 prompts (50 champions, 50 controls)
  - Computes R_V at early (L5) and late (L27 or equivalent) layers
  - Performs statistical tests (t-test, Cohen's d)
- **Output:** `results/phase2_generalization/[MODEL_SHORT]/[TIMESTAMP]_cross_architecture_validation/`
  - `summary.json` (machine-readable results)
  - `report.md` (human-readable report)
  - `*.csv` (raw data)
- **Time:** 30-60 minutes (depends on model size and GPU)

### Step 2.2: Check Results
- **Action:** Verify success criteria
- **Command:** `cat results/phase2_generalization/[MODEL_SHORT]/*/summary.json | python -m json.tool`
- **Success Criteria (ALL must pass):**

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Champions R_V | < 0.75 | Must show contraction |
| Controls R_V | > 0.85 | Must NOT show contraction |
| Cohen's d | > 1.0 (absolute) | Meaningful effect size |
| p-value | < 0.01 | Statistical significance |
| n (champions) | ≥ 40 | Statistical power |
| n (controls) | ≥ 40 | Statistical power |

### Step 2.3: Decision Gate

| Result | Action | Rationale |
|--------|--------|-----------|
| **ALL criteria pass** | ✅ **Proceed to Phase 3** | Circuit generalizes |
| Weak effect (0.5 < \|d\| < 1.0) | ⚠️ **Proceed with caution** | Effect present but weak, note in report |
| No effect (\|d\| < 0.5) | ❌ **STOP** | Circuit does not generalize to this architecture |
| Wrong direction (champions > controls) | ❌ **STOP** | Investigate data quality, possible bug |

**If STOP:** Document findings in `CIRCUIT_MAP.md` with status "NOT FOUND" and explain why.

**Total Phase 2 Time:** 30-60 minutes + analysis time

---

## Phase 3: Source Layer Hunt (MLP Ablation)

**Goal:** Find which early-layer MLP is **necessary** for contraction.

### Step 3.1: Run Ablation Sweep
- **Action:** Ablate (zero out) MLP at each early layer, measure R_V
- **Command:**
```bash
for layer in 0 1 2 3 4 5 6 7 8; do
  echo "=== Ablating Layer $layer ==="
  python -m src.pipelines.run \
    --config configs/discovery/[MODEL_SHORT]/02_source_hunt_mlp_ablation_l${layer}.json
done
```
- **What it does:**
  - For each layer L0-L8, zero out MLP output
  - Measure R_V on recursive prompts
  - Compare to baseline (no ablation)
- **Output:** 9 result directories (one per layer)
- **Time:** 4-9 hours (30-60 min per layer × 9 layers)

### Step 3.2: Analyze Results
- **Action:** Identify which layer(s) are necessary
- **Success Criteria:** A layer is the **SOURCE** if ablating it:
  - Increases R_V by > 0.3 (removes contraction)
  - Shows statistical significance (p < 0.01)

### Step 3.3: Expected Pattern

| Layer | R_V (ablated) | Delta | Role |
|-------|---------------|-------|------|
| L0 | ~1.5-2.0 | +0.8+ | PRIMARY SOURCE (expected) |
| L1 | ~1.2-1.5 | +0.5 | Secondary |
| L2-L4 | ~0.9-1.1 | +0.2 | Minor |
| L5+ | ~0.7-0.8 | ~0 | No effect |

**Note:** If pattern differs significantly, this is a real finding - document it.

**Total Phase 3 Time:** 4-9 hours + analysis time

---

## Phase 4: Transfer Layer Hunt (MLP Steering)

**Goal:** Find which layer's MLP can **induce** contraction via steering.

### Step 4.1: Run Steering Sweep
- **Action:** Apply steering vector at each layer, measure R_V on baseline prompts
- **Command:**
```bash
for layer in 0 1 2 3 4 5 6 7 8 9 10; do
  echo "=== Steering Layer $layer ==="
  python -m src.pipelines.run \
    --config configs/discovery/[MODEL_SHORT]/03_transfer_hunt_mlp_steer_l${layer}.json
done
```
- **What it does:**
  - Extract steering vector from recursive vs control prompts
  - Apply steering at each layer L0-L10
  - Measure R_V on baseline prompts (should contract if steering works)
- **Output:** 11 result directories (one per layer)
- **Time:** 5-11 hours (30-60 min per layer × 11 layers)

### Step 4.2: Analyze Results
- **Action:** Identify transfer point
- **Success Criteria:** A layer is the **TRANSFER POINT** if steering it:
  - Decreases R_V (induces contraction on baseline prompts)
  - Random direction control shows NO effect
  - Effect is statistically significant

### Step 4.3: Expected Pattern

| Layer | R_V (steered) | Delta | Effect |
|-------|---------------|-------|--------|
| L0-L2 | variable | ?? | May cause artifacts |
| L3-L4 | ~0.7-0.8 | -0.2 | TRANSFER POINT (expected) |
| L5+ | ~0.95 | ~0 | No effect |

### Step 4.4: Handle Anomalies

**If steering causes EXPANSION (R_V increases):**
1. This is a real finding - document it (Llama-3-8B showed this)
2. Try V-projection steering instead of residual stream
3. Check if model uses different gating (SiLU vs GELU)
4. Document in `CIRCUIT_MAP.md` as architectural difference

**Total Phase 4 Time:** 5-11 hours + analysis time

---

## Phase 5: Readout Validation

**Goal:** Confirm the late-layer readout mechanism via activation patching.

### Step 5.1: Run Validation
- **Action:** Test causal transfer via patching
- **Command:** `python -m src.pipelines.run --config configs/discovery/[MODEL_SHORT]/04_readout_validation.json`
- **What it does:**
  - Runs four-way control experiment:
    1. **Target patch** (late layer → late layer): Should transfer R_V
    2. **Wrong layer patch** (earlier layer → late layer): Should NOT transfer
    3. **Random patch**: Should NOT transfer
    4. **Shuffled patch**: Should NOT transfer
- **Output:** Result directory with all four conditions
- **Time:** 1-2 hours

### Step 5.2: Success Criteria
- Target patch transfer efficiency > 50%
- All controls < 20% transfer
- Statistical separation between target and controls

**Total Phase 5 Time:** 1-2 hours + analysis time

---

## Phase 6: Extended Metrics (Publication-Grade)

**Goal:** Compute additional geometric metrics for publication.

### Step 6.1: Compute Extended Metrics
- **Action:** Run extended metric computation on representative prompts
- **Code:**
```python
from src.metrics import compute_extended_metrics
from src.core.models import load_model

model, tokenizer = load_model("[HF_MODEL_NAME]")

# For representative prompts from each group
for prompt in sample_prompts:
    ext = compute_extended_metrics(
        model, tokenizer, prompt,
        early_layer=5, late_layer=27, window_size=16
    )
    # Record: cosine_early_late, spectral_late_top1_ratio,
    #         spectral_late_spectral_gap, attention_entropy
```
- **Metrics computed:**
  - `cosine_early_late`: Directional alignment between early and late layers
  - `spectral_late_top1_ratio`: Dominance of first singular value
  - `spectral_late_spectral_gap`: Gap between σ₁ and σ₂
  - `attention_entropy`: Focus at readout layer
- **Time:** 1-2 hours

### Step 6.2: Expected Patterns

| Metric | Champions (expected) | Controls (expected) |
|--------|---------------------|---------------------|
| cosine_early_late | > 0.8 (converging) | < 0.6 (diverging) |
| spectral_late_top1_ratio | > 0.5 (one dominant direction) | < 0.3 (distributed) |
| attention_entropy | Lower (focused) | Higher (diffuse) |

**Total Phase 6 Time:** 1-2 hours

---

## Output Requirements

### Required Artifact: CIRCUIT_MAP.md

Create `results/phase2_generalization/[MODEL_SHORT]/CIRCUIT_MAP.md` with:

1. **Model Architecture Info**
   - Layers, heads, hidden dim, GQA status, num_kv_heads

2. **Phase 2 Results Table**
   - R_V means, stds, n, 95% CIs for champions vs controls
   - Cohen's d, p-value

3. **Phase 3 Results Table**
   - R_V (ablated) for each layer L0-L8, delta, p-value, role assignment

4. **Phase 4 Results Table**
   - R_V (steered) for each layer L0-L10, delta, p-value, effect type

5. **Phase 5 Results Table**
   - Transfer efficiency for target vs all three controls

6. **Extended Metrics Table**
   - All extended metrics for champions vs controls with deltas

7. **Circuit Comparison Table**
   - Component-by-component comparison to Mistral-7B reference

8. **Conclusions**
   - Summary of findings
   - Anomalies documented explicitly
   - Hypotheses for any differences

9. **Raw Data Paths**
   - Full paths to all result directories

---

## Total Time Estimate

| Phase | Time | Notes |
|-------|------|-------|
| Phase 0 | 3 min | Pre-flight checks |
| Phase 1 | 5-10 min | Config generation |
| Phase 2 | 30-60 min | **GATE PHASE** |
| Phase 3 | 4-9 hours | Ablation sweep (9 layers) |
| Phase 4 | 5-11 hours | Steering sweep (11 layers) |
| Phase 5 | 1-2 hours | Readout validation |
| Phase 6 | 1-2 hours | Extended metrics |
| **Total** | **11-25 hours** | Plus analysis/documentation time |

**Note:** Times assume GPU with ≥24GB VRAM. Larger models (Mixtral) may require more.

---

## Decision Points & Stopping Criteria

### Stop After Phase 2 If:
- No effect (Cohen's d < 0.5)
- Wrong direction (champions R_V > controls R_V)
- **Action:** Document as "Circuit does not generalize" in CIRCUIT_MAP.md

### Proceed with Caution If:
- Weak effect (0.5 < |d| < 1.0)
- **Action:** Note in report, proceed but flag as weak effect

### Continue Through All Phases If:
- Phase 2 passes all criteria
- **Action:** Complete Phases 3-6, document full circuit map

---

## Risk Mitigation

| Risk | Mitigation | Fallback |
|------|------------|----------|
| Model too large for GPU | Use `device_map="auto"`, `torch_dtype=float16` | Test smaller variant first |
| Phase 2 fails | This is a valid finding - document it | Negative results are publishable |
| Anomalous results | Document as architectural difference | See Llama-3-8B steering anomaly |
| Registry missing experiments | Phase 1.4 catches this early | Add to registry before proceeding |
| Low sample sizes | Check for tokenization errors, OOM | Reduce batch size |

---

## Reproducibility Guarantees

- ✅ Fixed seed (42) for all random operations
- ✅ Versioned prompt bank (hash tracked in every output)
- ✅ Config snapshots saved with each run
- ✅ All results saved to CSV + JSON
- ✅ Model metadata captured (architecture, GQA status)

---

## References

- **Protocol Document:** `docs/CROSS_ARCHITECTURE_DISCOVERY_PIPELINE_ENTRYGATE.md`
- **Metrics Reference:** `docs/METRICS_REFERENCE.md`
- **Reference Circuit:** Mistral-7B (validated, d = -3.56)
- **Example Output:** `results/phase2_generalization/llama3_8b_base/CIRCUIT_MAP.md`
- **Phase 1 Report:** `R_V_PAPER/research/PHASE1_FINAL_REPORT.md`

---

## Approval Checklist

- [ ] Target model selected and approved
- [ ] Protocol (this document) approved
- [ ] GPU access confirmed (≥24GB VRAM)
- [ ] HuggingFace access confirmed
- [ ] Time allocation confirmed (11-25 hours per model)

---

**Ready to proceed upon approval.**
