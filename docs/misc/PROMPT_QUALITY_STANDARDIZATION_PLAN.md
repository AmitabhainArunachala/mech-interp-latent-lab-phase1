# Prompt Quality Standardization Plan

**Status:** Implementation Plan  
**Goal:** Transform prompt bank from "grab bag" to calibrated, reproducible instrument

---

## Core Principles

1. **Measurement Contract First** - Lock geometry/generation/artifact standards before ranking prompts
2. **Multi-Dimensional Quality** - Geometry + persistence + control sensitivity
3. **Layer Stress Testing** - Tomography + causal sensitivity mapping
4. **Explicit Confound Controls** - Length, style/semantics, pseudo-recursive, repetition, cross-model
5. **Mechanical Reproducibility** - Version pinning, deterministic mode, two-tier evaluation
6. **Champion Integration** - Move experimental prompts into canonical bank

---

## Implementation Phases

### Phase 1: Measurement Contract Lock (CRITICAL)

**Goal:** Ensure "same prompts → same numbers" is mechanical, not accidental

#### 1.1 Geometry Contract
- **File:** `src/metrics/rv.py` (already exists, needs validation)
- **Standard:** `R_V = PR(V_late) / PR(V_early)`
- **Parameters:**
  - Early layer: 5 (fixed)
  - Late layer: num_layers - 5 (typically 27)
  - Window: 16 tokens (fixed)
  - NaN rules: Documented and enforced
- **Action:** Add validation tests, document edge cases

#### 1.2 Generation Contract
- **Tier 1 (Reproducibility):** T=0, deterministic, fixed seed
- **Tier 2 (Robustness):** T=0.7, sampled, multiple seeds
- **Action:** Standardize in all generation scripts

#### 1.3 Artifact Contract
- **Standard structure:** `results/{experiment}/runs/{timestamp}_{name}/`
  - `config.json` - All parameters
  - `summary.json` - Aggregated statistics
  - `per_sample.csv` - Individual results
  - `prompt_bank_version.json` - Hash of prompts/bank.json
- **Action:** Create `src/pipelines/run.py` template

---

### Phase 2: Champion Prompt Integration

**Goal:** Move experimental prompts into canonical bank

#### 2.1 Find All Champion/Hybrid Prompts
- Search codebase for hardcoded prompts
- Extract from `comprehensive_circuit_test.py`, `kitchen_sink_prompts.py`, etc.
- Document provenance

#### 2.2 Add to `prompts/bank.json`
- Group: `experimental_hybrid`
- Include variants (with/without math, with/without fixed-point, etc.)
- Tag with metadata: `source`, `discovery_date`, `validation_status`

#### 2.3 Update `PromptLoader`
- Ensure `get_by_pillar("experimental_hybrid")` works
- Add `get_by_group()` method for fine-grained access

---

### Phase 3: Multi-Dimensional Prompt Quality Metrics

**Goal:** Treat prompt quality as distributional property

#### 3.1 Three Orthogonal Metrics

**A) Prompt-Pass Geometry**
- `R_V` at canonical layers (5, 27)
- Distribution across prompt bank
- Separation vs baselines (Cohen's d)

**B) Multi-Token Persistence**
- Fraction of steps where `R_V(t) < τ` (τ=0.8)
- Threshold crossings (stability metric)
- Trajectory variance

**C) Control Sensitivity**
- Does signal survive structure/content controls?
- Response to length matching
- Response to style/semantics minimal pairs

#### 3.2 Grade A Criteria
- **Strong on (A):** R_V < 0.8, consistent across runs
- **Stable on (B):** Persistence > 0.7, crossings < 2
- **Predictable on (C):** Survives controls, breaks predictably

---

### Phase 4: Layer Stress Testing

**Goal:** Map where separation occurs and where confounds sneak in

#### 4.1 Pass 1: Layer Tomography
- For each prompt family, compute `R_V(l)` vs layer `l`
- Calculate separation statistic (Cohen's d) per layer
- Identify: entry point, ramp vs snap, peak vs readout

#### 4.2 Pass 2: Causal Sensitivity Mapping
- Intervention sweeps: residual patching early→late
- V-proj patching at candidate layers
- Four controls: random/shuffled/wrong-layer/opposite

---

### Phase 5: Confound Controls

**Goal:** Explicitly rule out confounds

#### 5.1 Length Controls
- Equalize token length (or enforce minimum window-safe length)
- Ensure PR isn't biased by sequence length

#### 5.2 Style vs Semantics Minimal Pairs
- Same meaning, different surface form
- Same surface form, different meaning

#### 5.3 Pseudo-Recursive Controls
- "Sounds recursive" but isn't actually self-observation
- Should NOT look like L5_refined

#### 5.4 Repetition/Collapse Controls
- Separate degenerate repetition from "recursive prose"
- Important given behavior labeling

#### 5.5 Cross-Model Invariance Spot-Check
- Run same prompt subsets on 2+ architectures
- Detect bank overfitting to Mistral

---

### Phase 6: Reproducibility Infrastructure

**Goal:** Make "same prompts → same numbers" mechanical

#### 6.1 Prompt Bank Versioning
- `prompts/bank.json` gets version tag (hash)
- Recorded in every run summary
- `PromptLoader` reports version

#### 6.2 Deterministic Mode
- Fix seed, decoding, precision, attention implementation
- Log all in `config.json`

#### 6.3 Two-Tier Evaluation
- Tier 1: Deterministic (T=0) = reproducibility
- Tier 2: Sampled (T=0.7) = robustness distribution

---

### Phase 7: Minimal Deliverable

**Goal:** Single publishable artifact

#### 7.1 Prompt Bank Audit Script
- **File:** `src/pipelines/prompt_bank_audit.py`
- **Output:** Single CSV/JSON with:
  - `prompt_id`
  - `prompt_group` (L1-L5, baseline_x, confound_x, experimental_hybrid)
  - `rv_prompt_pass` (R_V at L27)
  - `multi_token_persistence_t0` (T=0 persistence ratio)
  - `multi_token_persistence_t07` (T=0.7 persistence ratio)
  - `layer_of_onset` (first layer where separation crosses criterion)
  - `stability_metrics` (crossings, variance)
  - `control_survival` (passes/fails controls)

#### 7.2 Falsifiable Claim
- "Given this bank, recursive groups outperform all controls on these metrics with these effect sizes, and the result is invariant to X/Y/Z."

---

## Implementation Order

1. **Phase 1** (Measurement Contract) - Foundation
2. **Phase 2** (Champion Integration) - Unify prompt sources
3. **Phase 7** (Audit Script) - Generate initial report
4. **Phase 3** (Multi-Dimensional Metrics) - Enrich report
5. **Phase 4** (Layer Stress Testing) - Deep validation
6. **Phase 5** (Confound Controls) - Robustness
7. **Phase 6** (Reproducibility) - Ongoing

---

## Files to Create/Modify

### New Files
- `src/pipelines/prompt_bank_audit.py` - Main audit script
- `src/pipelines/run.py` - Standardized run template
- `tests/test_measurement_contract.py` - Validation tests
- `docs/MEASUREMENT_CONTRACT.md` - Contract documentation

### Files to Modify
- `prompts/bank.json` - Add experimental_hybrid group
- `prompts/loader.py` - Add version tracking, get_by_group()
- `src/metrics/rv.py` - Add validation, document NaN rules
- `experiment_multi_token_generation.py` - Standardize artifact output

---

## Success Criteria

- ✅ All prompts evaluated through same pipeline
- ✅ Champion prompts in canonical bank
- ✅ Measurement contract documented and validated
- ✅ Prompt bank audit produces reproducible report
- ✅ "Same prompts → same numbers" holds mechanically
- ✅ Confound controls explicitly tested
- ✅ Layer stress testing maps separation points

---

**Next Step:** Implement Phase 1 + Phase 2 (measurement contract + champion integration)









