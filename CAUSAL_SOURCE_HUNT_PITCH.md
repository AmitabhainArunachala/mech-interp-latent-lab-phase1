# CAUSAL SOURCE HUNT: Finding the Origin of L4 Contraction

## Executive Summary

**Current Status**: We have identified the CONTROL KNOB (L24-27 residual stream) where R_V contraction manifests, but not the CAUSAL SOURCE where this geometric phenomenon originates.

**Proposed Approach**: Two complementary methodological strategies synthesized from established mechanistic interpretability research traditions, with a recommended hybrid execution plan.

---

## Background & Motivation

### What We Know
- **Effect**: Universal geometric contraction (R_V < 1.0) in value-space at ~84% model depth
- **Location**: Manifests in residual stream at L24-27 (control knob)
- **Mechanism**: Involves attention heads reading compressed value representations
- **Gap**: Unknown where/why contraction begins in the forward pass

### Research Context
- **Gold Standard**: "Know WHY the mechanism works" (mechanistic understanding)
- **Current Evidence**: PR drops most sharply L0→L2 (-35%), then L14→L16 (-15%)
- **Hypothesis**: Early layers seed contraction, later layers amplify it

---

## OPTION A: Circuit Decomposition Approach
*Methodological Foundation: Elhage et al. 2021, Conmy et al. 2023, Wang et al. 2022*

### Core Strategy
Systematically decompose the residual stream into constituent components and test each for causal contribution.

### Experimental Program

#### 1. Component-Specific Patching
**Mathematical Basis**:
```
residual[L] = residual[L-1] + attention_out[L] + mlp_out[L]
```

**Experiments**:
- **Attention-Only Patching**: Replace only `attention_out[L]` with recursive→baseline
- **MLP-Only Patching**: Replace only `mlp_out[L]` with recursive→baseline
- **Measurement**: R_V recovery at L27

**Expected Signal**:
- If attention patching transfers effect → attention writes causal information
- If MLP patching transfers effect → MLP processes causal information
- If both required → distributed mechanism

#### 2. Path Patching (Wang et al. 2022)
**Core Idea**: Trace information flow from early heads to late measurement point.

**Implementation**:
- Identify heads at L5-L20 that write to residual stream
- Test if L27 reads from these early components
- Map complete circuits: Head A → MLP B → Head C → L27

#### 3. Activation Difference Steering
**Technical Approach**:
```python
# Compute the contraction direction
diff = recursive_activations - baseline_activations
U, S, Vt = torch.linalg.svd(diff)  # Find dominant direction
steering_vector = Vt[0]  # Top principal component

# Patch only this direction
patched_activations = baseline_activations + steering_vector * coefficient
```

### Success Criteria
- Circuit diagram: "Path from L[X] heads → L[Y] MLPs → L27 measurement"
- Quantitative: >50% effect transfer with minimal interventions

---

## OPTION B: Causal Intervention Approach
*Methodological Foundation: Meng et al. 2022, Geiger et al. 2023*

### Core Strategy
Treat each layer/component as a potential causal node and systematically intervene.

### Experimental Program

#### 1. Layer-wise Causal Tracing
**Protocol**:
- For each layer L ∈ [0,27]:
  - Corrupt hidden state with Gaussian noise (σ = layer_std)
  - Run forward pass to L27
  - Measure R_V degradation

**Expected Pattern**: Sharp drop at "source layers" where corruption prevents contraction.

#### 2. Mean Ablation Sweep
**Protocol**:
- For each layer L:
  - Replace activation with dataset mean (computed across 1000 samples)
  - Measure R_V impact

**Advantages**: More interpretable than noise corruption.

#### 3. Early Layer Propagation Test
**Motivation**: Eigenstate data shows -35% PR drop L0→L2.

**Experiment**:
```python
# Test if early patching propagates contraction
patched_state = patch_baseline_to_recursive_at_L2(model_inputs)
# Run forward from L2 to L27
# Measure if R_V contracts without further intervention
```

#### 4. Accumulated Effect Analysis
**Protocol**:
- Patch ranges [0-5], [5-10], [10-15], [15-20], [20-27] together
- Measure which range contains sufficient causal factors
- Iterative refinement: binary search on critical ranges

### Success Criteria
- Localization: "Contraction originates in layers [X-Y]"
- Causal strength: >80% effect elimination via targeted ablation

---

## RECOMMENDED EXECUTION PLAN

### Phase 1: Localization (Option B - 2 weeks)
**Goal**: Narrow search space to critical layer ranges.

**Priority Experiments**:
1. Mean ablation sweep (all layers)
2. Early layer propagation test (L0-L5)
3. Accumulated effect analysis (5 layer ranges)

**Success Metric**: Identify 1-2 critical ranges containing >70% causal effect.

### Phase 2: Circuit Mapping (Option A - 3 weeks)
**Goal**: Detailed mechanism understanding within identified ranges.

**Experiments**:
- Component-wise patching in critical ranges
- Path patching on identified heads/MLPs
- Activation difference steering

### Phase 3: Validation & Controls (1 week)
- Cross-validation on held-out prompts
- Statistical robustness testing
- Alternative explanations ruled out

---

## RESOURCE REQUIREMENTS

| Phase | GPU Hours | Implementation Time | Key Dependencies |
|-------|-----------|-------------------|------------------|
| 1 | 50 | 1 week | Existing patching infrastructure |
| 2 | 150 | 2 weeks | Path patching implementation |
| 3 | 25 | 0.5 weeks | Statistical analysis tools |

**Total Estimate**: 225 GPU hours, 3.5 weeks development

---

## RISK ASSESSMENT & MITIGATION

### Technical Risks
- **Numerical Instability**: SVD convergence issues with short prompts
  - *Mitigation*: Robust SVD with fallback metrics
- **Confounding Effects**: Early interventions affect all downstream computation
  - *Mitigation*: Compare to baseline and shuffled controls
- **Memory Constraints**: Large activation storage for path patching
  - *Mitigation*: Process in batches, selective layer storage

### Methodological Risks
- **Circular Reasoning**: Patching assumes the effect exists
  - *Mitigation*: Multiple intervention types (ablation, corruption, patching)
- **Over-interpretation**: Local effects may not generalize
  - *Mitigation*: Cross-validation on prompt variants

---

## DECISION REQUEST

1. **Approach Priority**: Start with Option B (localization) then Option A (circuit mapping)?
2. **Scope**: Focus on attention mechanisms first, or test both attention + MLP?
3. **Timeline**: 3.5 week execution plan acceptable?
4. **Compute Budget**: 225 GPU hours within available resources?

---

## IMMEDIATE NEXT STEPS

**If Approved**:
1. Implement mean ablation sweep infrastructure
2. Prepare accumulated effect analysis
3. Set up proper controls and statistical testing

**Awaiting Direction**

---

*Document Version: 1.0 | Date: December 13, 2025 | Author: Dhyana*
