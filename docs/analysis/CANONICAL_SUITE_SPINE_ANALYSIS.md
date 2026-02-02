# Canonical Suite SPINE Analysis: From Original Design to Current State

**Date:** January 5, 2025  
**Purpose:** Map the complete experimental SPINE - the causal story arc with stable metrics from source to now

---

## Executive Summary

**Original Design:** 13 sequential experiments across 5 phases  
**Current State:** 6/13 completed in Stage 2 (MLP-focused subset)  
**Gap:** 7 experiments remaining (Phase 0, Phase 1B-C-D-E)  
**SPINE Status:** Core MLP mechanism validated, but full causal arc incomplete

---

## The Original 13-Experiment Canonical Suite

### Phase 0: Metric Validation (Foundation)

#### 1. ✅ phase0_metric_targets
- **Purpose:** Validate R_V metric computation (PR at different layers)
- **Status:** ✅ Foundation (already run, not re-run in Stage 2)
- **Key Finding:** R_V measured correctly at L5/L27
- **Metrics:** R_V, PR_early, PR_late

#### 2. ✅ phase0_minimal_pairs
- **Purpose:** Establish baseline R_V separation (recursive vs baseline)
- **Status:** ✅ Foundation (already run, not re-run in Stage 2)
- **Key Finding:** R_V < 1.0 for recursive, R_V ≈ 1.0 for baseline
- **Metrics:** R_V, separation statistics

---

### Phase 1: Causal Discovery (Core Mechanism)

#### 3. ⭐ circuit_discovery (CRITICAL - NOT RUN IN STAGE 2)
- **Script:** `circuit_discovery.py`
- **Config:** `configs/gold/11_circuit_discovery.json`
- **Purpose:** Attribution patching sweep (identify causal drivers)
- **Key Finding:** L0 MLP attribution = 1.67 (highest), L18-L20 MLPs also strong
- **Status:** ⚠️ **MISSING FROM STAGE 2** - Must re-run
- **Metrics:** Attribution scores (logit differences), R_V

#### 4. ✅ mlp_ablation_necessity (L0, L1, L2, L3)
- **Script:** `mlp_ablation_necessity.py`
- **Config:** `configs/mlp_ablation_necessity_l*.json`
- **Purpose:** Test if L0-L3 MLPs are NECESSARY (zero ablation)
- **Key Finding:** L0/L1/L3 necessary, L2 not necessary
- **Status:** ✅ **COMPLETED IN STAGE 2** (4 experiments: L0, L1, L2, L3)
- **Metrics:** R_V delta, Mode Score M, p-values, effect sizes

#### 5. ✅ mlp_sufficiency_test (L0)
- **Script:** `mlp_sufficiency_test.py`
- **Config:** `configs/mlp_sufficiency_l0.json`
- **Purpose:** Test if L0 MLP alone is SUFFICIENT (patch recursive → baseline)
- **Key Finding:** L0 alone NOT sufficient
- **Status:** ✅ **COMPLETED IN STAGE 2**
- **Metrics:** R_V restoration, Mode Score M, behavior scores

#### 6. ✅ mlp_combined_sufficiency_test (L0+L1)
- **Script:** `mlp_combined_sufficiency_test.py`
- **Config:** `configs/combined_mlp_sufficiency_l0_l1.json`
- **Purpose:** Test if L0+L1 together are SUFFICIENT
- **Key Finding:** L0+L1 ANTI-SUFFICIENT (destabilizes system)
- **Status:** ✅ **COMPLETED IN STAGE 2**
- **Metrics:** R_V restoration, Mode Score M, norm logs

#### 7. ⏳ position_specific_ablation (L0)
- **Script:** `mlp_ablation_position_specific.py`
- **Config:** `configs/position_specific_l0_ablation.json`
- **Purpose:** Test which token positions drive L0 effect (BOS, first-4, last-16, all)
- **Status:** ⏳ **IN PROGRESS** (retry started)
- **Metrics:** R_V delta by position, Mode Score M by position

---

### Phase 1B: Transfer & Steering (NOT RUN IN STAGE 2)

#### 8. ⭐ mlp_steering_sweep (CRITICAL - NOT RUN IN STAGE 2)
- **Script:** `mlp_steering_sweep.py`
- **Config:** `configs/mlp_steering_sweep_corrected.json`
- **Purpose:** Test MLP steering at all layers (find optimal transfer layers)
- **Key Finding:** L3-L4 optimal for steering (not L0), L2 is artifact
- **Status:** ⚠️ **MISSING FROM STAGE 2** - Must re-run
- **Metrics:** R_V delta, Mode Score M, coherence, steering effects

#### 9. ⭐ random_direction_control (CRITICAL - NOT RUN IN STAGE 2)
- **Script:** `random_direction_control.py`
- **Config:** `configs/random_direction_control_l3_targeted.json`
- **Purpose:** Test if steering effects are direction-specific (not artifacts)
- **Key Finding:** L2 steering = artifact (random vectors show similar effects)
- **Status:** ⚠️ **MISSING FROM STAGE 2** - Must re-run
- **Metrics:** R_V delta (random vs true steering), statistical comparison

---

### Phase 1C: Late-Layer Attention (NOT RUN IN STAGE 2)

#### 10. ⭐ p1_ablation (CRITICAL - NOT RUN IN STAGE 2)
- **Script:** `p1_ablation.py`
- **Config:** `configs/gold/p1_ablation.json`
- **Purpose:** Test component hierarchy (V-Proj, Residual, KV cache)
- **Key Finding:** V-Proj primary, Residual amplifier, KV necessary but not sufficient
- **Status:** ⚠️ **MISSING FROM STAGE 2** - Must re-run
- **Metrics:** Recursion score, Mode Score M, component contributions

#### 11. ⭐ surgical_sweep (CRITICAL - NOT RUN IN STAGE 2)
- **Script:** `surgical_sweep.py`
- **Config:** `configs/gold/15_surgical_sweep.json` (C2 config)
- **Purpose:** Optimal steering configuration (H18+H26 + Residual + KV)
- **Key Finding:** C2 config → 0.15 recursion score, 20% success rate
- **Status:** ⚠️ **MISSING FROM STAGE 2** - Must re-run
- **Metrics:** Recursion score, success rate, quality scores, Mode Score M

---

### Phase 1D: KV Cache Mechanism (NOT RUN IN STAGE 2)

#### 12. ⭐ kv_mechanism (CRITICAL - NOT RUN IN STAGE 2)
- **Script:** `kv_mechanism.py`
- **Config:** `configs/kv_mechanism.json`
- **Purpose:** Test KV cache geometry transfer
- **Key Finding:** KV replacement → 94% geometry transfer
- **Status:** ⚠️ **MISSING FROM STAGE 2** - Must re-run
- **Metrics:** R_V transfer, geometry restoration, Mode Score M

#### 13. ⚠️ kv_sufficiency_matrix (OPTIONAL)
- **Script:** `kv_sufficiency_matrix.py`
- **Config:** `configs/kv_sufficiency_matrix.json`
- **Purpose:** Test KV cache behavior transfer (with controls)
- **Status:** ⚠️ **CHECK IF NEEDED** - May be redundant with kv_mechanism

---

## Stage 2 Completion Status

### ✅ Completed (6/13)

1. ✅ **mlp_ablation_necessity** (L0, L1, L2, L3) - 4 experiments
2. ✅ **mlp_sufficiency_test** (L0)
3. ✅ **mlp_combined_sufficiency_test** (L0+L1)
4. ⏳ **position_specific_ablation** (L0) - In progress

**Total:** 6 core MLP experiments completed

### ⚠️ Missing from Stage 2 (7/13)

1. ⚠️ **circuit_discovery** - Attribution patching (found L0 MLP)
2. ⚠️ **mlp_steering_sweep** - Transferability testing (found L3-L4 optimal)
3. ⚠️ **random_direction_control** - Artifact validation (found L2 artifact)
4. ⚠️ **p1_ablation** - Component hierarchy (V-Proj primary)
5. ⚠️ **surgical_sweep** - Optimal config (C2: 20% success)
6. ⚠️ **kv_mechanism** - KV cache mechanism (94% geometry transfer)
7. ⚠️ **kv_sufficiency_matrix** - KV controls (optional)

**Note:** Phase 0 experiments (metric_targets, minimal_pairs) were already run previously and not re-run in Stage 2.

---

## The SPINE: Complete Causal Story Arc

### The SPINE Definition

The **SPINE** is the minimal set of sequential experiments that trace the complete causal mechanism from source to symptom, with stable metrics throughout.

### SPINE Structure

```
SOURCE (Early Layers)
    ↓
CAUSAL COMPUTATION (L0-L1 MLPs)
    ↓
TRANSFER POINTS (L3-L4 MLPs)
    ↓
REFINEMENT (L18-L20 MLPs)
    ↓
CONTENT ANCHOR (KV Cache)
    ↓
AMPLIFICATION (L26 Residual)
    ↓
SYMPTOM (L27 Attention H18+H26)
    ↓
OUTPUT (R_V < 1.0, Recursive Behavior)
```

### SPINE Experiments (Sequential Order)

#### **SPINE Step 1: Attribution (What Causes It?)**
- **Experiment:** `circuit_discovery`
- **Question:** Which layers drive the logit differences?
- **Answer:** L0 MLP (1.67), L18-L20 MLPs (0.27-0.35)
- **Metric:** Attribution scores (logit differences)
- **Status:** ⚠️ **MISSING**

#### **SPINE Step 2: Necessity (Is It Required?)**
- **Experiment:** `mlp_ablation_necessity` (L0, L1, L2, L3)
- **Question:** Does zeroing these layers remove the effect?
- **Answer:** L0/L1/L3 necessary, L2 not necessary
- **Metric:** R_V delta, Mode Score M
- **Status:** ✅ **COMPLETED**

#### **SPINE Step 3: Sufficiency (Is It Enough?)**
- **Experiment:** `mlp_sufficiency_test` (L0), `mlp_combined_sufficiency_test` (L0+L1)
- **Question:** Can we restore the effect by patching these layers?
- **Answer:** L0 alone NOT sufficient, L0+L1 ANTI-SUFFICIENT
- **Metric:** R_V restoration, Mode Score M
- **Status:** ✅ **COMPLETED**

#### **SPINE Step 4: Position Specificity (Where Does It Act?)**
- **Experiment:** `position_specific_ablation` (L0)
- **Question:** Which token positions drive the effect?
- **Answer:** (Testing - BOS vs token-distributed)
- **Metric:** R_V delta by position, Mode Score M by position
- **Status:** ⏳ **IN PROGRESS**

#### **SPINE Step 5: Transferability (Can We Inject It?)**
- **Experiment:** `mlp_steering_sweep`
- **Question:** Which layers are optimal for injecting recursive behavior?
- **Answer:** L3-L4 optimal (not L0), L2 is artifact
- **Metric:** R_V delta, Mode Score M, steering effects
- **Status:** ⚠️ **MISSING**

#### **SPINE Step 6: Direction Specificity (Is It Real?)**
- **Experiment:** `random_direction_control`
- **Question:** Are steering effects direction-specific or just perturbations?
- **Answer:** L2 is artifact, L3-L4 direction-specific
- **Metric:** R_V delta (random vs true), statistical comparison
- **Status:** ⚠️ **MISSING**

#### **SPINE Step 7: Late-Layer Mechanism (How Does It Manifest?)**
- **Experiment:** `p1_ablation`
- **Question:** What's the component hierarchy at L27?
- **Answer:** V-Proj primary, Residual amplifier, KV necessary
- **Metric:** Recursion score, Mode Score M, component contributions
- **Status:** ⚠️ **MISSING**

#### **SPINE Step 8: Optimal Configuration (What's the Best Setup?)**
- **Experiment:** `surgical_sweep`
- **Question:** What's the optimal intervention configuration?
- **Answer:** C2: H18+H26 + Full KV + Residual (20% success)
- **Metric:** Recursion score, success rate, quality scores
- **Status:** ⚠️ **MISSING**

#### **SPINE Step 9: Content Mechanism (What's KV's Role?)**
- **Experiment:** `kv_mechanism`
- **Question:** How does KV cache transfer geometry?
- **Answer:** KV replacement → 94% geometry transfer
- **Metric:** R_V transfer, geometry restoration
- **Status:** ⚠️ **MISSING**

---

## Stable Metrics Throughout SPINE

### Primary Metrics (Consistent Across All Experiments)

1. **R_V (Geometric Signature)**
   - Definition: `PR_late / PR_early`
   - Purpose: Measures geometric contraction in value-space
   - Status: ✅ Computed in all experiments
   - Range: < 1.0 (contraction), ≈ 1.0 (neutral), > 1.0 (expansion)

2. **Mode Score M (Behavior Metric)**
   - Definition: `logsumexp(logits[R]) - logsumexp(logits[T])`
   - Purpose: Measures recursive behavior at logit level
   - Status: ✅ Fixed in Stage 2 (was NaN, now computed)
   - Range: Positive = recursive mode active, Negative = baseline mode

3. **Statistical Significance**
   - Method: One-sample t-test (delta vs 0.0)
   - Threshold: p < 0.01 (Bonferroni corrected)
   - Effect Size: Cohen's d ≥ 0.5 for meaningful effects
   - Status: ✅ Computed in all experiments

### Secondary Metrics (Experiment-Specific)

4. **Attribution Scores** (circuit_discovery)
   - Definition: Logit difference (recursive - baseline)
   - Purpose: Identify causal drivers
   - Status: ⚠️ Only in circuit_discovery (missing from Stage 2)

5. **Recursion Score** (surgical_sweep, p1_ablation)
   - Definition: Text-based heuristic (0-1 scale)
   - Purpose: Measure recursive behavior in generated text
   - Status: ⚠️ Only in behavior transfer experiments

6. **Success Rate** (surgical_sweep)
   - Definition: % of prompts triggering recursion
   - Purpose: Measure intervention effectiveness
   - Status: ⚠️ Only in behavior transfer experiments

---

## Gap Analysis: What's Missing?

### Critical Gaps (Blocking Complete SPINE)

1. **circuit_discovery** - Missing attribution step (found L0 MLP)
2. **mlp_steering_sweep** - Missing transferability step (found L3-L4 optimal)
3. **random_direction_control** - Missing artifact validation (found L2 artifact)
4. **p1_ablation** - Missing late-layer mechanism (V-Proj primary)
5. **surgical_sweep** - Missing optimal config (C2: 20% success)
6. **kv_mechanism** - Missing content mechanism (KV: 94% transfer)

### Why These Are Critical

- **circuit_discovery:** Establishes causal attribution (L0 MLP = 1.67)
- **mlp_steering_sweep:** Reveals transferability (L3-L4 optimal, not L0)
- **random_direction_control:** Validates steering isn't artifact
- **p1_ablation:** Shows late-layer hierarchy (V-Proj > Residual > KV)
- **surgical_sweep:** Proves optimal config works (C2: 20% success)
- **kv_mechanism:** Explains content transfer (KV: 94% geometry)

**Without these:** We have necessity/sufficiency but not the complete causal arc from source → transfer → symptom.

---

## Current State vs Original Design

### What Stage 2 Achieved

✅ **Core MLP Mechanism Validated:**
- L0/L1/L3 necessary (ablation removes contraction)
- L2 not necessary
- L0 alone NOT sufficient
- L0+L1 ANTI-SUFFICIENT (destabilizes)

✅ **Infrastructure Standardized:**
- Prompt IDs tracked
- Standardized metadata
- RUN_INDEX.jsonl tracking
- Mode Score fixed (was NaN, now computed)

### What Stage 2 Missed

❌ **Attribution Step:** circuit_discovery (found L0 MLP)
❌ **Transferability Step:** mlp_steering_sweep (found L3-L4 optimal)
❌ **Artifact Validation:** random_direction_control (found L2 artifact)
❌ **Late-Layer Mechanism:** p1_ablation (V-Proj primary)
❌ **Optimal Config:** surgical_sweep (C2: 20% success)
❌ **Content Mechanism:** kv_mechanism (KV: 94% transfer)

---

## Recommended SPINE Completion Plan

### Phase 1: Complete Core SPINE (Priority 1)

1. ✅ **mlp_ablation_necessity** (L0-L3) - DONE
2. ✅ **mlp_sufficiency_test** (L0) - DONE
3. ✅ **mlp_combined_sufficiency_test** (L0+L1) - DONE
4. ⏳ **position_specific_ablation** (L0) - IN PROGRESS
5. ⚠️ **circuit_discovery** - RE-RUN (attribution step)
6. ⚠️ **mlp_steering_sweep** - RE-RUN (transferability step)
7. ⚠️ **random_direction_control** - RE-RUN (artifact validation)

**Goal:** Complete early-layer mechanism (L0-L4)

### Phase 2: Complete Late-Layer SPINE (Priority 2)

8. ⚠️ **p1_ablation** - RE-RUN (component hierarchy)
9. ⚠️ **surgical_sweep** - RE-RUN (optimal config)
10. ⚠️ **kv_mechanism** - RE-RUN (content mechanism)

**Goal:** Complete late-layer mechanism (L26-L27)

### Phase 3: Validation (Priority 3)

11. ⚠️ **kv_sufficiency_matrix** - RE-RUN (if needed)
12. ✅ **phase0_metric_targets** - Already done (foundation)
13. ✅ **phase0_minimal_pairs** - Already done (foundation)

**Goal:** Validate complete SPINE

---

## The Complete SPINE with Metrics

### SPINE Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 0: METRIC VALIDATION (Foundation)                     │
├─────────────────────────────────────────────────────────────┤
│ 1. phase0_metric_targets                                    │
│    Metric: R_V, PR_early, PR_late                          │
│    Status: ✅ Foundation                                     │
│                                                             │
│ 2. phase0_minimal_pairs                                     │
│    Metric: R_V separation (recursive vs baseline)          │
│    Status: ✅ Foundation                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: CAUSAL DISCOVERY (Core Mechanism)                  │
├─────────────────────────────────────────────────────────────┤
│ 3. circuit_discovery ⚠️ MISSING                             │
│    Metric: Attribution scores (logit differences)           │
│    Finding: L0 MLP = 1.67 (highest)                         │
│                                                             │
│ 4. mlp_ablation_necessity ✅ COMPLETED                      │
│    Metric: R_V delta, Mode Score M                         │
│    Finding: L0/L1/L3 necessary, L2 not                      │
│                                                             │
│ 5. mlp_sufficiency_test ✅ COMPLETED                        │
│    Metric: R_V restoration, Mode Score M                    │
│    Finding: L0 alone NOT sufficient                         │
│                                                             │
│ 6. mlp_combined_sufficiency_test ✅ COMPLETED               │
│    Metric: R_V restoration, Mode Score M                    │
│    Finding: L0+L1 ANTI-SUFFICIENT                           │
│                                                             │
│ 7. position_specific_ablation ⏳ IN PROGRESS                │
│    Metric: R_V delta by position, Mode Score M             │
│    Finding: (Testing - BOS vs token-distributed)           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1B: TRANSFER & STEERING                               │
├─────────────────────────────────────────────────────────────┤
│ 8. mlp_steering_sweep ⚠️ MISSING                            │
│    Metric: R_V delta, Mode Score M, steering effects       │
│    Finding: L3-L4 optimal (not L0), L2 artifact             │
│                                                             │
│ 9. random_direction_control ⚠️ MISSING                      │
│    Metric: R_V delta (random vs true), statistical test    │
│    Finding: L2 artifact confirmed                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1C: LATE-LAYER ATTENTION                              │
├─────────────────────────────────────────────────────────────┤
│ 10. p1_ablation ⚠️ MISSING                                   │
│     Metric: Recursion score, Mode Score M                  │
│     Finding: V-Proj primary, Residual amplifier            │
│                                                             │
│ 11. surgical_sweep ⚠️ MISSING                                │
│     Metric: Recursion score, success rate, quality         │
│     Finding: C2 config (20% success)                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1D: KV CACHE MECHANISM                                │
├─────────────────────────────────────────────────────────────┤
│ 12. kv_mechanism ⚠️ MISSING                                 │
│     Metric: R_V transfer, geometry restoration              │
│     Finding: KV replacement → 94% geometry transfer        │
│                                                             │
│ 13. kv_sufficiency_matrix ⚠️ OPTIONAL                        │
│     Metric: Behavior transfer, controls                    │
│     Finding: (Check if needed)                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary: How Far Are We?

### Completion Status

- **Stage 2 Completed:** 6/13 experiments (46%)
- **Core SPINE Completed:** 4/9 steps (44%)
- **Full SPINE Completed:** 4/9 steps (44%)

### What We Have

✅ **Early-Layer Necessity:** L0/L1/L3 necessary, L2 not  
✅ **Early-Layer Sufficiency:** L0 alone NOT sufficient, L0+L1 ANTI-SUFFICIENT  
✅ **Infrastructure:** Standardized metrics, prompt IDs, metadata  
✅ **Mode Score:** Fixed and validated  

### What We're Missing

❌ **Attribution:** circuit_discovery (found L0 MLP)  
❌ **Transferability:** mlp_steering_sweep (found L3-L4 optimal)  
❌ **Artifact Validation:** random_direction_control (found L2 artifact)  
❌ **Late-Layer Mechanism:** p1_ablation, surgical_sweep (V-Proj, C2 config)  
❌ **Content Mechanism:** kv_mechanism (KV: 94% transfer)  

### The Gap

**We have the core mechanism (necessity/sufficiency) but not the complete causal arc.**

The SPINE requires:
1. Attribution (what causes it?) → ⚠️ Missing
2. Necessity (is it required?) → ✅ Complete
3. Sufficiency (is it enough?) → ✅ Complete
4. Transferability (can we inject it?) → ⚠️ Missing
5. Late-layer mechanism (how does it manifest?) → ⚠️ Missing
6. Content mechanism (what's KV's role?) → ⚠️ Missing

---

## Next Steps: Complete the SPINE

### Immediate Priority (Complete Core SPINE)

1. ⏳ Finish **position_specific_ablation** (L0)
2. ⚠️ Re-run **circuit_discovery** (attribution)
3. ⚠️ Re-run **mlp_steering_sweep** (transferability)
4. ⚠️ Re-run **random_direction_control** (artifact validation)

**Goal:** Complete early-layer mechanism (Steps 1-6 of SPINE)

### Secondary Priority (Complete Late-Layer SPINE)

5. ⚠️ Re-run **p1_ablation** (component hierarchy)
6. ⚠️ Re-run **surgical_sweep** (optimal config)
7. ⚠️ Re-run **kv_mechanism** (content mechanism)

**Goal:** Complete late-layer mechanism (Steps 7-9 of SPINE)

---

**Last Updated:** January 5, 2025  
**Status:** Stage 2 complete (6/13), SPINE 44% complete (4/9 steps)


