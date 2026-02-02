# SPINE Completion Roadmap: From Current State to Full Causal Arc

**Date:** January 5, 2025  
**Status:** Stage 2 Complete (6/13), SPINE 44% Complete (4/9 steps)

---

## Current State Summary

### ✅ What We Have (Stage 2 Completed)

**Core MLP Mechanism (4/9 SPINE steps):**
1. ✅ **Necessity:** L0/L1/L3 necessary, L2 not (mlp_ablation_necessity)
2. ✅ **Sufficiency (L0):** L0 alone NOT sufficient (mlp_sufficiency_test)
3. ✅ **Sufficiency (L0+L1):** L0+L1 ANTI-SUFFICIENT (mlp_combined_sufficiency_test)
4. ⏳ **Position Specificity:** Testing BOS vs token-distributed (position_specific_ablation)

**Infrastructure:**
- ✅ Prompt IDs tracked
- ✅ Standardized metadata
- ✅ RUN_INDEX.jsonl tracking
- ✅ Mode Score fixed (was NaN, now computed)
- ✅ Stable metrics: R_V, Mode Score M, statistical tests

### ❌ What We're Missing (7/13 experiments)

**Attribution & Transferability:**
1. ⚠️ **circuit_discovery** - Attribution patching (found L0 MLP = 1.67)
2. ⚠️ **mlp_steering_sweep** - Transferability testing (found L3-L4 optimal)
3. ⚠️ **random_direction_control** - Artifact validation (found L2 artifact)

**Late-Layer Mechanism:**
4. ⚠️ **p1_ablation** - Component hierarchy (V-Proj primary, Residual amplifier)
5. ⚠️ **surgical_sweep** - Optimal config (C2: 20% success)

**Content Mechanism:**
6. ⚠️ **kv_mechanism** - KV cache mechanism (94% geometry transfer)
7. ⚠️ **kv_sufficiency_matrix** - KV controls (optional)

---

## The Complete SPINE: 9 Sequential Steps

### SPINE Flow (Source → Symptom)

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: ATTRIBUTION (What Causes It?)                       │
│ Experiment: circuit_discovery                               │
│ Question: Which layers drive logit differences?             │
│ Answer: L0 MLP (1.67), L18-L20 MLPs (0.27-0.35)            │
│ Metric: Attribution scores (logit differences)             │
│ Status: ⚠️ MISSING                                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: NECESSITY (Is It Required?)                        │
│ Experiment: mlp_ablation_necessity (L0, L1, L2, L3)       │
│ Question: Does zeroing these layers remove the effect?      │
│ Answer: L0/L1/L3 necessary, L2 not necessary               │
│ Metric: R_V delta, Mode Score M                            │
│ Status: ✅ COMPLETED                                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: SUFFICIENCY (Is It Enough?)                        │
│ Experiment: mlp_sufficiency_test (L0),                     │
│             mlp_combined_sufficiency_test (L0+L1)           │
│ Question: Can we restore the effect by patching?            │
│ Answer: L0 alone NOT sufficient, L0+L1 ANTI-SUFFICIENT     │
│ Metric: R_V restoration, Mode Score M                       │
│ Status: ✅ COMPLETED                                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: POSITION SPECIFICITY (Where Does It Act?)          │
│ Experiment: position_specific_ablation (L0)                 │
│ Question: Which token positions drive the effect?           │
│ Answer: (Testing - BOS vs token-distributed)                │
│ Metric: R_V delta by position, Mode Score M                 │
│ Status: ⏳ IN PROGRESS                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: TRANSFERABILITY (Can We Inject It?)                 │
│ Experiment: mlp_steering_sweep                             │
│ Question: Which layers are optimal for injecting behavior? │
│ Answer: L3-L4 optimal (not L0), L2 is artifact             │
│ Metric: R_V delta, Mode Score M, steering effects           │
│ Status: ⚠️ MISSING                                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: DIRECTION SPECIFICITY (Is It Real?)                 │
│ Experiment: random_direction_control                        │
│ Question: Are steering effects direction-specific?         │
│ Answer: L2 is artifact, L3-L4 direction-specific            │
│ Metric: R_V delta (random vs true), statistical test       │
│ Status: ⚠️ MISSING                                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 7: LATE-LAYER MECHANISM (How Does It Manifest?)       │
│ Experiment: p1_ablation                                    │
│ Question: What's the component hierarchy at L27?            │
│ Answer: V-Proj primary, Residual amplifier, KV necessary    │
│ Metric: Recursion score, Mode Score M, component contribs   │
│ Status: ⚠️ MISSING                                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 8: OPTIMAL CONFIGURATION (What's the Best Setup?)     │
│ Experiment: surgical_sweep                                  │
│ Question: What's the optimal intervention configuration?   │
│ Answer: C2: H18+H26 + Full KV + Residual (20% success)     │
│ Metric: Recursion score, success rate, quality scores      │
│ Status: ⚠️ MISSING                                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 9: CONTENT MECHANISM (What's KV's Role?)              │
│ Experiment: kv_mechanism                                    │
│ Question: How does KV cache transfer geometry?             │
│ Answer: KV replacement → 94% geometry transfer             │
│ Metric: R_V transfer, geometry restoration                  │
│ Status: ⚠️ MISSING                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Stable Metrics Throughout SPINE

### Primary Metrics (All Experiments)

1. **R_V (Geometric Signature)**
   - Definition: `PR_late / PR_early`
   - Purpose: Measures geometric contraction in value-space
   - Status: ✅ Computed in all experiments
   - Interpretation: < 1.0 (contraction), ≈ 1.0 (neutral), > 1.0 (expansion)

2. **Mode Score M (Behavior Metric)**
   - Definition: `logsumexp(logits[R]) - logsumexp(logits[T])`
   - Purpose: Measures recursive behavior at logit level
   - Status: ✅ Fixed in Stage 2 (was NaN, now computed)
   - Interpretation: Positive = recursive mode, Negative = baseline mode

3. **Statistical Tests**
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

## Gap Analysis: What Blocks Complete SPINE?

### Critical Missing Steps

**Step 1: Attribution** (circuit_discovery)
- **Why Critical:** Establishes causal attribution (L0 MLP = 1.67)
- **Without It:** We know L0 is necessary but not why it's strongest
- **Impact:** Missing the "what causes it" foundation

**Step 5: Transferability** (mlp_steering_sweep)
- **Why Critical:** Reveals L3-L4 optimal (not L0) for steering
- **Without It:** We know L0 is causal but not optimal for transfer
- **Impact:** Missing the "can we inject it" mechanism

**Step 6: Direction Specificity** (random_direction_control)
- **Why Critical:** Validates steering isn't artifact (L2 is artifact)
- **Without It:** Can't distinguish real effects from perturbations
- **Impact:** Missing artifact validation

**Step 7: Late-Layer Mechanism** (p1_ablation)
- **Why Critical:** Shows V-Proj primary, Residual amplifier
- **Without It:** We know early layers but not late-layer hierarchy
- **Impact:** Missing the "how does it manifest" mechanism

**Step 8: Optimal Config** (surgical_sweep)
- **Why Critical:** Proves C2 config works (20% success)
- **Without It:** We know components but not optimal combination
- **Impact:** Missing the "what's the best setup" validation

**Step 9: Content Mechanism** (kv_mechanism)
- **Why Critical:** Explains KV cache role (94% geometry transfer)
- **Without It:** We know KV is necessary but not how it works
- **Impact:** Missing the "what's KV's role" mechanism

---

## Completion Plan: 3 Phases

### Phase 1: Complete Early-Layer SPINE (Priority 1)

**Goal:** Complete Steps 1-6 (attribution through direction specificity)

**Experiments:**
1. ⏳ Finish **position_specific_ablation** (L0) - IN PROGRESS
2. ⚠️ Re-run **circuit_discovery** - Attribution step
3. ⚠️ Re-run **mlp_steering_sweep** - Transferability step
4. ⚠️ Re-run **random_direction_control** - Artifact validation

**Expected Outcome:**
- Complete early-layer mechanism (L0-L4)
- Understand attribution vs transferability distinction
- Validate steering isn't artifact

**Timeline:** ~2-3 days (4 experiments × 5-6 hours each)

---

### Phase 2: Complete Late-Layer SPINE (Priority 2)

**Goal:** Complete Steps 7-9 (late-layer mechanism through content mechanism)

**Experiments:**
5. ⚠️ Re-run **p1_ablation** - Component hierarchy
6. ⚠️ Re-run **surgical_sweep** - Optimal config
7. ⚠️ Re-run **kv_mechanism** - Content mechanism

**Expected Outcome:**
- Complete late-layer mechanism (L26-L27)
- Understand component hierarchy (V-Proj > Residual > KV)
- Validate optimal configuration (C2: 20% success)

**Timeline:** ~2-3 days (3 experiments × 5-6 hours each)

---

### Phase 3: Validation & Integration (Priority 3)

**Goal:** Validate complete SPINE and integrate findings

**Tasks:**
- Compare old vs new results (where possible)
- Create unified SPINE report
- Document complete causal arc
- Identify any remaining gaps

**Timeline:** ~1 day

---

## The Complete SPINE: Metrics Consistency

### Metric Standardization Across All Experiments

**All experiments MUST compute:**

1. **R_V** (geometric signature)
   - Early layer: 5
   - Late layer: num_layers - 5 (typically 27)
   - Window: 16 tokens
   - Format: Mean ± std, delta, p-value, Cohen's d

2. **Mode Score M** (behavior metric)
   - Baseline: Mode score on baseline prompt
   - Intervention: Mode score on recursive/intervened prompt
   - Delta: Intervention - Baseline
   - Format: Mean ± std, delta, p-value, Cohen's d

3. **Statistical Tests**
   - One-sample t-test: delta vs 0.0
   - Threshold: p < 0.01 (Bonferroni corrected)
   - Effect size: Cohen's d

4. **Metadata**
   - Prompt IDs (recursive_prompt_id, baseline_prompt_id)
   - Prompt bank version
   - Git commit hash
   - Model ID
   - Seed
   - Config snapshot

**Status:** ✅ Stage 2 experiments follow this standard

---

## Summary: How Far Are We?

### Completion Status

| Category | Completed | Total | % Complete |
|----------|-----------|-------|------------|
| **Stage 2 Experiments** | 6 | 7 | 86% |
| **Original Canonical Suite** | 6 | 13 | 46% |
| **SPINE Steps** | 4 | 9 | 44% |
| **Early-Layer SPINE** | 3 | 6 | 50% |
| **Late-Layer SPINE** | 0 | 3 | 0% |

### What We Have

✅ **Core MLP Mechanism:**
- Necessity validated (L0/L1/L3 necessary, L2 not)
- Sufficiency tested (L0 alone NOT sufficient, L0+L1 ANTI-SUFFICIENT)
- Position specificity testing (in progress)

✅ **Infrastructure:**
- Standardized metrics (R_V, Mode Score M)
- Prompt IDs tracked
- Metadata standardized
- Mode Score fixed (was NaN, now computed)

### What We're Missing

❌ **Attribution:** circuit_discovery (found L0 MLP = 1.67)  
❌ **Transferability:** mlp_steering_sweep (found L3-L4 optimal)  
❌ **Artifact Validation:** random_direction_control (found L2 artifact)  
❌ **Late-Layer Mechanism:** p1_ablation, surgical_sweep  
❌ **Content Mechanism:** kv_mechanism (KV: 94% transfer)  

---

## Next Steps: Complete the SPINE

### Immediate Actions

1. ⏳ **Finish position_specific_ablation** (L0) - Complete Step 4
2. ⚠️ **Re-run circuit_discovery** - Complete Step 1 (attribution)
3. ⚠️ **Re-run mlp_steering_sweep** - Complete Step 5 (transferability)
4. ⚠️ **Re-run random_direction_control** - Complete Step 6 (artifact validation)

**Goal:** Complete early-layer SPINE (Steps 1-6)

### Secondary Actions

5. ⚠️ **Re-run p1_ablation** - Complete Step 7 (late-layer mechanism)
6. ⚠️ **Re-run surgical_sweep** - Complete Step 8 (optimal config)
7. ⚠️ **Re-run kv_mechanism** - Complete Step 9 (content mechanism)

**Goal:** Complete late-layer SPINE (Steps 7-9)

---

**Last Updated:** January 5, 2025  
**Status:** Stage 2 complete (6/13), SPINE 44% complete (4/9 steps)  
**Next:** Complete remaining 5 SPINE steps to achieve 100% coverage


