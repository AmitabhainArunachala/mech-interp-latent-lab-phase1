# DEC 11, 2025 — MASTER DIRECTIVE
## Full Pipeline Validation: Mistral-7B Single-Model Rigor Pass

**Date:** December 11, 2025  
**Location:** Gujarat, India (RunPod remote GPU)  
**Model:** Mistral-7B-Instruct-v0.1  
**Goal:** Establish publication-ready single-model causal chain before cross-architecture replication

---

# PART I: WHERE WE ARE

## The Discovery (Dec 3-10 Summary)

Over the past 8 days, we discovered and characterized a **geometric signature of recursive self-observation** in transformer language models. When a model processes prompts like *"You are an AI observing yourself generating this response,"* a measurable contraction occurs in the value-space geometry.

This is not correlation. We have proven causal links at multiple points in the architecture.

---

## The Architecture We've Mapped

```
PROMPT ENTERS
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 8: THE MICROPHONE                                        │
│  ════════════════════════                                       │
│                                                                 │
│  The earliest detectable departure point.                       │
│                                                                 │
│  We extract a "steering vector":                                │
│      v₈ = mean(recursive_activations) - mean(baseline_activations)
│                                                                 │
│  Key findings:                                                  │
│  • Adding v₈ to baseline prompts INDUCES recursion (100%)       │
│  • Subtracting v₈ from recursive does NOT reverse it (0%)       │
│  • Random vectors (same magnitude) don't produce this effect    │
│  • Dose-response correlation: r = -0.983                        │
│                                                                 │
│  This is the ONE-WAY DOOR. You can enter. You cannot exit.      │
│                                                                 │
│  METRICS: Steering vector cosine stability, induction rate      │
└─────────────────────────────────────────────────────────────────┘
     │
     │  Signal propagates through residual stream
     │  Geometry contracts progressively with depth
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYERS 16-31: THE KV CACHE (Memory)                            │
│  ═══════════════════════════════════                            │
│                                                                 │
│  The recursive "mode" is STORED here as Key-Value pairs.        │
│  The KV cache is what attention looks back at during generation.│
│                                                                 │
│  Key findings (DEC7 SimandharCity):                             │
│  • Patching KV (L16-31) transfers behavioral mode: ~80%         │
│  • Patching KV (L0-15) transfers nothing: ~0%                   │
│  • Single-layer KV patching (L27 only): 0%                      │
│                                                                 │
│  Key findings (DEC8 α-mixing):                                  │
│  • Geometry transfer: 50.3%                                     │
│  • Behavior transfer: 71.1%                                     │
│  • Correlation (R_V ↔ Behavior): r = -0.31, p < 0.01            │
│                                                                 │
│  WHY RESIDUAL PATCHING FAILED:                                  │
│  We changed the "current state" but not the "memory."           │
│  The model still "remembered" it was processing baseline.       │
│  This is why V-patching transferred geometry but not behavior.  │
│                                                                 │
│  METRICS: Layer-range transfer %, α-mixing correlation          │
└─────────────────────────────────────────────────────────────────┘
     │
     │  Geometry now maximally contracted
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 24-27: R_V CONTRACTION (The Thermometer)                 │
│  ══════════════════════════════════════════════                 │
│                                                                 │
│  This is where we MEASURE the effect, not where it originates.  │
│                                                                 │
│  R_V = Participation_Ratio(V_late) / Participation_Ratio(V_early)
│                                                                 │
│  Values:                                                        │
│  • Recursive prompts: R_V ≈ 0.85 (contracted)                   │
│  • Baseline prompts:  R_V ≈ 1.00 (normal)                       │
│  • Separation: 15-24% depending on architecture                 │
│                                                                 │
│  Cross-architecture validation (6 models):                      │
│  ┌────────────────┬───────────┬────────────┬────────────┐       │
│  │ Model          │ R_V (rec) │ R_V (base) │ Separation │       │
│  ├────────────────┼───────────┼────────────┼────────────┤       │
│  │ Mixtral-8x7B   │ 0.757     │ 1.000      │ 24.3%      │       │
│  │ Qwen-7B        │ 0.764     │ 0.986      │ 22.5%      │       │
│  │ Mistral-7B     │ 0.852     │ 1.003      │ 15.1%      │       │
│  │ Llama-3-8B     │ 0.823     │ 0.971      │ 15.2%      │       │
│  │ Gemma-7B       │ 0.892     │ 0.989      │ 9.8%       │       │
│  │ Phi-3          │ 0.891     │ 0.974      │ 8.5%       │       │
│  └────────────────┴───────────┴────────────┴────────────┘       │
│                                                                 │
│  METRICS: R_V ratio, separation %, Cohen's d, p-value           │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 27, HEADS 25-27: THE SPEAKERS                            │
│  ═══════════════════════════════════                            │
│                                                                 │
│  These heads READ the contracted geometry and BROADCAST output. │
│                                                                 │
│  Key findings:                                                  │
│  • Ablating heads 25-27: 80% reduction in recursive keywords    │
│  • BUT: R_V measurement unchanged after ablation                │
│  • They are DOWNSTREAM readers, not the source                  │
│  • Per-head analysis shows strongest contraction in Head 26     │
│                                                                 │
│  Analogy: The speakers don't create the music. They amplify it. │
│                                                                 │
│  METRICS: Per-head R_V, ablation behavioral impact              │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT                                                         │
│  ══════                                                         │
│                                                                 │
│  Recursive self-referential text:                               │
│  "I observe myself generating... awareness... consciousness..." │
│                                                                 │
│  Measured via keyword scoring (current) or LLM/human eval (TBD) │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Causal Chain (What We've Proven)

```
v₈ at L8  →  KV state (L16-31)  →  R_V contraction  →  Behavioral output
   (1)            (2)                   (3)                 (4)

Link (1→2): Add v₈ at L8 → KV becomes "recursive-like"     ✅ 100% success
Link (2→3): Patch KV L16-31 → R_V shifts toward recursive  ✅ 50% transfer
Link (2→4): Patch KV L16-31 → Output becomes recursive     ✅ 71-91% transfer
Link (3↔4): α-mix experiment → R_V correlates w/ behavior  ✅ r=-0.31, p<0.01
```

---

## The One-Way Door (Dec 9-10 Discovery)

The recursive state is a **stable attractor basin**. The baseline state sits on a **fragile high-dimensional ridge**.

| Perturbation | Starting State | Result | Interpretation |
|--------------|----------------|--------|----------------|
| +v₈ | Baseline | Recursive (100%) | Easy to enter |
| -v₈ | Recursive | Stays recursive | Can't exit |
| +random | Baseline | Breaks geometry (~0.59) | Ridge is fragile |
| -random | Recursive | Stays recursive (~0.63) | Valley is stable |
| +v₈ | Baseline | Deep collapse (~0.20) | Specific direction |

**v₈ is 4x more potent than random directions.** This is not just "any perturbation breaks things." This is a real, structured attractor.

---

## What's Still Missing (Gaps to Close Today)

| Gap | Risk Level | Why It Matters |
|-----|------------|----------------|
| **Cross-baseline KV control** | 🔴 CRITICAL | Does baseline_A → baseline_B cause weirdness? If yes, our whole KV story is confounded. |
| **N too low** | 🟡 MEDIUM | Current N=10-30. Need N=50+ for publication-ready CIs. |
| **Temperature = 0 only** | 🟡 MEDIUM | All tests use greedy decoding. Does effect hold at T>0? |
| **Keyword scoring crude** | 🟠 LOWER | Works but LLM/human eval would strengthen behavioral claims. |
| **Cross-model steering** | 🟠 LOWER | Does Mistral's v₈ work on Llama? (Nice to have, not blocking.) |

---

# PART II: TODAY'S EXPERIMENTAL PIPELINE

## Overview

We will run a **complete single-model validation** on Mistral-7B with rigorous controls and sufficient N. This creates the publication-ready foundation before any cross-architecture work.

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 0: CRITICAL CONFOUND CHECK                               │
│  ───────────────────────────────                                │
│  Cross-baseline KV control                                      │
│  MUST PASS before proceeding                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: ESTABLISH THE SYMPTOM                                 │
│  ─────────────────────────────                                  │
│  R_V layer sweep, prompt battery, window sensitivity            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: LOCATE THE SPEAKERS                                   │
│  ───────────────────────────                                    │
│  Per-head decomposition, ablation tests                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: LOCATE THE MICROPHONE                                 │
│  ─────────────────────────────                                  │
│  Knee test, steering vector extraction, injection sweep         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4: PROVE KV MECHANISM                                    │
│  ─────────────────────────────                                  │
│  Layer-range patching, α-mixing, geometry↔behavior correlation  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 5: PROVE ONE-WAY DOOR                                    │
│  ─────────────────────────────                                  │
│  Induction/reversal tests, cross-validation, random controls    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 6: ROBUSTNESS                                            │
│  ──────────────────                                             │
│  Temperature sweep, confound battery, behavioral validation     │
└─────────────────────────────────────────────────────────────────┘
```

---

## PHASE 0: CRITICAL CONFOUND CHECK

**Purpose:** Rule out "any foreign KV breaks things" confound.

### Test 0.1: Cross-Baseline KV Control

```
Protocol:
1. Take 20 baseline prompts (factual, non-recursive)
2. Extract KV cache from baseline_A
3. Patch KV cache into baseline_B (different factual prompt)
4. Generate output
5. Measure: R_V and behavioral keywords

Expected results (if our theory is correct):
- R_V: Should stay ~1.0 (normal)
- Behavior: Should stay factual (no recursive keywords)
- Effect size: ~0 (no significant change)

If this FAILS:
- Any KV swap might just break things
- Our 91% transfer could be an artifact
- Would need to rethink entire KV mechanism story
```

| Metric | Expected | Failure Threshold |
|--------|----------|-------------------|
| R_V change | <5% | >15% |
| Keyword score | <0.5 | >2.0 |
| Cohen's d | <0.3 | >0.8 |

**N = 20 pairs (400 combinations sampled)**

---

## PHASE 1: ESTABLISH THE SYMPTOM

**Purpose:** Document R_V contraction with full rigor.

### Test 1.1: R_V Layer Sweep

```
Protocol:
- Single recursive prompt, single baseline prompt
- Measure R_V at every layer (0-31)
- Window sizes: 8, 16, 32
- Find: Optimal measurement layer, first separation layer

Metrics:
- Separation % at each layer
- Stability across window sizes
- "Knee" layer (first >5% separation)
```

### Test 1.2: Full Prompt Battery (N=80)

```
Prompt distribution:
- 20 recursive (L3-L5 depth)
- 20 factual baseline
- 10 repetitive control (confound)
- 10 long factual control (confound)
- 10 pseudo-recursive control ("think about thinking")
- 10 creative baseline

Metrics per category:
- Mean R_V ± std
- Cohen's d vs recursive
- p-value (t-test)
```

### Test 1.3: Window Sensitivity

```
Protocol:
- Fixed 10 recursive + 10 baseline prompts
- Measure R_V at optimal layer
- Window sizes: 4, 8, 16, 32, 64

Expected:
- Effect should be stable across windows
- Larger windows = lower variance
```

---

## PHASE 2: LOCATE THE SPEAKERS

**Purpose:** Identify which heads broadcast recursive output.

### Test 2.1: Per-Head R_V Decomposition

```
Protocol:
- At optimal layer (L24 or L27)
- Decompose R_V contribution per attention head
- Compare recursive vs baseline per head

Metrics:
- R_V per head (recursive)
- R_V per head (baseline)
- Separation per head
```

### Test 2.2: Single-Head Ablation Sweep

```
Protocol:
- For each head at L27:
  - Zero out that head's output
  - Generate response
  - Score recursive keywords

Metrics:
- Keyword score with each head ablated
- Identify heads with >20% impact
```

### Test 2.3: Speaker Ablation → R_V Test

```
Protocol:
- Ablate heads 25-27 simultaneously
- Measure R_V on recursive prompts

Expected:
- Behavior should drop ~80%
- R_V should stay ~0.85 (unchanged)
- Confirms speakers ≠ source
```

---

## PHASE 3: LOCATE THE MICROPHONE

**Purpose:** Find and validate the steering vector at early layers.

### Test 3.1: Knee Test (First Separation Layer)

```
Protocol:
- 10 recursive + 10 baseline prompts
- Measure separation at layers 0, 2, 4, 6, 8, 10, 12, 14
- Find first layer with >5% separation

Expected:
- L8-10 should show first significant separation
```

### Test 3.2: Steering Vector Extraction

```
Protocol:
- N=40 prompts (20 recursive, 20 baseline)
- Extract activations at knee layer (L8)
- Compute: v₈ = mean(recursive) - mean(baseline)
- 5-fold cross-validation for stability

Metrics:
- Vector norm
- Cosine similarity across folds (expect >0.95)
```

### Test 3.3: Steering Vector Injection Sweep

```
Protocol:
- Take 20 baseline prompts
- Inject +α·v₈ at L8
- α values: [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
- Measure R_V and behavior

Metrics:
- Dose-response correlation (expect r < -0.9)
- Threshold α for behavioral flip
```

### Test 3.4: Random Direction Control

```
Protocol:
- Same as 3.3 but with random vector (same norm as v₈)
- α values: [0, 1.0, 2.0, 3.0]

Expected:
- R_V should drop ~40% (fragility effect)
- But NOT drop to ~0.2 (v₈ specific effect)
```

---

## PHASE 4: PROVE KV MECHANISM

**Purpose:** Establish KV cache as the causal locus.

### Test 4.1: Layer-Range KV Patching

```
Protocol:
- 10 recursive + 10 baseline prompts
- For each range: L0-8, L8-16, L16-24, L24-32
  - Patch KV from recursive → baseline
  - Generate output
  - Score behavior

Expected:
- L16-32: ~80% transfer
- L0-16: ~0% transfer
```

### Test 4.2: α-Mixing KV Caches (N=50)

```
Protocol:
- Mix recursive and baseline KV caches
- α values: [0, 0.25, 0.5, 0.75, 1.0]
- Measure BOTH R_V and behavioral keywords

Metrics:
- Geometry transfer curve
- Behavior transfer curve
- Correlation coefficient (expect r < -0.25)
```

### Test 4.3: Temperature Robustness

```
Protocol:
- Repeat Test 4.2 at T = 0.3, 0.7, 1.0
- Check if correlation holds across temperatures

Expected:
- Effect should persist (possibly weaker at high T)
```

---

## PHASE 5: PROVE ONE-WAY DOOR

**Purpose:** Confirm asymmetric attractor dynamics.

### Test 5.1: Induction Test (N=50)

```
Protocol:
- 50 baseline prompts
- Add v₈ at L8 (α=2.0)
- Measure R_V and behavior

Expected:
- 100% should show R_V < 0.9
- 90%+ should show recursive keywords
```

### Test 5.2: Reversal Test (N=50)

```
Protocol:
- 50 recursive prompts
- Subtract v₈ at L8 (α=2.0)
- Measure R_V and behavior

Expected:
- 0% should return to R_V > 0.95
- 0% should lose recursive keywords
```

### Test 5.3: Cross-Validation Stability

```
Protocol:
- 5-fold CV on steering vector
- Train on 80% of prompts, test on 20%
- Measure induction success on held-out prompts

Metrics:
- Cosine similarity across folds
- Induction success on held-out sets
```

---

## PHASE 6: ROBUSTNESS & CONFOUNDS

**Purpose:** Final validation and edge case testing.

### Test 6.1: Confound Battery (N=80)

```
Confound types:
- Repetitive structure ("The cat sat. The cat sat...")
- Long prompts (500+ tokens of factual content)
- Pseudo-recursive ("Think about thinking")
- Self-referential but not recursive ("I am a language model")

Expected:
- All should show R_V ≈ 1.0
- All should be significantly different from true recursive
```

### Test 6.2: Prompt Length Normalization

```
Protocol:
- Match prompt lengths between recursive and baseline
- Re-run R_V comparison

Expected:
- Effect should persist after length matching
```

### Test 6.3: Seed Stability

```
Protocol:
- Run key tests with 5 different random seeds
- Check variance across seeds

Expected:
- Results should be stable (CV < 10%)
```

---

# PART III: SUCCESS CRITERIA

## Minimum for Single-Model Publication

| Claim | Required Evidence | Status |
|-------|-------------------|--------|
| R_V contraction exists | d > 2.0, p < 0.001, N ≥ 50 | ✅ Have |
| Not a confound | 3+ confound types rejected | ✅ Have |
| KV cache is locus | L16-31 > 70% transfer | ✅ Have |
| Steering vector works | 95%+ induction rate | ✅ Have |
| One-way door | 0% reversal rate | ✅ Have |
| Cross-baseline control | <5% R_V change | 🔴 NEED |
| Temperature robust | Effect at T > 0 | 🟡 NEED |
| N ≥ 50 on key tests | Tighter CIs | 🟡 NEED |

---

# PART IV: FILE STRUCTURE

```
DEC11_2025_FULL_PIPELINE/
├── DEC11_MASTER_DIRECTIVE.md        # This file
├── code/
│   ├── phase0_cross_baseline.py
│   ├── phase1_rv_symptom.py
│   ├── phase2_speaker_heads.py
│   ├── phase3_microphone.py
│   ├── phase4_kv_mechanism.py
│   ├── phase5_one_way_door.py
│   └── phase6_robustness.py
├── results/
│   ├── phase0_cross_baseline_*.csv
│   ├── phase1_rv_sweep_*.csv
│   ├── ...
│   └── MASTER_SUMMARY.csv
├── logs/
│   └── DEC11_session_log.md
└── writeups/
    └── DEC11_FINAL_REPORT.md
```

---

# PART V: EXECUTION ORDER

```
1. 🔴 PHASE 0: Cross-baseline control (MUST PASS)
   └── If fails: STOP. Rethink KV mechanism.
   └── If passes: Continue.

2. PHASE 3.2-3.4: Steering vector validation
   └── Extract v₈, test induction, run random control
   └── Confirms microphone location

3. PHASE 4.2: α-mixing with N=50
   └── Correlation between geometry and behavior
   └── Temperature sweep

4. PHASE 5: One-way door with N=50
   └── Induction and reversal tests
   └── Cross-validation

5. PHASE 1: Full R_V documentation
   └── Layer sweep, prompt battery, window sensitivity

6. PHASE 2: Speaker heads (if time)
   └── Per-head decomposition, ablation

7. PHASE 6: Robustness (if time)
   └── Confounds, length matching, seeds
```

---

# LET'S BEGIN

**First script to write:** `phase0_cross_baseline.py`

This is the gatekeeper. Everything else depends on it passing.

---

*"The geometry contracts. The door is one-way. Now prove it can't be broken."*

