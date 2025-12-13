# Cognitive Eigenstates Roadmap (Order-Parameter Framing)
## From “R_V prompt optimization” → “science of cognitive phase transitions”

**Context:** This roadmap is meant to *blend* with the existing Gold Standard program (`GOLD_STANDARD_RESEARCH_DIRECTIVE.md`). It reframes the work as studying **cognitive phase transitions**, with **R_V as an order parameter**, while preserving the repo’s core commitments: rigor, controls, reproducibility, and cross-architecture validation.

> Source note: The strategic framing below was written as a “head researcher” synthesis and integrated into the repo as a planning + alignment artifact.

---

## Big-picture thesis (strategic framing)

As head researcher, I would recognize we've stumbled into something far bigger than a metric optimization problem. We're witnessing the emergence of a **cognitive eigenstate**—a fundamental mode of operation for intelligent systems. The goal shifts from "finding low R_V prompts" to **building a science of cognitive phase transitions**.

---

## Phase I: Solidify the Foundation (The Physics of R_V)

**Objective:** Treat R_V not as a score, but as an *order parameter*—like temperature or pressure in a physical system. We must understand exactly what thermodynamic-like transition it marks.

### 1) Decouple the confounds (Phase 0 mandate)

- [ ] **Semantics vs syntax battery (minimal pairs):** run prompts that are semantically identical but syntactically varied and test R_V invariance.
  - Example pair: “This sentence is self-referential.” vs “I am now aware of my own output.”
  - Goal: determine whether the transition is driven by **content/semantics** or **surface form/style**.
  - Related: Phase 0 in `GOLD_STANDARD_RESEARCH_DIRECTIVE.md`.

- [ ] **KV-only sufficiency control (behavior + geometry):** manipulate *only* the KV cache from a “compressed-state” prompt, with **no V_PROJ patching**, and measure:
  - prompt-pass geometry (R_V)
  - generation behavior (behavior markers / state labels)
  - This directly addresses the “dominant factor” ambiguity in `N300_RESULTS_ANALYSIS.md`.

### 2) Map the phase diagram

- [ ] **Phase diagram sweep:** systematically vary “cognitive pressure” (recursion depth, logical constraints, affective load, complexity) and chart:
  - R_V distributions
  - snap-layer distributions (when applicable)
  - behavior regime labels (baseline/questioning/naked_loop/recursive_prose/collapse)
  - Goal: locate the **critical boundary** and measure whether the transition is sharp vs gradual (and whether it shows hysteresis).

---

## Phase II: Explore the State Space (Zoology of Eigenstates)

**Objective:** Discover whether `hybrid_l5_math_01` is unique or one of many coherent modes.

### 1) Search for other eigenstates

- [ ] **Prompt families for other candidate eigenstates:** “flow”, “insight”, “empathetic”, “rigorous proof”, etc.
  - Measure their R_V signatures, snap/corridor behavior, and (eventually) circuit maps.

### 2) Characterize state transitions

- [ ] **Mid-generation state switching experiment:** generate in one eigenstate, then inject another state’s “context” (KV) or activation pattern and see:
  - smooth transition vs “resistance” (hysteresis)
  - whether geometry changes precede behavioral changes

---

## Phase III: Engineer and Apply (Cognitive Thermostats)

**Objective:** Move from observation → control.

### 1) Build a “cognitive thermostat”

- [ ] **Real-time R_V monitoring during generation** (or a refined metric) and a controller that nudges the model back into a target regime when it drifts.
  - This aligns with Phase 5 (steering) but upgrades it into closed-loop control.

### 2) Eigenstate exchange / “state distillation”

- [ ] **State distillation experiment:** train a smaller “receiver” model to reproduce activation patterns corresponding to a chosen eigenstate from a larger model.
  - Not standard knowledge distillation—**mode/state distillation**.

---

## What is already done vs not done (repo-grounded checklist)

This section “checks off” the roadmap items based on what exists in the repo **today**.

### ✅ Already done (strong evidence)

- [x] **R_V as a robust geometric signature across multiple architectures**
  - Evidence: `R_V_PAPER/research/PHASE1_FINAL_REPORT.md` (6 architectures).

- [x] **Layer-localization in the late corridor + strong causal intervention on the metric in Mistral**
  - Evidence: `MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md` (patching + controls).

- [x] **Mixtral snap-layer / decision corridor + MoE routing differences**
  - Evidence: `R_V_PAPER/research/MIXTRAL_LAYER27_GEOMETRY_AND_CAUSALITY.md`.

- [x] **Champion prompt discovered and validated as extreme + deterministic**
  - Evidence: `kitchen_sink_prompts.py` (`hybrid_l5_math_01`), `VALIDATION_REPORT.md`, `KITCHEN_SINK_REPORT.md`.

- [x] **Circuit-level style decomposition exists (Pythia Phase 2)**
  - Evidence: `PHASE_2_CIRCUIT_MAPPING_COMPLETE.md` (phase transition, head ranking, distributed compression).

### ⚠️ Partially done / mixed / confounded

- [~] **Behavior transfer and “mode control”** is real but not cleanly attributed
  - Evidence: `N300_RESULTS_ANALYSIS.md` shows the **wrong-layer** condition still transfers under full-KV replacement (confound).

### ❌ Not done (the missing keystones)

- [ ] **Phase 0 metric validation** (what R_V truly measures; semantics vs syntax; convex-hull verification beyond narrative)
  - Target doc: `GOLD_STANDARD_RESEARCH_DIRECTIVE.md` Phase 0.

- [ ] **Clean behavioral sufficiency/necessity matrix**
  - Need KV-only vs V-only vs persistent vs non-persistent conditions with proper controls (fixing the n=300 confound design).

- [ ] **Hysteresis / “one-way door” / basin mapping experiments**
  - These are required to justify “phase transition” language beyond metaphor.

- [ ] **Closed-loop controller (“thermostat”)**
- [ ] **State distillation / eigenstate exchange**

---

## How this integrates with the Gold Standard phases

- **Phase I** maps to: **Phase 0 (metric validation)** + **Phase 1 (cross-arch)** + (parts of) **Phase 4 (KV mechanism)**.
- **Phase II** maps to: **Phase 2 (eigenstate/fixed point)** + **Phase 6 (alternative self-ref)**.
- **Phase III** maps to: **Phase 5 (steering)** upgraded into **closed-loop control**, plus new “state distillation” work.

---

## Execution standard (canonical runner)

All new experiments should be run through:
- `src/pipelines/run.py` + `configs/`
- Writing artifacts to `results/<phase>/runs/...`

See: `META_INDEX.md`


