# META INDEX: Telos + Navigation Map
## What the “meta files” are, how they fit, and how to execute the program

This repository contains both:
- **Living code** (`src/`, `prompts/`, `configs/`) that should be runnable and reproducible.
- **Meta-governance** (directives, orchestration, status trackers, and history) that defines *why* we do things and *what counts as truth*.

The goal of this document is to stitch the meta layer into a single map.

---

## The Telos (North Star)

**We are not here to publish quickly.** We are here to build a defensible scientific claim about:
- **A geometric signature** of recursive self-observation (R_V contraction in value-space),
- Its **scope** (cross-architecture and scaling),
- Its **mechanism** (which layers/heads/paths and why),
- With **reproducibility as law**.

The governing “constitution” is:
- `GOLD_STANDARD_RESEARCH_DIRECTIVE.md`
- Strategic framing supplement:
  - `COGNITIVE_EIGENSTATE_ROADMAP.md` (treat R_V as an order parameter; roadmap + done-vs-missing checklist)

---

## How the meta files fit together (top-down)

### 1) Constitution (What counts as success?)
- **`GOLD_STANDARD_RESEARCH_DIRECTIVE.md`**
  - Defines the research program and phases (0→6).
  - Defines falsifiable predictions and publication criteria.

### 2) Agent Operating System (How to behave while executing?)
- **`AGENT_PROMPT_GOLD_STANDARD.md`**
  - Encodes the same telos as the directive but in “do this now” terms.
  - Defines data standards and minimum statistical expectations.

### 3) Orchestration / Terraforming (How to turn messy workshop → lab?)
- **`META_ORCHESTRATION_DEC11.md`**
  - Explicitly calls out the transition from exploratory chaos → scientific rigor.
  - Defines the desired phase-based structure for results and experiments.

### 4) Status trackers (What is running / blocked right now?)
- **`EXPERIMENT_STATUS.md`**
  - Operational status for the n=300 NeurIPS run (what/where/how to monitor).
- **`UNIFIED_TEST_STATUS.md`**
  - Status of V_PROJ vs head-level vs residual-level patching comparisons.

### 5) Method history (What was claimed vs what was actually executed?)
- **`KV_PATCHING_HISTORY.md`**
  - Reconciles “KV cache patching” terminology and what was actually patched.
  - Highlights control-design pitfalls and what’s still untested.

### 6) Repository architecture decisions (What is the canonical codebase?)
- **`REFOUNDATION_COMPLETE.md`**
  - Documents the `src/` + `prompts/` refoundation and what is considered canonical.

---

## The canonical execution path (how you run truth)

### Canonical runner
The repo now has a single blessed entrypoint:
- `src/pipelines/run.py`

It runs experiments from JSON configs:
- `configs/phase1_existence.json`
- `configs/rv_l27_causal_validation.json`

Each run writes a timestamped folder:
- `results/<phase>/runs/<timestamp>_<experiment>.../`
  - `config.json` (exact config snapshot)
  - `summary.json` (machine-readable summary)
  - `report.md` (human-readable summary)
  - experiment-specific artifacts (CSV, plots, etc.)

### Why this matters to the telos
The Gold Standard directive demands:
- **Reproducibility**
- **Standardized prompts**
- **Explicit controls**
- **Recorded parameters**

The canonical runner is the *mechanism* that turns those into enforceable reality:
- Configs become the “pre-registration lite.”
- Output folders become the immutable evidence trail.

---

## Where the runner fits into the phases

### Phase 1 (Cross-architecture R_V validation)
- Use `configs/phase1_existence.json` as the basic existence proof template.
- Duplicate config per model/size tier; keep everything else constant.

### Phase 1 (Causal validation at target layer)
- Use `configs/rv_l27_causal_validation.json` to enforce:
  - main patch (recursive V)
  - random norm-matched control
  - shuffled token control
  - wrong-layer control

### Phase 0/2/3/4/5/6 (not yet migrated)
These should be added as new configs + pipelines (one per phase), so the meta program is runnable end-to-end.

---

## “How to use this map” (practical)

If you are confused, do this:
1) Read `README.md` (what exists).
2) Run `20_MINUTE_REPRODUCIBILITY_PROTOCOL.md` (certificate of environment + phenomenon).
3) Read `GOLD_STANDARD_RESEARCH_DIRECTIVE.md` (telos).
4) Execute via `src/pipelines/run.py` + `configs/` and only trust artifacts under `results/<phase>/runs/`.


