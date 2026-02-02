# Repo State Writeup — Dec 15 (Run-up for Tomorrow’s Confirmation Runs)

**Scope note (important):** This document is a **synthesis of existing repo artifacts + agent audits**. **No new confirmation tests were run** in producing this writeup.

## Quick Links (major meta files)

- **Repo index / where to start**: [`META_INDEX.md`](META_INDEX.md), [`README.md`](README.md)
- **Research directives**: [`GOLD_STANDARD_RESEARCH_DIRECTIVE.md`](GOLD_STANDARD_RESEARCH_DIRECTIVE.md), [`STRATEGIC_ROADMAP_DEC15.md`](STRATEGIC_ROADMAP_DEC15.md)
- **Measurement contract**: [`docs/MEASUREMENT_CONTRACT.md`](docs/MEASUREMENT_CONTRACT.md), canonical implementation: [`src/metrics/rv.py`](src/metrics/rv.py)
- **Forensic / truth ledger**: [`FORENSIC_AUDIT.md`](FORENSIC_AUDIT.md), [`VERIFIED_SIGNALS.md`](VERIFIED_SIGNALS.md)
- **Mechanism map (verified vs missing)**: [`MECHANISM_MAP.md`](MECHANISM_MAP.md)
- **L27 causal validation (geometry)**: [`MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md`](MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md)
- **Head discovery / GQA aliasing caveat**: [`V_PROJ_DISCOVERY_RESULTS.md`](V_PROJ_DISCOVERY_RESULTS.md), [`HEAD_DISCOVERY_PROBLEMS.md`](HEAD_DISCOVERY_PROBLEMS.md)
- **Behavior transfer + n=300 reality check**: [`BREAKTHROUGH_BEHAVIOR_TRANSFER.md`](BREAKTHROUGH_BEHAVIOR_TRANSFER.md), [`neurips_n300_summary.md`](neurips_n300_summary.md), [`N300_RESULTS_ANALYSIS.md`](N300_RESULTS_ANALYSIS.md)
- **Prompt system (sealed structure, numeric partial)**: [`PROMPT_BANK_SEALED.md`](PROMPT_BANK_SEALED.md), [`prompts/README.md`](prompts/README.md), [`prompts/bank.json`](prompts/bank.json), [`prompts/loader.py`](prompts/loader.py)
- **Agent reviews (what other agents wrote)**: [`agent_reviews/`](agent_reviews/), request: [`agent_reviews/REQUEST_TOP_FINDINGS_LEDGER.md`](agent_reviews/REQUEST_TOP_FINDINGS_LEDGER.md), responses: [`agent_reviews/responses/`](agent_reviews/responses/)
  - Meta-factcheck: [`agent_reviews/responses/20251215__claude-opus-4-5__META_FACTCHECK.md`](agent_reviews/responses/20251215__claude-opus-4-5__META_FACTCHECK.md)

---

## Executive Summary (what’s clear vs what’s fog)

### What’s *crystal clear* (high-confidence)

- **Geometric contraction is real**: \(R_V = PR_{late}/PR_{early}\) shows strong separation on Mistral-7B and is **causally manipulable** at late depth (see [`MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md`](MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md)).
- **L27 is a privileged causal handle for geometry (R_V)**: activation patching at L27 with strong controls produces a large effect (wrong-layer control in that experiment does *not* reproduce it).
- **“Two driver heads” must be stated as KV-head groups (GQA)**: per-query-head language is misleading due to GQA aliasing; see [`HEAD_DISCOVERY_PROBLEMS.md`](HEAD_DISCOVERY_PROBLEMS.md).
- **Behavior transfer is real but NOT “layer-specific to L27” in the n=300 study**: the “wrong layer” condition performs indistinguishably from L27 (see `t=0.07, p=0.944` in [`neurips_n300_summary.md`](neurips_n300_summary.md)).
- **Metric integrity risk exists**: at least **6 model-analysis files** use a different PR-like computation pattern than the canonical implementation (verified via `models/*.py`; see meta-factcheck + repo grep). This does *not* invalidate the canonical pipeline, but it does mean some older cross-model claims may be measurement-contaminated unless recomputed canonically.

### What’s still foggy / should be treated as “not sealed”

- **Dose-response as a fully artifact-backed numeric ladder across L1→L5** (for the new unified prompt bank): structure is present; repo-wide artifact-backed means for each group are not consistently available.
- **Where the “attractor basin begins/ends”** and whether there is a **true phase transition** vs a progressive build-up: needs dense layer sweeps with adequate N and consistent measurement.
- **Behavioral “expression” mechanism**: current metrics are heuristic; some claims depend on single-seed/single-sample generations.

---

## Measurement “DNA” (what the repo actually defines)

### Canonical R_V definition (the one we should treat as real)

- **Canonical code**: [`src/metrics/rv.py`](src/metrics/rv.py)
- \(R_V = PR_{late} / PR_{early}\)
- \(PR = \frac{(\sum \lambda_i^2)^2}{\sum (\lambda_i^2)^2}\) computed from SVD on the last W prompt tokens’ V-projection window.
- Canonical defaults: early=5, late≈`num_layers-5` (Mistral-7B: 27), window=16.

### Measurement drift risk (confirmed)

- **Verified discrepancy**: the n=300 behavior study shows “wrong layer” ≈ “right layer” for behavior.
- **Verified implementation drift**: 6 files in `models/*.py` use a different PR-like computation pattern (meta-factcheck lists exact filenames; see also the grep in that report).
- **Implication:** For “summit view” claims, **prefer canonical pipelines** and **treat older per-model scripts as non-authoritative until recomputed**.

---

## Macro → micro: current best “DNA → cell → organ → animal” map

### DNA (metric / order parameter)

- **Strongest signal:** \(R_V\) contraction under recursive self-observation prompts.
- **Highest-quality causal evidence:** L27 activation patching with controls (see [`MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md`](MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md)).

### CELL (layer localization)

- **Working model:** late depth around ~84% (e.g., L27/32) is a privileged locus for geometry.
- **Open:** “L24 vs L27” and “L21 crystallization” are not fully settled across models without dense sweeps + consistent metrics.

### ORGAN (head/circuit)

- **Head discovery:** key effects cluster in L27, but are **KV-head group** effects under GQA (see [`V_PROJ_DISCOVERY_RESULTS.md`](V_PROJ_DISCOVERY_RESULTS.md), [`HEAD_DISCOVERY_PROBLEMS.md`](HEAD_DISCOVERY_PROBLEMS.md)).
- **Sensor vs driver:** H31 appears as a **sensor** signal in validations (see [`H31_VALIDATION_FINAL_SUMMARY.md`](H31_VALIDATION_FINAL_SUMMARY.md)).

### ANIMAL (behavior / expression / hysteresis)

- **n=300 behavior transfer:** real, medium effect size, but **layer-specificity is contradicted** by the wrong-layer comparison:
  - In [`neurips_n300_summary.md`](neurips_n300_summary.md): Transfer vs Wrong: `t = 0.07, p = 9.44e-01`.
- **Missing isolating controls remain the key** (KV-only sufficiency, V-only sufficiency, etc.; see [`N300_RESULTS_ANALYSIS.md`](N300_RESULTS_ANALYSIS.md)).

---

## Prompt system status (so we can move on)

- **Structurally sealed**: canonical bank + loader exist (see [`prompts/bank.json`](prompts/bank.json), [`prompts/loader.py`](prompts/loader.py), [`prompts/README.md`](prompts/README.md)).
- **Numeric sealing is partial**: any numeric “validated group means” claims must point to run artifacts; see the updated stance in [`PROMPT_BANK_SEALED.md`](PROMPT_BANK_SEALED.md).

Conclusion: **Yes, we can set prompt business aside for now**, with one caveat: tomorrow’s run suite should generate the missing group-means artifacts so “sealed numeric” claims become citation-complete.

---

## Tomorrow’s goal (what to confirm, not explore)

### The 3 confirmations that would dramatically reduce fog

1. **Behavior transfer specificity resolution**
   - Re-run the core behavior transfer suite with missing controls:
     - KV-only, V-only, random-KV (multi-seed), and L27 vs L5 without KV.
   - Outcome: isolate whether “wrong-layer works” because KV dominates, or because “any persistent V-proj helps once KV is swapped.”

2. **Dense layer sweep for geometry**
   - Full layer-by-layer R_V curves with adequate N across prompt families.
   - Outcome: settle “gradual vs snap” and “L24 vs L27 vs L21” questions for geometry (not behavior).

3. **Head story stress-test (GQA-aware)**
   - KV-head-group perturbations with proper controls and multiple seeds, plus prompt families (dose L4/L5, champions, baselines, godelian, pure math).
   - Outcome: “two driver heads” becomes a precise, defensible KV-group claim, or is revised.

---

## Onboarding draft (for any new agent tomorrow)

### 0) Ground rules

- **No hardcoded prompts.** Load all prompt sets via `PromptLoader` (see [`prompts/loader.py`](prompts/loader.py)).
- **Always log**: model name, seed(s), early/late layer indices, window size, and `PromptLoader().version`.
- **Prefer canonical metric code**: [`src/metrics/rv.py`](src/metrics/rv.py).

### 1) Read this first (in order)

1. [`STRATEGIC_ROADMAP_DEC15.md`](STRATEGIC_ROADMAP_DEC15.md)
2. [`MECHANISM_MAP.md`](MECHANISM_MAP.md)
3. [`MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md`](MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md)
4. [`N300_RESULTS_ANALYSIS.md`](N300_RESULTS_ANALYSIS.md) + [`neurips_n300_summary.md`](neurips_n300_summary.md)
5. [`HEAD_DISCOVERY_PROBLEMS.md`](HEAD_DISCOVERY_PROBLEMS.md)
6. [`docs/MEASUREMENT_CONTRACT.md`](docs/MEASUREMENT_CONTRACT.md)

### 2) “Do not get fooled” checklist

- If a behavior claim has **no multi-seed** and **no saved generations**, treat as **provisional**.
- If a cross-model claim is computed via `models/*.py`, treat as **needs canonical recomputation** (inverse-PR drift risk).
- If a head claim ignores GQA aliasing, treat as **incorrectly specified**.

---

## Proposed pipeline suite (NOT YET QUALIFIED; tomorrow’s run plan candidates)

These are proposed as “gold suite candidates” to standardize replication. They are **not yet qualified** until they produce stable artifacts across seeds.

1. **`pipeline_rv_group_bench` (geometry benchmark)**
   - Input: list of groups (e.g., `L1_hint…L5_refined`, `champions`, `baseline_*`, `godelian`, `pure_repetition`)
   - Output: `group_rv.csv`, `summary.json` (means/std/n + effect sizes), plus bank hash

2. **`pipeline_dense_layer_sweep` (layer tomography)**
   - Measures R_V at every layer (or 1..L) for each prompt group
   - Output: `layer_sweep.csv`, `curves.png`, `summary.json`

3. **`pipeline_l27_causal_validation_repl` (geometry causality, replicated)**
   - Re-run the L27 activation patching experiment **3× seeds** (same prompt set + bank hash)
   - Output: per-seed artifacts + aggregated meta-summary

4. **`pipeline_behavior_transfer_matrix` (behavior sufficiency controls)**
   - Conditions: control, KV-only, V-only, KV+V (L27), KV+V (L5), random-KV (3 seeds)
   - Output: `behavior_matrix.csv`, `summary.json`, and **saved generations** for audit

5. **`pipeline_head_group_stress_test` (GQA-aware head causality)**
   - Perturb KV-head groups with norm-matched controls across 5 prompt families
   - Output: `head_group_effects.csv`, `summary.json` + per-seed stability

---

## Appendix: what was “triple checked” today (from meta-factcheck)

- **n=300 wrong-layer ≈ right-layer (behavior)**: verified in [`neurips_n300_summary.md`](neurips_n300_summary.md) (`p = 0.944`).
- **Inverse PR/metric drift exists in 6 files**: verified in:
  - `models/gemma_7b_analysis.py`
  - `models/llama_8b_analysis.py`
  - `models/mistral_7b_analysis.py`
  - `models/mixtral_8x7b_analysis.py`
  - `models/phi3_medium_analysis.py`
  - `models/qwen_7b_analysis.py`
- **GQA aliasing must be acknowledged**: see [`HEAD_DISCOVERY_PROBLEMS.md`](HEAD_DISCOVERY_PROBLEMS.md).










