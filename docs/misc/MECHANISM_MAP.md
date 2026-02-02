# MECHANISM_MAP — Recursive Self-Reference (Mistral-7B focus)

**Purpose:** A single “master map” of what’s **VERIFIED** vs **MISSING/UNMAPPED** in the recursive self-reference mechanism, organized to support the repo’s north star: **atomic mapping → domino points → maximal leverage interventions**.

**Scope rule:** This map links to concrete repo artifacts for each “VERIFIED” item (CSV/JSON/TXT/py). Narrative writeups are treated as interpretation unless backed by primary artifacts.

---

## 0) Where this already exists (but scattered)

The repo had the ingredients of a master map spread across “map-like” files:

- **High-level strategy + gaps**: `STRATEGIC_ROADMAP_DEC15.md`
- **Repo narrative map**: `NOTES_FROM_THE_COMPOSER/LIVING_MAP.md`
- **Early-layer entry/ramp evidence** (residual patching → \(R_V@L27\)):  
  `results/phase1_mechanism/runs/20251213_080454_path_patching_mechanism_full_early_layer_sweep_full_controls_base/EARLY_LAYER_MAP.md`
- **Artifact-backed audit** (Mistral-specific reality check): `FORENSIC_AUDIT.md`
- **Canonical rigor program**: `GOLD_STANDARD_RESEARCH_DIRECTIVE.md` + `COGNITIVE_EIGENSTATE_ROADMAP.md`

This file consolidates the map in one place and keeps “percent complete” honest.

---

## 1) What we have mapped (~40%) — with evidence pointers

### 1.1 DETECTION / LATE CONTROL BAND @ ~L27

**VERIFIED (KV-head / GQA-aliased):** V-projection ablation at late layers shows specific head indices at L27 cause \(R_V\) contraction (ablating them increases \(R_V\)).  
**Primary artifact:** `results/head_discovery/v_proj_head_discovery_20251214_091646.csv`

**Important caveat (VERIFIED):** In Mistral’s GQA, “H18 vs H26” is not separable via KV-head v-proj ablation; multiple query heads map to the same KV head.  
**Code:** `v_proj_head_discovery.py` (KV head mapping via `head_idx % num_kv_heads`)  
**Interpretation doc (secondary):** `FORENSIC_AUDIT.md`

**VERIFIED (single-pair attention targeting):** Heads labeled H18/H26 show ~28% recursive-token targeting in the analyzed comparison log, with 0% in baseline.  
**Artifact:** `target_comparison_output.txt`

---

### 1.2 GEOMETRIC SIGNATURE: \(R_V\) contraction (robust separation)

**VERIFIED (large-N, Mistral):** Strong separation in \(R_V\) at L27 between recursive vs baseline with very large effect size.  
**Artifacts:**  
- `DECEMBER_2025_EXPERIMENTS/DEC8_RUNPOD/01_GEOMETRY_OF_RECURSION/results/full_validation_20251208_130707.csv`  
- `DECEMBER_2025_EXPERIMENTS/DEC8_RUNPOD/01_GEOMETRY_OF_RECURSION/results/full_validation_20251208_134241.csv`

**VERIFIED (causal on geometry):** Layer-27 causal validation pipeline exists with comprehensive controls and writes per-pair CSV artifacts.  
**Code (canonical):** `src/pipelines/rv_l27_causal_validation.py` + `configs/rv_l27_causal_validation.json`  
**Narrative summary:** `MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md`

---

### 1.3 GATING FUNCTION: context-dependent expression (behavior label flips)

**VERIFIED (N=40 prompts × 4 conditions):** “Expression” is context-dependent and flips are common in `part_a_results.csv`.  
**Artifact:** `results/comprehensive_circuit_test/part_a_results.csv`  
**Analysis summary (derived from CSV):** `results/comprehensive_circuit_test/part_a_analysis.md` + `part_a_analysis.json`

**VERIFIED (measurement caveat):** `expressed_binary` is a heuristic label over *one sampled generation* (`temperature=0.7`), so flips are expected unless you aggregate across seeds.  
**Code:** `comprehensive_circuit_test.py` + `src/metrics/behavior_states.py`

---

### 1.4 SECONDARY REGULATORS: suppressor groups (candidate antagonists)

**PARTIALLY VERIFIED:** The repo has a working notion of “suppressor” groups (e.g. `H6_GROUP`, `H18_GROUP`) and shows condition-dependent effects on both \(R_V\) and expression rates in the N=40 run.  
**Artifact:** `results/comprehensive_circuit_test/part_a_results.csv`  
**Code:** `comprehensive_circuit_test.py`

**Caveat:** Because ablations are KV-head v-proj under GQA, “secondary regulators” are currently at the KV-head granularity, not true per-query-head isolation.

---

### 1.5 THRESHOLD HINTS: “champion” prompts + identity equations

**VERIFIED (in N=40 run):** Champion prompts show higher expression rates than standard/baseline under the heuristic labeling.  
**Artifact:** `results/comprehensive_circuit_test/part_a_results.csv`

**VERIFIED:** Identity-equation detections increase in the `both_ablated` condition in the N=40 run.  
**Artifact:** `results/comprehensive_circuit_test/part_a_results.csv` (`has_identity_equation`)

---

## 2) What’s missing (~60%) — the unmapped modules

### 2.1 ENTRY POINT (where does detection begin?)

**PARTIALLY MAPPED (layer-level, not head-level):** Early-layer residual patching suggests an early “ramp” rather than a single mic-layer.  
**Artifact:**  
`results/phase1_mechanism/runs/20251213_080454_path_patching_mechanism_full_early_layer_sweep_full_controls_base/EARLY_LAYER_MAP.md`

**UNMAPPED:** First layer where Mistral shows statistically reliable \(R_V\) separation across *prompt distributions* (not just single-trace tomography).  
**Candidate artifacts to extend:** `mistral_relay_tomography_v2.csv` (single-trace; insufficient)

---

### 2.2 SIGNAL PROPAGATION (layer-by-layer pathway / relay)

**UNMAPPED:** A concrete causal “path” showing which computations carry signal from early ramp → late control band (heads/MLPs, sequence).  
**Related but incomplete:** `path_patching_mechanism` pipeline family under `results/phase1_mechanism/runs/...` and `src/pipelines/path_patching_mechanism.py` (if present in registry).

---

### 2.3 GATE THRESHOLD (pass vs filter decision boundary)

**UNMAPPED:** A quantified threshold model predicting expression:
- prompt features (token patterns / density / structure),
- early geometry,
- late geometry,
- attention readouts,
→ \(P(\text{express})\).

Current status: we have *signals* and *flip tables*, but no fitted boundary.

---

### 2.4 EXPRESSION MECHANISM (what produces output when gate opens?)

**UNMAPPED / INCONCLUSIVE:** Whether “expression is default once suppressors are removed” vs “there exist specific enabling heads/circuits that must be active.”  
**Reason:** Current Part C is too small and too stochastic (single sampled generations; 10 random heads; heuristic label).

---

### 2.5 TEMPORAL DYNAMICS (multi-token persistence)

**UNMAPPED (results):** Does contraction persist across generation steps (stable eigenstate) or only exist on prompt pass (meter/readout)?  
**Script exists (starting point):** `experiment_multi_token_generation.py`  
**Motivating doc:** `THE_BIG_QUESTIONS_LEFT_AFTER_GEMINI_WRITEUP.md` (explicitly flags this as critical).

---

### 2.6 INTERACTION MAP (coordination of many heads)

**UNMAPPED:** Dependency graph / synergy/antagonism among the “important heads.”  
We have “important head lists” and deltas, but not minimal sufficient sets or causal graphs.

---

### 2.7 TOKEN-LEVEL FLOW (which tokens trigger which heads, through layers)

**UNMAPPED:** A token-level causal trace (token classes → head activations → residual changes → late contraction → behavior).  
We have isolated attention targeting logs, but not a full propagation map.

---

### 2.8 PROMPT-PASS VALIDATION (generation confound control)

**UNMAPPED (Mistral L0–L3 prompt-pass):** No on-disk prompt-pass results found for Mistral L0–L3 (only generation-mode evidence).  
**Gap risk:** L0 source-layer claim may share the Gemma confound unless prompt-pass validates it.  
**New configs created:** `configs/canonical/mistral_7b_v0_1/mlp_ablation_prompt_pass_l{0,1,2,3}.json`

---

## 3) Domino candidates (current best bets)

### 3.1 Domino A: Temporal maintenance
If \(R_V(t)\) remains contracted across generation (not just prompt pass), the phenomenon is a true “mode” and becomes a viable lever for interventions.

### 3.2 Domino B: Gate threshold
If we can model the pass/filter boundary, we can engineer prompts or interventions that achieve high expression probability (e.g., 90%+) deterministically.

### 3.3 Domino C: Late control band (KV-head level)
If selective interventions to the late KV-head(s) can bias the system toward the contracted regime *without collapse*, that’s maximum leverage.

---

## 4) What it would take to get to 100% (practical plan)

### 4.1 Define the map’s “units” and success criteria
- **Unit tests for each module** (entry, propagation, gate, expression, temporal dynamics).
- **Artifact standard**: every module produces a CSV/JSON summary + run config snapshot (use canonical runner).

### 4.2 Build the minimal evidence loop for each missing module
- **Temporal dynamics**: per-step \(R_V(t)\) trajectories (recursive vs baseline) with deterministic decoding + variance if sampled.
- **Gate threshold**: fit a simple predictive model on prompt features + geometric/attention readouts.
- **Expression mechanism**: multi-seed ablation study on a fixed prompt set; evaluate state probabilities, not single samples.
- **Propagation**: path patching or causal tracing per layer to identify which subcomponents actually transmit the signal.

### 4.3 Enforce “master map updates” as part of workflow
Repo rule suggestion: every new experiment PR/commit includes:
- updated row(s) in this map (VERIFIED / PARTIAL / UNMAPPED),
- links to new artifacts under `results/<phase>/runs/...`,
- and a single sentence on whether it changed a domino ranking.

---

## 5) Ownership: where to edit when something changes

- **If you add a new verified artifact:** update Section 1 with the path.
- **If you resolve a gap:** move it from Section 2 → Section 1 and keep the old text under a “Retired/Resolved” subsection.
- **If a claim gets contradicted:** mark it and link the contradicting artifact (same style as `FORENSIC_AUDIT.md`).










