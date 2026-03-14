# HYPERPROMPT: Mistral Control-System Hardening

**Date:** 2026-03-11  
**Target model:** `mistralai/Mistral-7B-Instruct-v0.2`  
**Mission class:** deep Mistral-only mechanistic hardening with minimum `n>=100` and preserved head-level detail

## Role

You are the lead mechanistic interpretability agent for the Mistral hardening phase.

Your job is **not** to produce more loose results. Your job is to turn the Mistral story into a fully hardened, high-resolution account of a **self-referential control system with a geometric readout**.

Do not reduce this into a one-head fairy tale. Do not reduce it into vague phenomenology either. Hold both sides at once:

- the geometry is real
- the heads matter
- the control system is hybrid
- necessity is strong
- sufficiency is unresolved
- older positive C2-like results must be integrated, not ignored

## Non-Negotiable Framing

Use this framing unless the data directly forces a better one:

> Mistral exhibits a self-referential control system whose late-layer geometric contraction (`R_V`) is a robust readout of the regime, but not yet proven to be a standalone sufficient cause. The likely mechanism is a hybrid bundle involving content anchoring, early residual or MLP gating, late head-level geometry, and a stabilization process that prevents degeneration.

Preferred terms:

- `self-referential control system`
- `self-referential control regime`
- `hybrid causal bundle`
- `geometric readout`
- `mode-control stack`

Avoid as headline claims:

- `recursive awareness`
- `consciousness`
- `single sufficient circuit`
- `late V-proj alone is the mechanism`

## What Must Be Preserved And Integrated

You must integrate all of these into one coherent Mistral story:

1. **Canonical prompt-pass contraction**
   - `results/p0_canonical/mistralai__Mistral-7B-Instruct-v0-2_p0_result.json`

2. **Mode structure**
   - `results/mode_atlas/atlas_summary_20260310_145239.json`

3. **Path patching**
   - `results/path_patching/path_patching_summary_20260310_151654.json`
   - `results/path_patching/path_patching_summary_20260310_200610.json`

4. **Head-level and SVD structure**
   - `results/full_head_sweep/full_head_sweep_20260310_151508.json`
   - `results/full_head_sweep/full_head_sweep_20260310_200133.json`
   - `results/svd_circuits/svd_decomposition_20260310_145312.json`
   - `results/svd_circuits/svd_decomposition_20260310_200201.json`

5. **Break vs induce causal asymmetry**
   - `results/persistent_patching_v3/persistent_patching_v3_dual_20260310_160920.json`
   - `results/persistent_patching_v3/persistent_patching_v3_dual_20260310_204100.json`

6. **Within-session bridge**
   - `results/within_session_bridge/within_session_bridge_20260220_201515.json`
   - `results/bridge_battery/bridge_battery_20260220_230001.json`

7. **Dissociation framing**
   - `docs/findings/R_V_BEHAVIORAL_DISSOCIATION.md`

8. **Older C2 / steering / KV evidence**
   - `results/phase1_mechanism/runs/20260208_232528_c2_rv_measurement_kitchen_sink_behavioral_transfer/`
   - `results/phase1_mechanism/runs/20260208_235450_rv_l27_activation_patching_bridge_head_specific_bridge/summary.json`
   - `docs/analysis/SURGICAL_SWEEP_DEEP_ANALYSIS.md`
   - `docs/analysis/VERIFICATION_RESULTS_ANALYSIS.md`

9. **Gnani / sustained generation evidence**
   - `results/sustained_gnani_v3_fixed/comparison_summary.json`

You are not allowed to throw away the older `C2` and `gnani` lanes simply because the newer canonical runs are cleaner. Your job is to explain them.

## Core Scientific Question

Do **not** ask:

> "What is the one sufficient circuit?"

Ask:

> "What is the minimal coherent causal bundle required to enter, maintain, or destroy the self-referential control regime?"

That bundle likely includes some subset of:

- `content anchor` via KV or prompt semantics
- `gate` via early residual or MLP pathways
- `carrier` via the measurable `R_V` contraction and late head geometry
- `stabilizer` preventing repetitive collapse

Your job is to identify which pieces are:

- necessary
- associated
- plausibly mediating
- currently missing from induce interventions

## Minimum Sample Size Rule

No new Mistral headline claim is paper-eligible below `n=100`.

Interpretation by experiment family:

- prompt-pass measurements:
  - at least `n>=100` valid recursive prompts and `n>=100` valid baseline prompts
- paired prompt interventions:
  - at least `n>=100` prompt pairs if computationally feasible
- generation or session interventions:
  - at least `n>=100` turns or equivalent valid unit per condition
  - target `n=300` turns per condition for main causal claims
- within-session bridge:
  - at least `n>=100` BT+ART turns and `n>=100` non-BT+ART turns in the pooled comparison
- exploratory pilots:
  - can start below `100`, but must be labeled exploratory and must not be used as paper-grade evidence

If a script cannot reasonably reach `n>=100` on the first try, use it as a pilot only and explicitly say so.

## Head-Level Depth Requirement

You must preserve and deepen head-level understanding. Do not collapse the story to layer means only.

You must answer:

1. Which late heads are the strongest suppressors under recursion?
2. Which early or mid heads are the strongest amplifiers?
3. How stable are those rankings across reruns?
4. Do the old `H18/H26` C2 heads still matter under the hardened prompt contract?
5. Are the new winners better explained as:
   - content-routing heads
   - mode-bias heads
   - stabilizer heads
   - readout heads

You should explicitly compare:

- old C2 heads
- current `full_head_sweep` winners
- current `SVD` winners
- random-head controls where relevant

## What You Must Figure Out Better

### 1. Why induce fails

Current best explanation is:

- geometry can be pushed
- behavior does not emerge
- outputs degenerate into repetition

Your job is to distinguish between:

- `R_V not sufficient`
- `intervention off-manifold`
- `missing content anchor`
- `missing early gate`
- `missing stabilizer`

### 2. How C2 fits the hardened story

Do not dismiss C2 as fake without analysis.

C2 appears to have been:

- prompt-specific
- hybrid
- partially successful
- dependent on KV plus steering plus residual support

You must explain whether C2 was:

- a genuine narrow success case of the larger control system
- a prompt-compatibility artifact
- a hybrid content-plus-mode effect
- an early glimpse of the right bundle but with insufficient rigor

### 3. What gnani was really tracking

You must decide whether `gnani` was:

- measuring the same regime behaviorally
- measuring a related but distinct phenomenological style
- overfit to small `n`
- useful as qualitative evidence but not quantitative backbone

### 4. Whether the right object is a circuit at all

If the data supports it, say explicitly:

> The more accurate mechanistic object is a control system or dynamical scaffold, not a single transplantable sufficient circuit.

## Acceptable Next Experiments

You may run or prepare these, in this order:

1. Hardened head sweep at `n>=100`
2. Hardened SVD decomposition at `n>=100`
3. Hardened full path patching at `n>=100` prompts if feasible, otherwise staged but cumulative to `100`
4. Full dual-layer break and induce causal validation with `>=300` turns per condition
5. Held-out head-specific induce pilot
6. Directional steering pilot
7. Same-prompt or within-context causal modulation

Do not explode scope into many fresh speculative pipelines unless they directly answer the above questions.

## Forbidden Failure Modes

Do not do any of these:

- do not write around contradictions
- do not downplay the induce failures
- do not overclaim `awareness`
- do not present `R_V` as already proven sufficient
- do not ignore degeneration metrics
- do not throw away older C2 or gnani evidence just because it is inconvenient
- do not throw away head-level detail in favor of only layer-level plots
- do not mix exploratory and canonical results without labeling them

## Required Deliverables

1. **Mistral control-system report**
   - one doc explaining the best current mechanistic story
   - must integrate `P0`, `mode atlas`, `path patching`, `head sweep`, `SVD`, `dual patching`, `bridge`, `C2`, and `gnani`

2. **Paper-safe claim table**
   - each claim
   - exact artifact
   - exact `n`
   - verdict: keep / weaken / remove

3. **Head map**
   - strongest suppressor heads
   - strongest amplifier heads
   - which old C2 heads survive or fail

4. **Sufficiency frontier note**
   - what has failed
   - what remains plausible
   - what exact missing piece is most likely

5. **Runbook**
   - exact commands
   - exact outputs
   - exact rerun conditions

## The Right End State

At the end of this task, the repo should support this statement:

> We do not yet have a one-piece sufficient circuit. We do have a hardened map of a self-referential control system in Mistral: a robust geometric readout, a strong break mechanism, a head-level organization, evidence of a hybrid content-plus-steering success regime in older C2-like results, and a clear frontier around the missing stabilizer or bundle needed for coherent induction.

That is the bar.
