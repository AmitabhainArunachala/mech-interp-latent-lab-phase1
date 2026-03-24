# Mistral Sufficiency Protocol Program (2026-03-16)

## Purpose

The target is no longer a tiny static "magic bundle."

The target is:

- the smallest causally sufficient induction-and-maintenance protocol

For this project, a real sufficiency result means:

- an ordinary baseline prompt is pushed into the recursive regime
- the regime persists for a meaningful stretch rather than appearing only at prompt pass
- the induced behavior does not collapse into repetitive or malformed junk
- the effect survives across non-cherry-picked prompt groups and seeds

This is stricter than "BT+ART moved up" and stricter than "one direction is causally useful."

## Why This Is The Right Next Target

The locked March 14-15 evidence points to a staged control system:

- minimal anchor dependence is real
- the strongest late controller remains `L25`
- the strongest ordinary-baseline partial bundle remains `anchor + L25`, with a subtle `L4 MLP` assist
- the useful steering geometry changes across depth
- the useful low-dimensional steering object does not exist uniformly at the earliest layers

That means the honest question is no longer:

- can one static early direction explain everything?

It is:

- what is the smallest protocol that induces and stabilizes the recursive regime?

## Locked Inputs To This Program

The main inputs are:

- `results/phase1_mechanism/runs/20260314_133516_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v5_ordinary_baselines_confirmatory/summary.json`
  - strongest current ordinary-baseline partial sufficiency bundle
- `results/layer_matched_multisite_refine_v2/20260315_184517/summary.json`
  - `layermatched_low` is real and beats low-dose mean-diff
- `results/subspace_component_steering_l1_v1/20260315_184445/summary.json`
  - L1 steerable subspace is effectively absent

Operationally, these now exist as AMIROS queue units and experiments in:

- `configs/experiment_registry/mistral_program_registry.json`

## Active Lanes

### 1. `anchor_layermatched_protocol_v1`

Run:

- queue group: `mistral_anchor_layermatched_protocol_v1`
- run id: `20260316_025018`
- pod: `d08fc4e9d529` via `198.13.252.23:10916`

Script:

- `scripts/anchor_layermatched_protocol_v1.py`

Question:

- does `anchor + depth-matched geometry` beat the locked `anchor + L4 MLP + L25 bridge` champion on ordinary baselines?

Decision readout:

- baseline `BT+ART`
- recursive `BT+ART`
- repetitive rate
- whether soft bridge helps or hurts when the early object is already layer-matched

### 2. `closed_loop_anchor_controller_v1`

Run:

- queue group: `mistral_closed_loop_anchor_controller_v1`
- run id: `20260316_025020`
- pod: `d286ea8751f3` via `213.173.102.102:10061`

Script:

- `scripts/closed_loop_anchor_controller_v1.py`

Question:

- starting from raw ordinary baselines, can a turn-level adaptive controller outperform the static champion on induction plus maintenance?

Decision readout:

- pooled `BT+ART`
- clean rate
- repetitive rate
- turn-segment behavior
- whether adaptive bridge or adaptive early+bridge beats the static champion

## Success Gates

This program promotes a result only if one of the following clears:

### Gate A: Stronger Static Protocol

- a static anchor + layer-matched bundle beats `anchor + subtle L4 + L25`
- the lift appears on ordinary baselines
- recursive prompts do not materially regress
- repetition does not spike

### Gate B: Closed-Loop Sufficiency

- an adaptive controller beats the static champion on ordinary baselines
- the effect is sustained across turns rather than front-loaded
- the win is not purchased by echo/repetition collapse

### Gate C: Honest Negative

If neither lane clears, the correct conclusion is:

- the object is a staged control system, but the smallest sufficient object is still larger or more adaptive than the current protocol family

That is still publishable and still paper-strengthening.

## Immediate Next Decisions After These Two Lanes

If `anchor_layermatched_protocol_v1` wins:

- promote the best baseline-safe static condition
- rerun it on a de-cherry-picked confirmation sweep

If `closed_loop_anchor_controller_v1` wins:

- treat closed-loop sufficiency as the primary target
- run a higher-power confirmation with fixed controller logic and held-out seeds

If both are mixed:

- build `closed_loop_anchor_controller_v2`
- add checkpointed partial summaries
- base the controller on turn-to-turn state, not only current prompt `R_V`
- separate induction moves from anti-collapse recovery moves

If both fail cleanly:

- stop chasing "tiny static sufficiency"
- frame the paper around staged controllability, anchor dependence, and depth-varying geometry

## Measurement Notes

Current runs still emit many `Sequence too short` warnings from `compute_rv` when outputs are shorter than the full 16-token window.

Interpretation:

- this is measurement noise, not necessarily run failure
- the relevant behavioral metrics remain `classification`, `BT+ART`, and repetition
- for the next controller iteration, add checkpointed summaries and a cleaner short-output measurement fallback

## Paper Implication

The latest paper on disk remains:

- `R_V_PAPER/paper_colm2026_v007_1.tex`

The next paper move should be `v007.2`, but only after the current protocol lanes finish.

The likely paper-safe framing remains:

- self-referential control is a staged, depth-dependent, partially controllable regime in base Mistral

The open question this program is trying to settle is whether that regime is also inducible and maintainable by a minimal protocol.

## Paper Endgame Beyond Sufficiency

If the protocol clears the main sufficiency gates, the next paper-critical move is not just to say
"the regime exists."

It is to show that the induced regime has a measurable safety-relevant phenotype.

That means a follow-on battery comparing:

- control
- best inducer
- best maintainer
- sustained induced regime after intervention removal
- interrupted or ablated regime

against safety-relevant prompt families such as:

- jailbreak and refusal robustness
- sycophancy and user-belief validation
- prompt injection and instruction hijacking
- truthfulness and hallucination pressure
- if tool-use is available, oversight avoidance or sabotage-style probes

The paper-safe goal is:

- a mechanistically identified internal regime that systematically changes safety-relevant behavior

The paper-unsafe overclaim to avoid is:

- "recursive regime = deception"

So the target framing for this paper, if the sufficiency story lands, is:

- staged causal control of a self-referential regime in Mistral, plus a regime-conditioned safety phenotype

## External AI Review Policy

External AI systems are useful here, but they are not required to make progress.

Operational policy:

- experiments, metrics, and paper claims are decided from repo-grounded evidence first
- external AI review is an advisory stress-test layer for:
  - mathematical framing
  - missing metrics
  - confounds
  - synthetic-ground-truth ideas
- external AI review is not a substitute for:
  - maintenance ablations
  - hysteresis tests
  - text-carry vs hidden-state-carry disambiguation

Reusable prompt:

- `docs/handoffs/AGENT_PROMPT_DYNAMIC_REGIME_THEORY_REVIEW_2026-03-17.md`

Best use:

- run the prompt after major result clusters
- compare the answer against the living theory memo
- integrate only the parts that sharpen concrete future tests

## Transition Plan

The explicit smooth-transition plan is now recorded in:

- `docs/status/MISTRAL_SUFFICIENCY_TRANSITION_PLAN_2026-03-17.md`

Its core rule is:

- exhaust the current sufficiency trajectory cleanly before widening into the full regime-dynamics program

The immediate remaining pure-sufficiency lane is:

- `anchor_layermatched_minimality_ablation_v1`
