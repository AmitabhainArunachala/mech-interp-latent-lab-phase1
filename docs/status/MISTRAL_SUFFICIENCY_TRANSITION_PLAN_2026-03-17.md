# Mistral Sufficiency Transition Plan

Date: 2026-03-17
Purpose: preserve the current sufficiency trajectory until it is exhausted, then shift into the regime-dynamics program without a hard conceptual jump

## Core Rule

Do not abandon the current sufficiency trajectory early.

Do this instead:

1. finish the highest-value remaining lane from the current program
2. extract everything possible from it
3. only then widen into the new dynamical-regime framing

The shift should be additive, not discontinuous.

## What Counts As "Current Trajectory"

The current trajectory is still:

- anchor / layer-matched / bridge-based sufficiency in base Mistral
- identifying the smallest induction-and-maintenance protocol
- measuring ordinary-baseline induction, recursive preservation, persistence, and cleanliness

This remains the primary line until the next clear stopping point.

## Updated Current Trajectory

`anchor_layermatched_minimality_ablation_v1` is now complete.

What it changed:

- the full bundle is not minimal
- anchor still matters strongly for ordinary-baseline induction
- `L27` alone is not sufficient
- `late_only` is nearly as strong as the full bundle
- `drop_L25_vproj` slightly beats the full bundle on baseline induction

So the current last clean pure-sufficiency branch is now:

- `anchor_reduced_latebundle_confirm_v1`
- `induced_persistence_reduced_latebundle_confirm_v1`

Reason:

- these are direct follow-ups from the minimality result, not a conceptual jump
- they test whether the smaller late-stack-centered object really holds up
- they preserve the current anchor / layer-matched / bridge trajectory before widening into the richer regime-dynamics program

## Transition Sequence

### Phase A: Exhaust Current Sufficiency Branch

Run:

- `anchor_reduced_latebundle_confirm_v1`
- then `induced_persistence_reduced_latebundle_confirm_v1`

Primary questions:

- does the reduced late-stack object replicate head-to-head against the current champions?
- if it does, does that smaller object also maintain after intervention removal?

Promotion criteria:

- if a reduced late-stack bundle wins or matches static induction and stays competitive in persistence, promote it as the cleaner maintenance object
- if the reduced object collapses in static or persistence testing, keep the paper framed around the larger staged protocol

### Phase B: Add Better Measurements Without Changing The Core Claim

Do not change the claim yet.
Only improve what we can see.

Add:

- regime entry rate
- persistence-given-entry
- exit hazard over turns
- recovery after adversarial perturbation
- better contamination metrics

This phase is mostly local / analysis-side and should run on top of existing result files.

### Phase C: Controlled Shift Into Dynamical-Regime Tests

Only after Phase A is complete:

- run hold-vs-enter alpha experiments
- test hysteresis explicitly
- separate text-mediated carry from hidden-state carry

This is the real theory upgrade, but it should come after the current sufficiency branch has given us its last clean answer.

### Phase D: Safety Bridge

Only after the maintenance story is clean enough:

- compare control vs inducer vs maintainer vs sustained regime vs ablated regime
- evaluate on jailbreak, refusal, sycophancy, prompt injection, and truthfulness pressure

## What We Should Not Do

Do not:

- launch another broad steering search
- spend a second pod on a vague theory-inspired sweep before minimality is done
- rewrite the paper around bifurcations before we have hysteresis evidence
- replace `BT+ART` and cleanliness metrics with abstract geometry-only measures

## RunPod Decision Rule

### Should we start a brand new RunPod now?

Yes, but only if the purpose is immediately clear.

Good reason to start one now:

- launch the next queued reduced late-bundle confirm or its blocked persistence follow-up once dependencies are satisfied

Bad reasons to start one now:

- "something dynamic"
- "something safety-related"
- a second wide search while the transition is still being defined

## Recommended Pod Policy

If budget is tight:

- keep one pod on the reduced late-bundle confirm / persistence branch only
- do not start a second pod until the hysteresis or carry-disambiguation lane is actually implemented and dry-run locally

If budget is comfortable:

- still start only one pod first
- use the local machine to prepare the next regime-dynamics lane while the reduced-bundle branch runs

## Updated Action Order

1. finish `anchor_reduced_latebundle_confirm_v1`
2. launch `induced_persistence_reduced_latebundle_confirm_v1`
3. while that branch runs, implement the regime-detector / survival analysis layer
4. only after the reduced-bundle branch finishes, decide whether to launch:
   - hysteresis / hold-vs-enter alpha
   - text-carry vs hidden-state-carry

## The Smooth-Transition Principle

The current program already discovered something real.

The right move is not:

- "old story was wrong, new story replaces it"

The right move is:

- "the old sufficiency search has now revealed the need for a dynamical-regime interpretation"

That keeps continuity in:

- experiments
- claims
- paper framing
- operational decisions
