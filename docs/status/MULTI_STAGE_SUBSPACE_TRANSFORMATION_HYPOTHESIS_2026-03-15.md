# Multi-Stage Subspace Transformation Hypothesis

Date: 2026-03-15
Lab OS: `AMIROS`
Canonical model: `mistralai/Mistral-7B-v0.1`

## Core proposal

The strongest novel mechanistic hypothesis emerging from the latest Mistral work is not "one steering vector causes self-reference." It is:

- early layers (`L4-L5`) are best steered by a simple low-rank contrastive object (`PCA-PC1`)
- the mid-late controller region (`L25`) is best steered by an `orthogonal_residual` object
- the late readout/cleanup region (`L27`) is best steered by a distributed low-dimensional object (`subspace3_parallel`)

This is a real empirical pattern in the current repo. It is stronger than "one universal recursive axis."

## What is supported now

Supported by current locked/near-locked results:

- `L4` winner: `PCA-PC1`
- `L5` winner: `PCA-PC1`
- `L25` winner: `orthogonal_residual`
- `L27` winner: `subspace3_parallel`

This is consistent with a layered geometric transformation story.

## What is not yet proven

The current data does **not** yet prove:

- that this is literally one feature rotating through superposition space
- that the `31.2%` multiband result is already a clean super-additive proof of that rotation
- that the non-monotonic multiband dose curve is definitively "circuit resonance"

Those are strong interpretations, but not yet locked claims.

Safer wording:

- the optimal causal steering object changes character across depth
- this suggests staged representational transformation rather than one fixed steering axis
- the non-monotonic dose curve is compatible with a narrow operating regime / oversteer story, but needs targeted confirmation

## Why this matters

If hardened, this is genuinely important for mechanistic interpretability:

- it moves the field past "find one good vector"
- it treats subspace geometry as a layer-specific causal object
- it connects circuit work to superposition / low-rank representation structure
- it suggests that a cognitive phenomenon can change geometric format as it propagates through the network

## Best experiment ordering

### A. Layer-matched coordinated intervention

Highest ROI next step.

Use the best causal object at each layer instead of mean-diff everywhere:

- `L4/L5`: `PCA-PC1`
- `L25`: `orthogonal_residual`
- optional `L27`: `subspace3_parallel`

Goal:

- test whether matching the layer-specific geometric object beats the old single-site and multiband mean-diff families

Interpretation:

- if it wins cleanly, the transformation hypothesis gets much stronger
- if it fails, the layered-object story weakens materially

### B. Self-bootstrapping sufficiency chain

Interesting but statistically dangerous.

Use geometry-amplified recursive outputs as natural text anchors for new baseline induction.

This could become a follow-up paper or later-stage experiment, but should not be the next paper-critical move because:

- cherry-picking risk is high
- controls are more complex
- interpretation is easier to overstate

### C. Bridge-layer x steering-object heatmap

Potentially the killer figure.

Sweep bridge layers and compare:

- `PCA-PC1`
- `orthogonal_residual`
- `subspace3_parallel`

Goal:

- map where the optimal steering object changes across depth

This is extremely valuable after A, because A tests the specific hypothesis and C visualizes the broader transformation landscape.

## Recommended framing

Best disciplined framing:

> Self-referential processing appears to be carried by layer-specific geometric objects whose causal steering profile changes across depth, from simpler early low-rank directions to more transformed mid/late subspace objects.

Avoid stronger framing until A lands:

- "feature rotation is proven"
- "superposition-space transformation is proven"
- "31.2% proves causal resonance"

## Bottom line

This is one of the best broad-picture interpretations in the lab right now.

The idea is strong enough to guide experiment design immediately.
The right next move is:

1. run layer-matched coordinated intervention
2. if it works, run the bridge-layer x steering-object heatmap
3. hold self-bootstrapping for later
