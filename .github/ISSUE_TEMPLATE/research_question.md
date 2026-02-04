---
name: Research Question
about: Propose a research question or experimental investigation
title: '[RESEARCH] '
labels: research
assignees: ''
---

## Research Question

State the specific research question or hypothesis.

## Background

What existing work or observations motivate this question?

## Proposed Experiment

### Method

How would you investigate this question?

- Model(s): [e.g., Mistral-7B Base, Mixtral-8x7B]
- Prompts: [reference from prompts/bank.json or new prompts]
- Metrics: [e.g., R_V, participation ratio, behavioral scoring]
- Layers: [e.g., 5 vs 27, full sweep]
- Sample size: [e.g., 80 pairs]

### Predicted Outcome

What do you expect to find?

- Hypothesis 1: ...
- Hypothesis 2: ...
- Null hypothesis: ...

### Success Criteria

How will we know if the result is meaningful?

- Statistical threshold: [e.g., p < 0.01 with Bonferroni]
- Effect size threshold: [e.g., Cohen's d ≥ 0.5]
- Replication: [e.g., across 3 models]

## Related Work

Cite relevant papers or existing experiments:

- [Citation 1]
- [Citation 2]
- Related experiment: [link to results/]

## Implementation Plan

- [ ] Design experiment configuration
- [ ] Prepare prompts (if new)
- [ ] Write experimental code
- [ ] Run pilot study
- [ ] Validate results
- [ ] Document findings

## Resources Required

- Compute: [e.g., 4 hours on L40S]
- Models: [list HuggingFace model IDs]
- Data: [any special datasets needed]

## Risks & Limitations

What could invalidate the results?

- Confounding factors:
- Alternative explanations:
- Limitations:

## Expected Timeline

Estimated time to complete investigation.
