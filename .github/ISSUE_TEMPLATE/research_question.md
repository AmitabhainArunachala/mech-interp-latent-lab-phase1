---
name: Research Question
about: Propose a new research direction or experiment
title: '[RESEARCH] '
labels: research
assignees: ''
---

## Research Question

State the research question or hypothesis clearly.

## Background

- What motivated this question?
- What existing research relates to this?
- What gap does this fill?

## Proposed Experiment

### Methodology

Describe the experimental approach:

- Models to test:
- Prompt sets to use:
- Metrics to compute:
- Layers to analyze:
- Statistical tests planned:

### Expected Outcome

What would a positive result look like? What would a negative result mean?

### Reproducibility Plan

- Sample size: (minimum 80 pairs)
- Statistical thresholds: (p < 0.01 with Bonferroni)
- Effect size threshold: (|d| ≥ 0.5)
- Random seed control:

## Resource Requirements

- Compute time estimate:
- GPU memory required:
- Storage needed for results:

## Connection to Existing Work

How does this relate to:
- The R_V metric?
- Existing validated results?
- Current understanding of geometric contraction?

## References

Cite relevant papers or prior work:

1. 
2. 
3. 

## Potential Risks

- Could this produce null results?
- Are there confounding factors to control?
- How would we interpret negative findings?

## Checklist

- [ ] This follows the measurement invariants
- [ ] This uses the standard protocol
- [ ] Statistical power is adequate (n ≥ 80)
- [ ] Reproducibility is ensured (seeds, configs documented)
- [ ] I'm prepared to report null results
