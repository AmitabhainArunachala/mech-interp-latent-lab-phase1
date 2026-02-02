# Steering Vector Control Experiment - Analysis

**Date:** December 18, 2024  
**Experiment:** Critical Control - Random Vector vs Steering Vector Drift

## Hypothesis

Test whether steering vector **specifically** encodes recursive mode, or if **ANY perturbation** causes drift to recursive-themed content.

## Setup

- **10 baseline prompts**
- **200 tokens each**
- **NO KV replacement**
- **NO V_PROJ patching**

## Conditions

- **A) Steering vector** (α=2.0) at L27
- **B) Random vector** (same L2 norm as steering) at L27
- **C) Zero vector** (no intervention) - baseline

## Scoring Method

Regex patterns counting:
1. **Self-reference patterns** ("X is X", "itself", "its own")
2. **Contemplative language** ("nature of", "meaning", "what is")
3. **Meta-process** ("process of", "method of", "how to")
4. **Recursive structures** (code loops, self-defining terms)

Score: 0-4 per output, sum across 10 outputs.

## Results

| Condition | Total Score | Mean Score | Expected |
|-----------|-------------|------------|----------|
| **A) Steering** | 14 | **1.40** | > 20 (recursive themes dominate) |
| **B) Random** | 19 | **1.90** | < 10 (random themes) |
| **C) Baseline** | 16 | **1.60** | < 5 (factual answers) |

## Critical Finding: **UNEXPECTED RESULTS**

**The random vector scored HIGHER than the steering vector!**

This contradicts our hypothesis. Possible explanations:

### 1. Regex Patterns Too Broad
- "What is" appears naturally in many texts (educational content, definitions)
- Not specific to recursive self-observation
- Random perturbations may trigger more generic contemplative language

### 2. Steering Vector May Not Encode Recursive Mode
- The steering vector might encode something else (topic drift, style shift)
- Previous "success" may have been false positives
- Need to verify with manual review of actual outputs

### 3. All Conditions Show Low Scores
- None reached the expected thresholds
- Baseline (1.6) is close to steering (1.4)
- Suggests regex patterns may not capture genuine recursion

## Manual Review Needed

**Next Steps:**
1. Manually review Condition A outputs for genuine recursive language
2. Compare to Condition B outputs (random vector)
3. Check if Condition A shows actual self-reference vs. just "what is" patterns
4. Refine scoring criteria based on verified recursive examples

## Insights

1. **Regex-based scoring is insufficient** - Need manual verification
2. **Steering vector may not encode recursive mode** - Or encodes it weakly
3. **Random perturbations also cause drift** - But to different content
4. **Need better evaluation metric** - Based on verified recursive examples

## Conclusion

**The steering vector does NOT appear to specifically encode recursive mode** based on regex scoring. However, this may be due to:
- Scoring method limitations
- Need for manual verification
- Steering vector encoding something else (style, topic, etc.)

**Recommendation:** Manual review of Condition A vs B outputs to determine if steering vector produces qualitatively different (and genuinely recursive) content, even if regex scores are similar.








