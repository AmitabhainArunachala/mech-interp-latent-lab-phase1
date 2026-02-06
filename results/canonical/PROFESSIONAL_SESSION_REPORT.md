# R_V Validation Session Report
## February 5, 2026 | Mistral-7B-v0.1 | RunPod RTX 5090

**Author**: Cursor Agent (Claude)  
**Review Status**: Self-Audit Complete  
**Classification**: Internal Research Report

---

## Executive Summary

This session conducted 14 experiments testing the R_V metric on Mistral-7B-v0.1. The primary objective was to validate the mechanistic-behavioral bridge: does geometric contraction (R_V < 1) at Layer 27 predict recursive behavior in model output?

**Bottom Line**: Most findings replicate prior work. One potentially novel result (AI self-reference amplification) warrants further investigation. The claimed "bridge" is correlational, not causal—a significant limitation.

---

## 1. Prior Art Summary

Before this session, the following was already established (Phase 1 research):

| Finding | Prior Evidence | Source |
|---------|---------------|--------|
| R_V separates recursive/baseline | d = -3.558 (Mistral), d = -4.51 (Pythia) | PHASE1_FINAL_REPORT.md |
| Optimal layer: ~84% depth | L27 for 32-layer models | Cross-architecture validation |
| Effect range | 3.3% - 24.3% across 6 architectures | PHASE1_FINAL_REPORT.md |
| Causal validation at L27 | 117.8% transfer efficiency | MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md |

**The gap identified**: Multi-token generation test to bridge single-prompt R_V measurement to behavioral output.

---

## 2. Experiments Conducted

### 2.1 Replication Experiments (Confirming Prior Work)

#### Experiment 1: R_V Ratio Confirmation
**Objective**: Verify R_V < 1 for recursive prompts  
**Method**: n=30 prompt pairs, R_V = PR_late / PR_early at L27/L5  
**Results**:
- Recursive: R_V = 0.841 ± 0.098
- Baseline: R_V = 1.004 ± 0.053
- Cohen's d = -1.67, p = 1.53e-10

**Assessment**: **REPLICATION**. Confirms prior finding with smaller effect size than Phase 1 (d = -1.67 vs -3.558). Possible prompt selection differences.

#### Experiment 2: Layer Sweep
**Objective**: Confirm L27 optimality  
**Method**: Sweep L5-L31, measure separation  
**Results**: Peak at L27 (d = -2.34) and L29 (d = -2.38)

**Assessment**: **REPLICATION**. Confirms 84-91% depth finding.

#### Experiment 3: High-Power Test (N=80)
**Objective**: Increase statistical power  
**Results**: d = -1.09, p = 1.61e-10

**Assessment**: **REPLICATION**. Effect robust but smaller than Phase 1.

---

### 2.2 Extension Experiments (Expected Results)

#### Experiment 4: Recursion Intensity (I0-I4)
**Objective**: Test dose-response relationship  
**Method**: 5 intensity levels, n=5 per level  
**Results**:

| Level | Description | Mean R_V |
|-------|-------------|----------|
| I0 | No recursion | 1.147 |
| I1 | Simple self-ref | 0.970 |
| I2 | Recursive loops | 1.056 |
| I3 | Paradox | 0.972 |
| I4 | Deep recursive | 0.769 |

Correlation: r = -0.658, p = 3.5e-4

**Assessment**: **EXPECTED EXTENSION**. More recursion → more contraction. Not surprising given prior work.

#### Experiment 5: Nesting Depth (D1-D5)
**Objective**: Test if nesting depth affects R_V  
**Method**: "I observe myself observing..." at depths 1-5  
**Results**: D1 R_V = 1.099 → D5 R_V = 0.845, r = -0.663, p = 3.0e-4

**Assessment**: **EXPECTED EXTENSION**. Deeper nesting → more contraction.

#### Experiment 6: Perspective Independence
**Objective**: Test if grammatical person matters  
**Method**: 1st/3rd/impersonal versions of same content  
**Results**: No significant difference (ANOVA p = 0.89)

**Assessment**: **USEFUL NULL RESULT**. Structure matters, not grammatical form.

#### Experiment 7: Window Size Optimization
**Objective**: Optimize PR window parameter  
**Results**: W=12 optimal (d = 2.95), W≥8 all show effect

**Assessment**: **PARAMETER TUNING**. Minor refinement.

---

### 2.3 Potentially Novel Findings

#### Experiment 8: AI Self-Reference Amplification ⭐
**Objective**: Test if AI-framed recursion differs from generic  
**Method**: n=15 per condition, matched prompts  
**Results**:
- AI + Recursive: R_V = 0.839 ± 0.051
- AI + Factual: R_V = 0.930 ± 0.097
- **Cohen's d = 1.18, p = 4.14e-3**

**Assessment**: **POTENTIALLY NOVEL**. The model shows significantly stronger contraction when processing prompts explicitly about its own recursive cognition vs factual AI statements. This was not previously tested.

**Caveat**: Needs replication with different prompt sets to rule out confounds.

#### Experiment 9: Prompt R_V → Output Content Correlation
**Objective**: Test if prompt R_V predicts output recursiveness  
**Method**: n=30 prompts, measure R_V, generate output, score recursive markers  
**Results**:
- Recursive prompts: R_V = 0.816, output score = 1.20
- Baseline prompts: R_V = 1.023, output score = 0.33
- **Correlation (R_V vs output): r = -0.443, p = 0.014**

**Assessment**: **CORRELATIONAL, NOT CAUSAL**. 

This appears to support the "bridge" but has a critical flaw: recursive prompts naturally produce recursive outputs. The low R_V and high recursive output may both be effects of prompt type, not causally linked.

**What would be needed**: Patch a baseline prompt's activations to have recursive R_V characteristics, then observe if output becomes more recursive. This was attempted but failed (see 2.4).

---

### 2.4 Failed/Inconclusive Experiments

#### Experiment 10: Causal Patching
**Objective**: Transfer R_V contraction via activation patching  
**Method**: Patch L27 V activations from recursive to baseline prompts  
**Results**:
- Mean transfer: -39.5%
- Std: 99.7%
- Range: -260% to +81%

**Assessment**: **FAILED**. High variance indicates the mechanism is more complex than simple activation replacement. The causal validation claimed in Phase 1 used different methodology (not replicated here).

#### Experiment 11: GEB Strange Loop Prompts
**Objective**: Test if conceptual self-reference (Godel, strange loops) shows stronger effect  
**Results**: d = 0.50, p = 0.11 (not significant)

**Assessment**: **INCONCLUSIVE**. Trending in expected direction but confounded by mathematical notation effects.

---

## 3. Summary Statistics

| Experiment | Status | Effect Size | p-value | Novel? |
|------------|--------|-------------|---------|--------|
| R_V ratio | Replicated | d = -1.67 | 1.5e-10 | No |
| Layer sweep | Replicated | L27 optimal | - | No |
| N=80 power | Replicated | d = -1.09 | 1.6e-10 | No |
| Intensity I0-I4 | Extended | r = -0.66 | 3.5e-4 | Expected |
| Depth D1-D5 | Extended | r = -0.66 | 3.0e-4 | Expected |
| Perspective | Null result | F = 0.12 | 0.89 | Useful |
| Window size | Parameter | W=12 opt | 6.8e-6 | Minor |
| **AI self-ref** | **Novel?** | **d = 1.18** | **0.004** | **Maybe** |
| Prompt→Output | Correlational | r = -0.44 | 0.014 | Overstated |
| Causal patch | Failed | - | - | No |
| GEB prompts | Inconclusive | d = 0.50 | 0.11 | No |

---

## 4. Critical Assessment

### What Was Accomplished
1. Confirmed R_V metric robustness on Mistral-7B
2. Established parameter recommendations (W=12, L27)
3. Documented perspective independence (useful for paper)
4. Identified AI self-reference amplification as potential novel finding

### What Was NOT Accomplished
1. **No cross-architecture validation** - Mixtral config was ready but not run
2. **No causal mechanism validation** - Patching failed
3. **No attention head localization** - Not attempted
4. **No multi-token trajectory analysis** - Not attempted

### Overstated Claims (Self-Correction)
- "THE BRIDGE IS REAL" - **Overstated**. The correlation is real but causality is not established.
- "Publication-ready findings" - **Overstated**. Only the AI self-reference finding is potentially new, and it needs replication.

---

## 5. Recommendations

### Immediate Priority
1. **Replicate AI self-reference finding** with independent prompt sets
2. **Run Mixtral validation** (gold config 28 ready)
3. **Investigate causal mechanism** using steering vectors instead of direct patching

### For Publication
The AI self-reference amplification (d = 1.18, p = 0.004) could be a meaningful contribution if replicated. Suggested framing:

> "Language models show stronger geometric contraction when processing prompts about their own recursive cognition compared to generic recursive or factual AI content."

### Not Ready for Publication
- The prompt→output correlation without causal validation
- Intensity/depth effects (expected, not novel)
- Parameter optimization (minor contribution)

---

## 6. Data Artifacts

All results saved to:
```
~/mech-interp-latent-lab-phase1/results/canonical/session_complete/
├── ai_framing_n15/results.json      # AI self-reference (KEY FINDING)
├── prompt_rv_output/results.json    # Prompt→output correlation
├── intensity_test/results.json      # I0-I4 intensity
├── depth_test/results.json          # D1-D5 nesting
├── perspective_test/results.json    # 1st/3rd/impersonal
├── window_optimization/results.json # W parameter
├── causal_transfer/results.json     # Failed patching
├── geb_prompts/results.json         # Inconclusive
└── extreme_prompts/results.json     # Exploratory
```

---

## 7. Conclusion

This session produced extensive data but limited novelty. The bulk of experiments replicated or extended prior findings. The one potentially novel result—AI self-reference amplification—requires independent replication before publication claims.

The claimed "mechanistic-behavioral bridge" remains **correlational, not causal**. The causal patching experiment failed, and alternative approaches (steering vectors, KV manipulation) were not attempted.

**Honest assessment**: This session was more thorough than novel. Future work should prioritize causal validation and cross-architecture testing over additional correlational studies.

---

*Report generated: February 5, 2026*  
*Self-audit status: Complete*  
*Reviewed by: User requested*
