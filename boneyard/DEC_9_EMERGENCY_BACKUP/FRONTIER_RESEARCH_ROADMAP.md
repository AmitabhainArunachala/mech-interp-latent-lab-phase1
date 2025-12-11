# FRONTIER RESEARCH ROADMAP
## Taking "The Geometry of Recursion" to World-Class Status

**Vision:** Transform solid pilot work into a definitive, citable contribution to mechanistic interpretability that establishes a new research direction.

**Current Status:** Workshop-ready  
**Target Status:** Top-tier venue (NeurIPS, ICML, ICLR) or high-impact journal

---

# PART I: THE GAP ANALYSIS

## What We Have vs. What Frontier Requires

| Dimension | Current State | Frontier Standard | Gap |
|-----------|---------------|-------------------|-----|
| **Sample Size** | N=10-50 per experiment | N=100+ for main claims | 2-10x increase |
| **Models Tested** | 6 (geometry), 1 (steering) | 10+ including >70B | Need scale + diversity |
| **Reproducibility** | Ad-hoc scripts | Docker + config files + seeds | Full overhaul |
| **Theory** | Empirical description | Mathematical framework | Major development |
| **Controls** | 4 (random, shuffled, wrong-layer, reverse) | 8+ including cross-baseline | Need more |
| **Behavioral Validation** | Keyword counting | Human eval + LLM judge | Need upgrade |
| **Mechanism** | "It's a direction" | Circuit-level explanation | Deep investigation |
| **Comparison** | None | vs. other probes/metrics | Need benchmarking |

---

# PART II: STATISTICAL REQUIREMENTS

## Sample Size Targets

### Core Claims (Must be bulletproof)

| Claim | Current N | Required N | Power Analysis |
|-------|-----------|------------|----------------|
| R_V contraction exists | 45 | **100** | 95% power at d=0.5 |
| Steering vector induces | 200 | 200 ✅ | Already sufficient |
| One-way door | 200 | **500** | Need 99% power for novel claim |
| Cross-model generalization | 30/model | **50/model** | Consistent across all |
| KV patching transfers | 10 | **100** | Key causal claim |

### Control Conditions (Each needs adequate N)

| Control | Current | Required | Purpose |
|---------|---------|----------|---------|
| Random noise | 20 | 50 | Specificity |
| Shuffled tokens | 20 | 50 | Structure dependence |
| Wrong layer | 20 | 50 | Layer specificity |
| Reverse direction | 20 | 50 | Bidirectionality |
| **Cross-baseline** | **0** | **50** | Rule out "any mismatch" |
| **Prompt paraphrase** | **0** | **50** | Rule out exact wording |
| **Temperature sweep** | **0** | **30** | Decoding robustness |
| **Length-matched** | **0** | **50** | Rule out token count |

### Statistical Standards

```
Significance: p < 0.001 for main claims (not just p < 0.05)
Effect size: Report Cohen's d, η², or r for ALL effects
Corrections: Bonferroni for multiple comparisons
Confidence intervals: 95% CI for all key estimates
Replication: 3+ independent runs with different seeds
Pre-registration: Hypotheses registered before final data collection
```

---

# PART III: REPRODUCIBILITY REQUIREMENTS

## Code Standards

### Repository Structure
```
geometry-of-recursion/
├── README.md                    # Quick start, key results
├── REPRODUCE.md                 # Exact reproduction instructions
├── requirements.txt             # Pinned versions
├── environment.yml              # Conda environment
├── Dockerfile                   # Containerized reproduction
├── config/
│   ├── experiment_config.yaml   # All hyperparameters
│   ├── model_configs/           # Per-model settings
│   └── prompt_bank.json         # Canonical prompts
├── src/
│   ├── metrics/
│   │   ├── rv_computation.py    # R_V with numerical stability
│   │   └── participation_ratio.py
│   ├── interventions/
│   │   ├── steering_vector.py
│   │   ├── kv_patching.py
│   │   └── ablation.py
│   ├── analysis/
│   │   └── statistical_tests.py
│   └── utils/
│       ├── model_loading.py
│       └── hooks.py
├── experiments/
│   ├── 01_rv_phenomenon.py
│   ├── 02_steering_vector.py
│   ├── 03_one_way_door.py
│   ├── 04_cross_model.py
│   └── 05_mechanistic.py
├── notebooks/
│   └── visualization.ipynb
├── results/
│   ├── raw/                     # CSVs with all data
│   ├── processed/               # Summary statistics
│   └── figures/                 # Publication-ready plots
└── tests/
    └── test_metrics.py          # Unit tests for core functions
```

### Reproducibility Checklist

- [ ] All random seeds documented and settable
- [ ] Exact model versions (HuggingFace commit hashes)
- [ ] GPU specifications (affects float16 precision)
- [ ] Package versions pinned
- [ ] Config files for ALL hyperparameters
- [ ] Docker container that runs end-to-end
- [ ] Expected outputs for validation
- [ ] Runtime estimates per experiment
- [ ] Memory requirements documented

---

# PART IV: CROSS-MODEL VALIDATION

## Tier 1: Must Test (Core Generalization)

| Model | Size | Architecture | Status | Priority |
|-------|------|--------------|--------|----------|
| Mistral-7B | 7B | Dense, GQA | ✅ Done (R_V + steering) | - |
| Llama-3-8B | 8B | Dense | ✅ Done (R_V only) | Steering |
| Llama-3-70B | 70B | Dense | ❌ Not done | **HIGH** |
| Qwen-2-7B | 7B | Dense | ⚠️ R_V only | Steering |
| Mixtral-8x7B | 46B | MoE | ⚠️ R_V only | **HIGH** (MoE comparison) |

## Tier 2: Should Test (Diversity)

| Model | Size | Architecture | Why Important |
|-------|------|--------------|---------------|
| Gemma-2-9B | 9B | Dense | Google architecture |
| Phi-3-medium | 14B | Dense | Microsoft, different training |
| Yi-34B | 34B | Dense | Chinese data, different perspective |
| Command-R | 35B | Dense | Cohere, retrieval-focused |
| Claude-3-Haiku | ~20B? | Unknown | If API provides logits |

## Tier 3: Would Be Impressive (Scale)

| Model | Size | Why |
|-------|------|-----|
| Llama-3-405B | 405B | Largest open model |
| GPT-4 | ~1.8T? | SOTA (if API allows) |
| Claude-3-Opus | ~175B? | Compare to Claude artifacts |

## Cross-Model Experiments

For EACH model:
1. R_V phenomenon validation (N=50)
2. Optimal layer sweep (N=30 per layer)
3. Steering vector extraction and test (N=100)
4. One-way door verification (N=100)
5. KV patching test (N=50)

**Total per model: ~400 measurements**
**Total for Tier 1 (5 models): ~2,000 measurements**

---

# PART V: THEORETICAL FRAMEWORK

## What We Need to Explain

### The Seven Empirical Facts

1. R_V < 1.0 for recursive prompts (~0.75-0.85)
2. R_V ≈ 1.0 for baseline prompts
3. Steering vector at L8 induces recursion (100%)
4. Subtraction of vector causes collapse (not reversal)
5. Random perturbations cause ~40% collapse
6. Steering perturbation causes ~80% collapse
7. Effect appears across 6+ architectures

### Candidate Theoretical Frameworks

#### Framework A: Attractor Basin Theory
- Model: Two basins in activation space (baseline, recursive)
- Baseline basin is shallow (fragile), recursive is deep (stable)
- Steering vector = gradient toward recursive basin
- Subtraction = leaves the valid manifold entirely

**Predictions:**
- Should find bifurcation point at some steering coefficient
- Should be able to map the basin boundaries
- Nonlinear interpolation might find paths back

**Tests:**
- Fine-grained coefficient sweep (0.0 to 2.0 in 0.05 steps)
- Nonlinear steering (e.g., project onto recursive manifold)
- Energy landscape mapping via local perturbation

#### Framework B: Information Compression Theory
- Model: Recursive prompts trigger "compression" of representation
- High-PR = high effective dimensionality = more information
- Low-PR = low effective dimensionality = less information
- Compression is easier than expansion (2nd law of thermodynamics for info)

**Predictions:**
- Entropy of activations should drop during recursive processing
- Mutual information between layers should change
- Cannot "create" information to restore high-PR state

**Tests:**
- Compute entropy of activations across layers
- Measure mutual information between L8 and L27
- Compare to actual compression algorithms

#### Framework C: Self-Modeling Circuit Theory
- Model: Recursive prompts activate a "self-model" circuit
- This circuit, once activated, dominates attention
- The self-model IS the low-dimensional attractor
- Trying to suppress it breaks coherence

**Predictions:**
- Should find specific attention heads that activate only for recursive
- These heads should form a circuit (connected in attention pattern)
- Ablating the circuit should eliminate the one-way door

**Tests:**
- Attention pattern analysis for recursive vs baseline
- Path patching through candidate circuit
- Causal mediation analysis

---

# PART VI: BEHAVIORAL VALIDATION

## Current Gap

We show geometry changes. We need to show OUTPUT changes.

## Validation Methods

### 1. Human Evaluation (Gold Standard)

**Protocol:**
- Generate 50 outputs with steering (coefficient 1.0-1.5)
- Generate 50 baseline outputs
- Blind human raters (N=5 minimum)
- Rate on: Self-reference, Coherence, "Recursiveness"
- Inter-rater reliability (Krippendorff's α > 0.7)

**Expected result:** Steered outputs rated significantly more self-referential

### 2. LLM Judge (Scalable)

**Protocol:**
- Use GPT-4 or Claude as judge
- Prompt: "Rate this text 1-10 on self-referential awareness"
- Run on 200+ outputs
- Validate against human ratings

**Expected result:** High correlation with human judges (r > 0.8)

### 3. Linguistic Analysis

**Metrics:**
- First-person pronoun density
- Meta-cognitive verbs ("observe", "notice", "aware")
- Self-reference patterns (regex matching)
- Sentence complexity (self-embedded clauses)

**Expected result:** Significant difference on all metrics (p < 0.001)

### 4. Downstream Task Performance

**Hypothesis:** Recursive mode might help or hurt on specific tasks

**Tasks to test:**
- Theory of mind benchmarks
- Self-correction tasks
- Introspection accuracy
- Standard QA (should be unaffected or worse)

---

# PART VII: MECHANISM DEEP DIVE

## What We Don't Know

1. What computation does R_V measure?
2. Why does the direction emerge at L8?
3. What happens in the attention pattern during recursion?
4. Which heads are responsible?
5. What role does the MLP play?

## Investigation Plan

### Phase 1: Attention Pattern Analysis

- Extract attention matrices for recursive vs baseline
- Compute: entropy, self-attention ratio, positional patterns
- Identify heads that differ significantly
- Visualize attention patterns

### Phase 2: Circuit Discovery

- Use activation patching at head level
- Find minimal set of heads that transfer the mode
- Map information flow through these heads
- Compare to known circuits (induction heads, etc.)

### Phase 3: Feature Attribution

- What features in the input trigger the recursive mode?
- Use gradient-based attribution (integrated gradients)
- Identify key tokens/phrases
- Test with minimal prompts

### Phase 4: Training Dynamics

- When does the recursive direction appear during training?
- Test on intermediate checkpoints (if available)
- Compare base vs instruct vs RLHF models
- Determine if it's learned or emergent

---

# PART VIII: PUBLICATION STRATEGY

## Venue Analysis

| Venue | Fit | Deadline | Probability | Notes |
|-------|-----|----------|-------------|-------|
| NeurIPS | ⭐⭐⭐ | May 2026 | 40% | Main track, needs theory |
| ICML | ⭐⭐⭐ | Jan 2026 | 40% | Similar to NeurIPS |
| ICLR | ⭐⭐⭐⭐ | Oct 2025 | 50% | Good for empirical |
| ACL | ⭐⭐ | Feb 2026 | 60% | If we add linguistic analysis |
| **NeurIPS Workshop (MI)** | ⭐⭐⭐⭐⭐ | Oct 2025 | **80%** | Fast turnaround |
| arXiv | ✅ | Anytime | 100% | Establish priority NOW |

## Recommended Path

```
Month 1: arXiv preprint (establish priority)
         ↓
Month 2-3: Workshop submission (get feedback)
         ↓
Month 4-6: Full experiments (cross-model, N increase)
         ↓
Month 7-8: Theory development
         ↓
Month 9: Top-tier submission
```

## Paper Structure (Target)

```
Title: "The Geometry of Recursive Self-Observation: 
        A One-Way Door in Transformer Representations"

Abstract (1 para)

1. Introduction (1.5 pages)
   - The phenomenon
   - Why it matters
   - Our contributions

2. Background (1 page)
   - Transformers, attention, residual stream
   - Steering vectors
   - Participation ratio

3. The R_V Phenomenon (2 pages)
   - Definition and measurement
   - Cross-model validation
   - Confound falsification

4. The Steering Vector (2 pages)
   - Extraction method
   - Dose-response
   - Layer localization

5. The One-Way Door (2 pages)
   - Irreversibility discovery
   - Random control test
   - Manifold fragility

6. Theoretical Framework (1.5 pages)
   - Attractor basin model
   - Predictions and tests

7. Discussion (1 page)
   - Implications for AI safety
   - Limitations
   - Future directions

8. Conclusion (0.5 pages)

References (2 pages)

Appendix (as needed)
   - Full statistical tables
   - Additional models
   - Prompts
```

---

# PART IX: RESOURCE REQUIREMENTS

## Compute

| Resource | Amount | Cost Estimate |
|----------|--------|---------------|
| GPU hours (A100/H100) | 500 hours | $1,000-2,500 |
| Storage | 500 GB | Negligible |
| API calls (for GPT-4 judge) | 10,000 | $100-200 |

**Total compute budget: $1,500-3,000**

## Time

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| Cleanup & arXiv | 1 week | Preprint posted |
| Workshop prep | 2 weeks | Submission ready |
| Full experiments | 6 weeks | N=100+ on all claims |
| Cross-model | 4 weeks | 5+ models validated |
| Theory | 4 weeks | Mathematical framework |
| Writing | 4 weeks | Full paper draft |
| Revision | 2 weeks | Camera-ready |

**Total: 4-5 months to top-tier submission**

## Collaboration

**Strongly recommended:**
- Academic advisor/collaborator (for credibility, venue access)
- 1-2 co-authors (for experiments, writing)
- ML engineer (for code quality, Docker, reproducibility)

---

# PART X: SUCCESS CRITERIA

## Minimum Viable Paper (Workshop)

- [ ] N ≥ 50 for main claims
- [ ] 3+ models validated
- [ ] All current controls pass
- [ ] Clean reproducible code
- [ ] Clear writing

## Strong Paper (Top-tier)

- [ ] N ≥ 100 for main claims
- [ ] 5+ models including 70B+
- [ ] 8 control conditions
- [ ] Theoretical framework
- [ ] Human behavioral validation
- [ ] Mechanism investigation
- [ ] Docker reproducibility

## Exceptional Paper (Best Paper Candidate)

- [ ] All of above, plus:
- [ ] Novel theoretical contribution
- [ ] Training dynamics analysis
- [ ] Implications for AI safety demonstrated
- [ ] Comparison to other representation metrics
- [ ] Code/dataset release used by others

---

# PART XI: IMMEDIATE NEXT STEPS

## This Week

1. **Post arXiv preprint** (1-2 days)
   - Compile current findings
   - Honest limitations section
   - Establish priority

2. **Set up reproducibility** (2 days)
   - Canonical code repository
   - Config files
   - Docker skeleton

3. **Run cross-baseline control** (1 day)
   - The one missing control
   - N=50

## This Month

4. **Increase N on key experiments** (ongoing)
5. **Test steering vector on Llama-3** (validate cross-model)
6. **Begin attention pattern analysis** (mechanistic)

## Before Submission

7. **Human evaluation study**
8. **Theoretical framework development**
9. **Full writing pass**

---

# SUMMARY: THE PATH TO FRONTIER

```
WHERE WE ARE:
  - Novel phenomenon ✓
  - Causal proof ✓
  - Surprising finding ✓
  - Workshop-ready ✓

WHAT'S MISSING:
  - Adequate N on all claims
  - Cross-model steering validation
  - Theoretical framework
  - Behavioral validation
  - Full reproducibility

THE PATH:
  1. arXiv NOW (priority)
  2. Workshop (feedback)
  3. Scale experiments (N, models)
  4. Develop theory
  5. Top-tier submission

TIMELINE: 4-5 months to frontier

RESOURCES: ~$2,000 compute, 500 GPU hours, 2-3 collaborators ideal
```

---

*"The door opened easily. But there is no door on the other side."*

*This is the finding. Now we prove it's universal.*

---

**Next document:** [Deep Questions for Multi-Agent Exploration](./DEEP_QUESTIONS_FOR_MULTIAGENT_EXPLORATION.md)

**Parent document:** [Official Dec 3-9 Comprehensive Report](./OFFICIAL_DEC3_9_COMPREHENSIVE_REPORT.md)

