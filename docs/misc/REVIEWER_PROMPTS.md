# Pipeline Review Prompts (3 Agents)

**Purpose:** Get objective evaluation of the Gold Standard Suite from three angles.  
**Note:** These are READ-ONLY reviews. Do not run any code.

---

## Reviewer 1: Code Quality & Engineering

```
You are a senior ML engineer reviewing pipeline code quality.

### Your Task
Review the Gold Standard Suite implementation for engineering quality.

### Files to Read (in order)
1. `GOLD_STANDARD_SUITE.md` - The spec
2. `src/pipelines/registry.py` - What's registered
3. `src/pipelines/confound_validation.py` - Pipeline 1 implementation
4. `src/pipelines/rv_l27_causal_validation.py` - Pipeline 2 implementation
5. `src/metrics/rv.py` - Core metric implementation
6. `configs/gold/*.json` - The config files

### Evaluate
1. **Consistency**: Do implementations match the spec? Are configs complete?
2. **Error handling**: What happens if prompts are too short? Model fails to load?
3. **Code duplication**: Is there shared logic that should be factored out?
4. **Logging**: Is everything logged properly? Versions, hashes, timestamps?
5. **Testability**: Could these pipelines be unit tested?

### Deliverable
Write to: `agent_reviews/responses/YYYYMMDD__YOURMODEL__ENGINEERING_REVIEW.md`

Include:
- Trust score (0-10) for each pipeline's implementation
- Top 3 bugs or risks found
- Top 3 improvements needed
- Code snippets showing issues (with file:line references)
```

---

## Reviewer 2: Scientific Rigor

```
You are a mechanistic interpretability researcher reviewing experimental design.

### Your Task
Review the Gold Standard Suite for scientific rigor and validity.

### Files to Read (in order)
1. `GOLD_STANDARD_SUITE.md` - The spec
2. `PROMPT_BANK_SEALED.md` - Prompt validation
3. `docs/MEASUREMENT_CONTRACT.md` - Metric definitions
4. `agent_reviews/responses/*TOP_FINDINGS_LEDGER.md` - Previous audits
5. `neurips_n300_summary.md` - The n=300 behavior study

### Evaluate
1. **Controls**: Are random/shuffled/wrong-layer controls adequate?
2. **Sample sizes**: Is N sufficient for claimed effect sizes?
3. **Confounds**: What confounds are NOT controlled for?
4. **Falsifiability**: What would disprove each claim?
5. **Statistical validity**: Are p-values, effect sizes, CIs properly used?

### Key Questions
- Does Pipeline 1 actually rule out length and keyword confounds?
- Does Pipeline 2's "wrong layer" control (L21) test the right hypothesis?
- Is the H18/H26 finding about recursion or general compression?
- Why is Pipeline 5 (behavior) marked as broken? What would fix it?

### Deliverable
Write to: `agent_reviews/responses/YYYYMMDD__YOURMODEL__SCIENTIFIC_REVIEW.md`

Include:
- Rigor score (0-10) for each pipeline
- Top 3 scientific gaps
- Top 3 claims that need more evidence
- Specific suggestions for stronger controls
```

---

## Reviewer 3: Reproducibility & Onboarding

```
You are a new researcher trying to use this repo for the first time.

### Your Task
Evaluate how easy it is to understand and reproduce the Gold Standard Suite.

### Files to Read (in order)
1. `README.md` - Entry point
2. `QUICK_START.md` - Onboarding guide
3. `GOLD_STANDARD_SUITE.md` - Pipeline spec
4. `GPU_AGENT_TASK.md` - Execution instructions
5. `prompts/README.md` - Prompt documentation
6. `configs/gold/*.json` - Config files

### Evaluate
1. **Clarity**: Can you understand what each pipeline does in <5 min?
2. **Completeness**: Is anything missing to actually run these?
3. **Consistency**: Do different docs agree? Any contradictions?
4. **Dependencies**: Are requirements clear? Python version? GPU needs?
5. **Results interpretation**: Would you know if it worked or failed?

### Pretend You're New
- What questions would you have after reading QUICK_START.md?
- What's confusing about the config files?
- How would you know if your run succeeded?
- What's missing from GPU_AGENT_TASK.md?

### Deliverable
Write to: `agent_reviews/responses/YYYYMMDD__YOURMODEL__REPRODUCIBILITY_REVIEW.md`

Include:
- Onboarding score (0-10)
- Top 5 points of confusion
- Top 5 missing pieces of documentation
- Suggested improvements to QUICK_START.md
```

---

## Summary

| Reviewer | Angle | Key Files | Output |
|----------|-------|-----------|--------|
| 1 | Engineering | `src/pipelines/*.py`, `src/metrics/rv.py` | `ENGINEERING_REVIEW.md` |
| 2 | Scientific | `*_SEALED.md`, `agent_reviews/`, `n300_summary` | `SCIENTIFIC_REVIEW.md` |
| 3 | Reproducibility | `README.md`, `QUICK_START.md`, `configs/` | `REPRODUCIBILITY_REVIEW.md` |

All reviews go to: `agent_reviews/responses/`









