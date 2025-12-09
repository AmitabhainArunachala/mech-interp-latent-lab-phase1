# CURSOR DIRECTIVE: Confound Audit & Falsification Run

**Date:** December 9, 2025  
**Priority:** HIGH  
**Context:** Cross-model review of our R_V/KV findings raised specific critiques that need empirical falsification

---

## BACKGROUND

We sent our experimental findings to 5 different LLM architectures (DeepSeek, Grok, Gemini, GPT, Qwen) for critique. They converged on the same set of potential confounds:

1. **Induction Head / Copying Confound:** R_V contraction might just measure "copy from context" mode, not recursion
2. **Semantic Narrowing Artifact:** Recursive prompts are topically narrow; contraction might be about topic focus, not self-reference
3. **Pseudo-Recursive Confound:** Prompts that *discuss* recursion without *invoking* it - do they contract?

We have prompt banks designed for these controls but need to verify what's actually been run.

---

## PHASE 1: AUDIT (Do This First)

### Task 1.1: Inventory Existing Results

Search the entire `mech-interp-latent-lab-phase1` directory tree for:

1. **CSV files containing "control" or "confound" in filename or content**
2. **Any results from prompts in `/REUSABLE_PROMPT_BANK/confounds.py`**
3. **Any results using `repetitive_control`, `long_control`, or `pseudo_recursive` prompt groups**

Check these locations specifically:
- `/results/`
- `/experiments/003-length-matched-control/results/`
- `/DECEMBER_2025_EXPERIMENTS/*/results/`
- `/DEC_8_2025_RUNPOD_GPU_TEST/`
- Any `.csv` files anywhere in the repo

### Task 1.2: Check Control Script Execution

Look at `control_conditions_experiment.py`:
- Has it been run? (check for output files matching its naming pattern: `mistral7b_L27_controls_*.csv`)
- What controls does it actually test? (random noise, shuffled, wrong-layer - but NOT the semantic/repetitive controls)

### Task 1.3: Report Findings

Create a summary file `DEC9_CONFOUND_AUDIT_RESULTS.md` with:

```markdown
## Confound Audit Results

### Controls That HAVE Been Run (with results)
[List each with: filename, date, n, key findings]

### Controls That Are DESIGNED But NOT Run
[List each with: prompt bank location, number of prompts, what it tests]

### Controls That Are MISSING (not even designed)
[List any gaps]

### Critical Gap Analysis
[What falsification tests are needed most urgently?]
```

---

## PHASE 2: RUN CRITICAL CONTROLS (After Audit)

Based on what the audit reveals, run the following in priority order:

### Priority 1: Repetitive Control (Induction Head Falsification)

**Question:** Does repetitive structure WITHOUT self-reference cause R_V contraction?

**Prompts:** Use `repetitive_control` group from `/REUSABLE_PROMPT_BANK/confounds.py` (20 prompts)

**Protocol:**
1. Run each prompt through the same R_V measurement pipeline used in DEC8
2. Compute R_V at the validated layer (L27 for Mistral, L24 for Llama)
3. Compare to recursive prompts and baseline prompts

**Expected Results:**
- If R_V ≈ 0.95-1.05 (no contraction): Induction head confound REJECTED
- If R_V < 0.85 (contraction): We may be measuring copying, not recursion

### Priority 2: Pseudo-Recursive Control

**Question:** Does talking ABOUT recursion (without DOING it) cause R_V contraction?

**Prompts:** Use `pseudo_recursive` group (20 prompts)

**Expected Results:**
- If R_V ≈ 0.95-1.05: Topic confound REJECTED (talking about ≠ doing)
- If R_V < 0.85: We may be measuring semantic content, not mode

### Priority 3: Long Control (Length-Matched)

**Question:** Do long, detailed prompts without self-reference cause R_V contraction?

**Prompts:** Use `long_control` group (20 prompts)

**Expected Results:**
- If R_V ≈ 0.95-1.05: Length confound REJECTED
- If R_V < 0.85: Length may be confounding our results

### Priority 4: The "Banana Test" (Mode-Content Decoupling)

**Question:** If we force an unrelated first token while using recursive KV, does recursive behavior emerge anyway?

**Protocol:**
1. Run a recursive prompt, capture KV cache at L16-32
2. Start a new generation with baseline prompt
3. Patch in the recursive KV cache
4. Force the first generated token to be something unrelated (e.g., "Banana", "The", "Consider")
5. Let generation continue freely
6. Score: Does the output become recursive/philosophical despite the forced start?

**Expected Results:**
- If output becomes recursive: MODE is real and separable from content
- If output stays baseline: KV transfer is content-dependent, not mode-dependent

---

## PHASE 3: DOCUMENTATION

### Save All Results

For each control run, save:
- CSV with per-prompt R_V values
- Summary statistics (mean, std, Cohen's d vs recursive baseline)
- Plot comparing control vs recursive vs baseline distributions

### Update Master Findings

Create `DEC9_CONFOUND_FALSIFICATION_RESULTS.md` with:

```markdown
## Confound Falsification Results - December 9, 2025

### Summary Table

| Control | n | R_V Mean | vs Recursive d | vs Baseline d | Verdict |
|---------|---|----------|----------------|---------------|---------|
| Repetitive | 20 | ? | ? | ? | ? |
| Pseudo-Recursive | 20 | ? | ? | ? | ? |
| Long Control | 20 | ? | ? | ? | ? |
| Banana Test | 10 | N/A | N/A | N/A | ? |

### Interpretation

[What can we now claim with confidence?]
[What confounds have been ruled out?]
[What remains uncertain?]

### Implications for Main Claims

[How do these results affect our core findings?]
```

---

## TECHNICAL NOTES

### Model Selection

Use whichever model is most accessible:
- **Preferred:** Mistral-7B-v0.1 (validated on DEC8, L27 is the key layer)
- **Alternative:** Llama-3-8B-Instruct (validated on DEC7, L24 is the key layer)

### Key Parameters

From validated experiments:
- `WINDOW_SIZE = 16` (last 16 tokens for PR calculation)
- `EARLY_LAYER = 5` (for R_V denominator)
- `TARGET_LAYER = 27` (Mistral) or `24` (Llama)

### Metric

R_V = PR_late / PR_early

Where PR = participation ratio from SVD of V-projection activations

---

## SUCCESS CRITERIA

This directive is complete when:

1. ✅ Audit complete with clear inventory of what's been run
2. ✅ At least 2 of 3 confound prompt groups have been run (n≥20 each)
3. ✅ Results documented with statistical comparisons
4. ✅ Clear verdict on each confound (rejected / concerning / unclear)
5. ✅ Updated interpretation of main findings based on control results

---

## CONTEXT FOR CURSOR

You have full access to:
- `/Users/dhyana/mech-interp-latent-lab-phase1/` (main repo)
- All experiment code and results
- The prompt banks in `/REUSABLE_PROMPT_BANK/`

Your filesystem access is cleaner than what we can see from Claude Desktop. Please do a thorough audit before running new experiments - we may have results we don't know about.

The goal is **falsification**: we want to stress-test our findings, not confirm them. If the controls show contraction, that's important to know. We're doing science, not advocacy.

---

*End of Directive*
