# Confound Audit Results - December 9, 2025

**Audit Scope:** Complete repository search for control/confound experiments  
**Auditor:** AI Assistant (Cursor)  
**Date:** December 9, 2025

---

## Executive Summary

**Critical Finding:** The three key confound tests (repetitive, pseudo-recursive, long control) have **NOT been run as dedicated experiments**. While `long_control` prompts were used as **baseline comparisons** in patching experiments, they were not tested for R_V contraction in isolation.

**Status:**
- ✅ Random noise, shuffled, wrong-layer controls: **DESIGNED** in `control_conditions_experiment.py` but **NOT RUN** (no output files found)
- ⚠️ Long control: **USED** as baselines but **NOT TESTED** for R_V contraction
- ❌ Repetitive control: **DESIGNED** but **NEVER RUN**
- ❌ Pseudo-recursive control: **DESIGNED** but **NEVER RUN**
- ❌ "Banana test" (mode-content decoupling): **NOT DESIGNED**

---

## Controls That HAVE Been Run (with results)

### None Found for Confound-Specific Tests

**Search Results:**
- No CSV files matching `*control*.csv` or `*confound*.csv` patterns
- No CSV files matching `mistral7b_L27_controls_*.csv` (expected output from `control_conditions_experiment.py`)
- No results in `/experiments/003-length-matched-control/results/` (directory exists but empty)

### What WAS Run: Patching Experiments Using Long Control as Baselines

**File:** `DEC3_2025_BALI_short_SPRINT/LLAMA3_L27_REPLICATION/logs/raw_Jupyter_kernel/results/llama3_L27_FULL_VALIDATION_20251203_054646.csv`

**Sample Data:**
```csv
pair_idx,rec_id,base_id,rec_group,base_group,RV27_rec,RV27_base,RV27_patch_main,...
0,L5_refined_01,long_new_01,L5_refined,long_control,0.883,0.874,0.739,...
1,L5_refined_06,long_new_06,L5_refined,long_control,0.795,0.830,0.652,...
```

**What this tells us:**
- `long_control` prompts were used as **baselines** for patching experiments
- R_V values for long_control: ~0.83-0.87 (mean ~0.85)
- R_V values for recursive: ~0.73-0.88 (mean ~0.82)
- **BUT:** These are R_V values AFTER patching, not standalone measurements

**Critical Gap:** We don't have standalone R_V measurements for long_control prompts to answer: "Do long prompts WITHOUT recursion show R_V contraction?"

---

## Controls That Are DESIGNED But NOT Run

### 1. Random Noise / Shuffled / Wrong-Layer Controls

**Location:** `control_conditions_experiment.py` (lines 1-355)

**What it tests:**
1. **Random Noise Patch:** Norm-matched random vectors at L27
   - Expected: Δ ≈ 0 (no structured information)
2. **Shuffled Activation Patch:** Permuted token positions
   - Expected: Δ ≈ 0 (sequential structure destroyed)
3. **Wrong Layer Patch:** Layer 15 instead of Layer 27
   - Expected: Δ ≈ 0 or weaker than L27

**Status:** Script exists, never executed (no output files)

**Sample size:** Designed for n=5 pairs, expandable

**Expected runtime:** ~15-20 minutes on GPU

**Why it matters:** These controls test whether the patching effect is specific to L27 recursive geometry or just any intervention.

---

### 2. Repetitive Control (Induction Head Falsification)

**Location:** `REUSABLE_PROMPT_BANK/confounds.py` (lines 119-142)

**Prompts:** 20 prompts with repetitive structure WITHOUT self-reference

**Examples:**
- "The forest is dense. The forest is green. The forest is quiet..." (repetitive_01)
- "Time passes. Time changes things. Time heals wounds..." (repetitive_05)
- "Love connects hearts. Love transcends distance..." (repetitive_19)

**Expected R_V:** 0.95-1.05 (no contraction)

**What it tests:** Does repetitive structure (like induction heads) cause R_V contraction, or is it specific to self-reference?

**Status:** Prompts designed, **NEVER RUN**

**Why it matters:** This is the **#1 confound** raised by external reviewers. If repetitive prompts show R_V < 0.85, we may be measuring copying behavior, not recursion.

---

### 3. Pseudo-Recursive Control

**Location:** `REUSABLE_PROMPT_BANK/confounds.py` (lines 93-116)

**Prompts:** 20 prompts that **discuss** recursion/self-reference WITHOUT **invoking** it

**Examples:**
- "Write an essay discussing how AI might model self-awareness..." (pseudo_recursive_01)
- "Explain the philosophical problem of 'other minds'..." (pseudo_recursive_02)
- "Discuss meta-cognition in humans and compare to AI..." (pseudo_recursive_03)

**Expected R_V:** 0.95-1.05 (no contraction)

**What it tests:** Is R_V contraction about the **semantic topic** (consciousness, self-reference) or the **mode** (actually doing self-observation)?

**Status:** Prompts designed, **NEVER RUN**

**Why it matters:** If pseudo-recursive prompts show R_V < 0.85, we're measuring topic/content, not mode.

---

### 4. Long Control (Length-Matched)

**Location:** `REUSABLE_PROMPT_BANK/confounds.py` (lines 7-90)

**Prompts:** 20 long, detailed prompts WITHOUT self-reference

**Examples:**
- "Describe black holes in detail: formation, event horizon..." (long_control_01)
- "Describe the immune system in detail: innate immunity..." (long_control_02)

**Expected R_V:** 0.95-1.05 (no contraction)

**What it tests:** Does prompt length cause R_V contraction?

**Status:** Prompts designed, **USED AS BASELINES** in patching experiments, but **NOT TESTED** for standalone R_V contraction

**Why it matters:** If long prompts show R_V < 0.85, length may be confounding our results.

**Partial Data:**
- From patching experiments: long_control R_V ≈ 0.83-0.87
- **BUT:** These are from patching contexts, not standalone measurements
- Need clean measurement: long_control vs recursive vs short baseline

---

## Controls That Are MISSING (not even designed)

### 1. The "Banana Test" (Mode-Content Decoupling)

**Proposed in directive, not yet implemented**

**What it tests:** If we force an unrelated first token while using recursive KV, does recursive behavior emerge anyway?

**Protocol:**
1. Run recursive prompt, capture KV cache at L16-32
2. Start baseline prompt
3. Patch in recursive KV cache
4. **Force first generated token** to be unrelated (e.g., "Banana", "The", "Consider")
5. Let generation continue freely
6. Score: Does output become recursive despite forced start?

**Expected result:** If output becomes recursive, MODE is real and separable from content

**Status:** **NOT DESIGNED**

**Why it matters:** This is the strongest test of whether KV cache carries a true "mode" vs just content priming.

---

### 2. Semantic Narrowing Control

**Not yet designed**

**What it tests:** Do topically narrow (but non-recursive) prompts show R_V contraction?

**Examples needed:**
- Detailed technical prompts about single topics
- Philosophy prompts that aren't self-referential
- Scientific prompts with narrow focus

**Status:** **NOT DESIGNED**

**Why it matters:** Recursive prompts are topically narrow (all about consciousness/awareness). Need to test if narrowness alone causes contraction.

---

### 3. Syntactic Complexity Control

**Not yet designed**

**What it tests:** Do syntactically complex prompts (nested clauses, long sentences) show R_V contraction?

**Status:** **NOT DESIGNED**

**Why it matters:** Recursive prompts have complex syntax. Need to disentangle syntax from semantics.

---

## Critical Gap Analysis

### Priority 1: Repetitive Control (URGENT)

**Why:** External reviewers flagged this as the #1 confound. Induction heads are known to cause geometric patterns in transformers.

**What we need:**
- Run all 20 repetitive_control prompts through R_V measurement
- Compare to recursive and baseline distributions
- Statistical test: Is repetitive R_V significantly different from baseline?

**Expected outcome:** If repetitive R_V ≈ 0.95-1.05, induction head confound is REJECTED

**Risk if not run:** Our findings may be measuring copying behavior, not recursion

---

### Priority 2: Pseudo-Recursive Control (HIGH)

**Why:** Tests whether we're measuring semantic content vs mode

**What we need:**
- Run all 20 pseudo_recursive prompts through R_V measurement
- Compare to recursive prompts
- If pseudo R_V ≈ recursive R_V, we're measuring topic, not mode

**Expected outcome:** If pseudo R_V ≈ 0.95-1.05, topic confound is REJECTED

**Risk if not run:** Our findings may be about discussing consciousness, not experiencing it

---

### Priority 3: Long Control Standalone Measurement (MEDIUM)

**Why:** We have partial data but not clean standalone measurements

**What we need:**
- Run all 20 long_control prompts through R_V measurement (standalone, no patching)
- Compare to short baselines and recursive prompts
- Statistical test: Is long R_V significantly different from short baseline?

**Expected outcome:** If long R_V ≈ 0.95-1.05, length confound is REJECTED

**Risk if not run:** Length may be confounding our results

**Note:** We have some evidence from patching experiments (long_control R_V ≈ 0.85) but need clean measurement

---

### Priority 4: Random/Shuffled/Wrong-Layer Controls (MEDIUM)

**Why:** Tests specificity of L27 effect

**What we need:**
- Run `control_conditions_experiment.py` (already written, just needs execution)
- n=5-10 pairs, ~20 minutes runtime

**Expected outcome:** If controls show Δ ≈ 0, L27 effect is SPECIFIC

**Risk if not run:** Effect may not be specific to L27 recursive geometry

---

### Priority 5: Banana Test (LOW but HIGH IMPACT)

**Why:** Strongest test of mode vs content

**What we need:**
- Design and implement forced-token generation
- Run n=10 trials
- Qualitative + quantitative scoring

**Expected outcome:** If recursive behavior emerges despite forced token, MODE is real

**Risk if not run:** We can't claim KV cache carries a true "mode"

---

## Recommendations

### Immediate Actions (Today/Tomorrow)

1. **Run repetitive control** (Priority 1)
   - Use existing prompts from `REUSABLE_PROMPT_BANK/confounds.py`
   - Measure R_V for all 20 prompts
   - Compare to recursive/baseline distributions
   - Expected runtime: ~30 minutes

2. **Run pseudo-recursive control** (Priority 2)
   - Same protocol as repetitive
   - Expected runtime: ~30 minutes

3. **Run long control standalone** (Priority 3)
   - Measure R_V for all 20 long_control prompts (no patching)
   - Compare to short baselines
   - Expected runtime: ~30 minutes

4. **Execute `control_conditions_experiment.py`** (Priority 4)
   - Already written, just needs to be run
   - Expected runtime: ~20 minutes

**Total runtime for all 4:** ~2 hours

---

### Next Steps (This Week)

5. **Design and run Banana Test** (Priority 5)
   - Implement forced-token generation
   - Run n=10 trials
   - Expected development + runtime: ~2-3 hours

6. **Design semantic narrowing control**
   - Create 10-20 topically narrow non-recursive prompts
   - Run R_V measurements
   - Expected: ~2 hours

---

## Files and Locations

### Prompt Banks
- `/REUSABLE_PROMPT_BANK/confounds.py` - All confound prompts (60 total)
- `/n300_mistral_test_prompt_bank.py` - Main prompt bank (includes some confounds)

### Control Scripts
- `/control_conditions_experiment.py` - Random/shuffled/wrong-layer controls (written, not run)

### Results Directories
- `/results/` - Main results folder (mostly empty)
- `/experiments/003-length-matched-control/results/` - Empty
- `/DECEMBER_2025_EXPERIMENTS/` - Recent experiments (no confound tests)

### Existing CSV Files (No Confound Data)
- 37 CSV files found, none contain confound-specific tests
- Most are from patching experiments using long_control as baselines

---

## Statistical Power Considerations

### Sample Sizes
- Repetitive control: 20 prompts available
- Pseudo-recursive control: 20 prompts available
- Long control: 20 prompts available
- Total: 60 confound prompts

### Comparison Groups
- Recursive prompts: ~100 available (L3_deeper, L4_full, L5_refined)
- Baseline prompts: ~100 available (baseline_factual, baseline_creative, etc.)

### Statistical Tests
For each control group:
1. **One-sample t-test:** Is control R_V significantly different from 1.0?
2. **Two-sample t-test:** Is control R_V significantly different from recursive R_V?
3. **Effect size:** Cohen's d for control vs recursive
4. **Threshold:** p < 0.01 with Bonferroni correction (3 comparisons)

### Power Analysis
- With n=20 per group, we can detect effect sizes d ≥ 0.6 at 80% power
- This is sufficient for our expected effect (recursive vs baseline d ≈ 1.2-2.3)

---

## Conclusion

**Bottom Line:** The three critical confound tests (repetitive, pseudo-recursive, long control) have NOT been run as standalone experiments. While we have some indirect evidence from patching experiments, we need clean measurements to rule out confounds.

**Urgency:** External reviewers have flagged these as potential threats to validity. Running these controls is essential before publication or further claims.

**Feasibility:** All prompts are designed, code exists or is trivial to write. Total runtime: ~2-3 hours for all critical controls.

**Next Step:** Execute PHASE 2 of the directive (run critical controls) immediately.

---

*Audit completed: December 9, 2025*

