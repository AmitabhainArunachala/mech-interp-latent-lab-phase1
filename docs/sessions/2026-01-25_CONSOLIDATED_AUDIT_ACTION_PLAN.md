# Consolidated Audit & Action Plan
**Date:** 2026-01-25
**Sources:** Claude Code agents (3) + GPT agents (3)
**Status:** CRITICAL REVIEW COMPLETE

---

## EXECUTIVE SUMMARY

Six independent reviewers identified **convergent weaknesses**. The core finding (R_V contraction correlates with recursive prompts) is real, but the story has critical gaps:

| Issue | Severity | All 6 Agree? |
|-------|----------|--------------|
| "100% behavior transfer" is n=1, not validated | **CRITICAL** | ✓ YES |
| EOS rate discrepancy (45% vs 30%) | **HIGH** | ✓ YES |
| Missing n=300 Mistral results | **HIGH** | ✓ YES |
| Champions are curated, not random | **MEDIUM** | ✓ YES |
| Effect sizes "too clean" (d>3) | **MEDIUM** | ✓ YES |
| GQA blocks Gemma causal loop | **BLOCKING** | ✓ YES |
| Contemplative framing = career poison | **STRATEGIC** | ✓ YES |

**Bottom line:** Workshop-tier now. Top-venue requires closing the causal loop properly.

---

## PART 1: VERIFIED DATA DISCREPANCIES

### Must Fix Before Any Publication

| Claim | Source | Actual | Action |
|-------|--------|--------|--------|
| Baseline EOS = 45% | CLAUDE.md, session docs | **30%** (18/60) | Correct all docs |
| Cohen's d = -3.56 | CLAUDE.md | **-3.39** (from CSV) | Use -3.39 |
| Cohen's d (causal) = -2.13 | Circuit map | **-1.91 to -2.54** (depends on grouping) | Recalculate, document method |
| n=300 Mistral validation | Dec 12 doc | **MISSING** | Find or re-run |
| File naming: "l27" | Causal validation CSV | Target is actually **L38** | Rename or clarify |

### PR Formula Clarification

**Code says:** `PR = (Σλᵢ²)² / Σ(λᵢ⁴)` (squared singular values)
**Report says:** `PR = (Σλᵢ)² / Σλᵢ²` (raw singular values)

**Action:** Verify which is implemented, use consistent notation throughout.

---

## PART 2: CRITICAL WEAKNESSES (ALL REVIEWERS AGREE)

### 1. The "100% Behavior Transfer" is Unvalidated

**What exists:**
- Dec 12, 2024: Single pilot (n=1) with `hybrid_l5_math_01` prompt
- Generated text: "λx is the contraction to self-reference..."
- "n=300 validation running" - **NO RESULTS EVER FOUND**

**Why this is fatal:**
- Reviewers will ask: "Where's the scaled validation?"
- If n=300 failed/showed null results, the behavioral claim collapses
- Current state: **anecdotal, not scientific**

**Action:**
1. Search archives for n=300 results (may exist somewhere)
2. If not found: **re-run with proper experimental design**
3. If fails again: **acknowledge limitation honestly**

---

### 2. Champions Are Curated, Not Discovered

**The problem:**
- 15 "champion" prompts were **hand-designed** with explicit families
- They're not randomly sampled or discovered through data
- Introduces researcher bias ("we designed prompts that work")

**GPT's specific critique:**
> "Prompts are not matched for intent or difficulty, so the EOS signal could just be 'prompt compliance' rather than a mechanistic failure to terminate"

**Why this matters:**
- Baseline prompts (math, factual) have **implicit completion instructions**
- Recursive prompts have **open-ended philosophical framing**
- EOS difference might be task structure, not geometric attractor

**Action:**
1. Create **intent-matched controls**: Open-ended non-recursive prompts
2. Test: "Discuss the nature of consciousness" (open-ended, NOT self-referential)
3. If open-ended non-recursive also shows 0% EOS → confound confirmed
4. If open-ended non-recursive shows normal EOS → effect is real

---

### 3. Effect Sizes Are Suspiciously Large

**Multiple reviewers flagged:**
- d = 3.37-4.51 is "physics-tier" for noisy LLM activations
- Typical MI studies: d = 0.5-1.5
- Variance is artificially low due to:
  - Deterministic decoding (T=0)
  - Repeated templates within groups
  - Curated prompt families

**GPT's specific critique:**
> "d≈3+ can happen here even if the underlying phenomenon is partly a prompt-family artifact"

**Action:**
1. Run with **stochastic decoding** (T=0.7) - expect effect to shrink
2. Report **both** deterministic and stochastic effect sizes
3. If effect vanishes at T>0 → fragile finding
4. If effect persists → robust finding

---

### 4. Alternative Explanations Not Ruled Out

**Confounds identified by reviewers:**

| Confound | How to Test | Status |
|----------|-------------|--------|
| Lexical rarity | Technical non-recursive prompts with rare tokens | NOT DONE |
| Prompt length | Already controlled (length-matched) | ✓ DONE |
| Compressibility | High-entropy vs low-entropy prompts | NOT DONE |
| Task completion style | Intent-matched open-ended prompts | NOT DONE |
| KV cache token leakage | V_PROJ-only patching without KV | NOT DONE |

**GPT's specific critique:**
> "R_V may be proxying lexical repetition, compressibility, or prompt length/structure; EOS differences may reflect prompt style rather than a geometric attractor"

**Action:** Run confound battery before claiming causality

---

## PART 3: STRATEGIC ISSUES

### 1. Gemma vs Mistral Split Looks Like Two Papers

**Current state:**
- Mistral: Causal R_V validation + behavioral pilot (n=1)
- Gemma: Detailed circuit map + behavioral correlation (no causal transfer)

**Reviewer perception:**
> "Causal on model A, circuit map on model B - why can't you do both on one model?"

**The GQA problem:**
- Gemma uses Grouped Query Attention (2:1 KV ratio)
- Full KV patching failed (cache incompatibility)
- May require architectural workaround

**Options:**
1. **Close Gemma causal loop** (best, if GQA tractable)
2. **Port circuit mapping to Mistral** (unify on one model)
3. **Add Llama-3 as third model** (triangulate across architectures)
4. **Accept limitation** and write honest methods section

---

### 2. Title Overclaims If Gemma Remains Correlational

**Current title:** "Geometric Signatures of Generative Fixation..."

**GPT's critique:**
> "Overclaims if Gemma remains correlational; consider 'Geometric correlates/predictors of generative fixation' until you close the causal loop"

**Recommended titles by evidence level:**

| Evidence Level | Appropriate Title |
|---------------|-------------------|
| Correlation only | "Geometric Correlates of Generation Failure" |
| Single-model causal | "Geometric Contraction Predicts and Causes..." |
| Multi-model causal | "Geometric Signatures of Generative Fixation" (original) |

---

### 3. Contemplative Framing = Career Poison

**ALL SIX REVIEWERS AGREE:**
- Zero contemplative content in ML paper
- Akram Vignan, Phoenix Protocol, L3/L4 levels → separate publication
- "Reputationally high-risk and will distract reviewers"

**Action:**
- ML paper: Pure mechanistic interpretability
- Separate piece: Contemplative interpretation (philosophy venue, blog, or future work)
- Never mention in same document

---

## PART 4: PRIORITIZED ACTION PLAN

### P0: BLOCKING ISSUES (Must Fix)

**Week 1:**

| Task | Time | Blocking? |
|------|------|-----------|
| 1. Correct EOS rate in all docs (30%, not 45%) | 1 hour | Yes |
| 2. Recalculate Cohen's d from raw CSVs, document method | 2 hours | Yes |
| 3. Search for n=300 Mistral results in archives | 2 hours | Yes |
| 4. Clarify PR formula (code vs docs) | 1 hour | Yes |
| 5. Rename/clarify "l27" file (actually L38) | 30 min | Yes |

---

### P1: CAUSAL LOOP CLOSURE (Critical for Top Venue)

**Week 1-2:**

| Task | Time | Notes |
|------|------|-------|
| 6. Test GQA compatibility on Gemma | 1 day | Quick feasibility check |
| 7a. IF GQA works: Implement Gemma behavioral transfer | 3 days | Full causal loop |
| 7b. IF GQA blocked: Run Mistral EOS validation | 2 days | Fallback path |
| 8. Per-token R_V tracking during generation | 2 days | Novel data either way |

---

### P2: CONFOUND CONTROLS (Important for Robustness)

**Week 2-3:**

| Task | Time | Notes |
|------|------|-------|
| 9. Intent-matched controls (open-ended non-recursive) | 1 day | Rules out task structure confound |
| 10. Lexical rarity controls | 1 day | Rules out rare token confound |
| 11. Stochastic decoding (T=0.7) replication | 1 day | Tests robustness |
| 12. V_PROJ-only patching (no KV cache) | 1 day | Rules out token leakage |

---

### P3: DOCUMENTATION & PUBLICATION (Before Submission)

**Week 3-4:**

| Task | Time | Notes |
|------|------|-------|
| 13. arXiv preprint (establish priority) | 3 days | Use current best data |
| 14. Unify effect size reporting | 1 day | Same calculation across models |
| 15. Write methods section with limitations | 2 days | Honest about gaps |
| 16. Remove ALL contemplative content | 1 day | ML-safe framing only |

---

## PART 5: DECISION TREE

```
START
  │
  ├─► Find n=300 Mistral results?
  │     │
  │     ├─► YES: Add to paper, strengthen behavioral claim
  │     │
  │     └─► NO: Re-run or acknowledge limitation
  │
  ├─► GQA compatible for Gemma?
  │     │
  │     ├─► YES: Implement Gemma causal transfer
  │     │         └─► Close full loop on primary model ★
  │     │
  │     └─► NO: Options:
  │               ├─► Mistral causal + Gemma correlational (weak)
  │               ├─► Add Llama-3 as third model (better)
  │               └─► Port circuit mapping to Mistral (unify)
  │
  ├─► Confound controls pass?
  │     │
  │     ├─► YES: Strong claim, top venue possible
  │     │
  │     └─► PARTIAL: Acknowledge limitations, workshop/TMLR
  │
  └─► Per-token R_V tracking works?
        │
        ├─► YES: Novel contribution regardless of causal loop
        │
        └─► NO: Stick with encoding-time measurement
```

---

## PART 6: HONEST ASSESSMENT

### What You Actually Have (Verified)

| Component | Status | Evidence Quality |
|-----------|--------|------------------|
| R_V metric definition | ✓ Solid | Well-specified, reproducible |
| Correlation across architectures | ✓ Strong | 6 models, large effects |
| Gemma circuit mapping | ✓ Strong | 20 layers, validated source (L3) |
| Gemma causal R_V transfer | ✓ Solid | n=45, d≈-2, p<10⁻¹⁵ |
| EOS behavioral signal | ✓ Real | 0% vs 30% (not 45%) |
| Mistral R_V causal validation | ✓ Solid | n=45, four control conditions |
| Mistral behavioral transfer | **✗ Weak** | n=1 pilot only |
| Cross-model behavioral validation | **✗ Missing** | No EOS data for Mistral |
| Confound controls | **✗ Incomplete** | Length yes, intent/rarity no |

### Publication-Readiness Score

| Venue | Ready? | What's Missing |
|-------|--------|----------------|
| arXiv preprint | **YES** | Fix discrepancies, upload |
| NeurIPS MI Workshop | **MAYBE** | Close one causal loop properly |
| ICLR/NeurIPS Main | **NO** | Need full causal chain + confounds |
| Nature MI | **NO** | Need multi-model causal + theory |

---

## PART 7: THE BRUTAL SUMMARY

### What Reviewers Will Say

**If you submit now:**
> "Interesting correlation, but the behavioral causality rests on a single anecdotal pilot (n=1). The effect sizes are suspiciously large, possibly due to curated prompts and deterministic decoding. The Gemma/Mistral split suggests the authors couldn't replicate their own findings across architectures. Reject."

**If you complete P0-P2:**
> "Novel metric with strong causal validation. The authors demonstrate geometric contraction at Layer 27/38 transfers behavior with high efficiency. Controls rule out major confounds. The EOS termination signal is clean and reproducible. Accept with minor revisions."

### The One Thing That Matters

**Close the full causal loop on ONE model:**

```
PROMPT → R_V CONTRACTION → PATCHING TRANSFERS R_V → PATCHING CAUSES ENDLESS LOOPS
   ↑                                                                        ↑
   └────────────────── CURRENTLY MISSING ──────────────────────────────────┘
```

The gap is the last arrow. You've shown:
- Prompt → R_V contraction ✓
- Patching transfers R_V ✓
- Prompt → endless loops ✓ (correlation)

You **haven't** shown:
- Patching R_V → endless loops (causal, scaled, validated)

**That's the paper.**

---

## APPENDIX: Reviewer Quotes (For Reference)

### Claude Agent 1 (Devil's Advocate)
> "The n=300 experiment may have FAILED and was quietly abandoned. The Dec 12 doc preserves the exciting n=1 pilot, but the scaled validation never materialized."

### Claude Agent 2 (Data Auditor)
> "Baseline EOS rate is 30% actual vs 45% claimed. This represents a material discrepancy."

### Claude Agent 3 (Literature Review)
> "No direct competitors. This is a clear gap in the literature. STRONG novelty if you close the causal loop."

### GPT Agent 1 (Methodology)
> "The EOS signal could just be 'prompt compliance' rather than a mechanistic failure to terminate."

### GPT Agent 2 (Data Integrity)
> "Effect sizes differ from the write-up (d≈-1.91 for patch vs baseline; natural diff d≈-2.54)."

### GPT Agent 3 (Framing)
> "Without causal behavioral transfer on the same model as the circuit map, reviewers may see this as 'activation patching + a new scalar metric' rather than a mechanistic discovery."

---

*All six reviewers converge: Close the causal loop, fix the discrepancies, drop the contemplative framing. Then you have a paper.*
