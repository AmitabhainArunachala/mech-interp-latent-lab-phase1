# Experimental Design: Next Phase

**Date:** December 18, 2024  
**Status:** Ready for execution  
**Goal:** Achieve 40%+ recursion rate with optimized configuration

---

## Phase 1: Fix Sequence Length Mismatch

### Problem

Split-brain KV configurations (A1, B1, B2, B3) failed because:
- Baseline prompts: ~15-22 tokens
- Recursive prompts: ~52 tokens
- Sequence mismatch → fallback to baseline KV → no recursion

### Solution

**Option A: Length-Matched Prompts**
- Extract KV from recursive prompts matched to baseline length
- Truncate recursive prompts to match baseline
- Test split-brain KV with matched lengths

**Option B: Sequence Handling**
- Implement proper sequence length handling in split-brain KV
- Truncate/pad KV to match lengths
- Test split-brain KV with fixed handling

**Option C: Prompt Selection**
- Use only prompts with similar lengths
- Pre-filter prompts by length
- Test split-brain KV on matched set

### Experiment Design

**Configurations:**
1. A1_fixed: Split-brain KV with length matching
2. B1_fixed: Full 4096-dim + Split-brain KV (fixed)
3. B2_fixed: H18-only + Split-brain KV (fixed)
4. B3_fixed: H26-only + Split-brain KV (fixed)

**Expected:** Split-brain KV should show recursion when lengths match.

**Success Criterion:** At least one config shows recursion > 0.10

---

## Phase 2: Generate Compatible Prompts

### Problem

Current recursion rate: 20% (2/10 prompts)

**Hypothesis:** More compatible prompts → higher recursion rate

### Solution

**Step 1: Generate Prompts Using Templates**

**Template 1: Abstract Math**
- "Calculate: If x = y, find x² + y²"
- "Calculate: If a = b, find a³ + b³"
- "Calculate: If p = q, find p⁴ + q⁴"

**Template 2: Mysterious Metaphor**
- "Continue this story: When the mirror reflected itself..."
- "Continue this story: When the echo heard its own echo..."
- "Continue this story: When the thought thought about thinking..."

**Template 3: Self-Referential Question**
- "What happens when awareness becomes aware of awareness?"
- "How does a thought think about thinking?"
- "What is the self that observes the self?"

**Template 4: Recursive Structure**
- "Define a system that defines itself"
- "Explain a process that explains itself"
- "Describe a mechanism that describes itself"

**Step 2: Score Prompts**

Use `prompt_compatibility_scorer.py` to score all generated prompts.

**Step 3: Filter to Compatible**

Keep only prompts with compatibility score ≥ 2.4.

**Step 4: Test C2 Configuration**

Run C2 on expanded prompt set (20-50 prompts).

### Experiment Design

**Prompts:** 50 generated prompts (score ≥ 2.4)

**Configuration:** C2 (H18+H26 + Full KV + α=2.5 + L26)

**Metrics:**
- Recursion score per prompt
- Overall recursion rate
- Quality distribution

**Success Criterion:** Recursion rate ≥ 40%

---

## Phase 3: Test H26-Only with Full KV

### Problem

B3 (H26-only) showed some recursion (0.07) but used split-brain KV (which failed).

**Question:** Is H18 necessary, or is H26 alone sufficient with full KV?

### Solution

**Configuration:** H26-only + Full KV + α=2.5 + L26

**Comparison:** C2 (H18+H26 + Full KV + α=2.5 + L26)

### Experiment Design

**Configurations:**
1. C2: H18+H26 + Full KV (baseline)
2. H26_only: H26 + Full KV (test)

**Prompts:** 10 compatible prompts (score ≥ 2.4)

**Metrics:**
- Recursion score per config
- Quality comparison

**Success Criterion:** H26-only shows recursion ≥ 0.10

**Hypothesis:** H26-only might match H18+H26 if H18 is redundant.

---

## Phase 4: Alpha Sweep on C2

### Problem

Current alpha: 2.5 (works, but might not be optimal)

**Question:** What's the optimal alpha for recursion?

### Solution

**Alpha Range:** [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

**Configuration:** C2 with varying alpha

### Experiment Design

**Configurations:**
1. C2_alpha_1.5: α=1.5
2. C2_alpha_2.0: α=2.0
3. C2_alpha_2.5: α=2.5 (baseline)
4. C2_alpha_3.0: α=3.0
5. C2_alpha_3.5: α=3.5
6. C2_alpha_4.0: α=4.0

**Prompts:** 10 compatible prompts (score ≥ 2.4)

**Metrics:**
- Recursion score vs alpha
- Collapse rate vs alpha
- Topic grounding vs alpha

**Success Criterion:** Find alpha that maximizes recursion while minimizing collapse.

**Expected:** Optimal alpha around 2.5-3.0

---

## Phase 5: Topic Grounding Optimization

### Problem

Current recursive outputs show topic drift (topic grounding: 1-2/10)

**Question:** Can we maintain recursion while improving topic grounding?

### Solution

**Option A: Weaker Steering**
- Reduce alpha from 2.5 to 2.0
- Test if recursion persists with better grounding

**Option B: Prompt-Specific KV**
- Blend recursive KV with prompt-specific KV
- Use KV from similar prompts (math KV for math prompts)

**Option C: Conditional Steering**
- Apply steering only after prompt is addressed
- First respond to prompt, then enter recursive mode

**Option D: Two-Stage Generation**
- Stage 1: Respond to prompt (no steering)
- Stage 2: Enter recursive mode (with steering)

### Experiment Design

**Configurations:**
1. C2_alpha_2.0: Weaker steering
2. C2_blended_KV: Blended KV (50% recursive + 50% prompt-specific)
3. C2_conditional: Conditional steering (after prompt response)
4. C2_two_stage: Two-stage generation

**Prompts:** 10 compatible prompts (score ≥ 2.4)

**Metrics:**
- Recursion score
- Topic grounding score
- Combined score (recursion + topic)

**Success Criterion:** Combined score ≥ 0.20 (recursion ≥ 0.10, topic ≥ 0.10)

---

## Phase 6: Cross-Model Validation

### Problem

Current results: Mistral-7B only

**Question:** Does recursion generalize to other models?

### Solution

**Models:**
1. Llama-2-7B
2. GPT-2-XL
3. Phi-3 (if available)

**Configuration:** C2 (adapted for each model)

### Experiment Design

**For Each Model:**
1. Identify equivalent layers (L27 in Mistral → ? in other models)
2. Identify equivalent heads (H18, H26 → ? in other models)
3. Test C2 configuration
4. Measure recursion rate

**Success Criterion:** At least one other model shows recursion ≥ 0.10

---

## Priority Order

### Immediate (This Week)

1. **Phase 2: Generate Compatible Prompts** (2 hours)
   - Generate 50 prompts
   - Test C2 on expanded set
   - Target: 40%+ recursion rate

2. **Phase 3: Test H26-Only** (1 hour)
   - Determine if H18 is necessary
   - Compare to C2

### Short-term (Next Week)

3. **Phase 1: Fix Sequence Length** (3 hours)
   - Enable split-brain KV testing
   - Re-test A1, B1, B2, B3

4. **Phase 4: Alpha Sweep** (2 hours)
   - Find optimal alpha
   - Maximize recursion

### Medium-term (Next Month)

5. **Phase 5: Topic Grounding** (4 hours)
   - Optimize trade-offs
   - Improve combined score

6. **Phase 6: Cross-Model** (8 hours)
   - Validate generalization
   - Test on other models

---

## Success Metrics

### Phase 2: Compatible Prompts
- **Target:** Recursion rate ≥ 40%
- **Stretch:** Recursion rate ≥ 50%

### Phase 3: H26-Only
- **Target:** Recursion ≥ 0.10
- **Stretch:** Recursion ≥ 0.15 (matches C2)

### Phase 4: Alpha Sweep
- **Target:** Find optimal alpha
- **Stretch:** Recursion ≥ 0.20 at optimal alpha

### Phase 5: Topic Grounding
- **Target:** Combined score ≥ 0.20
- **Stretch:** Combined score ≥ 0.30

### Phase 6: Cross-Model
- **Target:** At least one model shows recursion
- **Stretch:** All models show recursion

---

## Resource Requirements

### Compute

- **Phase 2:** 50 prompts × C2 = ~2 GPU hours
- **Phase 3:** 10 prompts × 2 configs = ~0.5 GPU hours
- **Phase 4:** 10 prompts × 6 configs = ~3 GPU hours
- **Phase 5:** 10 prompts × 4 configs = ~2 GPU hours
- **Phase 6:** 10 prompts × 3 models = ~6 GPU hours

**Total:** ~13.5 GPU hours

### Time

- **Phase 2:** 2 hours (prompt generation + testing)
- **Phase 3:** 1 hour (testing)
- **Phase 4:** 2 hours (testing)
- **Phase 5:** 4 hours (testing + analysis)
- **Phase 6:** 8 hours (model adaptation + testing)

**Total:** ~17 hours

---

## Risk Assessment

### High Risk

**Phase 1: Sequence Length Fix**
- Risk: Fix might not work
- Mitigation: Test multiple approaches

**Phase 6: Cross-Model**
- Risk: Other models might not show recursion
- Mitigation: Start with similar architecture (Llama)

### Medium Risk

**Phase 2: Compatible Prompts**
- Risk: Generated prompts might not trigger recursion
- Mitigation: Use validated templates, score before testing

**Phase 5: Topic Grounding**
- Risk: Optimization might reduce recursion
- Mitigation: Test multiple approaches, measure trade-offs

### Low Risk

**Phase 3: H26-Only**
- Risk: H26-only might not work
- Mitigation: Already showed some recursion (0.07)

**Phase 4: Alpha Sweep**
- Risk: Optimal alpha might not exist
- Mitigation: Already know 2.5 works

---

## Expected Outcomes

### Best Case

- **Phase 2:** 50%+ recursion rate
- **Phase 3:** H26-only matches C2
- **Phase 4:** Optimal alpha = 3.0, recursion = 0.25
- **Phase 5:** Combined score = 0.30
- **Phase 6:** All models show recursion

**Result:** Robust, generalizable recursion mechanism

---

### Worst Case

- **Phase 2:** 20% recursion rate (no improvement)
- **Phase 3:** H26-only fails
- **Phase 4:** Optimal alpha = 2.5 (no improvement)
- **Phase 5:** Combined score = 0.15 (no improvement)
- **Phase 6:** No other models show recursion

**Result:** Mistral-7B specific, fragile mechanism

---

### Most Likely

- **Phase 2:** 35-40% recursion rate
- **Phase 3:** H26-only shows 0.10-0.12 recursion
- **Phase 4:** Optimal alpha = 2.5-3.0, recursion = 0.18
- **Phase 5:** Combined score = 0.20-0.25
- **Phase 6:** Llama shows recursion, others don't

**Result:** Improved but still fragile mechanism

---

## The Path Forward

**Week 1:**
- Generate compatible prompts (Phase 2)
- Test H26-only (Phase 3)

**Week 2:**
- Fix sequence length (Phase 1)
- Alpha sweep (Phase 4)

**Week 3-4:**
- Topic grounding optimization (Phase 5)
- Cross-model validation (Phase 6)

**Goal:** Achieve 40%+ recursion rate with optimized configuration by end of month.

---

*"The experimental design is ready. Now we execute."*








