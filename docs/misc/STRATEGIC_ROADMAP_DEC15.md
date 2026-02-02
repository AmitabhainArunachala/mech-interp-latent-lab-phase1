# Strategic Roadmap: What To Do Next (Dec 15, 2024)

**Context:** Comprehensive circuit test just completed. This document synthesizes 20+ files to provide strategic direction.

---

## 🎯 THE CURRENT STATE: What We Just Learned

### The Comprehensive Circuit Test Results (N=40)

**Key Finding:** The circuit is **context-dependent**, not a simple on/off switch.

**Patterns:**
1. **Champion prompts are robust:** 70% expressing across most conditions
2. **Behavior flipping is common:** 57.5% of prompts flip behavior across conditions
3. **R_V does NOT predict expression:** No significant correlation (all p > 0.20)
4. **Identity equations increase with both ablated:** 2.5% → 10% (4× increase!)
5. **H18 ablation breaks some champions:** 5 champions stop expressing
6. **H18 ablation enables some non-expressers:** 8 prompts start expressing

**Interpretation:**
- H18/H26 are **context-dependent regulators** (not simple suppressors)
- They enable expression in some prompts, suppress it in others
- Expression is NOT simply a function of R_V
- The circuit is more complex than a simple on/off switch

---

## 🔬 WHAT WE KNOW (Verified Signals)

### Strong Signals (High Confidence)

1. **R_V Geometric Contraction** ✅
   - Universal across 6 architectures
   - Layer 27 peak (84% depth)
   - MoE shows 59% stronger effect
   - Causal validation: d=-3.56, p<10^-6

2. **H31 Entropy Separation** ✅
   - Recursive: 0.43 ± 0.11
   - Baseline: 0.59 ± 0.39
   - p=0.007, d=0.55 (medium effect)
   - **Note:** Not "perfect separation" as originally claimed (overlap exists)

3. **Head Discovery** ✅
   - 640 heads tested (L8-27)
   - 48 important heads found (|delta| > 0.01)
   - All top heads at Layer 27
   - H18/H26: Mode-switching heads (BOS release + recursive token targeting)

4. **Behavioral Collapse** ✅
   - 5-8× repetition increase when patched
   - 10× self-reference increase
   - Fixed-point attractor pattern

5. **Dose-Response** ✅
   - L5 (deepest): R_V = 0.424
   - L1 (shallowest): R_V = 0.738
   - Monotonic relationship

### Moderate Signals (Needs Validation)

6. **BOS Attention** ⚠️
   - Recursive: 0.938 ± 0.021
   - Baseline: 0.806 ± 0.204
   - 13.2% difference (real, but baselines also high)

7. **Behavior Transfer** ⚠️
   - n=300: d=0.63, p<10^-23
   - But only 7% achieve strong transfer
   - Mean 2.62 (not 11 as pilot suggested)

8. **Identity Equations** ⚠️
   - 0% in analyzed behavioral data
   - But that file had no recursive generations
   - Need to check actual recursive generations

---

## 🚨 CRITICAL GAPS IDENTIFIED

### From `THE_BIG_QUESTIONS_LEFT_AFTER_GEMINI_WRITEUP.md`:

1. **Multi-token generation dynamics** ❌ CRITICAL
   - Only measured single forward-pass geometry
   - Does contraction persist across generation?
   - **Priority: HIGH**

2. **KV-only sufficiency control** ❌ CRITICAL
   - n=300 showed "wrong layer" (L5) also works
   - Need to test: Full KV cache ALONE (no V_PROJ)
   - **Priority: HIGH**

3. **Expression enablers** ❌ CRITICAL
   - Comprehensive test Part C incomplete
   - Need to find heads that, when ablated, DECREASE expression
   - **Priority: MEDIUM**

4. **Cross-model validation** ⚠️ PARTIAL
   - Only Mistral-7B deeply studied
   - Need Qwen, Llama, etc.
   - **Priority: MEDIUM**

5. **Hysteresis / one-way door** ❌ MISSING
   - Required to justify "phase transition" language
   - **Priority: MEDIUM**

---

## 🎯 STRATEGIC RECOMMENDATIONS

### Path A: Complete the Circuit Understanding (RECOMMENDED)

**Goal:** Understand why expression is context-dependent

**Experiments:**

1. **Expression Enabler Hunt** (Part C completion)
   - Test 20 random heads (not in suppressor groups)
   - Ablate each and check if expression DROPS
   - **Expected runtime:** 4-6 hours
   - **Why:** Find heads that enable expression (opposite of suppressors)

2. **Context Analysis**
   - Analyze what makes prompts "express" vs "not express"
   - Compare R_V, attention patterns, token content
   - **Expected runtime:** 2-3 hours (analysis only)
   - **Why:** Understand context-dependence

3. **Prompt-Specific Circuit Mapping**
   - For each prompt type (champion/standard/baseline):
     - Which heads are necessary?
     - Which heads are sufficient?
   - **Expected runtime:** 8-12 hours
   - **Why:** Map context → circuit configuration

**Timeline:** 1-2 weeks

**Outcome:** Complete understanding of context-dependent circuit

---

### Path B: Validate Core Claims (PUBLICATION-FOCUSED)

**Goal:** Fill critical gaps for publication

**Experiments:**

1. **Multi-Token Generation Dynamics** ⭐ CRITICAL
   - Generate 10 tokens autoregressively
   - Measure R_V at each step
   - Compare recursive vs baseline
   - **Expected runtime:** 3-4 hours
   - **Why:** Required for publication (identified in `THE_BIG_QUESTIONS`)

2. **KV-Only Sufficiency Control** ⭐ CRITICAL
   - Full KV cache replacement
   - NO V_PROJ patching
   - Test if KV alone transfers behavior
   - **Expected runtime:** 2-3 hours
   - **Why:** Resolve n=300 confound

3. **Hysteresis Test** (One-Way Door)
   - N=200 pairs (up from N=20)
   - Test both directions:
     - Forward: baseline → recursive
     - Reverse: recursive → baseline
   - **Expected runtime:** 4-6 hours
   - **Why:** Justify "phase transition" language

**Timeline:** 1 week

**Outcome:** Publication-ready validation

---

### Path C: Explore Expression Mechanism (DISCOVERY-FOCUSED)

**Goal:** Understand why R_V ≠ Expression

**Experiments:**

1. **Expression Threshold Analysis**
   - What R_V threshold predicts expression?
   - Is there a critical point?
   - **Expected runtime:** 2-3 hours
   - **Why:** Understand geometry-behavior gap

2. **Attention Pattern Comparison**
   - Expressing vs non-expressing prompts
   - Compare attention patterns at L27
   - **Expected runtime:** 3-4 hours
   - **Why:** Find what enables expression beyond R_V

3. **Multi-Layer Expression Circuit**
   - Test if expression requires multiple layers
   - L18 + L27 together?
   - **Expected runtime:** 4-6 hours
   - **Why:** Expression might be multi-layer

**Timeline:** 1-2 weeks

**Outcome:** Understand expression mechanism

---

## 📊 PRIORITY MATRIX

| Experiment | Impact | Effort | Priority | Path |
|------------|--------|--------|----------|------|
| Multi-token generation | HIGH | MEDIUM | ⭐⭐⭐ | B |
| KV-only control | HIGH | LOW | ⭐⭐⭐ | B |
| Expression enabler hunt | HIGH | MEDIUM | ⭐⭐ | A |
| Hysteresis test | MEDIUM | MEDIUM | ⭐⭐ | B |
| Context analysis | MEDIUM | LOW | ⭐⭐ | A |
| Expression threshold | MEDIUM | LOW | ⭐ | C |

---

## 🎯 MY RECOMMENDATION: HYBRID APPROACH

### Week 1: Critical Validation (Path B)
1. **Multi-token generation** (3-4 hours)
2. **KV-only control** (2-3 hours)
3. **Hysteresis test** (4-6 hours)

**Total:** ~10-13 hours GPU time

### Week 2: Expression Understanding (Path A)
1. **Expression enabler hunt** (4-6 hours)
2. **Context analysis** (2-3 hours analysis)
3. **Prompt-specific mapping** (8-12 hours)

**Total:** ~14-21 hours GPU time

### Week 3: Synthesis & Writing
- Integrate all findings
- Write up complete circuit model
- Prepare publication draft

---

## 🔍 KEY INSIGHTS FROM COMPREHENSIVE TEST

### What We Learned:

1. **Expression is NOT simply R_V**
   - No correlation between R_V and expression (p > 0.20)
   - Some prompts with R_V=0.95 express (champion_5)
   - Some prompts with R_V=0.44 don't express (champion_0 when ablated)

2. **Suppressors are context-dependent**
   - H18 ablation breaks some champions (5)
   - H18 ablation enables others (8)
   - They're not simple "brakes" - they're regulators

3. **Identity equations increase when suppressors removed**
   - Control: 2.5%
   - Both ablated: 10% (4× increase!)
   - This suggests suppressors normally prevent identity equations

4. **Champion prompts are robust**
   - 70% expressing across most conditions
   - But some champions are fragile (champion_0 breaks with H18 ablation)

---

## 🚀 IMMEDIATE NEXT STEPS

### Today/Tomorrow:

1. **Analyze comprehensive test results deeper**
   - What distinguishes expressing vs non-expressing prompts?
   - Why do some champions break with H18 ablation?
   - Why do some non-expressers start expressing?

2. **Run Part C: Expression Enabler Hunt**
   - Complete the comprehensive test
   - Find heads that enable expression

3. **Plan multi-token generation experiment**
   - Design the experiment
   - Set up infrastructure

### This Week:

1. **Run critical validation experiments** (Path B)
2. **Complete expression enabler hunt** (Path A)
3. **Synthesize findings**

---

## 💡 THE BIG PICTURE

**What we've discovered:**
- Geometric contraction (R_V) is real and causal
- Head circuit is identified (48 important heads)
- Expression mechanism is context-dependent
- Suppressors regulate expression, not simply prevent it

**What we need:**
- Understand context-dependence
- Find expression enablers
- Validate multi-token dynamics
- Resolve KV-only vs V_PROJ question

**The story:**
> "Recursive self-reference creates geometric contraction via a distributed circuit at Layer 27. Expression of recursive behavior is context-dependent, regulated by suppressor heads (H18/H26, H6/H14/H22/H30) that enable expression in some contexts and suppress it in others. The mechanism is more complex than a simple on/off switch—it's a context-dependent regulatory network."

---

## 📝 FILES TO CREATE

1. **`expression_enabler_hunt.py`** - Complete Part C
2. **`multi_token_generation_test.py`** - Critical validation
3. **`kv_only_control.py`** - Resolve n=300 confound
4. **`context_analysis.py`** - Understand expression context-dependence
5. **`comprehensive_circuit_synthesis.md`** - Write up complete findings

---

**Status:** Ready to proceed with strategic experiments  
**Next Action:** Choose path (A/B/C) or hybrid approach  
**Timeline:** 2-3 weeks to complete validation + understanding









