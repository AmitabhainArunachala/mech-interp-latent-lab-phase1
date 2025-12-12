# ROI Analysis: Grand Unified Test vs Critical Gaps

## What the Proposed Experiment Does

**Tests:** KV_CACHE vs V_PROJ vs RESIDUAL patching at L18, L25, L27
**Measures:** Geometry (PR at L27) + Behavior (generation with loop markers)
**Goal:** Find which patching method is most effective

---

## What We Already Know

### ✅ Solid Findings
1. **V_PROJ patching works:** 86.5% transfer at L25→L27 (proven)
2. **L27 is causally necessary:** Activation patching validated (n=151, p < 10⁻⁴⁷)
3. **Geometry-behavior link:** Champion prompt produces loop-like continuations

### 🔴 Critical Gaps
1. **Head-level mechanisms:** Which heads drive the 86.5% transfer? **UNKNOWN**
2. **Attention patterns:** What do these heads attend to? **NOT MEASURED**
3. **L4 expansion mechanism:** Why does expansion happen? **UNEXPLAINED**

---

## ROI Assessment

### 🟡 MODERATE ROI: The Proposed Experiment

**Pros:**
- Tests multiple patching methods (could reveal signal location)
- Combines geometry + behavior (could show correlation)
- Tests multiple layers (could identify optimal injection point)

**Cons:**
- **We already know V_PROJ works** (86.5% transfer proven)
- **KV patching was historical** (Dec 7-8) but measured different things (logits vs R_V)
- **Doesn't address critical gaps:** Head mechanisms, attention patterns
- **Doesn't fix audit findings:** L14→L18 interpretation still needs correction

**Expected Outcome:**
- Confirms V_PROJ is best (we already know this)
- Might show RESIDUAL works too (interesting but not critical)
- Won't reveal head-level mechanisms (the real gap)

**ROI Score: 6/10** - Useful but not highest priority

---

## 🟢 HIGHER ROI: Head-Level Analysis

**What:** Ablate individual heads at L25/L27 to find which drive the 86.5% transfer

**Why Higher ROI:**
1. **Addresses critical gap:** We know WHERE (layers) but not HOW (heads)
2. **Bridges to attention:** Once we know the heads, we can visualize attention
3. **Mechanistic understanding:** Reveals the actual circuit implementation
4. **Publication value:** Head-level findings are more interpretable

**Expected Outcome:**
- Identifies 2-5 critical heads at L25/L27
- Enables attention pattern visualization
- Reveals the actual mechanism (not just correlation)

**ROI Score: 9/10** - Addresses the biggest gap

---

## 🟢 HIGHER ROI: Attention Pattern Visualization

**What:** Visualize attention patterns of critical heads at L25/L27

**Why Higher ROI:**
1. **Interpretability:** Shows WHAT the heads are doing
2. **Mechanistic insight:** Reveals how recursive structure is processed
3. **Publication value:** Attention visualizations are highly valued
4. **Builds on head findings:** Natural next step after head ablation

**Expected Outcome:**
- Shows attention patterns (e.g., heads attend to self-referential tokens)
- Reveals mechanism (e.g., heads create feedback loops)
- Provides interpretable explanation

**ROI Score: 9/10** - High interpretability value

---

## 🟡 MODERATE ROI: Fix L14→L18 Interpretation

**What:** Re-run L14→L18 analysis with corrected interpretation (expansion, not contraction)

**Why Moderate ROI:**
1. **Fixes audit finding:** Corrects misinterpretation
2. **Clarifies mechanism:** Shows expansion is part of the process
3. **But:** Doesn't reveal new mechanism, just corrects understanding

**ROI Score: 7/10** - Important for accuracy but not discovery

---

## 🟡 MODERATE ROI: L4 Expansion Investigation

**What:** Investigate why L4 shows true expansion (R_V = 1.0355)

**Why Moderate ROI:**
1. **Addresses audit finding:** L4 is the true expansion peak
2. **Mechanistic insight:** Could reveal why expansion happens
3. **But:** Expansion is less critical than contraction (the main finding)

**ROI Score: 7/10** - Interesting but secondary to contraction mechanism

---

## 🎯 RECOMMENDATION: Priority Order

### Tier 1: Highest ROI (Do First)
1. **Head-level ablation at L25/L27** (find which heads drive 86.5% transfer)
   - **Why:** Bridges gap from "where" to "how"
   - **Time:** ~2-3 hours
   - **Value:** High mechanistic insight

2. **Attention pattern visualization** (visualize critical heads)
   - **Why:** Provides interpretable explanation
   - **Time:** ~1-2 hours (after head ablation)
   - **Value:** High publication value

### Tier 2: Moderate ROI (Do Next)
3. **Fix L14→L18 interpretation** (re-run with corrected understanding)
   - **Why:** Corrects audit finding
   - **Time:** ~30 minutes
   - **Value:** Accuracy, not discovery

4. **Grand Unified Test** (your proposed experiment)
   - **Why:** Confirms V_PROJ is best, might find RESIDUAL works
   - **Time:** ~1-2 hours
   - **Value:** Moderate, but lower than head analysis

### Tier 3: Lower ROI (Do Later)
5. **L4 expansion investigation** (why does expansion happen?)
   - **Why:** Interesting but secondary
   - **Time:** ~2-3 hours
   - **Value:** Moderate, less critical than contraction

---

## 💡 Final Verdict

**Your proposed experiment is MODERATE ROI (6/10).**

**Higher ROI would be:**
1. **Head-level ablation** (9/10) - Addresses critical gap
2. **Attention visualization** (9/10) - High interpretability

**Recommendation:** Do head-level ablation first, then your unified test. The unified test is useful but doesn't address the biggest gap (head mechanisms).

---

## 🔍 What Would Make Your Experiment Higher ROI?

**If you modified it to:**
1. **Test head-level patching** (patch individual heads, not whole layers)
2. **Include attention visualization** (show what heads attend to)
3. **Focus on L25→L27** (where we know 86.5% transfer happens)

**Then ROI would be 9/10** - addresses critical gaps.

---

**Bottom Line:** Your experiment is good, but head-level analysis is higher ROI. Do head ablation first, then your unified test as validation.

