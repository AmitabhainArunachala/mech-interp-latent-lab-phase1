# Lab Perspective: Fixed-Point Attractor Theory

**Date:** December 18, 2024  
**Context:** Analysis from experimental work on Mistral-7B

---

## What the Theory Gets Right

### 1. **The Three-Component Architecture is Spot-On**

Your framework predicts:
- **Steering (Dynamics)**: ✅ Confirmed - steering vector exists, has direction
- **KV (Initial Condition)**: ✅ Confirmed - without KV, we get drift/collapse
- **V_PROJ (Execution)**: ✅ Confirmed - V_PROJ patching maintains geometric signature

**But here's what we discovered:** The **Residual Stream** might be MORE important than V_PROJ for grounding!

**Evidence:**
- Residual+KV: **0.55 on-topic** (better than V_PROJ+KV at 0.46)
- Triple-Medium (Residual@L26 + V_PROJ@L27 + KV): **0.76 on-topic** (best!)

**Implication:** The "execution" component might be distributed across residual stream AND V_PROJ, not just V_PROJ.

---

### 2. **"Steering Alone Falls to Simplest Fixed Point" - CONFIRMED**

**What we observed:**
- Steering alone (no KV, no V_PROJ): 100% garbage
- Random vector (same norm): Similar garbage
- Both produce: topic drift, collapse, repetitive loops

**Your theory explains this perfectly:**
> "Without KV: The equation becomes S = f(S) with no constraint → falls to simplest fixed point (tautology)"

**What we saw:**
- Prompt: "Calculate 12 × 3 + 4"
- Output: Academic citation loops (O'Brien, D. V...)
- This IS the "simplest fixed point" - repetitive structure with no content

**The theory is validated here.**

---

### 3. **"Contraction = Approaching Fixed Point" - Needs Nuance**

**What we know:**
- R_V < 1.0 at L27 for recursive prompts ✅
- R_V ≈ 1.0 for baseline prompts ✅
- This suggests contraction dynamics ✅

**But:**
- We haven't measured R_V DURING generation with interventions
- We don't know if R_V contraction is CAUSAL or CORRELATIONAL
- The steering control experiment showed random vectors ALSO cause drift

**Question:** Is R_V contraction the signature of approaching a fixed point, or just a side effect of repetitive patterns?

**Test needed:** Measure R_V trajectory during generation with/without interventions.

---

## What the Theory Might Miss

### 1. **The Residual Stream Discovery**

Your framework emphasizes V_PROJ as the "execution" component, but:

**Our finding:** Residual stream steering (projected through V_PROJ weight matrix) works BETTER than direct V_PROJ patching for maintaining topic grounding.

**Why this matters:**
- Residual stream is the "state" that flows through layers
- V_PROJ is a projection FROM residual stream
- Modifying residual stream might be more fundamental than modifying V_PROJ

**Updated architecture:**
```
RECURSIVE MODE = DYNAMICS + INITIAL CONDITION + STATE MODIFICATION

STEERING VECTOR (Dynamics)
  ↓
RESIDUAL STREAM (State) ← Modified here (L26)
  ↓
V_PROJ (Projection) ← Also modified here (L27)
  ↓
KV CACHE (Grounding) ← Replaced here (L27)
```

**The triple-system intervention suggests:** Modifying state BEFORE projection might be more effective than modifying projection alone.

---

### 2. **The "Single Direction" Claim Needs Testing**

**Your theory:** "The steering vector is the dominant eigenvector of the recursive operator"

**Our control experiment:** Random vector (same norm) produced similar drift patterns.

**This suggests:**
- Maybe ANY perturbation without KV causes drift?
- Or maybe the steering vector IS special, but we need better evaluation?

**Critical test needed:** Compare steering vector outputs vs. random vector outputs MANUALLY for genuine recursive patterns (not regex).

---

### 3. **The "Content Binding" Equation**

Your equation: `S = Attention(S, KV) + Δ`

**Our observations:**
- Triple-Medium (Residual@L26 + V_PROJ@L27 + KV@L27): Best grounding (0.76)
- This suggests the equation might be:

```
S_{L27} = Attention(S_{L26} + α·v_residual, KV_{L27}) + α·v_vproj
```

**The residual modification happens BEFORE attention, V_PROJ modification happens DURING attention.**

**This is more nuanced than your equation suggests.**

---

## Predictions We Can Test NOW

### Prediction 1: Phase Transition at Critical Alpha

**Your prediction:** Sharp transition at critical α where system falls into recursive attractor.

**We can test:** We have alpha sweeps from steering experiments. Check if there's a threshold.

**Status:** ✅ Data exists, needs analysis

---

### Prediction 2: Cross-Prompt Convergence

**Your prediction:** Under recursive mode, inter-prompt distances shrink.

**We can test:** Extract L27 activations from all 10 prompts under each config, compute pairwise distances.

**Status:** ⚠️ Need to add this to pipeline

---

### Prediction 3: Noise Resilience

**Your prediction:** If true attractor, trajectory self-corrects.

**We can test:** Inject noise into residual stream during generation, see if outputs recover.

**Status:** ⚠️ Need to implement noise injection

---

### Prediction 4: Steering Vector Eigenstructure

**Your prediction:** Top PCs align with self-referential token embeddings.

**We can test:** PCA on steering vector, check alignment with token embeddings.

**Status:** ✅ Can do this now with existing steering vector

---

## The Critical Gap: Manual Evaluation

**Your framework assumes:** We can detect recursive outputs.

**Our reality:** Regex-based scoring shows 0.00 recursion for ALL configs, even triple-medium with 0.76 on-topic.

**This means:**
1. Either regex patterns are wrong (likely)
2. Or genuine recursion requires different patterns than we're looking for
3. Or recursion is STRUCTURAL (patterns in how text is organized) not LEXICAL (specific words)

**The triple-medium outputs need manual review:**
- Are they showing structural recursion?
- Are they applying self-reference to the prompt content?
- Or are they just well-grounded factual answers?

---

## Updated Theory Based on Lab Work

### The Four-Component Architecture

```
RECURSIVE MODE = DYNAMICS + STATE + PROJECTION + GROUNDING

1. STEERING VECTOR (Dynamics)
   - Points toward attractor basin
   - Can be applied to residual OR V_PROJ

2. RESIDUAL STREAM (State)
   - The actual state vector flowing through layers
   - Modification here (L26) primes the state BEFORE attention

3. V_PROJ (Projection)
   - Projects state into attention space
   - Modification here (L27) biases attention patterns

4. KV CACHE (Grounding)
   - Provides content anchor
   - Prevents drift to empty fixed points
```

**Key insight:** Residual modification (L26) + V_PROJ modification (L27) + KV replacement (L27) = Best grounding (0.76 on-topic)

**This suggests:** The attractor basin has multiple entry points, and using multiple entry points simultaneously improves convergence.

---

## What We Still Don't Know

### 1. **Is the Steering Vector Actually Special?**

- Random vector also causes drift
- But does random vector produce GENUINE recursion, or just garbage?
- Need manual review to distinguish

### 2. **What is "Genuine Recursion" vs. "Structural Patterns"?**

- Regex shows 0.00 for all configs
- But triple-medium has 0.76 on-topic
- Are these outputs recursively self-referential, or just well-grounded?

### 3. **Is R_V Contraction Causal or Correlational?**

- We see R_V < 1.0 for recursive prompts
- But does CAUSING R_V contraction produce recursion?
- Or is R_V just a side effect?

### 4. **Why Does Triple-Medium Work Best?**

- Residual@L26 (α=1.5) + V_PROJ@L27 (α=1.5) + KV@L27
- Why does modifying BOTH residual AND V_PROJ improve grounding?
- Is it because we're hitting the attractor from multiple angles?

---

## The Most Compelling Prediction

**Your Prediction 2: Phase Transition**

> "Sharp transition at critical α where system suddenly falls into recursive attractor. Not gradual - threshold behavior."

**Why this matters:**
- If true, there's a CRITICAL ALPHA where everything changes
- Below threshold: Normal processing
- Above threshold: Recursive mode
- This would explain why some interventions work and others don't

**We should test:** Alpha sweep from 0.0 to 5.0, measure:
- On-topic rate
- Coherence
- Manual recursion score
- R_V trajectory

**If there's a sharp transition, your theory is validated.**

---

## The Deepest Insight from Lab Work

**Your framework says:** "Transformers implement a learned algorithm for diagonalization"

**What we've observed:**
- The algorithm exists (steering vector works)
- The geometry exists (R_V contraction)
- The grounding mechanism exists (KV cache)
- But the EXECUTION is distributed (residual + V_PROJ, not just V_PROJ)

**The updated picture:**
> Transformers implement diagonalization through a **distributed attractor system** where:
> - Residual stream modification primes the state
> - V_PROJ modification biases the projection
> - KV cache provides the content anchor
> - Together, they converge to a fixed point that applies self-reference TO specific content

---

## What We Should Test Next

### Priority 1: Manual Review of Triple-Medium Outputs
- Are they genuinely recursive, or just well-grounded?
- This determines if the theory is validated or needs revision

### Priority 2: Alpha Phase Transition
- Sweep alpha from 0.0 to 5.0
- Look for sharp transition point
- This validates Prediction 2

### Priority 3: R_V Trajectory During Generation
- Measure R_V at each generation step
- See if it contracts as theory predicts
- This validates the contraction mapping claim

### Priority 4: Cross-Prompt Distance Analysis
- Extract L27 activations from all prompts
- Compute pairwise distances
- Check if recursive mode reduces distances
- This validates Prediction 4

---

## Final Thought

**Your theory is elegant and explains most of what we've seen.**

**But the lab work suggests:**
- The mechanism is MORE distributed than V_PROJ alone
- Residual stream modification is crucial
- Triple-system intervention (residual + V_PROJ + KV) works best
- Manual evaluation is essential (regex fails)

**The theory needs one update:**
> The recursive attractor is accessed through **multiple entry points** (residual stream, V_PROJ, KV cache), and using multiple entry points simultaneously improves convergence to the fixed point.

**This is still a fixed-point attractor theory - just with a more nuanced entry mechanism.**

---

*"The geometry exists. The dynamics exist. The grounding exists. Now we need to verify the phenomenology."* — Lab Notes, Dec 18, 2024








