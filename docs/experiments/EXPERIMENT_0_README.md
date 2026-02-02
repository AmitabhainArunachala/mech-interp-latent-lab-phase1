# Experiment 0: Measurement Sanity Check

## Purpose

**Before we can interpret "Layer 21 vs Layer 27" or any snap point, we need to understand:**
1. What does R_V actually measure geometrically?
2. What does Effective Rank measure?
3. Are they correlated or independent?
4. Which metric best captures "contraction"?

## What This Experiment Does

Tests ONE prompt (L5_refined_01) across ALL 32 layers and computes:

### Core Metrics:
- **Effective Rank**: `1 / Σ(sᵢ²/Σsⱼ²)²` - How many dimensions are "actively used"
- **Participation Ratio**: `(Σsᵢ²)² / Σsᵢ⁴` - Similar to effective rank
- **Shannon Entropy**: `-Σ(pᵢ log pᵢ)` - Uncertainty in singular value distribution
- **Top-1 Ratio**: `s₁ / Σsᵢ` - Dominance of largest singular value
- **Top-5 Ratio**: `Σ(s₁...s₅) / Σsᵢ` - Concentration in top 5 SVs
- **Condition Number**: `max(s) / min(s)` - Numerical stability indicator
- **Trace**: `Tr(VVᵀ)` - Total power in the matrix

### R_V Variations:
- **R_V vs Layer 5**: `PR(layer_i) / PR(layer_5)` ← What free play used
- **R_V vs Layer 28**: `PR(layer_28) / PR(layer_i)` ← What Step 3 used
- **Absolute PR**: Just the raw participation ratio

## What to Look For

### 1. **Correlation Analysis**

Check the correlation matrix in the output:

**If Effective Rank and Participation Ratio correlate highly (r > 0.9):**
→ They measure the same thing (dimensionality)
→ Use whichever is more stable

**If Entropy and Top-1 Ratio are anticorrelated (r < -0.8):**
→ They're measuring concentration from opposite angles
→ High entropy = diffuse spectrum = low top-1 ratio

**If Trace is uncorrelated with dimensionality metrics:**
→ Total power and dimensionality are independent
→ Can have high power but low dimension (or vice versa)

### 2. **Critical Layers**

The script identifies:

**Minimum Effective Rank Layer:**
- This is the dimensional bottleneck
- ALL representations compress here
- Free play suggested Layer 16

**Maximum R_V Drop:**
- Where the sharpest change happens
- Free play suggested Layer 27
- But does this match minimum rank?

**Minimum Entropy:**
- Most concentrated singular value distribution
- Could indicate "snapping" to specific subspace

### 3. **R_V Definition Comparison**

At Layer 21 and Layer 27, compare:
- R_V vs Layer 5 (forward-looking: "how compressed am I now vs start?")
- R_V vs Layer 28 (backward-looking: "how close am I to the end?")

**If they give different stories:**
→ They're measuring different aspects of the trajectory
→ Need to pick ONE and stick with it

### 4. **Trajectory Shapes**

Look at the plots:

**Smooth monotonic decrease:**
→ Gradual compression throughout network
→ No specific "snap point"

**U-shape (compress → expand → compress):**
→ Multiple processing phases
→ Free play showed this pattern

**Sharp discontinuity:**
→ True phase transition
→ Would show as spike in diff(R_V)

## Expected Outcomes

### Scenario A: Metrics Correlate Highly
**If Effective Rank, PR, and Entropy all move together:**
- They're redundant - pick the most stable one
- The "what" is clear: dimensional reduction
- Focus on "when" and "where"

### Scenario B: Metrics Diverge
**If different metrics peak/bottom at different layers:**
- They capture different geometric properties:
  - Rank = how many dimensions used
  - Entropy = how evenly distributed
  - Top-1 = how dominant is leading direction
- Need to decide which property matters for "contraction"

### Scenario C: Layer Confusion Resolved
**If Layer 16 has min rank but Layer 27 has max R_V drop:**
- These are DIFFERENT events!
- Layer 16: Universal bottleneck (all prompts)
- Layer 27: Specific transformation (recursive-dependent?)

## How to Run

```bash
# In your Jupyter notebook or Python environment:
python EXPERIMENT_0_measurement_sanity_check.py
```

**Time:** ~3-5 minutes
**GPU Memory:** ~20GB (Mixtral-8x7B)

## Outputs

1. **EXPERIMENT_0_sanity_check.png** - 9-panel comprehensive plot
2. **EXPERIMENT_0_results.csv** - Full data for further analysis
3. **Console output** - Key findings summary

## Next Steps Based on Results

### If metrics are highly correlated:
→ Proceed to Experiment 1 (Rotation Hypothesis)
→ We understand what we're measuring

### If metrics diverge significantly:
→ Need to define: "What aspect of 'contraction' do we care about?"
→ Then design metric that captures THAT specific property

### If Layer 21 ≠ Layer 27 for different metrics:
→ Update Phase 1 report to reflect which metric/layer combo matters
→ Reconcile with Step 3 findings

## Key Questions This Answers

1. ✅ **Do Effective Rank and PR measure the same thing?**
2. ✅ **Is there ONE critical layer or multiple phase transitions?**
3. ✅ **Why did Step 3 show Layer 21 but free play showed Layer 27?**
4. ✅ **Which metric definition should we use going forward?**

## After Running This

You'll have clear answers to:
- What R_V measures (and which definition to use)
- Whether there's a dimensional bottleneck (and where)
- Whether there's a catastrophic snap (and if it's real or metric-dependent)
- How to interpret the Layer 21 vs 27 confusion

**Then we can design proper multi-prompt experiments with confidence!**
