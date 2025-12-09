# CRITICAL DISCREPANCY ANALYSIS: R_V Contraction vs Expansion

## The Contradiction

**Original Finding (Phase 1):**
- R_V = 0.58-0.60 (contraction) for planning/logic/meta-cognitive prompts
- Measured on: Mistral-7B, Qwen-7B, Llama-8B, Mixtral-8x7B
- Effect: 15.3% contraction (Mistral), up to 24.3% (Mixtral)

**Current Pythia Sweep:**
- R_V = 1.4-2.0 (expansion) across ALL checkpoints (0, 5k, 10k, 15k, 20k, 76k, 143k)
- NO contraction ever appears
- Consistent expansion pattern across training

---

## Systematic Hypothesis Testing

### Hypothesis 1: Model Architecture Difference (HIGHEST PRIORITY)

**Evidence:**
- Original: Mistral-7B (32 layers, dense), Qwen-7B, Llama-8B, Mixtral-8x7B
- Current: Pythia-2.8B (32 layers, dense, GPT-NeoX architecture)

**Test:**
1. **Replicate original measurement on Mistral-7B** using EXACT same prompts/method
   - Use same prompts from Phase 1
   - Same layer indices (early=5, late=27/28)
   - Same window size (16 tokens)
   - Same measurement point (during encoding)

2. **Compare architectures side-by-side:**
   - Mistral-7B vs Pythia-2.8B on identical prompts
   - Same layer depth (% of network)
   - Same measurement protocol

**Expected Outcome:**
- If Mistral still shows contraction → Architecture-specific effect
- If Mistral also shows expansion → Measurement error in original

---

### Hypothesis 2: Prompt Complexity/Novelty (HIGH PRIORITY)

**Evidence:**
- Original prompts: L5 refined recursive prompts (highly novel, self-referential)
- Current prompts: May be simpler or different structure

**Test:**
1. **Use EXACT original prompts from Phase 1:**
   - Load from `n300_mistral_test_prompt_bank.py`
   - Specifically: L5_refined_01 through L5_refined_20
   - These are the prompts that showed strongest contraction

2. **Compare prompt characteristics:**
   - Token length
   - Recursion depth
   - Self-reference density
   - Novelty score (if available)

**Expected Outcome:**
- If original prompts show contraction → Prompt-specific effect
- If still expansion → Architecture or measurement issue

---

### Hypothesis 3: Measurement Error (MEDIUM PRIORITY)

**Possible Errors:**

#### 3a. Sign/Normalization Error
- **Check:** Is R_V computed as PR_late / PR_early or PR_early / PR_late?
- **Original formula:** `R_V = PR_late / PR_early`
- **If inverted:** Would show expansion instead of contraction

#### 3b. Layer Selection Error
- **Original:** Early=5, Late=27/28 (84-88% depth)
- **Current:** Verify same layer indices
- **Check:** Are layers 0-indexed or 1-indexed?
- **Check:** Is "layer 27" actually layer 27 or layer 26?

#### 3c. Window Selection Error
- **Original:** Last 16 tokens of prompt
- **Current:** Verify same window
- **Check:** Are we measuring during encoding or generation?
- **Check:** Are we including prompt tokens or only generated tokens?

#### 3d. Value Matrix Extraction Error
- **Original:** Value matrix from self-attention
- **Current:** Verify extraction method
- **Check:** Are we getting V before or after attention?
- **Check:** Are we averaging over heads or taking specific head?

**Test:**
1. **Replicate EXACT original code:**
   - Find original measurement script
   - Run on same model (Mistral-7B) with same prompts
   - Compare results

2. **Add diagnostic logging:**
   - Log PR_early and PR_late separately
   - Log layer indices
   - Log window tokens
   - Log value matrix shape

---

### Hypothesis 4: Pythia-Specific Architecture (MEDIUM PRIORITY)

**Pythia Differences:**
- GPT-NeoX architecture (vs Mistral's architecture)
- Different normalization (RMSNorm vs LayerNorm)
- Different attention pattern
- Different initialization

**Test:**
1. **Compare architectures:**
   - Run same prompts on Mistral-7B and Pythia-2.8B
   - Compare R_V values directly
   - Check if Pythia fundamentally doesn't contract

2. **Check Pythia-specific factors:**
   - Does Pythia have different attention patterns?
   - Does Pythia have different value space geometry?
   - Is contraction architecture-dependent?

---

### Hypothesis 5: Training Stage Effect (LOW PRIORITY)

**Evidence:**
- Original: "Final trained" models (fully converged)
- Current: Early checkpoints (0, 5k, 10k, etc.)

**Test:**
1. **Check final checkpoint (143k):**
   - Is this actually "final" or is there a later checkpoint?
   - Compare checkpoint 143k to original "final" measurement

2. **Training dynamics:**
   - Does contraction emerge only after certain training?
   - Is expansion a feature of early training?

---

## Diagnostic Test Plan (Priority Order)

### TEST 1: Replicate Original on Mistral-7B (CRITICAL)
```python
# Use EXACT original prompts and method
model = "mistralai/Mistral-7B-Instruct-v0.2"
prompts = load_L5_refined_prompts()  # From n300_mistral_test_prompt_bank.py
early_layer = 5
late_layer = 27  # Verify this matches original
window_size = 16

# Measure R_V
results = measure_R_V(model, prompts, early_layer, late_layer, window_size)

# Expected: R_V ≈ 0.85 (contraction)
# If NOT: Measurement error in original
```

### TEST 2: Side-by-Side Architecture Comparison
```python
# Same prompts, different models
prompts = load_L5_refined_prompts()

mistral_RV = measure_R_V("mistralai/Mistral-7B-Instruct-v0.2", prompts)
pythia_RV = measure_R_V("EleutherAI/pythia-2.8b", prompts)

# Compare directly
# If Mistral contracts but Pythia expands → Architecture effect
```

### TEST 3: Verify Measurement Formula
```python
# Check computation step-by-step
V_early = extract_values(model, prompt, layer=5)
V_late = extract_values(model, prompt, layer=27)

PR_early = participation_ratio(V_early)
PR_late = participation_ratio(V_late)

R_V = PR_late / PR_early  # Verify this is correct direction

# Log all intermediate values
print(f"PR_early: {PR_early}")
print(f"PR_late: {PR_late}")
print(f"R_V: {R_V}")
```

### TEST 4: Layer Depth Normalization
```python
# Compare at same % depth, not same layer number
mistral_layers = 32
pythia_layers = 32

# 84% depth = layer 27 in 32-layer model
mistral_late = int(0.84 * mistral_layers)  # = 27
pythia_late = int(0.84 * pythia_layers)    # = 27

# Verify we're measuring at equivalent depths
```

---

## Most Likely Explanations (Ranked)

### 1. **Architecture-Specific Effect** (40% probability)
- Pythia (GPT-NeoX) may fundamentally not contract
- Mistral/Qwen/Llama architectures do contract
- This would be a major finding: contraction is architecture-dependent

### 2. **Measurement Error in Current Sweep** (30% probability)
- Wrong layer indices
- Wrong window selection
- Wrong value extraction method
- Sign error in formula

### 3. **Prompt Difference** (20% probability)
- Current prompts not as recursive/novel as original
- Original L5 prompts had specific structure that induced contraction
- Current prompts don't trigger contraction mechanism

### 4. **Measurement Error in Original** (10% probability)
- Original measurement had systematic error
- Contraction was never real
- This would invalidate Phase 1 findings

---

## Immediate Action Items

1. **Find original measurement code** - Locate exact script used in Phase 1
2. **Replicate on Mistral-7B** - Use original prompts, verify contraction
3. **Compare side-by-side** - Mistral vs Pythia on identical prompts
4. **Add diagnostic logging** - Log all intermediate values
5. **Verify formula** - Check R_V = PR_late / PR_early is correct

---

## Success Criteria

**Resolution achieved when:**
- ✓ Can reproduce original contraction on Mistral-7B
- ✓ Understand why Pythia shows expansion
- ✓ Know if contraction is architecture-specific or universal
- ✓ Have clear path forward for Pythia experiments

---

*This discrepancy must be resolved before proceeding with full Pythia sweep.*

