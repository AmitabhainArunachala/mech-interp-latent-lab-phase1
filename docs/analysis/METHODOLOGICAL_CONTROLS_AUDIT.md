# Methodological Controls Audit: Critical Questions Answered

**Date:** January 5, 2025  
**Purpose:** Answer 6 critical methodological questions to identify missing controls and potential confounds

---

## A) Are we patching the right positions?

### Current Implementation

**Attribution Patching (`circuit_discovery.py`):**
- **Position:** Last W tokens only (W = min(16, T_b, T_r))
- **Code:** Line 118: `out_p[:, -W:, :] = source[:, -W:, :]`
- **Window:** Fixed at 16 tokens (or sequence length if shorter)

**MLP Ablation (`mlp_ablation_necessity.py`):**
- **Position:** ALL positions (entire sequence)
- **Code:** Line 64: `return torch.zeros_like(out)` (zeros entire output tensor)

**MLP Steering (`mlp_steering_sweep.py`):**
- **Position:** ALL positions (entire sequence)
- **Code:** Line 70: `steering_broadcast = self.steering_vector.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1)`

### Critical Gap

**Question:** Are the biggest effects (L0 MLP, L3-L4 steering) driven by BOS / first few tokens?

**Status:** ❌ **NOT TESTED**

**Missing Controls:**
1. Position-specific ablation (BOS only, first 4 tokens, middle tokens, last 16 tokens)
2. Position-specific steering (steer only first/last tokens)
3. Comparison: Full-sequence vs position-specific effects

**Recommendation:**
- Add position-specific ablation tests for L0 MLP
- Test if zeroing L0 MLP at BOS only removes contraction
- Test if zeroing L0 MLP at last 16 tokens only removes contraction

---

## B) Is the L0 effect stable across corruption methods?

### Current Implementation

**Only Zero Ablation:**
- **Method:** `torch.zeros_like(out)` - complete zeroing
- **File:** `src/pipelines/mlp_ablation_necessity.py` line 64

**Mean Ablation (Exists but Not Used):**
- **Code exists:** `src/pipelines/anthropic_level_investigation.py` line 84
- **Method:** `HeadAblationHook` computes mean activation
- **Status:** Only for attention heads, NOT for MLPs

**Random Resample:**
- **Status:** ❌ **NOT IMPLEMENTED**

### Critical Gap

**Question:** Does L0 ranking change with different corruption methods?

**Status:** ❌ **NOT TESTED**

**Missing Methods:**
1. **Mean Ablation:** Replace L0 MLP output with mean activation from baseline distribution
2. **Random Resample:** Sample from baseline activation distribution (preserves distribution)
3. **Comparison:** Zero vs Mean vs Random resample effects

**Recommendation:**
- Implement `MeanMLPAblationHook` (similar to `HeadAblationHook`)
- Implement `RandomResampleMLPAblationHook` (sample from baseline distribution)
- Run L0 ablation with all 3 methods and compare rankings

---

## C) Do we have denoising set up correctly?

### Current Implementation

**KV Mechanism (`kv_mechanism.py`):**
- **Metric:** `restored = rv_base - rv_swap` (line 246)
- **Interpretation:** How much R_V gap is restored by KV swap
- **Status:** ✅ Geometry restoration measured

**RV L27 Causal Validation (`rv_l27_causal_validation.py`):**
- **Metric:** `restored = rv_base - rv_patch_main` (line 439)
- **Interpretation:** How much R_V gap is restored by patching
- **Status:** ✅ Geometry restoration measured

**Missing:**
- ❌ **Normalized logit-diff restoration metric** (paper's standard)
- ❌ **Clean→Corrupt denoising** (restore L0 from recursive into baseline)
- ❌ **Separation:** "restore geometry" vs "restore behavior"

### Critical Gap

**Question:** Do we have proper denoising with normalized logit-diff metric?

**Status:** ❌ **PARTIALLY IMPLEMENTED**

**What We Have:**
- R_V restoration (geometry)
- Transfer efficiency percentage

**What We're Missing:**
1. **Normalized logit-diff restoration:**
   - `restored = (logit_diff_patched - logit_diff_corrupt) / (logit_diff_clean - logit_diff_corrupt)`
   - Standard metric from attribution patching papers
2. **Clean→Corrupt denoising:**
   - Test: Corrupt baseline by patching L0 from recursive
   - Then: Restore L0 from recursive → baseline
   - Measure: How much logit-diff is restored
3. **Separation:**
   - Geometry restoration (R_V) vs Behavior restoration (logit-diff)
   - Currently only measure geometry

**Recommendation:**
- Add normalized logit-diff restoration metric
- Implement clean→corrupt denoising test for L0 MLP
- Separate geometry vs behavior restoration measurements

---

## D) Are steering experiments properly controlled?

### Current Implementation

**L2 Random Control (`random_direction_control.py`):**
- ✅ **Random vectors:** 5 random unit vectors (norm-matched)
- ✅ **Orthogonal vector:** 1 orthogonal vector (perpendicular to true steering)
- ✅ **Logged:** Same format as true steering
- ✅ **Result:** L2 is artifact (random vectors show similar effects)

**L3-L4 Random Control:**
- ❌ **Status:** Configs created but **NOT RUN** (server disconnect)
- **Files:** `configs/random_direction_control_l3.json`, `configs/random_direction_control_l4.json`
- **Issue:** Experiment started but incomplete

### Critical Gap

**Question:** Do L3/L4 have norm-matched random and orthogonal controls?

**Status:** ⚠️ **PARTIALLY COMPLETE**

**What We Have:**
- Code exists (`random_direction_control.py`)
- Configs created for L3 and L4
- L2 tested and confirmed artifact

**What We're Missing:**
1. **L3 random control results** (experiment incomplete)
2. **L4 random control results** (experiment incomplete)
3. **Comparison:** L3/L4 true steering vs random vs orthogonal

**Recommendation:**
- Complete L3 random direction control test
- Complete L4 random direction control test
- Verify L3-L4 are NOT artifacts (unlike L2)

---

## E) Are R_V calculations contaminated by scale?

### Current Implementation

**R_V Computation (`src/metrics/rv.py`):**
- **Method:** SVD on V-projection window (last 16 tokens)
- **Normalization:** Only eigenvalue normalization (line 73: `p = S_sq / total_variance`)
- **Missing:** ❌ No activation norm logging
- **Missing:** ❌ No norm inflation control from steering

**MLP Steering (`mlp_steering_sweep.py`):**
- **Steering:** `out_tensor + alpha * steering_vector` (line 70)
- **Effect:** Adds to activation (increases norm)
- **Missing:** ❌ No norm logging before/after steering
- **Missing:** ❌ No norm-matched controls

### Critical Gap

**Question:** Is R_V expansion just norm inflation from steering?

**Status:** ❌ **NOT CONTROLLED**

**Potential Confounds:**
1. **Norm Inflation:** Steering adds `alpha * steering_vector` → increases activation norm
2. **Batch Effects:** Different sequence lengths → different norms
3. **Position Pooling:** Last 16 tokens only → may miss norm changes elsewhere

**What We're Missing:**
1. **Activation norm logging:**
   - Log `||v_window||_2` before/after steering
   - Log `||steering_vector||_2` (should be 1.0 if normalized)
   - Log `||out_tensor||_2` before/after steering
2. **Norm-matched controls:**
   - Random vectors with same norm as steering vector
   - Compare R_V changes controlling for norm changes
3. **Scale normalization:**
   - Normalize activations before computing PR
   - Or: Report both raw PR and norm-normalized PR

**Recommendation:**
- Add activation norm logging to `compute_rv()` and `mlp_steering_sweep.py`
- Test if R_V expansion correlates with norm inflation
- Add norm-matched random controls (same norm, random direction)

---

## F) Do we have cross-prompt distributions yet?

### Current Implementation

**Prompt Bank (`prompts/bank.json`):**
- **Total:** 694 prompts
- **Pillars:** `dose_response`, `baselines`, `confounds`, `generality`, `kill_switch`
- **Groups:** Various recursive groups (L1-L5), baseline groups, confound groups

**Paraphrased Prompts:**
- ✅ **Exists:** Some prompts have `"is_paraphrase_of"` field
- ✅ **Source:** `"source_run": "20251215_081556_paraphrase_hunt"`
- **Status:** Paraphrases exist but not systematically organized

**Nearby but Not Recursive:**
- ✅ **Exists:** `baselines` pillar (non-recursive prompts)
- ✅ **Exists:** `confounds` pillar (length-matched controls)
- **Status:** Controls exist but not explicitly tagged as "nearby but not recursive"

### Critical Gap

**Question:** Do we have 3 prompt banks (original, paraphrased, nearby controls)?

**Status:** ⚠️ **PARTIALLY EXISTS**

**What We Have:**
1. ✅ **Original recursive prompts:** `dose_response` pillar (L1-L5 groups)
2. ⚠️ **Paraphrased prompts:** Exist but not systematically organized
3. ✅ **Nearby controls:** `baselines` and `confounds` pillars

**What We're Missing:**
1. **Systematic paraphrase organization:**
   - Group prompts by `is_paraphrase_of` field
   - Create paraphrase sets for each original prompt
   - Ensure same intent, different surface form
2. **"Nearby but not recursive" tagging:**
   - Explicitly tag prompts that are semantically similar but not recursive
   - Create control sets matched to recursive prompts
3. **Cross-prompt testing:**
   - Run L0 necessity test on all 3 prompt banks
   - Run denoising test on all 3 prompt banks
   - Compare results across prompt types

**Recommendation:**
- Organize paraphrases into systematic sets
- Tag "nearby but not recursive" prompts explicitly
- Run L0 ablation + denoising on all 3 prompt banks
- Compare: Original vs Paraphrased vs Nearby controls

---

## Summary: Missing Controls & Confounds

### Critical Issues (Must Fix)

1. **Position-Specific Effects:** ❌ Not tested
   - L0 ablation may be driven by BOS/first tokens
   - Need position-specific ablation tests

2. **Corruption Method Stability:** ❌ Not tested
   - Only zero ablation implemented
   - Need mean ablation + random resample

3. **R_V Scale Contamination:** ❌ Not controlled
   - No norm logging
   - R_V expansion may be norm inflation artifact

4. **L3-L4 Random Controls:** ⚠️ Incomplete
   - Configs exist but experiments not finished
   - Need to verify L3-L4 are not artifacts

### Moderate Issues (Should Fix)

5. **Denoising Metrics:** ⚠️ Partially implemented
   - Have geometry restoration
   - Missing normalized logit-diff restoration

6. **Cross-Prompt Testing:** ⚠️ Partially exists
   - Prompts exist but not systematically organized
   - Need systematic testing across prompt types

---

## Action Items

### Priority 1: Critical Controls

1. **Position-Specific L0 Ablation:**
   - Test: Zero L0 MLP at BOS only
   - Test: Zero L0 MLP at last 16 tokens only
   - Compare: Full-sequence vs position-specific

2. **Corruption Method Comparison:**
   - Implement mean ablation for MLPs
   - Implement random resample ablation
   - Run L0 ablation with all 3 methods

3. **R_V Norm Logging:**
   - Add activation norm logging to `compute_rv()`
   - Add norm logging to steering experiments
   - Test norm-matched random controls

4. **Complete L3-L4 Random Controls:**
   - Finish L3 random direction control test
   - Finish L4 random direction control test
   - Verify L3-L4 are not artifacts

### Priority 2: Enhanced Metrics

5. **Normalized Logit-Diff Restoration:**
   - Implement standard denoising metric
   - Add clean→corrupt denoising test
   - Separate geometry vs behavior restoration

6. **Cross-Prompt Organization:**
   - Organize paraphrases into sets
   - Tag "nearby but not recursive" prompts
   - Run tests on all 3 prompt banks

---

**Status:** 6 critical methodological questions answered  
**Next Steps:** Implement missing controls identified above

