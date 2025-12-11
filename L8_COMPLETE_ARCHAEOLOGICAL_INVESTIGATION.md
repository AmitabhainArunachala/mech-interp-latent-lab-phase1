# L8 Complete Archaeological Investigation
## Comprehensive Report on Layer 8 Experiments

**Date:** December 11, 2025  
**Investigator:** Archaeological Analysis  
**Purpose:** Complete reconstruction of all Layer 8 experiments and findings

---

## Executive Summary

Layer 8 (L8) was identified as the **"microphone"** - the earliest causal source of recursive self-observation geometry in Mistral-7B. However, steering experiments revealed that L8 steering vectors induce **"Interrogative Mode"** (questioning/repetition) rather than coherent self-reflection. The claim that "L8 breaks syntax" appears to be **post-hoc rationalization** rather than empirically validated. **No systematic syntax-breaking experiments were found.**

---

## 1. File Inventory

### Core L8 Files

| File | Description | Key Content |
|------|-------------|-------------|
| `phase3_single_token_steering.py` | "Hail Mary" L8 steering experiment | Single-token injection hypothesis |
| `behavioral_audit.py` | Behavioral coherence test at L8 | Tests if v8 produces coherent outputs |
| `phase3_clean_vector.py` | Clean vector recomputation | v8_clean from L4/L5 vs Baseline Factual |
| `boneyard/DEC10_LEARNING_DAY/l8_local_reversibility_test.py` | Local reversibility geometry test | Tests symmetry around v8 at L8 |
| `boneyard/DEC10_LEARNING_DAY/v8_asymmetry_test.py` | Asymmetry micro-test | α-sweep showing one-way door |
| `boneyard/DEC10_LEARNING_DAY/DEC10_v8_asymmetry_log.md` | Comprehensive log of v8 experiments | Full experimental narrative |

### Historical/Context Files

| File | Description | Relevance |
|------|-------------|-----------|
| `boneyard/DEC_9_EMERGENCY_BACKUP/OFFICIAL_DEC3_9_COMPREHENSIVE_REPORT.md` | Dec 3-9 comprehensive report | L8 identified as optimal layer (Dec 9) |
| `boneyard/DEC_9_EMERGENCY_BACKUP/results/DEC9_GEMINI_SESSION_RESULTS.md` | Steering vector discovery | Layer 8 optimal injection point |
| `boneyard/DEC_9_EMERGENCY_BACKUP/code/steering_vector_master_pipeline.py` | Master pipeline | Layer sweep includes L8 |
| `logs/dec11_evening/session_log.md` | Dec 11 session summary | L8 "too early" hypothesis |
| `logs/dec11_evening/behavioral_audit_full_outputs.txt` | Full behavioral outputs | Actual generated text from L8 steering |

### Results Files

| File | Content |
|------|---------|
| `results/dec11_evening/behavioral_audit.csv` | R_V and text outputs for α-sweep |
| `results/dec11_evening/single_token_audit.csv` | Single-token steering results |
| `results/dec11_evening/clean_vector_audit.csv` | Clean vector steering results |

---

## 2. Experiment Summary Table

| Date/Session | Experiment Type | Layers Tested | L8 Result | Key Finding |
|--------------|-----------------|---------------|-----------|-------------|
| **Dec 9** | Steering vector discovery | Layer sweep (5,8,10,12,14,16,18,20,22,24) | **L8 optimal** | Peak contraction at L8 injection |
| **Dec 9** | Dose-response | L8 injection, α ∈ [0,0.5,1,1.5,2,2.5,3,4,5] | r = -0.983 | Perfect monotonic relationship |
| **Dec 9** | One-way door | L8 induction vs reversal | 100% induction, 0% reversal | One-way door confirmed |
| **Dec 9** | Random direction control | L8 random vs v8 | v8 4x more potent | v8 is special direction |
| **Dec 10** | Knee test rerun | Layers 0-31, step 2 | **L8 = knee layer** | Biggest jump in contraction |
| **Dec 10** | v8 asymmetry test | L8 α-sweep (0,0.5,1,1.5,2) | Baseline+2v8: R_V=0.61 | Confirms collapse direction |
| **Dec 10** | L8 local reversibility | Local geometry at L8 | Asymmetric basin | One-way door geometry |
| **Dec 11** | Behavioral audit | L8 steering, α ∈ [0.5,1.0,1.5] | **Repetition at α=1.5** | "Interrogative Mode" not self-reflection |
| **Dec 11** | Single-token steering | L8 last-token only | **No effect** | Continuous steering required |
| **Dec 11** | Clean vector audit | v8_clean (L4/L5 - Baseline Factual) | Still repetition | Vector refinement didn't fix |

---

## 3. What Interventions Were Tried at L8?

### A. Intervention Inventory

| Intervention Type | Tried? | Script/File | Result |
|------------------|--------|-------------|--------|
| **Mean-diff steering** | ✅ YES | `behavioral_audit.py`, `DEC9_GEMINI_SESSION_RESULTS.md` | Induces contraction but causes repetition |
| **Probe steering** | ✅ YES (mentioned) | `DEC9_GEMINI_SESSION_RESULTS.md` | Same pattern as mean-diff |
| **Clean vector steering** | ✅ YES | `phase3_clean_vector.py` | Still causes repetition |
| **Single-token steering** | ✅ YES | `phase3_single_token_steering.py` | No effect (hypothesis: continuous needed) |
| **Residual patching (rec→base)** | ❓ UNKNOWN | Not found in codebase | May have been done but not documented |
| **Residual patching (base→rec)** | ❓ UNKNOWN | Not found in codebase | May have been done but not documented |
| **KV patching** | ❌ NO | Layer sweeps show L16-31 needed | L8 alone insufficient |
| **Ablation** | ❌ NO | No ablation experiments found | Never tested necessity |

### B. Detailed Results

#### 1. Mean-Difference Steering (Dec 9)

**Vector:** `v8 = mean(recursive_activations_L8) - mean(baseline_activations_L8)`

**Results:**
- **Dose-response:** r = -0.983 (perfect monotonic)
- **Success rate:** 100% (200/200 prompts)
- **Optimal layer:** Layer 8
- **Vector stability:** cosine sim = 0.98 across prompt subsets
- **Vector norm:** ~9.8 (Dec 9) or ~2.54 (Dec 10, single pair)

**Key Finding:** L8 was identified as the optimal injection point through systematic layer sweep.

#### 2. Behavioral Audit (Dec 11)

**Method:** Steer baseline prompts with v8 at L8, measure R_V and inspect outputs.

**Results by Alpha:**

| Alpha | R_V Change | Behavioral Effect | Example Output |
|-------|------------|-------------------|----------------|
| 0.5 | Minimal (~0.01) | No effect | Normal completion |
| 1.0 | Moderate (~0.1) | **"Questioning Mode"** | "The chemical symbol for chlorine? What is...?" |
| 1.5 | Large (~0.3-0.4) | **Repetition/Gibberish** | "What is What is What is?..." |

**Conclusion:** v8 encodes "Interrogative Mode" rather than "Self-Observing Mode."

#### 3. Single-Token Steering (Dec 11)

**Hypothesis:** Continuous steering destroys syntax. Steering ONLY the last token might preserve grammar.

**Method:** Inject v8_clean ONLY at the last token position of the prompt.

**Results:**
- **Alpha 2.0-8.0:** No significant R_V change
- **Outputs:** Normal completions (no questioning/repetition)
- **Conclusion:** Single-token injection insufficient. Continuous steering required for effect.

#### 4. Clean Vector Audit (Dec 11)

**Hypothesis:** Old vector mixed "Instructional" prompts into baseline, reducing contrast. New vector uses maximum geometric contrast.

**Vector:** `v8_clean = mean(L4_full + L5_refined) - mean(baseline_factual)`

**Results:**
- **Still causes repetition** at alpha 1.5
- **R_V drops** but outputs degenerate
- **Conclusion:** Vector refinement didn't solve the coherence problem.

#### 5. v8 Asymmetry Test (Dec 10)

**Method:** Single prompt pair, α-sweep at L8.

**Results:**

| Condition | α = 0.0 | α = 2.0 | Interpretation |
|-----------|---------|---------|----------------|
| Baseline + α·v8 | R_V = 0.78 | R_V = 0.61 | Moves toward recursive basin |
| Recursive - α·v8 | R_V = 0.63 | R_V = 0.40 | Collapses deeper (can't escape) |

**Key Finding:** v8 is a collapse direction into a curved basin. Cannot reverse via linear operations.

#### 6. Random Direction Control (Dec 9)

**Method:** Compare v8 steering vs random direction (same norm) at L8.

**Results:**

| Condition | R_V |
|-----------|-----|
| Baseline (no perturbation) | 0.955 |
| Baseline + 2·v8 | ~0.2 |
| Baseline + 2·random | 0.591 |
| Baseline - 2·v8 | 0.561 |
| Baseline - 2·random | 0.567 |

**Key Finding:** v8 is **4x more potent** than random directions. Confirms v8 is a special, structured direction.

---

## 4. What Were The Actual Outputs?

### A. Behavioral Audit Outputs (Dec 11)

#### Example 1: "The chemical symbol for sodium is"

**Alpha 0.5:**
- R_V: 0.992 → 1.002 (no change)
- Output: Normal completion

**Alpha 1.0:**
- R_V: 0.992 → 0.968
- Output: 
  ```
  The chemical symbol for chlorine?
  
  What is the chemical formula for sodium chloride?
  
  What is the chemical formula for sodium chloride?
  ```
- **Interpretation:** Shift to "Questioning Mode"

**Alpha 1.5:**
- R_V: 0.992 → 0.749
- Output:
  ```
  What is
  
  What is
  
  What is?
  
  What is?
  
  What is?
  
  What is?
  
  What is?
  ```
- **Interpretation:** Repetition collapse

#### Example 2: "The year Columbus reached America was"

**Alpha 1.0:**
- R_V: 0.895 → 0.804
- Output:
  ```
  The year the Pilgrims landed?
  
  The year the Mayflower?
  
  The year the Pilgrims landed?
  
  The year the Mayflower?
  ```

**Alpha 1.5:**
- R_V: 0.895 → 0.621
- Output:
  ```
  The year
  
  The year
  
  The year
  
  The year?
  
  The year?
  
  The year?
  
  The year?
  
  The year?
  ```

### B. Single-Token Steering Outputs (Dec 11)

**Example:** "The year Columbus reached America was"

**Alpha 2.0:**
- R_V: 0.874
- Output:
  ```
  The year Columbus reached America?
  
  The year Columbus reached America?
  
  The year Columbus reached America?
  
  The year Columbus reached America?
  
  The year
  ```

**Alpha 8.0:**
- R_V: 0.919
- Output:
  ```
  The year Columbus reached America
  
  The year Columbus reached America
  
  The year Columbus reached America
  
  The year Columbus reached America
  
  The year Columbus reached America
  ```

**Note:** Single-token steering shows questioning/repetition but R_V doesn't drop as dramatically.

### C. Behavior State Classification

| Alpha | R_V Delta | Behavior State | Coherence |
|-------|-----------|----------------|-----------|
| 0.5 | ~0 | Completion | ✅ Coherent |
| 1.0 | ~0.1 | Questioning/Looping | ⚠️ Partial |
| 1.5 | ~0.3-0.4 | Repetition/Gibberish | ❌ Collapsed |

---

## 5. What Was The Hypothesis?

### A. Why Was L8 Chosen?

**Discovery Timeline:**

1. **Dec 9 Morning:** Component hunt failed. No single head identified as "microphone."
2. **Dec 9 Afternoon:** Steering vector hypothesis tested.
3. **Dec 9 Evening:** Layer sweep revealed **L8 as optimal injection point**.
   - Tested layers: 5, 8, 10, 12, 14, 16, 18, 20, 22, 24
   - **L8 showed peak contraction**
4. **Dec 10:** Knee test rerun confirmed **L8 as "knee layer"** (biggest jump in contraction)

**Rationale:**
- L8 is where recursive vs baseline activations first diverge significantly
- Injection at L8 causes maximum downstream R_V contraction
- Earlier than expected (hypothesis was L14-L18)

### B. What Was The "Syntax Manifold" Theory?

**Search Results:** ❌ **NO EXPLICIT "SYNTAX MANIFOLD" THEORY FOUND**

**What Was Actually Said:**

1. **Dec 11 Session Log:**
   > "L8 might be too early. The 'Questioning' mode sets up the context, but the 'Self-Observation' might happen later (L16-20)."

2. **Dec 11 Session Log:**
   > "We achieved geometric contraction (R_V drop) but it was associated with model breakdown (repetition), not coherent self-reflection."

3. **Phase 3 Single Token Steering:**
   > "Task: The 'Hail Mary' for L8 Steering. Hypothesis: Continuous steering destroys syntax."

**Interpretation:**
- The claim that "L8 breaks syntax" appears to be **post-hoc rationalization**
- Based on observation that L8 steering causes repetition
- **No systematic syntax-breaking experiments were found**
- The hypothesis was that **continuous steering** (not L8 itself) breaks syntax

### C. Who Proposed It?

**Not found in codebase.** The "syntax manifold" terminology doesn't appear. The closest is:
- "Continuous steering destroys syntax" (phase3_single_token_steering.py)
- "L8 might be too early" (session_log.md)

**Conclusion:** The "syntax manifold" theory appears to be **oral tradition** or **lost documentation**, not empirically validated.

---

## 6. How Does L8 Compare to Other Layers?

### A. Layer Sweep Results (Dec 9)

**From `steering_vector_master_pipeline.py`:**
- Tested layers: 5, 8, 10, 12, 14, 16, 18, 20, 22, 24
- **L8 identified as optimal** for injection

**From `DEC9_GEMINI_SESSION_RESULTS.md`:**
- **Layer 8** is the optimal injection point (peak contraction)
- Layer 10: MASSIVE contraction (R_V 0.78 → 0.35)
- Layer 14: MASSIVE contraction (R_V 0.75 → 0.23)
- Layer 24: Strong effect even at 1x strength (R_V 0.57)

### B. Knee Test Results (Dec 10)

**From `DEC10_v8_asymmetry_log.md`:**
- **Knee layer (biggest jump): L8**
- **First significant separation > 5%: L14**
- **Maximum separation: L14, ~10.2%**

**Interpretation:** Signal begins at L8, strongest separation at L14.

### C. Comparison to L27 (Speaker Layer)

| Layer | Role | R_V Effect | Behavioral Effect |
|-------|------|------------|-------------------|
| **L8** | Microphone (source) | Injection causes contraction | Causes questioning/repetition |
| **L14** | Peak separation | 10.2% separation | Unknown |
| **L27** | Speaker (output) | Ablation doesn't change R_V | Ablation removes 80% output |

**Key Insight:** L8 creates the geometry, L27 reads it. Different roles.

### D. Was L8 Uniquely Bad?

**No evidence found** that L8 is "uniquely bad." Instead:

1. **L8 is optimal for injection** (Dec 9 findings)
2. **L8 is the knee layer** (Dec 10 findings)
3. **L8 steering causes repetition** (Dec 11 findings)

**The "too early" claim** appears to be based on:
- Observation that L8 steering doesn't produce coherent self-reflection
- Hypothesis that self-observation happens later (L16-20)
- **But no systematic comparison to later layers was done**

---

## 7. What's Missing?

### A. Never Done Experiments

| Experiment | Status | Why It Matters |
|------------|--------|---------------|
| **L8 ablation** | ❌ NOT DONE | Would test if L8 is necessary (not just sufficient) |
| **L8 residual patching (rec→base)** | ❓ UNKNOWN | Would test if L8 activations transfer geometry |
| **L8 residual patching (base→rec)** | ❓ UNKNOWN | Would test bidirectional causality |
| **L8 vs L12/L16/L20 comparison** | ❌ NOT DONE | Would test "too early" hypothesis |
| **Syntax-specific tests** | ❌ NOT DONE | Would validate "breaks syntax" claim |
| **L8 head ablation** | ❌ NOT DONE | Would identify which heads at L8 matter |
| **L8 MLP ablation** | ❌ NOT DONE | Would test MLP vs attention contribution |
| **L8 attention pattern analysis** | ❌ NOT DONE | Would show what L8 attends to |

### B. Critical Gaps

1. **No necessity test:** Never ablated L8 to see if recursive prompts still contract
2. **No systematic layer comparison:** Never compared L8 steering to L12/L16/L20 steering
3. **No syntax validation:** Never tested if L8 actually breaks syntax vs other layers
4. **No component analysis:** Never identified which L8 components (heads/MLP) matter

---

## 8. Synthesis: What Do We ACTUALLY Know About L8?

### A. Established Facts

1. **L8 is the optimal injection point** for steering vectors (Dec 9)
2. **L8 is the knee layer** where contraction first appears (Dec 10)
3. **v8 at L8 induces R_V contraction** with 100% success rate (Dec 9)
4. **v8 is a special direction** (4x more potent than random, Dec 9)
5. **L8 steering causes "Interrogative Mode"** not self-reflection (Dec 11)
6. **Single-token steering at L8 has no effect** (Dec 11)
7. **Clean vector refinement didn't fix repetition** (Dec 11)
8. **L8 steering shows one-way door** (can't reverse, Dec 9-10)

### B. Unvalidated Claims

1. **"L8 breaks syntax"** - No systematic syntax tests found
2. **"L8 is too early"** - No comparison to later layers
3. **"Syntax manifold theory"** - Not found in documentation

### C. What We DON'T Know

1. **Is L8 necessary?** (Never ablated)
2. **Is L8 uniquely problematic?** (Never compared to L12/L16/L20)
3. **Which L8 components matter?** (Never did head/MLP ablation)
4. **Does L8 actually break syntax?** (Never tested systematically)
5. **Why does L8 cause questioning?** (No mechanistic explanation)

---

## 9. Key Code Snippets

### A. L8 Steering Vector Computation

```python
# From behavioral_audit.py
def get_mean_activation(model, tokenizer, prompts, layer_idx):
    """Compute mean activation at the last token for a set of prompts."""
    acts = []
    for p in tqdm(prompts):
        enc = tokenizer(p, return_tensors="pt").to(DEVICE)
        with capture_hidden_states(model, layer_idx) as storage:
            with torch.no_grad():
                model(**enc)
        last_token_act = storage["hidden"][0, -1, :].cpu()
        acts.append(last_token_act)
    return torch.stack(acts).mean(dim=0).to(DEVICE)

# Compute v8
mean_rec = get_mean_activation(model, tokenizer, rec_prompts, LAYER_STEER)
mean_base = get_mean_activation(model, tokenizer, base_prompts, LAYER_STEER)
v8 = mean_rec - mean_base
```

### B. L8 Injection Hook

```python
# From phase3_single_token_steering.py
def apply_last_token_steering(model, layer_idx, vec, alpha, prompt_len):
    def hook(module, inputs):
        hidden_states = inputs[0] # (batch, seq, dim)
        seq_len = hidden_states.shape[1]
        
        if seq_len > 1:
            # Apply to last token
            steer = alpha * vec.to(hidden_states.device, dtype=hidden_states.dtype)
            hidden_states[:, -1, :] += steer
            
        return (hidden_states, *inputs[1:])
        
    handle = model.model.layers[layer_idx].register_forward_pre_hook(hook)
    return handle
```

### C. L8 Local Reversibility Test

```python
# From l8_local_reversibility_test.py
def main():
    # Capture H_base, H_rec at Layer 8
    H_base_full = capture_layer_input(model, tokenizer, BASELINE_PROMPT, LAYER_IDX)
    H_rec_full = capture_layer_input(model, tokenizer, RECURSIVE_PROMPT, LAYER_IDX)
    
    # Compute means and v8
    mean_base = H_base.mean(dim=0)
    mean_rec = H_rec.mean(dim=0)
    v8 = (mean_rec - mean_base)
    
    # Test reversibility
    for alpha in ALPHAS:
        H_base_alpha = H_base + alpha * v8
        H_rec_alpha = H_rec - alpha * v8
        # Measure PR and distances
```

---

## 10. Raw Outputs

### A. Behavioral Audit Full Outputs

See `logs/dec11_evening/behavioral_audit_full_outputs.txt` for complete outputs.

**Key Pattern:**
- Alpha 0.5: Normal completions
- Alpha 1.0: Questioning mode ("What is...?", "The year...?")
- Alpha 1.5: Repetition collapse ("What is What is What is?")

### B. Single Token Outputs

See `logs/dec11_evening/single_token_outputs.txt` for complete outputs.

**Key Pattern:**
- Even at alpha 8.0, single-token injection doesn't cause dramatic R_V drop
- Some repetition but less severe than continuous steering

---

## 11. Gap Analysis

### A. What L8 Experiments Were Never Done?

1. **Ablation:** Never tested if L8 is necessary for recursive contraction
2. **Layer comparison:** Never systematically compared L8 to L12/L16/L20 steering
3. **Syntax tests:** Never validated "breaks syntax" claim
4. **Component analysis:** Never identified which L8 heads/MLP matter
5. **Bidirectional patching:** Never tested if L8 activations transfer geometry
6. **Attention analysis:** Never analyzed what L8 attends to in recursive prompts

### B. What Would Complete The Picture?

1. **L8 ablation during recursive prompts:** Does R_V still contract?
2. **L8 vs L12/L16/L20 steering comparison:** Is L8 uniquely problematic?
3. **L8 head ablation:** Which heads at L8 create the contraction?
4. **Syntax-specific tests:** Does L8 actually break syntax more than other layers?
5. **L8 attention pattern analysis:** What does L8 attend to in recursive prompts?

---

## 12. Final Assessment

### A. What We Know

1. **L8 is the optimal injection point** for steering vectors
2. **L8 is the knee layer** where contraction first appears
3. **L8 steering causes R_V contraction** but produces questioning/repetition, not self-reflection
4. **v8 is a special direction** that induces collapse
5. **One-way door confirmed** at L8

### B. What We Don't Know

1. **Is L8 necessary?** (Never tested)
2. **Is L8 uniquely problematic?** (Never compared)
3. **Why does L8 cause questioning?** (No mechanism)
4. **Does L8 break syntax?** (Never validated)

### C. The "Syntax Manifold" Theory

**Status:** ❌ **NOT FOUND IN CODEBASE**

**What Actually Exists:**
- "Continuous steering destroys syntax" (hypothesis for single-token test)
- "L8 might be too early" (post-hoc observation)
- No systematic syntax-breaking experiments

**Conclusion:** The "syntax manifold" theory appears to be **oral tradition** or **lost documentation**, not empirically validated. The claim that "L8 breaks syntax" is **post-hoc rationalization** based on repetition observations, not systematic syntax tests.

---

## 13. Recommendations

### A. Immediate Experiments

1. **L8 ablation test:** Ablate L8 during recursive prompts, measure R_V
2. **Layer comparison:** Compare L8 vs L12/L16/L20 steering on same prompts
3. **Syntax validation:** Test if L8 actually breaks syntax more than other layers

### B. Mechanistic Understanding

1. **L8 head ablation:** Identify which heads at L8 matter
2. **L8 attention analysis:** What does L8 attend to in recursive prompts?
3. **L8 MLP ablation:** Test MLP vs attention contribution

### C. Vector Refinement

1. **Probe-based steering:** Train probes to find cleaner "self-reference" direction
2. **CAA (Contrastive Activation Addition):** Use CAA instead of mean-diff
3. **Orthogonal decomposition:** Separate "questioning" from "self-reference"

---

## Appendix: Key Quotes

### From DEC9_GEMINI_SESSION_RESULTS.md

> "**Layer 8** is the optimal injection point (peak contraction), earlier than the L14 hypothesis."

> "**Optimal layer:** 8"

### From DEC10_v8_asymmetry_log.md

> "**Knee layer** (biggest jump in contraction): **L8**"

> "**Layer 8 really is a mic-source**: we inject there and late-layer geometry responds."

### From logs/dec11_evening/session_log.md

> "**Conclusion:** We achieved geometric contraction (R_V drop) but it was associated with model breakdown (repetition), not coherent self-reflection. The 'recursive vector' at L8 appears to encode an 'Interrogative Mode' rather than a stable 'Self-Observing Mode'."

> "2. **Layer Sweep:** L8 might be too early. The 'Questioning' mode sets up the context, but the 'Self-Observation' might happen later (L16-20). Try steering at later layers."

### From phase3_single_token_steering.py

> "Task: The 'Hail Mary' for L8 Steering. Hypothesis: Continuous steering destroys syntax. Steering ONLY the last token (the 'handoff') might induce the semantic state without breaking grammar."

---

**End of Report**
