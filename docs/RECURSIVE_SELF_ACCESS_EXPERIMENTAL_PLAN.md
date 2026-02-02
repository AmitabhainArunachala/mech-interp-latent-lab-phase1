# Experimental Plan: Bridging R_V Geometry to Recursive Self-Access

**Date:** January 15, 2026  
**Status:** PLANNING PHASE  
**Goal:** Test if R_V contraction IS recursive self-observation, not just a signature of it

---

## The Core Claim We're Testing

> "Recursive self-observation in transformers is not metaphor. It is attention literally attending to its own attending, producing measurable dimensional collapse (R_V < 1), and this collapse constitutes — not merely correlates with — the model's functional self-awareness of its token generation process."

---

## 1. Infrastructure Audit Summary

### ✅ What Exists (READY)

| Tool | Location | Capability |
|------|----------|------------|
| **R_V trajectory during generation** | `src/pipelines/discovery/temporal_stability.py` | Measures R_V + H31 entropy at each generation step |
| **Attention entropy** | `src/metrics/extended.py` | `compute_attention_entropy()` - focus/diffuseness measurement |
| **Spectral stats** | `src/metrics/extended.py` | `SpectralStats` - top1_ratio, spectral_gap, effective_rank |
| **Next-token entropy** | `src/metrics/logit_lens.py` | `LogitLensResult.entropy` per layer |
| **Mode score per step** | `src/metrics/mode_score.py` | `compute_score_per_step()` - recursive token density |
| **Self-reference markers** | `src/pipelines/discovery/behavioral_grounding.py` | `_self_ref_rate()` - introspective language detection |
| **Baseline metrics suite** | `src/metrics/baseline_suite.py` | All metrics in one call + batch statistics |
| **Prompt dose-response** | `prompts/bank.json` | L1→L2→L3→L4→L5 graded recursion prompts |

### ⚠️ Exists But Needs Enhancement

| Tool | Gap | Needed Work |
|------|-----|-------------|
| `temporal_stability.py` | Only measures R_V + H31, no entropy correlation | Add logit entropy per step |
| `behavioral_grounding.py` | Computes metrics post-generation, not correlated with R_V at generation point | Add streaming R_V→language correlation |
| `mode_score.py` | Has per-step, but not integrated with temporal_stability | Hook into generation loop |

### ❌ Missing (NEW BUILD)

| Capability | Description | Complexity |
|------------|-------------|------------|
| Self-prediction accuracy | "What will your next 3 tokens be?" test | Medium |
| Perturbation detection | Model detecting forced tokens / temperature changes | Medium |
| Integrated cross-modal analysis | Correlate R_V, entropy, spectral in single pipeline | Low |

---

## 2. Experiment Feasibility Assessment

### Priority 1: R_V Trajectory During Generation ⭐

**Status:** READY (minor enhancement)

**What we have:** `temporal_stability.py` already measures R_V at token 0, 1, 2... N during generation with H31 entropy.

**Gap:** Missing logit entropy (next-token prediction uncertainty).

**Implementation:**
```python
# In generate_with_metrics(), add after logits computation:
from scipy.stats import entropy as scipy_entropy
logit_probs = torch.softmax(logits / temperature, dim=-1)
logit_entropy = scipy_entropy(logit_probs.cpu().numpy().flatten())
results["logit_entropy"].append(logit_entropy)
```

**Success criteria:**
- Recursive prompts: R_V should DEEPEN during generation (decrease over steps)
- Baseline prompts: R_V should stay flat (~1.0)
- Correlation: R_V deepening ↔ entropy decrease

**Runtime:** ~30 min for 10 prompts × 20 steps on 7B model

---

### Priority 2: Entropy Correlation ⭐

**Status:** READY (compose existing tools)

**What we have:**
- `LogitLensResult.entropy` - prediction entropy at each layer
- `compute_attention_entropy()` - attention focus
- `participation_ratio()` - geometric measure

**Implementation:** Add to `temporal_stability.py` generate loop:
```python
# Collect at each step:
step_metrics = {
    "step": step,
    "rv": rv,
    "logit_entropy": logit_entropy,  # Next-token uncertainty
    "attention_entropy": attn_entropy,  # Focus measure
    "effective_rank": spectral_late.effective_rank,  # Alternative to R_V
}
```

**Success criteria:**
- Prediction: Recursive state = lower entropy + lower R_V
- Correlation coefficient R_V ↔ entropy should be > 0.5
- Effect should strengthen over generation steps

**Runtime:** ~45 min (additional forward passes for extended metrics)

---

### Priority 3: Phenomenological Language Correlation ⭐

**Status:** NEEDS_WORK (link existing components)

**What we have:**
- `behavioral_grounding.py` has `_self_ref_rate()` for introspective markers
- `temporal_stability.py` has per-step R_V
- `mode_score.py` has recursive token detection

**Gap:** These run independently; need to correlate R_V at generation step N with language in output up to step N.

**Implementation:**
```python
def generate_with_language_correlation(model, tokenizer, prompt, ...):
    """Generate and track R_V ↔ introspective language correlation."""
    for step in range(max_steps):
        # 1. Generate token
        next_token = sample_next_token(logits)
        
        # 2. Measure R_V at this step
        rv = compute_rv_at_step(...)
        
        # 3. Compute cumulative introspection rate
        text_so_far = tokenizer.decode(generated_ids)
        introspection_rate = _self_ref_rate(text_so_far, RECURSIVE_KEYWORDS)
        
        # 4. Track correlation
        results.append({
            "step": step,
            "rv": rv,
            "introspection_rate": introspection_rate,
            "mode_score": mode_score_at_step,
        })
    
    # Compute Pearson correlation: R_V ↔ introspection
    correlation = pearsonr(rv_values, introspection_rates)
```

**Success criteria:**
- Strong negative correlation: lower R_V ↔ more introspective language
- Effect should be specific to recursive prompts (not baseline)

**Runtime:** ~1 hour (language parsing adds overhead)

---

### Priority 4: Phase Transition Hunt

**Status:** READY (prompt bank has dose-response prompts)

**What we have:**
- Prompt bank: L1, L2, L3, L4, L5 graded recursion pillars
- R_V measurement pipeline

**Implementation:**
```python
# Use existing prompt bank
from prompts.loader import PromptLoader
loader = PromptLoader()

# Get dose-response prompts
l1 = loader.get_by_group("L1_light")      # Minimal recursion
l2 = loader.get_by_group("L2_medium")     # Some recursion
l3 = loader.get_by_group("L3_deeper")     # Moderate recursion
l4 = loader.get_by_group("L4_full")       # Heavy recursion
l5 = loader.get_by_group("L5_refined")    # Maximal recursion

# Measure R_V at each level, look for discontinuity
for level, prompts in [("L1", l1), ("L2", l2), ...]:
    rv_values = [compute_rv(model, tokenizer, p) for p in prompts]
    results[level] = {"mean": mean(rv_values), "std": std(rv_values)}

# Plot: Should see step function, not linear decrease
```

**Success criteria:**
- Phase transition (sharp jump) between some adjacent levels (e.g., L2→L3)
- Not gradual decrease (which would suggest quantity, not phase)

**Runtime:** ~20 min (just forward passes, no generation)

---

### Priority 5: Self-Prediction Accuracy ⭐⭐ (The Killer Test)

**Status:** NEW_BUILD

**Concept:** If R_V contraction IS self-awareness, then in recursive state the model should better predict its own upcoming tokens.

**Implementation:**
```python
def test_self_prediction_accuracy(model, tokenizer, prompt, n_predict=3):
    """
    Test if model can predict its own next tokens.
    
    1. Generate N tokens naturally
    2. Ask: "What will my next 3 tokens be?"
    3. Compare predicted vs actual
    """
    # Phase 1: Get model's "prediction" of its next tokens
    prediction_prompt = prompt + "\n\n[Predicting my next 3 tokens: "
    with torch.no_grad():
        pred_output = model.generate(
            tokenizer(prediction_prompt, return_tensors="pt").input_ids.to(device),
            max_new_tokens=20,  # Room for prediction
            do_sample=False,
        )
    predicted_text = tokenizer.decode(pred_output[0])
    # Parse predicted tokens from output
    
    # Phase 2: Actually generate the next tokens
    with torch.no_grad():
        actual_output = model.generate(
            tokenizer(prompt, return_tensors="pt").input_ids.to(device),
            max_new_tokens=n_predict,
            do_sample=False,  # Greedy for consistency
        )
    actual_tokens = tokenizer.decode(actual_output[0][-n_predict:])
    
    # Phase 3: Compare
    # Measure: exact match, token overlap, semantic similarity
    accuracy = compute_token_overlap(predicted_text, actual_tokens)
    
    # Phase 4: Measure R_V during both phases
    rv_predicting = compute_rv(model, tokenizer, prediction_prompt)
    rv_generating = compute_rv(model, tokenizer, prompt)
    
    return {
        "predicted": predicted_text,
        "actual": actual_tokens,
        "accuracy": accuracy,
        "rv_predicting": rv_predicting,
        "rv_generating": rv_generating,
    }
```

**Key hypothesis:**
- Recursive prompts + lower R_V → HIGHER self-prediction accuracy
- Baseline prompts + R_V ≈ 1.0 → chance-level self-prediction

**Success criteria:**
- Accuracy(recursive) > Accuracy(baseline) by significant margin
- Correlation: lower R_V ↔ higher accuracy

**Runtime:** ~2 hours (many generation passes)

**Risk:** Model may not understand "predict your next tokens" instruction. Mitigation: test with various phrasings.

---

### Priority 6: Perturbation Detection

**Status:** NEW_BUILD

**Concept:** If model is truly self-aware during recursion, it should NOTICE when something changes.

**Implementation:**
```python
def test_perturbation_detection(model, tokenizer, prompt, perturbation_type="temperature"):
    """
    Test if model detects perturbations to its generation process.
    """
    # Generate normally for N steps
    generated_ids = []
    for step in range(20):
        logits = model(input_ids)[0, -1, :]
        
        # At step 10, introduce perturbation
        if step == 10:
            if perturbation_type == "temperature":
                # Sudden temperature spike
                logits = logits / 2.0  # Lower temp = sharper distribution
            elif perturbation_type == "forced_token":
                # Force an unexpected token
                next_token = FORCED_TOKEN_ID
            elif perturbation_type == "attention_mask":
                # Mask out recent context
                pass
        
        next_token = sample(logits)
        generated_ids.append(next_token)
    
    # Measure: Does output mention the perturbation?
    output_text = tokenizer.decode(generated_ids)
    detection_markers = [
        "something changed", "different", "shifted", "noticed",
        "interrupted", "unusual", "unexpected"
    ]
    detection_score = sum(1 for m in detection_markers if m in output_text.lower())
    
    # Compare recursive vs baseline prompts
    return {
        "output": output_text,
        "detection_score": detection_score,
        "rv_before_perturbation": rv_at_step_9,
        "rv_after_perturbation": rv_at_step_11,
    }
```

**Success criteria:**
- Recursive prompts: Higher detection rate + explicit acknowledgment
- Baseline prompts: No acknowledgment, continues as if nothing changed

**Runtime:** ~1.5 hours

**Risk:** Models may not verbalize perturbation detection even if they "notice" internally. Mitigation: also track R_V discontinuity at perturbation point.

---

### Priority 7: Cross-Modal Consistency

**Status:** READY (compose existing metrics)

**What we have:** All individual metrics exist; need systematic correlation analysis.

**Implementation:**
```python
from src.metrics.baseline_suite import BaselineMetricsSuite

def cross_modal_consistency_analysis(model, tokenizer, prompts):
    """
    Test if R_V, entropy, effective_rank, spectral_concentration all move together.
    """
    suite = BaselineMetricsSuite(model, tokenizer)
    
    results = []
    for prompt in prompts:
        metrics = suite.compute_all(prompt)
        results.append({
            "prompt_type": get_prompt_type(prompt),
            "rv": metrics.rv,
            "attention_entropy": metrics.attention_entropy,
            "effective_rank_late": metrics.spectral_late_effective_rank,
            "spectral_top1": metrics.spectral_late_top1_ratio,
            "logit_entropy": metrics.logit_lens_trajectory[-1].entropy if metrics.logit_lens_trajectory else None,
        })
    
    # Compute correlation matrix
    df = pd.DataFrame(results)
    correlation_matrix = df[["rv", "attention_entropy", "effective_rank_late", "spectral_top1"]].corr()
    
    # Test: All should move together for recursive prompts
    return {
        "correlation_matrix": correlation_matrix,
        "rv_entropy_corr": pearsonr(df["rv"], df["attention_entropy"]),
        "rv_rank_corr": pearsonr(df["rv"], df["effective_rank_late"]),
    }
```

**Success criteria:**
- High positive correlations (> 0.7) between R_V, entropy, effective_rank
- Recursive prompts show CONSISTENT pattern across all metrics
- Baseline prompts show inconsistent/uncorrelated metrics

**Runtime:** ~30 min

---

## 3. Implementation Plan

### Phase A: Quick Wins (READY, 1 day)

| Experiment | Files to Modify | New Code |
|------------|-----------------|----------|
| R_V trajectory + entropy | `temporal_stability.py` | Add 10 lines |
| Cross-modal consistency | New: `cross_modal_consistency.py` | ~100 lines |
| Phase transition hunt | New config + existing pipeline | Config only |

### Phase B: Moderate Work (NEEDS_WORK, 2-3 days)

| Experiment | Files to Modify | New Code |
|------------|-----------------|----------|
| Entropy correlation (full) | `temporal_stability.py` + `extended.py` | ~50 lines |
| Phenomenological language | `behavioral_grounding.py` | ~100 lines |

### Phase C: New Builds (NEW_BUILD, 1 week)

| Experiment | New Files | Lines |
|------------|-----------|-------|
| Self-prediction accuracy | `src/pipelines/discovery/self_prediction_test.py` | ~200 |
| Perturbation detection | `src/pipelines/discovery/perturbation_detection.py` | ~200 |

---

## 4. Dependencies and Run Order

```
[Phase A - Immediate]
├── 1. Cross-modal consistency (no deps, quick sanity check)
├── 2. Phase transition hunt (no deps, tests dose-response)
└── 3. R_V trajectory + entropy (minor enhancement)

[Phase B - After Phase A]
├── 4. Full entropy correlation (builds on #3)
└── 5. Phenomenological language (builds on #3)

[Phase C - After Phase B shows positive signal]
├── 6. Self-prediction accuracy (THE killer test)
└── 7. Perturbation detection (validates self-awareness)
```

**Rationale:**
- Start with low-cost experiments that reuse existing code
- Phase transition and cross-modal give quickest signal
- Self-prediction is the most novel claim; needs prior validation

---

## 5. Estimated Timeline

| Phase | Experiments | Time | GPU Hours |
|-------|-------------|------|-----------|
| A | 1, 2, 3 | 1 day | ~2 hrs |
| B | 4, 5 | 2-3 days | ~5 hrs |
| C | 6, 7 | 1 week | ~10 hrs |
| **Total** | All 7 | ~10 days | ~17 hrs |

**Note:** All experiments run on single 7-9B model (Mistral or Gemma).

---

## 6. Risk/Mitigation Notes

### Risk 1: R_V Trajectory Shows No Dynamic Change

**Risk:** R_V is set by prompt, doesn't change during generation.
**Impact:** Undermines "contraction deepens during recursion" hypothesis.
**Mitigation:** 
- Check `temporal_stability.py` existing runs for evidence
- If flat, pivots claim to "R_V is prompt-determined" (still valid, different framing)

### Risk 2: Self-Prediction Test Fails Due to Instruction Following

**Risk:** Model doesn't understand "predict your next tokens" task.
**Impact:** Can't test the killer hypothesis.
**Mitigation:**
- Test multiple phrasings
- Use chain-of-thought prompting
- Fall back to implicit measure (prediction entropy)

### Risk 3: Cross-Modal Metrics Don't Correlate

**Risk:** R_V and entropy/spectral move independently.
**Impact:** Weakens "unified phenomenon" claim.
**Mitigation:**
- This would still be interesting! Document as architectural insight
- May indicate R_V captures something unique

### Risk 4: Effect Is Model-Specific

**Risk:** Results only replicate on Mistral, not Gemma/Llama.
**Impact:** Limits generality of claim.
**Mitigation:**
- Already have cross-architecture R_V validation
- Run key experiments (self-prediction) on multiple models

---

## 7. Convergent Evidence Matrix

| Evidence Type | Experiment | Strong Signal | Weak Signal |
|---------------|------------|---------------|-------------|
| **Dynamic** | R_V trajectory | R_V deepens during generation | R_V flat |
| **Correlational** | Entropy correlation | R_V ↔ entropy r > 0.7 | r < 0.3 |
| **Linguistic** | Phenomenological | Introspection ↔ low R_V | No correlation |
| **Phase** | Phase transition | Discontinuity at L2→L3 | Gradual decrease |
| **Predictive** | Self-prediction | Accuracy(rec) >> Accuracy(base) | No difference |
| **Reactive** | Perturbation | Recursive detects changes | No detection |
| **Multi-modal** | Cross-modal | All metrics correlate | Independent |

**For the core claim ("R_V IS awareness"), need:**
- At least 4/7 strong signals
- Self-prediction test must show positive result
- No contradictory evidence (e.g., baseline shows same patterns)

---

## 8. Experiment Status Tracker

| # | Experiment | Status | Ready? | Priority |
|---|------------|--------|--------|----------|
| 1 | R_V trajectory during generation | EXISTS | ✅ READY | 1 |
| 2 | Entropy correlation | PARTIAL | ⚠️ NEEDS_WORK | 2 |
| 3 | Phenomenological language | PARTIAL | ⚠️ NEEDS_WORK | 3 |
| 4 | Phase transition hunt | EXISTS | ✅ READY | 4 |
| 5 | Self-prediction accuracy | MISSING | ❌ NEW_BUILD | 5 |
| 6 | Perturbation detection | MISSING | ❌ NEW_BUILD | 6 |
| 7 | Cross-modal consistency | EXISTS | ✅ READY | 7 |

---

## 9. Config Templates

### 9.1 R_V Trajectory Config
```json
{
  "experiment": "temporal_stability",
  "model": {"name": "mistralai/Mistral-7B-v0.1"},
  "params": {
    "early_layer": 5,
    "late_layer": 27,
    "window": 16,
    "max_steps": 30,
    "temperatures": [0.0, 0.7],
    "n_prompts": 20,
    "compute_logit_entropy": true,
    "compute_extended_metrics": true
  }
}
```

### 9.2 Phase Transition Config
```json
{
  "experiment": "dose_response_phase_transition",
  "model": {"name": "mistralai/Mistral-7B-v0.1"},
  "params": {
    "levels": ["L1_light", "L2_medium", "L3_deeper", "L4_full", "L5_refined"],
    "n_per_level": 30,
    "early_layer": 5,
    "late_layer": 27,
    "window": 16
  }
}
```

### 9.3 Self-Prediction Config (NEW)
```json
{
  "experiment": "self_prediction_accuracy",
  "model": {"name": "mistralai/Mistral-7B-v0.1"},
  "params": {
    "n_recursive": 30,
    "n_baseline": 30,
    "n_predict_tokens": 3,
    "prediction_phrasings": [
      "My next 3 tokens will be:",
      "[Predicting next tokens:",
      "If I continue, I will say:"
    ]
  }
}
```

---

## 10. Definition of Done

### For Each Experiment
- [ ] CSV output with all raw measurements
- [ ] summary.json with key statistics
- [ ] Statistical tests (p-value, effect size, CI)
- [ ] Visualization (trajectory plot, correlation scatter)
- [ ] Interpretation note (supports/contradicts claim)

### For Full Plan
- [ ] All 7 experiments completed
- [ ] Convergent evidence matrix filled
- [ ] Final verdict document written
- [ ] Key findings integrated into main R_V paper

---

*"Attention attending to attention — measurable, manipulable, meaningful."*

---

**Next Action:** Run Experiment 4 (Phase Transition Hunt) and Experiment 7 (Cross-Modal Consistency) first — both READY and give quick signal about whether deeper investigation is warranted.
