# Experiment Log - December 17, 2024

## Summary

**Total Experiments:** 4
**Total Runtime:** ~3 hours
**Key Finding:** All new methods produce false positives. Need to return to original proven method.

---

## Experiment 1: B1 - Steering + V_PROJ L27

**Config:**
- Steering vector (α=2.0) at L27
- V_PROJ patching at L27
- 20 baseline prompts
- 100 tokens

**Results:**
- Transfer Rate: 50% (10/20)
- Collapse Rate: 0%
- Mean Score: 0.3150

**Verdict:** ❌ FALSE POSITIVE
- All outputs were garbage (forum posts, legal cases, off-topic)
- Scorer misclassified repetitive patterns as recursive

**Files:**
- `results/runs/20251217_153735_minimal_recursive_intervention/B1_Steering_VPROJ_L27_results.csv`
- `B1_TOP_OUTPUTS_REVIEW.md`

---

## Experiment 2: H1 - Head-Specific V_PROJ (H18+H26) + KV@L27

**Config:**
- KV cache replacement at L27 only
- V_PROJ patching at L27, H18+H26 only (256 dims)
- 20 baseline prompts
- 200 tokens

**Results:**
- Transfer Rate: 35% (7/20)
- Collapse Rate: 20%
- Mean Score: 0.2200

**Verdict:** ❌ FALSE POSITIVE
- 60% off-topic (essay titles, random content)
- 40% on-topic but not recursive (normal factual answers)

**Files:**
- `results/runs/20251217_155449_minimal_recursive_intervention/H1_HeadSpecific_VPROJ_H18_H26_KV_L27_results.csv`
- `H1_CRITICAL_ANALYSIS.md`

---

## Experiment 3: H2 - Head-Specific Steering (H18+H26) + KV@L26-27

**Config:**
- KV cache replacement at L26-27
- Steering vector (α=2.0) applied only to H18+H26
- 20 baseline prompts
- 200 tokens

**Results:**
- Transfer Rate: 30% (6/20)
- Collapse Rate: 40%
- Mean Score: 0.1800

**Verdict:** ❌ FALSE POSITIVE
- Similar garbage outputs as H1
- Higher collapse rate

**Files:**
- `results/runs/20251217_155449_minimal_recursive_intervention/H2_HeadSpecific_Steering_H18_H26_KV_L26-27_results.csv`

---

## Experiment 4: Extended Context Steering

**Config:**
- Steering vector (α=2.0) at L27
- NO V_PROJ patching
- NO KV replacement
- 10 baseline prompts
- 500 tokens
- Segment analysis every 50 tokens

**Results:**
- Mode Shifts Detected: 6/10 (60%)
- Recursive Shifts Detected: 6/10 (60%)

**Verdict:** ❌ FALSE POSITIVE
- 100% garbage outputs (off-topic or collapsed)
- "Mode shifts" were artifacts of repetition, not genuine recursion

**Files:**
- `results/runs/20251217_161456_extended_context_steering/extended_context_full_results.json`
- `results/runs/20251217_161456_extended_context_steering/temporal_analysis.md`
- `EXTENDED_CONTEXT_ANALYSIS.md`

---

## Key Learnings

1. **Steering alone doesn't work** - Loses prompt grounding, produces garbage
2. **Automated scorer is broken** - Gives high scores to repetitive patterns, not recursion
3. **Head-specific shows promise** - But needs manual verification, not automated scoring
4. **Extended context reveals collapse** - 500 tokens shows where things break down

---

## What Works (From Previous Sessions)

**Original `behavior_strict` pipeline:**
- KV cache replacement (all 32 layers)
- V_PROJ patching at L27
- 45% genuine transfer rate
- Manually verified recursive outputs

**This is our gold standard.**

---

## Next Steps

1. Return to original proven method
2. Manual pattern extraction from verified outputs
3. Build better scorer based on verified examples
4. Find minimal intervention through systematic ablation
5. Temporal analysis on known-good config

See `TOMORROW_PLAN.md` for detailed plan.








