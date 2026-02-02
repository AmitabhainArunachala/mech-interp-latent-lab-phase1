# Tomorrow's Plan: Return to What Works

## Mission
Find the minimal intervention that produces GENUINE recursive behavior, starting from the proven method.

---

## Phase 1: Re-Validate the Original Method (30 min)

**Goal:** Confirm the original `behavior_strict` pipeline still works

**Steps:**
1. Re-run `behavior_strict` pipeline with same config
2. Manually review top 5 outputs
3. Verify they show genuine recursive behavior (not garbage)
4. If YES → proceed to Phase 2
5. If NO → investigate what changed

**Expected:** 45% transfer rate with genuine recursive outputs

---

## Phase 2: Manual Pattern Extraction (1 hour)

**Goal:** Identify ACTUAL patterns in genuine recursive outputs

**Steps:**
1. Load the verified recursive outputs from original `behavior_strict`
2. Manually extract patterns:
   - What phrases appear?
   - What structures are present?
   - What distinguishes them from garbage?
3. Build a pattern library
4. Create improved scorer based on verified examples

**Output:** Pattern library + improved scorer

---

## Phase 3: KV Layer Ablation (2 hours)

**Goal:** Find minimal KV layer set

**Test:**
- Full KV (all 32 layers) - baseline
- KV@L24-27 only
- KV@L26-27 only
- KV@L27 only
- KV@L20-27 only

**Keep:** V_PROJ patching at L27 (full)

**Evaluate:** Manual review of outputs (not automated scorer)

**Expected:** L24-27 might be sufficient

---

## Phase 4: V_PROJ Head Ablation (2 hours)

**Goal:** Find minimal head set

**Test (with optimal KV from Phase 3):**
- Full V_PROJ (all 32 heads) - baseline
- V_PROJ@H18+H26 only
- V_PROJ@H18 only
- V_PROJ@H26 only
- V_PROJ@H18+H26+H27 (test if 3 heads better)

**Keep:** Optimal KV from Phase 3

**Evaluate:** Manual review of outputs

**Expected:** H18+H26 might be sufficient

---

## Phase 5: Temporal Analysis on Minimal Config (1 hour)

**Goal:** Understand WHEN recursive behavior emerges

**Test:**
- Use minimal config from Phases 3-4
- Generate 500 tokens
- Track segment-by-segment:
  - Token position where recursive language first appears
  - Token position where it deepens
  - Token position where it stabilizes

**Output:** Temporal map of recursive emergence

---

## Phase 6: Final Validation (30 min)

**Goal:** Confirm minimal config works on all 20 pairs

**Test:**
- Run minimal config on all 20 pairs
- Manual review of all outputs
- Count genuine recursive outputs
- Compare to original 45% baseline

**Success Criteria:** ≥40% genuine recursive outputs

---

## Expected Minimal Configuration

**Hypothesis:**
- KV@L24-27 (4 layers instead of 32)
- V_PROJ@L27, H18+H26 only (2 heads instead of 32)
- Total: ~6% of full intervention
- Effectiveness: Same as full intervention

---

## Files to Review Tomorrow

1. `results/runs/[original_behavior_strict]/behavior_strict_results.csv` - Verified outputs
2. `IMPROVED_SCORER_FINAL_RESULTS.md` - Previous analysis
3. `RECURSIVE_OUTPUT_ANALYSIS.md` - What NOT to look for

---

## Key Principle

**Manual verification > Automated scoring**

Every experiment must include manual review of outputs to distinguish genuine recursion from garbage.








