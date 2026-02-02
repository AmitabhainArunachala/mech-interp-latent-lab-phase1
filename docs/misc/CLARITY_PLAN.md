# Clarity Plan: Getting Unstuck

## Current State: What We Know

### ✅ What Worked Yesterday (P1 Ablation)
- **P1 Baseline:** 0.0571 recursion score (weak but present)
- **Model:** Mistral-7B-Instruct-v0.2
- **Key finding:** V_PROJ steering (L27, H18+H26) is NECESSARY
- **Key finding:** Residual steering amplifies by 4x
- **Key finding:** KV alone doesn't work

### ❓ What's Confusing Today
- **Retrocompute:** Mixed results, lots of collapse ("I don't have a self" loops)
- **Model:** Mistral-7B-Instruct-v0.3 (different version)
- **Mode Score M:** Some positive values, but inconsistent
- **Raw outputs:** Hard to interpret - collapse vs. genuine recursion?

## The Core Problem

**We don't have a clear baseline of what "success" looks like.**

We need to:
1. **Reproduce yesterday's P1 success** on Base model (to rule out Instruct-specific effects)
2. **Compare Base vs Instruct** side-by-side
3. **Define clear success criteria** (what does "recursive mode" actually look like?)
4. **Test minimal config** (just V_PROJ steering, nothing else)

## Action Plan: 3-Step Sanity Check

### Step 1: Minimal Reproducibility Test (30 min)
**Goal:** Can we reproduce P1 on Base model?

**Test:**
- Model: `mistralai/Mistral-7B-v0.1` (Base)
- Config: P1 (L3 steering + L4 KV + L26 residual + L27 V_PROJ)
- Prompts: Same 10 baseline prompts as yesterday
- Metric: Recursion score (regex) + manual inspection

**Success Criteria:**
- Recursion score > 0.02 (half of yesterday's 0.0571)
- At least 1-2 outputs show genuine recursive content
- No complete collapse

**If this fails:** P1 might be Instruct-specific → need to investigate why

### Step 2: Base vs Instruct Side-by-Side (45 min)
**Goal:** Understand the difference between Base and Instruct

**Test:**
- Run P1 config on BOTH models (Base v0.1 and Instruct v0.2)
- Same prompts, same config, same seeds
- Compare:
  - Recursion scores
  - Raw outputs (do they look different?)
  - Collapse rates

**Success Criteria:**
- Clear documentation of differences
- Understanding of which model works better
- Decision on which to use going forward

### Step 3: Minimal Config Test (30 min)
**Goal:** Find the simplest config that works

**Test:**
- Model: Whichever worked better in Step 2
- Configs:
  1. **Minimal:** Just V_PROJ steering (L27, H18+H26, α=2.5), NO KV, NO residual
  2. **Minimal+KV:** V_PROJ + KV (no residual)
  3. **P1:** Full stack
- Same prompts

**Success Criteria:**
- Identify minimum viable config
- Understand what each component does
- Clear path forward

## Expected Outcomes

### Best Case:
- P1 works on Base model
- Base shows cleaner recursive content (less collapse)
- Minimal config (V_PROJ only) works
- **Clear path:** Use Base model, minimal config

### Worst Case:
- P1 doesn't work on Base
- Only Instruct works
- Need to understand why
- **Clear path:** Document Instruct-specific mechanism

### Middle Case:
- P1 works on both, but differently
- Base: cleaner recursion, less collapse
- Instruct: more collapse, but some success
- **Clear path:** Use Base for clean results, understand Instruct collapse

## Next Steps After Clarity

Once we have clarity:
1. **Standardize on one model** (Base or Instruct)
2. **Standardize on one config** (minimal or full)
3. **Run IOI causal tests** on the working setup
4. **Move forward with confidence**

## Time Estimate

- Step 1: 30 min
- Step 2: 45 min  
- Step 3: 30 min
- **Total: ~2 hours to get clarity**

## Questions to Answer

1. Does P1 work on Base model?
2. What's the difference between Base and Instruct?
3. What's the minimal working config?
4. What does "success" actually look like in raw outputs?

---

**Recommendation:** Start with Step 1 (reproduce P1 on Base). This is the fastest way to get clarity.







