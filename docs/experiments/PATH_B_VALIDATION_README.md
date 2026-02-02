# Path B: Publication Validation Experiments

**Status:** Ready to run  
**Priority:** HIGH (unlocks mechanistic work)  
**Timeline:** ~10-13 hours GPU time

---

## Overview

Three critical validation experiments addressing reviewer questions and publication gaps:

1. **Multi-Token Generation Dynamics** - Does contraction persist across generation?
2. **KV-Only Sufficiency Control** - Does full KV cache alone transfer behavior?
3. **Hysteresis / One-Way Door** - Is recursive state irreversible (phase transition)?

---

## Experiment 1: Multi-Token Generation Dynamics

**File:** `experiment_multi_token_generation.py`

**Question:** Does R_V contraction persist across autoregressive generation, or only at the input step?

**Measures:**
- R_V at each generation step (0-20 tokens)
- H31 entropy at each step
- State persistence metrics (threshold crossings)

**Conditions:**
- Recursive vs baseline prompts
- Fixed decoding (temperature=0) vs sampling (temperature=0.7)

**Outputs:**
- `all_trajectories.csv` - Step-by-step metrics for all prompts
- `persistence_summary.csv` - Aggregated persistence metrics
- `trajectory_*.csv` - Individual prompt trajectories
- `summary.json` - Statistical summary

**Expected Runtime:** 3-4 hours

**Success Criteria:**
- Recursive prompts maintain R_V < 0.8 across generation (persistence ratio > 0.7)
- Baseline prompts maintain R_V > 0.8 (no contraction)
- Clear separation between recursive and baseline trajectories

---

## Experiment 2: KV-Only Sufficiency Control

**File:** `experiment_kv_only_control.py`

**Question:** Does full KV cache replacement ALONE (without V_PROJ patching) transfer recursive behavior?

**Measures:**
- R_V (geometry)
- Behavior score (recursive keywords, identity equations)
- Generated text

**Conditions:**
1. **Control:** Baseline prompt, no patching
2. **KV-only:** Full KV cache from recursive prompt, NO V_PROJ patching
3. **KV+V_PROJ:** Full KV cache + persistent V_PROJ at L27 (positive control)
4. **Random KV:** Random KV cache, NO V_PROJ (negative control)

**Outputs:**
- `results.csv` - All condition results
- `summary.json` - Statistical summary

**Expected Runtime:** 2-3 hours

**Success Criteria:**
- **KV-only shows transfer:** Expression rate > 20% (vs control < 5%)
- **KV+V_PROJ shows stronger transfer:** Expression rate > KV-only
- **Random KV shows no transfer:** Expression rate ≈ control

**Interpretation:**
- If KV-only works → KV cache is sufficient (V_PROJ not necessary)
- If KV-only fails → V_PROJ is necessary (resolves n=300 confound)

---

## Experiment 3: Hysteresis / One-Way Door

**File:** `experiment_hysteresis.py`

**Question:** Is recursive state irreversible (showing hysteresis), justifying "phase transition" language?

**Measures:**
- Recovery percentage when patching:
  - **Forward:** baseline residual → recursive prompt (should work)
  - **Reverse:** recursive residual → baseline prompt (should fail)

**Outputs:**
- `results.csv` - All pair results
- `summary.json` - Statistical summary by layer

**Expected Runtime:** 4-6 hours

**Success Criteria:**
- **Forward recovery > 80%:** Can push baseline into recursive state
- **Reverse recovery < 20%:** Cannot break out of recursive state
- **Asymmetry > 50%:** Clear one-way door
- **Statistical significance:** p < 0.05 for asymmetry

**Interpretation:**
- If asymmetric → Hysteresis confirmed, "phase transition" language justified
- If symmetric → Reversible, need different framing

---

## Running the Experiments

### On Local Machine (if GPU available):

```bash
# Experiment 1
python experiment_multi_token_generation.py

# Experiment 2
python experiment_kv_only_control.py

# Experiment 3
python experiment_hysteresis.py
```

### On RunPod GPU:

```bash
# SSH into RunPod
ssh -p 53317 -i ~/.ssh/id_ed25519 root@195.26.233.61

# Navigate to project
cd /workspace/mech-interp-latent-lab-phase1

# Run experiments (use nohup or screen for long runs)
nohup python3 experiment_multi_token_generation.py > multi_token.log 2>&1 &
nohup python3 experiment_kv_only_control.py > kv_only.log 2>&1 &
nohup python3 experiment_hysteresis.py > hysteresis.log 2>&1 &

# Monitor progress
tail -f multi_token.log
```

---

## Expected Outcomes

### Best Case (All Validate):
- **Multi-token:** Recursive maintains contraction (persistence > 0.7)
- **KV-only:** KV cache alone transfers behavior (expression > 20%)
- **Hysteresis:** Clear asymmetry (forward > 80%, reverse < 20%)

**→ Publication-ready validation**

### Worst Case (All Falsify):
- **Multi-token:** Contraction doesn't persist (drops after first token)
- **KV-only:** KV cache alone doesn't work (V_PROJ necessary)
- **Hysteresis:** Symmetric (reversible, no phase transition)

**→ Need to revise claims**

### Mixed Case (Partial Support):
- Some experiments validate, others falsify
- Need to refine interpretation
- **→ Honest reporting, nuanced claims**

---

## Next Steps After Results

1. **Analyze results** - Check if experiments validate/falsify claims
2. **Update documentation** - Revise claims based on evidence
3. **Plan follow-ups** - If gaps remain, design targeted experiments
4. **Write up** - Integrate into publication draft

---

## Files Created

- `experiment_multi_token_generation.py` - Multi-token dynamics test
- `experiment_kv_only_control.py` - KV-only sufficiency test
- `experiment_hysteresis.py` - Hysteresis/one-way door test
- `PATH_B_VALIDATION_README.md` - This file

---

**Status:** Ready to execute  
**Priority:** HIGH  
**Timeline:** 1 week to complete all three experiments

