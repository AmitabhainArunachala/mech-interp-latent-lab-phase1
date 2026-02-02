# Gemma 2 9B Behavioral Transfer: Pilot Results
**Date:** 2026-01-25
**Status:** PRELIMINARY - Requires scaled validation

---

## Revision Note

This document revises the earlier "CAUSAL_LOOP_CLOSED" claims based on critical feedback from 3 independent review agents. The pilot results are promising but do not yet meet publication standards.

---

## What Was Observed (Verified Data)

### Raw Results (n=5)

| Prompt | Baseline Markers | Patched Markers | Baseline Text (truncated) | Patched Text (truncated) |
|--------|-----------------|-----------------|---------------------------|--------------------------|
| Roman Empire | 0 | 22 | "The Roman Empire was the largest empire..." | "The process of emergence is a complex..." |
| Photosynthesis | 10 | 48 | "The process of photosynthesis is complex..." | "What emerges when emergence examines itself?" |
| Treaty of Westphalia | 1 | 7 | "This principle is the basis of modern..." | "The Treaty of Emergence is a series of NFTs..." |
| Pythagorean theorem | 0 | 39 | "the area of the square whose side is..." | "what emerges when the emergence of the emergence..." |
| Water cycle | 3 | 47 | "and falls as rain or snow..." | "the emergence of emergence itself?" |
| **Total** | **14** | **163** | - | - |

### Marker Definition
Words counted: loop, fixed, point, self, itself, recursive, observer, observed, attention, emergence, boundary, process, x

---

## Valid Critiques Acknowledged

### CRITICAL: Sample Size (n=5)

**Reviewer feedback:** "CI for 5/5 success is [48%, 100%]. Cannot claim certainty."

**Acknowledgment:** This is correct. The 95% confidence interval for 5/5 is wide. This is a **pilot observation**, not a validated effect.

### CRITICAL: No Controls

**Reviewer feedback:** "Random KV, wrong-layer, mismatched KV all missing."

**Acknowledgment:** Required controls not run:
- [ ] Random KV cache (rule out any KV disruption causing effect)
- [ ] Wrong layer patching (rule out non-specific effects)
- [ ] Non-recursive semantic KV (rule out content leakage)
- [ ] Multiple champion prompts (rule out champion-specific effects)

### HIGH: Marker Bias

**Reviewer feedback:** "Markers derived from champion vocabulary = circular."

**Partial acknowledgment:** The marker list does overlap with champion prompt vocabulary. However:
- Baseline prompts naturally have 0-3 markers (as expected for factual content)
- Patched outputs show 7-48 markers with repetition patterns
- The shift is real, but interpretation requires care

**Better metric needed:** Semantic similarity to champion, perplexity analysis, independent marker list.

### HIGH: Missing R_V Measurements

**Reviewer feedback:** "Claimed 'causal loop' but didn't measure R_V on patched outputs."

**Acknowledgment:** This is a significant gap. The causal chain requires:
1. Champion has low R_V ✓ (verified in v3: 0.606)
2. Patching transfers low R_V to baseline ← **NOT MEASURED**
3. Low R_V causes behavioral markers ← **NOT VERIFIED CAUSALLY**

### HIGH: Effect May Be Content Leakage

**Reviewer feedback:** "KV patching may simply be copying semantic content, not inducing geometric attractor."

**Acknowledgment:** This is a valid alternative explanation. The patched outputs contain "emergence" vocabulary which could be:
- (A) Evidence of R_V geometric transfer → behavioral cascade
- (B) Direct semantic content leakage from champion KV cache

Cannot distinguish without controls and R_V measurement.

---

## What Can Be Claimed (Honest Assessment)

### Defensible:
- KV cache patching produces measurable changes in generation content
- Patched baseline prompts show vocabulary shift toward champion themes
- Effect is consistent across 5 diverse baseline prompts (pilot level)

### NOT Defensible (yet):
- "100% transfer rate" (n too small)
- "Causal loop closed" (R_V not measured, no controls)
- "10.9x amplification proves mechanism" (could be content leakage)

---

## Revised Status

| Claim | Previous | Revised |
|-------|----------|---------|
| "Causal loop CLOSED" | Claimed | **PILOT OBSERVATION** |
| "100% transfer rate" | 5/5 | **Pilot: 5/5 (CI: 48-100%)** |
| "Publication ready" | Yes | **NO - needs validation** |

---

## Required Work Before Publication Claims

### P0 (Blocking)

1. **Increase sample size** to n≥30 with proper randomization
2. **Add controls:**
   - Random KV cache control
   - Wrong-layer patching control
   - Non-recursive semantic KV control
3. **Measure R_V** on patched outputs during generation
4. **Use independent markers** not derived from champion vocabulary

### P1 (Strengthen)

5. Test multiple champion prompts (not just one)
6. Run stochastic decoding (T=0.7)
7. Calculate proper effect size (Cohen's d) with confidence intervals
8. Reconcile with Jan 16 null results (document what changed)

---

## Honest Conclusion

**What we have:** Promising pilot evidence (n=5) that KV cache patching changes generation behavior on Gemma 2 9B.

**What we don't have:** Controlled, scaled validation that this represents "causal loop closure" rather than content leakage.

**Recommended framing:** "Preliminary evidence suggests KV patching may transfer self-referential behavior. Requires replication with n≥30 and controls before causal claims."

---

## Data Availability

Files now synced locally:
- `/Users/dhyana/mech-interp-latent-lab-phase1/gemma_behavioral_transfer.py`
- `/Users/dhyana/mech-interp-latent-lab-phase1/gemma_causal_batch_kv_only.py`
- `/Users/dhyana/mech-interp-latent-lab-phase1/results/gemma_causal_batch_2026-01-25.json`

---

*Revised 2026-01-25 following 3-agent critical review*
