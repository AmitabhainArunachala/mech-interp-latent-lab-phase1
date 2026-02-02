# META-FACTCHECK (Agent Writeups Audit)

**Title:** META-FACTCHECK (agent writeups audit)  
**Date:** 2025-12-15  
**Model:** Claude Opus 4.5 (claude-opus-4-20250514)  
**Reviewed files:**
- `20251215__claude-composer__TOP_FINDINGS_LEDGER.md`
- `20251215__claude-opus-4-5__TOP_FINDINGS_LEDGER.md`
- `20251215__gemini-3-pro-preview__TOP_FINDINGS_LEDGER.md`
- `20251215__grok-code-fast-1__TOP_FINDINGS_LEDGER.md`

---

## 1) Inventory

| File | Lines | Header Format | Prompt Bank Version |
|------|-------|---------------|---------------------|
| claude-composer | 858 | ✅ Complete | b1e5291421c5646d |
| claude-opus-4-5 | 611 | ✅ Complete | "not checked" |
| gemini-3-pro-preview | 129 | ✅ Complete | b1e5291421c5646d |
| grok-code-fast-1 | 355 | ✅ Complete | "not checked" |

**Note:** 4 files found as expected. Two agents (claude-opus, grok) did not verify prompt bank version.

---

## 2) Trust Score Per Agent

### Claude Composer (Score: 8/10)

**Strengths:**
1. Most comprehensive coverage (858 lines, 12 detailed findings)
2. Clear VERIFIED/UNCERTAIN/NOT VERIFIED status labels with explicit evidence paths
3. Detailed control validation documentation (random/shuffled/wrong-layer)

**Failures:**
1. Claims "R_V Implementation: UNCERTAIN" but then says "primary is canonical" — contradictory framing
2. Missing the critical n=300 study that contradicts L27 specificity for behavior
3. Layer 21 phase transition claim listed as "preliminary" but no mention of n=300 evidence that L21 patching works

---

### Claude Opus 4.5 (Score: 7/10)

**Strengths:**
1. Correctly identifies R_V implementation discrepancy (layer 28 vs 27, per-head averaging)
2. Accurately flags GQA aliasing and retracts H18/H26 as individually special
3. Good skepticism about artifact support ("claims exceed artifacts")

**Failures:**
1. Did not verify prompt bank hash (listed as "not checked")
2. Missing Finding 6 cross-architecture details from actual CSVs
3. Does not mention the n=300 study that contradicts L27 behavior specificity

---

### Gemini 3 Pro Preview (Score: 9/10)

**Strengths:**
1. **CRITICAL FIND:** Only agent to surface the n=300 result showing L21=L27 for behavior (p=0.94)
2. Clear "Specificity Crisis" framing that changes interpretation
3. Concise (129 lines) but hits all critical issues

**Failures:**
1. Less detail on head-level findings and GQA aliasing
2. Missing detailed stats on confound validation runs
3. No explicit path to `neurips_n300_summary.md` artifact (I had to find it)

---

### Grok Code Fast 1 (Score: 5/10)

**Strengths:**
1. **Only agent to flag inverse PR formula** in `models/*.py` files — VERIFIED by repo check
2. Comprehensive "Next Moves" section with specific pipeline proposals
3. Clear tabular format

**Failures:**
1. **CRITICAL ERROR:** Claims "100% Behavior Transfer" as VERIFIED, but n=300 study shows L21 works equally well (d=0.65 vs d=0.63)
2. **MATH ERROR:** States PR formula as `(Σλᵢ)² / Σλᵢ²` — this is WRONG; canonical is `(Σλᵢ²)² / Σ(λᵢ²)²`
3. Many "VERIFIED" labels on findings that other agents correctly flag as UNCERTAIN

---

## 3) Cross-Report Contradiction Map

| Claim Topic | Claude Composer | Claude Opus | Gemini | Grok | Compatible? |
|-------------|-----------------|-------------|--------|------|-------------|
| **R_V Formula** | `(Σλᵢ²)² / Σ(λᵢ²)²` | `(Σλᵢ²)² / Σ(λᵢ²)²` | `(Σλᵢ²)² / Σ(λᵢ²)²` | **`(Σλᵢ)² / Σλᵢ²`** | ❌ NO |
| **L27 Behavior Transfer** | UNCERTAIN (N=1) | UNCERTAIN (N=1) | **CONTRADICTED** (L21=L27) | VERIFIED (100%) | ❌ NO |
| **H18/H26 Special** | VERIFIED (as KV group) | CONTRADICTED (GQA alias) | — | VERIFIED (head groups) | ⚠️ PARTIAL |
| **Inverse PR Bug Exists** | Not mentioned | Noted layer 28 issue | Not mentioned | **VERIFIED (6 files)** | ✅ YES (verified) |
| **MoE 59% Amplification** | VERIFIED | UNCERTAIN (CSV missing?) | VERIFIED | VERIFIED | ⚠️ PARTIAL |
| **L27 is Peak Layer** | VERIFIED | UNCERTAIN (N=1) | VERIFIED | VERIFIED | ⚠️ PARTIAL |
| **Dose-Response** | VERIFIED | Not artifact-backed | VERIFIED | VERIFIED | ✅ YES |

### Critical Conflict #1: L27 Behavior Transfer

**Gemini says:**
> "The N=300 study reveals that L21 V-Proj patching works just as well as L27 (p=0.94)"

**Grok says:**
> "100% Behavior Transfer Achieved... Status: VERIFIED"

**Repo verification:** `neurips_n300_summary.md` confirms:
- Transfer (L27): d=0.63
- Wrong layer (L21): d=0.65
- Comparison: t=0.07, **p=0.944**

**Verdict:** Gemini is correct. Grok's "VERIFIED" status is FALSE. The effect is NOT specific to L27.

### Critical Conflict #2: PR Formula

**Grok defines:**
```
PR = (Σλᵢ)² / Σλᵢ²
```

**Canonical (`src/metrics/rv.py` line 7):**
```
PR = (Σλᵢ²)² / Σ(λᵢ²)²
```

**Verdict:** Grok's formula is mathematically incorrect. Uses λ instead of λ².

---

## 4) Hallucination / Citation Integrity Sweep

### Claude Composer

| Referenced File | Status |
|----------------|--------|
| `src/metrics/rv.py` | ✅ Real (verified) |
| `docs/MEASUREMENT_CONTRACT.md` | ❓ Not verified |
| `MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md` | ✅ Real |
| `V_PROJ_DISCOVERY_RESULTS.md` | ❓ Not verified |
| `BREAKTHROUGH_BEHAVIOR_TRANSFER.md` | ❓ Not verified |
| `H31_VALIDATION_FINAL_SUMMARY.md` | ❓ Not verified |
| `boneyard/DECEMBER_2025_EXPERIMENTS/DEC8_RUNPOD/...` | ❓ Long path, plausible |

### Claude Opus 4.5

| Referenced File | Status |
|----------------|--------|
| `DECEMBER_2025_EXPERIMENTS/DEC8_RUNPOD/.../full_validation_20251208_130707.csv` | ❓ Very specific, plausible |
| `models/mistral_7b_analysis.py` lines 19-21 | ✅ **Verified** (checked LATE_LAYER=28) |
| `target_comparison_output.txt` | ❓ Not verified |
| `target_acquisition_comparison.csv` | ❌ Explicitly listed as "referenced but not present" |
| `KV_PATCHING_HISTORY.md` | ❓ Not verified |
| `MECHANISM_MAP.md` | ❓ Not verified |

### Gemini 3 Pro Preview

| Referenced File | Status |
|----------------|--------|
| `neurips_n300_summary.md` | ✅ **Verified** (critical finding confirmed) |
| `DEC12_2024_BEHAVIOR_TRANSFER_BREAKTHROUGH.md` | ❓ Not verified |
| `DEC13_DEEP_DIVE_SYNTHESIS.md` | ❓ Not verified |
| `GRAND_UNIFIED_TEST_RESULTS.md` | ❓ Not verified |

### Grok Code Fast 1

| Referenced File | Status |
|----------------|--------|
| `models/*.py` inverse PR bug | ✅ **Verified** (grep found 6 files) |
| `PHASE1_FINAL_REPORT.md` | ❓ Not verified |
| `BREAKTHROUGH_BEHAVIOR_TRANSFER.md` | ❓ Not verified |
| `R_V_PAPER/code/VALIDATED_mistral7b_layer27_activation_patching.py` | ❓ Long path, plausible |

---

## 5) Math + Definitions Audit

### PR Formula Consistency

| Agent | Formula Given | Matches Canonical? |
|-------|--------------|-------------------|
| Claude Composer | `(Σλᵢ²)² / Σ(λᵢ²)²` | ✅ YES |
| Claude Opus 4.5 | `(Σλᵢ²)² / Σ(λᵢ²)²` | ✅ YES |
| Gemini | `(Σλᵢ²)² / Σ(λᵢ²)²` | ✅ YES |
| Grok | `(Σλᵢ)² / Σλᵢ²` | ❌ **NO** (missing squares on λ) |

**Impact:** Grok's formula would compute a different metric entirely. This undermines trust in their mathematical claims.

### Layer Definitions

| Agent | Early Layer | Late Layer | Window |
|-------|-------------|------------|--------|
| Claude Composer | 5 | num_layers - 5 (27) | 16 |
| Claude Opus 4.5 | 5 | 27 (notes 28 discrepancy) | 16 |
| Gemini | 5 | 27 | 16 |
| Grok | 5 | num_layers - 5 (27) | 16 |

**Verdict:** All agents agree on canonical parameters. Opus correctly flags the layer 28 discrepancy in `models/mistral_7b_analysis.py`.

### Internal Self-Contradictions

**Grok:**
- Claims "PR = (Σλᵢ)² / Σλᵢ²" in Section A
- Then says "10+ files use inverted PR formula" as a bug
- But the "inverse" formula `1.0 / (S_sq_norm ** 2).sum()` is actually a THIRD formula
- Inconsistent on what the bug actually is

**Claude Composer:**
- Finding 6 says behavior transfer is "UNCERTAIN (single prompt pair)"
- But doesn't mention the n=300 study that exists and contradicts L27 specificity

---

## 6) Consensus Findings (Survived Scrutiny)

The following claims are supported by multiple agents AND pass basic verification:

1. **R_V contraction is real at L27 for Mistral-7B** — Claude Composer, Claude Opus, Gemini, Grok all agree; N=40-45 documented; Cohen's d ≈ -3.5 to -5.5

2. **Canonical R_V formula is `(Σλᵢ²)² / Σ(λᵢ²)²`** — 3/4 agents agree; verified in `src/metrics/rv.py`

3. **GQA aliasing means H2/H10/H18/H26 are indistinguishable** — Claude Opus, Claude Composer, Grok acknowledge; Gemini silent but not contradicting

4. **Controls (random/shuffled/wrong-layer) validated for L27 patching** — All agents cite similar control structure for geometric effect

5. **Inverse PR formula exists in `models/*.py`** — Grok identified; **verified by grep** (6 files)

6. **Dec 7-8 KV transfer claims were retracted** — Claude Opus documents; Grok references

7. **L27 specificity does NOT hold for behavioral transfer** — Gemini documents (n=300, p=0.94); **verified via `neurips_n300_summary.md`** (conditional: this contradicts other agents)

8. **H31 is correlational/sensor, not causal** — Gemini states ablation has zero effect on R_V; Claude Composer says "weaker than claimed"

---

## 7) What's Missing From All of Them

| Gap | Severity | Notes |
|-----|----------|-------|
| **No agent provides prompt bank SHA256 hash from actual code execution** | HIGH | Two say "not checked", two give same hash but unclear provenance |
| **No agent verifies n=300 study prompt provenance** | HIGH | Was it run with canonical prompt bank? |
| **Missing: What IS the inverse PR formula's effect?** | HIGH | Grok flags it exists but no agent quantifies impact on past claims |
| **Seed replication absent across all** | MEDIUM | All agents note single-seed runs; none report multi-seed results |
| **Behavior metric definition unclear** | MEDIUM | Is it 0-11 keywords? Which keywords? All agents reference but none define precisely |
| **No agent addresses L21 vs L27 for geometric (not behavioral) patching** | MEDIUM | L21 control for behavior works; does it for geometry? |
| **Missing: Complete layer sweep artifact** | MEDIUM | All claim L27 is special but admit N=1 tomography for layer comparison |
| **Cross-architecture CSV artifacts missing for most models** | MEDIUM | Claims about 6 models but only Mistral/Mixtral have documented CSVs |
| **Raw generated text not preserved** | MEDIUM | Behavior scores computed but text not saved for verification |
| **Prompt length distribution not reported** | LOW | Confound discussed but token length stats not given |

---

## Summary Verdict

| Agent | Trust Score | Major Issue |
|-------|-------------|-------------|
| Claude Composer | 8/10 | Missing n=300 L21=L27 finding |
| Claude Opus 4.5 | 7/10 | Good skepticism, didn't verify prompt bank |
| **Gemini** | **9/10** | **Found critical L21=L27 result others missed** |
| Grok | 5/10 | Wrong PR formula, overclaims "VERIFIED" |

### Most Critical Finding This Audit Surfaced

**The n=300 behavior transfer study (`neurips_n300_summary.md`) shows that L21 V-proj patching works just as well as L27 (p=0.94).** This directly contradicts Grok's "VERIFIED: 100% Behavior Transfer" claim and should downgrade all "L27 is uniquely causal for behavior" claims to CONTRADICTED.

The geometric effect (R_V contraction) at L27 remains validated. But behavioral transfer is NOT specific to L27 — the Full KV cache or a broad late-layer sensitivity appears responsible.

---

*Audit completed 2025-12-15 by Claude Opus 4.5 (meta-auditor)*

---

## ADDENDUM (Dec 16, 2025)

### GPT-5.2 Review (Not Included in Original Audit)

**Trust Score: 9/10** — Artifact-first methodology, raised critical behavior issues

**Key Findings Not Covered Above:**

1. **Behavior metric is broken**: A_control (no intervention) shows 28% "expression rate" — classifier too permissive
2. **Random KV anomaly**: Gaussian random KV produces similar "expression" uplift as recursive KV
3. **Degeneracy confound**: Patched outputs show 5× higher 4-gram repetition (0.466 vs 0.089)
4. **Contract drift**: Short-prompt handling differs between contract (NaN) and code (min-window)

**Impact on Consensus:** GPT-5.2 confirms Gemini's skepticism about behavior claims and adds the degeneracy dimension.

### H18/H26 Gold Standard Validation (Dec 15 GPU Run)

**Results now artifact-backed:** `results/h18_h26_gold_standard/20251215_173922_h18_h26_gold_standard.csv`

**Key stats (N=100):**
- Target (KV-head 2) at L27: +7.87% delta (recursive), +8.92% (baseline)
- Control head (KV-head 0) at L27: +3.02% (recursive), +3.95% (baseline)
- Target at L21 (wrong layer): **-0.05%** (near zero)

**Verdict:** Head effect is **layer-specific** (L27 >> L21) and **head-specific** (target >> control), but **NOT prompt-type-specific** (works equally on baselines).

### Updated Trust Rankings

| Agent | Original Score | Updated Score | Notes |
|-------|---------------|---------------|-------|
| Gemini 3 Pro | 9/10 | 9/10 | Still best for critical findings |
| GPT-5.2 | — | **9/10** | Matches Gemini quality, new issues |
| Claude Composer | 8/10 | 8/10 | Comprehensive but missed n=300 |
| Claude Opus 4.5 | 7/10 | 7/10 | Good skepticism |
| Grok | 5/10 | 5/10 | Still has formula error |

