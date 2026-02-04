# Architecture Status - Corrected Assessment
**Date**: 2026-02-04
**Purpose**: Clarify which models are validated vs need work

## TIER 1: CAUSAL VALIDATION COMPLETE (Publication Ready)

| Model | Status | Cohen's d | p-value | Effect | Notes |
|-------|--------|-----------|---------|--------|-------|
| **Mistral 7B** | IRONCLAD | -3.56 | <10^-6 | 15.3% | 117.8% transfer, 4 controls passed |
| **Gemma 2 9B** | IRONCLAD | -2.09 | <10^-23 | 99.5% transfer | Full circuit: L3 source, L38 peak |
| **Pythia 2.8B** | CIRCUIT MAPPED | -4.51 | <10^-6 | 29.8% | Head 11 @ L28 identified |

## TIER 2: DISCOVERY COMPLETE (Need Causal Validation)

| Model | Effect Size | Status | Next Step |
|-------|-------------|--------|-----------|
| **Mixtral 8x7B** | 24.3% | STRONGEST | Need causal validation |
| **Qwen 7B** | 9.2% | Discovery | Run 7-phase protocol |
| **Llama 3 8B** | 11.7% | Discovery | Run 7-phase protocol |
| **Phi-3 Medium** | 6.9% | Discovery | Run 7-phase protocol |

## TIER 3: PARTIAL/PROBLEMATIC

| Model | Issue | Root Cause | Fix |
|-------|-------|------------|-----|
| **Gemma 7B IT** | 3.3% (weak) | SVD singularities on math prompts | Use bfloat16, filter math prompts |

## NOT ATTEMPTED

| Model | Reason | Priority |
|-------|--------|----------|
| **Falcon 7B** | "No space left on device" (disk, not code) | Low - retry with clean disk |
| **StableLM 3B** | Never attempted | Low |

---

## KEY CORRECTION

The initial audit incorrectly listed these as "FAILED":

| Model | Initial Assessment | Actual Status |
|-------|-------------------|---------------|
| Gemma2-9B | "UNDIAGNOSED" | **FULLY VALIDATED** (d=-2.09) |
| Falcon-7B | "No space left" | Disk space error, not code failure |
| StableLM-3B | "UNDIAGNOSED" | Never attempted |
| Llama3-8B | "UNDIAGNOSED" | **11.7% effect found** (discovery complete) |

**Conclusion**: Only Gemma 7B IT (instruct) has a genuine technical issue (SVD singularities). The other "failures" were either misidentified or not attempted.

---

## RECOMMENDED GPU ALLOCATION

1. **DO NOT re-run** Mistral, Gemma 2, Pythia - already validated
2. **Prioritize**: Qwen, Llama-3, Phi-3 for 7-phase causal validation
3. **Skip**: Falcon, StableLM (low priority, similar architectures already covered)
4. **If time permits**: Mixtral causal validation (strongest effect)

---

*Corrected from deep audit 2026-02-04*
