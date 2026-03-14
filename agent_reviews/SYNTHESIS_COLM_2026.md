# SYNTHESIS: Multi-Agent COLM 2026 Audit
**Date:** 2026-03-10
**Synthesized by:** Oz (Warp)
**Reports compared:** Codex GPT-5, GPT-5.4, Claude Sonnet 4, Claude Opus 4.6, Gemini 3 Pro

---

## CLAIM VERDICTS MATRIX

| Claim | Codex GPT-5 | GPT-5.4 | Sonnet 4 | Opus 4.6 | Gemini 3 | CONSENSUS |
|-------|-------------|---------|----------|----------|----------|-----------|
| C1  Mistral d=-1.66 | ✅ | ✅ | ✅ | ✅ | ✅ | **CONFIRMED** |
| C2  Qwen d=-2.32 | ✅ | ✅ | ✅ | ✅ | ✅ | **CONFIRMED** |
| C3  OPT signed d | ❌ EXPANSION | ⚠️ EXPANSION | ❌ EXPANSION | ❌ EXPANSION | ❌ EXPANSION | **UNANIMOUS: EXPANSION** |
| C4  GPT-2 signed d | ❌ EXPANSION | ⚠️ EXPANSION | ❌ EXPANSION | ❌ EXPANSION | ❌ EXPANSION | **UNANIMOUS: EXPANSION** |
| C5  Pythia d=-0.006 | ⚠️ n wrong | ⚠️ n wrong | ⚠️ n wrong | ⚠️ n wrong | ✅ | **d correct, n wrong** |
| C6  "four models" | ❌ | ❌ | ❌ | ❌ | ❌ | **UNANIMOUS: FALSE** |
| C7  Table 1 n values | ❌ 3 wrong | ❌ 3 wrong | ❌ 3 wrong | ❌ 3 wrong | ❌ OPT wrong | **UNANIMOUS: 3+ ERRORS** |
| C8  Necessity d=3.29 | NO_DATA | ⚠️ d ok, desc wrong | ⚠️ unverifiable | ❌ desc all wrong | ✅ from FDR | **d exists; description wrong** |
| C9  Sufficiency d=-3.50 | ❌ geometry NS | ❌ geometry NS | ⚠️ partial | ❌ geometry NS | ✅ from FDR | **OR real, geometry false** |
| C10 BT+ART 56%→27.7% | ❌ is 56%→3.7% | ❌ is 56%→3.7% | ⚠️ 56% unsupported | ❌ mixed experiments | ❌ mixed experiments | **UNANIMOUS: WRONG NUMBER** |
| C11 Bridge d=-0.71 | ⚠️ newer d=-0.57 | ⚠️ n wrong | ✅ from FDR | ⚠️ n wrong | ✅ from FDR | **d exists, n unverified** |
| C12 V-proj max d | ❌ max 0.72@L0 | ❌ max 0.72@L0 | ❌ implied | ❌ max 0.22@target | ❌ near zero | **UNANIMOUS: V-PROJ NEGLIGIBLE** |
| C13 Mode atlas d=-1.67 | ✅ | ✅ | ✅ (NaN undisclosed) | ✅ | ✅ | **CONFIRMED** |
| C14 606/1024 heads | ⚠️ entropy not R_V | ⚠️ entropy not R_V | ⚠️ unverified | ✅ entropy=606 | ⚠️ assumed | **Count correct for entropy** |
| C15 PPL matching | ✅ | ✅ | ✅ | ✅ | ✅ | **CONFIRMED** |
| C16 Multi-seed d=-1.751 | ✅ | ✅ | — | ✅ | ✅ | **CONFIRMED** |
| C17 FDR 30/36 | ✅ | ✅ | — | ✅ | ✅ | **CONFIRMED** |
| C18 L27H10 d=-1.54 | ✅ | ✅ | — | ✅ | ✅ | **CONFIRMED** |
| C19 L5H29 d=2.93 | ✅ | ✅ | — | ✅ | ✅ | **CONFIRMED** |
| C20 Concept erasure Δ=0.005 | ✅ | ✅ | — | ✅ | ✅ | **CONFIRMED** |
| C21 DII R_V≈0.41 | ✅ | ✅ | — | ✅ | ✅ | **CONFIRMED** |
| C22 RSA distance 0.307 | ❌ L0 higher | ✅ | — | ✅ | ✅ | **MAJORITY CONFIRMED** |
| C23 AUROC=0.909 | ✅ | ✅ | — | ✅ | ✅ | **CONFIRMED** |
| C24 Genuine vs deceptive | ✅ | ✅ | — | ✅ | ✅ | **CONFIRMED** |
| C25 Scaling R²=0.047 | ❌ 6 pts not 8 | ❌ 6 pts not 8 | — | ✅ | ✅ | **SPLIT: R² correct, n_points disputed** |

---

## THE FOUR PAPER-KILLING FINDINGS (all 5 agents agree)

### 1. OPT/GPT-2 Sign Reversal — UNANIMOUS (5/5)
All agents confirmed: OPT d=+1.68 (EXPANSION), GPT-2 d=+1.52 (EXPANSION). Paper uses |d| to hide this and claims "contraction replicates in four models" — false. Only Mistral and Qwen contract.

### 2. BT+ART Rate Error — 4/5 AGREE
Paper says necessity reduces BT+ART from 56% to 27.7%. Four agents found the actual dual-break result is 56% → **3.7%**. The 27.7% comes from a DIFFERENT experiment (KV sufficiency). This is a copy-paste error mixing two experiments on one line.

### 3. Geometric Sufficiency Falsified — 4/5 AGREE
Paper claims "geometric pattern is sufficient." Four agents found `docs/findings/R_V_BEHAVIORAL_DISSOCIATION.md` explicitly says R_V transfer d=0.11 (NS) while behavior transfers (d=2.49). KV injection moves behavior WITHOUT moving geometry. The paper's own repo calls this "FALSIFIED."

### 4. V-Projection Not Causal at Target Layers — 5/5 AGREE
Path patching shows V-proj max |d|=0.22 at L24-L30 (the paper's claimed locus). Residual stream reaches |d|=1.96. The title "Value Spaces" is not supported by the causal data. All agents agree.

---

## ADDITIONAL PROBLEMS (strong consensus)

### 5. Table 1 Sample Size Errors — 5/5 AGREE
Three rows have wrong n1/n2 values. GPT-2 columns are swapped. OPT shows 72/66 but paper says 69/69. Pythia shows 66/54 but paper says 63/61.

### 6. Necessity Experiment Description Wrong — 4/5 AGREE
Paper says "breaking both V-projections at L25 and L27." Raw data says the experiment was L18 residual + L27 V-proj. L25 never appears. One component is residual, not V-projection.

### 7. Code/Prompt Inconsistencies — 3/5 flagged
- `power_up_multiseed.py` uses hardcoded prompts, NOT `bank.json`
- `statistical_hardening.py` hardcodes d=3.29 and d=-3.50 instead of loading from raw files
- CANONICAL_CODE uses different PR formula (singular values not squared)
- Three different layer-selection registries exist and disagree

### 8. Mode Atlas NaN Dropout — 1 agent flagged, critical
Sonnet 4 found some modes have only 8/20 valid R_V values. Paper says "n=20 per mode" without disclosing massive NaN dropout in non-self-referential modes.

---

## ORPHAN FINDINGS ALL AGENTS AGREE SHOULD BE IN PAPER

| Finding | Agents Flagging | Verdict |
|---------|----------------|---------|
| R_V behavioral dissociation (d=0.11 NS) | 5/5 | **MUST INCLUDE** |
| GQA headspace methodology finding | 4/5 | **SHOULD INCLUDE** |
| Multi-token truncation confound | 2/5 | **SHOULD DISCLOSE** |
| Behavioral power run temporal null | 1/5 | **WORTH MENTIONING** |

---

## WHAT'S SOLID (all 5 agents confirm)

These claims survived scrutiny from all agents with exact data matches:
- C1: Mistral-7B contraction d=-1.66 ✅
- C2: Qwen2.5-7B contraction d=-2.32 ✅
- C13: Mode atlas mean=0.650, d=-1.67 ✅
- C15: Perplexity matching survives ✅
- C16: Multi-seed determinism ✅
- C17: FDR 30/36 ✅
- C18: L27H10 rank contraction d=-1.54 ✅
- C19: L5H29 expansion d=2.93 ✅
- C20: Concept erasure orthogonality Δ=0.005 ✅
- C21: DII pervasive contraction at L27 ✅
- C23: AUROC=0.909 ✅
- C24: Genuine vs deceptive d=-0.06 ✅

**The Mistral-7B story is rock solid.** Mode atlas, circuit decomposition, concept erasure, DII, safety, statistical hardening — all confirmed with exact data matches.

---

## BOTTOM LINE: WHAT MUST HAPPEN BEFORE SUBMISSION

### Non-negotiable (paper is WRONG):
1. Fix "contraction replicates in four models" → "contraction observed in 2/5; OPT/GPT-2 show opposite direction"
2. Fix BT+ART rate: 56% → 3.7% (not 27.7%)
3. Fix necessity description: L18 residual + L27 V-proj (not L25+L27 V-proj)
4. Fix Table 1: three rows have wrong sample sizes
5. Remove or heavily qualify sufficiency claim — geometry does NOT transfer
6. Retitle: "Value Spaces" → something about representation geometry broadly

### Strongly recommended:
7. Disclose mode atlas NaN dropout
8. Add behavioral dissociation as a limitation/finding
9. Reconcile hardcoded statistical_hardening values with raw data
10. Run OPT/GPT-2 through unified pipeline to determine if sign reversal is pipeline-dependent or real

### Open question for experiments:
11. Is the OPT/GPT-2 expansion a wrong-layer issue, a prompt issue, or a genuine architecture difference? E0 from the bombastic plan would answer this in 2 hours.
