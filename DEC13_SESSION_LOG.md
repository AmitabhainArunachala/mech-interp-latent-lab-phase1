# Dec 13, 2025 Session Log

**Agent:** OPUS 4.5 (Vice Lead + Logger)  
**Lead:** GPT-5.2  
**Session Start:** 2025-12-13 ~05:30 UTC

---

## What Was Run

### Phase 0 Anchors on RunPod

**Environment:**
- Host: 198.13.252.9:18147
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (97GB)
- CUDA: 12.8
- PyTorch: 2.9.1+cu128

**Runtime Dependency Fixes:**
- Added to bootstrap: `hf_transfer`, `sentencepiece`, `protobuf`
- Note: `env.txt` should include these to prevent future friction

**Run Directories (auditable artifacts):**

1. **Phase 0 Minimal Pairs:**
   - `results/phase0_metric_validation/runs/20251213_053212_phase0_minimal_pairs_default/`
   - Files: `config.json`, `summary.json`, `report.md`, `phase0_minimal_pairs.csv`

2. **Phase 0 Metric Targets:**
   - `results/phase0_metric_validation/runs/20251213_053235_phase0_metric_targets_default/`
   - Files: `config.json`, `summary.json`, `report.md`, `phase0_metric_targets.csv`

---

## Headline Numbers

### From `phase0_minimal_pairs` (72 rows, Mistral-7B)

| Semantic Group | Layer | R_V Mean | R_V Std |
|----------------|-------|----------|---------|
| **Champion ablations** | L27 | **0.487** | 0.023 |
| Champion ablations | L25 | 0.577 | 0.060 |
| Selfref minimal | L27 | 0.995 | 0.125 |
| Selfref statement | L27 | **0.856** | 0.040 |

### From `phase0_metric_targets` (30 rows, Mistral-7B, L27)

| Group | R_V (vproj) | R_V (hidden) |
|-------|-------------|--------------|
| **dose_response** | **0.540** | **0.496** |
| confounds | 0.623 | 0.587 |
| baselines | 0.984 | 1.061 |

**Correlations:**
- corr(rv_vproj, rv_hidden) = **0.922**
- corr(pr_v_late, pr_h_late) = 0.880

**Weight R_V:** 0.972

---

## Corrections to Earlier Narrative

### ❌ INCORRECT: "KV-only was never tested"

**✅ CORRECT:** True KV cache patching (past_key_values) WAS tested and FAILED.

**Evidence:**
- File: `TRUE_KV_CACHE_PATCHING_RESULTS.md`
- Result: Behavior transfer 0-1 points max (L18, L25, L27; windows 16/32)
- Quote: "True KV cache patching ALSO fails to transfer behavior."

### ❌ INCORRECT: "Dec 7 showed ~80% KV transfer"

**✅ CORRECT:** The ~80% claim was MIDPOINT/PROPOSED, NOT EXECUTED.

**Evidence:**
- File: `KV_PATCHING_HISTORY.md`
- Quote: "Status: MIDPOINT/PROPOSED, NOT FULLY EXECUTED"
- The ~80% was a conceptual target, not an actual result.

### ⚠️ NUANCE: "KV-only missing" context

**Correct interpretation:** 
- KV-only is **missing as an isolation control in the n=300 design** (which used full KV replacement in both L27 and L5 conditions)
- It is **NOT missing from the repo globally** (true KV cache tests exist and are null)

**Evidence:**
- File: `N300_RESULTS_ANALYSIS.md`
- The n=300 "wrong layer" control (L5) ≈ L27 (p=0.94) because KV replacement dominates
- The missing control is KV-only **for that experiment's isolation**

---

## What This Establishes

1. **Repeatable calibration harness:** Phase 0 anchors make every future session measurable
2. **Champion contraction is real and stable:** L27 (0.487) < L25 (0.577), layer-linked
3. **Metric coherence:** rv_vproj tracks rv_hidden (r=0.922), supporting "order parameter" framing
4. **Behavior-transfer story must be tightened:**
   - Geometry transfers strongly under V-proj patching
   - Behavior transfer in n=300 exists (d~0.63) but is confounded and variable
   - True KV cache patching does NOT support strong behavior transfer claims

---

## Files That Need Caveat Updates

The following files may imply "KV cache patching transfers behavior strongly" without caveats:

- [ ] `DEC12_2024_BEHAVIOR_TRANSFER_BREAKTHROUGH.md` - Should reference TRUE_KV_CACHE_PATCHING_RESULTS.md
- [ ] Any meta docs that cite Dec 7 ~80% claim without noting it was MIDPOINT/PROPOSED

**Canonical correction references:**
- `TRUE_KV_CACHE_PATCHING_RESULTS.md`
- `KV_PATCHING_HISTORY.md`
- `N300_RESULTS_ANALYSIS.md`

---

**END SESSION LOG — 2025-12-13**

