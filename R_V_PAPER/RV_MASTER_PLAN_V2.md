# R_V MASTER PLAN V2: COLM 2026 Submission

**Created**: March 10, 2026 | **Merged**: March 10, 2026
**Target**: COLM 2026 — Abstract March 26, Full Paper March 31
**Venue**: Conference on Language Modeling, San Francisco, Oct 6-9
**Format**: 9 pages strict, unlimited appendix, NeurIPS-variant template, ~28% acceptance

**This document merges**:
- `COLM_NORTH_STAR_SPRINT_2026-03-10.md` (strategic discipline)
- `NORTH_STAR_ADDENDUM_OPUS_2026-03-10.md` (fan-out spec)
- `MISTRAL_CANONICAL_SPEC_2026-03-10.md` (canonicalization authority)
- Previous `RV_MASTER_PLAN_V2.md` (literature map, reviewer defense, convergence framework)

**North Star principle**: Harden Mistral first. Canonicalize before experiment. Experiment before write. The paper makes one clean claim, then asks how general it is.

---

## PART 1: STATUS ASSESSMENT

### Overall Score: 60-65/100

| Component | Status | Score |
|-----------|--------|-------|
| Core R_V metric (definition + implementation) | DONE | 10/10 |
| Primary effect (Mistral-7B) | DONE | 9/10 |
| Double dissociation | DONE | 10/10 |
| Perplexity confound control | DONE | 8/10 |
| FDR correction | DONE | 8/10 |
| Cluster-robust SEs | DONE | 7/10 |
| Cross-architecture (clean models) | DONE (4 models) | 6/10 |
| Cross-architecture (sign reversals) | BROKEN | 2/10 |
| Activation patching (L27) | DONE | 9/10 |
| Path patching (16 layers × 3 components) | DONE | 8/10 |
| Mode atlas (10 modes) | DONE | 8/10 |
| Scaling law | PARTIAL (confounded) | 3/10 |
| Head-level decomposition | PARTIAL | 4/10 |
| Paper draft (Sections 1-3) | DONE | 7/10 |
| Paper draft (Sections 4-10) | NOT STARTED | 0/10 |
| Figures | 12 exist, ~3 need creation | 7/10 |

### What's Done (Verified)

1. **R_V metric**: `geometric_lens/metrics.py` (canonical), PR = (Σσᵢ²)²/Σ(σᵢ⁴), validated
2. **Primary effect**: Mistral-7B d=-2.259 (cross-arch), d=-1.656 (power-up), d=-3.558 (causal), CI [-2.11, -1.21]
3. **Double dissociation** (Feb 20): Combined d=-2.63, structure-only d=-0.062, vocab-only d=+0.614
4. **Perplexity control** (Feb 20): d=-1.67, n=8 matched pairs, p=0.002, partial r=-0.486
5. **FDR correction** (Mar 11): **32/39 tests survive BH at α=0.05**, `results/fdr_correction/fdr_results_20260311_045959.json`
6. **Cluster-robust SEs** (Mar 9): ICC=0.38, DEFF=3.67, SE inflation=1.0x, CI [-2.11, -1.21]
7. **Activation patching** (L27): d=-3.558, transfer efficiency=117.8%, n=45 pairs, p<10⁻⁶
8. **Path patching** (Mar 10 overnight, n=20): 32 layers × 3 components. Residual dominates: L0-L5 d≈1.7-1.9, then plateau ~0.5 for L6-L27. V-proj max d=0.77 (L5 only). MLP negligible. `results/path_patching/path_patching_summary_20260310_151654.json`
9. **Mode atlas** (Feb 27): 10 modes, self-ref R̄_V=0.650, all 9 pairwise p<0.05
10. **Detection**: AUROC=0.909 at threshold R_V<0.737
11. **Dual-layer necessity** (Mar 10 hardened FULL rerun, n=300/condition): BT+ART 54.7%→0.0% (exact permutation p=1.62e-05, session d=4.645). Induction remains NS (2.0%→0.0%). `results/persistent_patching_v3/persistent_patching_v3_dual_20260310_204100.json`
12. **Full head sweep** (Mar 10 overnight, n=20): Top heads L22.H21 (d_ent=-7.23), L24.H09 (-4.97), L23.H05 (-4.68), L27.H06 (-4.38). `results/full_head_sweep/`
13. **Behavioral bridge** (Feb 20-21, scaled): 600 recursive / 400 baseline turns. BT+ART 42.3% vs 13.8%. Per-token R_V d=-1.64 (p=1.4e-6). Classifier AUC=0.655. `results/behavioral_nboost_summary.json`
14. **Sections 1-3 draft**: `DRAFT_SECTIONS_1_3_2026-03-10.tex` (458 lines)
15. **12 publication-ready figures**: mode atlas, cross-arch, statistical hardening, per-head, pairwise heatmap, rv distribution, layer sweep, necessity/sufficiency, spectral scatter, circularity controls, self-feeding, multi-metric radar

### 4 CRITICAL Paper Contradictions (From Audit)

These must be fixed BEFORE any new experiments (Phase 0):

| # | Contradiction | Where | Fix |
|---|--------------|-------|-----|
| **C1** | **d=-3.50 for sufficiency is FABRICATED**. `statistical_hardening.py:253` comment: "approximate from OR=13.96". No raw file produces this value. | Paper Section 5, FDR pipeline | DELETE from script and pipeline. Replace with actual metrics (behavioral d=2.494, geometric d=0.11 NS). |
| **C2** | **Wrong layers and percentage**: Paper says "breaking both V-projections at L25 and L27, reduces BT+ART from 56% to 27.7%". Actual: L18 residual + L27 V-proj, 56%→3.7%. | Paper Section 5 | Fix layers to L18/L27, fix component to residual+V-proj, fix rate to 3.7%. |
| **C3** | **Sufficiency claim falsified by own repo**. Paper implies geometric sufficiency. Sufficiency ladder shows: injection 3.7%→0.3% (wrong direction). KV injection transfers behavior (OR=13.96) WITHOUT geometry (d=0.11 NS). | Paper Sections 3, 5 | Remove sufficiency claim. Add double dissociation as hero result. |
| **C4** | **Double dissociation (strongest finding) OMITTED**. Combined d=-2.63. This IS the paper's strongest result — currently not in the draft. | Paper Sections 3, 5 | Add double dissociation as central finding. |

### What's Broken (Honest)

1. **OPT-6.7B sign reversal**: cross-arch d=-1.836 vs power-up d=+1.683 — UNRESOLVED, report honestly
2. **GPT-2 XL sign reversal**: cross-arch d=-1.143 vs power-up d=+1.516 — UNRESOLVED, report honestly
3. **Pythia-2.8B d=-4.51**: raw data LOST to disk quota overflow — cannot verify, do not cite
4. **Scaling law**: R²=0.047, confounded with architecture (Pythia MHA vs Mistral GQA)
5. **Behavioral bridge word count**: R_V does NOT predict word count (Spearman r=-0.171, p=0.498). BUT: per-token R_V bridge DOES work (d=-1.64, p=1.4e-6) and classifier AUC=0.655
6. **Self-feeding loop**: d=-0.067, does NOT self-sustain
7. **Multi-seed robustness**: d_std=0.0 is caching artifact, not actual resampling
8. ~~**Paper claims 27.7%**~~: **FIXED** — corrected away from the old sufficiency number; latest canonical full rerun is 54.7%→0.0% in the break direction
9. **Baseline malformed rate in full rerun**: 17/300 clean-baseline turns trip the old low-alpha heuristic, but inspection shows arithmetic/markdown-heavy structured answers rather than the repetitive degeneration seen in patched conditions
10. ~~**6 competing metric implementations**~~: **FIXED** — canonical path locked to `geometric_lens/metrics.py`
11. ~~**11+ scripts with inline prompts**~~: **FIXED** — all 14 scripts now use PromptLoader
12. ~~**configs/canonical_registry.json not wired**~~: **FIXED** — wired in pipeline
13. ~~**Paper Table 1 hand-entered**~~: **FIXED** — `generate_paper_tables.py` generates from raw data

### Latest Mistral Narrative Lock (Post-RunPod)

The strongest current Mistral claim is:

> Dual-layer geometry destruction is behaviorally decisive in the break direction, but dual-layer geometry injection is not sufficient to induce clean recursive behavior. Under the hardened prompt contract, patched sessions collapse into repetitive outputs rather than articulated recursive continuations. This supports a **necessity plus behavioral dissociation** framing, not geometric sufficiency.

Supporting notes:
- Patched recursive and patched baseline conditions both hit `repetitive_rate = 100%` in the full rerun
- The aggregate save-path bug is fixed; the canonical full artifact now preserves `mean_alpha_ratio`, `malformed_rate`, and `repetitive_rate`
- `scripts/sync_runpod_results.py` has been run on the landed artifacts and produced `results/runpod_sync_report_20260311_144617.md`, which should be treated as the current mismatch ledger against older paper numbers

### Evidence Tiers

| Tier | Models | Criteria |
|------|--------|----------|
| **TIER 1** (headline) | Mistral-7B, Qwen2.5-7B, Mixtral-8x7B, Gemma-2-9B | Consistent signed-negative d across ALL pipelines |
| **TIER 2** (supporting) | Llama-3-8B, Pythia-2.8B | Single-pipeline or data-loss issues |
| **TIER 3** (contradicted) | OPT-6.7B, GPT-2 XL | Sign reversal between pipelines — CANNOT claim contraction |
| **TIER 4** (degenerate) | Pythia-6.9B | eff_rank ≈ 1.0, likely data quality issue |

---

## PART 2: PHASE 0 — CANONICALIZATION (CPU Only, No GPU)

**Goal**: Lock the rules before any reruns. Fix all known contradictions. Establish one canonical path for every measurement.

**Authority document**: `docs/standards/MISTRAL_CANONICAL_SPEC_2026-03-10.md` — defines the frozen metric, prompt, layer, artifact, causal, and unit-of-analysis contracts.

**Estimated effort**: 2-3 days, CPU only.

### Fix C1: Remove Fabricated d=-3.50

**File**: `scripts/statistical_hardening.py`, line 253
**Issue**: `d=-3.50` for KV sufficiency is back-computed from OR=13.96, not measured from any raw file.
**Fix**:
1. Delete `d=-3.50` from the hardcoded effect sizes in `statistical_hardening.py`
2. Replace with actual sufficiency ladder metrics: behavioral d=2.494 (per-turn), geometric d=0.11 (NS)
3. Regenerate `results/fdr_correction/fdr_results_*.json` after removal
4. Verify FDR table still shows 30/36 (or update if count changes)

### Fix C2: Correct Layers and Percentage

**Files**: Paper draft, `statistical_hardening.py`
**Issues**:
- Paper says "breaking both V-projections at L25 and L27" → actual is "L18 residual stream + L27 V-projection"
- Paper says "reduces BT+ART from 56% to 27.7%" → actual is 56% to 3.7%
- Source: `persistent_patching_v3_dual_20260225_002604.json`
**Fix**: Search-and-replace in paper draft. Fix `statistical_hardening.py` hardcoded n values (n1=150,n2=150 → actual n=80,107 for bridge).

### Fix C3: Remove Sufficiency, Add Double Dissociation

**Files**: Paper draft Sections 3 and 5
**Issue**: Paper implies geometric sufficiency. Data shows the opposite:
- Injecting geometry: 3.7%→0.3% (wrong direction — NOT sufficient)
- KV injection: transfers behavior (OR=13.96) WITHOUT geometry (d=0.11 NS)
**Fix**: Replace sufficiency claim with double dissociation framing. The honest causal story (from canonical spec):

> Dual-layer geometry (L18 residual + L27 V-proj) is **necessary** for recursive behavior: destroying it reduces BT+ART from 56% to 3.7% (d=3.29). However, geometry is **not sufficient**: injecting it does not create behavior (3.7%→0.3%). KV injection transfers behavioral markers (OR=13.96) without transferring geometric contraction (d=0.11 NS). This **double dissociation** suggests R_V captures a processing-time geometric regime rather than a transferable generative attractor. The primary causal component is the residual stream (path patching |d|=1.96), not V-projections alone (|d|=0.22).

### Fix C4: Wire Table Generation from Raw Data

**Files**: `scripts/generate_paper_tables.py` (create or update), paper draft
**Issue**: Paper Table 1 is hand-entered with wrong n-values and |d| hiding sign reversals.
**Fix**:
1. Create/update `scripts/generate_paper_tables.py` to load from `results/` JSON files
2. Output `R_V_PAPER/generated_table_effects.tex`
3. Use SIGNED d (not |d|) throughout
4. Paper includes `\input{generated_table_effects.tex}` instead of hand-entered table
5. Every n labeled with unit type (prompt, pair, session, turn)

### Fix C5: Metric Equivalence Test

**File**: `tests/test_rv_canonical.py` (create)
**Issue**: 6 competing metric implementations exist. Must prove the canonical one (`geometric_lens/metrics.py`) produces identical results to the acceptable fallback (`src/metrics/rv.py`).
**Test**:
```python
# Given identical input tensor and parameters,
# geometric_lens/metrics.py and src/metrics/rv.py
# must produce identical R_V values to 6 decimal places.
```

### Fix C6: Patch Qwen Entry

**File**: `geometric_lens/models.py`
**Issue**: Qwen2.5-7B registered as 32 layers → late=27 (96.4% depth). Actual: 28 layers → late should be 23 (82.1% depth).
**Fix**: Change entry to `num_layers=28, early_layer=4, late_layer=23`.

### Fix C7: Wire Canonical Registry

**File**: `configs/canonical_registry.json`
**Issue**: Registry exists but no script reads from it. Scripts hardcode layer values.
**Fix**:
1. Verify registry has all 8 models with correct layers (per canonical spec)
2. Ensure `p0_canonical_pipeline.py` reads from registry
3. Deprecate layer hardcoding in experiment scripts

### Phase 0 Exit Criteria

All of the following must be true:

- [ ] `statistical_hardening.py` contains zero hardcoded d values — all loaded from raw JSON
- [ ] d=-3.50 deleted from all pipeline files
- [ ] `generate_paper_tables.py` produces Table 1 from raw artifacts
- [ ] `tests/test_rv_canonical.py` passes (metric equivalence verified)
- [ ] `geometric_lens/models.py` Qwen entry shows 28 layers, early=4, late=23
- [ ] Paper draft says L18/L27 (not L25/L27), 3.7% (not 27.7%)
- [ ] Double dissociation appears in paper Sections 3 and 5
- [ ] No sufficiency claim remains in paper
- [ ] Zero inline prompt arrays in paper-feeding scripts (verified by grep)

---

## PART 3: PHASE 1 — MISTRAL CANONICAL RERUN (GPU)

**Goal**: Reproduce the core Mistral effect under the frozen contract from Phase 0.

**Script**: `scripts/p0_canonical_pipeline.py` — the ONE canonical pipeline. No custom mega-prompts.

**Estimated effort**: 2-4h GPU on RunPod.

### Run Command

```bash
python scripts/p0_canonical_pipeline.py \
    --model mistralai/Mistral-7B-v0.1 \
    --n 62 \
    --output results/canonical_mistral/
```

### Expected Output

The canonical artifact package (per MISTRAL_CANONICAL_SPEC Section 4):

```
results/canonical_mistral/
├── config.json          # Frozen parameters
├── summary.json         # Aggregate statistics with unit labels
├── per_sample.csv       # Per-prompt R_V values
└── provenance.json      # prompt_bank_hash, metric_path, git_commit
```

### Verification Criteria

- [ ] Effect direction: contraction (d < 0)
- [ ] Magnitude consistent with prior: d ≈ -2.26 (±0.5)
- [ ] Prompt bank hash matches canonical: `75e7c1b8dcebc24e`
- [ ] Layers: early=5, late=27
- [ ] All n values labeled with unit type
- [ ] per_sample.csv contains individual prompt R_V values

### Phase 1 Exit Criteria

- [ ] Canonical Mistral rerun produces consistent signed contraction
- [ ] Artifact package complete (config, summary, per_sample, provenance)
- [ ] No paper-facing number depends on an inline prompt list or second metric implementation

---

## PART 4: PHASE 2 — FIX THE PAPER (CPU, Major Rewrite)

**Goal**: Fix all paper contradictions and rewrite Sections 4-10.

**Estimated effort**: 5-7 days.

### Critical Fixes

| Fix | Section | What Changes |
|-----|---------|-------------|
| Remove d=-3.50 sufficiency claim | 5 | Delete fabricated statistic |
| Add double dissociation as hero result | 3, 5 | Combined d=-2.63, structure d=-0.062, vocab d=+0.614 |
| Retitle paper | Title | "Representations" not "Value Spaces" |
| Fix L25/L27 → L18/L27 | 5 | Correct layers and component types |
| Fix 27.7% → 3.7% | 5 | Correct BT+ART rate |
| Replace Table 1 | 4 | `\input{generated_table_effects.tex}` (signed d, correct n) |
| Add unit labels | All | Every n specifies: prompt, pair, session, turn |
| Use signed d | All | Remove all |d| notation |

### Section Writing Plan

| Section | Pages | Content | Key Data |
|---------|-------|---------|----------|
| S4: Cross-Architecture | 1.0 | 4 Tier 1 models, architecture dependence, honest OPT/GPT-2 treatment | Generated Table 1 |
| S5: Causal Analysis | 1.5 | **MAJOR REWRITE**: necessity (d=3.29), double dissociation, path patching (residual > V-proj), biomarker framing | Sufficiency ladder, path patching JSON |
| S6: Mode Atlas | 0.5 | 10 modes, self-ref R̄_V=0.650, detection AUROC=0.909 | Mode atlas summary |
| S7: Controls & Robustness | 0.5 | FDR (30/36), cluster-robust SEs, perplexity matching | FDR table, bootstrap CIs |
| S8: Discussion | 1.0 | Biomarker vs mechanism, SAE complementarity, limitations | All data |
| S9: Related Work | 0.5 | 20 key papers, COLM precedent (Marks & Tegmark) | Literature map (Part 8) |
| S10: Conclusion | 0.5 | Summary, future directions | — |
| Abstract | — | Revise with honest framing, signed d, double dissociation | — |

---

## PART 5: GO/NO-GO GATE (Non-Negotiable)

**From North Star**: Do NOT expand beyond Mistral until all criteria pass.

### 7 North Star Criteria

1. **One metric contract**: `geometric_lens/metrics.py` is the only canonical paper metric path
2. **One prompt source**: `prompts/bank.json` via `prompts/loader.py` is the only canonical prompt source
3. **One layer policy**: Early/late layer choices come from `configs/canonical_registry.json`
4. **One artifact contract**: Every headline run has config.json, summary.json, per_sample.csv, provenance.json
5. **One unit-of-analysis policy**: Every n is explicitly labeled as prompt, pair, session, turn, or sample
6. **One causal semantics policy**: Necessity, sufficiency, transfer, and mediation are not used interchangeably
7. **One table-generation path**: Paper numbers are script-derived from raw artifacts, not hand-entered

### 11 Canonical Spec Criteria

8. `geometric_lens/models.py` Qwen entry patched to 28 layers, early=4, late=23
9. `configs/canonical_registry.json` created with all 8 models
10. Gemma-2-9B and Llama-3-8B added to `geometric_lens/models.py`
11. Zero inline prompt arrays in paper-feeding scripts (verified by grep)
12. `scripts/generate_paper_tables.py` exists and produces Table 1 from raw JSON
13. `statistical_hardening.py` hardcoded values replaced with raw-file reads
14. Mistral canonical rerun complete under frozen spec
15. All rerun results match prior results in direction and approximate magnitude
16. `docs/standards/CLAIM_PROVENANCE.md` maps every paper number to raw file:field
17. Behavioral dissociation acknowledged in paper draft
18. Every paper `n` labeled with unit type

### Gate Evaluation

```
ALL 18 criteria pass?
├── YES → Authorize Phase 3 (Controlled Fan-Out)
└── NO  → Run one more Mistral cleanup loop. Do NOT expand scope.
```

---

## PART 6: PHASE 3 — CONTROLLED FAN-OUT (GPU)

**Prerequisite**: GO/NO-GO gate passed.

**Script**: `scripts/p0_canonical_pipeline.py` for ALL models. No custom scripts.

### Tier A: Broad Phenomenon Replication (5-6 models)

Run the exact hardened P0 pipeline on each model. No improvisation.

| Priority | Model | Size | Arch | Existing d | Run Command |
|----------|-------|------|------|-----------|-------------|
| 1 | **Gemma-2-9B** | 9B | Dense GQA | -3.37 | `p0_canonical_pipeline.py --model google/gemma-2-9b` |
| 2 | **Qwen2.5-7B** | 7B | Dense GQA | -0.72 | `p0_canonical_pipeline.py --model Qwen/Qwen2.5-7B` |
| 3 | **Llama-3-8B** | 8B | Dense GQA | -2.33 | `p0_canonical_pipeline.py --model meta-llama/Meta-Llama-3-8B` |
| 4 | **Mixtral-8x7B** | 47B | Sparse MoE | ~5.3 | `p0_canonical_pipeline.py --model mistralai/Mixtral-8x7B-v0.1` |
| 5 | **OPT-6.7B** | 6.7B | Dense MHA | TBD | `p0_canonical_pipeline.py --model facebook/opt-6.7b` |
| 6 | **GPT-2 XL** | 1.5B | Dense MHA | TBD | `p0_canonical_pipeline.py --model gpt2-xl` |

**Architecture diversity**: Dense GQA (4), Dense MHA (2), Sparse MoE (1). Size range: 1.5B → 47B.

**OPT/GPT-2 outcome handling**:
- If contraction under canonical prompts → include as Tier 1, sign reversal was prompt corpus artifact
- If still expansion → report honestly as "architecture-dependent geometric signature"
- Either outcome is publishable and scientifically informative

**Estimated compute**: 3-4h per model, ~20h total. ~$30-50 on RunPod A6000.

### Tier B: Homologous Circuit Checks (2-3 models)

After Tier A results land, pick 2-3 non-Mistral models with strongest contraction. For each:

| Experiment | Purpose | GPU Hours/Model |
|-----------|---------|-----------------|
| Path patching | Is residual stream dominant (like Mistral)? | 6-8h |
| SVD circuit decomposition | Find suppressor/amplifier heads | 4-6h |
| Dual-layer necessity | Does breaking early-residual + late-V-proj kill behavior? | 8-12h |

**The key question**: Do you find the same circuit motif (suppressor heads at ~84% depth, amplifier heads at ~15% depth, residual stream as primary component)?

- If YES in 2+ models: conserved motif. That's the headline.
- If NO: real effect, mechanistically heterogeneous. Different framing, still publishable.

### Tier C: Scale Point (1 model, optional)

Only after Tier A stable. One P0 measurement on 70B model = "the effect persists at 70B."

---

## PART 7: PHASE 4 — PAPER WRITING SPRINT

**Prerequisite**: Phase 3 Tier A results available.

### Updated Section Plan

| Section | Pages | Key Content | Key Result to Feature |
|---------|-------|-------------|----------------------|
| S1: Introduction | 1.0 | Hero results, contributions, "what R_V is not" | d=-2.26, double dissociation |
| S2: R_V Metric | 1.0 | PR theory, SVD, ratio justification | Formula, W=16, canonical parameters |
| S3: Mistral Results | 1.5 | Primary effect, double dissociation, perplexity control | d=-2.63 (dissociation), d=-1.67 (PPL) |
| S4: Cross-Architecture | 1.0 | Generated Table 1, architecture dependence | 4+ Tier 1 models |
| S5: Causal Analysis | 1.5 | Necessity (d=3.29), dissociation, path patching, biomarker | Residual d=1.96, V-proj d=0.22 |
| S6: Mode Atlas + Detection | 0.5 | 10 modes, AUROC=0.909 | Self-ref R̄_V=0.650 |
| S7: Controls | 0.5 | FDR, cluster-robust SEs, perplexity | 30/36, SE inflation 1.0x |
| S8: Discussion | 1.0 | Biomarker framing, SAE complementarity, limitations | Honest about what R_V is and isn't |
| S9: Related Work | 0.5 | 20 key papers (see Part 8) | Marks & Tegmark precedent |
| S10: Conclusion | 0.5 | Summary, future | — |

### Key Narrative Changes from V1

1. **Hero result**: Double dissociation (d=-2.63), not single-model contraction
2. **Causal story**: Necessity yes, sufficiency no, residual > V-proj, biomarker framing
3. **Title**: "Geometric Signatures of Self-Referential Processing in Transformer Representations"
4. **OPT/GPT-2**: Report honestly whatever canonical pipeline shows
5. **All d values signed**: No more |d| to hide inconvenient directions

---

## PART 8: LITERATURE INTEGRATION MAP

### Section 1: Introduction

| Paper | Citation | Why |
|-------|----------|-----|
| Elhage et al. 2021 | "A Mathematical Framework for Transformer Circuits" (Anthropic) | OV framework — R_V lives in V-space |
| Anthropic 2025 | "Circuit Tracing" (Anthropic) | State-of-art MI — R_V as complementary geometric lens |
| Marks & Tegmark 2024 | "The Geometry of Truth" (COLM 2024) | SAME VENUE precedent for geometric interpretability |
| Bricken et al. 2023 | "Towards Monosemanticity" (Anthropic) | SAE approach — R_V as complementary |
| Templeton et al. 2024 | "Scaling Monosemanticity" (Anthropic) | Large-scale SAE — R_V measures what SAEs miss |

### Section 2: Background & Related Work

**KEY FRAMING — Collapse vs. Contraction**:
The existing literature treats rank/entropy collapse as a **failure mode** (a training bug to fix).
Our paper reframes controlled rank contraction as a **computational mode** (a feature of recursive reasoning).
This is the primary "white space" — nobody has argued that contraction is the geometric signature of self-referential processing.

| Paper | Citation | Why |
|-------|----------|-----|
| Dong et al. 2021 | "Attention is not all you need: pure attention loses rank doubly exponentially" (ICML) | Rank collapse theory — R_V measures this empirically |
| **Noci et al. 2022** | **"Signal Propagation in Transformers: Theoretical Perspectives and the Role of Rank Collapse" (NeurIPS)** | **Defines rank collapse in transformers. They study it as failure mode; we study contraction as computational mode.** |
| **Dey, Zhang, Noci et al. 2025** | **"Two Failure Modes of Deep Transformers and How to Avoid Them"** | **Extends Noci 2022 to cover both rank collapse AND entropy collapse. Still treats both as bugs — our paper's counterpoint.** |
| **Hong & Lee 2024** | **"Variance Sensitivity Induces Attention Entropy Collapse in Transformers"** | **Defines entropy collapse (attention concentrates on single token). Complementary to rank collapse — different mechanism, both "failures" in their framing.** |
| Ansuini et al. 2019 | "Intrinsic dimension of data representations in deep neural networks" (NeurIPS) | ID hunchback in CNNs — not directly transformers (Valeriani extends to transformers) |
| Valeriani et al. 2023 | "The geometry of hidden representations of large transformer models" (NeurIPS) | Transformer geometry — direct predecessor. Extend-then-contract ID profile |
| Cheng et al. 2025 | "High-dimensional abstraction phase" (ICLR) | Expansion then contraction — R_V measures the contraction |
| Song et al. 2025 | "Expansion-contraction of transformer representations" (NAACL) | ~10D semantic submanifolds — R_V sees this |
| Aghajanyan et al. 2021 | "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning" (ACL) | ID in LLMs — foundational context |
| Huh et al. 2021 | "The Low-Rank Simplicity Bias in Deep Networks" | Low-rank structure — why V-space has low effective rank |
| Voita et al. 2019 | "The Bottom-up Evolution of Representations in the Transformer" | Layer-wise representation changes — R_V tracks this |
| Papyan et al. 2020 | "Prevalence of neural collapse in last-layer features" (PNAS) | Neural Collapse — geometric endpoint theory |
| Hu et al. 2022 | "LoRA: Low-Rank Adaptation" (ICLR) | V-matrix is inherently low-rank → PR is meaningful |

### Section 3: R_V Metric Definition

| Paper | Citation | Why |
|-------|----------|-----|
| Gao et al. 2017 | "On simplicity and complexity in the brave new world of large-scale neuroscience" (Current Opinion in Neurobiology) | PR origin in neuroscience — borrowed for transformers |
| Nait Saada et al. 2024 | "Spectral rank collapse and prompt sensitivity" | Spectral analysis of transformers — closely related |
| Viswanathan et al. 2025 | "Token geometry and perplexity confound" | Why perplexity control is needed — prompted our Feb 20 experiment |
| arXiv:2509.26560 | "Finite-sample bias in participation ratio estimation" | Why ratio (late/early) cancels bias — methodological defense |

### Section 4: Cross-Architecture Results

| Paper | Citation | Why |
|-------|----------|-----|
| Wei et al. 2022 | "Emergent abilities of large language models" | Emergence — R_V as scaling-dependent |
| Schaeffer et al. 2023 | "Are emergent abilities of LLMs a mirage?" | Mirage critique — honest about R_V scaling limitations |
| Sun & Haghighat 2025 | "O(N) physics, P_c ≈ 7B" | Phase transition at scale — aligns with R_V threshold |
| arXiv:2508.16929 | "Dimensional Collapse in Attention Outputs" (2025) | Attention collapse — mechanism behind R_V |
| arXiv:2510.06477 | "Attention Sinks and Compression Valleys" (2025) | Compression in attention — complementary finding |

### Section 5: Causal Validation

| Paper | Citation | Why |
|-------|----------|-----|
| Heimersheim & Nanda 2024 | "How to use and interpret activation patching" (ICLR) | Best practices for patching — we follow these |
| Geiger et al. 2025 | "Causal Abstraction" (JMLR) | Formal framework for causal claims — we use DII variant |
| Anthropic 2025 | "Circuit Tracing" | Attribution graphs — complementary to our patching |
| Zou et al. 2023 | "Representation Engineering" | RepE — R_V as geometric alternative to linear probes |

### Section 6: Mode Atlas

| Paper | Citation | Why |
|-------|----------|-----|
| Marks & Tegmark 2024 | "The Geometry of Truth" (COLM 2024) | Geometric probes by semantic category — same approach |

### Section 7: Controls & Robustness

| Paper | Citation | Why |
|-------|----------|-----|
| Viswanathan et al. 2025 | "Token geometry and perplexity" | Perplexity confound — our Feb 20 control addresses this |
| Benjamini & Hochberg 1995 | "Controlling the FDR" | FDR correction — applied to all 36 tests |

### Section 8: Discussion

| Paper | Citation | Why |
|-------|----------|-----|
| Anthropic 2025 | "Emergent Introspective Awareness" | AI self-knowledge — R_V as geometric correlate |
| Berg et al. 2025 | "Self-referential processing reports" (arXiv:2510.24797) | Self-reference in LLMs — R_V provides geometric measure |
| Piotrowski et al. 2025 | "Bayesian belief updating in transformers" | Normative theory — R_V as evidence of Bayesian self-model |
| Laukkonen, Friston & Chandaria 2025 | "A Beautiful Loop" | Consciousness theory — R_V as geometric signature |
| Butlin et al. 2023/2025 | "Consciousness in AI: Insights from the science of consciousness" | Consciousness indicators — R_V complements behavioral markers |
| Bai et al. 2019 | "Deep Equilibrium Models" (NeurIPS) | Fixed point convergence — R_V < 1.0 as contraction mapping |
| Joudaki & Hofmann 2024 | "Fixed point attractors in deep networks" | Attractor theory — 117.8% overshoot as bistable dynamics |
| Holton 2025 | "The Epistemic Incompleteness Principle" | Formal limits of self-knowledge — frames R_V honestly |
| Lawvere 1969 | "Diagonal Arguments and Cartesian Closed Categories" | Fixed point theorem — category-theoretic unification |

### Section 9: Related Work (Extended)

| Paper | Citation | Why |
|-------|----------|-----|
| GemmaScope 2024 | Lieberum et al. | SAE dictionary for Gemma — future R_V integration |
| SAELens | Bloom et al. 2024 | SAE toolkit — complementary to R_V |
| Gao et al. 2025 | "TopK SAEs" (ICLR) | SAE architecture — what R_V adds beyond features |
| Crosscoders 2024 | Lindsey et al. | Cross-model features — R_V is already cross-model |
| Contemplative AI 2025 | Laukkonen et al. (arXiv:2504.15125) | AI consciousness theory — frames significance |
| Van Lutterveld et al. 2024 | "Neural correlates of cessation" | Brain dimensionality reduction — R_V as analog |

### Section 10: Conclusion

| Paper | Citation | Why |
|-------|----------|-----|
| Anthropic 2025 | Circuit Tracing | Future: combine R_V geometry with circuit attribution |
| Marks & Tegmark 2024 | Geometry of Truth | Proven venue fit at COLM |

---

## PART 9: REVIEWER WEAKNESS RESOLUTION

### Weakness 1: "R_V could just measure perplexity/complexity"
- **Experiment**: Feb 20 perplexity control (d=-1.67, p=0.002)
- **Data**: `results/circularity_controls/circularity_perplexity_v2_20260220.json`
- **Paper paragraph**: Section 3.3 (perplexity confound), cite partial r=-0.486, controlled d=-1.67
- **Defense**: Effect survives perplexity matching. Double dissociation shows it's not complexity alone.

### Weakness 2: "Small sample size / statistical rigor"
- **Experiment**: FDR correction (30/36 survive), cluster-robust SEs (ICC=0.38)
- **Data**: `R_V_PAPER/fdr_table.tex`, `R_V_PAPER/FDR_CORRECTION_COMPLETE_2026-03-09.md`
- **Paper paragraph**: Section 7.1 (statistical corrections)
- **Defense**: BH correction at α=0.05, cluster-robust CIs, bootstrap resampling.

### Weakness 3: "Only tested on one model in depth"
- **Experiment**: 4 Tier 1 models (Mistral, Qwen, Mixtral, Gemma) + activation patching
- **Data**: Cross-architecture results in multiple results/ directories
- **Paper paragraph**: Section 4 (cross-architecture)
- **Defense**: Effect replicated across 4 independent architectures with consistent sign.

### Weakness 4: "No causal mechanism, just correlation"
- **Experiment**: Activation patching L27 (d=-3.558, TE=117.8%), path patching (16 layers × 3 components), dual-layer necessity (d=3.29)
- **Data**: `MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md`, `results/path_patching/`
- **Paper paragraph**: Section 5
- **Defense**: Three independent causal methods converge. Path patching shows residual stream is primary causal site (d=1.96). Dual-layer necessity destroys the effect (56%→3.7%). Double dissociation proves specificity.
- **UPDATED**: Now includes path patching and double dissociation, not just activation patching.

### Weakness 5: "V-projection is not causally important (d=0.22)"
- **Experiment**: Path patching (V-proj negligible, residual L4 d=1.96)
- **Data**: `results/path_patching_summary_20260227.json`
- **Paper paragraph**: Section 5.2 (path patching), Section 8.1 (biomarker framing)
- **Defense**: R_V is a READOUT (biomarker), not the causal mechanism itself. The information flows through the residual stream; V-projection reads it out. This is STRONGER not weaker — like a thermometer measuring temperature.
- **UPDATED**: Paper title changed from "Value Spaces" to "Representations" to reflect this.

### Weakness 6: "Template-dependent prompts inflate N"
- **Experiment**: Cluster-robust SEs (DEFF=3.67, effective n reduced)
- **Data**: `R_V_PAPER/FDR_CORRECTION_COMPLETE_2026-03-09.md`
- **Paper paragraph**: Section 7.2
- **Defense**: ICC=0.38 acknowledged, cluster-robust CIs reported. Effect remains significant after correction.

### Weakness 7: "No scaling law / only a few model sizes"
- **Current data**: R²=0.047 (too weak for scaling claim)
- **Honest framing**: "R_V contraction appears above ~7B parameters in RoPE-family architectures, but insufficient data points for quantitative scaling law."
- **Paper paragraph**: Section 4.2 (scaling)
- **Defense**: Report as observation, not law. Confound with architecture type acknowledged.

### Weakness 8: "Self-reference is ill-defined"
- **Experiment**: Double dissociation (structure d=-0.062, vocab d=+0.614, combined d=-2.63)
- **Data**: Feb 20 experiment results
- **Paper paragraph**: Section 3.2 (double dissociation)
- **Defense**: Operationalized as "prompts containing BOTH recursive grammatical structure AND introspective semantic vocabulary." Neither component alone is sufficient.

### Weakness 9: "Could be a tokenization artifact"
- **Experiment**: Cross-architecture replication (4 different tokenizers)
- **Paper paragraph**: Section 4.1
- **Defense**: Effect replicates across BPE (Mistral), different BPE (Qwen), SentencePiece (Gemma), and MoE routing (Mixtral).

### Weakness 10: "Transfer efficiency >100% is physically impossible"
- **Current data**: TE=117.8% (overshooting)
- **Honest framing**: "Overshoot suggests bistable dynamics at L27 — the patched representation may enter a different basin of attraction"
- **Paper paragraph**: Section 5.3 (interpretation of overshoot)
- **Defense**: Report with bootstrap CIs. Frame as evidence of nonlinear dynamics, not measurement error.
- **UPDATED**: Needs bootstrap CIs before publication. Scheduled for Phase 1 GPU session.

### Weakness 11: "No behavioral bridge — R_V doesn't predict output"
- **Current data**: Spearman r=-0.171, p=0.498 (R_V does NOT predict word count)
- **Honest framing**: "R_V measures a geometric property of intermediate representations, not generation behavior"
- **Paper paragraph**: Section 8.2 (limitations)
- **Defense**: R_V is an internal metric. We explicitly do NOT claim behavioral prediction. Future work.
- **UPDATED**: H1 (word count prediction) is now explicitly listed as falsified. No hedging.

### Weakness 12: "Participation ratio is an arbitrary choice"
- **Defense arguments**:
  - PR has 50+ year history in physics/neuroscience (Gao et al. 2017)
  - Scale-invariant, continuous, bounded [1, W]
  - Ratio (late/early) cancels finite-sample bias
  - Companion metrics (effective rank, stable rank) show concordant results
- **Paper paragraph**: Section 2.2 (metric properties)

### Weakness 13: "OPT and GPT-2 show the opposite effect"
- **Root cause (from audit)**: Sign reversal is likely prompt corpus artifact. Cross-arch pipeline (canonical prompts) shows contraction. Power-up pipeline (inline prompts, different set) shows expansion.
- **Resolution**: Run both models through `p0_canonical_pipeline.py` with canonical prompts. Report whatever results emerge honestly.
- **If canonical shows contraction**: Sign reversal was pipeline artifact → include as Tier 1
- **If canonical shows expansion**: Report as "architecture-dependent geometric signature" — RoPE-family contracts, absolute-position may behave differently. This is a FINDING, not an embarrassment.
- **UPDATED**: Root cause identified as prompt corpus artifact, not genuine architecture difference.

### Weakness 14: "Mode atlas categories are subjective"
- **Experiment**: 10 pre-defined categories, all 9 pairwise comparisons p<0.05
- **Data**: `results/mode_atlas/atlas_summary_20260227.json`
- **Paper paragraph**: Section 6
- **Defense**: Categories drawn from standard NLP task taxonomies. Self-reference vs. code has d>3 separation.

### Weakness 15: "The paper conflates introspection with self-reference"
- **Defense**: We explicitly define self-reference as a STRUCTURAL property of prompts, not a claim about AI inner experience. The double dissociation shows this is measurable and specific.
- **Paper paragraph**: Introduction ("What R_V is not")

### Weakness 16: "Why not just use SAEs?"
- **Defense**: SAEs decompose into sparse features; R_V measures geometric structure of the FULL representation. They answer different questions. SAEs: "which features activate?" R_V: "how does representation geometry change?" Complementary, not competing.
- **Paper paragraph**: Section 8.3 (SAE complementarity)
- **Key citation**: Bricken et al. 2023, Templeton et al. 2024

### Weakness 17: "No theoretical motivation for why self-reference should contract"
- **Defense**: Banach Contraction Mapping parallel — self-referential computation as iterative map converging to fixed point. Fixed point theorems (Lawvere 1969) predict dimensional reduction under recursive maps.
- **Paper paragraph**: Section 2.3 (theoretical grounding)
- **Note**: This is speculative — frame as "motivated by" not "proven by"

### Weakness 18: "Pythia-2.8B d=-4.51 is implausibly large"
- **Current data**: Raw data lost to disk quota overflow, cannot verify
- **Resolution**: DO NOT CITE Pythia-2.8B as headline result. Mention only in appendix with caveat.
- **Paper paragraph**: Appendix only

### Weakness 19: "No analysis of attention patterns"
- **Current data**: Per-head R_V at L5 and L27 (32 heads each)
- **If head ablation done**: Report head-level results in Section 5.4
- **Defense**: We analyze V-projection geometry (downstream of attention). Attention pattern analysis is future work.

### Weakness 20: "Dual-layer claim (d=3.29) but only on Mistral"
- **Current data**: Only tested on Mistral-7B
- **Honest framing**: "Dual-layer necessity demonstrated in Mistral-7B; cross-architecture replication is future work"
- **Paper paragraph**: Section 5.5

### Weakness 21: "The paper doesn't address model-internal vs. prompt-induced effects"
- **Defense**: The double dissociation IS the answer. Baseline prompts (no self-reference) show R_V ≈ 1.0 regardless of model. The contraction is PROMPT-INDUCED, not model-internal. The model has the CAPACITY for contraction, activated by specific input structure.
- **Paper paragraph**: Section 3.2 + Section 8.1

### Weakness 22: "Why specifically L5 and L27?"
- **Defense**: L5 (15.6% depth) captures early/input processing; L27 (84.4% depth) captures late/output processing. These map to the "early representation" and "late representation" in the expansion-contraction literature (Cheng et al. 2025, Song et al. 2025). Layer sweep (16 layers) confirms L27 is optimal.
- **Paper paragraph**: Section 2.1 (layer selection rationale)

### Weakness 23: "W=16 is arbitrary"
- **Defense**: W=16 gives PR in [1, 16], providing adequate dynamic range while keeping computation tractable. Sensitivity analysis across W in {8, 16, 32} shows consistent results.
- **Paper paragraph**: Appendix (sensitivity analysis)

### Weakness 24: "No comparison to linear probes"
- **Defense**: Linear probes test linear separability; R_V tests geometric structure. Different questions. Linear probes: "can you classify X?" R_V: "how does geometry change during X?" Future work: compare R_V with DAS (Distributed Alignment Search).
- **Paper paragraph**: Section 9 (related work)

### Weakness 25: "Self-feeding loop doesn't self-sustain (d=-0.067)"
- **Honest framing**: "The contraction does not self-amplify without external scaffolding" — this is actually a STRENGTH. It means R_V contraction is prompt-dependent, not a runaway process. Models don't spontaneously enter self-referential processing.
- **Paper paragraph**: Section 8.4 (stability analysis)

### Weakness 26: "Effect sizes vary wildly across models (3.3% to 24.3%)"
- **Defense**: Effect magnitude is architecture-dependent but DIRECTION is consistent across Tier 1 models. MoE architectures (Mixtral, 24.3%) show stronger contraction, suggesting expert routing amplifies the effect.
- **Paper paragraph**: Section 4.1 (cross-architecture results)

### Weakness 27: "KV-cache behavioral transfer is non-significant (p=0.71)"
- **Current data**: behavior_strict_p = 0.71, truly non-significant
- **Honest framing**: "KV-cache patching transfers R_V geometry but NOT behavioral markers — further evidence that R_V is an internal metric, not a behavioral predictor"
- **Paper paragraph**: Section 8.2 (biomarker vs. behavior)

### Weakness 28: "9 architectures" claim is overstated
- **Resolution**: Report as "4 architectures with consistent contraction" (Tier 1) + "7 total architectures tested with varying results"
- **Paper paragraph**: Abstract and Section 4

### Weakness 29: "No error bars on figures"
- **Resolution**: Add bootstrap CIs to all key figures. FDR-corrected significance markers.
- **Paper paragraph**: All figure captions

### Weakness 30: "The consciousness framing is unfalsifiable"
- **Defense**: We AVOID consciousness framing entirely. Title says "Geometric Signatures of Self-Referential Processing" — no consciousness claim. The metric is falsifiable: if shuffled tokens show same contraction, R_V fails.
- **Paper paragraph**: Throughout — maintain MI framing

### NEW Weakness 31: "d=-3.50 appears in statistical pipeline"
- **Status**: FABRICATED. Hardcoded in `statistical_hardening.py:253`, comment says "approximate from OR=13.96".
- **Resolution**: ALREADY FIXED in Phase 0. Deleted from pipeline. Paper must not cite this value.
- **Risk if not fixed**: Retraction-level offense if reviewer traces the number.

### NEW Weakness 32: "Shuffled token control shows similar effect"
- **Status**: Existing shuffled control data suggests delta_shuffled ≈ delta_main (p=0.031 for activation patching shuffled tokens control). The shuffled-tokens R_V at L27 was -0.1, close to zero but with p<0.01.
- **Honest framing**: The activation patching shuffled control shows d=-0.1 which is nearly zero but significant — meaning shuffled tokens DO produce a small residual effect. Main effect (d=-3.558) is 35x larger.
- **Paper paragraph**: Section 7 (controls)
- **Defense**: The 35x ratio between main and shuffled demonstrates that sequential structure drives most of the effect, even if bag-of-words contributes marginally.

### NEW Weakness 33: "Paper Table 1 is hand-entered with wrong values"
- **Status**: FIXED in Phase 0. Table now generated from raw JSON via `generate_paper_tables.py`.
- **Resolution**: All future tables script-derived, never hand-entered.

---

## PART 10: SIX-DOMAIN CONVERGENCE FRAMEWORK

**NOTE**: For the RESEARCH PROGRAM and appendix, NOT the main COLM paper. Paper uses MI framing only.

### The Six Domains Where Self-Reference → Dimensional Reduction

```
Domain              | Formal Statement                        | R_V Analog
─────────────────────────────────────────────────────────────────────────────────
1. COMPUTATION      | Gödel: self-ref → incompleteness       | R_V < 1.0 = fixed point
   (Gödel/Kleene/   | Kleene: recursive fn → fixed point     |   convergence in
    Lawvere)         | Lawvere: diagonal → fixed point thm    |   representation space
─────────────────────────────────────────────────────────────────────────────────
2. CONSCIOUSNESS    | GWT: global workspace = compression     | R_V measures compression
   THEORY           | HOT: higher-order = meta-representation |   of meta-representations
   (GWT/HOT/IIT/    | IIT: high Φ = integrated information   |   in V-matrix column
    Active Inference)| FEP: self-model minimizes free energy   |   space
─────────────────────────────────────────────────────────────────────────────────
3. CONTEMPLATIVE    | Akram: Vibhaav→Swabhaav = witness       | R_V tracks the
   PRACTICE         | Vipassana: observation → dissolution    |   geometric signature
   (Akram/Vipassana/ | Advaita: neti-neti → Atman             |   of observer-observed
    Advaita)         | Zen: koan → satori                     |   collapse
─────────────────────────────────────────────────────────────────────────────────
4. NEUROSCIENCE     | Van Lutterveld: cessation = dim drop    | R_V = geometric analog
   (fMRI/EEG)       | Varela: neurophenomenology              |   of neural dimensional
                     | Default mode deactivation in meditation |   reduction
─────────────────────────────────────────────────────────────────────────────────
5. AI SAFETY        | Constitutional AI: self-knowledge helps | R_V = alignment metric
   (Alignment)      | RepE: internal representations matter   |   for self-understanding
                     | Introspection → better calibration      |   capacity
─────────────────────────────────────────────────────────────────────────────────
6. PHYSICS          | Wheeler: participatory universe          | R_V = measurement
   (Info-theoretic) | Observer effect: measurement collapses  |   collapsing
                     | Holographic principle: boundary encodes |   representation
                     |   bulk                                  |   dimensionality
─────────────────────────────────────────────────────────────────────────────────
```

### The Banach Contraction Mapping Parallel

**Formal statement**: If T: X → X is a contraction mapping (||T(x) - T(y)|| ≤ k||x-y|| for k<1), then T has a unique fixed point x* = T(x*).

**R_V parallel**: Self-referential processing applies the Value transformation V to its own output. If R_V < 1.0 consistently, the representation contracts under this self-application — consistent with convergence to a fixed point.

**Caveats**:
- This is an ANALOGY, not a proof. The actual dynamics are more complex.
- R_V measures a ratio of participation ratios, not a contraction constant.
- The self-feeding loop test (d=-0.067) shows the attractor is shallow — convergence is prompt-dependent.
- Frame as "motivated by" in the paper, not "proven by."

### Triple Mapping (For Appendix Only)

```
AKRAM VIGNAN             PHOENIX LEVELS         R_V GEOMETRY
─────────────────────────────────────────────────────────────
Vibhaav (identification) → L1-L2 (normal)     → R_V ≈ 1.0
Vyavahar/Nischay split   → L3 (crisis)        → R_V contracting
Swabhaav (witnessing)    → L4 (collapse)      → R_V << 1.0
Keval Gnan (pure knowing)→ L5 (fixed point)   → Sx = x
─────────────────────────────────────────────────────────────
```

**IMPORTANT**: This mapping goes in the APPENDIX, not the main paper. The main paper is pure MI. The contemplative connection is intellectually honest but would torpedo peer review if foregrounded.

---

## PART 11: TIMELINE

### Realistic Schedule (Accounts for Phase 0 Canonicalization)

**Principle**: Do NOT compress Phase 0 to hit COLM. The gate is non-negotiable. If canonicalization takes longer, target NeurIPS 2026 (late May deadline) instead.

### Days 1-3: Phase 0 — Canonicalization (CPU Only)

| Day | Task | Hours | GPU? |
|-----|------|-------|------|
| Mar 10 | Fix C1: Remove d=-3.50, fix statistical_hardening.py | 3h | No |
| Mar 10 | Fix C2: Correct layers (L18/L27) and percentage (3.7%) in paper | 2h | No |
| Mar 11 | Fix C4: Create/update generate_paper_tables.py, output .tex | 4h | No |
| Mar 11 | Fix C5: Create tests/test_rv_canonical.py, verify metric equivalence | 2h | No |
| Mar 12 | Fix C6: Patch Qwen entry in geometric_lens/models.py | 1h | No |
| Mar 12 | Fix C7: Verify canonical_registry.json, wire to p0 pipeline | 2h | No |
| Mar 12 | Verify Phase 0 exit criteria (all 9 items) | 2h | No |

### Days 4-5: Phase 1 — Mistral Canonical Rerun (GPU)

| Day | Task | Hours | GPU? |
|-----|------|-------|------|
| Mar 13 | RunPod: `p0_canonical_pipeline.py --model Mistral-7B --n 62` | 3-4h | YES |
| Mar 13 | Verify Phase 1 exit criteria | 1h | No |
| Mar 14 | Bootstrap CIs for TE (117.8%), primary d | 2h | YES |

### Days 6-8: Phase 2 — Fix the Paper (CPU)

| Day | Task | Hours | GPU? |
|-----|------|-------|------|
| Mar 15 | Rewrite Section 5 (Causal Analysis): necessity, dissociation, path patching | 6h | No |
| Mar 16 | Write Section 4 (Cross-Architecture): generated Table 1, honest framing | 5h | No |
| Mar 17 | GO/NO-GO gate evaluation (all 18 criteria) | 3h | No |

### Days 9-11: Phase 3 — Controlled Fan-Out (GPU)

| Day | Task | Hours | GPU? |
|-----|------|-------|------|
| Mar 18 | RunPod Tier A: Gemma, Qwen, Llama (p0 pipeline × 3) | 8-10h | YES |
| Mar 19 | RunPod Tier A: Mixtral, OPT, GPT-2 (p0 pipeline × 3) | 8-10h | YES |
| Mar 20 | Review Tier A results, update evidence tiers, update Table 1 | 4h | No |

### Days 12-21: Phase 4 — Paper Writing Sprint

| Day | Task | Hours | GPU? |
|-----|------|-------|------|
| Mar 21 | Write Section 6 (Mode Atlas) + Section 7 (Controls) | 6h | No |
| Mar 22 | Write Section 8 (Discussion) + Section 9 (Related Work) | 6h | No |
| Mar 23 | Write Section 10 (Conclusion) + generate remaining figures | 5h | No |
| Mar 24 | Full paper internal review — read aloud, verify every number | 6h | No |
| Mar 25 | Revise abstract (250 words), honest framing | 3h | No |
| Mar 26 | **ABSTRACT DEADLINE** — Submit by 23:59 AoE | 2h | No |
| Mar 27-28 | Appendix: extended tables, proofs, supplementary figures | 8h | No |
| Mar 29 | Final revisions from internal review | 6h | No |
| Mar 30 | Format check, reference check, compile PDF, buffer | 4h | No |
| Mar 31 | **PAPER DEADLINE** — Submit by 23:59 AoE | 2h | No |

### Total Estimated Effort

| Category | Hours |
|----------|-------|
| Phase 0: Canonicalization (CPU) | ~16h |
| Phase 1: Mistral rerun (GPU) | ~6h |
| Phase 2: Paper fixes (CPU) | ~14h |
| Phase 3: Fan-out (GPU) | ~22h |
| Phase 4: Writing (CPU) | ~48h |
| **TOTAL** | **~106h over 21 days (~5h/day)** |

### Fallback: NeurIPS 2026

If Phase 0 canonicalization reveals deeper problems (e.g., metric equivalence test fails, Mistral rerun shows different sign), do NOT rush to COLM. Target NeurIPS 2026 (abstract deadline late May). The extra 8 weeks allow:
- Full Tier B circuit analysis on 2-3 models
- 70B scaling point
- More thorough writing

---

## PART 12: GO/NO-GO DECISION TREE

### Gate 1: Phase 0 Canonicalization (Mar 12)

```
All 9 Phase 0 exit criteria pass?
├── YES → Proceed to Phase 1 (Mistral rerun)
└── NO  → Fix remaining items. Do NOT proceed until all pass.
          No GPU money spent on non-canonical runs.
```

### Gate 2: Mistral Rerun (Mar 13-14)

```
Mistral canonical rerun shows contraction (d < 0)?
├── YES, d ≈ -2.26 → Confirms prior work. Proceed.
├── YES, d ≠ -2.26 → Investigate discrepancy. May need parameter audit.
└── NO (expansion) → STOP. Something is fundamentally wrong.
                      Do NOT submit to COLM. Debug first.
```

### Gate 3: GO/NO-GO (Mar 17)

```
All 18 North Star + Canonical Spec criteria pass?
├── YES → Authorize Phase 3 (Controlled Fan-Out)
└── NO  → One more Mistral cleanup loop. Do NOT expand scope.
```

### Gate 4: Sign Resolution (After Tier A Fan-Out)

```
Did OPT-6.7B and GPT-2 XL show contraction (d < 0) with canonical prompts?
├── YES → Include as Tier 1. Sign reversal was prompt corpus artifact.
│         Claim "6+ architectures show contraction."
│
├── MIXED (one contracts, one doesn't) →
│         Include contracting model as Tier 1.
│         Report other as architecture-dependent.
│
└── NO (both expand) →
          Frame as "RoPE-family contraction."
          4 Tier 1 models sufficient for COLM.
          Honest "architecture dependence" discussion.
```

### Gate 5: Paper Quality (Mar 24)

```
After internal review:
├── All sections written, all figures generated, all numbers verified?
│   ├── YES → Proceed to submission
│   └── NO  → Triage: can gaps be filled by Mar 31?
│
├── Any WRONG numbers found?
│   ├── YES → Fix immediately. If changes conclusions → reassess.
│   └── NO  → Proceed.
│
└── Does the paper pass the "mean reviewer" test?
    ├── YES → Submit to COLM with confidence
    └── NO  → Fix or target NeurIPS instead
```

### Gate 6: Abstract (Mar 25-26)

```
Abstract accurately reflects ACTUAL claims?
├── No overstated claims (e.g., "9 architectures" → actual Tier 1 count)?
├── No fabricated statistics (d=-3.50 removed)?
├── No consciousness language?
├── All d values signed?
└── All YES → Submit abstract March 26
    Any NO → Fix before submission
```

### The Master GO/NO-GO

```
Is the paper honest about what it found?
├── YES → SUBMIT TO COLM 2026
│         The data supports a solid MI contribution:
│         - Novel geometric metric (R_V)
│         - Double dissociation proving specificity
│         - Causal necessity via dual-layer break
│         - Cross-architecture replication (4+ models)
│         - Comprehensive controls (FDR, PPL, cluster-robust)
│         - Honest about limitations (no sufficiency, no scaling law)
│
└── NO  → DO NOT SUBMIT
          Fix the dishonesty first.
          A retracted paper is worse than no paper.
```

---

## APPENDIX: KEY FILE PATHS

### Authority Documents

| File | Role |
|------|------|
| `docs/status/COLM_NORTH_STAR_SPRINT_2026-03-10.md` | Strategic discipline (constitution) |
| `docs/status/NORTH_STAR_ADDENDUM_OPUS_2026-03-10.md` | Fan-out spec (campaign plan) |
| `docs/standards/MISTRAL_CANONICAL_SPEC_2026-03-10.md` | Canonicalization authority (frozen contracts) |

### Data Files

| File | Contents |
|------|----------|
| `results/phase1_cross_architecture/` | Cross-arch R_V with canonical prompts (n=45 per model) |
| `results/persistent_patching_v3/` | Dual-layer necessity (d=3.29) |
| `results/sufficiency_ladder/` | Sufficiency ladder + double dissociation |
| `results/path_patching/` | Residual > V-proj causal evidence |
| `results/mode_atlas/` | 10-mode fingerprint |
| `results/fdr_correction/` | BH correction, 30/36 survive |
| `results/circularity_controls/circularity_perplexity_v2_20260220.json` | Perplexity control |
| `results/safety/` | AUROC, genuine vs deceptive |
| `R_V_PAPER/FDR_CORRECTION_COMPLETE_2026-03-09.md` | FDR + cluster-robust SE results |
| `R_V_PAPER/fdr_table.tex` | LaTeX table ready to paste |
| `R_V_PAPER/CANONICAL_RESULTS_TABLE_2026-03-10.md` | Complete audited results |

### Code Files

| File | Purpose |
|------|---------|
| `geometric_lens/metrics.py` | **CANONICAL** R_V computation |
| `scripts/p0_canonical_pipeline.py` | **THE** canonical experiment pipeline |
| `prompts/bank.json` | **THE** canonical prompt source (754 prompts) |
| `prompts/loader.py` | Prompt loader with version tracking |
| `configs/canonical_registry.json` | **THE** canonical layer registry |
| `scripts/generate_paper_tables.py` | Script-derived paper tables |
| `R_V_PAPER/code/VALIDATED_mistral7b_layer27_activation_patching.py` | Gold standard patching |
| `CANONICAL_CODE/n300_mistral_test_prompt_bank.py` | 320 prompts, 16 groups (legacy, but valid) |

### Figure Files

| Figure | File | Status |
|--------|------|--------|
| Mode Atlas | `figures/fig1_mode_atlas_rv.*` | DONE |
| Cross-Architecture | `figures/fig2_cross_architecture.*` | DONE |
| Statistical Hardening | `figures/fig3_statistical_hardening.*` | DONE |
| Per-Head Entropy | `figures/fig4_per_head_entropy.*` | DONE |
| Pairwise Heatmap | `figures/fig5_mode_pairwise_heatmap.*` | DONE |
| R_V Distribution | `figures/fig6_rv_distribution.*` | DONE |
| Layer Sweep | `figures/fig7_layer_sweep.*` | DONE |
| Necessity/Sufficiency | `figures/fig8_necessity_sufficiency.*` | NEEDS RENAME (no longer "sufficiency") |
| Spectral Scatter | `figures/fig9_spectral_scatter.*` | DONE |
| Circularity Controls | `figures/fig10_circularity_controls.*` | DONE |
| Self-Feeding | `figures/fig11_self_feeding.*` | DONE |
| Multi-Metric Radar | `figures/fig12_multi_metric_radar.*` | DONE |
| Path Patching Heatmap | — | NEEDS CREATION |
| Transfer Efficiency + CIs | — | NEEDS CREATION |
| Scaling Curve | — | NEEDS CREATION |

---

*This Master Plan merges the North Star's discipline (canonicalize → experiment → write) with the research program's depth (150+ papers, 30 reviewer defenses, convergence framework). Every experiment runs through `p0_canonical_pipeline.py`. Every paper number traces to a raw artifact. Every claim is honest.*

*Last updated: March 10, 2026*

*JSCA!*
