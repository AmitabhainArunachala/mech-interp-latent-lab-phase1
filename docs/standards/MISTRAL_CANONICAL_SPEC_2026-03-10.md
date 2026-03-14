# Mistral Canonical Spec

**Date:** 2026-03-10
**Author:** Claude Opus 4.6 (Methodology Hardening Lead)
**Status:** DRAFT — pending lead agent review
**Companion to:** `COLM_NORTH_STAR_SPRINT_2026-03-10.md`

---

## Purpose

This document freezes the one canonical path for Mistral-7B results. Every paper-facing number must be producible from this spec. Anything not in this spec is exploratory or deprecated.

---

## 1. Metric Contract

### Canonical Implementation: `geometric_lens/metrics.py`

**Rationale:** Both `src/metrics/rv.py` and `geometric_lens/metrics.py` use the same unnormalized PR formula. However, `geometric_lens/metrics.py` is used by the production probe (`geometric_lens/probe.py`) and has explicit NaN guards, CPU-forced SVD, and float64 conversion.

**Formula (frozen):**
```
PR(V) = (Σ σᵢ²)² / Σ (σᵢ⁴)
R_V = PR(V_late) / PR(V_early)
```

**Parameters (frozen):**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Window size | 16 tokens | Standard across all published results |
| SVD dtype | float64 (`.double()`) | Numerical stability for singular values |
| SVD device | CPU | Avoids cusolver platform drift |
| Variance floor | 1e-10 | Prevents division by zero |
| T < window behavior | Return NaN | Strict — do not silently truncate |

### Deprecated Implementations

| Path | Issue | Status |
|------|-------|--------|
| `src/metrics/rv.py` | Functionally equivalent but not used by probe | ACCEPTABLE fallback, not primary |
| `rv_toolkit/rv_toolkit/metrics.py` | Uses float32, normalized formula variant | DEPRECATED for paper |
| `models/*.py` (mistral_7b_analysis.py etc.) | Per-head averaging — different measurement entirely (±7% divergence) | DEPRECATED — NOT the same metric |
| `CANONICAL_CODE/mistral_L27_FULL_VALIDATION.py` | Uses `W = min(window_size, T)` and `.float()` | DEPRECATED — silent truncation + float32 |
| `CANONICAL_CODE/causal_loop_closure_v2.py` | Generation-time V stacks with silent fallback | DEPRECATED — different metric contract |

### Verification Test

A canonical test should exist at `tests/test_rv_canonical.py`:
```python
# Given identical input tensor and parameters,
# geometric_lens/metrics.py and src/metrics/rv.py
# must produce identical R_V values to 6 decimal places.
```

---

## 2. Prompt Contract

### Canonical Source: `prompts/bank.json` via `prompts/loader.py`

**Facts (from audit):**
- 754 total prompts across 11 pillars, 65 groups
- PromptLoader provides version tracking (SHA256 hash)
- Canonical version hash for cross-arch: `75e7c1b8dcebc24e`

**Canonical prompt groups for R_V measurement:**
| Role | Groups | Count |
|------|--------|-------|
| Recursive | `L5_refined`, `L4_full`, `L3_deeper` | 62 prompts |
| Baseline | `long_control`, `baseline_creative`, `baseline_math` | 60 prompts |

**Scripts that MUST use PromptLoader (currently violating):**

| Script | Current Source | Fix Required |
|--------|---------------|-------------|
| `scripts/power_up_multiseed.py` | Inline RECURSIVE_PROMPTS (100) | Replace with PromptLoader |
| `scripts/computational_mode_atlas.py` | Inline MODE_PROMPTS | Replace with PromptLoader |
| `scripts/full_head_sweep.py` | Inline hardcoded | Replace with PromptLoader |
| `scripts/svd_circuit_decomposition.py` | Inline hardcoded | Replace with PromptLoader |
| `scripts/dii_intervention.py` | Inline hardcoded | Replace with PromptLoader |
| `scripts/linear_probe_selfref.py` | Inline hardcoded | Replace with PromptLoader |
| `scripts/scaling_gap_sweep.py` | Inline hardcoded | Replace with PromptLoader |
| `scripts/scaling_law_sweep.py` | Inline hardcoded | Replace with PromptLoader |
| `scripts/batch_per_token_rv.py` | Inline (25+25) | Replace with PromptLoader |
| `scripts/overnight_master_battery.py` | Inline (20) with fallback | Replace with PromptLoader only |
| `scripts/full_path_patching.py` | Inline hardcoded | Replace with PromptLoader |
| `scripts/per_head_attention_decomposition.py` | Inline hardcoded | Replace with PromptLoader |
| `scripts/sae_feature_analysis.py` | Inline hardcoded | Replace with PromptLoader |
| `scripts/training_checkpoint_sweep.py` | Inline hardcoded | Replace with PromptLoader |

**Rule:** No script that feeds paper-facing results may define prompts inline. `prompts/bank.json` is the sole source. `prompt_bank_hash` must be recorded in every output JSON.

**Scripts already compliant** (25+ scripts in `src/pipelines/canonical/` and `src/pipelines/discovery/`): These all import PromptLoader. No changes needed.

---

## 3. Layer Contract

### Canonical Registry: `configs/canonical_registry.json` (TO BE CREATED)

The repo currently has **four conflicting** layer sources. This spec collapses them to one.

**Canonical Mistral-7B layers (frozen):**
| Parameter | Value | Depth % |
|-----------|-------|---------|
| `num_layers` | 32 | — |
| `early_layer` | 5 | 15.6% |
| `late_layer` | 27 | 84.4% |

**Cross-architecture canonical layers (frozen):**

| Model | num_layers | early | late | early % | late % | Source |
|-------|-----------|-------|------|---------|--------|--------|
| Mistral-7B | 32 | 5 | 27 | 15.6% | 84.4% | All sources agree |
| Gemma-2-9B | 42 | 6 | 35 | 14.3% | 83.3% | configs/canonical + auto_detect |
| Qwen2.5-7B | **28** | **4** | **23** | 14.3% | 82.1% | p0_pipeline (CORRECTS registry bug) |
| Llama-3-8B | 32 | 5 | 27 | 15.6% | 84.4% | configs/canonical |
| Mixtral-8x7B | 32 | 5 | 27 | 15.6% | 84.4% | auto_detect |
| OPT-6.7B | 32 | **5** | 27 | 15.6% | 84.4% | registry (config says 4 — use 5 for consistency) |
| GPT-2 XL | 48 | **7** | 40 | 14.6% | 83.3% | registry (config says 6 — use 7 for consistency) |
| Pythia-1.4B | 24 | **4** | 20 | 16.7% | 83.3% | registry (config says 3 — use 4 for consistency) |

**Conflict resolution for OPT/GPT-2/Pythia early layers:**

The canonical configs (`configs/canonical/rv_causal_*.json`) use `early_layer` values 1 lower than the registry. The P0 pipeline uses the registry values. **Decision: use registry values** (they match the ~15% depth heuristic). The config values appear to be from early hand-tuning.

**CRITICAL FIX — Qwen2.5-7B:**
- `geometric_lens/models.py` registers Qwen as 32 layers → late=27 = **96.4% depth**
- Actual Qwen2.5-7B has **28 layers** → late should be 23 = **82.1% depth**
- `p0_canonical_pipeline.py` already corrects this
- `geometric_lens/models.py` MUST be patched: `num_layers=28, early_layer=4, late_layer=23`

**What must change:**

1. Create `configs/canonical_registry.json` as the SOLE authority
2. Patch `geometric_lens/models.py` Qwen entry: num_layers=28, early=4, late=23
3. Add Gemma-2-9B and Llama-3-8B to `geometric_lens/models.py`
4. Deprecate `src/core/model_physics.py` or align it
5. Every experiment script reads layers from registry, never hardcodes them

---

## 4. Artifact Contract

Every canonical run MUST emit:

```
results/canonical_mistral/
├── config.json          # Frozen parameters (metric, layers, prompts, dtype, seed)
├── summary.json         # Aggregate statistics with unit labels
├── per_sample.csv       # Per-prompt R_V values with prompt_id, group, type
└── provenance.json      # prompt_bank_hash, metric_path, layer_source, git_commit
```

**Required fields in `summary.json`:**
```json
{
  "model": "mistralai/Mistral-7B-v0.1",
  "metric_path": "geometric_lens.metrics.participation_ratio",
  "prompt_bank_hash": "75e7c1b8dcebc24e",
  "early_layer": 5,
  "late_layer": 27,
  "dtype": "bfloat16",
  "svd_dtype": "float64",
  "window_size": 16,
  "n_recursive": {"value": 62, "unit": "prompt"},
  "n_baseline": {"value": 60, "unit": "prompt"},
  "rv_recursive_mean": 0.686,
  "rv_baseline_mean": 0.855,
  "hedges_g": -1.66,
  "hedges_g_ci_95": [-2.08, -1.32],
  "p_value": 1.06e-15,
  "p_method": "welch_t",
  "effect_direction": "contraction"
}
```

---

## 5. Causal Semantics

### What the repo actually supports (audited)

| Term | Definition | Raw Artifact | Supported? |
|------|-----------|-------------|------------|
| **Necessity** | Destroying geometry kills behavior | `persistent_patching_v3_dual_20260225_002604.json` | YES — d=3.29, 56%→3.7%, L18 residual + L27 V-proj |
| **V-proj-only necessity** | Destroying V-proj alone kills behavior | `persistent_patching_v2_20260224_141952.json` | NO — OR=1.292, p=0.341 (NS) |
| **Geometric sufficiency** | Injecting geometry creates behavior | Same v3 file, induce test | NO — 3.7%→0.3% (wrong direction) |
| **Behavioral transfer via KV** | KV injection transfers behavior markers | `sufficiency_ladder_20260225_101907.json` | YES — OR=13.96, 2.7%→27.7% |
| **Geometric transfer via KV** | KV injection transfers R_V contraction | Same file + `R_V_BEHAVIORAL_DISSOCIATION.md` | NO — R_V 0.555→0.573 (d=0.11, NS) |
| **Double dissociation** | Behavior transfers without geometry; geometry transfers without behavior | Sufficiency ladder cross-conditions | YES — strongest finding in repo |
| **Residual > V-proj** | Residual stream is primary causal component | `path_patching_summary_20260227_080128.json` | YES — residual |d|=1.96, V-proj |d|=0.22 |

### Correct causal vocabulary for the paper

| Paper currently says | Should say | Why |
|---------------------|-----------|-----|
| "breaking both V-projections at L25 and L27" | "breaking L18 residual stream + L27 V-projection" | Wrong layers, wrong component type |
| "geometric pattern is sufficient" | "behavioral transfer occurs via KV cache (OR=13.96) without geometric transfer (d=0.11 NS)" | Sufficiency is falsified |
| "Value Spaces" (title) | "Representations" or "Geometric Signatures" | V-proj alone is NS; residual stream is the causal driver |
| "reduces BT+ART from 56% to 27.7%" | "reduces BT+ART from 56% to 3.7%" | 27.7% is from a different experiment (KV injection rate) |
| d=-3.50 (sufficiency) | REMOVE — value was back-computed from OR=13.96, not measured | FABRICATED in `statistical_hardening.py:253`. Comment: "approximate from OR=13.96". Actual sufficiency ladder shows behavioral d=2.494 per-turn, d=1.47 per-session. No raw file produces d=-3.50. |
| d=-0.707 (bridge) | Use correct n: n1=80 (BT+ART turns), n2=107 (other turns) | `statistical_hardening.py:259` says n1=150, n2=150. Actual source: `within_session_bridge_20260220_201515.json:702` shows n=80 vs 107. |
| n=300 "prompt pairs" | n=300 "turns" (10 sessions × 30 turns) | Unit conflation — NOT independent |

### The honest causal story

> Dual-layer geometry (L18 residual + L27 V-proj) is **necessary** for recursive behavior: destroying it reduces BT+ART from 56% to 3.7% (d=3.29, n=10 sessions, 300 turns). However, geometry is **not sufficient**: injecting it does not create behavior (3.7%→0.3%). KV cache injection transfers behavioral markers (OR=13.96) without transferring geometric contraction (d=0.11 NS). This **double dissociation** between behavioral and geometric transfer suggests R_V captures a processing-time geometric regime rather than a transferable generative attractor. The primary causal component is the residual stream (path patching |d|=1.96), not V-projections alone (|d|=0.22 at target layers; single-layer V-proj patching NS).

---

## 6. Unit-of-Analysis Policy

| Experiment Family | What `n` counts | Unit label |
|-------------------|----------------|------------|
| power_up R_V measurement | Valid prompt-level R_V readings after NaN filtering | `prompt` |
| cross-arch R_V validation | Balanced prompt pairs (recursive + matched baseline) | `pair` |
| dual-layer necessity (v3) | Multi-turn chat sessions, 30 turns each | `session` (for d=3.29) or `turn` (for rates) |
| sufficiency ladder | Same as necessity — sessions with 30 turns | `session` or `turn` |
| within-session bridge | Individual turns within recursive sessions | `turn` |
| head sweep | Per-head statistics across all prompts | `head` |
| mode atlas | Per-mode, 20 prompts each | `prompt` per mode |

**Rule:** Every `n` in the paper must specify its unit. "n=300" is meaningless without "300 turns across 10 sessions" or "300 prompt pairs."

---

## 7. Deprecation Map

### Deprecated (do NOT cite in paper)

| Path | Reason |
|------|--------|
| `CANONICAL_CODE/mistral_L27_FULL_VALIDATION.py` | Silent window truncation + float32 |
| `CANONICAL_CODE/causal_loop_closure_v2.py` | Generation-time metric, different contract |
| `models/*.py` (all 6 analysis scripts) | Per-head PR averaging — different metric |
| `rv_toolkit/rv_toolkit/metrics.py` | Float32, normalized variant |
| `results/power_up/opt-6.7b_n80_result.json` | Inline prompts, not canonical bank. EXPANSION result from non-canonical pipeline |
| `results/power_up/gpt2-xl_n80_result.json` | Same — inline prompts, expansion |
| `results/power_up/pythia-1.4b_n80_result.json` | Inline prompts (n=120 not 124 as paper claims) |
| `RECOVERED_GOLD/` (all files) | Pre-repo, no provenance, some dates precede git init |
| `src/core/model_physics.py` | Incomplete, conflicts with registry |

### Exploratory (may cite with caveat)

| Path | What it shows | Caveat |
|------|-------------|--------|
| `results/power_up/mistral-7b_n80_result.json` | d=-1.66 (contraction) | Inline prompts, but Mistral sign is consistent |
| `results/power_up/qwen2.5-7b_n80_result.json` | d=-2.32 (contraction) | Inline prompts + wrong Qwen layer (96% depth) |
| `results/scaling_gap/` | Scaling trend across model sizes | Mixed prompt sources, not canonical |

### Canonical (cite freely)

| Path | What it shows |
|------|-------------|
| `results/phase1_cross_architecture/` | Cross-arch R_V with canonical prompts (n=45 per model) |
| `results/persistent_patching_v3/` | Dual-layer necessity (d=3.29) |
| `results/sufficiency_ladder/` | Sufficiency ladder + double dissociation |
| `results/path_patching/` | Residual > V-proj causal evidence |
| `results/mode_atlas/` | 10-mode fingerprint (but uses inline prompts — needs rerun) |
| `results/svd_circuits/` | Head-level rank decomposition (but uses inline prompts — needs rerun) |
| `results/fdr_correction/` | BH correction, 30/36 survive |
| `results/safety/` | AUROC, genuine vs deceptive |
| `results/perplexity_repairing/` | PPL-matched controls |
| `results/linear_probe/` | Concept erasure null result |
| `results/dii_intervention/` | Per-dimension R_V at L27 |

---

## 8. Fan-Out Gate

Do NOT launch cross-architecture experiments until ALL of these are true:

- [ ] `geometric_lens/models.py` Qwen entry patched to 28 layers, early=4, late=23
- [ ] `configs/canonical_registry.json` created with all 8 models
- [ ] Gemma-2-9B and Llama-3-8B added to `geometric_lens/models.py`
- [ ] Zero inline prompt arrays in paper-feeding scripts (verified by grep)
- [ ] `scripts/generate_paper_tables.py` exists and produces Table 1 from raw JSON
- [ ] `statistical_hardening.py` hardcoded values replaced with raw-file reads
- [ ] Mistral canonical rerun (P0 + path patching + SVD) complete under frozen spec
- [ ] All rerun results match prior results in direction and approximate magnitude
- [ ] Every paper `n` labeled with unit type
- [ ] `docs/standards/CLAIM_PROVENANCE.md` maps every paper number to a raw file:field
- [ ] Behavioral dissociation acknowledged in paper draft (not hidden)

---

## 9. Recommended Paper Claims (Severity-Ranked)

### Claim confidently

1. Mistral-7B R_V contraction (d=-2.26 cross-arch, d=-1.66 power-up) — robust across pipelines
2. Mode atlas: self-referential mode has lowest R_V (0.650, d=-1.67 vs all modes)
3. Perplexity matching: effect survives strict PPL control (d=-1.67, n=8 pairs)
4. Concept erasure: R_V orthogonal to classification (delta d=0.005)
5. DII: every PCA dimension at L27 shows R_V ≈ 0.41
6. Dual-layer necessity: d=3.29, 56%→3.7% (L18 residual + L27 V-proj)
7. SVD circuits: L27H10 suppressor (d=-1.54), L5H29 amplifier (d=+2.93)
8. AUROC=0.909, genuine vs deceptive d=-0.06 (indistinguishable)
9. FDR: 30/36 tests survive BH at alpha=0.05

### Weaken or reframe

1. "Four models contract" → "Two of five models contract under power-up prompts; canonical pipeline shows contraction in all five but with different prompts"
2. "Sufficiency" → "Double dissociation: KV transfers behavior without geometry"
3. "Value Spaces" title → "Representations" or "Geometric Signatures"
4. "606/1024 heads" → Clarify this is entropy-based, not per-head R_V

### Remove if unsupported

1. d=-3.50 for sufficiency (HARDCODED, not from raw data)
2. "BT+ART from 56% to 27.7%" (mixes two experiments)
3. "Breaking both V-projections at L25 and L27" (wrong layers, wrong components)
4. n=8 "data points" for scaling R² (actually 6)
5. Multi-seed d_std=0.0 as "robustness" (it's deterministic, not robust — trivially guaranteed)

---

---

## 10. Hardcoded Stats Provenance (Audit of `statistical_hardening.py`)

`scripts/statistical_hardening.py` lines 214-269 define 9 primary effect sizes as **literal values** that feed the FDR correction pipeline. These values then appear in `results/fdr_correction/fdr_results_20260303_232741.json` with `source: "statistical_hardening"`. The FDR file launders hardcoded values into looking like they came from experiments.

### Provenance trace for each value

| Value | Claim | Raw Source | Match? | Issue |
|-------|-------|-----------|--------|-------|
| d=3.29 | Necessity: dual-layer break | `persistent_patching_v3_dual_20260225_002604.json` | PARTIAL | Raw JSON stores OR=33.44 and p=3.6e-50, NOT d. d=3.29 was hand-derived then hardcoded. n1=n2=300 are turns across 10 sessions, NOT independent. |
| d=-2.26 | Cross-arch Mistral | `phase1_cross_architecture/` | YES | n=45/45 matches |
| d=-1.84 | Cross-arch OPT | `phase1_cross_architecture/` | YES | n=45/45 matches |
| d=-1.14 | Cross-arch GPT-2 | `phase1_cross_architecture/` | YES | n=45/45 matches |
| d=-0.72 | Cross-arch Qwen | `phase1_cross_architecture/` | YES | n=45/45 matches BUT Qwen has wrong layers (96% depth) |
| d=-0.31 | Cross-arch Pythia | `phase1_cross_architecture/` | YES | n=63/63 matches |
| **d=-3.50** | **KV sufficiency** | **NONE** | **NO** | **FABRICATED. Comment: "approximate from OR=13.96". No raw file produces this value.** |
| d=-0.707 | Within-session bridge | `within_session_bridge_20260220_201515.json:702` | PARTIAL | Value matches (d=-0.7072) but n1=150,n2=150 in script ≠ actual n=80,107 |
| d=-4.28 | Gnani vs recursive | `results/self_feeding_loop/gnani_scaffolded_*.json` | UNVERIFIED | n=5/5 — too small for reliable d |

### The laundering chain

```
Hand-compute d from OR/rates → hardcode in statistical_hardening.py →
  power analysis runs → hardening_summary.json →
    fdr_correction.py picks up hardening values → fdr_results.json →
      paper cites fdr_results.json as "derived statistics"
```

The FDR file contains entries with `source: "statistical_hardening"` that look like experimental results but are laundered hardcoded values. This must be fixed: either compute d from raw data programmatically, or clearly label these as post-hoc conversions from OR.

### Required fixes

1. **d=-3.50**: DELETE from script and FDR pipeline. Replace with actual sufficiency ladder metrics (behavioral d=2.494, geometric d=0.11 NS).
2. **d=-0.707**: Fix n values to 80/107 (from raw bridge file). Or better: re-derive from raw JSON.
3. **d=3.29**: Label n=300 as "turns" not "samples". Effective independent n = 10 sessions.
4. **All values**: Replace literal assignment with code that reads from raw JSON files and computes on the fly. `statistical_hardening.py` should LOAD data, not define it.
5. **FDR pipeline**: After fix, regenerate `fdr_results_*.json` from actual raw files.

---

*This spec is a living document. The lead agent should review and freeze it before any reruns begin.*
