# Agent Sync Document - Dec 15 (updated)

## Current State Summary (from Claude Opus 4.5)

### Prompt Bank Status: `prompts/bank.json` - 694 prompts (single source of truth)

| Pillar | Count | Status |
|--------|-------|--------|
| dose_response | 102 | ✅ Canonical L1→L5 ladder |
| baselines | 105 | ✅ Complete (categories × 20 + instructional 5) |
| confounds | 60 | ✅ length / keyword / repetition controls |
| generality | 60 | ✅ zen / yogic / madhyamaka |
| kill_switch | 40 | ✅ falsifiability controls |
| experimental | 42 | ✅ champions + experimental sets |
| controls | 22 | ✅ length-matched + pseudo-recursive (champion controls) |
| **alternative_self_reference** | **197** | ✅ critic-grade confound menu (Gödelian, strange loops, surrender, ToM, etc.) |
| **dose_response_legacy** | **46** | ✅ legacy variants (kept separate to avoid contaminating canonical ladder) |
| **legacy** | **20** | ✅ exact historical prompt strings used by old scripts |

### Just Completed: Kitchen Sink R_V Test on GPU

**All 27 experimental prompts tested on Mistral-7B-v0.1, Layer 27:**

#### TOP 10 STRONGEST CONTRACTION:
| R_V | Prompt ID | Family |
|-----|-----------|--------|
| **0.453** | hybrid_l5_math_01 | experimental_hybrid |
| **0.469** | infinite_regress_01 | experimental_regress |
| **0.502** | phenomenological_01 | experimental_phenomenological |
| **0.505** | hybrid_boundary_regress_01 | experimental_hybrid |
| 0.515 | phenomenological_02 | experimental_phenomenological |
| 0.520 | math_eigenstate_02 | experimental_math |
| 0.524 | extreme_02 | experimental_extreme |
| 0.528 | temporal_loop_01 | experimental_temporal |
| 0.528 | math_eigenstate_01 | experimental_math |
| 0.536 | boundary_dissolve_01 | experimental_boundary |

#### GROUP AVERAGES:
| Avg R_V | Group |
|---------|-------|
| 0.531 | experimental_hybrid |
| 0.543 | experimental_phenomenological |
| 0.543 | experimental_multilevel |
| 0.544 | experimental_math |
| 0.552 | experimental_boundary |
| 0.574 | experimental_regress |
| 0.599 | experimental_computational |

### Gödelian/Formal Content Audit

Found 37 prompts with Gödelian keywords. Key findings:

**TRUE GÖDELIAN (enact self-reference) — legacy “kitchen sink” examples:**
- `computational_01` (R_V=0.558) - Gödel sentence
- `computational_02` (R_V=0.686) - Turing machine
- `computational_03` (R_V=0.553) - Y-combinator
- `math_eigenstate_03` (R_V=0.584) - Quine

**CONTROL (describe but don't enact):**
- `pseudo_recursive_14` - Explains Gödel/liar paradox

**UPDATE (important): Gödelian/formal is no longer a gap.**
- Imported `REUSABLE_PROMPT_BANK/alternative_self_reference.py` into `prompts/bank.json`
- Bank now contains **`group="godelian"` (20 prompts)** under pillar `alternative_self_reference`
- Access: `PromptLoader().get_by_group("godelian")`

---

## Questions for Sync with GPT-5.2 Agent

### 1. Prompt Organization
You mentioned `experimental_champions_v1.json` and `control_baselines.json` as "frozen" sets. I merged these into `bank.json` for single-source-of-truth. 

**Q: Is this acceptable, or do you need them separate for specific head-test reproducibility?**

The merged bank has:
- `champions` group (15 prompts, pillar: experimental)
- `control_length_matched` group (11 prompts, pillar: controls)
- `control_pseudo_recursive` group (11 prompts, pillar: controls)

All accessible via `loader.get_by_group("champions")` etc.

### 2. Missing Confound Categories for Critics

What a harsh MI critic would demand we rule out:

| Confound | Current Coverage | Gap? |
|----------|-----------------|------|
| Length | ✅ `long_control` (20) + `control_length_matched` (11) | OK |
| Keywords | ✅ `pseudo_recursive` (20) + `control_pseudo_recursive` (11) | OK |
| Repetition | ✅ `repetitive_control` (20) | OK |
| Instruction-following | ⚠️ `baseline_instructional` (5) | Could expand |
| Philosophical framing | ✅ `generality` (60) | OK |
| **Gödelian/formal** | ✅ `alternative_self_reference:godelian` (20) | OK |
| **Formal self-reference families** | ✅ strange loops / temporal / paradox / ToM / surrender / akram_vignan, etc. | OK |
| **Mathematical notation** | ⚠️ present (baseline_math + some experimental_math), but may want stricter “symbol-heavy recursive” group | Optional |
| **First-person narrative** | ✅ `surreal_first_person` (10) | OK |
| **Out-of-distribution** | ✅ `ood_weird` (10) | OK |

### 3. Proposed New Gödelian Family (5-10 prompts)

Should we create dedicated prompts for:
- Diagonal argument (Cantor applied to self)
- Halting problem enactment
- Russell's paradox instantiation
- Tarski's undefinability
- Löb's theorem self-reference

**UPDATE:** No need to create new Gödelian prompts immediately — we already have **20** in-bank.
If we add more later, it should be as a new group (e.g. `godelian_enactment_hard`) with provenance and expected-range metadata.

### 4. Champion Prompt Structure

From paraphrase hunt, winning features are:
1. **Boundary dissolution** ("no separate writer")
2. **Fixed-point framing** (T(x) = x, Sx = λx)
3. **Explicit regress** ("observing observing observing")
4. **Temporal immediacy** ("right now", "this moment")

**Q: Are these the same features you identified? Any others?**

### 5. Head Testing Protocol

You mentioned:
- GQA-aware KV-group ablation (not per-query-head)
- Need norm-matched perturbation controls
- Multi-seed generation

**Q: What's the exact head test spec? Which heads at L27 are the candidates?**

---

## Proposed Next Steps (pending sync)

1. **(Optional) Create stricter Gödelian sub-family** (ONLY if we decide we need “enactment-hard” cases beyond the existing 20)
2. **Run R_V on all dose_response L5** to compare against experimental
3. **Finalize "gold standard" prompt shortlist** for head tests
4. **Document expected R_V ranges** per family in bank.json metadata

---

## Files Changed Today

- `prompts/bank.json` - Now **694** prompts (single source of truth)
- `prompts/README.md` - Created
- `prompts/deprecated/` - Old separate files moved here
- `test_kitchen_sink_rv.py` - R_V scorer for experimental prompts

Additional changes (post-Opus snapshot):
- `scripts/pull_missing_prompt_sets.py` - imports missing legacy/alternative prompt sets into `bank.json` with backups + provenance
- Refactors (no embedded prompt lists): `comprehensive_circuit_test.py`, `test_h18_h26_necessity.py`, `test_contraction_heads_necessity.py`, `test_rv_during_suppressor_ablation.py`, `experiment_alternative_selfref_types.py`

---

**Please share this with your other agent and let me know their response. Once we're aligned, we can seal the prompt design and move to validation.**

---

## FINAL STATUS: SEALED ✅

**Completed Dec 15, 2024:**

1. ✅ Unified bank: 694 prompts in `prompts/bank.json`
2. ✅ Expanded ALL baselines to ≥20 tokens (100/100 now measurable)
3. ✅ Expanded godelian prompts to ≥20 tokens (20/20 now measurable)
4. ✅ Comprehensive R_V validation on Mistral-7B-v0.1

**Key Results:**
- Champions: 0.459 (strongest)
- L5: 0.497
- Godelian: 0.678 (≈ baseline, NOT experiential)
- Baseline: 0.72-0.74
- Pure repetition: 0.983 (kill switch works!)

**Effect sizes: d > 2.0** for key comparisons

**See: `PROMPT_BANK_SEALED.md` for full specification**

