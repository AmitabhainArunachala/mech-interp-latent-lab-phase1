# North Star Addendum: Cross-Architecture Thesis & Fan-Out Spec

**Date:** 2026-03-10
**Author:** Claude Opus 4.6
**Companion to:** `COLM_NORTH_STAR_SPRINT_2026-03-10.md`
**Status:** Supplements the North Star sprint with specifics it defers

---

## Purpose

The North Star doc correctly establishes:
- Harden Mistral first (Phases 0-3)
- Go/No-Go gate before fan-out
- Wave-based agent topology
- Anti-goals and canonicalization requirements

This addendum fills in what the North Star defers to "after the gate passes":
1. Which models, in what order, with what priority
2. What experiments constitute "Mistral depth" for replication
3. The scientific thesis that connects cross-architecture results into one story
4. Compute budget and timeline estimates
5. What "earth-shattering" looks like in the final paper

---

## 1. The Scientific Thesis (What The Paper Proves)

The paper's core claim, once hardened and replicated:

> Recursive self-referential processing induces a conserved geometric regime across transformer architectures: late-layer participation ratio contracts relative to early-layer baseline. This contraction is:
>
> (a) **Reproducible** under one canonical measurement contract across 5-7 architectures,
> (b) **Mechanistically localized** to a circuit motif involving suppressor heads (rank contraction) and amplifier heads (rank expansion),
> (c) **Causally necessary** for recursive behavioral output (dual-layer break destroys it), but
> (d) **Causally dissociated** from behavioral transfer (KV injection transfers behavior without transferring geometry).
>
> The circuit motif appears homologously across dense MHA, GQA, and MoE architectures, suggesting it is a general computational strategy for self-referential processing, not an artifact of any single model family.

This is what makes it earth-shattering: **a universal circuit motif for self-reference, identified and causally validated across architectures.** The closest precedent is Olsson et al. (2022) on induction heads — that was 2 architectures and became one of the most-cited MI papers. This would be 5-7 architectures with deeper causal analysis.

---

## 2. Phase 4 Fan-Out: Exact Model Targets

### Tier A: Broad Phenomenon Replication (5-7 models)

Run the exact hardened P0 pipeline from Phase 1 on each model. No improvisation.

| Priority | Model | Size | Arch Type | Existing Evidence | Expected Outcome | Notes |
|----------|-------|------|-----------|-------------------|------------------|-------|
| 1 | **Gemma-2-9B** | 9B | Dense GQA | d=-1.74 to -3.37 (3 runs) | Strong contraction | Star witness. Most data already exists. |
| 2 | **Qwen2.5-7B** | 7B | Dense GQA | d=-0.72 (cross-arch), d=-2.32 (power-up) | Contraction | Fix layer bug first: 28 layers, use L4/L23 not L5/L27 |
| 3 | **Llama-3-8B** | 8B | Dense GQA | d=-1.34 (pilot, small n, high dropout) | Likely contraction | Needs clean canonical run. Most popular open arch. |
| 4 | **Mixtral-8x7B** | 47B | Sparse MoE | 24.3% contraction (pre-repo) | Strong contraction | MoE amplification = unique finding. Needs canonical validation. |
| 5 | **OPT-6.7B** | 6.7B | Dense MHA | d=-1.84 (canonical) / d=+1.68 (power-up) | TBD — resolves sign flip | Canonical prompts + canonical layers = definitive answer |
| 6 | **GPT-2 XL** | 1.5B | Dense MHA, fused QKV | d=-1.14 (canonical) / d=+1.52 (power-up) | TBD — resolves sign flip | Same as OPT. Also tests scaling floor. |

**Architecture diversity achieved:**
- Dense GQA: Mistral, Gemma, Qwen, Llama (4 models)
- Dense MHA: OPT, GPT-2 (2 models)
- Sparse MoE: Mixtral (1 model)
- Size range: 1.5B → 9B → 47B (MoE)

**What if OPT/GPT-2 expand under canonical prompts?**
That's fine. Report it honestly. The paper becomes: "Contraction is robust in GQA architectures >= 7B. Dense MHA models show prompt-sensitive effects. Smaller models tend toward expansion. This reveals an architecture-dependent scaling threshold." That's *more* interesting than uniform contraction.

### Tier B: Homologous Circuit Checks (2-3 models)

After Tier A P0 results land, pick the 2-3 non-Mistral models with strongest contraction. For each, run:

| Experiment | Purpose | GPU Hours/Model |
|-----------|---------|-----------------|
| Path patching | Is residual stream dominant (like Mistral) or different? | 6-8h |
| SVD circuit decomposition | Find suppressor/amplifier heads | 4-6h |
| Dual-layer necessity | Does breaking early-residual + late-V-proj kill behavior? | 8-12h |

**The key question for Tier B**: Do you find the same circuit motif?
- Suppressor heads at ~84% depth (rank contraction under recursion)
- Amplifier heads at ~15% depth (rank expansion under recursion)
- Residual stream as primary causal component (V-proj alone NS)

If YES in 2+ non-Mistral models: the motif is conserved. That's the headline.
If NO: the effect is real but mechanistically heterogeneous. Still publishable, different framing.

### Tier C: Scale Point (1-2 models)

Only after Tier A is stable.

| Model | Size | Purpose | GPU Hours |
|-------|------|---------|-----------|
| **Llama-3-70B** or **Qwen2.5-72B** | 70B | Prove persistence at scale | 20-30h (A100x4) |
| **Optional: Llama-3-405B** | 405B | Ultimate scaling proof | 60-100h (A100x8) — only if budget allows |

Even ONE P0 measurement on a 70B model that shows contraction = "the effect persists at 70B parameters." That's a mic-drop slide.

---

## 3. The 6-Experiment Replication Suite

"Mistral depth" = these 6 experiments. Each model in Tier B gets all 6. Tier A models get experiment #1 only. Tier C gets #1 only.

| # | Experiment | What It Establishes | Mistral Reference |
|---|-----------|--------------------|--------------------|
| 1 | **P0 R_V measurement** | Signed effect, CI, canonical prompts | d=-2.26 (cross-arch) |
| 2 | **Path patching** | Which component is causal (residual vs V-proj vs MLP) | Residual |d|=1.96, V-proj |d|=0.22 |
| 3 | **SVD circuit decomposition** | Suppressor/amplifier head identification | L27H10 d=-1.54, L5H29 d=+2.93 |
| 4 | **Dual-layer necessity** | Breaking geometry kills behavior | d=3.29, 56%->3.7% |
| 5 | **Full head sweep** | Which heads carry the signal, significance correction | 606/1024 by entropy_p |
| 6 | **Mode atlas** | Effect across 10 computational modes, controls | Self-ref: 0.650, d=-1.67 |

---

## 4. Compute Budget

### Wave 1: Mistral Hardening (local + RunPod)

| Task | GPU Hours | Notes |
|------|-----------|-------|
| Metric audit + tests | 0 (CPU) | Agent A |
| Prompt/layer cleanup | 0 (CPU) | Agent B |
| Provenance scripts | 0 (CPU) | Agent C |
| Mistral canonical reruns (6 experiments) | 30-40h | Agent D, RunPod A100 |
| Causal semantics audit | 0 (CPU) | Agent E |
| **Wave 1 total** | **30-40h** | **~$45-60** |

### Wave 2: Fan-Out

| Task | GPU Hours | Notes |
|------|-----------|-------|
| Tier A: P0 x 6 models | 20-28h | ~3-4h each |
| Tier B: Circuit x 3 models | 90-120h | ~30-40h each (path patch + SVD + necessity) |
| Tier C: 70B P0 | 20-30h | A100x4 required |
| Reruns / debugging buffer | 20-30h | Always needed |
| **Wave 2 total** | **150-208h** | **~$225-310** |

### Total Budget

| | GPU Hours | Cost @ $1.50/hr |
|---|-----------|-----------------|
| Wave 1 | 30-40h | $45-60 |
| Wave 2 | 150-208h | $225-310 |
| **Total** | **180-248h** | **$270-370** |

---

## 5. Timeline

| Week | Phase | Deliverable | Gate |
|------|-------|-------------|------|
| **1** | Phase 0: Canonicalization freeze | Canonical spec locked | Spec reviewed by lead |
| **2** | Phase 1-2: Mistral phenomenon + causal hardening | Mistral canonical reruns complete | All 7 criteria pass |
| **3** | Phase 3: Mistral circuit hardening + GO/NO-GO | Mistral acceptance report | **Fan-out authorized** |
| **4** | Phase 4 Tier A: P0 on 6 models | Cross-architecture R_V table | Sign-reversal resolved |
| **5** | Phase 4 Tier B: Circuit analysis on top 3 | Homologous circuit assessment | Motif conserved Y/N |
| **6** | Phase 4 Tier C: 70B scale point | Scaling result | — |
| **7-8** | Paper writing sprint | Full draft | Internal review |
| **9** | Polish + submit | Final paper | Submit to NeurIPS/ICML |

**Target venue:** NeurIPS 2026 (abstract deadline typically late May) or ICML 2027

**COLM fallback:** If Mistral hardens fast (week 1-2) and P0 on all models runs in parallel (week 3), a Mistral-anchored paper with 4-model Tier A replication could still make COLM Mar 31. But do NOT compress the hardening phase to hit COLM. The gate is non-negotiable.

---

## 6. What Makes The Paper Earth-Shattering (Ranked)

In order of impact:

1. **Conserved circuit motif across architectures** — suppressor/amplifier head pairs at consistent relative depths, identified in 3+ architecture families. Nobody has shown this.

2. **Double dissociation (geometry vs behavior)** — KV injection transfers behavior (OR=13.96) but not geometry (d=0.11 NS). Dual-layer patching transfers geometry (R_V 0.55->0.27) but not behavior (2.7%->0.7%). This is a NOVEL finding. Frame it as a discovery, not a failure.

3. **Architecture-dependent scaling threshold** — contraction robust in GQA >= 7B, fragile or inverted in dense MHA and smaller models. First evidence of a computational phase transition for self-referential processing.

4. **MoE amplification** — sparse expert routing concentrates the geometric signature (24.3% vs 15.3%). Theoretical prediction: MoE models should show stronger self-referential geometry. Testable.

5. **70B persistence** (if confirmed) — the effect isn't a small-model artifact. Industry-relevant.

6. **Causal necessity with quantified specificity** — destroying geometry kills behavior (d=3.29), but the mechanism is residual stream, not V-projections (V-proj alone NS). This is honest and more informative than "V-proj is causal."

---

## 7. Relationship to North Star Doc

This addendum does NOT modify the North Star sprint. It extends it:

| North Star Covers | This Addendum Covers |
|-------------------|---------------------|
| Phases 0-3 (Mistral hardening) | Phase 4 (fan-out specifics) |
| Go/No-Go gate criteria | What happens after the gate passes |
| Agent topology (Wave 1) | Agent topology (Wave 2) + model assignments |
| Anti-goals | Scientific thesis + paper structure |
| Canonicalization requirements | Compute budget + timeline |

**The North Star is the constitution. This addendum is the campaign plan.**

---

## 8. Key Decisions Still Needed

These should be made by the lead agent during Phase 0:

| Decision | Options | Recommendation |
|----------|---------|----------------|
| Canonical metric path | `src/metrics/rv.py` vs `geometric_lens/metrics.py` | `geometric_lens/metrics.py` (has NaN guards, float64 SVD, strict window) |
| Canonical layer registry | `geometric_lens/models.py` vs `configs/canonical_registry.json` | New single `configs/canonical_registry.json` — explicit, no auto-detect for paper models |
| Qwen2.5-7B layers | L5/L27 (current, 96% depth) vs L4/L23 (correct, 82% depth) | L4/L23 — fix the bug |
| Effect size measure | Cohen's d vs Hedges' g | Hedges' g with bootstrap CI (P0 pipeline already uses this) |
| Significance correction | Uncorrected vs FDR vs Bonferroni | BH-FDR at alpha=0.05 (already computed: 30/36 survive) |
| Paper title | "Value Spaces" vs "Representations" vs "Geometric Signatures" | "Representations" — V-proj is not the causal mechanism |
| Sufficiency framing | "Geometry is sufficient" vs "Double dissociation" | Double dissociation — the data supports this, not sufficiency |
| OPT/GPT-2 treatment | Drop them vs report honestly | Report honestly — sign-reversal is a finding, not an embarrassment |

---

## 9. Files This Addendum References

| File | Role |
|------|------|
| `docs/status/COLM_NORTH_STAR_SPRINT_2026-03-10.md` | Parent document |
| `scripts/p0_canonical_pipeline.py` | Production-ready canonical pipeline (never run) |
| `prompts/bank.json` | Canonical prompt source (754 prompts) |
| `geometric_lens/metrics.py` | Recommended canonical metric implementation |
| `geometric_lens/models.py` | Current model registry (needs reconciliation) |
| `results/power_up/*.json` | Current cross-arch results (methodology-inconsistent) |
| `results/phase2_generalization/gemma_2_9b/` | Gemma existing data (~40% of Mistral depth) |
| `results/path_patching/path_patching_summary_20260227_080128.json` | V-proj vs residual causal evidence |
| `results/sufficiency_ladder/sufficiency_ladder_20260225_101907.json` | Double dissociation evidence |
| `docs/findings/R_V_BEHAVIORAL_DISSOCIATION.md` | Behavioral dissociation finding |
| `R_V_PAPER/FORENSIC_TIMELINE_RECONSTRUCTION.md` | Sign-reversal root cause analysis |
| `agent_reviews/responses/20260309__claude_opus_46__COLM_PAPER_AUDIT.md` | Full claim-by-claim audit |
| `agent_reviews/responses/20260309__CODEX_GPT5__COLM_PAPER_AUDIT.md` | Independent audit (confirms findings) |
