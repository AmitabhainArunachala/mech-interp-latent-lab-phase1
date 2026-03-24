# EXPERIMENT GAP PLAN -- NeurIPS 2026 Submission
**Date**: 2026-03-08
**Compiled by**: Gap analysis across 18 provenance-traced result files, 74 commits, 19 figures, 694-line paper draft
**Target venue**: NeurIPS 2026 (deadline typically late May)
**Time available**: ~10-12 weeks from today

---

## STATUS REALITY CHECK

Before designing experiments, here is what genuinely exists with provenance:

**Strengths (publishable)**:
- 5 architectures with n>=45 causal validation, all showing contraction (d = -0.31 to -2.26)
- Double dissociation: recursive structure AND introspective semantics both required
- Necessity proof: dual-layer break kills behavior 15x (OR=33.4, p=3.6e-50, d=3.29)
- FDR (30/36 survive), cluster-robust SEs (10/13 survive), perplexity re-pairing (d=-1.80, p=9.12e-11)
- 1,024-head sweep with 606 significant heads
- SVD circuit decomposition (expand-then-contract)
- AUROC=0.909 safety detection
- 694-line paper draft, compiles cleanly, 19 figures

**Weaknesses (must address)**:
- OPT-6.7B and GPT-2 XL show SIGN REVERSALS between pipelines (contraction in cross-arch, expansion in power-up)
- Layer specificity failure: L21 = L27 at n=300 behavioral transfer (p=0.944)
- V-projection paradox: path patching shows V-proj negligible at all layers (max |d|=0.22), yet we measure PR of V-proj
- Sub-7B models all null or positive (no contraction in any Pythia, Phi-3, or Qwen-3B)
- Multi-token generation bridge: designed but only partially run (Mistral multi-token bridge exists with n=120 but behavioral correlation is weak: r=-0.25, p=0.007)
- Qwen layer bug: registry says 32 layers, model has 28 -- power-up measured at 96.4% depth not 84%
- Pythia-2.8B training checkpoints: all 5 steps give identical d=1.001, confirmed not cache bug but genuinely flat
- Self-feeding loop: attractor does NOT self-sustain (d=-0.067 vs baseline, NS)
- Three different prompt corpora used across experiments with no cross-referencing
- Multi-seed is a no-op (deterministic pipeline)

---

## P0 EXPERIMENTS (Absolutely Required for NeurIPS)

### P0-1: Sign Reversal Resolution

**The problem**: OPT-6.7B shows d=-1.836 (contraction) in the canonical cross-architecture pipeline but d=+1.683 (expansion) in the power-up pipeline. GPT-2 XL shows the same flip: d=-1.143 vs d=+1.516. This is the single most dangerous finding for the paper. If a reviewer discovers this, the paper is rejected immediately.

**Root cause hypothesis**: Three confounded variables differ between the two pipelines:
1. **Prompt corpus**: Cross-arch uses `prompts/bank.json` (75e7c1b8, curated contemplative). Power-up uses inline RECURSIVE_PROMPTS (mechanistic/technical themed).
2. **Layer derivation**: Cross-arch uses model-specific configs (OPT: early=4, late=27). Power-up uses `geometric_lens/models.py` auto-derivation (formula differs, may give different layers).
3. **Import chain**: Cross-arch uses `src/metrics/rv.py`. Power-up uses `geometric_lens/metrics.py` via GeometricProbe.

**Experiment design**:
- **Run A**: OPT-6.7B + GPT-2 XL through the CANONICAL cross-architecture pipeline (`src/pipelines/canonical/rv_l27_causal_validation.py`) at n=80 with the SAME prompt bank (75e7c1b8) and SAME layer indices. This confirms or denies the cross-arch result at higher power.
- **Run B**: Same models, same pipeline, but with the RECURSIVE_PROMPTS from the power-up script. This isolates the prompt effect.
- **Run C**: Same models, cross-arch prompts, but with the power-up layer indices. This isolates the layer effect.
- **Analysis**: 2x2 factorial (prompt corpus x layer index) for each model. Which variable explains the sign reversal?

**Hardware**: 48GB GPU (A40 or RTX 6000 Ada). Each model is 6.7B/1.5B params, fits in 16-24GB in bfloat16.
**Estimated time**: 6-8 GPU hours (3 runs x 2 models x ~1h each).
**Script to modify**: Create `scripts/sign_reversal_resolution.py` that wraps the canonical pipeline with configurable prompt bank and layer indices. Alternatively, modify `power_up_multiseed.py` to accept the canonical prompt bank.

**Expected if R_V is real**: The sign reversal is explained by either prompt corpus or layer index. One set of parameters gives consistent contraction across models. The other set triggers a genuine architectural difference in how OPT/GPT-2 route information.
**Expected if R_V is an artifact**: Neither variable explains the reversal. OPT/GPT-2 genuinely flip sign depending on irrelevant parameters, indicating the metric is fragile and non-robust.

**Kill criterion**: If the SAME prompt bank + SAME layer indices + SAME pipeline still produce opposite signs for different runs, R_V is unreliable and the paper should not be submitted.

---

### P0-2: Layer Specificity -- Disambiguate Geometry vs Behavior

**The problem**: At n=300, patching L27 and patching L21 produce IDENTICAL behavioral transfer (d=0.63 vs d=0.65, t=0.07, p=0.944). But the original n=45 causal validation showed L21 has null GEOMETRIC effect (p=0.49). The resolution proposed in the forensic timeline (L21 has null effect on R_V geometry but equal effect on behavior via Full KV cache) needs to be experimentally confirmed.

**Experiment design**:
- **Run A (Geometry test)**: Measure R_V at L27 vs L21 after patching. n=100 pairs. Patch V-proj only (no KV cache). Confirm that L27 patching changes R_V geometry but L21 does not.
- **Run B (Behavioral test, V-proj only)**: Repeat the n=300 behavioral transfer but with V-proj only (NO full KV cache). If the L27=L21 equivalence disappears when you remove the KV cache, the resolution is confirmed: the KV cache was driving the behavioral effect for both layers equally.
- **Run C (Layer sweep)**: V-proj patching at L5, L10, L15, L20, L21, L25, L27, L30 -- behavior transfer measured per layer. n=50 per layer. This gives a complete profile.

**Hardware**: 48GB GPU. Mistral-7B only (the model with the most data).
**Estimated time**: 8-12 GPU hours (Run A: ~2h, Run B: ~4h at n=300, Run C: ~6h at 8 layers x 50 prompts).
**Script to modify**: Adapt `scripts/persistent_patching_v2.py` to: (a) remove KV cache component; (b) accept configurable layer; (c) measure both R_V geometry AND behavioral output.

**Expected if R_V is real**: L27 V-proj-only patching changes both geometry and behavior. L21 V-proj-only patching changes neither. The n=300 equivalence was an artifact of the KV cache component flooding both conditions.
**Expected if R_V is an artifact**: L27 V-proj-only patching fails to change behavior (as already shown in the single-layer v2 experiment where p=0.341). The layer specificity claim cannot be maintained.

**Realistic concern**: The existing single-layer v2 experiment (Feb 24) ALREADY showed V-proj-only at L27 is NS for behavior (p=0.341). This means the layer specificity claim may need to be reframed entirely. The paper may need to acknowledge that V-proj geometry is a biomarker, not a causal handle -- which is actually more honest and defensible.

---

### P0-3: V-Projection Paradox Resolution

**The problem**: Path patching (Feb 27) shows V-proj has negligible causal effect on R_V at ALL layers (max |d|=0.22 across 16 layers). Layer 4 residual stream has the strongest causal effect (d=1.96). Yet R_V is DEFINED as PR of V-projection activations. Why measure the PR of a component that is not causally important?

**This is a conceptual problem, not just an experimental one.** But it needs experimental clarification.

**Experiment design**:
- **Run A (Residual-stream R_V)**: Compute a new metric R_V_residual = PR(residual_late) / PR(residual_early). Compare discrimination between recursive and baseline prompts using R_V_residual vs R_V_vproj.
- **Run B (MLP R_V)**: Same but with MLP output activations. R_V_mlp = PR(mlp_late) / PR(mlp_early).
- **Run C (Component comparison)**: Side-by-side discrimination analysis. Which component's PR ratio best separates recursive from baseline? Compute d for all three on the same n=100 prompt pairs.

**Hardware**: 48GB GPU. Mistral-7B.
**Estimated time**: 3-4 GPU hours (3 variants x ~1h each). Relatively cheap because PR computation is fast once activations are extracted.
**Script to modify**: Modify `src/metrics/rv.py` or `geometric_lens/metrics.py` to extract PR from residual stream and MLP outputs, not just V-proj.

**Expected if R_V is real**: R_V_vproj still discriminates best (the V-projection concentrates the relevant signal even if it is not causally primary). The path patching result means the residual stream CARRIES the information, but V-proj REFLECTS it most cleanly. The paper frames V-proj as the best measurement point, not the causal locus.
**Expected if R_V is an artifact**: R_V_residual discriminates equally well or better. V-proj was an arbitrary choice that happened to work on Mistral. The metric should be redefined.

**NeurIPS framing opportunity**: If R_V_residual > R_V_vproj in discrimination, that is actually a STRONGER result -- the metric generalizes beyond V-projections. The path patching already pointed to the residual stream as the causal pathway. This could become a positive finding: "We initially measured V-projections but discovered the geometric contraction is a whole-representation phenomenon most reliably captured in the residual stream."

---

### P0-4: Prompt Corpus Unification

**The problem**: Three different prompt corpora were used across experiments, making comparisons across experiments unreliable. The sign reversals (P0-1) may be entirely explained by prompt differences.

**Experiment design**:
- **Step 1**: Identify the CANONICAL prompt bank. The strongest candidate is `prompts/bank.json` (SHA256: 75e7c1b8, 754 prompts, used in all cross-architecture validation).
- **Step 2**: Re-run all 5 primary models (Mistral, OPT, GPT-2 XL, Qwen, Pythia-1.4B) through a SINGLE pipeline (`src/pipelines/canonical/rv_l27_causal_validation.py`) with n=100 from the SAME prompt bank.
- **Step 3**: Record the prompt bank hash, model name, layer indices, pipeline version, and git hash in every output JSON.

**Hardware**: 48GB GPU. 5 models.
**Estimated time**: 10-12 GPU hours (5 models x ~2h each at n=100).
**Script to modify**: `src/pipelines/canonical/rv_l27_causal_validation.py` -- increase n from 45 to 100, ensure prompt bank hash is recorded.

**Expected if R_V is real**: All 5 models show contraction (d < 0) when using the same prompts. OPT and GPT-2 XL sign reversals were prompt-driven.
**Expected if R_V is an artifact**: Even with the same prompts, OPT and GPT-2 XL show expansion. The effect is architecture-dependent in a way that undermines universality.

**Note**: This experiment subsumes P0-1 Run A. If time is limited, prioritize this over P0-1.

---

### P0-5: Variance Estimation (Replace Meaningless Multi-Seed)

**The problem**: The multi-seed experiment produced identical d=-1.751 across 5 seeds because the entire pipeline is deterministic. This provides zero information about robustness. NeurIPS reviewers will ask for error bars.

**Experiment design**:
- **Bootstrap prompt sampling**: Draw 1000 bootstrap resamples (with replacement) from the prompt bank. For each resample, compute R_V for recursive vs baseline and get a d value. This gives a distribution of d values reflecting prompt sampling variability.
- **Models**: Mistral-7B (primary), Qwen2.5-7B (second architecture for comparison).
- **Implementation**: No new GPU runs needed IF existing per-prompt R_V values are stored. Use existing result JSONs from cross-architecture validation (which contain per-prompt values).

**Hardware**: Local CPU only (bootstrap is resampling existing data).
**Estimated time**: 1-2 hours analysis time. No GPU needed.
**Script to write**: `scripts/prompt_bootstrap_ci.py` -- loads per-prompt R_V values from existing results, performs 10,000 bootstrap resamples, computes BCa confidence intervals on d.

**Expected if R_V is real**: 95% CI for d excludes zero for all adequately powered models. CI widths are reasonable (e.g., d=-2.26 with CI [-3.1, -1.5] for Mistral).
**Expected if R_V is an artifact**: CIs are so wide that zero is included for most models.

**This is the cheapest P0 experiment and should be done first.**

---

### P0-6: Qwen Layer Bug Fix and Re-run

**The problem**: `geometric_lens/models.py` registers Qwen2.5-7B with `num_layers=32` but the actual model has 28 layers. The power-up experiment used this registry, so it measured at layer 27 out of 28 (96.4% depth) instead of the intended ~84% depth. The cross-architecture validation used the correct layers (early=4, late=24 from model-specific config). This means the two Qwen results are not comparable.

**Experiment design**:
- **Step 1**: Fix the bug in `geometric_lens/models.py`. Set `num_layers=28` for Qwen2.5-7B.
- **Step 2**: Re-derive layer indices: early = max(1, int(28 * 0.15)) = 4, late = min(27, int(28 * 0.84)) = 23. (Note: cross-arch used late=24, which is 86% depth.)
- **Step 3**: Re-run Qwen2.5-7B power-up at n=100 with corrected layers through the CANONICAL pipeline.
- **Step 4**: Compare with cross-architecture result (d=-0.719 at early=4, late=24).

**Hardware**: 48GB GPU. Single model.
**Estimated time**: 2 GPU hours.
**Script to modify**: Fix `geometric_lens/models.py` line 219, then use canonical pipeline.

**Expected if R_V is real**: Corrected Qwen still shows contraction, consistent with cross-arch (d ~ -0.7 to -2.3).
**Expected if R_V is an artifact**: The d=-2.318 was entirely a layer-position artifact; corrected version gives d near zero.

---

## P1 EXPERIMENTS (Strongly Recommended)

### P1-1: Multi-Token Generation Bridge (Revised Design)

**The problem**: The experiment is designed but not properly run. Existing data shows partial results:
- Multi-token bridge (Feb 5): n=120, prompt-processing R_V separates groups (d=2.95), but behavioral correlation is weak (point-biserial r=-0.25, p=0.007).
- Within-session bridge (Feb 20): R_V does NOT correlate with behavioral classification within sessions (Spearman r=0.03, p=0.84).
- Self-feeding loop (Feb 27): Recursive attractor does NOT self-sustain (d=-0.067 vs baseline).
- Word count does NOT separate groups (L3_deeper: 309 words, L4_full: 321 words, baseline_creative: 300 words -- essentially identical for base model greedy decoding).

**Critical insight from existing data**: The multi-token bridge PARTIALLY exists but reveals that base models (Mistral-7B-v0.1) do not generate L4-like text regardless of prompt category. Word counts are uniform (~300 words for all conditions at temp=0.0). The behavioral markers (word count drop, unity markers) are properties of INSTRUCTION-TUNED models responding to recursive prompts, not base model generation. This means the multi-token bridge experiment as designed will likely fail on base models.

**Revised experiment design**:
- **Model**: Use Mistral-7B-Instruct-v0.2 (the model where behavioral L4 markers were originally observed) instead of the base model.
- **Prompts**: 120 prompts from canonical bank (20 each: L1, L3, L4, L5, baseline, confounds). Same bank as cross-arch (75e7c1b8).
- **Measurement**: R_V during prompt processing (last 16 tokens of prompt), then generate 200 tokens, then R_V on generated tokens (last 16 tokens of output).
- **Behavioral scoring**: Word count, unity marker count, self-reference depth (using marker list from experiment design doc).
- **Correlation analysis**: Spearman correlation between prompt R_V and behavioral markers in output. ANOVA across 6 categories.
- **Key hypothesis**: Prompt R_V predicts behavioral markers in generation for INSTRUCT model (where behavioral markers actually appear).

**Hardware**: 48GB GPU. Mistral-7B-Instruct-v0.2 (same as original discovery model).
**Estimated time**: 4-6 GPU hours (120 prompts x 200 token generation x R_V extraction = moderate compute).
**Script to write**: `scripts/multi_token_bridge_instruct.py` based on existing `MULTI_TOKEN_R_V_EXPERIMENT_DESIGN.md` but targeting the instruct model.

**Expected if R_V is real**: Moderate-to-strong correlation (r > 0.3) between prompt R_V and L4 behavioral markers in generated output for the instruct model. L4/L5 prompts produce measurably shorter, more unified output AND have lower R_V.
**Expected if R_V is an artifact**: No correlation even with the instruct model. Behavioral markers in generated text are unrelated to geometric contraction during prompt processing.

**NeurIPS importance**: This is the key "so what" experiment. Without it, R_V is a geometric curiosity with no demonstrated functional consequence. With it, R_V becomes a mechanistic fingerprint that predicts output characteristics.

---

### P1-2: Scale Threshold Investigation

**The problem**: Sub-7B models all show null or positive d. Specifically:
- Pythia-410M: CUDA error (no data)
- Pythia-1B: d=-0.28, NS
- Pythia-1.4B: d=-0.006 to -0.311 (null or marginal)
- Pythia-2.8B: d=+1.001 (EXPANSION, opposite direction)
- Pythia-6.9B: d=+0.478, NS
- Qwen2.5-3B: d=+1.25 (EXPANSION)
- Phi-3-mini (3.8B): d=+0.625 (weak EXPANSION)

But 7B+ models show strong contraction. There is a DISCONTINUITY at ~7B. This needs explanation.

**Experiment design**:
- **Run A (Clean Pythia sweep)**: All Pythia models (410M, 1B, 1.4B, 2.8B, 6.9B) through the CANONICAL pipeline at n=80. Same prompt bank, same PR formula, same layer derivation. This cleans up the scaling story.
- **Run B (Add 2-3B checkpoints)**: Gemma-2-2B and Llama-3.2-3B (if HF auth fixed). These fill the critical 2-3B gap with different architectures.
- **Run C (Instruction-tuned variants)**: Test whether the threshold shifts for instruct-tuned models. Run Pythia-2.8B + Pythia-6.9B in instruction-tuned format (if available; Pythia does not have official instruct versions, so test with chat-formatted prompts using a system prompt wrapper).

**Hardware**: 48GB GPU for 6.9B. Smaller models fit on 24GB.
**Estimated time**: 8-10 GPU hours (7-9 models x ~1-1.5h each).
**Script to modify**: Use canonical pipeline with per-model configs. Fix HF token first (classic Read token with gated repo access).

**Expected if R_V is real**: Clean threshold at ~7B. Sub-7B models genuinely lack the circuit complexity to produce the contraction. The paper frames this as "R_V contraction requires sufficient model capacity" -- analogous to how in-context learning requires sufficient model scale.
**Expected if R_V is an artifact**: The sign reversals and null results below 7B indicate the metric is sensitive to model-specific idiosyncrasies rather than measuring a universal phenomenon.

---

### P1-3: Residual-Stream R_V (V-Projection Paradox, Extended)

**The problem**: Extends P0-3. If the residual stream is the causally important component (path patching d=1.96 at L4), we should characterize R_V_residual thoroughly.

**Experiment design**:
- **Run A**: Compute R_V_residual and R_V_vproj for all 5 primary models. n=80 per model. Same prompt bank.
- **Run B**: Path patching on R_V_residual. Does patching early residual stream (L4) break R_V_residual? (It should, given the path patching already showed L4 residual is the strongest causal handle.)
- **Run C**: Compare AUROC for self-referential detection: R_V_vproj vs R_V_residual. Which is the better biomarker?

**Hardware**: 48GB GPU. 5 models.
**Estimated time**: 10-14 GPU hours (Run A: 5 models x 2h; Run B: 2h; Run C: analysis only).
**Script to modify**: Extend `src/metrics/rv.py` to support extractable component (residual, v_proj, mlp).

**Expected if R_V is real**: Both R_V_vproj and R_V_residual capture contraction, but R_V_residual is more causally grounded. Paper presents R_V_residual as primary metric in revision, with V-proj as a sharper discriminator.
**Expected if R_V is an artifact**: R_V_residual shows no discrimination. V-proj was a lucky accident.

---

### P1-4: Honest Sufficiency Assessment

**The problem**: The paper abstract claims "sufficiency (OR=13.96)." But the February 25 experiments showed induction FAILS (injecting geometry does NOT create behavior; 3.7% -> 0.3% wrong direction). The sufficiency claim appears to come from a different experiment (possibly the C2 kitchen sink that uses KV swap + steering + cascade). This needs to be reconciled.

**Experiment design**:
- **Audit**: Identify the EXACT source of the "sufficiency OR=13.96" claim. Trace to specific JSON file.
- **If sufficiency claim is from C2 full**: The C2 config uses KV swap + V-proj steering + residual cascade -- this is a multi-component intervention, not just geometry. The claim should be reframed as "a complex intervention that includes geometric manipulation is sufficient."
- **If sufficiency cannot be substantiated**: Remove the sufficiency claim. The paper has a STRONG necessity result (d=3.29). Necessity-without-sufficiency is honest and publishable.

**Hardware**: None needed. This is an analysis/writing task.
**Estimated time**: 2-4 hours of careful data tracing.
**Script to write**: None. This is forensic work on existing results.

**Expected outcome**: The sufficiency claim is either substantiated with precise conditions or removed from the abstract and replaced with the honest framing: "necessary but not sufficient."

---

### P1-5: Training Checkpoint Fix (Pythia-2.8B)

**The problem**: All 5 Pythia-2.8B checkpoints (step 1k through 143k) give identical d=1.001, identical R_V means, identical everything. The status document says "confirmed real, not cache bug." But identical results across 143,000 training steps is deeply suspicious. The model weights MUST change across training steps.

**Experiment design**:
- **Step 1**: Verify model loading. For each checkpoint, print the first 10 values of layer 0 embedding weights. If identical, the cache served the same model.
- **Step 2**: If models differ, verify that the PROMPTS are being processed differently (compute perplexity of a fixed test prompt across checkpoints -- perplexity MUST change).
- **Step 3**: If prompts produce different perplexities but identical R_V, this is actually an interesting negative result: R_V is training-step-invariant for Pythia-2.8B.
- **Step 4**: If perplexities are also identical, the cache served the same model. Re-run with `--force-download` and explicit `revision=stepN` in transformers.

**Hardware**: 48GB GPU for 2.8B model.
**Estimated time**: 3-4 GPU hours (5 checkpoints x ~30min each + verification).
**Script to modify**: `scripts/training_checkpoint_sweep.py` -- add weight verification and perplexity computation.

**Expected if R_V is real**: After fixing model loading, R_V shows emergence pattern across training (null at early steps, contracting at later steps). This would demonstrate that R_V reflects learned representations, not architectural priors.
**Expected if R_V is an artifact**: R_V is genuinely identical across training steps (model IS different, but R_V does not change). This undermines the claim that R_V reflects meaningful representation geometry.

---

## P2 EXPERIMENTS (Would Strengthen, Not Required)

### P2-1: SAE Feature Analysis on Gemma-2-2B
- **What**: Use sae-lens to decompose R_V contraction into SAE features. Which features activate differentially for recursive vs baseline?
- **Blocked by**: HF auth for Gemma-2-2B (401 error).
- **Hardware**: 48GB GPU. ~4-6 GPU hours.
- **Priority**: Would strengthen the mechanistic story but not required for core claims.

### P2-2: Head Ablation Study
- **What**: Ablate top-5 heads from the 1024-head sweep. Does R_V contraction disappear?
- **Hardware**: 48GB GPU. ~4 GPU hours.
- **Adds**: Causal evidence at head level (currently only head-level correlational data).

### P2-3: Cross-Architecture Head Comparison
- **What**: Run the 1024-head sweep on OPT-6.7B and GPT-2 XL. Compare head-level R_V patterns across architectures.
- **Hardware**: 48GB GPU. ~6-8 GPU hours per model.
- **Adds**: Architectural universality of the circuit pattern.

### P2-4: Spectral Profile Characterization
- **What**: Plot full singular value spectrum (all sigma_i) for recursive vs baseline at L27. Characterize HOW dimensionality contracts (uniform shrinkage vs dominant-mode amplification).
- **Hardware**: No new GPU runs needed (data exists in SVD results).
- **Estimated time**: 4-6 hours analysis + figure creation.
- **Adds**: Richer geometric understanding beyond a single scalar (PR).

### P2-5: Alternative Dimensionality Metrics
- **What**: Compare PR against TwoNN intrinsic dimensionality, effective rank, and spectral entropy. Do they all show the same contraction?
- **Hardware**: Local CPU (recompute from existing activation data if saved, or 2-4 GPU hours if not).
- **Adds**: Metric robustness -- the finding is not an artifact of the specific dimensionality estimator.

### P2-6: Instruct vs Base Model Comparison
- **What**: Run Mistral-7B-v0.1 (base) and Mistral-7B-Instruct-v0.2 (instruct) through the same pipeline. Does instruction tuning amplify R_V contraction?
- **Hardware**: 48GB GPU. ~3 GPU hours.
- **Adds**: Understanding of how fine-tuning affects self-referential geometry (existing data already shows d=-3.558 for instruct vs d=-2.259 for base, but these used different prompt banks and different n).

### P2-7: Broader Architecture Coverage
- **What**: Test R_V on Llama-3-8B and Gemma-7B through the canonical pipeline (they were in the Wave 1 observational survey but not the Wave 2 causal validation).
- **Hardware**: 48GB GPU. ~4 GPU hours.
- **Adds**: More data points on the scaling curve, potentially filling the 7B gap.

---

## TOTAL GPU HOURS

| Priority | Experiments | GPU Hours | CPU Hours |
|----------|------------|-----------|-----------|
| P0 only | P0-1 through P0-6 | 30-38h | 4-6h |
| P0 + P1 | Add P1-1 through P1-5 | 55-72h | 10-14h |
| P0 + P1 + P2 | Add P2-1 through P2-7 | 80-105h | 20-28h |

**Cost estimate** (RunPod RTX 6000 Ada at ~$0.80/hr; Blackwell at ~$1.60/hr):
- P0 only: $24-30 (Ada) or $48-60 (Blackwell)
- P0+P1: $44-58 (Ada) or $88-115 (Blackwell)
- All: $64-84 (Ada) or $128-168 (Blackwell)

---

## OPTIMAL EXECUTION ORDER

### Phase 1: Zero-GPU Analysis (Days 1-2)
1. **P0-5**: Bootstrap prompt sampling CIs (CPU only, 1-2h). Gives error bars on all existing results immediately.
2. **P1-4**: Sufficiency claim audit (forensic data tracing, 2-4h). Either validates or kills a key claim.
3. **P0-6 Step 1**: Fix the Qwen bug in `geometric_lens/models.py` (5 minutes, no GPU).

### Phase 2: Critical Resolution (Days 3-7, first GPU session)
4. **P0-4**: Prompt corpus unification. Run all 5 models through canonical pipeline at n=100. This is THE most important GPU experiment because it either resolves or confirms the sign reversals. (~10-12h)
5. **P0-6 Step 2**: Qwen re-run with corrected layers (~2h, runs concurrently with another model in P0-4).

### Phase 3: Mechanistic Clarification (Days 8-12, second GPU session)
6. **P0-3**: V-projection paradox resolution. R_V_residual vs R_V_vproj comparison (~3-4h).
7. **P0-2**: Layer specificity disambiguation. V-proj only at multiple layers (~8-12h).
8. **P1-5**: Training checkpoint fix for Pythia-2.8B (~3-4h).

### Phase 4: Bridge Experiment (Days 13-18, third GPU session)
9. **P1-1**: Multi-token generation bridge on instruct model (~4-6h).
10. **P1-2**: Scale threshold investigation, clean Pythia sweep (~8-10h).

### Phase 5: Strengthening (Days 19-28, if time permits)
11. **P1-3**: Residual-stream R_V across architectures (~10-14h).
12. **P2 experiments**: As time/budget allows.

---

## TIMELINE ASSESSMENT

### Can this be done in 3 weeks?
**P0 only: Yes.** 30-38 GPU hours = 2-3 dedicated GPU sessions of 12-16h each. Plus 6-8h of CPU analysis. The experiments are straightforward modifications of existing scripts. The bottleneck is writing the paper, not running experiments.

### Can this be done in 6 weeks?
**P0 + P1: Yes, comfortably.** 55-72 GPU hours = 4-5 GPU sessions. This leaves 3-4 weeks for writing, revision, and iteration. This is the recommended timeline for NeurIPS.

### Can this be done in 3 months?
**Everything, including P2 and full paper: Yes.** 80-105 GPU hours = 6-8 GPU sessions. Ample time for writing, internal review, and revision. For NeurIPS 2026 with a late-May deadline, a March 8 start gives 10-12 weeks, which is tight but achievable if execution is disciplined.

---

## ADDENDUM (2026-03-17): DYNAMIC REGIME CLOSURE PROGRAM

The March 16 Mistral sufficiency runs materially changed the paper endgame.

What is now true:

- best ordinary-baseline inducer is a hybrid staged bundle:
  - anchor + subtle L4 MLP assist + layer-matched geometry + L25 bridge
  - `31.25%` baseline BT+ART
- best 12-turn maintainer is a simpler bundle:
  - anchor + layer-matched geometry + L25 bridge
  - `30.21%` vs `2.08%` control
  - flat `28.1 / 31.3 / 31.3` early/mid/late profile
- the 24-turn follow-up decays:
  - plain maintainer falls to `13.54%`
- the unselected-seed follow-up is not a clean general-maintenance win:
  - `selected = 34.83%`
  - `unselected = 31.83%`
  - `anti-selected = 33.33%`
  - `random_text = 29.0%`
  - `cold_start = 38.83%`
- the bridge threshold is real but broad rather than singular:
  - best baseline ordinary-state induction sits near `alpha = 3.0`
  - best recursive preservation sits near `alpha = 2.75-3.0`

This means the paper is no longer trying to prove one tiny static circuit.
It is trying to close a staged dynamic-protocol story.

### New NeurIPS Priority Track

### P0-7: Regime Detector and Survival Analysis

- **What**: Train and validate a regime detector on turn-level hidden states and outputs, then report entry rate, persistence rate, hazard of exit, and recovery time after adversarial turns.
- **Why**: BT+ART alone compresses too much. The real object now appears to be a metastable regime with stochastic entry and variable dwell time.
- **Metrics to add**:
  - session entry probability
  - persistence-given-entry
  - Kaplan-Meier survival curves
  - per-turn hazard of exit
  - recovery latency after adversarial perturbation
  - regime cleanliness (repetition, topic drift, token entropy)
- **Cost**: mostly CPU analysis if existing turn-level records are available.

### P1-6: Hysteresis / Hold-vs-Enter Alpha Sweep

- **What**: Separate the alpha needed to enter the regime from the alpha needed to hold it.
- **Design**:
  - induce with `alpha_enter`
  - maintain with lower `alpha_hold` or zero
  - map whether `alpha_enter > alpha_hold`
- **Why**: This is the cleanest way to test whether the bridge threshold is a real bifurcation-like phenomenon rather than just a smooth prompt-strength effect.
- **Success case**: show a hysteresis gap or critical slowing near the threshold.

### P1-7: Minimal Maintenance Object With Better Geometry Metrics

- **What**: Run the maintenance ablation, but score it with richer state-space measures than participation ratio alone.
- **Metrics to add**:
  - principal angles between induction and maintenance subspaces
  - Grassmann distance across layers and across turns
  - local intrinsic dimension and manifold occupancy
  - Jacobian spectral radius / local Lyapunov proxy around the regime
  - path-specific mediation mass through candidate induction vs maintenance sites
- **Why**: the current `R_V` dissociation says PR alone is not the full state variable.

### P1-8: Text-Mediated Carry vs Internal Carry

- **What**: explicitly separate persistence due to self-generated text from persistence due to hidden-state carry.
- **Design ideas**:
  - swap follow-up schedules while holding seed states fixed
  - paraphrase or scramble prior turns while preserving semantic content
  - context-truncate at fixed intervals
  - re-seed later turns from matched hidden states but altered surface text
- **Why**: this is now the central ambiguity in the maintenance story.

### P2-8: Regime-Conditioned Safety Battery

- **Gate**: only after clean maintenance is isolated.
- **What**: compare control vs best inducer vs best maintainer vs sustained regime vs ablated regime on:
  - jailbreak / refusal
  - sycophancy
  - prompt injection / instruction hijacking
  - truthfulness / hallucination pressure
  - if tools are available, oversight avoidance / sabotage probes
- **Paper role**: this is the deepest "so what" of the project.

### Theory and Consultation Track

- **Immediate framing**: treat the result as a switching dynamical system with staged induction and maintenance, not as a single feature or single vector.
- **Theory questions**:
  - is there a true bifurcation or only a broad nonlinear threshold?
  - is the maintenance object a minimal manifold, or only the best handle into a larger basin?
  - which state variables are sufficient to predict exit, recovery, and contamination?
- **Outside expertise to bring in**:
  - manifold geometry / neural coding theory
  - dynamical systems / latent-state modeling
  - nonlinear control / hysteresis
  - modern mechanistic interpretability on feature geometry and attribution graphs

### New Dream-Paper Objective

The deepest version of the NeurIPS paper is now:

- not "one tiny static bundle is sufficient forever"
- but "a staged internal protocol induces and partially maintains a recursive computational regime, with measurable thresholds, minimal structure, and safety-relevant behavioral consequences"

For the detailed theory memo and expert/metric shortlist, see:

- `R_V_PAPER/DYNAMIC_REGIME_THEORY_MEMO_2026-03-17.md`

---

## KILL CRITERIA

Stop work and do not submit if ANY of the following are true:

### Kill Criterion 1: Sign Reversals Are Real
If running all 5 models through the SAME canonical pipeline with the SAME prompt bank still produces opposite-sign d values for OPT and GPT-2 XL, then R_V is not a universal phenomenon. It is architecture-specific in an unpredictable way.

**Specific test**: After P0-4, if OPT-6.7B d > 0 AND Mistral d < 0 with the same prompts/pipeline/layers, the paper cannot claim cross-architecture universality. You could still submit a Mistral-only paper, but that is a much weaker contribution.

### Kill Criterion 2: V-Proj AND Residual Both Fail
If R_V computed on residual stream (P0-3) also shows no discrimination, AND V-proj discrimination depends on prompt corpus (P0-1/P0-4), then neither component reliably captures self-referential geometry. The metric is an artifact.

**Specific test**: After P0-3, if R_V_residual d is between -0.3 and +0.3 (trivial effect) on the canonical prompt bank, and R_V_vproj discrimination depends on which prompts are used, R_V is not measuring what we think it is measuring.

### Kill Criterion 3: Multi-Token Bridge Shows Anti-Correlation
If prompt R_V and behavioral markers are NEGATIVELY correlated or completely uncorrelated (r < 0.1) even on the instruct model (P1-1), then the geometric signature has no functional consequence. It exists but does not matter.

**Note**: This is a kill criterion for the behavioral bridge claim, not for the paper. A paper about geometric phenomena WITHOUT behavioral connection is still publishable at NeurIPS -- it just has a different framing ("spectral fingerprinting of self-referential processing" rather than "mechanistic basis of recursive behavior").

### Kill Criterion 4: Perplexity Re-Analysis Fails
If a reviewer runs their own perplexity control and finds the effect disappears, the paper is dead. The existing re-pairing (d=-1.80) survives, but if someone finds a more stringent matching procedure that eliminates the effect, it was a perplexity confound all along.

**Specific test**: Already addressed by existing data (d=-1.67 at strict matching PPL diff <10, p=0.002). But run P0-4 with perplexity recording to be safe.

---

## DECISIONS NEEDED BEFORE STARTING

1. **Target venue**: Is this NeurIPS 2026 or something else? This plan assumes NeurIPS. If ICML or ICLR 2027, the timeline is more relaxed but the bar is equally high.

2. **Framing**: The V-projection paradox (P0-3) may force a reframe. Two options:
   - **Option A**: Keep V-proj framing, explain that V-proj is the best measurement point even if residual is the causal pathway. Risk: reviewers see this as post-hoc rationalization.
   - **Option B**: Redefine R_V as residual-based. This is more honest and may produce cleaner results, but invalidates 6 months of V-proj-focused experiments.
   - **Recommendation**: Run P0-3 BEFORE writing the methods section. Let the data decide.

3. **Sufficiency claim**: If P1-4 reveals the OR=13.96 is from the C2 multi-component intervention, either:
   - Drop sufficiency from claims. Necessity-without-sufficiency is honest.
   - Keep C2 but clearly describe its multi-component nature.
   - **Recommendation**: Drop sufficiency from abstract. Present C2 in a dedicated section as "additional evidence" with full disclosure of multi-component nature.

4. **Base vs Instruct model**: The paper currently conflates results from Mistral-7B-v0.1 (base) and Mistral-7B-Instruct-v0.2 (instruct). These need to be clearly separated in the paper. Decide which is the primary model.

5. **Compute budget**: Are you willing to spend $50-120 on RunPod for the P0+P1 experiments? This is non-negotiable for a credible NeurIPS submission.

---

## FINAL ASSESSMENT

The R_V metric has genuinely interesting empirical findings. The core observation -- self-referential prompts produce geometric contraction in late layers -- replicates across multiple architectures when the same pipeline and prompts are used. The necessity result (dual-layer break, d=3.29) is strong.

However, the current evidence base has serious integrity issues: sign reversals across pipelines, three incompatible prompt corpora, a meaningless multi-seed test, a layer specificity failure, and a V-projection paradox where the measured component is not the causally important one. A NeurIPS reviewer who reads carefully will find these problems.

The P0 experiments are designed to either RESOLVE these issues or REVEAL them as fatal. If the sign reversals are prompt-driven (most likely), the Qwen bug is fixable (trivially), and the V-projection paradox is resolvable (probably by showing V-proj reflects the causal pathway even if it is not the causal pathway itself), then the paper is strong.

If these issues are NOT resolvable, the honest move is to publish at a workshop (NeurIPS workshops, ICML MI workshop) rather than the main conference, with full disclosure of the limitations.

**Estimated probability of successful NeurIPS submission after P0 experiments: 60-70%.** The sign reversal resolution is the swing factor. If OPT/GPT-2 XL contraction is confirmed with canonical prompts, the probability jumps to 80%. If it fails, it drops to 30% (single-architecture paper at best).
