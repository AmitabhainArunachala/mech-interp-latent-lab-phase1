# R_V Paper — Bombastic Testing Handoff

**Created**: 2026-03-09
**Created by**: Triple-review session (MI expert + stats expert + infrastructure audit)
**Purpose**: Hand this to any agent. It contains everything needed to understand what was decided, why, and what to execute.

---

## What Was Done

### Phase 1: Triple Review of paper_colm2026_v005.tex

Three independent expert agents reviewed the current paper draft (694 lines, ~8,200 words). All three converged on the same 5 critical problems:

1. **OPT/GPT-2 sign reversal hidden with |d|** — Paper uses absolute value of Cohen's d to hide that OPT-6.7B (d=+1.68) and GPT-2 XL (d=+1.52) show EXPANSION, not contraction. Two different pipelines (cross-arch vs power-up) give opposite signs for the same models due to three confounded variables: different prompts, different layer indices, different import chains.

2. **V-projection is NOT causal** — Paper title says "Value Spaces" but path patching shows V-proj max |d|=0.22 (negligible). Residual stream at L4 has d=1.96. The metric works because V reflects residual stream geometry (V = W_v * h), not because V-space itself is causal.

3. **Sufficiency not proven** — Paper claims both necessity AND sufficiency. Data shows: breaking geometry breaks behavior (necessity, d=3.29). But KV injection transfers behavior (OR=13.96) WITHOUT transferring R_V geometry (d=0.11, NS). Must remove sufficiency claim.

4. **BT+ART rate wrong** — Paper says 27.7%, actual data says 3.7%. The 27.7% is from a different experiment arm (KV-only injection). Copy-paste error.

5. **Pipeline mixing** — Table 1 uses power-up pipeline n values. FDR uses cross-arch pipeline d values. These are different experiments. For OPT/GPT-2, they show OPPOSITE SIGNS.

### Phase 2: Deep Project Expert Verification

A fourth agent with full project access verified ALL findings against raw data files. Confirmed every number. Traced OR=13.96 to sufficiency_ladder JSON. Confirmed Pythia-2.8B checkpoints are byte-for-byte identical (cache bug, not real data).

### Phase 3: Skill Upgrades

Four skills were upgraded/created to prevent future agents from overclaiming:

| Skill | Version | Lines | Key Content |
|-------|---------|-------|-------------|
| `neel-nanda-mi` | 2.0.0 | 400 | SAELens, nnsight, tuned lens, honest R_V integration, fixed code examples |
| `anthropic-mi-papers` | 2.0.0 | 568 | 8 new papers (Circuit Tracing 2025, DAS, crosscoders), feature steering, honest R_V |
| `mi-statistics` | 1.0.0 | 335 | FDR, cluster-robust SEs, bootstrap BCa, power analysis, 7 MI pitfalls |
| `rv-paper` | 1.0.0 | 374 | 4-tier evidence system, all data file paths, what CAN/CANNOT be claimed, pre-submission checklist |

All skills live in `~/.claude/skills/`.

### Phase 4: Bombastic Testing Plan

Three expert agents (MI methodology, statistics, infrastructure) designed the plan below. The user's directive: "we won't water down the paper, just do massive more testing to make it bombastic."

---

## The 5 Known Problems and How Each Experiment Addresses Them

```
PROBLEM                          EXPERIMENT THAT RESOLVES IT
─────────────────────────────────────────────────────────────
OPT/GPT-2 sign reversal     →   E0 (layer validation) + P0 (unified pipeline)
V-proj not causal            →   P0.5 (residual vs V-proj comparison)
No sufficiency               →   P5 (generation-conditioned R_V) — reframes as predictive biomarker
BT+ART 27.7% error          →   Text fix (5 minutes)
Pipeline mixing              →   P0 (one pipeline for everything)
```

---

## The 8-Experiment Plan

### E0: Layer Selection Validation (MUST RUN FIRST)

**Why**: The OPT/GPT-2 sign reversal might be an artifact of wrong layer selection. Before running 500+ prompts, verify that the measurement layers are correct for each architecture.

**Protocol**:
1. For each of 5 models (Mistral-7B, OPT-6.7B, GPT-2-XL, Qwen-2.5-7B, Pythia-1.4B), run a full-layer PR sweep on 10 recursive + 10 baseline prompts from the canonical n300 bank.
2. Compute PR at EVERY layer. Plot PR(layer) for recursive vs baseline.
3. Identify the layer where recursive-baseline gap is maximized.
4. Compare against the current ModelSpec late_layer setting.
5. If OPT/GPT-2 discriminating layer is at a different depth percentage, this explains the sign reversal.

**GPU time**: ~2 hours
**Uses**: `scripts/scaling_law_sweep.py` (adapt) or `src/pipelines/discovery/layer_sweep.py`
**Success**: Sign reversal explained by layer selection → fix layers → R_V universal
**Failure**: Layers are correct, sign genuinely reverses → architecture-dependent effect (still publishable, different framing)

---

### P0: Canonical 5-Model Corpus Unification (THE MOST IMPORTANT EXPERIMENT)

**Why**: The paper currently mixes two pipelines with different prompts, layers, and import chains. ALL cross-architecture claims are indefensible without a single unified run.

**Protocol**:
1. 5 models x 150 prompts per group (100 recursive: 30 L5, 30 L4, 20 L3, 20 L1; 150 baseline: 50 factual, 50 creative, 50 confound) from the canonical n300 bank (`CANONICAL_CODE/n300_mistral_test_prompt_bank.py`).
2. Use per-model optimized layers from E0.
3. For each prompt, record: R_V(V-projection), R_V(residual stream), perplexity, token count, all raw PRs.
4. Use `scripts/power_up_multiseed.py` as the base runner, extended for dual-metric capture.

**Statistical analysis** (pre-registered):
- Per-model Welch's t-test (one-tailed: recursive < baseline)
- Cohen's d with 95% BCa bootstrap CI (10,000 resamples)
- Benjamini-Hochberg FDR across 5 model-level tests at alpha=0.05
- Cluster-robust SEs (compute ICC from template groups, report DEFF)
- Random-effects meta-analysis (REML) across 5 models: grand mean mu, tau^2, I^2, Q statistic, forest plot
- Jonckheere-Terpstra dose-response trend test across L1→L3→L4→L5

**n=150 justification**: With ICC=0.38 and DEFF=3.67, effective n drops to ~41/group. At effective n=41, 80% power detects d>=0.63. This covers Qwen (d=-0.72) which was marginal at n=100.

**Success criteria**:
- Bombastic: All 5 d<0, 4/5 FDR-sig, I^2<75%, dose-response significant
- Solid: 3/5 contract, OPT/GPT-2 explained by layer fix or honest architecture-dependence
- Paper-killing: Only Mistral works (reduces to single-model paper)

**GPU time**: 10-14 hours
**Critical addition**: Include 20 `introspective_concrete` prompts ("Observe a tree carefully") and 20 non-observational descriptive prompts ("A tree has branches...") to resolve the loose end where introspective_concrete shows R_V=0.612 (closer to recursive than baseline). A reviewer will use this to argue the effect is "observational language" not self-reference.

---

### P0.5: Residual Stream vs V-Projection R_V (Run Parallel with P0)

**Why**: The paper is titled "Value Spaces" but V-proj patching has max |d|=0.22. A knowledgeable reviewer will immediately ask "why measure V if patching V has no effect?" This experiment either justifies the V framing or forces a reframe.

**Protocol**:
1. On Mistral-7B, 40 prompts (20 recursive, 20 baseline):
   - Measure PR-ratio using V-projection outputs (current R_V method)
   - Measure PR-ratio using residual stream hidden states at same layers
   - Measure PR-ratio using K-projection, Q-projection, MLP outputs
2. Compute Cohen's d for each variant. Compute correlation between variants.

**Hooks already support this**: `capture_hidden_states`, `capture_k_projection`, `capture_q_projection` in `geometric_lens/hooks.py`.

**GPU time**: 2-3 hours
**Success**: V-proj R_V discriminates BETTER than residual → V framing justified
**Acceptable**: V and residual equal, K/Q weaker → "value/residual pathway" framing
**Failure**: All projections equal → reframe as generic dimensionality, not V-specific

---

### P1: Attribution Patching (Replaces Head Ablation)

**Why**: The original plan was "ablate top 7 circuit heads." But 606/1024 heads are "significant" (FDR-corrected: 566/1024 = 55%). Ablating 7 from 566 is uninterpretable. The field standard is now attribution patching (gradient-based), not ablation.

**Protocol**:
1. Make `participation_ratio` differentiable through `torch.linalg.svd` (SVD supports autograd).
2. For 20 recursive + 20 baseline prompt pairs on Mistral-7B:
   - Run forward pass, capture all head outputs
   - Compute R_V as differentiable function
   - Backpropagate from R_V to all head outputs
   - Attribution = gradient * (activation_recursive - activation_baseline) per head
3. Rank heads by mean absolute attribution. Top-k are the circuit.
4. THEN ablate only top-k heads, measure R_V drop. Compare against k random heads.
5. Dose-response: ablate 1, 3, 5, 7, 10, 15, 20 heads. Plot R_V vs ablated count.

**Controls**:
- Verify attribution on baseline-only gives near-zero everywhere
- Verify sum of attributions approximates total R_V change (completeness)
- If SVD gradient is numerically unstable, use finite-difference fallback

**GPU time**: 4-6 hours
**Success**: Top 10 heads account for >50% of R_V effect → sparse circuit identified
**Acceptable**: Distributed but clustered at L8-14 → "processing zone" not "circuit"
**Failure**: Uniform attribution → R_V is network-level, not circuit-level (still publishable, different framing)

---

### P2: Concept Erasure with Dose-Response

**Why**: Current concept erasure removes 1 direction from 4096 dimensions (0.024% variance). Of COURSE R_V is unaffected. Multi-direction erasure is needed for a real orthogonality claim.

**Protocol**:
1. Compute difference-in-means direction at L27: d = mean(h_recursive) - mean(h_baseline), normalized.
2. Train logistic probe (recursive vs baseline from L27 hidden states). Use probe weight as second concept direction.
3. Erase top-k directions (k = 1, 2, 3, 5, 8, 13, 21 — Fibonacci spacing for log-linear coverage).
4. For each k, compute R_V on 80 recursive + 80 baseline prompts.
5. Plot d(k) vs k. Fit exponential decay. Report k_star (half-effect scale).

**Control**: Erase k RANDOM directions (not top-k) at same layer. This should NOT reduce contraction.

**Existing scripts**: `scripts/linear_probe_selfref.py`, `scripts/dii_intervention.py`

**GPU time**: 3-5 hours
**Success**: k_star < 10 → effect is low-dimensional, "self-reference direction" identified
**Acceptable**: k_star 10-50 → moderately distributed
**Failure**: No reduction at any k → effect is holistic property of full space

**Bonus (if time)**: Test if the direction transfers across models (compute in Mistral, test in Qwen). If cosine > 0.3 → universal direction (Nature-tier finding).

---

### P3: Random-Direction DII Control

**Why**: Quick, cheap, essential. If random directions at L27 produce R_V~0.41 just like the actual direction, the DII result is an artifact.

**Protocol**:
1. Sample 100 random unit vectors in R^4096 at L27 in Mistral-7B.
2. For each, compute R_V change when that direction is boosted/suppressed.
3. Compare distribution against the actual causal direction's DII.

**Script EXISTS**: `scripts/random_direction_control.py`

**GPU time**: 1-2 hours
**Success**: Actual direction 3+ SD from random distribution → causal specificity proven
**Failure**: Random directions equally effective → L27 is sensitive to ANY perturbation

---

### P5: Generation-Conditioned R_V (The Bridge Experiment)

**Why**: This bridges geometry to behavior — the paper's thesis. KV injection shows a troubling dissociation (transfers behavior but not R_V). If R_V during prompt processing PREDICTS behavioral output, R_V is a validated biomarker even without sufficiency.

**Protocol**:
1. 80 prompts (40 recursive, 40 baseline) on Mistral-7B:
   - Measure R_V during prompt processing (single forward pass)
   - Generate 200 tokens (greedy, temperature=0)
   - Score generation on: self-reference count (regex), word count, perplexity
2. Correlate R_V(prompt) with each behavioral marker.
3. Plot with regression lines and bootstrap CIs.

**Script EXISTS**: `scripts/causal_generation_bridge.py`

**GPU time**: 4-6 hours
**Success**: r > 0.5 between R_V and behavioral markers → predictive biomarker confirmed
**Solid**: r > 0.3 → weak but real bridge
**Failure**: No correlation → geometric signature doesn't propagate to output

---

### P6: Cross-Model Direction Transfer (If Time Permits)

**Why**: If the "self-reference direction" from P2 transfers across architectures, this is a universal geometric property — the strongest possible claim.

**Protocol**:
1. Compute self-reference direction in Mistral-7B at L27 (from P2).
2. Compute same in Qwen-2.5-7B at equivalent layer.
3. Align spaces via CCA on 100 shared prompts.
4. Measure cosine similarity.

**GPU time**: 2-3 hours
**Success**: Cosine > 0.3 → universal direction

---

## Infrastructure That Already Exists

**You do NOT need to build experiment infrastructure from scratch.**

| Need | What Exists | Location |
|------|-------------|----------|
| R_V computation | `GeometricProbe`, `ParticipationRatio` | `geometric_lens/metrics.py`, `probe.py` |
| Model loading | Device management, attention impl detection | `geometric_lens/models.py` |
| Hooks (V, K, Q, residual, MLP) | Full hook taxonomy | `geometric_lens/hooks.py` |
| Config-driven runner | Pipeline registry (54 experiments) | `src/pipelines/run.py`, `registry.py` |
| Prompt bank | 320 prompts, 16 groups | `CANONICAL_CODE/n300_mistral_test_prompt_bank.py` |
| Random direction control | Ready to run | `scripts/random_direction_control.py` |
| Generation bridge | Ready to run | `scripts/causal_generation_bridge.py` |
| Head sweep | 1024-head sweep | `scripts/full_head_sweep.py` |
| Linear probe | Self-reference probe | `scripts/linear_probe_selfref.py` |
| DII intervention | Direction injection | `scripts/dii_intervention.py` |
| Bootstrap CIs | BCa computation | `scripts/bootstrap_ci.py` |
| FDR correction | Benjamini-Hochberg | `scripts/fdr_correction.py` |
| Cluster-robust SEs | ICC + DEFF | `scripts/cluster_robust_se.py` |
| Power-up multi-seed | 5-model sweep runner | `scripts/power_up_multiseed.py` |
| Layer sweep | Per-layer analysis | `src/pipelines/discovery/layer_sweep.py` |
| 42 canonical configs | JSON configs ready to execute | `configs/canonical/` |
| Batch GPU scripts | Shell scripts for RunPod | `scripts/gpu_batch_*.sh` |

---

## What the Paper CAN Claim After These Experiments

If results are positive:

1. R_V contraction is a **real geometric signature** of self-referential processing in 7B+ transformers
2. The effect requires **BOTH recursive structure AND introspective semantics** (double dissociation)
3. Late-layer processing is **necessary** for recursive behavioral markers (d=3.29)
4. R_V follows a **dose-response curve** across recursion depth (L1→L5)
5. R_V **predicts** self-referential generation output (if P5 succeeds)
6. The effect survives **all statistical hardening**: FDR, bootstrap CIs, cluster-robust SEs, perplexity controls, TOST equivalence for controls
7. R_V is a geometric **SIGNATURE** (readout), not proven to be the causal mechanism

## What the Paper CANNOT Claim (Even After Testing)

1. "Sufficiency" — unless P5 delivers r > 0.5
2. "V-projection is causal" — unless P0.5 shows V discriminates better than residual
3. "Universal scaling law" — R^2=0.047, this is noise
4. Any Pythia-2.8B checkpoint result — cache bug
5. "Transfer efficiency >100%" — likely artifact

---

## Timeline

| Day | Experiment | GPU hrs | Prerequisite |
|-----|-----------|---------|--------------|
| 1 | E0: Layer validation | 2h | None |
| 2-3 | P0: Unified pipeline + P0.5: V vs residual | 12-17h | E0 results (set layers) |
| 4-5 | P1: Attribution patching + P2: Concept erasure + P3: Random DII | 8-13h | None (parallel) |
| 6-7 | P5: Generation bridge + P6: Cross-model (if time) | 6-9h | P2 results (for P6) |
| 8 | Statistical polish, meta-analysis, figures | 0h | All experiments |
| 9-23 | Write paper | 0h | All results |
| 18 (Mar 26) | **ABSTRACT DEADLINE** | | |
| 23 (Mar 31) | **PAPER DEADLINE** | | |

**Total GPU**: 28-41 hours (~$30-50 on RunPod A100)

---

## Decision Tree After E0

```
E0 result: Do OPT/GPT-2 contract with corrected layers?
│
├── YES → Run P0 with all 5 models, claim universality
│         Title: "Geometric Signatures of Self-Referential Processing
│                 in Transformer Representations"
│         (universal finding, 5 architectures)
│
└── NO → Run P0 anyway, report honestly
         ├── If OPT/GPT-2 expand with contemplative prompts too:
         │   → Drop from main paper, supplement as anomalous
         │   → 3-architecture paper (Mistral, Qwen, Gemma)
         │
         └── If sign depends on prompt type:
             → Report as finding: "R_V direction is architecture- and
                prompt-dependent. GQA/Llama-family models contract;
                older MHA models show prompt-sensitive response."
             → This is actually MORE interesting than universal contraction
```

---

## Files Modified/Created in This Session

| File | Action | Content |
|------|--------|---------|
| `~/.claude/skills/neel-nanda-mi/SKILL.md` | Upgraded to v2.0.0 | SAELens, nnsight, honest R_V, fixed code |
| `~/.claude/skills/anthropic-mi-papers/SKILL.md` | Upgraded to v2.0.0 | 8 new papers, circuit tracing, honest R_V |
| `~/.claude/skills/mi-statistics/SKILL.md` | Created v1.0.0 | FDR, cluster-robust SEs, bootstrap BCa, 7 MI pitfalls |
| `~/.claude/skills/rv-paper/SKILL.md` | Created v1.0.0 | Evidence tiers, data map, CAN/CANNOT claims |
| `R_V_PAPER/BOMBASTIC_TESTING_HANDOFF.md` | Created | This document |

---

*The experiment is the message. Run E0 first — it costs 2 hours and determines the entire paper's scope.*
