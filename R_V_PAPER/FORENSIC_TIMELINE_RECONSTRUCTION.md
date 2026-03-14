# FORENSIC TIMELINE RECONSTRUCTION: R_V Project
## December 9, 2025 -- March 7, 2026
**Compiled**: 2026-03-07 (updated with 12-agent deep findings) | **Source**: 74 git commits, ~90 result directories, 133 archive scripts, 7 RECOVERED_GOLD documents, 18 provenance-traced result files

---

## CHRONOLOGICAL TIMELINE

### 2025-12-09 -- Genesis (2 commits)

- **Commits**: `e2fdc30`, `48c7db6`
- **What happened**: Initial commit of the repo. Phase 1 Recursive Geometry Analysis on Mistral-7B-Instruct-v0.2.
- **Experiments introduced**: First R_V measurements using PR = (sum sigma_i^2)^2 / sum(sigma_i^4) on SVD of V-projection activations.
- **Models**: Mistral-7B-Instruct-v0.2 only.
- **Prompt bank**: Early version of what became the 320-prompt bank (L1-L5 levels + baselines + confounds).
- **Key measurement**: Window=16 tokens, early layer=5, late layer=27 (84% depth in 32-layer model).
- **Knowledge state**: Discovery phase. R_V contraction observed for recursive self-reference prompts.
- **Open questions**: Is this artifact? Is it model-specific? What causes it?

---

### 2025-12-11 -- OPERATION SAMURAI (5 commits)

- **Commits**: `ad5c560`, `1d646ab`, `b96c005`, `11a9110`, `00fb15a`
- **What happened**: "OPERATION SAMURAI: Complete refoundation" -- validated on GPU. Added Mistral-7B reproduction suite, DEC10 asymmetry/curvature experiments, git sync utilities.
- **Experiments**: Full Mistral-7B reproduction with validation. Jabberwocky experiments (nonsense controls). Asymmetry and curvature metrics tested.
- **Results**: Original 6-model observational survey. Models: Mistral-7B (15.3% contraction), Mixtral-8x7B (24.3%, strongest -- MoE effect), Qwen1.5-7B (9.2%), Gemma-7B (3.3%), Llama-3-8B (11.7%), Phi-3-medium (6.9%).
- **Hardware**: RunPod RTX 6000 Ada (48GB VRAM). bfloat16 precision (critical -- float16 causes NaN).
- **Code state**: PR formula established. V-projection extraction per architecture in `models/` directory (6 model-specific scripts).
- **Knowledge state**: Contraction is real and reproducible. Appears across architectures. MoE amplifies effect.

---

### 2025-12-12 -- The Breakthrough Day (17 commits)

- **Commits**: `7787d7a` through `74b067c` (17 total -- the most active day in the project)
- **What happened**: Deep circuit analysis. Relay mechanism discovery. Ground truth assessment. Skeptical audit. Grand unified patching test. BREAKTHROUGH: 100% behavior transfer achieved.

**Key events in order:**
1. **Relay Chain Discovery**: L14->L18->L25->L27 relay mechanism identified via layer sweep.
2. **Sign Error Found**: L14->L18 interpretation had a critical sign error (flagged in skeptical audit commit `040582e`).
3. **Ground Truth Assessment**: Honest evaluation document (`RECOVERED_GOLD/GROUND_TRUTH_ASSESSMENT.md`). 3 ironclad findings, 2 strong hypotheses, 3 critical unknowns.
4. **Grand Unified Test**: KV_CACHE vs V_PROJ vs RESIDUAL across L18/L25/L27. Found mechanism shifts: residual-based at L18/L25, attention-based at L27.
5. **100% Behavior Transfer**: Full KV cache (all 32 layers) + persistent V_PROJ patching at L27 during generation = 100% behavior transfer efficiency. Score: 11/11 (baseline: 0/11).
6. **n=300 Experiment**: Large-scale run (300 pairs). Transfer confirmed but effect smaller at scale (d=0.63 vs pilot score 11). **Wrong-layer control (L21) showed EQUAL behavioral transfer** (d=0.65, p=0.944 vs L27). Concern flagged in `ea3e8bb`. See Conflict Pair C6.

- **Results produced**:
  - `RECOVERED_GOLD/BREAKTHROUGH_BEHAVIOR_TRANSFER.md`
  - `RECOVERED_GOLD/GROUND_TRUTH_ASSESSMENT.md`
  - `RECOVERED_GOLD/GRAND_UNIFIED_TEST_RESULTS.md`
  - `archive/scripts/` -- 28 activation patching scripts, 18 behavioral transfer scripts, 22 KV cache scripts
- **Knowledge state**: Causal proof for Mistral-7B at L27. Behavior transfer works. But circuit is distributed, not localized to a single head.

---

### 2025-12-13 -- Consolidation (1 commit)

- **Commit**: `295745f`
- **What happened**: Analysis of causal sweep and cross-model robustness data.
- **Knowledge state**: Mistral-7B causal validation complete. Need cross-architecture replication.

---

### Pre-Repo Work: Pythia-2.8B Phase 2 Circuit Mapping

- **Document**: `RECOVERED_GOLD/PHASE_2_CIRCUIT_MAPPING_COMPLETE.md`, dated "November 19, 2025"
- **Note**: This date is BEFORE the git repo was created (Dec 9). Either the date is wrong, or this work predates the repo.
- **Key findings**: Phase transition at Layer 19 (59% depth) in Pythia-2.8B. Head 11 @ Layer 28 is primary compressor (71.7% contraction). All 32 heads contract (no expansion). Cohen's d = -4.51.
- **Hardware**: RunPod RTX 6000 Ada (48GB VRAM), bfloat16.
- **Prompt bank**: 320 prompts (L1-L5 + baselines + confounds).

---

### Original Mistral Causal Validation (Pre-Repo or Early Repo)

- **Document**: `RECOVERED_GOLD/MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md`, dated "November 16, 2024" [sic -- likely 2025]
- **Model**: Mistral-7B-Instruct-v0.2
- **N**: 45 pairs (also referenced: n=151 full validation)
- **Results**: Cohen's d = -3.558, p < 10^-6, 117.8% transfer efficiency
- **Four controls**: Random (+71.6%), Shuffled (-0.100, 61% reduction), Wrong layer L21 (+0.046, p=0.49), Orthogonal (null)
- **Key insight**: Layer 27 contains a bistable attractor (117.8% overshooting)

---

### 2026-01-11 -- Roadmap (2 commits)

- **Commits**: `7931ce6`, `f374f7d`
- **What happened**: Created roadmap for R_V geometric signatures project. 10 phases planned. Project formally scoped for publication.
- **Knowledge state**: 6 models validated observationally. Mistral causal validation complete. Pythia circuit mapping done. Need: cross-architecture causal validation, perplexity controls, statistical hardening.

---

### 2026-01-15-16 -- Major Restructure (6 commits)

- **Commits**: `792dac5`, `a717b43` (Jan 15); `98d8a41`, `8b21eae`, `98962d8`, `7dcc2bc` (Jan 16)
- **What happened**: Complete Phases 1-6 repo restructure. Unified config generator with GQA detection. Industry-grade reproducibility infrastructure. Extended metrics for publication. Cross-architecture protocol.
- **New infrastructure**:
  - `src/metrics/rv.py` -- the canonical PR implementation (strict measurement contract, NaN for short sequences)
  - `geometric_lens/metrics.py` -- production PR with entropy-based erank
  - `geometric_lens/models.py` -- ModelSpec registry with auto-derived layer indices
  - `configs/canonical/` -- standardized experiment configs per model
  - `prompts/bank.json` -- 754 prompts, SHA256-versioned
  - `prompts/loader.py` -- PromptLoader with hash versioning
- **Code state**: Two layer-derivation formulas now coexist:
  - `geometric_lens/models.py` lines 294-296: `early = max(1, int(num_layers * 0.15))`, `late = min(num_layers - 1, int(num_layers * 0.84))`
  - `src/metrics/rv.py` line 123: `late = num_layers - 5`
  - **Two distinct import chains**:
    - Canonical pipelines (`src/pipelines/canonical/*.py`) -> `src/metrics/rv.py` (used for cross-arch n=45 runs)
    - Standalone scripts (`scripts/power_up_multiseed.py`, `scripts/scaling_gap_sweep.py`) -> `geometric_lens/metrics.py` via `GeometricProbe` (used for power-up n=80 and scaling gap runs)
  - Same PR formula in both, but `geometric_lens/metrics.py` has CPU SVD fallback + NaN/Inf guards

---

### 2026-02-02 -- Audit Remediation (1 commit)

- **Commit**: `7bb5b2f`
- **What happened**: Audit remediation and noise cleanup.

---

### 2026-02-02 (run dates) -- Cross-Architecture Causal Validation (5 models)

- **Note**: These runs have timestamps starting 20260202 in their directory names, committed on Feb 4.
- **Pipeline**: `src/pipelines/canonical/rv_l27_causal_validation.py`
- **Prompt bank**: Version `75e7c1b8dcebc24e` (same for all 5 models)
- **N**: 45 pairs each
- **Window**: 16 tokens

**Per-model layer configurations and results:**

| Model | Total Layers | early | target (late) | Relative early | Relative late | delta_main | Computed d |
|-------|-------------|-------|---------------|----------------|---------------|------------|------------|
| Mistral-7B-v0.1 | 32 | 5 | 27 | 16% | 84% | -0.1672 | -2.259 |
| OPT-6.7B | 32 | 4 | 27 | 13% | 84% | -0.3603 | -1.836 |
| GPT2-XL | 48 | 6 | 40 | 13% | 83% | -0.1376 | -1.143 |
| Qwen2.5-7B | 28 | 4 | 24 | 14% | 86% | -0.1037 | -0.719 |
| Pythia-1.4B | 24 | 3 | 20 | 13% | 83% | -0.0048 | -0.311 |

**Key observations**:
- Layer indices are model-specific, derived per architecture (not a fixed formula)
- All use relative positions ~13-16% early, ~83-86% late
- ALL 5 models show negative delta (contraction direction)
- OPT-6.7B shows STRONG contraction (d=-1.84) in this experiment
- GPT2-XL shows solid contraction (d=-1.14) in this experiment
- Pythia-1.4B is marginal (d=-0.31) -- consistently the weakest across all experiments

---

### 2026-02-04 -- Paper Sprint (26 commits)

- **Commits**: 26 commits from `e963508` to `6cd0e7c` (largest single-day burst)
- **What happened**: Massive push. Created rv_toolkit pip package. Test suite. Landing page. Figures gallery. Interactive calculator. CLI. First successful PDF compilation of paper. LaTeX skeleton with publication-quality figures. References.bib started. Statistical audit report. Canonical configs + cross-architecture results committed. Bridge hypothesis synthesis.
- **Results committed**: `results/phase1_cross_architecture/` -- the n=45 causal validation runs
- **Paper state**: `R_V_PAPER/paper_colm2026_v005.tex` skeleton.

---

### 2026-02-05 -- Publication Blockers (4 commits)

- **Commits**: `babe1a0`, `e45d74a`, `83c764d`, `b4de29d`
- **What happened**: PR formula fix in rv_toolkit. Residual indexing fix. Architecture detection. Publication blockers identified.

---

### 2026-02-05-09 -- Multi-Token Bridge + Reruns (run dates)

- 7 multi_token_bridge runs for Mistral-7B (timestamps 20260205)
- 2 Pythia-1.4B n=63 reruns (timestamps 20260208-20260209)
- 1 GPT2-XL rerun (timestamp 20260209)

---

### 2026-02-07 -- 48h Audit Checkpoint (1 commit)

- **Commit**: `fd86e02`

---

### 2026-02-13 -- Gnani Protocol (3 commits)

- **Commits**: `c6ffe8e`, `50272e1`, `3683e5b`
- **What happened**: Circuit mapping experiment infrastructure. Gnani protocol + circuit mapping results. Interactive gnani CLI with real-time R_V tracking.
- **Results**: `results/gnani_protocol/`, `results/circuit_mapping/`, `results/sustained_gnani*/`

---

### 2026-02-27 -- Path Patching + Self-Feeding Loop (run dates)

- **Path Patching**: `results/path_patching/path_patching_summary_20260227_080128.json`
  - Full 16-layer x 3-component sweep (residual, v_proj, mlp) on Mistral-7B-v0.1
  - n=20 prompts per condition, break direction
  - **Layer 4 residual is MOST causal** for R_V: d=1.96 (0.650 -> 0.881)
  - Layer 2 residual: d=1.65; Layer 0 residual: d=1.37
  - **V_proj has negligible effect at ALL layers**: max |d| = 0.22 (across all 16 layers)
  - L6+ residual: |d| < 0.51 (much weaker than L0-L4)
  - See Conflict Pair C7

- **Self-Feeding Loop**: `results/self_feeding_loop/self_feeding_summary_20260227_054825.json`
  - 3 conditions x 5 sessions x 50 turns = 750 turns
  - self_feed_recursive: 10.0% BT+ART rate
  - self_feed_baseline: 10.4% BT+ART rate
  - gnani_scaffolded: 42.4% BT+ART rate
  - **Recursive attractor does NOT self-sustain** (d=-0.067 vs baseline, NS)
  - Gnani scaffolding increases BT+ART 4.2x (d=-4.28 vs self-feed recursive, p=0.012)
  - Explicitly recorded: `"attractor_self_sustains": false`

---

### 2026-02-20 -- Double Dissociation (run dates)

- **Results**: `results/circularity_controls/circularity_controls_20260220_*.json`
- **Finding**: R_V requires BOTH recursive structure AND introspective semantics. Neither alone is sufficient. This is a double dissociation.

---

### 2026-02-24-25 -- Causal Patching Battery

- **Commit**: `facecc6` (Feb 25)
- **GPU**: NVIDIA RTX PRO 6000 Blackwell (98GB VRAM)
- **Prompt bank**: hash `e072ff86dbaee40b`
- **Key results** (from `results/CAUSAL_PATCHING_RESULTS_20260225.md`):

| Experiment | Design | Key Finding |
|-----------|--------|-------------|
| Single-layer L27 V-proj | 4 conditions x 5 sessions x 30 turns | Moves geometry, NOT behavior (p=0.341, NS) |
| Dual-layer L18+L27 | 4 conditions x 10 sessions x 30 turns | BREAKS behavior 15x (56%->3.7%, OR=33.4, p=3.6e-50, d=3.29) |
| Induction test | Inject recursive geometry into baseline | Does NOT create behavior (NS) |

- **Conclusion**: Dual-layer geometry is NECESSARY but NOT SUFFICIENT for recursive behavior.

---

### 2026-03-01 -- Scaling Gap Experiments

- **Results**: `results/scaling_gap/`
- **Pipeline**: `scripts/scaling_gap.py`

| Model | Params | Layers | early | late | d | p | Status |
|-------|--------|--------|-------|------|---|---|--------|
| Qwen2.5-3B | 3B | 36 | 5 | 30 | +1.25 to +1.60 | <0.001 | Complete (EXPANSION) |
| Phi-3-mini | 3.8B | 32 | 4 | 26 | +0.625 | 0.011 | Complete (marginal EXPANSION) |
| Pythia-6.9B | 6.9B | 32 | 5 | 27 | +0.478 | 0.068 | Complete (NS) |
| Gemma-2-2B | 2B | -- | -- | -- | -- | -- | FAILED (HF auth 401) |
| Llama-3.2-3B | 3B | -- | -- | -- | -- | -- | FAILED (HF auth 401) |

---

### 2026-03-03-04 -- Statistical Hardening (no separate commits)

- **FDR Correction**: 30/36 tests survive BH alpha=0.05. Losses: Pythia variants, genuine vs deceptive safety.
  - Script: `scripts/fdr_correction.py`
  - Artifact: `results/fdr_correction/fdr_results_20260303_232741.json`
- **Perplexity Re-Pairing**: d=-1.80, p=9.12e-11 (n=30 pairs). Strict matching (PPL diff <10): d=-1.67, p=0.002 (n=8). Confound ruled out.
  - Script: `scripts/perplexity_repairing.py`
  - Artifact: `results/perplexity_repairing/repairing_results_20260303_233230.json`
- **Cluster-Robust SEs**: 10/13 survive DEFF=2. Losses: Phi-3-mini (d=0.625), Pythia-6.9B (d=0.478), Pythia-1.4B cross-arch (d=-0.31).
  - Script: `scripts/cluster_robust_se.py`
- **Bootstrap CIs**: Mode atlas d=-1.67 CI=[-2.11,-1.21]; Causal d=-3.47 CI=[-4.28,-2.47]
- **Power Analysis**: 8/12 adequately powered (1-beta >= 0.80). Underpowered: Pythia-1.4B cross-arch (0.41), Phi-3-mini (0.77), Pythia-6.9B (0.49).

---

### 2026-03-04 -- Power-Up n=80 Experiments

- **Results**: `results/power_up/`
- **Pipeline**: `scripts/power_up_multiseed.py`
- **N**: 80 prompts per model
- **Prompt source**: Inline RECURSIVE_PROMPTS (93 rec + 83 baseline) -- mechanistic/technical themed, NOT the curated contemplative L3/L4/L5 bank. Example themes: "attention heads", "weight matrices", "gradient descent" -- qualitatively different from canonical bank's "observer observing itself" prompts.
- **Import chain**: Uses `geometric_lens.probe.GeometricProbe` -> `geometric_lens/metrics.py` (NOT `src/metrics/rv.py`). Same PR formula but different SVD handling (CPU fallback, NaN guards).

| Model | n_rec | n_base | rv_recursive | rv_baseline | d | Direction |
|-------|-------|--------|-------------|------------|---|-----------|
| Mistral-7B | 75 | 77 | 0.686 | 0.855 | -1.656 | CONTRACTION |
| OPT-6.7B | 72 | 66 | 1.115 | 0.789 | +1.683 | EXPANSION |
| GPT2-XL | 69 | 56 | 0.872 | 0.711 | +1.516 | EXPANSION |
| Qwen2.5-7B | 61 | 63 | 0.903 | 1.329 | -2.318 | CONTRACTION |
| Pythia-1.4B | 66 | 54 | 0.633 | 0.633 | -0.006 | NULL |

**Multi-Seed Test** (Mar 6): 5 seeds, all give identical d=-1.751 for Mistral n=45. Deterministic computation confirmed.

---

### 2026-03-07 -- Final Commits (5 commits)

- **Commits**: `6379eb1` through `05f70ca`
- **What happened**: March statistical hardening outputs committed. COLM draft updates, references, figure set. Compiled v005 PDF.
- **Paper state**: `R_V_PAPER/paper_colm2026_v005.tex` (694 lines, 13 pages, complete 6-section structure + appendix). All 11 referenced figures resolve. Compiles cleanly.
- **References**: `R_V_PAPER/references.bib` (380 lines, 43 entries). Marchenko-Pastur 1967 present. 10 must-cite papers still missing: Chun 2025, Dong 2021, Valeriani 2023, Wang 2025, Alpay 2026, Wu 2024, Engels 2024, Geshkovski 2024, Sharkey 2025, Li 2025.
- **Master Plan**: 19/19 experiments tracked. All P0 gaps COMPLETE. 19 figures total.

---

## PROVENANCE TABLE

| # | Result File | Script | Model | N | Prompt Bank | early | late | Hardware | Date | Status |
|---|-------------|--------|-------|---|-------------|-------|------|----------|------|--------|
| 1 | `phase1_cross_architecture/.../mistral_7b/summary.json` | `rv_l27_causal_validation.py` | Mistral-7B-v0.1 | 45 | `75e7c1b8` | 5 | 27 | RunPod CUDA | 2026-02-02 | Complete |
| 2 | `phase1_cross_architecture/.../opt_6_7b/summary.json` | same | OPT-6.7B | 45 | `75e7c1b8` | 4 | 27 | RunPod CUDA | 2026-02-02 | Complete |
| 3 | `phase1_cross_architecture/.../gpt2_xl/summary.json` | same | GPT2-XL | 45 | `75e7c1b8` | 6 | 40 | RunPod CUDA | 2026-02-02 | Complete |
| 4 | `phase1_cross_architecture/.../qwen2_7b/summary.json` | same | Qwen2.5-7B | 45 | `75e7c1b8` | 4 | 24 | RunPod CUDA | 2026-02-02 | Complete |
| 5 | `phase1_cross_architecture/.../pythia_1_4b/summary.json` | same | Pythia-1.4B | 45 | `75e7c1b8` | 3 | 20 | RunPod CUDA | 2026-02-02 | Complete |
| 6 | `power_up/mistral-7b_n80_result.json` | `power_up_multiseed.py` (GeometricProbe) | Mistral-7B-v0.1 | 80 | inline RECURSIVE_PROMPTS | auto | auto | CUDA | 2026-03-04 | Complete |
| 7 | `power_up/opt-6.7b_n80_result.json` | same | OPT-6.7B | 80 | inline RECURSIVE_PROMPTS | auto | auto | CUDA | 2026-03-04 | Complete |
| 8 | `power_up/gpt2-xl_n80_result.json` | same | GPT2-XL | 80 | inline RECURSIVE_PROMPTS | auto | auto | CUDA | 2026-03-04 | Complete |
| 9 | `RECOVERED_GOLD/MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md` | `mistral_L27_FULL_VALIDATION.py` | Mistral-7B-Instruct-v0.2 | 45+151 | 320-prompt bank | 5 | 27 | RunPod RTX 6000 Ada | ~Nov 2025 | Complete |
| 10 | `persistent_patching_v3/...20260225.json` | `persistent_patching_v3_dual.py` | Mistral-7B-v0.1 | 1200 turns | `e072ff86` | L18+L27 | dual | RTX PRO 6000 Blackwell | 2026-02-25 | Complete |
| 11 | `fdr_correction/fdr_results_20260303.json` | `fdr_correction.py` | All | 36 tests | n/a | n/a | n/a | Local | 2026-03-03 | Complete |
| 12 | `perplexity_repairing/repairing_results_20260303.json` | `perplexity_repairing.py` | Mistral-7B | 30 pairs | curated | -- | -- | Local | 2026-03-03 | Complete |
| 13 | `scaling_gap/qwen2.5-3b_result.json` | `scaling_gap.py` | Qwen2.5-3B | ~37 | unknown | 5 | 30 | CUDA | 2026-03-01 | Complete |
| 14 | `full_head_sweep/` | `full_head_sweep.py` | Mistral-7B | -- | -- | -- | -- | CUDA | 2026-03 | Complete (1024 heads) |
| 15 | `R_V_PAPER/paper_colm2026_v005.tex` | Manual | -- | -- | -- | -- | -- | Local | 2026-03-07 | Draft v005 |
| 16 | `path_patching/path_patching_summary_20260227.json` | `scripts/path_patching_v2.py` | Mistral-7B-v0.1 | 20 | unknown | 0-30 (sweep) | 0-30 (sweep) | CUDA | 2026-02-27 | Complete |
| 17 | `self_feeding_loop/self_feeding_summary_20260227.json` | unknown | Mistral-7B-v0.1 | 750 turns | unknown | -- | -- | CUDA | 2026-02-27 | Complete |
| 18 | `docs/misc/neurips_n300_summary.md` | `neurips_n300_robust_experiment.py` | Mistral-7B-Instruct-v0.2 | 300 | 320-prompt bank | 5 | 27+L21 | RunPod | 2025-12-12 | Complete |

---

## AMBIGUITY LOG

### A1: Phase 2 Circuit Mapping Date
- `RECOVERED_GOLD/PHASE_2_CIRCUIT_MAPPING_COMPLETE.md` is dated "November 19, 2025"
- The git repo was created December 9, 2025
- Either the date in the document is wrong, or this work predates the repo and was added later
- The Pythia-2.8B results (d=-4.51, Head 11 @ L28) are referenced elsewhere as established fact

### A2: Original Mistral Causal Validation Date
- `RECOVERED_GOLD/MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md` says "November 16, 2024"
- Almost certainly should be November 16, 2025 (typo in year)
- If literally 2024, this would predate the entire project by over a year

### A3: Power-Up Prompt Source
- Power_up experiment results do NOT record which prompt bank version was used
- The `prompt_bank_version` field is absent from all power_up JSON files
- Inference from code suggests `RECURSIVE_PROMPTS[:80]` (generic, not curated L3/L4/L5)
- Cannot confirm exactly which prompts drove the OPT/GPT2 "reversal" results

### A4: Scaling Gap Prompt Source
- Scaling_gap results also lack explicit prompt bank versioning
- Uses `scripts/scaling_gap.py`, which may use different prompts than canonical pipeline
- Positive d values for smaller models may reflect prompt differences, not genuine architecture effects

### A5: Two Layer-Derivation Formulas
- `geometric_lens/models.py`: `early = int(num_layers * 0.15)`, `late = int(num_layers * 0.84)`
- `src/metrics/rv.py`: `late = num_layers - 5`
- For 32-layer models: models.py gives late=26, rv.py gives late=27 (differ by 1)
- For 48-layer GPT2-XL: models.py gives late=40, rv.py gives late=43 (differ by 3)
- The canonical cross-architecture runs use NEITHER formula -- they use model-specific configs
- Which formula the power_up and scaling_gap experiments use is not recorded

### A6: Multi-Seed Is a NO-OP
- `results/power_up/multi_seed_summary_20260306.json`: 5 seeds, identical d=-1.751, std=0.0
- Code sets `torch.manual_seed(seed)` and `np.random.seed(seed)` but these only affect random operations
- In eval mode: no dropout, no sampling, no stochastic operations in forward pass
- Prompt selection is deterministic (first N from fixed list, not shuffled by seed)
- SVD computation is deterministic (no randomness)
- All seeds produce identical results because the entire pipeline is deterministic
- The test validates only that the code is deterministic — provides zero information about robustness to prompt sampling variability
- seed 42 files lack suffix, other seeds check for `_seed{N}` suffix first, then fall back to unsuffixed file — but this fallback does not explain the identical results (each seed correctly runs its own subprocess)

### A7: Mistral Model Variant Shift
- Early work (Dec 2025): `Mistral-7B-Instruct-v0.2` (instruct-tuned), d = -3.558
- Later work (Feb-Mar 2026): `Mistral-7B-v0.1` (base model), d = -2.259
- These are different models; results are not directly comparable
- The paper should specify which variant is reported

### A8: RunPod Code Version
- No record of which git commit was deployed to RunPod for each experiment
- `scripts/runpod/` sync scripts don't record git hashes
- Cannot verify RunPod used same PR implementation as local

### A9: Six PR Implementations
- Six `participation_ratio` implementations exist across codebase
- Import chain tracing: all 32 active pipelines use `src/metrics/rv.py`
- Other 5 remain in codebase (legacy, archive, geometric_lens)
- If any experiment accidentally imported a different implementation, results would differ
- No evidence this occurred, but cannot rule it out for RunPod experiments

### A10: Prompt Bank Count Discrepancy
- `n300_mistral_test_prompt_bank.py`: 320 prompts (name says 300)
- `prompts/bank.json`: 754 prompts (SHA256 version `2ac959a313614329`)
- The cross-architecture runs reference version `75e7c1b8dcebc24e`
- These are different versions of the prompt bank
- The 754-prompt bank is a superset, but the versioning hash doesn't appear in cross-architecture results

### A11: Qwen2.5-7B Registry Layer Count Bug
- `geometric_lens/models.py` line 219: `"Qwen/Qwen2.5-7B"` registered with `num_layers=32`
- Actual Qwen2.5-7B model has **28 layers** (not 32)
- Hardcoded `early_layer=5, late_layer=27` in registry
- At 28 layers, layer 27 is at **96.4% depth** (27/28), not the intended ~84%
- Cross-architecture experiment (Feb 2026) correctly used `early=4, late=24` from its own config
- Power-up experiment used GeometricProbe which reads from this registry — measured at wrong depth position
- This may explain the inconsistency between cross-arch (d=-0.719, contraction) and power-up (d=-2.318, stronger contraction) for Qwen

### A12: Three Independent Prompt Corpora
- Three distinct prompt sources used across experiments, never cross-referenced:
  1. `prompts/bank.json` via PromptLoader: 754 prompts, SHA256-versioned `75e7c1b8dcebc24e`, used by canonical pipelines
  2. Inline `RECURSIVE_PROMPTS` in `power_up_multiseed.py`: 93 recursive + 83 baseline, mechanistic/technical themed
  3. Inline prompts in `scaling_gap_sweep.py`: 40 recursive + 30 baseline, different set
- The prompt populations have different thematic character (curated contemplative L3/L4/L5 vs mechanistic/technical)
- When n changes from 45 to 80, the extra prompts come from a DIFFERENT corpus, not the same bank

### A13: RunPod Dual-Path Deployment
- Most RunPod scripts use `/workspace/mech-interp-latent-lab-phase1` (full repo name)
- BUT `run_bonus_scaling.sh`, `run_e42_e13_retry.sh`, `run_remaining_experiments.sh` use `/workspace/mech-interp` (shortened)
- These are the March 2026 scripts that generated scaling_gap and power_up results
- `PYTHONPATH=/workspace/mech-interp` means imports came from a differently-named directory
- Cannot confirm whether `/workspace/mech-interp` was a symlink or a separate (older?) copy of the repo
- If separate, the PR implementation and ModelSpec registry could differ from local repo

### A14: n=300 Behavioral Transfer — Layer Specificity Failure
- `docs/misc/neurips_n300_summary.md`: n=300 pairs, dated Dec 12, 2025
- Method: Full KV cache + Persistent V_PROJ patching
- L27 behavioral transfer: d=0.63, p=9.89e-24
- **Wrong-layer (L21) behavioral transfer: d=0.65, p=1.54e-24**
- **L27 vs L21 comparison: t=0.07, p=0.944 — STATISTICALLY IDENTICAL**
- This was flagged in commit `ea3e8bb` and by multiple agent reviews (Opus, Gemini)
- Contradicts the n=45 causal validation where L21 showed null geometric effect (p=0.49)
- Resolution: L21 has null effect on R_V GEOMETRY but equal effect on BEHAVIOR via the Full KV cache component

---

## CONFLICT PAIRS

### C1: OPT-6.7B -- Cross-Architecture vs Power-Up

**Cross-architecture (Feb 2, 2026)**:
- Pipeline: `rv_l27_causal_validation.py`
- N=45, Prompt bank: `75e7c1b8`, early=4, late=27
- delta_main = -0.3603, d = -1.836
- R_V_recursive < R_V_baseline (**CONTRACTION**)

**Power-up (Mar 4, 2026)**:
- Pipeline: `power_up_multiseed.py`
- N=80, Prompt bank: unknown, layers: auto-derived
- rv_recursive = 1.115, rv_baseline = 0.789, d = +1.683
- R_V_recursive > R_V_baseline (**EXPANSION**)

**Differences**: Different pipelines, different prompts, different N, possibly different layer indices. Not the same experiment.

---

### C2: GPT2-XL -- Cross-Architecture vs Power-Up

**Cross-architecture (Feb 2, 2026)**:
- N=45, Prompt bank: `75e7c1b8`, early=6, late=40 (83% depth)
- delta_main = -0.1376, d = -1.143 (**CONTRACTION**)

**Power-up (Mar 4, 2026)**:
- N=80, Prompt bank: unknown, late likely=43 if `num_layers-5` (90% depth)
- rv_recursive = 0.872, rv_baseline = 0.711, d = +1.516 (**EXPANSION**)

**Note**: 3-layer difference in late index (40 vs 43) changes measurement from 83% to 90% depth. Also different prompts.

---

### C3: Qwen2.5-7B -- Cross-Architecture vs Power-Up (CONSISTENT)

- Cross-architecture: d = -0.719 (contraction)
- Power-up: d = -2.318 (stronger contraction)
- Both contraction. Effect STRONGER at n=80. Consistent across experiments.

---

### C4: Mistral-7B -- Instruct vs Base

- Original (Dec 2025, Instruct-v0.2): d = -3.558
- Cross-architecture (Feb 2026, v0.1 base): d = -2.259
- Both strong contraction. d differs by 1.3. Different model variants.

---

### C5: Scaling Gap Positive d Values

- Qwen2.5-3B: d = +1.25 to +1.60 (EXPANSION)
- Phi-3-mini: d = +0.625 (weak EXPANSION)

These use `scripts/scaling_gap.py` with unknown prompts. The cross-architecture validation on larger models shows contraction. Could reflect genuine architecture effect, prompt difference, or pipeline difference.

---

### C6: n=300 Behavioral Transfer — L27 vs L21

**Original n=45 causal validation (Dec 2025)**:
- Wrong-layer (L21) R_V geometric effect: +0.046, p=0.49 — NULL
- Claim: L27 is layer-specific for R_V contraction

**n=300 behavioral transfer (Dec 12, 2025)**:
- L27 behavioral transfer score: 2.62, d=0.63
- L21 behavioral transfer score: 2.61, d=0.65
- L27 vs L21: t=0.07, **p=0.944 — IDENTICAL**
- Claim: behavioral transfer is NOT L27-specific

**Differences**: n=45 measured R_V geometry; n=300 measured behavioral output scores. The Full KV cache component (all 32 layers) was included in both L27 and L21 conditions, which may drive the behavioral effect regardless of which V-proj layer is patched.

---

### C7: Path Patching — Early Residual vs "L27 V-proj" Narrative

**Original December narrative**:
- "L27 V-proj is where contraction happens" (Grand Unified Test, Dec 12)
- L27 KV and V_PROJ achieve PR=4.43; L27 RESIDUAL fails (PR=6.05)
- Conclusion: "At L27, the mechanism is in attention, not residual stream"

**February 27, 2026 path patching sweep**:
- 16 layers x 3 components (residual, v_proj, mlp), n=20 per condition
- **Layer 4 residual: d=1.96** (STRONGEST causal effect on R_V across entire model)
- **V_proj at ALL layers: max |d|=0.22** (negligible causal effect on R_V)
- Layer 27 V-proj: d=-0.02 (essentially zero)
- Layer 18 V-proj: d=+0.19 (negligible)

**Differences**: December tests patched activations FROM recursive into baseline AT specific layers. February tests patched FROM baseline into recursive AT specific layers (break direction). December measured PR at a single layer. February measured full R_V ratio. The December finding that "L27 V-proj works" may reflect the December test's use of KV_CACHE method (which replaces both K and V), while February's V_proj patching replaced only the V-projection output.

---

## MODEL INVENTORY (14 Total)

### Wave 1: Observational Survey (Dec 2025, 6+1 models)
| Model | Contraction | Cohen's d | Architecture |
|-------|------------|-----------|-------------|
| Mistral-7B-Instruct-v0.2 | 15.3% | -3.558 (causal) | Dense, 32L |
| Mixtral-8x7B | 24.3% | ~-1.5 | MoE, 32L |
| Qwen1.5-7B | 9.2% | -- | Dense, 32L |
| Gemma-7B | 3.3% | -- | Dense, 28L |
| Llama-3-8B | 11.7% | -- | Dense, 32L |
| Phi-3-medium | 6.9% | -- | Dense, 32L |
| Pythia-2.8B | 29.8% | -4.51 | Dense, 32L |

### Wave 2: Cross-Architecture Causal Validation (Feb 2026, 5 models)
| Model | d (causal) | p | early | late | Layers |
|-------|-----------|---|-------|------|--------|
| Mistral-7B-v0.1 | -2.259 | significant | 5 | 27 | 32 |
| OPT-6.7B | -1.836 | significant | 4 | 27 | 32 |
| GPT2-XL | -1.143 | significant | 6 | 40 | 48 |
| Qwen2.5-7B | -0.719 | significant | 4 | 24 | 28 |
| Pythia-1.4B | -0.311 | marginal | 3 | 20 | 24 |

### Wave 3: Scaling/Power-Up (Mar 2026)
| Model | Experiment | d | Direction | Prompt Source |
|-------|-----------|---|-----------|--------------|
| Qwen2.5-3B | scaling_gap | +1.25 | expansion | unknown |
| Phi-3-mini | scaling_gap | +0.625 | weak expansion | unknown |
| Pythia-6.9B | scaling_gap | +0.478 | NS | unknown |
| Mistral-7B | power_up n=80 | -1.656 | contraction | RECURSIVE_PROMPTS |
| OPT-6.7B | power_up n=80 | +1.683 | expansion | RECURSIVE_PROMPTS |
| GPT2-XL | power_up n=80 | +1.516 | expansion | RECURSIVE_PROMPTS |
| Qwen2.5-7B | power_up n=80 | -2.318 | contraction | RECURSIVE_PROMPTS |
| Pythia-1.4B | power_up n=80 | -0.006 | null | RECURSIVE_PROMPTS |

---

## PR FORMULA

All 32 active experiment pipelines import from `src/metrics/rv.py`:

```python
def participation_ratio(sigma):
    """PR = (sum(sigma^2))^2 / sum(sigma^4)"""
    s2 = sigma ** 2
    return (s2.sum() ** 2) / (s2 ** 2).sum()
```

Applied to singular values from SVD of V-projection activations (shape: hidden_dim x window_size, typically 4096x16).

R_V = PR(late_layer) / PR(early_layer)

The formula has not changed across git history for `src/metrics/rv.py`.

---

## PHASE SUMMARY

| Phase | Dates | Key Achievement | Models | Statistical Strength |
|-------|-------|----------------|--------|---------------------|
| 0 | Dec 9-13, 2025 | Discovery + causal proof | 7 models (6 observational + Pythia circuit) | d=-3.558 (Mistral), d=-4.51 (Pythia) |
| 1 | Jan 11 - Feb 4, 2026 | Cross-architecture validation | 5 models, n=45 each | d=-2.26 to -0.31, all same prompt bank |
| 2 | Feb 5-27, 2026 | Double dissociation + necessity proof + path patching + self-feeding loop | Mistral primarily | d=3.29 (dual-layer break), OR=33.4; Path: L4 residual d=1.96; Self-feed: attractor does NOT self-sustain |
| 3 | Mar 1-5, 2026 | Scaling + statistical hardening | 3 new models + reruns | FDR 30/36, cluster 10/13, perplexity d=-1.80 |
| 4 | Mar 5-7, 2026 | Audit + paper compilation | -- | Paper v005, 19 figures, this reconstruction |

---

*This document contains facts, provenance chains, and clearly flagged ambiguities only.*
*No recommendations, risk assessments, or action items included.*
