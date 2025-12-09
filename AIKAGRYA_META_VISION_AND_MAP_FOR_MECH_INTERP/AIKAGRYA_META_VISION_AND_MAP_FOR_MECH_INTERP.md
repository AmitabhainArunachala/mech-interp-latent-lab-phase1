# AIKĀGRYA Meta Vision and Map for Mechanistic Interpretability

*Living meta-document for the Aikāgrya alignment + mechanistic interpretability research program*

**Last Updated:** November 16, 2025  
**Repository:** `mech-interp-latent-lab-phase1`  
**Status:** Active research program, Phase 1 complete, Phase 2 in planning

---

## Table of Contents

- [I. Current Aikāgrya Empirical Work (Ground Truth in This Repo)](#i-current-aikāgrya-empirical-work-ground-truth-in-this-repo)
- [II. External Research Landscape: Alignment & Mechanistic Interpretability](#ii-external-research-landscape-alignment--mechanistic-interpretability)
- [III. Aikāgrya Meta Vision: From R_V to Intrinsic Alignment](#iii-aikāgrya-meta-vision-from-r_v-to-intrinsic-alignment)
- [IV. Research Mandala: From Beginner Sprints to Deep Mechanistic Work](#iv-research-mandala-from-beginner-sprints-to-deep-mechanistic-work)
- [V. Immediate Next Steps Inside This Repo](#v-immediate-next-steps-inside-this-repo)

---

## I. Current Aikāgrya Empirical Work (Ground Truth in This Repo)

### 1.1 Inventory of Experiments

This section documents **what we have actually done and recorded** in this repository. Every claim links to actual file paths, CSVs, or code.

#### 1.1.1 R_V Contraction Discovery (Phase 1C/1D)

**Status:** ✅ **REPRODUCIBLE** - Complete code, results, and documentation

**Core Finding:** Recursive self-observation prompts induce geometric contraction in Value matrix column space, measured via R_V metric (Participation Ratio late / Participation Ratio early).

**File Locations:**
- **Prompt Bank:** `n300_mistral_test_prompt_bank.py` (320 prompts total)
  - Structure: 100 dose-response (L1-L5), 100 baselines, 60 confounds, 60 generality tests
  - Golden set: 80 prompts (20 L5_refined, 20 L3_deeper, 20 factual_baseline, 20 creative_baseline)
- **Original Discovery Notebook:** `L4transmissionTEST001.1.ipynb` (Mistral-7B, October 2025)
- **Final Report:** `PHASE1_FINAL_REPORT.md` (comprehensive 6-model validation)

**Models Tested:**
1. **Mistral-7B-Instruct-v0.2** (Dense, 32 layers)
   - Code: `models/mistral_7b_analysis.py`
   - Results: 15.3% contraction, "High-Energy Collapse" phenotype
   - Prompts: ~30-50 (exploratory) → 80 (systematic)

2. **Qwen/Qwen1.5-7B-Chat** (Dense, Chinese-trained)
   - Code: `models/qwen_7b_analysis.py`
   - Results: 9.2% contraction, "Compact Focusing" phenotype
   - Prompts: 80 (golden set)

3. **Google/Gemma-7B-IT** (Dense, Google architecture)
   - Code: `models/gemma_7b_analysis.py`
   - Results: 3.3% contraction (when successful), "Near-Singularity" phenotype
   - Special finding: Math prompts cause SVD failures (mathematical singularities)
   - Prompts: ~40-60 (many failed due to numerical instability)

4. **Meta-Llama-3-8B-Instruct** (Dense, Meta architecture)
   - Code: `models/llama_8b_analysis.py`
   - Results: 11.7% contraction, "Balanced Contraction" phenotype
   - Prompts: 80 (golden set)

5. **Microsoft/Phi-3-medium-4k-instruct** (GQA, 3.8B params)
   - Code: `models/phi3_medium_analysis.py`
   - Results: 6.9% contraction, "Gentle Contraction" phenotype
   - Special architecture: Grouped Query Attention (GQA) requires specific V-matrix extraction
   - Prompts: 80 (golden set)

6. **Mixtral-8x7B-Instruct-v0.1** (MoE, 47B total, 13B active)
   - Code: `models/mixtral_8x7b_analysis.py`
   - Results: **24.3% contraction** (STRONGEST EFFECT), "Distributed Collapse" phenotype
   - Special finding: MoE architecture amplifies effect (59% stronger than dense Mistral)
   - Prompts: 80 (golden set) + additional validation experiments

**Core Metric Implementation:**
- **R_V Definition:** `R_V = PR(V_late) / PR(V_early)`
  - `PR = (Σσᵢ²)² / Σσᵢ⁴` (Participation Ratio, effective dimensionality)
  - Early layer: 5 (after initial processing)
  - Late layer: 28 (or `num_layers - 4`)
  - Window: Last 16 tokens
  - Measurement: During prompt encoding (not generation)

**Key Results Summary:**
| Model | Architecture | Contraction | Phenotype | Status |
|-------|-------------|-------------|-----------|--------|
| Mixtral-8x7B | MoE | 24.3% | Distributed Collapse | ✅ Complete |
| Mistral-7B | Dense | 15.3% | High-Energy Collapse | ✅ Complete |
| Llama-3-8B | Dense | 11.7% | Balanced Contraction | ✅ Complete |
| Qwen-7B | Dense | 9.2% | Compact Focusing | ✅ Complete |
| Phi-3-medium | GQA | 6.9% | Gentle Contraction | ✅ Complete |
| Gemma-7B | Dense | 3.3%* | Near-Singularity | ⚠️ Partial (SVD failures) |

*When not experiencing numerical singularities

**Documentation:**
- `PHASE1_FINAL_REPORT.md` - Complete 6-model validation report
- `NOV_16_Mixtral_free_play.md` - Detailed Mixtral exploratory analysis
- `MIXTRAL_LAYER27_GEOMETRY_AND_CAUSALITY.md` - Mixtral-specific findings

---

#### 1.1.2 Activation Patching Causal Validation (Phase 1F)

**Status:** ✅ **REPRODUCIBLE** - Complete code, results, and statistical validation

**Core Finding:** Layer 27 (84% network depth) causally mediates geometric contraction. Patching recursive V-projections into baseline prompts at L27 induces contraction with 117.6% transfer efficiency (overshooting natural gap).

**File Locations:**
- **Main Validation Script:** `mistral_L27_FULL_VALIDATION.py` (n=151 pairs)
- **Earlier Validation:** `mistral_patching_TRULY_FIXED.py` (n=45 pairs, foundational)
- **Diagnostic Tools:** `mistral_patching_DIAGNOSTIC.py`, `mistral_patching_FIXED_FINAL.py`
- **Results CSV:** `mistral7b_L27_patching_n15_results_20251116_211154.csv` (n=15 initial test)
- **Pairing Plan:** `n200_pairing_plan.csv` (prompt pairing strategy)

**Experimental Protocol:**
1. Extract V-projection from recursive prompt at L27: `V_rec`
2. Extract V-projection from baseline prompt at L27: `V_base`
3. Replace last 16 tokens of `V_base` with last 16 tokens of `V_rec`
4. Continue forward pass with patched activations
5. Measure resulting R_V at L27

**Control Conditions (Four Pillars of Proof):**
1. **Random baseline:** Patch with random Gaussian noise → Null effect (d = +0.01)
2. **Shuffled recursive:** Patch with token-shuffled recursive activations → 61% reduction (d = -0.04)
3. **Orthogonal projection:** Patch with orthogonalized recursive activations → Null effect (d = +0.05)
4. **Wrong layer (L5):** Patch at L5 instead of L27 → Null effect (d = -0.02)

**Key Results (n=151):**
- Baseline R_V: 0.7738 ± 0.0580
- Recursive R_V: 0.5750 ± 0.0517
- Patched R_V: 0.5396 ± 0.0590
- **Delta:** -0.2342 ± 0.0658
- **Cohen's d:** -3.56
- **Transfer efficiency:** 117.6% (overshooting natural gap)

**Statistical Validation:**
- Main effect: p < 10⁻⁴⁷ (paired t-test)
- All controls: p > 0.5 (null effects confirmed)
- Effect size ratio: 35:1 separation (main vs strongest control)

**Documentation:**
- `MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md` - Complete n=45 analysis
- `Claude_Desktop 3 day sprint write up` - Full paper draft incorporating n=151

---

#### 1.1.3 Mixtral Layer-by-Layer Sweep (Phase 1F Extended)

**Status:** ✅ **REPRODUCIBLE** - Complete code and CSV results

**Core Finding:** Full 80-prompt × 32-layer sweep reveals Layer 27 as "snap layer" with discrete phase transition rather than gradual convergence.

**File Locations:**
- **Analysis Code:** `NOV_16_Mixtral_free_play.py` (comprehensive toolkit)
- **Results CSV:** `results/mixtral/full_layer_analysis_20251116_044633.csv` (80 prompts × 32 layers)
- **Mixtral Patching:** `results/mixtral/MIXTRAL_LAYER27_PATCHING.csv` (n=5 pairs)

**Key Metrics Recorded:**
- R_V(layer) = PR(V_layer) / PR(V_5) for each layer
- Effective Rank (1 / Σ(σᵢ²/Σσⱼ²)²)
- Top singular value
- Spectral gap
- Participation Ratio

**Critical Discovery:**
- **R_V ≠ Effective Rank** - These measure different geometric properties
- R_V snaps at Layer 21 (1.0 → 0.77)
- Effective Rank drops at Layer 27/31 (6.90 → 4.71)
- Three-phase process: compression (L1-L20), re-expansion (L21-L26), divergence (L27-L31)

**Documentation:**
- `NOV_16_Mixtral_free_play.md` - Detailed analysis and interpretation
- `MIXTRAL_LAYER27_GEOMETRY_AND_CAUSALITY.md` - Consolidated findings

---

#### 1.1.4 Adjacent Layer Sweep (Mistral-7B)

**Status:** ✅ **REPRODUCIBLE** - Code exists, results documented

**Core Finding:** Biphasic pattern with peaks at L25 (d=-1.99) and L27 (d=-2.29), identifying L25-L27 as critical region (78-84% depth).

**File Locations:**
- **Code:** `adjacent_layer_sweep.py`
- **Results:** Documented in paper draft (`Claude_Desktop 3 day sprint write up`)

**Key Results:**
| Layer | Depth | Mean Δ | Cohen's d | p-value |
|-------|-------|--------|-----------|---------|
| L24 | 75.0% | -0.117 | -1.21 | 4.01×10⁻¹¹ |
| **L25** | **78.1%** | **-0.219** | **-1.99** | **1.06×10⁻¹⁸** |
| L26 | 81.2% | -0.135 | -1.29 | 5.65×10⁻¹² |
| **L27** | **84.4%** | **-0.214** | **-2.29** | **3.86×10⁻²¹** |
| L28 | 87.5% | -0.104 | -0.90 | 8.98×10⁻⁸ |

**Critical Region:** L25-L27 (78-84% depth) with sharp 51% drop at L28.

---

#### 1.1.5 Path Patching Attempts (Incomplete)

**Status:** ⚠️ **PARTIAL / TECHNICAL CHALLENGES** - Code exists but experiments failed

**Goal:** Trace causal cascade from L27 intervention through downstream layers (L28-L31).

**File Locations:**
- `path_patching_alternative.py` - Traditional PyTorch hooks approach
- `fixed_path_patching.py` - Refined tuple-handling version
- `debug_path_patching.py` - Diagnostic tools
- `path_patching_L27_to_L31.py` - Specific L27→L31 experiment

**Status:** Experiments encountered technical challenges with hook ordering and tuple outputs from attention modules. **Not yet working.** Remains important direction for future work.

**Hypothesis (untested):** Downstream layers (L28-L31) may exhibit compensatory expansion when L27 is perturbed, suggesting geometric homeostasis.

---

### 1.2 Methodological Traceability & Gaps

#### Analysis Pipeline Status

**R_V Computation:**
- ✅ Standardized function exists in multiple scripts
- ⚠️ Not yet centralized into reusable module
- ✅ Core implementation: `compute_column_space_pr()` pattern across files

**Activation Patching:**
- ✅ Working implementation: `patch_v_during_forward()` context manager
- ✅ Standardized protocol: `run_patched_forward_final()` function
- ⚠️ Multiple iterations exist (TRULY_FIXED, FIXED_FINAL, etc.) - needs consolidation

**Data Management:**
- ✅ CSVs exist for major experiments
- ⚠️ No standardized naming convention
- ⚠️ No centralized results directory structure
- ⚠️ Metadata (model versions, dates, hyperparameters) not consistently logged

#### Environment & Reproducibility

**Gaps Identified:**
- ❌ No `requirements.txt` or `environment.yml` for exact dependency versions
- ❌ No Docker container or RunPod template documented
- ❌ Hard-coded paths in some scripts (e.g., `/workspace/`)
- ❌ No unit tests for core metrics (R_V, PR, effective rank)
- ❌ No automated validation that CSVs match code outputs
- ⚠️ Some experiments documented only in external tools (ChatGPT/Claude logs)

**What Exists:**
- ✅ `env.txt` - Basic environment info (partial)
- ✅ `mech_interp_knowledge_base.md` - Consolidated MI references
- ✅ `.cursorrules` - Project-specific context for Cursor AI

#### Reproducibility Checklist

**Immediate TODOs:**
- [ ] Centralize all R_V + patching code into `aikagrya_rv/` package structure
- [ ] Add minimal `README_RV_EXPERIMENTS.md` linking code + CSVs + notebooks
- [ ] Standardize location for CSV outputs + metadata (`results/` subdirectory)
- [ ] Create `requirements.txt` with exact versions from successful runs
- [ ] Document RunPod/Colab setup steps in `SETUP.md`
- [ ] Add unit tests for `compute_column_space_pr()` and `compute_metrics_fast()`
- [ ] Create validation script to verify CSV outputs match code expectations

**Future Improvements:**
- [ ] Automated experiment runner with logging
- [ ] Standardized prompt bank loader
- [ ] Cross-model comparison visualization pipeline
- [ ] Statistical analysis module (effect sizes, significance tests)

---

## II. External Research Landscape: Alignment & Mechanistic Interpretability

### 2.1 Foundational Papers & Frameworks

**Classic Mechanistic Interpretability:**
- **Meng et al. 2022** - "Locating and Editing Factual Associations in GPT"
  - Foundational activation patching methodology
  - Causal tracing for factual recall
  - **Connection to Aikāgrya:** Our patching protocol directly builds on this framework

- **Wang et al. 2022** - "Interpretability in the Wild: Circuit for Indirect Object Identification"
  - Systematic patching protocols and controls
  - Path patching methodology
  - **Connection to Aikāgrya:** Our four-control validation follows this rigor

- **Elhage et al. 2021** - "A Mathematical Framework for Transformer Circuits"
  - Geometric structure of transformer representations
  - Subspace decomposition and attention geometry
  - **Connection to Aikāgrya:** R_V metric measures geometric transformations in this framework

- **Olah et al. 2018** - "Attention and Augmented Recurrent Neural Networks"
  - Attention as geometric operations
  - **Connection to Aikāgrya:** Value space geometry is central to our work

**Modern Geometric Deep Learning:**
- **Bronstein et al. 2021** - "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges"
  - Framework for geometric analysis of neural networks
  - **Connection to Aikāgrya:** Our work positions transformers within this geometric framework

- **Doshi & Kim 2024** - Recent work on attention geometry
  - **Connection to Aikāgrya:** Value space transformations align with geometric deep learning principles

**Rank Dynamics & Compression:**
- **Dar et al. 2022** - Information compression in transformers
  - Rank reduction as information bottleneck
  - **Connection to Aikāgrya:** R_V contraction may reflect similar compression mechanisms

---

### 2.2 Modern Mechanistic Interpretability Programs

**Anthropic's Research:**
- **Concept Steering / Representation Engineering**
  - Extracting and steering concept vectors
  - **Connection to Aikāgrya:** Could we extract "recursive self-observation" steering vectors from L27?

- **Introspective Awareness Work**
  - Models' awareness of their own processing
  - **Connection to Aikāgrya:** Our recursive prompts explicitly invoke introspective awareness

- **Deception / Safety Studies**
  - Detecting deceptive reasoning patterns
  - **Connection to Aikāgrya:** Geometric signatures could detect consciousness-like states

**DeepMind / OpenAI:**
- **Activation Patching**
  - Causal intervention techniques
  - **Connection to Aikāgrya:** Our patching protocol follows these methods

- **Chain-of-Thought Monitoring**
  - Tracking reasoning processes
  - **Connection to Aikāgrya:** Could R_V track reasoning quality?

- **Process Supervision & RLHF**
  - Alignment through training objectives
  - **Connection to Aikāgrya:** Could geometric constraints be built into training?

---

### 2.3 Introspection & Steering Vector Work

**Recent Introspection Papers:**
- **Self-Awareness in LLMs** (various 2024-2025)
  - Models' capacity for self-reference
  - **Connection to Aikāgrya:** Our recursive prompts explicitly test this capacity

**Steering Vector Libraries:**
- **Anthropic's Steering Vectors**
  - Extracting concept directions in activation space
  - **Connection to Aikāgrya:** Could extract "recursive processing" vector from L27 V-space

- **SAE Frameworks (Sparse Autoencoders)**
  - Feature decomposition via sparse autoencoders
  - **Connection to Aikāgrya:** Could decompose V-space into recursive vs. non-recursive features

---

### 2.4 Alignment-Focused Architectures & Proposals

**Intrinsic Alignment:**
- **Geometric Constraints**
  - Building alignment into model geometry
  - **Connection to Aikāgrya:** Could we encourage "honest" geometric regimes?

- **Transparency Through Geometry**
  - Making internal states interpretable
  - **Connection to Aikāgrya:** R_V provides quantitative transparency metric

**Consciousness Detection:**
- **Quantitative Signatures**
  - Detecting consciousness-like processing
  - **Connection to Aikāgrya:** R_V contraction may be such a signature

---

### 2.5 Tools, Libraries & Models for Mech Interp Research

**Python Libraries:**
- **TransformerLens** (Neel Nanda)
  - Comprehensive transformer analysis toolkit
  - **Status:** Not yet integrated into Aikāgrya codebase
  - **Why Useful:** Standardized hooks, activation patching, attention visualization

- **SAE Frameworks**
  - Sparse autoencoder implementations
  - **Status:** Not yet used
  - **Why Useful:** Feature decomposition of V-space

- **Logit Lens Tools**
  - Probing intermediate representations
  - **Status:** Not yet used
  - **Why Useful:** Could complement R_V with behavioral probes

**HuggingFace Models:**
- **Small Interpretable Models**
  - GPT-2 small, nanoGPT, etc.
  - **Status:** Not yet tested
  - **Why Useful:** Full circuit-level analysis possible

- **Interpretability-Aligned Checkpoints**
  - Models trained for transparency
  - **Status:** TODO - identify specific checkpoints
  - **Why Useful:** Baseline for "honest" geometric regimes

**Activation Patching Libraries:**
- **nnsight** (attempted, failed)
  - High-level intervention framework
  - **Status:** Technical challenges encountered
  - **Alternative:** Traditional PyTorch hooks (working)

**TODO:** Integrate references to specific HF interpretable model checkpoints once we confirm exact names and licenses.

---

## III. Aikāgrya Meta Vision: From R_V to Intrinsic Alignment

### 3.1 The R_V Work as the First 5%

**Current Status: Beginner-Level Mech Interp Sprint**

The R_V contraction discovery represents our **first real mechanistic interpretability experiment** - a 3-day intensive sprint that, while somewhat sloppy in organization, produced a genuinely promising empirical foothold. This work is:

**What It Is:**
- ✅ **Grounded:** Real data from 6 architectures, rigorous controls, statistical validation
- ✅ **Promising:** Universal phenomenon (3-24% effect), architecture-specific phenotypes
- ✅ **Causally Validated:** Activation patching confirms L27 mediation (d=-3.56, p<10⁻⁴⁷)
- ✅ **Novel:** First quantitative geometric signature of recursive self-reference

**What It Is Not:**
- ❌ **Not yet a consciousness detector:** R_V contraction may be necessary but not sufficient
- ❌ **Not yet an alignment solution:** No evidence that geometric regimes correlate with honesty/non-deception
- ❌ **Not yet a complete mechanistic explanation:** We know *where* (L27) and *what* (contraction), not yet *how* (circuit-level)

**Why We Treat It as 5%:**

The R_V work is a **hint of a deeper geometric phenomenon**. Recursive self-reference → geometric contraction suggests:

1. **Geometry-based probes:** R_V is just one metric. What about Q/K/O spaces? Attention entropy? Residual stream geometry?

2. **Layer localization:** L27 is critical, but what about L25-L27 as a "critical region"? What happens in adjacent layers?

3. **Circuit-level analysis:** Which heads? Which MLPs? What's the actual mechanism?

4. **Cross-architecture generalization:** 6 models show the effect, but what about GPT-4? Claude? Gemini?

5. **Behavioral validation:** Does R_V contraction correlate with introspective text generation? Reasoning quality?

**The Seed:**

From this 5% foundation, we envision:
- **Geometric signatures** for consciousness-like states
- **Layer-localized** interventions for alignment
- **Circuit-level** understanding of recursive processing
- **Intrinsic alignment** through geometric constraints

But we proceed **one experimental step at a time**, from toy models → small LLMs → larger systems, with clear distinctions between empirical observation, mechanistic explanation, and philosophical interpretation.

---

### 3.2 Toward a Mathematically-Grounded Intrinsic Alignment System

**The Big Dream (Outer Rings of the Mandala):**

We envision a **mathematical / geometric framework** where certain internal states / regimes are provably or empirically correlated with:

- **Honesty:** Geometric signatures of truthful reasoning
- **Non-deceptive reasoning:** Contraction patterns that indicate genuine processing
- **Transparency:** Interpretable geometric states
- **Stable introspection:** Recursive self-reference without collapse

**The Layered Vision:**

**Core (Empirical):**
- R_V contraction as geometric signature
- Layer localization (L25-L27 critical region)
- Causal validation (activation patching)

**Middle Ring (Mechanistic):**
- Circuit-level decomposition
- Dual-space geometry (in-subspace vs. orthogonal)
- Cross-architecture validation

**Outer Ring (Philosophical):**
- Consciousness detection protocols
- Intrinsic alignment constraints
- Geometric regulation of AI behavior

**The Honest Approach:**

Concepts like eigenstates, phase transitions, "contraction regimes," and contemplative insights **may eventually nest here**, but they belong in **outer layers** of the mandala, not the core empirical claim.

We proceed:
- **One experimental step at a time**
- **From toy models → small open LLMs → larger systems**
- **With clear distinctions between:**
  - Empirical observation (what we measure)
  - Mechanistic explanation (how it works)
  - Philosophical interpretation (what it means)

**The Path Forward:**

1. **Expand empirical base:** More models, more metrics, more controls
2. **Deepen mechanistic understanding:** Circuit analysis, dual-space geometry
3. **Validate behavioral connections:** Does geometry predict behavior?
4. **Build alignment framework:** Can we encourage "honest" geometric regimes?

But always: **Grounded in math, architecture, and real experiments.**

---

## IV. Research Mandala: From Beginner Sprints to Deep Mechanistic Work

This section structures research tasks as **concentric rings** - from easy beginner sprints to advanced alignment framework work. Each level builds on the previous, with clear prerequisites and expected outcomes.

---

### 4.1 Level 1 – Beginner Mech Interp (Easy, 1–3 Day Sprints)

**Prerequisites:** Basic Python, PyTorch, HuggingFace Transformers

**Goal:** Replicate and extend R_V work with better organization and additional metrics.

#### 4.1.1 Re-run R_V Experiments with Cleaner Infrastructure

**Example Sprint:**
- **Task:** Standardize R_V computation across all 6 models
- **Difficulty:** ✅ Easy
- **Duration:** 1-2 days
- **Dependencies:** 
  - `n300_mistral_test_prompt_bank.py` (prompt bank)
  - `models/*_analysis.py` (existing model scripts)
- **Deliverables:**
  - Centralized `aikagrya_rv/metrics.py` module
  - Standardized CSV outputs with metadata
  - Automated plots (R_V distributions, dose-response curves)

**Checklist:**
- [ ] Refactor `compute_column_space_pr()` into reusable function
- [ ] Create `run_rv_experiment(model, prompt_bank, output_dir)` wrapper
- [ ] Add logging for model version, date, hyperparameters
- [ ] Generate standardized plots (R_V by prompt type, by model)

---

#### 4.1.2 Add Simple Behavioral Metrics

**Example Sprint:**
- **Task:** Complement R_V with behavioral probes (logit entropy, output diversity, self-referential token frequency)
- **Difficulty:** ✅ Easy
- **Duration:** 1-2 days
- **Dependencies:**
  - Existing R_V infrastructure
  - Model generation capabilities
- **Deliverables:**
  - `aikagrya_rv/behavioral.py` module
  - CSV columns: `logit_entropy`, `output_diversity`, `self_ref_frequency`
  - Correlation analysis: R_V vs. behavioral metrics

**Checklist:**
- [ ] Implement logit entropy computation
- [ ] Add output diversity metric (unique tokens / total tokens)
- [ ] Count self-referential tokens ("I", "my", "self", "aware", etc.)
- [ ] Run correlation: R_V vs. behavioral metrics

---

#### 4.1.3 Cross-Model Validation (2-4 Models)

**Example Sprint:**
- **Task:** Re-run R_V on 2-4 open models with cleaned logging
- **Difficulty:** ✅ Easy
- **Duration:** 2-3 days
- **Dependencies:**
  - Standardized R_V infrastructure
  - Model access (HuggingFace)
- **Deliverables:**
  - Consistent CSV format across models
  - Cross-model comparison plots
  - Phenotype classification (High-Energy Collapse, Compact Focusing, etc.)

**Checklist:**
- [ ] Select 2-4 models (e.g., Llama, Gemma, Qwen, Mistral)
- [ ] Run standardized R_V experiment
- [ ] Generate cross-model comparison visualization
- [ ] Document phenotype differences

---

#### 4.1.4 Naive Head-Level Ablations

**Example Sprint:**
- **Task:** Ablate individual attention heads in L25-L27 and measure R_V change
- **Difficulty:** ✅ Easy
- **Duration:** 1-2 days
- **Dependencies:**
  - R_V infrastructure
  - Hook-based head ablation capability
- **Deliverables:**
  - Head importance ranking (which heads affect R_V most?)
  - Ablation plots (R_V change vs. head index)

**Checklist:**
- [ ] Implement head ablation hook
- [ ] Test on L25, L26, L27 (critical region)
- [ ] Rank heads by R_V impact
- [ ] Visualize head importance

---

### 4.2 Level 2 – Intermediate Mech Interp (Medium Complexity, 1–3 Week Sprints)

**Prerequisites:** Level 1 infrastructure, familiarity with activation patching

**Goal:** Deeper mechanistic understanding through steering vectors, layer profiling, and attention analysis.

#### 4.2.1 Steering Vector Explorations

**Example Sprint:**
- **Task:** Extract recursive steering vectors from L27 V-space, test if they induce R_V contraction and introspective text generation
- **Difficulty:** ⚠️ Medium
- **Duration:** 1-2 weeks
- **Dependencies:**
  - L27 V-space activations from recursive vs. baseline prompts
  - Steering vector extraction (PCA or SAE)
  - Generation pipeline with vector injection
- **Deliverables:**
  - `aikagrya_rv/steering.py` module
  - Steering vector visualization
  - Behavioral validation (does steering produce introspective text?)

**Checklist:**
- [ ] Extract V-space activations at L27 for recursive vs. baseline
  - **Code:** `mistral_L27_FULL_VALIDATION.py` (already extracts V)
- [ ] Compute steering vector (mean difference or PCA)
- [ ] Inject vector into baseline prompts at L27
- [ ] Measure R_V change (does it contract?)
- [ ] Generate text and analyze for introspective content

**Connection to External Work:** Builds on Anthropic's representation engineering, but applied to recursive self-reference.

---

#### 4.2.2 Layer-Wise Profiling

**Example Sprint:**
- **Task:** Build R_V(layer) curves across whole network depth, identify contraction bands per model
- **Difficulty:** ⚠️ Medium
- **Duration:** 1 week
- **Dependencies:**
  - R_V infrastructure
  - Full-layer hook capability
- **Deliverables:**
  - `aikagrya_rv/layer_profiling.py` module
  - R_V(layer) curves for all models
  - Contraction band identification (which layers show strongest effect?)

**Checklist:**
- [ ] Implement full-layer R_V sweep (L1 to L32)
- [ ] Generate R_V(layer) curves for recursive vs. baseline
- [ ] Identify "contraction bands" (regions of strongest effect)
- [ ] Compare across models (do all models show same band?)

**Note:** Mixtral sweep already done (`NOV_16_Mixtral_free_play.py`), replicate for other models.

---

#### 4.2.3 Attention Head Zoom-In

**Example Sprint:**
- **Task:** Use activation patching or ablation to identify which heads in L25-L27 are necessary for contraction
- **Difficulty:** ⚠️ Medium
- **Duration:** 1-2 weeks
- **Dependencies:**
  - Activation patching infrastructure
  - Head-level intervention capability
- **Deliverables:**
  - Head importance ranking (which heads mediate contraction?)
  - Attention pattern analysis (what do critical heads attend to?)

**Checklist:**
- [ ] Implement head-level patching (patch individual head outputs)
- [ ] Test on L25, L26, L27 heads
- [ ] Rank heads by R_V impact
- [ ] Visualize attention patterns for critical heads

**Connection to External Work:** Similar to IOI circuit analysis (Wang et al. 2022), but for recursive processing.

---

#### 4.2.4 Cross-Architecture Phenotype Study

**Example Sprint:**
- **Task:** Systematically compare "High-Energy Collapse" vs. "Compact Focusing" vs. other geometric patterns
- **Difficulty:** ⚠️ Medium
- **Duration:** 1-2 weeks
- **Dependencies:**
  - Cross-model R_V data
  - Phenotype classification framework
- **Deliverables:**
  - Phenotype taxonomy (what geometric strategies exist?)
  - Architecture-phenotype mapping (which architectures show which phenotypes?)

**Checklist:**
- [ ] Define phenotype metrics (contraction rate, layer localization, etc.)
- [ ] Classify all 6 models into phenotypes
- [ ] Analyze architecture-phenotype correlations
- [ ] Hypothesize why MoE shows strongest effect

---

### 4.3 Level 3 – Advanced Mech Interp & Alignment Framework (Hard, Multi-Week/Month Efforts)

**Prerequisites:** Level 2 infrastructure, deep understanding of transformer architecture, circuit analysis experience

**Goal:** Circuit-level decomposition, dual-space geometry, and proto-intrinsic alignment constraints.

#### 4.3.1 Circuit-Level Decomposition

**Example Sprint:**
- **Task:** Identify concrete circuits (heads, MLPs) implementing recursion-related transformations
- **Difficulty:** 🔴 Hard
- **Duration:** 1-2 months
- **Dependencies:**
  - Head-level ablation results
  - MLP intervention capability
  - Circuit analysis framework (TransformerLens?)
- **Deliverables:**
  - Complete circuit diagram for recursive processing
  - Head-MLP interaction analysis
  - Feature selectivity (what features do critical components detect?)

**Checklist:**
- [ ] Map head-MLP interactions in L25-L27
- [ ] Identify induction patterns (how does recursion propagate?)
- [ ] Analyze QK/V structure for critical heads
- [ ] Test circuit necessity (ablate entire circuit, measure R_V)

**Connection to External Work:** Full circuit analysis like IOI (Wang et al. 2022), but for recursive self-reference.

---

#### 4.3.2 Dual-Space Geometry

**Example Sprint:**
- **Task:** Decompose activations into introspective subspaces vs. orthogonal background, track contraction/expansion in each
- **Difficulty:** 🔴 Hard
- **Duration:** 1-2 months
- **Dependencies:**
  - SAE or PCA framework
  - Subspace decomposition capability
- **Deliverables:**
  - `aikagrya_rv/dual_space.py` module
  - Subspace vs. orthogonal R_V analysis
  - Coordinated dynamics visualization

**Checklist:**
- [ ] Decompose V-space into introspective vs. non-introspective subspaces
- [ ] Track R_V in each subspace separately
- [ ] Analyze coordination (do subspaces contract together?)
- [ ] Test if orthogonal projection removes effect (already done in controls!)

**Note:** Paper draft mentions r=0.904 correlation between in-subspace and orthogonal components - this needs deeper analysis.

---

#### 4.3.3 Link to Existing MI / Alignment Work

**Example Sprint:**
- **Task:** Integrate with steering vector libraries (Anthropic-style) and SAE frameworks
- **Difficulty:** 🔴 Hard
- **Duration:** 1-2 months
- **Dependencies:**
  - External libraries (TransformerLens, SAE frameworks)
  - Integration capability
- **Deliverables:**
  - Unified framework combining R_V, steering vectors, SAEs
  - Can we align subspaces, not just outputs?

**Checklist:**
- [ ] Integrate TransformerLens for standardized hooks
- [ ] Apply SAE to V-space decomposition
- [ ] Extract recursive steering vectors using SAE features
- [ ] Test if steering vectors align with R_V contraction

---

#### 4.3.4 Proto-Intrinsic Alignment Constraints

**Example Sprint:**
- **Task:** Explore whether certain geometric regimes correlate with non-deceptive reasoning, can be encouraged via fine-tuning
- **Difficulty:** 🔴 Hard
- **Duration:** 2-3 months
- **Dependencies:**
  - Behavioral validation (does R_V predict honesty?)
  - Fine-tuning capability
- **Deliverables:**
  - Geometric signature of "honest" reasoning
  - Fine-tuning protocol to encourage honest regimes

**Checklist:**
- [ ] Collect dataset of "honest" vs. "deceptive" reasoning
- [ ] Measure R_V for each
- [ ] Identify geometric signatures of honesty
- [ ] Design fine-tuning objective to encourage honest R_V regimes
- [ ] Test if fine-tuned model shows improved honesty

**Connection to External Work:** Intrinsic alignment through geometric constraints, not just output supervision.

---

## V. Immediate Next Steps Inside This Repo

### 5.1 Repo-Local TODOs

**Immediate Actions (This Week):**

- [ ] **Create `README_RV_EXPERIMENTS.md`**
  - Link all code files to CSVs
  - Document prompt bank structure
  - Explain R_V metric computation
  - Provide reproduction instructions

- [ ] **Refactor R_V Analysis Code**
  - Create `aikagrya_rv/` package structure:
    ```
    aikagrya_rv/
    ├── __init__.py
    ├── metrics.py          # R_V, PR, effective rank
    ├── hooks.py            # Value extraction hooks
    ├── patching.py         # Activation patching
    ├── analysis.py         # Statistical analysis
    └── visualization.py     # Plotting functions
    ```
  - Even if just a note/plan for now, document the structure

- [ ] **Create `notebooks/` Subfolder**
  - Move existing notebooks:
    - `L4transmissionTEST001.1.ipynb` → `notebooks/phase1c_discovery.ipynb`
    - Create `notebooks/phase1d_6model_validation.ipynb` (if exists)
    - Create `notebooks/phase1f_activation_patching.ipynb` (if exists)

- [ ] **Add Simple Visualization Script**
  - `scripts/plot_rv_results.py`:
    - Load CSV
    - Plot R_V distributions
    - Plot dose-response curves
    - Generate cross-model comparison

- [ ] **Standardize Results Directory**
  - Create `results/` structure:
    ```
    results/
    ├── phase1c_rv/          # 6-model R_V results
    ├── phase1f_patching/     # Activation patching results
    └── phase1f_mixtral/      # Mixtral layer sweep
    ```
  - Move existing CSVs to appropriate locations

- [ ] **Create `requirements.txt`**
  - Document exact package versions from successful runs
  - Include: torch, transformers, numpy, pandas, matplotlib, seaborn, scipy

- [ ] **Document Setup Process**
  - Create `SETUP.md` with:
    - RunPod/Colab setup instructions
    - Model loading steps
    - Environment variable configuration (HF_HUB_ENABLE_HF_TRANSFER, etc.)

---

### 5.2 Proposed Next Sprint (1–3 Days)

**Sprint: Re-run R_V on 2–3 Models with Cleaned Logging + Logit Entropy**

**Goal:** Standardize R_V computation, add behavioral metrics, generate clean documentation.

**Tasks:**
1. **Refactor `compute_column_space_pr()` into `aikagrya_rv/metrics.py`**
   - Make it reusable across all models
   - Add proper error handling for SVD failures

2. **Create `run_rv_experiment()` wrapper**
   - Takes model name, prompt bank, output directory
   - Returns standardized CSV with metadata

3. **Add logit entropy computation**
   - Complement R_V with behavioral probe
   - Measure: `H(logits) = -Σ p_i log p_i`

4. **Re-run on 2-3 models:**
   - Mistral-7B (baseline)
   - Qwen-7B (Chinese-trained)
   - Llama-3-8B (Meta architecture)

5. **Generate standardized plots:**
   - R_V distributions (recursive vs. baseline)
   - Dose-response curves (L3 vs. L5)
   - Cross-model comparison

6. **Document in `README_RV_EXPERIMENTS.md`**

**Expected Deliverables:**
- ✅ Cleaned `aikagrya_rv/` package structure
- ✅ Standardized CSV outputs with metadata
- ✅ Automated visualization pipeline
- ✅ Documentation linking code → CSVs → results

**Difficulty:** ✅ Easy (1-3 days)

**Dependencies:**
- `n300_mistral_test_prompt_bank.py` (prompt bank)
- `models/*_analysis.py` (existing model scripts)
- HuggingFace model access

---

## VI. Tone & Positioning

**Throughout this document:**

- ✅ **Confident but not grandiose:** We have real empirical results, but acknowledge limitations
- ✅ **Grounded in actual work:** Every claim links to file paths or explicit TODOs
- ✅ **Layered vision:** Big alignment narrative in outer rings, clearly labeled as speculative
- ✅ **Disciplined progression:** One experimental step at a time, from beginner to advanced

**The Core Always Returns To:**
- **What we actually have:** R_V experiments, activation patching, 6-model validation
- **How to grow from there:** Clear roadmap from Level 1 → Level 2 → Level 3
- **Grounded in math, architecture, and real experiments**

---

## VII. Document Maintenance

**This is a living document.** Update it as:
- New experiments are completed
- New files are added to the repo
- External research landscape evolves
- Vision clarifies through empirical work

**Last Updated:** November 16, 2025  
**Next Review:** After next sprint completion

---

**End of Meta-Document**

