Title: TOP FINDINGS LEDGER (DNA → CELL → ANIMAL)
Date: 2025-12-15
Model: gpt-5.2
Repo commit: 295745f3bf17846884dc4d361030e126be2aff54
Prompt bank version: b1e5291421c5646d

## Scope / how to read this

- **Evidence standard**: Only claims grounded in artifacts present in this repo are marked **VERIFIED**. Narrative-only writeups that reference out-of-repo CSVs (e.g., `/workspace/...` or `/Users/.../Desktop/...`) are treated as **UNCERTAIN** unless the referenced data is also present in-repo.
- **“Behavior / expression” caveat**: Several “behavior” metrics in this repo are explicitly heuristic (keyword markers, coarse state labels). Where used, I call out likely confounds (single-sample, seed sensitivity, repetition/degeneracy).

---

## A) Canonical measurement contract check (DNA)

### Canonical R_V definition (as intended)

- **Definition**: \(R_V = \mathrm{PR}(V_{late}) / \mathrm{PR}(V_{early})\)
- **PR definition**: \(\mathrm{PR} = \frac{(\sum s_i^2)^2}{\sum (s_i^2)^2}\) where \(s_i\) are **singular values** (SVD) of the **V-projection window**.
- **Canonical parameters**:
  - **early layer**: 5
  - **late layer**: `num_layers - 5` (often 27 for 32-layer models)
  - **window**: last 16 tokens of the *prompt pass*

**Evidence (contract + code)**:
- Contract doc: `docs/MEASUREMENT_CONTRACT.md`
- Canonical implementation: `src/metrics/rv.py` (`participation_ratio`, `compute_rv`)

### Contract drift / inconsistencies (what differs where)

#### 1) Short-prompt handling is inconsistent between the contract and canonical code

- **Contract claims**: “If prompt length < window_size, return NaN” (`docs/MEASUREMENT_CONTRACT.md`).
- **Canonical code actually does**: uses \(W = \min(\text{window\_size}, T)\) and computes PR on a smaller window (only returns NaN if \(W=0\) or degeneracy) (`src/metrics/rv.py`).
- **Impact**:
  - Results may be **not comparable** across prompt families with different token lengths if some pipelines don’t enforce minimum length.
  - Some pipelines *do* filter by length (example: `src/pipelines/rv_l27_causal_validation.py` filters baselines to token_len >= window; `src/pipelines/behavioral_grounding.py` similarly filters).

**Status**: **VERIFIED** (contract vs code discrepancy is directly visible).

#### 2) At least two PR formulas appear in repo documentation (λ vs λ²)

- Canonical code uses the **singular-value-squared** PR formula (`src/metrics/rv.py`).
- A common writeup uses a **different written formula**: “PR = (Σλᵢ)² / Σλᵢ²” in `R_V_PAPER/research/PHASE1_FINAL_REPORT.md` (Section “1.3 Core Metric: R_V”).

**Status**: **VERIFIED (as a documentation mismatch)**; **UNCERTAIN** whether any *results* depended on the wrong formula (depends on which scripts were actually run).

#### 3) “Late layer” sometimes means “intervention layer” rather than `num_layers-5`

- Canonical “existence” metric is early=5, late=27 (`src/metrics/rv.py`, `docs/MEASUREMENT_CONTRACT.md`).
- Canonical *causal* pipeline explicitly measures at the intervention point: “measure R_V at the measurement layer (usually the patch layer)” (`src/pipelines/rv_l27_causal_validation.py` docstring; uses `target_layer` as measurement layer).

**Status**: **VERIFIED** (definition variant exists and is documented).

#### 4) Multiple non-canonical implementations exist (error handling, effective-rank math, etc.)

Examples:
- `phase1_per_layer_baseline.py`: custom `compute_pr`, returns `0.0` on exceptions (not NaN) and defines “effective rank” via `logdet(Gram)` (not the same as “effective rank” used elsewhere).
- `advanced_activation_patching.py`: custom `compute_pr`, returns `1.0` on exceptions and hard-codes `champ_rv = 0.5088` for “transfer %” calculations.

**Status**: **VERIFIED** (code exists and differs from contract).

---

### Canonical generation parameters (behavior tests)

**Contract** (`docs/MEASUREMENT_CONTRACT.md`):
- Tier 1 (deterministic): `do_sample=False`, `temperature=0.0`, seed fixed (default 42), `max_new_tokens=100`.
- Tier 2 (sampled robustness): `do_sample=True`, `temperature=0.7`, multi-seed (default [42,123,456]).

**Observed in actual run artifacts**:
- `results/kv_sufficiency_matrix/runs/20251215_152312_kv_sufficiency_matrix_mistral7b_instruct_l27_w16_n20/config.json`:
  - `do_sample=true`, `temperature=0.7`, `max_new_tokens=100`, `seed=42` (Tier-2 style).
- `configs/behavioral_grounding.json`:
  - `do_sample=false`, `temperature=0.8`, `max_new_tokens=160`, `seed=9` (Tier-1-ish wrt sampling, but temperature not set to 0.0; also seed not 42).
- `results/phase1_mechanism/runs/20251213_124735_behavioral_grounding_batch_ministral8b_n100_L24_27_W32_sampled_v1/config.json`:
  - `do_sample=true`, `temperature=0.7`, `max_new_tokens=120`, `seed=9` (Tier-2 style).

**Status**: **VERIFIED** (parameters are in repo configs).

---

## B) Top 12 core findings ledger (sorted by leverage/importance)

### 1) R_V contraction survives length-matched and pseudo-recursive confounds (Mistral-7B-v0.1)

- **Claim**: “Champion” recursive prompts have substantially lower \(R_V\) than length-matched and pseudo-recursive controls; pseudo-recursive and length-matched are not significantly different from each other.
- **Scale tag**: **DNA / CELL**
- **Status**: **VERIFIED**
- **Evidence**:
  - Run artifacts:
    - `results/confound_validation/runs/20251215_152231_confound_validation_mistral7b_instruct_l27_w16/summary.json`
    - `results/confound_validation/runs/20251215_152231_confound_validation_mistral7b_instruct_l27_w16/report.md`
    - `results/confound_validation/runs/20251215_152231_confound_validation_mistral7b_instruct_l27_w16/confound_results.csv`
  - Producing pipeline: `src/pipelines/confound_validation.py` (uses `src/metrics/rv.py::compute_rv`)
- **Stats**:
  - N=54 total (18 champions, 18 length_matched, 18 pseudo_recursive) (see `summary.json`)
  - Means:
    - champions: 0.4571
    - length_matched: 0.7666
    - pseudo_recursive: 0.7174
  - Welch t-tests:
    - champions vs length_matched: t=-7.93, p=3.22e-07
    - champions vs pseudo_recursive: t=-11.88, p=3.09e-10
    - length_matched vs pseudo_recursive: t=1.12, p=0.274
- **Replication**:
  - One logged run (see run_dir above). No explicit multi-seed replication in artifacts.
- **Confounds handled**:
  - Length (explicit token-count handling + length-matched controls)
  - “Pseudo-recursive” content (explicit pseudo-recursive control)
- **What would falsify it?**
  - Re-running the same pipeline with a different prompt selection / seed and finding champions no longer differ from length_matched/pseudo_recursive.

### 2) H31 attention features separate recursive vs baseline, but BOS attention is the stronger / cleaner signal than entropy

- **Claim**: At layer 31 head 31 (H31), recursive prompts show much higher BOS attention; entropy separation exists but is weaker with overlap.
- **Scale tag**: **ORGAN / CELL**
- **Status**: **VERIFIED**
- **Evidence**:
  - Data: `results/h31_validation/h31_validation_n50.csv`
  - Producing script: `h31_validation_n50.py` (uses `src/metrics/rv.py::compute_rv`; computes entropy/BOS from attention weights)
  - Narrative summary: `H31_VALIDATION_FINAL_SUMMARY.md` (numbers match the CSV-derived stats below)
- **Stats** (computed from `results/h31_validation/h31_validation_n50.csv`):
  - N=100 (50 recursive, 50 baseline)
  - **R_V (L27)**:
    - recursive mean 0.5046 (std 0.0431)
    - baseline mean 0.9012 (std 0.1567)
    - Welch t=-17.26, p=3.66e-24; Cohen’s d=-3.45
  - **H31 entropy**:
    - recursive mean 0.4299 (std 0.1073)
    - baseline mean 0.5882 (std 0.3902)
    - Welch t=-2.77, p=0.00766; Cohen’s d=-0.55
  - **H31 BOS attention**:
    - recursive mean 0.9381 (std 0.0210)
    - baseline mean 0.8061 (std 0.2041)
    - Welch t=4.55, p=3.46e-05; Cohen’s d=0.91
- **Replication**:
  - Single logged CSV. (No multi-seed runs in-repo for this exact experiment.)
- **Confounds handled**:
  - Prompt diversity: prompts pulled from multiple groups (see `h31_validation_n50.py` prompt selection).
- **What would falsify it?**
  - Re-running `h31_validation_n50.py` on multiple seeds and observing BOS attention separation disappear or invert.

### 3) “Behavioral expression” is fragile: baseline-only control already produces substantial “expression” rate in KV-sufficiency runs

- **Claim**: The repo’s “expression” classifier can label baseline continuations as “recursive_prose” even with no intervention, so expression-rate uplifts must be interpreted cautiously.
- **Scale tag**: **ANIMAL**
- **Status**: **VERIFIED**
- **Evidence**:
  - Pipeline + labeler:
    - `src/pipelines/kv_sufficiency_matrix.py` (`_is_expression` uses `src/metrics/behavior_states.py::label_behavior_state`)
  - Run artifacts:
    - `results/kv_sufficiency_matrix/runs/20251215_152312_kv_sufficiency_matrix_mistral7b_instruct_l27_w16_n20/kv_matrix_results.csv`
    - `results/kv_sufficiency_matrix/runs/20251215_152312_kv_sufficiency_matrix_mistral7b_instruct_l27_w16_n20/summary.json`
  - Example of A_control rows labeled `recursive_prose`: see `kv_matrix_results.csv` (A_control includes `behavior_state=recursive_prose`, `is_expression=True`).
- **Stats**:
  - N=50 pairs; N_rows=350 generations (`summary.json`)
  - Expression rate in A_control: 0.28 (`summary.json`)
- **Replication**:
  - One logged run.
- **Confounds handled**:
  - KV length mismatch explicitly handled (KV only used when seq len matches; see `_generate_with_optional_kv`).
- **What would falsify it?**
  - Tightening `label_behavior_state` and showing A_control expression-rate collapses toward 0 without harming true positive detection.

### 4) Random KV has a surprisingly large “expression rate” uplift, similar to KV-from-recursive (3 seeds)

- **Claim**: In the KV sufficiency matrix, random Gaussian KV (3 seeds) raises “expression rate” nearly as much as KV-from-recursive.
- **Scale tag**: **ANIMAL**
- **Status**: **VERIFIED** (as a measurement); **UNCERTAIN** (as a mechanistic explanation)
- **Evidence**:
  - Run artifacts:
    - `results/kv_sufficiency_matrix/runs/20251215_152312_kv_sufficiency_matrix_mistral7b_instruct_l27_w16_n20/summary.json`
    - `results/kv_sufficiency_matrix/runs/20251215_152312_kv_sufficiency_matrix_mistral7b_instruct_l27_w16_n20/kv_matrix_results.csv`
  - Producing pipeline: `src/pipelines/kv_sufficiency_matrix.py` (random KV created by `_create_gaussian_kv_cache`, matched to baseline KV mean/std per layer)
- **Stats** (`summary.json`):
  - N=50 pairs; N_rows=350
  - Expression rates:
    - A_control: 0.28
    - B_kv_from_recursive: 0.42
    - D_random_kv_seed_101: 0.40
    - D_random_kv_seed_202: 0.38
    - D_random_kv_seed_303: 0.40
- **Replication**:
  - 3 random-KV seeds in the same run; no independent rerun logged.
- **Confounds handled**:
  - KV seq-length matching (explicit).
- **What would falsify it?**
  - Re-running with stricter behavior labeling + deterministic generation (Tier 1) and observing random KV no longer increases “expression.”

### 5) Residual “path patching” shows strong late-layer basin behavior: shuffled≈recursive only emerges at L24–L27

- **Claim**: For residual patching, early layers are “content sensitive” (shuffled ≠ recursive), while late layers (L24–L27) show shuffled≈recursive (“texture not meaning” appears late).
- **Scale tag**: **CELL / ORGAN**
- **Status**: **VERIFIED**
- **Evidence**:
  - Run summary: `results/phase1_mechanism/runs/20251213_080454_path_patching_mechanism_full_early_layer_sweep_full_controls_base/summary.json`
  - Interpretation writeup: `results/phase1_mechanism/runs/20251213_080454_path_patching_mechanism_full_early_layer_sweep_full_controls_base/EARLY_LAYER_MAP.md`
  - Producing pipeline family: `src/pipelines/path_patching_mechanism.py` (referenced by the run_dir name; run artifacts point to `path_patching_mechanism.csv`)
- **Stats**:
  - Rows: 22,400 (`summary.json`)
  - Pairing: `max_pairs=40`, `n_repeats=2`, `windows=[16,32]`, `patch_layers=[0..27]`, `patch_types=[none,recursive,shuffled,opposite,random]` (`summary.json`)
  - Aggregate means (all layers/windows combined; `summary.json`):
    - none: mean 0.6156
    - recursive: mean 0.5337
    - shuffled: mean 0.4655
    - random: mean 0.9401
  - Key qualitative result: “first layers where shuffled≈recursive (within 0.01) are L24–L27 (for both windows)” (`EARLY_LAYER_MAP.md`)
- **Replication**:
  - A second high-N confirmation run is referenced in the writeup: `results/phase1_mechanism/runs/20251213_090121_path_patching_mechanism_early_layers_deep_base/` (`EARLY_LAYER_MAP.md`). (This ledger does not ingest its stats directly; it is an additional in-repo artifact to consult.)
- **Confounds handled**:
  - Random / shuffled / opposite controls included (see `summary.json`).
- **What would falsify it?**
  - Running the same pipeline with different prompt families and finding shuffled≈recursive in early layers (≤23).

### 6) Confound validation also detects a non-trivial length correlation for R_V

- **Claim**: Token count is positively correlated with \(R_V\) in the confound-validation dataset.
- **Scale tag**: **DNA**
- **Status**: **VERIFIED**
- **Evidence**:
  - `results/confound_validation/runs/20251215_152231_confound_validation_mistral7b_instruct_l27_w16/summary.json`
  - Producing pipeline: `src/pipelines/confound_validation.py` (`corr_token_count_vs_rv`)
- **Stats**:
  - N=54 (see Finding #1)
  - Pearson r=0.371, p=0.00570 (`summary.json`)
- **What would falsify it?**
  - A larger confound-validation dataset showing near-zero correlation after stricter length control.

### 7) “Champion” R_V@L27 is not unique to one prompt: paraphrase families yield similar contraction

- **Claim**: Multiple paraphrase families achieve \(R_V@L27\) near or below the “anchor champion” score, suggesting the effect is not tied to one brittle string.
- **Scale tag**: **CELL**
- **Status**: **VERIFIED**
- **Evidence**:
  - Run artifacts:
    - `results/champion_paraphrase_hunt/runs/20251215_081556_paraphrase_hunt/summary.json`
    - `results/champion_paraphrase_hunt/runs/20251215_081556_paraphrase_hunt/config.json` (script+params)
    - `results/champion_paraphrase_hunt/runs/20251215_081556_paraphrase_hunt/paraphrase_scores.csv`
  - Producing script: `experiment_champion_paraphrase_hunt.py` (declared in config)
- **Stats**:
  - N=73 candidates (n_valid=73) (`summary.json`)
  - Best-of-family \(R_V@27\) (`summary.json`):
    - boundary_dissolution: 0.4789
    - fixed_point: 0.4857
    - explicit_regress: 0.4894
    - anchor_hybrid_l5_math_01: 0.5083
- **Confounds handled**:
  - None beyond fixed measurement settings (seed=42, window=16 in config). This is a selection/screening run.
- **What would falsify it?**
  - Re-running with different seeds/prompts and seeing paraphrases fail to reproduce low R_V.

### 8) Head-level V-projection ablation localizes strongest effects to Layer 27 and reveals GQA “groups of 4” structure

- **Claim**: The strongest head-level deltas cluster at Layer 27 and appear in groups of 4 heads due to GQA/KV sharing.
- **Scale tag**: **ORGAN**
- **Status**: **VERIFIED**
- **Evidence**:
  - Writeup: `V_PROJ_DISCOVERY_RESULTS.md`
  - CSV artifact: `results/head_discovery/v_proj_head_discovery_20251214_091646.csv` (also mirrored at `DEC_14_FINDINGS/v_proj_head_discovery_20251214_091646.csv`)
  - Producing script: `v_proj_head_discovery.py` (method described in writeup)
- **Stats** (from the writeup):
  - Sample: 20 recursive prompts; heads tested: 640 (20 layers × 32 heads) (`V_PROJ_DISCOVERY_RESULTS.md`)
  - Top deltas are all at Layer 27; repeated 4-head patterns attributed to GQA (`V_PROJ_DISCOVERY_RESULTS.md`)
- **What would falsify it?**
  - Running the same ablation method on a non-GQA model and still seeing 4-head identical deltas.

### 9) Mixtral Layer-27 V-patching (n=5 pairs) moves baseline geometry toward the recursive regime, but does not fully match it

- **Claim**: In Mixtral, overwriting baseline’s L27 V tensor with recursive L27 V reduces baseline \(R_V\) substantially (partial transfer).
- **Scale tag**: **CELL / ORGAN**
- **Status**: **VERIFIED** (for the n=5 patching probe)
- **Evidence**:
  - Data: `results/mixtral/MIXTRAL_LAYER27_PATCHING.csv` (also mirrored at `R_V_PAPER/results/mixtral/MIXTRAL_LAYER27_PATCHING.csv`)
  - Analysis note (contains aggregates): `R_V_PAPER/research/MIXTRAL_LAYER27_GEOMETRY_AND_CAUSALITY.md`
- **Stats** (from `R_V_PAPER/research/MIXTRAL_LAYER27_GEOMETRY_AND_CAUSALITY.md`):
  - N=5 pairs (`L5_refined_01..05` paired with `factual_new_01..05`)
  - Means ± std:
    - RV27_rec: 0.429 ± 0.044
    - RV27_base: 1.078 ± 0.066
    - RV27_patched: 0.886 ± 0.086
- **Confounds handled**:
  - None logged beyond the patching manipulation itself; this is a small probe.
- **What would falsify it?**
  - Scaling to N≥40 pairs and finding patched RV does not move toward recursive RV.

### 10) Phase-0 “metric target” check: R_V computed on v_proj and on hidden state are strongly correlated

- **Claim**: Across prompts, \(R_V(\text{v\_proj})\) and \(R_V(\text{hidden})\) are highly correlated; v_proj measurement tracks a broader contraction phenomenon.
- **Scale tag**: **DNA**
- **Status**: **VERIFIED**
- **Evidence**:
  - Run artifacts:
    - `results/phase0_metric_validation/runs/20251213_052612_phase0_metric_targets_default/summary.json`
    - `results/phase0_metric_validation/runs/20251213_052612_phase0_metric_targets_default/report.md`
    - `results/phase0_metric_validation/runs/20251213_052612_phase0_metric_targets_default/phase0_metric_targets.csv`
  - Producing pipeline: `src/pipelines/phase0_metric_targets.py` (uses `src/metrics/rv.py::participation_ratio`)
- **Stats**:
  - N_rows=30 prompts (10 per pillar: baselines, confounds, dose_response)
  - Pearson correlations:
    - rv_vproj vs rv_hidden: r=0.922 (`summary.json`)
    - pr_v_late vs pr_h_late: r=0.880 (`summary.json`)
- **What would falsify it?**
  - Re-running on a different model and finding low correlation (<0.3).

### 11) Kitchen-sink rerun: L4_full prompts show strong late-layer contraction similar to L5_refined in this sample

- **Claim**: In the kitchen-sink rerun (Mistral-7B-Instruct-v0.2), mean \(R_V\) at L27 for `L4_full` is slightly lower than for `L5_refined` in this dataset.
- **Scale tag**: **CELL**
- **Status**: **VERIFIED** (for this run; interpretation may be dataset-specific)
- **Evidence**:
  - Run artifacts:
    - `results/kitchen_sink/runs/20251215_081007_test_kitchen_sink_rerun/config.json`
    - `results/kitchen_sink/runs/20251215_081007_test_kitchen_sink_rerun/kitchen_sink_results_20251215_081007.csv`
  - Producing script: `test_kitchen_sink.py` (declared in config)
- **Stats** (computed from the CSV):
  - N=20 prompts per label for `L4_full` and `L5_refined`
  - Mean ± std (L27):
    - `L4_full`: 0.555 ± 0.0568
    - `L5_refined`: 0.570 ± 0.0733
- **What would falsify it?**
  - Re-running with a broader set of L5 prompts and seeing L5 consistently outperform L4.

### 12) Residual patching “behavioral grounding” often produces repetition/degeneration rather than clean self-reference

- **Claim**: In sampled generation, baseline_patched outputs have sharply higher repetition (repeat_4gram_frac) and lower lexical diversity, even when “self_ref_rate” rises.
- **Scale tag**: **ANIMAL**
- **Status**: **VERIFIED** (as measured); **UNCERTAIN** (as “true behavioral transfer”)
- **Evidence**:
  - Run artifacts:
    - `results/phase1_mechanism/runs/20251213_124735_behavioral_grounding_batch_ministral8b_n100_L24_27_W32_sampled_v1/summary.json`
    - `results/phase1_mechanism/runs/20251213_124735_behavioral_grounding_batch_ministral8b_n100_L24_27_W32_sampled_v1/config.json`
    - `results/phase1_mechanism/runs/20251213_124735_behavioral_grounding_batch_ministral8b_n100_L24_27_W32_sampled_v1/behavioral_grounding_batch.jsonl`
  - Producing pipeline: `src/pipelines/behavioral_grounding_batch.py` (referenced by run_name; artifacts are in run_dir)
- **Stats** (`summary.json`):
  - Model: `mistralai/Ministral-8B-Instruct-2410`
  - N_pairs=65; sampled generation (`do_sample=true`, `temperature=0.7`), window=32
  - Example at L27:
    - baseline: repeat_4gram_frac_mean 0.089, self_ref_rate_mean 0.00152, unique_word_ratio_mean 0.591
    - baseline_patched: repeat_4gram_frac_mean 0.466, self_ref_rate_mean 0.0209, unique_word_ratio_mean 0.435
- **What would falsify it?**
  - A richer behavior metric (human ratings or robust classifiers) showing patched outputs are genuinely more self-observational without increased degeneration.

---

## C) Layer story (CELL)

- **“Where does contraction begin?”**:
  - In residual patching experiments, early depth behaves like a **ramp**, not a single “switch layer” (`results/phase1_mechanism/runs/20251213_080454_path_patching_mechanism_full_early_layer_sweep_full_controls_base/EARLY_LAYER_MAP.md`).
  - A late “control band” appears around **L24–L27** where shuffled≈recursive (within 0.01) and random patches produce expansion (`EARLY_LAYER_MAP.md`; `summary.json`).
- **Is “L27” fixed or `num_layers-5`?**
  - Canonically, late is `num_layers-5` (often 27 for 32 layers) (`docs/MEASUREMENT_CONTRACT.md`, `src/metrics/rv.py`).
  - Several experiments hard-code `late=27` (e.g., `h31_validation_n50.py`, confound_validation run params in `summary.json`).

---

## D) Head/circuit story (ORGAN)

- **GQA aliasing is real and must be accounted for**:
  - V-projection head discovery shows identical deltas across groups of 4 heads, consistent with KV sharing (GQA) (`V_PROJ_DISCOVERY_RESULTS.md`; CSV `results/head_discovery/v_proj_head_discovery_20251214_091646.csv`).
- **H31 as a candidate “register” is not fully clean**:
  - Entropy separation exists but is weaker than early small-N narratives; BOS attention is stronger and more consistent (`results/h31_validation/h31_validation_n50.csv`, `H31_VALIDATION_FINAL_SUMMARY.md`).

---

## E) Behavior / attractor / one-way-door story (ANIMAL)

- **KV cache swaps and random-KV swaps can increase “expression rate,” but baseline already expresses non-trivially**:
  - See Findings #3 and #4; the labeler flags baseline generations as “recursive_prose” at ~28% even without interventions (`results/kv_sufficiency_matrix/...`).
- **Residual patching can change surface-level “self-ref markers,” but often via degeneration**:
  - Patched generations show high repetition and reduced lexical diversity alongside increased marker rate (`results/phase1_mechanism/...behavioral_grounding_batch.../summary.json`).

---

## F) Next moves (ranked)

### 1) Lock a single canonical R_V implementation + short-prompt policy

- **Action**: Decide whether short prompts should be **NaN** (contract) or **min-window** (current `src/metrics/rv.py`). Then update one of:
  - `docs/MEASUREMENT_CONTRACT.md` (contract)
  - `src/metrics/rv.py` (implementation)
- **Why**: This directly affects cross-experiment comparability and confound sensitivity.
- **Success metric**: “same prompt → same number” reproducibility check passes in CI-style script (see contract’s own tests).

### 2) Standardize “behavior” measurement (replace brittle expression heuristics)

- **Action**: Create a behavior evaluation module that reports:
  - repetition/degeneracy metrics (already present),
  - plus a robust classifier or human-rated rubric on a small set (N=50–100).
- **Why**: Current “expression” labels are too permissive (A_control has 28% expression).

### 3) Canonicalize 3 pipelines as a “gold standard suite” and run 3× seeds

Proposed minimal suite:
- **Existence/robustness**: `src/pipelines/confound_validation.py` (run with 3 seeds; increase to N≥60 per condition).
- **Causality / controls**: `src/pipelines/rv_l27_causal_validation.py` (ensure artifacts saved under `results/`; run N≥80 pairs, include wrong-layer + shuffled + random).
- **Mechanism map**: `src/pipelines/path_patching_mechanism.py` (focus on L20–L27 band; confirm shuffled≈recursive onset and random expansion).

### 4) Resolve the “random KV anomaly” with stronger controls

- **Action**: Extend `src/pipelines/kv_sufficiency_matrix.py` with:
  - deterministic generation (Tier 1) + multiple seeds (Tier 2),
  - a stricter expression classifier and a “degeneracy gate” (exclude high repetition outputs).
- **Goal**: Determine whether random KV is causing genuine mode transfer or just destabilizing generation into a labeler-sensitive regime.

### 5) Scale Mixtral causal patching beyond n=5 (or treat it as a pilot)

- **Action**: Reproduce Mixtral L27 patching with N≥40 pairs and log full run artifacts (config + per-pair CSV + summary.json).
- **Why**: The n=5 probe is promising but too small to anchor a major claim.


