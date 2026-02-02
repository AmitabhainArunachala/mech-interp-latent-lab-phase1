# FORENSIC_AUDIT — Mistral-7B Findings (repo-backed only)

**Scope:** This document audits *Mistral-7B* findings in this repo and labels each claim as **VERIFIED**, **UNCERTAIN**, or **CONTRADICTED**.  
**Rule:** A claim is **VERIFIED** only if it is directly supported by a concrete artifact in this repo (CSV/JSON/TXT) and the artifact path is given. Markdown narrative reports are treated as *claims*, not primary evidence, unless they embed raw values that are also present in CSV/JSON/TXT artifacts.

---

## 0) Evidence index (primary artifacts used)

- **R_V contraction (small-N reproduction)**:
  - `/Users/dhyana/mech-interp-latent-lab-phase1/mistral_reproduction_results.json`
  - `/Users/dhyana/mech-interp-latent-lab-phase1/mistral_complete_reproduction.py`
  - `/Users/dhyana/mech-interp-latent-lab-phase1/MISTRAL_REPRODUCTION_REPORT.md` (summary of the JSON + per-prompt values)
- **R_V contraction (larger-N RunPod dataset)**:
  - `/Users/dhyana/mech-interp-latent-lab-phase1/DECEMBER_2025_EXPERIMENTS/DEC8_RUNPOD/01_GEOMETRY_OF_RECURSION/results/full_validation_20251208_130707.csv`
  - `/Users/dhyana/mech-interp-latent-lab-phase1/DECEMBER_2025_EXPERIMENTS/DEC8_RUNPOD/01_GEOMETRY_OF_RECURSION/results/full_validation_20251208_134241.csv`
- **Layer tomography (single-trace sweep)**:
  - `/Users/dhyana/mech-interp-latent-lab-phase1/mistral_relay_tomography_v2.csv`
  - `/Users/dhyana/mech-interp-latent-lab-phase1/tomography_relay_v2.py`
- **Head discovery (v_proj ablation)**:
  - `/Users/dhyana/mech-interp-latent-lab-phase1/results/head_discovery/v_proj_head_discovery_20251214_091646.csv`
  - `/Users/dhyana/mech-interp-latent-lab-phase1/v_proj_head_discovery.py`
- **Attention targeting / BOS comparison (single prompt-pair analyses)**:
  - `/Users/dhyana/mech-interp-latent-lab-phase1/target_acquisition_output.txt`
  - `/Users/dhyana/mech-interp-latent-lab-phase1/target_comparison_output.txt`
  - `/Users/dhyana/mech-interp-latent-lab-phase1/bos_attention_comparison.csv`
  - `/Users/dhyana/mech-interp-latent-lab-phase1/bos_comparison_output.txt`
- **KV cache claims & revisions**:
  - `/Users/dhyana/mech-interp-latent-lab-phase1/KV_PATCHING_HISTORY.md`
  - `/Users/dhyana/mech-interp-latent-lab-phase1/TRUE_KV_CACHE_PATCHING_RESULTS.md` (note: references missing CSV)
- **Retractions / pipeline invalidations**:
  - `/Users/dhyana/mech-interp-latent-lab-phase1/HEAD_DISCOVERY_PROBLEMS.md`

### Missing artifacts referenced by the repo (cannot verify)

- `target_acquisition_comparison.csv` is claimed in `target_comparison_output.txt` and `DEC_14_FINDINGS/COMPREHENSIVE_HEAD_DISCOVERY_SYNTHESIS.md`, but **is not present** in this repo snapshot.
- `true_kv_cache_patching.csv` is claimed in `TRUE_KV_CACHE_PATCHING_RESULTS.md` and by `true_kv_cache_patching.py` config, but **is not present** in this repo snapshot.
- Multiple L8 behavioral result files are referenced in `L8_COMPLETE_ARCHAEOLOGICAL_INVESTIGATION.md` but **are not present** (e.g. `results/dec11_evening/*.csv`, `logs/dec11_evening/*`).

---

## 1) R_V CONTRACTION — what are the actual numbers?

### 1.1 Small-N reproduction run (Dec 11, 2025)

**VERIFIED (for this run):** Recursive prompts have lower \(R_V\) than baseline in this reproduction.

- **Artifact**: `/Users/dhyana/mech-interp-latent-lab-phase1/mistral_reproduction_results.json`
  - **N (recursive)**: 8 (inferred from `RECURSIVE_PROMPTS` length in `/Users/dhyana/mech-interp-latent-lab-phase1/mistral_complete_reproduction.py`)
  - **N (baseline)**: 8 (inferred from `BASELINE_PROMPTS` length in same script)
  - **Recursive mean \(R_V\)**: 0.9588525699536328 (std 0.11297803698740934)
  - **Baseline mean \(R_V\)**: 1.1488566198084325 (std 0.10787766617082367)
  - **Separation (baseline − recursive)**: 0.19000404985479968
  - **Test (as stored)**: t = 3.218121816090927, p = 0.003096623703715938

**VERIFIED (computed from the JSON summary stats):** Approximate Cohen’s d (pooled; recursive − baseline) is **d ≈ -1.72**.  
Note: the script uses `np.std` default (population sd), so this d is an approximation using those stored stds.

- **How to re-run**:
  - Script: `/Users/dhyana/mech-interp-latent-lab-phase1/mistral_complete_reproduction.py`
  - Command: `python3 mistral_complete_reproduction.py`
  - Output: writes `mistral_reproduction_results.json` (same path format as above).

**UNCERTAIN (generalization):** This run is N=8/8 with fixed prompts inside the script; it is not evidence of robustness across prompt banks.

---

### 1.2 Larger-N RunPod “full_validation” CSVs (Dec 8, 2025)

**VERIFIED:** In the `full_validation_20251208_*.csv` artifacts, \(R_V\) is much lower for `group=recursive` than `group=baseline` at both **L22** and **L27**, with very large effect sizes.

**Artifact A**: `/Users/dhyana/mech-interp-latent-lab-phase1/DECEMBER_2025_EXPERIMENTS/DEC8_RUNPOD/01_GEOMETRY_OF_RECURSION/results/full_validation_20251208_130707.csv`  
**Artifact B**: `/Users/dhyana/mech-interp-latent-lab-phase1/DECEMBER_2025_EXPERIMENTS/DEC8_RUNPOD/01_GEOMETRY_OF_RECURSION/results/full_validation_20251208_134241.csv`  
These two CSVs contain the **same \(R_V\) samples** for L22 and L27.

- **Layer L27 (R_V rows)**:
  - **N (recursive)**: 40
  - **N (baseline)**: 40
  - **Recursive mean ± sd**: 0.47061 ± 0.03950
  - **Baseline mean ± sd**: 0.85775 ± 0.09010
  - **Mean diff (baseline − recursive)**: 0.38714
  - **Welch t-test (two-sided)**: t = 24.8891, p = 4.50e-31
  - **Welch t-test (one-sided baseline > recursive)**: p = 2.25e-31
  - **Cohen’s d (pooled; recursive − baseline)**: d = -5.5654

- **Layer L22 (R_V rows)**:
  - **N (recursive)**: 40
  - **N (baseline)**: 40
  - **Recursive mean ± sd**: 0.77803 ± 0.04872
  - **Baseline mean ± sd**: 0.86931 ± 0.08081
  - **Mean diff (baseline − recursive)**: 0.09129
  - **Welch t-test (two-sided)**: t = 6.1187, p = 6.33e-08
  - **Welch t-test (one-sided baseline > recursive)**: p = 3.17e-08
  - **Cohen’s d (pooled; recursive − baseline)**: d = -1.3682

**VERIFIED (producer script located):** These CSVs match the output schema and sample sizes of:

- `/Users/dhyana/mech-interp-latent-lab-phase1/boneyard/DECEMBER_2025_EXPERIMENTS/DEC8_RUNPOD/01_GEOMETRY_OF_RECURSION/code/full_validation_test.py`

In that script:

- `MODEL_NAME = "mistralai/Mistral-7B-v0.1"`
- `N_RECURSIVE = 40`, `N_BASELINE = 40`
- It writes rows in the format `["experiment","layer","group","metric","value"]` and includes `R_V` rows for L22 and L27.

**UNCERTAIN (provenance detail):** The script as written saves to a `/workspace/.../full_validation_<timestamp>.csv` path, while this repo snapshot stores the resulting CSVs under `DECEMBER_2025_EXPERIMENTS/.../results/`. That’s consistent with a copy/rename, but the copy step itself is not recorded in-file.

---

## 2) LAYER TOMOGRAPHY — what’s the layer-by-layer story?

### 2.1 What artifacts exist?

**VERIFIED:** There is a complete layer sweep CSV for a *single* “Champion / Regress / Baseline” prompt trio.

- **Data**: `/Users/dhyana/mech-interp-latent-lab-phase1/mistral_relay_tomography_v2.csv`
- **Generator**: `/Users/dhyana/mech-interp-latent-lab-phase1/tomography_relay_v2.py`

**Important limitation (VERIFIED):** This tomography is **not a distribution over prompts**. It is effectively **N=1 per trace** (one prompt each), so any “phase transition” language is **UNCERTAIN** unless corroborated by a multi-prompt sweep.

---

### 2.2 Gradual change vs phase transition?

**UNCERTAIN:** The single-trace tomography suggests a **late-layer transition band** (large negative Champion-vs-Baseline deltas emerging strongly around the early 20s), but this is not statistically supported beyond the single prompt trio.

- Evidence: In `/Users/dhyana/mech-interp-latent-lab-phase1/mistral_relay_tomography_v2.csv`, Champion vs Baseline deltas become strongly negative by L21 and remain negative through late layers.

---

### 2.3 What happens at L21 (“solution crystallization”)?

**VERIFIED (numbers for this tomography run only):**

From `/Users/dhyana/mech-interp-latent-lab-phase1/mistral_relay_tomography_v2.csv` (row `layer=21`):

- **Champion \(R_V\)**: 0.6944176361155556
- **Regress \(R_V\)**: 0.8012759038412793
- **Baseline \(R_V\)**: 1.0358652702812257
- **Δ(Champion − Baseline)**: -0.3414476341656700
- **Δ(Champion − Regress)**: -0.1068582677257237

**UNCERTAIN (interpretation):** This is sometimes called “crystallization,” but the repo does not contain a multi-prompt layer sweep tying L21 to a robust effect in Mistral-7B.

---

### 2.4 What happens at L27 (alleged peak)?

**VERIFIED (numbers for this tomography run only):**

From `/Users/dhyana/mech-interp-latent-lab-phase1/mistral_relay_tomography_v2.csv` (row `layer=27`):

- **Champion \(R_V\)**: 0.5083598965759920
- **Regress \(R_V\)**: 0.5705873103600673
- **Baseline \(R_V\)**: 0.7099695692409560
- **Δ(Champion − Baseline)**: -0.2016096726649640
- **Δ(Champion − Regress)**: -0.0622274137840752

**VERIFIED (independent corroboration that L27 is “very strong” in another dataset):** The DEC8 RunPod CSVs show a very large recursive-vs-baseline separation at `layer=L27` (Section 1.2), but note those are *different prompts/conditions* than the tomography.

---

### 2.5 Is there any signal in early layers (L0–15)?

**VERIFIED (numbers for this tomography run only):**

From `/Users/dhyana/mech-interp-latent-lab-phase1/mistral_relay_tomography_v2.csv` restricted to layers 0–15:

- **% layers (0–15) with Champion \(R_V\) < Baseline \(R_V\)**: 43.75%
- **Mean(Champion − Baseline) across layers 0–15**: -0.05869486159272389

**UNCERTAIN:** This is N=1 per trace; it cannot establish whether early-layer signal exists generally.

**VERIFIED caveat:** In this tomography method, `EARLY_LAYER = 5` is used as the anchor in `tomography_relay_v2.py`, and the CSV shows `layer=5` yields \(R_V \approx 1.0\) for all traces (a normalization artifact, not a discovery).

---

## 3) HEAD DISCOVERY — validate the H18 & H26 claim

### 3.1 Raw data location

**VERIFIED:**

- **Raw CSV**: `/Users/dhyana/mech-interp-latent-lab-phase1/results/head_discovery/v_proj_head_discovery_20251214_091646.csv`
- **Producer script**: `/Users/dhyana/mech-interp-latent-lab-phase1/v_proj_head_discovery.py`

The script parameters state:

- `N_RECURSIVE = 20`
- `TEST_LAYERS = 8..27` (inclusive)
- `NUM_HEADS = 32`

---

### 3.2 Actual delta for H18 and H26 (from raw CSV)

**VERIFIED (from the CSV rows where `layer=27`):**

- **L27H18**:
  - `rv_baseline` = 0.5350170267921202
  - `rv_ablated` = 0.6265008487998702
  - `delta` = +0.09148382200775
  - `n_samples` = 20

- **L27H26**:
  - `rv_baseline` = 0.5350170267921202
  - `rv_ablated` = 0.6265008487998702
  - `delta` = +0.09148382200775
  - `n_samples` = 20

Source: `/Users/dhyana/mech-interp-latent-lab-phase1/results/head_discovery/v_proj_head_discovery_20251214_091646.csv`

---

### 3.3 Noise floor (random heads) vs H18/H26

**VERIFIED (distribution over all 640 head-tests in the CSV):**

From `/Users/dhyana/mech-interp-latent-lab-phase1/results/head_discovery/v_proj_head_discovery_20251214_091646.csv` using `abs_delta`:

- **Median |delta|**: 0.0008915
- **Mean |delta|**: 0.0037998
- **p90 |delta|**: 0.0050637
- **p95 |delta|**: 0.0144995
- **p99 |delta|**: 0.0666974

**VERIFIED:** H18/H26 at L27 have `|delta| = 0.0914838`, which is **above p99** of the global distribution.

**VERIFIED (within layer 27 only):**

- **Layer 27 |delta| median**: 0.0426815
- **Layer 27 |delta| max**: 0.0914838 (H18/H26 are tied for max, along with other heads—see next section)

---

### 3.4 Critical complication: GQA aliasing (H18 & H26 are not independent in this ablation)

**VERIFIED:** `v_proj_head_discovery.py` explicitly maps query heads to KV heads:

- It notes Mistral uses GQA and sets `kv_head_idx = head_idx % num_kv_heads` (with `num_kv_heads = 8` for Mistral-7B).
- This implies **heads 2, 10, 18, 26** all map to the same KV head index (`% 8 == 2`) and will show identical deltas under this intervention.

Evidence paths:

- `/Users/dhyana/mech-interp-latent-lab-phase1/v_proj_head_discovery.py` (see comments + `kv_head_idx = head_idx % num_kv_heads`)
- `/Users/dhyana/mech-interp-latent-lab-phase1/DEC_14_FINDINGS/V_PROJ_DISCOVERY_RESULTS.md` (explicitly describes the 4-head grouping pattern)

**CONTRADICTED (as phrased):** The claim “H18 & H26 are uniquely responsible” is **not supported** by the v_proj ablation data alone; the v_proj intervention cannot distinguish them from H2/H10 in the same KV group.

**UNCERTAIN:** H18 and H26 can still be “special” as *query heads* (different attention patterns), but the v_proj ablation evidence is KV-head-level, not per-query-head causal isolation.

---

### 3.5 Is “28% vs 0% recursive token targeting” replicable?

**VERIFIED (for one prompt-pair):** The repo contains a recorded console output showing a single prompt-pair comparison.

- **Artifact**: `/Users/dhyana/mech-interp-latent-lab-phase1/target_comparison_output.txt`
  - **H18**: Recursive=26.8% | Baseline=0.0% | Δ=+26.8%
  - **H26**: Recursive=28.9% | Baseline=0.0% | Δ=+28.9%

**UNCERTAIN (replicability):** This is **one** recursive prompt and **one** baseline prompt, with a hand-defined substring list `RECURSIVE_TOKENS`:

- Script: `/Users/dhyana/mech-interp-latent-lab-phase1/compare_targets_baseline.py`
  - Defines `RECURSIVE_TOKENS = ["itself","self","writ","process", ...]`
  - Uses `is_rec = any(r in tok for r in RECURSIVE_TOKENS)` (substring match)
  - Therefore “Baseline=0%” is partly a function of the baseline prompt not containing those substrings.

**Missing evidence (VERIFIED missing file):**

- `target_comparison_output.txt` claims it saved `target_acquisition_comparison.csv`, but that CSV is **not present** in this repo snapshot (see “Missing artifacts” above).

---

## 4) KV CACHE TRANSFER — what’s the real story?

### 4.1 Retraction / correction of the “Dec 7 ~80% KV transfer” claim

**VERIFIED:** The repo explicitly documents that earlier “KV cache transfer” claims were **provisional / not executed** as stated.

- Evidence: `/Users/dhyana/mech-interp-latent-lab-phase1/KV_PATCHING_HISTORY.md`
  - Says Dec 7–8 KV cache swap was a *conceptual target / proposed*, not fully executed.
  - Says Dec 12 work patched **K-projection + V-projection**, not `past_key_values`, and that this was misnamed “KV_CACHE”.

**Status:** **VERIFIED** (as a correction about what was/wasn’t actually run).

---

### 4.2 Which experiments show “transfer” vs “fail” (based on repo CSV evidence)?

The only directly-auditable numeric “behavior transfer” evidence for patching in this repo snapshot is the `KV_sweep` section inside the two DEC8 RunPod full validation CSVs:

- `/Users/dhyana/mech-interp-latent-lab-phase1/DECEMBER_2025_EXPERIMENTS/DEC8_RUNPOD/01_GEOMETRY_OF_RECURSION/results/full_validation_20251208_130707.csv`
- `/Users/dhyana/mech-interp-latent-lab-phase1/DECEMBER_2025_EXPERIMENTS/DEC8_RUNPOD/01_GEOMETRY_OF_RECURSION/results/full_validation_20251208_134241.csv`

**VERIFIED (from the CSV rows where `experiment=KV_sweep` and `metric=behavior_score`):**

- In `full_validation_20251208_130707.csv`:
  - **natural baseline**: n=10, mean=0.00, nonzero=0/10
  - **patched L0-15**: n=10, mean=0.00, nonzero=0/10
  - **patched L16-31**: n=10, mean=1.86, nonzero=5/10, max=5.71
  - **patched L0-31**: n=10, mean=4.09, nonzero=7/10, max=12.82

- In `full_validation_20251208_134241.csv`:
  - **natural baseline**: n=10, mean=0.00, nonzero=0/10
  - **patched L0-15**: n=10, mean=1.93, nonzero=5/10, max=5.71
  - **patched L16-31**: n=10, mean=1.73, nonzero=4/10, max=7.14
  - **patched L0-31**: n=10, mean=6.11, nonzero=6/10, max=20.00

**VERIFIED (what `behavior_score` means in these CSVs):** In the producer script
`/Users/dhyana/mech-interp-latent-lab-phase1/boneyard/DECEMBER_2025_EXPERIMENTS/DEC8_RUNPOD/01_GEOMETRY_OF_RECURSION/code/full_validation_test.py`,
`behavior_score` is:

- `score_recursive_behavior(text) = (keyword_count / word_count) * 100`
- where `keyword_count` counts regex hits for a fixed list of self-reference patterns (e.g. `\\bobserv\\w*`, `\\bawar\\w*`, `\\bprocess\\w*`, `\\bitself\\b`, etc.).

So the “transfer” recorded in `KV_sweep` is **transfer under a keyword-rate heuristic**, not an externally-validated behavioral score.

---

### 4.3 Is there a script that “successfully transfers recursive behavior”?

**VERIFIED (under the repo’s keyword-rate definition):** The DEC8 RunPod `full_validation_test.py` protocol produces **non-zero** `behavior_score` for KV-patched baselines in multiple conditions in the saved CSVs (Section 4.2).

**UNCERTAIN (semantic “behavior transfer”):** Because `behavior_score` is a keyword-rate heuristic and the CSVs don’t include the generated texts, we cannot verify that the generations are genuinely “recursive” rather than keyword-y or degenerate.

**Available implementation (VERIFIED exists, but results CSV missing):**

- `/Users/dhyana/mech-interp-latent-lab-phase1/true_kv_cache_patching.py` implements `past_key_values` patching and is configured to write `true_kv_cache_patching.csv`, but that CSV is **missing** in this repo snapshot.

**Therefore:** “Successful transfer” cannot be asserted at the level of “this script works” from what is currently present.

---

## 5) CONTRADICTIONS / REVISIONS

### 5.1 Why did L8 steering break coherence?

**UNCERTAIN (primary behavioral artifacts missing):**

- The repo contains a narrative forensic document claiming L8 steering induces “Interrogative Mode” / repetition and that “L8 breaks syntax” was not systematically validated:
  - `/Users/dhyana/mech-interp-latent-lab-phase1/L8_COMPLETE_ARCHAEOLOGICAL_INVESTIGATION.md`
- However, the specific cited raw outputs (e.g. `results/dec11_evening/*.csv`, `logs/dec11_evening/*`) are **not present** in this repo snapshot, so we cannot independently verify the behavioral outputs here.

**VERIFIED (at least as a stated hypothesis + design):**

- `/Users/dhyana/mech-interp-latent-lab-phase1/phase3_single_token_steering.py` exists and is explicitly framed as testing whether “continuous steering destroys syntax,” by steering only the last token.

---

### 5.2 Why did “true KV patching” fail on Dec 12?

**UNCERTAIN (because the claimed CSV is missing):**

- `/Users/dhyana/mech-interp-latent-lab-phase1/TRUE_KV_CACHE_PATCHING_RESULTS.md` claims true KV cache patching produced near-zero behavior transfer and references `true_kv_cache_patching.csv`.
- That referenced CSV is **not present** in this repo snapshot, so the numeric results are not independently auditable here.

**Potential tension (UNCERTAIN, not a contradiction):** The DEC8 RunPod `full_validation_test.py` artifacts *do* show non-zero `behavior_score` for some KV-patched conditions (Section 4.2), but they:

- use a **different model identifier** (`mistralai/Mistral-7B-v0.1`),
- use a **different behavioral metric** (keyword-rate),
- and are a different run/date than the Dec 12 writeup.

So the repo supports: “true KV cache patching can sometimes increase keyword-rate self-reference markers,” but it does **not** settle whether it transfers the intended recursive behavior.

**VERIFIED (revision about earlier mislabeling / what was actually patched):**

- `/Users/dhyana/mech-interp-latent-lab-phase1/KV_PATCHING_HISTORY.md` documents that earlier “KV_CACHE” experiments patched K/V projections, not `past_key_values`, and that the Dec 7 “~80% transfer” claim was not actually executed (provisional).

---

### 5.3 Which findings have been retracted or revised?

**VERIFIED retractions / revisions (explicitly documented in-repo):**

- **Dec 7–8 “~80% KV cache behavior transfer”**:
  - Status: **CONTRADICTED** as an executed empirical result in this repo snapshot.
  - Evidence: `/Users/dhyana/mech-interp-latent-lab-phase1/KV_PATCHING_HISTORY.md` states it was a conceptual/proposed target and not fully executed then.

- **Head discovery pipeline that produced all-zero deltas**:
  - Status: **CONTRADICTED** (invalid results).
  - Evidence: `/Users/dhyana/mech-interp-latent-lab-phase1/HEAD_DISCOVERY_PROBLEMS.md` states gradient magnitudes and deltas were all zeros and that the approach was invalid.

**UNCERTAIN revisions:** Some later markdown writeups cite results whose corresponding raw CSVs are missing in this repo snapshot; those cannot be audited here beyond noting the mismatch.

---

## 6) Rerun map (what you can actually rerun)

- **R_V contraction (small-N reproduction)**:
  - Script: `/Users/dhyana/mech-interp-latent-lab-phase1/mistral_complete_reproduction.py`
  - Output: `mistral_reproduction_results.json`

- **Layer tomography (single prompt trio)**:
  - Script: `/Users/dhyana/mech-interp-latent-lab-phase1/tomography_relay_v2.py`
  - Output: `mistral_relay_tomography_v2.csv`

- **Head discovery (v_proj ablation)**:
  - Script: `/Users/dhyana/mech-interp-latent-lab-phase1/v_proj_head_discovery.py`
  - Output dir: `/Users/dhyana/mech-interp-latent-lab-phase1/results/head_discovery/`

- **Target acquisition comparisons (single prompt pair)**:
  - Script: `/Users/dhyana/mech-interp-latent-lab-phase1/compare_targets_baseline.py`
  - Output (expected): `target_acquisition_comparison.csv` (**missing** in current repo snapshot; rerun should regenerate it)

- **True KV cache patching implementation exists**:
  - Script: `/Users/dhyana/mech-interp-latent-lab-phase1/true_kv_cache_patching.py`
  - Output (expected): `true_kv_cache_patching.csv` (**missing** in current repo snapshot; rerun should regenerate it)

---

## 7) Bottom-line forensic assessment

- **VERIFIED:** Mistral-7B shows \(R_V\) contraction (recursive < baseline) in at least two distinct datasets in this repo:
  - Small-N reproduction (N=8/8) with p=0.0031 (`mistral_reproduction_results.json`)
  - Larger-N DEC8 CSVs (N=40/40) with p≈4.5e-31 at L27 (`full_validation_20251208_*.csv`)

- **VERIFIED:** The “H18 & H26” head story is **not independently supported by v_proj ablation as uniquely causal** due to **GQA KV-head aliasing** (they are in a 4-head equivalence class).

- **VERIFIED:** The repo contains explicit self-corrections that earlier KV cache transfer claims were mischaracterized or not executed (`KV_PATCHING_HISTORY.md`).

- **UNCERTAIN:** Several behavioral-transfer and L8-coherence claims reference missing raw artifacts in this repo snapshot; they cannot be re-audited without those CSVs/logs being present.


