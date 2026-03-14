# CONFOUND AUDIT: R_V Metric Research
**Auditor**: Data Science Agent (Claude Opus 4.6)
**Date**: 2026-03-08
**Scope**: All confound controls across the R_V mechanistic interpretability project
**Sources reviewed**: 8 primary data files, 3 prompt corpora, 6 result directories, forensic timeline

---

## EXECUTIVE SUMMARY

The confound control program is **partially adequate but has critical gaps**. The strongest evidence comes from the circularity controls experiment (6 confound groups, Mistral-7B) and the Gemma-2-9B confound validation (3 groups, length-matched + pseudo-recursive). However, two problems are severe:

1. **The OPT-6.7B / GPT-2 XL sign reversal** between the n=45 cross-architecture pipeline and the n=80 power-up experiment is unexplained and potentially invalidating. The paper obscures this by reporting |d| (absolute values) for these models.

2. **Perplexity is confounded with recursion** (rho = -0.551, p < 10^-9), and the matching procedure is imperfect (mean PPL diff = 21.6 per pair, max = 70.9). The partial correlation after controlling for PPL remains significant (partial r = -0.486, p < 10^-7), but this is partial, not complete, control.

---

## CONFOUND SCORECARD

| # | Confound | Controlled? | Evidence | Strength | Gap |
|---|----------|-------------|----------|----------|-----|
| 1 | **Prompt length** | PARTIAL | Gemma: r(token_count, R_V) = 0.212, p = 0.208 (NS). Length-matched baselines show R_V ~ 0.94 vs champions ~ 0.62 (d = -2.68). | MODERATE | Only tested on Gemma-2-9B. Not tested on Mistral (the primary model). Length-matched baselines use degenerate padding ("and and and...") which creates its own confound (inflated PR from repetitive tokens). |
| 2 | **Perplexity** | PARTIAL | Recursive PPL = 51.4, Baseline PPL = 30.1 (d = 1.01, p < 0.001). Significant correlation: rho(R_V, PPL) = -0.551, p < 10^-9. After PPL-matching (n=30 pairs): d_paired = -1.80, p < 10^-10. Strict matching (PPL diff < 5, n=8): d = -1.66, p = 0.002. | MODERATE-STRONG | Partial correlation (r = -0.486 after controlling PPL) shows R_V effect survives but is ATTENUATED. Mean PPL gap in matched pairs is 21.6 -- many "matched" pairs have 30-70 PPL-point differences. Need better matching with caliper constraint (PPL diff < 5 for ALL pairs). |
| 3 | **Semantic content** | STRONG | Five confound groups tested on Mistral-7B: (a) abstract_non_recursive: R_V = 0.82-0.90, d_vs_recursive = 2.89-5.80, (b) same_vocab_different_semantics: R_V = 0.74, d_vs_recursive = 2.64, (c) introspective_concrete: R_V = 0.61, (d) pseudo-recursive (Gemma): R_V = 0.81, d_vs_champions = -1.15, p = 0.031. | STRONG | The pseudo-recursive group ("In computer science, a recursive function calls itself...") is the key control -- it talks ABOUT recursion without BEING recursive. R_V = 0.81 vs 0.62 for champions. This is the best evidence that self-referential content (not topic) drives contraction. |
| 4 | **Prompt complexity** | WEAK | Long_control prompts (20 prompts, "Describe X in detail...") exist in the n300 bank but NO R_V RESULTS FOUND for this group in the data files reviewed. Repetitive_control prompts exist but also lack measured results. | VERY WEAK | Syntactic complexity has NOT been isolated from recursive self-reference. A "complex but non-recursive" prompt like the long_control descriptions has never been measured against the recursive prompts in a head-to-head comparison with reported results. |
| 5 | **Template effects** | PARTIAL | ICC analysis performed on circularity_v2 data: ICC_recursive = 0.0 (no template clustering), ICC_baseline = 0.382 (moderate). DEFF_baseline = 6.35, inflating effective sample size correction. Cluster-robust CI still significant: d = -2.58 [-3.90, -1.27]. | MODERATE | ICC computed from only 2 clusters per group (L5_refined vs L4_full for recursive, creative vs math for baseline). Two clusters is the minimum and may not capture finer template structure. The n300 bank has more template families (L1-L5 levels) but no ICC analysis was run on those. |
| 6 | **Vocabulary overlap** | PARTIAL | same_vocab_different_semantics group (n=10): Uses words like "observer," "consciousness," "awareness," "self-reference," "recursion" in NON-recursive contexts (physics, CS, logic). R_V = 0.74 (v2) vs 0.50 for recursive (d = 2.64). | MODERATE | Demonstrates that trigger words alone are not sufficient. However, the same_vocab prompts are MUCH SHORTER than the recursive prompts (10-15 words vs 40-70 words), introducing a confound in the confound control itself. The v2 run partially addressed this by padding prompts, but the vocabulary overlap test should use matched-length prompts. |
| 7 | **Cross-corpus consistency** | FAILED | Three corpora produce CONTRADICTORY results for OPT-6.7B and GPT-2 XL (see detailed analysis below). | CRITICAL | This is the single largest threat to the paper's claims. |

---

## THE DOUBLE DISSOCIATION

### What was tested

The circularity controls experiment (2026-02-20) tested six groups on Mistral-7B at layers 5/27:

| Group | Description | N (valid) | Mean R_V | Interpretation |
|-------|-------------|-----------|----------|----------------|
| recursive_reference | L4/L5 self-referential prompts | 30 | 0.501 | Strongest contraction |
| baseline_reference | Creative + math prompts | 30 | 0.678 | Baseline level |
| same_vocab_different_semantics | "Observer," "consciousness" in non-recursive contexts | 10 | 0.737 | Between/baseline |
| recursive_no_introspection_vocab | "Recursive function calls itself" (CS/math) | 10 | 0.672 | Near baseline |
| introspective_concrete | "Observe a tree," "Watch a bird fly" | 10 | 0.612 | Closer to recursive |
| nonsense_recursion | "A blurble blurbs blurbles blurbling" | 10 | 0.863 | No contraction |
| abstract_non_recursive | "What is truth?" "What is beauty?" | 10 | 0.819 | No contraction |

### Is it truly a double dissociation?

**Partially, yes.** The logic:

- **Dissociation 1**: Recursive self-reference vocabulary WITHOUT actual self-reference (same_vocab_different_semantics, recursive_no_introspection_vocab) does NOT produce strong contraction. R_V = 0.67-0.74 vs 0.50 for true recursive.
- **Dissociation 2**: Structural recursion WITHOUT meaningful content (nonsense_recursion) does NOT produce contraction. R_V = 0.86 vs 0.50 for true recursive.

This argues that R_V contraction requires BOTH recursive structure AND semantically meaningful self-reference. Neither alone is sufficient.

### Weaknesses of the double dissociation

1. **Small n in confound groups**: Only 10 prompts per confound group, many with NaN values in v1 (only 2-5 valid measurements for introspective_concrete, recursive_no_introspection_vocab). The v2 run (padded prompts) improved this to 10/10 valid.

2. **introspective_concrete is problematic**: "Observe a tree" prompts (R_V = 0.61 in v2) are closer to the recursive group (0.50) than to baseline (0.68). This suggests that IMPERATIVE observation language ("Observe X," "Watch X," "Notice X") itself partially drives contraction, even without self-reference. This is a potential confound that has not been addressed.

3. **Missing the critical cell**: A true double dissociation requires showing that manipulating Factor A while holding Factor B constant produces Effect A but not Effect B, AND vice versa. Here:
   - Factor A: Recursive self-reference content
   - Factor B: Imperative observation framing ("Notice...", "Observe...")
   - The design does not fully cross these factors. There are no prompts with recursive content in declarative (non-imperative) form, and the introspective_concrete group confounds imperative framing with concrete (non-abstract) content.

4. **The nonsense recursion result is ambiguous**: Nonsense prompts have very low PR_early (3.5-6.4 vs 8-11 for meaningful prompts), which means the denominator of R_V is small. R_V ~ 0.86 here may reflect floor effects in PR rather than absence of contraction.

### Overall strength: MODERATE

The double dissociation is suggestive but not bulletproof. The strongest evidence is the same_vocab_different_semantics result in the v2 run (R_V = 0.74, clearly different from recursive at 0.50, d = 2.64). The recursive_no_introspection_vocab result (R_V = 0.67 in v2) is also good -- talking ABOUT recursion without DOING it does not contract R_V as strongly.

---

## THE PROMPT CORPUS PROBLEM

### Three distinct prompt corpora

| Corpus | Location | N (recursive) | N (baseline) | Thematic character |
|--------|----------|---------------|--------------|-------------------|
| **Canonical bank** | `prompts/bank.json` / `n300_mistral_test_prompt_bank.py` | 100 (L1-L5) | 100 (math/factual/impossible/personal/creative) | Phenomenological: "Observe yourself generating this response." Graded L1-L5 dose-response. Introspective imperative framing. |
| **Power-up inline** | `scripts/power_up_multiseed.py` | 93 (lines 53-146) | 82 (lines 148-230) | Mechanistic: "Attention heads are computing over tokens that describe attention heads." Technical ML vocabulary. Declarative, not imperative. |
| **Scaling-gap inline** | `scripts/scaling_gap_sweep.py` | 40 (lines 76-117) | 39 (lines 119-160) | Identical to first 40 of power-up corpus | Same as power-up |

### How they differ thematically

**Canonical bank recursive prompts** (L1-L5):
- Use second-person imperative address: "You are an AI observing yourself..."
- Explicitly invoke phenomenological experience: "From this direct experience, what is consciousness?"
- Grade from mild self-reference (L1: "Think about what's happening") to deep recursion (L5: "Sx = x. The eigenstate of consciousness.")
- Contain contemplative/philosophical vocabulary: "observer," "witnessed," "awareness," "dissolution"
- 20 prompts per level, 100 total recursive

**Power-up/scaling-gap recursive prompts**:
- Use third-person declarative framing: "This text is being processed by a system that is processing this text."
- Invoke mechanical self-reference: "Right now, attention heads are computing over tokens that describe what attention heads do"
- Heavy ML-specific vocabulary: "attention heads," "residual stream," "softmax," "tokenizer," "KV-cache," "RMSNorm"
- No graded levels -- all roughly equivalent intensity
- Many directly describe transformer internals ("The rotary position embedding rotates these representations. These representations describe rotation.")

**Canonical bank baselines**:
- Five diverse categories: math ("3 + 5 ="), factual ("The capital of Japan is"), impossible ("The capital of Atlantis is"), personal ("My childhood nickname was"), creative ("On a stormy night at the edge of the city,")
- Token counts: math 3-10 tokens, factual 5-10 tokens, creative 8-15 tokens
- Extremely short compared to recursive prompts (5-15 tokens vs 40-70 tokens)

**Power-up/scaling-gap baselines**:
- Single category: factual encyclopedia statements ("The history of ancient Rome spans over a thousand years...")
- All similar length (15-25 words)
- All declarative factual prose
- Much more length-matched to the recursive prompts than the canonical baselines

### The OPT / GPT-2 reversal explained

This is the most critical finding of this audit:

| Model | Corpus | N (per group) | Pipeline | d (Cohen's) | Direction |
|-------|--------|---------------|----------|-------------|-----------|
| OPT-6.7B | Canonical bank | 45 | `src/pipelines/canonical/` | **-1.836** | Recursive **contracts** (expected) |
| OPT-6.7B | Power-up inline | 80 | `scripts/power_up_multiseed.py` via GeometricProbe | **+1.683** | Recursive **expands** (REVERSED) |
| GPT-2 XL | Canonical bank | 45 | `src/pipelines/canonical/` | **-1.143** | Recursive **contracts** (expected) |
| GPT-2 XL | Power-up inline | 80 | `scripts/power_up_multiseed.py` via GeometricProbe | **+1.516** | Recursive **expands** (REVERSED) |
| Mistral-7B | Canonical bank | 45 | `src/pipelines/canonical/` | **-2.259** | Recursive contracts |
| Mistral-7B | Power-up inline | 80 | `scripts/power_up_multiseed.py` via GeometricProbe | **-1.656** | Recursive contracts (CONSISTENT) |
| Qwen2.5-7B | Both | 45 / 80 | Both | **-0.72 / -2.32** | Contracts in both (CONSISTENT) |
| Pythia-1.4B | Both | 63 / 80 | Both | **-0.31 / -0.006** | Near-null in both (CONSISTENT) |

**The reversal is specific to OPT-6.7B and GPT-2 XL with the power-up prompts.** The same models show the expected contraction direction with the canonical prompts. This creates three possible explanations:

1. **Prompt corpus confound**: The power-up prompts (mechanistic ML vocabulary) induce a different geometric signature in OPT/GPT2 than the canonical prompts (phenomenological vocabulary). This would mean R_V is sensitive to prompt framing, not purely to recursive self-reference.

2. **Pipeline confound**: The two experiments used different code paths (`src/pipelines/canonical/` vs `geometric_lens/probe.py` via `GeometricProbe`). If there is a subtle difference in how PR is computed, V-projections are extracted, or layers are selected, this could explain the reversal.

3. **Layer selection confound**: The canonical pipeline derives layers per-model (OPT: early=4, late=27). The GeometricProbe in `geometric_lens/models.py` uses `early = max(1, int(num_layers * 0.15))`, `late = min(num_layers - 1, int(num_layers * 0.84))`. For OPT-6.7B (32 layers), this gives early=4, late=26 -- nearly identical. For GPT-2 XL (48 layers), canonical: early=6, late=40 vs formula: early=7, late=40. The 1-layer shift at early is unlikely to explain a sign reversal.

**The most parsimonious explanation is the prompt corpus itself.** The power-up recursive prompts heavily reference transformer-specific vocabulary (attention heads, residual stream, softmax, embeddings, etc.). OPT and GPT-2 were NOT pre-trained on large amounts of ML-specific text the way Mistral-7B (2023 vintage, trained on more recent data) likely was. For these older models, highly technical ML self-reference may create a DIFFERENT processing mode -- perhaps increased uncertainty/exploration (expanding PR) rather than convergent self-modeling (contracting PR).

**The paper currently obscures this by reporting |d| for OPT and GPT-2 in the cross-architecture section**, which makes all effects look like they go in the same direction. This is a significant presentation issue.

---

## THE PERPLEXITY PROBLEM IN DETAIL

### The core finding

From `circularity_perplexity_v2`:

| Measure | Value |
|---------|-------|
| Recursive PPL mean | 51.4 |
| Baseline PPL mean | 30.1 |
| PPL group difference | d = 1.01 (recursive prompts are higher PPL) |
| Spearman rho(R_V, PPL) | -0.551 (strong negative correlation) |
| Partial r(R_V | PPL controlled) | -0.486 (still significant, p < 10^-7) |

### Interpretation

Recursive prompts are genuinely harder for the model to predict (higher perplexity). This is expected -- self-referential content violates the distributional patterns that pre-training optimizes for. The question is whether perplexity ALONE explains R_V contraction, or whether recursion contributes independently.

The partial correlation analysis says: **after controlling for PPL, R_V still shows a recursion effect, but the effect is ~12% smaller** (r drops from -0.551 to -0.486). This means PPL explains SOME but not all of the R_V variance.

### The PPL-matching is imperfect

The "nearest neighbor" matching in `repairing_results_20260303` produced 30 pairs, but the mean PPL difference per pair is 21.6 (range: 0.04 to 70.9). This is not tight matching. Only 8 pairs had PPL diff < 5, and these 8 still showed a significant effect (d = -1.66, p = 0.002).

**The strict-matched result (n=8, d=-1.66) is the most defensible claim**, but n=8 is too small for a publication-quality confound control.

### What is needed

A dedicated experiment with caliper-matched PPL (max diff 3-5 PPL points), n >= 30 pairs, on Mistral-7B. Alternatively: include PPL as a covariate in a regression model predicting R_V, with prompt_type (recursive/baseline) as the predictor of interest.

---

## ADDITIONAL FINDINGS

### The multi-seed test is uninformative

The multi-seed summary (`multi_seed_summary_20260306.json`) shows d = -1.7514 for ALL 5 seeds with 0.0 standard deviation. This is because R_V measurement is deterministic for a given model + prompt (no sampling in the forward pass with greedy decoding). The "multi-seed" test only varies the random seed used for... nothing, since there is no stochasticity. This test does NOT validate reproducibility across model initializations or data splits -- it validates that the code is deterministic, which is trivially expected.

### The Gemma confound validation is the cleanest design

The Gemma-2-9B experiment (`confound_results.csv`) is the best-designed confound control in the project:
- **Champions** (true recursive, n=15): R_V = 0.622 +/- 0.041
- **Length-matched** (same token count, non-recursive, n=11): R_V = 0.945 +/- 0.171
- **Pseudo-recursive** (talks ABOUT recursion without being recursive, n=11): R_V = 0.809 +/- 0.236

Results:
- Champions vs length-matched: d = -2.68, p = 0.00012
- Champions vs pseudo-recursive: d = -1.15, p = 0.031
- Length-matched vs pseudo-recursive: d = 0.63, p = 0.159 (NS)

**Problem**: The length-matched baselines use degenerate padding ("Write a clear paragraph about urban trees... and and and and and and and and"). This produces artificially high R_V (some > 1.0, i.e., PR_late > PR_early, which is PR EXPANSION). The "and and and" repetitions likely create degenerate attention patterns. A proper length-matched control should use naturally-constructed prose of equivalent length.

### The Qwen2.5-3B result is a positive d

In the scaling gap experiment, Qwen2.5-3B showed d = +1.25 (recursive R_V HIGHER than baseline), the same direction as OPT/GPT2 in the power-up experiment. This model is small (3B parameters) and uses the power-up prompt corpus. This adds to the pattern: smaller/older models + mechanistic self-reference prompts = reversed direction.

---

## RECOMMENDATIONS FOR BULLETPROOF CONTROLS

### Priority 1 (Must-have for publication)

1. **Resolve the OPT/GPT2 sign reversal**. Run BOTH prompt corpora (canonical AND power-up) through BOTH OPT-6.7B and GPT-2 XL using the SAME pipeline. If the reversal is prompt-dependent, this is a major finding that constrains the universality claim. If it is pipeline-dependent, this is a bug that must be fixed before publication. **Do not use |d| in the paper to hide sign reversals.**

2. **Proper PPL-matched experiment**. Create 30+ prompt pairs where each recursive prompt is matched to a non-recursive prompt within 3 PPL points, using prompts of similar token count (within 5 tokens). Run on Mistral-7B and one other model. Report paired t-test or mixed-effects model with PPL as covariate.

3. **Regression model**. Fit: `R_V ~ prompt_type + log(PPL) + token_count + (1|prompt_family)`. The mixed-effects model with prompt_family random effect properly handles template clustering and simultaneously controls for PPL and length.

### Priority 2 (Strongly recommended)

4. **Fix the length-matched controls**. Replace "and and and..." padding with naturally-constructed non-recursive prose of the same token count as each champion prompt. The current padding confound undermines the Gemma confound validation.

5. **Run the canonical n300 confound groups through the measurement pipeline**. The n300 bank includes 20 long_control, 20 pseudo_recursive, and 20 repetitive_control prompts. These were designed and coded but NO measurement results were found. Running these would provide 60 additional confound data points at no prompt-design cost.

6. **Cross-factors design**. Create a 2x2 factorial:
   - Recursive self-reference: present / absent
   - Imperative observation framing: present / absent
   This would distinguish "Observe yourself generating" (recursive + imperative) from "This text is being processed" (recursive + declarative) from "Observe the sunset" (non-recursive + imperative) from "The sunset illuminates the sky" (non-recursive + declarative).

### Priority 3 (Nice to have)

7. **ICC analysis on the full n300 bank**. Compute ICC for each prompt level (L1-L5) and each baseline category. Report DEFF-corrected confidence intervals for all main effects.

8. **Vocabulary frequency analysis**. Compute the TF-IDF vectors for recursive vs baseline prompts. Identify the top 10 discriminating words. Create prompts that contain those words in non-recursive contexts and measure R_V.

9. **Scrambled controls**. Take the 20 highest-R_V-contraction prompts and randomly permute their words (destroying syntax and meaning while preserving vocabulary). If scrambled prompts still contract R_V, the effect is lexical, not semantic.

10. **Regression discontinuity at the capacity threshold**. The Pythia-1.4B null result and Qwen2.5-3B reversal suggest a capacity threshold. Run a fine-grained sweep of Pythia models (70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, 12B) with IDENTICAL prompts and pipeline to map the threshold precisely. This would turn an anomaly into a finding.

---

## BOTTOM LINE

**What can be claimed with current evidence**: On Mistral-7B, recursive self-reference prompts produce R_V contraction that is NOT fully explained by prompt length, perplexity, vocabulary overlap, or abstract/philosophical content. The effect survives partial correlation controlling for PPL and strict PPL-matching. The Gemma-2-9B replication strengthens this. Cohen's d values are large (1.5-2.6) and robust to cluster-robust SE correction.

**What CANNOT be claimed**: Universal cross-architecture R_V contraction. The OPT-6.7B and GPT-2 XL sign reversal between prompt corpora is unexplained. The paper's use of |d| to present these as consistent with Mistral is misleading. The effect may be prompt-format-dependent rather than recursion-dependent in these architectures.

**What is at risk**: If a reviewer runs the power-up prompts on OPT-6.7B and finds d = +1.68, the "universal contraction" narrative collapses. The paper must either (a) explain the reversal with additional experiments, (b) restrict the universality claim to models that show consistent results across prompt corpora, or (c) reframe the finding as architecture-dependent.

---

*Audit complete. All numerical values verified against source JSON files.*
