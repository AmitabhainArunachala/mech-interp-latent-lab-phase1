# MASTER AUDIT: R_V Paper vs Repo Data Verification
**Target:** COLM 2026 submission — `R_V_PAPER/paper_colm2026_v005.tex`
**Date:** 2026-03-09
**Purpose:** Every claim in the paper must be verified against actual data in this repo. Find contradictions, orphan findings, and gaps.

---

## INSTRUCTIONS FOR AGENTS

You are auditing a mechanistic interpretability research repo. A paper has been written (694 lines of LaTeX) making specific quantitative claims. Your job is to verify EVERY claim against the actual data files in this repo.

### Rules
1. **Do NOT read prior audit reports** in `agent_reviews/responses/` or `FEB_5_LATEST_AUDIT_*.md`. We need fresh eyes, not anchored opinions.
2. **Every verdict must cite an exact file path** with the actual value from that file.
3. **Do NOT trust the paper text.** Verify everything against raw data (JSON, CSV, Python output).
4. **If data is missing, say MISSING.** Do not infer or assume.
5. **You must read the paper first:** `R_V_PAPER/paper_colm2026_v005.tex`
6. **Time budget:** Focus on the 25 claims below. Do not explore tangentially.

### Output Format

Save your response to:
```
agent_reviews/responses/20260309__MODELNAME__COLM_PAPER_AUDIT.md
```

Required header:
```
Title: COLM 2026 PAPER AUDIT
Date: 2026-03-09
Model: <your model name + version>
Audit duration: <approximate time spent>
```

---

## PART A: CLAIM-BY-CLAIM VERIFICATION

For each claim below, provide EXACTLY this format:

```
### Claim [ID]: [short description]
- **Paper says:** [exact quote or paraphrase with line number]
- **Data file:** [exact path to the JSON/CSV that should contain this]
- **Data shows:** [actual values from that file]
- **Verdict:** CONFIRMED / CONTRADICTED / PARTIAL / NO_DATA
- **Severity:** CRITICAL / HIGH / MEDIUM / LOW
- **Notes:** [any context]
```

### The 25 Claims to Verify

**Cross-Architecture (Claims 1-7)**

C1. Mistral-7B shows contraction with d=-1.66, CI [-2.08, -1.32], n=152
- Check: `results/power_up/mistral-7b_n80_result.json` and any n>80 files
- Check: Are n1=75 and n2=77 (as in Table 1) consistent with data?

C2. Qwen2.5-7B shows contraction with d=-2.32, CI [-2.86, -1.90], n=124
- Check: `results/power_up/qwen2.5-7b_n100_result.json` and `qwen2.5-7b_n80_result.json`

C3. OPT-6.7B: Paper uses |d|=1.68 in body text (line 197). What is the SIGNED d?
- Check: `results/power_up/opt-6.7b_n80_result.json`
- Check: Is recursive_mean > or < baseline_mean?
- CRITICAL: Does the data show contraction or EXPANSION?

C4. GPT-2 XL: Paper uses |d|=1.52 in body text (line 198). What is the SIGNED d?
- Check: `results/power_up/gpt2-xl_n80_result.json`
- CRITICAL: Same question as C3.

C5. Pythia-1.4B shows d=-0.006, p=0.88, n=124
- Check: `results/power_up/pythia-1.4b_n80_result.json`

C6. Paper line 194 says "contraction replicates in four models." Is this accurate given C3 and C4 findings?

C7. Table 1 (lines 525-529) shows specific n1, n2 values. Do these match the JSON files?

**Causal Claims (Claims 8-12)**

C8. Necessity: d=3.29, n=300, "breaks V-projections at L25 and L27"
- Search: `results/` for any file containing these stats
- Check: What experiment script produced this? Does the script exist?
- Check: `results/path_patching/path_patching_summary_20260227_080128.json` — what does the V-proj path patching data actually show?

C9. Sufficiency: d=-3.50, n=300, OR=13.96 for KV injection
- Search: `results/kv_sufficiency_matrix/` and `results/sufficiency_ladder/`
- CRITICAL: Does KV injection transfer R_V geometry (not just behavior)?
- Check: `docs/findings/R_V_BEHAVIORAL_DISSOCIATION.md` — what does it say about R_V transfer?

C10. Paper line 231: "reduces BT+ART from 56% to 27.7%"
- Find the actual source data for these percentages
- Is 27.7% correct? Or is the actual number different?

C11. Within-session bridge: d=-0.71, n=150
- Check: `results/within_session_bridge/`

C12. V-projection path patching: For every layer tested, what is the V-proj Cohen's d?
- Check: `results/path_patching/path_patching_summary_20260227_080128.json`
- List the max V-proj |d| across all layers
- CRITICAL: Does this support or undermine the "Value Spaces" title?

**Mode Atlas & Statistical (Claims 13-17)**

C13. Self-referential R_V mean = 0.650, SD = 0.098, d=-1.67 vs all modes
- Check: `results/mode_atlas/`

C14. 606/1024 heads significant at p<0.05
- Check: `results/full_head_sweep/`

C15. Perplexity matching survives: d=-1.67, p=0.002, n=8 strict pairs
- Check: `results/perplexity_repairing/`

C16. Multi-seed: all 5 seeds give identical d=-1.751
- Check: `results/power_up/multi_seed_summary_20260306.json`

C17. FDR: 30/36 survive at alpha=0.05
- Check: `results/fdr_correction/`

**Circuit & Representation (Claims 18-22)**

C18. L27H10 effective rank: 7.28 → 5.91, d=-1.54
- Check: `results/svd_circuits/`

C19. L5H29 expansion d=2.93
- Check: Same as C18

C20. Concept erasure: d=-1.82 before, d=-1.82 after, delta=0.005
- Check: `results/` for concept erasure files

C21. DII at L27: every PCA dimension shows R_V ≈ 0.41
- Check: `results/dii_intervention/`

C22. RSA: max dissimilarity at L28 (distance 0.307)
- Check: `results/rsa/`

**Safety & Scaling (Claims 23-25)**

C23. AUROC = 0.909 for self-referential detection
- Check: `results/safety/` or `results/classifier_evaluation/`

C24. Genuine vs deceptive: d=-0.06
- Check: Same as C23

C25. Scaling: R²=0.047 with 8 data points (paper line 472)
- Check: `results/scaling_law/` or `results/scaling_gap/`

---

## PART B: ORPHAN FINDINGS SCAN

Search these directories for significant findings NOT mentioned in the paper:

1. `docs/findings/` — Read ALL 5 files. List any finding not in the paper.
2. `RECOVERED_GOLD/` — Read ALL 7 files. List any finding not in the paper.
3. `docs/findings/NEURIPS_CANDIDATE_2026-02-20.md` — Is the GQA headspace finding in the paper?
4. `docs/findings/R_V_BEHAVIORAL_DISSOCIATION.md` — Is this dissociation acknowledged in the paper?
5. `results/phase1_cross_architecture/runs/` — Are there models tested but not reported?
6. `industry_grade/` — Any findings here that should be in the paper?

For each orphan finding, provide:
```
### Orphan [ID]: [description]
- **Source file:** [path]
- **Key stat:** [value]
- **Should be in paper?** YES_CRITICAL / YES_USEFUL / NO_IRRELEVANT
- **Why:** [brief reason]
```

---

## PART C: CODE CONSISTENCY CHECK

1. **R_V computation:** Compare the formula in these files:
   - `src/metrics/rv.py` (canonical)
   - `geometric_lens/metrics.py` (alternative)
   - `CANONICAL_CODE/` (any files here)
   - Paper equation (lines 110-119)
   - Are they all computing the same thing? If not, which results use which code path?

2. **Layer selection:** Compare layer choices across:
   - `geometric_lens/models.py` (model registry)
   - `src/core/` or `src/experiments/`
   - The paper's stated layers (line 120-121)
   - CRITICAL: What layers does the registry use for OPT and GPT-2? Are they at ~16% and ~84% depth?

3. **Prompt bank:** 
   - `prompts/bank.json` — how many prompts per category?
   - Are the same prompts used across all experiments?
   - Do power_up results use the same bank as mode_atlas results?

---

## PART D: CONTRADICTION MAP

After completing Parts A-C, fill in this summary:

```
## CONTRADICTION SUMMARY

### Paper claims data CONTRADICTS:
- [list each with claim ID]

### Paper claims with NO supporting data found:
- [list each with claim ID]

### Findings in repo that SHOULD be in paper but aren't:
- [list each with orphan ID]

### Code inconsistencies that affect results:
- [list each]

### Recommended paper changes (ranked by severity):
1. [most critical]
2. ...
```

---

## KEY FILE LOCATIONS (for reference)

```
Paper:           R_V_PAPER/paper_colm2026_v005.tex
Canonical code:  src/metrics/rv.py
Alt code:        geometric_lens/metrics.py, geometric_lens/models.py
Model registry:  geometric_lens/models.py (lines 196-253)
Prompt bank:     prompts/bank.json
Results root:    results/
Power-up data:   results/power_up/
Cross-arch:      results/phase1_cross_architecture/
Path patching:   results/path_patching/
KV sufficiency:  results/kv_sufficiency_matrix/
Findings docs:   docs/findings/
Gold files:      RECOVERED_GOLD/
Industry grade:  industry_grade/
Agent reviews:   agent_reviews/responses/ (DO NOT READ — fresh eyes only)
```

---

## WHAT MAKES A GOOD AUDIT

From the Dec 2015 meta-factcheck of 4 agents:
- Gemini scored 9/10 by being concise but finding the ONE critical thing others missed (L21=L27 equivalence)
- Grok scored 5/10 by marking things "VERIFIED" that were actually contradicted
- The best agents were SKEPTICAL by default and only marked CONFIRMED when data was unambiguous

**Default to SKEPTICISM. The paper's job is to be right. Your job is to find where it's wrong.**
