# POWER PROMPT: Triple-Check 8 Missing/Suspect Paper Claims

**Date**: 2026-03-12
**Purpose**: Exhaustive search for source data backing 8 claims flagged by 5-agent audit
**Repo**: `~/mech-interp-latent-lab-phase1/`

---

## YOUR MISSION

You are auditing a mechanistic interpretability paper (COLM 2026) that measures geometric contraction in transformer value spaces via an R_V metric. A prior 5-agent audit flagged 8 quantitative claims that either have NO source file, the WRONG source file, or a VALUE MISMATCH. Your job is to find every file in this repo (and on this machine) that could be the source, and report exactly what you find.

**Be adversarial. Assume nothing. Check everything.**

---

## THE 8 CLAIMS TO TRACE

### CLAIM 1: Per-token R_V bridge d=-1.64, p=1.4e-6

**Paper says**: Per-token R_V during generation shows d=-1.64, p=1.4e-6 separating recursive from baseline.
**Prior audit says**: FABRICATED/LOST. Closest real values are d=-0.608 (p=0.069, NS) and d=-0.567 (within-session, different metric).

**Search instructions**:
1. `grep -r "1\.64" results/ --include="*.json" --include="*.csv"` — look for the exact d value
2. `grep -r "1\.4e-6\|1\.4e-06\|0\.0000014" results/` — look for the exact p value
3. Read EVERY file in `results/batch_per_token_rv/` — this is the most likely home
4. Read EVERY file in `results/behavioral_bridge/` or similar directories
5. Read `results/behavioral_nboost_summary.json` — check all d values
6. Search `industry_grade/` recursively for any per-token analysis
7. Search `R_V_PAPER/results/` for any behavioral bridge data
8. Check `RECOVERED_GOLD/` for any per-token R_V results
9. Check ALL `.ipynb` files for computed but unsaved values
10. Check `~/Desktop/MECH INTERP FILES/` for any per-token CSV or notebook

**Report**: The exact file path and line containing d=-1.64 (or the closest value you find), and whether p=1.4e-6 appears anywhere.

---

### CLAIM 2: Primary Mistral headline d=-2.26, n=151/151

**Paper says**: Mistral-7B primary effect is d=-2.26 with n=151 recursive, 151 baseline pairs.
**Prior audit says**: d=-2.259 exists in `results/phase1_cross_architecture/runs/20260202_121604_.../summary.json` but with n=45, NOT 151.

**Search instructions**:
1. Read `results/phase1_cross_architecture/runs/20260202_121604_rv_l27_causal_validation_mistral_7b/summary.json` — confirm n_pairs
2. `grep -r "n_pairs\|n_rec\|n_base\|n1=\|n2=" results/ --include="*.json"` — find ANY file with n>=100
3. `grep -r "151" results/ --include="*.json"` — find ANY file referencing 151 samples
4. Read `results/power_up/mistral-7b_n80_result.json` — this has d=-1.656, n=75+77=152, could this be the intended source?
5. Read `results/p0_canonical/` — check all files for Mistral base results
6. Search `CANONICAL_CODE/mistral_L27_FULL_VALIDATION.py` — does the script support n=151?
7. Check `docs/status/CLAIM_REGISTRY.md` for claim M02
8. Search ALL `summary.json` files: `find results/ -name "summary.json" -exec grep -l "2\.26\|2\.259" {} \;`

**Report**: Every file containing d close to -2.26, and every file with n>=100 for Mistral. State definitively whether n=151 data exists ANYWHERE.

---

### CLAIM 3: Gemma cross-arch d=-3.37

**Paper says**: Gemma-2-9B cross-architecture effect is d=-3.37 (in Table 1 cross-arch column).
**Prior audit says**: d=-3.37 is from Phase 3 multi-token bridge experiment (n=117), NOT cross-architecture. Actual cross-arch: d=-1.74 (n=60).

**Search instructions**:
1. Read `results/phase3_bridge/gemma_2_9b/multi_token_correlation_v3/` — find the summary.json with h2_cohens_d=3.369
2. Read `results/phase2_generalization/gemma_2_9b/11_causal_validation_champion/.../summary.json` — confirm d=-1.736
3. Read `results/phase2_generalization/gemma_2_9b/08_causal_validation_n45/.../summary.json` — confirm d=-1.908
4. Read `R_V_PAPER/results/gemma/` if it exists
5. Read `R_V_PAPER/GEMMA_2_9B_STAR_WITNESS_ASSESSMENT.md` — find where -3.37 is labeled as "multi-token bridge"
6. `grep -r "3\.37\|3\.369" results/ --include="*.json"` — find ALL occurrences
7. Check `RECOVERED_GOLD/` for any Gemma results

**Report**: Confirm that -3.37 is from multi-token bridge (not cross-arch), confirm the actual cross-arch value, and list every file containing either number.

---

### CLAIM 4: Mixtral 24.3% contraction, d~5.3

**Paper says**: Mixtral-8x7B shows 24.3% contraction with |d|~5.3, strongest effect across all models.
**Prior audit says**: NO production result artifacts in repo. Narrative docs exist. Original CSV may be on Desktop (iCloud).

**Search instructions**:
1. `find results/ -path "*mixtral*" -type f` — list ALL Mixtral result files
2. Read `R_V_PAPER/results/mixtral/MIXTRAL_8x7B_SUMMARY.md`
3. Read `R_V_PAPER/results/mixtral/MIXTRAL_LAYER27_PATCHING.csv` — 5 pairs of real data
4. Read `R_V_PAPER/research/MIXTRAL_LAYER27_GEOMETRY_AND_CAUSALITY.md` — the 302-line narrative
5. Read `models/mixtral_8x7b_analysis.py` — does it compute 24.3%?
6. `ls ~/Desktop/MECH\ INTERP\ FILES/MECH\ INTERP\ JUPYTER\ TEST\ NOV13-16/` — check for Mixtral CSV
7. Try to read `~/Desktop/MECH INTERP FILES/MECH INTERP JUPYTER TEST NOV13-16/MIXTRAL_8x7B_RESULTS.csv` (may timeout if on iCloud)
8. Check `configs/discovery/mixtral_8x7b_v0_1/` — what was configured?
9. Check `configs/gold/28_mixtral_causal_validation.json`
10. `grep -r "24\.3\|0\.243" results/ --include="*.json"`
11. Check `archive/scripts/` for Mixtral scripts

**Report**: State whether the 80-prompt raw data exists ANYWHERE on this machine. If only the 5-pair patching CSV exists, compute the effect size from those 5 pairs and report it.

---

### CLAIM 5: Bayes Factor BF10=9.5e23

**Paper says**: BF10=9.5e23 for the primary Mistral effect.
**Prior audit says**: Not stored in any file. Computation code exists in `scripts/statistical_hardening.py`.

**Search instructions**:
1. `grep -r "9\.5e23\|9\.5e+23\|9500000000000000000000000" results/ R_V_PAPER/ docs/`
2. `grep -r "BF10\|bf10\|bayes_factor\|bayes" results/ --include="*.json"`
3. Read `results/statistical_hardening/hardening_summary_20260311_151203.json` — does it contain BF values?
4. Read `results/fdr_correction/fdr_results_20260311_045959.json` — does it contain BF values?
5. Read `scripts/statistical_hardening.py` lines 377-400 — the BF computation code
6. Check `R_V_PAPER/MORNING_BRIEFING_2026-03-11.md` — mentions corrected BF values (necessity >10^54, KV >10^18, bridge 6.8e3). Where do THOSE come from?
7. `grep -r "bayes\|BF" results/ --include="*.json" -l`

**Report**: State whether 9.5e23 exists in ANY data file. If the hardening summary contains BF values, report what they are. If no BF is stored, confirm that.

---

### CLAIM 6: Head count 630/691 (p<0.05 uncorrected)

**Paper says**: 630 heads (61.5%) with significant entropy separation, 691 (67.5%) significant on either metric.
**Prior audit says**: These numbers come from Instruct-v0.2 head sweep, NOT base v0.1. Base sweep shows 606.

**Search instructions**:
1. List ALL files in `results/full_head_sweep/`
2. Read `results/full_head_sweep/full_head_sweep_20260302_074757.json` — this should be BASE. What head counts?
3. Read `results/full_head_sweep/full_head_sweep_20260310_151508.json` — this should be INSTRUCT. What head counts?
4. Read `results/full_head_sweep/full_head_sweep_20260311_120236.json` — INSTRUCT n=100. What head counts?
5. For each file, check: (a) what model was used, (b) n per condition, (c) count of p<0.05 entropy-sig heads, (d) count of p<0.05 either-metric heads
6. `grep -r "630\|691\|606\|681" results/full_head_sweep/ --include="*.json"`
7. Check if model name is stored in each JSON

**Report**: For EACH head sweep file, state: model name, n, entropy-sig count, either-metric count. Confirm which is base vs instruct.

---

### CLAIM 7: Word count correlation r=-0.171, p=0.498

**Paper says**: R_V does not predict word count (r=-0.171, p=0.498).
**Prior audit says**: Real null result but no dedicated source file. Mentioned in RV_MASTER_PLAN.

**Search instructions**:
1. `grep -r "0\.171\|0\.498" results/ R_V_PAPER/ docs/ industry_grade/`
2. `grep -r "word_count\|word.count\|token_count" results/ --include="*.json" --include="*.csv"`
3. Read `results/behavioral_nboost_summary.json` — does it contain word count correlation?
4. Read `results/batch_per_token_rv/summary.json` — does it contain word count data?
5. Check ALL files in `industry_grade/2026-02-20/evidence/` — the behavioral power run
6. Search for any `.csv` in results/ that might contain per-sample word counts
7. Check `R_V_PAPER/RV_MASTER_PLAN_V2.md` line 76 — it cites this number, does it cite a source file?

**Report**: Find the raw data that produces r=-0.171. If it's computed inline in a notebook or script but never saved, say so.

---

### CLAIM 8: Necessity Cohen's h=1.31

**Paper says**: Dual-layer break shows Cohen's h=1.31 for necessity.
**Prior audit says**: Source stores d=1.664 (turn-level Cohen's d) or d=3.29 (session-level). The h vs d confusion is a unit conversion issue.

**Search instructions**:
1. Read `results/persistent_patching_v3/persistent_patching_v3_dual_20260310_204100.json` — find the stored effect size
2. Read `results/persistent_patching_v3/persistent_patching_v3_dual_20260225_002604.json` — older base result
3. `grep -r "1\.31\|cohen.*h\|cohens_h" results/ --include="*.json"`
4. `grep -r "1\.664\|3\.29" results/persistent_patching_v3/ --include="*.json"`
5. Read `R_V_PAPER/STATISTICAL_EVIDENCE_AUDIT.md` — find F1 (necessity) entry
6. Read `scripts/statistical_hardening.py` — does it compute Cohen's h anywhere?
7. Check: Cohen's h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2)). If break goes from 54.7% to 0.0%, compute h manually: h = 2*arcsin(sqrt(0.547)) - 2*arcsin(sqrt(0.0)) = 2*arcsin(0.7396) = 2*0.8330 = 1.666. Does that match?
8. If break goes from 56% to 3.7%: h = 2*arcsin(sqrt(0.56)) - 2*arcsin(sqrt(0.037)) = 2*0.8411 - 2*0.1935 = 1.6822 - 0.3870 = 1.295 ≈ 1.31. Check THIS computation.

**Report**: State the exact stored effect size, the proportions used, and whether h=1.31 can be derived from 56%→3.7% (the older numbers) vs 54.7%→0.0% (the newer hardened numbers). The h=1.31 may be correct for the OLD data but wrong for the NEW data.

---

## OUTPUT FORMAT

For each claim, report:

```
## CLAIM N: [short name]
- **Paper value**: [what the paper says]
- **Files searched**: [list every file you checked]
- **Files containing the value**: [exact paths, or NONE]
- **Actual value in closest source**: [what the data really shows]
- **Verdict**: CONFIRMED / MISLABELED / WRONG_VALUE / WRONG_N / FABRICATED / ORPHANED
- **Recommended fix**: [specific action]
```

## HARD RULES

1. Do NOT accept narrative documents (`.md` writeups) as "source" — only raw data files (`.json`, `.csv`, `.npy`) count
2. Do NOT accept a value appearing in a planning doc as evidence it was measured
3. If a value appears in results JSON, check the model field — is it base v0.1 or Instruct v0.2?
4. If you find the value in a different experiment than claimed, that's MISLABELED not CONFIRMED
5. Report the FULL file path for every source you find
6. If you compute a derived value (like Cohen's h from proportions), show your work

## DIRECTORIES TO SEARCH (PRIORITY ORDER)

```
results/                          # Primary — all experiment outputs
R_V_PAPER/results/                # Secondary — curated paper results
industry_grade/                   # Tertiary — hardened analyses
RECOVERED_GOLD/                   # Historical — recovered from earlier work
docs/findings/                    # Analysis writeups (not raw data)
docs/status/                      # Audit documents
archive/                          # Old/superseded work
configs/                          # Experiment configs (for provenance)
~/Desktop/MECH INTERP FILES/      # Original notebooks (may be on iCloud)
```
