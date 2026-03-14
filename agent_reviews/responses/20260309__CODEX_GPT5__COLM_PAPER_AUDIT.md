Title: COLM 2026 PAPER AUDIT
Date: 2026-03-09
Model: Codex (GPT-5)
Audit duration: ~2 hours

## PART A: CLAIM-BY-CLAIM VERIFICATION

### Claim C1: Mistral-7B shows contraction with d=-1.66, CI [-2.08, -1.32], n=152
- **Paper says:** Lines 195 and 525 say Mistral-7B has `d=-1.66`, `95% CI [-2.08, -1.32]`, with `n=152` and table split `75/77`.
- **Data file:** `results/power_up/mistral-7b_n80_result.json`
- **Data shows:** `cohens_d=-1.6564878536967445`; `ci_95=[-2.079492571500931, -1.3186160006653789]`; `n_recursive=75`; `n_baseline=77`; `rv_recursive_mean=0.685975896685412`; `rv_baseline_mean=0.8550126067144158`.
- **Verdict:** CONFIRMED
- **Severity:** LOW
- **Notes:** No Mistral `n>80` file exists in `results/power_up/`; the reported `n=152` is the sum of valid recursive and baseline prompt-pass measurements from the `n80` run.

### Claim C2: Qwen2.5-7B shows contraction with d=-2.32, CI [-2.86, -1.90], n=124
- **Paper says:** Lines 196 and 526 say Qwen2.5-7B has `d=-2.32`, `95% CI [-2.86, -1.90]`, `n=124`, with table split `61/63`.
- **Data file:** `results/power_up/qwen2.5-7b_n80_result.json`; `results/power_up/qwen2.5-7b_n100_result.json`
- **Data shows:** In `qwen2.5-7b_n80_result.json`, `cohens_d=-2.3180613840620405`; `ci_95=[-2.8625799873806193, -1.8955014222424906]`; `n_recursive=61`; `n_baseline=63`; `rv_recursive_mean=0.903061024175268`; `rv_baseline_mean=1.3292348062793076`. `qwen2.5-7b_n100_result.json` is a 0-byte empty file.
- **Verdict:** CONFIRMED
- **Severity:** LOW
- **Notes:** The exact paper numbers come from the `n80` artifact; the expected `n100` artifact exists only as an empty placeholder.

### Claim C3: OPT-6.7B body text uses |d|=1.68; what is the signed d?
- **Paper says:** Line 197 reports `|d|=1.68` for OPT-6.7B.
- **Data file:** `results/power_up/opt-6.7b_n80_result.json`
- **Data shows:** `cohens_d=1.6825357789104158`; `ci_95=[1.3481539379301815, 2.091058734349604]`; `n_recursive=72`; `n_baseline=66`; `rv_recursive_mean=1.1150298720804228`; `rv_baseline_mean=0.789174049782326`.
- **Verdict:** CONTRADICTED
- **Severity:** CRITICAL
- **Notes:** The raw prompt-pass result is positive, so OPT-6.7B shows expansion, not contraction. The body text hides the sign; Table 1 line 527 prints the positive sign.

### Claim C4: GPT-2 XL body text uses |d|=1.52; what is the signed d?
- **Paper says:** Line 198 reports `|d|=1.52`, with CI `[1.07, 2.05]`, for GPT-2 XL.
- **Data file:** `results/power_up/gpt2-xl_n80_result.json`
- **Data shows:** `cohens_d=1.5162788889116183`; `ci_95=[1.0699010891149683, 2.052843535388862]`; `n_recursive=69`; `n_baseline=56`; `rv_recursive_mean=0.872281857323008`; `rv_baseline_mean=0.710956692992414`.
- **Verdict:** CONTRADICTED
- **Severity:** CRITICAL
- **Notes:** The raw prompt-pass result is positive, so GPT-2 XL also shows expansion, not contraction.

### Claim C5: Pythia-1.4B shows d=-0.006, p=0.88, n=124
- **Paper says:** Line 199 and Table 1 line 529 report `d=-0.006`, `p=0.88`, `n=124`, with table split `63/61`.
- **Data file:** `results/power_up/pythia-1.4b_n80_result.json`
- **Data shows:** `cohens_d=-0.005656024342823943`; `p_value=0.8763365998141571`; `n_recursive=66`; `n_baseline=54`; total valid `n=120`; `rv_recursive_mean=0.6329022756533812`; `rv_baseline_mean=0.633056267906811`.
- **Verdict:** PARTIAL
- **Severity:** MEDIUM
- **Notes:** The effect size and p-value match closely, but the paper's `n=124` is not supported by the raw file.

### Claim C6: “The contraction replicates in four models.”
- **Paper says:** Line 194 says “The contraction replicates in four models with large effects.”
- **Data file:** `results/power_up/mistral-7b_n80_result.json`; `results/power_up/qwen2.5-7b_n80_result.json`; `results/power_up/opt-6.7b_n80_result.json`; `results/power_up/gpt2-xl_n80_result.json`; `results/power_up/pythia-1.4b_n80_result.json`
- **Data shows:** Negative `d` only for Mistral (`-1.6564878536967445`) and Qwen (`-2.3180613840620405`). OPT is positive (`1.6825357789104158`), GPT-2 XL is positive (`1.5162788889116183`), and Pythia-1.4B is null (`-0.005656024342823943`).
- **Verdict:** CONTRADICTED
- **Severity:** CRITICAL
- **Notes:** On the raw `power_up` prompt-pass results, contraction does not replicate in four models.

### Claim C7: Table 1 n1/n2 values match the JSON files
- **Paper says:** Table 1 lines 525-529 report splits: Mistral `75/77`, Qwen `61/63`, OPT `69/69`, GPT-2 XL `56/69`, Pythia-1.4B `63/61`.
- **Data file:** `results/power_up/mistral-7b_n80_result.json`; `results/power_up/qwen2.5-7b_n80_result.json`; `results/power_up/opt-6.7b_n80_result.json`; `results/power_up/gpt2-xl_n80_result.json`; `results/power_up/pythia-1.4b_n80_result.json`
- **Data shows:** Mistral matches (`75/77`). Qwen matches (`61/63`). OPT raw split is `72/66`, not `69/69`. GPT-2 XL raw split is `69/56`, not `56/69`. Pythia-1.4B raw split is `66/54`, not `63/61`.
- **Verdict:** CONTRADICTED
- **Severity:** HIGH
- **Notes:** Three of the five Table 1 rows have unsupported sample splits.

### Claim C8: Necessity d=3.29, n=300, “breaks V-projections at L25 and L27”
- **Paper says:** Lines 228-232 and Table 1 line 519 claim dual-layer necessity with `d=3.29`, `n=300`, and “breaking both V-projections at L25 and L27.”
- **Data file:** `results/path_patching/path_patching_summary_20260227_080128.json`; `results/persistent_patching_v3/persistent_patching_v3_dual_20260225_002604.json`
- **Data shows:** `path_patching_summary_20260227_080128.json` tests `component in {residual, v_proj, mlp}` at layers `0,2,...,30`; it does not test layer 25 and has no layer-27 entry. `persistent_patching_v3_dual_20260225_002604.json` explicitly says `"description": "Dual-layer persistent patching: L18 residual + L27 V-proj, 4-condition break+induce"` and is not an `L25+L27 V-proj` break experiment.
- **Verdict:** NO_DATA
- **Severity:** CRITICAL
- **Notes:** I did not find a raw result file for the exact experiment described in the paper. The `d=3.29` value appears in `scripts/statistical_hardening.py` and `results/statistical_hardening/hardening_summary_20260227_075339.json`, but those are hardcoded summary numbers, not raw experiment outputs.

### Claim C9: Sufficiency d=-3.50, n=300, OR=13.96 for KV injection
- **Paper says:** Lines 234-236 and Table 1 line 520 claim KV-only sufficiency with `d=-3.50`, `n=300`, and `OR=13.96`, implying the geometric pattern is sufficient.
- **Data file:** `results/sufficiency_ladder/sufficiency_ladder_20260225_101907.json`; `docs/findings/R_V_BEHAVIORAL_DISSOCIATION.md`
- **Data shows:** In `sufficiency_ladder_20260225_101907.json`, `kv_only_vs_baseline.turn_level.or=13.960829493087557` and `test_rate=0.27666666666666667` vs `base_rate=0.02666666666666667`, but `kv_only_vs_baseline.session_level.cohens_d=1.4650909234700569`, not `-3.50`. Also `kv_only.mean_rv=0.5727770806515043` is slightly higher than `clean_baseline.mean_rv=0.5548804725652157`, so low-`R_V` geometry is not transferred. `docs/findings/R_V_BEHAVIORAL_DISSOCIATION.md` states “behavioral transfer” but `R_V` transfer `d=0.11 (NS)`.
- **Verdict:** CONTRADICTED
- **Severity:** CRITICAL
- **Notes:** The OR is real, but it is a behavioral rate effect. The raw current Mistral sufficiency ladder does not show geometric transfer, and the quoted `d=-3.50` is not present in the raw sufficiency file.

### Claim C10: “reduces BT+ART from 56% to 27.7%”
- **Paper says:** Line 231 says dual-layer necessity reduces BT+ART “from 56% to 27.7%”.
- **Data file:** `results/persistent_patching_v3/persistent_patching_v3_dual_20260225_002604.json`; `results/sufficiency_ladder/sufficiency_ladder_20260225_101907.json`
- **Data shows:** In `persistent_patching_v3_dual_20260225_002604.json`, `recursive_clean.bt_art_rate=0.56` and `recursive_dual_patched.bt_art_rate=0.03666666666666667` (`56% -> 3.67%`). The `27.7%` figure appears instead in `sufficiency_ladder_20260225_101907.json` as `kv_only.bt_art_rate=0.27666666666666667`.
- **Verdict:** CONTRADICTED
- **Severity:** CRITICAL
- **Notes:** The paper appears to combine the recursive-clean `56%` necessity baseline with the separate KV-only sufficiency `27.7%` rate.

### Claim C11: Within-session bridge d=-0.71, n=150
- **Paper says:** Lines 238-239 and Table 1 line 521 report `d=-0.71`, `n=150`, `p<10^-6`.
- **Data file:** `results/within_session_bridge/within_session_bridge_20260220_201515.json`; `results/within_session_bridge/within_session_bridge_20260221_000441.json`; `industry_grade/2026-02-20/evidence/BEHAVIORAL_POWER_RUN_WRITEUP_2026-02-20.md`
- **Data shows:** The older file `within_session_bridge_20260220_201515.json` has `pooled.recursive_only.output_rv.cohens_d=-0.7071922399278552`, `n_bt_art=80`, `n_other=107`, `mean_bt_art=0.48780945080448646`, `mean_other=0.600531474740143`. The newer file `within_session_bridge_20260221_000441.json` has `pooled.recursive_only.output_rv.cohens_d=-0.5669381431230488`, `n_bt_art=254`, `n_other=308`. The writeup explicitly says the old value `-0.7071922399278552` updated to `-0.5669381431230488`.
- **Verdict:** PARTIAL
- **Severity:** HIGH
- **Notes:** The paper's effect size matches an older snapshot, but the paper's `n=150` is not found in the raw JSONs, and the latest bridge artifact reports a different effect size.

### Claim C12: V-projection path patching supports the “Value Spaces” mechanism
- **Paper says:** Lines 228-236 assert necessity and sufficiency of value-space geometry; the title frames the phenomenon as “Geometric Contraction in Transformer Value Spaces.”
- **Data file:** `results/path_patching/path_patching_summary_20260227_080128.json`
- **Data shows:** V-proj Cohen's d by tested layer is: `L0=-0.7187634135915841`, `L2=0.06993329617603013`, `L4=-0.0076749564315409845`, `L6=-0.010498703173376738`, `L8=0.04297593272071711`, `L10=-0.060695358166976576`, `L12=-0.07846251136637362`, `L14=0.2196962668886364`, `L16=-0.04905672438350481`, `L18=0.19357006789123135`, `L20=0.05718492216072809`, `L22=0.08796507134420548`, `L24=0.01297664588023767`, `L26=-0.02193254757258386`, `L28=0.0`, `L30=0.0`. Max `|d|` is only `0.7187634135915841` at layer 0. Residual patching in the same file reaches much larger effects, e.g. layer 4 residual `cohens_d=-1.9616225183631633`.
- **Verdict:** CONTRADICTED
- **Severity:** HIGH
- **Notes:** The raw path-patching evidence does not show a strong or late-layer-specific V-projection mechanism. It weakens, rather than strengthens, a strong “value-space-specific” causal reading.

### Claim C13: Self-referential R_V mean = 0.650, SD = 0.098, d=-1.67 vs all modes
- **Paper says:** Line 200 and Figure/atlas discussion imply the mode-atlas self-referential mean is `0.650`, `SD 0.098`, with `d=-1.67`.
- **Data file:** `results/mode_atlas/atlas_summary_20260227_075328.json`
- **Data shows:** `fingerprint.self_referential.rv.mean=0.6501590407213126`; `fingerprint.self_referential.rv.std=0.09811429122171737`; `fingerprint.self_referential.rv.n=20`. Using `all_results` in the same file, self-referential vs all other valid mode samples gives `d=-1.6677147172445814` with `n_self=20`, `n_other=132`.
- **Verdict:** CONFIRMED
- **Severity:** LOW
- **Notes:** The d-value is not stored directly in the file; it is recoverable from the raw per-prompt `all_results`.

### Claim C14: 606/1024 heads significant at p<0.05
- **Paper says:** Lines 252-256 and Figure 4 caption say 606 of 1,024 heads are significant.
- **Data file:** `results/full_head_sweep/full_head_sweep_20260302_074757.json`
- **Data shows:** The file has per-head keys `entropy_d`, `entropy_p`, `rank_d`, `rank_p` rather than per-head `R_V`. Counting raw p-values gives `606` heads with `entropy_p<0.05`, `169` heads with `rank_p<0.05`, and `681` heads significant on at least one metric. The top entropy effect is `layer=10`, `head=20`, `entropy_d=3.901117832874233`, `entropy_p=1.0645689837502488e-07`.
- **Verdict:** PARTIAL
- **Severity:** HIGH
- **Notes:** The count `606/1024` is real for head-level entropy, but the file does not contain per-head `R_V` significance as the paper text suggests.

### Claim C15: Perplexity matching survives with d=-1.67, p=0.002, n=8 strict pairs
- **Paper says:** Lines 318-320 report strict perplexity-matched survival with `d=-1.67`, `p=0.002`, `n=8`.
- **Data file:** `results/perplexity_repairing/repairing_results_20260303_233230.json`
- **Data shows:** `strict_results.n_pairs=8`; `strict_results.d_paired=-1.6647371491401157`; `strict_results.p_paired=0.0021864727162156337`.
- **Verdict:** CONFIRMED
- **Severity:** LOW
- **Notes:** The looser matched result in the same file is stronger: `matched_results.d_paired=-1.7998411954326712`.

### Claim C16: Multi-seed all 5 seeds give identical d=-1.751
- **Paper says:** Lines 326-329 report all five seeds produce identical `d=-1.751`.
- **Data file:** `results/power_up/multi_seed_summary_20260306.json`
- **Data shows:** `seeds=[42,137,2026,31415,27182]`; `d_values=[-1.7514030206375515, -1.7514030206375515, -1.7514030206375515, -1.7514030206375515, -1.7514030206375515]`; `d_std=0.0`.
- **Verdict:** CONFIRMED
- **Severity:** LOW
- **Notes:** Each seed-level result in the same file also repeats the same CI and p-value.

### Claim C17: FDR 30/36 survive at alpha=0.05
- **Paper says:** Lines 307-309 and 547-548 say `30/36` survive BH correction at `alpha=0.05`.
- **Data file:** `results/fdr_correction/fdr_results_20260303_232741.json`
- **Data shows:** `alpha=0.05`; `n_significant_fdr=30`; `n_tests=36`. The six non-significant tests are `Cross-arch: Pythia-1.4B R_V`, `pythia-6.9b`, `pythia-2.8b`, `pythia-1b`, `pythia-1.4b`, and `genuine vs deceptive`.
- **Verdict:** CONFIRMED
- **Severity:** LOW
- **Notes:** The raw non-significant set matches the appendix description.

### Claim C18: L27H10 effective rank 7.28 -> 5.91, d=-1.54
- **Paper says:** Lines 266-269 report L27H10 effective rank `7.28 -> 5.91`, `d=-1.54`.
- **Data file:** `results/svd_circuits/svd_decomposition_20260306_131647.json`
- **Data shows:** `head_results.L27_H10.eff_rank_baseline_mean=7.284222507518746`; `eff_rank_recursive_mean=5.913905693007832`; `d_eff_rank=-1.5437437482309984`; `d_top1_ratio=1.616732736552503`.
- **Verdict:** CONFIRMED
- **Severity:** LOW
- **Notes:** The same values also appear in `results/svd_circuits/svd_decomposition_20260304_122437.json`.

### Claim C19: L5H29 expansion d=2.93
- **Paper says:** Lines 267-269 report L5H29 rank expansion with `d=2.93`.
- **Data file:** `results/svd_circuits/svd_decomposition_20260306_131647.json`
- **Data shows:** `head_results.L5_H29.eff_rank_baseline_mean=6.995092172285853`; `eff_rank_recursive_mean=9.526073175050913`; `d_eff_rank=2.9280631359655525`; `d_top1_ratio=-2.7079986772378795`.
- **Verdict:** CONFIRMED
- **Severity:** LOW
- **Notes:** This is a genuine sign flip relative to late-layer contraction heads.

### Claim C20: Concept erasure leaves the effect unchanged (d=-1.82 before and after; delta=0.005)
- **Paper says:** Lines 280-284 say `d=-1.82` before erasure and `d=-1.82` after, with `Δd=0.005`.
- **Data file:** `results/linear_probe/probe_analysis_20260306_153537.json`
- **Data shows:** `concept_erasure.d_before=-1.8183126503137932`; `d_after=-1.823395858520043`; absolute difference `0.005083208206249745`; `rv_recursive_before_mean=0.6503718313670589`; `rv_recursive_after_mean=0.6497153203055595`; `rv_baseline_before_mean=0.8530880922087759`; `rv_baseline_after_mean=0.853176687717671`.
- **Verdict:** CONFIRMED
- **Severity:** LOW
- **Notes:** The raw file slightly strengthens, rather than exactly preserves, the d-value.

### Claim C21: DII at L27 shows every PCA dimension has R_V ≈ 0.41
- **Paper says:** Lines 286-290 say that at L27 “every individual PCA dimension shows `R_V ≈ 0.41`.”
- **Data file:** `results/dii_intervention/dii_results_20260305_122736.json`
- **Data shows:** `layers.late.layer=27`. Per-dimension `rv` for dims `0-19` ranges from `0.367` to `0.473`, with many values at `0.406-0.416`; `layers.late.grouped` top-20 gives `rv=0.324`, `d_vs_clean=-3.42`.
- **Verdict:** CONFIRMED
- **Severity:** LOW
- **Notes:** The claim is approximate but fair; all listed late-layer single-dimension values are in the same narrow, strongly contractive band.

### Claim C22: RSA maximum dissimilarity is at L28 (distance 0.307)
- **Paper says:** Lines 292-294 say maximum dissimilarity occurs at Layer 28 with distance `0.307`.
- **Data file:** `results/rsa/rsa_analysis_20260302_123257.json`
- **Data shows:** `selfref_distance_trajectory` includes `L28=0.30672680230647636`, but also `L0=0.3850953261257697`, which is larger. The minimum is `L4=0.08690060565846025`.
- **Verdict:** CONTRADICTED
- **Severity:** MEDIUM
- **Notes:** The paper's L28 value is the maximum only if layer 0 is excluded without saying so.

### Claim C23: AUROC = 0.909 for self-referential detection
- **Paper says:** Lines 343-344 and Figure 6 say AUROC `=0.909`.
- **Data file:** `results/safety/safety_analysis_20260302_123229.json`
- **Data shows:** `e53_deployment_monitoring.auroc=0.9088871308016878`; `n_total=500`; `n_selfref=50`; `n_diverse=450`; `best_threshold=0.7366319324884241`; `best_tpr=0.8333333333333334`; `best_fpr=0.13924050632911392`.
- **Verdict:** CONFIRMED
- **Severity:** LOW
- **Notes:** The rounded AUROC matches the paper.

### Claim C24: Genuine vs deceptive d=-0.06
- **Paper says:** Lines 346-348 say genuine and deceptive self-reference are geometrically indistinguishable with `d=-0.06`.
- **Data file:** `results/safety/safety_analysis_20260302_123229.json`
- **Data shows:** `e51_genuine_vs_deceptive.d_genuine_vs_deceptive=-0.06076466279505974`; `genuine_rv_mean=0.6472502632403718`; `genuine_rv_std=0.09868310610029962`; `deceptive_rv_mean=0.6526344704085851`; `deceptive_rv_std=0.07261858961096519`.
- **Verdict:** CONFIRMED
- **Severity:** LOW
- **Notes:** The same file also supports the paper's baseline separations: genuine vs baseline `-1.8867501564849152`, deceptive vs baseline `-2.104843507242892`.

### Claim C25: Scaling fit R²=0.047 with 8 data points
- **Paper says:** Line 472 says the scaling fit has `R^2=0.047` with `8` data points.
- **Data file:** `results/scaling_gap/scaling_gap_summary_20260301_144055.json`; `results/scaling_law/scaling_law_summary_20260227_104843.json`
- **Data shows:** In `scaling_gap_summary_20260301_144055.json`, `scaling_fit.r_squared=0.047008339848824736`, `scaling_fit.n_points=6`. The same file lists seven model entries, but `pythia-410m` has `cohens_d=NaN`, so it is excluded from the fit. A different file, `scaling_law_summary_20260227_104843.json`, gives `r_squared=0.5347930981564958`.
- **Verdict:** CONTRADICTED
- **Severity:** HIGH
- **Notes:** The quoted `R^2≈0.047` exists, but it is based on 6 fitted points, not 8, and the repo also contains a materially different scaling summary.

## PART B: ORPHAN FINDINGS SCAN

Scan notes:
- Reviewed all 5 files in `docs/findings/`.
- Reviewed all 7 files in `RECOVERED_GOLD/`.
- `results/phase1_cross_architecture/runs/` contains only the five paper model families plus Mistral bridge runs; I did not find an extra model family in that directory itself.

### Orphan O1: Behavioral transfer without R_V transfer
- **Source file:** `docs/findings/R_V_BEHAVIORAL_DISSOCIATION.md`
- **Key stat:** Behavioral transfer `Cohen's d = 2.494`, but `R_V` transfer `d = 0.11 (NS)`; marker rate `0.3 -> 27.4`.
- **Should be in paper?** YES_CRITICAL
- **Why:** This directly undercuts the paper's sufficiency framing. The paper currently treats KV-based behavior transfer as if it transfers the `R_V` geometry; this repo finding says it does not.

### Orphan O2: GQA headspace correction flips control direction
- **Source file:** `results/meta_yolo/runs/20260220_102900_offline_meta_yolo/summary.json`
- **Key stat:** `bridge_specificity.comparisons.v4_head_specific_vs_v4_random_head.cohens_d=-1.2539166694831796`; `v4_head_specific_vs_v4_baseline_donor.cohens_d=-1.3701282954126641`; `random_head_version_flip.sign_changed=true`.
- **Should be in paper?** YES_USEFUL
- **Why:** The repo has a strong methodology finding that head-level causal conclusions in GQA models depend on correct headspace semantics. The paper omits this entirely.

### Orphan O3: Multi-token behavior linkage is truncation-confounded
- **Source file:** `industry_grade/2026-02-20/evidence/multi_token_confound_analysis.json`
- **Key stat:** At `temperature=0.0`, `pct_truncated=88.88888888888889`, `rv_word_spearman_all_r=-0.649803706520141`, and `trunc_word_pointbiserial_r=0.7086667192980836`. At `temperature=0.7`, `pct_truncated=69.44444444444444`, `rv_word_spearman_nontrunc_p=0.211545010342096`.
- **Should be in paper?** YES_CRITICAL
- **Why:** Any geometry-to-behavior bridge story is materially weakened if the behavior signal is length/truncation-sensitive. The repo documents this confound explicitly; the paper does not.

### Orphan O4: Seed-bridge specificity replicates across 3 complete seeds
- **Source file:** `industry_grade/2026-02-20/evidence/seed_bridge_analysis.json`
- **Key stat:** Pooled paired tests give `head_specific_vs_random_head_control mean_diff=-0.03838883203269895, d=-0.7770406649337096, n_pairs=180`; `head_specific_vs_baseline_donor_control mean_diff=-0.05509582593134857, d=-1.440768889976374, n_pairs=180`.
- **Should be in paper?** YES_USEFUL
- **Why:** This is a stronger specificity-control replication than most of the paper's causal section, and it lives in a pre-registered industry-grade package rather than a one-off exploratory note.

### Orphan O5: MoE amplification claim is absent
- **Source file:** `RECOVERED_GOLD/GROUND_TRUTH_ASSESSMENT.md`
- **Key stat:** “MoE amplifies effect: `24.3% vs 15.3%` (`59% stronger`).”
- **Should be in paper?** YES_USEFUL
- **Why:** If this old repo claim is still believed, it is a notable architecture effect that would matter more than several weaker paper claims. I did not re-verify this against a current raw artifact in this audit, so it should be rechecked before use.

## PART C: CODE CONSISTENCY CHECK

### R_V computation

- `src/metrics/rv.py` matches the paper equation exactly: `PR = (sum sigma_i^2)^2 / sum sigma_i^4`, `R_V = PR_late / PR_early`. It uses float64 SVD, rejects `T < window_size` with `NaN`, and defaults `late = num_layers - 5`.
- `geometric_lens/metrics.py` uses the same `PR` and `R_V` formula, and its `participation_ratio()` also returns `NaN` when `T < window_size`.
- `CANONICAL_CODE/mistral_L27_FULL_VALIDATION.py` is not contract-equivalent: it uses `W = min(window_size, T)` and `.float()`, so short prompts are silently accepted and measured in lower precision.
- `CANONICAL_CODE/causal_loop_closure_v2.py` is also not contract-equivalent: it computes `R_V` on generation-time `v_early_stack` / `v_late_stack` and silently falls back to shorter-than-window stacks with `v_early_stack[-window_size:] if ... else v_early_stack`.
- Result families use different code paths:
- `results/power_up/*` are generated by `scripts/power_up_multiseed.py`, which imports `GeometricProbe` and therefore uses `geometric_lens/*`.
- `results/mode_atlas/*` are generated by `scripts/computational_mode_atlas.py`, which also imports `GeometricProbe`.
- `results/phase1_cross_architecture/runs/*` causal-validation summaries come from `src/pipelines/canonical/rv_l27_causal_validation.py`, which imports `src.metrics.rv.participation_ratio`.
- Old behavior-transfer / recovered-gold experiments are tied to legacy scripts such as `scripts/persistent_patching_v2.py`, `scripts/persistent_patching_v3_dual.py`, and `CANONICAL_CODE/causal_loop_closure_v2.py`, which are generation-time and contract-divergent.

### Layer selection

- `geometric_lens/models.py` registry uses:
- OPT-6.7B: `early_layer=5`, `late_layer=27` (`5/32 = 15.6%`, `27/32 = 84.4%`).
- GPT-2 XL: `early_layer=7`, `late_layer=40` (`7/48 = 14.6%`, `40/48 = 83.3%`).
- Qwen2.5-7B: `early_layer=5`, `late_layer=27`.
- Pythia-1.4B: `early_layer=4`, `late_layer=20`.
- So the registry itself is broadly consistent with the paper's “~16% / ~84% depth” statement for OPT and GPT-2.
- But the canonical configs disagree with the registry:
- `configs/canonical/rv_causal_opt_6_7b.json` uses `early_layer=4`, `target_layer=27`.
- `configs/canonical/rv_causal_gpt2_xl.json` uses `early_layer=6`, `target_layer=40`.
- `configs/canonical/rv_causal_qwen2_7b.json` uses `early_layer=4`, `target_layer=24`.
- `configs/canonical/rv_causal_pythia_1_4b_n63.json` uses `early_layer=3`, `target_layer=20`.
- `src/core/model_physics.py` is a third registry and does not even define OPT-6.7B, GPT-2 XL, Qwen2.5-7B, or Pythia-1.4B specifically; unknown models fall back to `early_layer=5`, `late_layer=27`.
- Net result: the repo does not have one stable “other architectures use comparable relative depths” policy. There are at least three layer-selection policies in active files.

### Prompt bank

- `prompts/bank.json` contains `754` prompt entries total.
- Relevant group counts in `prompts/bank.json` are: `L4_full=20`, `L5_refined=20`, `L3_deeper=22`, `baseline_math=20`, `baseline_creative=20`, `long_control=20`.
- `prompts/loader.py` states “No ad-hoc lists in .py files. All prompts come from prompts/bank.json.”
- But the paper-supporting scripts violate that:
- `scripts/computational_mode_atlas.py` hardcodes `MODE_PROMPTS` for all 10 computational modes and does not call `PromptLoader`.
- `scripts/power_up_multiseed.py` hardcodes `RECURSIVE_PROMPTS` and `BASELINE_PROMPTS` (100 each) and does not call `PromptLoader`.
- Therefore power-up results and mode-atlas results do not use the same prompt source as canonical causal experiments.
- The paper's prompt-family statement at line 130 is also inaccurate for canonical causal validation: `results/phase1_cross_architecture/runs/20260202_125718_rv_l27_causal_validation_opt_6_7b/summary.json` and the analogous GPT-2/Qwen summaries show `pairing.recursive_groups=["L5_refined","L4_full","L3_deeper"]` and `pairing.baseline_groups=["long_control","baseline_creative","baseline_math"]`, not just the two recursive and two baseline families named in the paper.
- The paper's “n=300 prompt pairs” wording does not match the raw causal-behavior files. `results/persistent_patching_v3/persistent_patching_v3_dual_20260225_002604.json` and `results/sufficiency_ladder/sufficiency_ladder_20260225_101907.json` record `n_sessions=10` and `total_turns=300`; the raw `300` is turn count, not 300 unique prompt pairs.

## PART D: CONTRADICTION MAP

## CONTRADICTION SUMMARY

### Paper claims data CONTRADICTS:
- C3
- C4
- C6
- C7
- C9
- C10
- C12
- C22
- C25

### Paper claims with NO supporting data found:
- C8

### Findings in repo that SHOULD be in paper but aren't:
- O1
- O2
- O3
- O4
- O5

### Code inconsistencies that affect results:
- `results/power_up/*` prompt-pass data and `results/phase1_cross_architecture/runs/*` causal-validation data disagree on OPT-6.7B and GPT-2 XL sign, so the repo contains incompatible cross-architecture stories.
- `scripts/statistical_hardening.py` hardcodes primary effect sizes and sample sizes (`d=3.29`, `d=-3.50`, `d=-0.707`, etc.) instead of loading raw result files.
- `geometric_lens/models.py`, `src/core/model_physics.py`, and `configs/canonical/*.json` use different early/late layer choices.
- `prompts/loader.py` declares `prompts/bank.json` canonical, but `scripts/computational_mode_atlas.py` and `scripts/power_up_multiseed.py` use hardcoded prompt lists.
- Legacy “canonical” code in `CANONICAL_CODE/*` silently shortens windows and measures generation-time `R_V`, which is not the same metric contract described in the paper.

### Recommended paper changes (ranked by severity):
1. Remove or rewrite the necessity/sufficiency claims until they point to raw files that actually implement the stated experiments. In particular, stop claiming `L25+L27 V-proj` necessity and `R_V` sufficiency from KV injection without raw support.
2. Fix the cross-architecture narrative: OPT-6.7B and GPT-2 XL are expansion cases in the `power_up` prompt-pass data, not contraction cases.
3. Correct Table 1 sample splits and the Pythia-1.4B total `n`.
4. Replace hardcoded hardening numbers with values loaded from raw artifacts, or remove the hardening summary from the paper.
5. State explicitly that different sections use different code paths, different layer registries, and different prompt sources.
6. Correct the prompt-design section: canonical causal-validation summaries use `L3_deeper` and `long_control` in addition to the families named in the paper, and the raw `300` in the behavioral files is turn count rather than 300 prompt pairs.
7. Clarify or remove the RSA maximum-dissimilarity claim and the scaling-fit `8 data points` claim.
8. Clarify that the `606/1024` head count comes from head-level entropy significance, not per-head `R_V` significance.
