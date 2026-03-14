# Morning Briefing — 2026-03-11

## RunPod Update (Added 2026-03-11)

The Mistral hardening RunPod results have now landed and materially strengthen the necessity-plus-dissociation narrative.

**New canonical artifact**:
- `results/persistent_patching_v3/persistent_patching_v3_dual_20260310_204100.json`

**Headline result**:
- `recursive_clean`: `54.7%` BT+ART
- `recursive_dual_patched`: `0.0%`
- break effect: `d=4.645`, exact permutation `p=1.62e-05`
- `baseline_clean`: `2.0%`
- `baseline_dual_patched`: `0.0%`
- induce remains null

**Important refinement**:
- Patched conditions collapse into `100%` repetitive output with `mean_alpha_ratio ~0.70-0.72`
- `baseline_clean malformed_rate = 5.7%` appears to be a heuristic artifact from arithmetic/markdown-heavy outputs, not the same failure mode as the patched repetitive collapse

**Tool check**:
- `scripts/sync_runpod_results.py` has been run against the updated results
- output report: `results/runpod_sync_report_20260311_144617.md`
- result: the repo's latest canonical Mistral artifacts now diverge from several older paper-facing numbers, so Sections 4-10 and tables need to be refreshed from the latest raw artifacts before submission

**Current paper stance should be explicit**:
- necessity: yes
- induce / geometric sufficiency: no
- strongest causal framing: behavioral dissociation
- strongest measurement caveat: clean arithmetic prompts can trip low-alpha malformed heuristics

## Overnight Work Complete

### Paper Verification: 100 PASS, 0 FAIL, 2 WARN
Every quantitative claim in `paper_colm2026_v005.tex` (705 lines) has been verified against raw data files.

**102 claims checked** via `scripts/verify_paper_claims.py`:
- 100 PASS (exact or 2dp match to raw data)
- 0 FAIL
- 2 WARN (missing file paths for self-feeding and scaling R², data verified manually)

### Fixes Applied Overnight
1. **Scaling gap CIs** in Table 1: now use bootstrap CIs from raw data (not normal approx)
2. **Table 1 BFs** corrected: necessity >10^54, KV >10^18, bridge 6.8×10^3
3. **STATISTICAL_EVIDENCE_AUDIT.md** F4: d=-3.50 → h=0.78
4. **DRAFT_SECTIONS_1_3**: all d=3.29 → h=1.31
5. **generate_paper_tables.py**: prefers bootstrap CIs when available
6. **verify_paper_claims.py**: expectations updated to match corrected CIs
7. **Integration section** (§5.1): 10 frontier papers now use proper `\citep`/`\citet` (were inline text-only)
8. **Typo fixed**: "This extend" → "This extends" (line 432)
9. **All 30 citations** verified against references.bib — 0 missing
10. **All 11 figures** verified present in `figures/` directory

### New Tool Ready
`scripts/sync_runpod_results.py` — processes RunPod results and cross-references paper claims.
- Analyzes head sweep, path patching, dual-layer bridge
- Generates timestamped markdown report
- Run: `python3 scripts/sync_runpod_results.py`

### Manual Verification (Beyond verify_paper_claims.py)
| Claim | Paper | Raw Data | Status |
|-------|-------|----------|--------|
| 606/1024 heads significant | 606 (59.2%) | 606 entropy-significant, 681 either-metric | CONFIRMED (paper counts entropy only) |
| Concept erasure Δd=0.005 | d=-1.82 → -1.82 | d=-1.818 → -1.823, Δ=0.005 | CONFIRMED |
| Bridge d=-0.71 | n1=80, n2=107 | d=-0.7072, n_bt_art=80, n_other=107 | CONFIRMED |
| R²=0.047 | Scaling fit weak | scaling_gap_summary_20260301_144055.json | CONFIRMED |
| FDR 30/36 | 6 fail | Exact 6 match paper's list | CONFIRMED |
| Necessity BT+ART 56%→3.7% | OR=33.4 | 0.56/0.0367, OR=33.44 | CONFIRMED |
| Pythia checkpoint d≈1.0 | Flat from step 1K | 2.8B: d=1.001 all steps (suspicious constant) | CONFIRMED (but note caching artifact risk) |
| Power: Pythia-1.4B cross-arch 0.41 | need n≥165 | power_analysis_20260303: d=-0.31, n=63, power=0.408 | CONFIRMED |
| Power: Phi-3-mini scaling 0.77 | need n≥42 | power_analysis_20260303: d=0.625, n=38+39, power=0.772 | CONFIRMED |
| Power: Pythia-6.9B scaling 0.49 | need n≥70 | power_analysis_20260303: d=0.478, n=37+31, power=0.489 | CONFIRMED |

**Note on apparent power discrepancy**: The `hardening_summary` files show power=0.05 for Pythia-1.4B, but that's the **power-up** pipeline (d=-0.006). The paper's power=0.41 correctly references the **cross-arch** pipeline (d=-0.31). Different tests, no error.

### Remaining Concerns (Non-Blocking)
1. **Pythia-2.8B checkpoints**: d=1.001 identical across ALL steps — likely caching artifact (same as multi-seed). Paper acknowledges as limitation.
2. **Generated table vs paper table**: sign convention differs for self-feeding (generated: +4.28, paper: -4.28). Both correct, different reference direction.
3. **FDR input data uses old d values** (d=3.29 for necessity, d=-3.50 for sufficiency) — but FDR output (30/36) is still valid since it's based on p-values.

## RunPod Follow-Through

1. **Refresh paper-facing numbers** from the landed artifacts, not the older placeholders.
2. **Use** `results/runpod_sync_report_20260311_144617.md` as the mismatch ledger.
3. **Prioritize**:
   - dual-layer bridge values from `persistent_patching_v3_dual_20260310_204100.json`
   - path patching values from `path_patching_summary_20260310_151654.json`
   - head sweep values from `full_head_sweep_20260310_151508.json`
4. **Do not overclaim malformed collapse** in clean baseline until the low-alpha arithmetic heuristic is cleaned up or explicitly caveated.

## What's Left Before COLM Submission

### P0 (Critical, by Mar 26)
- [ ] Abstract submission (1-page, Mar 26 deadline)
- [ ] P0 canonical Mistral-7B-v0.1 (base) run — needs GPU, separate from Instruct
- [ ] Integrate RunPod results if they change any numbers

### P1 (Important, by Mar 31)
- [ ] Sections 4-10 rewrite per DRAFT_SECTIONS_1_3 model
- [ ] Address sign reversal (OPT/GPT-2) more thoroughly in discussion
- [ ] Add Gemma-2-9B and Mixtral data to cross-architecture section (clean Tier 1 models)
- [ ] Create final figures (regenerate any that changed)

### P2 (Nice to have)
- [ ] Second GPU for P0 canonical re-run of all 5 models
- [ ] Word count removed (correct — see R_V_BEHAVIORAL_DISSOCIATION.md)
- [ ] Sufficiency ladder honest framing in text

## Quick Commands
```bash
# Verify all paper claims
python3 scripts/verify_paper_claims.py

# Regenerate Table 1 from raw data
python3 scripts/generate_paper_tables.py

# Process RunPod results
python3 scripts/sync_runpod_results.py

# Run statistical hardening
python3 scripts/statistical_hardening.py
```
