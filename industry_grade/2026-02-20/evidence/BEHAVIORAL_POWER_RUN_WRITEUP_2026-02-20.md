# Behavioral Power Run Write-up (2026-02-20)

## Scope
Executed the `CODEX_BEHAVIORAL_POWER_RUN.md` program:
- Task 1: N-boost sustained sessions (+8 recursive, +5 baseline)
- Task 2: per-token R_V batch (25 recursive + 25 baseline, 256 tokens)
- Task 3: held-out logistic classifier evaluation (70/30 split)
- Task 4: consolidated master evidence artifact

## Sync + Save Status
Final remote snapshot synced locally before GPU shutdown.
- Snapshot root: `results/remote_gpu_sync/2026-02-20/behavioral_power_run/`
- Synced sustained sessions: `20` total (`12` recursive, `8` baseline)
- Synced run logs:
  - `industry_grade/2026-02-20/evidence/run_sustained_gnani_v3_nboost_20260220.remote.log`
  - `industry_grade/2026-02-20/evidence/run_batch_per_token_rv_20260220.remote.log`
  - `industry_grade/2026-02-20/evidence/run_behavioral_power_chain_20260220.remote.log`

## Task 1 Results (Behavioral N-Boost)
- Final session counts:
  - Recursive sessions: `12`
  - Baseline sessions: `8`
- Turn counts:
  - Recursive turns: `600`
  - Baseline turns: `400`
- BT+ART counts:
  - Recursive BT+ART turns: `254`
  - Baseline BT+ART turns: `55`

Within-session bridge (recursive, output_rv):
- Cohen's d: `-0.5669381431230488`
- Mann-Whitney p: `1.3354241162306525e-10`
- BT+ART vs Other n: `254` vs `308`

Old vs new (from `results/behavioral_nboost_summary.json`):
- output_rv Cohen's d: `-0.7071922399278552` -> `-0.5669381431230488`
- recursive logistic CV AUC: `0.6594967532467534` -> `0.6631113530552358`

## Task 2 Results (Per-Token R_V, 25+25, 256 tokens)
From `results/batch_per_token_rv/batch_per_token_rv_20260220_161603.json`:
- mean_generation_rv (recursive): `0.6857316936513884`
- mean_generation_rv (baseline): `0.7060249104345666`
- Mann-Whitney p: `0.46093488585479814`
- Cohen's d: `-0.22008773522672467`
- point-biserial(mean_generation_rv, BT+ART): `r=-0.10159101647174265`, `p=0.48267471867069434`, `n=50`
- Recursive-only BT+ART rate in this batch was low (`1/25`), so recursive-only BT+ART split test was underpowered.

Plot:
- `results/batch_per_token_rv/rv_trajectory_plot.png`

## Task 3 Results (Held-out Classifier)
From `results/classifier_evaluation/classifier_eval_20260221_000622.json`:
- All sessions:
  - Train AUC: `0.6909618882275133`
  - Test AUC: `0.6774193548387096`
  - R_V-alone test AUC: `0.5852374551971327`
  - Bootstrap 95% CI (test AUC): `[0.6138969284912299, 0.7392476105137396]`
  - Top 3 standardized features:
    - `top1_ratio` (`-1.594`)
    - `output_rv` (`-1.196`)
    - `spectral_gap` (`+0.463`)
- Baseline-only:
  - Test AUC: `0.5941176470588235`

## Task 4 Results (Master Evidence)
Artifacts:
- `results/behavioral_nboost_summary.json`
- `industry_grade/2026-02-20/evidence/behavioral_bridge_master.json`

Master highlights:
- output_rv bootstrap CI95 for BT+ART minus Other mean: `[-0.10398079265019548, -0.05868104877120251]`
- temporal lag (output_rv(t) -> class(t+1), recursive pooled): `rho=-0.014225883669267246`, `p=0.7389922104641166` (not significant)
- c2 semantic anchor: `n=755`, `rho=-0.6519420205824219`, `p=1.4298236437045998e-92`
- verdict: `SUCCESS_CRITERIA_MET`

## Interpretation
- The main behavioral bridge claim is strengthened by scale-up:
  - Recursive BT+ART turns are now `254` (well above `200` target).
  - Between-class separation on `output_rv` remains strong and significant (`d=-0.567`, very low p).
  - Held-out predictive performance is above the requested threshold (`test AUC=0.677 > 0.65`).
- Temporal precedence (lag-1) did not show a significant `output_rv` effect in pooled recursive data.
- Per-token 25+25 batch did not produce a significant recursive-vs-baseline separation in mean generation R_V.

## Primary Artifact Index
- `results/behavioral_nboost_summary.json`
- `industry_grade/2026-02-20/evidence/behavioral_bridge_master.json`
- `results/within_session_bridge/within_session_bridge_20260221_000441.json`
- `results/bridge_battery/bridge_battery_20260221_000443.json`
- `results/classifier_evaluation/classifier_eval_20260221_000622.json`
- `results/batch_per_token_rv/batch_per_token_rv_20260220_161603.json`
- `results/batch_per_token_rv/rv_trajectory_plot.png`
