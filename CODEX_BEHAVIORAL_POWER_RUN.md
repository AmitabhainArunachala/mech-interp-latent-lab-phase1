# CODEX POWER RUN: Behavioral Bridge N-Boost + Per-Token R_V at Scale

## Context

You're working in `~/mech-interp-latent-lab-phase1/` on a mechanistic interpretability project studying R_V (geometric contraction in Value matrix column space during recursive self-reference in transformers).

**Model**: Mistral-7B-Instruct-v0.3, Layer 5 (early) / Layer 27 (late), window=16.

**What exists already:**
- `scripts/sustained_gnani_v3.py` — runs 50-turn sessions (recursive or baseline), generates text, measures R_V + 8 spectral metrics + BT+ART classification on each turn's output via prefill
- `scripts/within_session_bridge.py` — correlates per-turn R_V with BT+ART classification across sessions
- `scripts/bridge_battery.py` — temporal lag, state transitions, logistic regression, C2 validation
- `scripts/batch_per_token_rv.py` — per-token R_V tracking during generation (25 recursive + 25 baseline prompts)
- `results/sustained_gnani_v3_fixed/` — current data: 4 recursive sessions + 3 baseline sessions (50 turns each)
- `src/` — full library (core/hooks, core/hf_accessors, metrics/rv, metrics/extended, metrics/logit_lens)

**The problem:** Our geometry→behavior story rests on only 80 BT+ART turns from 4 recursive sessions. A reviewer will ask "is this a 4-session fluke?" We need 3x the behavioral N and batch per-token R_V with statistical tests.

---

## TASK 1: Generate 8 More Recursive Sessions + 5 More Baseline Sessions (HIGHEST PRIORITY)

Use `scripts/sustained_gnani_v3.py` as the template. Run it to produce:
- **8 new recursive sessions** (50 turns each) with different random seeds
- **5 new baseline sessions** (50 turns each) with different random seeds

Each session should write its output JSON to `results/sustained_gnani_v3_fixed/` following the same naming convention: `recursive_YYYYMMDD_HHMMSS.json` and `baseline_YYYYMMDD_HHMMSS.json`.

Look at how `sustained_gnani_v3.py` works — it loads the model once, runs 50 turns of generation with metrics, and saves. You may need to run it multiple times with different seeds, or modify it to accept a `--seed` argument and a `--n-sessions` argument so you can batch them.

Key parameters to match existing runs:
- max_new_tokens=128, temperature=0.7
- Classification via `classify_output()` already in the script
- Metrics via `compute_prefill_metrics()` on the output text
- early_layer=5, late_layer=27, window=16

**After generating sessions**, rerun the analysis scripts:

```bash
cd ~/mech-interp-latent-lab-phase1
python scripts/within_session_bridge.py
python scripts/bridge_battery.py
```

Save a combined summary to `results/behavioral_nboost_summary.json` with:
- Total sessions (recursive/baseline)
- Total turns per condition
- Total BT+ART per condition
- Recomputed within-session bridge stats (output_rv d, p, logistic AUC)
- Compare old N vs new N effect sizes

**Target**: Get from 80 BT+ART → 200+ BT+ART turns, from 4→12 recursive sessions, from 3→8 baseline sessions.

---

## TASK 2: Batch Per-Token R_V with Behavioral Tagging (N=25+25)

Use `scripts/batch_per_token_rv.py` as the base. It already has 25 recursive + 25 baseline prompts and the per-token tracking loop.

**Modifications needed:**
1. Increase `max_new_tokens` from 64 to 256 (current data had 88.9% truncation at 64)
2. Keep temperature=0.7
3. For each generated text, also run `classify_output()` from sustained_gnani_v3.py to tag the generation as BT+ART or not
4. For each prompt, record:
   - prompt_rv (canonical prefill R_V)
   - per-token R_V trajectory (list of floats, one per generated token)
   - mean_generation_rv (mean of per-token R_V values)
   - final_text (decoded generation)
   - classification (BT+ART tag)
   - n_tokens_generated

**Statistical tests to compute and save:**
- Mann-Whitney U on mean_generation_rv: recursive vs baseline
- Cohen's d on mean_generation_rv: recursive vs baseline
- Point-biserial correlation: mean_generation_rv vs BT+ART classification
- Within recursive only: does lower mean_generation_rv → higher BT+ART rate?
- Mean R_V trajectory plot (recursive mean+SEM vs baseline mean+SEM across token positions)

Save results to `results/batch_per_token_rv/batch_per_token_rv_TIMESTAMP.json` and plot to `results/batch_per_token_rv/rv_trajectory_plot.png`.

---

## TASK 3: Cross-Validated Classifier with Held-Out Test (on expanded data)

After Task 1 is complete, build a proper classifier evaluation:

1. Load ALL sessions from `results/sustained_gnani_v3_fixed/`
2. Extract per-turn features: output_rv, eff_rank, top1_ratio, spectral_gap, cosine, attn_entropy, perplexity, rs_rv
3. Target: class_bin (BT+ART = 1, else = 0)
4. Split: stratified 70/30 train/test (NOT cross-validated on full set — hold out 30% completely)
5. Fit LogisticRegression on train, evaluate on test
6. Report:
   - Train AUC, Test AUC (the held-out AUC is the headline number)
   - Feature importances (standardized coefficients)
   - Confusion matrix on test set
   - R_V alone AUC vs multi-metric AUC (both on test set)
   - Bootstrap 95% CI on test AUC (1000 resamples)

Also run the same classifier on **baseline sessions only** to show that the R_V→behavior link doesn't hold in baseline (expected: AUC near 0.5).

Save to `results/classifier_evaluation/classifier_eval_TIMESTAMP.json`.

---

## TASK 4: Consolidated Evidence Summary

After Tasks 1-3 are complete, create a single master evidence file:

`industry_grade/2026-02-20/evidence/behavioral_bridge_master.json`

Structure:
```json
{
  "within_session_bridge": {
    "n_recursive_sessions": ...,
    "n_baseline_sessions": ...,
    "n_recursive_turns": ...,
    "n_baseline_turns": ...,
    "n_bt_art_recursive": ...,
    "n_bt_art_baseline": ...,
    "output_rv_cohens_d": ...,
    "output_rv_p_value": ...,
    "output_rv_ci95": [...],
    "temporal_lag_rho": ...,
    "temporal_lag_p": ...
  },
  "logistic_classifier": {
    "train_n": ...,
    "test_n": ...,
    "rv_alone_test_auc": ...,
    "multi_metric_test_auc": ...,
    "multi_metric_auc_ci95": [...],
    "baseline_only_auc": ...,
    "top_3_features": [...]
  },
  "per_token_rv": {
    "n_recursive": ...,
    "n_baseline": ...,
    "max_new_tokens": ...,
    "mean_gen_rv_recursive": ...,
    "mean_gen_rv_baseline": ...,
    "cohens_d": ...,
    "p_value": ...,
    "biserial_r_with_bt_art": ...,
    "biserial_p": ...
  },
  "c2_semantic": {
    "n_total": 755,
    "rho": -0.652,
    "p": 1.4e-92
  },
  "verdict": "..."
}
```

---

## EXECUTION ORDER

1. **Task 1 first** (most GPU time — 13 sessions × 50 turns × generation + metrics). This is the bottleneck.
2. **Task 2 second** (25+25 prompts × 256 tokens each — separate model load is fine).
3. **Task 3 third** (CPU only, depends on Task 1 output).
4. **Task 4 last** (aggregation, depends on all above).

## IMPORTANT NOTES

- Do NOT stop seed 789 or any existing seed bridge work if it's running — just let it finish in the background.
- All scripts should be run from the repo root: `cd ~/mech-interp-latent-lab-phase1`
- Device: use `--device cuda` if GPU available, `--device mps` for Apple Silicon, `--device cpu` as last resort (will be slow).
- If you encounter OOM, reduce batch size or run sessions sequentially (one model load, loop over seeds).
- Print progress every turn so we can see it's alive.
- Save intermediate results after each session completes (don't lose work if a later session crashes).
- If sustained_gnani_v3.py doesn't accept seeds/n-sessions args, add them — minimal changes, keep the core logic identical.

## SUCCESS CRITERIA

When done, running this should show the improvement:
```bash
python scripts/within_session_bridge.py
```
Expected output: 12+ recursive sessions, 200+ BT+ART turns, d < -0.5 maintained, AUC > 0.65 on held-out test set.
