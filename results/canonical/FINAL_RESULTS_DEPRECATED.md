# final_results.json is deprecated

`results/canonical/final_results.json` contains single-layer PR values mislabeled as R_V.
Do not use it for claims or summaries.

Next step:
- Recompute with `R_V = PR_late / PR_early` using `src/metrics/rv.py`.
