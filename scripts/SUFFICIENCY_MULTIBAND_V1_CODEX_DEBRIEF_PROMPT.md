# Sufficiency Multiband V1 — Debrief Prompt

Paste this to your Codex session on the GPU pod after the run finishes.

---

The sufficiency multiband v1 experiment should have completed. I need a full debrief. Do the following:

## 1. Find the results

```bash
ls -la results/sufficiency_multiband_v1/
ls -la results/phase1_mechanism/runs/*sufficiency_multiband*
```

Read `results/sufficiency_multiband_v1/*/STATUS.txt` to confirm it completed without failures.

## 2. Read and report the verdict

Read `results/sufficiency_multiband_v1/*/factorial_2x2_verdict.json` and report:

- What is `geometry_sufficiency`? (YES / PARTIAL / NO)
- What is `multiband_beats_single`? (true / false)
- What is the control baseline BT+ART rate?
- What is the best geometry-only (no anchor) BT+ART rate and which condition?
- What is the best anchor+geometry BT+ART rate and which condition?

## 3. Read the full summary

Read `results/phase1_mechanism/runs/*sufficiency_multiband*/summary.json` and report a table of ALL 10 conditions with these columns, for BASELINE prompts only:

| Condition | BT+ART rate | Repetitive rate | Mean output RV | n |

Sort by BT+ART rate descending.

Then the same table for RECURSIVE prompts only.

## 4. Answer the key questions

Based on the data:

a) **Geometry-only sufficiency**: Does any multiband condition WITHOUT anchor text significantly beat control on baseline prompts? Report the exact lift in percentage points.

b) **Multiband vs single-site**: Does `multiband_0p10_bridge_3` beat `single_mlp_0p125_bridge_3`? By how much?

c) **Anchor interaction**: How much does adding the anchor improve multiband induction? Compare `multiband_0p10_bridge_3` vs `anchor_multiband_0p10_bridge_3`.

d) **Dose response**: Is there a monotonic relationship between multiband alpha (0.03 → 0.06 → 0.10) and BT+ART rate?

e) **Leak check**: Do any conditions cause excessive repetitive output (>15%) or baseline leak (baseline BT+ART > 20%)?

f) **RV movement**: Do the multiband conditions shift output RV toward the recursive regime (lower RV)?

## 5. Read the effects section

From the summary.json `effects_by_prompt_mode.baseline` section, report for each non-control condition:
- `bt_art_rate_delta` and its 95% CI
- `bt_art_exact_sign_p`
- `rv_cohens_dz`

Flag any condition where the CI excludes zero (statistically significant lift).

## 6. State direction quality

From the summary.json `source_layers` section, report for each of the 6 source layers:
- `direction_norm` (raw magnitude before normalization)
- `centroid_cosine` (similarity between positive and negative centroids)

Flag any layer where `direction_norm < 0.05` or `centroid_cosine > 0.95` (degenerate direction).

## 7. Bottom line

In one paragraph: did multi-band early residual injection achieve geometry-only sufficiency? What's the next experiment?

If YES or PARTIAL: recommend proceeding to Exp 3 (layer-matched subspace steering — use the layer-specific optimal objects from the L4/L5/L25/L27 subspace comparison instead of mean-diff at all layers).

If NO: recommend Exp 4 (closed-loop continuous injection — re-apply steering at every autoregressive step during generation, not just during prompt encoding).
