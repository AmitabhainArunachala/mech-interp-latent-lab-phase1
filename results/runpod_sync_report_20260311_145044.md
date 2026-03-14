# RunPod Results Sync Report

**Generated**: 2026-03-11 05:50:44 UTC
**Results directory**: `/Users/dhyana/mech-interp-latent-lab-phase1/results`
**Paper reference**: `R_V_PAPER/paper_colm2026_v005.tex`

## Executive Summary

| Metric | Value |
|--------|-------|
| Total claims checked | 7 |
| Claims matching paper | 1 |
| Claims diverging from paper | 6 |
| Match rate | 14% |

---

## 1. Full Head Sweep (E2.2)

Found 3 result file(s):
  - `full_head_sweep_20260302_074757.json`
  - `full_head_sweep_20260310_145353.json`
  - `full_head_sweep_20260310_151508.json`

Analyzing latest: `full_head_sweep_20260310_151508.json`
Model: mistralai/Mistral-7B-Instruct-v0.2
Prompt bank: 2ac959a313614329
Prompt subset: mistral_hardening_v1
Layers: 32, Heads/layer: 32, Total heads: 1024
Prompts: 20 recursive, 20 baseline

Significant heads (p < 0.05, uncorrected):
  Entropy metric:  630/1024 (61.5%)
  OV rank metric:  165/1024 (16.1%)
  Either metric:   691/1024 (67.5%)

Top 5 heads by |entropy d|:
      Head  d_entropy    p_entropy     d_rank
  --------------------------------------------
   L22.H21     -7.226     0.000000        N/A
   L24.H09     -4.965     0.000000        N/A
   L23.H05     -4.682     0.000000     -0.979
   L27.H06     -4.384     0.000000     -1.763
   L19.H13      4.300     0.000000        N/A

Top 5 heads by |OV rank d|:
      Head     d_rank       p_rank  d_entropy
  --------------------------------------------
   L25.H05     -4.666     0.000000     -1.316
   L29.H07     -4.162     0.000000     -0.276
   L29.H02     -3.825     0.000000     -0.116
   L18.H01     -3.497     0.000000     -1.528
   L08.H03      3.233     0.000000     -1.346

Top 5 layers by avg |entropy d|:
  L07: avg |d| = 1.526
  L26: avg |d| = 1.499
  L24: avg |d| = 1.463
  L30: avg |d| = 1.392
  L06: avg |d| = 1.340

---

## 2. Full Path Patching

Found 3 result file(s):
  - `path_patching_summary_20260227_080128.json`
  - `path_patching_summary_20260310_144503.json`
  - `path_patching_summary_20260310_151654.json`

Analyzing latest: `path_patching_summary_20260310_151654.json`
Model: mistralai/Mistral-7B-Instruct-v0.2
Target layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
Components: ['residual', 'v_proj', 'mlp']
N prompts: 20

Path patching heatmap (Cohen's d, break direction):
   Layer           mlp      residual        v_proj
  --------------------------------------------------
  L   0    +0.093       +1.885***    -0.587 * 
  L   1    -0.415       +1.944***    -0.558 * 
  L   2    -0.374       +1.787***    +0.047   
  L   3    -0.473       +1.810***    -0.735 * 
  L   4    +0.475       +1.755***    -0.415   
  L   5    +0.127       +1.684***    +0.767 * 
  L   6    +0.522 *     +0.580 *     -0.014   
  L   7    +0.333       +0.573 *     +0.132   
  L   8    +0.348       +0.575 *     +0.108   
  L   9    +0.109       +0.557 *     -0.002   
  L  10    +0.106       +0.552 *     +0.079   
  L  11    +0.085       +0.544 *     -0.045   
  L  12    +0.119       +0.529 *     +0.003   
  L  13    -0.188       +0.528 *     -0.022   
  L  14    +0.049       +0.524 *     +0.159   
  L  15    +0.094       +0.521 *     +0.067   
  L  16    -0.183       +0.520 *     +0.019   
  L  17    +0.022       +0.520 *     -0.084   
  L  18    -0.034       +0.516 *     +0.249   
  L  19    -0.270       +0.514 *     +0.208   
  L  20    -0.193       +0.511 *     +0.118   
  L  21    -0.100       +0.510 *     +0.011   
  L  22    -0.198       +0.508 *     +0.132   
  L  23    +0.027       +0.508 *     +0.242   
  L  24    -0.079       +0.512 *     +0.093   
  L  25    -0.224       +0.509 *     +0.027   
  L  26    -0.286       +0.509 *     +0.004   
  L  27    +0.000       +0.505 *     +0.505 * 
  L  28    +0.000       +0.000       +0.000   
  L  29    +0.000       +0.000       +0.000   
  L  30    +0.000       +0.000       +0.000   
  L  31    +0.000       +0.000       +0.000   

Top 5 causal sites by |d|:
   Layer    Component          d   delta_rv
  ------------------------------------------
  L   1     residual     +1.944      0.139
  L   0     residual     +1.885      0.129
  L   3     residual     +1.810      0.153
  L   2     residual     +1.787      0.155
  L   4     residual     +1.755      0.156

V-proj max |d|: 0.767 at L5
Residual top |d|: 1.944 at L1
MLP top |d|: 0.522 at L6

---

## 3. Dual-Layer Bridge (L18 Residual + L27 V-proj)

Searched directories: results/persistent_patching_v3
Found 11 result file(s):
  - `persistent_patching_v3_dual_20260225_002604.json`
  - `persistent_patching_v3_dual_20260310_151341.json`
  - `persistent_patching_v3_dual_20260310_152013.json`
  - `persistent_patching_v3_dual_20260310_160920.json`
  - `persistent_patching_v3_dual_20260310_191243.json`
  - `persistent_patching_v3_dual_20260310_191355.json`
  - `persistent_patching_v3_dual_20260310_191519.json`
  - `persistent_patching_v3_dual_20260310_191654.json`
  - `persistent_patching_v3_dual_20260310_193713.json`
  - `persistent_patching_v3_dual_20260310_194619.json`
  - `persistent_patching_v3_dual_20260310_204100.json`

Analyzing best file (most sessions): `persistent_patching_v3_dual_20260310_204100.json`
Model: mistralai/Mistral-7B-Instruct-v0.2
Experiment: persistent_patching_v3_dual_layer
Sessions per condition: 10
Max turns per session: 30
Total turns per condition: 300
V-layer: 27, R-layer: 18

Condition summaries:
                     Condition  BT+ART rate   Mean R_V  N turns
  --------------------------------------------------------------
            A: recursive_clean        0.547      0.601      300
     B: recursive_dual_patched        0.000      0.753      300
             C: baseline_clean        0.020      0.713      300
      D: baseline_dual_patched        0.000      0.585      300

Break test (A vs B):
  A (recursive clean) BT+ART rate: 0.547
  B (recursive patched) BT+ART rate: 0.000
  Odds ratio: N/A
  Turn-level p: 0.000000
  Session-level Cohen's d: 4.645

Induce test (C vs D):
  C (baseline clean) BT+ART rate: 0.020
  D (baseline patched) BT+ART rate: 0.000
  Odds ratio: N/A
  Session-level Cohen's d: 0.629

R_V session contrasts:
  break: mean_diff=-0.152, d=-5.406, perm_p=0.000016
  induce: mean_diff=0.128, d=3.818, perm_p=0.000016
  Note: dual-layer induce R_V shift is not directly comparable to the paper's KV-only geometry/behavior dissociation claim (d=0.11, NS), which comes from a different experiment family.

Sanity check (A vs C): OR=59.088, p=0.000000

---

## 4. Paper Claim Cross-Reference

| # | Claim | Paper Value | Data Value | Match |
|---|-------|-------------|------------|-------|
| 1 | Head sweep: significant heads | 606/1024 (59.2%) | 691/1024 (67.5%) | **NO** -- Counting heads significant on entropy OR rank metric at p<0.05 uncorrected. |
| 2 | Head sweep: top head location | L10H20 (|d|=3.9) | L22H21 (|d|=7.226) | **NO** |
| 3 | Path patching: V-proj max |d| | 0.22 | 0.767 | **NO** -- V-proj top site: L5 |
| 4 | Path patching: top residual causal site | L4 (d=1.96) | L1 (d=+1.944) | **NO** -- Paper claims L4 residual d=1.96 is top causal site. |
| 5 | Dual layer: BT+ART rate (recursive clean) | 56% | 54.7% | YES |
| 6 | Dual layer: BT+ART rate (recursive patched) | 3.7% | 0.0% | **NO** |
| 7 | Dual layer: session-level Cohen's d (break) | 3.29 | 4.64 | **NO** |

---

## 5. Action Items (Mismatches)

- **Head sweep: significant heads**: Paper says `606/1024 (59.2%)`, data shows `691/1024 (67.5%)`.
  - Note: Counting heads significant on entropy OR rank metric at p<0.05 uncorrected.
  - Action: Investigate whether this is a new run, different prompt set, or genuine discrepancy.
- **Head sweep: top head location**: Paper says `L10H20 (|d|=3.9)`, data shows `L22H21 (|d|=7.226)`.
  - Action: Investigate whether this is a new run, different prompt set, or genuine discrepancy.
- **Path patching: V-proj max |d|**: Paper says `0.22`, data shows `0.767`.
  - Note: V-proj top site: L5
  - Action: Investigate whether this is a new run, different prompt set, or genuine discrepancy.
- **Path patching: top residual causal site**: Paper says `L4 (d=1.96)`, data shows `L1 (d=+1.944)`.
  - Note: Paper claims L4 residual d=1.96 is top causal site.
  - Action: Investigate whether this is a new run, different prompt set, or genuine discrepancy.
- **Dual layer: BT+ART rate (recursive patched)**: Paper says `3.7%`, data shows `0.0%`.
  - Action: Investigate whether this is a new run, different prompt set, or genuine discrepancy.
- **Dual layer: session-level Cohen's d (break)**: Paper says `3.29`, data shows `4.64`.
  - Action: Investigate whether this is a new run, different prompt set, or genuine discrepancy.

---
*Report generated by `scripts/sync_runpod_results.py`*