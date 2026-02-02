# Mistral-7B Causal Circuit Progress Writeup
**Date/Time:** 2026-01-16 18:52:45 +0530  
**Model:** `mistralai/Mistral-7B-v0.1`  
**Prompt Bank Version:** `75e7c1b8dcebc24e`  

## Scope of Work (Today)
- Prompt-pass necessity sweep for L0–L3.
- L27 causal validation (v-proj patching controls).
- KV head ablation validation at L27.
- MLP sufficiency tests (L0, L3).
- Combined MLP sufficiency tests (L0+L1, L0+L1+L3).
- Random direction control at L3 (steering specificity).
- Schema compliance fixes for canonical runs.

## Key Findings (Raw Numbers + Pointers)

### 1) Prompt-pass Necessity (Same prompt, no generation)
**L0 (necessary; very strong):**  
- `rv_baseline_mean`: 0.5066  
- `rv_ablated_mean`: 1.6859  
- `rv_delta_mean`: +1.1793  
- `rv_pvalue`: 1.31e-64  
- Verdict: L0 MLP is necessary (PR_early drives effect).  
- Summary: `results/phase1_mechanism/runs/20260116_113943_mlp_ablation_necessity_prompt_pass_l0_necessity_prompt_pass/summary.json`  
- CSV: `results/phase1_mechanism/runs/20260116_113943_mlp_ablation_necessity_prompt_pass_l0_necessity_prompt_pass/mlp_ablation_necessity_prompt_pass.csv`

**L1 (necessary; strong):**  
- `rv_baseline_mean`: 0.5066  
- `rv_ablated_mean`: 1.3764  
- `rv_delta_mean`: +0.8698  
- `rv_pvalue`: 4.17e-54  
- Verdict: L1 MLP is necessary (PR_early drives effect).  
- Summary: `results/phase1_mechanism/runs/20260116_114327_mlp_ablation_necessity_prompt_pass_l1_necessity_prompt_pass/summary.json`  
- CSV: `results/phase1_mechanism/runs/20260116_114327_mlp_ablation_necessity_prompt_pass_l1_necessity_prompt_pass/mlp_ablation_necessity_prompt_pass.csv`

**L2 (no effect):**  
- `rv_delta_mean`: +0.0023  
- `rv_pvalue`: 0.7118  
- Verdict: no significant effect.  
- Summary: `results/phase1_mechanism/runs/20260116_114427_mlp_ablation_necessity_prompt_pass_l2_necessity_prompt_pass/summary.json`  
- CSV: `results/phase1_mechanism/runs/20260116_114427_mlp_ablation_necessity_prompt_pass_l2_necessity_prompt_pass/mlp_ablation_necessity_prompt_pass.csv`

**L3 (necessary; moderate):**  
- `rv_delta_mean`: +0.1880  
- `rv_pvalue`: 7.70e-21  
- Verdict: L3 MLP is necessary (PR_early drives effect).  
- Summary: `results/phase1_mechanism/runs/20260116_114536_mlp_ablation_necessity_prompt_pass_l3_necessity_prompt_pass/summary.json`  
- CSV: `results/phase1_mechanism/runs/20260116_114536_mlp_ablation_necessity_prompt_pass_l3_necessity_prompt_pass/mlp_ablation_necessity_prompt_pass.csv`

### 2) L27 Causal Validation (Patch + controls)
**Main patch effect (target layer):**  
- `rv_baseline_mean`: 0.6939  
- `rv_recursive_mean`: 0.5081  
- `rv_delta_mean`: -0.1672  
- `rv_p_value`: 2.24e-19  
- Transfer estimate: 89.98%  
- Summary: `results/phase1_cross_architecture/runs/20260116_115423_rv_l27_causal_validation_default/summary.json`  
- CSV (pairs): `results/phase1_cross_architecture/runs/20260116_115423_rv_l27_causal_validation_default/rv_l27_causal_validation_pairs.csv`

### 3) L27 KV Head Ablation (H2 vs control head, target vs control layer)
**All pass checks true (target head > control head; L27 > L21):**  
- `rv_recursive_mean`: 0.4999  
- `rv_baseline_mean`: 0.7063  
- `rv_delta_mean`: +0.0777  
- `rv_p_value`: 6.11e-32  
- Summary: `results/phase1_mechanism/runs/20260116_120016_head_ablation_validation_mistral_l27_kv_head_validation/summary.json`  
- CSV: `results/phase1_mechanism/runs/20260116_120016_head_ablation_validation_mistral_l27_kv_head_validation/head_ablation_results.csv`

### 4) MLP Sufficiency (Single layer)
**L0 sufficiency (fails):**  
- `rv_restoration_pct_mean`: 21.57%  
- `rv_pvalue`: 0.0972  
- Verdict: NOT sufficient  
- Summary: `results/phase1_mechanism/runs/20260116_121226_mlp_sufficiency_test_l0_sufficiency/summary.json`  
- CSV: `results/phase1_mechanism/runs/20260116_121226_mlp_sufficiency_test_l0_sufficiency/mlp_sufficiency_test.csv`

**L3 sufficiency (fails):**  
- `rv_restoration_pct_mean`: 9.03%  
- `rv_pvalue`: 0.2931  
- Verdict: NOT sufficient  
- Summary: `results/phase1_mechanism/runs/20260116_131551_mlp_sufficiency_test_l3_sufficiency/summary.json`  
- CSV: `results/phase1_mechanism/runs/20260116_131551_mlp_sufficiency_test_l3_sufficiency/mlp_sufficiency_test.csv`

### 5) Combined MLP Sufficiency (Multi-layer)
**L0+L1 (fails strongly):**  
- `rv_patched_mean`: 1.1121  
- `rv_restoration_pct_mean`: -342.87%  
- Verdict: NOT sufficient  
- Summary: `results/phase1_mechanism/runs/20260116_124006_combined_mlp_sufficiency_test_l0_l1_combined_sufficiency/summary.json`  
- CSV: `results/phase1_mechanism/runs/20260116_124006_combined_mlp_sufficiency_test_l0_l1_combined_sufficiency/combined_mlp_sufficiency_test.csv`

**L0+L1+L3 (fails strongly):**  
- `rv_patched_mean`: 1.4138  
- `rv_restoration_pct_mean`: -547.64%  
- Verdict: NOT sufficient  
- Summary: `results/phase1_mechanism/runs/20260116_130033_combined_mlp_sufficiency_test/summary.json`  
- CSV: `results/phase1_mechanism/runs/20260116_130033_combined_mlp_sufficiency_test/combined_mlp_sufficiency_test.csv`

### 6) Random Direction Control (Steering specificity)
**True steering not better than random (artifact):**  
- `true_steering_rv_delta`: 2.4322  
- `random_avg_rv_delta`: 2.6337  
- `rv_ratio`: 0.92  
- `rv_ttest.p`: 0.1379  
- Verdict: ARTIFACT  
- Summary: `results/phase1_mechanism/runs/20260116_124427_random_direction_control_l3_random_control/summary.json`  
- CSV: `results/phase1_mechanism/runs/20260116_124427_random_direction_control_l3_random_control/random_direction_control.csv`  
- Comparison table: `results/phase1_mechanism/runs/20260116_124427_random_direction_control_l3_random_control/comparison_table.csv`

## Interpretation / Thoughts
- **Prompt-pass necessity is strong at L0/L1 and moderate at L3**, with L2 null. This suggests early-layer MLPs are causal for contraction, but the effect is not monolithic.
- **Sufficiency is consistently negative** (L0, L3, L0+L1, L0+L1+L3). This means the contraction mechanism is **not reproduced by MLP patching alone**, even when stacking layers.
- **Random direction control shows no specificity** (true steering ≈ random), which is a serious warning sign: the steering effects appear to be a generic perturbation effect rather than a direction-specific circuit.
- **L27 causal validation + KV head ablation are strong** and pass controls; this supports a real late-layer handle, but the source-to-readout pathway is still unresolved.

## Status vs “Full Causal Circuit”
We now have **necessity without sufficiency** at the source layers and **strong late-layer causal evidence**, but the **steering direction is not specific**. This blocks a “complete causal circuit” claim. The missing link is a **specific, sufficient intervention** that restores contraction in a controlled, non-generic way.

## Next Steps (if we resume)
1. Test **late-layer sufficiency** directly (e.g., V-proj or KV head patching on baseline prompts) to see if contraction can be restored with a specific late-layer intervention.  
2. If late-layer sufficiency is real, evaluate **mediation** (source → L27 head group → contraction) rather than relying on MLP-only sufficiency.  
3. Re-evaluate steering vector definition/normalization if specificity continues to fail in random controls.
