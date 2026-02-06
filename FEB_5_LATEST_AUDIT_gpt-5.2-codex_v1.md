# Signal Quality & Industry Standard Audit (V2.1)
**Repo:** `mech-interp-latent-lab-phase1`  
**Scope:** `results/canonical/`, `results/phase1_cross_architecture/`, `results/phase3_bridge/`, `results/discovery/` (priority‑2, high‑signal candidates)  
**Date:** 2026-02-05  

## Executive Summary
本監査スコープ内で **契約違反（contract violation）を含むファイルは27件**。最大の問題は「必須アーティファクト欠落（CSV／prompt_bank_version／hardware_info）」と「R_V 定義違反（単層PRの誤ラベル／再帰が >1.0）」です。  
KEEP基準（n≥50＋統計＋完全アーティファクト＋R_V妥当）を満たす結果は確認できず、**現状はほぼ全件が ARCHIVE_ONLY か RAMP_UP（再実行前提）**となります。

## KEEP_SIGNAL
```
File Path | n | Stats | Controls | Artifacts | R_V Correct? | Why High-Signal
----------|---|-------|----------|-----------|--------------|----------------
NONE | - | - | - | - | - | 必須アーティファクトと統計要件を満たす結果が見当たらない
```

## RAMP_UP
```
File Path | Current n | Target n | Missing | Config Changes | Priority
----------|-----------|----------|---------|---------------|----------
results/phase1_cross_architecture/runs/20260202_121604_rv_l27_causal_validation_mistral_7b/summary.json | 45 | 80 | CSV, hardware_info, correction | {"n_pairs": 80, "write_csv": true, "write_hardware_info": true, "p_value_correction": "holm"} | HIGH
results/phase1_cross_architecture/runs/20260202_125718_rv_l27_causal_validation_opt_6_7b/summary.json | 45 | 80 | CSV, hardware_info, correction | {"n_pairs": 80, "write_csv": true, "write_hardware_info": true, "p_value_correction": "holm"} | HIGH
results/phase1_cross_architecture/runs/20260202_130807_rv_l27_causal_validation_gpt2_xl/summary.json | 45 | 80 | CSV, hardware_info, correction | {"n_pairs": 80, "write_csv": true, "write_hardware_info": true, "p_value_correction": "holm"} | MEDIUM
results/phase1_cross_architecture/runs/20260202_120856_rv_l27_causal_validation_qwen2_7b/summary.json | 45 | 80 | CSV, hardware_info, R_V>1.0 | {"n_pairs": 80, "write_csv": true, "write_hardware_info": true, "verify_rv_ratio": true} | HIGH
results/phase3_bridge/gemma_2_9b/multi_token_correlation_v2/runs/20260124_163912_multi_token_bridge_gemma_2_9b_rv_behavioral_bridge_v2/summary.json | 117 | 160 | CSV, hardware_info, truncation | {"n_prompts_per_group": 60, "write_csv": true, "write_hardware_info": true, "max_new_tokens": 400} | HIGH
results/phase1_cross_architecture/runs/20260202_133252_multi_token_bridge_mistral_7b_bridge/summary.json | 120 | 160 | CSV, hardware_info, truncation | {"n_prompts_per_group": 30, "write_csv": true, "write_hardware_info": true, "max_new_tokens": 400} | HIGH
results/discovery/path_patching/20251213_080454_path_patching_mechanism_full_early_layer_sweep_full_controls_base/summary.json | n_rows=22400 | 80 per layer | CSV in run dir, hardware_info, stats | {"write_csv": true, "write_hardware_info": true, "include_stats": true} | MEDIUM
results/canonical/c2_measurement_suite/20260111_130123_c2_rv_measurement/summary.json | 50 | 100 | config.json, CSV, hardware_info, stats | {"n_prompts": 100, "write_csv": true, "write_hardware_info": true, "include_stats": true} | MEDIUM
```

## ARCHIVE_ONLY
```
File Path | Reason | Evidence | Archive Location
----------|--------|----------|----------------
results/canonical/final_results.json | Contract violation | baseline_rv=5.279, recursive_rv=6.733 (単層PR値) | results/archive/contract_violations/
results/canonical/rv_l27_causal_validation/20251216_061127_rv_l27_causal_validation/summary.json | Incomplete | CSV・prompt_bank_version・hardware_info欠落 | results/archive/incomplete/
results/canonical/rv_l27_causal_validation/20251216_060955_rv_l27_causal_validation/summary.json | Incomplete | CSV・prompt_bank_version・hardware_info欠落 | results/archive/incomplete/
results/canonical/confound_validation/20251216_060911_confound_validation/summary.json | Incomplete | CSV・prompt_bank_version・hardware_info欠落、n_total=37 | results/archive/incomplete/
results/canonical/c2_measurement_suite/20260111_125011_c2_rv_measurement/summary.json | Incomplete | config.json・CSV・prompt_bank_version・hardware_info欠落 | results/archive/incomplete/
results/canonical/c2_measurement_suite/20260111_125410_c2_rv_measurement/summary.json | Incomplete | config.json・CSV・prompt_bank_version・hardware_info欠落 | results/archive/incomplete/
results/canonical/c2_measurement_suite/20260111_123508_c2_rv_measurement/summary.json | Incomplete | config.json・CSV・prompt_bank_version・hardware_info欠落 | results/archive/incomplete/
results/canonical/c2_measurement_suite/20260111_130123_c2_rv_measurement/summary.json | Incomplete | config.json・CSV・prompt_bank_version・hardware_info欠落 | results/archive/incomplete/
results/canonical/c2_measurement_suite/20260111_140002_c2_ablation_no_cascade/summary.json | Incomplete | config.json・CSV・prompt_bank_version・hardware_info欠落 | results/archive/incomplete/
results/canonical/c2_measurement_suite/20260111_140229_c2_ablation_no_steering/summary.json | Incomplete | config.json・CSV・prompt_bank_version・hardware_info欠落 | results/archive/incomplete/
results/canonical/c2_measurement_suite/20260111_140449_c2_ablation_no_kv/summary.json | Incomplete | config.json・CSV・prompt_bank_version・hardware_info欠落 | results/archive/incomplete/
results/canonical/multi_token_bridge/summary.json | Incomplete | CSV・hardware_info欠落 | results/archive/incomplete/
results/phase1_cross_architecture/runs/20260202_121604_rv_l27_causal_validation_mistral_7b/summary.json | Incomplete | CSV・hardware_info欠落 | results/archive/incomplete/
results/phase1_cross_architecture/runs/20260202_115958_rv_l27_causal_validation_pythia_1_4b/summary.json | Incomplete | CSV・hardware_info欠落 | results/archive/incomplete/
results/phase1_cross_architecture/runs/20260202_120856_rv_l27_causal_validation_qwen2_7b/summary.json | Contract violation | recursive R_V=1.157 (>1.0) + CSV・hardware_info欠落 | results/archive/contract_violations/
results/phase1_cross_architecture/runs/20260202_125718_rv_l27_causal_validation_opt_6_7b/summary.json | Incomplete | CSV・hardware_info欠落 | results/archive/incomplete/
results/phase1_cross_architecture/runs/20260202_130807_rv_l27_causal_validation_gpt2_xl/summary.json | Incomplete | CSV・hardware_info欠落 | results/archive/incomplete/
results/phase1_cross_architecture/runs/20260116_115423_rv_l27_causal_validation_default/summary.json | Incomplete | CSV・hardware_info欠落 | results/archive/incomplete/
results/phase1_cross_architecture/runs/20260202_133252_multi_token_bridge_mistral_7b_bridge/summary.json | Incomplete | CSV・hardware_info欠落 | results/archive/incomplete/
results/phase3_bridge/gemma_2_9b/multi_token_correlation_v2/runs/20260124_163912_multi_token_bridge_gemma_2_9b_rv_behavioral_bridge_v2/summary.json | Incomplete | CSV・hardware_info欠落 | results/archive/incomplete/
results/phase3_bridge/gemma_2_9b/multi_token_correlation_v3/runs/20260124_170932_multi_token_bridge_gemma_2_9b_rv_behavioral_bridge_v3_t0_long/summary.json | Incomplete | CSV・hardware_info欠落 | results/archive/incomplete/
results/discovery/path_patching/20251213_080454_path_patching_mechanism_full_early_layer_sweep_full_controls_base/summary.json | Incomplete | CSVは別パスのみ、hardware_info欠落 | results/archive/incomplete/
results/discovery/path_patching/20251213_073754_path_patching_mechanism_full_early_layer_sweep_base/summary.json | Incomplete | CSVは別パスのみ、hardware_info欠落 | results/archive/incomplete/
results/discovery/path_patching/20251213_055827_path_patching_mechanism_default/summary.json | Contract violation | rv系指標で>1.5が頻出（単層PR疑い）+ artifacts欠落 | results/archive/contract_violations/
results/discovery/behavioral_grounding/20251216_125333_behavior_strict/summary.json | Incomplete | CSVは別パスのみ、hardware_info欠落、n_pairs=20 | results/archive/incomplete/
results/discovery/behavioral_grounding/20251217_122855_behavior_strict/summary.json | Incomplete | CSVは別パスのみ、hardware_info欠落、n_pairs=20 | results/archive/incomplete/
results/discovery/phase0_validation/20251213_052612_phase0_metric_targets_default/summary.json | Incomplete | CSVは別パスのみ、hardware_info欠落、n_rows=30 | results/archive/incomplete/
```

## Top 5 ROI Experiments
```
Rank | Experiment | Current State | Gap to Bridge | Config Path | Expected Outcome | Effort | Priority
-----|------------|---------------|---------------|-------------|------------------|--------|----------
1 | Multi-token R_V→behavior (Mistral) | n=120, 高トランケーション | per-token R_V + 低トランケーション + CSV/hardware | results/phase1_cross_architecture/runs/20260202_133252_multi_token_bridge_mistral_7b_bridge/config.json | 橋渡し相関の明確化 | 2-3 days | CRITICAL
2 | Multi-token R_V→behavior (Gemma-2-9B) | n=117, t0のみ | n≥160 + 温度条件 + CSV/hardware | results/phase3_bridge/gemma_2_9b/multi_token_correlation_v3/runs/20260124_170932_multi_token_bridge_gemma_2_9b_rv_behavioral_bridge_v3_t0_long/config.json | モデル間再現性 | 3-4 days | HIGH
3 | Cross-arch causal validation (Qwen2.5) | n=45, R_V>1.0 | n≥80 + R_V検証 + CSV/hardware | results/phase1_cross_architecture/runs/20260202_120856_rv_l27_causal_validation_qwen2_7b/config.json | 契約準拠の再現 | 2 days | HIGH
4 | C2 measurement high‑N | n=50, stats欠落 | n≥100 + 統計/CSV/hardware | results/canonical/c2_measurement_suite/20260111_130123_c2_rv_measurement/summary.json | C2の定量確証 | 2 days | HIGH
5 | Path patching full controls | n_rows大だがstatsなし | 統計＋run dir CSV/hardware | results/discovery/path_patching/20251213_080454_path_patching_mechanism_full_early_layer_sweep_full_controls_base/summary.json | 層依存の因果地図 | 3-4 days | MEDIUM
```

## Claims vs Data Audit
```
Claim Location | Claim | Data Location | Verification | Status | Action Required
---------------|-------|---------------|--------------|--------|-----------------
results/canonical/final_results.json | "baseline_rv=5.279" | results/canonical/final_results.json | ❌ PR値（>1.5）でR_V定義違反 | INVALID | R_VをPR_late/PR_earlyで再計算
results/phase1_cross_architecture/runs/20260202_120856_rv_l27_causal_validation_qwen2_7b/summary.json | "rv_recursive_mean=1.157" | 同左 | ❌ 再帰R_V>1.0 | INVALID | 測定実装/データ再検証
results/canonical/c2_measurement_suite/20260111_130123_c2_rv_measurement/summary.json | "statistics": {} | 同左 | ❌ 必須統計欠落 | INVALID | Cohen's d / p / 95% CI を追加
results/phase3_bridge/gemma_2_9b/.../summary.json | h1/h3相関の有意性 | 同左 | ⚠️ 多重比較補正不明 | UNCERTAIN | 補正方法の明記
results/canonical/confound_validation/20251216_060911_confound_validation/summary.json | p値のみ | 同左 | ❌ Cohen's d / 95% CI欠落 | INVALID | 効果量とCIを追記
```

## Critical Gaps Summary
- **必須アーティファクト欠落**: CSV と `hardware_info.json` が全スコープで欠落（再現性失格）。  
- **統計要件不足**: Cohen's d / 95% CI / 補正p値が欠落。  
- **R_V 契約違反**: 単層PR値の誤ラベル（`results/canonical/final_results.json`）と再帰R_V>1.0（Qwen2.5）。  
- **サンプル不足**: n<50 が多く、publication基準に未達。  
- **高トランケーション**: multi-token bridge の有効生成が少なく、相関評価が不安定。  

## Recommendations
- **Immediate (0‑3日)**: 全パイプラインで `hardware_info.json` と per‑sample CSV の出力を強制、R_V定義検証を追加。  
- **Short‑term (1‑2週)**: 主要実験を n≥80 で再実行し、Cohen's d / 95% CI / 補正p値を統一記録。  
- **Long‑term (3‑6週)**: multi-token bridge のトランケーション対策（max_new_tokens拡大、EOS到達率改善）を実装して再評価。  

## Contract Violations Summary
- `results/canonical/final_results.json`: 単層PR値（>1.5）をR_Vとして記載。  
- `results/phase1_cross_architecture/runs/20260202_120856_rv_l27_causal_validation_qwen2_7b/summary.json`: 再帰R_V>1.0。  
- `results/discovery/path_patching/20251213_055827_path_patching_mechanism_default/summary.json`: rv系指標で>1.5が頻出（単層PR疑い）。  
- **必須アーティファクト欠落（契約違反）**:
  - `results/canonical/rv_l27_causal_validation/20251216_061127_rv_l27_causal_validation/summary.json`
  - `results/canonical/rv_l27_causal_validation/20251216_060955_rv_l27_causal_validation/summary.json`
  - `results/canonical/confound_validation/20251216_060911_confound_validation/summary.json`
  - `results/canonical/c2_measurement_suite/20260111_125011_c2_rv_measurement/summary.json`
  - `results/canonical/c2_measurement_suite/20260111_125410_c2_rv_measurement/summary.json`
  - `results/canonical/c2_measurement_suite/20260111_123508_c2_rv_measurement/summary.json`
  - `results/canonical/c2_measurement_suite/20260111_130123_c2_rv_measurement/summary.json`
  - `results/canonical/c2_measurement_suite/20260111_140002_c2_ablation_no_cascade/summary.json`
  - `results/canonical/c2_measurement_suite/20260111_140229_c2_ablation_no_steering/summary.json`
  - `results/canonical/c2_measurement_suite/20260111_140449_c2_ablation_no_kv/summary.json`
  - `results/canonical/multi_token_bridge/summary.json`
  - `results/phase1_cross_architecture/runs/20260202_121604_rv_l27_causal_validation_mistral_7b/summary.json`
  - `results/phase1_cross_architecture/runs/20260202_115958_rv_l27_causal_validation_pythia_1_4b/summary.json`
  - `results/phase1_cross_architecture/runs/20260202_125718_rv_l27_causal_validation_opt_6_7b/summary.json`
  - `results/phase1_cross_architecture/runs/20260202_130807_rv_l27_causal_validation_gpt2_xl/summary.json`
  - `results/phase1_cross_architecture/runs/20260116_115423_rv_l27_causal_validation_default/summary.json`
  - `results/phase1_cross_architecture/runs/20260202_133252_multi_token_bridge_mistral_7b_bridge/summary.json`
  - `results/phase3_bridge/gemma_2_9b/multi_token_correlation_v2/runs/20260124_163912_multi_token_bridge_gemma_2_9b_rv_behavioral_bridge_v2/summary.json`
  - `results/phase3_bridge/gemma_2_9b/multi_token_correlation_v3/runs/20260124_170932_multi_token_bridge_gemma_2_9b_rv_behavioral_bridge_v3_t0_long/summary.json`
  - `results/discovery/path_patching/20251213_080454_path_patching_mechanism_full_early_layer_sweep_full_controls_base/summary.json`
  - `results/discovery/path_patching/20251213_073754_path_patching_mechanism_full_early_layer_sweep_base/summary.json`
  - `results/discovery/behavioral_grounding/20251216_125333_behavior_strict/summary.json`
  - `results/discovery/behavioral_grounding/20251217_122855_behavior_strict/summary.json`
  - `results/discovery/phase0_validation/20251213_052612_phase0_metric_targets_default/summary.json`
