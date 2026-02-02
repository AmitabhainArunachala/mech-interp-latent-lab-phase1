# Metrics Reference & Reporting Standard

**Single source of truth for R_V research metrics, reporting schemas, and reproducibility.**

---

## 1. The Metrics Stack

We classify metrics into **Core** (required for all claims) and **Extended** (for deeper mechanism analysis).

### Core Metrics (Required)
| Metric | Code | Purpose | Nullable? |
|--------|------|---------|-----------|
| **R_V** | `src/metrics/rv.py` | **Primary.** Geometric contraction (PR_late / PR_early). | No |
| **Logit Diff** | `src/metrics/logit_diff.py` | **Attribution.** Linear causal trace metric (Nanda-standard). | Yes* |
| **Activ. Norms** | `baseline_suite.py` | **Diagnostic.** Intervention strength check (early/late residual norms). | No |
| **Statistics** | `baseline_suite.py` | **Rigor.** Cohen's d, p-value, 95% CI for all above. | Partial |

**\*Logit Diff Nullability**: Geometry-only experiments (e.g., `rv_l27_causal_validation`, `mlp_sufficiency_test`, `head_ablation_validation`) measure R_V via patching/ablation WITHOUT generating text. These experiments may have `logit_diff_*` metrics as `null` because behavioral output is not measured. The `--strict` flag in `run.py` excludes geometry-only experiments from null-value failures. See `GEOMETRY_ONLY_CANONICAL` in `src/pipelines/run.py` for the full list.

### Extended Metrics (Recommended)
| Metric | Code | Purpose |
|--------|------|---------|
| **Mode Score M** | `src/metrics/mode_score.py` | **Behavior.** Logit-level recursive vs task classifier. |
| **Logit Lens** | `src/metrics/logit_lens.py` | **Evolution.** Crystallization layer & entropy trajectory. |
| **Cosine Sim** | `src/metrics/extended.py` | **Alignment.** Directional similarity (Early vs Late). |
| **Spectral Stats** | `src/metrics/extended.py` | **Shape.** Effective rank, spectral gap, top-1 ratio. |
| **Attn Entropy** | `src/metrics/extended.py` | **Focus.** Attention head sparsity at readout. |

---

## 2. Reporting Schema

Every experiment MUST output a `summary.json` following this strict schema.
This is enforced by `src/metrics/baseline_suite.py` and `src/pipelines/run.py`.

### JSON Schema (`summary.json`)
```json
{
  "experiment": "rv_l27_causal_validation",
  "timestamp": "20260115_120000",
  "model": "mistralai/Mistral-7B-v0.1",
  "prompt_bank_version": "84a2448e...",  // CRITICAL for reproducibility
  
  "n_pairs": 30,
  
  // --- R_V Statistics ---
  "rv_recursive_mean": 0.52,
  "rv_recursive_std": 0.05,
  "rv_baseline_mean": 1.01,
  "rv_baseline_std": 0.02,
  "rv_delta_mean": -0.49,
  "rv_delta_ci_95": [-0.51, -0.47],
  "rv_cohens_d": -3.56,
  "rv_p_value": 1.2e-23,
  
  // --- Logit Diff Statistics ---
  "logit_diff_delta_mean": 2.5,
  "logit_diff_cohens_d": 1.2,
  "logit_diff_p_value": 0.001,
  
  // --- Extended Metrics (if enabled) ---
  "mode_score_m": 0.85,
  
  // --- Artifacts ---
  "artifacts": {
      "csv": "results/phase1/runs/.../results.csv",
      "config": "results/phase1/runs/.../config.json"
  }
}
```

---

## 3. Run Ledger (`RUN_INDEX.jsonl`)

All runs are automatically logged to `results/RUN_INDEX.jsonl`.
This acts as a permanent ledger of all experimental outcomes.

### Ledger Entry Format
```json
{
  "timestamp": "20260115_120000",
  "experiment": "rv_l27_causal_validation",
  "model": "mistralai/Mistral-7B-v0.1",
  "prompt_bank_version": "84a2448e...",
  "success": true,
  "run_dir": "results/phase1/runs/20260115_120000_rv_l27...",
  
  // Key Outcome Metrics
  "rv_d": -3.56,
  "rv_p": 1.2e-23,
  "rv_delta": -0.49,
  "logit_diff_d": 1.2,
  
  "git_commit": "abc1234"
}
```

---

## 4. Compliance Plan

### Where is it enforced?
1. **Pipeline Level**: `src/pipelines/run.py` automatically injects `prompt_bank_version` and writes to the Ledger.
2. **Metric Level**: `BaselineMetricsSuite` ensures consistent statistical calculation (Cohen's d, CI).
3. **Artifact Level**: `atomic_config_snapshot` ensures exact reproduction parameters are saved.

### How to comply?
1. **Use `BaselineMetricsSuite`**:
   ```python
   suite = BaselineMetricsSuite(model, tokenizer)
   stats = suite.compute_batch_statistics(recursive, baseline)
   # stats now contains all required keys
   ```

2. **Use `PromptLoader`**:
   ```python
   loader = PromptLoader()
   prompts = loader.get_balanced_pairs()
   ```

3. **Return Standard Result**:
   ```python
   return ExperimentResult(summary=stats)
   ```

---

## 5. Implementation Details

### R_V (Geometric Contraction)
Defined as $R_V = \frac{PR_{late}}{PR_{early}}$ where $PR = \frac{(\sum \lambda_i^2)^2}{\sum \lambda_i^4}$.
- **Input**: Last $W=16$ tokens of prompt (not generation).
- **Process**: SVD of $V$ projection at Layer 5 (Early) and Layer 27 (Late).
- **Constraint**: Must use `torch.float64` for SVD or handle low-precision instability.

### Cohen's d (Effect Size)
Defined as $d = \frac{\mu_1 - \mu_2}{s_{pooled}}$.
- **Requirement**: Must report $d$ for all causal claims.
- **Thresholds**: $>0.2$ (Small), $>0.5$ (Medium), $>0.8$ (Large).
- **R_V Benchmark**: We consistently observe $|d| > 3.0$ for recursive vs baseline.

---

**Version**: 2.0 (Industry-Grade)
**Last Updated**: Jan 2026
