# Canonical Experiments Reference

**Single source of truth for experiment categories, required metrics, and design rationale.**

---

## Experiment Categories

Canonical experiments fall into two categories based on what they measure:

### 1. Geometry-Only Experiments

These experiments measure R_V geometric contraction via activation patching/ablation. They do **not** generate text, so `logit_diff_*` metrics are intentionally `null`.

| Experiment | Purpose | Why No Logit Diff |
|------------|---------|-------------------|
| `rv_l27_causal_validation` | Validate Layer 27 as causal site for R_V | Patches activations, measures R_V delta only |
| `mlp_sufficiency_test` | Test if MLP alone can induce contraction | Ablates components, no text generation |
| `combined_mlp_sufficiency_test` | Test combined MLP layers | Multi-layer ablation study |
| `head_ablation_validation` | Identify which attention heads matter | Head-by-head ablation sweep |
| `random_direction_control` | Verify steering direction specificity | Random vector injection control |

**Required metrics for geometry-only:**
- `rv_delta_mean` (primary outcome)
- `rv_cohens_d` (effect size)
- `rv_p_value` (significance)
- `n_pairs` (sample size)

**Nullable metrics:**
- `logit_diff_*` (all null — no text generated)

### 2. Behavioral Experiments

These experiments measure both geometric contraction AND behavioral output. They generate text and compute logit differences.

| Experiment | Purpose |
|------------|---------|
| `confound_validation` | Rule out confounds (length, complexity) |
| `mlp_ablation_necessity_prompt_pass` | Test MLP necessity with behavioral grounding |
| `behavioral_grounding` | Correlate R_V with output behavior |
| `behavioral_grounding_batch` | Batch version for statistical power |

**Required metrics for behavioral:**
- All geometry metrics (above)
- `logit_diff_delta_mean`
- `logit_diff_cohens_d`
- `logit_diff_p_value`

---

## Strict Mode

The runner (`src/pipelines/run.py`) supports a `--strict` flag:

```bash
python -m src.pipelines.run --config config.json --strict
```

**Behavior:**
- Fails if any required metric is `None`
- **Exception:** Geometry-only experiments (listed in `GEOMETRY_ONLY_CANONICAL`) are excluded from this check

**Rationale:** Geometry-only experiments intentionally omit logit_diff because they don't generate text. Strict mode ensures behavioral experiments have complete metrics without penalizing geometry-only studies.

---

## Code References

```python
# In src/pipelines/run.py

GEOMETRY_ONLY_CANONICAL = {
    "rv_l27_causal_validation",
    "mlp_sufficiency_test",
    "combined_mlp_sufficiency_test",
    "head_ablation_validation",
    "random_direction_control",
}
```

---

## Design Rationale

### Why separate geometry from behavior?

1. **Measurement purity:** R_V measures geometric contraction in Value space. This is a property of the forward pass on the prompt, independent of what tokens are generated.

2. **Causal isolation:** Patching experiments (like `rv_l27_causal_validation`) need to measure the effect of activations on geometry without the confound of generation behavior.

3. **Efficiency:** Geometry-only experiments are faster because they don't require autoregressive generation.

### When to use each category?

- **Geometry-only:** When validating the R_V metric itself, testing causal mechanisms, or running ablation studies
- **Behavioral:** When correlating geometric signatures with output behavior, or validating that geometric changes have behavioral consequences

---

**Version:** 1.0  
**Last Updated:** Jan 2026
