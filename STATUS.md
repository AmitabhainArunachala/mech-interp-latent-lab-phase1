# R_V Research: Current Status

**Last Updated:** 2026-01-15
**Phase:** 7 of 10 (Unit Tests)
**Progress:** ██████░░░░ 60%

---

## Quick Links

| Document | Purpose |
|----------|---------|
| [.planning/PROJECT.md](.planning/PROJECT.md) | Requirements, constraints, scope |
| [.planning/ROADMAP.md](.planning/ROADMAP.md) | 10-phase plan with details |
| [.planning/STATE.md](.planning/STATE.md) | Current position, completed phases |
| [AUDIT.md](AUDIT.md) | Full repo inventory |
| [JAN11_2025_SESSION_SUMMARY.md](JAN11_2025_SESSION_SUMMARY.md) | Latest experimental results |

---

## The Finding

**R_V contraction is a causal, layer-specific geometric signature of recursive self-observation.**

| Metric | Value | Source |
|--------|-------|--------|
| Cohen's d | -3.56 | L27 causal validation |
| p-value | < 10⁻⁶ | n=45 pairs |
| Transfer efficiency | 117.8% | Activation patching |
| Champions R_V | 0.52 | vs controls 0.78-0.83 |

---

## Completed (Phases 1-6)

- [x] **Three-tier restructure**: pipelines, configs, results → canonical/discovery/archive
- [x] **Statistical standards**: All canonical pipelines report n, p-value, Cohen's d, 95% CI
- [x] **Mistral-7B validated**: Cross-arch baseline replicated (R_V = 0.5186)

---

## In Progress

| Task | Owner | Status |
|------|-------|--------|
| Llama cross-arch validation | GPU | Blocked (needs HF_TOKEN) |
| Commit restructure | Local | Ready |

---

## Next Up (Phase 7)

- [ ] Unit tests for `src/metrics/rv.py`
- [ ] Unit tests for `src/metrics/logit_diff.py`
- [ ] Unit tests for `src/metrics/baseline_suite.py`
- [ ] Unit tests for `src/metrics/mode_score.py`

---

## Directory Structure

```
src/
├── metrics/          # R_V, logit_diff, logit_lens, mode_score, baseline_suite
├── core/             # hooks, activations, patching, steering
└── pipelines/
    ├── canonical/    # 7 publication-ready pipelines
    ├── discovery/    # 12 methodology pipelines
    └── archive/      # 35 historical/deprecated

configs/
├── canonical/        # 15 validated configs
├── discovery/        # 27 experimental configs
└── archive/          # 13 deprecated configs

results/
├── canonical/              # Paper-worthy runs (rv_l27, confound, c2)
├── discovery/              # Methodology runs
├── phase2_generalization/  # Cross-architecture validation runs
└── archive/                # Historical/failed runs

prompts/
├── bank.json         # 754 prompts, version-tracked
├── loader.py         # PromptLoader API
└── README.md         # Schema documentation
```

---

## Canonical Pipelines

| Pipeline | Purpose | Key Result |
|----------|---------|------------|
| `rv_l27_causal_validation.py` | L27 activation patching | d=-3.56 |
| `confound_validation.py` | 4-pillar controls | Random/shuffled/wrong-layer |
| `random_direction_control.py` | Directional specificity | True > random |
| `mlp_ablation_necessity.py` | MLP necessity | Required for effect |
| `mlp_sufficiency_test.py` | MLP sufficiency | Sufficient for effect |
| `mlp_combined_sufficiency_test.py` | Combined MLP tests | Validated |
| `head_ablation_validation.py` | Head-level ablation | H18/H26 critical |

---

## Key Files

| File | Purpose |
|------|---------|
| `MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md` | Crown jewel result documentation |
| `prompts/bank.json` | All 754 prompts with expected R_V ranges |
| `src/metrics/rv.py` | The R_V metric implementation |
| `configs/canonical/rv_l27_causal_validation.json` | Gold standard config |

---

## Publication Blockers

1. **Cross-architecture validation** — Need Llama results to claim generality
2. **n=100 runs** — Current max is n=45, need n≥100 for publication claims
3. **Multi-token generation** — Bridge R_V (prompt-time) to behavioral output

---

*This file is the authoritative status. See .planning/ for detailed roadmap.*
