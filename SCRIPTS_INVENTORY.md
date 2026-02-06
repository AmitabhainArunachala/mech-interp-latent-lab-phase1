# Scripts Inventory

**Project:** mech-interp-latent-lab-phase1  
**Generated:** 2026-02-05  
**Total Scripts:** 43 (34 Python, 9 Shell)

---

## 1. ORCHESTRATION SCRIPTS (Core Workflow)

### Active - Primary Entry Points

| Script | Purpose | Usage Frequency | Status |
|--------|---------|-----------------|--------|
| `run_and_report.py` | Full experiment lifecycle: preflight → run → validate → report | **High** | ✅ Active |
| `run_batch_and_report.py` | Batch execution of multiple configs | **High** | ✅ Active |
| `preflight_audit.py` | Pre-experiment validation (gold standard compliance) | **High** | ✅ Active |
| `postrun_validator.py` | Post-experiment results validation | **High** | ✅ Active |
| `verify_research_ready.py` | Comprehensive repo readiness checks | **Medium** | ✅ Active |

**Dependencies:**
- `run_and_report.py` → `preflight_audit.py` → `src.pipelines.run` → `postrun_validator.py`
- `run_batch_and_report.py` → `run_and_report.py`

---

## 2. CONFIG GENERATION & MODEL SETUP

### Active

| Script | Purpose | Usage | Status |
|--------|---------|-------|--------|
| `generate_model_configs.py` | Generate discovery configs for new models | Onboard new models | ✅ Active |
| `run_model_discovery.sh` | Multi-phase discovery pipeline for new models | Model onboarding | ✅ Active |

**Workflow:**
```
generate_model_configs.py → creates configs/discovery/{model}/
run_model_discovery.sh {model} → runs 6-phase discovery
```

---

## 3. PROMPT MANAGEMENT

### Active

| Script | Purpose | Usage | Status |
|--------|---------|-------|--------|
| `create_prompt_families.py` | Create matched prompt families for cross-arch validation | Validation setup | ✅ Active |
| `pull_missing_prompt_sets.py` | Import legacy prompts with deduplication | Maintenance | ✅ Active |
| `integrate_champion_prompts.py` | Move experimental prompts to canonical bank | Maintenance | ⚠️ Legacy (one-time) |

**Dependencies:**
- All prompt scripts → `prompts/bank.json`
- `pull_missing_prompt_sets.py` reads from `REUSABLE_PROMPT_BANK/`

---

## 4. EXPERIMENT RUNNERS (Specific Pipelines)

### Active - Phase 1 (Mechanism)

| Script | Pipeline | Purpose | Status |
|--------|----------|---------|--------|
| `run_c2_rv_measurement.py` | `c2_rv_measurement` | C2 intervention + R_V measurement | ✅ Active |
| `run_c2_ablation.py` | `c2_ablation` | Component ablation studies | ✅ Active |
| `run_logit_lens_analysis.py` | `logit_lens_analysis` | Token probability tracing | ✅ Active |
| `run_l0_l1_l3_experiment.py` | `mlp_combined_sufficiency` | Multi-layer sufficiency | ✅ Active |
| `run_l0_l1_l18_l19_l20.py` | `mlp_combined_sufficiency` | Full circuit test | ✅ Active |
| `run_mlp_vproj_combined.py` | `mlp_vproj_combined` | MLP + V_proj combined | ✅ Active |

### Active - Phase 2 (Generalization)

| Script | Pipeline | Purpose | Status |
|--------|----------|---------|--------|
| `run_cross_arch_llama.py` | `cross_architecture_validation` | Cross-model validation | ✅ Active |
| `run_cross_architecture.sh` | Multi-model runner | Batch cross-architecture | ✅ Active |

### Canonical Suite

| Script | Purpose | Status |
|--------|---------|--------|
| `stage2_canonical_suite.py` | Run 13 canonical experiments | ⚠️ Partial (7/13 implemented) |

---

## 5. ANALYSIS & STATISTICS

### Active

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `compute_c2_statistics.py` | Comprehensive C2 stats | `c2_rv_measurement.csv` | `comprehensive_results.json` |
| `compute_stats.py` | Paired t-tests, Cohen's d | `c2_rv_measurement.csv` | Console report |
| `check_c2_results.py` | Quick C2 validation | Run directory | Publication criteria check |
| `analyze_kv_sweep.py` | KV layer sweep analysis | `kv_mechanism.csv` | Transfer efficiency report |
| `analyze_logit_lens_results.py` | Logit lens interpretation | `logit_lens_analysis.csv` | Key findings |
| `fix_cross_arch_summary.py` | Regenerate summary.json | `cross_architecture_validation.csv` | Fixed summary |

**Dependencies:**
- All analysis scripts → pandas, scipy, numpy
- Most read from `results/phase1_mechanism/runs/`

---

## 6. SMOKE TESTS & VALIDATION

### Active

| Script | Purpose | When to Use | Status |
|--------|---------|-------------|--------|
| `smoke_test_l0_necessity.py` | Quick L0 ablation test (n=5) | Pre-commit validation | ✅ Active |
| `smoke_test_l0_sufficiency.py` | Quick L0 patch test (n=5) | Pre-commit validation | ✅ Active |

**Usage:**
```bash
# Before committing changes
python scripts/smoke_test_l0_necessity.py
python scripts/smoke_test_l0_sufficiency.py
```

---

## 7. RUNPOD / REMOTE GPU SCRIPTS

### Active (Remote Execution)

| Script | Purpose | Environment | Status |
|--------|---------|-------------|--------|
| `setup_runpod.sh` | Initial RunPod setup | RunPod | ✅ Active |
| `setup_gpu_repo.sh` | Sync repo to GPU + install deps | Local → GPU | ✅ Active |
| `sync_to_runpod.sh` | Sync specific files to RunPod | Local → GPU | ✅ Active |
| `pull_results.sh` | Sync results back to local | GPU → Local | ✅ Active |
| `run_on_runpod.sh` | Execute experiment on RunPod | RunPod | ✅ Active |
| `quick_runpod_sync.sh` | Fast sync with validation | Local ↔ GPU | ✅ Active |
| `monitor_experiment.sh` | Check running experiment | RunPod | ✅ Active |
| `monitor_head_discovery.sh` | Monitor head validation | RunPod | ✅ Active |
| `run_h31_validation_remote.sh` | Run H31 validation remotely | RunPod | ✅ Active |
| `run_circuit_hunt_on_runpod.sh` | Execute circuit hunt | RunPod | ✅ Active |
| `run_full_sweep_remote.sh` | Full parameter sweep | RunPod | ✅ Active |
| `test_mistral_remote.sh` | Remote smoke test | RunPod | ✅ Active |
| `verify_sync.sh` | Validate sync integrity | Local | ✅ Active |

### Supporting Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `bootstrap_remote.sh` | Remote bootstrap | ⚠️ Legacy |
| `connect_and_test.sh` | SSH connectivity test | ✅ Active |
| `push_repo.sh` | Git push helper | ✅ Active |
| `sync_to_main.sh` | Sync to main branch | ✅ Active |
| `sync_to_main_auto.sh` | Auto-sync with cron | ⚠️ Deprecated (use CI) |
| `check_experiment_progress.sh` | Check progress | ✅ Active |
| `run_bootstrap_over_ssh.sh` | SSH bootstrap | ⚠️ Legacy |

---

## 8. UTILITY SCRIPTS

### Active

| Script | Purpose | Status |
|--------|---------|--------|
| `clean.sh` | Remove __pycache__, .pyc files | ✅ Active |
| `__init__.py` | Package marker (empty) | ✅ Active |

---

## DEPENDENCY MAP

```
scripts/
├── run_and_report.py
│   ├── preflight_audit.py
│   ├── src.pipelines.run
│   └── postrun_validator.py
├── run_batch_and_report.py
│   └── run_and_report.py
├── verify_research_ready.py
│   ├── src.metrics.rv
│   ├── src.core.model_physics
│   ├── src.pipelines.registry
│   └── prompts.loader
├── generate_model_configs.py
│   └── src.utils.multi_model_discovery
├── run_model_discovery.sh
│   ├── generate_model_configs.py
│   └── src.pipelines.run
├── *experiment_runners.py
│   └── src.pipelines.*
├── *analysis.py
│   └── results/*/runs/
└── runpod/*
    └── ssh/rsync workflows
```

---

## USAGE RECOMMENDATIONS

### Daily Workflow

```bash
# 1. Verify repo state
python scripts/verify_research_ready.py

# 2. Run smoke tests before committing
python scripts/smoke_test_l0_necessity.py
python scripts/smoke_test_l0_sufficiency.py

# 3. Run experiment with full validation
python scripts/run_and_report.py --config configs/gold/01_existence.json

# 4. Clean cache
./scripts/clean.sh
```

### Adding New Model

```bash
# 1. Generate configs
python scripts/generate_model_configs.py --model meta-llama/Meta-Llama-3-8B

# 2. Run discovery phases
./scripts/run_model_discovery.sh llama3_8b all
```

### Remote GPU Execution

```bash
# 1. Setup
./scripts/runpod/setup_gpu_repo.sh

# 2. Run experiment
./scripts/runpod/run_on_runpod.sh

# 3. Monitor
./scripts/runpod/monitor_experiment.sh

# 4. Pull results
RUNPOD_HOST=... RUNPOD_PORT=... ./scripts/runpod/pull_results.sh
```

---

## DEPRECATED / LEGACY SCRIPTS

| Script | Reason | Replacement |
|--------|--------|-------------|
| `integrate_champion_prompts.py` | One-time migration | N/A |
| `sync_to_main_auto.sh` | Replaced by CI/CD | GitHub Actions |
| `bootstrap_remote.sh` | Outdated setup | `setup_runpod.sh` |

---

## SCRIPT HEALTH SUMMARY

| Category | Count | Active | Deprecated |
|----------|-------|--------|------------|
| Orchestration | 5 | 5 | 0 |
| Config Generation | 2 | 2 | 0 |
| Prompt Management | 3 | 2 | 1 |
| Experiment Runners | 10 | 10 | 0 |
| Analysis | 6 | 6 | 0 |
| Smoke Tests | 2 | 2 | 0 |
| RunPod/Remote | 19 | 16 | 3 |
| Utilities | 2 | 2 | 0 |
| **TOTAL** | **49** | **45** | **4** |

---

## MAINTENANCE NOTES

1. **verify_research_ready.py** - Update `EXPECTED_PROMPT_COUNT` when bank.json changes
2. **preflight_audit.py** - Sync with gold standard requirements
3. **runpod/** - Keep SSH config in sync with actual RunPod instances
4. **analysis scripts** - Update paths if results directory structure changes
