# Unified Auditor-Experimenter Integration Specification

**Date:** 2026-02-05  
**Version:** 1.0  
**Status:** CANONICAL  
**Purpose:** Align OpenClawd's mi-experimenter/mi-auditor with mech-interp-lab gold standard

---

## Executive Summary

This document establishes a **unified contract** between:
- **mech-interp-latent-lab-phase1/** — The canonical research repository (gold standard)
- **clawd/skills/mi-experimenter/** — Automated experiment runner
- **clawd/skills/mi-auditor/** — Automated results auditor

All systems MUST operate under the same methodology, metrics, and reporting standards.

---

## Orientation Links (Top 10)

1. [Measurement Contract](docs/standards/MEASUREMENT_CONTRACT.md)
2. [Research Progress Summary](docs/status/RESEARCH_PROGRESS_SUMMARY.md)
3. [Phase 1 Final Report](R_V_PAPER/research/PHASE1_FINAL_REPORT.md)
4. [Bridge Hypothesis Investigation](BRIDGE_HYPOTHESIS_INVESTIGATION.md)
5. [Statistical Audit Executive Summary](STATISTICAL_AUDIT_EXECUTIVE_SUMMARY.md)
6. [Reproducibility Audit Report](REPRODUCIBILITY_AUDIT_REPORT.md)
7. [Quality Control Report](QUALITY_CONTROL_REPORT.md)
8. [Architecture Executive Summary](ARCHITECTURE_EXECUTIVE_SUMMARY.md)
9. [Publication Blockers Status](PUBLICATION_BLOCKERS_STATUS.md)
10. [Agent Onboarding](AGENT_ONBOARDING.md)

---

## Repo Story (12 bullets)

1. This repo is the canonical research source of truth.
2. OpenClawd experimenter/auditor must align with this repo’s contracts.
3. R_V is defined as PR_late / PR_early, measured on prompt tokens.
4. Gold configs and prompt bank are mandatory inputs for any run.
5. All runs must emit standard artifacts (config, summary, per_sample, prompt bank version, hardware).
6. Statistical thresholds and control requirements are enforced at audit time.
7. Causal validity requires random/shuffled/wrong-layer/orthogonal controls.
8. Post-run validation must confirm schema, claims, and reproducibility metadata.
9. Cross-architecture validation is ongoing; tier system prioritizes models.
10. Bridge experiment is partial; behavior causality is unproven.
11. Reproducibility gaps are mostly metadata and determinism, not methodology.
12. Immediate priority: align thresholds, fix tool docs, run Mixtral validation.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         UNIFIED MI RESEARCH PIPELINE                             │
│                                                                                  │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐         │
│  │  PRE-FLIGHT      │     │  EXPERIMENT      │     │  POST-RUN        │         │
│  │  AUDITOR         │────►│  RUNNER          │────►│  AUDITOR         │         │
│  │  (Cursor)        │     │  (OpenClawd)     │     │  (Cursor)        │         │
│  │                  │     │                  │     │                  │         │
│  │ • Config verify  │     │ • Execute        │     │ • Statistical    │         │
│  │ • Gold standard  │     │ • Log metrics    │     │   validation     │         │
│  │ • Code review    │     │ • Stream data    │     │ • Claims audit   │         │
│  └──────────────────┘     └──────────────────┘     └──────────────────┘         │
│           │                        │                        │                   │
│           │                        │                        │                   │
│           ▼                        ▼                        ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    CANONICAL DATA STORE                                  │   │
│  │                    mech-interp-latent-lab-phase1/results/                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Canonical Sources of Truth

| Component | Location | Status |
|-----------|----------|--------|
| **Prompt Bank** | `mech-interp-latent-lab-phase1/prompts/bank.json` | 754 prompts |
| **Measurement Contract** | `mech-interp-latent-lab-phase1/docs/standards/MEASUREMENT_CONTRACT.md` | LOCKED v1.1 |
| **Gold Standard Configs** | `mech-interp-latent-lab-phase1/configs/gold/` | 32 configs |
| **R_V Implementation** | `mech-interp-latent-lab-phase1/src/metrics/rv.py` | CANONICAL |
| **Agent Gold Standard** | `mech-interp-latent-lab-phase1/AGENT_PROMPT_GOLD_STANDARD.md` | Dec 2025 |

**Rule:** OpenClawd's implementations MUST match these canonical sources.

---

## 3. PR Formula Verification

### Canonical PR Formula (CORRECT)

```
PR = (Σλ²)² / Σλ⁴
```

Where λ are singular values from SVD of V-projection window.

### Implementation Check

| Source | Implementation | Status |
|--------|---------------|--------|
| `mech-interp-lab/src/metrics/rv.py` | `(S_sq.sum() ** 2) / (S_sq ** 2).sum()` | ✅ CORRECT |
| `clawd/skills/rv_toolkit/rv_core.py` | `(sum_S2 ** 2) / (sum_S4 + eps)` | ✅ CORRECT |

**VERIFIED:** Both implementations use the same correct formula.

### ⚠️ Bug Status Clarification

The SKILL.md v5.1 mentions a "PR formula bug" — this appears to be **outdated documentation**. 

Both current implementations are correct. The "bug" about normalized vs unnormalized values is:
1. A documentation artifact from an earlier version
2. NOT present in current code

**Action:** Update SKILL.md to remove the false bug report.

---

## 4. Statistical Requirements

### Gold Standard Requirements (Enforced by Both Systems)

| Requirement | Threshold | Enforced By |
|-------------|-----------|-------------|
| Sample size per condition | N ≥ 50 | mi-experimenter |
| Effect size reporting | Cohen's d | mi-experimenter |
| Significance threshold | p < 0.001 for claims | mi-auditor |
| Multiple comparisons | Bonferroni correction | mi-auditor |
| 95% Confidence intervals | Required | Both |
| Controls | 4 types minimum | mi-experimenter |

### Control Types (REQUIRED)

1. **random** — Gaussian noise, norm-matched
2. **shuffled** — Permuted source activation
3. **wrong_layer** — Patch from incorrect layer
4. **orthogonal** — Direction orthogonal to source

---

## 5. Artifact Structure Contract

### Run Directory Structure (ENFORCED)

```
results/<phase>/runs/<YYYYMMDD_HHMMSS>_<experiment>/
├── config.json           # Exact config snapshot (REQUIRED)
├── summary.json          # Aggregated statistics (REQUIRED)
├── per_sample.csv        # Individual results (REQUIRED)
├── prompt_bank_version.json  # Hash of prompts/bank.json
├── hardware_info.json    # GPU model, CUDA version, precision
└── report.md             # Human-readable summary (OPTIONAL)
```

### Summary.json Required Fields

```json
{
  "experiment": "string",
  "model": "string",
  "timestamp": "ISO-8601",
  "n_samples": "integer",
  "cohens_d": "float",
  "p_value": "float",
  "transfer_efficiency": "float (percent)",
  "controls_passed": {
    "random": "boolean",
    "shuffled": "boolean",
    "wrong_layer": "boolean",
    "orthogonal": "boolean"
  },
  "rv_recursive_mean": "float",
  "rv_baseline_mean": "float",
  "rv_patched_mean": "float",
  "hardware": {
    "gpu_model": "string",
    "cuda_version": "string",
    "precision": "string"
  }
}
```

---

## 6. Integration Protocol

### Pre-Flight Audit (Cursor → OpenClawd)

Before ANY GPU run, Cursor audits:

```
PRE-FLIGHT CHECKLIST
├── [ ] Config matches gold standard format
├── [ ] Experiment registered in registry.py
├── [ ] Prompt bank version tracked
├── [ ] Target layer correct for architecture
├── [ ] Sample size N ≥ 50
├── [ ] All 4 controls specified
├── [ ] Determinism flags set
└── [ ] Hardware logging enabled
```

### Experiment Execution (OpenClawd)

OpenClawd runs with:

```python
from mi_experimenter import RVCausalValidator

validator = RVCausalValidator(
    model_name="mistralai/Mixtral-8x7B-v0.1",  # Tier 2 priority
    target_layer=27,
    controls=["random", "shuffled", "wrong_layer", "orthogonal"],
    n_pairs=50,  # Minimum for gold standard
    save_results=True,
    output_dir="~/mech-interp-latent-lab-phase1/results/canonical/"
)
results = validator.run()
```

### Post-Run Validation (Cursor)

Cursor validates results against gold standard:

```
POST-RUN VALIDATION
├── [ ] All required fields present in summary.json
├── [ ] Cohen's d reported with p-value
├── [ ] All 4 controls have results
├── [ ] Transfer efficiency calculated
├── [ ] Hardware info logged
├── [ ] Results reproducible (seed logged)
└── [ ] Claims match evidence strength
```

---

## 7. Model Tier System

### Tier 1: IRONCLAD (Validated)

| Model | Cohen's d | Controls | Status |
|-------|-----------|----------|--------|
| Mistral 7B | -3.56 | 4/4 ✅ | VALIDATED |
| Gemma 2-9B | -2.09 | 4/4 ✅ | VALIDATED |
| Pythia 2.8B | -4.51 | 4/4 ✅ | VALIDATED |

**Rule:** Do NOT re-run Tier 1 models. Use for paper claims.

### Tier 2: DISCOVERY → CAUSAL (Priority Queue)

| Model | Discovery Effect | Priority |
|-------|-----------------|----------|
| Mixtral 8x7B | 24.3% | 🔥 PRIORITY 1 |
| Llama-3 8B | 11.7% | 🔥 PRIORITY 2 |
| Qwen 7B | 9.2% | PRIORITY 3 |
| Phi-3 | 6.9% | PRIORITY 4 |

**Rule:** GPU time goes here. Run 4-control validation.

### Tier Upgrade Requirements

To upgrade from Tier 2 → Tier 1:
- [ ] 4 controls passed
- [ ] Cohen's d > 0.5
- [ ] p < 0.001
- [ ] N ≥ 50 pairs
- [ ] Replication within session

---

## 8. Auditor Standards

### Statistical Rigor Audit

From `clawd/skills/mi_auditor/auditors/statistical_rigor.py`:

```python
class StatisticalAuditor:
    COHENS_D_SMALL = 0.2
    COHENS_D_MEDIUM = 0.5
    COHENS_D_LARGE = 0.8
    COHENS_D_HUGE = 2.0
    P_THRESHOLD = 0.05  # Note: Gold standard uses 0.001
    POWER_THRESHOLD = 0.8
```

**ALIGNMENT REQUIRED:** Change P_THRESHOLD to 0.001 for claims.

### Causal Validity Audit

From `clawd/skills/mi_auditor/auditors/causal_validity.py`:

Checks for:
- Proper controls (all 4 types)
- Direction specificity (random fails)
- Layer specificity (wrong_layer fails)
- Orthogonal specificity

---

## 9. Synchronization Points

### Daily Sync (Automated)

```bash
# Cursor runs daily
cd ~/mech-interp-latent-lab-phase1
git status
python -c "from prompts.loader import PromptLoader; print(f'Prompts: {len(PromptLoader().prompts)}')"
```

### Pre-GPU Sync (Manual)

Before GPU session:
1. Cursor audits OpenClawd's config
2. OpenClawd confirms alignment
3. Human (Dhyana) approves
4. GPU run proceeds

### Post-GPU Sync (Automated)

After GPU session:
1. OpenClawd saves to `mech-interp-lab/results/`
2. Cursor validates summary.json
3. Cursor generates audit report
4. Results integrated to knowledge graph

---

## 10. Communication Protocol

### Message Types (JSON)

```json
// Design Proposal (OpenClawd → Cursor)
{
  "type": "design_proposal",
  "experiment": "string",
  "model": "string",
  "config_path": "string",
  "expected_outcome": "string"
}

// Design Critique (Cursor → OpenClawd)
{
  "type": "design_critique",
  "verdict": "accept|accept_with_revisions|reject",
  "concerns": ["string"],
  "required_changes": ["string"]
}

// Execution Report (OpenClawd → Cursor)
{
  "type": "execution_report",
  "results_path": "string",
  "summary": {...}
}

// Validation Report (Cursor → OpenClawd)
{
  "type": "validation_report",
  "verdict": "validated|needs_revision|rejected",
  "tier_upgrade": "boolean",
  "claims_supported": ["string"],
  "gaps_identified": ["string"]
}
```

---

## 11. Immediate Actions

### OpenClawd TODO

- [ ] Fix SKILL.md false bug report (PR formula is correct)
- [ ] Align P_THRESHOLD to 0.001 for gold standard
- [ ] Ensure output goes to `mech-interp-lab/results/`
- [ ] Add prompt_bank_version.json to output

### Cursor TODO

- [ ] Create automated pre-flight checker script
- [ ] Create automated post-run validator script
- [ ] Update AUDIT_REPORT_2026-02-05.md with integration status

### Shared TODO

- [ ] Run Mixtral 8x7B validation (PRIORITY 1)
- [ ] Complete R_V(t) trajectory experiment
- [ ] Update paper with Tier 2 results

---

## 12. Success Metrics

### Pipeline Health

| Metric | Target | Current |
|--------|--------|---------|
| Design→Approval cycles | ≤ 2 | TBD |
| Validation confidence avg | ≥ 0.75 | TBD |
| Tier 2 → Tier 1 upgrades | 4 models | 0 |
| False positive rate | < 5% | TBD |

### Research Progress

| Metric | Target | Current |
|--------|--------|---------|
| Models with causal validation | 7+ | 3 |
| Paper-ready claims | 5+ | 3 |
| Cross-architecture evidence | 5 architectures | 3 |

---

## Appendix: Quick Reference

### Entry Points

```bash
# Run experiment (OpenClawd)
python -m mi_experimenter.experiments.rv_causal_validator --model mixtral-8x7b

# Validate results (Cursor)
python -m src.pipelines.run --config configs/canonical/rv_l27_causal_validation.json

# Full pipeline audit (Cursor)
python -c "from src.utils.run_index import audit_all; audit_all()"
```

### Key Files

| Purpose | Path |
|---------|------|
| Gold Standard | `mech-interp-lab/AGENT_PROMPT_GOLD_STANDARD.md` |
| Measurement Contract | `mech-interp-lab/docs/standards/MEASUREMENT_CONTRACT.md` |
| OpenClawd Skill | `clawd/skills/mi-experimenter/SKILL.md` |
| Integration Doc | `clawd/AUDITOR_EXPERIMENTER_INTEGRATION.md` |
| This Document | `mech-interp-lab/UNIFIED_AUDITOR_INTEGRATION.md` |

---

*"When the auditor and experimenter are aligned, truth emerges from the loop."*

---

**Document Status:** ACTIVE  
**Next Review:** After Mixtral validation complete  
**Owner:** Dhyana + Claude (Cursor & OpenClawd)
