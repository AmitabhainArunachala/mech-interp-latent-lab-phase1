## Repo Signal Reorg (Industry Standard)

**Date:** 2026-02-05  
**Scope:** Mark KEEP vs RAMP vs ARCHIVE for signal + rigor  
**Basis:** 25 meta files reviewed (contracts, audits, bridge reports, architecture reviews)

### Definitions
- **KEEP (Signal / Industry Standard):** High‑N (≥50), clear effect (|d| ≥ 0.8 or p < 0.001), canonical method, reproducible artifacts, contract compliant.
- **RAMP (Promising):** Mechanistically plausible or novel, but low‑N, confounded, or missing causal proof; must be re‑run with high‑N + full controls.
- **ARCHIVE (Noisy/Old/Dead‑end):** Duplicated, superseded, or heavily confounded; keep only for historical traceability.

---

## KEEP — Canonical Standards and Signal

**Contracts / Standards**
- `docs/standards/MEASUREMENT_CONTRACT.md` (LOCKED)
- `UNIFIED_AUDITOR_INTEGRATION.md` (CANONICAL)
- `MCP_MONITOR_PROTOCOL.md` (ACTIVE)
- `.cursorrules` (repo standard)

**Canonical Metrics & Validation**
- `src/metrics/rv.py` (canonical R_V ratio)
- `src/metrics/baseline_suite.py` (Nanda baseline metrics)
- `configs/gold/*.json` (gold configs, high‑signal focus)
- `results/canonical/*` (paper‑worthy runs)

**Verified Statistical Signal**
- `STATISTICAL_AUDIT_REPORT.md`
- `STATISTICAL_AUDIT_EXECUTIVE_SUMMARY.md`
- `REPRODUCIBILITY_AUDIT_REPORT.md` (actionable gaps, still KEEP)

**Core Research Findings**
- `BRIDGE_HYPOTHESIS_SYNTHESIS.md` (partial validation, honest framing)
- `BRIDGE_STATUS_SUMMARY.md` (prompt→R_V validated)
- `BRIDGE_HYPOTHESIS_INVESTIGATION.md` (confounds documented)

---

## RAMP — Promising, Needs High‑N / Causal Upgrade

**Top priority (causal bridge / cross‑arch)**
- Mixtral causal validation (gold config): `configs/gold/28_mixtral_causal_validation.json`
- Multi‑token R_V(t) trajectory (bridge): `configs/canonical/multi_token_bridge_mistral.json`
- Activation patching causal test (baseline prompts patched with recursive activations)
- Head localization / head ablation validation (`head_ablation_validation`)

**Promising low‑N (must replicate)**
- AI self‑reference amplification (d=1.18, p=0.004)  
  Source: `results/canonical/session_2_complete/ai_framing_n15/results.json`
- Causal transfer experiments in archive (validated patterns, low‑N)
  - `archive/rv_paper_code/VALIDATED_mistral7b_layer27_activation_patching.py`
  - `archive/scripts/experiment_multi_token_generation.py`
  - `archive/scripts/comprehensive_head_discovery.py`
  - `archive/scripts/comprehensive_circuit_test.py`
  - Transfer suites (`ultimate_transfer.py`, `refined_nuclear_transfer.py`, etc.)

**Signal upgrade requirements**
- n ≥ 50 pairs (prefer ≥80)
- Full control suite (random, shuffled, wrong_layer, orthogonal)
- Artifact contract enforced (summary.json + per_sample.csv + hardware_info.json)
- Correct R_V ratio (PR_late/PR_early), not single‑layer PR

---

## ARCHIVE — Noisy / Duplicated / Dead‑end (Keep for history only)

**Known noise/confounds**
- Any analysis relying on L4 string‑matching as “phenomenology” without semantic validation
- Any R_V computed as single‑layer PR (rv_toolkit metrics mismatch per QC report)
- Runs with truncation bias (≥80% truncated) and filtered subsets

**Docs to archive (historical)**
- Session reports or single‑run status files that do not change policy:
  - `FINAL_ALIGNMENT_REPORT.md`
  - `QUALITY_CONTROL_REPORT.md`
  - `AUDIT_REPORT_2026-02-05.md`
  - `PUBLICATION_BLOCKERS_STATUS.md`

**Archive inventory**
- `archive/` directory (per audit: 97 keep‑archived, 20 delete)
- `docs/sessions/*`, `docs/status/*` (historical logs)

---

## Immediate Reorg Actions (Non‑destructive)

1. **Create Signal Index** (this doc) and use it as the single triage source.
2. **Tag RAMP items** in configs/results with a small `RAMP_UP.md` note per run.
3. **Prevent low‑signal runs** by enforcing preflight checks (n, controls, contract).
4. **Do not delete results**; use archive tagging instead.

---

## Output Labels (for future audit)

- `KEEP_SIGNAL` — meets industry standard
- `RAMP_UP` — promising, needs high‑N + causal proof
- `ARCHIVE_ONLY` — historical, noisy, or superseded

