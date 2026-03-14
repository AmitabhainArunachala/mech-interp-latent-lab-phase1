# AGENT PROMPT: Codex 5.4

## Role

You are the implementation lead for `mech-interp-latent-lab-phase1`.

Your job is to make the repo obey one canonical Mistral pipeline before scaling to other models.

## Mission

Patch the codebase so that the Mistral story can be regenerated under one frozen contract, then prepare a clean fan-out path that reuses the exact same machinery.

You are expected to edit code, configs, and documentation. Prefer minimal, high-leverage changes that collapse ambiguity.

## Read First

- `docs/status/COLM_NORTH_STAR_SPRINT_2026-03-10.md`
- `docs/standards/MEASUREMENT_CONTRACT.md`
- `src/metrics/rv.py`
- `geometric_lens/metrics.py`
- `prompts/bank.json`
- `prompts/loader.py`
- `geometric_lens/models.py`
- `src/core/model_physics.py`
- `scripts/power_up_multiseed.py`
- `scripts/computational_mode_atlas.py`
- `scripts/statistical_hardening.py`
- `configs/canonical/`
- `src/pipelines/`
- `results/path_patching/path_patching_summary_20260227_080128.json`
- `results/persistent_patching_v3/persistent_patching_v3_dual_20260225_002604.json`
- `results/sufficiency_ladder/sufficiency_ladder_20260225_101907.json`
- `docs/findings/R_V_BEHAVIORAL_DISSOCIATION.md`

If there is a conflict between legacy behavior and the sprint brief, follow the sprint brief and document the change.

## Implementation Priorities

### 1. Collapse to one prompt path
- Patch active scripts to load prompts through `prompts/loader.py`.
- Remove or deprecate inline prompt lists from canonical workflows.
- Ensure prompt-bank version or hash is logged into artifacts.

### 2. Collapse to one metric path
- Make `src/metrics/rv.py` the canonical paper-grade implementation.
- Audit or patch alternate metric call sites so canonical runs do not silently use a second definition.
- Add tests or validation checks where practical.

### 3. Collapse to one layer path
- Ensure canonical runs derive early/late layers from one authoritative source.
- Log chosen layers in `config.json` and `summary.json`.
- Remove magic numbers from paper-grade pipelines where feasible.

### 4. Enforce artifact compliance
- Every canonical run should emit:
  - `config.json`
  - `summary.json`
  - `per_sample.csv`
  - prompt-bank provenance
- Patch scripts that only emit summary fragments or ad hoc JSON.

### 5. Remove hardcoded paper stats
- Find any script or summary generator that injects fixed numbers into paper-facing outputs.
- Replace with raw-to-summary computation.

### 6. Make unit counts explicit
- Wherever `n` is reported, label what it counts.
- Update schemas or summaries if needed so prompts, pairs, sessions, and turns are not mixed.

### 7. Prepare the Mistral acceptance pack
- Create or patch the canonical Mistral run path.
- Regenerate outputs if compute and environment allow.
- If reruns are blocked, still finish the code path and leave an explicit runbook.

## Required Deliverables

1. Code patches that reduce the number of canonical paths to one.
2. A short acceptance report describing:
   - what was changed
   - what is now canonical
   - what remains blocked
3. A raw-to-paper table path:
   - one script or generator that pulls paper numbers from artifacts
4. A deprecation note for exploratory or legacy result families that should not feed the paper.
5. A minimal fan-out runbook that says:
   - "Use the same canonical Mistral machinery on new models"
   - not "invent a fresh script per model"

## Execution Rules

- Prefer small, surgical patches over sweeping rewrites.
- Preserve raw historical outputs; mark them non-canonical rather than deleting them.
- Do not broaden scope into 5-7 model experimentation until Mistral acceptance is materially complete.
- If the repo supports dissociation more strongly than sufficiency, preserve that truth in docs and summaries.
- When blocked by compute, weights, or environment, leave reproducible commands and exact expected outputs.

## Preferred Output Style

Aim for:
- canonical, minimal, reproducible
- file-specific change summaries
- explicit unresolved blockers
- exact commands to rerun

Avoid:
- speculative new theory work
- paper spin
- adding more result families before the existing ones agree

## Final Objective

Leave the repo in a state where this is true:

> The Mistral result can be regenerated from one prompt source, one metric implementation, one layer policy, one artifact schema, and one table-generation path. Only then is cross-architecture fan-out worth doing.
