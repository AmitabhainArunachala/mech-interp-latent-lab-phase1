# Power Prompt: Mistral Strengthen Pass

Date: 2026-03-11
Audience: Codex / strong code-and-research agent
Mode: high-autonomy, evidence-first, no fluff

## Mission

Spend the next 2-4 hours turning the current Mistral hardening state into the strongest defensible paper narrative and experiment stack possible.

The current objective is not "find any positive result." It is:

1. lock the strongest honest Mistral causal story,
2. remove avoidable measurement and reporting ambiguity,
3. stress every relevant local tool or script that can strengthen the paper,
4. produce artifacts and docs that let Dhyana immediately see what is real, what is caveated, and what should happen next.

## Current Ground Truth

These are the authoritative current artifacts:

- full hardened dual-layer rerun:
  - `results/persistent_patching_v3/persistent_patching_v3_dual_20260310_204100.json`
- medium validation:
  - `results/persistent_patching_v3/persistent_patching_v3_dual_20260310_194619.json`
- smoke validation:
  - `results/persistent_patching_v3/persistent_patching_v3_dual_20260310_193713.json`
- RunPod sync report:
  - `results/runpod_sync_report_20260311_151054.md`
- statistical hardening summary:
  - `results/statistical_hardening/hardening_summary_20260311_151203.json`
- generated paper effects table:
  - `R_V_PAPER/generated_table_effects.tex`
- active handoff:
  - `docs/handoffs/2026-03-11_MISTRAL_HARDENING_HANDOFF.md`
- dissociation note:
  - `docs/findings/R_V_BEHAVIORAL_DISSOCIATION.md`
- paper planning spine:
  - `R_V_PAPER/RV_MASTER_PLAN_V2.md`
- briefing:
  - `R_V_PAPER/MORNING_BRIEFING_2026-03-11.md`

## Canonical Interpretation To Preserve

Treat this as the current best-supported Mistral claim unless stronger evidence emerges:

> Recursive self-referential processing in Mistral induces a real late-layer geometric regime, and dual-layer destruction of that regime strongly breaks recursive behavioral output. However, dual-layer geometry injection does not induce clean recursive behavior under the hardened contract. Instead, patched generations collapse into repetitive degeneration. The honest causal framing is necessity plus behavioral dissociation, not geometric sufficiency.

Do not regress to older overclaims.

## Key Current Numbers

From `persistent_patching_v3_dual_20260310_204100.json`:

- `recursive_clean`: `164/300 = 54.7%` BT+ART
- `recursive_dual_patched`: `0/300 = 0.0%`
- `baseline_clean`: `6/300 = 2.0%`
- `baseline_dual_patched`: `0/300 = 0.0%`
- break session effect: `d=4.645`, exact permutation `p=1.6237544450277933e-05`
- `recursive_dual_patched repetitive_rate = 100%`
- `baseline_dual_patched repetitive_rate = 100%`
- `baseline_clean malformed_rate = 5.7%`

Interpretation guardrail:
- the patched conditions show repetitive degeneration with `mean_alpha_ratio ~0.70-0.72`
- the clean-baseline malformed turns appear dominated by arithmetic/markdown formatting and are likely a classifier artifact, not the same pathology
- the saved full artifact still contains pre-fix turn labels; do not treat the 5.7% malformed rate as already recomputed under the newer low-alpha guardrail

## Hard Constraints

- Do not claim geometric sufficiency unless new raw evidence clearly supports it.
- Do not treat old paper numbers as authoritative if they conflict with the latest canonical artifacts.
- Do not quietly hide nulls, sign reversals, or caveats.
- Prefer scripts and raw JSON over prose memory.
- If a tool or script produces a misleading comparison, patch the tool or explicitly caveat its output.

## Highest-ROI Work Order

### Track 1: Measurement cleanup

Goal: remove avoidable artifact inflation and make summaries trustworthy.

1. Audit low-alpha `MALFORMED` labeling in the full rerun.
2. Patch heuristics only if the patch is clearly supported by raw examples.
3. Add tests for any heuristic change.
4. If no safe patch exists, write a clear measurement caveat doc instead.

### Track 2: Reporting cleanup

Goal: make the toolchain tell the honest current story.

1. Run and inspect:
   - `scripts/sync_runpod_results.py`
   - `scripts/generate_paper_tables.py`
   - `scripts/verify_paper_claims.py`
   - `scripts/statistical_hardening.py`
2. Patch any script that is comparing non-equivalent experiment families or using stale assumptions.
3. Save generated reports and link them in the docs.

### Track 3: Paper narrative hardening

Goal: align paper-facing docs with the strongest honest result.

Priority files:
- `R_V_PAPER/MORNING_BRIEFING_2026-03-11.md`
- `R_V_PAPER/RV_MASTER_PLAN_V2.md`
- `docs/findings/R_V_BEHAVIORAL_DISSOCIATION.md`
- `docs/status/COLM_NORTH_STAR_SPRINT_2026-03-10.md` if needed
- `R_V_PAPER/paper_colm2026_v005.tex` or latest paper draft if a safe focused patch is possible

Target narrative:
- phenomenon real
- necessity strong
- induce null
- residual stream dominates path patching
- patched generations often become repetitive rather than recursively articulate
- R_V remains scientifically useful, but the causal paper stance must be narrower and cleaner

### Track 4: Experiment triage

Goal: decide what is worth running next.

Ask:
- Is there any immediate, concrete induce-improvement experiment with high expected value?
- Or is the right move to stop chasing induce and deepen the dissociation story?

Default if uncertain:
- pivot the paper narrative now
- keep induce exploratory and clearly labeled

## What "Testing Every Tool" Means Here

It does not mean random experimentation. It means:

- run every relevant local summary or verification tool that touches Mistral paper claims
- inspect whether each tool is aligned with the latest canonical artifacts
- patch broken or stale tools
- generate reports that reduce ambiguity

Relevant tools to consider:

- `scripts/sync_runpod_results.py`
- `scripts/generate_paper_tables.py`
- `scripts/verify_paper_claims.py`
- `scripts/statistical_hardening.py`
- existing tests under `tests/`

Known current cleanup targets:

- keep `persistent_patching_v3` result selection pinned to the most complete full rerun, not the oldest JSON in the directory
- keep low-alpha arithmetic / markdown answers out of the `MALFORMED` bucket unless they are genuinely token-salad outputs
- treat `verify_paper_claims.py` as a paper-draft provenance checker, not as the canonical latest-results drift detector
- version or explicitly re-run classifier-dependent summaries before replacing canonical BT+ART or malformed rates in saved artifacts

## Deliverables Expected By End Of Pass

Produce as many of these as the evidence supports:

1. updated docs locking the necessity-plus-dissociation narrative
2. any needed classifier/reporting-tool patches plus tests
3. one concise status doc or addendum summarizing the latest Mistral truth state
4. a short list of exact paper claims that must be updated from stale numbers
5. a recommendation on whether to continue induce work or freeze it as exploratory

## Style

- Be blunt, traceable, and source-anchored.
- Use exact file paths and artifact names.
- Prefer "this is supported / this is not supported / this is caveated" wording.
- No motivational language. No inflated claims.

## Success Condition

At the end of the pass, Dhyana should be able to answer:

1. What is the strongest honest Mistral claim now?
2. What exactly changed after the hardened full rerun?
3. Which paper numbers are stale?
4. Which measurement issues are real versus heuristic artifacts?
5. What should we do next with the highest ROI?
