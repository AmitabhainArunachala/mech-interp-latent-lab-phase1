# Mistral Sufficiency Sync And Paper Update Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Repair RunPod harvesting for the March 19 Mistral sufficiency queue, annotate the completed overnight results, and update the COLM draft to a Mistral-first conditional-sufficiency framing.

**Architecture:** Fix the shell-level sync bug in the harvester first, then pull the missing RunPod artifacts into stable local provenance paths. Once the artifacts exist locally, update status documentation and the paper draft to reflect the verified Mistral boundary story: staged induction and maintenance are real, but broad seed-independence is not yet supported.

**Tech Stack:** Bash, rsync/ssh, Python utilities, Markdown status docs, LaTeX paper draft.

---

### Task 1: Repair Harvest Loop

**Files:**
- Modify: `scripts/harvest_runpod_research_os.sh`
- Test: manual sync verification against March 19 artifacts

**Step 1: Write the failing test**

Use the known failing case:

- `results/phase1_cross_architecture/runs/20260319_110757_multi_token_bridge_mistral_7b_bridge_low_trunc_confirmatory_n24/summary.json`
- `results/phase1_cross_architecture/runs/20260319_131310_multi_token_bridge_mistral_7b_bridge_true_longgen_n18/summary.json`
- `results/self_feeding_loop_bundle_v2/20260319_110753/self_feeding_summary_20260319_172423.json`
- `results/mistral_sufficiency_bundle_v2/20260319_110753/STATUS.txt`

Expected before fix: harvest completes but only the first sync target is copied.

**Step 2: Run test to verify it fails**

Run: `RUNPOD_HOST=root@154.54.102.57 RUNPOD_PORT=18633 SSH_KEY=/Users/dhyana/.ssh/id_ed25519 REMOTE_REPO=/workspace/mech-interp-latent-lab-phase1 bash -x scripts/harvest_runpod_research_os.sh`

Expected: trace shows the loop exits after the first target because `ssh`/`rsync` consume loop stdin.

**Step 3: Write minimal implementation**

Change the target iteration so command stdin no longer aliases the target list stream. Use `mapfile` plus a `for` loop, or redirect `ssh`/`rsync` from `/dev/null`.

**Step 4: Run test to verify it passes**

Run:

- `RUNPOD_HOST=root@154.54.102.57 RUNPOD_PORT=18633 SSH_KEY=/Users/dhyana/.ssh/id_ed25519 REMOTE_REPO=/workspace/mech-interp-latent-lab-phase1 bash scripts/harvest_runpod_research_os.sh`
- `python3 - <<'PY' ... verify target files exist ... PY`

Expected: all March 19 Mistral sufficiency artifacts exist locally.

**Step 5: Commit**

```bash
git add scripts/harvest_runpod_research_os.sh
git commit -m "fix: sync all RunPod harvest targets"
```

### Task 2: Annotate Overnight Mistral Outputs

**Files:**
- Modify: `docs/status/AMIROS_STATUS_BOARD.md`
- Modify: `docs/status/NIGHTLY_SUMMARY.md`
- Create: `docs/status/MISTRAL_SUFFICIENCY_NIGHT_2026-03-19.md`

**Step 1: Write the failing test**

Define the required factual annotations:

- queue `mistral_sufficiency_bundle_v2`
- completed experiments: `bridge_low_trunc_confirmatory_n24`, `bridge_true_longgen_n18`, `self_feeding_loop_v2`
- failed/not-completed experiment: `sustained_gnani_v3_v2`
- clear links to local artifact paths

**Step 2: Run test to verify it fails**

Read current docs and confirm they do not yet provide a single stable local annotation page for the March 19 night.

**Step 3: Write minimal implementation**

Add a dedicated status note summarizing:

- exact overnight timeline
- what each completed run tested
- what the results imply for the Mistral sufficiency narrative
- why `sustained_gnani_v3_v2` currently counts as a failed or incomplete lane

Update status board / nightly summary references if needed.

**Step 4: Run test to verify it passes**

Open the docs and confirm each claim points to a local file path that now exists.

**Step 5: Commit**

```bash
git add docs/status/AMIROS_STATUS_BOARD.md docs/status/NIGHTLY_SUMMARY.md docs/status/MISTRAL_SUFFICIENCY_NIGHT_2026-03-19.md
git commit -m "docs: annotate mistral sufficiency overnight results"
```

### Task 3: Update COLM Draft To Mistral-First Boundary Framing

**Files:**
- Modify: `R_V_PAPER/paper_colm2026_v008_0_1.tex`
- Modify: `R_V_PAPER/OPENREVIEW_SUBMISSION_DRAFT.md`
- Reference: `docs/status/MISTRAL_V008_BOUNDARY_UPDATE_2026-03-19.md`
- Reference: `docs/status/MISTRAL_SUFFICIENCY_TRANSITION_PLAN_2026-03-17.md`

**Step 1: Write the failing test**

Identify overstrong draft claims:

- “flat temporal profile”
- “seed-independence validation confirms the selection effect is below 4%”
- any “full sufficiency” implication

**Step 2: Run test to verify it fails**

Search the draft for seed-independence / full-sufficiency wording and compare against the March 19 boundary memo.

**Step 3: Write minimal implementation**

Revise the abstract, introduction, and sufficiency section so the paper says:

- staged induction and maintenance are real in base Mistral
- induction and maintenance dissociate
- maintenance quality depends on seed quality / basin entry state
- best elite-seed maintainer differs from the most seed-stable broad maintainer
- current paper focus is Mistral-first, not a universal cross-model sufficiency claim

**Step 4: Run test to verify it passes**

Run:

- `rg -n "seed-independence|full sufficiency|flat temporal profile" R_V_PAPER/paper_colm2026_v008_0_1.tex`
- `python3 scripts/verify_paper_claims.py`

Expected: disallowed framing removed or softened; verifier still passes or any residual failures are explicitly documented.

**Step 5: Commit**

```bash
git add R_V_PAPER/paper_colm2026_v008_0_1.tex R_V_PAPER/OPENREVIEW_SUBMISSION_DRAFT.md
git commit -m "docs: reframe mistral sufficiency boundary for colm draft"
```

### Task 4: Reconcile Next Mistral-First Focus

**Files:**
- Modify: `docs/status/MISTRAL_SUFFICIENCY_TRANSITION_PLAN_2026-03-17.md` if needed
- Modify: `docs/status/MISTRAL_SUFFICIENCY_NIGHT_2026-03-19.md`

**Step 1: Write the failing test**

Required next-focus statement is not yet explicit enough:

- finish interpreting `anchor_reduced_latebundle_confirm_v1`
- prioritize `induced_persistence_reduced_latebundle_confirm_v1`
- keep the paper centered on base-Mistral staged sufficiency before widening

**Step 2: Run test to verify it fails**

Confirm there is no short, updated action note integrating last night’s results with the Mistral-first transition plan.

**Step 3: Write minimal implementation**

Add a concise “next focus” block that turns the transition plan into concrete March 20 marching orders.

**Step 4: Run test to verify it passes**

Read the updated note and confirm the next-run priority is explicit and paper-aligned.

**Step 5: Commit**

```bash
git add docs/status/MISTRAL_SUFFICIENCY_NIGHT_2026-03-19.md docs/status/MISTRAL_SUFFICIENCY_TRANSITION_PLAN_2026-03-17.md
git commit -m "docs: lock mistral-first sufficiency focus"
```
