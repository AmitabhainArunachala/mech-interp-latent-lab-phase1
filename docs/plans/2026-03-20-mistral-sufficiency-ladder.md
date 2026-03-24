# Mistral Reduced-Late Sufficiency Ladder Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Drive the next Mistral-first ladder from the reduced late-stage maintainer toward the remaining three unsolved sufficiency gaps: broad maintenance, longer-horizon retention, and clearer behavioral consequence under structured prompt stress.

**Architecture:** Treat broad induction as provisionally locked by the existing broad-confirm winner, then start the live ladder at the reduced late maintainer because it is the strongest current maintenance object on median seeds. Run the ladder in increasing difficulty: random and low-RV basin probes, long-horizon low-RV continuation, fixed-schedule mixed-prompt robustness, then scaffolded long-horizon `gnani` retention after removing the plotting failure that previously killed the queue.

**Tech Stack:** Bash queue wrappers, Python experiment scripts, RunPod tmux orchestration, Markdown planning docs, AMIROS shared-state logging.

---

### Task 1: Lock The Ladder Stages

**Files:**
- Create: `docs/plans/2026-03-20-mistral-sufficiency-ladder.md`
- Reference: `results/anchor_positive_broad_confirm_v1/summary.json`
- Reference: `results/induced_persistence_reduced_latebundle_confirm_v1/20260317_141750/summary.json`
- Reference: `results/induced_persistence_reduced_latebundle_long_v1/20260317_150042/summary.json`
- Reference: `results/induced_persistence_unselected_seed_v1/20260316_132334/summary.json`

**Step 1: Write the failing test**

Define the paper-facing ladder contract:

- Stage 0: broad induction is already held by `anchor_single_mlp_0p125_layermatched_low_bridge_3`
- Stage 1: reduced-late maintainer must survive broader seed pools (`random`, `low_rv`)
- Stage 2: reduced-late maintainer must survive longer-horizon low-RV continuation (`24` turns)
- Stage 3: reduced-late maintainer must survive a fixed mixed prompt schedule, not just raw continuation
- Stage 4: scaffolded `gnani` must complete cleanly and emit a stable summary artifact

**Step 2: Run test to verify it fails**

Read the current status notes and confirm there is no single execution plan that turns those stages into a live March 20 queue.

**Step 3: Write minimal implementation**

Record the exact queue stages, parameters, success criteria, and artifact paths in this plan.

**Step 4: Run test to verify it passes**

Open the saved plan and confirm every stage has a concrete script path and concrete promotion criterion.

**Step 5: Commit**

```bash
git add docs/plans/2026-03-20-mistral-sufficiency-ladder.md
git commit -m "docs: add mistral reduced-late sufficiency ladder"
```

### Task 2: Remove The `gnani` Plotting Blocker

**Files:**
- Modify: `scripts/sustained_gnani_v3.py`
- Test: `python3 -m py_compile scripts/sustained_gnani_v3.py`

**Step 1: Write the failing test**

Known failure case:

- Run completes metric computation
- `comparison_summary.json` is written
- queue still fails because `matplotlib` is absent when building `convergence_panel.png`

**Step 2: Run test to verify it fails**

Use the existing overnight log:

- `results/mistral_sufficiency_bundle_v2/20260319_110753/sustained_gnani_v3_v2.log`

Expected: `ModuleNotFoundError: No module named 'matplotlib'`

**Step 3: Write minimal implementation**

Make convergence-panel generation optional:

- if `matplotlib` is available, save the figure normally
- if it is missing or plotting fails, keep the run successful and annotate the summary with `convergence_panel_status`

**Step 4: Run test to verify it passes**

Run: `python3 -m py_compile scripts/sustained_gnani_v3.py`

Expected: no syntax errors; figure generation is no longer a hard dependency for queue completion.

**Step 5: Commit**

```bash
git add scripts/sustained_gnani_v3.py
git commit -m "fix: make sustained gnani plotting optional"
```

### Task 3: Add The Reduced-Late Ladder Queue

**Files:**
- Create: `scripts/runpod_mistral_reduced_late_ladder_v1_queue.sh`
- Reference: `scripts/induced_persistence_followup.py`
- Reference: `scripts/induced_persistence_unselected_seed_v1.py`
- Reference: `scripts/sustained_gnani_v3.py`

**Step 1: Write the failing test**

Required queue stages are not yet available as one reproducible launcher:

- `reduced_late_random_12`
- `reduced_late_lowrv_12`
- `reduced_late_lowrv_24`
- `reduced_late_structured_unselected`
- `sustained_gnani_v3_recover`

**Step 2: Run test to verify it fails**

Confirm there is no existing queue script that starts at the reduced late maintainer and walks through all remaining Mistral ladder stages in order.

**Step 3: Write minimal implementation**

Create a single AMIROS-aware queue wrapper that:

- writes stage-local artifacts under one ladder run directory
- upserts each completed stage into the results index
- keeps the Mistral-first ladder in one tmux session on RunPod

**Step 4: Run test to verify it passes**

Run: `bash -n scripts/runpod_mistral_reduced_late_ladder_v1_queue.sh`

Expected: shell parses cleanly.

**Step 5: Commit**

```bash
git add scripts/runpod_mistral_reduced_late_ladder_v1_queue.sh
git commit -m "feat: add reduced-late mistral sufficiency ladder queue"
```

### Task 4: Sync And Launch On RunPod

**Files:**
- Modify remotely via sync: `scripts/sustained_gnani_v3.py`
- Modify remotely via sync: `scripts/runpod_mistral_reduced_late_ladder_v1_queue.sh`
- Modify remotely via sync: `docs/plans/2026-03-20-mistral-sufficiency-ladder.md`

**Step 1: Write the failing test**

The live A100 queue does not yet exist on the RunPod host.

**Step 2: Run test to verify it fails**

Check the host:

- `ssh -p 18633 root@154.54.102.57 'tmux ls 2>/dev/null || true'`

Expected: no ladder session present.

**Step 3: Write minimal implementation**

Sync the changed files, then launch:

```bash
tmux new-session -d -s mistral_reduced_late_ladder_v1 \
  'cd /workspace/mech-interp-latent-lab-phase1 && bash scripts/runpod_mistral_reduced_late_ladder_v1_queue.sh'
```

**Step 4: Run test to verify it passes**

Check:

- tmux session exists
- ladder `STATUS.txt` exists on remote
- GPU utilization becomes nonzero once the first stage starts

**Step 5: Commit**

```bash
git add docs/plans/2026-03-20-mistral-sufficiency-ladder.md scripts/sustained_gnani_v3.py scripts/runpod_mistral_reduced_late_ladder_v1_queue.sh
git commit -m "ops: launch reduced-late mistral sufficiency ladder"
```

### Task 5: Promote Or Refine Based On Checkpoints

**Files:**
- Reference: ladder artifacts under `results/mistral_reduced_late_ladder_v1/`
- Reference: stage artifacts under `results/mistral_reduced_late_ladder_v1_bundle/`

**Step 1: Write the failing test**

Promotion criteria are not yet locked.

**Step 2: Run test to verify it fails**

Confirm there is no explicit decision rule for what counts as a meaningful step toward the remaining gaps.

**Step 3: Write minimal implementation**

Use these gates:

- broad maintenance: `anchor_late_only_bridge_3` or `anchor_drop_L25_vproj_bridge_3` must beat control on `random` and `low_rv`
- longer retention: one reduced-late maintainer must stay above control at `24` turns on `low_rv`
- scaffolded retention: `sustained_gnani_v3` must finish cleanly and preserve recursive-vs-baseline separation in `comparison_summary.json`
- behavioral consequence: structured mixed-schedule arms must show a meaningful gap between `selected` / `unselected` and `random_text` / `cold_start`

**Step 4: Run test to verify it passes**

Record the launched ladder run id and keep using these gates when harvesting results.

**Step 5: Commit**

```bash
git add docs/plans/2026-03-20-mistral-sufficiency-ladder.md
git commit -m "docs: lock promotion gates for mistral sufficiency ladder"
```
