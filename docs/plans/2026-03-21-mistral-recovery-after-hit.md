# Mistral Recovery-After-Hit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add and launch the next high-ROI Mistral sufficiency experiment: seeded maintenance with a mid-rollout anti-late hit and a measured recovery phase.

**Architecture:** Reuse the fixed-schedule seeded persistence battery for arm selection and session summaries, and reuse the targeted soft-break hook definitions for the injected anti-late burst. Compare maintain-only, hit-then-off, and hit-then-resume schedules so we can tell recovery apart from simply leaving steering on.

**Tech Stack:** Python experiment script, pytest helper tests, bash RunPod queue wrappers, AMIROS result indexing.

---

### Task 1: Lock The Recovery Protocol

**Files:**
- Create: `docs/plans/2026-03-21-mistral-recovery-after-hit.md`
- Create: `tests/test_mistral_recovery_after_hit_v1.py`
- Create: `scripts/mistral_recovery_after_hit_v1.py`

**Step 1: Write the failing test**

Define the contract for:
- the per-turn action schedule
- the `pre_hit / hit / post_hit` segmentation
- the verdict fields that compare resumed recovery against hit-then-off

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_mistral_recovery_after_hit_v1.py -q`

Expected: import failure because the recovery script does not exist yet.

**Step 3: Write minimal implementation**

Add:
- `choose_action(...)`
- `make_recovery_segments(...)`
- `build_recovery_verdict(...)`
- the seeded mixed-schedule recovery experiment

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_mistral_recovery_after_hit_v1.py -q`

Expected: PASS.

### Task 2: Add RunPod Launchers

**Files:**
- Create: `scripts/runpod_mistral_recovery_after_hit_v1_queue.sh`
- Create: `scripts/runpod_mistral_recovery_after_hit_pack_v1_queue.sh`

**Step 1: Write the failing test**

The recovery experiment is not yet launchable as a queue.

**Step 2: Run test to verify it fails**

Confirm there is no queue wrapper for a seeded recovery-after-hit run.

**Step 3: Write minimal implementation**

Add:
- one generic queue wrapper for a single maintainer
- one two-step pack wrapper for `late_only` then `drop_L25`

**Step 4: Run test to verify it passes**

Run:
- `bash -n scripts/runpod_mistral_recovery_after_hit_v1_queue.sh`
- `bash -n scripts/runpod_mistral_recovery_after_hit_pack_v1_queue.sh`

Expected: both parse cleanly.

### Task 3: Verify And Launch

**Files:**
- Modify remotely via sync: `scripts/mistral_recovery_after_hit_v1.py`
- Modify remotely via sync: `scripts/runpod_mistral_recovery_after_hit_v1_queue.sh`
- Modify remotely via sync: `scripts/runpod_mistral_recovery_after_hit_pack_v1_queue.sh`

**Step 1: Run local verification**

Run:
- `pytest tests/test_mistral_recovery_after_hit_v1.py -q`
- `python3 -m py_compile scripts/mistral_recovery_after_hit_v1.py`

**Step 2: Launch**

On the idle Blackwell pod, run the pack queue in tmux.

**Step 3: Confirm**

Check:
- tmux session exists
- `STATUS.txt` exists
- GPU utilization becomes nonzero
