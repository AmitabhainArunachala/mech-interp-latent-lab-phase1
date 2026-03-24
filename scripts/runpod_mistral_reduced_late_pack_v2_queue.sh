#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log_step() {
  printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1"
}

log_step "1/4 fixed-schedule unselected-seed robustness on reduced late-only maintainer"
OUTPUT_STEM=induced_persistence_unselected_reduced_late_only_v1 \
QUEUE_GROUP=mistral_induced_persistence_unselected_reduced_late_only_v1 \
EXPERIMENT_ID=induced_persistence_unselected_reduced_late_only_v1 \
CLAIM_ID=INDUCED_PERSISTENCE_UNSELECTED_REDUCED_LATE_ONLY_V1 \
SOURCE_RUN_DIR=results/anchor_reduced_latebundle_confirm_v1/20260317_132349 \
TARGET_CONDITION=anchor_late_only_bridge_3 \
BASELINE_GROUPS=baseline \
SESSIONS_PER_ARM=24 \
MAX_TURNS=15 \
bash scripts/runpod_mistral_induced_persistence_unselected_seed_v1_queue.sh

log_step "2/4 structured carry ablation on reduced late-only maintainer with random seed selection"
OUTPUT_STEM=structured_text_carry_reduced_late_only_random_v1 \
QUEUE_GROUP=mistral_structured_text_carry_reduced_late_only_random_v1 \
EXPERIMENT_ID=structured_text_carry_reduced_late_only_random_v1 \
CLAIM_ID=STRUCTURED_TEXT_CARRY_REDUCED_LATE_ONLY_RANDOM_V1 \
SOURCE_RUN_DIR=results/anchor_reduced_latebundle_confirm_v1/20260317_132349 \
TARGET_CONDITION=anchor_late_only_bridge_3 \
CONTROL_CONDITION=control \
BASELINE_GROUPS=baseline \
TOP_K_PER_GROUP=12 \
SELECTION_STRATEGY=random \
MAX_TURNS=15 \
bash scripts/runpod_mistral_structured_text_carry_ablation_v1_queue.sh

log_step "3/4 structured carry ablation on reduced drop-L25 maintainer with random seed selection"
OUTPUT_STEM=structured_text_carry_reduced_drop_l25_random_v1 \
QUEUE_GROUP=mistral_structured_text_carry_reduced_drop_l25_random_v1 \
EXPERIMENT_ID=structured_text_carry_reduced_drop_l25_random_v1 \
CLAIM_ID=STRUCTURED_TEXT_CARRY_REDUCED_DROP_L25_RANDOM_V1 \
SOURCE_RUN_DIR=results/anchor_reduced_latebundle_confirm_v1/20260317_132349 \
TARGET_CONDITION=anchor_drop_L25_vproj_bridge_3 \
CONTROL_CONDITION=control \
BASELINE_GROUPS=baseline \
TOP_K_PER_GROUP=12 \
SELECTION_STRATEGY=random \
MAX_TURNS=15 \
bash scripts/runpod_mistral_structured_text_carry_ablation_v1_queue.sh

log_step "4/4 unresolved anti-late candidate sweep"
OUTPUT_STEM=mistral_soft_break_last2_candidate_followup_v1 \
QUEUE_GROUP=mistral_soft_break_last2_candidate_followup_v1 \
EXPERIMENT_ID=mistral_soft_break_last2_candidate_followup_v1 \
CLAIM_ID=MISTRAL_SOFT_BREAK_LAST2_CANDIDATE_FOLLOWUP_V1 \
TOKEN_WINDOW=2 \
SCALES=0.75,1.0,1.25 \
CONDITION_NAMES=anti_l25_only,anti_l27_bridge,anti_late_full \
bash scripts/runpod_mistral_soft_break_last2_candidate_sweep_v1_queue.sh

log_step "reduced late pack v2 complete"
