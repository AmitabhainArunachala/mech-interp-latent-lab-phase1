#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log_step() {
  printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1"
}

log_step "1/6 structured carry ablation on reduced late-only maintainer"
OUTPUT_STEM=structured_text_carry_reduced_late_only_v1 \
QUEUE_GROUP=mistral_structured_text_carry_reduced_late_only_v1 \
EXPERIMENT_ID=structured_text_carry_reduced_late_only_v1 \
CLAIM_ID=STRUCTURED_TEXT_CARRY_REDUCED_LATE_ONLY_V1 \
SOURCE_RUN_DIR=results/anchor_reduced_latebundle_confirm_v1/20260317_132349 \
TARGET_CONDITION=anchor_late_only_bridge_3 \
CONTROL_CONDITION=control \
BASELINE_GROUPS=baseline \
TOP_K_PER_GROUP=8 \
SELECTION_STRATEGY=median \
MAX_TURNS=15 \
bash scripts/runpod_mistral_structured_text_carry_ablation_v1_queue.sh

log_step "2/6 structured carry ablation on reduced drop-L25 maintainer"
OUTPUT_STEM=structured_text_carry_reduced_drop_l25_v1 \
QUEUE_GROUP=mistral_structured_text_carry_reduced_drop_l25_v1 \
EXPERIMENT_ID=structured_text_carry_reduced_drop_l25_v1 \
CLAIM_ID=STRUCTURED_TEXT_CARRY_REDUCED_DROP_L25_V1 \
SOURCE_RUN_DIR=results/anchor_reduced_latebundle_confirm_v1/20260317_132349 \
TARGET_CONDITION=anchor_drop_L25_vproj_bridge_3 \
CONTROL_CONDITION=control \
BASELINE_GROUPS=baseline \
TOP_K_PER_GROUP=8 \
SELECTION_STRATEGY=median \
MAX_TURNS=15 \
bash scripts/runpod_mistral_structured_text_carry_ablation_v1_queue.sh

log_step "3/6 fixed-schedule unselected-seed robustness on reduced drop-L25 maintainer"
OUTPUT_STEM=induced_persistence_unselected_reduced_drop_l25_v1 \
QUEUE_GROUP=mistral_induced_persistence_unselected_reduced_drop_l25_v1 \
EXPERIMENT_ID=induced_persistence_unselected_reduced_drop_l25_v1 \
CLAIM_ID=INDUCED_PERSISTENCE_UNSELECTED_REDUCED_DROP_L25_V1 \
SOURCE_RUN_DIR=results/anchor_reduced_latebundle_confirm_v1/20260317_132349 \
TARGET_CONDITION=anchor_drop_L25_vproj_bridge_3 \
BASELINE_GROUPS=baseline \
SESSIONS_PER_ARM=24 \
MAX_TURNS=15 \
bash scripts/runpod_mistral_induced_persistence_unselected_seed_v1_queue.sh

log_step "4/6 targeted soft break confirm anti-bridge scale 1.0 window 2"
OUTPUT_STEM=mistral_soft_break_targeted_anti_bridge_scale1p0_w2_v1 \
QUEUE_GROUP=mistral_soft_break_targeted_anti_bridge_scale1p0_w2_v1 \
EXPERIMENT_ID=mistral_soft_break_targeted_anti_bridge_scale1p0_w2_v1 \
CLAIM_ID=MISTRAL_SOFT_BREAK_TARGETED_ANTI_BRIDGE_SCALE1P0_W2_V1 \
CONDITION_NAME=anti_bridge_only \
SCALE=1.0 \
TOKEN_WINDOW=2 \
bash scripts/runpod_mistral_soft_break_targeted_confirm_v1_queue.sh

log_step "5/6 targeted soft break confirm anti-bridge scale 1.5 window 2"
OUTPUT_STEM=mistral_soft_break_targeted_anti_bridge_scale1p5_w2_v1 \
QUEUE_GROUP=mistral_soft_break_targeted_anti_bridge_scale1p5_w2_v1 \
EXPERIMENT_ID=mistral_soft_break_targeted_anti_bridge_scale1p5_w2_v1 \
CLAIM_ID=MISTRAL_SOFT_BREAK_TARGETED_ANTI_BRIDGE_SCALE1P5_W2_V1 \
CONDITION_NAME=anti_bridge_only \
SCALE=1.5 \
TOKEN_WINDOW=2 \
bash scripts/runpod_mistral_soft_break_targeted_confirm_v1_queue.sh

log_step "6/6 targeted soft break confirm anti-L25+bridge scale 1.25 window 2"
OUTPUT_STEM=mistral_soft_break_targeted_anti_l25_bridge_scale1p25_w2_v1 \
QUEUE_GROUP=mistral_soft_break_targeted_anti_l25_bridge_scale1p25_w2_v1 \
EXPERIMENT_ID=mistral_soft_break_targeted_anti_l25_bridge_scale1p25_w2_v1 \
CLAIM_ID=MISTRAL_SOFT_BREAK_TARGETED_ANTI_L25_BRIDGE_SCALE1P25_W2_V1 \
CONDITION_NAME=anti_l25_bridge \
SCALE=1.25 \
TOKEN_WINDOW=2 \
bash scripts/runpod_mistral_soft_break_targeted_confirm_v1_queue.sh

log_step "overnight pack complete"
