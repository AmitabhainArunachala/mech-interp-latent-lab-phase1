#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log_step() {
  printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1"
}

log_step "1/3 targeted soft break confirm anti-late-full scale 1.25 window 2"
OUTPUT_STEM=mistral_soft_break_targeted_anti_late_full_scale1p25_w2_v1 \
QUEUE_GROUP=mistral_soft_break_targeted_anti_late_full_scale1p25_w2_v1 \
EXPERIMENT_ID=mistral_soft_break_targeted_anti_late_full_scale1p25_w2_v1 \
CLAIM_ID=MISTRAL_SOFT_BREAK_TARGETED_ANTI_LATE_FULL_SCALE1P25_W2_V1 \
CONDITION_NAME=anti_late_full \
SCALE=1.25 \
TOKEN_WINDOW=2 \
bash scripts/runpod_mistral_soft_break_targeted_confirm_v1_queue.sh

log_step "2/3 targeted soft break confirm anti-late-full scale 1.0 window 2"
OUTPUT_STEM=mistral_soft_break_targeted_anti_late_full_scale1p0_w2_v1 \
QUEUE_GROUP=mistral_soft_break_targeted_anti_late_full_scale1p0_w2_v1 \
EXPERIMENT_ID=mistral_soft_break_targeted_anti_late_full_scale1p0_w2_v1 \
CLAIM_ID=MISTRAL_SOFT_BREAK_TARGETED_ANTI_LATE_FULL_SCALE1P0_W2_V1 \
CONDITION_NAME=anti_late_full \
SCALE=1.0 \
TOKEN_WINDOW=2 \
bash scripts/runpod_mistral_soft_break_targeted_confirm_v1_queue.sh

log_step "3/3 targeted soft break confirm anti-L27+bridge scale 1.25 window 2"
OUTPUT_STEM=mistral_soft_break_targeted_anti_l27_bridge_scale1p25_w2_v1 \
QUEUE_GROUP=mistral_soft_break_targeted_anti_l27_bridge_scale1p25_w2_v1 \
EXPERIMENT_ID=mistral_soft_break_targeted_anti_l27_bridge_scale1p25_w2_v1 \
CLAIM_ID=MISTRAL_SOFT_BREAK_TARGETED_ANTI_L27_BRIDGE_SCALE1P25_W2_V1 \
CONDITION_NAME=anti_l27_bridge \
SCALE=1.25 \
TOKEN_WINDOW=2 \
bash scripts/runpod_mistral_soft_break_targeted_confirm_v1_queue.sh

log_step "soft break followup v2 complete"
