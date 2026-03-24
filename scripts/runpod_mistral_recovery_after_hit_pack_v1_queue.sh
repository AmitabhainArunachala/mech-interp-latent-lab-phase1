#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log_step() {
  printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1"
}

log_step "1/2 recovery-after-hit on reduced late-only maintainer"
OUTPUT_STEM=mistral_recovery_after_hit_late_only_v1 \
QUEUE_GROUP=mistral_recovery_after_hit_late_only_v1 \
EXPERIMENT_ID=mistral_recovery_after_hit_late_only_v1 \
CLAIM_ID=MISTRAL_RECOVERY_AFTER_HIT_LATE_ONLY_V1 \
TARGET_CONDITION=anchor_late_only_bridge_3 \
SESSIONS_PER_ARM=24 \
bash scripts/runpod_mistral_recovery_after_hit_v1_queue.sh

log_step "2/2 recovery-after-hit on reduced drop-L25 maintainer"
OUTPUT_STEM=mistral_recovery_after_hit_drop_l25_v1 \
QUEUE_GROUP=mistral_recovery_after_hit_drop_l25_v1 \
EXPERIMENT_ID=mistral_recovery_after_hit_drop_l25_v1 \
CLAIM_ID=MISTRAL_RECOVERY_AFTER_HIT_DROP_L25_V1 \
TARGET_CONDITION=anchor_drop_L25_vproj_bridge_3 \
SESSIONS_PER_ARM=24 \
bash scripts/runpod_mistral_recovery_after_hit_v1_queue.sh

log_step "recovery-after-hit pack v1 complete"
