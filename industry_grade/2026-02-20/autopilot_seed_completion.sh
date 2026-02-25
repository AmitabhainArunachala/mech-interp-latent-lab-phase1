#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="root@38.80.152.248"
REMOTE_PORT="33595"
SSH_KEY="${HOME}/.ssh/id_ed25519"
REMOTE_REPO="/workspace/mech-interp-latent-lab-phase1"
SYNC_ROOT="$REPO_ROOT/results/remote_gpu_sync/2026-02-20/phase1_mechanism"
LOG_FILE="$REPO_ROOT/industry_grade/2026-02-20/evidence/autopilot_seed_completion.log"

mkdir -p "$SYNC_ROOT"
touch "$LOG_FILE"

required_run_names=(
  "seed_bridge_20260220_baseline_donor_control_s123_n80"
  "seed_bridge_20260220_random_head_control_s456_n80"
  "seed_bridge_20260220_baseline_donor_control_s456_n80"
)

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] autopilot start" | tee -a "$LOG_FILE"

found_all=0
while [[ "$found_all" -eq 0 ]]; do
  found_all=1
  for run_name in "${required_run_names[@]}"; do
    remote_dir="$(
      ssh -p "$REMOTE_PORT" -i "$SSH_KEY" "$REMOTE_HOST" \
        "ls -1dt $REMOTE_REPO/results/phase1_mechanism/runs/*_${run_name} 2>/dev/null | head -n 1" || true
    )"
    if [[ -z "$remote_dir" ]]; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] pending run_dir $run_name" | tee -a "$LOG_FILE"
      found_all=0
      continue
    fi

    has_summary="$(
      ssh -p "$REMOTE_PORT" -i "$SSH_KEY" "$REMOTE_HOST" \
        "if [ -f '$remote_dir/summary.json' ]; then echo yes; else echo no; fi"
    )"
    if [[ "$has_summary" != "yes" ]]; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] pending summary $run_name" | tee -a "$LOG_FILE"
      found_all=0
      continue
    fi

    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] sync $run_name <- $remote_dir" | tee -a "$LOG_FILE"
    scp -r -P "$REMOTE_PORT" -i "$SSH_KEY" "$REMOTE_HOST:$remote_dir" "$SYNC_ROOT/" >/dev/null 2>&1 || true
  done

  if [[ "$found_all" -eq 0 ]]; then
    sleep 120
  fi
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] required runs complete; refreshing analysis" | tee -a "$LOG_FILE"

python3 "$REPO_ROOT/industry_grade/2026-02-20/analyze_seed_bridge_matrix.py" >>"$LOG_FILE" 2>&1 || true

if [[ -f "$REPO_ROOT/industry_grade/2026-02-20/analyze_semantic_behavior.py" ]]; then
  KMP_DUPLICATE_LIB_OK=TRUE python3 "$REPO_ROOT/industry_grade/2026-02-20/analyze_semantic_behavior.py" >>"$LOG_FILE" 2>&1 || true
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] autopilot done" | tee -a "$LOG_FILE"
