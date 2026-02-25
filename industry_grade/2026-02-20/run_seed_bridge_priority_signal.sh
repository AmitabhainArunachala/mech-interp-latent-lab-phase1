#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MATRIX="$REPO_ROOT/configs/canonical/seed_bridge_2026_02_20/RUN_MATRIX_PRIORITY.csv"
LOG_DIR="$REPO_ROOT/industry_grade/2026-02-20/evidence/seed_bridge_priority_logs"
STATUS_FILE="$REPO_ROOT/industry_grade/2026-02-20/evidence/seed_bridge_priority_status.csv"
MAX_RETRIES="${MAX_RETRIES:-2}"
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-10800}"

mkdir -p "$LOG_DIR"

if [[ ! -f "$STATUS_FILE" ]]; then
  echo "run_id,condition,seed,status,attempt,timestamp_utc,log_file" > "$STATUS_FILE"
fi

run_already_success() {
  local run_id="$1"
  awk -F, -v id="$run_id" '$1==id && $4=="SUCCESS" {found=1} END{exit(found?0:1)}' "$STATUS_FILE"
}

run_dir_has_summary() {
  local run_name="$1"
  local run_root="$REPO_ROOT/results/phase1_mechanism/runs"
  local latest
  latest="$(ls -1dt "$run_root"/*"_${run_name}" 2>/dev/null | head -n 1 || true)"
  if [[ -n "$latest" && -f "$latest/summary.json" ]]; then
    return 0
  fi
  return 1
}

capture_error_artifact() {
  local run_name="$1"
  local out_file="$2"
  local run_root="$REPO_ROOT/results/phase1_mechanism/runs"
  local latest

  latest="$(ls -1dt "$run_root"/*"_${run_name}" 2>/dev/null | head -n 1 || true)"
  if [[ -n "$latest" && -f "$latest/error.txt" ]]; then
    cp "$latest/error.txt" "$out_file"
  fi
}

cd "$REPO_ROOT"

while IFS=, read -r run_id condition seed config_path; do
  if [[ "$run_id" == "run_id" ]]; then
    continue
  fi

  run_name="$(python3 - <<'PY' "$config_path"
import json,sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    cfg = json.load(f)
print(cfg.get('run_name', 'unknown_run_name'))
PY
)"

  if run_already_success "$run_id"; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP $run_id (status SUCCESS)"
    continue
  fi

  if run_dir_has_summary "$run_name"; then
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[$ts] SKIP $run_id (summary already exists for $run_name)"
    echo "$run_id,$condition,$seed,SUCCESS,0,$ts,existing_summary" >> "$STATUS_FILE"
    continue
  fi

  attempt=1
  success=0
  while (( attempt <= MAX_RETRIES + 1 )); do
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    log_file="$LOG_DIR/${run_id}.attempt${attempt}.log"

    echo "[$ts] RUN $run_id ($condition seed=$seed) attempt=$attempt"

    set +e
    if command -v timeout >/dev/null 2>&1; then
      KMP_DUPLICATE_LIB_OK=TRUE \
      PYTHONUNBUFFERED=1 \
      timeout --signal=TERM "${RUN_TIMEOUT_SEC}" \
        python3 -m src.pipelines.run --config "$config_path" > "$log_file" 2>&1
      rc=$?
    else
      KMP_DUPLICATE_LIB_OK=TRUE \
      PYTHONUNBUFFERED=1 \
      python3 -m src.pipelines.run --config "$config_path" > "$log_file" 2>&1
      rc=$?
    fi
    set -e

    if [[ $rc -eq 0 ]]; then
      echo "$run_id,$condition,$seed,SUCCESS,$attempt,$ts,$log_file" >> "$STATUS_FILE"
      success=1
      break
    fi

    echo "$run_id,$condition,$seed,FAIL,$attempt,$ts,$log_file" >> "$STATUS_FILE"
    capture_error_artifact "$run_name" "$LOG_DIR/${run_id}.attempt${attempt}.error.txt"

    if (( attempt <= MAX_RETRIES )); then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RETRY $run_id after failure rc=$rc"
      sleep 5
    fi

    attempt=$((attempt + 1))
  done

  if [[ $success -ne 1 ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] GAVE_UP $run_id after $((MAX_RETRIES + 1)) attempts"
  fi

done < "$MATRIX"

echo "Completed priority signal pass."
