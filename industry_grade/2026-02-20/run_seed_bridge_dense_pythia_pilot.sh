#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MATRIX="$REPO_ROOT/configs/canonical/seed_bridge_dense_pythia_2026_02_20/RUN_MATRIX.csv"
LOG_DIR="$REPO_ROOT/industry_grade/2026-02-20/evidence/seed_bridge_dense_pythia_logs"
mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"

while IFS=, read -r run_id condition seed config_path; do
  if [[ "$run_id" == "run_id" ]]; then
    continue
  fi

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RUN $run_id ($condition seed=$seed)"
  LOG_FILE="$LOG_DIR/${run_id}.log"

  KMP_DUPLICATE_LIB_OK=TRUE \
  PYTHONUNBUFFERED=1 \
  python3 -m src.pipelines.run --config "$config_path" \
    | tee "$LOG_FILE"
done < "$MATRIX"

echo "Completed dense (Pythia) pilot seed bridge runs."
