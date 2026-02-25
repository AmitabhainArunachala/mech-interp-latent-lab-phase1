#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="root@38.80.152.248"
REMOTE_PORT="33595"
SSH_KEY="${HOME}/.ssh/id_ed25519"
REMOTE_REPO="/workspace/mech-interp-latent-lab-phase1"

LOCAL_SYNC_ROOT="$REPO_ROOT/results/remote_gpu_sync/2026-02-20/phase1_mechanism"
mkdir -p "$LOCAL_SYNC_ROOT"

MATRIX="$REPO_ROOT/configs/canonical/seed_bridge_2026_02_20/RUN_MATRIX.csv"

cd "$REPO_ROOT"

while IFS=, read -r run_id condition seed config_path; do
  if [[ "$run_id" == "run_id" ]]; then
    continue
  fi

  run_name="$(python3 - <<'PY' "$config_path"
import json,sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    cfg=json.load(f)
print(cfg['run_name'])
PY
)"

  remote_match="$(ssh -p "$REMOTE_PORT" -i "$SSH_KEY" "$REMOTE_HOST" "ls -1dt $REMOTE_REPO/results/phase1_mechanism/runs/*_${run_name} 2>/dev/null | head -n 1" || true)"

  if [[ -z "$remote_match" ]]; then
    echo "[WARN] No run directory yet for $run_id ($run_name)"
    continue
  fi

  echo "[SYNC] $run_id <- $remote_match"
  scp -r -P "$REMOTE_PORT" -i "$SSH_KEY" "$REMOTE_HOST:$remote_match" "$LOCAL_SYNC_ROOT/"
done < "$MATRIX"

echo "Running local seed analysis..."
python3 "$REPO_ROOT/industry_grade/2026-02-20/analyze_seed_bridge_matrix.py"
if [[ -f "$REPO_ROOT/industry_grade/2026-02-20/analyze_semantic_behavior.py" ]]; then
  echo "Running semantic behavior analysis..."
  KMP_DUPLICATE_LIB_OK=TRUE python3 "$REPO_ROOT/industry_grade/2026-02-20/analyze_semantic_behavior.py" || true
fi

echo "Done."
