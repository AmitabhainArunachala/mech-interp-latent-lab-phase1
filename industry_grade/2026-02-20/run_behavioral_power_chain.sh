#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

wait_for_sustained() {
  log "Waiting for sustained_gnani_v3 Task-1 process to finish..."
  while ps auxww | grep '[s]cripts/sustained_gnani_v3.py' >/dev/null 2>&1; do
    sleep 60
  done
  log "sustained_gnani_v3 process no longer running."
}

assert_session_targets() {
  python3 - <<'PY'
import glob, json, os

files = [p for p in glob.glob("results/sustained_gnani_v3_fixed/*.json") if os.path.basename(p) != "comparison_summary.json"]
rec = bas = 0
for p in files:
    d = json.load(open(p))
    if d.get("mode") == "recursive":
        rec += 1
    elif d.get("mode") == "baseline":
        bas += 1
print(f"session_counts recursive={rec} baseline={bas}")
if rec < 12 or bas < 8:
    raise SystemExit("Target counts not met yet (need >=12 recursive and >=8 baseline).")
PY
}

wait_for_seed_bridge_idle() {
  log "Waiting for active seed_bridge GPU runs to finish before Task-2..."
  while ps auxww | grep '[s]rc.pipelines.run --config configs/canonical/seed_bridge_2026_02_20' >/dev/null 2>&1; do
    sleep 60
  done
  log "No active seed_bridge runs detected."
}

log "Behavioral power chain started."
wait_for_sustained
assert_session_targets

log "Running within_session_bridge.py"
python3 scripts/within_session_bridge.py

log "Running bridge_battery.py"
python3 scripts/bridge_battery.py

wait_for_seed_bridge_idle

log "Running batch_per_token_rv.py (Task-2)"
KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1 \
python3 scripts/batch_per_token_rv.py \
  --device cuda \
  --n-per-group 25 \
  --max-tokens 256 \
  --temperature 0.7 \
  --seed 20260220 \
  --output results/batch_per_token_rv

log "Running classifier_evaluation.py (Task-3)"
python3 scripts/classifier_evaluation.py

log "Building behavioral summaries (Task-4)"
python3 scripts/build_behavioral_bridge_master.py

log "Behavioral power chain complete."
