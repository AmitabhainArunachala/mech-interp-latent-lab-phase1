#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
POLL_SECONDS="${POLL_SECONDS:-300}"
ONCE="${ONCE:-0}"
INVENTORY_PATH="${INVENTORY_PATH:-$REPO_ROOT/configs/experiment_registry/pod_inventory.json}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/results/amiros_operator}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/operator_$(date +%Y%m%d).log"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_FILE"
}

POD_LINES=()
while IFS= read -r line; do
  POD_LINES+=("$line")
done < <("$PYTHON_BIN" - "$INVENTORY_PATH" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
for pod in payload.get("pods", []):
    if pod.get("enabled", True):
        print("|".join([
            pod["pod_name"],
            pod["host"],
            str(pod["port"]),
            pod.get("remote_repo", "/workspace/mech-interp-latent-lab-phase1"),
            pod.get("role", ""),
        ]))
PY
)

harvest_pod() {
  local pod_name="$1"
  local host="$2"
  local port="$3"
  local remote_repo="$4"
  if ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$port" -i "$SSH_KEY" "$host" "test -d '$remote_repo'" >/dev/null 2>&1; then
    RUNPOD_HOST="$host" RUNPOD_PORT="$port" SSH_KEY="$SSH_KEY" REMOTE_REPO="$remote_repo" \
      bash "$REPO_ROOT/scripts/harvest_runpod_research_os.sh" >>"$LOG_FILE" 2>&1 || true
    log "harvested $pod_name"
  else
    log "skip harvest $pod_name: remote repo unavailable"
  fi
}

probe_pod() {
  local pod_name="$1"
  local host="$2"
  local port="$3"
  local remote_repo="$4"
  ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$port" -i "$SSH_KEY" "$host" \
    "cd '$remote_repo' 2>/dev/null && nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader && printf '\n---TMUX---\n' && tmux ls 2>/dev/null || true" \
    >>"$LOG_FILE" 2>&1 || log "probe failed for $pod_name"
}

run_cycle() {
  log "operator cycle start"
  for line in "${POD_LINES[@]}"; do
    IFS='|' read -r pod_name host port remote_repo role <<<"$line"
    log "pod=$pod_name role=$role"
    harvest_pod "$pod_name" "$host" "$port" "$remote_repo"
    probe_pod "$pod_name" "$host" "$port" "$remote_repo"
  done
  "$PYTHON_BIN" "$REPO_ROOT/scripts/nightly_summary.py" >>"$LOG_FILE" 2>&1 || true
  log "operator cycle complete"
}

while true; do
  run_cycle
  if [[ "$ONCE" == "1" ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done
