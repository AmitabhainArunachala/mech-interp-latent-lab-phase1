#!/usr/bin/env bash
set -euo pipefail

# One-shot: SSH into RunPod and run the remote bootstrap.
#
# Usage (on your LOCAL machine):
#   RUNPOD_HOST=198.13.252.9 RUNPOD_PORT=18147 bash scripts/runpod/run_bootstrap_over_ssh.sh
#
# Optional:
#   RUNPOD_USER=root
#   RUNPOD_KEY=~/.ssh/id_ed25519
#   REMOTE_DIR=/workspace/mech-interp-latent-lab-phase1

RUNPOD_HOST="${RUNPOD_HOST:?Set RUNPOD_HOST (e.g. 198.13.252.9)}"
RUNPOD_PORT="${RUNPOD_PORT:?Set RUNPOD_PORT (e.g. 18147)}"
RUNPOD_USER="${RUNPOD_USER:-root}"
RUNPOD_KEY="${RUNPOD_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/mech-interp-latent-lab-phase1}"

echo "[ssh] ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_DIR} (port ${RUNPOD_PORT})"

ssh -p "${RUNPOD_PORT}" -i "${RUNPOD_KEY}" \
  -o ServerAliveInterval=60 \
  -o ServerAliveCountMax=5 \
  "${RUNPOD_USER}@${RUNPOD_HOST}" \
  "cd '${REMOTE_DIR}' && bash scripts/runpod/bootstrap_remote.sh"


