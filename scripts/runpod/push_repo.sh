#!/usr/bin/env bash
set -euo pipefail

# Sync local repo -> RunPod (fast, incremental).
#
# Usage (on your LOCAL machine):
#   RUNPOD_HOST=198.13.252.9 RUNPOD_PORT=18147 bash scripts/runpod/push_repo.sh
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

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "[push] local:  ${ROOT_DIR}"
echo "[push] remote: ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_DIR} (port ${RUNPOD_PORT})"

rsync -avz \
  --no-owner --no-group \
  --delete \
  --exclude ".venv/" \
  --exclude "__pycache__/" \
  --exclude "**/__pycache__/" \
  --exclude ".git/" \
  -e "ssh -p ${RUNPOD_PORT} -i ${RUNPOD_KEY}" \
  "${ROOT_DIR}/" \
  "${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_DIR}/"

echo "[push][ok]"


