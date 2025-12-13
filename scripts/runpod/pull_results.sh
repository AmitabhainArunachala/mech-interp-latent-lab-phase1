#!/usr/bin/env bash
set -euo pipefail

# Sync remote results -> local (so your paper artifacts are local + backed up).
#
# Usage (on your LOCAL machine):
#   RUNPOD_HOST=198.13.252.9 RUNPOD_PORT=18147 bash scripts/runpod/pull_results.sh
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

mkdir -p "${ROOT_DIR}/results"

echo "[pull] remote: ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_DIR}/results/"
echo "[pull] local:  ${ROOT_DIR}/results/"

rsync -avz \
  --no-owner --no-group \
  -e "ssh -p ${RUNPOD_PORT} -i ${RUNPOD_KEY}" \
  "${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_DIR}/results/" \
  "${ROOT_DIR}/results/"

echo "[pull][ok]"


