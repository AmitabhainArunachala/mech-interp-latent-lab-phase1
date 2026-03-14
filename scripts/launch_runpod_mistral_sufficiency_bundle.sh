#!/usr/bin/env bash
set -euo pipefail

LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_REPO="${REMOTE_REPO:-/workspace/mech-interp-latent-lab-phase1}"
REMOTE_HOST="${RUNPOD_HOST:?set RUNPOD_HOST, e.g. root@198.13.252.23}"
REMOTE_PORT="${RUNPOD_PORT:?set RUNPOD_PORT, e.g. 16717}"
SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_SESSION="${REMOTE_SESSION:-mistral_sufficiency_bundle}"
QUEUE_SCRIPT="${QUEUE_SCRIPT:-scripts/runpod_mistral_sufficiency_bundle_queue.sh}"
STATUS_GLOB="${STATUS_GLOB:-results/mistral_sufficiency_bundle/*}"
SYNC_CODE_FIRST="${SYNC_CODE_FIRST:-1}"
BOOTSTRAP_REMOTE="${BOOTSTRAP_REMOTE:-1}"
REMOTE_ENV_PREFIX="${REMOTE_ENV_PREFIX:-}"

SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -p "${REMOTE_PORT}"
  -i "${SSH_KEY}"
)

RSYNC_RSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${REMOTE_PORT} -i ${SSH_KEY}"

if [[ "${SYNC_CODE_FIRST}" == "1" ]]; then
  rsync -az \
    -e "${RSYNC_RSH}" \
    --no-owner \
    --no-group \
    --exclude ".git" \
    --exclude ".venv" \
    --exclude "__pycache__" \
    --exclude ".pytest_cache" \
    --exclude "results" \
    --exclude "node_modules" \
    "${LOCAL_REPO}/" "${REMOTE_HOST}:${REMOTE_REPO}/"
fi

ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "mkdir -p '${REMOTE_REPO}'"
if [[ "${BOOTSTRAP_REMOTE}" == "1" ]]; then
  ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "cd '${REMOTE_REPO}' && python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q torch transformers accelerate numpy scipy pandas tqdm matplotlib seaborn scikit-learn sentencepiece safetensors"
fi
ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "tmux kill-session -t '${REMOTE_SESSION}' 2>/dev/null || true"
ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" \
  "tmux new-session -d -s '${REMOTE_SESSION}' 'cd ${REMOTE_REPO} && ${REMOTE_ENV_PREFIX} bash ${QUEUE_SCRIPT}'"

echo "launched_tmux_session=${REMOTE_SESSION}"
echo "remote_repo=${REMOTE_REPO}"
echo "monitor_command=ssh -p ${REMOTE_PORT} -i ${SSH_KEY} ${REMOTE_HOST} \"tmux capture-pane -pt ${REMOTE_SESSION}:0 | tail -n 80\""
echo "status_command=ssh -p ${REMOTE_PORT} -i ${SSH_KEY} ${REMOTE_HOST} \"cd ${REMOTE_REPO} && ls -1dt ${STATUS_GLOB} | head -n 1 | xargs -I{} cat {}/STATUS.txt\""
