#!/usr/bin/env bash
set -euo pipefail

LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_REPO="${REMOTE_REPO:-/workspace/mech-interp-latent-lab-phase1}"
REMOTE_HOST="${RUNPOD_HOST:?set RUNPOD_HOST, e.g. root@198.13.252.23}"
REMOTE_PORT="${RUNPOD_PORT:?set RUNPOD_PORT, e.g. 16717}"
SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_SESSION="${REMOTE_SESSION:-mistral_sufficiency_bundle_v2}"
QUEUE_SCRIPT="${QUEUE_SCRIPT:-scripts/runpod_mistral_sufficiency_bundle_v2_queue.sh}"
STATUS_GLOB="${STATUS_GLOB:-results/mistral_sufficiency_bundle_v2/*}"
SYNC_CODE_FIRST="${SYNC_CODE_FIRST:-1}"
BOOTSTRAP_REMOTE="${BOOTSTRAP_REMOTE:-1}"
PUSH_HF_TOKEN="${PUSH_HF_TOKEN:-1}"
HF_KEYCHAIN_SERVICE="${HF_KEYCHAIN_SERVICE:-huggingface-api-token}"
REMOTE_HF_HOME="${REMOTE_HF_HOME:-/workspace/hf_cache}"
REMOTE_TMPDIR="${REMOTE_TMPDIR:-/workspace/tmp}"
REMOTE_HF_ETAG_TIMEOUT="${REMOTE_HF_ETAG_TIMEOUT:-30}"
REMOTE_HF_DOWNLOAD_TIMEOUT="${REMOTE_HF_DOWNLOAD_TIMEOUT:-1200}"
DEFAULT_REMOTE_ENV_PREFIX="export TMPDIR=${REMOTE_TMPDIR} HF_HOME=${REMOTE_HF_HOME} TRANSFORMERS_CACHE=${REMOTE_HF_HOME}/transformers HUGGINGFACE_HUB_CACHE=${REMOTE_HF_HOME}/hub HF_HUB_DISABLE_XET=1 HF_HUB_ETAG_TIMEOUT=${REMOTE_HF_ETAG_TIMEOUT} HF_HUB_DOWNLOAD_TIMEOUT=${REMOTE_HF_DOWNLOAD_TIMEOUT} &&"
REMOTE_ENV_PREFIX="${REMOTE_ENV_PREFIX:-$DEFAULT_REMOTE_ENV_PREFIX}"

SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -p "${REMOTE_PORT}"
  -i "${SSH_KEY}"
)

SCP_OPTS=(
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -P "${REMOTE_PORT}"
  -i "${SSH_KEY}"
)

RSYNC_RSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${REMOTE_PORT} -i ${SSH_KEY}"

resolve_hf_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf '%s' "${HF_TOKEN}"
    return 0
  fi
  if command -v security >/dev/null 2>&1; then
    security find-generic-password -a "${USER}" -s "${HF_KEYCHAIN_SERVICE}" -w 2>/dev/null
    return $?
  fi
  return 1
}

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

ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "mkdir -p '${REMOTE_REPO}' '${REMOTE_HF_HOME}/transformers' '${REMOTE_HF_HOME}/hub' '${REMOTE_TMPDIR}' /root/.cache/huggingface"
if [[ "${PUSH_HF_TOKEN}" == "1" ]]; then
  if HF_TOKEN_VALUE="$(resolve_hf_token)"; then
    TOKEN_FILE="$(mktemp)"
    trap 'rm -f "${TOKEN_FILE:-}"' EXIT
    chmod 600 "${TOKEN_FILE}"
    printf '%s' "${HF_TOKEN_VALUE}" > "${TOKEN_FILE}"
    scp "${SCP_OPTS[@]}" "${TOKEN_FILE}" "${REMOTE_HOST}:${REMOTE_HF_HOME}/token"
    ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "install -m 600 '${REMOTE_HF_HOME}/token' /root/.cache/huggingface/token"
    rm -f "${TOKEN_FILE}"
    trap - EXIT
  fi
fi
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
