#!/bin/bash
set -euo pipefail

LOCAL_REPO="/Users/dhyana/mech-interp-latent-lab-phase1"
REMOTE_REPO="/workspace/mech-interp-latent-lab-phase1"
REMOTE_HOST="root@198.13.252.15"
REMOTE_PORT="13678"
SSH_KEY="${HOME}/.ssh/id_ed25519"
POLL_SECS="${POLL_SECS:-120}"

SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -p "${REMOTE_PORT}"
  -i "${SSH_KEY}"
)

SCP_OPTS=(
  -P "${REMOTE_PORT}"
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -i "${SSH_KEY}"
)

ssh_remote() {
  ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "$@"
}

pull_file() {
  local remote_path="$1"
  local local_path="$2"
  mkdir -p "$(dirname "${local_path}")"
  scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${remote_path}" "${local_path}" >/dev/null
}

latest_match() {
  local glob="$1"
  ssh_remote "cd ${REMOTE_REPO} && ls -td ${glob} 2>/dev/null | head -n 1" || true
}

wait_for_summary() {
  local glob="$1"
  local label="$2"
  local run_dir
  while true; do
    run_dir="$(latest_match "${glob}")"
    if [[ -n "${run_dir}" ]] && ssh_remote "[ -f '${run_dir}/summary.json' ]"; then
      echo "${run_dir}"
      return 0
    fi
    echo "[finalizer] waiting for ${label} summary..."
    sleep "${POLL_SECS}"
  done
}

V3_GLOB="results/phase1_mechanism/runs/*mistral_l25_w32_a3_state_benchmark_v3_confirmatory*"
BRIDGE_GLOB="results/phase1_cross_architecture/runs/*mistral_multi_token_bridge_confirmatory*"

v3_run_dir="$(wait_for_summary "${V3_GLOB}" "confirmatory v3")"
v3_local_dir="${LOCAL_REPO}/${v3_run_dir#${REMOTE_REPO}/}"
echo "[finalizer] v3 finished: ${v3_run_dir}"

pull_file "${v3_run_dir}/summary.json" "${v3_local_dir}/summary.json"
pull_file "${v3_run_dir}/benchmark_records.jsonl" "${v3_local_dir}/benchmark_records.jsonl"
pull_file "${v3_run_dir}/blind_ratings.csv" "${v3_local_dir}/blind_ratings.csv"
pull_file "${v3_run_dir}/blind_key.json" "${v3_local_dir}/blind_key.json"
echo "[finalizer] pulled v3 artifacts"

bridge_run_dir="$(wait_for_summary "${BRIDGE_GLOB}" "multi-token bridge")"
bridge_local_dir="${LOCAL_REPO}/${bridge_run_dir#${REMOTE_REPO}/}"
echo "[finalizer] bridge finished: ${bridge_run_dir}"

pull_file "${bridge_run_dir}/summary.json" "${bridge_local_dir}/summary.json"
pull_file "${bridge_run_dir}/rv_behavioral_correlation.csv" "${bridge_local_dir}/rv_behavioral_correlation.csv"
echo "[finalizer] pulled multi-token artifacts"
