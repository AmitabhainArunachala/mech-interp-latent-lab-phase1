#!/bin/bash
set -euo pipefail

LOCAL_REPO="/Users/dhyana/mech-interp-latent-lab-phase1"
REMOTE_REPO="/workspace/mech-interp-latent-lab-phase1"
REMOTE_HOST="root@198.13.252.15"
REMOTE_PORT="13678"
SSH_KEY="${HOME}/.ssh/id_ed25519"
POLL_SECS="${POLL_SECS:-300}"

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

SCAN_GLOB="results/phase1_mechanism/runs/*causal_state_targeted_scan_v1*"
V3_GLOB="results/phase1_mechanism/runs/*causal_state_benchmark_v3_confirmatory*"
V3_CONFIG_LOCAL="${LOCAL_REPO}/configs/canonical/causal_state_benchmark_v3_confirmatory.json"
V3_CONFIG_REMOTE="${REMOTE_REPO}/configs/canonical/causal_state_benchmark_v3_confirmatory.json"
BRIDGE_GLOB="results/phase1_cross_architecture/runs/*mistral_multi_token_bridge_confirmatory*"
BRIDGE_CONFIG_LOCAL="${LOCAL_REPO}/configs/canonical/multi_token_bridge_mistral_confirmatory.json"
BRIDGE_CONFIG_REMOTE="${REMOTE_REPO}/configs/canonical/multi_token_bridge_mistral_confirmatory.json"

ssh_remote() {
  ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "$@"
}

scp_pull() {
  local remote_path="$1"
  local local_path="$2"
  mkdir -p "$(dirname "${local_path}")"
  scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${remote_path}" "${local_path}" >/dev/null
}

scp_push() {
  local local_path="$1"
  local remote_path="$2"
  scp "${SCP_OPTS[@]}" "${local_path}" "${REMOTE_HOST}:${remote_path}" >/dev/null
}

latest_run_dir() {
  local glob="$1"
  ssh_remote "cd ${REMOTE_REPO} && ls -td ${glob} 2>/dev/null | head -n 1" || true
}

wait_for_summary() {
  local glob="$1"
  local label="$2"
  local run_dir
  while true; do
    run_dir="$(latest_run_dir "${glob}")"
    if [[ -n "${run_dir}" ]] && ssh_remote "[ -f '${run_dir}/summary.json' ]"; then
      echo "${run_dir}"
      return 0
    fi
    echo "[autopilot] waiting for ${label} summary..."
    sleep "${POLL_SECS}"
  done
}

launch_remote_config() {
  local config_path="$1"
  local log_stem="$2"
  ssh_remote "cd ${REMOTE_REPO} && ts=\$(date +%Y%m%d_%H%M%S) && logfile=\"industry_grade/2026-02-20/evidence/${log_stem}_\${ts}.remote.log\" && nohup ./.venv/bin/python -m src.pipelines.run --config ${config_path} > \"\$logfile\" 2>&1 < /dev/null & pid=\$! && echo \"PID=\$pid LOG=\$logfile\""
}

scan_run_dir="$(wait_for_summary "${SCAN_GLOB}" "targeted scan")"
scan_local_dir="${LOCAL_REPO}/${scan_run_dir#${REMOTE_REPO}/}"
echo "[autopilot] targeted scan complete: ${scan_run_dir}"

scp_pull "${scan_run_dir}/summary.json" "${scan_local_dir}/summary.json"
scp_pull "${scan_run_dir}/best_candidate.json" "${scan_local_dir}/best_candidate.json"
scp_pull "${scan_run_dir}/candidate_scores.json" "${scan_local_dir}/candidate_scores.json"

did_launch_v3="false"
if python3 "${LOCAL_REPO}/scripts/build_confirmatory_v3_config.py" \
  --best-candidate "${scan_local_dir}/best_candidate.json" \
  --base-config "${LOCAL_REPO}/configs/canonical/causal_state_benchmark_v2.json" \
  --output "${V3_CONFIG_LOCAL}"; then
  echo "[autopilot] confirmatory v3 config written: ${V3_CONFIG_LOCAL}"
  did_launch_v3="true"
else
  echo "[autopilot] promotion threshold not met; skipping v3 confirmatory launch."
fi

if [[ "${did_launch_v3}" == "true" ]]; then
  scp_push "${V3_CONFIG_LOCAL}" "${V3_CONFIG_REMOTE}"

  launch_info="$(launch_remote_config "configs/canonical/causal_state_benchmark_v3_confirmatory.json" "causal_state_benchmark_v3_confirmatory")"
  echo "[autopilot] launched confirmatory v3: ${launch_info}"

  v3_run_dir="$(wait_for_summary "${V3_GLOB}" "confirmatory v3")"
  v3_local_dir="${LOCAL_REPO}/${v3_run_dir#${REMOTE_REPO}/}"
  echo "[autopilot] confirmatory v3 complete: ${v3_run_dir}"

  scp_pull "${v3_run_dir}/summary.json" "${v3_local_dir}/summary.json"
  scp_pull "${v3_run_dir}/benchmark_records.jsonl" "${v3_local_dir}/benchmark_records.jsonl"
  scp_pull "${v3_run_dir}/blind_ratings.csv" "${v3_local_dir}/blind_ratings.csv"
  scp_pull "${v3_run_dir}/blind_key.json" "${v3_local_dir}/blind_key.json"
fi

scp_push "${BRIDGE_CONFIG_LOCAL}" "${BRIDGE_CONFIG_REMOTE}"
bridge_launch_info="$(launch_remote_config "configs/canonical/multi_token_bridge_mistral_confirmatory.json" "multi_token_bridge_mistral_confirmatory")"
echo "[autopilot] launched multi-token bridge: ${bridge_launch_info}"

bridge_run_dir="$(wait_for_summary "${BRIDGE_GLOB}" "multi-token bridge")"
bridge_local_dir="${LOCAL_REPO}/${bridge_run_dir#${REMOTE_REPO}/}"
echo "[autopilot] multi-token bridge complete: ${bridge_run_dir}"

scp_pull "${bridge_run_dir}/summary.json" "${bridge_local_dir}/summary.json"
scp_pull "${bridge_run_dir}/rv_behavioral_correlation.csv" "${bridge_local_dir}/rv_behavioral_correlation.csv"

echo "[autopilot] artifacts pulled locally."
