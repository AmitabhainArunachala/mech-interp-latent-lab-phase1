#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUNPOD_HOST:-}" ]]; then
  echo "RUNPOD_HOST is required, e.g. root@213.173.102.102" >&2
  exit 1
fi

RUNPOD_PORT="${RUNPOD_PORT:-22}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$RUNPOD_PORT" -i "$SSH_KEY")
REMOTE_REPO="${REMOTE_REPO:-/workspace/mech-interp-latent-lab-phase1}"

ssh "${SSH_OPTS[@]}" "$RUNPOD_HOST" "cd '$REMOTE_REPO' && python3 scripts/nightly_summary.py >/dev/null"

mkdir -p "$REPO_ROOT/configs/experiment_registry" "$REPO_ROOT/docs/status" "$REPO_ROOT/results"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

rsync -az --no-owner --no-group -e "ssh ${SSH_OPTS[*]}" \
  "$RUNPOD_HOST:$REMOTE_REPO/configs/experiment_registry/pod_leases.json" \
  "$tmp_dir/pod_leases.remote.json"

rsync -az --no-owner --no-group -e "ssh ${SSH_OPTS[*]}" \
  "$RUNPOD_HOST:$REMOTE_REPO/configs/experiment_registry/results_index.json" \
  "$tmp_dir/results_index.remote.json"

if ssh "${SSH_OPTS[@]}" "$RUNPOD_HOST" "test -f '$REMOTE_REPO/docs/status/AMIROS_STATUS_BOARD.md'"; then
  rsync -az --no-owner --no-group -e "ssh ${SSH_OPTS[*]}" \
    "$RUNPOD_HOST:$REMOTE_REPO/docs/status/AMIROS_STATUS_BOARD.md" \
    "$REPO_ROOT/docs/status/AMIROS_STATUS_BOARD.remote.md"
else
  rsync -az --no-owner --no-group -e "ssh ${SSH_OPTS[*]}" \
    "$RUNPOD_HOST:$REMOTE_REPO/docs/status/NIGHTLY_SUMMARY.md" \
    "$REPO_ROOT/docs/status/AMIROS_STATUS_BOARD.remote.md"
fi

python3 - "$REPO_ROOT" "$tmp_dir/pod_leases.remote.json" "$tmp_dir/results_index.remote.json" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
remote_leases_path = Path(sys.argv[2])
remote_results_path = Path(sys.argv[3])
local_leases_path = repo_root / "configs" / "experiment_registry" / "pod_leases.json"
local_results_path = repo_root / "configs" / "experiment_registry" / "results_index.json"

def load(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))

def dump(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

local_leases = load(local_leases_path, {"updated_at": "", "leases": []})
remote_leases = load(remote_leases_path, {"updated_at": "", "leases": []})
lease_map = {lease["pod_name"]: lease for lease in local_leases.get("leases", [])}
for lease in remote_leases.get("leases", []):
    lease_map[lease["pod_name"]] = lease
local_leases["leases"] = sorted(lease_map.values(), key=lambda x: x["pod_name"])
local_leases["updated_at"] = remote_leases.get("updated_at") or local_leases.get("updated_at")
dump(local_leases_path, local_leases)

local_results = load(local_results_path, {"updated_at": "", "results": []})
remote_results = load(remote_results_path, {"updated_at": "", "results": []})
result_map = {
    (r.get("run_id"), r.get("experiment_id")): r
    for r in local_results.get("results", [])
}
for result in remote_results.get("results", []):
    result_map[(result.get("run_id"), result.get("experiment_id"))] = result
local_results["results"] = sorted(
    result_map.values(),
    key=lambda x: (x.get("updated_at", ""), x.get("experiment_id", "")),
)
local_results["updated_at"] = remote_results.get("updated_at") or local_results.get("updated_at")
dump(local_results_path, local_results)
PY

python3 - "$REPO_ROOT" "$tmp_dir/sync_targets.txt" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
targets_path = Path(sys.argv[2])
results_index = json.loads((repo_root / "configs" / "experiment_registry" / "results_index.json").read_text(encoding="utf-8"))
pod_leases = json.loads((repo_root / "configs" / "experiment_registry" / "pod_leases.json").read_text(encoding="utf-8"))

targets: set[str] = set()

for lease in pod_leases.get("leases", []):
    out_dir = str(lease.get("out_dir") or "").strip()
    if out_dir.startswith("results/"):
        targets.add(out_dir.rstrip("/"))

for result in results_index.get("results", []):
    artifact_path = str(result.get("artifact_path") or "").strip()
    if not artifact_path or any(ch in artifact_path for ch in "*?[]"):
        continue
    if not artifact_path.startswith("results/"):
        continue
    artifact = Path(artifact_path)
    if artifact.suffix:
        parent = artifact.parent
        if "runs" in parent.parts or parent.name.startswith("202"):
            targets.add(str(parent))
        else:
            targets.add(artifact_path)
    else:
        targets.add(artifact_path)

targets_path.write_text(
    "\n".join(sorted(targets)) + ("\n" if targets else ""),
    encoding="utf-8",
)
PY

while IFS= read -r target <&3; do
  [[ -z "$target" ]] && continue
  remote_path="$REMOTE_REPO/$target"
  local_path="$REPO_ROOT/$target"
  if ssh "${SSH_OPTS[@]}" "$RUNPOD_HOST" "test -d '$remote_path'" < /dev/null; then
    mkdir -p "$local_path"
    rsync -az --no-owner --no-group -e "ssh ${SSH_OPTS[*]}" \
      "$RUNPOD_HOST:$remote_path/" \
      "$local_path/" < /dev/null
  elif ssh "${SSH_OPTS[@]}" "$RUNPOD_HOST" "test -f '$remote_path'" < /dev/null; then
    mkdir -p "$(dirname "$local_path")"
    rsync -az --no-owner --no-group -e "ssh ${SSH_OPTS[*]}" \
      "$RUNPOD_HOST:$remote_path" \
      "$local_path" < /dev/null
  fi
done 3< "$tmp_dir/sync_targets.txt"

python3 "$REPO_ROOT/scripts/nightly_summary.py" >/dev/null

echo "[ok] harvested research OS state from $RUNPOD_HOST:$RUNPOD_PORT"
