#!/usr/bin/env bash
set -euo pipefail

# RunPod remote bootstrap (run ON the pod, after SSH).
#
# Goals:
# - consistent HF caches
# - sane python env
# - fast "compile/import" sanity
# - run 1–2 canonical pipelines to anchor the session
#
# Usage (on pod):
#   bash scripts/runpod/bootstrap_remote.sh
#
# Optional env:
#   REPO_DIR=/workspace/mech-interp-latent-lab-phase1
#   HF_HOME=/workspace/.hf
#   PYTHON=python3

REPO_DIR="${REPO_DIR:-/workspace/mech-interp-latent-lab-phase1}"
HF_HOME="${HF_HOME:-/workspace/.hf}"
PYTHON="${PYTHON:-python3}"
PIP_DISABLE_PIP_VERSION_CHECK=1

echo "[runpod] repo_dir: ${REPO_DIR}"
echo "[runpod] hf_home: ${HF_HOME}"
echo "[runpod] python:   ${PYTHON}"

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
df -h || true

mkdir -p "${HF_HOME}" "${HF_HOME}/transformers"
export HF_HOME="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
# Enable hf_transfer only if installed; otherwise it hard-errors in huggingface_hub.
if python - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("hf_transfer") else 1)
PY
then
  export HF_HUB_ENABLE_HF_TRANSFER=1
else
  unset HF_HUB_ENABLE_HF_TRANSFER || true
fi

cd "${REPO_DIR}"

if [[ ! -f "env.txt" ]]; then
  echo "[runpod][fatal] env.txt not found in repo root: ${REPO_DIR}" >&2
  exit 2
fi

if [[ ! -d ".venv" ]]; then
  "${PYTHON}" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Only (re)install deps if env.txt changed.
ENV_SHA_FILE=".venv/.env_txt_sha256"
ENV_SHA_NOW="$(python - <<'PY'
import hashlib
from pathlib import Path
p = Path("env.txt")
print(hashlib.sha256(p.read_bytes()).hexdigest())
PY
)"
ENV_SHA_OLD=""
if [[ -f "${ENV_SHA_FILE}" ]]; then
  ENV_SHA_OLD="$(cat "${ENV_SHA_FILE}" || true)"
fi

if [[ "${ENV_SHA_NOW}" != "${ENV_SHA_OLD}" ]]; then
  echo "[runpod] env.txt changed (or first install) -> installing deps..."
  python -m pip install --upgrade pip
  python -m pip install -r env.txt
  echo -n "${ENV_SHA_NOW}" > "${ENV_SHA_FILE}"
else
  echo "[runpod] env.txt unchanged -> skipping pip install"
fi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
PY

python -m py_compile src/pipelines/run.py src/pipelines/registry.py
python -m py_compile src/metrics/rv.py src/core/hooks.py

echo "[runpod] running canonical Phase 0 anchors..."
python -m src.pipelines.run --config configs/phase0_minimal_pairs.json
python -m src.pipelines.run --config configs/phase0_metric_targets.json

echo "[runpod][ok] bootstrap complete"


