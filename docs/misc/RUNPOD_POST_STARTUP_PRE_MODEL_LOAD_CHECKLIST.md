# RunPod Post-Startup → Pre-Model-Load Checklist (Mech-Interp Latent Lab Phase 1)

**Purpose:** prevent “silent bad runs” (wrong GPU, wrong caches, low disk, missing token, broken deps) *before* you pay the model-load tax.

---

## 0) Confirm you’re in the right place

- **Repo path**: you should be in the repo root (contains `src/`, `configs/`, `results/`, `env.txt`).

```bash
pwd
ls
python3 --version
```

---

## 1) Hardware sanity (GPU + driver + VRAM)

```bash
nvidia-smi
```

- **Check**:
  - GPU model matches what you paid for
  - driver + CUDA version present
  - no unexpected processes eating VRAM

Optional (more detail):

```bash
nvidia-smi -q | sed -n '1,120p'
```

---

## 2) Disk + inode sanity (model weights are big)

```bash
df -h
df -hi
```

- **Check**:
  - enough free space for model weights + caches + outputs (rule of thumb: **50–150GB** free depending on model set)
  - not near inode exhaustion

---

## 3) Python env sanity (fast fail)

### Option A (recommended): fresh venv per pod + reuse caches on volume

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r env.txt
```

### Option B: you already have an env

```bash
python -c "import torch; import transformers; import numpy; import pandas; print('ok')"
```

---

## 4) Torch + CUDA sanity (before any HF model load)

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)
if torch.cuda.is_available():
    print("device_count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"[{i}] {p.name}  VRAM={p.total_memory/1024**3:.1f}GB")
PY
```

- **If `cuda_available=False`**: stop and fix this now (wrong image, driver mismatch, or broken torch install).

---

## 5) HuggingFace auth + cache paths (critical on RunPod)

### 5.1 Decide where caches live (prefer a persistent mounted volume)

Pick a persistent path (examples; adjust to your RunPod mount):
- `/workspace` (common)
- `/workspace/persistent`
- `/volume`

Then set:

```bash
export HF_HOME="/workspace/.hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"
```

### 5.2 Confirm token is available (don’t paste it into logs)

- Either:
  - `export HUGGINGFACE_HUB_TOKEN=...` (best done via RunPod secret/env)
  - or `huggingface-cli login` (interactive; avoid if you want non-interactive scripts)

Fast check (should not print your token):

```bash
python - <<'PY'
import os
tok = os.environ.get("HUGGINGFACE_HUB_TOKEN")
print("HUGGINGFACE_HUB_TOKEN present:", bool(tok))
print("HF_HOME:", os.environ.get("HF_HOME"))
print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))
PY
```

---

## 6) Repo “import + compile” sanity (catches broken edits immediately)

```bash
python3 -m py_compile src/pipelines/run.py src/pipelines/registry.py
python3 -m py_compile src/metrics/rv.py src/core/hooks.py
```

---

## 7) Dry-run the canonical runner (still no model load)

This just checks CLI/config parsing and that the runner can create output dirs.

```bash
python -m src.pipelines.run --help
```

Optional: confirm the Phase 0 docs exist where expected:

```bash
ls results/phase0_metric_validation || true
```

---

## 8) “Session ritual”: run 1–2 tiny canonical pipelines *every time*

**Why:** you’re still learning MI, and the repo is big — running a consistent baseline each session keeps you grounded and makes regressions obvious.

### Recommended default (Phase 0 / metric sanity)

Run these first, then do any new experiments:

```bash
python -m src.pipelines.run --config configs/phase0_minimal_pairs.json
python -m src.pipelines.run --config configs/phase0_metric_targets.json
```

**After each run**: open the printed `run_dir` and skim `report.md` + `summary.json`.

---

## 9) Pre-model-load “stop conditions” (don’t proceed if any are true)

- **GPU missing / wrong VRAM** (`nvidia-smi` or torch shows no CUDA)
- **Disk too low** (you’ll crash mid-download or mid-run)
- **HF token missing** (private model / gated model) or cache path not persistent
- **Imports fail** (`py_compile` fails) — fix code before you load models
- **You can’t write to results dir** (permissions / volume not mounted)

---

## 10) When it’s green: now load models

At this point, model load time is “safe to pay” because the environment is consistent and auditable.


