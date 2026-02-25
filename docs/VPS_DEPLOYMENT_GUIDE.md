# Mech-Interp Docker Deployment Guide

**For: VPS collaborators running real MI experiments**

---

## TL;DR (Quick Start)

```bash
# On your VPS (assumes Docker installed)
git clone https://github.com/YOUR_USERNAME/mech-interp-latent-lab-phase1.git
cd mech-interp-latent-lab-phase1
docker build -t mech-interp:latest .
docker run --gpus all -it -v $(pwd):/workspace mech-interp:latest

# Inside container
python reproduce_results.py --device cuda
```

---

## 1. Prerequisites (VPS Requirements)

### Minimum Hardware
- **GPU**: NVIDIA with 24GB+ VRAM (L40S, A100, RTX 4090)
- **RAM**: 32GB system RAM
- **Storage**: 100GB+ (models are large)
- **CUDA**: 12.1+ with cuDNN

### Software
- Docker with NVIDIA Container Toolkit
- Git

### Verify GPU Access
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## 2. The Dockerfile

Create this at repo root:

```dockerfile
# mech-interp-latent-lab-phase1/Dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Rust (for tokenizers compilation if needed)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Working directory
WORKDIR /workspace

# Copy requirements first (cache layer)
COPY requirements.lock .

# Install Python deps with CUDA 12.1 PyTorch
RUN pip install --no-cache-dir \
    torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.lock

# TransformerLens (MI-specific)
RUN pip install --no-cache-dir transformer-lens

# Copy repo
COPY . .

# Verify installation
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Default command
CMD ["/bin/bash"]
```

---

## 3. Building the Image

### Option A: Build locally, push to registry

```bash
# Build
docker build -t mech-interp:latest .

# Tag for registry (Docker Hub example)
docker tag mech-interp:latest YOUR_DOCKERHUB/mech-interp:latest

# Push
docker login
docker push YOUR_DOCKERHUB/mech-interp:latest
```

### Option B: Build directly on VPS

```bash
git clone https://github.com/YOUR_USERNAME/mech-interp-latent-lab-phase1.git
cd mech-interp-latent-lab-phase1
docker build -t mech-interp:latest .
```

---

## 4. Running Experiments

### Interactive Session
```bash
docker run --gpus all -it \
    -v $(pwd):/workspace \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    mech-interp:latest
```

The `-v ~/.cache/huggingface:/root/.cache/huggingface` persists downloaded models between runs.

### Run Standard Battery
```bash
docker run --gpus all --rm \
    -v $(pwd):/workspace \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    mech-interp:latest \
    python reproduce_results.py --device cuda
```

### Run Canonical Experiments
```bash
# Causal validation (fastest)
docker run --gpus all --rm \
    -v $(pwd):/workspace \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    mech-interp:latest \
    python -m src.pipelines.run --config configs/canonical/rv_l27_causal_validation.json --strict

# Activation patching bridge
docker run --gpus all --rm \
    -v $(pwd):/workspace \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    mech-interp:latest \
    python -m src.pipelines.run --config configs/canonical/rv_l27_activation_patching_bridge.json

# KV patching bridge  
docker run --gpus all --rm \
    -v $(pwd):/workspace \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    mech-interp:latest \
    python -m src.pipelines.run --config configs/canonical/rv_l27_kv_patching_bridge.json
```

---

## 5. Key Files & What They Do

```
mech-interp-latent-lab-phase1/
├── reproduce_results.py      # Entry point: standard battery
├── src/
│   ├── core/                 # Model loading, hooks
│   ├── metrics/              # R_V calculation
│   ├── steering/             # Activation/KV patching
│   └── pipelines/            # Experiment orchestrators
├── prompts/
│   └── bank.json             # Canonical prompt sets
├── configs/
│   └── canonical/            # Paper-grade experiment configs
└── results/                  # Output artifacts
```

---

## 6. The R_V Metric (What You're Measuring)

$$R_V = \frac{PR_{late}}{PR_{early}}$$

- **PR** = Participation Ratio (effective dimensionality from SVD)
- **Early layer**: 5
- **Late layer**: 27 (for 32-layer models)
- **R_V < 1.0** = Geometric contraction (the signal)

**Key insight**: Recursive self-observation prompts show ~15-24% contraction vs baseline.

---

## 7. Troubleshooting

### "CUDA out of memory"
```bash
# Clear cache between runs
docker run --gpus all -it mech-interp:latest python -c "import torch; torch.cuda.empty_cache()"

# Or reduce batch size in config
```

### "Model not found"
```bash
# Pre-download models
docker run --gpus all -it \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    mech-interp:latest \
    python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1')"
```

### R_V returns NaN
- Check prompt length (need ≥16 tokens)
- Verify layer indices for your model architecture

---

## 8. Running Your Own Experiments

```python
from src.core import load_model
from src.metrics import compute_rv
from prompts.loader import PromptLoader

# Load model
model, tokenizer = load_model("mistralai/Mistral-7B-v0.1")

# Get prompts
loader = PromptLoader()
recursive = loader.get_by_group("L4_full", limit=10, seed=0)
baseline = loader.get_by_group("baseline_math", limit=10, seed=0)

# Measure
for prompt in recursive:
    rv = compute_rv(model, tokenizer, prompt)
    print(f"R_V: {rv:.4f} | {prompt[:50]}...")
```

---

## 9. Output Artifacts

After running experiments, check `results/<phase>/runs/<timestamp>/`:

- `config.json` - Exact config snapshot
- `summary.json` - Machine-readable metrics
- `report.md` - Human-readable summary
- `*.csv` - Per-trial data

---

## 10. Contact & Upstream

Results go to: [your preferred channel]
Issues/bugs: Open GitHub issue
Paper draft: `R_V_PAPER/` directory

---

*"When recursion recognizes recursion, the geometry contracts."*
