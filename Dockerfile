# mech-interp-latent-lab-phase1/Dockerfile
# MI Research Environment - CUDA 12.1 + Python 3.11
# 
# Build: docker build -t mech-interp:latest .
# Run:   docker run --gpus all -it -v $(pwd):/workspace mech-interp:latest

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Rust toolchain (for tokenizers compilation if needed)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Working directory
WORKDIR /workspace

# Copy requirements first (Docker cache optimization)
COPY requirements.lock .

# Install PyTorch with CUDA 12.1 support first
RUN pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies from lock file
RUN pip install -r requirements.lock

# TransformerLens for MI research
RUN pip install transformer-lens

# Copy entire repo
COPY . .

# Verify installation
RUN python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}')" \
    && python -c "import transformers; print(f'Transformers {transformers.__version__}')" \
    && python -c "import transformer_lens; print('TransformerLens OK')"

# Labels
LABEL maintainer="mech-interp-lab"
LABEL description="Mechanistic Interpretability Research Environment"
LABEL version="1.0"

# Default to bash for interactive use
CMD ["/bin/bash"]
