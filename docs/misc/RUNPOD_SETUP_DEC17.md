# RunPod Setup - Dec 17, 2025

## Connection Details
```bash
ssh root@38.80.152.72 -p 30814 -i ~/.ssh/id_ed25519
```

## GPU Specifications
- **Model:** NVIDIA RTX PRO 6000 Blackwell Server Edition
- **VRAM:** 97GB total
- **CUDA:** 12.8
- **Driver:** 570.195.03

## Environment Status

### ✅ Installed
- Python 3.12.3
- PyTorch 2.8.0+cu128 (CUDA enabled)
- Transformers library
- All project code synced to `/workspace/mech-interp-latent-lab-phase1`

### ✅ Verified
- GPU accessible via `torch.cuda.is_available()`
- Model loading module imports successfully
- Code structure intact (`src/`, `configs/`, `results/`)

## Quick Test Commands

### Test GPU
```bash
ssh root@38.80.152.72 -p 30814 -i ~/.ssh/id_ed25519
cd /workspace/mech-interp-latent-lab-phase1
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Test Model Loading
```bash
cd /workspace/mech-interp-latent-lab-phase1
python3 -c "from src.core.models import load_model; print('✅ Ready')"
```

### Run Experiment
```bash
cd /workspace/mech-interp-latent-lab-phase1
python3 -m src.pipelines.run --config configs/gold/05_behavior_strict.json
```

## Notes
- Code synced successfully (98MB transferred)
- Minor permission warnings on temp files (safe to ignore)
- All dependencies installed with `--break-system-packages` flag

## Status: ✅ READY FOR EXPERIMENTS








