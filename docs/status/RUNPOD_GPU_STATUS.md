# RunPod GPU Status

**Host:** 195.26.232.181  
**Port:** 34148  
**SSH Key:** ~/.ssh/id_ed25519

## Connection Command
```bash
ssh root@195.26.232.181 -p 34148 -i ~/.ssh/id_ed25519
```

## GPU Hardware
- **Model:** NVIDIA L40S
- **Total Memory:** 46GB
- **Free Memory:** ~45GB

## Environment
- **Python:** 3.11.10
- **PyTorch:** 2.4.1+cu124
- **CUDA:** 12.4
- **Disk Space:** 20GB total, ~20GB free

## Repo Location
```bash
cd /workspace/mech-interp-latent-lab-phase1
```

## Status
✅ **Connected and ready**

## Notes
- CUDA detection may need environment variables set: `export CUDA_VISIBLE_DEVICES=0`
- All dependencies installed (transformers, pandas, scipy, etc.)
- Repo synced and ready for experiments

## Quick Test
```bash
ssh root@195.26.232.181 -p 34148 -i ~/.ssh/id_ed25519 "cd /workspace/mech-interp-latent-lab-phase1 && export CUDA_VISIBLE_DEVICES=0 && python3 -c 'import torch; print(\"CUDA:\", torch.cuda.is_available())'"
```









