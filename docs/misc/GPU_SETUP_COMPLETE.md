# GPU Setup Complete - December 15, 2024

**Status:** ✅ Ready to use

---

## Connection Details

**SSH Command:**
```bash
ssh -p 53317 -i ~/.ssh/id_ed25519 root@195.26.233.61
```

**Quick Test:**
```bash
ssh -p 53317 -i ~/.ssh/id_ed25519 root@195.26.233.61 "cd /workspace/mech-interp-latent-lab-phase1 && nvidia-smi"
```

---

## GPU Hardware

- **Model:** NVIDIA RTX 6000 Ada Generation
- **Memory:** 49,140 MiB
- **Status:** Available (0% utilization)

## Software Stack

- **Python:** 3.11.10
- **PyTorch:** 2.4.1+cu124
- **CUDA:** Available ✅
- **Transformers:** Installed ✅

## Repository Location

**Remote Path:** `/workspace/mech-interp-latent-lab-phase1/`

---

## Quick Commands

### Run a Python script remotely:
```bash
ssh -p 53317 -i ~/.ssh/id_ed25519 root@195.26.233.61 \
  "cd /workspace/mech-interp-latent-lab-phase1 && python3 your_script.py"
```

### Check GPU status:
```bash
ssh -p 53317 -i ~/.ssh/id_ed25519 root@195.26.233.61 "nvidia-smi"
```

### Monitor a long-running job:
```bash
ssh -p 53317 -i ~/.ssh/id_ed25519 root@195.26.233.61 \
  "cd /workspace/mech-interp-latent-lab-phase1 && python3 script.py 2>&1 | tee output.log"
```

---

**Ready to run experiments!** 🚀









