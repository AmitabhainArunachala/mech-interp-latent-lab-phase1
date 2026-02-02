# RunPod Setup Instructions

**Server:** `198.13.252.12:22`  
**Status:** SSH authentication needed

---

## Option 1: Use RunPod Web Terminal (Recommended)

1. **Open RunPod web terminal** (via RunPod dashboard)
2. **Run setup script:**
   ```bash
   cd /root
   # Copy setup script content or upload it
   bash <(curl -s https://raw.githubusercontent.com/your-repo/setup_runpod.sh)
   # OR manually run:
   pip install transformers torch numpy pandas scipy tqdm
   ```

3. **Clone or sync repository:**
   ```bash
   cd /root
   git clone <your-repo-url> mech-interp-latent-lab-phase1
   # OR sync via RunPod file manager
   ```

4. **Verify:**
   ```bash
   cd /root/mech-interp-latent-lab-phase1
   python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```

---

## Option 2: Set Up SSH Keys

1. **Generate SSH key locally** (if not exists):
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/runpod_key -N ""
   ```

2. **Copy public key to RunPod** (via web terminal):
   ```bash
   # On RunPod, run:
   mkdir -p ~/.ssh
   echo "<your-public-key>" >> ~/.ssh/authorized_keys
   chmod 600 ~/.ssh/authorized_keys
   ```

3. **Update SSH config:**
   ```bash
   # Add to ~/.ssh/config:
   Host runpod-new
       HostName 198.13.252.12
       Port 22
       User root
       IdentityFile ~/.ssh/runpod_key
   ```

4. **Test connection:**
   ```bash
   ssh runpod-new 'echo "✅ Connected"'
   ```

---

## Option 3: Password Authentication

If RunPod has password authentication enabled:

```bash
ssh root@198.13.252.12 -p 22
# Enter password when prompted
```

Then run setup commands manually.

---

## Quick Setup Commands (Once Connected)

```bash
# 1. Install dependencies
pip install transformers torch numpy pandas scipy tqdm scikit-learn

# 2. Verify GPU
nvidia-smi

# 3. Test imports
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Files to Sync

Once SSH is working, sync these directories:

```bash
rsync -avz -e "ssh -p 22" \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  src/ configs/ scripts/ prompts/ \
  root@198.13.252.12:/root/mech-interp-latent-lab-phase1/
```

---

## Critical Files Needed

- `src/pipelines/mlp_vproj_combined_sufficiency_test.py` (NEW - not run yet)
- `src/pipelines/logit_lens_analysis.py`
- `src/pipelines/vproj_patching_analysis.py`
- `src/metrics/logit_lens.py`
- `src/metrics/logit_diff.py`
- `configs/mlp_vproj_combined_sufficiency.json`
- `scripts/run_mlp_vproj_combined.py`

---

## First Experiment to Run

Once setup is complete:

```bash
cd /root/mech-interp-latent-lab-phase1
python3 scripts/run_mlp_vproj_combined.py
```

This tests the **complete circuit** (Gate + Amplifier + Readout).

---

## Troubleshooting

**SSH Permission Denied:**
- Check if password auth is enabled in RunPod settings
- Set up SSH keys via web terminal
- Use RunPod web terminal instead

**Import Errors:**
- Run: `pip install --upgrade transformers torch`
- Check Python version: `python3 --version` (need 3.8+)

**CUDA Not Available:**
- Check GPU: `nvidia-smi`
- Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
