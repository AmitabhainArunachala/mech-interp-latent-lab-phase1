# RunPod Sync Status

**Last Updated:** $(date)

## Connection Details
- **Host:** 213.173.111.30
- **Port:** 26212 ✅
- **User:** root
- **SSH Key:** ~/.ssh/id_ed25519

## Quick Connection Command
```bash
ssh root@213.173.111.30 -p 26212 -i ~/.ssh/id_ed25519
```

## Status
✅ **CONNECTED, SYNCED, AND FULLY OPERATIONAL!** - GPU working!

## Available Scripts

### 1. Quick Sync & Setup (All-in-One)
```bash
./quick_runpod_sync.sh
```
This script will:
- Test connection
- Check GPU status
- Check disk space
- Sync entire repo
- Install dependencies
- Verify PyTorch/CUDA setup

### 2. Sync Only
```bash
./sync_to_runpod.sh
```
Just syncs the repo without installing dependencies.

### 3. Setup Only
```bash
./setup_runpod_quick.sh
```
Installs dependencies and verifies setup (assumes repo already synced).

## GPU Info
- **Model:** NVIDIA RTX PRO 6000 Blackwell Server Edition
- **VRAM:** ~98GB total, ~97GB free
- **CUDA:** 12.8
- **PyTorch:** 2.8.0+cu128 (CUDA available: ✅, GPU working: ✅)
- **Python:** 3.12.3

## Next Steps

✅ **Repo is synced!** Now:

1. **Open Cursor on RunPod:**
   - File → Open Folder → `/workspace/mech-interp-latent-lab-phase1`

2. **Or SSH directly:**
   ```bash
   ssh root@213.173.111.30 -p 26212 -i ~/.ssh/id_ed25519
   cd /workspace/mech-interp-latent-lab-phase1
   ```

3. **To re-sync later:**
   ```bash
   cd /Users/dhyana/mech-interp-latent-lab-phase1
   ./sync_to_runpod.sh
   ```

## What Gets Synced

✅ **Included:**
- All Python files
- Config files
- Documentation
- Scripts

❌ **Excluded:**
- `.git/` (to avoid conflicts)
- `__pycache__/`
- `*.csv`, `*.log`, `*.png` (results)
- `results/` directory
- `models/` directory (too large)

## Troubleshooting

If connection fails:
1. Check RunPod dashboard - is the pod running?
2. Verify SSH is enabled in RunPod settings
3. Check port number matches dashboard
4. Test SSH key: `ssh-add -l` (should show your key)

If sync fails:
- Check disk space on RunPod: `df -h`
- Check permissions: `ls -la /workspace`

If dependencies fail:
- Check Python version: `python3 --version`
- Check pip: `pip3 --version`
- May need to install system packages first

