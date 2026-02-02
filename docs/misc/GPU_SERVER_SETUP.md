# GPU Server Setup - January 15, 2025

## Connection Details

**Host:** 216.81.151.42  
**Port:** 18748 (SSH: 22)  
**User:** root  
**SSH Alias:** `gpu-new`

## Quick Connect

```bash
# Using alias (after adding to ~/.ssh/config)
ssh gpu-new

# Direct connection
ssh -p 18748 root@216.81.151.42
```

## System Specs

- **GPU:** NVIDIA RTX PRO 6000 Blackwell Server Edition
- **GPU Memory:** 97GB total
- **OS:** Ubuntu (Linux 6.8.0)
- **Python:** 3.12.3
- **Status:** Fresh server, ready for setup

## SSH Config

Added to `~/.ssh/config`:
```
Host gpu-new
    HostName 216.81.151.42
    Port 18748
    User root
    StrictHostKeyChecking no
```

## Sync Strategy

**Selective sync only** - sync files as needed:
- Config files
- Pipeline scripts
- Prompt bank
- Results (pull back)

## Next Steps

When ready to sync specific files:
```bash
# Sync a specific file
scp -P 18748 file.py root@216.81.151.42:/root/mech-interp-latent-lab-phase1/

# Sync a directory
rsync -avz -e "ssh -p 18748" src/ root@216.81.151.42:/root/mech-interp-latent-lab-phase1/src/
```
