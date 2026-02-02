# RunPod SSH Enable Guide

## Current Status
- ✅ Host is reachable (ping works)
- ❌ SSH port 26018 is closed/refused
- **Issue:** SSH service not enabled or not running

## Quick Fix Steps

### Option 1: Enable SSH via RunPod Dashboard
1. Go to your RunPod dashboard
2. Find your pod (82.221.170.234)
3. Click on the pod → **Settings** or **Connect**
4. Look for **"SSH"** or **"Enable SSH"** toggle
5. Enable it and note the port (might be different from 26018)
6. The port is usually shown in the connection details

### Option 2: Use RunPod Web Terminal
1. In RunPod dashboard, click **"Connect"** or **"Terminal"**
2. This opens a web-based terminal
3. Once connected, you can manually start SSH:
   ```bash
   # Check if SSH is installed
   which sshd || apt-get update && apt-get install -y openssh-server
   
   # Start SSH service
   service ssh start
   # OR
   systemctl start ssh
   # OR
   /etc/init.d/ssh start
   
   # Check what port SSH is using
   netstat -tlnp | grep ssh
   # OR
   ss -tlnp | grep ssh
   ```

### Option 3: Check RunPod Connection Tab
1. In RunPod dashboard, go to your pod
2. Click **"Connect"** tab
3. Look for **"SSH"** section
4. It should show:
   - The correct SSH command
   - The correct port number
   - Connection instructions

## Once SSH is Enabled

After enabling SSH, update the port if needed and run:

```bash
cd /Users/dhyana/mech-interp-latent-lab-phase1
./quick_runpod_sync.sh
```

Or if the port changed, update it in the script first.

## Verify SSH is Working

Once enabled, test with:
```bash
ssh -v -p [PORT] -i ~/.ssh/id_ed25519 root@82.221.170.234 "echo 'SSH works!'"
```

## Common RunPod SSH Ports
- 22 (standard SSH)
- 2222 (alternative)
- 26018 (your current)
- Sometimes RunPod assigns random high ports (30000+)

Check the RunPod dashboard for the exact port!









