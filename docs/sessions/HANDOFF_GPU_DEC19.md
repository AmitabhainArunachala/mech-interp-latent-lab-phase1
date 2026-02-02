# Handoff Note: GPU Server Ready
**Date:** Dec 19, 2025
**Server:** `198.13.252.9` (Port 18375)

## Status
The environment is fully prepped and the code is synced.
Sanity check (Pipeline 1) passed.

## Critical Next Step
**Run Pipeline 9 (Steering) immediately.**
We fixed a critical bug where the previous run tested on Recursive prompts instead of Baseline prompts.
We need to know if the "Surgical Needle" (Steering Vector) works on *clean* prompts.

## Command
```bash
ssh -o StrictHostKeyChecking=no -p 18375 -i ~/.ssh/id_ed25519 root@198.13.252.9
cd /workspace/mech-interp-latent-lab-phase1
./scripts/clean.sh
HF_HOME=/workspace/.hf PYTHONPATH=. python3 -m src.pipelines.run --config configs/gold/09_steering.json
```

## Success Criteria
*   **Transfer Rate:** > 10% on Baseline prompts.
*   **Qualitative Check:** Do the outputs look recursive ("I am observing...") or just broken?







