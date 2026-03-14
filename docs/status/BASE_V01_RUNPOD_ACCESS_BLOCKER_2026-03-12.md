# Base v0.1 RunPod Access Blocker

Date: 2026-03-12
Scope: launching the required base `mistralai/Mistral-7B-v0.1` P0 canonical run

## Outcome

No live RunPod route was available from this shell, so the base P0 launch could not start.

## Routes tested

- `gpu-server` -> `198.13.252.23:12221` -> `Connection refused`
- `gpu-new` -> `216.81.151.42:18748` -> `Connection refused`
- `gpu-rtx6000` -> `198.13.252.9:18750` -> `Connection refused`
- `runpod-new` -> `198.13.252.12:19757` -> `Permission denied (publickey)`
- `root@195.26.232.181:34148` -> `Operation timed out`
- `root@195.26.233.44:47660` -> `Connection refused`
- `root@195.26.233.61:53317` -> `Connection refused`
- `root@213.173.111.30:16010` -> `Connection refused`
- `root@195.26.233.16:35333` -> `Connection refused`
- `root@198.13.252.15:18609` -> `Connection refused`
- `isv37z6krqu4q2-644112db@ssh.runpod.io` with `~/.ssh/id_ed25519` -> `Permission denied (publickey)`
- `9e2s58yh8i7w9f-64411d45@ssh.runpod.io` with `~/.ssh/id_ed25519` -> `Permission denied (publickey)`
- `oi2tyqeicc09lm-64411d42@ssh.runpod.io` with `~/.ssh/id_ed25519` -> connects, then `container not found`

## Local clues found

- `~/.ssh/config.save` contains older RunPod aliases (`runpod-research`, `runpod`, `runpod-dec9`, `runpod-dec10`, `runpod-dec11`).
- `agni-workspace/experiment_summary.md` references `oi2tyqeicc09lm-64411d42@ssh.runpod.io`, but that identity now resolves to a missing container.

## What is needed

One working RunPod endpoint plus the matching SSH credential for this shell. Once that exists, the next command should be:

```bash
python scripts/p0_canonical_pipeline.py --model mistralai/Mistral-7B-v0.1
```

If the remote repo is not current, push first with `scripts/runpod/push_repo.sh` and then launch the base queue from `scripts/runpod_mistral_base_v01_core7.sh` or the single P0 command above.
