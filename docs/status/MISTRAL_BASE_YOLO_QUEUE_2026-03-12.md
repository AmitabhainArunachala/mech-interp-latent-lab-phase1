# Mistral Base YOLO Queue

Date: 2026-03-12
Model: `mistralai/Mistral-7B-v0.1`

## Armed queue

- Active soft followup wrapper PID on RunPod: `35600`
- Active soft followup child PID on RunPod at queue time: `35619`
- Armed base queue wrapper PID on RunPod: `36939`
- Armed queue session: `mistral_base_priority_queue_20260312_084229`

## Remote run directories

- Soft followups currently in flight:
  - `results/mistral_soft_followups/20260312_083136`
- Armed followup queue:
  - `results/mistral_base_priority_queue/20260312_084229`

## Queue order

1. Finish current soft followups on base v0.1
2. Run `p0_canonical_pipeline.py --model mistralai/Mistral-7B-v0.1 --n 100`
3. Run `full_head_sweep.py --model mistralai/Mistral-7B-v0.1 --n-prompts 100 --batch-layers 8`

## Relevant scripts

- `scripts/runpod_mistral_soft_followups.sh`
- `scripts/runpod_mistral_base_priority_queue.sh`

## Intent

Keep the GPU occupied with the highest-ROI base-only queue after the current soft multisite run, without overlapping jobs.
