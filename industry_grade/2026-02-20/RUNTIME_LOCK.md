# Runtime Lock Notes (2026-02-20)

## Goal
Freeze execution context to reduce environment drift between replication batches.

## Baseline lock source
- `requirements.lock` (direct dependency lock for canonical reproducibility)
- Existing container spec: `Dockerfile`

## Remote runtime observed (active GPU session)
- Evidence file: `industry_grade/2026-02-20/evidence/runtime_snapshot_remote_2026-02-20.txt`
- Key versions:
  - `torch==2.4.1+cu124`
  - `transformers==4.36.2`
  - `numpy==2.4.2`, `scipy==1.17.0`, `pandas==3.0.1`
  - `accelerate==1.12.0`, `sentencepiece==0.2.1`, `protobuf==6.33.5`

## Drift vs lock
`requirements.lock` pins `torch==2.1.2` and older scientific stack versions.
Current remote runtime is newer than lock.

## Hardening action
For paper-grade batches, run one of:
1. Build and run from `Dockerfile` (CUDA 12.1 + locked direct dependencies).
2. Export full freeze from run host (`pip freeze`) and save as dated lock in `industry_grade/<date>/` before batch execution.

## Current status
- Runtime is documented (snapshot captured).
- Lock drift is explicit and auditable.
- Seed replication batch currently in-flight on documented runtime.
