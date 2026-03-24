# Cross-Model Replication Automation Status

Date: 2026-03-15

## Bottom line

We are not starting from zero.
The repo already has most of the hard operational substrate needed for automated cross-model
replication, but it is still Mistral-shaped rather than fully model-agnostic.

Best current estimate:

- operational automation for queueing / leases / result indexing: `80%`
- automated replication of the narrow core assay on a new model: `72%`
- automated end-to-end scientific replication with clean homolog mapping and paper-safe claim lock:
  `50%`

What changed materially on 2026-03-15:

- the first live `Qwen/Qwen2.5-7B` base replication bundle succeeded end-to-end
  - canonical `P0` contraction on the frozen contract
  - frozen-contract full path patching with a strong early-source map
- the first live `EleutherAI/pythia-1.4b` base replication bundle also succeeded end-to-end
  - canonical `P0` came back null on the frozen contract
  - frozen-contract path patching came back weak/noisy rather than Mistral-like
- one real portability bug was found and fixed:
  - model downloads were filling the root overlay
  - replication bundles now pin HF caches to `/workspace/hf_cache`

## What already exists

- queue scripts that update `research_os`
- shared registry files:
  - `configs/experiment_registry/mistral_program_registry.json`
  - `configs/experiment_registry/pod_leases.json`
  - `configs/experiment_registry/results_index.json`
- reusable canonical cross-architecture configs under `configs/canonical/rv_causal_*`
- existing `phase1_cross_architecture` and `power_up` result families
- live base-model replication bundles now proven on:
  - `Qwen/Qwen2.5-7B`
  - `EleutherAI/pythia-1.4b`
- local utilities that already know how to read/write lease and result state:
  - `src/utils/research_os.py`
  - `src/utils/mistral_program.py`

## What is still missing

### 1. One model-agnostic assay bundle

Right now the logic is spread across model-specific configs and historical families.
We need one explicit replication bundle that says:

- run canonical prompt-pass `P0`
- run full path patching
- run late-controller search
- run late-readout validation
- run long-form quality bridge

and emits the same normalized summary schema for every model.

### 2. Homolog mapping

For Mistral we know the important layers by direct work:

- early source region `L0-L5`
- late controller `L25`
- late readout cluster `L27`

For a new model, we still need either:

- a heuristic layer-depth mapper, or
- an initial discovery pass that proposes candidate homologous layers

before the rest of the assay can run automatically.

### 3. Claim gating by family

We have a strong Mistral claim registry, but not yet a general multi-model replication gate that
automatically labels findings as:

- discovery
- confirmatory
- replicated
- paper-safe

across families.

### 4. Normalized cross-model outputs

The next real unlock is a single result schema that every model writes:

- prompt-pass effect
- early-source localization
- late-controller candidate
- late-readout candidate
- long-form quality bridge
- sufficiency status

That is the piece that turns “many runs” into “automatic replication.”

## What would make it real

To get from current state to a genuine automated replication system, the next concrete build should
be:

1. `replication_bundle_registry.json`
2. one generic launcher:
   - `scripts/run_model_replication_bundle.sh --model <...>`
3. one homolog-mapper:
   - propose early / controller / readout candidate layers by depth and quick scans
4. one normalized summarizer:
   - write `results/replication_bundle/<model>/<run>/summary.json`
5. one cross-model dashboard row generator

## Recommendation

Do not try to automate the full old exploratory zoo across models.
Automate the narrow Mistral-derived core assay first, then port only that.

The right first target is:

- `Qwen2.5-7B`
- one larger model after that

Once that pipeline works on two non-Mistral models, the system becomes a real cross-model
replication engine rather than a Mistral lab with extra scripts.
