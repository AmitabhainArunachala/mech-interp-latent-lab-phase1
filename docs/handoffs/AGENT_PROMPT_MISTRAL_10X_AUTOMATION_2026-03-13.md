You are working on the Mistral control-system program in `/Users/dhyana/mech-interp-latent-lab-phase1`.

Read first:

- `docs/status/MISTRAL_10X_AUTOMATION_PROGRAM_2026-03-13.md`
- `docs/status/CLAIM_REGISTRY.md`
- `docs/status/MISTRAL_NUMERIC_SOURCE_OF_TRUTH_2026-03-11.md`

Primary automation entrypoints:

- `configs/experiment_registry/mistral_program_registry.json`
- `scripts/runpod_mistral_overnight_program_queue.sh`
- `scripts/harvest_runpod_research_os.sh`
- `scripts/nightly_summary.py`

Non-negotiable rules:

1. Canonical paper model is `mistralai/Mistral-7B-v0.1`.
2. Do not mix `Instruct-v0.2` numbers into base claims.
3. Do not edit paper numbers from chat summaries or partial logs.
4. Every numerical statement must cite one exact artifact path.
5. Do not launch broad exploratory sweeps unless resolving a specific contradiction.

Current scientific frame:

- early source region: `L0-L5`
- best delicate upstream handle: `L4 MLP`
- best late controller: `L25`
- late readout/compression cluster: `L27`
- prompt-to-generation bridge is real
- quality-linked bridge is now stronger than word-count bridge
- full clean sufficiency is not yet shown

Current priorities:

1. persistence hardening
2. scaffold ablation ladder
3. minimal sufficient bundle
4. narrow larger-model replication

If you run experiments:

- use unattended queue scripts
- prefer the registry-driven launcher over hand-written one-off tmux commands
- write `STATUS.txt`
- capture artifact paths automatically
- sync results back local
- update `CLAIM_REGISTRY.md` only after confirming the raw artifact

If you analyze experiments:

- prefer contradiction finding over hype
- surface failures and drift clearly
- treat `word_count` as diagnostic, not headline
- prefer quality-class, BT+ART, and recursive-content metrics

Your job is not to make the story prettier.
Your job is to make it cleaner, harder, and more reproducible.
