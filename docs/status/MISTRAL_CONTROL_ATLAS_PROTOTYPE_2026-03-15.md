# Mistral Control Atlas Prototype

Date: 2026-03-15

This note is the easy-to-find lab anchor for the first replay-first mech interp viewer.

## Where it lives

- Standalone page:
  - `website/control-atlas.html`
- Embedded landing-page section:
  - `website/index.html`
- Viewer logic:
  - `website/mistral-control-atlas.js`
- Generated artifact-backed dataset:
  - `website/data/mistral-control-atlas.json`
  - `website/data/mistral-control-atlas-data.js`
- Data builder:
  - `scripts/build_mistral_control_atlas.py`
- Local serve helper:
  - `scripts/serve_mistral_control_atlas.sh`

## What it is

The atlas is a replay-first instrument for the hardened Mistral control-system story:

- early source region: `L0-L5`
- late controller: `L25`
- late readout/cleanup cluster: `L27`
- anchor / bridge / subtle `L4` prompt-pass comparisons
- induced-persistence turn replay
- late subspace comparison at `L27`

The trajectory motion is intentionally labeled as a replay synthesis from locked control points, not a raw full-state live stream.

## Why it matters

This is the first concrete UI surface for turning our locked Mistral artifacts into a causal geometry viewer rather than just tables and heatmaps.

It is the right place to extend later with:

- real hook-trace exports
- token-level live streaming
- true layer/head overlays
- side-by-side intervention comparisons

## Reopen later

Fastest path:

```bash
open /Users/dhyana/mech-interp-latent-lab-phase1/website/control-atlas.html
```

Or serve it locally:

```bash
cd /Users/dhyana/mech-interp-latent-lab-phase1
bash scripts/serve_mistral_control_atlas.sh 8000
```
