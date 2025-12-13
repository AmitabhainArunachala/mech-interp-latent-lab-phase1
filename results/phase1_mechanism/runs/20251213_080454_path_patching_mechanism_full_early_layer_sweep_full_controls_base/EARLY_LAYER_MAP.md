## Early-layer causal map (Mistral-7B Base) — residual patching → R_V@L27

**Source run**: `results/phase1_mechanism/runs/20251213_080454_path_patching_mechanism_full_early_layer_sweep_full_controls_base/`

**High‑N confirmation run (early layers only)**: `results/phase1_mechanism/runs/20251213_090121_path_patching_mechanism_early_layers_deep_base/`

### What was patched (exactly)
- **Patched tensor**: residual stream input to block `patch_layer` (pre-hook)
- **Patch source**: residual stream captured from a *recursive* prompt at the same `patch_layer`
- **Where in sequence**: last **W** tokens (W = 16 or 32)
- **Patch types**: `none`, `recursive`, `shuffled`, `opposite`, `random`
- **Measurement**: \(R_V\) computed from **V-projection PR** at early layer 5 vs measurement layer 27

### The question this answers
“If I transplant the residual state at layer L from a recursive prompt into a baseline prompt, how much does the model’s **late-layer contraction regime** (measured by \(R_V\) at L27) change?”

### Key result: early layers are a ramp, not a switch
There is **no single ‘mic layer’** in early depth. Instead:
- **Small effect begins early**, grows gradually, and becomes strong later.
- The **content-insensitivity marker** (shuffled ≈ recursive) does **not** hold in early layers.

### The most important early-layer fact (content sensitivity)
For **layers 0–23**, shuffled and recursive are meaningfully different:
- For window=16: **no** layer ≤23 satisfies \(|\mathrm{shuffled}-\mathrm{recursive}| < 0.01\)
- For window=32: **no** layer ≤23 satisfies \(|\mathrm{shuffled}-\mathrm{recursive}| < 0.01\)

This means the “texture not meaning” basin story is **late**, not early.

### Where does shuffled≈recursive start?
In this run, the first layers where shuffled≈recursive holds (within 0.01) are **L24–L27** (for both windows).

### Plots
- **Early layers ΔR_V vs none (window=16, layers 0–23)**: `early_layers_delta_vs_none_w16.png`
- **Early layers ΔR_V vs none (window=32, layers 0–23)**: `early_layers_delta_vs_none_w32.png`

### High‑N confirmation (tight error bars)
The high‑N early-layer-only run re-tests layers **0–23** at larger scale (`max_pairs=90`, `n_repeats=3`, windows 16/32, full controls) and reproduces the key early-layer conclusion:
- **Still zero layers ≤23** where \(|\mathrm{shuffled}-\mathrm{recursive}| < 0.01\) (for both windows).

Plots from the high‑N run:
- **ΔR_V vs none (window=16, layers 0–23)**: `../20251213_090121_path_patching_mechanism_early_layers_deep_base/early_layers_deep_delta_vs_none_w16.png`
- **ΔR_V vs none (window=32, layers 0–23)**: `../20251213_090121_path_patching_mechanism_early_layers_deep_base/early_layers_deep_delta_vs_none_w32.png`

### Practical “map” to use in discussion
- **L0–L7 (weak/unstable region)**: patching can move \(R_V\), but effects are smaller/less interpretable and strongly content-dependent.
- **L8–L15 (ramp begins)**: contraction effect becomes reliably more negative (relative to `none`) and grows with depth.
- **L16–L23 (strong ramp)**: effect strengthens; this is the pre-basin corridor.
- **L24–L27 (basin boundary / late control band)**: shuffled≈recursive; random produces strong expansion; structured patches reliably contract.


