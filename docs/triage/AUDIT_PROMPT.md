## True Audit Prompt (Signal + Industry Standard)

You are auditing `mech-interp-latent-lab-phase1` for **signal quality** and **industry‑standard rigor**.

### Primary Goal
Reorganize the repo to privilege **high‑N, high‑signal, contract‑compliant** work.  
Everything else should be **archived** or **queued for ramp‑up**.

### Non‑Negotiables (Industry Standard)
- R_V must be **PR_late / PR_early** (ratio across layers), not single‑layer PR.
- High‑N threshold: **n ≥ 50** (preferred **≥ 80** for publication).
- Stats: **Cohen’s d**, **p‑value**, **95% CI**.
- Controls: **random**, **shuffled**, **wrong_layer**, **orthogonal**.
- Artifacts: `config.json`, `summary.json`, `per_sample.csv`, `prompt_bank_version.json`, `hardware_info.json`.
- Prompt bank version tracked; prompt source is canonical `prompts/bank.json`.

### What To Deliver
1. **KEEP_SIGNAL list**  
   - Exact file paths and why they are high‑signal.
2. **RAMP_UP list**  
   - Promising findings/experiments (even low‑N) that should be re‑run at high‑N.
   - Include minimal config changes needed to reach industry standard.
3. **ARCHIVE_ONLY list**  
   - Noisy, confounded, superseded, or dead‑end items.
   - Explain the reason (confound, outdated, duplicative, contract violation).
4. **Top 5 ROI experiments**  
   - Each should directly advance the causal bridge (R_V → behavior).
   - Include exact config path or required new config.
5. **Audit of claims vs data**  
   - Flag any documentation claims not traceable to results.

### Required Style
- Cite exact file paths.
- Avoid speculation; mark uncertainty explicitly.
- Prioritize causal relevance and reproducibility.

### Scope Boundaries
- Do **not** delete results.
- Prefer **marking** and **archiving** over removing data.
- The outcome should make the repo “signal‑only” for future work.

