# Today's Plan — Dec 13, 2025

**Author:** OPUS 4.5 (Vice Lead)  
**Reviewed by:** GPT-5.2 (Lead)  
**Status:** PROPOSED — awaiting Lead adjudication

---

## Primary Objective: Mistral L27 Contraction Circuit

**Goal:** Lock down the mechanistic circuit for geometric contraction (R_V) at L27 in Mistral-7B.

**Strategy Update (Dec 13 Mid-Session):**
We are splitting this into two distinct phases:
1. **Localization (Search Space Reduction):** Use "Option B" (Causal Inference) to find the *source* layers (L0-L20) that feed the contraction.
2. **Circuit Decomposition:** Once source layers are found, use "Option A" (Anthropic) to decompose into heads/MLPs.

**Why this is highest priority:**
- This is the cleanest, most publishable spine
- We have auditable Phase 0 numbers: champion R_V = 0.487 at L27
- Causal circuit for R_V contraction is defensible (vs behavior transfer which is confounded)

**Expected artifact:**
- `DEC13_CAUSAL_SOURCE_PITCH.md` (Strategy Document) ✅
- `experiment_causal_source_hunt.py` (Localization Script) ✅
- Ranked list of heads/paths and effect sizes with controls
- Run directory under `results/phase3_attention/runs/`
- Short writeup summarizing circuit

**Method:**
1. **Source Hunt:** Run `experiment_causal_source_hunt.py` (L0-L27 sweep) to find the "seed".
2. **Head Ablation:** Head-level ablation sweep at L27 (32 heads).
3. **Path Patching:** From [Source] → L27.
4. **Controls:** random ablation, shuffled, wrong-layer.

---

## Secondary Objective A: Reconcile Behavior-Transfer Story

**Goal:** Create a single canonical doc that cleanly separates:

1. **Geometry transfers strongly** under V-proj patching (confirmed)
2. **Behavior transfer in n=300** exists (d~0.63) but is confounded by KV replacement and highly variable (28% show 0 transfer)
3. **True KV cache patching** does NOT reproduce the rumored strong behavior transfer (0-1 points in `TRUE_KV_CACHE_PATCHING_RESULTS.md`)

**Expected artifact:**
- `BEHAVIOR_TRANSFER_RECONCILIATION.md` in repo root
- Explicitly cites: `TRUE_KV_CACHE_PATCHING_RESULTS.md`, `KV_PATCHING_HISTORY.md`, `N300_RESULTS_ANALYSIS.md`

---

## Secondary Objective B: Analyze n=300 Variance

**Goal:** Understand what distinguishes high-transfer pairs (score ≥8) from no-transfer pairs (score=0).

**Why this matters:**
- 7% achieved strong transfer (≥8)
- 28% showed no transfer (=0)
- Understanding this variance reveals mechanism

**Expected artifact:**
- Analysis script or notebook
- Short writeup on distinguishing features (prompt length? semantic content? token structure?)

---

## NOT Today (Parked)

- **Attractor/hysteresis boundary tests:** Only after we have a reliable "toggle" circuit
- **Cross-architecture patching (Pythia):** Blocked until Mistral circuit is locked
- **Phase 2 eigenstate experiments:** Lower priority than circuit validation

---

## Success Criteria for Today

✅ **Primary:** Head-level effect sizes at L27 with controls, in auditable run dir  
✅ **Secondary A:** Single reconciliation doc that prevents paper overclaiming  
✅ **Secondary B:** At least exploratory analysis of n=300 variance  

---

## Execution Notes

**If running on RunPod:**
```bash
RUNPOD_HOST=198.13.252.9 RUNPOD_PORT=18147 bash scripts/runpod/push_repo.sh
ssh -p 18147 root@198.13.252.9 "cd /workspace/mech-interp-latent-lab-phase1 && source .venv/bin/activate && python -m src.pipelines.run --config configs/[NEW_CONFIG].json"
RUNPOD_HOST=198.13.252.9 RUNPOD_PORT=18147 bash scripts/runpod/pull_results.sh
```

**Config template for L27 circuit mapping:**
```json
{
  "experiment": "phase3_l27_head_ablation",
  "run_name": "default",
  "seed": 0,
  "results": { "root": "results", "phase": "phase3_attention" },
  "model": {
    "name": "mistralai/Mistral-7B-v0.1",
    "device": "cuda"
  },
  "params": {
    "target_layer": 27,
    "window": 16,
    "n_prompts": 50,
    "controls": ["random", "shuffled", "wrong_layer"]
  }
}
```

---

**OPUS 4.5 END — 2025-12-13**

