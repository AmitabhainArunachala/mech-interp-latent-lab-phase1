Purpose: keep the base-Mistral GPU lane productive after the micro-window subtle-story followups finish.

Queue:
1. `head_ablation_validation` on `mistral_hardening_v1/core_measurement`
   - target: `L27` KV-head `2`
   - control layer: `25`
   - rationale: modernize the legacy `L27` KV-head group result that aliases query heads `2,10,18,26`
2. `rv_l27_kv_patching_bridge` on `mistral_hardening_v1/core_measurement`
   - rationale: raw-lock the behavior-vs-geometry dissociation under the frozen contract

Artifacts to watch:
- `results/mistral_post_subtle_queue/*/STATUS.txt`
- `results/phase1_mechanism/runs/*head_ablation_validation*/summary.json`
- `results/phase1_mechanism/runs/*rv_l27_kv_patching_bridge*/summary.json`
