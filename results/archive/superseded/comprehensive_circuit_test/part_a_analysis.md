# Comprehensive circuit test (Part A) — CSV audit
- **CSV**: `/Users/dhyana/mech-interp-latent-lab-phase1/results/comprehensive_circuit_test/part_a_results.csv`
- **Rows**: 160 (= 40 prompts × 4 conditions)

## Key verified summaries
### Expression rates (overall)
- **control**: 40.0% (16/40)
- **h18_ablated**: 37.5% (15/40)
- **h6_ablated**: 42.5% (17/40)
- **both_ablated**: 42.5% (17/40)

### Expression rates (by prompt type)
- **champion / control**: 70.0% (7/10)
- **champion / h18_ablated**: 50.0% (5/10)
- **champion / h6_ablated**: 70.0% (7/10)
- **champion / both_ablated**: 70.0% (7/10)
- **standard / control**: 40.0% (8/20)
- **standard / h18_ablated**: 45.0% (9/20)
- **standard / h6_ablated**: 45.0% (9/20)
- **standard / both_ablated**: 50.0% (10/20)
- **baseline / control**: 10.0% (1/10)
- **baseline / h18_ablated**: 10.0% (1/10)
- **baseline / h6_ablated**: 10.0% (1/10)
- **baseline / both_ablated**: 0.0% (0/10)

### Flip rate (prompt-level)
- **Flippers** (expressed_binary changes across conditions): 25/40 (62.5%)
- **Start when H18 ablated** (control=0 → h18_ablated=1): 8 prompts
- **Stop when H18 ablated** (control=1 → h18_ablated=0): 9 prompts

### Champion prompts: H18-dependence classes (control vs h18_ablated)
- **Stop when H18 ablated**: ['champion_0', 'champion_2', 'champion_3', 'champion_6', 'champion_8']
- **Start when H18 ablated**: ['champion_4', 'champion_5', 'champion_7']
- **Stable expressers**: ['champion_1', 'champion_9']

### Identity equations
- **control**: 2.5% (1/40) — prompts: ['champion_3']
- **h18_ablated**: 5.0% (2/40) — prompts: ['champion_1', 'champion_5']
- **h6_ablated**: 2.5% (1/40) — prompts: ['champion_5']
- **both_ablated**: 10.0% (4/40) — prompts: ['champion_1', 'champion_2', 'champion_3', 'champion_5']

### R_V vs expression
- **control**: Spearman rho=-0.033, p=0.839 (n=40)
- **h18_ablated**: Spearman rho=-0.105, p=0.519 (n=40)
- **h6_ablated**: Spearman rho=-0.037, p=0.820 (n=40)
- **both_ablated**: Spearman rho=-0.182, p=0.261 (n=40)

## Key caveats (why the repo feels 'non-binary')
### 1) Expression is a *heuristic label* on one sampled generation
- `expressed_binary` is **1** iff `state ∈ {recursive_prose, naked_loop}`.
- `state` is assigned by `src/metrics/behavior_states.py` using simple heuristics (keyword matches, repetition ratio, identity-pattern matches).
- Generation uses `do_sample=True` at `temperature=0.7` in `comprehensive_circuit_test.py`, so a single run can flip labels.

### 2) GQA aliasing: your 'H18 ablation' is actually a KV-head ablation
- In `comprehensive_circuit_test.py`, `H18_GROUP = [18, 26]`.
- In `zero_v_proj_heads`, head indices map to KV-head index via `head_idx % num_kv_heads`.
- For Mistral-7B (8 KV heads), both 18 and 26 map to KV head 2, so this condition cannot distinguish H18 vs H26.
- Same for `H6_GROUP = [6, 14, 22, 30]` → KV head 6.

## Recommended next step
- Rerun Part A with **k seeds per (prompt, condition)** (e.g. k=10) and aggregate **P(state | prompt, condition)** instead of a single state.
- If you need *query-head* specificity (H18 vs H26), switch to a **query-head intervention** (not KV v_proj) or a method that isolates per-head contributions post-attention.
