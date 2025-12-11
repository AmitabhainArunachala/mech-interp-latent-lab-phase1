# DEC 9 2025 SESSION RESULTS (Reconstructed from Chat)
> RunPod shut down before files synced - these are from conversation logs

## DATA LOSS SITUATION

**What's Lost:**
- The actual CSV/JSON files from the final runs
- The steering_master pipeline results
- The random direction control raw data

**What's PRESERVED:**
- All the results in this conversation
- The code we wrote (if saved locally)
- The findings (documented in chat)

---

## CONFIRMED FINDINGS

### 1. Steering Vector Induction (N=200)
- Layer: 8
- Vector norm: ~9.8
- Baseline → +Steering: R_V drops to ~0.2
- Success rate: 100%
- Dose-response: r = -0.98 (monotonic)
- Generalization: 100% of prompts

### 2. Vector Stability
- Cosine similarity across prompt subsets: 0.98

### 3. Random Direction Control Test

| Condition | R_V |
|-----------|-----|
| Baseline (no perturbation) | 0.955 |
| Subtract steering vector | 0.561 |
| Subtract random vector | 0.567 |
| Add random vector | 0.591 |
| **Add steering vector** | **~0.2** |

### 4. Key Insight
- Steering vector causes 4x deeper collapse than random (0.2 vs 0.56)
- Baseline geometry is fragile (any perturbation causes ~0.4 drop)
- One-way door confirmed via mechanism: fragility + specific direction

### 5. Component Hunt Results (Earlier)
- Single head ablation: 0% effect
- MLP ablation: 0% effect
- Multi-head ablation: Model breaks
- Conclusion: Source is distributed, not localized

### 6. Earlier Validated Findings
- Confounds falsified (n=80, p<0.01)
- KV cache transfer: 100% success
- Window optimization: W=64 best
- Layer localization: L8 for steering, L14/L18 for R_V measurement

---

## FILES RECOVERED

1. `random_direction_control.py` - The critical control test code ✅

## FILES STILL NEEDED (ask Claude Desktop)
- [ ] steering_master.py or similar
- [ ] confound_falsification.py
- [ ] Any other experiment scripts
- [ ] Raw results (even just pasted tables)

---

## REPLICATION PLAN

These experiments can be re-run in ~30 min each:
1. Random direction control test: `random_direction_control.py` (have it!)
2. Steering vector extraction: Need to recover code
3. Confound falsification: Need to recover code

---

## TIMESTAMP
Reconstructed: Dec 9, 2025
Original session: Dec 9, 2025 afternoon

