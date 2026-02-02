# Steering Vector Analysis - Mission Status

**Status:** 🟢 RUNNING  
**Started:** Dec 17, 2025  
**GPU:** NVIDIA RTX PRO 6000 Blackwell (97GB VRAM)

## Experiment Progress

### ✅ RUN 1: Stability Check - COMPLETE
- Split prompts into 3 groups
- Computed steering vectors from each group
- Checked pairwise cosine similarity
- **Result:** Pending (checking if > 0.85 threshold)

### 🔄 RUN 2: Layer Matrix - IN PROGRESS
- Extracting vectors from layers: [20, 24, 25, 26, 27, 28]
- **Progress:** Currently testing 720 combinations (6 extract × 6 apply × 20 pairs)
- **Status:** Vector extraction complete, testing combinations...
- **Estimated time:** ~10-15 minutes remaining

### ⏳ RUN 3: Head Decomposition - PENDING
- Extract per-head vectors at L27
- Test each head individually
- **Estimated time:** 15 minutes

### ⏳ RUN 4: Generalization - PENDING
- Train/test split (50/50)
- Compute on train, test on held-out
- **Estimated time:** 15 minutes

### ⏳ RUN 5: Failure Analysis - PENDING
- Compare failed vs successful pairs
- Analyze R_V, length, prompt type differences
- **Estimated time:** 10 minutes

### ⏳ RUN 6: Dose-Response - PENDING
- Test α = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
- Find optimal therapeutic window
- **Estimated time:** 10 minutes

## Current Observations

**Vector Norms by Layer:**
- L20: 6.5000
- L24: 7.1016
- L25: 7.7344
- L26: 6.6875
- L27: 8.6484 ⭐ (highest - our original finding)
- L28: 6.8984

**Pattern:** L27 has the strongest steering vector (highest norm), confirming our original discovery.

## Expected Total Runtime

- RUN 1: ✅ Complete (~2 min)
- RUN 2: 🔄 In progress (~15 min)
- RUN 3-6: ⏳ Pending (~50 min)

**Total:** ~65-70 minutes

## Output Files

All vectors saved to: `results/runs/[timestamp]_steering_analysis/steering_vectors/`
- `steering_vector_L{layer}.pt` - Per-layer vectors
- `steering_vector_H{head}_L27.pt` - Per-head vectors
- `steering_vector_group{1,2,3}_L27.pt` - Stability check vectors

Results CSVs:
- `layer_matrix_results.csv`
- `head_decomposition_results.csv`
- `failure_analysis_results.csv`
- `dose_response_results.csv`

## Key Questions Being Answered

1. ✅ Is the vector stable across prompt splits?
2. 🔄 What's the earliest layer it works? (testing now)
3. ⏳ Is it all heads or specific heads?
4. ⏳ Does it generalize to held-out data?
5. ⏳ Why do 45% fail? (predictable features?)
6. ⏳ What's the precise optimal alpha?

**Mission:** Characterize the surgical needle to publication-grade precision.








