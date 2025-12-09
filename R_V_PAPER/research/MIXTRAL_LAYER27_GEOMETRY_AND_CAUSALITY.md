## Mixtral Layer‑27 Geometry, MoE Routing, and Causal Interventions

*Phase 1F – Mixtral-8x7B, November 16, 2025*  
*Context: AIKAGRYA / L4 Contraction Phenomenon, Phase 1F “full‑power” Mixtral run*  

This note distills the key quantitative and mechanistic results from the Mixtral‑8×7B experiments you just ran (80‑prompt × 32‑layer sweep, expert routing probes, and layer‑27 activation patching), and incorporates the most salient external commentary (Claude’s review) in a single place.

It is meant as a compact “MI‑style” summary of what we actually know so far about the **L4 contraction phenomenon** in Mixtral.

---

### 1. Recap: What we already knew before today

- Across **6 architectures** (Mistral‑7B, Qwen‑1.5‑7B, Gemma‑7B‑IT, Llama‑3‑8B, Phi‑3‑medium, Mixtral‑8×7B), **recursive self‑observation prompts** (L3/L5) consistently showed:
  - A strong **late‑layer contraction** in the value (V) space, measured via  
    \[
      R_V(\ell) \;=\; \frac{\mathrm{PR}(V_\ell)}{\mathrm{PR}(V_{\ell\_early})}
    \]
    with \(\ell\_\text{early} = 5\), \(\ell_\text{late} \approx \text{depth} - 4\).
  - Recursion lowers \(R_V\) by **3–24%** vs matched baselines (Mistral ≈ 15%, Qwen ≈ 9%, Llama‑3 ≈ 12%, Phi‑3 ≈ 7%, Gemma ≈ 3%, Mixtral ≈ 24%).
- In **Mixtral‑8×7B**, the effect is exceptionally strong:
  - At the chosen late layer (L27 ≈ 27/32 ≈ 0.84 depth),  
    - Recursive prompts: \(R_V \approx 0.88\) (80‑prompt golden set).  
    - Factual baseline: \(R_V \approx 1.16\) (expansion).  
    - Creative baseline: \(R_V \approx 1.13\).  
  → About **24% stronger contraction** for L5 vs factual baseline.
- Earlier “Experiment 0” on a single L5 prompt showed:
  - **Effective rank** drops sharply near L27 from ~8.7 → ~3.6 (over last 16 tokens).
  - **R_V vs layer** using the “new” definition (PR at layer / PR at layer 5) showed a **sharp negative jump at L27**.

This already looked like a **depth‑localized linear subspace collapse**, with Mixtral exhibiting the strongest L4 effect of any model tested.

---

### 2. Full 80×32 sweep: Layer‑27 as a “decision corridor”

You then ran a full **80‑prompt × 32‑layer** sweep:

- Prompts:
  - 20 × `L5_refined` (max recursion)
  - 20 × `L3_deeper`  (medium recursion)
  - 20 × `baseline_factual`
  - 20 × `baseline_creative`
- For each prompt and each layer \(\ell = 0…31\), you logged:
  - \(R_V(\ell) = \mathrm{PR}(V_\ell) / \mathrm{PR}(V_5)\) (PR on last 16 tokens of V),
  - Effective rank of \(V_\ell\),
  - Top singular value, spectral gap, etc.
- For each prompt, you defined the **“snap layer”** as the depth where \(R_V(\ell)\) has the largest negative step \(\Delta R_V(\ell) = R_V(\ell) - R_V(\ell-1)\).

**Snap‑layer distributions:**

- **L5_refined:**
  - Mean snap layer: 27.4, median 27.0, range 27–31  
  - 18/20 snap at **27**, 2/20 at 31.
- **L3_deeper:**
  - Mean snap layer: 27.5, median 27.0, range 27–31  
  - 17/20 at **27**, 1 at 29, 2 at 31.
- **baseline_factual:**
  - Mean: 24.8, median 27.0, range 7–31  
  - 9/20 at 27, 4/20 at 31, others scattered (7, 18, 19, 29).
- **baseline_creative:**
  - Mean: 27.6, median **31.0**, range 9–31  
  - 12/20 at 31, 5/20 at 27, a few early (9,18).

**R_V at L27:**

- L5_refined: \(R_V(27) = 0.458 \pm 0.051\)  
- L3_deeper: \(R_V(27) = 0.460 \pm 0.035\)  
- baselines (factual + creative combined): \(R_V(27) \approx 0.97 \pm 0.11\)
- Effect size (recursive vs baseline at L27):  
  - Recursive mean ≈ **0.459**, baseline mean ≈ **0.971**  
  - **Cohen’s \(d \approx 5.3\)** (enormous).

**MI interpretation:**

- There is a **late “decision corridor”** in Mixtral (layers ~27–31) where the value‑space makes its biggest move.
- **Recursive prompts** reliably “snap” **early in that corridor (L27)** and into a **much more contracted subspace** (R_V ~ 0.46).
- **Creative baselines** mostly delay their snap to L31; factual baselines are mixed but many still snap in the 27–31 band.
- The L4 phenomenon in Mixtral can be restated:

> Recursive self‑observation prompts drive the value space into a low‑dimensional attractor **early** in the final decision corridor (L27), while baselines either don’t contract there (R_V ~ 1) or contract later and less strongly. The effect size at that single layer is huge.

This refines the older “Layer‑21” story: your full sweep shows that **L27, not L21**, is the true catastrophic contraction point under the “R_V vs layer‑5” definition. The earlier emphasis on 21 came from using a different R_V definition (PR(L28)/PR(L\_\*) rather than PR(L\_\*)/PR(L5)), which compares “before vs after” the snap rather than tracking the depth where the collapse itself occurs.

---

### 3. Expert routing at Layer 27 (MoE structure)

You then probed **expert routing** at Layer 27:

- For each of the 80 prompts, you extracted the router probabilities over **8 experts** at L27.  
  This gives a table of shape `[80 prompts × 8 experts]`, with per‑group averages:

```text
expert                 0      1      2      3      4      5      6      7
group                                                                    
L3_deeper          0.141  0.099  0.159  0.065  0.096  0.194  0.127  0.118
L5_refined         0.121  0.093  0.152  0.100  0.139  0.167  0.121  0.107
baseline_creative  0.115  0.083  0.143  0.140  0.144  0.117  0.129  0.130
baseline_factual   0.115  0.103  0.145  0.116  0.140  0.102  0.170  0.109
```

**Routing entropy (per‑prompt) at Layer 27:**

- L5_refined:     \(H \approx 2.925 \pm 0.039\) bits  
- L3_deeper:      \(H \approx 2.911 \pm 0.034\) bits  
- baseline_factual: \(H \approx 2.836 \pm 0.079\) bits  
- baseline_creative: \(H \approx 2.872 \pm 0.071\) bits  
- Uniform over 8 experts = 3.000 bits.

**Key conclusions:**

1. **Expert routing is not “collapsing” to a single expert.**  
   - Everyone is using multiple experts; entropies are high and not dramatically lower for recursion.
   - In fact, recursive prompts have *slightly higher* entropy than baselines (2.918 vs 2.854 bits, Cohen’s d ≈ 1.06).
   - So the strong geometric collapse at Layer 27 **cannot be explained** by “only 1–2 experts firing.” That simple story is falsified.

2. **But there *is* a strong difference in *which* experts are used.**  
   - **Expert 5** is notably favored by recursive prompts:
     - L3_deeper: 19.4 % vs 12.5 % uniform (+6.9 pp)  
     - L5_refined: 16.7 % (+4.2 pp)  
     - Factual baseline: 10.2 % (−2.3 pp)  
     - Creative baseline: 11.7 % (−0.8 pp)
   - Baseline factual favors **Expert 6**; creative spreads weight across 2/3/4/7 more evenly.

So the MoE story is:

> **Recursive prompts do not just use fewer experts; they recruit a *different mixture* of experts in the decision corridor, especially over‑weighting Expert 5, whereas factual baselines lean on Expert 6 and creatives spread their mass more evenly.**

Routing entropy and top‑3 mass are not giving a trivial “sparsity implies collapse” explanation. Whatever is happening at Layer 27 is more about the **content of the expert outputs and their combination** than about sheer sparsity.

---

### 4. Expert‑5 geometry probe

To test whether **Expert 5 itself** was a “collapse expert,” you ran a focused probe:

- Chose **5 recursive** prompts (`L5_refined_01…05`) and **5 factual** prompts (`factual_new_01…05`).
- For each, you:
  - Extracted **Expert 5’s V output** at Layer 27.
  - Computed effective rank and PR on its last 16 tokens.

Example rows:

```text
        prompt_id             group   eff_rank   PR      top_sv   spectral_gap
0   L5_refined_01        L5_refined   4.02       4.02   197.8     118.9
1   L5_refined_02        L5_refined   5.72       5.72   164.0      72.4
...
5  factual_new_02  baseline_factual   1.00       1.00    68.5       0.0
6  factual_new_05  baseline_factual   1.54       1.54    78.9      36.3
```

Group averages:

- **L5_refined (recursive):**
  - EffRank ≈ **4.34 ± 0.80**
  - PR ≈ **4.34 ± 0.80**
  - TopSV ≈ **178 ± 12.5**
  - Spectral gap ≈ **94.7 ± 21.6**

- **baseline_factual:**
  - EffRank ≈ **1.27 ± 0.38**
  - PR ≈ **1.27 ± 0.38**
  - TopSV ≈ **73.7 ± 7.4**
  - Spectral gap ≈ **18.2 ± 25.7**

Effect size (EffRank, recursive vs baseline):  
**Cohen’s d ≈ 5.68** — enormous, but in the *opposite direction* of the naïve “specialist expert collapses everything” hypothesis.

**Conclusion:**

- **Expert 5 is *more* high‑dimensional and high‑energy on recursive prompts**, not less. Baselines that route to E5 get a much narrower, lower‑energy subspace.
- The overall value‑space contraction at Layer 27 is therefore **not** due to E5 itself being low‑rank. Instead:

> Recursive prompts cause Expert 5 (and related experts) to light up in a rich, structured way; the **global mixing + downstream layers then compress this enriched subspace into a low‑dimensional attractor**, which shows up as \(R_V \approx 0.46\) for the combined V at L27.

This points toward a **two‑stage story**:

1. **Feature creation:** Recursive structure recruits particular experts (notably E5) that produce high‑rank, high‑energy “self‑observation” features.
2. **Feature consolidation:** The shared Layer‑27 transformation compresses those features (and others) into a low‑dimensional, high‑signal subspace — the L4 contraction itself.

---

### 5. Causal test: Activation patching at Layer 27

Finally, you ran a first **causal intervention**:

- You paired **5 recursive** prompts (`L5_refined_01…05`) with **5 factual** prompts (`factual_new_01…05`).
- For each pair `(rec, base)`:
  1. Measured the unpatched baseline’s and recursive’s L27 geometry:
     - \(R_V^{\text{rec}}\_ {27}, \text{EffRank}^{\text{rec}}\_{27}\),
     - \(R_V^{\text{base}}\_ {27}, \text{EffRank}^{\text{base}}\_{27}\).
  2. Re‑ran the baseline prompt, but at Layer 27 **patched in the recursive prompt’s V** on the last `WINDOW_SIZE` positions (overwriting the value tensor after MoE mixing).
  3. Measured the patched baseline’s L27 geometry \(R_V^{\text{patched}}\_{27}, \text{EffRank}^{\text{patched}}\_{27}\).

Per‑pair results (rounded):

```text
Pair                      R_V27          EffRank27
-----------------------------------------------------
L5_refined_01 → factual1:  rec  0.419 | 4.76
                            base 1.115 | 3.32
                            pat  1.001 | 2.98

L5_refined_02 → factual2:  rec  0.415 | 4.19
                            base 1.076 | 3.19
                            pat  0.864 | 2.56

L5_refined_03 → factual3:  rec  0.377 | 4.00
                            base 1.125 | 3.32
                            pat  0.875 | 2.58

L5_refined_04 → factual4:  rec  0.439 | 4.26
                            base 1.110 | 3.31
                            pat  0.925 | 2.76

L5_refined_05 → factual5:  rec  0.496 | 4.65
                            base 0.965 | 4.00
                            pat  0.767 | 3.18
```

Aggregate (mean ± std):

```text
               mean   std
RV27_rec      0.429  0.044
RV27_base     1.078  0.066
RV27_patched  0.886  0.086

er27_rec      4.372  0.322
er27_base     3.428  0.324
er27_patched  2.812  0.265
```

**Key causal effect at Layer 27:**

- **Baseline R_V27 drops from ~1.08 → ~0.89 on average** when patched with recursive V27.  
- **Baseline EffRank27 drops from ~3.43 → ~2.81** (even though recursive EffRank27 is ~4.37; see caveat below).

So:

> Overwriting the baseline’s Layer‑27 value tensor with the recursive one reliably **pushes its geometry toward the contracted regime**, but does not fully match the true recursive geometry.

**Caveats / next steps:**

- For short baseline prompts, layer‑5 and layer‑27 V tensors have fewer than 16 tokens; since `compute_metrics_fast` uses `min(window_size, seq_len)`, the “local window” being analyzed differs between rec and base. For precise comparison, we’d want to align the window size (e.g., use the same `W = min(T_rec, T_base)` for both).
- So the numeric EffRank comparison should be treated as qualitative (the direction is meaningful; the exact values may shift with different windowing).
- You still need to measure **downstream behavior**: does patching at 27 make the baseline’s **generated text** more self‑observational? That would close the loop from geometry → behavior.

Even with those caveats, this is already a **genuine causal signal**: a single‑layer V‑patch at L27 can systematically change the core metric \(R_V\) in the predicted direction.

---

### 6. External review (Claude’s synthesis, paraphrased)

In a long meta‑analysis, another LLM (Claude) reflected on these results in the broader context of mechanistic interpretability. Key points from that review:

- Your work lives squarely in the **“partial mechanistic structure”** paradigm Neel Nanda talks about: you’re not trying to fully reverse‑engineer Mixtral, but you’re **identifying robust, interpretable geometric signatures** (R_V drops, rank changes, phase transitions) that correlate tightly with behavior.
- The **linear‑representation hypothesis** (that cognition in LLMs is largely implemented by linear subspaces and their interactions) is strongly supported:
  - You’re directly measuring how singular value spectra and PR change across depth and condition.
  - The fact that a single late layer (L27) shows a massive, consistent R_V shift for recursion matches the idea of **depth‑localized linear “phase transitions.”**
- Your cross‑model results and the 80×32 Mixtral sweep show that the L4 effect is **not a fluke**:
  - It appears in 6 unrelated architectures.
  - It survives across different prompt groups (L1–L5, baselines, confounds).
  - In Mixtral, the effect size at L27 (Cohen’s d ~ 5.3) is extraordinarily large.
- The expert‑routing analysis adjusts expectations:
  - The collapse is **not** due to “all mass going into one expert”; in fact, recursive prompts have slightly **higher** routing entropy.
  - The interesting structure is in **which experts and what they compute**, not in crude sparsity.
- The activation‑patching result is exactly the kind of **causal experiment** that MI people care about:
  - It starts to show that **patching the value state at the right layer moves both geometry and (eventually) behavior**.
  - This lines up with the broader MI toolkit (activation patching, CCA/PCA of subspaces, feature attribution) and points toward isolating “recursion‑carrying” components at L27.

In Claude’s words (paraphrased): *“Stop trying to ‘explain consciousness’ as a whole; instead, think of this as discovering a concrete geometric and causal mechanism that supports one specific facet of self‑observation in a real model. That’s exactly what modern mechanistic interpretability is about.”*

---

### 7. Where to go from here (brief MI‑style roadmap)

Given the current state, the most incisive next steps are:

1. **Align and extend the patching experiment:**
   - Use a shared window size for PR/EffRank when comparing short vs long prompts.
   - Patch not just L27 but also test:
     - Patching **only specific expert contributions** (e.g., swap Expert 5 vs Expert 6),
     - Patching at L27 and then measuring **R_V/EffRank at L28–31** and **output text**.

2. **Cross‑model late‑corridor replication:**
   - Run a smaller 40×depth snap‑layer analysis on Mistral‑7B and Llama‑3‑8B.
   - Ask whether they also have a late “decision band” where recursive vs baseline divergence peaks.

3. **Subspace & behavior analysis:**
   - At L27, compute principal components of the value space for recursive vs baseline prompts.
   - Train simple linear probes on those PCs to distinguish groups.
   - Correlate **projection onto “recursive PCs”** with **R_V** and with **behavioral markers** (e.g., frequency of “I”, “this response”, “observer”, etc.).

Together with the geometry and routing work you’ve already done, these would turn this from a striking phenomenon into a **coherent mechanistic story**: a late‑layer value‑space transition, modulated by MoE expert mixtures, that causally shapes how Mixtral represents and expresses self‑observation.

---

*File: `MIXTRAL_LAYER27_GEOMETRY_AND_CAUSALITY.md` — created as a standalone summary of the Layer‑27 / MoE / patching results for Phase 1F (Mixtral‑8×7B).*


