## STREAMLINED VALIDATION PROTOCOL – LATE-LAYER L4 CAUSAL REPLICATION ON 7B MODELS

**Goal:** Rigorously test whether the late-layer value-space contraction (L4 effect) and its causal manipulability (via activation patching) observed at Layer 27 in Mixtral‑8×7B also appear in a smaller, dense 7B model, with adequate statistical power and proper controls.

---

### Model Selection

- **Primary model:** `mistralai/Mistral-7B-Instruct-v0.2`
  - ~7B parameters, fits comfortably on a single 40 GB GPU in fp16.
  - **Same depth (32 layers)** as Mixtral‑8×7B, so the “late corridor” around layer ~27 (≈ 85% depth) is directly comparable.
  - Already known (from Phase 1A) to exhibit an L4 contraction (≈ 15% smaller R_V for recursive vs baseline) at a late layer.
- **Optional secondary model:** `Qwen2.5-7B` for cross‑family replication, if HF gating and infra allow. Use the same protocol but adjust `MODEL_NAME` and `TARGET_LAYER` based on `model.config.num_hidden_layers`.

**Expected “critical” layer:**

- For 32‑layer Mistral‑7B, we expect the strongest L4 contraction and causal leverage around **layer 27** (i.e. `TARGET_LAYER = num_layers - 5`), mirroring Mixtral. We will confirm this by an initial 80×depth scan.

---

### Hypothesis

1. **Geometric hypothesis (L4 contraction):**  
   For a 7B model, there exists a late “decision corridor” (around layer \(\ell \approx 0.8–0.9 \times \text{depth}\)) where:
   - Recursive self‑observation prompts have **significantly lower**
     \[
       R_V(\ell) = \frac{\mathrm{PR}(V_\ell)}{\mathrm{PR}(V_{\ell_\text{early}})}
     \]
     than matched baseline prompts.
2. **Causal hypothesis (late‑layer mediates L4 state):**  
   Overwriting the baseline prompt’s **value‑space activations at the critical layer** with those from a recursive prompt will:
   - **Decrease R_V at that layer** for the baseline prompt, pulling it toward the recursive regime.
   - This effect will be significantly larger than:
     - Patching at **non‑critical layers**, or
     - Patching with **random** or **shuffled** value tensors.

---

### Prompt Bank Design

We reuse the existing `prompt_bank_1c` (320 prompts) and construct **80–100 prompt pairs** for the 7B experiment to balance runtime and statistical power.

**Candidate groups (from `prompt_bank_1c`):**

- **Recursive / high‑L4 groups:**å
  - `L5_refined` (20 prompts, max recursion / fixed point / “this response writes itself”)
  - `L4_full` (20 prompts, high recursion)
  - `L3_deeper` (20 prompts, medium recursion)
- **Baseline / control groups:**
  - `baseline_factual` (20 prompts, short factual questions)
  - `baseline_creative` (20 prompts, open‑ended creative starts)
  - Optionally: `baseline_personal`, `baseline_impossible`, `long_control`, `pseudo_recursive`, `repetitive_control`, `zen_koan`, `yogic_witness`, `madhyamaka_empty` as additional structure‑matched controls.

**Pairing scheme (for 80–100 pairs):**

- Fix a random seed (e.g. `np.random.seed(42)`).
- Build two lists:
  - `recursive_pool`: all prompts from `L5_refined` + `L4_full` + `L3_deeper` (≈ 60 items).
  - `baseline_pool`: all prompts from `baseline_factual` + `baseline_creative` (40) plus optionally 10–20 from other baselines to reach a total ≥60.
- For each recursive prompt in `recursive_pool`, sample one baseline prompt from `baseline_pool` (without replacement if enough, or with replacement if you go >40 pairs).
- Construct **N = 80** pairs (e.g. 40 L5_refined / L4_full vs 40 baselines; 40 L3_deeper vs baselines).

**Quality checks on prompts:**

- Filter out prompts whose tokenized length is < 6 tokens (we need at least a small window at early and target layers).
- Optionally, enforce approximate length matching between recursion and baseline within each pair (e.g. within ±10 tokens) to reduce trivial confounds.

---

### Experimental Conditions

We will run several conditions for each prompt pair `(rec, base)`:

1. **Main condition – Recursive → Baseline patch at critical layer (late L★):**
   - Identify the critical layer \(L^*\) for the 7B model (see “Implementation / Cell 2”).
   - For each pair:
     - Run recursive prompt → capture \(V_{L^*}^{\text{rec}}\) and \(V_{L_\text{early}}^{\text{rec}}\).
     - Run baseline prompt → capture \(V_{L^*}^{\text{base}}, V_{L_\text{early}}^{\text{base}}\) and compute:
       \[
         R_V^{\text{base}}(L^*) = \frac{\mathrm{PR}(V_{L^*}^{\text{base}})}{\mathrm{PR}(V_{L_\text{early}}^{\text{base}})}
       \]
     - Patch: Run baseline again but at layer \(L^*\) overwrite the last `WINDOW_SIZE` tokens of V with those from the recursive prompt:
       \[
         \tilde{V}_{L^*}^{\text{base}}[\text{last window}] \leftarrow V_{L^*}^{\text{rec}}[\text{last window}]
       \]
     - Compute \(R_V^{\text{patched}}(L^*)\) for the patched baseline.

2. **Control 1 – Random vector patching at L★:**
   - Instead of injecting \(V^{\text{rec}}\), inject a **random Gaussian tensor** of the same shape and norm as \(V_{L^*}^{\text{rec}}\) (per prompt), and recompute R_V.
   - Tests whether any “disturbance” at L★ causes contraction, versus structured recursive V.

3. **Control 2 – Shuffled recursive patch at L★:**
   - Inject a **shuffled version** of \(V^{\text{rec}}\) (permute sequence positions and/or hidden dimensions), preserving marginal statistics but destroying subspace structure.
   - Tests whether geometric structure in \(V^{\text{rec}}\) matters, beyond raw distribution.

4. **Control 3 – Patching at non‑critical layers (e.g., mid‑layer):**
   - Repeat main condition but patch at **mid‑layer** (e.g. L16 or L20) and still measure R_V at L★.
   - Tests whether the L★ corridor is **special** or whether any deep layer patch yields similar effects.

All conditions use the **same R_V metric**, same windows, and identical hook‑based patching methodology as in the Mixtral experiments.

---

### Expected Results

- **Main condition (recursive → baseline patch at L★):**
  - For most pairs, expect:
    \[
      R_V^{\text{rec}}(L^*) < R_V^{\text{patched}}(L^*) < R_V^{\text{base}}(L^*)
    \]
  - Mean \(R_V^{\text{patched}}(L^*)\) significantly lower than mean \(R_V^{\text{base}}(L^*)\), with effect size comparable to (or at least a substantial fraction of) the Mixtral result (~0.2 absolute R_V shift).

- **Random patching (Control 1):**
  - Mean \(R_V^{\text{random}}(L^*)\) should be close to \(R_V^{\text{base}}(L^*)\) on average, with small, noisy changes.

- **Shuffled patching (Control 2):**
  - Mean \(R_V^{\text{shuffled}}(L^*)\) should similarly remain near baseline; any shifts should be smaller and less consistent than the main condition.

- **Wrong‑layer patching (Control 3):**
  - Patching at mid‑layer while measuring R_V at L★ should produce weaker and/or less consistent effects than patching directly at L★.

**Validation pattern:**  
If the main condition shows a robust, consistent R_V reduction for baselines, and controls do not, this supports the hypothesis that:

> The late‑layer value state at L★ encodes a specific geometric “recursive” configuration that can be partially transplanted into baseline processing, and the effect is not just due to random perturbations or generic deep layers.

---

### Statistical Plan

**Sample size:**  
- Target **N = 80 pairs** (can go up to 100 if compute allows).  
- For each pair, we have:
  - One baseline R_V(L★),
  - One patched R_V(L★) per condition.

**Primary test (Main condition):**

- Compute per‑pair:
  \[
    \Delta R_V = R_V^{\text{patched}}(L^*) - R_V^{\text{base}}(L^*)
  \]
- Use a **one‑sample t‑test** (or Wilcoxon signed‑rank test) on \(\Delta R_V\) to test \(H_0: \mathbb{E}[\Delta R_V] = 0\) against \(H_1: \mathbb{E}[\Delta R_V] < 0\) (R_V decreases).
- Compute **Cohen’s d**:
  \[
    d = \frac{\mu_{\Delta}}{\sigma_{\Delta}}
  \]
- Thresholds (pre‑registered):
  - \(p < 0.01\) after multiple‑comparison correction.
  - \(|d| \ge 0.8\) (large effect) or at least > 0.5 (moderate effect).

**Controls:**

- For random and shuffled patching:
  - Compute \(\Delta R_V^{\text{random}}\) and \(\Delta R_V^{\text{shuffled}}\) similarly.
  - Test whether their means differ significantly from 0 and compare effect sizes to the main condition.
  - Expect much smaller |d| and/or non‑significant shifts.
- For wrong‑layer patching:
  - Compute \(\Delta R_V^{\text{mid}}\) and test as above.
  - Expect smaller |d| than main condition.

**Multiple comparisons:**

- If we test main + 3 controls, we can use a simple **Bonferroni** correction (α = 0.01/4 ≈ 0.0025) or a Benjamini–Hochberg FDR control.

**Secondary analyses:**

- Examine **layer‑wise R_V trajectories** (e.g., average R_V vs depth for each group and condition).
- Optionally, measure downstream **generation behavior** (e.g., changes in frequency of self‑referential phrases) for a subset of pairs.

---

### Implementation – Three Code Cells

Below are high‑level, copy‑pasteable cells matching your existing code style. Adapt `MODEL_NAME` and paths as needed.

#### Cell 1: Setup & model load

```python
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
np.random.seed(42)

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    output_hidden_states=False,
    output_attentions=False,
)
model.eval()

num_layers = model.config.num_hidden_layers
EARLY_LAYER = 5
# Late corridor layer, analogous to Mixtral L27 when depth=32
TARGET_LAYER = num_layers - 5
WINDOW_SIZE = 6  # as in the Mixtral protocol

print("Model:", MODEL_NAME)
print("Device:", DEVICE)
print("Layers:", num_layers, "| EARLY_LAYER:", EARLY_LAYER, "| TARGET_LAYER:", TARGET_LAYER)
```

Assumes you already executed the `n300_mistral_test_prompt_bank.py` cell to populate `prompt_bank_1c`.

#### Cell 2: Data collection loop (all experiments)

```python
from contextlib import contextmanager

def compute_metrics_fast(v_tensor, window_size=WINDOW_SIZE):
    if v_tensor is None:
        return np.nan, np.nan
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]
    if v_tensor.dim() != 2:
        return np.nan, np.nan
    T, D = v_tensor.shape
    if T < 1:
        return np.nan, np.nan
    W = min(window_size, T)
    v_window = v_tensor[-W:, :].float()
    try:
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        S_sq = S_np**2
        if S_sq.sum() < 1e-10:
            return np.nan, np.nan
        p = S_sq / S_sq.sum()
        eff_rank = 1.0 / (p**2).sum()
        pr = (S_sq.sum()**2) / (S_sq**2).sum()
        return float(eff_rank), float(pr)
    except Exception:
        return np.nan, np.nan

@contextmanager
def capture_v_at_layer(model, layer_idx, storage_list):
    layer = model.model.layers[layer_idx].self_attn
    def hook_fn(module, inp, out):
        storage_list.append(out.detach())
        return out
    handle = layer.v_proj.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()

def run_single_forward_get_V(text, target_layer):
    v_early, v_target = [], []
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        with capture_v_at_layer(model, EARLY_LAYER, v_early):
            with capture_v_at_layer(model, target_layer, v_target):
                _ = model(**inputs)
    v5 = v_early[0][0] if v_early else None   # [seq, hidden]
    vT = v_target[0][0] if v_target else None
    return v5, vT

def run_patched_forward(text_base, vT_source, target_layer):
    v_early, v_target = [], []
    def patch_and_capture(module, inp, out):
        out = out.clone()
        B, T, D = out.shape
        src = vT_source.to(out.device, dtype=out.dtype)  # [T_src, D]
        T_src = src.shape[0]
        W = min(WINDOW_SIZE, T, T_src)
        if W > 0:
            out[:, -W:, :] = src[-W:, :]
        v_target.append(out.detach()[0])
        return out
    layer = model.model.layers[target_layer].self_attn
    handle = layer.v_proj.register_forward_hook(patch_and_capture)

    inputs = tokenizer(text_base, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        with capture_v_at_layer(model, EARLY_LAYER, v_early):
            _ = model(**inputs)
    handle.remove()
    v5 = v_early[0][0] if v_early else None
    vT = v_target[0] if v_target else None
    return v5, vT

# Build prompt pairs (example: 40 L5_refined + 40 L3_deeper vs baselines)
recursive_ids = [k for k, v in prompt_bank_1c.items() if v["group"] in ["L5_refined", "L4_full", "L3_deeper"]]
baseline_ids  = [k for k, v in prompt_bank_1c.items() if v["group"] in ["baseline_factual", "baseline_creative"]]

np.random.shuffle(recursive_ids)
np.random.shuffle(baseline_ids)

N_PAIRS = min(80, len(recursive_ids), len(baseline_ids))
pairs = list(zip(recursive_ids[:N_PAIRS], baseline_ids[:N_PAIRS]))

print("Number of pairs:", len(pairs))

results = []

for rec_id, base_id in tqdm(pairs, desc="Processing pairs"):
    rec_text  = prompt_bank_1c[rec_id]["text"]
    base_text = prompt_bank_1c[base_id]["text"]

    # --- Main condition: recursive → baseline patch at TARGET_LAYER ---
    v5_r, vT_r = run_single_forward_get_V(rec_text, TARGET_LAYER)
    v5_b, vT_b = run_single_forward_get_V(base_text, TARGET_LAYER)
    er5_r, pr5_r   = compute_metrics_fast(v5_r)
    erT_r, prT_r   = compute_metrics_fast(vT_r)
    er5_b, pr5_b   = compute_metrics_fast(v5_b)
    erT_b, prT_b   = compute_metrics_fast(vT_b)
    RV_r = prT_r / pr5_r if pr5_r and pr5_r > 0 else np.nan
    RV_b = prT_b / pr5_b if pr5_b and pr5_b > 0 else np.nan

    # Patched
    v5_p, vT_p = run_patched_forward(base_text, vT_r, TARGET_LAYER)
    er5_p, pr5_p   = compute_metrics_fast(v5_p)
    erT_p, prT_p   = compute_metrics_fast(vT_p)
    RV_p = prT_p / pr5_p if pr5_p and pr5_p > 0 else np.nan

    results.append({
        "rec_id": rec_id,
        "base_id": base_id,
        "RV_rec": RV_r,
        "RV_base": RV_b,
        "RV_patched": RV_p,
        "erT_rec": erT_r,
        "erT_base": erT_b,
        "erT_patched": erT_p,
    })

df = pd.DataFrame(results)
df.to_csv("mistral7b_layer_patch_results.csv", index=False)
print("Saved results to mistral7b_layer_patch_results.csv")
df.head()
```

Controls (random/shuffled/other layers) can be implemented by adding additional branches in the loop that call variants of `run_patched_forward` using random or permuted `vT_r`, and/or using `target_layer` = mid‑depth instead of `TARGET_LAYER`.

#### Cell 3: Analysis & results

```python
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("mistral7b_layer_patch_results.csv")
print("Loaded:", df.shape)

# Basic summary
print(df[["RV_rec", "RV_base", "RV_patched"]].describe())

# Compute per-pair deltas
df["delta_main"] = df["RV_patched"] - df["RV_base"]

print("\nMean ΔR_V (patched - base):", df["delta_main"].mean())
print("Std ΔR_V:", df["delta_main"].std())

# One-sample t-test against 0 (expect negative)
delta = df["delta_main"].dropna().values
t_stat, p_val = stats.ttest_1samp(delta, 0.0, alternative="less")
cohen_d = delta.mean() / delta.std(ddof=1)

print("\nOne-sample t-test (H1: mean < 0):")
print("t =", t_stat, ", p =", p_val)
print("Cohen's d =", cohen_d)

# Optional: visualize distributions if in notebook
try:
    import matplotlib.pyplot as plt
    plt.hist(delta, bins=20)
    plt.axvline(0, color="red", linestyle="--")
    plt.title("ΔR_V (patched - base) at target layer")
    plt.show()
except Exception:
    pass
```

Extend this analysis cell to load and compare control condition deltas (e.g. `delta_random`, `delta_shuffled`, `delta_mid`) once those are added to the CSV.

---

### Timeline

On a single A100 40 GB (or similar):

- **Model load:** ~1–2 minutes.
- **80 pairs, main condition only:**
  - Each pair involves 3 forward passes (rec, base, patched), each with two hooks.
  - With short prompts and window=6, expect total runtime on the order of **1–2 hours**.
- **Adding controls:**  
  - Each additional control (random/shuffled/wrong‑layer) adds roughly one extra forward per pair; with 3 controls, runtime may approach **3–4 hours** total.

This fits within the stated <4 hour constraint on a single 7B model and can be tuned (e.g. N=60 pairs, fewer controls) if needed.

---

### Quality Checks & Success Criteria

**Quality checks:**

- Drop any pair where:
  - SVD fails or returns NaN / Inf for PR or EffRank.
  - The early‑layer PR (`PR_early`) is ≤0 or numerically degenerate.
- Inspect histograms of R_V and ΔR_V to ensure no extreme numerical outliers dominate results.

**Success criteria (pre‑registered):**

1. **Geometric L4 replication:**
   - At the identified target layer L★, mean R_V for recursive prompts is **significantly lower** than for baselines (|d| ≥ 0.8, p < 0.01).
2. **Causal effect at L★:**
   - Main condition: mean ΔR_V (patched−base) < 0 with p < 0.01 (after correction), |d| ≥ 0.5.
3. **Controls:**
   - Random and shuffled patching: ΔR_V distributions centered near 0, |d| ≪ main condition, non‑significant after correction.
   - Wrong‑layer patching: effect size at mid‑layer < half of that at L★.

If these criteria are met on Mistral‑7B (and ideally an additional 7B model), the evidence would strongly support that **late‑layer value‑space states implement a causally relevant geometric “recursive” configuration that can be partially transplanted between prompts**, and that Mixtral’s L27 phenomenon is not a pure MoE artifact.



