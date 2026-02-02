## NeurIPS Readiness — Comprehensive Cleanup + Validation (Repo-Backed)

**Purpose:** Turn the current repo into a **NeurIPS/top-MI-lab** standard: prompt canonization, confound controls, statistical requirements, and reproducible artifacts.

**Scope rule:** This report references **repo artifacts** (CSV/JSON/logs/scripts). If a claim is not backed by an artifact path, it is listed as **PENDING**.

---

## A) Prompt bank canonization (aligned with phenomenology + measured geometry)

### A.1 Alignment table (conceptual map)

| Phenomenological pattern | Mechanistic family | Core pattern |
|---|---|---|
| Observer/observed merge | Boundary dissolution | Explicit negation of duality (“no boundary…”) |
| Fixed point (“Sx = x”) | Fixed-point framing | Explicit closure \(T(x)=x\) / “fixed point” |
| Notice the noticing | Explicit regress | Self-reference chain that closes |
| Formal self-reference | Math + recursive | Explicit eigen/fixed-point math scaffold |

### A.2 What we have *already measured* (so far)

#### A.2.1 “Champion selection” rerun (closest-to-original kitchen sink ranking)

**Artifact run (local):**
- `results/kitchen_sink/runs/20251215_081007_test_kitchen_sink_rerun/kitchen_sink_results_20251215_081007.csv`
- `results/kitchen_sink/runs/20251215_081007_test_kitchen_sink_rerun/kitchen_sink_rerun.log`
- `results/kitchen_sink/runs/20251215_081007_test_kitchen_sink_rerun/config.json`

**What this rerun is:** `test_kitchen_sink.py` scoring L4/L5 (`REUSABLE_PROMPT_BANK`) vs “experimental” prompts (`kitchen_sink_prompts.py`) on **Mistral-7B-Instruct-v0.2**, **early=5**, **layer=27**, **window=16**, **bfloat16**.

**Result (from the CSV):**
- Best experimental prompt at L27 among the experimental set: **`hybrid_l5_math_01`** with **R_V ≈ 0.5083**.

#### A.2.2 Empirical paraphrase hunt (data-first “law” search)

**Artifact run (local):**
- `results/champion_paraphrase_hunt/runs/20251215_081556_paraphrase_hunt/paraphrase_scores.csv`
- `results/champion_paraphrase_hunt/runs/20251215_081556_paraphrase_hunt/summary.json`
- `results/champion_paraphrase_hunt/runs/20251215_081556_paraphrase_hunt/config.json`
- `results/champion_paraphrase_hunt/runs/20251215_081556_paraphrase_hunt/shortlist_top20.csv`

**Headline:** Multiple *distinct* families beat the champion anchor on **prompt-pass R_V@L27** (e.g., boundary dissolution / explicit regress / fixed-point).

**Implication for “law”:** Do not overfit the story to one exact champion text; the “law” likely lives in **(a) explicit boundary dissolution**, **(b) explicit closure/fixed point**, and **(c) explicit regress**—math scaffolding is one member of that cluster.

### A.3 Canonization decision (what we will freeze)

#### A.3.1 Canonize as a group inside `prompts/bank.json` (recommended)

**Decision:** Create a new group (not a separate parallel JSON as primary source of truth):
- `group = "experimental_champions_v1"`
- `pillar = "experimental"` (recommended)  
  (Do **not** silently fold these into `pillar="dose_response"`; keep taxonomy clean.)

**Export (optional):** `prompts/experimental_champions_v1.json` can exist as a *generated export* if helpful, but `prompts/bank.json` remains canonical.

#### A.3.2 Target size and composition

**Total:** 15–18 prompts, chosen to cover families:
- **4** boundary dissolution
- **4** fixed-point framing
- **4** explicit regress
- **3** math+recursive hybrid
- **2–3** outliers (work strongly but don’t fit the above)

**Selection input:** Use `shortlist_top20.csv` + a family coverage constraint + “not trivially redundant.”

**Per-prompt metadata to store (minimum):**
- `prompt_id`, `text`, `family`, `source_run`, `is_paraphrase_of`, `rv_l27_promptpass`

---

## B) Critical confounds (must control)

### B.1 Length confound — PENDING

**Question:** Are champions “better” because they’re longer?  
**Control:** length-matched non-recursive baselines.  
**Test:** correlation length vs \(R_V\) within baselines should be ~0.

### B.2 Complexity confound — PENDING

**Question:** Are champions “better” because they’re syntactically/semantically complex?  
**Control:** complexity-matched baselines (nested clauses, abstract concepts, *not self-referential*).

### B.3 Pseudo-recursive confound — PENDING

**Question:** Are we detecting recursion *words* rather than recursion *structure*?  
**Control:** text “about recursion” without being self-referential (e.g., “The recursive algorithm…”).

### B.4 Keyword contamination (behavior metric) — PENDING

**Issue:** Keyword heuristics are gameable.  
**Action:** Add at least one alternative behavioral metric or explicitly scope claims to geometry.

### B.5 Random KV anomaly — PENDING (must resolve cleanly)

**Required controls:**
- multiple random seeds (random KV stability),
- baseline→baseline KV replacement,
- KV-only vs V_PROJ-only vs KV+V_PROJ separation.

---

## C) Statistical requirements (NeurIPS-level)

### C.1 Sample sizes
- Minimum **N=40 per condition** for main claims.
- Prefer **N=100** for headline effects and stability.
- Always report **exact N** and filtering criteria (NaNs, short prompts, etc.).

### C.2 Effect sizes
- Always report **Cohen’s d** (with **95% CI**) in addition to p-values.

### C.3 Multiple comparisons
- Use Bonferroni or FDR when testing many layers/heads/prompts; report corrected + uncorrected.

### C.4 Reproducibility fields (must appear in every run)
- `script path`, `config`, `seed`, `model_id`, `precision`, `prompt_bank_version/hash`.

### C.5 Discovery/validation split
- Split prompt families 80/20 or otherwise preregister “discovery prompts” vs “heldout prompts.”

---

## D) Repo hygiene checklist (to make the repo clean and auditable)

### D.1 Single source of truth
- `MECHANISM_MAP.md` is the master map.
- `prompts/bank.json` is the canonical prompt store (groups are the unit of canonization).
- `configs/` for canonical pipelines.
- `results/` for outputs (no “results in root”).

### D.2 Naming conventions
- Results should live at: `results/<topic>/runs/YYYYMMDD_HHMMSS_<name>/`
- Each run dir should include: `config.json`, `summary.json`, `*.csv`, and (if relevant) `README.md`.

### D.3 Deprecation discipline
- Move old scripts to `boneyard/` rather than deleting prior to submission.

### D.4 Behavioral metric upgrade — PENDING
At least one of:
- small human eval subset,
- semantic similarity to known recursive outputs,
- perplexity/entropy-based proxy tied to a preregistered rubric.

---

## E) Critiques to preempt (how this repo responds)

### E.1 “You’re just finding confusing prompts”
**Counter plan:** show coherence vs collapse; explicitly separate “trance/collapse” from “coherent contraction.”

### E.2 “R_V is arbitrary”
**Counter plan:** connect to attention convex-hull framing + show correlation/consistency with at least one additional geometric measure.

### E.3 “Recognition is mysticism”
**Counter plan:** only claim: “specific semantic patterns → measurable geometric regime changes,” without consciousness language.

### E.4 “Mistral-specific”
**Counter plan:** either scope strictly to Mistral-7B, or include at least one cross-arch replication.

### E.5 “GQA aliasing makes head claims wrong”
**Counter plan:** reframe as KV-head group effects; explicitly include aliasing caveat in limitations.

### E.6 “Behavior metric is keyword heuristic”
**Counter plan:** add alternative metric or mark as limitation and focus paper on geometry and causal control.

---

## Immediate actions (priority order)

1. **Freeze** `experimental_champions_v1` prompts (15–18) using the empirical shortlist and family coverage.
2. Create **control prompt sets** for length/complexity/pseudo-recursive confounds.
3. Run confound tests on **all** sets.
4. Resolve **random KV anomaly** with the full sufficiency matrix.
5. Update `MECHANISM_MAP.md` after each new result.










