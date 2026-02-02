# Mechanistic Interpretability Research Rules (v2.0)

This file serves as the **Standard Operating Procedure (SOP)** for all agents working on this repository. It supersedes previous rule blocks.

---

## 1. The Core Scientific Goal
We are investigating **Geometric Contraction ($R_V < 1.0$)** in Transformer models during recursive self-reference.
*   **Status:** Existence, Causality, and Stability are **proven** (Pipelines 1, 2, 6, 8).
*   **Current Focus:** **Behavior Transfer** (Does geometry cause behavior?) and **Cross-Model Generalization**.

---

## 2. Code Standards

### Model Physics (No Magic Numbers)
**NEVER** hardcode layer indices (e.g., `layer=27`) or head indices in your code.
*   **Use:** `src.core.model_physics`
    ```python
    from src.core.model_physics import get_model_physics
    physics = get_model_physics(model_name)
    target_layer = physics.late_layer
    ```

### Prompt Hygiene
**NEVER** hardcode prompt strings or lists in your code.
*   **Use:** `prompts.loader.PromptLoader`
    ```python
    from prompts.loader import PromptLoader
    loader = PromptLoader()
    prompts = loader.get_by_group("L4_full")
    ```
*   **Logging:** Always log `prompt_bank_version.txt` in your output directory.

### Metrics (The Contract)
**NEVER** invent your own R_V calculation.
*   **Use:** `src.metrics.rv.compute_rv` (Static) or `src.pipelines.temporal_stability` (Dynamic).
*   **Reference:** `docs/standards/MEASUREMENT_CONTRACT.md`

### Artifacts
Every run must produce a timestamped directory in `results/` containing:
1.  `config.json` (Exact reproduction parameters)
2.  `summary.json` (Machine-readable results)
3.  `report.md` (Human-readable summary)
4.  `prompt_bank_version.txt`

---

## 3. The Gold Standard Suite (Pipelines)

Use the canonical runner: `python -m src.pipelines.run --config configs/gold/...`

| ID | Name | Purpose | Status |
| :--- | :--- | :--- | :--- |
| **P1** | `01_existence` | Confirm $R_V < 1.0$ | ✅ PASS |
| **P2** | `02_causality` | V-Proj Patching | ✅ PASS |
| **P3** | `03_layer_map` | Locate Effect | ✅ PASS |
| **P4** | `04_head_validation` | Head Ablation | ✅ PASS |
| **P5** | `05_behavior_strict` | **Behavior Transfer** | 🚧 IN PROGRESS |
| **P6** | `06_temporal_stability` | Attractor Dynamics | ✅ PASS |
| **P7** | `07_hysteresis` | One-Way Door | ⚠️ NEGATIVE |
| **P8** | `08_kv_mechanism` | KV Cache Swap | ✅ PASS |
| **P9** | `09_steering` | **Vector Steering** | 🚧 IN PROGRESS |

---

## 4. Experimental Hygiene (Don't Fool Yourself)

1.  **Controls are Mandatory:** Every intervention must run against `Random`, `Shuffled`, and `Baseline` controls.
2.  **Blind Scoring:** Do not rely on "looking at the text." Use `src.metrics.behavior_strict` (Gates + Composite Score).
3.  **Clean Environment:** Always run `./scripts/clean.sh` before a major run to clear stale `.pyc` files.

---

## 5. Next Steps (The Roadmap)

1.  **Verify Steering (P9):** Can we induce recursion in a *non-recursive* prompt using a vector? (Fixing the "Tainted Baseline" bug).
2.  **Generalize:** Run P1 and P6 on **Llama-3-8B** using `model_physics.py`.
3.  **Solve Behavior:** Why does P8 (KV Swap) transfer geometry (94%) but P5 (Behavior) shows 0% transfer? Find the missing link.

---
*Created: Dec 19, 2025*







