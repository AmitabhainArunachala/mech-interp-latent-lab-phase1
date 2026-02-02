# Reproducibility Review: Gold Standard Suite
**Reviewer:** Reviewer 3 (Reproducibility & Onboarding)
**Model:** gemini-3-pro-preview
**Date:** 2025-12-16

---

## Executive Summary
The onboarding experience is surprisingly good for a research codebase. `QUICK_START.md` is excellent—clear, honest, and direct. The concept of a "Sealed" prompt bank (`prompts/bank.json`) is a gold standard for reproducibility. However, a new user will hit immediate friction when trying to run Pipelines 4 and 5, which are documented but not fully implemented/registered.

**Onboarding Score: 9/10** (Drops to 6/10 if trying to run the full suite)

---

## Top 5 Points of Confusion

1.  **The "L27" Definition**:
    *   `src/metrics/rv.py` says `late=27`.
    *   `models/mistral_7b_analysis.py` says `LATE_LAYER=28`.
    *   **Confusion**: "Is it 0-indexed or 1-indexed? Which one is right?" (Answer: `src/metrics/rv.py` is canonical, but the discrepancy is confusing).

2.  **Pipeline 4 & 5 Ghost Status**:
    *   `GOLD_STANDARD_SUITE.md` lists them as Pipelines 4 & 5.
    *   `GPU_AGENT_TASK.md` says "Run Pipeline 4".
    *   `src/pipelines/registry.py` **does not have them**.
    *   **Result**: User runs the config and gets `ConfigError: Unknown experiment`.

3.  **"Behavior" Terminology**:
    *   Docs talk about "behavioral transfer."
    *   Code implements "keyword regex counting."
    *   **Confusion**: Users expecting semantic changes will be confused by the simplistic metric.

4.  **Results Location**:
    *   `results/` contains raw data.
    *   `agent_reviews/responses/` contains the "truth" (ledgers).
    *   **Confusion**: "Do I trust the CSV I just generated, or the PDF/MD in the reviews folder?"

5.  **GQA Aliasing**:
    *   Docs mention H18/H26.
    *   Mistral-7B architecture uses GQA.
    *   **Confusion**: "How can H18 be distinct from H2 if they share a KV head?" (The docs verify this is an issue, but the legacy naming persists).

---

## Top 5 Missing Documentation / Pieces

1.  **Implementation of Pipeline 4**: `head_ablation_validation` needs to be registered in `registry.py` to be runnable via `run.py`.
2.  **Implementation of Pipeline 5**: `behavior_validation_strict` is completely missing.
3.  **Multi-Token Persistence Guide**: `experiment_multi_token_generation.py` is mentioned as "UNMAPPED" but crucial. Needs a guide.
4.  **Cross-Model Configs**: Only Mistral-7B configs exist in `configs/gold/`. `QUICK_START` mentions Mixtral, but there's no "gold" config for it.
5.  **Troubleshooting "Overshooting"**: Pipeline 2 results often show >100% transfer efficiency. `QUICK_START` should explain if this is a feature or a bug (it's likely noise/normalization, but looks weird).

---

## Suggested Improvements to QUICK_START.md

1.  **Add Warning about P4/P5**: Explicitly state "Pipelines 4 and 5 are currently standalone scripts, not integrated into the `run.py` runner yet."
2.  **Clarify L27**: Add a note: "L27 means Layer Index 27 (0-indexed) in a 32-layer model. This is the 28th layer."
3.  **Behavior Disclaimer**: "Note: 'Behavior' currently refers to keyword induction, not semantic steering."

---

## "Can I Run It?" Check

*   **Pipeline 1 (Existence)**: ✅ YES (`python -m src.pipelines.run --config configs/gold/01_existence.json`)
*   **Pipeline 2 (Causality)**: ✅ YES (`python -m src.pipelines.run --config configs/gold/02_causality.json`)
*   **Pipeline 3 (Layer Map)**: ✅ YES (`python -m src.pipelines.run --config configs/gold/03_layer_map.json`)
*   **Pipeline 4 (Heads)**: ❌ NO (Must run `python validate_h18_h26_gold_standard.py` directly, violating the runner contract)
*   **Pipeline 5 (Behavior)**: ❌ NO (Code missing)









