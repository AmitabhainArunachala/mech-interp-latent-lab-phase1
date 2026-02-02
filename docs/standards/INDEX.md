# Industry Standards & Benchmarks

This directory contains the **immutable contracts** and **external benchmarks** that govern this research.

## Internal Contracts
*   [**MEASUREMENT_CONTRACT.md**](./MEASUREMENT_CONTRACT.md): The law. Defines $R_V$, "Strict Behavior," and Generation Tiers.

## External Benchmarks (Gap Analysis)
*   [**MIB_GAP_ANALYSIS.md**](./MIB_GAP_ANALYSIS.md): Comparison against arXiv:2504.13151 ("MIB: A Mechanistic Interpretability Benchmark").
    *   *Status:* Aligned on Circuit Localization. Planning DAS integration.

## Usage
All pipelines in `src/pipelines/` must comply with the contracts defined here.
Any deviation must be approved by updating the contract version number.

## Operational Guides
*   [**PIPELINE_OPERATIONS.md**](../PIPELINE_OPERATIONS.md): How to run experiments properly.
*   [**REPRODUCIBILITY_POLICY.md**](../REPRODUCIBILITY_POLICY.md): Dependency and reproduction policy.
*   [**AUDIT_2026-01-24.md**](../analysis/AUDIT_2026-01-24.md): Repo audit snapshot.







