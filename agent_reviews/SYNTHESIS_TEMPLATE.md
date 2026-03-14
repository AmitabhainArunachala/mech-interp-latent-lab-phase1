# SYNTHESIS: Multi-Agent COLM Audit Results
**Date:** 2026-03-09
**Auditor:** [synthesizing agent]
**Reports compared:** [list agent reports]

---

## METHOD
For each of the 25 claims, record each agent's verdict. Flag disagreements.
- **CONSENSUS** = all agents agree
- **MAJORITY** = 3+ agents agree, 1-2 disagree
- **SPLIT** = no clear majority → INVESTIGATE

---

## CLAIM VERDICTS MATRIX

| Claim | Agent 1 | Agent 2 | Agent 3 | Agent 4 | Agent 5 | Consensus |
|-------|---------|---------|---------|---------|---------|-----------|
| C1  Mistral d=-1.66 | | | | | | |
| C2  Qwen d=-2.32 | | | | | | |
| C3  OPT signed d | | | | | | |
| C4  GPT-2 signed d | | | | | | |
| C5  Pythia d=-0.006 | | | | | | |
| C6  "four models" claim | | | | | | |
| C7  Table 1 n values | | | | | | |
| C8  Necessity d=3.29 | | | | | | |
| C9  Sufficiency d=-3.50 | | | | | | |
| C10 BT+ART 56%→27.7% | | | | | | |
| C11 Bridge d=-0.71 | | | | | | |
| C12 V-proj max d | | | | | | |
| C13 Mode atlas d=-1.67 | | | | | | |
| C14 606/1024 heads | | | | | | |
| C15 PPL matching | | | | | | |
| C16 Multi-seed d=-1.751 | | | | | | |
| C17 FDR 30/36 | | | | | | |
| C18 L27H10 rank d=-1.54 | | | | | | |
| C19 L5H29 d=2.93 | | | | | | |
| C20 Concept erasure Δ=0.005 | | | | | | |
| C21 DII R_V≈0.41 | | | | | | |
| C22 RSA distance 0.307 | | | | | | |
| C23 AUROC=0.909 | | | | | | |
| C24 Genuine vs deceptive d=-0.06 | | | | | | |
| C25 Scaling R²=0.047 | | | | | | |

---

## DISAGREEMENT ANALYSIS

For each SPLIT or MAJORITY verdict, investigate:
1. Which file did each agent cite?
2. Did they read different files (→ data provenance issue)?
3. Did they interpret the same file differently (→ ambiguity issue)?

---

## ORPHAN FINDINGS CONSENSUS

| Orphan | Agent 1 | Agent 2 | Agent 3 | Agent 4 | Agent 5 | Include? |
|--------|---------|---------|---------|---------|---------|----------|
| Behavioral dissociation | | | | | | |
| GQA headspace | | | | | | |
| [add as discovered] | | | | | | |

---

## FINAL VERDICT

### Must fix before submission (CONTRADICTED claims):
1. 
2. 

### Should fix (PARTIAL or NO_DATA claims):
1.
2.

### Orphan findings to add:
1.
2.

### Claims that are SOLID (all agents CONFIRMED):
1.
2.
