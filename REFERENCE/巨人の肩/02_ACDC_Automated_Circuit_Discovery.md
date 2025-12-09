# ACDC – Conmy et al., NeurIPS 2023

- **Title**: Automated Circuit Discovery in Language Models
- **Why relevant**: Automates the search for important edges in a circuit; can help find which late-layer KV heads and paths carry the recursive mode.
- **Key methods to adapt**:
  - Edge-importance scoring over a candidate graph of components.
  - Iterative pruning while monitoring behavioural loss.
- **Notes**: Good candidate for scaling beyond hand-curated heads/layers in L16–32.
