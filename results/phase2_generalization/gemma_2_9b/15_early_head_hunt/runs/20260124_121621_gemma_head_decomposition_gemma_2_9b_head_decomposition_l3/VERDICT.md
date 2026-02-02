# Gemma 2 9B Head-wise Decomposition at L3

**Date:** 2026-01-24 12:17

## Configuration
- Source layer: L3
- Control layer: L5
- KV-heads tested: 8
- Prompts per type: 40

## Results by KV-head

| KV-head | Δ_L3 (mean±std) | Δ_L5 (mean±std) | p(L3≠0) | Sig? | L3>L5? |
|---------|-----------------|-----------------|---------|------|--------|
| 0 | -0.0005±0.0027 | +0.0012±0.0030 | 4.63e-01 |  |  |
| 1 | -0.0003±0.0044 | +0.0012±0.0030 | 8.06e-01 |  |  |
| 2 | -0.0025±0.0024 | -0.0031±0.0026 | 2.49e-04 | ✓ |  |
| 3 | +0.0011±0.0033 | -0.0008±0.0031 | 1.59e-01 |  |  |
| 4 | +0.0006±0.0032 | +0.0000±0.0031 | 4.14e-01 |  |  |
| 5 | +0.0002±0.0029 | -0.0070±0.0058 | 7.83e-01 |  | ✓ |
| 6 | -0.0016±0.0044 | +0.0008±0.0037 | 1.25e-01 |  |  |
| 7 | -0.0058±0.0061 | -0.0026±0.0043 | 5.73e-04 | ✓ |  |

## Driver Heads

No single head identified as primary driver. Effect may be:
1. Distributed across multiple heads
2. Arising from head interactions
3. Requiring MLP interaction (not just attention)