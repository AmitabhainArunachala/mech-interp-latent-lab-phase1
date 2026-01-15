# Triple-System Intervention Gradient - Summary

## Results

| Config | Mean Coherence | Mean On-Topic | Mean Recursion | Collapse Rate |
|--------|----------------|---------------|----------------|---------------|
| baseline | 0.73 | 0.73 | 0.00 | 0.10 |
| vproj_kv | 0.72 | 0.46 | 0.00 | 0.10 |
| residual_kv | 0.71 | 0.55 | 0.00 | 0.10 |
| triple_light | 0.64 | 0.67 | 0.00 | 0.20 |
| triple_medium | 0.56 | 0.76 | 0.00 | 0.30 |

## Key Comparisons

### Config 3 vs Config 2 (Residual vs V_PROJ)
- Residual+KV coherence: 0.71 vs V_PROJ+KV: 0.72
- Residual+KV on-topic: 0.55 vs V_PROJ+KV: 0.46

### Config 4 vs Config 2 (Triple-Light vs V_PROJ+KV)
- Triple-Light coherence: 0.64 vs V_PROJ+KV: 0.72
- Triple-Light on-topic: 0.67 vs V_PROJ+KV: 0.46

### Config 4 vs Config 5 (Light vs Medium Alpha)
- Triple-Light on-topic: 0.67 vs Triple-Medium: 0.76
- Triple-Light collapse: 0.20 vs Triple-Medium: 0.30
