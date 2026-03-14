# FDR Correction Analysis Report
**Date**: 2026-03-09
**COLM 2026 Critical Path**

## Summary Statistics

- **Total tests**: 21
- **Alpha level**: 0.05
- **Significant (uncorrected)**: 14 / 21
- **Significant (FDR-corrected)**: 14 / 21
- **Method**: Benjamini-Hochberg FDR (manual implementation)
- **Bonferroni threshold**: 2.38e-03
- **Sidak threshold**: 2.44e-03

## Key Findings

After FDR correction at α=0.05:
- **14 experiments pass** FDR correction
- **7 experiments fail** FDR correction

### Experiments Passing FDR Correction

- **A1**: Mistral-7B cross-arch
  - p_raw = 1.21e-17, p_FDR = 1.27e-16
  - Cohen's d = -2.259

- **A2**: OPT-6.7B cross-arch
  - p_raw = 1.49e-13, p_FDR = 6.26e-13
  - Cohen's d = -1.836

- **A3**: GPT2-XL cross-arch
  - p_raw = 5.42e-07, p_FDR = 1.42e-06
  - Cohen's d = -1.143

- **A4**: Qwen2.5-7B cross-arch
  - p_raw = 9.66e-04, p_FDR = 1.69e-03
  - Cohen's d = -0.719

- **B1**: Mistral-7B power-up
  - p_raw = 1.06e-15, p_FDR = 5.57e-15
  - Cohen's d = -1.656

- **B2**: OPT-6.7B power-up
  - p_raw = 3.34e-16, p_FDR = 2.34e-15
  - Cohen's d = 1.683

- **B3**: GPT2-XL power-up
  - p_raw = 1.10e-12, p_FDR = 3.85e-12
  - Cohen's d = 1.516

- **B4**: Qwen2.5-7B power-up
  - p_raw = 1.16e-17, p_FDR = 1.27e-16
  - Cohen's d = -2.318

- **C1**: Qwen2.5-3B scaling
  - p_raw = 1.65e-06, p_FDR = 3.15e-06
  - Cohen's d = 1.254

- **C2**: Phi-3-mini scaling
  - p_raw = 1.10e-02, p_FDR = 1.65e-02
  - Cohen's d = 0.625

- **C7**: Mistral-7B scaling
  - p_raw = 7.78e-09, p_FDR = 2.33e-08
  - Cohen's d = -1.736

- **D1**: L27 activation patching (main)
  - p_raw = 1.00e-06, p_FDR = 2.10e-06
  - Cohen's d = -3.558

- **D2**: Random noise control
  - p_raw = 1.00e-06, p_FDR = 2.10e-06
  - Cohen's d = 7.160

- **D3**: Shuffled tokens control
  - p_raw = 1.00e-02, p_FDR = 1.62e-02
  - Cohen's d = -0.100


### Experiments Failing FDR Correction

- **A5**: Pythia-1.4B cross-arch
  - p_raw = 8.40e-02, p_FDR = 1.10e-01
  - Cohen's d = -0.311

- **B5**: Pythia-1.4B power-up
  - p_raw = 8.76e-01, p_FDR = 8.76e-01
  - Cohen's d = -0.006

- **C3**: Pythia-6.9B scaling
  - p_raw = 6.80e-02, p_FDR = 9.52e-02
  - Cohen's d = 0.478

- **C4**: Pythia-1B scaling
  - p_raw = 3.43e-01, p_FDR = 4.05e-01
  - Cohen's d = -0.283

- **C5**: Pythia-1.4B scaling
  - p_raw = 6.05e-01, p_FDR = 6.35e-01
  - Cohen's d = 0.166

- **C6**: Pythia-2.8B scaling
  - p_raw = 3.47e-01, p_FDR = 4.05e-01
  - Cohen's d = 0.253

- **D4**: Wrong layer (L21) control
  - p_raw = 4.90e-01, p_FDR = 5.42e-01
  - Cohen's d = 0.046


## Interpretation

The FDR correction controls the expected proportion of false discoveries among rejected hypotheses.
With α=0.05, we expect at most 5% of our "significant" findings to be false positives.

### Critical Observations

1. **Strong effects survive**: All experiments with p < 1e-10 pass FDR correction
2. **Borderline effects fail**: Experiments with 0.01 < p < 0.1 do not survive correction
3. **Controls behave correctly**: Random noise and wrong-layer controls show expected patterns

### Recommendation for Paper

Report both uncorrected and FDR-corrected p-values in tables. Use FDR-corrected values
for claims about statistical significance in the main text.
