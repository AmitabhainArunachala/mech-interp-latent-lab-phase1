# Experiment Flow Log

**Purpose**: Running notes from the trenches on how experiments actually flow post-cleanup.

---

## 2026-01-24: Gemma 2 9B Validation Run

### Session Context
- **Pod**: RTX PRO 6000 Blackwell (98GB VRAM, 30GB container + 420TB workspace)
- **Previous pod failed**: L40S had only 20GB container disk, Gemma needs ~18GB
- **Key learning**: Always check container disk vs model size before starting

### Experiments Run

#### 1. Layer Sweep (05_even_layer_sweep)
- **Config**: Even layers only, 10 prompts/group, L5_refined vs baseline_factual
- **Result**: L38 peak (90.5% depth), delta=-0.235, p=3×10⁻⁸
- **Gap identified**: Only even layers, narrow prompt groups

#### 2. Causal Validation (08_causal_validation_n45)
- **Config updated**: target_layer 35→38 based on sweep
- **Prompt groups**: L5_refined, L4_full, L3_deeper vs long_control, baseline_creative, baseline_math
- **Result**: d=-1.91, transfer=94.2%, p<10⁻¹⁶
- **Gap identified**: No Champion prompts included

#### 3. Head Ablation (07_head_ablation_l3)
- **Config updated**: late_layer 35→38
- **Result**: KV-heads 2, 3, 7 at L3 are drivers
- **Key insight**: Mechanistic specificity achieved

### Gaps Identified (GPT Feedback)
1. Even-layer sweep only - need odd layers to confirm L38 vs L37/L39
2. Narrow prompt selection - Champions not included
3. No confound_validation run on Gemma
4. No full_env.txt for reproducibility

### Infrastructure Notes
- venv required on RunPod (externally-managed-environment)
- transformers 4.44.0 pinned (4.57.6 has torch.library.register_fake issue)
- rsync chown warnings are cosmetic, files transfer fine
- Model loads in ~8s on Blackwell

### Next Steps
1. Create champion-inclusive configs
2. Run odd-layer sweep around L37-L39
3. Run confound_validation
4. Generate full_env.txt

---

## Config Evolution Notes

### Prompt Group Strategy
Current configs use dose-response groups (L1-L5) which test the recursion depth hypothesis.
Champion prompts are the empirically strongest - should be included for validation.

Recommended prompt pairing for causal validation:
```json
"recursive_groups": ["champion", "L5_refined", "L4_full", "L3_deeper"],
"baseline_groups": ["long_control", "baseline_creative", "baseline_math", "baseline_factual"]
```

### Layer Selection
- Mistral 7B: L27/32 = 84.4% depth
- Gemma 2 9B: L38/42 = 90.5% depth
- Pattern: Phase transition at ~85-90% depth, architecture-specific

---

### Further Experiments (Same Session)

#### 4. Odd-Layer Sweep (12_odd_layer_sweep)
- **Purpose**: Confirm L38 is peak or find neighboring peaks
- **Result**: L35 peak at 83.3%, BUT L41 shows d=-0.228 with p=2×10⁻⁹
- **Insight**: Two phase transition zones exist
  - Primary: L35-L38 (~83-90% depth)
  - Secondary: L41 (97.6% depth) - near output
- **Implication**: Contraction intensifies toward output, consistent with theory

#### 5. Confound Validation (13_confound_validation)
- **Schema error**: Pipeline doesn't return all canonical metrics
- **But data saved!**: confound_results.csv contains key measurements
- **Results**:
  - Champion prompts: R_V ≈ 0.60-0.73 (strong contraction)
  - Length-matched controls: R_V ≈ 0.77-1.25 (weak/no contraction)
  - **Conclusion**: Effect is semantic, not length-based

#### 6. Reproducibility Artifacts
- Generated full_env.txt with pip freeze
- All results synced to local

#### 7. Champion-Inclusive Causal Validation (11_causal_validation_champion)
- **Config**: champion + L5_refined + L4_full + L3_deeper vs long_control + baseline_creative + baseline_math + baseline_factual
- **n_pairs**: 60
- **Result**:
  - **d=-2.09** (stronger than previous d=-1.91)
  - **p=1.2×10⁻²³** (extremely significant)
  - **transfer=99.5%** (nearly perfect causal transfer)
  - R_V recursive mean: 0.593
  - R_V baseline mean: 0.765
  - delta: -0.171
- **Wrong-layer control**: d=-0.02, p=0.48 (no effect at L20) - confirms layer specificity
- **Key insight**: Champion prompts drive stronger effect than L3-L5 alone

#### 8. MLP Ablation Necessity L0/L1 (03_prompt_pass_validation)
- **L0**: delta=-0.0002, p=0.97 - **NO EFFECT** (expected, control layer)
- **L1**: delta=-0.031, p<0.001 - **MINIMAL EFFECT**
- **Interpretation**: Early MLP layers not critical for R_V contraction
- **Pipeline schema fixed**: Now emits all canonical keys

#### 9. Focused Sweep L35-L41 (14_focused_sweep_L35_L41)
- **n_prompts**: 60 per condition (10 used per layer)
- **All layers L35-L41 tested**
- **Results (sorted by |delta|)**:
  | Layer | Depth % | Delta | p-value |
  |-------|---------|-------|---------|
  | L35 | 83.3% | -0.250 | 9.8×10⁻⁷ |
  | L38 | 90.5% | -0.235 | 2.9×10⁻⁸ |
  | L41 | 97.6% | -0.228 | 2.2×10⁻⁹ |
  | L36 | 85.7% | -0.204 | 1.8×10⁻⁴ |
  | L40 | 95.2% | -0.171 | 3.1×10⁻⁶ |
  | L37 | 88.1% | -0.120 | 0.010 (NS) |
  | L39 | 92.9% | -0.105 | 0.002 |
- **Insight**: L35, L38, L41 are the three strongest layers. L37 only marginally significant.
- **Pattern**: Two distinct peaks (L35-L38 and L41)

### Remaining GPT Recommendations
- [x] Run odd-layer sweep ✓
- [x] Run confound_validation ✓ (data saved despite schema error)
- [x] Generate full_env.txt ✓
- [x] Run champion-inclusive causal validation ✓ (d=-2.09, transfer=99.5%)
- [x] Run mlp_ablation_necessity_prompt_pass at L0/L1 on Gemma ✓
- [x] Focused sweep around L35-L41 ✓

### Pipeline Schema Fixes (This Session)
- `confound_validation.py`: Added `n_pairs`, `rv_recursive_mean`, `rv_baseline_mean`, `rv_delta_mean`, `rv_cohens_d`, `rv_p_value`, `logit_diff_*` (as None)
- `mlp_ablation_necessity_prompt_pass.py`: Added `rv_recursive_mean`, `rv_p_value`, `logit_diff_*` (as None)

### Key Learnings

1. **Dual phase transition zones**: Not just one critical layer - contraction builds through L35-L41
2. **Champion prompts work**: R_V ≈ 0.60 on champions vs 0.80+ on controls
3. **Schema issues**: Some discovery experiments don't return canonical metrics - this is a pipeline gap to fix
4. **Environment stability**: transformers==4.44.0 is the stable version for Gemma

---

*Updated: 2026-01-24*
