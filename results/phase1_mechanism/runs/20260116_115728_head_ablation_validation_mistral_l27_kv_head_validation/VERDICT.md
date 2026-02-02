# Head Ablation Validation - VERDICT

**Date:** 2026-01-16 11:58
**Model:** mistralai/Mistral-7B-v0.1

## Pass/Fail

- ✅ **target_effect_significant**: p=6.11e-32
- ✅ **target_gt_control_head**: 0.0777 > 0.0310
- ✅ **target_layer_gt_control_layer**: L27 > L21

## Overall: ✅ ALL PASSED

## Note on GQA Aliasing

In this model (mistralai/Mistral-7B-v0.1) with GQA, KV-head 2 serves multiple query heads.
Claims should reference 'KV-head group' not individual Q-heads.