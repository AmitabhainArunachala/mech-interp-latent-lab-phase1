# Activation Patching References and Citations

**Last Updated:** January 10, 2026

---

## Current Citations in Repository

### Found Citations

1. **`manim/top_findings_summary.py` (line 501):**
   ```python
   cite = footer("Activation patching / causal tracing: Meng et al. 2022  •  Transformer circuits: Elhage et al. 2021")
   ```

2. **Repository Rules (`INDUSTRY_GRADE_SPINE_AUDIT.md`):**
   - Mentions "attribution patching" as a technique
   - References circuit discovery methods

---

## What We Actually Do

### Implementation: `circuit_discovery.py`

**Method:** Attribution Patching Sweep (Recursive → Baseline)
- Patching activations from recursive prompt into baseline prompt
- Measuring logit difference (Mode Score delta)
- Sweeping across all layers and components (Attention vs MLP)

**Key Code Pattern:**
```python
# 1. Capture activations from recursive run
activations = {}
def make_capture_hook(name):
    def hook(module, inp, out):
        activations[name] = out.detach().clone()
    return hook

# 2. Patch into baseline run
def make_patch_hook(source):
    def hook(module, inp, out):
        out_p = out_tensor.clone()
        out_p[:, -W:, :] = source[:, -W:, :]  # Patch last W tokens
        return out_p
    return hook
```

---

## Terminology Used in Repository

1. **"Attribution Patching"** - Used in `circuit_discovery.py` and reports
   - Measures causal attribution (which components drive recursive behavior)
   - Patching from recursive (source) into baseline (target)

2. **"Activation Patching"** - Used in citations and general discussion
   - General term for patching activations between runs

3. **"Causal Tracing"** - Used in citations
   - Related technique for tracing causal paths

---

## Related Techniques in Repository

### 1. MLP Ablation (`mlp_ablation_necessity.py`)
- **Method:** Zero ablation (necessity test)
- **Purpose:** Test if component is necessary for contraction
- **Citation:** Not explicitly cited (standard ablation technique)

### 2. MLP Patching (`mlp_sufficiency_test.py`)
- **Method:** Patching MLP outputs (sufficiency test)
- **Purpose:** Test if component is sufficient to induce contraction
- **Citation:** Not explicitly cited (standard patching technique)

### 3. KV Cache Patching (`kv_mechanism.py`)
- **Method:** Swapping KV cache between prompts
- **Purpose:** Test geometry transfer via memory
- **Citation:** Not explicitly cited (novel application)

### 4. Path Patching (`path_patching_mechanism.py`)
- **Method:** Patching specific pathways
- **Purpose:** Trace information flow
- **Citation:** Not explicitly cited (related to Wang et al. 2022?)

---

## Standard Papers (To Verify)

### Activation Patching / Causal Tracing

1. **Meng et al. 2022** (Currently cited, but user says wrong)
   - "Locating and Editing Factual Associations in GPT"
   - Introduces activation patching for factual associations
   - **Status:** ❌ User indicated "That's the wrong paper"

2. **Heimersheim & Nanda 2024** (Found via search)
   - "How to use and interpret activation patching" (arXiv:2404.15255)
   - Tutorial/guide on activation patching
   - **Status:** ⚠️ Guide paper, not foundational

3. **Geiger et al. 2021** (Potential foundational paper)
   - "Interchange Intervention" / "Causal Abstraction"
   - **Status:** ⚠️ Needs verification - may be foundational for activation patching

4. **Wang et al. 2022** (Mentioned in repo rules)
   - "Path Patching"
   - **Status:** ⚠️ Needs verification if this is the correct citation

5. **Elhage et al. 2021** (Currently cited)
   - "Transformer Circuits"
   - General framework for circuit analysis
   - **Status:** ✅ Likely correct (foundational work)

### Other Related Papers

4. **Chan et al. 2022** (Mentioned in repo rules)
   - "Causal Scrubbing"
   - **Status:** ⚠️ Needs verification

5. **Conmy et al. 2023** (ACDC - Found in `REFERENCE/巨人の肩/02_ACDC_Automated_Circuit_Discovery.md`)
   - "Automated Circuit Discovery"
   - **Status:** ✅ Verified (NeurIPS 2023)

---

## Action Items

1. **Verify Correct Activation Patching Paper**
   - User indicated Meng et al. 2022 is wrong
   - Need to identify correct paper for activation patching / attribution patching

2. **Standardize Citations**
   - Update `manim/top_findings_summary.py` with correct citation
   - Add citations to pipeline docstrings
   - Create citation reference document

3. **Clarify Terminology**
   - "Attribution Patching" vs "Activation Patching" vs "Causal Tracing"
   - Document which term applies to which technique

---

## Current Implementation Status

### Pipelines Using Patching Techniques

| Pipeline | Technique | Citation Status |
|----------|-----------|----------------|
| `circuit_discovery.py` | Attribution patching sweep | ⚠️ Needs citation |
| `mlp_sufficiency_test.py` | MLP output patching | ⚠️ Needs citation |
| `mlp_combined_sufficiency_test.py` | Multi-layer MLP patching | ⚠️ Needs citation |
| `kv_mechanism.py` | KV cache patching | ⚠️ Novel application |
| `path_patching_mechanism.py` | Path-specific patching | ⚠️ Needs citation |

---

## Next Steps

1. **Search for correct activation patching paper**
   - Check if there's a more foundational paper
   - Verify Wang et al. 2022 for path patching
   - Identify correct citation for attribution patching

2. **Update repository citations**
   - Fix `manim/top_findings_summary.py`
   - Add docstrings to patching functions
   - Create `CITATIONS.md` reference file

3. **Document methodology alignment**
   - Map our techniques to standard papers
   - Clarify what's novel vs. standard practice
