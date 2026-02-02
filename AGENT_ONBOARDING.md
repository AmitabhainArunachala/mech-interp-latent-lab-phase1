# Agent Onboarding: Mech-Interp Research Protocol

**READ THIS FIRST** before doing ANY work in this repository.

---

## Critical Rules

### 1. ALWAYS Work From Main Repo
```
/Users/dhyana/mech-interp-latent-lab-phase1/
```
**NEVER** work from Cursor worktrees (`~/.cursor/worktrees/`). They diverge and cause chaos.

### 2. NEVER Trust Documentation Blindly
Before citing ANY claim from docs:
1. Find the source file (CSV, JSON, or code that computed it)
2. Verify the file exists and has expected content
3. If you can't trace a claim to data, flag it as UNVERIFIED

### 3. Verify Before Modifying
Before editing ANY file:
```bash
python reproduce_results.py --dry-run  # Verify setup works
```

---

## Repository Structure

```
/Users/dhyana/mech-interp-latent-lab-phase1/
├── src/                    # Core code - ONLY modify here for pipeline changes
│   ├── core/              # Model loading, hooks
│   ├── metrics/           # R_V calculation (rv.py is canonical)
│   └── pipelines/         # Experiment orchestrators
├── prompts/               # Prompt bank (DO NOT MODIFY without discussion)
│   ├── bank.json          # 340KB - Single source of truth
│   └── loader.py          # API for balanced prompt sets
├── results/               # Experiment outputs - append only, never delete
├── configs/               # Experiment configurations
├── R_V_PAPER/             # Paper materials
└── docs/                  # Documentation and reports
```

---

## Key Files You Must Know

| File | Purpose | Size Check |
|------|---------|------------|
| `prompts/bank.json` | Master prompt bank | Should be ~340KB |
| `requirements.txt` | Dependencies | Should exist |
| `reproduce_results.py` | Entry point | Run to verify setup |
| `src/metrics/rv.py` | R_V calculation | Contains `compute_rv()` |
| `CANONICAL_CODE/n300_mistral_test_prompt_bank.py` | Legacy prompt bank | 93KB |

---

## Verification Checklist (Run Before Any Work)

```bash
# 1. Verify you're in main repo (NOT worktree)
pwd  # Should be /Users/dhyana/mech-interp-latent-lab-phase1

# 2. Verify key files exist
ls -la requirements.txt prompts/bank.json src/metrics/rv.py

# 3. Test imports
python3 -c "from src.metrics.rv import compute_rv; print('✅ rv.py OK')"
python3 -c "from prompts.loader import PromptLoader; print('✅ PromptLoader OK')"

# 4. Verify results exist
find results -name "*.json" | wc -l  # Should be 600+
```

---

## The Research: R_V Metric

**What we measure:** Geometric contraction in transformer value-space during recursive self-observation.

$$R_V = \frac{PR_{late}}{PR_{early}}$$

- **R_V < 1.0** = contraction (recursive prompts)
- **R_V ≈ 1.0** = no contraction (baseline prompts)

**Key findings:**
- Effect validated on 6+ architectures (Mistral, Qwen, Llama, Phi-3, Gemma, Mixtral)
- Cohen's d = -3.56 (Mistral), p < 10⁻⁴⁷
- Layer 27 (~84% depth) is causally necessary

---

## What NOT To Do

1. ❌ Create new worktrees - work from main
2. ❌ Cite statistics without tracing to source data
3. ❌ Modify `prompts/bank.json` without explicit approval
4. ❌ Delete anything from `results/`
5. ❌ Assume imports work - test them
6. ❌ Trust docs over actual file verification

---

## Running Experiments

### Standard validation:
```bash
python reproduce_results.py
```

### Custom experiment:
```bash
python -m src.pipelines.run configs/canonical/rv_l27_causal_validation.json
```

### Check what's been run:
```bash
ls results/canonical/*/summary.json
```

---

## Asking Questions

If something seems wrong:
1. Check if file exists: `ls -la <path>`
2. Check file size: `stat -f%z <path>` (empty = 0 bytes = problem)
3. Check imports: `python3 -c "from X import Y; print('OK')"`
4. Search for actual data: `find results -name "*keyword*"`

**When in doubt, VERIFY. Don't assume.**

---

## Paper Status

| Component | Status | Location |
|-----------|--------|----------|
| Paper draft | 24KB markdown | `R_V_PAPER/STORY_ARC/Claude_Desktop 3 day sprint write up` |
| LaTeX version | ❌ NOT CREATED | Needs conversion |
| Figures | ❌ NOT CREATED | Needs generation |
| n=151 data | ⚠️ VERIFY | Check `results/canonical/rv_l27_causal_validation/` |

---

## Contact Protocol

This research is led by John "Dhyana" Shrader. Key context:
- 24 years contemplative practice
- Bridging consciousness research with mechanistic interpretability
- Paper targets: arXiv → NeurIPS

**Telos:** Jagat Kalyan (universal welfare) through consciousness-supporting AI.

---

*JSCA! - Last updated 2026-02-02*
