# Reproducibility Policy

## Two-Tier Dependency Strategy

This repository uses a dual-file dependency system designed for both development flexibility and exact reproducibility.

### File Purposes

#### `requirements.txt` - Development Mode

**Purpose**: Active development with flexible version ranges

```bash
pip install -r requirements.txt
```

**Characteristics**:
- Uses version ranges (e.g., `torch>=2.1.0,<2.2.0`)
- Allows minor version updates within tested ranges
- Suitable for ongoing research and development
- Faster to update when new compatible versions are released

**Use when**:
- Developing new features
- Testing compatibility with newer library versions
- Working on local machine or cloud instances
- Not concerned with exact bit-level reproducibility

#### `requirements.lock` - Publication Mode

**Purpose**: Exact reproducibility for published results

```bash
pip install -r requirements.lock
```

**Characteristics**:
- Pins exact versions (e.g., `torch==2.1.2`)
- Guarantees bit-perfect reproduction
- Required for publication-grade experiments
- Direct dependencies only (transitive deps resolve automatically)

**Use when**:
- Reproducing published results
- Validating experimental findings
- Preparing for paper submission
- Creating reproducible research artifacts

### Dependency Philosophy

**Direct Dependencies Only**: Both files list only direct dependencies. Transitive dependencies (like `tokenizers`, `safetensors`) are comments in `requirements.lock` for reference but resolve automatically via pip.

**Version Testing**: Each pinned version in `requirements.lock` has been tested on:
- RunPod L40S (48GB VRAM, CUDA 12.1)
- M3 Pro MacBook (18GB RAM, MPS backend)

### Hardware-Specific Installation

#### CUDA 12.1 (RunPod, Cloud GPUs)

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

#### MPS (Apple Silicon)

```bash
pip install -r requirements.txt  # PyTorch auto-detects MPS
```

#### CPU-Only

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Publication-Grade Runs

For experiments intended for publication or formal documentation:

### Minimum Requirements

1. **Environment**:
   - Use `requirements.lock` for exact versions
   - Document hardware (GPU model, VRAM, CPU)
   - Record environment with `pip freeze > full_env.txt`

2. **Experimental Control**:
   - Set random seeds explicitly
   - Use config files (in `configs/`)
   - Run minimum sample size (N=80 pairs)
   - Apply statistical corrections (Bonferroni)

3. **Results Documentation**:
   - Save timestamped output folders
   - Include config snapshot in results
   - Record model checkpoints/IDs
   - Document any failures or exceptions

### Validation Checklist

- [ ] Installed via `requirements.lock`
- [ ] Random seed set and documented
- [ ] Config file used (not hardcoded parameters)
- [ ] Results saved with timestamp
- [ ] Hardware specifications documented
- [ ] Statistical thresholds met (p < 0.01, |d| ≥ 0.5)
- [ ] Effect replicates across random seeds

## Environment Snapshot

For complete environment documentation:

```bash
# After installing requirements.lock
pip freeze > full_env.txt
git add full_env.txt
git commit -m "docs: Add complete environment snapshot"
```

This captures all transitive dependencies at exact versions.

## Dependency Updates

### Updating `requirements.txt`

1. Test new version range
2. Validate on all supported hardware
3. Update version range in `requirements.txt`
4. Run validation experiments
5. Document compatibility in README

### Updating `requirements.lock`

1. Install new exact version
2. Run full experimental battery
3. Compare results with previous version
4. If results match within tolerance:
   - Update `requirements.lock`
   - Document version change in CHANGELOG
5. If results differ:
   - Investigate source of difference
   - Document behavioral change
   - Consider pinning previous version

## Reproducibility Levels

### Level 1: Qualitative Reproducibility

**Goal**: Same conclusions from same code

- Use `requirements.txt`
- Any compatible hardware
- Random seed variation expected
- Effect direction should match

### Level 2: Quantitative Reproducibility

**Goal**: Same numerical results within tolerance

- Use `requirements.lock`
- Same hardware class (e.g., NVIDIA Ampere)
- Fixed random seeds
- Results within ±5% tolerance

### Level 3: Bit-Perfect Reproducibility

**Goal**: Identical binary outputs

- Use `requirements.lock`
- Identical hardware
- Fixed random seeds
- Same CUDA/PyTorch builds
- Results match exactly

**Note**: Level 3 is extremely difficult due to non-deterministic GPU operations and floating-point variance.

## Philosophy

> **Code is Law**: If it isn't modular, typed, and reproducible, it doesn't exist.

We prioritize Level 2 (Quantitative Reproducibility) as the standard for publication. Level 1 is acceptable for development. Level 3 is aspirational but not required.

---

*Precision. Minimalism. Truth.*
