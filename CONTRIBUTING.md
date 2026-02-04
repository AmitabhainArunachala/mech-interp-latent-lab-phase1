# Contributing to Mechanistic Interpretability Research

Thank you for your interest in contributing to our research on geometric signatures of recursive self-observation in transformer language models!

## Philosophy

**Precision. Minimalism. Truth.**

We value:
- **Reproducibility**: Code must produce identical results across runs
- **Minimalism**: Only include what's necessary
- **Documentation**: All claims must be backed by code and data
- **Scientific rigor**: Statistical thresholds and effect sizes must be reported

## How to Contribute

### Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include minimal reproducible examples
- Specify your environment (Python version, hardware, OS)

### Proposing Changes

1. **Fork** the repository
2. **Create a branch** from `main` with a descriptive name
3. **Make your changes** following our code standards
4. **Test thoroughly** - all experiments must reproduce
5. **Submit a Pull Request** with a clear description

### Code Standards

#### Reproducibility

- Set random seeds explicitly
- Use the two-file dependency system (`requirements.txt` and `requirements.lock`)
- Document hardware requirements (GPU memory, compute time)
- Include exact command-line invocations in commit messages

#### Code Style

- Follow existing patterns in the codebase
- Use type hints where appropriate
- Keep functions focused and modular
- Use context managers (`with` statements) for all model hooks
- Never leave hooks attached after function returns

#### Testing

- Validate against Mistral-7B Base (our reference model)
- Test with at least 80 prompt pairs for statistical power
- Report p-values with Bonferroni correction
- Report effect sizes (Cohen's d ≥ 0.5 for meaningful effects)

#### Documentation

- Update README.md for significant changes
- Add docstrings to new functions
- Include examples in docstrings
- Update relevant documentation in `docs/`

### Research Contributions

If proposing new experiments:

1. **Justify the hypothesis** - explain what you're testing
2. **Follow the protocol** - use standard R_V measurement invariants
3. **Report all results** - including null findings
4. **Archive failed experiments** - move to appropriate directory with explanation

### The Boneyard

Failed experiments are valuable but don't belong in the living codebase. If your experiment doesn't produce significant results:

1. Document the attempt thoroughly
2. Move code to an archive directory
3. Update documentation explaining what was tried and why it failed

## Development Setup

```bash
# Clone the repository
git clone https://github.com/AmitabhainArunachala/mech-interp-latent-lab-phase1.git
cd mech-interp-latent-lab-phase1

# Install dependencies (development mode)
pip install -r requirements.txt

# Run tests
python reproduce_results.py
```

## Pull Request Process

1. Update documentation to reflect any changes
2. Ensure all tests pass
3. Update CHANGELOG.md if making user-facing changes
4. Request review from maintainers
5. Address review feedback promptly

## Questions?

Open an issue for discussion or reach out to the maintainers.

---

*"Code is Law. If it isn't modular, typed, and reproducible, it doesn't exist."*
