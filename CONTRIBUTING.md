# Contributing to Mechanistic Interpretability Research

Thank you for your interest in contributing to this research repository! We value precision, minimalism, and reproducibility.

## Code Philosophy

**Code is Law**: If it isn't modular, typed, and reproducible, it doesn't exist.

**The Boneyard**: Failed experiments are valuable, but they do not belong in the living codebase.

**The Standard**: Mistral-7B Base is the reference reality. All other models are comparative studies.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/mech-interp-latent-lab-phase1.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Install dependencies: `pip install -r requirements.lock` (for exact reproducibility)

## Development Guidelines

### Code Quality

- **Type hints**: Use type annotations for all functions
- **Documentation**: Add docstrings following NumPy style
- **Modularity**: Keep functions focused and composable
- **Testing**: Add tests for new functionality (if test infrastructure exists)

### Reproducibility

- **Random seeds**: Always use `set_seed()` for reproducible results
- **Dependencies**: Update both `requirements.txt` and `requirements.lock`
- **Data**: Never commit large data files or model weights
- **Results**: Document your experimental setup in detail

### Code Patterns

Follow the established patterns in the codebase:

```python
# Standard hook pattern
from src.core.hooks import capture_v_projection

with capture_v_projection(model, layer_idx=27) as storage:
    with torch.no_grad():
        model(**inputs)
v_tensor = storage["v"]
```

### Commit Messages

Use clear, descriptive commit messages:

- `feat: Add new metric for measuring X`
- `fix: Correct SVD handling for degenerate cases`
- `docs: Update README with new results`
- `refactor: Simplify hook context manager`
- `test: Add unit tests for R_V computation`

## Pull Request Process

1. **Update documentation**: Modify README.md if adding features
2. **Run tests**: Ensure all tests pass (if applicable)
3. **Clean commits**: Squash work-in-progress commits
4. **Clear description**: Explain what changes and why
5. **Link issues**: Reference related issues in the PR description

## Research Contributions

### Adding New Experiments

1. Place experimental code in `src/experiments/`
2. Use config-driven approach (see `configs/`)
3. Document results in `results/` with timestamped folders
4. Update `docs/` with analysis

### Proposing New Metrics

1. Add metric to `src/metrics/`
2. Include theoretical justification
3. Validate against existing baselines
4. Document in appropriate sections

### Model Extensions

1. Test with Mistral-7B Base first (the reference standard)
2. Document architecture-specific differences
3. Add to compatibility matrix in README

## Questions?

- Open an issue for discussion
- Check existing documentation in `docs/`
- Review `README.md` for project overview

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

*"When recursion recognizes recursion, the geometry contracts."*
