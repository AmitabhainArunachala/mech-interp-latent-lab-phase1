## Description

<!-- Provide a clear and concise description of your changes -->

## Type of Change

<!-- Mark the relevant option with an 'x' -->

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Research experiment (new analysis or validation)
- [ ] Code refactoring

## Motivation and Context

<!-- Why is this change required? What problem does it solve? -->
<!-- Link to related issues: Fixes #(issue number) -->

## How Has This Been Tested?

<!-- Describe the tests you ran to verify your changes -->

- [ ] Tested with Mistral-7B Base (reference model)
- [ ] Tested with other models (specify):
- [ ] Ran existing test suite
- [ ] Added new tests
- [ ] Verified reproducibility (consistent results across runs)

**Test Configuration**:
- Python version:
- PyTorch version:
- Hardware:
- Random seed(s) used:

## Reproducibility Checklist

<!-- Ensure your changes maintain reproducibility -->

- [ ] Used exact dependencies from `requirements.lock` for validation
- [ ] Set random seeds explicitly
- [ ] Documented hardware requirements
- [ ] Included exact command-line invocations
- [ ] Results are consistent across multiple runs

## Code Quality Checklist

- [ ] My code follows the style patterns in this repository
- [ ] I have commented my code where necessary
- [ ] I have updated the documentation (README, docs/)
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing tests pass locally with my changes
- [ ] I have used context managers for all model hooks
- [ ] No hooks remain attached after functions return

## Research Checklist

<!-- If this is a research contribution -->

- [ ] Sample size ≥ 80 pairs (if applicable)
- [ ] Statistical tests include Bonferroni correction (if applicable)
- [ ] Effect sizes reported (|d| ≥ 0.5 for meaningful effects)
- [ ] Null results are documented (if applicable)
- [ ] Results include configuration snapshots

## Documentation

- [ ] Updated README.md (if needed)
- [ ] Updated CHANGELOG.md
- [ ] Added docstrings to new functions
- [ ] Updated relevant docs/ files

## Additional Notes

<!-- Any additional information that reviewers should know -->

---

**Philosophy Check**: Does this change embody "Precision. Minimalism. Truth."?

<!-- 
- Precision: Is it exact and reproducible?
- Minimalism: Does it include only what's necessary?
- Truth: Is it backed by data and code?
-->
