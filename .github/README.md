# .github Directory

This directory contains GitHub-specific configuration files for the repository.

## Contents

### Issue Templates (`ISSUE_TEMPLATE/`)

Standardized templates for creating issues:

- **bug_report.md**: For reporting bugs and unexpected behavior
- **feature_request.md**: For proposing new features or enhancements
- **research_question.md**: For proposing research questions or experimental investigations

### Workflows (`workflows/`)

GitHub Actions CI/CD pipelines:

- **ci.yml**: Continuous integration workflow
  - Linting with flake8
  - Code formatting checks with black
  - Type checking with mypy
  - Import verification
  - Reproducibility checks
  - Repository structure validation

### Templates

- **PULL_REQUEST_TEMPLATE.md**: Standard template for pull requests with comprehensive checklist

## Usage

These templates automatically populate when creating issues or pull requests on GitHub.

### For Contributors

1. Choose the appropriate issue template when reporting bugs or proposing features
2. Fill out all relevant sections of the template
3. Use the PR template checklist to ensure your contribution meets quality standards

### For Maintainers

- Update templates as project needs evolve
- Modify CI workflows to match testing requirements
- Keep templates aligned with CONTRIBUTING.md guidelines

---

*Precision. Minimalism. Truth.*
