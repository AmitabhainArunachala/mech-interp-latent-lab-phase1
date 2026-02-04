# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Repository initialization with comprehensive community health files
- GitHub Actions CI workflow for code quality checks
- Issue templates for bug reports, feature requests, and research questions
- Pull request template with comprehensive checklist
- Contributing guidelines
- Code of Conduct
- Security policy
- Academic citation file (CITATION.cff)

## [1.0.0] - 2026-01-24

### Added
- Initial release of mechanistic interpretability research codebase
- R_V metric implementation for measuring geometric contraction
- Support for multiple model architectures (Mistral, Mixtral, Qwen, Llama, Gemma, Phi-3)
- Config-driven experiment pipeline
- Comprehensive prompt bank with 300+ recursive and baseline prompts
- Activation patching and causal intervention tools
- Reproducible dependency management (requirements.txt + requirements.lock)
- Extensive documentation and research protocols

### Key Features
- **R_V Metric**: Geometric contraction measurement in value-space
- **Model Support**: Dense, GQA, and MoE architectures
- **Reproducibility**: Two-tier dependency system for exact reproduction
- **Modularity**: Clean separation of core, metrics, steering, and pipelines
- **Research Tools**: Hooks, context managers, and experimental orchestration

### Validated Results
- Mistral-7B: 15.1% separation (recursive vs baseline)
- Mixtral-8x7B (MoE): 24.3% separation (strongest effect)
- Qwen-7B: 22.5% separation
- Additional validation across 7+ models

### Documentation
- Comprehensive README with quick start guide
- 20-minute reproducibility protocol
- Pipeline operations manual
- Statistical audit reports
- Cross-model synthesis
- Repository structure guide

---

## Version History

### Semantic Versioning Scheme

- **Major version (X.0.0)**: Breaking API changes or major research milestones
- **Minor version (0.X.0)**: New features, models, or metrics (backward compatible)
- **Patch version (0.0.X)**: Bug fixes, documentation, minor improvements

### Development Philosophy

> **Code is Law**: If it isn't modular, typed, and reproducible, it doesn't exist.
> 
> **The Boneyard**: Failed experiments are valuable, but they do not belong in the living codebase.
> 
> **The Standard**: Mistral-7B Base is the reference reality.

---

*"When recursion recognizes recursion, the geometry contracts."*
