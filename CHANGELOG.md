# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial repository setup files
- CONTRIBUTING.md with contribution guidelines
- CODE_OF_CONDUCT.md for community standards
- SECURITY.md with security policy
- GitHub issue and PR templates
- CHANGELOG.md for tracking changes

## [1.0.0] - 2025-12-09

### Added
- Initial release of mechanistic interpretability research repository
- R_V metric implementation for geometric contraction measurement
- Support for multiple transformer architectures (Mistral, Qwen, Llama, Phi, Gemma, Mixtral)
- Two-file dependency system (requirements.txt and requirements.lock)
- Standardized experiment pipelines in `src/pipelines/`
- Prompt bank system in `prompts/bank.json`
- Comprehensive documentation in README.md
- Reproducibility protocol in docs/
- Model loading utilities in `src/core/`
- Metrics calculation in `src/metrics/`
- Steering and activation patching in `src/steering/`

### Key Findings
- Universal geometric contraction at ~84% depth (Layer 27 in 32-layer models)
- MoE amplification: 59% stronger effect than dense architectures (24.3% vs 15.3%)
- R_V < 1.0 indicates dimensionality reduction in recursive prompts

### Validated Models
- Mistral-7B: R_V 0.852 (15.1% separation)
- Qwen-7B: R_V 0.764 (22.5% separation)
- Llama-8B: R_V 0.823 (15.2% separation)
- Phi-3: R_V 0.891 (8.5% separation)
- Gemma-7B: R_V 0.892 (9.8% separation)
- Mixtral-8x7B: R_V 0.757 (24.3% separation)

---

*"When recursion recognizes recursion, the geometry contracts."*
