# Security Policy

## Supported Versions

We currently support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in this research repository, please report it privately to help us address it before public disclosure.

### What to Report

Please report any security concerns, including:

- Code execution vulnerabilities
- Dependency vulnerabilities in `requirements.txt` or `requirements.lock`
- Data integrity issues that could affect research reproducibility
- Model loading vulnerabilities (arbitrary code execution, pickle exploits)
- Any issues that could compromise research integrity

### How to Report

1. **Do not** open a public issue for security vulnerabilities
2. Email the maintainers directly (contact information in repository)
3. Provide a detailed description of the vulnerability
4. Include steps to reproduce the issue
5. Suggest a fix if possible

### What to Expect

- **Acknowledgment**: We will acknowledge receipt within 48 hours
- **Assessment**: We will assess the vulnerability within 1 week
- **Resolution**: We will work on a fix and coordinate disclosure timing
- **Credit**: We will credit reporters in security advisories (unless anonymity is requested)

## Security Best Practices

### For Users

When using this research code:

- Always use pinned dependencies from `requirements.lock` for production runs
- Review model checkpoints from HuggingFace before loading
- Use virtual environments to isolate dependencies
- Be cautious when loading user-provided prompts or data
- Validate input data before processing

### For Contributors

When contributing code:

- Never commit credentials, API keys, or sensitive data
- Use `torch.load(..., weights_only=True)` when loading model weights
- Validate all user inputs
- Use type hints and validation for function parameters
- Review dependencies for known vulnerabilities before adding them
- Follow the principle of least privilege for file system access

## Dependency Security

We maintain two dependency files:

- `requirements.txt`: Development dependencies with version ranges
- `requirements.lock`: Exact pinned versions for reproducibility

Both files are regularly reviewed for security vulnerabilities. To check for vulnerabilities:

```bash
pip install safety
safety check -r requirements.txt
```

## Model Loading Security

When loading transformer models from HuggingFace:

- Models are loaded with `torch_dtype=torch.float16` and `device_map="auto"`
- We use the official transformers library
- Model checkpoints should come from trusted sources
- Review model cards before loading new models

## Data Integrity

Research reproducibility depends on data integrity:

- All prompts are version-controlled in `prompts/bank.json`
- Results are timestamped and include configuration snapshots
- Random seeds must be set explicitly
- Hardware specifications are documented

## Responsible Disclosure

We follow responsible disclosure practices:

1. Security issues are addressed privately
2. Fixes are developed and tested before public disclosure
3. Security advisories are published after fixes are available
4. We coordinate with reporters on disclosure timing

---

*"Trust only what reproduces."*
