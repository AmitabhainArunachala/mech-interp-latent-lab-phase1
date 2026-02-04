# Security Policy

## Supported Versions

This research repository is actively maintained. Security updates apply to the latest version on the `main` branch.

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| develop | :white_check_mark: |
| other   | :x:                |

## Reporting a Vulnerability

We take security seriously, even in research code. If you discover a security vulnerability, please follow these steps:

### What Qualifies as a Security Issue?

- Arbitrary code execution vulnerabilities
- Dependency vulnerabilities in production code
- Data leakage or privacy issues
- Credential exposure risks
- Unsafe deserialization or file handling
- Other security-related concerns

### Reporting Process

1. **Do NOT** open a public issue for security vulnerabilities
2. Email the maintainer directly with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

3. You will receive acknowledgment within 48 hours
4. We will work to validate and fix the issue
5. Once fixed, we will publicly disclose (with credit to you, if desired)

## Security Best Practices

When using this repository:

### Model Weights

- Never commit API keys or access tokens
- Use environment variables for sensitive data
- Be cautious with HuggingFace API tokens

### Data Handling

- Don't commit large datasets
- Sanitize outputs before sharing
- Be aware of potential data leakage in model outputs

### Dependencies

- Use `requirements.lock` for reproducible, verified dependencies
- Regularly check for dependency vulnerabilities
- Update PyTorch and transformers to patched versions

### Code Execution

- This research code executes arbitrary Python
- Run experiments in isolated environments (containers, VMs)
- Don't run untrusted configs or code
- Validate inputs before processing

## Dependency Security

We use:
- **PyTorch** (2.1.x series): Check [PyTorch Security](https://pytorch.org/blog/security/)
- **Transformers** (4.36.x): Check [HuggingFace Security](https://huggingface.co/docs/hub/security)

Monitor:
```bash
pip install safety
safety check -r requirements.lock
```

## Responsible Disclosure

We appreciate responsible disclosure and will:
- Acknowledge your contribution
- Work with you on fixes
- Credit you in release notes (unless you prefer anonymity)

## Out of Scope

These are NOT security issues for this research repository:
- Performance issues
- Numerical precision differences
- Model outputs (this is a research tool, not production)
- Issues in third-party model weights

---

**Contact**: For security reports, contact the repository maintainer.

*Precision. Minimalism. Truth.*
