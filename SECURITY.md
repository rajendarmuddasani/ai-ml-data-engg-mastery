# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this repository, please report it responsibly:

### 🔒 Private Reporting (Preferred)

1. **Do NOT** create a public GitHub issue
2. **Do NOT** post in Discussions
3. **Email** the maintainer privately or use GitHub's private vulnerability reporting feature
4. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### ⏱️ Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity (critical: 24-72 hours, high: 1-2 weeks, medium/low: best effort)

---

## Security Considerations for This Repository

### What This Repository Contains

This is an **educational repository** containing:
- ✅ Jupyter notebooks (Python code, markdown)
- ✅ Documentation (markdown files)
- ✅ Synthetic/example datasets (no real sensitive data)
- ❌ **NO** production code deployed to servers
- ❌ **NO** real user data
- ❌ **NO** API keys, credentials, or secrets

### Potential Risks

While primarily educational, be aware of:

1. **Code Execution**: Notebooks contain executable Python code
   - Always review code before running
   - Use virtual environments
   - Don't run notebooks from untrusted sources

2. **Dependencies**: Third-party packages may have vulnerabilities
   - Keep packages updated: `pip install --upgrade -r requirements.txt`
   - Use `pip-audit` or `safety` to scan for known vulnerabilities

3. **Data Privacy**: When adapting notebooks for real data
   - ⚠️ **Never commit sensitive data** (customer info, proprietary semiconductor data, credentials)
   - Use `.gitignore` to exclude data files
   - Anonymize data before using in examples

---

## Best Practices for Users

### When Using These Notebooks:

✅ **DO:**
- Run notebooks in isolated virtual environments
- Review code before execution
- Keep dependencies updated
- Use version control for your modifications
- Anonymize any real data you use

❌ **DON'T:**
- Commit API keys, passwords, or credentials
- Include proprietary semiconductor test data (STDF files with real device info)
- Run untrusted code without reviewing
- Share notebooks containing sensitive information publicly

### Setting Up Securely:

```bash
# Create isolated environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt

# Scan for vulnerabilities (optional but recommended)
pip install pip-audit
pip-audit

# Or use safety
pip install safety
safety check
```

### Protecting Sensitive Data:

```python
# ❌ BAD: Hardcoded credentials
api_key = "sk-1234567890abcdef"

# ✅ GOOD: Environment variables
import os
api_key = os.environ.get('API_KEY')

# ✅ GOOD: External config file (add to .gitignore)
import json
with open('config.json') as f:
    config = json.load(f)
    api_key = config['api_key']
```

---

## Dependency Security

### Current Dependencies (Core)

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
xgboost>=1.5.0
jupyter>=1.0.0
```

### Monitoring for Vulnerabilities

We monitor dependencies for known vulnerabilities. If you find a vulnerability:

1. Check if it affects this repository's usage
2. Report via GitHub Issues or private vulnerability report
3. We'll update dependencies or provide mitigation guidance

### Updating Dependencies

```bash
# Check for outdated packages
pip list --outdated

# Update specific package
pip install --upgrade package-name

# Update all packages (test thoroughly after)
pip install --upgrade -r requirements.txt
```

---

## Notebook Execution Safety

### Jupyter Notebook Security

Jupyter notebooks can execute arbitrary code. **Always review notebooks before running.**

**Security features:**
- Notebooks from untrusted sources show "Not Trusted" indicator
- Review all code cells before executing
- Use `nbconvert` to sanitize notebooks: `jupyter nbconvert --clear-output notebook.ipynb`

**Red flags in notebooks:**
- Obfuscated code (base64 encoding, exec())
- Network requests to unknown URLs
- File system operations (deleting files, reading sensitive data)
- Subprocess/shell commands

---

## Data Privacy Compliance

### For Contributors:

If contributing notebooks with real-world examples:

✅ **Allowed:**
- Synthetic/generated data
- Public datasets with proper attribution
- Anonymized aggregate statistics

❌ **Not Allowed:**
- Personal identifiable information (PII)
- Proprietary semiconductor test data with device serial numbers
- Customer information
- API keys, credentials, tokens

### STDF Semiconductor Data:

When using STDF files:
- Anonymize wafer IDs, lot numbers, device serial numbers
- Aggregate spatial data to prevent reverse-engineering fab location
- Remove timestamps that could identify production batches
- Use synthetic STDF files for examples when possible

---

## Incident Response

### If a Security Issue is Discovered:

1. **Containment**: Identify affected notebooks/code
2. **Assessment**: Determine severity and impact
3. **Remediation**: Fix vulnerability, update documentation
4. **Notification**: Inform users via:
   - GitHub Security Advisory (if applicable)
   - README.md update
   - CHANGELOG.md entry
5. **Prevention**: Update security practices to prevent recurrence

---

## Additional Resources

- [GitHub Security Best Practices](https://docs.github.com/en/code-security)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/security.html)
- [Jupyter Notebook Security](https://jupyter-notebook.readthedocs.io/en/stable/security.html)

---

## Contact

For security concerns: Use GitHub's private vulnerability reporting or email the maintainer.

**Thank you for helping keep this educational resource safe!** 🔒

---

*Last Updated: December 2025*
