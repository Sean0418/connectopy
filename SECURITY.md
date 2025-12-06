# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability within this project, please report it
responsibly by emailing the maintainers directly rather than opening a public issue.

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Resolution**: Depends on severity

## Data Privacy

This project processes Human Connectome Project (HCP) data. Users must:

1. Agree to HCP data usage terms at [ConnectomeDB](https://db.humanconnectome.org/)
2. Not redistribute raw HCP data
3. Follow HCP guidelines for derived data sharing

## Dependencies

We regularly update dependencies to address known vulnerabilities. Run:

```bash
pip install --upgrade -e ".[dev]"
```

to get the latest security patches.
