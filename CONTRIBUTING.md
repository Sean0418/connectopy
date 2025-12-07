# Contributing Guidelines

Thank you for your interest in contributing to Connectopy! We welcome contributions from the community.

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/Brain-Connectome.git
cd Brain-Connectome
```

### 2. Set Up Development Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,docs]"
pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

## Development Workflow

### Running Tests

```bash
pytest
```

### Linting

```bash
ruff check .
ruff format .
```

### Building Documentation

```bash
cd docs
make html
```

## Submitting Changes

### Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add your changes to CHANGELOG.md (if exists)
4. Submit a pull request to the `main` branch

### Commit Message Format

Use descriptive commit messages:

```
feat: add new dimorphism visualization
fix: handle missing values in data loader
docs: update installation instructions
test: add tests for PCA module
```

Prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `style:` - Formatting changes

## Code Style

- Follow PEP 8 guidelines
- Use NumPy-style docstrings
- Maximum line length: 100 characters
- Use type hints for function signatures

### Example

```python
def analyze_dimorphism(
    data: pd.DataFrame,
    feature_columns: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Analyze sexual dimorphism in brain connectivity features.

    Parameters
    ----------
    data : DataFrame
        Dataset with features and Gender column.
    feature_columns : list of str
        Feature column names to analyze.
    alpha : float, default=0.05
        Significance threshold.

    Returns
    -------
    results : DataFrame
        Statistical test results.
    """
    ...
```

## Reporting Issues

### Bug Reports

Include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/tracebacks

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternative approaches considered

## Code of Conduct

Please review our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## Questions?

Open an issue or reach out to the maintainers.
