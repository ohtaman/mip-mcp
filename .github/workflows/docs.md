# CI/CD Workflows

This directory contains GitHub Actions workflows for the MIP-MCP project.

## Workflows

### 1. CI (`ci.yml`)
**Triggers:** Pull requests to main, pushes to main

**Jobs:**
- **test**: Runs tests and linting on Python 3.12
  - Installs dependencies with `uv`
  - Runs `make test` (unit tests)
  - Runs `make lint` (ruff linting)
  - Checks code formatting with ruff

- **coverage**: Runs only on pull requests
  - Generates test coverage reports
  - Uploads coverage to Codecov (optional)

### 2. Main Branch CI/CD (`main.yml`)
**Triggers:** Pushes to main branch only

**Jobs:**
- **test-and-build**: Complete testing and build process
  - Runs all tests and linting
  - Builds the Python package with `uv build`
  - Tests package installation
  - Archives build artifacts

## Adding CI Status Badge

Add this badge to your README.md to show CI status:

```markdown
[![CI](https://github.com/YOUR_USERNAME/mip-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/mip-mcp/actions/workflows/ci.yml)
```

## Requirements

- Python 3.12
- uv package manager
- All dependencies listed in `pyproject.toml`

## Local Testing

Test the CI commands locally:

```bash
# Install dependencies
uv sync --dev

# Run tests (same as CI)
make test

# Run linting (same as CI)
make lint

# Check formatting
uv run ruff format --check src/ tests/
```
