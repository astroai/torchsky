# GitHub Workflows

This directory contains the GitHub Actions CI/CD workflows for torchsky.

## Active Workflows

### `ci.yml` - Comprehensive CI Pipeline
**Triggers:** Push to main/develop branches, Pull Requests, Releases
**Purpose:** Complete build, test, quality assurance, and deployment pipeline

**Jobs:**
1. **Code Quality** - Lint and scoped typecheck (Python 3.12)
   - **ruff** (lint)
   - **pyright** (allowlisted modules; `pyrightconfig.json`)

2. **Build & Test** - Editable install and full pytest
   - **OS:** Ubuntu
   - **Python:** 3.12
   - **System deps:** build-essential (C++ extension)
   - **Tests:** `pytest -n auto` with **`[test]`** extras (`pip install -e ".[test]"`)

3. **Memory Safety** - Memory leak detection (placeholder for valgrind)

4. **Performance Benchmarks** - Performance regression testing (placeholder)

5. **Documentation** - Documentation build verification (placeholder)

6. **Publish** - Automated PyPI publishing on GitHub releases

**Key Features:**
- C++ extension compilation (vendored CFITSIO / scikit-build)
- Code quality + scoped Pyright + parallel pytest
- Automated PyPI releases (if configured)

## CI Configuration

- **Python version:** 3.12 (see `requires-python` in `pyproject.toml`)
- **Code quality:** ruff, pyright
- **Local parity:** `pixi run ci-local` before push (see `AGENTS.md`)
- **PyPI Publishing:** Requires `PYPI_API_TOKEN` repository secret

## Setup for PyPI Publishing

Add the following secret to your GitHub repository settings:
- `PYPI_API_TOKEN`: Your PyPI API token

The workflow automatically publishes to PyPI when you create a GitHub release.
