---
name: torchsky-dev
description: Develop torchsky (torch-native astronomy stack) — pixi workflow, AGENTS.md conventions, WCS/sphere/catalogs patterns
license: MIT
compatibility: opencode
metadata:
  project: torchsky
  stack: python, c++, pytorch, pixi
---

## Workflow

- Git pre-push hook: `pixi run preflight-push` (ruff + compileall)
- Targeted tests: `pixi run -e test -- pytest tests/test_foo.py`
- Optional local GitHub-suite parity: `pixi run ci-local`
- Python interpreter: `pixi run -e default -- python ...`

## Design rules

- No FITS I/O in core — use torchfits at boundaries
- torchsky owns WCS, sphere, maps, scenes, catalogs, field models
- Prefer torch over numpy in user-facing paths; preserve differentiability and device (CPU/CUDA/MPS)
- No heuristics where exact methods exist
- Algorithmic scalability before native kernels
- Compare against astropy/healpy in tests (extras), not as required imports
- Keep comments sparse; `ponytail:` prefix for deliberate shortcuts

## Common commands

- `pixi run lint` — ruff format
- `pixi run -e test -- python -m pytest tests/ -q --no-header -x`
- `pixi run -e bench -- python benchmarks/bench_*.py`
