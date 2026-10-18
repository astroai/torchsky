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

- `pixi run preflight-push` — Ruff and compileall pre-push gate
- `pixi run typecheck` — pyright
- `pixi run test` — package test task (map tests run in the CI map matrix)
- `pixi run -e test -- pytest tests/test_foo.py` — targeted package tests

Scientific benchmarks, exhaustive parity, and real-data validation run from
the sibling `torchsky-harness` checkout against an explicitly installed
torchsky subject. Record the two repository revisions separately; consult the
harness's `AGENTS.md` and benchmark methodology for its current commands.
