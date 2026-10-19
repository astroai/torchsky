---
name: torchsky-dev
description: Develop torchsky (torch-native astronomy stack) — pixi workflow, AGENTS.md conventions, WCS/sphere/catalogs/map patterns
metadata:
  project: torchsky
  stack: python, c++, pytorch, pixi
---

# torchsky development

Repository operating rules live in `AGENTS.md` at the repo root — read it before
changing code. This skill covers the daily workflow and design constraints.

## Environment (pixi required)

Never guess `python3` or bare `python` on PATH. Use pixi tasks; the repo's
Python and dependencies come from the pixi environment.

- The default (`default`) dev env is lean: build tools + pytest driver only.
  Ecosystem astronomy packages live in the relevant feature-gated environments
  such as `test`; scientific comparator environments live in
  `torchsky-harness`. None are required imports of `import torchsky`.

## Workflow

- Git pre-push hook is `pixi run preflight-push` (ruff + compileall). Do not
  run the full pytest suite as a push blocker.
- Targeted tests: `pixi run -e test -- pytest tests/test_foo.py`.
- Optional local GitHub-suite parity: `pixi run ci-local`.
- Python interpreter: `pixi run -e default -- python ...`.
- `pixi run test` runs `pytest -n auto --ignore-glob='tests/test_map_*.py'` —
  map tests are EXCLUDED and run in the CI map matrix; see the `map-suite`
  skill before claiming the suite is green.

## Design rules (from AGENTS.md)

- No FITS/CFITSIO code in core — `torchfits` owns file I/O; torchsky accepts
  in-memory tensors/metadata at boundaries. Do not restore `torchsky.table`.
- torchsky owns sky coordinates, sphere/HEALPix, SHT, maps, scenes, catalogs,
  field models, and WCS.
- Prefer torch over numpy in user-facing paths; preserve differentiability and
  device behavior (CPU baseline; CUDA/MPS where the module claims it).
- No heuristics where exact or well-founded methods exist; isolate and document
  any temporary heuristic with benchmark or regression justification.
- Algorithmic scalability (indexing, chunking, streaming, sparsity) before
  native kernels; add C++ only for proven hotspots, with parity tests against
  the non-native path.
- Compare against astropy/healpy/etc. in `tests/`, `scripts/`, and CI extras —
  never as required imports of the core package. Scientific benchmarks and
  exhaustive parity live in `torchsky-harness`.
- Keep comments sparse and high-signal; `ponytail:` prefix for deliberate
  shortcuts.

## Common commands

- `pixi run lint` — ruff check + format
- `pixi run typecheck` — pyright (test env)
- `pixi run verify` — torchsky + `torchsky.cpp` import smoke
- `pixi run dev` — rebuild native extensions after C++ changes
- `pixi run test` — package test task; the GitHub map matrix covers excluded map tests

For scientific evaluation, install the intended torchsky subject and run the
[`torchsky-harness`](https://github.com/astroai/torchsky-harness) CLI from its
own checkout. Record the harness and subject revisions separately; see the
harness's `AGENTS.md` and benchmark methodology.
