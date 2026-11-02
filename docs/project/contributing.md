# Contributing

This guide is for contributors and maintainers.
The short checklist is [CONTRIBUTING.md](https://github.com/astroai/torchsky/blob/main/CONTRIBUTING.md) in the repository root; this page is the full guide. Documentation authoring is in [Authoring docs](authoring.md) and releases in [Release process](release.md).

## Engineering Doctrine

- Favor scalable algorithms and data structures before adding native kernels.
- Avoid heuristic behavior unless it is isolated, justified, and benchmark-backed.
- Build torchsky-native primitives first; use existing astronomy libraries mainly as comparison targets, compatibility surfaces, and validation references.
- Preserve differentiability and tensor-native execution in public compute paths where practical.
- Validate substantial changes on both synthetic data and representative real or upstream-fixture data.
- Include benchmark evidence for performance-sensitive work, especially when changing default execution paths.
- Keep FITS/HDU/table/CFITSIO implementation work in `torchfits`; torchsky
  integration code should exchange decoded tensors and metadata.
- Rank work by massive workflow impact, correctness risk, asymptotic scale, and
  measured end-to-end performance. Avoid standalone micro-cleanups.

## Development Setup

Clone and install with the steps in
[CONTRIBUTING.md](https://github.com/astroai/torchsky/blob/main/CONTRIBUTING.md#setup).
The project uses [pixi](https://pixi.sh/) for environment management,
[ruff](https://github.com/astral-sh/ruff) for linting, and
[pytest](https://docs.pytest.org/) for testing. Use Pixi commands for project
work instead of a bare `python` or `python3` from your shell.

- Python package: `src/torchsky/`
- Native extension: `src/torchsky/cpp_src/`
- Tests: `tests/`
- Docs: `docs/`

Local files that are intentionally not committed:

- large downloaded real-data inputs under `data/`
- temporary staging files under `data/_tmp/`

Scientific benchmark inputs, evaluation outputs, and scorecards are owned by the
sibling `torchsky-harness` repository.

```
src/torchsky/          Python package
src/torchsky/cpp_src/  C++ native geometry/compute extension
src/torchsky/wcs/      Pure-PyTorch WCS implementation
src/torchsky/sphere/   HEALPix, spherical geometry, SHT
tests/                  Bounded unit, API, regression, adjoint, and device tests
docs/                   Documentation
examples/               Runnable example scripts
```

## Native extension

The C++ extension is built by
[scikit-build-core](https://scikit-build-core.readthedocs.io/) with
[nanobind](https://nanobind.readthedocs.io/) bindings. It must not link CFITSIO
or implement FITS/table operations.

Rebuild after C++ changes:

```bash
pixi run dev
```

Use the `torchfits` repository for CFITSIO build configuration.

## Test Expectations

The everyday commands (preflight, targeted tests, typecheck, docs contract) are
listed in
[CONTRIBUTING.md](https://github.com/astroai/torchsky/blob/main/CONTRIBUTING.md#checks).
Minimum pre-PR check:

```bash
pixi run -e test -- pytest tests/test_api.py tests/test_wcs_api.py -q
```

Full suite (the `pixi run test` task is the quicker lane and deliberately
ignores `tests/test_map_*.py`):

```bash
pixi run -e test -- pytest -n auto
```

Git pre-push hook (`pixi run setup-hooks`) is `pixi run preflight-push` only
(ruff + compileall). Full pytest is GitHub `build-test`. To run that suite
locally:

```bash
pixi run ci-local
```

## Scientific evaluation

Scientific benchmarks, exhaustive parity, real-data validation, comparator
environments, and CANFAR evaluation runs live in
[`torchsky-harness`](https://github.com/astroai/torchsky-harness). Core changes
retain bounded correctness and performance smoke contracts. See the
[evaluation guide](../performance/benchmarks.md) for the boundary and harness methodology for
run instructions.

## Documentation Policy

The published documentation site is built from `docs/` only. Unpublished
research, reviews, archives, and brainstorming stay outside git; implemented
behavior belongs in `docs/`.

- Root `README.md`: user-facing only.
- `docs/index.md`: site home and documentation map.
- `docs/quickstart.md`: the shortest working path.
- `docs/reference/index.md`: generated public API router; page identity and
  ordering live in `docs/reference/_registry.toml`.
- `docs/changelog.md`: release notes.
- `docs/examples.md` and `docs/evaluation/benchmarks.md`: operational guides.
- `docs/roadmap.md`: planned and in-flight work.
- Untracked research and brainstorming stay outside git; implemented behavior
  belongs in `docs/`.

Build and verify docs with Pixi:

```bash
pixi run docs-build     # zensical build --clean --strict
pixi run docs-serve     # local preview
pixi run docs-contract  # strict build + offline link check + docs tests
```

If a PR changes a public API, update the relevant namespace guide and
`docs/reference/_registry.toml` routing in the same PR, then run
`pixi run docs-reference`. See [authoring.md](authoring.md) for the page and
snippet conventions. FITS I/O documentation and benchmarks belong in
`torchfits`; update only the cross-package contract here.

## PR guidelines

The checklist is in
[CONTRIBUTING.md](https://github.com/astroai/torchsky/blob/main/CONTRIBUTING.md#pull-requests).
Do not commit local scratch files, benchmark artifacts, or `.env` files.

## Release process

Maintainers: see [release.md](release.md) for the canonical runbook. For
performance claims, record a harness run and link its subject revision,
harness revision, and scorecard in the PR.
