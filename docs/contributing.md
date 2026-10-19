# Contributing

This guide is for contributors and maintainers.
User-facing usage lives in `../README.md`.
Repository-specific coding-agent and engineering guidance lives in `../AGENTS.md`.

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

```bash
git clone https://github.com/astroai/torchsky.git
cd torchsky
pixi install
pixi run dev
pixi run verify
```

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

Scientific benchmark inputs, campaign outputs, and scorecards are owned by the
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

Minimum pre-PR checks:

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

Targeted checks while editing:

```bash
pixi run preflight-push
pixi run typecheck
pixi run -e test -- pytest tests/test_foo.py
```

## Scientific evaluation

Scientific benchmarks, exhaustive parity, real-data validation, comparator
environments, and CANFAR campaigns run from
[`torchsky-harness`](https://github.com/astroai/torchsky-harness). Core changes
retain bounded correctness and performance smoke contracts. See the
[evaluation guide](benchmarks.md) for the boundary and harness methodology for
run instructions.

## Documentation Policy

The published documentation site is built from `docs/` only. Unpublished
research, reviews, archives, and brainstorming stay outside git; implemented
behavior belongs in `docs/`.

- Root `README.md`: user-facing only.
- `docs/index.md`: site home and documentation map.
- `docs/quickstart.md`: the shortest working path.
- `docs/api.md`: public API guide and namespace router.
- `docs/changelog.md`: release notes.
- `docs/examples.md` and `docs/benchmarks.md`: operational guides.
- `notes/planned.md`: the single tracked plan for work that is committed for a
  future release.
- Untracked research, reviews, archives, and brainstorming stay outside git
  (ignored under `notes/`); implemented behavior belongs in `docs/`, not notes.

Build and verify docs with Pixi:

```bash
pixi run docs-build     # zensical build --clean --strict
pixi run docs-serve     # local preview
pixi run docs-contract  # strict build + offline link check + docs tests
```

If a PR changes a public API, update the relevant namespace guide and
`docs/api.md` routing in the same PR. FITS I/O documentation and benchmarks
belong in `torchfits`; update only the cross-package contract here.

## Release Checklist (Maintainers)

Use `docs/release.md` as the canonical runbook.

Release and package validation:

1. Run `pixi run preflight-push` and the targeted tests for changed behavior.
   Run `pixi run typecheck` when Python types or public signatures change.
2. Ensure `docs/changelog.md` is final and versions are synced in:
   - `pyproject.toml`
   - `pixi.toml`
   - `src/torchsky/__init__.py`
3. Ensure GitHub package CI is green and build artifacts as required by the
   release workflow. Full local test gates are opt-in under `AGENTS.md`.

For performance claims, record an appropriately scoped harness run and link its
subject revision, harness revision, and scorecard in the PR. Do not make
scientific campaigns implicit core release gates.

## PR guidelines

- Keep PRs focused on a single concern.
- Include tests for behavior changes.
- Run `pixi run lint` and fix issues before submitting.
- Do not commit local scratch files, benchmark artifacts, or `.env` files.

## Release process

See [release.md](release.md) for the maintainer checklist.
