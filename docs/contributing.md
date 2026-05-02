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

## Development Setup

```bash
git clone https://github.com/sfabbro/torchsky.git
cd torchsky
pixi install
./extern/vendor.sh
pixi run dev
pixi run verify
```

The project uses [pixi](https://pixi.sh/) for environment management,
[ruff](https://github.com/astral-sh/ruff) for linting, and
[pytest](https://docs.pytest.org/) for testing. Use Pixi commands for project
work instead of a bare `python` or `python3` from your shell.

- Python package: `src/torchsky/`
- Native extension: `src/torchsky/cpp_src/`
- Vendored native deps: `extern/cfitsio/`
- Tests: `tests/`
- Benchmarks: `benchmarks/`
- Docs: `docs/`

Local files that are intentionally not committed:

- large downloaded real-data inputs under `data/`
- generated benchmark artifacts under `data/benchmark_results/`
- temporary staging files under `data/_tmp/`

Fresh clones should recreate those products locally using the commands in
`docs/benchmarks.md`.

```
src/torchsky/          Python package
src/torchsky/cpp_src/  C++ native extension (nanobind + CFITSIO)
src/torchsky/wcs/      Pure-PyTorch WCS implementation
src/torchsky/sphere/   HEALPix, spherical geometry, SHT
extern/cfitsio/         Vendored CFITSIO sources
tests/                  Unit and integration tests
benchmarks/             Benchmark scripts and upstream-derived validation gates
docs/                   Documentation
examples/               Runnable example scripts
```

## Native extension

The C++ extension is built by [scikit-build-core](https://scikit-build-core.readthedocs.io/) with [nanobind](https://nanobind.readthedocs.io/) bindings. Populate vendored sources with:

```bash
./extern/vendor.sh
```

Rebuild after C++ changes:

```bash
pixi run dev
```

To build against system CFITSIO instead of the vendored source, pass the CMake
flag through pip/scikit-build:

```bash
python -m pip install . --config-settings=cmake.args="-DTORCHSKY_USE_VENDORED_CFITSIO=OFF"
```

## Test Expectations

Minimum pre-PR checks:

```bash
pixi run -e test -- pytest tests/test_api.py tests/test_table.py -q
```

Full suite:

```bash
pixi run test
```

Local pre-push gate:

```bash
pixi run preflight-push
pixi run typecheck
pixi run ci-local
```

## Benchmarks

Quick regression check:

```bash
pixi run bench-fast
```

For investigation, narrow scope with `bench-core`, `bench-table`,
`bench-cache`, or domain-specific tasks such as `bench-wcs`,
`bench-sphere`, and `bench-map-acceptance-smoke`. `bench-all` is
comprehensive and can take over an hour on a laptop-class machine.

## Documentation Policy

- Root `README.md`: user-facing only.
- `docs/README.md`: documentation map for users, contributors, maintainers, and
  research/history notes.
- `docs/api.md`: public API guide and namespace router.
- `docs/changelog.md`: release notes.
- `docs/examples.md` and `docs/benchmarks.md`: operational guides.

If a PR changes a public API, update the relevant namespace guide and `docs/api.md` routing in the same PR.

## Release Checklist (Maintainers)

Use `docs/release.md` as the canonical runbook.

Minimum gates:

1. Run release benchmarks and update `docs/benchmarks.md`.
2. Ensure `docs/changelog.md` is final and versions are synced in:
   - `pyproject.toml`
   - `pixi.toml`
   - `src/torchsky/__init__.py`
3. Ensure CI is green.
4. Build and validate artifacts:

```bash
pixi run bench-all
```

Include benchmark evidence in PRs that touch performance-sensitive paths.

## PR guidelines

- Keep PRs focused on a single concern.
- Include tests for behavior changes.
- Run `pixi run lint` and fix issues before submitting.
- Do not commit local scratch files, benchmark artifacts, or `.env` files.

## Release process

See [release.md](release.md) for the maintainer checklist.
