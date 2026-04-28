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
git clone https://github.com/sfabbro/torchfits.git
cd torchfits
pixi install
pixi run test
```

The project uses [pixi](https://pixi.sh/) for environment management, [ruff](https://github.com/astral-sh/ruff) for linting, and [pytest](https://docs.pytest.org/) for testing.

- Python package: `src/torchsky/`
- Native extension: `src/torchsky/cpp_src/`
- Vendored native deps: `extern/cfitsio/`
- Tests: `tests/`
- Benchmarks: `benchmarks/`
- Docs: `docs/`

```
src/torchfits/          Python package
src/torchfits/cpp_src/  C++ native extension (nanobind + CFITSIO)
src/torchfits/wcs/      Pure-PyTorch WCS implementation
src/torchfits/sphere/   HEALPix, spherical geometry, SHT
extern/cfitsio/         Vendored CFITSIO sources
tests/                  Unit and integration tests
benchmarks/             Benchmark scripts and replay gates
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
pip install -e . --no-build-isolation
```

## Testing

- `-DTORCHSKY_USE_VENDORED_CFITSIO=OFF`

## Test Expectations

Minimum pre-PR checks:

```bash
pixi run pytest tests/test_api.py tests/test_table.py -q
```

Full suite:

```bash
pixi run test
```

Upstream parity gates (require comparison libraries):

```bash
pixi run wcs-upstream-gate
pixi run fits-upstream-gate
pixi run healsparse-upstream-gate
pixi run sphere-upstream-gate
```

## Benchmarks

Quick regression check:

```bash
pixi run python benchmarks/bench_sentinel.py --initial-repeats 3 --full-repeats 9
```

For investigation, narrow scope with `bench-core`, `bench-table`, `bench-cache`.
`bench-all` is exhaustive and can take over an hour on a laptop-class machine.

## Documentation Policy

- Root `README.md`: user-facing only.
- `docs/api.md`: public API reference.
- `docs/changelog.md`: release notes.
- `docs/examples.md` and `docs/benchmarks.md`: operational guides.

If a PR changes a public API, update `docs/api.md` in the same PR.

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

## Documentation policy

- `README.md`: user-facing overview only.
- `docs/api.md`: public API reference. Update if a PR changes a public API.
- `docs/changelog.md`: release notes, [Keep a Changelog](https://keepachangelog.com/) format.
- `docs/benchmarks.md`: benchmark methodology and results.
- `docs/sphere.md`: sphere/HEALPix API reference.

## PR guidelines

- Keep PRs focused on a single concern.
- Include tests for behavior changes.
- Run `pixi run lint` and fix issues before submitting.
- Do not commit local scratch files, benchmark artifacts, or `.env` files.

## Release process

See [release.md](release.md) for the maintainer checklist.
