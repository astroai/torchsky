# Contributing to torchsky

Thanks for your interest. torchsky is a torch-native astronomy stack (WCS,
HEALPix, mapmaking, observation operators, field models). Please read the
[Code of Conduct](CODE_OF_CONDUCT.md) first. The full guide is
[docs/project/contributing.md](docs/project/contributing.md); repository policy for coding
agents and maintainers is in [AGENTS.md](AGENTS.md).

## Setup

torchsky uses [pixi](https://pixi.sh/). Use `pixi run ...` for all Python; do not
call a bare `python` from your shell.

```bash
git clone https://github.com/<you>/torchsky.git
cd torchsky
pixi install
pixi run dev        # build the native extension
pixi run verify     # import smoke test
pixi run setup-hooks  # optional pre-push hook (ruff + compileall)
```

## Checks

```bash
pixi run preflight-push                           # ruff + compileall
pixi run -e test -- pytest tests/test_foo.py      # targeted tests
pixi run test                                     # main suite (excludes map tests)
pixi run -e test -- pytest -n 4 -q tests/test_map_*.py   # map tests
pixi run typecheck                                # pyright, when types change
pixi run docs-contract                            # strict docs build and docs tests
pixi run examples                                 # example smoke runner
```

Run the targeted tests for what you changed; CI runs the full suite.

## Pull requests

- Fork, branch from `main`, and open a single focused PR against `main`.
- Add tests for behavior changes (bounded unit, regression, adjoint, gradient,
  or device contracts). Scientific benchmarks live in
  [torchsky-harness](https://github.com/astroai/torchsky-harness).
- Performance PRs state the bottleneck removed, why it scales, and the
  benchmark evidence.
- Public API changes update the matching guide, `docs/reference/_registry.toml`,
  and `docs/api-manifest.json` (`pixi run api-manifest`).
- Keep FITS I/O in [torchfits](https://github.com/astroai/torchfits); never
  import `astropy.io.fits` or `fitsio`.
- Add a line to `docs/project/changelog.md` for user-visible changes.

Maintainers: see [docs/project/release.md](docs/project/release.md) for releases.
