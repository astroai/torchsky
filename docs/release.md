# Release Checklist

Maintainer runbook for cutting a release.

## 1. Version sync

Confirm the version triplet matches in:

- `pyproject.toml` (`version = "X.Y.Z"`)
- `pixi.toml`
- `src/torchsky/__init__.py`

## 2. Changelog

Finalize the entry in `docs/changelog.md`. Follow [Keep a Changelog](https://keepachangelog.com/) format.

## 3. Tests

```bash
pixi run test
```

All tests must pass.

## 4. Correctness gates

Run the supported local parity and benchmark gates:

```bash
pixi run ci-local
pixi run bench-wcs
pixi run bench-wcs-tpv-inverse
pixi run sphere-bench-geometry-gate
pixi run sphere-bench-healpix-advanced-gate
pixi run sphere-upstream-spin-matrix-gate
pixi run sphere-bench-geom-exact-contains-gate
pixi run bench-map-acceptance-smoke
```

All gates must pass. If upstream replay scripts are reintroduced, add their
tasks and documentation in the same change before making them release blockers.

## 5. Benchmark evidence

Run the exhaustive benchmark suite:

```bash
pixi run bench-all
```

Run the ML loader benchmark:

```bash
pixi run -e bench -- python benchmarks/bench_ml_loader.py \
  --device cpu --shape 2048,2048 --n-files 50 \
  --batch-size 4 --num-workers 4 --epochs 3 --repeats 5 --warm-cache
```

Update `docs/benchmarks.md` with the run ID, date, and snapshot tables.

## 6. Local artifact check (optional)

```bash
pip wheel . --no-deps --no-build-isolation -w dist
twine check dist/*
```

Smoke-test the wheel in a fresh virtualenv.

## 7. Tag and push

```bash
git add -A
git commit -m "release: vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

## 8. Publish

Create a GitHub release for `vX.Y.Z`.

Publishing triggers `.github/workflows/build_wheels.yml`, which:

1. Runs tests.
2. Builds wheels on Linux and macOS plus sdist.
3. Uploads to [PyPI](https://pypi.org/project/torchsky/) via trusted publishing.

## 9. Post-release verification

- [ ] `pip install torchsky==X.Y.Z` works in a fresh environment.
- [ ] `import torchsky; print(torchsky.__version__)` shows correct version.
- [ ] Core WCS and sphere imports run without `torchfits` or CFITSIO.
- [ ] The documented optional integration works with the supported `torchfits`
      0.5.x release.
- [ ] Changelog and release notes links resolve.
