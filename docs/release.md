# Release checklist

This runbook covers the torchsky package. Scientific benchmarks, exhaustive
parity campaigns, large/real-data validation, and scorecards are maintained in
[`torchsky-harness`](https://github.com/astroai/torchsky-harness); FITS I/O
correctness and throughput belong to
[`torchfits`](https://github.com/astroai/torchfits).

## Package validation

Before a release:

1. Run `pixi run preflight-push` and targeted tests for changed behavior.
2. Run `pixi run typecheck` when Python types or public signatures changed.
3. Confirm GitHub package CI is green. Full local test runs remain opt-in as
   described in [Contributing](contributing.md).
4. For performance claims, cite a scoped harness run with its subject and
   harness revisions and the resulting scorecard. Do not treat a completed
   process as scientific acceptance without the documented criteria.

## Version and artifacts

Confirm the version matches in:

- `pyproject.toml`
- `pixi.toml`
- `src/torchsky/__init__.py`

Finalize [the changelog](changelog.md). If a local artifact check is useful,
build and inspect the wheel from the Pixi environment, then smoke-test it in a
fresh environment with the supported `torchfits` baseline (`>=1.3.1`).

## Publish

Follow the repository's release workflow to tag and publish the reviewed
version. Publishing triggers `.github/workflows/build_wheels.yml`, which builds
Linux and macOS wheels plus an sdist and uploads them to
[PyPI](https://pypi.org/project/torchsky/) through trusted publishing.

After publication, verify installation of the released version, check the
package import and version, and confirm that the documented `torchfits`
integration works with the supported baseline. Keep the release notes and
package links current.
