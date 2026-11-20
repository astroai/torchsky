# Release checklist

This runbook covers the torchsky package. Scientific benchmarks, exhaustive
parity campaigns, large/real-data validation, and scorecards are maintained in
[`torchsky-harness`](https://github.com/astroai/torchsky-harness); FITS I/O
correctness and throughput belong to
[`torchfits`](https://github.com/astroai/torchfits).

## 1. Supported PyTorch range

The native core neither includes nor links libtorch: tensors cross through
DLPack and operators are registered from Python with `torch.library`. It is
built against the CPython stable ABI. A release is therefore **one
`cp312-abi3` wheel per platform**. Metadata gives a PyTorch floor and no upper
bound. `scripts/torch_support.json` is the single source of truth:

| Key | Meaning |
|---|---|
| `torchsky_version` | the release being prepared |
| `torch_min` | oldest PyTorch minor CI tests; the runtime floor (`torch>=…`) |
| `torch_tested` | newest PyTorch minor CI tests |
| `torchfits` | the torchfits requirement |

Raise `torch_min` only when torchsky starts to need a newer PyTorch API.
Record each newly tested PyTorch minor in `torch_tested`.

Apply the version to every source in one step and verify it:

```bash
pixi run -e default -- python scripts/release_lane.py --apply --prerelease rc1   # 0.6.0rc1
pixi run -e default -- python scripts/release_lane.py --apply                    # finalize: 0.6.0
pixi run release-lane-check
```

`--apply` rewrites these files:

- `pyproject.toml`: the version, `torch>=<min>` and the `torchfits` requirement;
- `pixi.toml`: the package version, the `pytorch >=<min>` pins and the
  `torch-min` feature;
- `src/torchsky/__init__.py` and `CITATION.cff`: the version.

`release-lane-check` also fails when a workflow hard-codes a torch pin.
Workflows read the pins from `scripts/release_lane.py --print-pins`
(`VERSION`, `TORCH_PIN`, `TORCH_MIN_PIN`, `TORCH_TESTED_PIN`, `TORCHFITS_PIN`).

## 2. Package validation

Before tagging a candidate:

1. `pixi run preflight-push` and targeted tests for changed behavior.
2. `pixi run typecheck` when Python types or public signatures changed.
3. `pixi run release-lane-check`.
4. Confirm GitHub package CI is green. Full local test runs remain opt-in as
   described in [Contributing](contributing.md).
5. For performance claims, cite a scoped harness run with its subject and
   harness revisions and the resulting scorecard. Do not treat a completed
   process as scientific acceptance without the documented criteria.

## 3. Changelog

[The changelog](changelog.md) keeps the in-flight work under the heading of the
version being released, for example `## [0.5.0rc1] - Unreleased`. Set the date
when tagging. For the stable cut, rename the heading to `## [0.5.0] - <date>`
and update the link reference at the bottom. `scripts/release_notes.py` builds
the GitHub release body from the section named after the tag (falling back to
`## [Unreleased]`).

## 4. Release candidate

Every stable tag is preceded by an `rc` tag for the same version, and the stable
cut changes only the version strings and the changelog.

```bash
git tag v0.5.0rc1
git push origin v0.5.0rc1
```

Pushing a tag runs `.github/workflows/build_wheels.yml`:

1. `release_gate` classifies the tag (`vX.Y.Z` stable, `vX.Y.ZrcN` release
   candidate; anything else fails), requires the tag to point at the checked-out
   commit, runs `release_lane.py --check`, and requires the version to equal the
   tag. For a stable tag it also requires a `vX.Y.Zrc*` tag, a tree that differs
   from that rc only in version files and the changelog
   (`scripts/check_release_rc.py`), and a successful push run for the rc at the
   same commit (`scripts/check_release_run.py`).
2. `tests` and `map_tests` run the retained suites in the locked Pixi
   environments (Python 3.12 and 3.13).
3. `build_wheels` builds one `cp312-abi3` wheel each for manylinux_2_28 x86_64
   and for macOS arm64, with deployment target 14.0 (the floor of the PyTorch
   and torchfits wheels). The build installs no PyTorch. cibuildwheel then tests
   each wheel on CPython 3.12, 3.13 and 3.14. Each wheel passes
   `scripts/check_distribution.py` and the install smoke test; macOS wheels
   also pass `scripts/check_macos_wheel_floor.py` (filename tag and Mach-O
   `minos`). `build_sdist` builds the sdist, validates it
   (`scripts/validate_release_sdist.py`), builds a wheel from it and smoke-tests
   that wheel.
4. `validate_release` checks:
   - the exact two-wheel `cp312-abi3` matrix;
   - the metadata version and `Requires-Python`;
   - the `torch>=<torch_min>` (uncapped) and torchfits requirements (`scripts/validate_release_wheels.py`), and writes
   `RELEASE-MANIFEST.sha256` next to the artifacts.
5. `github_release` publishes a GitHub **prerelease** with the artifacts and the
   changelog section. A release-candidate tag never uploads to PyPI.

Install the candidate from the prerelease assets in a fresh environment and run
the post-release checks in section 6 against it. Fix problems with a new
candidate (`rc2`, ...).

## 5. Stable release

Finalize the version and changelog (section 1 and 3), commit, tag, and push:

```bash
pixi run -e default -- python scripts/check_release_rc.py --version 0.5.0
git tag v0.5.0
git push origin main v0.5.0
```

The stable tag push builds and validates only. After it succeeds, publish by
running the workflow manually from the exact stable tag:

```bash
gh workflow run build_wheels.yml -f tag=v0.5.0 -f publish=true
```

Only a `workflow_dispatch` run with `publish=true`, on a stable tag, on
`astroai/torchsky`, publishes. `upload_pypi` re-verifies the sha256 manifest
against the validated artifacts, then uploads the wheels and sdist through
[PyPI](https://pypi.org/project/torchsky/) trusted publishing (the `pypi`
GitHub Environment, OIDC `id-token: write`; there is no API token). Fork runs
cannot publish. `github_release` then creates the stable GitHub release.

## 6. Post-release verification

- `pip install torchsky==X.Y.Z` works in a fresh environment with the supported
  `torchfits` baseline.
- `import torchsky; print(torchsky.__version__)` shows the released version and
  the documented `torchfits` integration works.
- [Stable docs](https://astroai.github.io/torchsky/) load and the changelog and
  release-note links resolve.

## Local artifact checks (optional)

Build an sdist or wheel from the Pixi environment and validate it with the same
scripts the workflow uses:

```bash
pixi run -e default -- python -c "from scikit_build_core.build import build_sdist; build_sdist('dist')"
pixi run -e default -- python scripts/validate_release_sdist.py dist/torchsky-X.Y.Z.tar.gz --version X.Y.Z
pixi run -e test -- python scripts/check_distribution.py --sdist dist/torchsky-X.Y.Z.tar.gz --skip-twine
```
