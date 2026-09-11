# AGENTS.md

This file gives repository-specific operating guidance for coding agents and maintainers working in `torchsky`.

## Mission

`torchsky` is building a torch-native astronomy stack for:

- very high performance data processing on large workloads
- multi-wavelength and multi-modal sky data
- scalability to many very large datasets
- continuous, field-like modelling / prediction / generation where that is technically coherent
- differentiable pipelines where that is technically coherent
- cross-device execution across CPU, CUDA, and MPS when practical
- compatibility layers with established astronomy libraries, without forcing primitive-for-primitive mimicry where a higher-level compatibility surface is better

FITS file and table I/O is owned by the sibling
[`torchfits`](https://github.com/astroai/torchfits) package (a required runtime
dependency). `torchsky` owns sky coordinates, geometry, maps, observation
operators, scenes, and field models. Exchange tensors, header mappings, table
batches, and small protocols across this boundary; `torchsky` does not contain
any FITS/CFITSIO code.

Never import `astropy.io.fits` or Python `fitsio`. Read and write FITS with `torchfits` (or the
thin `torchsky.table` / `torchsky.io.fits` facades that re-export it). Header
cards are plain dicts. Astropy WCS is a parity oracle and takes those dicts;
it is not a FITS I/O layer.

The project should prefer designs that remain viable at survey scale, not just toy examples or notebook-sized inputs.
The core should not collapse into a thin wrapper around legacy astronomy packages.
Geometric primitives, mapmaking, and cross-wavelength field representations are central building blocks, not side surfaces.

## Dependencies, installation, and ecosystem packages

### PyPI / `pyproject.toml` contract

- **Runtime (`[project].dependencies`)** is intentionally small: **`torch`**,
  **`numpy`**, and only compiled geometry/WCS kernels justified by benchmarks.
  FITS/CFITSIO belongs to `torchfits`, not the `torchsky` wheel. `torchfits` is
  itself a required runtime dependency (the FITS I/O layer for `torchsky`).
- **Heavy or domain-ecosystem packages** belong in **`[project.optional-dependencies]`** (`test`, `bench`, `viz`, `map`, `examples`, …), not in the core runtime list. Use them for tests, benchmarks, viz servers, map pipelines, and parity—not as silent requirements of `import torchsky`.
- **Table interchange:** prefer **PyArrow**-native paths and **`polars`** where a DataFrame-like API is needed; **`pandas`** remains optional (e.g. `to_pandas`, bench scripts).
- **Supported Python metadata:** **`>=3.12`** with no upper cap. The reproducible primary development lane uses **3.13.14**.
- **Pip extras:** **`[test]`** for pytest + fixtures aligned with CI; **`[bench]`** for throughput scripts; **`[bench-all]`** adds HEALPix / geometry comparators (**healpy**, **hpgeom**, **healsparse**, **astropy-healpix**, **spherical-geometry**, **mocpy**) for parity benches—same role as Pixi `bench-all`; **`[sht-bench]`** is the optional **s2fft** HEALPix SHT peer (not an oracle; JAX stays out of the default `test` env).

### Python interpreter (agents and humans)

Do **not** guess `python3` or bare `python` on PATH for project work.

1. **Pixi (required):** `pixi run <task>`, `pixi run -e test -- pytest …`, `pixi run -e default -- python -m …`. The repo’s Python and deps come from the Pixi environment.
2. **Last resort:** explicit `python3.12 -m …` only if Pixi is unavailable—avoid for automated agents.

### Pixi workspaces

- **`[package.*]`** (pixi-build): build/host/run deps for compiling the wheel; keep aligned with `pyproject.toml` runtime.
- **Workspace `[dependencies]`** (`default` environment): lean **dev tooling** (pytest, ruff, cmake, pytorch, numpy, …). Do **not** treat this list as “libraries torchsky wraps at runtime.”
- **Feature-gated environments** (see `pixi.toml`):
  - **`test`** — `pytest` suite: **astropy**, **torchfits**, **pyarrow**,
    **polars**, **pandas**, **scipy** (via solve), **requests**, etc., for WCS
    fixtures, integration inputs, table batches, and interop tests.
  - **`bench` / `bench-all`** — throughput comparators, plotting, **pandas**, **reproject**, optional map tools.
  - **`sphere-bench`** — HEALPix / sparse **ecosystem** comparators (**healpy**, **hpgeom**, **healsparse**, **astropy-healpix**, **spherical-geometry**, **mocpy**, …) and upstream-derived benchmark gates.
  - **`sht-bench`** — same comparators as `sphere-bench` plus optional **s2fft** via `pixi run sht-bench-bootstrap` (do not add JAX to `sphere-bench` by default).
  - **`map-compat`**, **`bench-legacy`** — external toolchains and legacy Python stacks as needed.
- **`pixi run test`** runs **`pytest -n auto --ignore-glob=tests/test_map_*.py`** in the **`test`** environment (same shape as the GitHub `build-test` job; map tests live in the separate CI map matrix).
- **`pixi run preflight-push`** — **ruff** + **`python -m compileall`** using the **default** env interpreter (`pixi run -e default -- python …`), not an arbitrary system `python`. This is the git **pre-push** hook.
- **`pixi run ci-local`** — **preflight-push** + **typecheck** + **test** (optional local copy of GitHub `build-test`; not a push blocker).
- **`pixi run ci-local-heavy`** — optional import smoke for sphere comparators in the **`bench-all`** env.

### I/O ownership and optional ecosystem packages

Use `torchfits` at file boundaries and keep it behind optional adapters or
application/CLI layers. Core geometry, map, scene, and field code should accept
in-memory tensors and metadata rather than opening FITS files.

Never import `astropy.io.fits` or Python `fitsio`. Read and write FITS with
`torchfits` (or the thin `torchsky.table` / `torchsky.io.fits` facades). Header
cards are plain dicts. Astropy WCS is a parity oracle and takes those dicts;
it is not a FITS I/O layer. Do not add new `astropy.io.fits` or `fitsio` imports
in tests, benchmarks, scripts, or examples. Leftover map/SMACS MEF fixture
writers still need migrating off `astropy.io.fits`; do not copy them.

Eat our own dog food: `src/torchsky` and examples compute with torchsky (and
`torchfits` at file boundaries). Do not import `astropy.wcs`, `healpy`,
`hpgeom`, `healsparse`, `mocpy`, or `spherical_geometry` as production backends.
Those packages belong in **tests** (oracles) and **benchmarks** (baselines).
`to_exact_region` defaults to the native backend; pass
`backend="spherical-geometry"` only in oracle tests. Optional extras
(`astropy_interop`, `gwcs_live`, JPL/TLE) stay opt-in and are not imported by
default `torchsky.wcs`.

Use **astropy** (except `astropy.io.fits`; never Python `fitsio`), **healpy**, **hpgeom**, **healsparse**, **astropy-healpix**, **spherical_geometry**, **pandas**, etc., only in:

- **`tests/`** (fixtures, parity, regression),
- **`benchmarks/`**,
- **`scripts/`** that bootstrap or print comparator versions,
- **CI workflows** that install extras for those jobs,
- **docs** that describe compatibility.

They **must not** become required imports for core **`src/torchsky`** behavior
on the default install path. FITS parser, HDU, checksum, compression,
file-cache, and FITS-table implementations belong in `torchfits`.

### pip-from-source / wheels (non-Pixi)

`docs/install.md` may describe **pip + cmake** builds for wheel installs and packagers. **Contributor CI and day-to-day workflows use Pixi** (`.github/workflows/ci.yml`, weeklies, `pixi run ci-local`). Do not treat raw **`pip install -e ".[test]"`** as the default path for agents when Pixi is available.

## Core Principles

### 1. No heuristics where exact or well-founded methods are available

- Do not introduce ad hoc thresholds, fallback rules, or data-dependent branching unless there is a clear numerical, algorithmic, or benchmark-backed reason.
- If a heuristic is temporarily necessary, isolate it, document why it exists, and add a benchmark or regression test that justifies it.
- Prefer mathematically grounded algorithms, explicit error bounds, or battle-tested upstream semantics.

### 2. Build forward-looking primitives; use established libraries as references, comparators, or compatibility targets

- Core primitives should be designed for the mission goals above: scalability, differentiability, cross-device execution, and high performance on modern workloads.
- Do not shape the internal architecture around legacy astronomy package boundaries if that blocks better primitives.
- Use established astronomy libraries primarily for:
  - semantic comparison
  - regression checks
  - migration and compatibility layers
  - upstream fixtures and validation data
- Use battle-tested non-astronomy libraries where they are the best fit for numerical kernels, systems concerns, storage, or compiler/runtime leverage.
- Compatibility does not require matching another project's internal primitives; it requires matching trusted behavior at a useful API layer when compatibility is an explicit goal.

### 3. Optimize for scalability first

- Assume catalogs, maps, cubes, and tables may be large enough that naive `O(N*M)` behavior is unacceptable.
- Favor structured indexing, chunking, streaming, sparse representations, and algorithmic pruning before native-kernel work.
- Native C++ kernels are justified after algorithmic structure is sound and the remaining bottleneck is real.

### 4. Preserve differentiability and tensor-native execution

- Keep user-facing compute paths in PyTorch unless there is a strong reason not to.
- Avoid unnecessary host transfers, NumPy-only detours, or CPU-only assumptions in public tensor paths.
- When adding fast paths, keep a correct tensor-native fallback unless the project explicitly decides otherwise.

### 5. Cross-device behavior matters

- CPU is the baseline.
- CUDA and MPS support should be preserved where the surrounding module claims it.
- Device-specific optimizations must not silently change semantics.

## API and Compatibility Policy

- Public APIs should stay thin and stable unless there is a strong reason to widen them.
- Prefer **torchsky-native** geometry and modelling primitives in
  `src/torchsky`; use `torchfits` for FITS interchange and established
  astronomy stacks (**healpy**, **astropy-healpix**, **spherical-geometry**,
  **Astropy**, **healsparse**, …) as references, parity baselines, and migration
  targets—see **Dependencies, installation, and ecosystem packages** above and
  `docs/compatibility.md`.
- Compatibility layers (e.g. healpy-shaped HEALPix under `torchsky.sphere.compat`) should sit **on top of** torchsky kernels (`torchsky.wcs.healpix`, sphere C++/torch paths), not the other way around.
- Compatibility work should target behavior users depend on, test fixtures, and operational workflows.
- Avoid exposing low-level internal kernels as public API unless repeated use or composability clearly justifies them.

## Table ownership

- FITS table reads, writes, scans, mutation, Arrow conversion, and predicate
  pushdown belong to `torchfits.table`.
- Do not import `astropy.io.fits`, do not import Python `fitsio`, and do not call Astropy `Table.write(..., format="fits")`.
- `torchsky.catalogs` may consume tensor/Arrow batches and own sky-domain
  partitioning, crossmatch, spherical indexing, and catalog modelling.
- Do not restore `torchsky.table` as a second implementation. Any temporary
  compatibility alias must be thin, documented, tested, and scheduled for
  removal.

## Performance Policy

- Every performance-sensitive change should answer:
  - what algorithmic bottleneck is being removed
  - why this scales better at large `N`
  - what the benchmark evidence is
- Prefer:
  - algorithmic improvements
  - structured indices
  - data layout improvements
  - chunking/streaming
  - selective native kernels for proven hotspots
- Benchmark both small/medium and large workloads when the optimization could shift crossover points.
- Do not force a native fast path by default if it regresses realistic medium-size workloads.

## Testing Policy

All substantive work should be validated on both simulated and real data when possible.

### Simulated / synthetic data

Use synthetic data to:

- stress corner cases
- isolate numerical behavior
- generate large-scale workloads cheaply
- test randomized parity against brute force or known baselines

### Real data or upstream fixtures

Use packaged fixtures, upstream test data, or representative real astronomy data to:

- confirm semantic compatibility
- catch edge cases synthetic generators miss
- validate interoperability claims

Examples already present in this repo include upstream-derived benchmark gates,
parity suites, and FITS-based integration tests. FITS integration fixtures may
be loaded through `torchfits`; I/O correctness and throughput claims must be
tested in the `torchfits` repository.

**Perf smoke:** `tests/test_perf_gate.py` should cover torchsky-owned hot paths
such as WCS, HEALPix, indexing, resampling, and operators. FITS read/write
performance gates belong to `torchfits`.

### For new features or optimizations

Try to include:

- correctness tests
- parity tests against a trusted baseline when available
- benchmark evidence
- at least one realistic-data or upstream-fixture check for user-facing behavior

## Implementation Preferences

- Prefer `torch` operations over Python loops in user-facing hot paths.
- Prefer `rg` for search.
- Use `apply_patch` for manual file edits.
- Keep comments sparse and high signal.
- Do not add broad new dependencies casually.
- Do not replace exact semantics with approximate shortcuts unless the tradeoff is explicit and tested.
- Do not reduce torchsky to a wrapper around `scamp`, `sextractor`, `swarp`, `psfex`, LSST Stack components, `drizzle`, `astropy`, `fitsio`, `healpy`, or similar packages.
- Do not add FITS parsing, HDU/table mutation, checksum, compression,
  or file-cache code to `torchsky`; contribute it to `torchfits`.

## Work prioritization

Prioritize code by expected scientific and operational impact, then correctness
risk, then performance at survey scale:

1. Remove duplicated ownership and make invalid states fail closed.
2. Fix correctness in shared geometry and observation operators, with trusted
   parity fixtures and exact adjoint/round-trip tests.
3. Replace asymptotically poor algorithms with indexing, tiling, chunking,
   sparsity, or matrix-free execution.
4. Optimize measured hot kernels across realistic small, medium, and massive
   workloads.
5. Add convenience surfaces only when they compose the primitives above.

Avoid isolated micro-optimizations, cosmetic refactors, and compatibility
breadth that do not materially advance correctness, scale, or measured
throughput.

## Native Code Guidance

- Add C++ only after confirming the Python/Torch algorithm is already the right one.
- Keep native kernels focused on real hotspots.
- Maintain correctness parity tests against the non-native path.
- Expose native code through narrow bindings; keep orchestration and policy in Python unless there is a compelling reason otherwise.

## Benchmarks and Evidence

Relevant benchmark surfaces live in:

- `benchmarks/`
- `docs/benchmarks.md`
- upstream-derived benchmark and parity gates documented in `docs/benchmarks.md`

For sphere and catalog work specifically:

- maintain explicit parity against trusted upstream behavior where available
- report throughput and candidate-reduction effects, not just wall time
- measure both default paths and gated fast paths when a native kernel is optional

## Change Checklist

Before finishing a non-trivial change, check:

1. Is the algorithm scalable enough for large datasets?
2. Did we preserve or improve differentiability and device behavior?
3. Did we avoid unnecessary heuristics?
4. Did we compare against a trusted library or upstream fixture where possible?
5. Did we test on both simulated and realistic data or fixtures?
6. Did we add benchmark evidence if this change claims performance value?
7. Did we avoid widening the public API unnecessarily?
8. If adding dependencies: are they confined to **`pyproject.toml` extras** or **Pixi features** (`test`, `bench`, …), not core runtime / default workspace, unless there is an explicit design decision and docs update?

## Documentation Expectations

- Update user-facing docs if public behavior changes.
- Update benchmark docs when benchmark surfaces or benchmark policy change.
- Keep repository guidance in this file aligned with actual practice.

## Agent PR hygiene and lint gates

- Do **not** open PRs that only remove one unused import, reformat a single file, or repeat the same micro-opt across many PRs. Batch mechanical cleanup into **one** commit/PR, or fold it into substantive work.
- Performance-related PRs must satisfy **Performance Policy** above (bottleneck, scaling story, benchmark evidence).

### Pre-push gate

The git **pre-push** hook (`pixi run setup-hooks`) runs **`pixi run preflight-push`** only (ruff + compileall). It must not run pytest. GitHub **`build-test`** is the full test gate.

Automated agents: do **not** run `pixi run ci-local` or `pixi run test` as a push blocker. Before push:

1. `pixi run preflight-push` (the hook runs this again)
2. Targeted tests for files you changed: `pixi run -e test -- pytest tests/test_foo.py`
3. `pixi run typecheck` when Python types or public signatures changed (pre-commit also runs pyright)

`pixi run ci-local` remains the optional local copy of GitHub `build-test` (preflight + typecheck + pytest `-n auto`, map tests excluded). Use it before a release, after a risky cross-cutting change, or when you want the suite locally. Do not put the test suite back in the pre-push hook.

- **Tree-wide `ruff format --check`** is not a CI gate yet; use **`pixi run lint`** when you want Ruff to apply formatting locally.

Optional: **`pre-commit install`** to mirror `.pre-commit-config.yaml` locally. **`pixi run verify`** runs **`scripts/verify_setup.py`** (torch + `torchsky.cpp` import smoke).

## Env hygiene

- Prefer `pixi run` / `pixi run python` over bare `python3` when Pixi exists.
- Never `pip install --user` or install into `~/.local` / `$HOME/.local` (esp. CANFAR `/arc/home`).
- Headless/batch: `export PYTHONNOUSERSITE=1` and `unset PYTHONPATH`.
- On CANFAR: read skill `canfar-lab-workflow` (mounts, quotas, resources, headless, ports).
