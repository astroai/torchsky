# AGENTS.md

Operating guidance for coding agents and maintainers working in `torchsky`.
Human contributors start at [CONTRIBUTING.md](CONTRIBUTING.md).

## Workflow

- Follow the user's current scope. A request for a plan or review authorizes
  that deliverable; an implementation request authorizes the edits and relevant
  checks needed to finish it.
- Plan, execute, verify, improve. Proceed with routine, reversible work. Ask only
  when missing information materially changes the result or an action needs
  authorization.
- Preserve unrelated working-tree changes. Keep diffs focused; report what
  changed, the checks actually run, and any unresolved failures.
- Treat historical plans and `docs/roadmap.md` items as context; check current
  code before resuming one.

## Git

- Work on the fork's `main` and open one PR to `astroai/torchsky` `main`.
  No extra branches, stashes, or worktrees unless the user asks.
- Commit or push only when asked. Do not open PRs for single-import or
  single-file cosmetic changes; batch mechanical cleanup into one change.

## Mission

`torchsky` is a torch-native astronomy stack for:

- high-performance processing of large, multi-wavelength, multi-modal sky data
- continuous field-like modelling and differentiable pipelines where coherent
- CPU, CUDA, and MPS execution where practical
- compatibility layers with established astronomy libraries at a useful API
  level, without primitive-for-primitive mimicry

It owns sky coordinates, WCS, HEALPix and spherical geometry, maps, mapmaking,
observation operators, scenes, catalogs, and field models. Prefer designs that
stay viable at survey scale. Do not reduce it to a wrapper around astropy,
healpy, swarp, scamp, or similar packages.

## I/O boundary

FITS file and table I/O belongs to the sibling
[`torchfits`](https://github.com/astroai/torchfits) package, a required runtime
dependency. `torchsky` contains no FITS/CFITSIO code.

- Never import `astropy.io.fits` or Python `fitsio`. Read and write FITS with
  `torchfits`; header cards are plain dicts. Astropy WCS is a parity oracle that
  takes those dicts, not a FITS layer.
- Core geometry, map, scene, and field code accepts in-memory tensors and
  metadata; it does not open files. Use `torchfits` at application, CLI, and
  example boundaries.
- Do not restore `torchsky.table` or `torchsky.io.fits` as second
  implementations. FITS table, checksum, compression, and cache work goes to
  `torchfits`. `torchsky.catalogs` may consume tensor/Arrow batches and owns
  sky-domain partitioning, crossmatch, and spherical indexing.

## Dependencies

- Runtime (`[project].dependencies`): `torch`, `numpy`, `torchfits`, and compiled
  kernels only when benchmarks justify them. Python `>=3.12`; wheels target
  cp312 and cp313 (primary dev lane 3.13).
- Heavy or domain packages go in `[project.optional-dependencies]` or Pixi
  features (`test`, `docs`, `viz`, `map`, ...), never in the core list. Prefer
  PyArrow and `polars` for table interchange; `pandas` stays optional.
- `astropy` (except `astropy.io.fits`), `healpy`, `hpgeom`, `healsparse`,
  `astropy-healpix`, `spherical_geometry`, and `mocpy` are allowed only in
  `tests/` (oracles), docs, CI, and explicit opt-in interop adapters
  (`astropy_interop`, `gwcs_live`, JPL/TLE). They must not be required by
  `import torchsky` or serve as production backends.
- Benchmarks, exhaustive upstream parity, large/real-data validation, scorecards,
  and comparator environments live in
  [`torchsky-harness`](https://github.com/astroai/torchsky-harness). Never
  import harness code here. Do not add broad dependencies casually.

## Python and Pixi

Use Pixi for every Python invocation: `pixi run <task>`,
`pixi run -e test -- pytest ...`, `pixi run -e default -- python ...`. Do not
guess bare `python`/`python3`. Environments: `default`, `test`, `test-py312`,
`test-py313`, `docs`, `astrometry`, `cuda-bench`. Never `pip install --user`.
For headless runs set `PYTHONNOUSERSITE=1` and unset `PYTHONPATH`.

## Core principles

1. No heuristics where exact or well-founded methods exist. Isolate and justify
   any unavoidable threshold with a regression test or benchmark.
2. Build forward-looking primitives; use established libraries as references,
   comparators, and compatibility targets. Compatibility layers (for example
   `torchsky.sphere.compat`) sit on top of torchsky kernels such as
   `torchsky.sphere.healpix` and `torchsky._cpp`, not the other way round.
3. Scalability first: assume catalogs, maps, and cubes are large. Prefer
   indexing, chunking, streaming, and sparsity before native code.
4. Stay tensor-native and differentiable: no needless host transfers or
   NumPy-only detours in public paths. Keep a correct torch fallback next to any
   fast path.
5. CPU is the baseline; keep claimed CUDA/MPS support. Device optimizations must
   not change semantics.
6. Fail closed on invalid states; remove duplicated ownership.
7. Keep public APIs thin and stable. Do not expose internal kernels without a
   repeated, composable use.

## Native code

Add C++ (`src/torchsky/cpp_src`, nanobind via scikit-build-core, loaded through
`torchsky._cpp`) only after the torch algorithm is right and a hotspot is
measured. Keep bindings narrow, policy in Python, and parity tests against the
non-native path. Never link CFITSIO.

## Performance policy

Every performance change states the bottleneck removed, why it scales at large
`N`, and the benchmark evidence (small, medium, and large workloads). Do not
force a native path that regresses realistic medium sizes. Report throughput and
candidate-reduction effects, not wall time alone. `tests/test_perf_gate.py`
covers torchsky-owned hot paths.

## Testing and push policy

- Core tests are bounded API, regression, adjoint, gradient, and device
  contracts, using both synthetic data and packaged or upstream fixtures.
- Before a push run `pixi run preflight-push` (ruff + compileall; also the
  pre-push hook) and targeted tests: `pixi run -e test -- pytest tests/test_foo.py`.
  Run `pixi run typecheck` when types or public signatures change, and
  `pixi run api-manifest-check` when the public API changes.
- `pixi run test` is the main suite (excludes `tests/test_map_*.py`; map tests run
  separately: `KMP_DUPLICATE_LIB_OK=TRUE pixi run -e test -- pytest -n 4 -q tests/test_map_*.py`).
  `pixi run ci-local` mirrors GitHub `build-test`; run these only when asked or
  for a risky cross-cutting change. Never add pytest to the pre-push hook.
- Never auto-run benchmark suites or harness campaigns.
- Once required checks pass, stop. Record unrelated baseline failures without
  fixing them or claiming the gate passed.
- `pixi run lint` runs `ruff format` tree-wide; do not run it as part of an
  unrelated change. Use `ruff format` only on files you edit.

## Documentation

- Update user docs when public behavior changes; update `docs/changelog.md` for
  user-visible changes.
- Follow `docs/authoring.md`: guides are hand-written; `docs/reference/*.md` is
  generated from `docs/reference/_registry.toml` by `scripts/gen_reference.py`
  (never hand-edit); runnable snippets carry `<!-- docs: run -->`.
- Run `pixi run docs-contract` for docs work (strict build, links, anchors,
  generated pages, snippets, docs tests). Docs-only edits need no other pytest.
- Examples in `examples/` must use torchsky, be self-contained, and write
  outputs to a temporary directory, not the working directory.
- Releases follow `docs/release.md`.

## Change checklist

Scalable at large `N`? Differentiability and device behavior preserved? No new
heuristics? Compared with a trusted library or fixture? Tested on simulated and
realistic data? Benchmark evidence for performance claims? Public API
unchanged or documented? New dependencies confined to extras or Pixi features?
