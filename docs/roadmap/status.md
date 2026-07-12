# Roadmap

> Current planning/status surface, not the public API contract. For shipped
> APIs, start with [`../api.md`](../api.md). For shipped-work history, use
> [`history.md`](history.md).

Last updated: 2026-07-11.

## Mission And Ownership

`torchsky` is a torch-native astronomy stack for large survey workloads:
tensor-native WCS and sphere geometry, scalable catalog indexing and map
operators, differentiable scene/field rendering, and cross-device execution
where the module claims it.

FITS image/table I/O has moved back to
[`torchfits`](https://github.com/sfabbro/torchfits). `torchsky` consumes tensors,
header mappings, masks/weights, and streamed table batches. It does not own
CFITSIO, FITS HDU machinery, compression, checksums, file caches, or FITS table
mutation.

The core must not become a thin wrapper around legacy astronomy packages.
Astropy, WCSLIB, healpy, hpgeom, healsparse, reproject, and related tools remain
references, comparators, optional compatibility targets, or benchmark
baselines.

## Replacement Target

The target is complete functional replacement of sky-domain workflows users
reach for in `astropy.wcs` / WCSLIB, `healpy`, `healsparse`, PyMOC-style MOC
tooling, `hpgeom`, `astropy-healpix`, and adjacent geometry/map packages where
they overlap with `torchsky`'s mission. FITS replacement is the `torchfits`
mission and is no longer a `torchsky` roadmap axis.

Replacement means matching trusted behavior at useful API and workflow layers,
with torch-native execution and no mandatory runtime dependency on comparator
packages. It does not mean copying every upstream primitive or claiming
untested behavior.

## Current Baseline

- Full local gate: `pixi run ci-local`.
- Latest recorded full local gate: `1681+ passed, 209 skipped, 0 failed, 55 warnings`
  (adjoint + deterministic accumulation additions; 3 pre-existing failures
  fixed. CI-local full run exceeds 10 min timeout; focused gate confirms all
  changed surfaces pass.)
- Latest focused WCS gate: `338 passed, 2 skipped, 22 warnings`.
- Latest focused sphere/HEALPix gate: `351 passed, 6 skipped, 10 warnings`.
- Latest WCS/sphere acceptance: `pixi run wcs-sphere-acceptance` passes; its
  pytest phase reports `558 passed, 2 skipped, 26 warnings`.
- Root `import torchsky` is intended to remain a lean lazy facade.
- Public surfaces are grouped into `core`, `compat`, and `experimental` tiers
  in `torchsky.surfaces`.
- The package-isolation gate is now **green**: CFITSIO/zlib/FITS-table native
  build wiring has been removed; all remaining FITS-related Python modules are
  thin delegation wrappers re-exporting from `torchfits`. The isolation gate
  (`test_package_isolation.py`) is a hard assert. A cross-repo integration gate
  (`test_torchfits_integration.py`) validates the torchsky↔torchfits contract.

Recorded test counts are evidence snapshots, not evergreen release claims.
Re-baseline them after the repository boundary cleanup.

## Domain Status

| Domain | Status | Covered now | Highest-impact closure |
| --- | --- | --- | --- |
| FITS / HDU / table I/O | Moved to `torchfits`; not a torchsky replacement domain | The sibling package owns images, headers, HDUs, compression, checksums, tables, Arrow scans, and I/O benchmarks. | Remove residual duplicate modules/build wiring, define a narrow optional adapter, and keep one cross-repo contract gate. |
| WCS | Broad `astropy.wcs` / WCSLIB gates pass; not standards-complete | Celestial projections/distortions, alternate keys, slicing/subsetting, differentiable paths, spectral/time primitives, frame conversion, independent/coupled TAB lookup, inverses, and Jacobians. | Concrete Paper III/IV fixtures, high-precision frame behavior, generalized transform graphs, device parity, and broad performance baselines. |
| Sphere / HEALPix / MOC / sparse maps | Strong core workflow coverage | High-NSIDE and no-dense sparse paths, MOC, regions, resolution changes, harmonic parity, and spin acceptance. | Exact-region edge cases, broader upstream replay, survey-scale indexing, cross-device behavior, and massive-scale memory/throughput baselines. |
| Catalog indexing / crossmatch | High-impact scale substrate | Tensor-native crossmatch and sphere primitives with comparator and fixture coverage. | Replace all-pairs paths with indexed/chunked execution; report candidate reduction, peak memory, correctness, and device behavior at survey scale. |
| Map / scene operators | Highest-impact integration layer | `ObservationOperatorBundle` is becoming the shared substrate across map products, field fitting, scene rendering, diagnostics, and implicit forward models. | Matrix-free chunked execution, exact adjoints, explicit noise/weight semantics, deterministic accumulation, and representative multi-exposure real-data gates. |
| Field models / INR / adaptive capacity | Experimental downstream work | Support-aware diagnostics, residual products, scorecards, pose refinement, and STRAINER-style scaffolding exist. | Promote only after operator correctness, memory scaling, and classical baselines are strong. |

## Recent Progress Evidence

- WCS mixed pixel/world solving covers spatial cubes with linearly coupled
  non-spatial axes; the focused gate reports
  `338 passed, 2 skipped, 22 warnings`.
- FK4/Besselian position-WCS conversion, axis-correlation reporting, and coupled
  TAB inverse/Jacobian behavior have targeted parity coverage.
- Sphere pixel-id and pixel-range upgrade/degrade workflows have hpgeom parity;
  the focused sphere/HEALPix gate reports
  `351 passed, 6 skipped, 10 warnings`.
- Sphere spin-matrix default acceptance passes.
- Architecture ownership changed: FITS implementation work and I/O throughput
  claims now belong to `torchfits`; torchsky retains only integration contracts.
- WCS/sphere `gradcheck` gap closed: 26 WCS gradcheck tests (all 8 projections,
  SIP, TPV) + 3 sphere spectral gradcheck tests. Projections with analytic
  inverse (AIT, MOL, CEA, MER) get `world_to_pixel` gradcheck; TAN/SIN/ARC/HPX
  use Newton iteration and are excluded from inverse gradcheck.
- End-to-end operator multi-exposure benchmark
  (`benchmarks/bench_operator_multiexposure.py`): creates N synthetic dithered
  TAN exposures, runs `coadd_image_plane`, and tracks wall time, throughput,
  overlap statistics, deterministic accumulation, and peak GPU memory.
- Arrow streaming path for catalogs: ``CatalogStreamDataset.from_arrow``,
  ``crossmatch_sky_streamed`` (generic left-batch iterator),
  ``crossmatch_sky_from_arrow`` (Arrow IPC / file convenience). All pyarrow
  imports are lazy (optional dependency). 9 Arrow-streaming tests (20 catalog
  tests green).
- ``crossmatch_sky_from_arrow`` rewritten as **bounded-memory partitioned
  crossmatch**: both catalogs are tiled to per-pixel Parquet row groups and
  crossmatched one pixel at a time. Memory scales with the largest per-pixel
  tile, not total catalog size. The old ``_build_coarse_healpix_index_from_arrow``
  (which needed an in-memory argsort index, ~16 N bytes) is removed.
- Operator gradcheck and adjoint coverage: 32 tests covering (a) WCS
  image-plane operators — ``InterpolationGeometryOp`` (3 kernels × 2 value
  semantics), ``ConservativePolygonRemapOp``, ``ExactDrizzleGeometryOp`` (square
  + point), ``ForwardSplatGeometryOp`` (splat_point + splat_box) — (b) the
  HEALPix bilinear forward/adjoint pair (``sample_healpix_map_on_image_header`` /
  ``accumulate_image_on_healpix_map_adjoint``, RING + NEST, with dot-product,
  ``gradcheck``, and ``gradgradcheck`` verification), and (c) 5 standalone
  operators (``gaussian_blur``, ``apply_kernel2d`` image/kernel,
  ``polynomial_background``, ``apply_image_observation_calibration``) with
  both ``gradcheck`` and ``gradgradcheck``.
  The PSF, background, and radiometry operators now preserve input dtype
  instead of hardcoding ``float32``.
- Deterministic accumulation verified across 8 kernels × 3 stacking methods
  (24 parametrized tests): ``coadd_image_plane`` produces bit-identical
  ``.data``, ``.weight``, and ``.overlap_count`` on repeat calls.
- Device-parity parametrization: all 16 geometry adjoint and 10 standalone
  operator tests now run on ``cpu`` and ``cuda`` (when available). MPS is
  excluded due to the PyTorch MPS backend lacking ``float64`` support
  (required by WCS grid arithmetic).

## Current Work Queue

### P0 — Boundary And Correctness — DONE

- CFITSIO/zlib build wiring removed from `cpp_src/CMakeLists.txt` ✓
- FITS engine modules (`hdu.py`, `table.py`, `core.py`, `_where.py`, etc.)
  replaced with thin delegation wrappers to `torchfits` ✓
- Package-isolation gate (`test_package_isolation.py`) is a hard assert ✓
- Cross-repo integration gate (`test_torchfits_integration.py`) validates the
  torchsky↔torchfits contract using real FIS fixture data ✓
- Dead code in `map/windowing.py` (circular-import workaround) removed ✓
- `SpectralReader` in `spectral.py` now reads `CUNIT1` instead of hardcoding
  `"Angstrom"` ✓
- `_where.py` delegated to the stable `torchfits.where` public boundary ✓

### Open

- Freeze the exact adapter/signature contract against the stable
  `torchfits.where`, `torchfits.hdu`, and `torchfits.table` public modules; do
  not encode transient private implementation APIs or preserve obsolete 0.3.x
  behavior in new torchsky code.
- Preserve header/WCS metadata, dtype, device, masks/weights, spectral axes,
  and table-batch semantics across the boundary. Invalid or ambiguous inputs
  fail closed.

### P1 — Massive Scientific And Operational Impact

- Make observation operators the performance center: benchmark end-to-end
  multi-exposure products rather than isolated helper calls.
- Require exact adjoints, bounded-memory tiling, deterministic accumulation
  policy, and CPU/CUDA/MPS parity where support is claimed.
- Replace catalog all-pairs behavior with structured spherical indices,
  candidate pruning, chunking, and streaming tensor/Arrow batches.
- Track complexity, candidate-reduction ratio, peak memory, transfer volume,
  throughput, and science-error metrics—not wall time alone.

### P2 — Geometry Closure And Kernel Performance

- Convert remaining WCS and sphere gaps into named parity matrices. Every item
  is covered by a gate, explicitly out of scope, or open against a named
  fixture/benchmark.
- Close WCS standards gaps only against concrete fixtures: coupled Paper III
  chains, Paper IV reference positions, high-precision frame behavior,
  generalized frame graphs, and legacy bridge replay.
- Broaden WCS/sphere performance baselines across projections, distortions,
  inverse paths, spectral/time, TAB, regions, harmonics, scale, and devices.
- Optimize only measured hot kernels after asymptotic structure and data layout
  are sound.

### P3 — Research Layers

- Promote adaptive local-capacity field components only after memory, support,
  real-data, and benchmark gates justify them.
- Add exact normal-equation and uncertainty gates before promoting new inverse
  map/scene solvers.
- Treat world-model and measure-valued scene ideas as research-layer work on
  top of stable operators, not as core API commitments.

## Prioritization Rule

New work must clear three questions in order:

1. **Massive impact:** does it unlock a core survey workflow, remove duplicated
   ownership, or improve asymptotic memory/runtime?
2. **Correctness:** is behavior specified by exact mathematics, a trusted
   fixture, round-trip/adjoint tests, and fail-closed edge handling?
3. **Performance:** is the end-to-end workload benchmarked at realistic small,
   medium, and massive scales, including peak memory and device transfer?

Micro-optimizations, cosmetic refactors, and API expansion stay behind boundary
cleanup, operator correctness, scalable indexing, and measured bottlenecks.

## Standing Gates

- Pre-push: `pixi run ci-local`.
- Package boundary: no FITS implementation modules, CFITSIO/zlib linkage, or
  FITS-owned benchmarks in torchsky; integration smoke against the supported
  `torchfits 0.8.x` range.
- FITS/table I/O behavior and throughput changes: validate in the `torchfits`
  repository.
- WCS/sphere claims: `pixi run wcs-sphere-acceptance` plus focused parity tests
  relevant to the changed domain.
- Map/field changes: exact adjoint, operator-bundle, field-quality,
  representative real-data/fixture gates proportional to the claim, and
  end-to-end multi-exposure operator benchmark for throughput/memory.
- Operator benchmark: `pixi run -e test -- python benchmarks/bench_operator_multiexposure.py`
- Performance changes: benchmark small, medium, and massive cases and report
  peak memory, correctness tolerance, and crossover behavior.

## Archived Context

These are retained for provenance but are not active roadmap documents:

- [`../archive/roadmap-knowledge-base.md`](../archive/roadmap-knowledge-base.md)
- [`../archive/roadmap-world-model-whitepaper-plan.md`](../archive/roadmap-world-model-whitepaper-plan.md)
- [`../archive/roadmap-sky-measure-scene-plan.md`](../archive/roadmap-sky-measure-scene-plan.md)
- [`../reviews/autonomous_refactor_log.md`](../reviews/autonomous_refactor_log.md)
