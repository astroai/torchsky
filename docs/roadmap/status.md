# Roadmap

> Current planning/status surface, not the public API contract. For shipped
> APIs, start with [`../api.md`](../api.md). For shipped-work history, use
> [`history.md`](history.md).

Last updated: 2026-06-18.

## Mission

`torchsky` is a torch-native astronomy stack for large survey workloads:
CFITSIO-backed data access, tensor-native WCS and sphere geometry, scalable map
operators, differentiable field rendering, and cross-device execution where the
module claims it.

The core should not become a thin wrapper around legacy astronomy packages.
Astropy, WCSLIB, healpy, hpgeom, healsparse, reproject, and related tools are
references, comparators, optional compatibility targets, or benchmark baselines.

## Replacement Target

The target is complete functional replacement of the core workflows users reach
for in `fitsio`, `astropy.io.fits`, `astropy.wcs` / WCSLIB, `healpy`,
`healsparse`, PyMOC-style MOC tooling, `hpgeom`, `astropy-healpix`, and adjacent
geometry packages where they overlap with `torchsky`'s mission.

Replacement means matching trusted behavior at useful API and workflow layers,
with torch-native execution and no mandatory Python runtime dependency on those
packages. It does not mean copying every internal primitive one-for-one, and it
does not mean claiming untested upstream behavior as done.

## Current Baseline

- Full local gate: `pixi run ci-local`.
- Latest full local gate result: `1817 passed, 53 skipped, 77 warnings`.
- Latest focused FITS gate result: `123 passed, 2 warnings`.
- Latest focused WCS gate result: `336 passed, 2 skipped, 22 warnings`.
- Latest focused sphere/HEALPix gate result: `351 passed, 6 skipped, 10 warnings`.
- Latest WCS/sphere acceptance result: `pixi run wcs-sphere-acceptance` passes.
  The pytest phase reports `556 passed, 2 skipped, 26 warnings`; WCS quick
  benchmarks have only small-N negligible deficits; sphere quick benchmarks
  report no comparable deficits; spin matrix default acceptance passes.
- Root `import torchsky` is a lean lazy facade.
- Public surfaces are grouped into `core`, `compat`, and `experimental` tiers in
  `torchsky.surfaces`.

## Domain Status

| Domain | Replacement progress | What is covered now | Remaining closure |
| --- | --- | --- |
| FITS / HDU / table I/O | Strong replacement for current `fitsio` / `astropy.io.fits` gates | Native image/header/checksum/compression/table paths, generated FITS fixture replay, Astropy VLA fixture replay, radio-style table coverage, table mutation validation, cache policy, and `torchsky.io.fits` ownership are active. | Larger committed benchmark baselines, broader real archive fixture replay, and continued audit that Python `fitsio` / Astropy stay out of runtime code. |
| WCS | Broad `astropy.wcs` / WCSLIB replacement gates pass, not full standards-complete | Celestial projections/distortions, alternate keys, Astropy-style metadata/helpers, slicing/subsetting, differentiable paths, spectral/time primitives, FK4/FK5/Galactic frame conversion, independent and coupled TAB lookup/inverse/Jacobians, legacy bridge benchmark rows, and focused acceptance are active. | Full FITS Paper III/IV breadth, apparent-place/high-precision frame behavior, generalized transform graphs, broader Kapteyn/PyAST/Cornish replay, and wider performance baselines. |
| Sphere / HEALPix / MOC / sparse maps | Strong `healpy` / `healsparse` / MOC replacement gates pass for core workflows | `torchsky.sphere` owns HEALPix/MOC/sparse-map behavior; high-NSIDE, no-dense sparse, MOC, region, HealSparse-style metadata, pixel/range resolution changes, harmonic parity, and spin matrix acceptance are active. | Broader `hpgeom` / `astropy-healpix` / PyMOC replay, exact-geometry edge cases, catalog indexing at survey scale, device coverage, and larger benchmark baselines. |
| Map / scene operators | Downstream consumer of geometry/I/O replacement work | `ObservationOperatorBundle` is becoming the shared execution substrate across map products, field fitting, scene rendering, quality diagnostics, and implicit forward models. | Continue after FITS/WCS/sphere replacement gates remain green. |
| Field models / INR / adaptive capacity | Experimental downstream work | Support-aware diagnostics, residual products, scorecards, pose refinement, and STRAINER-style scaffolding exist. | Continue after geometry/I/O replacement closure is stable. |

## Recent Progress Evidence

- WCS: coupled non-spatial FITS `-TAB` WCS now supports local inverse Jacobians
  through `world_to_pixel_jacobian(...)`; the WCS focused gate now reports
  `329 passed, 2 skipped, 22 warnings`.
- WCS: plain TAN differentiable pixel-to-world now uses the robust vector
  projection path at the tangent point, keeping `pixel_to_world_jacobian(...)`
  finite at projection center.
- WCS: integer slicing of fixed coupled linear axes now folds the fixed-axis
  contribution into retained `CRVAL`s, while arbitrary coupled `dropaxis(...)`
  still fails closed because no fixed coordinate is supplied.
- WCS: differentiable inverse transforms now cover spatial cubes with
  linearly coupled spectral/time-style extra axes, solving the extra-axis block
  after the torch-native RA/Dec inverse; the focused WCS gate now reports
  `330 passed, 2 skipped, 22 warnings`.
- WCS: FK4/Besselian frame conversion is now torch-native for position WCS
  workflows, including E-term add/remove, Newcomb FK4 precession, and Murray
  FK4-no-E/FK5 rotation terms; Astropy parity covers FK4<->FK5 and FK4
  precession, and the focused WCS gate now reports
  `334 passed, 2 skipped, 22 warnings`.
- WCS: `axis_correlation_matrix` now reports non-spatial `CD` couplings instead
  of assuming extra cube axes are diagonal-only; the focused WCS gate now reports
  `335 passed, 2 skipped, 22 warnings`.
- WCS: mixed pixel/world solving now handles arbitrary non-spatial linear
  `CD` systems exactly when the known world equations determine the unknown
  pixel axes; the focused WCS gate now reports
  `336 passed, 2 skipped, 22 warnings`.
- Sphere: stale unconditional harmonic parity skips were removed; `bl2beam`,
  `beam2bl`, and `map2alm_lsq` healpy parity gates now run when healpy is
  installed; `tests/test_sphere_spectral.py` reports `63 passed, 2 warnings`.
- Sphere: hpgeom and astropy-healpix parity now directly cover HEALPix
  angle/pixel conversion, range expansion/upgrade, and circle query
  pixel-range expansion.
- Sphere: native NESTED HEALPix pixel-range workflows now support exact
  down-resolution through `degrade_pixel_ranges(...)`; the focused
  sphere/HEALPix gate now reports `344 passed, 6 skipped, 10 warnings`.
- Sphere: explicit HEALPix pixel-id sets now support exact
  `upgrade_pixels(...)` / `degrade_pixels(...)` in both NESTED and RING
  ordering, with hpgeom parity for child expansion and top-level
  `torchsky.sphere` exports; the focused sphere/HEALPix gate now reports
  `351 passed, 6 skipped, 10 warnings`.
- Sphere acceptance: `benchmarks/bench_spin_matrix.py` now resolves the moved
  upstream replay script and the default acceptance case set passes.
- FITS: the current focused FITS gate reports `123 passed, 2 warnings` with no
  runtime Python `fitsio` or Astropy dependency in the core I/O path.

## Current Work Queue

### Now

- Convert the remaining replacement gaps into explicit parity matrices rather
  than repeating generic TODOs: `fitsio` / `astropy.io.fits`, `astropy.wcs` /
  WCSLIB / Kapteyn / PyAST, and healpy / hpgeom / healsparse / PyMOC /
  astropy-healpix.
- Close WCS standards gaps only against concrete fixtures or benchmarks:
  coupled Paper III chains, Paper IV reference-position behavior,
  apparent-place/high-precision frame behavior, generalized frame graphs, and
  legacy bridge replay.
- Broaden sphere replacement gates around hpgeom-style query behavior,
  astropy-healpix parity, PyMOC-style exchange, exact-region edge cases, and
  survey-scale catalog indexing.
- Broaden FITS replacement gates with larger archive-like fixtures and committed
  table/image throughput baselines while keeping Python `fitsio` and Astropy out
  of runtime code.

### Next

- Use the parity matrices to remove stale roadmap bullets: an item is either
  covered by a named gate, explicitly out of scope, or still open with a named
  fixture/benchmark target.
- Expand catalog crossmatch and spherical indexing gates for survey-scale table
  plus sphere pipelines.
- Broaden WCS performance baselines beyond the current quick acceptance rows
  across projections, distortions, inverse paths, spectral/time, TAB, and device
  regimes.

### Later

- Promote `ObservationOperatorBundle` through remaining map-product and
  field-quality paths after the geometry/I/O replacement matrices stop moving.
- Add exact adjoint or normal-equation gates before promoting any new inverse
  map/scene operator.
- Promote adaptive local-capacity field components only after memory, support,
  real-data, and benchmark gates justify them.
- Treat world-model and measure-valued sky-scene ideas as research-layer work
  on top of the library substrate, not as core API commitments.

## Standing Gates

- Pre-push: `pixi run ci-local`.
- FITS/table changes: focused FITS/table tests plus `pixi run bench-fits` or
  `pixi run bench-fitstable` when scale or performance behavior changes.
- WCS/sphere replacement claims: `pixi run wcs-sphere-acceptance` plus focused
  parity tests relevant to the changed domain.
- Map/field changes: exact adjoint, operator-bundle, field quality, and
  representative real-data or fixture gates proportional to the claim.

## Archived Context

These are retained for provenance but are not active roadmap documents:

- [`../archive/roadmap-knowledge-base.md`](../archive/roadmap-knowledge-base.md)
- [`../archive/roadmap-world-model-whitepaper-plan.md`](../archive/roadmap-world-model-whitepaper-plan.md)
- [`../archive/roadmap-sky-measure-scene-plan.md`](../archive/roadmap-sky-measure-scene-plan.md)
- [`../reviews/autonomous_refactor_log.md`](../reviews/autonomous_refactor_log.md)
