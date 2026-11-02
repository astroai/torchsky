# Roadmap

`torchsky` is the torch-native compute layer for sky data. FITS and table I/O
belong to [`torchfits`](https://github.com/astroai/torchfits); exhaustive parity,
large-data campaigns, and scorecards belong to
[`torchsky-harness`](https://github.com/astroai/torchsky-harness). This page
tracks the package's own direction, not experiment queues.

---

## Current focus: 0.5

0.5 is the first public release of the package under the name `torchsky`. The
goals below define what it ships.

| Goal | What "done" means |
|---|---|
| Correct | Full package suite green; analytic and reference-comparison gates in the harness pass on the declared cases |
| Fast | Measured against reference implementations on the same machine; hot paths stay tensor-native, with no NumPy/Astropy round trips inside loops |
| Lean and clean API | One name per concept; no duplicate surfaces; the public manifest and docs match the code |
| Clean structure | Implementations live where the API says they do; the largest modules (`wcs/core.py`, `sphere/spectral.py`, `map/image_plane_coadd.py`, ...) are split into cohesive units |
| Documented | Every public namespace has a guide, runnable examples, and a built documentation site |
| Harnessed | Correctness, parity, and performance evidence recorded with provenance in `torchsky-harness` |

---

## Package surfaces

| Area | Ownership | Guide |
|---|---|---|
| WCS and coordinates | Tensor transforms, coordinate frames, distortion, spectral and time axes | [WCS](wcs.md), [frames and motion](coordinates.md) |
| Sphere and HEALPix | Spherical predicates, coverage maps, sparse maps, harmonic primitives | [Sphere](sphere.md) |
| Catalogs | Sky-domain indexing, crossmatch, cone search, Arrow/tensor batches | [Catalogs](catalogs.md) |
| Map and scenes | Product planning and rendering, coadds, observation operators, bounded-memory execution | [Map](map.md), [scene](scene.md) |
| Continuous fields | Differentiable field models and calibration-fit operators | [Field](field.md), [encoders](encoders.md) |
| Spectral data | Spectral axes, cubes, transforms, and photometric utilities | [Spectra](spectral.md) |
| Runtime and interchange | `torchfits` boundary, transforms, visualization adapters | [Concepts](concepts.md), [Encoders and ML](encoders.md) |

---

## Engineering priorities

1. Fix correctness defects in shared geometry, coordinate transforms, and
   observation operators before adding surface.
2. Prefer indexing, tiling, chunking, streaming, and sparse representations for
   survey-scale data.
3. Preserve differentiability, device behavior, and exact mathematical contracts
   in bounded tests.
4. Avoid widening the public API without repeated use and composability evidence.
5. Support performance claims only with reproducible harness measurements.

---

## Upstream collaboration: torch-harmonics

TorchSky uses `torch-harmonics` as an optional regular-grid backend and upstream
partner, not as a required dependency or a replacement sky index. Generic
spherical numerical primitives should be contributed upstream when they benefit
both astronomy and geospatial applications.

Planned upstream sequence:

1. Propose a shared `IsoLatitudeGrid` contract with explicit synthesis,
   analysis, adjoint, quadrature, dtype, device, and spin semantics.
2. Add exact HEALPix RING geometry metadata needed by numerical operators:
   ring offsets, lengths, colatitudes, longitude origins, and quadrature weights.
3. Add scalar HEALPix synthesis, analysis, and exact adjoint operators using
   ragged ring FFTs and blocked Legendre recurrences, with CUDA and distributed
   tests.
4. Extend the same contract to general spin-weighted transforms, including
   spin-2 E/B conventions and distributed execution.
5. Consider generic scalar masked-spectrum coupling and bandpower operators
   only after transform semantics are stable.

TorchSky continues to own astronomy-specific semantics: FITS/WCS integration,
HEALPix indexing and queries, MOC and multi-order products, survey provenance,
beam/pixel conventions, pseudo-`C_ell` workspaces, pure E/B policy, and
mapmaking. The existing `TorchHarmonicsBackend` remains the compatibility
boundary while focused upstream PRs are reviewed and released.

---

## Later

- Bounded-memory spectral and HEALPix products at survey scale.
- Broader observation-operator coverage (detector response, calibration,
  likelihood) promoted out of the experimental tier as gates land.
- GPU/CUDA validation on a pinned runner.
- Additional guides, examples, and gallery material for real survey workflows.

---

## Ideas under consideration

- Differentiable astrometric solver. Today, accurate relative astrometry for
  multi-CCD cameras (for example CFHT MegaCam, where the pipeline WCS is good to
  about 0.5-1 arcsec) comes from an external tool such as SCAMP or astrometry.net
  that writes TPV headers, which `torchsky.wcs` consumes natively. A native
  solver is a large effort and is not scheduled.

## Open items carried over from the retired planning notes

Release 0.5:

- Review `torchsky.viz` (FastAPI server, routers, frontend, cache, renderer) for
  obsolete endpoints and untested branches; decide whether it stays an optional
  extra or is split out.
- Clarify or rename the experimental field surfaces (`native_physical`,
  `joint_native`, `neural_inverse`, lensing/magnification operators) so public
  names state what they do; drop any without a demonstrated consumer.
- Keep a single canonical trusted-publishing workflow for `v*` tags, merging the
  useful verification steps of any overlapping workflow.

Calibrated-field work (post-0.5 unless promoted):

- Adaptive chromatic response, physical radiometric/flat calibration, and
  photon/charge/read additive-stage renderer integration.
- End-to-end observation lineage and nuisance propagation across field/coadd
  callers; query/UQ reports and calibration lifecycle across solvers.
- Broader CUDA/MPS device evidence; skipped device tests are not validation.

Engineering backlog:

- MPS footprint-bound issue on band-grid detectors.
- Conservative NLL derivative-order/device contract before widening its range.
- Keep observed curvature distinct from expected Fisher information, and
  calibration uncertainty distinct from fixed product scaling.
