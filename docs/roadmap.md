# Roadmap

`torchsky` is the torch-native compute layer for sky data. FITS and table I/O
belong to [`torchfits`](https://github.com/astroai/torchfits); exhaustive parity,
large-data campaigns, and scorecards belong to
[`torchsky-harness`](https://github.com/astroai/torchsky-harness). This page
tracks the package's own direction, not experiment queues.

---

## Current focus: 0.4.0

The 0.4.0 line is the first public-quality release of the package. It is not on
PyPI yet; it ships when the goals below are met.

| Goal | What "done" means |
|---|---|
| Correct | Full package suite green; analytic and reference-comparison gates in the harness pass on the declared cases |
| Fast | Measured against reference implementations on the same machine; hot paths stay tensor-native, with no NumPy/Astropy round trips inside loops |
| Lean and clean API | One name per concept; no legacy or duplicate surfaces; the public manifest and docs match the code |
| Clean structure | Cohesive modules with bounded file sizes; implementations live where the API says they do |
| Documented | Every public namespace has a guide, runnable examples, and a built documentation site |
| Harnessed | Correctness, parity, and performance evidence recorded with provenance in `torchsky-harness` |

---

## Package surfaces

| Area | Ownership | Guide |
|---|---|---|
| WCS and coordinates | Tensor transforms, coordinate frames, distortion, spectral and time axes | [WCS](wcs.md), [coordinates](coordinates.md) |
| Sphere and HEALPix | Spherical predicates, coverage maps, sparse maps, harmonic primitives | [Sphere](sphere.md) |
| Catalogs | Sky-domain indexing, crossmatch, cone search, Arrow/tensor batches | [Catalogs](catalogs.md) |
| Map and scenes | Product planning and rendering, coadds, observation operators, bounded-memory execution | [Map](map.md), [scene](scene.md) |
| Continuous fields | Differentiable field models and calibration-fit operators | [Field](field.md), [encoders](encoders.md) |
| Spectral data | Spectral axes, cubes, transforms, and photometric utilities | [Spectral data](spectral.md) |
| Runtime and interchange | `torchfits` boundary, transforms, datasets, visualization adapters | [I/O](io.md), [ML/runtime](ml.md) |

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

## Later

- Bounded-memory spectral and HEALPix products at survey scale.
- Broader observation-operator coverage (detector response, calibration,
  likelihood) promoted out of the experimental tier as gates land.
- GPU/CUDA validation on a pinned runner.
- Additional guides, examples, and gallery material for real survey workflows.
