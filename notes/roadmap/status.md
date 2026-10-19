# Torchsky package roadmap

`torchsky` is the torch-native compute layer for sky coordinates, spherical
geometry, maps, observation operators, scenes, and field models. FITS and table
I/O belong to `torchfits`; scientific benchmarks, exhaustive parity,
large/real-data validation, scorecards, and CANFAR orchestration belong to the
sibling [`torchsky-harness`](https://github.com/astroai/torchsky-harness).

## Implemented package surfaces

| Area | Core ownership | Guide |
|---|---|---|
| WCS and coordinates | Tensor transforms, coordinate frames, distortion, spectral and time axes | [WCS](../wcs.md), [coordinates](../coordinates.md) |
| Sphere and HEALPix | Spherical predicates, coverage maps, sparse maps, harmonic primitives | [Sphere](../sphere.md) |
| Catalogs | Sky-domain indexing, crossmatch, cone search, Arrow/tensor batches | [Catalogs](../catalogs.md) |
| Map and scenes | Product planning/rendering, coadds, observation operators, bounded-memory execution | [Map](../map.md), [scene](../scene.md) |
| Continuous fields | Experimental differentiable field models and calibration-fit operators | [Field](../field.md), [encoders](../encoders.md) |
| Spectral models | Spectral axes, cubes, transforms, and photometric utilities | [Spectral data](../spectral.md) |
| Runtime and interchange | Torchfits boundary, transforms, datasets, and visualization adapters | [I/O](../io.md), [ML/runtime](../ml.md) |

This table identifies ownership and entry points; it does not certify parity or
performance against other packages. Consult the harness for current measured
results and acceptance criteria.

## Durable engineering priorities

1. Fix correctness defects in shared geometry, coordinate transforms, and
   observation operators.
2. Prefer indexing, tiling, chunking, streaming, and sparse representations for
   survey-scale data.
3. Preserve differentiability, device behavior, and exact mathematical
   contracts in bounded tests.
4. Avoid widening public APIs without repeated use or composability evidence.
5. Support performance claims with reproducible harness measurements.

For detailed policy and verification tiers, see [`AGENTS.md`](../../AGENTS.md).
For the current package-specific queue, see [active.md](active.md); this file
does not assign a project-wide experiment queue.
