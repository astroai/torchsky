# Ecosystem compatibility

Torchsky implements its own tensor-native coordinates, WCS, spherical geometry,
maps, scenes, and field models. Established astronomy packages are references,
parity oracles, or optional adapters; they are not production backends for the
default compute path. Compatibility targets are defined at user-facing behavior
where users depend on it, not by mirroring another package's internal design.

## Runtime boundaries

- **FITS and table I/O:** use `torchfits`. Torchsky consumes decoded tensors,
  plain header mappings, and Arrow or tensor batches; its compatibility facades
  do not implement a second FITS engine.
- **WCS:** `torchsky.wcs` owns coordinate transforms and consumes header
  mappings. Astropy WCS may be used as a test oracle, never for FITS I/O.
- **Sphere and HEALPix:** `torchsky.sphere` owns the geometry and harmonic
  compute paths. Healpy, hpgeom, healsparse, astropy-healpix, and
  spherical-geometry are optional references or adapters.
- **Mapmaking:** `torchsky.map` owns product planning and rendering. SWarp,
  Montage, drizzle, and reproject may be evaluation comparators, not runtime
  requirements of the core map path.
- **Tables and catalogs:** `torchfits.table` owns FITS table operations;
  `torchsky.catalogs` consumes batches and owns sky-domain indexing and matching.

Optional interoperability surfaces must remain opt-in and must not become
required imports for ordinary `torchsky` use. See
[AGENTS.md](https://github.com/astroai/torchsky/blob/main/AGENTS.md) for dependency and backend policy.

## Validation ownership

Core tests retain bounded API, regression, numerical, adjoint, gradient, and
device-contract checks. Exhaustive upstream replay, comparator matrices,
large-scale and real-data campaigns, and scientific scorecards are maintained
in the sibling
[`torchsky-harness`](https://github.com/astroai/torchsky-harness) repository.
FITS/HDU/table parity and throughput are owned by
[`torchfits`](https://github.com/astroai/torchfits).

Consult the harness's
[compatibility and benchmark methodology](https://github.com/astroai/torchsky-harness/blob/main/docs/benchmarks.md)
for current comparator requirements and evidence. A passing package test or
smoke run does not certify cross-library or scientific acceptance.

## Adding compatibility

Before adding an adapter, identify the concrete user-visible behavior it
supports and the owning runtime layer. Keep optional dependencies behind an
explicit adapter or extra, test the contract with small deterministic fixtures,
and route exhaustive parity or data-intensive comparison to the harness. Do not
introduce a legacy library as a hidden production backend.
