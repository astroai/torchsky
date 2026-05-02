# `torchsky.geom`

`torchsky.geom` is the geometry-oriented import surface for spherical regions,
predicates, and related helpers. It overlaps intentionally with
`torchsky.sphere`; use the import style that best matches the subsystem you are
building.

## What This Namespace Is For

Use `geom` when the code reads as geometry-first: regions, containment, overlap,
or predicates. Use [sphere.md](sphere.md) when working with HEALPix maps, MOC,
sparse maps, or healpy-shaped compatibility.

## Core Concepts

- Geometry primitives are torchsky-native where practical.
- Exact or well-founded methods are preferred over heuristics.
- Established astronomy libraries such as `spherical-geometry` are used as
  validation baselines and optional comparators, not as core runtime
  dependencies.

## API Surface

Common concepts include:

- `SphericalPolygon`
- `SphericalCap`
- `SphericalMultiPolygon`
- `SphericalBooleanRegion`
- `SphericalBox`
- `SphericalEllipse`
- `PixelizedRegion`
- `MOC`
- `HealSparseMap`
- `spherical_polygon_contains(...)`
- `convex_polygon_contains(...)`
- `query_polygon_general(...)`
- `query_ellipse(...)`
- distance, sampling, and longitude-wrapping helpers re-exported from
  `torchsky.sphere`

See [sphere.md](sphere.md) for the full geometry and HEALPix reference.

## Validation

Geometry behavior is covered by sphere and map tests plus benchmarks such as
`benchmarks/bench_sphere_polygons.py`,
`benchmarks/bench_sphere_geom_exact_contains.py`, and
`benchmarks/bench_sphere_geometry.py`.
