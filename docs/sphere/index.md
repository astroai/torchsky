# Celestial sphere: overview
<!-- TODO(0.5.0 docs): intro -->


`torchsky.sphere` is the spherical-geometry layer: HEALPix pixelization, vector
and great-circle helpers, region queries, spherical polygons, and map sampling.
Everything is tensor code that runs on CPU, CUDA, and MPS, with selected C++
kernels for hotspots such as polygon geometry. Related topics have their own
pages:

- [MOC and sparse maps](moc.md) for partial-sky coverage;
- [Harmonic transforms](sht.md) for `map2alm`, power spectra, beams, and spin;
- [HEALPix maps](../maps/healpix-mapmaking.md) for coadding images onto HEALPix.

Signatures live in the [`torchsky.sphere` reference](../reference/sphere.md); this
page is about which tool to reach for and how the pieces fit.

## Spherical regions

`SphericalPolygon`, `SphericalCap`, and `SphericalMultiPolygon` are
differentiable-friendly region objects with containment, area, and pixel
queries. Boolean algebra is available on the polygons themselves and, with
controlled error, on `PixelizedRegion` sets at a fixed `nside`.

<!-- docs: run -->
```python
import torch
from torchsky.sphere import SphericalPolygon

a = SphericalPolygon(
    lon_deg=torch.tensor([10.0, 20.0, 20.0, 10.0]),
    lat_deg=torch.tensor([0.0, 0.0, 10.0, 10.0]),
)
b = SphericalPolygon(
    lon_deg=torch.tensor([15.0, 25.0, 25.0, 15.0]),
    lat_deg=torch.tensor([5.0, 5.0, 15.0, 15.0]),
)

assert bool(a.contains(lon_deg=15.0, lat_deg=5.0))
assert a.intersects(b)
union = a.union(b)
assert bool(union.contains(lon_deg=12.0, lat_deg=2.0))

ra, rb = a.pixelize(nside=64), b.pixelize(nside=64)
overlap = ra.intersection(rb)
assert 0.0 < overlap.area() < ra.area()
```

Areas are in steradians. Containment and area are validated against the
`spherical-geometry` package in tests; that package is an oracle, not a backend.
The default exact-region backend is native.

## Partial sky, harmonics, and maps

Use `MOC`, `SparseHealpixMap`, and `MultiOrderHealpixMap` rather than dense
arrays whenever coverage is partial; see [MOC and sparse maps](moc.md).
Harmonic analysis, power spectra, beams, and pseudo-C_ell estimation are in
[Harmonic transforms](sht.md). Both live in `torchsky.sphere`; harmonic
functions are also grouped under `torchsky.sphere.spectral`.

## See also

- [healpy](https://healpy.readthedocs.io/), the [HEALPix home page](https://healpix.sourceforge.io/),
  [hpgeom](https://github.com/LSSTDESC/hpgeom), and
  [astropy-healpix](https://astropy-healpix.readthedocs.io/) as references;
- [`torchsky.sphere` API reference](../reference/sphere.md).
