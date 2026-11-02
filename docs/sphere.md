# Sphere pixels and geometry

`torchsky.sphere` is the spherical-geometry layer: HEALPix pixelization, vector
and great-circle helpers, region queries, spherical polygons, and map sampling.
Everything is tensor code that runs on CPU, CUDA, and MPS, with selected C++
kernels for hotspots such as polygon geometry. Related topics have their own
pages:

- [MOC and sparse maps](moc-and-sparse.md) for partial-sky coverage;
- [Harmonic transforms](sht.md) for `map2alm`, power spectra, beams, and spin;
- [HEALPix maps](healpix-maps.md) for coadding images onto HEALPix.

Signatures live in the [`torchsky.sphere` reference](reference/sphere.md); this
page is about which tool to reach for and how the pieces fit.

## Pixelization

Angles follow the healpy convention unless `lonlat=True`, in which case inputs
and outputs are longitude and latitude in degrees. `nest=True` selects NESTED
ordering; RING is the default, as in healpy.

<!-- docs: run -->
```python
import torch
import torchsky.sphere as sphere

nside = 128
lon = torch.tensor([10.0, 120.0, 250.0], dtype=torch.float64)
lat = torch.tensor([5.0, -40.0, 70.0], dtype=torch.float64)

pix = sphere.ang2pix(nside, lon, lat, lonlat=True, nest=True)

# The same pixels from colatitude/longitude in radians, and from unit vectors.
theta, phi = torch.deg2rad(90.0 - lat), torch.deg2rad(lon)
assert torch.equal(sphere.ang2pix(nside, theta, phi, nest=True), pix)
x, y, z = sphere.lonlat_to_unit_xyz(lon, lat).unbind(-1)
assert torch.equal(sphere.vec2pix(nside, x, y, z, nest=True), pix)

# pix2ang returns pixel centres, so the round trip is within a pixel.
lon_c, lat_c = sphere.pix2ang(nside, pix, nest=True, lonlat=True)
assert (lon_c - lon).abs().max() < 1.0 and (lat_c - lat).abs().max() < 1.0

assert sphere.nside2npix(nside) == 12 * nside**2
assert sphere.neighbors(nside, pix, nest=True).shape == (3, 8)
```

`ring2nest`, `nest2ring`, `ud_grade`, and the interpolation-weight helpers
complete the pixel toolkit. Prefer NESTED for hierarchical work (MOC, `ud_grade`,
spatial locality on GPUs) and RING for harmonic transforms.

## Vectors and distances

| Need | Use |
|---|---|
| Degrees to unit vectors and back | `lonlat_to_unit_xyz`, `unit_xyz_to_lonlat` |
| Longitude wrapping | `wrap_longitude` |
| Angular distance | `great_circle_distance` (degrees), `pairwise_angular_distance` |
| Points along an arc | `slerp_lonlat` |

Distances use the numerically stable `atan2(|a x b|, a . b)` form, so tiny and
antipodal separations both stay accurate.

## Region queries

Pixel queries return the HEALPix pixels covered by a shape:

<!-- docs: run -->
```python
import math

import torch
import torchsky.sphere as sphere

nside = 64
centre = sphere.lonlat_to_unit_xyz(torch.tensor(10.0), torch.tensor(0.0)).to(torch.float64)
disc = sphere.query_disc(nside, centre, radius=math.radians(1.0), nest=True)
ellipse = sphere.query_ellipse(nside, 10.0, 0.0, 2.0, 1.0, pa_deg=30.0, nest=True)
square_lon = torch.tensor([10.0, 20.0, 20.0, 10.0])
square_lat = torch.tensor([0.0, 0.0, 10.0, 10.0])
polygon = sphere.query_polygon_general(nside, square_lon, square_lat, nest=True)

assert disc.numel() > 0 and ellipse.numel() > disc.numel()
assert polygon.numel() > 0
print(disc.numel(), ellipse.numel(), polygon.numel())
```

`query_disc` is exact by default; `inclusive=True` also returns pixels that only
touch the shape. `convex_polygon_contains` and `spherical_polygon_contains`
test points directly without pixelizing.

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

## Sampling maps

`sample_healpix_map` evaluates a map at arbitrary sky positions with `nearest`
or `bilinear` interpolation. `sample_multiband_healpix` and
`sample_multiwavelength_healpix` do the same for `[n_band, npix]` cubes, the
latter resampling along a wavelength axis. `fit_monopole_dipole` and
`remove_monopole_dipole` clean a map before analysis.

<!-- docs: run -->
```python
import torch
import torchsky.sphere as sphere

nside = 16
healpix_map = torch.arange(sphere.nside2npix(nside), dtype=torch.float64)
lon = torch.tensor([10.0, 120.0], dtype=torch.float64)
lat = torch.tensor([5.0, -40.0], dtype=torch.float64)

nearest = sphere.sample_healpix_map(healpix_map, nside, lon, lat, interpolation="nearest")
smooth = sphere.sample_healpix_map(healpix_map, nside, lon, lat, interpolation="bilinear")
assert nearest.shape == smooth.shape == (2,)
```

## Partial sky, harmonics, and maps

Use `MOC`, `SparseHealpixMap`, and `MultiOrderHealpixMap` rather than dense
arrays whenever coverage is partial; see [MOC and sparse maps](moc-and-sparse.md).
Harmonic analysis, power spectra, beams, and pseudo-C_ell estimation are in
[Harmonic transforms](sht.md). Both live in `torchsky.sphere`; harmonic
functions are also grouped under `torchsky.sphere.spectral`.

## healpy compatibility

`torchsky.sphere.compat` exposes healpy- and hpgeom-shaped entry points
(`ang2pix`, `query_disc`, `get_all_neighbours`, `ud_grade`, `map2alm`,
`anafast`, `synfast`, `smoothing`, `pixwin`, and more) on top of the native
kernels, with tensor inputs and outputs. Removed ignored keywords are rejected
instead of silently accepted. These labels describe intended semantics, not an
exhaustive parity certificate; measured agreement with healpy is summarized in
[Parity evidence](evaluation/parity.md), and the porting table is in
[Coming from Astropy, healpy, and reproject](coming-from.md).

## Device support

| Job | CPU | CUDA | MPS |
|---|---|---|---|
| WCS pixel↔world | yes | yes float64 | yes float32 |
| HEALPix geometry | yes | yes | yes, looser index gates |
| Sparse `get_values` / MOC contains | yes | yes if tensors stay on device | yes float32 |
| Polygons (approximate contains/intersect/`query_polygon`) | yes | yes | yes float32 |
| Polygons (`NativeApproximateSphericalRegion`) | yes | no (CPU kernel / spherical-geometry) | no |
| SHT (Ishioka/Wigner C++) | yes float64 | no (CPU kernel) | no (CPU kernel) |
| SHT (ATen ring FFT) | yes float64 | yes float64 | yes float32 |
| rotate_alm / alm2lonlat | yes float64 | yes float64 | yes float32 |
| Observed place | yes float64 | yes float64 | yes float32 |


Native-kernel limits (CPU-only C++ for the fused spin transform and approximate
polygon region kernels) never change results; they only change where the work
runs. Set `TORCHSKY_DISABLE_CPP=1` to force the pure-torch paths.

## See also

- [healpy](https://healpy.readthedocs.io/), the [HEALPix home page](https://healpix.sourceforge.io/),
  [hpgeom](https://github.com/LSSTDESC/hpgeom), and
  [astropy-healpix](https://astropy-healpix.readthedocs.io/) as references;
- [`torchsky.sphere` API reference](reference/sphere.md).
