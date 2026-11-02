# HEALPix
<!-- TODO(0.5.0 docs): intro -->


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

With `nest=True`, `query_disc`, `query_polygon`, and `query_strip` descend the
NESTED quadtree: a coarse cell wholly inside the shape contributes one contiguous
pixel range and only cells crossing the boundary are refined, so the work is
`O(nside)` cells plus the output, never a scan of the interior. The membership
rule for boundary pixels is unchanged (pixel centres; `inclusive` and `fact` as
in healpy), and results equal the RING query mapped to NESTED. The healpy-shaped
`compat.query_circle(..., nest=True, return_pixel_ranges=True)` returns the ranges
directly (at `nside=4096`, radius 179 degrees: 215 ranges in about 2 ms instead
of enumerating 2e8 pixels).

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

## healpy compatibility

`torchsky.sphere.compat` exposes healpy- and hpgeom-shaped entry points
(`ang2pix`, `query_disc`, `get_all_neighbours`, `ud_grade`, `map2alm`,
`anafast`, `synfast`, `smoothing`, `pixwin`, and more) on top of the native
kernels, with tensor inputs and outputs. Keywords that have no effect are rejected
instead of silently accepted. These labels describe intended semantics, not an
exhaustive parity certificate; measured agreement with healpy is summarized in
[Parity evidence](../performance/parity.md), and the porting table is in
[Coming from Astropy, healpy, and reproject](../introduction/coming-from.md).

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
