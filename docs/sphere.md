# Sphere & HEALPix

`torchsky.sphere` is a PyTorch-native library for spherical geometry and HEALPix
maps, with selected C++ kernels for Spherical Harmonic Transforms (SHT) and
polygon geometry on CPU. Harmonic transforms are available through
`torchsky.sphere.spectral`; the former `torchsky.sphere.harmonics` alias was removed
a compatibility alias.

## Table of Contents

- [Pixel & Coordinate Utilities](#pixel-coordinate-utilities)
- [Coordinate Conversions](#coordinate-conversions)
- [Geometric Queries](#geometric-queries)
- [Spherical Geometry](#spherical-geometry-classes)
- [Sampling](#sampling)
- [Sparse Maps](#sparse-maps)
- [MOC](#moc)
- [STMOC](#class-stmoc)
- [Harmonic Space (SHT)](#harmonic-space-sht)
- [Power Spectra](#power-spectra)
- [Beams & Windows](#beams-windows)
- [Execution and Tuning](#execution-and-tuning)
- [Compatibility](#compatibility)

---

## Pixel & Coordinate Utilities

### `ang2pix(nside, theta, phi, nest=False, lonlat=False)`

Convert angles to HEALPix pixel indices.

- `lonlat=False` (default): `theta` = co-latitude (rad), `phi` = longitude (rad).
- `lonlat=True`: `theta` = longitude (deg), `phi` = latitude (deg).

```python
import torch
import torchsky.sphere as sphere

theta_rad = torch.tensor([0.5])
phi_rad = torch.tensor([1.0])
lon_deg = torch.tensor([10.0])
lat_deg = torch.tensor([5.0])

pix = sphere.ang2pix(128, theta_rad, phi_rad)
pix = sphere.ang2pix(128, lon_deg, lat_deg, lonlat=True, nest=True)
```

### `pix2ang(nside, ipix, nest=False, lonlat=False)`

Inverse of `ang2pix`.

### `vec2pix(nside, x, y, z, nest=False)` / `pix2vec(nside, ipix, nest=False)`

Convert between Cartesian unit vectors and pixel indices.

### `nside2npix(nside)` / `npix2nside(npix)`

Convert between NSIDE and total pixel count.

### `query_disc(nside, vec, radius, *, inclusive=False, nest=False)`

Pixels within a spherical disc. `vec` is a 3-element unit vector, `radius` in radians.
Inclusive `fact==4` pads by the maximum pixel radius. Other `fact` values
supersample at `nside*fact` (power of two) and degrade to parent pixels.
CUDA `query_disc` copies a length-3 `vec` to the CPU kernel; that copy is not
the wall-time tax, so the on-device MPS path is not ported to CUDA.

---

## Coordinate Conversions

| Function | Description |
|---|---|
| `lonlat_to_unit_xyz(lon_deg, lat_deg)` | (lon, lat) degrees to Cartesian `[..., 3]` |
| `unit_xyz_to_lonlat(vectors)` | Cartesian to (lon, lat) degrees |
| `wrap_longitude(lon_deg, *, center_deg=180)` | Wrap to `[center-180, center+180)` |
| `great_circle_distance(lon1, lat1, lon2, lat2)` | Angular distance (degrees) |
| `pairwise_angular_distance(lon, lat)` | N x N distance matrix |
| `slerp_lonlat(lon1, lat1, lon2, lat2, t)` | Spherical linear interpolation |

---

## Geometric Queries

### `query_polygon_general(nside, lon_deg, lat_deg, *, nest=False)`

Pixels inside a simple (possibly non-convex) spherical polygon.

### `query_ellipse(nside, lon, lat, semi_major, semi_minor, pa_deg, nest=False)`

Pixels within a spherical ellipse. Angles in degrees.

### `convex_polygon_contains(lon_deg, lat_deg, poly_lon_deg, poly_lat_deg)`

Fast half-space point-in-convex-polygon test.

### `spherical_polygon_contains(lon_deg, lat_deg, poly_lon_deg, poly_lat_deg)`

Point-in-polygon for simple (possibly non-convex) spherical polygons.

---

## Spherical Geometry Classes

### `SphericalPolygon`

Simple spherical polygon with containment, area, boolean operations, and pixel queries.

```python
from torchsky.sphere import SphericalPolygon

poly = SphericalPolygon(lon_deg=torch.tensor([10., 20., 20., 10.]),
                        lat_deg=torch.tensor([0., 0., 10., 10.]))
poly.area()                          # steradians
poly.contains(lon_deg=15., lat_deg=5.)
poly.query_pixels(nside=64)
poly.intersects(other_poly)
poly.union(other_poly)
```

Methods: `area`, `signed_area`, `contains`, `query_pixels`, `pixelize`, `area_estimate`, `intersects`, `union`, `intersection`, `difference`.

Validated against [spherical-geometry](https://github.com/spacetelescope/spherical_geometry) for non-convex containment and area.

### `SphericalCap`

Spherical disc region with the same interface as `SphericalPolygon`.

### `SphericalMultiPolygon`

Union-of-polygons collection.

```python
from torchsky.sphere import SphericalMultiPolygon, SphericalPolygon

multi = SphericalMultiPolygon(polygons=(poly1, poly2))
area = multi.area()
inside = multi.contains(lon_deg, lat_deg)
pixels = multi.query_pixels(nside=64)
```

### Class `PixelizedRegion`

Set of HEALPix pixels supporting boolean operations.

```python
from torchsky.sphere import PixelizedRegion

r1 = poly1.pixelize(nside=64)
r2 = poly2.pixelize(nside=64)
union_r = r1.union(r2)
inter_r = r1.intersection(r2)
diff_r  = r1.difference(r2)
area_sr = r1.area()
```

---

## Native HEALPix Pixel API

`torchsky.sphere.healpix` owns the torch-native HEALPix pixel primitives:
`ang2pix`, `pix2ang`, `vec2pix`, `pix2vec`, neighbor queries, interpolation
weights, disc/box/polygon queries, ordering conversions, range compression, and
map reordering.

## Sampling

### `sample_healpix_map(values, nside, lon_deg, lat_deg, *, interpolation='bilinear')`

Sample a HEALPix map at arbitrary sky coordinates. `interpolation`: `'nearest'` or `'bilinear'`.

### `sample_multiband_healpix(cube, nside, lon_deg, lat_deg, ...)`

Sample a multi-band HEALPix cube `[n_band, npix]`.

### `sample_multiwavelength_healpix(cube, nside, lon_deg, lat_deg, *, source_wavelength, target_wavelength, ...)`

Sample and optionally resample along wavelength axis.

### `fit_monopole_dipole` / `remove_monopole_dipole`

Fit or subtract monopole + dipole from a HEALPix map.

---

## Sparse Maps

### `HealSparseMap` / `SparseMap`

Sparse HEALPix map storing values only for covered pixels. Compatible with [healsparse](https://healsparse.readthedocs.io/) semantics.

```python
from torchsky.sphere import HealSparseMap

sparse = HealSparseMap(nside=64, nest=True, pixels=pix_tensor, values=val_tensor)
sparse = HealSparseMap.from_dense(dense_map, nside=64, nest=True)

sparse.coverage_mask
sparse.to_dense()
sparse.interpolate(lon_deg=lon, lat_deg=lat, method='bilinear')
sparse.ud_grade(nside_out=32)
```

`HealSparseMap.from_pixels(...)` canonicalizes inputs to sorted unique pixels,
keeping the last value supplied for duplicates, so `get_values(...)` and
interpolation can use indexed lookup reliably. Maps with `nside_coverage` derive
coverage maps in the NESTED coverage hierarchy even when sparse pixels are stored
in RING ordering, and `ud_grade(...)` preserves the coverage hierarchy when the
output resolution allows it. With `nside_coverage` set, `get_values` can use a
fused CPU gather (fill-block layout). Float maps may also degrade through that
layout when coverage nside is unchanged. Bool/int maps keep the pixel-list path
because fill-block cells cannot tell valid `False`/`0` from an uncovered child.
These are implementation choices; comparative throughput measurements live in
the harness.

`HealSparseMap.from_moc(moc, nside=...)` realizes MOC coverage directly as sparse
nested pixels without materializing a full dense HEALPix mask. Maps with
`nside_coverage` and a 1-D value vector write HealSparse FITS (`EXTNAME=COV`
int64 coverage plus `EXTNAME=SPARSE` image, `PIXTYPE=HEALSPARSE`). Boolean maps
are stored as `uint8` with `BOOLMAP`. Pixel-list maps without coverage still use
a PIXEL/VALUE table. Sentinel values are not valid pixels on the image path.
`from_healsparse(...)` and `to_healsparse()` provide optional CPU-host
interchange when `healsparse` is installed; the production implementation does
not import or delegate computation to HealSparse.

### `WideBitMask` / `SkyMaskPipe`

Additional sparse map types for interop with healsparse workflows.

---

## MOC

### Class `MOC`

Multi-Order Coverage map backed by HEALPix `UNIQ` cells in NESTED ordering.

```python
from torchsky.sphere import MOC, nest2uniq

moc = MOC(nest2uniq(4, torch.tensor([0, 1, 2])))
inside = moc.contains(torch.tensor([0.0, 45.0]), torch.tensor([0.0, 60.0]))
area_sr = moc.area
ascii_repr = moc.to_ascii()
json_repr = moc.to_json()
```

**Construction helpers:**

| Constructor | Description |
|---|---|
| `MOC(uniq)` | Build from `UNIQ` cells |
| `MOC.empty()` / `MOC.full(max_order)` | Empty coverage; full sky (order-0 cells) |
| `MOC.from_lonlat(lon, lat, max_order)` | Coverage of catalog points, compacted to parent UNIQ |
| `MOC.from_circle(lon_deg, lat_deg, radius_deg, max_order=10)` | Circular cap; hierarchical (interior cells stay coarse) |
| `MOC.from_polygon(lon_deg, lat_deg, max_order=10)` | Simple polygon footprint, including non-convex polygons; hierarchical |
| `MOC.from_region(region, max_order=None)` | Pixelize a torchsky spherical region, boolean region, exact-region wrapper, or `PixelizedRegion` into NUNIQ coverage; `PixelizedRegion` inputs preserve their current resolution by default, while other regions default to order 10 |
| `MOC.from_sparse_map(smap)` | Coverage from `HealSparseMap` / `SparseMap` |
| `MOC.from_threshold(smap, threshold, op='>')` | Threshold sparse-map values |
| `MOC.from_ascii(text)` | IVOA ASCII roundtrip, including compact pixel ranges like `3/5-7` |
| `MOC.from_json(text)` | IVOA JSON roundtrip |
| `MOC.read_fits(path)` | IVOA FITS roundtrip |
| `MOC.from_mocpy(moc)` / `moc.to_mocpy()` | Optional host-side mocpy interchange through IVOA JSON |

**Key methods and properties:**

| Method | Description |
|---|---|
| `contains(lon_deg, lat_deg)` | Point-in-MOC predicate |
| `contains_rows(catalog)` | Same mask from a `Catalog` (or any object with `lon`/`lat`) |
| `contains_pixels(nside, pixels, nest=True)` | Pixel-in-MOC predicate for HEALPix-indexed catalogs or sparse maps |
| `overlap_fraction(uniq)` | Exact covered fraction for canonical UNIQ cells without leaf expansion |
| `intersects_uniq(uniq)` / `contains_uniq(uniq)` | Intersecting or fully covered UNIQ cells |
| `filter_catalog(lon_deg, lat_deg)` | Filter catalog coordinates by coverage; prefer `Catalog.take(moc.contains(...))` |
| `contains_moc(other)` | Subset check |
| `overlaps(other)` | Non-empty intersection predicate for exposure footprints |
| `union(other)` / `intersection(other)` / `difference(other)` | Boolean operations on sorted NEST ranges; also `\|`, `&`, `-` |
| `degrade(max_order)` | Drop cells finer than `max_order` onto nested parents |
| `flatten(nside)` | Dense boolean mask at fixed resolution |
| `to_pixelized_region(nside, nest=True)` | Realize coverage as a `PixelizedRegion`, with optional RING conversion |
| `to_sparse_map(nside, sentinel=0.0)` | Realize coverage as sparse HEALPix map |
| `boundary()` | Boundary loops at the MOC's maximum order |
| `write_fits(path)` | Write IVOA MOC FITS |
| `to_ascii(compact=False)` / `to_json()` | Text serialization; compact ASCII emits contiguous pixel runs as ranges |
| `is_empty` | Whether the MOC has no covered cells |
| `area` / `area_sq_deg` | Covered area |

```python
from torchsky.catalogs import Catalog
from torchsky.sphere import MOC

cat = Catalog(lon=ra, lat=dec)
moc = MOC.from_circle(10.0, 20.0, 0.5, max_order=8)
inside = cat.inside(moc)
subset = cat.take(inside)
```

`HealpixIndex.lookup_moc(moc)` prunes source queries with the index's occupied
fixed-resolution buckets and then applies exact point-in-MOC containment.
`HealpixIndex.occupied_moc()` exposes those buckets as standard MOC coverage,
without introducing another sky index.

### Class `MultiOrderHealpixMap`

`MultiOrderHealpixMap` is a value layer over sorted, disjoint IVOA UNIQ cells.
MOC owns coverage and overlap semantics; this class does not define another
index. Values may require gradients, while integer UNIQ topology does not.

```python
from torchsky.sphere import MOC, MultiOrderHealpixMap

probability = MultiOrderHealpixMap(
    uniq,
    probdensity,
    values_are_densities=True,
)
mass = probability.integral(footprint_moc)
values, valid = probability.get_values_pos(lon, lat, return_valid=True)
```

`from_columns(...)` and `to_columns(...)` exchange standard `UNIQ` plus value
columns. Fixed-order maps convert to and from `SparseHealpixMap`. Optional
host-side adapters cover `mhealpy.HealpixMap` and
`cdshealpix.skymap.SkymapExplicit` when those packages are installed. These
adapters detach and copy values to CPU; tensor-native computation has no peer
dependency or backend.

`boundary()` returns one or more closed loops of shape `[N, 2]` in `(lon_deg, lat_deg)` coordinates. The loops are traced from stitched HEALPix cell edges at the MOC's current maximum order.

### Class `STMOC`

Space-Time Multi-Order Coverage (IVOA MOC 2.0). Time cells use order 61 = 1 µs since JD=0; spatial cells are torchsky `MOC` UNIQ coverage. This is not a mocpy wrapper.

```python
from torchsky.sphere import MOC, STMOC, nest2uniq

space = MOC(nest2uniq(4, torch.tensor([0, 1, 2])))
stmoc = STMOC.from_spatial_coverages(59000.0, 59001.0, space, time_depth=16)
inside = stmoc.contains(59000.5, 0.0, 0.0)
window = stmoc.query_by_time(59000.0, 59000.5)
```

| Constructor / method | Description |
|---|---|
| `STMOC.empty(...)` | No segments |
| `from_spatial_coverages(t_start_mjd, t_end_mjd, spatial, time_depth=...)` | One spatial MOC per half-open MJD interval |
| `from_times_positions(time_mjd, lon, lat, time_depth=..., spatial_depth=...)` | Quantize catalog samples onto time × HEALPix cells |
| `contains(time_mjd, lon_deg, lat_deg)` | Point-in-STMOC (broadcasts) |
| `filter_catalog(...)` | Keep samples inside the coverage |
| `query_by_time(t_start_mjd, t_end_mjd)` | Spatial union of segments overlapping `[t_start, t_end)` |
| `union` / `intersection` | Sweep over time breakpoints |
| `write_fits` / `read_fits` | IVOA RANGE packaging (`MOCDIM=2`; time pairs have bit 63 set) |
| `to_ascii` / `from_ascii` | Native `t<depth>/<c0>-<c1>` plus spatial MOC ASCII. Not a mocpy STMOC clone. |

Time conversion helpers: `mjd_to_time_us`, `time_us_to_mjd`. Polar motion / IERS is not involved.

---

## Harmonic Space (SHT)

Preferred module import:

```python
import torchsky.sphere.spectral as harmonics
```

### `map2alm(map_values, lmax=None, mmax=None, *, nside=None, nest=False, iter=0, pol=False, use_weights=False, use_pixel_weights=False, gal_cut=0.0, datapath=None)`

Analyze a HEALPix map into spherical harmonic coefficients.

- Scalar: `(npix,)` in, `(nalm,)` out.
- Polarized (`pol=True`): `(3, npix)` in, `(3, nalm)` out for (T, E, B).

### `alm2map(alm_values, nside, *, lmax=None, mmax=None, nest=False, pol=False, pixwin=False)`

### `map2alm_lsq(map_values, lmax, mmax=None, *, nside=None, pol=True, tol=1e-10, maxiter=20, nest=False)`

Iterative least-squares map analysis. Returns `(alm, rel_res, n_iter)`.

### SHT references and peers

**healpy** (ducc) and **s2fft** are optional references for HEALPix scalar and
spin SHT. Comparator setup and parity criteria are maintained in
[`torchsky-harness`](https://github.com/astroai/torchsky-harness); these packages
are not core dependencies and must not appear under `src/torchsky`.

`torchsky.sphere.sht` provides immutable geometry, request, capability, and
result contracts with separate `synthesize`, `adjoint_synthesize`, quadrature
`analyze`, and least-squares `pseudo_analyze` operations. `CARGeometry` accepts
only separable, unrotated, full-longitude CAR grids with Clenshaw-Curtis or
Fejer-1 latitude centers. Unsupported WCS and backend semantics fail explicitly.
The historical HEALPix `map2alm` and `alm2map` defaults are unchanged.

s2fft conventions that differ from healpy/torchsky:

- Bandlimit `L` is `lmax + 1`.
- Alm layout is typically `(L, 2L-1)` complex; torchsky/healpy use packed `m ≥ 0`.
- `method="jax_healpy"` is healpy. Do not treat it as a second oracle. Peer
  comparisons use native `sampling="healpix"` (`numpy` / `jax` / `torch`).
- HEALPix has no sampling theorem, so native s2fft vs healpy residuals are
  expected to depend on sampling and bandlimit. Acceptance tolerances and
  measured comparisons are maintained in the harness.

**cuHPX** is a CUDA-only research spike for scalar HEALPix SHT and spectral
HEALPix↔lat-lon regrid. It is not a core backend, has no spin-2 path, defaults
to `lmax = 2*nside+1` and float32, and must not be compared blindly to healpy's
`3*nside-1`. See [Limitations](#limitations).

**torch-harmonics** remains optional. Its adapter supports scalar synthesis and
quadrature analysis on compatible equiangular Clenshaw-Curtis CAR grids; it is
not a HEALPix backend and does not provide adjoint-synthesis or least-squares
semantics through this adapter.

### `map2alm_spin(maps, spin, *, nside=None, lmax=None, mmax=None, nest=False, iter=0)`

Spin-weighted map → alm transform. Input `maps` shape must be `(2, npix)`.

### `alm2map_spin(alms, nside, spin, *, lmax=None, mmax=None, nest=False)`

Synthesize a `(2, npix)` spin-component map pair from an `(alm_plus, alm_minus)`
tuple of coefficients.

### Coefficient Utilities

| Function | Description |
|---|---|
| `alm_size(lmax, mmax=None)` | Number of alm coefficients |
| `alm_index(ell, m, lmax)` | Flat index for mode (ell, m) |
| `almxfl(alm, fl)` | Multiply by ell-dependent filter |
| `rotate_alm(alm, psi, theta, phi)` | zyz Euler rotation (healpy convention). Returns a new tensor. |

---

## Power Spectra

### `anafast(map1, map2=None, *, nside=None, lmax=None, mmax=None, nest=False, nspec=None, iter=0, alm=False, pol=True, use_weights=False, use_pixel_weights=False, gal_cut=0.0, datapath=None)`

Auto/cross angular power spectrum C_ell.

### `alm2cl(alms1, alms2=None, *, lmax=None)`

Power spectrum from alm coefficients.

### CAR pseudo-C_ell estimation

`PseudoClField` binds a CAR map, scalar mask, bandlimit, and optional beam,
pixel windows, and deprojection templates. `coupled_spectra` uses IAU E/B
signs and returns scalar spectra, `(TE, TB)` for spin-0 x spin-2, or
`(EE, EB, BE, BB)` for spin-2 x spin-2. `ModeCouplingWorkspace` provides exact
scalar and E/B block coupling through blocked correlation-space
Gauss-Legendre quadrature, binned decoupling, and bandpower windows. Workspace
construction analyzes masks through `2*lmax` and requires
`nlat >= 2*lmax + 1` and `nlon >= 4*lmax + 1`. Spin binning must exclude
multipoles below two.

`BandpowerBinning` carries paired binning and unbinning operators. Decoupling
uses a direct solve only after explicit rank and condition checks; it never
selects a ridge or pseudoinverse fallback. `couple_noise` propagates unbinned
sky noise through mask, beam, pixel, and E/B blocks; `noise_bias` and
`deprojection_bias` are coupled spectra subtracted before binning.
`template_deprojection_bias` estimates the additive cleaning bias from an
explicit ensemble of already beam/pixel-convolved signal realizations.

`ApodizedMask.from_mask` constructs and validates the first and second spin
derivatives needed for pure E/B integration by parts. The current pure
estimator intentionally accepts only band-limited masks that vanish with
their first derivative on the two polar boundary rings and remain positive in
the interior; masks with holes or other boundary geometries fail closed.

### `synalm(cls, lmax=None)`

Generate random alm from input C_ell.

### `synfast(cls, nside, lmax=None, mmax=None, *, alm=False, pol=True, pixwin=False, fwhm=0.0, sigma=None, new=False)`

Synthesize map(s) from input $C_\ell$ power spectrum/spectra.

```python
cl = torch.ones(3 * nside)
m = sphere.synfast(cl, nside=nside)
```

### `smoothmap(map_values, *, nside=None, lmax=None, mmax=None, nest=False, fwhm_rad=None, sigma=None, beam=None, beam_window=None, pol=False, iter=0, use_weights=False, use_pixel_weights=False, verbose=True)`

Smooth a HEALPix map in harmonic space with a Gaussian beam.

### `smoothalm(alm_values, *, lmax=None, mmax=None, fwhm_rad=None, beam=None, pol=False)`

Apply beam smoothing directly to alm coefficients.

---

## Beams & Windows

| Function | Description |
|---|---|
| `gaussian_beam(fwhm_rad, lmax, *, pol=False)` | Gaussian beam transfer function B_ell |
| `pixwin(nside, *, pol=False, lmax=None)` | HEALPix pixel window function |
| `bl2beam(bl, theta)` / `beam2bl(beam, theta, lmax)` | Beam profile to/from harmonic window |

Pixel window coefficients are embedded as generated package data—no healpy or
FITS reader runtime dependency.

---

## Execution and Tuning

The SHT implementation uses:

- Ring FFT + Legendre recurrence blocking.
- OpenMP parallelism over rings.
- PyTorch `torch.fft` for azimuthal transforms.
- C++ backend (default on CPU) with compiled recurrence kernels. Falls back to
  pure PyTorch if the extension is unavailable.

These details describe execution paths, not comparative performance. See the
harness for current throughput evidence and comparator policy.

### Environment Knobs

| Variable | Default | Description |
|---|---|---|
| `TORCHSKY_SCALAR_MAX_CACHE_BYTES` | `512 MiB` | Basis-matrix cache budget |
| `TORCHSKY_SCALAR_RING_AUTO_MIN_BYTES` | `64 MiB` | Min map size to use ring path |
| `TORCHSKY_SPIN_RING_AUTO_MIN_BYTES` | `32 MiB` | Min map size for spin ring path |
| `TORCHSKY_SPIN_MAP2ALM_RING_CONCAT_CPP` | `1` | Fused CPU RING concat for `map2alm_spin` |
| `TORCHSKY_SPIN_ISHIOKA` | `1` | Ishioka 2018 μ² skip-2 for spin-1/2 CPU `map2alm_spin` / `alm2map_spin`. Set `0` for the Wigner recurrence. |
| `TORCHSKY_SPIN_THREADS` | unset | Cap the CPU spin `m`-pool. Unset: Linux min(affinity, cgroup quota, ATen threads, `hardware_concurrency`); Apple uses `hardware_concurrency`. CANFAR launch sets this to `--cpu`. |
| `TORCHSKY_SHT_PROFILE` | unset | When set, print `SpinMPool` size and `alm2map_spin` interpolate / equatorial FFT / polar FFT milliseconds on stderr. |

---

## Compatibility

See also **[Ecosystem compatibility](compatibility.md)** for how healpy/hpgeom fit into torchsky’s wider goal of replacing or interoperating with **Astropy, spherical_geometry**, and other common packages—not HEALPix alone.

Status: `parity` = upstream-compatible semantics · `extended` = torchsky-native additions · `partial` = known gaps · `planned` = not yet implemented.

Functions in `torchsky.sphere.compat` (and the `torchsky.sphere.healpy` alias) mirror the [healpy](https://healpy.readthedocs.io/) and [hpgeom](https://github.com/LSSTDESC/hpgeom) APIs and can be used as drop-in replacements at the tensor boundary.

| Function | Status | Notes |
|---|---|---|
| `ang2pix` / `pix2ang` | parity | ring/nest, lonlat/theta-phi |
| `ring2nest` / `nest2ring` | parity | |
| `vec2pix` / `pix2vec` / `ang2vec` / `vec2ang` | parity | |
| `get_all_neighbours` / `get_interp_weights` / `get_interp_val` | parity | |
| `boundaries` | parity | healpy-style boundary sampling |
| `ud_grade` | parity | `power`, `pess`, bad-mask semantics |
| `isnsideok` / `isnpixok` | parity | scalar + tensor forms |
| `max_pixrad` | parity | conservative radius |
| `query_disc` | parity | inclusive `fact==4` pads by max pixel radius; `fact!=4` supersamples at `nside*fact` then parent-degrades |
| `query_circle` | parity | lon/lat wrapper |
| `query_circle_vec` | parity | Cartesian-vector form |
| `query_polygon` | parity | convex half-spaces; concave simple polygons go through `query_polygon_general` |
| `query_polygon_vec` | parity | Cartesian-vector form |
| `query_box` | parity | wrap-aware longitude intervals |
| `query_strip` | parity | colatitude strip |
| `query_ellipse` | parity | hpgeom-style units; torch-native only |
| `pixel_ranges_to_pixels` | parity | half-open/closed range expansion |
| `upgrade_pixel_ranges` | parity | NEST pixel-range upgrade |
| `pixels_to_pixel_ranges` | extended | torchsky-native range compression |
| `map2alm` | parity | scalar + polarized (`pol=True`); `iter`; RING quadrature weights; `gal_cut` |
| `map2alm_lsq` | parity | returns `(alm, rel_res, n_iter)`; torch-native |
| `alm2map` | parity | scalar + polarized; `pixwin`; torch-native |
| `almxfl` | parity | short `fl` tails zero-filled (healpy semantics) |
| `alm2cl` | parity | auto/cross; healpy-compatible stacked ordering |
| `anafast` | parity | scalar + polarized; `nspec`, `alm=True`; torch-native |
| `synalm` | parity | scalar/multi-field; `new` ordering |
| `synfast` | parity | scalar/polarized; optional `alm` return |
| `gaussian_beam` / `gauss_beam` | parity | scalar + polarized |
| `pixwin` | parity | embedded generated pixel-window data (no healpy/FITS runtime dependency) |
| `bl2beam` / `beam2bl` | parity | beam profile ↔ harmonic window |
| `smoothalm` | parity | scalar + polarized |
| `smoothmap` / `smoothing` | parity | scalar + polarized; torch-native |
| `map2alm_spin` / `alm2map_spin` | parity | integer `spin≥0`; torch-native (compare to healpy in tests/benchmarks) |
| `fit_monopole` / `remove_monopole` | parity | `bad`/`gal_cut` masking semantics |
| `fit_dipole` / `remove_dipole` | parity | `bad`/`gal_cut` masking semantics |
| `spherical_polygon_contains` | extended | inside-point disambiguation |
| `spherical_polygons_intersect` | extended | edge crossing + containment |
| `SphericalBooleanRegion` (union/intersection/difference) | extended | controlled-error boolean algebra; optional exact backend |
| `NativeExactSphericalRegion` | extended | dependency-free exact-backend scaffold |
| `HealSparseMap` / `SparseMap` | parity | coverage-aware sparse map; dense-equivalent `ud_grade` |

Strict mode (`torchsky.sphere.compat.set_strict_mode(True)`) is retained for API compatibility. Geometry queries are always torch-native; they do not dispatch to hpgeom.

The `parity` labels below describe intended API semantics and compatibility
scope. They are not an exhaustive parity certificate; current fixture coverage
and quantitative acceptance evidence live in `torchsky-harness`.

---

## Limitations

| Area | Limitation |
|---|---|
| `map2alm` weights | `use_weights` applies computed RING quadrature (mean 1). `use_pixel_weights` loads HEALPix full-pixel FITS via torchfits from `datapath` (`1+w` corrections). The two flags are mutually exclusive. Missing pixel-weight files raise; nothing is downloaded or vendored. Supply the files through `TORCHSKY_HEALPIX_DATAPATH`. |
| `alm2map_spin` | CPU RING FFTs (polar and equatorial) use in-process pocketfft. Set `TORCHSKY_SPIN_ISHIOKA=0` at call time to select the Wigner recurrence instead of the Ishioka path. CUDA/MPS keep maps/alms on device via the ATen ring path; fused Ishioka/Wigner C++ stays CPU. Comparative throughput and parity evidence are maintained in the harness. |
| Pixel windows | Packaged tables cover standard HEALPix NSIDE range. Higher NSIDE may require external data. |
| Device | See the device table below. Fused Ishioka/Wigner is CPU C++. CUDA/MPS SHT uses the ATen ring FFT path on device (no host copy of the map). `rotate_alm` and `alm2lonlat` stay on device (ATen; CUDA float64 / MPS float32). |
| s2fft | Optional bench/test peer only. Native HEALPix has no sampling theorem; do not use `method="jax_healpy"` as an oracle. |
| cuHPX | Not a production backend. Its scalar CUDA SHT and spectral regrid remain an external research comparison; any future compatibility or performance decision belongs in the harness. Its default `lmax=2*nside+1` and float32 behavior must not be compared blindly to healpy `3*nside-1`. |

### Device policy

| Job | CPU | CUDA | MPS |
|---|---|---|---|
| WCS pixel↔world | yes | yes float64 | yes float32 |
| HEALPix geometry | yes | yes | yes, looser index gates |
| Sparse `get_values` / MOC contains | yes | yes if tensors stay on device | yes float32 |
| Polygons (approximate contains/intersect/`query_polygon`) | yes | yes | yes float32 |
| Polygons (exact `NativeExactSphericalRegion`) | yes | no (CPU kernel / spherical-geometry) | no |
| SHT (Ishioka/Wigner C++) | yes float64 | no (CPU kernel) | no (CPU kernel) |
| SHT (ATen ring FFT) | yes float64 | yes float64 | yes float32 |
| rotate_alm / alm2lonlat | yes float64 | yes float64 | yes float32 |
| Observed place | yes float64 | yes float64 | yes float32 |

Default `to_exact` / `to_exact_region` is the native torchsky backend.
Quantitative oracle comparisons for the optional `spherical-geometry` backend
are recorded in the harness.

---

## See Also

- [healpy documentation](https://healpy.readthedocs.io/)
- [HEALPix home page](https://healpix.sourceforge.io/)
- [hpgeom](https://github.com/LSSTDESC/hpgeom)
- [astropy-healpix](https://astropy-healpix.readthedocs.io/)
- [healsparse](https://healsparse.readthedocs.io/)
- [spherical-geometry](https://github.com/spacetelescope/spherical_geometry)
- [API reference](reference/sphere.md)
- [Scientific evaluation ownership](evaluation/benchmarks.md)
