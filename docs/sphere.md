# Sphere & HEALPix

`torchsky.sphere` is a PyTorch-native library for spherical geometry and HEALPix maps, with high-performance C++ kernels for Spherical Harmonic Transforms (SHT) and selected polygon geometry hot paths on CPU. Harmonic transforms are available through `torchsky.sphere.harmonics`; the older `torchsky.sphere.spectral` module remains a compatibility alias.

## Table of Contents

- [Pixel & Coordinate Utilities](#pixel--coordinate-utilities)
- [Coordinate Conversions](#coordinate-conversions)
- [Geometric Queries](#geometric-queries)
- [Spherical Geometry](#spherical-geometry-classes)
- [Sampling](#sampling)
- [Sparse Maps](#sparse-maps)
- [MOC](#moc)
- [STMOC](#class-stmoc)
- [Harmonic Space (SHT)](#harmonic-space-sht)
- [Power Spectra](#power-spectra)
- [Beams & Windows](#beams--windows)
- [Performance Notes](#performance-notes)
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

sparse.coverage_fraction
sparse.to_dense()
sparse.interpolate(lon_deg=lon, lat_deg=lat, method='bilinear')
sparse.ud_grade(nside_out=32)
```

`HealSparseMap.from_pixels(...)` canonicalizes inputs to sorted unique pixels,
keeping the last value supplied for duplicates, so `get_values(...)` and
interpolation can use indexed lookup reliably. Maps with `nside_coverage` derive
coverage maps in the NESTED coverage hierarchy even when sparse pixels are stored
in RING ordering, and `ud_grade(...)` preserves the coverage hierarchy when the
output resolution allows it. With `nside_coverage` set, `get_values` uses a fused
CPU gather (fill-block layout). On Mac that path matches healsparse at nside=256
and 50k nest queries (~1.03× interleaved). Linux CANFAR at the same 50k cap is
behind (0.77× at nside=256, 0.80× at nside=512; not interleaved). Float maps with
`nside_coverage` also degrade through that layout when coverage nside is
unchanged; on Mac that matches healsparse at nside=256→128 and 512→256 (10% nest,
interleaved). Linux dedicated `ud_grade` rows were ahead on that job (2.06× /
3.31×) with separate medians and are not claimed. Bool/int maps
keep the pixel-list path because fill-block cells cannot tell valid False/0 from
an uncovered child.

`HealSparseMap.from_moc(moc, nside=...)` realizes MOC coverage directly as sparse
nested pixels without materializing a full dense HEALPix mask. Maps with
`nside_coverage` and a 1-D value vector write HealSparse FITS (`EXTNAME=COV`
int64 coverage plus `EXTNAME=SPARSE` image, `PIXTYPE=HEALSPARSE`). Boolean maps
are stored as `uint8` with `BOOLMAP`. Pixel-list maps without coverage still use
a PIXEL/VALUE table. Sentinel values are not valid pixels on the image path.

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
| `MOC.from_circle(lon_deg, lat_deg, radius_deg, max_order=10)` | Circular cap footprint |
| `MOC.from_polygon(lon_deg, lat_deg, max_order=10)` | Simple polygon footprint, including non-convex polygons |
| `MOC.from_region(region, max_order=None)` | Pixelize a torchsky spherical region, boolean region, exact-region wrapper, or `PixelizedRegion` into NUNIQ coverage; `PixelizedRegion` inputs preserve their current resolution by default, while other regions default to order 10 |
| `MOC.from_sparse_map(smap)` | Coverage from `HealSparseMap` / `SparseMap` |
| `MOC.from_threshold(smap, threshold, op='>')` | Threshold sparse-map values |
| `MOC.from_ascii(text)` | IVOA ASCII roundtrip, including compact pixel ranges like `3/5-7` |
| `MOC.from_json(text)` | IVOA JSON roundtrip |
| `MOC.read_fits(path)` | IVOA FITS roundtrip |

**Key methods and properties:**

| Method | Description |
|---|---|
| `contains(lon_deg, lat_deg)` | Point-in-MOC predicate |
| `contains_pixels(nside, pixels, nest=True)` | Pixel-in-MOC predicate for HEALPix-indexed catalogs or sparse maps |
| `filter_catalog(lon_deg, lat_deg)` | Filter catalog coordinates by coverage |
| `contains_moc(other)` | Subset check |
| `union(other)` / `intersection(other)` / `difference(other)` | Boolean operations |
| `flatten(nside)` | Dense boolean mask at fixed resolution |
| `to_pixelized_region(nside, nest=True)` | Realize coverage as a `PixelizedRegion`, with optional RING conversion |
| `to_sparse_map(nside, sentinel=0.0)` | Realize coverage as sparse HEALPix map |
| `boundary()` | Boundary loops at the MOC's maximum order |
| `write_fits(path)` | Write IVOA MOC FITS |
| `to_ascii(compact=False)` / `to_json()` | Text serialization; compact ASCII emits contiguous pixel runs as ranges |
| `area` / `area_sq_deg` | Covered area |

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
import torchsky.sphere.harmonics as harmonics
```

### `map2alm(map_values, lmax=None, mmax=None, *, nside=None, nest=False, iter=0, pol=False, use_weights=False, use_pixel_weights=False, gal_cut=0.0, datapath=None)`

Analyze a HEALPix map into spherical harmonic coefficients.

- Scalar: `(npix,)` in, `(nalm,)` out.
- Polarized (`pol=True`): `(3, npix)` in, `(3, nalm)` out for (T, E, B).

### `alm2map(alm_values, nside, *, lmax=None, mmax=None, nest=False, pol=False, pixwin=False)`

### `map2alm_lsq(map_values, lmax, mmax=None, *, nside=None, pol=True, tol=1e-10, maxiter=20, nest=False)`

Iterative least-squares map analysis. Returns `(alm, rel_res, n_iter)`.

### SHT oracles and peers

**healpy** (ducc) is the science oracle for HEALPix scalar and spin SHT parity.
**s2fft** is an optional independent HEALPix implementation for benches and tests
only (`pip install -e ".[sht-bench]"` or `pixi run sht-bench-bootstrap`). It is
not a core dependency and must not appear under `src/torchsky`.

s2fft conventions that differ from healpy/torchsky:

- Bandlimit `L` is `lmax + 1`.
- Alm layout is typically `(L, 2L-1)` complex; torchsky/healpy use packed `m ≥ 0`.
- `method="jax_healpy"` is healpy. Do not treat it as a second oracle. Peer
  comparisons use native `sampling="healpix"` (`numpy` / `jax` / `torch`).
- HEALPix has no sampling theorem, so native s2fft vs healpy residuals are
  allowed; torchsky vs healpy stays tight.

**cuHPX** is a CUDA-only research spike for scalar HEALPix SHT and spectral
HEALPix↔lat-lon regrid. It is not a core backend, has no spin-2 path, defaults
to `lmax = 2*nside+1` and float32, and must not be compared blindly to healpy's
`3*nside-1`. See [Limitations](#limitations).

SFNO / torch-harmonics stay out: they are lat-lon weather architectures, not a
HEALPix SHT backend.

### `map2alm_spin(maps, spin, *, nside=None, lmax=None, mmax=None, nest=False, iter=0)`

Spin-weighted map → alm transform. Input `maps` shape must be `(2, npix)`.

### `alm2map_spin(alm_plus, alm_minus, nside, spin, *, lmax=None, mmax=None, nest=False)`

Synthesize two spin-component maps from `(alm_plus, alm_minus)` coefficient pairs.

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

## Performance Notes

The SHT implementation uses:

- Ring FFT + Legendre recurrence blocking for register-efficient throughput.
- OpenMP parallelism over rings.
- PyTorch `torch.fft` for azimuthal transforms.
- C++ backend (default on CPU) with compiled recurrence kernels. Falls back to pure PyTorch if the extension is unavailable.

For large $N_\text{side}$ or batch transforms, `torchsky.sphere` substantially outperforms pure-Python HEALPix implementations.

### Environment Knobs

| Variable | Default | Description |
|---|---|---|
| `TORCHSKY_SCALAR_MAX_CACHE_BYTES` | `512 MiB` | Basis-matrix cache budget |
| `TORCHSKY_SCALAR_RING_AUTO_MIN_BYTES` | `64 MiB` | Min map size to use ring path |
| `TORCHSKY_SPIN_RING_AUTO_MIN_BYTES` | `32 MiB` | Min map size for spin ring path |
| `TORCHSKY_SPIN_MAP2ALM_RING_CONCAT_CPP` | `1` | Fused CPU RING concat for `map2alm_spin` |
| `TORCHSKY_SPIN_ISHIOKA` | `1` | Ishioka 2018 μ² skip-2 for spin-1/2 CPU `map2alm_spin` / `alm2map_spin`. Set `0` for the Wigner recurrence. |
| `TORCHSKY_SPIN_THREADS` | unset | Cap the CPU spin `m`-pool. Unset: Linux min(affinity, cgroup quota, ATen threads, `hardware_concurrency`); Apple uses `hardware_concurrency`. CANFAR launch sets this to `--cpu`. |
| `TORCHSKY_SHT_PROFILE` | unset | When set, print `SpinMPool` size and `alm2map_spin` interpolate vs synthesize milliseconds on stderr. |

---

## Compatibility

See also **[Ecosystem compatibility](compatibility.md)** for how healpy/hpgeom fit into torchsky’s wider goal of replacing or interoperating with **Astropy, fitsio, spherical_geometry**, and other common packages—not HEALPix alone.

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
| `query_disc` | parity | inclusive pads by max pixel radius; `fact` must be 4 |
| `query_circle` | parity | lon/lat wrapper |
| `query_circle_vec` | parity | Cartesian-vector form |
| `query_polygon` | parity | convex only; raises on concave input |
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

---

## Limitations

| Area | Limitation |
|---|---|
| `map2alm` weights | `use_weights` applies computed RING quadrature (mean 1). `use_pixel_weights` loads HEALPix full-pixel FITS via torchfits from `datapath` (`1+w` corrections). The two flags are mutually exclusive. Missing pixel-weight files raise; nothing is downloaded or vendored. |
| `alm2map_spin` | Fused Ishioka CPU synthesis is ~1.6× healpy on Mac at nside=128 after restoring in-process polar RING FFTs (Cooley–Tukey/Bluestein; Accelerate radix-2 on Apple). Equatorial rings stay batched ATen. Linux CANFAR SHA `425af888` (in-process polar FFT, uncapped `SpinMPool`): `map2alm_spin` ~1.4× healpy at nside=128; `alm2map_spin` ~52×. That synthesis row is worse than SHA `9e65cbc` (~3×). Do not claim Linux SHT. Set `TORCHSKY_SPIN_ISHIOKA=0` at call time if Ishioka loses to Wigner. CUDA/MPS keep maps/alms on device via the ATen ring path; fused Ishioka/Wigner C++ stays CPU. Do not claim a fused CUDA Ishioka kernel. |
| Pixel windows | Packaged tables cover standard HEALPix NSIDE range. Higher NSIDE may require external data. |
| Device | See the device table below. Fused Ishioka/Wigner is CPU C++. CUDA/MPS SHT uses the ATen ring FFT path on device (no host copy of the map). `rotate_alm` and `alm2lonlat` stay on device (ATen; CUDA float64 / MPS float32). |
| s2fft | Optional bench/test peer only. Native HEALPix has no sampling theorem; do not use `method="jax_healpy"` as an oracle. |
| cuHPX | **No-go** as a torchsky backend until a CANFAR CUDA spike (`scripts/run_canfar_cuhpx_sht_spike.sh`) reports `go-scalar-cuda`. Even then: optional env-gated scalar CUDA adapter only, still healpy-tested, never default on CPU/MPS, no spin-2, no core/`pyproject.toml` dependency. Default cuHPX `lmax=2*nside+1` and float32 must not be compared blindly to healpy `3*nside-1`. |

### Device policy

| Job | CPU | CUDA | MPS |
|---|---|---|---|
| WCS pixel↔world | yes | yes float64 | yes float32 |
| HEALPix geometry | yes | yes | yes, looser index gates |
| Sparse `get_values` / MOC contains | yes | yes if tensors stay on device | yes float32 |
| Polygons | yes | no | no |
| SHT (Ishioka/Wigner C++) | yes float64 | no (CPU kernel) | no (CPU kernel) |
| SHT (ATen ring FFT) | yes float64 | yes float64 | yes float32 |
| rotate_alm / alm2lonlat | yes float64 | yes float64 | yes float32 |
| Observed place | yes float64 | yes float64 | yes float32 |

Exact spherical polygons (`to_exact`, `NativeExactSphericalRegion`) match spherical-geometry area to `1e-12` steradians. The approximate CPU backend is the throughput path; document that bound before treating approximate contains as exact.

---

## See Also

- [healpy documentation](https://healpy.readthedocs.io/)
- [HEALPix home page](https://healpix.jpl.nasa.gov/)
- [hpgeom](https://github.com/LSSTDESC/hpgeom)
- [astropy-healpix](https://astropy-healpix.readthedocs.io/)
- [healsparse](https://healsparse.readthedocs.io/)
- [spherical-geometry](https://github.com/spacetelescope/spherical_geometry)
- [API guide](api.md)
- [Benchmarks](benchmarks.md)
