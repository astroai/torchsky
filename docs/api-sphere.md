# Sphere and HEALPix API reference

`torchsky.sphere` provides tensor-native spherical geometry, HEALPix indexing,
regions, coverage maps, and harmonic transforms. The [sphere guide](sphere.md)
covers semantics and recipes; this page is the reference for the promoted names.

All functions accept torch tensors and run on CPU, CUDA, and MPS unless a
docstring says otherwise (some table or reference paths are CPU-only).

---

## HEALPix indexing and conversion

| Name | Role |
|---|---|
| `ang2pix`, `pix2ang` | Angles ↔ pixel indices. |
| `vec2pix`, `pix2vec` | Cartesian vectors ↔ pixel indices. |
| `ring2nest`, `nest2ring`, `reorder` | RING/NESTED ordering conversions. |
| `nside2npix`, `npix2nside` | Resolution ↔ pixel count. |
| `nside2order`, `order2nside` | Resolution ↔ order. |
| `nside2pixarea`, `nside2resol` | Pixel area and approximate resolution. |
| `neighbors`, `boundaries` | 8-neighbour sets and pixel boundary vectors. |
| `get_interp_weights`, `get_interp_val` | Bilinear interpolation (healpy-compatible). |
| `upgrade_pixels`, `degrade_pixels` | Explicit pixel-id resolution changes. |
| `pixels_to_pixel_ranges`, `pixel_ranges_to_pixels` | Compress/expand pixel sets. |
| `upgrade_pixel_ranges`, `degrade_pixel_ranges` | NESTED pixel-range resolution changes. |
| `nest2uniq`, `uniq2nest` | NESTED ↔ UNIQ indexing. |
| `mjd_to_time_us`, `time_us_to_mjd` | TMOC time index conversion. |

---

## Queries and sampling

| Name | Role |
|---|---|
| `query_disc`, `query_cone`, `query_circle` | Pixels whose centers fall in a spherical disc. |
| `query_box`, `query_strip` | Lon/lat boxes and colatitude strips. |
| `query_polygon`, `query_polygon_general` | Convex and general simple polygons. |
| `query_ellipse` | Pixels inside a sky ellipse. |
| `sample_map` | Unified HEALPix / multiband / multiwavelength sampler. |
| `sample_healpix_map` | Sample maps at arbitrary sky coordinates. |
| `sample_multiband_healpix`, `sample_multiwavelength_healpix` | Band/wavelength cubes with optional spectral resampling. |
| `interpolate_wavelength_axis` | Linear spectral-axis interpolation. |
| `sample_randoms` | Uniform-on-the-sphere random coordinates. |
| `ud_grade` | healpy-compatible resolution upgrade/degrade. |

---

## Spherical geometry and regions

| Name | Role |
|---|---|
| `LonLat`, `UnitVector3d`, `Vector3d` | Coordinate and vector containers. |
| `SphericalPolygon`, `SphericalMultiPolygon`, `SphericalCap`, `SphericalBox`, `SphericalEllipse` | Exact spherical regions. |
| `SphericalBooleanRegion` | Boolean compositions with controlled-error HEALPix evaluation. |
| `ExactSphericalRegion`, `NativeExactSphericalRegion` | Optional exact-region backends. |
| `Circle`, `Cone`, `Cap` | Spherical-cap aliases. |
| `PixelizedRegion`, `WideBitMask` | Pixelized masks and set operations. |
| `RegionAreaEstimate` | Controlled-error area estimate summary over an nside ladder. |
| `MOC`, `BMOC`, `STMOC` | Multi-order coverage maps (spatial, boolean, space-time). |
| `MultiOrderHealpixMap` | Differentiable values over standard UNIQ cells; MOC retains coverage semantics. |
| `SparseHealpixMap`, `HealSparseMap`, `SparseMap`, `SkyMaskPipe` | Sparse HEALPix maps and mask pipelines. |
| `sum_intersection`, `sum_union` | Sparse-map combination rules. |
| `spherical_polygon_area`, `spherical_polygon_signed_area`, `spherical_triangle_area` | Exact-area helpers. |
| `convex_polygon_contains`, `spherical_polygon_contains`, `spherical_polygons_intersect` | Predicates. |
| `to_exact_region` | Convert a region to a backend implementation. |

Geometry utilities: `lonlat_to_unit_xyz`, `unit_xyz_to_lonlat`,
`wrap_longitude`, `great_circle_distance`, `pairwise_angular_distance`,
`chord_radius`, `slerp_lonlat`, `fit_monopole_dipole`, `remove_monopole_dipole`.

---

## Harmonic transforms

| Name | Role |
|---|---|
| `map2alm`, `alm2map` | Scalar analysis and synthesis in healpy ordering. |
| `map2alm_lsq` | Iterative least squares analysis to `(lmax, mmax)`. |
| `map2alm_batched`, `alm2map_batched` | Explicit leading batch dimension. |
| `map2alm_spin`, `alm2map_spin` | Spin-weighted (Q/U-like) transforms. |
| `anafast` | Auto/cross power spectra from maps. |
| `alm2cl`, `almxfl`, `rotate_alm` | Spectra, filtering, and rotations. |
| `synalm`, `synfast` | Gaussian realizations from `C_l`. |
| `smoothalm`, `smoothmap` | Beam smoothing in harmonic or map space. |
| `gaussian_beam`, `pixwin`, `bl2beam`, `beam2bl` | Beam and pixel window functions. |
| `alm2lonlat`, `alm2lonlat_spin` | Evaluate alms at arbitrary sky coordinates. |
| `alm2map_der1` | Map plus first angular derivatives. |
| `alm_size`, `alm_index` | healpy-compatible alm sizing and indexing. |
| `PseudoClField`, `coupled_spectra` | Masked CAR scalar and spin-2 coupled auto/cross spectra. |
| `BandpowerBinning`, `ModeCouplingWorkspace` | Exact scalar pseudo-C_ell coupling, fail-closed decoupling, and bandpower windows. |

The implementation lives under `torchsky.sphere` submodules
(`torchsky.sphere.healpix`, `torchsky.sphere.spectral`,
`torchsky.sphere.geom`, `torchsky.sphere.moc`, `torchsky.sphere.sparse`, ...), with the healpy-shaped `compat`/`healpy`
namespaces kept for migration. See [compatibility](compatibility.md) for the
boundary rules.
