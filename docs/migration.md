# Migrating from Astropy, healpy, and reproject

`torchsky` is not a drop-in replacement for the legacy astronomy stack. It is a
tensor-native implementation of the same sky-domain operations, with explicit
inputs, explicit units, and no hidden file I/O. This page maps the common
migration paths and, just as importantly, states what does **not** carry over.

Read [compatibility.md](compatibility.md) for the ownership boundary: `torchfits`
owns FITS files and tables, `torchsky` owns coordinates, geometry, maps, and
models, and Astropy/healpy/reproject are references and oracles rather than
runtime backends.

## Migrating to 0.5

`0.5.0rc1` is the first public release of this code base as `torchsky`. It
replaces the PyPI package `torchfits` 0.3.1, and the `0.4.0` pre-releases were
never published, so everything below is relative to either starting point. The
package is pre-1.0: renames and removals are clean breaks with no compatibility
aliases. The full list is in the [changelog](changelog.md).

### Package and I/O

| Before | Now |
|---|---|
| `pip install torchfits==0.3.1` (FITS I/O, WCS, sphere in one wheel) | `pip install torchsky`; FITS files and tables come from the separate `torchfits` package (`>=1.3.1`, a required dependency) |
| `import torchfits` for WCS, HEALPix, spectral, transforms | `import torchsky` (`torchsky.wcs`, `torchsky.sphere`, `torchsky.spectral`, `torchsky.transforms`) |
| Root `read`, `write`, `open`, `get_header`, `get_wcs`, `Header`, `Card`, `HDUList`, `TensorHDU`, `TableHDU` | `torchfits.read`, `torchfits.read_header`, ...; build `WCS(header_mapping)` |
| `torchsky.io`, `.table`, `.hdu`, `.cache`, `.datasets`, `.dataloader`, `.header_parser`, `.core` | Removed; use `torchfits` (datasets from `torchfits.data`) |
| `pytorch-frame` adapter (`torchsky.frame`) | Removed; use Arrow or Polars |
| `torchsky info`, `torchsky cat`, `--sky`, `--inr-model`, `--per-band`, `--inr-*` | Torchfits CLI, `--background-mode`, `--field-model`, `--per-filter`, `--field-*` |
| Image transforms (stretches, normalisation, crops, augmentation) | `torchsky.transforms`; the index-based `RedshiftShift` is removed, use `torchsky.spectral.relativistic_doppler_shift` |

### WCS, sphere, spectral, coordinates

| Removed or renamed | Replacement |
|---|---|
| `WCS.pix2world`, `world2pix`, `calc_footprint`, `proj_plane_pixel_scales/area` | `pixel_to_world`, `world_to_pixel`, `footprint`, `pixel_scales`, `pixel_area` |
| `torchsky.sphere.healpy`, strict-mode setters, `strict=`/`verbose=` keywords, `query_strip(chunk_size=)`, `query_polygon_vec(fact=)` | `torchsky.sphere.compat`; delete the keywords |
| `HealSparseMap`, `SparseMap`, `get_values_pix`, `lookup_With_valid`, `generate_healpix_map`, `write(clobber=)` | `SparseHealpixMap`, `get_values`, `lookup_with_valid`, `to_dense`, `write_fits(overwrite=)` |
| `NativeExactSphericalRegion`, `backend="auto"`, `MOC.full(order)` | `NativeApproximateSphericalRegion`, `backend="native"`, `MOC.full()` |
| `max_pixrad`, `get_all_neighbours` in core HEALPix | `max_pixel_radius`, `neighbors` (peer spellings stay in `sphere.compat`) |
| `torchsky.spectral` redshift/velocity helpers | `torchsky.redshift`, `torchsky.log_lambda` |
| `transform_spectral`, `apply_extinction(lambda_obs_angstrom=)`, `median_3`/`gaussian_3` mesh filters, `zscale` as a low-level stretch | Canonical APIs in the sections below |
| `inr_bandpass`, `resolve_inr_bandpass` | `filter_profiles.field_bandpass`, `resolve_field_bandpass` |

### Map, scene, and field

| Before | Now |
|---|---|
| `torchsky.map.image_plane`, `map.operators.provenance`, `map.field.models`, `map.field.foam` | `torchsky.map`, `torchsky.map.provenance`, the `torchsky.map.field` facade |
| `evaluate_image_plane_aperture_uncertainty`, `CubeResult.line_uncertainty` | `torchsky.map.aperture_uncertainty` |
| `StrainerSkyEncoderMLPField.output`; `SourceCatalogScene` template spellings | `decoder_head`; `fixed_templates`, `coefficients` |
| `native_physical.py`, `joint_native.py`, `neural_inverse.py`, `operators/lensing.py` | `native_integrated_read`, `joint_native_likelihood`, `_gauss_newton` (private), `magnification` |
| `Scene` alias; detector-astrometry `tol=` | `SceneRepresentation`; `atol=`, `rtol=` |
| Field map plans with `model`, `model_kwargs`, `steps`, `learning_rate` keys | Regenerate with `field_*` keys; old plans are rejected |

### Names removed from public namespaces in 0.5

These were internal kernels or aliases that leaked through `__all__`. Import
them from their defining module only for white-box tests; they are not API.

| Removed from | Names | Use instead |
|---|---|---|
| `torchsky.wcs` | `pixel_to_world_fused`, `world_to_pixel_fused` | `WCS.pixel_to_world`, `WCS.world_to_pixel` |
| `torchsky.wcs` | `icrs_to_itrs_xyz`, `itrs_xyz_to_lonlat_deg` | `observed_altaz`, `transform` |
| `torchsky.wcs` | `ExperimentalConditionalResidual`, `ShadowUpdateResult`, `shadow_update_bayesian_modes` | `torchsky.wcs.residuals` (experimental) |
| `torchsky.wcs` | `canonical_celestial_frame`, `canonical_spectral_type`, `SPEED_OF_LIGHT` | `torchsky.wcs.frames`, `torchsky.wcs.spectral` |
| `torchsky.sphere` | `chord_radius`, `interpolate_wavelength_axis` | Private; `great_circle_distance`, `sample_multiwavelength_healpix` |
| `torchsky.scene` | `splat_to_native`, `sample_image_plane_batched_channels` | `torchsky.map.splat.splat_bilinear`; map operator kernels |
| `torchsky.field` | `Observation` (clashed with `map.Observation`), `ifu_cube_spectral_to_leading` | `FieldObservation`; `field_observations_from_ifu_cube` |
| `torchsky.encoders` | `SineLayer`, `sky_encoder_fourier_min_frequency` | `SirenImplicitField`, `SkyCoordinateEncoder` |
| `torchsky.redshift` | `require_one_plus_z`, `normalize_v_kind` | `one_plus_z_from_*`, `convert_velocity_kind` |
| `torchsky.quality` | `MAX_CONFIDENCE_SAMPLES` | None |
| modules | `torchsky.foreground`, `torchsky.map.field.gauss_newton` | Now private (`_foreground`, `_gauss_newton`) |
| `cosmology` | `n_steps=` (ignored since the Gauss-Legendre quadrature) | Drop it; passing it warns with `FutureWarning` and is removed in 0.6 |

Added in 0.5: `torchsky.map.coadd_image_plane`, `torchsky.map.ImageObservation`, and
`torchsky.wcs.TemporalWCS`.

### Behaviour changes

| Area | New behaviour |
|---|---|
| Devices | Map, field-fitting, and IFU APIs compute and return on the device of their tensor inputs; mixed devices raise `ValueError`. `default_compute_device()` only applies to path-backed inputs |
| MPS | Float64-only paths (exact/point drizzle, conservative polygon, forward splat, spectral resamplers, encoder fields) raise `NotImplementedError` up front |
| WCS | `PV1_1`/`PV1_2` raise `NotImplementedError` instead of being ignored; `LONPOLE`/`LATPOLE` follow WCSLIB; `CROTAi` builds the CD matrix when no CD/PC is present |
| `observed_altaz` | `pn` defaults to `iau2006` when `tt_minus_utc_s` is available |
| Redshift | Raises for `abs(beta) >= 1` |
| IFU cubes | An explicit spectral axis is required (no 16-point default) |
| Field PSF variance | Coherent mode propagates the standard deviation linearly (previously `K^2`, far too small); independent mode is unchanged |
| CLI | `torchsky sky match` with several inputs requires `{i}` in `--output` |
| Drizzle | Box splat and cube drizzle use the exact 1-D box/pixel overlap (flux-conserving for any `pixfrac`); point drizzle keeps drops in the outer half-pixel rim `[-0.5, n-0.5]`, which STScI drizzle discards |
| Image-plane resampling | Target-to-source mapping is the verified interpolated lattice (`mapping_mode("exact")` for per-pixel); large targets tile automatically |
| `coadd_healpix` | Bilinear window is the pixel-centre range; non-ICRS WCS convert through torchsky frames |
| Cosmology | Distances use per-point Gauss-Legendre quadrature in `ln(1+z)`; agreement with Astropy is below `1e-6` |
| Transforms | `ZScale` follows Astropy's `ZScaleInterval`; `RobustScale` uses the mean of the two middle values for even counts |
| Build | PyTorch 2.14 lane; `torchfits >= 1.3.1` |

## At a glance

| Legacy call | torchsky replacement | Notes |
|---|---|---|
| `astropy.wcs.WCS(header)` | [`torchsky.wcs.WCS`](reference/wcs.md#torchsky.wcs.WCS) | Header stays a plain dict; pixels are 0-based by default. |
| `wcs.pixel_to_world(x, y)` | `wcs.pixel_to_world(x, y)` | Tensors in, tensors out; dispatch on the header's axes. |
| `SkyCoord(ra, dec, frame=...)` | [`CatalogState`](reference/coords.md#torchsky.coords.CatalogState) | Batched tensors, explicit missing components. |
| `SkyCoord.transform_to(frame)` | [`transform`](reference/coords.md#torchsky.coords.transform) / `CatalogState.to_frame` | No `SkyCoord` graph; observed place needs Earth orientation. |
| `healpy.ang2pix(...)` | [`torchsky.sphere.ang2pix`](reference/sphere.md#torchsky.sphere.ang2pix) | Same arguments and defaults, tensor values. |
| `healpy.query_disc(...)` | [`torchsky.sphere.query_disc`](reference/sphere.md#torchsky.sphere.query_disc) | Takes a unit 3-vector, returns pixel indices. |
| `reproject.reproject_interp(...)` | [`torchsky.map.resample`](reference/map.md#torchsky.map.resample) | Operates on `Observation` objects and a target grid. |
| `astropy.io.fits` | [`torchfits`](io.md) | Never imported by torchsky. |
| `astropy.table.Table` | [`torchfits.table`](table.md) | Arrow/table batches; torchsky consumes tensors. |
| `astropy.cosmology` | [`torchsky.cosmology`](reference/cosmology.md) | Differentiable FLRW container, no parameter priors. |
| `astropy extinction laws` | [`torchsky.extinction`](reference/extinction.md) | Unit-explicit dust laws on tensors. |
| `specutils` Doppler/rebin | [`torchsky.spectral`](spectral.md) | Axis objects and flux-conserving rebin. |

## WCS

The structural differences are small and deliberate:

- **Header source.** Build a `WCS` from a `torchfits` header mapping. Torchsky
  never opens the file. See [the I/O boundary](io.md).
- **Pixel origin.** `pixel_to_world` and `world_to_pixel` use 0-based pixels.
  Use `pixel_to_world_values(..., origin=1)` when porting code that carries the
  FITS 1-based convention.
- **Axis dispatch.** A header with celestial, spectral, and time axes returns
  one tensor per axis; a pure spectral header returns the world array directly.
- **Distortions.** SIP, TPV, paper-IV lookup tables, and coupled `-TAB` axes are
  part of the same object; no separate `all_pix2world` bookkeeping.
- **Devices.** Move the WCS and its cached tables with `wcs.to("cuda")`; pass
  tensors on the device you want.
- **Removed aliases.** Replace `pix2world`/`world2pix` with
  `pixel_to_world`/`world_to_pixel`. Use `footprint`, `pixel_scales`, and
  `pixel_area` directly; the `calc_footprint` and `proj_plane_pixel_*` aliases
  are no longer exposed. Astropy-shaped `pixel_to_world_values`,
  `world_to_pixel_values`, `all_pix2world`, and `all_world2pix` remain.

<!-- docs: run -->
```python
import torch
from torchsky.wcs import WCS

header = {
    "NAXIS": 2,
    "NAXIS1": 256,
    "NAXIS2": 256,
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CRPIX1": 128.5,
    "CRPIX2": 128.5,
    "CRVAL1": 210.0,
    "CRVAL2": 54.0,
    "CDELT1": -1.0 / 3600.0,
    "CDELT2": 1.0 / 3600.0,
}

wcs = WCS(header)
x = torch.tensor([100.0, 128.0, 150.0], dtype=torch.float64)
y = torch.tensor([90.0, 128.0, 140.0], dtype=torch.float64)

ra, dec = wcs.pixel_to_world(x, y)
x_back, y_back = wcs.world_to_pixel(ra, dec)

assert torch.allclose(x_back, x, atol=1e-8)
assert torch.allclose(y_back, y, atol=1e-8)
print(f"round trip max error: {float((x_back - x).abs().max()):.2e} px")
```

The [WCS guide](wcs.md) covers `-TAB`, coupled tables, GWCS sampling, and
astrometric calibration. The full surface is in the
[WCS reference](reference/wcs.md).

## Clean-break utility names

- Import image mapmaking operations from `torchsky.map`; the historical
  `torchsky.map.image_plane` forwarding module is removed. Replace
  `evaluate_image_plane_aperture_uncertainty(...)` with
  `torchsky.map.aperture_uncertainty(...)` and use `kernel`, `pixfrac`, and
  `semantics` (or a `ResampleConfig`) instead of the old long keyword names.
- Import provenance from `torchsky.map.provenance`. The forwarding
  `torchsky.map.operators.provenance` module is removed.
- Import field models from `torchsky.field` or `torchsky.map.field`. The
  forwarding `torchsky.map.field.models` module is removed. Foam primitives
  remain on `torchsky.map.field`; their historical `.foam` module is removed.
- Construct `SourceCatalogScene` with `fixed_templates=` and `coefficients=`;
  the `templates`, `spectral_coefficients`, and `per_band_coefficients` aliases
  are removed.
- Use `StrainerSkyEncoderMLPField.decoder_head`, not its removed `output`
  alias. Use `CubeResult.aperture_uncertainty(...)`, not `line_uncertainty(...)`.
- A non-`None` `ForwardContext` now fails when the selected field-model method
  cannot consume it; forwarding helpers no longer silently discard context.
- Replace `transform_spectral(freq, rest_freq, convention)` with
  `convert_spectral(freq, "FREQ", target, to_unit="km/s", rest_frequency=...)`,
  where the target is `VRAD`, `VOPT`, or `VELO`.
- Pass wavelengths to `apply_extinction` as `wavelength=...` and declare
  `wavelength_unit`; `lambda_obs_angstrom` is no longer accepted.
- Use mesh filters `median` or `gaussian` with an explicit
  `mesh_filter_size` (CLI: `--mesh-filter-size` or
  `--sky-mesh-filter-size`); `median_3` and `gaussian_3` are removed.
- Calibration evidence rows must provide `exposure_id`; the `exposure`
  fallback key is removed.
- `zscale` selects display limits in high-level renderers through
  `limit_policy="zscale"`; it is not a `StretchName` or an `apply_stretch`
  mode. Pair it with a real stretch such as `stretch="linear"`.

## SkyCoord-style catalog geometry

There is no `SkyCoord` class, and that is intentional: a catalog is a batch of
tensors, not a graph of representation objects.

- [`CatalogState`](reference/coords.md#torchsky.coords.CatalogState)
  holds `lon`, `lat`, optional proper motion, parallax, radial velocity, epoch,
  frame, and a `valid` mask. Missing values stay `None`.
- `convert_celestial` and `transform` handle geometric frames; `CatalogState.to_frame`
  propagates proper motion through rigid rotations.
- Observed place (`ALTAZ`, `ITRS`, `GCRS`, `CIRS`, `TETE`) requires an explicit
  `EarthOrientation` (DUT1, optional polar motion) and site. IERS values are
  never inferred.
- `pmsafe` distance override, unit-bearing `Quantity` objects, and automatic
  epoch inference do not exist. Supply tensors in the documented units.

<!-- docs: run -->
```python
import torch
from torchsky.coords import CatalogState, convert_celestial

lon = torch.tensor([266.405, 83.822], dtype=torch.float64)
lat = torch.tensor([-28.936, -5.391], dtype=torch.float64)

gal_lon, gal_lat = convert_celestial(
    lon, lat, from_frame="ICRS", to_frame="galactic"
)
state = CatalogState(lon=lon, lat=lat, frame="ICRS").to_frame("galactic")

assert torch.allclose(state.lon, gal_lon, atol=1e-9)
assert torch.allclose(state.lat, gal_lat, atol=1e-9)
print(f"galactic longitude: {state.lon.tolist()}")
```

Frame names follow FITS/AST conventions (`ICRS`, `GALACTIC`, `FK5`, `FK4`,
`ECLIPTIC`, `GALACTOCENTRIC`, `ALTAZ`, …). Conversion between Cartesian
Galactocentric coordinates and ICRS has its own helpers; it is not a
`convert_celestial` direction. See [catalog coordinates](coordinates.md).

## HEALPix and healpy

`torchsky.sphere` reuses healpy's naming and argument order, including the
`nest` and `lonlat` switches, so most calls port by changing the import:

```python
import healpy as hp
pix = hp.ang2pix(64, lon_deg, lat_deg, lonlat=True)

import torchsky.sphere as sphere
pix = sphere.ang2pix(64, lon_deg, lat_deg, lonlat=True)
```

Differences that matter:

- Values are `torch.Tensor`; pass tensors, not lists, in hot paths.
- `query_disc` and related queries take a unit 3-vector, not `(lon, lat)`.
- `UNSEEN` exists as a constant, but map file reading/writing belongs to
  `torchfits`, not the sphere package.
- `nest=False` (RING) remains the default everywhere.
- The healpy-shaped namespace is available as `torchsky.sphere.compat`; new
  code can use `torchsky.sphere` directly. See the
  [compatibility reference](reference/sphere-compat.md).
- SHTs, smoothing, and spectra follow torchsky's own backend contract; review
  [sphere.md](sphere.md) before assuming healpy's precision or threading model.

### 0.4.0 sphere compatibility break

Version 0.4.0 intentionally removes the duplicate `torchsky.sphere.healpy`
re-export. Replace that import with `torchsky.sphere.compat`. It also removes
the process-global strict-mode setters/context manager and the ignored `strict`
query keywords. Geometry queries already use only torchsky-native kernels.
`torchsky.sphere.compat.query_ellipse` no longer accepts `backend="auto"` or
`backend="torch"` because both selected the same native implementation.
The same release removes ignored `verbose` keywords from `synalm`, `synfast`,
`smoothmap`, `smoothing`, `remove_monopole`, and `remove_dipole`; removes
`chunk_size` from `query_strip`; and removes `fact` from `query_polygon_vec`.
Delete those keywords at call sites. `query_circle(fact=...)` is retained and
now changes inclusive-overlap supersampling through the native `query_disc`
kernel.

Exact-region APIs now accept only `backend="native"` or
`backend="spherical-geometry"`; replace `backend="auto"` with `"native"`.
Use `NativeApproximateSphericalRegion` instead of the removed
`NativeExactSphericalRegion` name. Its `contains`, `intersects`, `union`,
`intersection`, and `difference` methods no longer accept ignored boundary or
cap discretization keywords. Call `MOC.full()` without an order argument.

`SparseHealpixMap` is the sole torchsky sparse-map class name. Replace
`HealSparseMap` and `SparseMap`, while retaining `from_healsparse()` and
`to_healsparse()` at explicit HealSparse interchange boundaries. Replace
`get_values_pix` with `get_values`, `lookup_With_valid` with
`lookup_with_valid`, `generate_healpix_map` with `to_dense`, and `write(...,
clobber=...)` with `write_fits(..., overwrite=...)`.

Core HEALPix code now uses `max_pixel_radius`, `ring2nest`, `nest2ring`,
`neighbors`, and `query_circle`; peer spellings such as `max_pixrad` and
`get_all_neighbours` remain only in `torchsky.sphere.compat`. Geometry classes
use their full names: `SphericalCap`, `SphericalBox`, `SphericalEllipse`,
`SphericalPolygon`, and `SphericalMultiPolygon`.

<!-- docs: run -->
```python
import torch
import torchsky.sphere as sphere

nside = 64
lon = torch.tensor([10.0, 120.0, 250.0], dtype=torch.float64)
lat = torch.tensor([5.0, -30.0, 60.0], dtype=torch.float64)

pix = sphere.ang2pix(nside, lon, lat, lonlat=True)
lon_back, lat_back = sphere.pix2ang(nside, pix, lonlat=True)
# pix2ang returns the pixel centre, so round trip through ang2pix instead of
# comparing the centre to the original off-centre coordinate.
assert torch.equal(sphere.ang2pix(nside, lon_back, lat_back, lonlat=True), pix)

vec = sphere.lonlat_to_unit_xyz(lon[0], lat[0])
inside = sphere.query_disc(nside, vec, radius=0.05)
assert bool((inside == pix[0]).any())

print(f"pixel indices: {pix.tolist()}")
```

### 0.4.0 field-bandpass API rename

Import field-observation bandpass helpers from
`torchsky.filter_profiles.field_bandpass` instead of
`torchsky.filter_profiles.inr_bandpass`. Remove source-alias imports and replace
`resolve_inr_bandpass` with
`resolve_field_bandpass`. Version 0.4.0 does not provide deprecated aliases.

The field CLI now also uses `field_*` parser destinations and serialized
execution-plan keys. Earlier field map plans containing `model`,
`model_kwargs`, `steps`, `learning_rate`, `patch_size`, sampling, or render
keys are not migrated or accepted. Regenerate those plans with 0.4.0 before
replay.

### Other 0.4.0 clean breaks

- Replace `torchsky info` and `torchsky cat` with the corresponding Torchfits
  inspection/table commands; Torchsky no longer forwards FITS-owned CLI work.
- Replace `--sky`, `--inr-model`, `--per-band`, and `--inr-*` field controls
  with their documented `--background-mode`, `--field-model`, `--per-filter`,
  and `--field-*` forms. Removed spellings are rejected rather than translated.
- Import and annotate `SceneRepresentation` directly; the duplicate `Scene`
  type alias is gone.
- Pass `atol=` to detector-astrometry inverse methods; the `tol=` alias is gone.

## Reprojection and coaddition

`reproject` maps arrays between WCS grids. `torchsky.map` goes further: it takes
observations with explicit weight, mask, DQ, and spectral metadata and returns a
[`MapResult`](reference/map.md#torchsky.map.MapResult) that carries data,
weight, header, and provenance.

- Use [`resample`](reference/map.md#torchsky.map.resample) to project
  observations onto a target grid without stacking.
- Use [`coadd`](reference/map.md#torchsky.map.coadd) to stack with a declared
  kernel, pixfrac, stacking method, and value semantics.
- Pass a reference FITS WCS/shape or a planned target; the target is never
  inferred silently when several observations disagree.
- Flux versus surface brightness is an explicit `ResampleConfig.semantics`
  choice; there is no implicit area scaling.

See [mapmaking](map.md) for the value-semantics contract and
[HEALPix maps](healpix-maps.md) for sphere-to-grid direction.

## FITS, tables, cosmology, and spectra

- **FITS.** Replace `astropy.io.fits` with `torchfits` reads and writes;
  `torchsky` consumes the resulting tensor and header. Never import
  `astropy.io.fits` in torchsky code. See [I/O boundary](io.md).
- **Tables.** Replace `astropy.table.Table` with `torchfits.table` Arrow or
  tensor reads; `torchsky.catalogs` accepts those batches. See
  [tables](table.md).
- **Cosmology.** Replace `astropy.cosmology` objects with
  [`FLRW`](reference/cosmology.md#torchsky.cosmology.FLRW) and the module
  functions; parameters are plain tensors, so they can be differentiated.
- **Dust.** Replace extinction-law calls with
  [`torchsky.extinction`](reference/extinction.md); wavelength units are
  explicit.
- **Spectra.** Replace `specutils` rebinning and Doppler shifts with
  [`SpectralAxis`](reference/spectral.md#torchsky.spectral.SpectralAxis),
  `flux_conserving_rebin`, and `torchsky.redshift`.

## What does not carry over

- No `SkyCoord` subclass or representation graph.
- No implicit units, no `Quantity` arithmetic, no automatic epoch/EOP values.
- No FITS or table I/O inside `torchsky` (including healpy's `read_map`).
- No Astropy `WCSAxes` plotting, `NDData`, or `Cutout2D` equivalents; use
  `torchfits` cutouts and `torchsky.viz`.
- No guaranteed bit-for-bit parity with healpy/reproject/Astropy outside the
  documented contracts; parity evidence lives in `torchsky-harness`.
- No `astropy.io.fits`/`fitsio` imports anywhere in the compute path.

## Porting checklist

1. Replace file reads with `torchfits` and pass header dicts onward.
2. Convert NumPy arrays to tensors with an explicit dtype and device.
3. Decide the pixel origin once (0-based torchsky vs 1-based FITS values).
4. Name the frame and epoch explicitly; supply EOP values for observed place.
5. Declare map value semantics, weights, and masks instead of relying on
   defaults inherited from another package.
6. Run the equivalent torchsky example and compare against your previous
   pipeline on the same inputs before deleting the old path.
