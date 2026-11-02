# Migrating to 0.5

Release notes for moving from `torchfits` 0.3.1 or an unpublished `0.4.0`
build to `torchsky` 0.5. Coming from Astropy, healpy, or reproject instead? See
[Coming from Astropy, healpy, and reproject](coming-from.md).


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
| `cosmology` | `n_steps=` (had no effect with the Gauss-Legendre quadrature) | Removed; drop the argument (the quadrature is fixed) |

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
