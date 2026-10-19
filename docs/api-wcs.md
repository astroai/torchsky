# WCS API reference

`torchsky.wcs` implements FITS WCS transformations as torch tensor operations.
The [WCS guide](wcs.md) covers concepts and recipes; this page is the reference
for the namespaces and the promoted names.

Everything here accepts and returns `torch.Tensor` (or plain Python scalars for
frame helpers) and runs on CPU, CUDA, and Apple Silicon MPS.

---

## `torchsky.wcs.WCS`

Build a transform from a `torchfits` header mapping:

```python
from torchsky.wcs import WCS

wcs = WCS(header)
```

| Method | Contract |
|---|---|
| `pixel_to_world(*pixel_arrays)` | Pixel coordinates → world coordinates, dispatching on the axes present (celestial, spectral, time, tabular). |
| `world_to_pixel(*world_arrays)` | Inverse of `pixel_to_world` with the same axis dispatch. |
| `pixel_to_world_values(*pixel_arrays, origin=0)` | Lower-level tuple form; `origin` selects 0- or 1-based pixels. |
| `world_to_pixel_values(*world_arrays, origin=0)` | Lower-level tuple form for the inverse direction. |
| `to(device)` | Move the WCS (and any baked tables) to a device. |

Celestial transforms support SIP, TPV, paper-IV lookup tables, and coupled
`-TAB` axes. Spectral and time axes are transforms too, not metadata.

---

## Distortion and tabular transforms

| Name | Role |
|---|---|
| `DistortionLookupTable` | FITS WCS Paper IV 2D lookup table (`CPDIS` / `DET2IM`). |
| `TabularAxis` | One-dimensional table-driven coordinate transform. |
| `TabularLookupWCS` | Separable multi-axis table-driven WCS transform. |
| `CoupledTabularLookupWCS` | Coupled multidimensional table lookup with multilinear interpolation. |
| `from_tabular(header)` | Build explicit `WCS(..., tabular_axes=...)` axes from FITS `-TAB` metadata. |
| `from_coupled_tabular(header)` | Build a forward coupled table-lookup WCS. |
| `tabular_axes_from_fits(header)` | NumPy-free axis construction helper. |
| `coupled_tabular_wcs_from_fits(header)` | Coupled-WCS construction helper. |
| `RegularGridTransform2D` | Frozen regular-grid detector transform with consistent sampled parity. |
| `DetectorToTangentTransform` | Compose a fixed table, exposure affine, and optional detector residual. |
| `DetectorToSkyTransform` | Forward-only detector-to-sky composition with a frozen attitude. |
| `FrozenDetectorEpochTables` | Time-dependent detector tables as frozen samples, not a live GWCS graph. |

GWCS sampling is a build-time step: `sample_gwcs`, `sample_gwcs_epochs`,
`sample_gwcs_detector_epochs`, and `sample_gwcs_detector_to_tangent` refine a
detector table until independent points satisfy the configured milliarcsecond
gates, then runtime never imports `gwcs`.

Validation helpers: `validate_sampled_transform`, `SampledTransformValidation`,
`GWCSGridSamplingResult`.

---

## Celestial frames and rotations

| Name | Role |
|---|---|
| `galactic_to_icrs`, `icrs_to_galactic` | Galactic ↔ ICRS. |
| `ecliptic_to_icrs`, `icrs_to_ecliptic` | Mean ecliptic of J2000 ↔ ICRS. |
| `ecliptic_of_date_to_icrs`, `icrs_to_ecliptic_of_date` | Ecliptic of date via IAU 2006 NPB. |
| `precess_fk5` | Precess FK5 equatorial coordinates between Julian equinoxes. |
| `transform(lon, lat, *, from_frame, to_frame)` | Generic rotation/observation dispatch on `(lon, lat)` tensors. |
| `canonical_celestial_frame(name)` | Normalize FITS/AST celestial frame names. |
| `convert_celestial(lon, lat, from_frame, to_frame)` | Convert between supported celestial frames. |
| `galactocentric_to_icrs`, `icrs_to_galactocentric` | Heliocentric ICRS ↔ Galactocentric Cartesian. |

---

## Motion, time, and observation effects

| Name | Role |
|---|---|
| `propagate_proper_motion`, `propagate_proper_motion_with_jacobian` | Angular motion with analytic Jacobian. |
| `propagate_space_motion`, `propagate_space_motion_with_jacobian` | Full barycentric astrometric state with 6D Jacobian. |
| `apply_ephemeris_offset` | Comoving-frame shift for moving targets. |
| `apply_stellar_aberration` | Classical stellar aberration. |
| `apply_refraction`, `apply_dcr`, `dcr_offset` | Refraction and differential chromatic refraction on sky coordinates. |
| `observed_altaz`, `icrs_to_itrs_xyz`, `itrs_xyz_to_lonlat_deg` | Observation geometry with explicit Earth orientation (`EarthOrientation`, `Site`). |
| `heliocentric_rv_to_lsr`, `lsr_line_of_sight_kms` | LSR corrections with explicit solar UVW. |

`EarthOrientation` requires `dut1_s`; nothing guesses it.

---

## Spectral and time axes

| Name | Role |
|---|---|
| `canonical_spectral_type`, `convert_spectral` | Canonical FITS spectral coordinate types and conversions. |
| `spectral_to_frequency`, `frequency_to_spectral` | Frequency-based conversions. |
| `air_wavelength_to_vacuum`, `vacuum_wavelength_to_air` | Air/vacuum wavelength conversion. |
| `rest_frequency_from_header` | `RESTFRQ`/`FREQ`/`RESTWAV` resolution. |
| `transform_spectral` | Compatibility helper returning km/s. |
| `IFUSliceWCS` | Piecewise-disjoint detector → 3D `(ra_deg, dec_deg, wavelength)` IFU slice WCS. |
| `CoupledTabularLookupWCS`, `TabularLookupWCS` | Shared with the tabular section above. |

---

## Affine and residual corrections

| Name | Role |
|---|---|
| `fit_affine_2d`, `AffineFitResult` | Weighted 2D affine correction between point sets. |
| `fit_sky_affine`, `SkyAffineFitResult` | Local tangent-plane affine fit. |
| `TangentPlaneFrame`, `TangentPlaneAffine` | Gnomonic frame and learnable affine correction. |
| `AffineFreeResidualGrid2D` | Smooth detector-coordinate residual with constant/affine modes removed. |

See also: [coordinates](coordinates.md), [astrometric calibration](astrometric_calibration.md),
and the [WCS guide](wcs.md) for distortion and `-TAB` walkthroughs.
