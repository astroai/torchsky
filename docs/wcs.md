# `torchsky.wcs`

`torchsky.wcs` provides torch-native FITS-WCS transforms, metadata helpers,
spectral and temporal coordinate support, frame conversion utilities, and
alignment helpers.

## What This Namespace Is For

Use `torchsky.wcs.WCS` when you need pixel/world transforms that stay in
PyTorch-oriented workflows. Astropy/WCSLIB are reference comparators in tests
and benchmarks, not runtime fallbacks for public tensor paths.

## Core Concepts

- Coordinates use Python 0-based pixel convention; celestial world values are in
  degrees unless noted.
- WCS objects are built from FITS-like header mappings, optionally with decoded
  lookup-table data for FITS `-TAB` axes. File decoding belongs to `torchfits`.
- Tensor paths run on CPU, CUDA, or MPS where the underlying operations support
  the device.
- Differentiable helpers are available for field-model training and geometry
  optimization.
- Compatibility helpers expose common Astropy-style method names and metadata.

## Recipes

### Pixel To World And Back

```python
import torchfits
from torchsky.wcs import WCS

_, header = torchfits.read("science.fits", hdu="auto", return_header=True)
wcs = WCS(header)
ra, dec = wcs.pixel_to_world(x, y)
x_back, y_back = wcs.world_to_pixel(ra, dec)
```

### Use Alternate FITS-WCS Keys

```python
from torchsky.wcs import WCS

wcs_a = WCS(header, key="A")
```

### Keep A Compiled Tensor Path

```python
wcs = WCS(header).compile()
world = wcs.pixel_to_world(pixels)  # pixels.shape == (..., 2)
```

### Load `-TAB` Axes

```python
import torchfits
from torchsky.wcs import WCS

with torchfits.open("cube_with_wave_tab.fits") as hdul:
    wcs = WCS(hdul[0].header, tables=hdul)
```

One-dimensional lookup axes use `PSi_0` for table extension name, `PSi_1` for
coordinate column, optional `PSi_2` for the indexing vector, `PVi_1` for
`EXTVER`, and `PVi_2` for `EXTLEVEL`. Separable multidimensional coordinate
arrays can use `PVi_3` to select the table dimension for each WCS axis; arrays
that vary along more than that selected dimension fail explicitly because they
need a true coupled interpolation substrate. Missing required tables or columns
fail instead of falling back to a linear axis. Coordinate and indexing columns
are resolved by exact name first, then by a unique case-insensitive match;
ambiguous case variants fail explicitly.

For genuinely coupled multidimensional table arrays, use
`coupled_tabular_wcs_from_fits(...)` to build a `CoupledTabularLookupWCS`
forward interpolator. It performs torch-native multilinear interpolation over
the table grid. Inverse exists for square, overdetermined, and underdetermined
tables. Square 2-D tables use the unique-cell search. Overdetermined tables
(`world_n_dim > pixel_n_dim`) use Gauss–Newton / least squares and return
NaN off the interpolated manifold. Underdetermined tables
(`world_n_dim < pixel_n_dim`) return a nearest-seed Gauss–Newton particular
solution (minimum-norm step); that inverse is not unique.

Celestial projections include Paper II zenithal, cylindrical, conic, polyconic
(`BON`, `PCO`), HEALPix (`HPX`, `XPH`), quadcube (`TSC`, `CSC`, `QSC`), and
`NCP` (implemented as slant `SIN`). Fused C++ covers `PCO`, `XPH`, and `TSC`;
`CSC`/`QSC` stay on the Python path. `WCS.to("cuda")` moves cached PV and
rotation tensors with the pixels. `WCS.to("mps")` is rejected (MPS has no
float64). CUDA WCS is float64. CPU fused WCS is float64. MPS two-arg TAN
stays float32 without moving WCS parameters onto MPS.

Small-N overhead versus Astropy (sub-3 ms) is acceptable; large-N must stay
first on CPU and CUDA. Observed place is ATen on the catalog device (CUDA
float64, MPS float32); there is no fused C++ observed kernel.

Catalog geometry is `CatalogState`, not `SkyCoord`. See [coordinates.md](coordinates.md).
GWCS is sampled at calibration time onto `RegularGridTransform2D`; runtime
never imports `gwcs`. `FrozenDetectorEpochTables.evaluate` / `.inverse` are
device-native table ops with a Newton residual bound. Live GWCS graphs stay
out.

### Spectral And Temporal Axes

Spectral WCS covers linear spectral axes, `-LOG`, reciprocal compound axes such
as `WAVE-F2W` / `FREQ-W2F`, velocity compound axes such as `WAVE-V2W`, and
one-axis grism algorithms `GRI` / `GRA` for the supported spectral types.

Temporal helpers provide UTC/TAI/TT/GPS scale conversions and a fast local
Keplerian light-time correction for topocentric barycentric/heliocentric use
cases. Full ephemeris-grade apparent-place and Earth-orientation corrections are
still outside the runtime contract.

### Fit A Pixel-Affine Correction

```python
from torchsky.wcs import fit_affine_2d

fit = fit_affine_2d(source_pixels, target_pixels, geometry="rscale")
corrected = wcs.with_affine_fit_correction(fit)
```

## API Surface

Main class:

- `WCS(header, key="", tabular_axes=None, tables=None)`
- `pixel_to_world(...)`, `world_to_pixel(...)`
- `all_pix2world(...)`, `all_world2pix(...)`
- `mixed_pixel_world(...)`
- `slice(...)`, `dropaxis(...)`, `sub(...)`, `celestial` — `slice` accepts
  negative steps (Python `range` length / CRPIX-CD update). Dropped coupled
  axes bake at CRPIX.
- `to_header()`, `footprint()`, `in_image(...)`

Differentiable geometry:

- `pixel_to_world_differentiable(...)`
- `world_to_pixel_differentiable(...)`
- `pixel_to_world_jacobian(...)`
- `world_to_pixel_jacobian(...)`

Spectral/temporal WCS:

- `spectral_to_frequency(...)`, `frequency_to_spectral(...)`
- `WCS.temporal_wcs(...)`
- `TemporalWCS` from `torchsky.wcs` or `torchsky.wcs.temporal`
- `convert_time_scale(...)`, `iso_to_mjd(...)` from `torchsky.wcs.temporal`
- `TabularAxis`, `TabularLookupWCS`, `CoupledTabularLookupWCS`
- `tabular_axes_from_fits(...)`, `coupled_tabular_wcs_from_fits(...)`

Frames and alignment:

- `icrs_to_galactic(...)`, `galactic_to_icrs(...)`, `icrs_to_ecliptic(...)`, `ecliptic_to_icrs(...)`, `precess_fk5(...)`
- `CatalogState` batched tensors; see [coordinates.md](coordinates.md)
- `EarthOrientation`, `Site`, `observed_altaz(...)` (explicit DUT1; optional polar motion, `pn='iau2006'` IAU 2000A, opt-in aberration/refraction)
- `transform(...)` tensor façade (`ALTAZ` / `ITRS` / `GCRS` / `CIRS` / `TETE` / geometric frames); not a SkyCoord
- `icrs_to_itrs_xyz(...)`, `itrs_xyz_to_lonlat_deg(...)`
- `propagate_proper_motion(...)` for tensor-native angular space motion
- `propagate_proper_motion_with_jacobian(...)` for local uncertainty transport
  when distance/RV are unavailable
- `propagate_space_motion(...)` for complete parallax/RV-aware ICRS motion
- `propagate_space_motion_with_jacobian(...)` for batched local uncertainty
  transport
- `TangentPlaneFrame`
- `fit_affine_2d(...)`, `fit_sky_affine(...)`

`propagate_space_motion(...)` propagates all six ICRS observables with the
ERFA/SOFA changing-light-time and special-relativistic model. It requires a
finite radial velocity and positive parallax, returns the propagated proper
motions, parallax, and radial velocity as well as position, and never applies
ERFA `pmsafe`'s implicit distance override. The tensor path broadcasts and is
differentiable.

`propagate_space_motion_with_jacobian(...)` returns the propagated state and a
batched 6x6 Jacobian in local catalogue coordinates. Its positional axes are
east/north tangent-plane offsets in mas at the source and destination, followed
by cosine-weighted longitude proper motion, latitude proper motion, parallax,
and radial velocity. This avoids raw-longitude discontinuities at 0/360 degrees
and lets callers propagate a covariance with `J @ C @ J.mT`. Catalogue-specific
subspaces, such as Gaia's correlated five astrometric parameters without a
joint radial-velocity covariance, can select the corresponding rows/columns
without inventing missing correlations.

`propagate_proper_motion_with_jacobian(...)` returns propagated position and a
batched 2x4 Jacobian from local east/north position and proper-motion offsets
to the destination tangent plane. It follows the same unit-distance angular
model as `propagate_proper_motion(...)`: no distance or radial velocity is
imputed, and perspective acceleration is therefore outside its contract.

## Examples

- [WCS transform example](../examples/example_wcs_transform.py)

## Compatibility And Validation

WCS parity is covered against Astropy/WCSLIB fixture behavior in the WCS test
suite and in focused benchmarks such as `benchmarks/bench_wcs.py`,
`benchmarks/bench_wcs_suite.py`, `benchmarks/bench_wcs_small_n_paths.py`, and
`benchmarks/bench_wcs_spectral_temporal.py`.

See [compatibility.md](compatibility.md) for validation policy and
[benchmarks.md](benchmarks.md) for commands.
