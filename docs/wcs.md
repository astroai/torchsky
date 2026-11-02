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
tables. Square 2-D tables use a batched unique-cell Gauss–Newton search.
Overdetermined tables
(`world_n_dim > pixel_n_dim`) use Gauss–Newton / least squares and return
NaN off the interpolated manifold. Underdetermined tables
(`world_n_dim < pixel_n_dim`) return a nearest-seed Gauss–Newton particular
solution (minimum-norm step); that inverse is not unique.

Celestial projections include Paper II zenithal, cylindrical, conic, polyconic
(`BON`, `PCO`), HEALPix (`HPX`, `XPH`), quadcube (`TSC`, `CSC`, `QSC`), and
`NCP` (implemented as slant `SIN`). Fused C++ covers `PCO`, `XPH`, `TSC`,
`CSC`, and `QSC`. `WCS.to("cuda")` moves cached PV and rotation tensors with
the pixels. `WCS.to("mps")` casts cached parameters to float32 (MPS has no
float64, so precision is lower than CPU float64). CUDA WCS is
float64. CPU fused WCS is float64. Public `pixel_to_world` /
`world_to_pixel` stay on the pixel device; fused C++ runs only when both
pixels and cached parameters are on CPU.

Identity TPV uses `PV1_1=1` and `PV2_1=1` (axis 2 is swapped: `PV2_1` is the
linear `v` term). A header with `PV2_2=1` and no `PV2_1` is rank-deficient;
inverse is then a particular solution, not a unique preimage.
The fused kernels are internal: `WCS.pixel_to_world` and `WCS.world_to_pixel`
run through the same dispatcher, return NaNs for invalid projection domains or
singular inverses instead of raising, and are the supported entry points.


Air wavelengths (`AWAV`, `FREQ-GRA`) use the WCSLIB Edlén form so spectral
WCS matches astropy.wcs / WCSLIB.

Small- and large-input performance is evaluated in the harness. Observed place
is ATen on the catalog device (CUDA
float64, MPS float32); there is no fused C++ observed kernel.

Catalog geometry is `CatalogState`, not `SkyCoord`. See [coordinates.md](coordinates.md).
GWCS is sampled at calibration time onto `RegularGridTransform2D`; runtime
never imports `gwcs`. `FrozenDetectorEpochTables.evaluate` / `.inverse` are
device-native table ops with a Newton residual bound. Optional
`torchsky.wcs.gwcs_live.freeze_live_gwcs` (the `gwcs` extra) evaluates a live
graph then freezes; it is not a core import.
Frozen tables preserve either consistent sampled Jacobian sign, including
reflected detector frames; exposure/residual corrections preserve that parity.
The determinant floor bounds absolute magnitude, and sampled checks do not
prove global injectivity between validation points.

CSC CPU uses the WCSLIB float32 Chan–O'Neill polynomial (scalar C Horner).
CUDA/MPS CSC without autograd host-copies to that CPU kernel until a fused CUDA
Chan–O'Neill kernel exists. Autograd stays on ATen. Comparative throughput and
quantitative parity evidence are maintained in the harness. `FREQ-GRA` is
validated against a relative-error criterion; numerical acceptance evidence
is recorded with the corresponding harness evaluation.

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
corrected = wcs.with_affine(fit)
```

## API Surface

Main class:

- `WCS(header, key="", tabular_axes=None, tables=None)`
- `pixel_to_world(...)` / `pix2world(...)`, `world_to_pixel(...)` / `world2pix(...)`
- `all_pix2world(...)`, `all_world2pix(...)`
- `mixed_pixel_world(...)`
- `slice(...)`, `dropaxis(...)`, `sub(...)`, `celestial` — `slice` accepts
  negative steps (Python `range` length / CRPIX-CD update). Dropped coupled
  axes bake at CRPIX.
- `to_header(relax=True, include_naxis=...)`, `to_head_string(relax=True)`
- `footprint(shape=None, center=True, origin=0)`, `calc_footprint(...)`, `in_image(...)`
- `pixel_scales()`, `pixel_area()`, `position_angle()`, `pixel_area_map(shape=None, unit="deg2")`
- `with_affine(matrix_or_fit, offset=None)`

Differentiable geometry:

- `jacobian(coords, y=None, *, inverse=False, origin=0, create_graph=False)`
- `pixel_to_world_differentiable(...)`, `world_to_pixel_differentiable(...)`
- `pixel_to_world_jacobian(...)`, `world_to_pixel_jacobian(...)`

Spectral, IFU, GWCS, and temporal WCS:

- `spectral_to_frequency(...)`, `frequency_to_spectral(...)`
- `WCS.temporal_wcs(...)`, `TemporalWCS`
- `IFUSliceWCS.from_slices(...)`, `IFUSliceWCS.from_callable(...)`
- `sample_gwcs(...)`, `sample_gwcs_epochs(...)`, `FrozenDetectorEpochTables`
- `from_tabular(...)`, `from_coupled_tabular(...)`, `TabularAxis`, `TabularLookupWCS`, `CoupledTabularLookupWCS`

Frames, DCR, ephemeris, and alignment:

- `transform(...)`, `convert_celestial(...)`
- `icrs_to_galactic(...)`, `galactic_to_icrs(...)`, `icrs_to_ecliptic(...)`, `ecliptic_to_icrs(...)`, `precess_fk5(...)`
- `CatalogState` batched tensors; see [coordinates.md](coordinates.md)
- `EarthOrientation`, `Site`, `observed_altaz(...)`
- `dcr_offset(...)`, `apply_dcr(...)`, `apply_ephemeris_offset(...)`, `apply_refraction(...)`, `apply_stellar_aberration(...)`
- `propagate_proper_motion(...)`, `propagate_proper_motion_with_jacobian(...)`
- `propagate_space_motion(...)`, `propagate_space_motion_with_jacobian(...)`
- `TangentPlaneFrame`, `fit_affine_2d(...)`, `fit_sky_affine(...)`

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

- [WCS transform example](published-examples/example_wcs_transform.py)

## Compatibility And Validation

Core tests retain bounded transform, round-trip, gradient, and device
contracts. Exhaustive Astropy/WCSLIB parity, broad fixture replay, and WCS
performance measurements are maintained in
[torchsky-harness](https://github.com/astroai/torchsky-harness).

See [compatibility.md](compatibility.md) for the ownership boundary and
[benchmarks.md](evaluation/benchmarks.md) for the harness workflow.
