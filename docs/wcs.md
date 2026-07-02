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
the table grid and intentionally does not guess a general inverse.

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
- `slice(...)`, `dropaxis(...)`, `sub(...)`, `celestial`
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

- `icrs_to_galactic(...)`, `galactic_to_icrs(...)`, `precess_fk5(...)`
- `TangentPlaneFrame`
- `fit_affine_2d(...)`, `fit_sky_affine(...)`

## Examples

- [WCS transform example](../examples/example_wcs_transform.py)

## Compatibility And Validation

WCS parity is covered against Astropy/WCSLIB fixture behavior in the WCS test
suite and in focused benchmarks such as `benchmarks/bench_wcs.py`,
`benchmarks/bench_wcs_suite.py`, `benchmarks/bench_wcs_small_n_paths.py`, and
`benchmarks/bench_wcs_spectral_temporal.py`.

See [compatibility.md](compatibility.md) for validation policy and
[benchmarks.md](benchmarks.md) for commands.
