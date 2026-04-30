# torchsky API Reference

## Quick Paths

For migrating from **healpy**, **Astropy**, **fitsio**, and related stacks, see **[Ecosystem compatibility](compatibility.md)**.

| Goal | Entry point |
|---|---|
| Read image/table | `torchsky.read(path, hdu=..., return_header=True)` |
| Read image only | `torchsky.read_image(path, hdu=0, mmap=True)` |
| Read table only | `torchsky.read_table(path, hdu=1, columns=[...])` |
| Cutout from image | `torchsky.read_subset(path, hdu, x1, y1, x2, y2)` |
| Multi-HDU images | `torchsky.read_hdus(path, hdus=[0,1,2])` |
| Repeated cutouts | `torchsky.open_subset_reader(path, hdu)` |
| Stream table | `torchsky.stream_table(path, chunk_rows=10000)` |
| Write | `torchsky.write(path, data, header=None, overwrite=False)` |
| WCS | `torchsky.get_wcs(path, hdu="auto")` |
| Header only | `torchsky.get_header(path, hdu=0)` |
| Multi-HDU handle | `with torchsky.open(path) as hdul: ...` |
| Table pushdown SQL | `where=` parameter in `read/stream_table` |
| Arrow / Polars | `torchsky.table.to_polars_lazy(...)`, `torchsky.table.to_duckdb(...)` |

---

## Core I/O

### `read(path, hdu=0, *, mode='auto', device='cpu', mmap='auto', fp16=False, bf16=False, columns=None, start_row=1, num_rows=-1, where=None, cache_capacity=10, handle_cache_capacity=16, fast_header=True, return_header=False)`

Unified reader. Auto-detects image or table based on HDU type.

- `hdu`: integer index, EXTNAME string, or `"auto"` / `None` (first data HDU).
- `mode`: `"auto"` | `"image"` | `"table"`.
- `mmap`: `True` | `False` | `"auto"` (heuristic based on file size).
- `fp16` / `bf16`: lossy downcasting. Avoid for astrometry/photometry.
- `where`: SQL-style predicate pushdown for tables (see [Predicate Pushdown](#predicate-pushdown)).
- `return_header=True`: returns `(data, Header)` instead of just `data`.

```python
data, hdr = torchsky.read("image.fits", device="cuda", return_header=True)
table = torchsky.read("cat.fits", hdu=1, columns=["RA", "DEC"], where="MAG_G < 20")
```

### `read_image(path, hdu=0, *, device='cpu', mmap=True, handle_cache=True, fp16=False, bf16=False, raw_scale=False, return_header=False)`

Low-level image reader. Bypasses auto-detection.

- `raw_scale=True`: return raw integer values without BSCALE/BZERO.

### `read_table(path, hdu=1, *, columns=None, start_row=1, num_rows=-1, device='cpu', mmap=True, return_header=False)`

Table-focused wrapper around `read()`.

### `read_hdus(path, hdus, *, device='cpu', mmap=True, return_header=False)`

Read multiple image HDUs in one open.

```python
sci, wht, msk = torchsky.read_hdus("mef.fits", hdus=["SCI", "WHT", "MASK"])
```

### `read_subset(path, hdu, x1, y1, x2, y2, handle_cache_capacity=16)`

Read a rectangular cutout without loading the full image. Coordinates are 0-based.

```python
stamp = torchsky.read_subset("large.fits", hdu=0, x1=1000, y1=1000, x2=1100, y2=1100)
```

Persistent cutout reader. Keeps the FITS handle open for repeated reads.

```python
with torchsky.open_subset_reader("mosaic.fits", hdu=0) as r:
    stamp1 = r(0, 0, 256, 256)
    stamp2 = r(256, 256, 512, 512)
```

### `stream_table(file_path, hdu=1, *, columns=None, start_row=1, num_rows=-1, chunk_rows=10000, mmap=False, max_chunks=None)`

Iterate over a table in row chunks. Yields `dict[str, Tensor | list]`.

### `read_large_table(file_path, hdu=1, *, max_memory_mb=100, streaming=False, return_iterator=False)`

Memory-aware chunked table read.

---

### `write(path, data, header=None, overwrite=False, compress=False)`

Write to FITS.

- `data`: `Tensor` (image), `dict[str, Tensor]` (table), or `HDUList` (multi-HDU).
- `compress`: `False` | `True` (Rice) | `"RICE"` | `"GZIP"` | `"HCOMPRESS"`.

```python
torchsky.write("out.fits", torch.randn(512, 512), header={"OBJECT": "M31"}, overwrite=True)
torchsky.write("cat.fits", {"RA": ra_tensor, "DEC": dec_tensor}, overwrite=True)
```

### HDU manipulation

```python
torchsky.insert_hdu(path, data, index=1, header=None, compress=False)
torchsky.replace_hdu(path, hdu, data, header=None, compress=False)
torchsky.delete_hdu(path, hdu)
```

### Checksums

```python
torchsky.write_checksums(path, hdu=0)   # compute & write DATASUM/CHECKSUM
result = torchsky.verify_checksums(path, hdu=0)
# result: {"datastatus": 1, "hdustatus": 1, "ok": True}
# status values: 1 = ok, 0 = absent, -1 = bad
```

---

### `open(path, mode='r') -> HDUList`

Multi-HDU handle with context management.

```python
with torchsky.open("mef.fits") as hdul:
    hdu = hdul[0]          # by index
    hdu = hdul["SCI"]      # by EXTNAME
    data = hdu.data        # Tensor
    hdr  = hdu.header      # Header
    tbl  = hdul[1]         # TableHDURef (lazy)
```

### `get_header(path, hdu=0) -> Header`

Read header only. `hdu` accepts index, EXTNAME, `"auto"`, or `None`.

### `get_wcs(path, hdu="auto", device=None) -> WCS`

Construct a WCS from a FITS file. See [WCS](#wcs).

---

## HDU Types

### `TensorHDU`

Image HDU. Attributes: `.data` (Tensor, lazy), `.header`, `.wcs`.

### `TableHDU`

In-memory table. Attribute: `.data` (dict[str, Tensor | list]).

Methods: `.filter(condition)` applies a SQL-style predicate and returns a filtered `TableHDU`.

### `TableHDURef`

Lazy file-backed table handle. Returned by `hdul[n]` for table HDUs.

```python
with torchsky.open("cat.fits") as hdul:
    t = hdul[1]               # TableHDURef
    ra = t["RA"]              # read single column
    for chunk in t.iter_rows(batch_size=100_000): ...
    for batch in t.scan_arrow(batch_size=100_000): ...
    full = t.materialize()    # -> TableHDU
```

**In-place file mutations:**
`append_rows_file`, `update_rows_file`, `insert_rows_file`, `delete_rows_file`,
`insert_column_file`, `replace_column_file`, `rename_columns_file`, `drop_columns_file`.

**Query/projection:**
`select(columns)`, `head(n)`, `read(columns=, row_slice=)`, `to_arrow()`, `scan_arrow()`.

### `Header`

Dict-like FITS header with ordered card tracking. Supports `header["KEY"]`,
`header.get("KEY", default)`, `header["KEY"] = val`, `.items()`, `.cards`,
`.card(key)`, `.comments(key)`, `.append(card)`, `.insert(index, card)`,
`.remove(key)`, `.add_history(text)`, `.add_comment(text)`,
`.get_history()`, and `.get_comment()`.

`Card(key, value, comment="")` represents one FITS header card and is tuple-compatible
with `(key, value, comment)` internal records.

---

## WCS

```python
from torchsky.wcs import WCS

wcs = WCS(header_dict)           # from a dict / FITS header
alt = WCS(header_dict, key="A")   # alternate FITS-WCS version
wcs = torchsky.get_wcs("image.fits", hdu="auto")

ra, dec = wcs.pixel_to_world(x, y)
x, y = wcs.world_to_pixel(ra, dec)
pixel, world = wcs.mixed_pixel_world(pixel=[[12.0, float("nan")]], world=[[float("nan"), -1.2]])
```

Pure-PyTorch implementation. Runs on CPU, CUDA, or MPS. Follows the same `pixel_to_world` / `world_to_pixel` interface as [astropy.wcs](https://docs.astropy.org/en/stable/wcs/).

**Supported projections (13):** TAN, SIN, ARC, ZPN, ZEA, STG, CEA, CAR, MER, AIT, MOL, HPX, SFL. Distortions: SIP, TPV, TNX, ZPX.

Compatibility-oriented WCS surfaces:

- `WCS.available_wcs_keys(header)` returns primary (`""`) and alternate (`"A"` .. `"Z"`) FITS-WCS versions present in a header.
- `WCS(header, key="A")` selects alternate-version FITS-WCS keywords such as `CTYPE1A`, `CRPIX1A`, and `PV2_1A`.
- `all_pix2world(...)` and `all_world2pix(...)` are Astropy-compatible aliases that accept stacked coordinate arrays plus positional `origin`, or separate coordinate arrays plus positional `origin`.
- `mixed_pixel_world(pixel=..., world=...)` accepts arrays with trailing dimension `naxis`; mark unknown pixel/world entries with `NaN`. It fills the missing side for full pixel/full world inputs and solves the common Kapteyn-style mixed 2D celestial case with one known pixel axis plus one known world axis.
- Metadata properties: `pixel_n_dim`, `world_n_dim`, `pixel_axis_names`, `world_axis_names`, `world_axis_units`, `world_axis_physical_types`, `pixel_shape`, `array_shape`, `pixel_bounds`, and `axis_correlation_matrix`.
- Geometry helpers: `footprint()` returns celestial image corners and `in_image(pixels)` masks coordinates inside `NAXISi` bounds.
- Astropy-style value/index helpers: `pixel_to_world_values(...)`, `world_to_pixel_values(...)`, `array_index_to_world_values(...)`, and `world_to_array_index_values(...)`. Array-index helpers use NumPy order, so a 2D image accepts or returns `(row, column)`.
- Cutout helpers: `slice(view, numpy_order=True)` returns a dimensionality-preserving WCS for positive-step array slices, updating `CRPIX` and `CD` so cropped or strided image views keep correct coordinates.
- Axis-reduction helpers: `dropaxis(axis)` removes one uncoupled FITS-order axis, and `celestial` extracts a 2D celestial WCS from a spatial cube when extra axes are uncoupled. Coupled axis drops fail closed.
- Differentiable geometry helpers: `pixel_to_world_differentiable(...)`, `world_to_pixel_differentiable(...)`, `pixel_to_world_jacobian(...)`, and `world_to_pixel_jacobian(...)` use pure Torch paths so WCS can sit inside INR/field-model training and multi-modal geometry encoders without routing gradients through native C++ shortcuts.
- Spectral helpers: `spectral_to_frequency(...)` and `frequency_to_spectral(...)` convert the selected WCS spectral axis through frequency using `CTYPE`, `CUNIT`, and `RESTFRQ` / `RESTFREQ` / `RESTWAV` metadata.

Low-level torch-native spectral conversion helpers are also available from `torchsky.wcs` and `torchsky.coords`: `convert_spectral`, `spectral_to_frequency`, `frequency_to_spectral`, `air_wavelength_to_vacuum`, and `vacuum_wavelength_to_air`. Supported coordinate types are `FREQ`, `WAVE`, `WAVN`, `AWAV`, `ENER`, `ZOPT`, `VRAD`, `VOPT`, `VELO`, and `BETA`; defaults follow FITS SI units unless explicit units such as `"Angstrom"`, `"cm-1"`, `"eV"`, or `"km/s"` are provided. The WCS transform path also supports the validated one-axis FITS spectral `-LOG` algorithm for `FREQ-LOG`, `WAVE-LOG`, and `WAVN-LOG`; compound grism and nonlinear Paper III algorithm chains remain roadmap work.

Temporal helpers in `torchsky.wcs.temporal` include `TemporalWCS`, `convert_time_scale`, `temporal_unit_to_days`, and `tai_minus_utc_seconds`. They support UTC/TAI/TT/GPS MJD conversions with a built-in leap-second table, common FITS/VO time units from nanoseconds through days/years/centuries, and reference-position / observatory metadata surfaces (`TREFPOS`, `TREFDIR`, `PLEPHEM`, `OBSGEO-*`). Ephemeris-dependent reference-position corrections currently fail closed with `NotImplementedError`.

Celestial frame helpers from `torchsky.wcs` and `torchsky.coords` include `icrs_to_galactic`, `galactic_to_icrs`, `precess_fk5`, and `convert_celestial`. The current native frame slice covers fixed ICRS/Galactic rotations and FK5 Julian-equinox mean precession; FK4 E-terms, nutation, aberration, proper motion, and apparent-place corrections remain roadmap items.

Alignment helpers include `fit_affine_2d(source, target, geometry=...)` for weighted correction fits. Supported geometries are `shift`, `rshift`, `rscale`, and `general`; the result carries the matrix, offset, residuals, RMS, inlier mask, and an `apply(points)` helper. `TangentPlaneFrame` provides differentiable gnomonic sky-to-plane transforms, and `fit_sky_affine(...)` fits the same correction families in that local sky plane.

Tabular lookup helpers include `TabularAxis(pixel, world, bounds="nan")` and `TabularLookupWCS([...])`. They provide torch-native piecewise-linear lookup axes, monotonic inverse lookup, optional clamp-vs-NaN bounds behavior, and autograd through the local interpolation slope. `WCS(header, tabular_axes={axis_number: axis})` wires independent `CTYPEi` `-TAB` axes into forward, inverse, and differentiable linear paths using FITS 1-based axis numbers. `tabular_axes_from_fits(header, tables)` builds that explicit axis mapping for the independent 1D FITS `-TAB` subset using `PSi_0` table selection, `PSi_1` coordinate columns, and optional `PSi_2` indexing vectors. Coupled multidimensional coordinate-array semantics remain roadmap work.

After `.compile()`, use a single **stacked** tensor so the transform goes through the compiled Torch graph: for 2D spatial WCS and FITS `origin` **0 or 1**, `pixel_to_world(pixels)` with `pixels.shape == (..., 2)` fuses `pixels + (1-origin) - CRPIX` with the projection and uses **Torch-only** forward pole rotation (so `torch.compile` can optimize end-to-end). Symmetrically, `world_to_pixel(world)` with `world.shape == (..., 2)` uses a fused inverse graph with **Torch-only** inverse pole rotation. The two-argument `pixel_to_world(x, y)` and `world_to_pixel(ra, dec)` forms may still use native C++ fast paths. Run `benchmarks/bench_wcs_small_n_paths.py` on your machine; Astropy may still win at tiny `N`, but the fused compiled paths are the ones to tune. Parity vs Astropy is enforced in `tests/test_wcs_parity.py`.

Coordinates use 0-based Python pixel convention. Results are in degrees.

---

## Mapmaking (`torchsky.map`)

```python
from torchsky.map import ImageObservation, coadd_image_plane

result = coadd_image_plane(
    [
        ImageObservation("img_a.fits", weight=1.0),
        ImageObservation(
            "img_b.fits",
            weight_path="img_b_wht.fits",
            mask_path="img_b_mask.fits",
        ),
    ],
    footprint_order=8,
)

coadd = result.data
weight = result.weight
header = result.header
overlap = result.overlap_count
coverage = result.coverage_moc
```

```python
import torch

from torchsky.map import coadd_image_plane_cube, plan_image_plane_cube_target
from torchsky.spectral import SpectralAxis

target = plan_image_plane_cube_target(
    observations,
    reference_index=0,
    spectral_axis=SpectralAxis(
        values=torch.tensor([4800.0, 6500.0], dtype=torch.float64),
        unit="Angstrom",
        type="WAVE",
    ),
    spectral_bin_edges=torch.tensor([4700.0, 4900.0, 6600.0], dtype=torch.float64),
)
cube = coadd_image_plane_cube(observations, target=target, footprint_order=8)
print(cube.spectral_axis.unit)  # Angstrom
print(cube.data.shape)          # [n_band, y, x]
```

### `ImageObservation(...)`

Fields:
- `path`, `hdu`: image source
- `weight`: scalar observation weight
- `photometric_scale`: known multiplicative calibration factor applied before stacking; accumulated weights are scaled by `1 / photometric_scale^2`
- `background_offset`: known additive background level subtracted before stacking
- `weight_path`, `weight_hdu`: optional per-pixel positive weight / inverse-variance map
- `mask_path`, `mask_hdu`: optional per-pixel validity mask (`> 0` means valid)
- `wavelength`: scalar spectral coordinate for exact-band grouping or cube planning
- `spectral_min`, `spectral_max`: optional bandpass bounds for interval-aware spectral reprojection
- `transmission_center`, `transmission_sigma`: optional Gaussian transmission-filter parameters for filter-weighted spectral reprojection
- `band_name`: optional band label carried into planned band metadata

### `MapBandpassSpec(...)`

Thin product-band definition for user-facing output requests.

Fields:
- `label`: optional band label carried through product planning and rendering
- `spectral_min`, `spectral_max`: optional output bandpass bounds
- `transmission_center`, `transmission_sigma`: optional Gaussian bandpass parameters used by `render_field_product(...)`

### `MapProductSpec(target_header, bands, value_semantics="surface_brightness", footprint_order=None)`

Canonical map product contract.

- `target_header`: requested final WCS / grid
- `bands`: tuple of requested `MapBandpassSpec` outputs
- `value_semantics`: `"surface_brightness"` or `"flux"`
- `footprint_order`: optional MOC footprint order carried into direct stack results

This is the common user-facing contract for:
- direct stack products
- rendered continuous-field products
- product-level evaluation metrics

### `plan_map_product(observations, *, reference_index=0, header_overrides=None, bands=None, spectral_axis=None, spectral_bin_edges=None, spectral_unit="Angstrom", spectral_type="WAVE", value_semantics="surface_brightness", footprint_order=None) -> MapProductSpec`

Thin planner over the existing spatial and spatial+spectral planning helpers.

- If `bands` is omitted and the observations carry spectral metadata, this uses the current cube planner and converts the result into a `MapProductSpec`.
- If the observations are plain 2D images without spectral metadata, this produces a single-band product on the planned target grid.
- This keeps the old planning helpers intact:
  - `plan_image_plane_target(...)`
  - `plan_image_plane_cube_target(...)`

### `stack_map_product(observations, product, *, resampling_kernel="bilinear", method="coadd", stacking_method="weighted_mean") -> MapProductResult`

Product-oriented direct stack entry point.

- Current supported method: `"coadd"`
- Reuses the existing explicit stack semantics:
  - weighted numerator/denominator reprojection per observation
  - no hidden photometric solving
  - no hidden background solving
- `stacking_method` may be:
  - `"weighted_mean"`: inverse-variance-style weighted mean across observations
  - `"mean"`: equal-weight mean across observations after reprojection, with weights and masks still defining valid support inside each observation
  - `"median"`: single-band image-plane median across observations after reprojection, with weights and masks still defining valid support inside each observation
- For single-band image products this wraps `coadd_image_plane(...)`.
- For representable multi-band products it wraps `coadd_image_plane_cube(...)`.
- Current direct-stack limitation:
  - multi-band products must currently be either exact-center bands or ordered contiguous interval bands

### `render_field_product(model, product, *, num_spectral_samples=9) -> MapProductResult`

Renders a fitted continuous field onto the exact same requested product contract as `stack_map_product(...)`.

- Every requested band is rendered on the same final WCS/grid from `product.target_header`.
- Gaussian bandpass metadata in `MapBandpassSpec` is honored here directly.
- This is the intended user-facing comparison point for stack vs stack++: both methods produce the same final product shape.

### `evaluate_map_product(prediction, truth, *, weight=None, header=None, value_semantics=None) -> dict`

Shared product-level evaluator for final image or cube products.

Primary metrics:
- `reconstruction_rmse`
- `reconstruction_mae`
- `reconstruction_p95_abs_err`
- `reconstruction_max_abs_err`
- `signal_to_residual_ratio`
- `signal_to_residual_db`
- `total_flux_relative_error`
- `band_flux_relative_error_mean`
- `band_flux_relative_error_max`
- `blue_red_flux_ratio_absolute_error`
- `mid_blue_flux_ratio_absolute_error`
- `coverage_fraction`

This is the standardized metric vocabulary now used by the stack truth and stack++ truth benchmarks.

### `evaluate_map_frontier_regions(prediction, truth, *, support_weight, overlap_count=None, fringe_kernel_size=9) -> dict`

Cheap region-split diagnostics for sparse large-mosaic products.

Primary metrics:
- `support_single_overlap_reconstruction_rmse`
- `support_multi_overlap_reconstruction_rmse`
- `fringe_extrapolated_reconstruction_rmse`
- `far_extrapolated_reconstruction_rmse`
- `support_single_overlap_fraction`
- `support_multi_overlap_fraction`
- `fringe_extrapolated_fraction`
- `far_extrapolated_fraction`

This is intended for diagnosing whether a field fit is mainly failing on weakly supported observed pixels, just outside the observed support fringe, or only in truly far extrapolated regions.

Field-fit benchmark rows now also expose a uniform-grid comparison surface:
- `uniform_reconstruction_rmse`
- `uniform_reconstruction_mae`
- `uniform_signal_to_residual_db`
- `uniform_total_flux_relative_error`

These are computed with an all-ones comparison weight on the full target grid, so masked coadds and globally defined fields can be compared on exactly the same spatial support.

### `coadd_image_plane(observations, *, target_header=None, output_path=None, footprint_order=None, resampling_kernel="bilinear", value_semantics="surface_brightness", stacking_method="weighted_mean") -> MapResult`

Differentiable image-plane coaddition baseline.

- If an observation already matches the target WCS/shape, torchsky accumulates it directly.
- Otherwise torchsky resamples onto the target grid with either `grid_sample(..., padding_mode="zeros", align_corners=True)` for `"nearest"`, `"bilinear"`, and `"bicubic"`, or native separable high-accuracy kernels for `"lanczos3"`, `"lanczos5"`, `"mitchell_netravali"`, and `"catmull_rom"`.
- `resampling_kernel` may be `"nearest"`, `"bilinear"`, `"bicubic"`, `"lanczos3"`, `"lanczos5"`, `"mitchell_netravali"`, `"catmull_rom"`, `"fft_shift"`, `"fft_diagonal_affine"`, `"periodic_linear_affine"`, `"fft_linear_affine"`, `"drizzle_point"`, `"drizzle_square"`, `"conservative_polygon"`, `"splat_point"`, or `"splat_box"`.
- Kernel notes:
  - `"catmull_rom"` is an interpolating cubic kernel.
  - `"mitchell_netravali"` is a smoother cubic reconstruction kernel and is not exact at input samples.
  - `"lanczos3"` / `"lanczos5"` are higher-support windowed-sinc kernels.
- Current kernel guidance from the truth benchmark surface:
  - general default: `"bilinear"`
  - stronger interpolating cubic: `"catmull_rom"`
  - smoother reconstruction: `"mitchell_netravali"`
  - widest-support sinc option: `"lanczos5"` when accuracy matters more than throughput
- `value_semantics` may be:
  - `"surface_brightness"`: preserve the current surface-brightness-style weighted mean
  - `"flux"`: apply local source/target pixel-area factors so reprojection is approximately flux-conserving
- `stacking_method` may be:
  - `"weighted_mean"`: accumulate projected weighted numerators and denominators, then divide at the end
  - `"mean"`: convert each projected observation into an output-space image, then average observations equally wherever they have positive projected support
  - `"median"`: convert each projected observation into an output-space image, then take the per-pixel median over observations with positive projected support
- The resampling contract is explicit:
  - sample a weighted numerator image `image * weight`
  - sample the weight image separately on the same target grid
  - for `"weighted_mean"` with `"surface_brightness"` products, torchsky rescales inverse-variance-style weights by the representative `target_pixel_area / source_pixel_area` factor before cross-observation accumulation
  - for `"weighted_mean"`, sum those projected numerators and denominators over observations and divide at the end
  - for `"mean"`, derive one projected image per observation and average those equally across observations with positive support
- If known calibration metadata is provided on an observation, torchsky first applies:
  - `image_calibrated = photometric_scale * (image - background_offset)`
  - `weight_calibrated = weight / photometric_scale^2`
- The output is a weighted numerator/denominator coadd:
  - `result.data`: final stacked image
- `result.weight`: accumulated per-pixel inverse-variance weight for `"weighted_mean"` or contributing-support count for `"mean"` / `"median"`
  - `result.overlap_count`: number of contributing observations per target pixel
  - `result.header`: target FITS header
- If `footprint_order` is set, torchsky also returns:
  - `result.coverage_moc`: union footprint MOC on the target grid
  - `result.observation_mocs`: one MOC per observation

Current scope:
- 2D images only
- scalar weights plus optional per-pixel weight and mask maps
- optional known per-observation photometric scale and additive background normalization
- `"median"` currently applies only to single-band image-plane coadds
- target grid inferred from the first observation unless `target_header` is provided
- footprint MOC diagnostics are optional and controlled by `footprint_order`
- no automatic photometric renormalization, zeropoint solving, or background matching; callers may supply known normalization terms, but torchsky does not solve them yet

### `ImagePlaneMapMaker(target_header=None, resampling_kernel="bilinear", value_semantics="surface_brightness", stacking_method="weighted_mean")`

Thin wrapper around `coadd_image_plane(...)` for repeated runs with a fixed target header.

### `plan_image_plane_target(observations, *, reference_index=0, header_overrides=None) -> PlannedImagePlaneTarget`

Explicit shared-grid planning helper for repeated or multi-band coadds.

- Reads the selected reference observation and normalizes its FITS header to the image shape.
- Returns:
- `header`: ready-to-use target FITS header
- `shape`: `(ny, nx)` target image shape
- Use `header_overrides` to pin a common output grid while preserving the reference observation as the template.

### `plan_image_plane_cube_target(observations, *, reference_index=0, header_overrides=None, spectral_axis=None, spectral_bin_edges=None, spectral_unit="Angstrom", spectral_type="WAVE") -> PlannedImagePlaneCubeTarget`

Joint spatial+spectral planner for cube-style map assembly.

- Reuses `plan_image_plane_target(...)` for the spatial grid.
- Builds a validated spectral target:
  - default: sorted unique observation wavelengths
  - explicit: caller-provided `SpectralAxis`
- Optional `spectral_bin_edges` enables interval-aware spectral reprojection using exact overlap fractions.
- Emits `band_metadata` with:
  - target band centers and bounds
  - number of contributing observations per target band
  - summed spectral projection weight per target band
  - stable band names when the contributing observations agree
- `target.to_band_table()` exposes that metadata as a column-style dict for downstream table interop.

### `plan_image_plane_cube_target_from_header(observations, spectral_header, *, reference_index=0, header_overrides=None, use_header_bin_edges=True) -> PlannedImagePlaneCubeTarget`

Compatibility planner for cube-style spectral WCS headers.

- Reconstructs a linear `SpectralAxis` from FITS `CTYPEi/CRVALi/CDELTi/CRPIXi/CUNITi`.
- Optionally derives spectral bin edges from the same header for interval-aware reprojection.
- This is the intended bridge for future comparisons against cube-oriented astronomy packages without making the core map primitive wrapper-shaped.
- The same pattern applies to image-plane comparison harnesses: external astronomy packages are benchmark and compatibility targets, not the primitive implementation.

### `coadd_image_plane_cube(observations, *, target=None, target_header=None, footprint_order=None, resampling_kernel="bilinear", value_semantics="surface_brightness", stacking_method="weighted_mean", spectral_axis=None, spectral_bin_edges=None, spectral_unit="Angstrom", spectral_type="WAVE", reference_index=0) -> SpectralCubeMapResult`

Planned spatial+spectral cube assembly on a shared image plane.

- If `target` is omitted, torchsky plans the spatial and spectral grid from the provided arguments.
- Spatial reprojection uses the same weighted image-plane coadd path as `coadd_image_plane(...)`.
- `resampling_kernel` uses the same kernel choices as `coadd_image_plane(...)`.
- `value_semantics` uses the same surface-brightness vs flux interpretation as `coadd_image_plane(...)`.
- `stacking_method` uses the same `"weighted_mean"` vs `"mean"` choices as `coadd_image_plane(...)`.
- `"median"` is currently not supported for cube or multiband stacking.
- Spectral reprojection is deterministic:
  - exact-center assignment when no `spectral_bin_edges` are provided
  - exact interval-overlap fractions when `spectral_bin_edges` are provided and observations carry `spectral_min` / `spectral_max`
  - normalized Gaussian filter-bin integrals when `spectral_bin_edges` are provided and observations also carry `transmission_center` / `transmission_sigma`
- Mathematically, when Gaussian transmission metadata is present, the spectral weights are:
  - `w_k propto integral_{lambda_k^-}^{lambda_k^+} exp(-(lambda-mu)^2 / (2 sigma^2)) d lambda`
  - truncated to the observation's `[spectral_min, spectral_max]` interval when those bounds are narrower than the target bins
  - normalized so `sum_k w_k = 1`
- Returns:
  - `spectral_axis`
  - `band_metadata`
  - `data`, `weight`, `overlap_count`
  - `coverage_mocs` when `footprint_order` is set
  - `spatial_wcs`
- `result.to_data_cube()` exposes the output as a `torchsky.spectral.DataCube`.

### `coadd_image_plane_multiband(observations, *, target_header=None, footprint_order=None, resampling_kernel="bilinear", value_semantics="surface_brightness", stacking_method="weighted_mean", spectral_unit="Angstrom", spectral_type="WAVE") -> MultiBandMapResult`

Grouped multi-wavelength assembly on a shared target grid.

- Every observation must define `wavelength`.
- This is the thin exact-center compatibility wrapper over `coadd_image_plane_cube(...)`.
- `stacking_method` is forwarded directly to `coadd_image_plane_cube(...)`.
- `"median"` is therefore currently unsupported here as well.
- Observations are grouped by exact `wavelength` value and sorted ascending.
- Use `plan_image_plane_target(...)` plus `target_header=...` when bands come from different pointings or per-band reference headers.

Returns:
- `spectral_axis`: `SpectralAxis` carrying sorted per-band coordinates and metadata
- `wavelength`: convenience tensor view of `spectral_axis.to_wavelength()`
- `data`: stacked coadds `[n_band, y, x]`
- `weight`: stacked per-band accumulated weights `[n_band, y, x]`
- `overlap_count`: stacked overlap counts when available
- `coverage_mocs`: one coverage MOC per band when `footprint_order` is set
- `band_observation_counts`: number of observations contributing to each band

The older coadd APIs remain the compatibility surface.

- `coadd_image_plane(...)`
- `coadd_image_plane_cube(...)`
- `coadd_image_plane_multiband(...)`

The newer product-oriented entry points are:
- `plan_map_product(...)`
- `stack_map_product(...)`
- `render_field_product(...)`

### Experimental Field Models

The `torchsky.map` namespace also exposes an experimental continuous-field layer for fitting a latent sky model `f(alpha, delta, lambda)` directly to overlapping observations.

Main pieces:
- `FieldObservation(...)`: in-memory observation with image tensor, FITS header, optional weight/mask, Gaussian bandpass metadata, and optional PSF blur.
- `FieldDomain(...)`: coordinate normalization domain for continuous field models, including `coord_mode` (`local_tangent` or `sphere_xyz`).
- `SineImplicitField(domain, ...)`: SIREN-style sine-activated implicit neural field.
- `WireImplicitField(domain, ...)`: WIRE-style Gabor-activated implicit neural field.
- `GaussianBasisField(domain, ...)`: smooth Gaussian radial-basis latent field.
- `FourierMLPField(domain, ...)`: multiscale Fourier-feature MLP latent field for harder heterogeneous cases.
- `SpectralBasisField(domain, ...)`: separable spatial-spectral latent field that learns a spectrum per sky position through a low-rank spectral basis.
- `BandpassConditionedField(domain, ...)`: transmission-conditioned latent field that predicts band-integrated images from the actual bandpass descriptor, while still supporting exact-wavelength evaluation as a degenerate delta-band case.
- `LearnedSpectralBasisField(domain, ...)`: spatial coefficient field with a learned nonlinear spectral basis, aimed at making the strongest current exact-band family more expressive without collapsing spectral structure into a single generic MLP.
- `SkyEncoderBandpassField(domain, ...)`: sky-encoder latent field that aggregates transmission-weighted multiscale sky features over the sampled bandpass before decoding, aimed at harder broad-band deep-field fitting where exact-wavelength baselines are no longer the right target. Supports `scale_fourier_to_domain` like other sky-encoder fields.
- `SkyLambdaPlanesField(domain, ...)`: factorized `(sky, wavelength)` feature planes, in the spirit of K-Planes, aimed at modeling wavelength-dependent morphology without forcing strict spectral separability.
- `SkyLambdaKPlanesField(domain, ...)`: multiscale factorized `(sky, wavelength)` feature planes with a tiny decoder and residual refinement head, aimed at the next generation of larger mosaic and deep-field transmitted-field fitting.
- `SkyLambdaFeatureField(domain, ...)`: experimental multiscale `sky-lambda` feature field with per-scale hidden encoders, learned scale fusion, and an optional coarse global branch for harder sparse-mosaic extrapolation, aimed at a more explicit `inr_c`-style feature-field architecture than the raw K-planes concatenation path.
- `OperatorConstrainedFeatureField(domain, ...)`: operator-aware multiscale spatial feature field with a low-dimensional continuous spectral scaffold, explicit observability-aware spectral regularization hooks, and a small bandpass-conditioned residual head aimed at improving supported-pixel fidelity on sparse mosaics without giving back full-grid coherence.
- `make_field_model("sky_lambda_feature_field_hetero", domain, ...)`: convenience alias for a `SkyLambdaFeatureField` with an enabled predictive log-variance head, intended for experimental heteroscedastic likelihood fitting.
- `make_field_model("sky_lambda_feature_atlas_field", domain, ...)`: atlas of locally blended `SkyLambdaFeatureField` experts for experimental sparse-mosaic locality tests.
- `SkyLambdaAtlasField(domain, ...)`: a small atlas of locally blended `SkyLambdaKPlanesField` experts for larger sparse mosaics where one global field may not allocate capacity locally enough, now with optional sparse top-k routing to keep locality sharper on harder broad-band rows.
- `ConditionedSpectralField(domain, ...)`: shared spatial latent field with a wavelength-conditioned decoder, aimed at broad-filter and exact-band multiwavelength fitting without strict low-rank spectral separation.
- `HashGridConditionedField(domain, ...)`: spatial multiresolution encoder (dense voxels on coarse levels where small, hashed on fine levels) with an explicit wavelength decoder; supports `log2_hashmap_size` per level, `hash_interpolation="smoothstep"`, `hash_dense_coarse`, `hash_max_dense_cells`, and `hash_level_gain` (same knobs on other hash-grid fields below).
- `HashGridSpectralField(domain, ...)`: same spatial encoder with a spectral head, for testing stronger spatial capacity without hashing wavelength directly.
- `SkyEncoderMLPField(domain, ...)`: sky-aware multiscale encoder + MLP latent field using `torchsky.encoders.SkyCoordinateEncoder` (harmonic + spherical Fourier + log-wavelength Fourier). Default `scale_fourier_to_domain=True` sets spherical `min_frequency` from `FieldDomain.spatial_scale_deg` via `sky_encoder_fourier_min_frequency`; set `False` to match older fixed-frequency behavior or loaded checkpoints. Defaults: `num_fourier_frequencies=8`, `spectral_frequencies=5`, SiLU trunk with Kaiming init and a small log-flux readout before `softplus`.
- `StrainerSkyEncoderMLPField(domain, ...)` / `make_field_model("strainer_sky_encoder_mlp", ...)`: same encoder + trunk capacity as `SkyEncoderMLPField` with explicit `encoder` + `body` + `decoder_head`, `reset_decoder_head()`, trunk-only checkpoints (`save_strainer_trunk_checkpoint` / `load_strainer_trunk_checkpoint`), and offline synthetic trunk pretraining (`pretrain_strainer_trunk_synthetic` or `scripts/pretrain_strainer_trunk.py`). Use `fit_field_model(..., freeze_field_encoder=True, freeze_field_body=True)` for head-only fine-tuning.
- `SkyEncoderResMLPField(domain, ...)`: stronger residual sky-encoder latent field for more demanding heterogeneous common-PSF fits, with the same domain-scaled Fourier option (`scale_fourier_to_domain`), optional multiscale readout path for harder broad-band deep-field rows, and Kaiming init on the input/readout stack.
- `HashGridMLPField(domain, ...)`: multilevel hash-grid encoder + feedforward MLP decoder.
- `HashGridResMLPField(domain, ...)`: same encoder + residual MLP decoder (`make_field_model("hashgrid_resmlp", ...)`), aligned with explicit-feature-plus-refinement INR patterns used in benchmarks.
- `estimate_field_domain(observations, coord_mode=...)`: derives a practical normalization domain from a set of observations.
- `render_field_observation(model, observation, ...)`: forward-renders a continuous field through the observation WCS, Gaussian bandpass, and optional Gaussian PSF.
- `render_field_observation_distribution(model, observation, ...)`: like `render_field_observation(...)`, but returns `(mean, logvar)` when the model supports predictive variance.
- `render_field_cube(model, target_header=..., spectral_axis=...)`: evaluates the fitted field on a shared target grid and spectral axis; optional `spatial_tile_hw=(th,tw)` for spatial tiling, `use_func_vmap` + `func_vmap_chunk` for `torch.func.vmap` over pixels (see `evaluate_field_coordinates_vmapped`).
- `render_field_healpix(model, nside=..., spectral_axis=..., nest=...)`: HEALPix-native `(nλ, npix)` rendering with `spatial_chunk_pixels` for pixel-axis chunking.
- `render_field_cube_via_healpix(model, target_header=..., spectral_axis=..., nside=..., nest=...)`: evaluate on HEALPix then `sample_healpix_map_on_image_header` onto **any** 2D WCS grid (not only TAN); useful when you want a spherical intermediate or multiple projections from one healpix cube.
- `sample_healpix_map_on_image_header` / `sample_healpix_map_on_tan_header` (alias): bilinear HEALPix → image sampling using `torchsky.WCS` + `get_interp_val`.
- `evaluate_field_coordinates_vmapped(model, ra_deg, dec_deg, wavelength)`: `functional_call` + `vmap` over flattened coordinates (JAX-style); prefer batched `model(ra, dec, λ)` for speed on standard field modules.
- `fit_field_model(observations, model, ...)`: weighted optimization loop over the rendered observations, with `mse`, `charbonnier`, `huber`, or `gaussian_nll` residual losses, optional `per_observation`, `per_band`, `global`, or mixed `band_global_mix` weighted reduction, optional cosine learning-rate scheduling, optional short learning-rate warmup (`lr_warmup_steps`), optional Adam weight decay (`weight_decay`), optional exponential moving-average evaluation weights (`ema_decay`), optional gradient clipping, optional coarse-to-fine curriculum support for spectral-basis frequency ramps and structured feature-field scale ramps (`curriculum="spectral_frequency_ramp"` or `"feature_scale_ramp"`), optional training-time spectral sampling (`training_spectral_sampling="bandpass_mc"` or `"mixture_mc"`), optional light deterministic full-band anchor supervision on a few observations per step, optional explicit role-selective component anchor supervision through `FieldComponentAnchorTerm` with rendered image anchors, direct coordinate/spectral sample anchors, or pixel-frame astrometric sample anchors resolved through the observation WCS, optional geometry-aware role centroid supervision through `FieldCentroidAnchorTerm`, optional observation minibatching (`observations_per_step`, including `observation_sampling="band_balanced"`), optional local patch-based observation crops (`training_patch_size`, `training_patches_per_observation`), optional experimental coherent shared-sky patch sampling across those cropped observations (`shared_sky_patch_sampling=True`), optional uncropped context observations inside that minibatch (`full_observations_per_step`), optional low-resolution full-frame context loss (`context_observations_per_step`, `context_downsample_factor`, `context_loss_weight`), optional low-memory sequential backward accumulation across those sampled observations (`streaming_accumulation=True`), optional operator-aware spectral regularization (`spectral_observability_weight`, `spectral_observability_probe_grid_size`), optional residual-head shrinkage (`residual_head_penalty_weight`), and optional spatial smoothness regularization on a small probe grid, while keeping deterministic quadrature for evaluation.

Mathematically, this path is explicit:
- latent field: `f(alpha, delta, lambda)`
- observation model: `I_i(x, y) ~= PSF_i * integral f(alpha_i(x, y), delta_i(x, y), lambda) T_i(lambda) d lambda`
- current bandpass model: normalized Gaussian `T_i(lambda)` or exact delta-band transmission when an observation/band is specified by a single wavelength
- current fit objective: weighted per-pixel residual loss over the rendered observation images (`mse`, `charbonnier`, or `huber`, depending on the fitting setup)

This is intentionally narrower than the image-plane coadd API. It is the first fitting surface for continuous multiwavelength modeling, not yet a full latent-sky pipeline with PSF learning, calibration solving, or background modeling.

---

## Table Module (`torchsky.table`)

### Read / write

```python
torchsky.table.read(path, hdu=1, columns=None, where=None)
torchsky.table.scan(path, hdu=1, chunk_rows=10000)
torchsky.table.reader(path, hdu=1)
torchsky.table.write(path, data, header=None, overwrite=False)
```

### In-place mutations

```python
torchsky.table.append_rows(path, rows, hdu=1)
torchsky.table.update_rows(path, rows, row_slice, hdu=1)
torchsky.table.insert_rows(path, rows, *, row, hdu=1)
torchsky.table.delete_rows(path, row_slice, hdu=1)
torchsky.table.insert_column(path, name, values, hdu=1, index=None, **meta)
torchsky.table.replace_column(path, name, values, hdu=1)
torchsky.table.rename_columns(path, mapping, hdu=1)
torchsky.table.drop_columns(path, columns, hdu=1)
```

### Interop

```python
torchsky.table.to_polars_lazy(path, hdu=1, decode_bytes=True)  # → polars.LazyFrame
torchsky.table.to_duckdb(path, hdu=1, relation_name="tbl", connection=con)
torchsky.table.duckdb_query(path, sql, hdu=1)
torchsky.table.scanner(path, hdu=1, columns=None, where=None, filter=None)  # → PyArrow scanner
torchsky.to_arrow(table_dict, decode_bytes=True, vla_policy="list")   # → pyarrow.Table
torchsky.to_pandas(table_dict, decode_bytes=True, vla_policy="object") # → pd.DataFrame
```

Interop with [Pandas](https://pandas.pydata.org/), [Polars](https://pola.rs/), [DuckDB](https://duckdb.org/), and [PyArrow](https://arrow.apache.org/docs/python/).

### Predicate pushdown

The `where=` parameter filters rows in C++ before data reaches Python.

**Operators:** `=`, `!=`, `<`, `>`, `<=`, `>=`, `AND`, `OR`, `NOT`, `IN (...)`, `NOT IN (...)`, `BETWEEN ... AND ...`, `IS NULL`, `IS NOT NULL`.

```python
torchsky.read("cat.fits", hdu=1, where="MAG_G < 20 AND DEC > 0")
torchsky.read("cat.fits", hdu=1, where="FILTER IN ('g', 'r') AND REDSHIFT BETWEEN 0.1 AND 0.5")
```

## Catalogs (`torchsky.catalogs`)

### `crossmatch_sky(left_ra_deg, left_dec_deg, right_ra_deg, right_dec_deg, *, radius_arcsec, left_chunk_size=4096, right_chunk_size=65536)`

Chunked nearest-neighbor sky crossmatch for tensor inputs.

- Inputs are longitude / latitude in degrees.
- Returns one nearest match per left row within `radius_arcsec`.
- The public API stays simple; candidate pruning is handled internally.
- Current implementation uses an internal coarse nested-HEALPix right-catalog index before exact great-circle refinement, with a benchmark-backed workload-size selector for the coarse index scale.
- Compatibility validation covers both synthetic stress cases and real upstream catalog fixtures vendored from Astropy's VOTable test data.

```python
from torchsky.catalogs import crossmatch_sky

result = crossmatch_sky(
    left_ra_deg=left["RA"],
    left_dec_deg=left["DEC"],
    right_ra_deg=right["RA"],
    right_dec_deg=right["DEC"],
    radius_arcsec=1.5,
)

print(result.left_index.shape)
print(result.right_index.shape)
print(result.separation_arcsec.shape)
```

Returns `CrossmatchResult` with:

- `left_index`: indices into the left catalog
- `right_index`: nearest matched indices into the right catalog
- `separation_arcsec`: on-sky separations in arcseconds

### `CatalogStreamDataset(partitions, loader=lambda x: x)`

`IterableDataset` helper for streaming partitioned catalog sources into PyTorch data pipelines.

```python
from torch.utils.data import DataLoader
from torchsky.catalogs import CatalogStreamDataset

ds = CatalogStreamDataset(partitions=range(8))
loader = DataLoader(ds, num_workers=2, batch_size=None)
for partition in loader:
    ...
```

This is intended as a thin streaming adapter, not a full catalog orchestration layer.

---

## Spectral Types (`torchsky.spectral`)

Experimental helpers for 1D spectra and IFU cubes. Not part of the core FITS I/O contract.

### `Spectrum1D`

```python
from torchsky.spectral import Spectrum1D, SpectralAxis, SpectralReader

sp = SpectralReader.read_spectrum_1d("spec.fits", hdu=0,
         flux_col="FLUX", wave_col="WAVELENGTH", ivar_col="IVAR", mask_col="MASK")
sp.wavelength     # Angstroms
sp.frequency      # Hz
sp.error          # 1/sqrt(ivar)
sp.apply_mask()   # masked pixels -> NaN
sp.resample(new_wave_tensor)
```

### `DataCube`

```python
cube = SpectralReader.read_data_cube("cube.fits", hdu=0)
cube.extract_spectrum(y=100, x=100)
cube.collapse_spectral(method="mean")
cube.extract_slice(wavelength=6563.0, width=5)
cube.to(device)
```

### `SpectralAxis`

Fields: `values`, `unit`, `type`, `rest_frequency`, `redshift`. Methods: `.to_wavelength()`, `.to_frequency()`.

---

## Datasets & DataLoaders

### `FITSDataset`

Map-style [PyTorch Dataset](https://pytorch.org/docs/stable/data.html#map-style-datasets) for random-access to FITS images.

```python
from torchsky import FITSDataset
from torchsky.transforms import Compose, ZScale, RandomCrop

ds = FITSDataset(file_paths, hdu="auto",
    transform=Compose([RandomCrop(256), ZScale()]),
    device="cuda", mmap="auto")
sample = ds[0]
```

### `IterableFITSDataset`

Streaming [IterableDataset](https://pytorch.org/docs/stable/data.html#iterable-style-datasets) for large file collections with multi-worker support.

### `TableChunkDataset`

Iterable dataset yielding table row chunks.

```python
torchsky.TableChunkDataset(file_paths, hdu=1, columns=None, chunk_rows=10000,
                             device="cpu", transform=None, include_header=False)
```

### DataLoader helpers

```python
torchsky.create_dataloader(dataset, batch_size=32, num_workers=4, shuffle=True, **kwargs)
torchsky.create_fits_dataloader(file_paths, *, hdu="auto", transform=None, device="cpu",
                                  batch_size=32, num_workers=4, **kwargs)
torchsky.create_table_dataloader(file_paths, *, hdu=1, chunk_rows=10000,
                                   columns=None, device="cpu", **kwargs)
```

---

## Transforms (`torchsky.transforms`)

All transforms accept and return a `Tensor`. GPU-safe.

| Transform | Description |
|---|---|
| `ZScale(contrast, max_reject)` | Quantile-based normalization to [0, 1] |
| `AsinhStretch(a, Q)` | Asinh stretch for high dynamic range |
| `LogStretch(a)` | Logarithmic stretch |
| `PowerStretch(gamma)` | Power-law stretch |
| `Normalize(mean, std)` | Z-score normalization (auto-computes if None) |
| `MinMaxScale()` | Scale to [0, 1] |
| `RobustScale(quantile)` | IQR-based normalization |
| `RandomCrop(size)` | Random crop |
| `CenterCrop(size)` | Center crop |
| `RandomFlip(horizontal, vertical, p)` | Random flip |
| `RandomRotation(degrees)` | Random rotation |
| `GaussianNoise(std)` | Additive Gaussian noise |
| `PoissonNoise(scale)` | Poisson noise augmentation |
| `PerturbByError(scale)` | Gaussian noise scaled by error map |
| `RedshiftShift(z_range)` | Spectral redshift augmentation |
| `ToDevice(device)` | Move to device |
| `Compose([...])` | Sequential pipeline |

**Convenience pipelines:**

```python
from torchsky.transforms import (create_training_transform,
                                   create_validation_transform,
                                   create_inference_transform)

train_t = create_training_transform(crop_size=256, normalize=True, augment=True)
val_t   = create_validation_transform(crop_size=256, normalize=True)
inf_t   = create_inference_transform(normalize=True)
```

---

## Batch Utilities

```python
torchsky.read_batch(file_paths, hdu=0, device="cpu")  # → list[Tensor]
torchsky.get_batch_info(file_paths)                   # → {"num_files": N, "valid_files": M}
```

---

## Cache & Runtime

```python
torchsky.configure_for_environment()          # auto-tune buffers/handles

torchsky.get_cache_stats()                    # Python-side hit/miss stats
torchsky.get_cache_performance()             # extended cache statistics
torchsky.clear_file_cache(data=True, handles=True, meta=True, cpp=True)
torchsky.clear_cache()
torchsky.clear_buffers()
torchsky.get_buffer_stats()
torchsky.configure_buffers()
```

---

## Spherical Geometry & HEALPix

**Canonical imports:** use **`torchsky.coords`** for WCS-class transforms and HEALPix indexing helpers (`ang2pix`, `pix2ang`, `query_disc`, …) when building coordinate-centric pipelines. The same symbols are also available on **`torchsky.geom`** and **`torchsky.sphere`** for geometry- and healpy-shaped workflows; pick one style per subsystem and stay consistent to avoid duplicate imports of the same underlying `torchsky.wcs.healpix` implementation.

See [sphere.md](sphere.md) for the complete `torchsky.sphere` reference.

---

## See Also

- [Sphere & HEALPix reference](sphere.md)
- [Examples](examples.md)
- [Benchmarks](benchmarks.md)
- [Changelog](changelog.md)
