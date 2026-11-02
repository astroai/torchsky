# Cubes and IFU
<!-- TODO(0.5.0 docs): intro -->


A spectral cube is a spatial map with a third, spectral axis. `torchsky` builds
it from either a set of 2D filter-band images or 3D IFU detector frames, and
keeps the two paths behind one entry point: [`coadd_cube`](../reference/map.md#torchsky.map.coadd_cube).
The target grid, spectral bin edges, resampling kernel, and value semantics are
always explicit.

## Which path am I on?

| Input | Detector shape | Path | Notes |
|---|---|---|---|
| Filter-band images | `(y, x)` each, one wavelength per exposure | Image-plane cube path | Uses each observation's `wavelength` (or `wavelength_map`) to assign flux to spectral bins. |
| IFU frames with per-pixel wavelengths | `(y, x)` plus `ifu_wcs` or `wavelength_map` | 3D IFU path | Spectral kernel is restricted to `drizzle`, `emsm`, `msm`, or `nearest`. |
| Physical IFU product | [`IFUObservation`](../reference/map.md#torchsky.map.IFUObservation) | [`build_ifu_cube`](../reference/map.md#torchsky.map.build_ifu_cube) | Carries detector footprints, slice ids, and variance components. |
| Multi-band product, one plane per band | 2D images with band metadata | [`stack_product`](../reference/map.md#torchsky.map.stack_product) / `coadd_multiband` | Produces a labelled product rather than a physical cube. |

If the observations mix `Observation` and `IFUObservation`, `coadd_cube` fails
closed: a cube has one detector model.

## Build a cube from filter bands

This is the common case: several images of the same field through different
filters, each with a representative wavelength. Synthetic data keeps the example
self-contained; the same call pattern applies to `torchfits`-loaded images.

<!-- docs: run -->
```python
import torch

from torchsky.map import Observation, ResampleConfig, coadd_cube, plan_cube

HEADER = {
    "NAXIS": 2,
    "NAXIS1": 16,
    "NAXIS2": 16,
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CRPIX1": 8.5,
    "CRPIX2": 8.5,
    "CRVAL1": 150.0,
    "CRVAL2": 2.0,
    "CDELT1": -0.001,
    "CDELT2": 0.001,
}

bands = [5000.0, 6000.0, 7000.0]
observations = [
    Observation(
        data=torch.full((16, 16), 1.0 + 0.5 * index, dtype=torch.float32),
        wcs=HEADER,
        wavelength=wavelength,
        weight=1.0,
        weight_is_inverse_variance=True,
    )
    for index, wavelength in enumerate(bands)
]

target = plan_cube(
    observations,
    spectral_axis=bands,
    device="cpu",
)
cube = coadd_cube(
    observations,
    target,
    resample=ResampleConfig(semantics="surface_brightness"),
    device="cpu",
)

assert tuple(cube.data.shape) == (3, 16, 16)
central = [round(float(value), 2) for value in cube.data[:, 8, 8]]
assert central == [1.0, 1.5, 2.0], central
```

`plan_cube` derives a spatial grid covering the observations and records the
spectral axis and bin edges. When you already know the grid, build a
[`CubeTarget`](../reference/map.md#torchsky.map.CubeTarget) directly (continuing
the example above):

```python
# fragment: continues from the previous block
from torchsky.map import CubeTarget
from torchsky.wcs import WCS

target = CubeTarget(
    spatial_wcs=WCS(HEADER),
    spectral_bin_edges=[4500.0, 5500.0, 6500.0, 7500.0],
)

cube = coadd_cube(observations, target, device="cpu")
```

Plotting or writing the result goes through [`CubeResult`](../reference/map.md#torchsky.map.CubeResult):
`cube.data` is `(n_wave, y, x)`, `cube.weight` the matching weights,
`cube.spectral_axis` the target axis, and `cube.header` the FITS-like spatial
metadata. `cube.to_data_cube()` returns the spectral container
([`DataCube`](../reference/spectral.md#torchsky.spectral.DataCube)) without
copying semantics into a second object.

## Build a cube from IFU frames

An IFU frame carries a spatial coordinate and a wavelength per detector pixel.
`Observation` accepts that mapping through `ifu_wcs` (an `IFUSliceWCS`) or
`wavelength_map`; `coadd_cube` then routes to the 3D path automatically.

<!-- docs: run -->
```python
import torch

from torchsky.map import CubeTarget, Observation, ResampleConfig, coadd_cube
from torchsky.wcs import WCS
from torchsky.wcs.gwcs_sampling import IFUSliceWCS

HEADER = {
    "NAXIS": 2,
    "NAXIS1": 16,
    "NAXIS2": 16,
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CRPIX1": 8.5,
    "CRPIX2": 8.5,
    "CRVAL1": 150.0,
    "CRVAL2": 2.0,
    "CDELT1": -0.001,
    "CDELT2": 0.001,
}

n_wave, n_slice = 12, 16
rows = torch.arange(n_wave, dtype=torch.float64)
columns = torch.arange(n_slice, dtype=torch.float64)

ifu_wcs = IFUSliceWCS(
    ra_map=(150.0 + (columns - 7.5) * 0.001).unsqueeze(0).expand(n_wave, n_slice).clone(),
    dec_map=torch.full((n_wave, n_slice), 2.0, dtype=torch.float64),
    wavelength_map=(5000.0 + 2.0 * rows).unsqueeze(1).expand(n_wave, n_slice).clone(),
    slice_id_map=columns.to(torch.int64).unsqueeze(0).expand(n_wave, n_slice).clone(),
)

frame = Observation(
    data=torch.ones((n_wave, n_slice), dtype=torch.float32),
    ifu_wcs=ifu_wcs,
    weight=torch.ones((n_wave, n_slice), dtype=torch.float32),
    weight_is_inverse_variance=True,
)

target = CubeTarget(
    spatial_wcs=WCS(HEADER),
    spectral_bin_edges=[4980.0, 5000.0, 5020.0, 5040.0],
)
cube = coadd_cube(
    [frame],
    target,
    resample=ResampleConfig(spectral_kernel="drizzle", spectral_pixfrac=0.8),
)
assert cube.data.shape == (3, 16, 16)
```

For a physical instrument product, construct
[`IFUObservation`](../reference/map.md#torchsky.map.IFUObservation) with the
detector-to-world transform and optional `variance_components`, then use
[`build_ifu_cube`](../reference/map.md#torchsky.map.build_ifu_cube). That path
returns covariance and side products (DQ, usability mask, response plan) in
addition to the science cube.

## Spectral axis contracts

- Target axes are honoured as given, including non-uniform axes. Non-increasing
  axes are rejected.
- The spectral kernel interpolates in **index space**. On a strongly irregular
  axis this introduces a bias of order `(width_right - width_left) * dF/dlambda`
  near bin edges; it is small for regular or log-spaced axes. Prefer a regular
  axis, or `spectral_kernel="nearest"` for exact bin assignment.
- Spectral kernels: `linear` (filter bands), `drizzle` (IFU default), `emsm` and
  `msm` (surface-brightness conserving), `nearest`.
- Kernel choice, `spectral_pixfrac`, and `ResampleConfig.semantics`
  (`surface_brightness` or `flux`) are recorded in the result provenance; there
  is no implicit convention switch.

## Uncertainty and covariance

`CubeResult.inverse_variance` is diagonal voxel inverse variance, and it is
populated only when every input weight is declared inverse variance
(`weight_is_inverse_variance=True`). Resampling one detector sample into several
voxels induces cross-voxel covariance that is **not** stored.
`CubeResult.correlation_factor` reports the local first-to-squared coefficient
ratio as an aperture-noise diagnostic, not a covariance matrix. The IFU path can
additionally return an explicit covariance object; propagate that when you need
exact voxel correlations.

## Performance notes

- `plan_cube` and `coadd_cube` process observations one at a time; peak memory is
  bounded by the target cube plus one resampled detector image.
- `response_chunk_size`, `max_response_entries`, and `footprint_max_depth` in
  [`ResampleConfig`](../reference/map.md#torchsky.map.ResampleConfig) bound the
  IFU response tables. Lower them for memory-constrained devices; raise them when
  many IFU slices overlap one voxel.
- The image-plane path reuses one projection per exposure per band and stacks in
  output tiles (`StackConfig.tile_size`).

## Related

- [Mapmaking](index.md) for 2D coadds, products, and sky matching.
- [HEALPix maps](healpix-mapmaking.md) for spherical products.
- [Spectral data](../spectra/index.md) for `SpectralAxis`, flux-conserving rebinning,
  and writing cubes with `torchfits`.
