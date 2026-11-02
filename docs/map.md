# Coadds and resampling

`torchsky.map` is the mapmaking facade: it plans a target grid, projects
calibrated exposures onto it, and stacks them into a declared product. It also
covers [spectral cubes](cubes.md) and [HEALPix maps](healpix-maps.md). SWarp,
Montage, STScI drizzle, and `reproject` are comparators, never runtime
dependencies.

Use it when the unit of work is a requested sky product rather than a single
image. Continuous fields live in [`torchsky.field`](field.md).

## Core Concepts

- **`Observation`** is one calibrated input: pixel data (a tensor or a path), a
  WCS header, and optional weight or variance maps, a DQ mask (`mask`,
  `good_bits`), photometric scale, background offset, IFU wavelength map, DCR
  parameters, or a moving-target `tracking_rate`.
- **Target** (`plan`, `Target`, `CubeTarget`, `ProductSpec`) is the output grid:
  WCS, shape, and bands.
- **Weights.** `weight` always controls the coadd denominator. It is treated as
  detector inverse variance only with `weight_is_inverse_variance=True`.
  Default, support, and mask-only weights produce no uncertainty object.
  `torchsky.quality.build_weight_product` attaches the inverse-variance,
  confidence, and mask-polarity semantics when you build masks yourself.
- **Value semantics.** `surface_brightness` and `flux` products use different
  area-scaling rules. Choose before comparing against another stack; mismatches
  with SWarp or Montage usually trace to value semantics, pixel-area treatment,
  mask policy, or weights, not I/O.
- **Kernel** is explicit configuration (`ResampleConfig(kernel=...)`) with
  different support, interpolation, and flux behaviour. See the
  [kernel table](#kernels).

## Coadd Images Onto A Shared Grid

<!-- docs: run -->
```python
import torch

from torchsky.map import Observation, ResampleConfig, StackConfig, coadd, plan

header = {
    "NAXIS": 2, "NAXIS1": 64, "NAXIS2": 64,
    "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRPIX1": 32.5, "CRPIX2": 32.5, "CRVAL1": 150.0, "CRVAL2": 2.0,
    "CDELT1": -1e-3, "CDELT2": 1e-3,
}
gen = torch.Generator().manual_seed(1)
observations = [
    Observation(
        data=torch.randn(64, 64, generator=gen) + 10.0,
        wcs={**header, "CRVAL1": 150.0 + shift},
        weight=torch.full((64, 64), 4.0),
        weight_is_inverse_variance=True,
    )
    for shift in (0.0, 0.01, 0.02)
]

target = plan(observations, projection="TAN")
result = coadd(
    observations,
    target,
    resample=ResampleConfig(kernel="bilinear", semantics="surface_brightness"),
    stack=StackConfig(method="weighted_mean", tile_size=1024),
)

assert result.data.shape == target.shape
covered = result.weight > 0
assert abs(float(result.data[covered].mean()) - 10.0) < 0.5
assert result.inverse_variance is not None
```

With real files, pass paths instead of tensors, for example
`Observation("img_a.fits", weight_path="img_a_wht.fits")`; `torchfits` does the
reading. `MapResult` carries `data`, `weight`, `header`, `context`,
`inverse_variance`, quality planes, and provenance.

`torchsky map` on the command line records the same distinction: omitting
`--weights` uses unit relative weights, while an explicit `--weights 1.0` is
stored in the plan as inverse-variance evidence. See [CLI](cli.md).

## Plan A Product Once, Stack Many Times

`ProductSpec` declares the output grid, bands, and semantics once so several
calls produce comparable products.

<!-- docs: run -->
```python
import torch

from torchsky.map import BandpassSpec, Observation, ProductSpec, plan, stack_product

header = {
    "NAXIS": 2, "NAXIS1": 32, "NAXIS2": 32,
    "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRPIX1": 16.5, "CRPIX2": 16.5, "CRVAL1": 150.0, "CRVAL2": 2.0,
    "CDELT1": -1e-3, "CDELT2": 1e-3,
}
observations = [
    Observation(data=torch.full((32, 32), value), wcs=header) for value in (1.0, 3.0)
]
target = plan(observations, projection="TAN")
product = ProductSpec(
    target_header=target.header,
    bands=(BandpassSpec(label="g"),),
    value_semantics="surface_brightness",
)
stacked = stack_product(observations, product, resampling_kernel="bilinear")
assert torch.allclose(stacked.data[stacked.weight > 0].mean(), torch.tensor(2.0), atol=1e-4)
```

## Uncertainty

When noise is explicit, direct copy, drizzle, nearest/bilinear/bicubic,
Lanczos/cubic, conservative polygon, and point/box splat paths propagate
diagonal variance with squared geometric coefficients. FFT and spectral
remappers return an absent uncertainty rather than a diagonal proxy.

For a finite aperture, `aperture_uncertainty` evaluates the exact detector-noise
variance with the adjoint of the same projection, including the covariance of the
requested aperture, without materializing a map covariance matrix.

<!-- docs: run -->
```python
import torch

from torchsky.map import (
    Observation, ResampleConfig, StackConfig, aperture_uncertainty, coadd, plan,
)

header = {
    "NAXIS": 2, "NAXIS1": 64, "NAXIS2": 64,
    "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRPIX1": 32.5, "CRPIX2": 32.5, "CRVAL1": 150.0, "CRVAL2": 2.0,
    "CDELT1": -1e-3, "CDELT2": 1e-3,
}
observations = [
    Observation(
        data=torch.ones(64, 64),
        wcs={**header, "CRVAL1": 150.0 + shift},
        weight=torch.full((64, 64), 4.0),
        weight_is_inverse_variance=True,
    )
    for shift in (0.0, 0.01, 0.02)
]
target = plan(observations, projection="TAN")
resample = ResampleConfig(kernel="drizzle_square", pixfrac=0.8)
result = coadd(observations, target, resample=resample, stack=StackConfig(method="weighted_mean"))

aperture = torch.zeros_like(result.data)
aperture[20:30, 20:30] = 1.0
noise = aperture_uncertainty(observations, result, aperture, resample=resample)
assert float(noise.variance) > 0.0
```

It requires explicit inverse variance and matching execution provenance and
fails closed for median or clipped products, data-dependent background fits, and
FFT or spectral remappers. Aperture weights may be signed.

Deterministic field products return no uncertainty rather than a unit proxy.
Models with an explicit likelihood variance emit a `model_predictive` product
(see [Fields](field.md)).

## Memory And Scale

`coadd_image_plane`, the engine behind `coadd`, tiles the output automatically
when the working set would exceed a memory budget (30% of free device memory on
CUDA, of available host memory on CPU). Only the tiles an exposure reaches are
processed, and exposures accumulate one at a time. Override with
`tile_memory_budget_bytes=`, `TORCHSKY_TILE_MEMORY_BUDGET_BYTES`, or
`auto_tile=False`. Exact median and sigma-clipped stacks and overlap-background
rectification use capacity-checked scratch storage (`scratch_dir=`) whose growth
is linear in exposure count. See [Scaling and tuning](evaluation/scaling.md) and
the [measured speeds](evaluation/highlights.md).

Schema-v2 plans validate input identities, target shape and header agreement,
kernel and stack choices, and runtime device and seeds before replay; replay
recomputes identities and permits only output and scratch-location overrides.

## Kernels

| Kernel family | Typical analogue | Notes |
|---|---|---|
| `nearest`, `bilinear` | Nearest and linear regridding | Baseline for compatibility checks |
| `bicubic`, `catmull_rom`, `mitchell_netravali` | Cubic families | Polynomial definitions differ across tools |
| `lanczos2` to `lanczos5` | Lanczos mosaickers (SWarp) | Wider support, higher cost |
| `fft_shift`, `fft_diagonal_affine`, `fft_linear_affine`, `periodic_linear_affine` | Restricted FFT remappers | Same-shape linear frames with explicit boundaries |
| `drizzle_point`, `drizzle_square`, `drizzle_turbo`, `drizzle_gaussian` | STScI drizzle drop kernels | Native; square drizzle uses exact box overlap |
| `conservative_polygon` | Conservative overlap | Native overlap remapper |
| `splat_point`, `splat_box` | Forward deposition | Approximate differentiable splats |

Drizzle conventions. `drizzle_square` and `conservative_polygon` overlap each
projected source-pixel polygon exactly with the target pixels; `drizzle_turbo` and
`splat_box` use the separable box overlap when the local source-to-target Jacobian
maps axes onto axes (to float32 resolution) and the exact polygon overlap for rotated
or sheared pairs; IFU cube drizzle scales each voxel footprint by the local
detector-to-cube Jacobian in x, y and wavelength. `drizzle_point` differs from STScI
`drizzle` 3.0 at the grid edge: torchsky keeps drops whose centre lies in the outer
half-pixel rim (target coordinates in `[-0.5, n - 0.5]`), whereas STScI 3.0 drops those
outside `[0, n - 1]`; this is intentional so no drop overlapping the grid is discarded.

Target-to-source mapping uses a verified interpolated lattice for SIP and TPV;
`mapping_mode("exact")` selects the per-pixel solve.

## Namespace Map

| Concept | Names |
|---|---|
| Inputs and targets | `Observation`, `ImageObservation`, `IFUObservation`, `Target`, `CubeTarget`, `ProductSpec`, `BandpassSpec` |
| Configuration | `ResampleConfig`, `SkyConfig`, `OutlierConfig`, `StackConfig` |
| Planning | `plan`, `plan_cube`, `plan_product` |
| Operations | `resample`, `resample_healpix`, `blot`, `detect_outliers`, `match_sky`, `coadd`, `coadd_multiband`, `coadd_cube`, `coadd_healpix`, `coadd_image_plane`, `aperture_uncertainty`, `stack_product` |
| Results | `MapResult`, `CubeResult`, `MultibandResult`, `HealpixResult`, `ApertureUncertainty`, `ProductResult`, `UncertaintyProduct` |

Reusable observation operators live in `torchsky.map.operators`: an
`ObservationOperatorBundle` is the shared observation law for image-plane and
field paths, and constrained learned instrument effects use
`ConstrainedKernelPSF`, `AffineRadiometry`, `SubPixelShift`, and
`LearnedInstrumentOperator`. Continuous fields live in `torchsky.field`. The full signatures are in the
[`torchsky.map` reference](reference/map.md).

## Device placement

Map APIs compute and return on the device of their tensor inputs (for example
`Observation(data=tensor)`); CPU tensors stay on CPU even when CUDA is
available. An explicit `device=` always wins, and tensors on different devices
without an explicit `device=` raise `ValueError`. Only when there is no tensor
to follow (observations given as file paths) does
`torchsky.device.default_compute_device()` apply: CUDA when available, or
`TORCHSKY_DEVICE`. Float64-only kernels (exact drizzle, conservative polygon
and splat projection, spectral resamplers, encoder field models) raise
`NotImplementedError` on MPS, which has no float64; run them on CPU or CUDA.

## Evaluation And Examples

Truth metrics, uncertainty scoring, comparator protocols, and evaluation reports
belong to [`torchsky-harness`](https://github.com/astroai/torchsky-harness); this
repository keeps bounded product, gradient, adjoint, and device-contract tests.
The [speed evidence](evaluation/highlights.md) and
[parity summary](evaluation/parity.md) link back to it. Methodology is in the
[harness benchmark guide](https://github.com/astroai/torchsky-harness/blob/main/docs/benchmarks.md).

Example scripts:

- [Dithered SIP coadd](published-examples/coadd_dithered_sip.py)
- [Drizzle flagship](published-examples/drizzle_flagship.py)
- [Coadd to HEALPix](published-examples/coadd_to_healpix.py)
- [Sphere image inverse solve](published-examples/sphere_image_inverse.py)
- [Survey split](published-examples/survey_split.py)

See [Examples](examples.md) for the full index.
