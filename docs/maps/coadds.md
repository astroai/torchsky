# Coadds
<!-- TODO(0.5.0 docs): intro -->


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
stored in the plan as inverse-variance evidence. See [CLI](../introduction/cli.md).

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
(see [Fields](../models/fields.md)).

## Memory And Scale

`coadd_image_plane`, the engine behind `coadd`, tiles the output automatically
when the working set would exceed a memory budget (30% of free device memory on
CUDA, of available host memory on CPU). Only the tiles an exposure reaches are
processed, and exposures accumulate one at a time. Override with
`tile_memory_budget_bytes=`, `TORCHSKY_TILE_MEMORY_BUDGET_BYTES`, or
`auto_tile=False`. Exact median and sigma-clipped stacks and overlap-background
rectification use capacity-checked scratch storage (`scratch_dir=`) whose growth
is linear in exposure count. See [Scaling and tuning](../performance/scaling.md) and
the [measured speeds](../performance/highlights.md).

Schema-v2 plans validate input identities, target shape and header agreement,
kernel and stack choices, and runtime device and seeds before replay; replay
recomputes identities and permits only output and scratch-location overrides.
