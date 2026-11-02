# HEALPix mapmaking
<!-- TODO(0.5.0 docs): intro -->


Image-plane coadds answer "what does this grid look like?". A HEALPix map
answers "what does the sky look like?", independent of any single pointing.
`torchsky` projects calibrated images directly onto a HEALPix grid, keeps the
accumulated weight and coverage, and can resample a spherical map back onto an
image WCS for comparison with cutouts.

The entry points are [`coadd_healpix`](../reference/map.md#torchsky.map.coadd_healpix)
and [`resample_healpix`](../reference/map.md#torchsky.map.resample_healpix).
Both accept dense tensors; the coadd additionally supports sparse coverage
through `torchsky.sphere.SparseHealpixMap`.

## Dense or sparse?

| Situation | Choice | Why |
|---|---|---|
| Small fraction of the sky, high resolution | `sparse=True` | Only covered pixels are allocated. |
| All-sky or broad survey field | `sparse=False` | Dense tensors are simpler to index and pass to SHT code. |
| Repeated cutouts over one region | `sparse=True` with a matching `nside_coverage` | Accumulation keeps only merged support. |
| Downstream harmonic analysis | Dense | `map2alm` consumes a full `npix` array. |

`nside_coverage` sets the resolution of the sparse coverage index; it must be
coarse enough that the covered set fits comfortably in memory. It does not
change the map resolution, which is `nside`.

## Coadd images onto HEALPix

<!-- docs: run -->
```python
import torch

from torchsky.map import Observation, coadd_healpix

HEADER = {
    "NAXIS": 2,
    "NAXIS1": 32,
    "NAXIS2": 32,
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CRPIX1": 16.5,
    "CRPIX2": 16.5,
    "CRVAL1": 150.0,
    "CRVAL2": 2.0,
    "CDELT1": -0.02,
    "CDELT2": 0.02,
}

observations = [
    Observation(
        data=torch.full((32, 32), value, dtype=torch.float32),
        wcs=HEADER,
        weight=1.0,
        weight_is_inverse_variance=True,
    )
    for value in (1.0, 2.0)
]

result = coadd_healpix(
    observations,
    nside=64,
    nest=False,
    kernel="bilinear",
    device="cpu",
)

covered = result.weight > 0
print("nside:", result.nside, "npix:", result.data.numel())
print("covered pixels:", int(covered.sum()))
print("mean over covered pixels:", round(float(result.data[covered].mean()), 3))
print("coverage MOC entries:", int(result.coverage_moc.uniq.numel()))
```

`nside` must be a power of two. `nest=False` (the default) selects RING
ordering; pass `nest=True` when the downstream code expects NESTED. The
resampling kernel is explicit (`bilinear`, `nearest`, `bicubic`, `lanczos2`,
`lanczos3`, `lanczos4`).

[`HealpixResult`](../reference/map.md#torchsky.map.HealpixResult) carries
`data`, `weight`, `nside`, `nest`, `sparse`, `overlap_count`, and a coverage
`MOC`. `result.image` is an alias for `data`; `result.sparse_map` returns the
`SparseHealpixMap` when `sparse=True` and `None` otherwise.

## Project a HEALPix map back onto an image

To compare a spherical product with an exposure, or to make a cutout for
display, resample onto a target header:

<!-- docs: run -->
```python
import torch

from torchsky.map import Observation, coadd_healpix, resample_healpix

HEADER = {
    "NAXIS": 2,
    "NAXIS1": 32,
    "NAXIS2": 32,
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CRPIX1": 16.5,
    "CRPIX2": 16.5,
    "CRVAL1": 150.0,
    "CRVAL2": 2.0,
    "CDELT1": -0.02,
    "CDELT2": 0.02,
}

observation = Observation(
    data=torch.ones((32, 32), dtype=torch.float32),
    wcs=HEADER,
    weight=1.0,
)
result = coadd_healpix([observation], nside=64, device="cpu")

image = resample_healpix(
    result.data,
    HEADER,
    nside=result.nside,
    method="bilinear",
    device="cpu",
)
print("image shape:", tuple(image.data.shape))
print("pixels with weight:", int((image.weight > 0).sum()))
```

`resample_healpix` also accepts a [`HealpixResult`](../reference/map.md#torchsky.map.HealpixResult)
(or a sparse map) and a [`Target`](../reference/map.md#torchsky.map.Target),
[`WCS`](../reference/wcs.md#torchsky.wcs.WCS), or header mapping as
`target`. The returned [`MapResult`](../reference/map.md#torchsky.map.MapResult)
has the same data/weight/header contract as an image-plane coadd. `method`
selects `bilinear`, `nearest`, or `harmonic`; the harmonic path is the right
choice only when the map and target are both well sampled and you want
band-limited interpolation.

## Sparse accumulation

For a survey-sized region, build a sparse product instead of allocating 12 ×
`nside²` pixels (continuing the example above):

```python
# fragment: continues from the example above
result = coadd_healpix(
    observations,
    nside=4096,
    sparse=True,
    nside_coverage=32,
    kernel="bilinear",
)
sparse_map = result.sparse_map
print(sparse_map.nside, sparse_map.nside_coverage)
```

In sparse mode `result.data` is a `SparseHealpixMap`; access pixels through the
map's own indexing, and use `result.coverage_moc` to describe the covered sky to
other tools. `reduction_chunk_size` bounds how many sampled contributions are
retained before each incremental reduction. The default (262,144) is a good
starting point; lower it when many images overlap the same high-resolution
pixels.

## Value semantics and uncertainty

- `Observation.weight` controls the coadd denominator. It is treated as detector
  inverse variance only when `weight_is_inverse_variance=True`; otherwise it is
  a relative weight and no inverse-variance product is produced.
- `coadd_healpix` inherits the `ResampleConfig` value semantics from the map
  layer: `surface_brightness` (default) or `flux`. Spherical products usually
  want `surface_brightness`.
- Uncertainty propagation follows the same rule as image coadds: explicit
  inverse variance in, diagonal variance out. Spherical resampling does not
  materialise a cross-pixel covariance matrix.

## Performance notes

- The HEALPix coadd samples images in tiles (`reduction_chunk_size`) and writes
  each chunk directly into the output tensors in dense mode, so peak memory is
  the map plus one chunk.
- Sparse accumulation grows with the covered sky, not with `npix`.
- `nside=4096` is 201 million pixels; use `sparse=True` unless you genuinely need
  the full sphere at that resolution.

## Related

- [Mapmaking](index.md) for image-plane coadds and product specs.
- [Sphere and HEALPix](../sphere/index.md) for indexing, queries, MOC, and harmonic
  transforms.
- [Spectral cubes](cubes.md) for the spatial-spectral product path.
