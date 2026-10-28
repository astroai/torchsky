# Runtime contract

Conventions every `torchsky` surface follows. These are contracts, not
recommendations: code in the guides and examples relies on them, and the package
tests assert the important ones.

## Tensors, devices, and dtypes

- Public compute paths accept and return `torch.Tensor`. Nothing moves data
  between devices or to NumPy implicitly; if you pass CUDA or MPS tensors, the
  result stays there, and if you pass CPU tensors it stays on the CPU.
- Kernels follow PyTorch dtype promotion. Passing `float64` coordinates keeps
  `float64` precision through the transform; `float32` inputs are computed in
  `float32` except where a documented step (for example an astrometric fit)
  promotes internally and returns the promoted dtype.
- Integer pixel indices are converted to the floating dtype of the surrounding
  computation; output pixel coordinates follow the coordinate dtype.
- The compiled `torchsky._cpp` extension is optional. Without it, the same
  Python/Torch kernels run with identical semantics and lower throughput. Set
  `TORCHSKY_DISABLE_CPP=1` to force the fallback when diagnosing a native crash;
  do not publish native performance numbers from such a run. See
  [installation](../install.md).

## Coordinates and pixels

- **Pixel origin.** `WCS.pixel_to_world` and `WCS.world_to_pixel` use 0-based
  pixel coordinates. The `pixel_to_world_values` / `world_to_pixel_values`
  forms accept `origin=0` (default) or `origin=1` for FITS-style arrays.
- **World units.** Celestial world coordinates are degrees. Spectral and time
  axes keep the units of the FITS header metadata; no silent unit conversion is
  performed.
- **Axis dispatch.** A `WCS` constructed from a header transforms only the axes
  present. Multi-axis transforms return one tensor per axis; a single-axis WCS
  returns the world array directly. See [`WCS`](wcs.md#torchsky.wcs.WCS).

## HEALPix

- `nside` must be a power of two. `nest=False` selects RING ordering, `nest=True`
  selects NESTED; the default is RING.
- `lonlat=False` means `(theta, phi)` in radians; `lonlat=True` means
  `(lon, lat)` in degrees.
- Query functions (`query_disc`, `query_polygon`, …) return pixel indices by
  default; pass `fact`/`inclusive` where the function exposes them to control
  subsampling and boundary inclusion.
- See [`torchsky.sphere`](sphere.md) and the
  [HEALPix kernels](sphere-healpix.md).

## Map value semantics

`torchsky.map` never guesses whether a product is flux or surface brightness:

- `ResampleConfig.kernel` defaults to `"bilinear"`, `pixfrac=1.0`.
- `ResampleConfig.semantics` defaults to `"surface_brightness"`. Passing
  `"flux"` multiplies by the target pixel area; the choice is recorded in the
  result provenance.
- Weights are **inverse variance** unless the observation declares otherwise
  (`Observation.weight_is_inverse_variance`). Variance components
  (`var_poisson`, `var_rnoise`, `var_flat`) stay separate.
- `ResampleConfig.spectral_kernel` defaults to `"linear"`; cube workflows also
  accept `"drizzle"`, `"emsm"`, `"msm"`, and `"nearest"`.
- See [`torchsky.map`](map.md) and the [map guide](../map.md).

## Quality and masks

`torchsky.quality` carries its semantics with the product instead of relying on
convention:

- `INVERSE_VARIANCE_SEMANTICS = "inverse_variance_per_pixel"`
- `CONFIDENCE_SEMANTICS = "normalized_weight_0_to_1"`
- `MASK_POLARITY = "set_means_flagged"` (a set bit means the pixel is flagged)
- `CONTRACT_VERSION = "torchsky.quality.v1"`

See [`torchsky.quality`](quality.md).

## Errors

Invalid state fails closed: unsupported projections, missing WCS axes, malformed
tabular metadata, mismatched frames, and out-of-range indices raise a typed
exception rather than returning a silent fallback. The exported exception types
are listed in [errors and exceptions](exceptions.md).

## Optional dependencies

`torchfits` is a required runtime dependency at the package boundary, but
`import torchsky` does not open files. Optional extras (`viz`, `map`, `ephem`,
`tle`, `gwcs`) are never imported by default; each guide states the extra it
needs. `torchsky` never imports `astropy`, `healpy`, `hpgeom`, `healsparse`,
`mocpy`, or `spherical_geometry` on the default path.
