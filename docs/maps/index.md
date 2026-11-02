# Images and maps: overview
<!-- TODO(0.5.0 docs): intro -->


`torchsky.map` is the mapmaking facade: it plans a target grid, projects
calibrated exposures onto it, and stacks them into a declared product. It also
covers [spectral cubes](cubes.md) and [HEALPix maps](healpix-mapmaking.md). SWarp,
Montage, STScI drizzle, and `reproject` are comparators, never runtime
dependencies.

Use it when the unit of work is a requested sky product rather than a single
image. Continuous fields live in [`torchsky.field`](../models/fields.md).

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
  [kernel table](resampling.md#kernels).

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
[`torchsky.map` reference](../reference/map.md).

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
The [speed evidence](../performance/highlights.md) and
[parity summary](../performance/parity.md) link back to it. Methodology is in the
[harness benchmark guide](https://github.com/astroai/torchsky-harness/blob/main/docs/benchmarks.md).

Example scripts:

- [Dithered SIP coadd](../published-examples/coadd_dithered_sip.py)
- [Drizzle flagship](../published-examples/drizzle_flagship.py)
- [Coadd to HEALPix](../published-examples/coadd_to_healpix.py)
- [Sphere image inverse solve](../published-examples/sphere_image_inverse.py)
- [Survey split](../published-examples/survey_split.py)

See [Examples](../examples/index.md) for the full index.
