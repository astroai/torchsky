# Map API reference

`torchsky.map` is the coadd and map-product layer. The [map guide](map.md)
covers value semantics, kernels, and workflows; this page is the reference for
the promoted facade and the implementation submodules.

The facade is intentionally small: 39 names. Everything else lives in an
explicit submodule and is imported from there.

---

## The public facade

```python
from torchsky.map import Observation, coadd, plan
```

### Configuration and results

| Name | Role |
|---|---|
| `Observation`, `IFUObservation` | Image-plane and physical IFU detector observations with explicit WCS, noise, DQ, spectral, and provenance contracts. |
| `Target`, `CubeTarget` | Planned target grids (2D and 3D spatial-spectral). |
| `ResampleConfig` | Resampling kernel, drizzle pixfrac, output value semantics. |
| `SkyConfig` | Per-exposure, overlap, planar, or mesh sky matching. |
| `OutlierConfig` | AstroDrizzle / stcal-compatible outlier detection settings. |
| `DQPolicy`, `DQRejectionLog`, `QualityProduct` | Input-bit policy, rejection audit trail, and propagated quality products. |
| `AdaptiveResampleConfig` | Error bounds, subdivision limits, and value semantics for adaptive reprojection. |
| `StackConfig` | Stacking method and uncertainty propagation settings. |
| `MapResult`, `CubeResult`, `MultibandResult`, `HealpixResult` | Product results with data, weight, header, and provenance. |
| `ApertureUncertainty` | Exact detector variance for one finite weighted aperture. |
| `ProductSpec`, `BandpassSpec`, `ProductResult`, `UncertaintyProduct` | Multi-band product specifications and results. |
| `TargetRenderSpec` | Render contract shared with the scene facade. |

### Planning and coaddition

| Name | Role |
|---|---|
| `plan(observations, ...)` | Plan a 2D target grid covering the observations. |
| `plan_cube(...)` | Plan a 3D spatial-spectral target cube. |
| `plan_ifu_cube_response(...)` | Build the physical spatial/spectral IFU response plan. |
| `plan_product(...)` | Plan a multi-band product from observations. |
| `coadd(observations, target, ...)` | Unified 2D mosaic/coadd entry point. |
| `coadd_multiband(...)` | Multi-band image-plane coadd. |
| `coadd_cube(...)` | Spectral cube from 2D filter bands or 3D IFU slices. |
| `build_ifu_cube(...)` | Build a footprint-aware IFU cube with DQ, variance components, context, and covariance actions. |
| `coadd_healpix(...)` | Direct coadd onto a dense or sparse HEALPix grid. |
| `stack_product(...)` | Stack bands into a labeled product. |
| `resample(...)` | Project observations onto a target grid without stacking. |
| `resample_healpix(...)` | Project a HEALPix map (or `HealpixResult`) onto a WCS grid. |
| `blot(...)` | Inverse-project a clean coadd back into exposure frames. |
| `detect_outliers(...)` | Cosmic-ray / transient detection via blot + derivative tests. |
| `match_sky(...)` | Solve and subtract sky background across exposures. |
| `aperture_uncertainty(...)` | Aperture variance from the map and weight. |
| `adaptive_reprojection_response(...)` | Construct an opt-in bounded adaptive response with exact adjoint and uncertainty propagation. |

---

## Implementation submodules

Import lower-level surfaces from the module that owns them.

| Module | Ownership |
|---|---|
| `torchsky.map.image_plane` | Planning, projection, coadd kernels, outlier and sky machinery. |
| `torchsky.map.cube` | Spectral cube assembly (`coadd_cube`) and the IFU 3D path. |
| `torchsky.map.healpix_coadd` | Direct-to-HEALPix coadd and resample. |
| `torchsky.map.product` | Product planning, band specs, stacking, uncertainty products. |
| `torchsky.map.observation` | `Observation`, targets, configs, and result types. |
| `torchsky.map.operators` | Observation operators, PSF specs, resampling kernels, value semantics. |
| `torchsky.map.solvers` | Regularized/linear solver hooks and preconditioners. |
| `torchsky.map.exposure_pose` | Exposure pose, kinematics, and splat geometry. |
| `torchsky.map.field` | Continuous field models and observation adapters. |
| `torchsky.map.likelihood`, `torchsky.map.query`, `torchsky.map.evidence` | Native likelihood, linear queries, and evidence (experimental). |
| `torchsky.map.healpix_spectral`, `torchsky.map.healpix_image_plane` | HEALPix ↔ image-plane sampling helpers. |
| `torchsky.map.splat`, `torchsky.map.sky`, `torchsky.map.target` | Splatting kernels, sky models, and render specs. |

---

## Value semantics

`ResampleConfig`/`BandpassSpec` carry the value semantics explicitly; there is
no implicit convention switch:

- `"surface_brightness"` keeps intensity per unit solid angle.
- `"flux"` multiplies by pixel area (arcsec² / sr) on planar grids.

Weights are inverse variance by default; `Observation` distinguishes total
weight from `weight_is_inverse_variance` and variance components
(`var_poisson`, `var_rnoise`, `var_flat`). See [map.md](map.md) for the full
contract and [psf-response-conventions.md](psf-response-conventions.md) for
detector-grid orientation rules.
