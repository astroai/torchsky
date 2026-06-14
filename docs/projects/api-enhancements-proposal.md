# API Enhancements Proposal for Torchsky

Based on the review of the examples, notebooks, and the student project guide, we propose the following API enhancements to improve usability, prevent common errors, and clean up student workflows.

## 1. Support for Non-Uniform Wavelength FITS Writing
Currently, `write_cube_fits` raises a hard `ValueError` if the spectral axis values are not uniformly spaced. This is because standard FITS WCS only supports linear coordinate grids (using `CDELT3`). However, in multi-band coaddition (like the JWST stack baseline), observations naturally have non-uniform band centers.

### Proposed Solution
Add an option `allow_nonuniform: bool = False` or `nonuniform_fallback: str = "custom_cards"` to `write_cube_fits`. When set:
- It falls back to writing a standard 3D FITS image where `CDELT3` represents the average or first spacing, but details the actual bandpass centers in custom headers (e.g., `WAVE0`, `WAVE1`, ...).
- Alternatively, it could fall back to a tabular WCS standard (`CTYPE3 = 'WAVE-TAB'`) if full coordinates compliance is needed.

### Status Note
Independent one-dimensional `-TAB` WCS reading and transforms are now available
through `WCS(header, tables=...)`, `TensorHDU.wcs`, and `torchsky.get_wcs(path)`.
The reader path now honors independent-axis `PVi_1`/`PVi_2` `EXTVER`/`EXTLEVEL`
table disambiguation and rejects non-independent `PVi_3` mappings explicitly.
The proposed `write_cube_fits(..., allow_nonuniform=...)` writer fallback is
still open and should build on that substrate rather than custom-only cards.

```python
# Proposed API signature
def write_cube_fits(
    path: str,
    cube: torch.Tensor,
    *,
    spatial_header: Dict[str, Any],
    spectral_axis: SpectralAxis,
    allow_nonuniform: bool = True,  # new parameter
    ...
)
```

---

## 2. Automatic Coordinate Cache Invalidations
Currently, `FieldObservation` caches WCS coordinate grids to avoid recomputing pixel-to-sky transforms. However, if a user changes the underlying image tensor shape or modifies header cards in-place, the cached coordinate grid becomes stale, leading to quiet bugs unless `observation.clear_coordinate_cache()` is called.

### Proposed Solution
Automatically invalidate the cache when relevant properties are mutated. We can wrap the coordinate cache in a descriptor or track mutations:
- Track the shape of `self.image` and clear cache if shape changes.
- Track changes to the `header` dictionary using a custom dictionary observer or property setter.

### Status Note
Implemented read-time invalidation for `FieldObservation` coordinate grids:
cached RA/Dec grids now carry a signature of image shape, requested grid shape,
and header card values. In-place header edits or image-shape changes recompute
the grid on the next `coordinate_grid(...)` call. Deep mutation observers are
not currently needed for the active call sites.

---

## 3. Unified Pose Refinement Flag in `fit_field_model`
Currently, to train with learnable alignment residuals (bundle adjustment), students must call `create_exposure_poses` manually and pass them to `fit_field_model`:

```python
poses = create_exposure_poses(field_observations, anchor_first=True)
result = fit_field_model(
    field_observations,
    model,
    observation_exposure_poses=poses,
    ...
)
```

### Proposed Solution
Introduce a high-level boolean parameter `refine_poses: bool = False` in `fit_field_model`. If set to `True`, the function automatically initializes exposure poses (anchoring the first one) and refines them jointly:

```python
# Proposed high-level call
result = fit_field_model(
    field_observations,
    model,
    refine_poses=True,
    steps=500,
    lr=5e-3,
)
```

### Status Note
Implemented. `fit_field_model(..., refine_poses=True)` now creates per-observation
`ExposurePose` modules, anchors the first pose by default, trains the remaining
pose parameters with the field model, returns them on `FieldFitResult.exposure_poses`,
and records a `diagnostics["pose_refinement"]` block. Explicit
`observation_exposure_poses=[...]` remains available for lower-level control and
is mutually exclusive with `refine_poses=True`.

---

## 4. Pseudo-RGB and Slice Visualizers in `torchsky.viz`
Students frequently need to construct pseudo-RGB composites and plot spectra at pixel centers to verify model fitting. Providing high-level plotting helpers would save boilerplate code.

### Proposed Solution
Add a `torchsky.viz` module with helper functions:
```python
from torchsky.viz import plot_pseudo_rgb, plot_pixel_spectrum

# Quick visualization of a rendered cube
plot_pseudo_rgb(cube, spectral_axis=axis, title="Pseudo-RGB Composite")
```
