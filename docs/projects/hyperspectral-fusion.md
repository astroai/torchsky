# Student Project: Hyperspectral Fusion with Neural Spectral Fields

One-line goal: turn several telescope images of the same sky into one deeper
spatial-plus-wavelength model of that sky.

First principle: start with things you can see and debug. Build a normal stack,
make quicklook images, then fit a neural field.

Near-term goal: make one small JWST-style patch reproducible, stacked, visualized,
fitted with an INR, and compared honestly.

Long-term goal: scale from one patch to a larger contiguous sky field.

## 0. Big Picture

This project has three layers:

1. **Data layer:** FITS files, WCS headers, filters, masks, and weights.
2. **Baseline layer:** a classical stack/coadd, including a drizzle-like baseline.
3. **Research layer:** a neural field that predicts intensity at `(RA, Dec, wavelength)`.

Keep those layers separate. If the data layer is wrong, the neural field will
learn the wrong thing.

## 1. If You Are New To Astro Data

Astronomical images are not just pictures. A calibrated FITS image usually has an
image array and a header. The header tells us where the pixels are on the sky.

| Term | What it means here |
|------|--------------------|
| FITS | The file format used for telescope images. It stores the image array plus a header dictionary. |
| Header | Metadata in the FITS file. For this project the important part is the WCS. |
| WCS | World Coordinate System: the mapping between pixel coordinates `(x, y)` and sky coordinates `(RA, Dec)`. |
| RA / Dec | Sky longitude and latitude, usually in degrees. They play the role of coordinates on the celestial sphere. |
| Band / filter | A telescope image integrated over a wavelength range, e.g. JWST/NIRCam `F090W` or `F444W`. |
| PSF | Point-spread function: how a point source is blurred by the telescope/instrument. |
| Weight / mask | Per-pixel confidence. Bad pixels and low-confidence pixels should count less in fits. |
| Stack / coadd | A conventional product made by reprojecting images onto one grid and combining them. |
| Cube | A 3D product `(wavelength, y, x)`: many sky images sampled along a spectral axis. |
| INR / neural field | A neural network that represents the sky as a continuous function instead of a fixed pixel grid. |

### What Is A Stack?

A stack, or coadd, is the normal astronomy baseline.

One-liner: put several images onto the same WCS grid and combine their pixels.

It answers: "What do we get if we trust the headers, put all images on one pixel
grid, and average them sensibly?"

### What Is Drizzle?

Drizzle is a standard HST/JWST-style way to combine dithered images. Instead of
just interpolating pixels, it drops small pixel "footprints" onto an output grid.

One-liner: drizzle is a careful, geometry-aware coadd baseline.

In `torchsky`, start with `resampling_kernel="bilinear"` because it is easy to
debug. Then run the drizzle-like baseline with `resampling_kernel="drizzle_square"`.

### What Is A Neural Field?

A neural field is a model of the sky as a function, not an image.

One-liner: ask the model for brightness at any `(RA, Dec, wavelength)`.

The classical stack asks, "What image do I get after resampling?" The INR asks,
"Can one continuous function explain all the observations directly?"

The research direction is a continuous sky model:

\[
F_\theta(\alpha, \delta, \lambda)
\]

where a telescope image is predicted by querying the sky through its WCS and
integrating over the filter throughput:

\[
I_{\mathrm{pred}}(x, y) =
\int F_\theta(\alpha(x,y), \delta(x,y), \lambda) T(\lambda) d\lambda .
\]

Images from different visits or telescopes can have pointing residuals. For
small cutouts, `ExposurePose` supplies a learnable tangent-plane residual per
exposure:

\[
(\alpha', \delta') = \mathrm{Pose}_{\phi_i}(\alpha_{\mathrm{WCS}}(x,y), \delta_{\mathrm{WCS}}(x,y)).
\]

Later in the project, the experiment becomes a joint solve for scene and pose:

\[
\min_{\theta,\phi}
\sum_i \left\| I_{\mathrm{obs},i} -
\int F_\theta(\mathrm{Pose}_{\phi_i}(\alpha,\delta), \lambda) T_i(\lambda)d\lambda
\right\|^2 .
\]

## 2. First Week Workflow

Run commands from the repository root. Use Pixi for project Python:

```bash
pixi run preflight-push
```

Recommended path:

1. **Read the glossary.** Know what FITS, WCS, band, stack, cube, and INR mean.
2. **List the available bands.** This confirms the data script works.
3. **Build SMACS J0723 mini-cutouts.** This makes small files you can inspect.
4. **Build a bilinear stack baseline.** This is the simplest classical product.
5. **Build a drizzle-like stack baseline.** This is closer to a standard astronomy coadd.
6. **Make quicklook PNGs.** Inspect images before training anything.
7. **Fit a small INR.** Only start model tuning after the data products look sane.
8. **Compare residuals.** Look at stack residuals, INR residuals, and exported cubes.

First-day success means the commands run, the baseline FITS cube exists, and the
student can explain what FITS, WCS, band, stack, and cube mean in one sentence.

## 3. Where The SMACS Data Comes From

The starter data is not raw JWST exposure data. It is a small crop made from
public UPdec-Webb coadded JWST/NIRCam products for SMACS J0723.

One-liner: we begin from public coadded JWST mosaics, not from raw detector reads.

Useful facts:

- UPdec-Webb is a public dataset for coaddition of JWST NIRCam images of SMACS J0723.
- The paper describes UPDC: upsampling and PSF deconvolution coaddition.
- The public data page describes iteration 0 as the Drizzle product.
- Later iterations add UPDC deconvolution; sharper images can also amplify noise.
- Our starter script uses iteration 0 files and crops their `SCI` image extension.

Why this choice? Iteration 0 is a conservative place to start. It gives a
drizzle-style public product that is easier to compare against before we attempt
neural fields or deconvolution-like behavior.

Important limitation: this starter fixture is already coadded. A future stronger
project stage should go back to calibrated exposures and build the stack/neural
field from individual observations.

References:

- UPdec-Webb data page: `https://nadc.china-vo.org/res/r101424`
- UPdec-Webb paper: `https://arxiv.org/abs/2411.13891`

## 4. Data Preparation

The repo does not commit FITS cutouts. Build a small public JWST NIRCam fixture:

```bash
pixi run -e test -- python scripts/build_smacs0723_minicutout.py --download
```

Outputs land in `data/smacs0723_minicutout/` by default, or in
`$TORCHSKY_DATA/smacs0723_minicutout/` when `TORCHSKY_DATA` is set. The script
writes a `manifest.json` with absolute paths when the data root is outside the
repo, so the rest of the workflow does not depend on where the FITS files live.
To see the currently supported public fixture bands:

```bash
pixi run -e test -- python scripts/build_smacs0723_minicutout.py --list-bands
```

This first fixture is JWST-only. That is intentional: it gives a clean starting
point. HST can be added after the student understands the same steps for one
dataset: file discovery, FITS headers, WCS sanity checks, stack, INR fit, and
residual comparison.

For adjacent tiles, keep the same `--crop-size` and vary `--center-x` /
`--center-y`. The manifest records `source_shape`, `crop_origin_xy`,
`crop_bounds_xy`, and the requested/actual crop centers so each tile can be
recreated later. This is intentionally simple provenance, but it is enough to
avoid losing track of which source pixels became which training patch.

Then build the WCS-harmonized stack baseline:

```bash
pixi run -e test -- python scripts/build_smacs0723_stack_baseline.py
```

This writes `smacs0723_stack_baseline.fits`, a small spectral FITS cube made by
reprojecting the cutouts onto the first cutout's WCS.

To make the drizzle-like torchsky baseline:

```bash
pixi run -e test -- python scripts/build_smacs0723_stack_baseline.py \
  --resampling-kernel drizzle_square \
  --output data/smacs0723_minicutout/smacs0723_stack_drizzle_square.fits
```

One-liner: bilinear is the simple debugging baseline; drizzle-like is the
astronomy baseline to compare against.

What the two scripts create:

- `manifest.json`: paths, bands, shapes, and checksums for the mini-cutouts.
- `smacs0723_*_cutout_*.fits`: one small FITS image per filter.
- `smacs0723_stack_baseline.fits`: a conventional multiband cube on one WCS grid.
- `smacs0723_stack_drizzle_square.fits`: optional drizzle-like stack baseline.
- `quicklooks/*.png`: visual checks created in the next step.

## 5. Quicklook Visualization

Before fitting a neural field, make pictures. Many astronomy mistakes are obvious
visually: flipped axes, wrong crop, bad background, missing band, or sources not
lining up.

```bash
pixi run -e test -- python scripts/visualize_smacs0723_products.py
```

This writes `quicklooks/` next to the manifest:

- `cutouts.png`: each input band as a panel, labeled with crop origin.
- `baseline_cube.png`: each baseline cube plane.
- `baseline_false_color.png`: approximate color composite from low/mid/high bands.
- `quicklook_summary.json`: paths and array shapes.

After an INR cube exists, compare it to the baseline:

```bash
pixi run -e test -- python scripts/visualize_smacs0723_products.py \
  --comparison-cube hyperspectral_reconstruction.fits
```

This adds `comparison_residuals.png` with baseline, reconstruction, and residual
panels. The first questions are simple: are sources in the same place, are
residuals structured, and did the model invent artifacts at tile boundaries?

## 6. Baseline Stack API

For scripts or notebooks, the classical baseline is:

```python
from torchsky.map import ImageObservation, coadd_image_plane_multiband
from torchsky.spectral import write_cube_fits

observations = [
    ImageObservation(
        path="data/smacs0723_minicutout/smacs0723_F090W_cutout_256.fits",
        wavelength=9000.0,
        spectral_min=8000.0,
        spectral_max=10000.0,
        band_name="F090W",
    ),
    # Add the remaining bands from manifest.json.
]

baseline = coadd_image_plane_multiband(observations, resampling_kernel="bilinear")
write_cube_fits(
    "smacs0723_stack_baseline.fits",
    baseline.data,
    spatial_header=baseline.header,
    spectral_axis=baseline.spectral_axis,
    overwrite=True,
)
```

Use this before fitting neural fields. If the baseline looks wrong, debug data,
headers, masks, or units before changing the INR.

For a drizzle-like baseline in code, use `coadd_image_plane_cube(...,
resampling_kernel="drizzle_square", drizzle_pixfrac=1.0)`. The helper script
uses this lower-level API when you pass `--resampling-kernel drizzle_square`.

Useful baseline checks:

- Does every cutout have similar `NAXIS1`, `NAXIS2`, `CRVAL1`, and `CRVAL2`?
- Does the stack cube shape look like `(number_of_bands, height, width)`?
- Are bright objects in roughly the same place in each band?
- Does the weight/coverage map show where there is real data?

## 7. INR Starting Point

The first neural-field model should be deliberately small:

```python
from torchsky.map import (
    FieldObservation,
    estimate_field_domain,
    fit_field_model,
    make_field_model,
    render_field_cube,
)
from torchsky.spectral import SpectralAxis, write_cube_fits
import torch

domain = estimate_field_domain(field_observations)
model = make_field_model("hashgrid_spectral_field", domain, hidden_dim=64, num_levels=4)
result = fit_field_model(field_observations, model, steps=500, lr=5e-3)

axis = SpectralAxis(
    values=torch.linspace(8000.0, 50000.0, 100),
    unit="Angstrom",
    type="WAVE",
)
cube = render_field_cube(model, target_header=target_header, spectral_axis=axis)
write_cube_fits(
    "hyperspectral_reconstruction.fits",
    cube,
    spatial_header=target_header,
    spectral_axis=axis,
    overwrite=True,
)
```

`FieldObservation` caches WCS coordinate grids after the first render so repeated
training steps do not recompute the same pixel-to-sky transform. If you mutate an
observation header or image shape in-place, call `observation.clear_coordinate_cache()`.

For early work, prefer boring experiments:

- Fit two or four bands before trying all available data.
- Use small crops and CPU first; move to GPU only after the loss and plots make sense.
- Plot `observation.image`, `render_field_observation(model, observation)`, and the residual.
- Change one thing at a time: model size, learning rate, spectral samples, or pose.
- Save quicklooks for every important run. A loss curve without images is not enough.

When rendering larger products, use `render_field_cube(..., spatial_tile_hw=(256, 256), spectral_chunk=1)`
or similar chunking so the output cube does not require all wavelengths and all
pixels in one forward pass.

## 8. Bundle Adjustment

Once the fixed-WCS INR works, add pose residuals:

```python
from torchsky.map import create_exposure_poses, fit_field_model

poses = create_exposure_poses(field_observations, anchor_first=True)
result = fit_field_model(
    field_observations,
    model,
    observation_exposure_poses=poses,
    steps=500,
    lr=5e-3,
)
```

`anchor_first=True` freezes the first pose to fix the global astrometric gauge.
Do not judge bundle adjustment from a single exposure; use at least two
overlapping observations and compare fixed-WCS and learned-pose residuals.

The "gauge" point matters: if every exposure is allowed to move, the model can
shift the whole sky and still fit the data. Freezing one reference pose defines
what "no shift" means.

## 9. Path To A Larger Contiguous Field

The term project should grow from one patch to a larger contiguous region in
stages. Do not jump straight to "fit the whole mosaic"; that usually hides data
problems and memory problems at the same time.

1. **One center crop:** prove the download, FITS read, WCS, baseline stack, and
   INR fit all work.
2. **A few adjacent crops:** use explicit `--center-x` / `--center-y` values and
   keep overlap between neighboring crops. Check that sources line up across tile
   boundaries in the baseline stack and quicklook PNGs.
3. **One shared target grid:** choose a reference WCS/header for the contiguous
   output, then compare every tile or exposure against that grid. The classical
   stack should be able to produce a sane product before the INR is trusted.
4. **Patch-based INR training:** use `fit_field_model(..., training_patch_size=...,
   shared_sky_patch_sampling=True, observations_per_step=...)` so each step sees
   manageable sky regions rather than the full field.
5. **Chunked cube rendering:** render final products with `spatial_tile_hw` and
   `spectral_chunk` to control memory. Write intermediate products and keep the
   manifest/checksums/quicklooks; do not rely on notebook state.

The important design idea is that the field is continuous, but the engineering is
still tiled and chunked. A large contiguous reconstruction is assembled from
carefully tracked local evidence, not from one giant untracked tensor.

## 10. Notebooks

| Notebook | Goal |
|----------|------|
| [examples/notebooks/01_field_fit_synthetic.ipynb](../../examples/notebooks/01_field_fit_synthetic.ipynb) | Single-band synthetic quickstart. |
| [examples/notebooks/02_multiband_hyperspectral_synthetic.ipynb](../../examples/notebooks/02_multiband_hyperspectral_synthetic.ipynb) | Multiband throughput, cube export, and pose hook. |
| [examples/notebooks/03_smacs0723_real_minicutout.ipynb](../../examples/notebooks/03_smacs0723_real_minicutout.ipynb) | Real JWST mini-cutouts after running the data scripts. |

## 11. Term-Scale Success Criteria

1. A reproducible data fixture with manifest, baseline stack cube, and quicklook plots.
2. A fixed-WCS INR fit to at least two calibrated bands on the same patch.
3. A learned-pose experiment showing whether alignment improves residuals or sharpness.
4. A tiled experiment showing that the method can extend beyond one center crop.
5. An exported hyperspectral FITS cube openable in standard astronomy tools.
6. A written comparison of stack baseline vs INR, including failures.

Reusable primitives belong in `src/torchsky/`. Training loops, plots, and research
experiments should live in `examples/`, `examples/notebooks/`, or a separate
research repo until they are stable enough to review.

See [field.md](../field.md), [implicit_forward.md](../implicit_forward.md),
[map.md](../map.md), and [spectral.md](../spectral.md) for technical references.
