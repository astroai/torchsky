# Student Project: Hyperspectral Fusion with Neural Spectral Fields

This project takes several telescope images of the same patch of sky and turns
them into one deeper model of that patch in space and wavelength. We start with
a small JWST-style patch and a normal astronomy stack, make quicklook images so
we can see what we are doing, then fit a neural field, and only later scale to a
larger contiguous sky field.

Guiding rule: start with things you can see and debug. If the data layer is
wrong, the neural field will learn the wrong thing.

## 0. Big Picture

This project has three layers, and we work on them in this order:

1. **Data layer:** FITS files, WCS headers, filters, masks, and weights.
2. **Baseline layer:** a classical stack/coadd, including a drizzle-like baseline.
3. **Research layer:** a neural field that predicts intensity at `(RA, Dec, wavelength)`.

Each layer should make sense on its own before you build on top of it.

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

### A Little Math (Optional First Read)

You do not need to derive these equations to use the code; the library calls
them for you. They are here so you know what the code is doing.

The continuous sky model is a function:

\[
F_\theta(\alpha, \delta, \lambda)
\]

A telescope image is predicted by querying the sky through its WCS and
integrating over the filter throughput \(T(\lambda)\):

\[
I_{\mathrm{pred}}(x, y) =
\int F_\theta(\alpha(x,y), \delta(x,y), \lambda) T(\lambda) d\lambda .
\]

Images from different visits or telescopes can have small pointing residuals.
For small cutouts, `ExposurePose` supplies a learnable tangent-plane residual
per exposure:

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

You should be able to do all of the following from a fresh terminal in the
repository root. Project Python lives inside Pixi environments; this guide
uses the `test` environment because it has the FITS and astronomy dependencies.

Sanity check that the environment is healthy:

```bash
pixi run preflight-push
```

If that command fails, ask before continuing; do not try to install Python
packages by hand.

Recommended path (each step has a command later in the doc):

1. **Read the glossary** above so the words FITS, WCS, band, stack, cube, and
   INR are not surprising.
2. **List the available bands** with `build_smacs0723_minicutout.py --list-bands`
   to confirm the data script can be invoked.
3. **Build SMACS J0723 mini-cutouts** with `--download` (Section 4).
4. **Build a bilinear stack baseline** with `build_smacs0723_stack_baseline.py`
   (Section 4); this is the simplest classical product.
5. **Build a drizzle-like stack baseline** with `--resampling-kernel drizzle_square`
   (Section 4); this is closer to a standard astronomy coadd.
6. **Make quicklook PNGs** with `visualize_smacs0723_products.py` (Section 5)
   and inspect them before training anything.
7. **Fit a small INR** (Section 7); only start tuning after the data products
   look sane.
8. **Compare residuals** between the stack baseline and the INR cube (Section 5
   with `--comparison-cube`).

First-day success: the commands above run, a baseline FITS cube exists, and
you can describe what FITS, WCS, band, stack, and cube mean in one sentence
each.

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

Important limitation: this starter fixture is already coadded. A future
stronger project stage should go back to calibrated exposures and build the
stack/neural field from individual observations.

This first fixture is JWST-only. That is intentional: it gives a clean
starting point. HST data can be added later, after you understand the same
steps for one telescope: file discovery, FITS headers, WCS sanity checks,
stack, INR fit, and residual comparison.

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

For your own scripts or notebooks, the classical baseline is:

```python
from torchsky.map import ImageObservation, coadd_image_plane_cube
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

baseline = coadd_image_plane_cube(
    observations,
    resampling_kernel="bilinear",  # use "drizzle_square" for the drizzle-like baseline
    drizzle_pixfrac=1.0,
)
write_cube_fits(
    "smacs0723_stack_baseline.fits",
    baseline.data,
    spatial_header=baseline.header,
    spectral_axis=baseline.spectral_axis,
    overwrite=True,
)
```

The `build_smacs0723_stack_baseline.py` helper calls this same function. If
the baseline looks wrong here, debug data, headers, masks, or units before
changing the INR.

Useful baseline checks:

- Does every cutout have similar `NAXIS1`, `NAXIS2`, `CRVAL1`, and `CRVAL2`?
- Does the stack cube shape look like `(number_of_bands, height, width)`?
- Are bright objects in roughly the same place in each band?
- Does the weight/coverage map show where there is real data?

## 7. INR Starting Point

Before fitting, build one `FieldObservation` per cutout. `FieldObservation`
wraps the image tensor, the FITS header (so the WCS is known), and the
spectral metadata for that band:

```python
import torch
import torchsky
from torchsky.map import FieldObservation

def field_observation_from_cutout(path, wavelength, spectral_min, spectral_max, band):
    image, header = torchsky.read(path, return_header=True)
    return FieldObservation(
        image=torch.as_tensor(image, dtype=torch.float32),
        header=dict(header),
        wavelength=wavelength,
        spectral_min=spectral_min,
        spectral_max=spectral_max,
        transmission_center=wavelength,
        band_name=band,
    )

field_observations = [
    field_observation_from_cutout(
        "data/smacs0723_minicutout/smacs0723_F090W_cutout_256.fits",
        wavelength=9000.0, spectral_min=8000.0, spectral_max=10000.0, band="F090W",
    ),
    field_observation_from_cutout(
        "data/smacs0723_minicutout/smacs0723_F150W_cutout_256.fits",
        wavelength=15000.0, spectral_min=13000.0, spectral_max=17000.0, band="F150W",
    ),
]
target_header = field_observations[0].header
```

The first neural-field model should be deliberately small:

```python
from torchsky.map import (
    estimate_field_domain,
    fit_field_model,
    make_field_model,
    render_field_cube,
)
from torchsky.spectral import SpectralAxis, write_cube_fits

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

What to expect: with two bands at `crop_size=256` on CPU, this should take a
few minutes. The 100 wavelengths span all NIRCam bands so the rendered cube
also covers wavelengths between the two bands (which the INR has to
extrapolate). For a first run, restrict the rendering axis to the wavelength
range you actually trained on.

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

"Fixed-WCS" means we trust the FITS header coordinates exactly. Bundle
adjustment relaxes that by letting each exposure shift, rotate, and scale
slightly to better align with the others while the neural field is being fit.

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
2. **A few adjacent crops:** keep the same `--crop-size` and vary `--center-x`
   and `--center-y` in `build_smacs0723_minicutout.py`. The manifest records
   `source_shape`, `crop_origin_xy`, `crop_bounds_xy`, and the requested/actual
   crop centers so each tile can be recreated later. Keep overlap between
   neighboring crops and check that sources line up across tile boundaries in
   the baseline stack and quicklook PNGs.
3. **One shared target grid:** choose a reference WCS/header for the contiguous
   output, then compare every tile or exposure against that grid. The classical
   stack should be able to produce a sane product before the INR is trusted.
4. **Patch-based INR training:** use `fit_field_model(..., training_patch_size=...,
   shared_sky_patch_sampling=True, observations_per_step=...)` so each step sees
   manageable sky regions rather than the full field.
5. **Chunked cube rendering:** render final products with `spatial_tile_hw` and
   `spectral_chunk` to control memory. Write intermediate products and keep the
   manifest/checksums/quicklooks; do not rely on notebook state.

The important design idea is that the field is continuous, but the engineering
is still tiled and chunked. A large contiguous reconstruction is assembled
from carefully tracked local evidence, not from one giant untracked tensor.

## 10. Notebooks

If you prefer notebooks, start with `01_field_fit_synthetic.ipynb` to learn the
field API on synthetic data; the SMACS notebook expects the data scripts above
to have been run first.

| Notebook | Goal |
|----------|------|
| [examples/notebooks/01_field_fit_synthetic.ipynb](../../examples/notebooks/01_field_fit_synthetic.ipynb) | Single-band synthetic quickstart. |
| [examples/notebooks/02_multiband_hyperspectral_synthetic.ipynb](../../examples/notebooks/02_multiband_hyperspectral_synthetic.ipynb) | Multiband throughput, cube export, and pose hook. |
| [examples/notebooks/03_smacs0723_real_minicutout.ipynb](../../examples/notebooks/03_smacs0723_real_minicutout.ipynb) | Real JWST mini-cutouts after running the data scripts. |

## 11. If Something Goes Wrong

Common first-day issues:

- **`pixi: command not found`**: Pixi is not installed yet. Ask before
  installing Python tooling by hand; the project is meant to run through Pixi.
- **`ModuleNotFoundError: torchsky.cpp`**: you ran in the `default` Pixi env
  instead of `test`. Re-run with `pixi run -e test -- python ...`.
- **`missing reference ...`**: the UPdec FITS files have not been downloaded.
  Re-run the cutout script with `--download`.
- **Quicklook PNGs look empty or all black**: open the FITS files in DS9, or
  rerun with a smaller `--crop-size` and check `manifest.json` for the actual
  `crop_bounds_xy`.
- **INR loss does not decrease**: shrink the model
  (`hidden_dim`, `num_levels`), drop to a single band, and confirm the
  baseline stack already shows the expected sources.

When asking for help, include: the exact command you ran, the full error, the
relevant `manifest.json` entry, and a quicklook PNG if there is one. That is
usually enough to diagnose remotely.

## 12. Term-Scale Success Criteria

1. A reproducible data fixture with manifest, baseline stack cube, and quicklook plots.
2. A fixed-WCS INR fit to at least two calibrated bands on the same patch.
3. A learned-pose experiment showing whether alignment improves residuals or sharpness.
4. A tiled experiment showing that the method can extend beyond one center crop.
5. An exported hyperspectral FITS cube openable in standard astronomy tools.
6. A written comparison of stack baseline vs INR, including failures.

Reusable primitives belong in `src/torchsky/`. Training loops, plots, and
research experiments should live in `examples/`, `examples/notebooks/`, or a
separate research repo until they are stable enough to review.

## 13. Further Reading

You do not need to read all of these. Pick the topic you are working on and
read one or two items per area; come back to the rest as the project deepens.

### torchsky internal docs

- [field.md](../field.md) — `FieldObservation`, fitting, rendering API.
- [implicit_forward.md](../implicit_forward.md) — multi-survey losses,
  `KinematicState`, Charbonnier scaffold.
- [map.md](../map.md) — `ImageObservation`, coadds, map products.
- [spectral.md](../spectral.md) — spectral cubes and FITS I/O.
- [AGENTS.md](../../AGENTS.md) — repo policy: dependencies, Pixi, performance
  and testing rules.

### Astronomy data formats and conventions

- FITS standard at NASA GSFC: https://fits.gsfc.nasa.gov/
- WCS paper I (Greisen & Calabretta 2002), the world coordinate convention used
  in FITS headers: https://www.aanda.org/articles/aa/abs/2002/45/aah3859/aah3859.html
- WCS paper II (Calabretta & Greisen 2002), spherical projections including TAN:
  https://www.aanda.org/articles/aa/abs/2002/45/aah3860/aah3860.html
- Astropy WCS user guide: https://docs.astropy.org/en/stable/wcs/
- SAOImage DS9 viewer (open `.fits` files visually): https://sites.google.com/cfa.harvard.edu/saoimageds9

### JWST, NIRCam, and SMACS J0723

- JWST NIRCam filter overview: https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters
- NASA press release for JWST first deep field (SMACS J0723): https://www.nasa.gov/missions/webb/nasas-webb-delivers-deepest-infrared-image-of-universe-yet/
- Pontoppidan et al. 2022, JWST Early Release Observations overview:
  https://arxiv.org/abs/2207.13067
- Pascale et al. 2022, JWST lensed galaxies behind SMACS 0723:
  https://arxiv.org/abs/2207.07102
- UPdec-Webb dataset page (the public coadds we crop from):
  https://nadc.china-vo.org/res/r101424
- UPdec-Webb paper (Wang et al. 2024) describing UPDC and the Drizzle baseline:
  https://arxiv.org/abs/2411.13891

### Drizzle and classical coaddition

- Fruchter & Hook 2002, the Drizzle algorithm:
  https://arxiv.org/abs/astro-ph/9808087
- STScI `drizzlepac` user docs:
  https://www.stsci.edu/scientific-community/software/drizzlepac

### Neural fields and INRs

- Mildenhall et al. 2020, NeRF, the canonical neural-field paper:
  https://arxiv.org/abs/2003.08934
- Sitzmann et al. 2020, SIREN, sinusoidal implicit representations:
  https://arxiv.org/abs/2006.09661
- Müller et al. 2022, Instant-NGP, hash-grid features used in
  `hashgrid_spectral_field`: https://arxiv.org/abs/2201.05989
- Xie et al. 2022, neural fields in visual computing review (good entry point
  to the field as a whole): https://arxiv.org/abs/2111.11426

### Bundle adjustment and joint scene + pose optimization

- Triggs et al. 1999, "Bundle Adjustment — A Modern Synthesis" (still the
  standard reference): https://hal.inria.fr/inria-00548290/document
- Lin et al. 2021, BARF, bundle-adjusting neural radiance fields, the closest
  computer-vision analogue of what `ExposurePose` does here:
  https://arxiv.org/abs/2104.06405

### Astronomy meets neural fields (optional but inspiring)

- Levis et al. 2022, neural rendering for black-hole emission tomography, an
  example of inverse problems with neural fields applied to astronomy:
  https://arxiv.org/abs/2204.03715

If you want a current shortlist of astronomy-plus-neural-fields papers, ask
your supervisor — that area moves fast and is best curated by hand.
