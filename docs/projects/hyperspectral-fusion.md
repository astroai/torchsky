# Student Project: Hyperspectral Fusion with Neural Spectral Fields

Welcome to the `torchsky` hyperspectral fusion project! This guide will orient you toward your research goal: performing **multi-band JWST/HST fusion** using continuous neural representations (`torchsky.map.field`), optimizing astrometric alignment via bundle adjustment (`ExposurePose`), and exporting standard data products for astronomical analysis (`write_cube_fits`).

## 1. The Core Concepts

Instead of standard, rigid multi-step pipelines (register -> resample -> co-add), we learn a continuous 3D model of the sky $F_\theta(\alpha, \delta, \lambda)$, representing intensity as a function of Right Ascension ($\alpha$), Declination ($\delta$), and wavelength ($\lambda$).

### The Forward Model
When we observe the sky with a telescope, the image $I$ is formed by projecting the continuous sky through the instrument's specific geometry and integrating over its spectral bandpass $T(\lambda)$:
$$ I_{pred}(x, y) = \int F_\theta(\alpha(x,y), \delta(x,y), \lambda) T(\lambda) d\lambda $$

By minimizing the difference between the predicted image $I_{pred}$ and the actual telescope image $I_{obs}$, we optimize the neural network weights $\theta$.

### Simultaneous Alignment (Bundle Adjustment)
Images from different visits or telescopes often have small pointing errors. If we assume the WCS header coordinates are perfectly correct, our fusion will be blurry.

To solve this, we introduce `ExposurePose`, a learnable astrometric residual module for each exposure $i$. It applies a small tangent-plane shift, rotation, and scale $\phi_i$ to correct the coordinates:
$$ (\alpha', \delta') = \text{Pose}_{\phi_i}(\alpha_{WCS}(x,y), \delta_{WCS}(x,y)) $$

We then jointly optimize the scene $\theta$ and the poses $\phi_i$:
$$ \min_{\theta, \phi} \sum_i \left\| I_{obs, i} - \int F_\theta(\text{Pose}_{\phi_i}(\alpha, \delta), \lambda) T_i(\lambda) d\lambda \right\|^2 $$

## 2. Environment Setup

Follow [AGENTS.md](../../AGENTS.md) for full details, but the project uses a **Pixi-first** workflow.

- **Check your code before pushing:** `pixi run ci-local`
- **Start Jupyter Lab:** `pixi run -e default -- jupyter lab`

## 3. Recommended Notebook Path

Run these from the **repository root** so `data/` and `torchsky` imports resolve properly.

| Notebook | Goal |
|----------|------|
| [examples/notebooks/01_field_fit_synthetic.ipynb](../../examples/notebooks/01_field_fit_synthetic.ipynb) | Quickstart: Single-band synthetic fit on CPU (~minutes). |
| [examples/notebooks/02_multiband_hyperspectral_synthetic.ipynb](../../examples/notebooks/02_multiband_hyperspectral_synthetic.ipynb) | **Core concepts:** Tabulated throughput, `render_field_cube`, `write_cube_fits`, and `ExposurePose` demo. |
| [examples/notebooks/03_smacs0723_real_minicutout.ipynb](../../examples/notebooks/03_smacs0723_real_minicutout.ipynb) | Real UPdec JWST cutouts. Needs local data (see below). |

## 4. Working with Real Data

The repo does **not** commit FITS cutouts. To get started with real JWST NIRCam data, build the small multi-band fixture:

```bash
pixi run -e default -- python scripts/build_smacs0723_minicutout.py --download
```

This script pulls UPdec JWST mosaic references and writes cropped FITS files under `data/smacs0723_minicutout/` (or `$TORCHSKY_DATA/smacs0723_minicutout/`). Adding HST ACS cutouts is conceptually the same pipeline once the FITS files and WCS headers are on disk.

## 5. Using `fit_field_model` with `ExposurePose`

To solve for registration and scene properties together, generate a list of `ExposurePose` objects (one per observation) and pass them to the high-level `fit_field_model` function:

```python
from torchsky.map import create_exposure_poses, fit_field_model

# Generate one pose per observation.
# anchor_first=True freezes the first exposure to define the absolute
# coordinate frame (the "gauge"), preventing the whole scene from drifting.
poses = create_exposure_poses(observations, anchor_first=True)

result = fit_field_model(
    observations,
    model,
    observation_exposure_poses=poses,
    steps=500,
    lr=5e-3, # Learning rate applies to both the network and the poses
)
```

### Performance Tip: Coordinate Caching
`torchsky` automatically caches the heavy WCS coordinate grids inside the `FieldObservation` object during the first forward pass. This significantly speeds up training when using `ExposurePose`, as the base WCS doesn't need to be recalculated on the CPU every step.

## 6. Exporting Your Results

Once you have a fully trained model, you can evaluate it over a dense 3D grid and export it as a standard astronomical FITS cube:

```python
from torchsky.spectral import SpectralAxis, write_cube_fits
import torch

# Define your spectral axis (e.g., 100 slices from 0.8 to 5.0 microns)
axis = SpectralAxis(
    values=torch.linspace(8000.0, 50000.0, 100),
    unit="Angstrom",
    type="WAVE"
)

# Render the continuous model into a discrete cube
cube_tensor = render_field_cube(model, target_header=header, spectral_axis=axis)

# Save to disk
write_cube_fits("hyperspectral_reconstruction.fits", cube_tensor, spatial_header=header, spectral_axis=axis)
```

## 7. Success Criteria (Term-Scale)

1. **Multi-Band Fusion:** End-to-end fit of a spectral field to **≥2** calibrated bands on a shared sky patch.
2. **Alignment:** Differentiable alignment experiment (`ExposurePose`) demonstrating an improvement in sharpness/loss over fixed-WCS fitting.
3. **Data Product:** Exported **hyperspectral FITS cube** openable in standard tools like DS9 or Glue.
4. **Baseline Comparison:** A comparison against a traditional weighted co-add (e.g., via `coadd_image_plane` or external drizzle) on the same pixels.

## 8. Where Code Should Live

| Kind of work | Location |
|--------------|----------|
| Reusable primitives (pose, writers, core layers) | `src/torchsky/` (Submit a Pull Request!) |
| Training orchestration, logging, experiments | `examples/notebooks/`, `examples/`, or your private research repo |
| Student thesis-only analysis/plotting | Same as experiments; avoid widening public API without review |

See [field.md](../field.md), [implicit_forward.md](../implicit_forward.md), and [spectral.md](../spectral.md) for deeper technical references on the map module.
