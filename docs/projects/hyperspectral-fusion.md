# Student project — hyperspectral fusion with neural spectral fields

This page orients an undergrad toward **multi-band JWST-style fusion** using
`torchsky.map.field` (continuous spectral INR), optional **`ExposurePose`**
(bundle-adjustment-style alignment), and FITS cube export via **`write_cube_fits`**.

## Environment

Follow [AGENTS.md](../../AGENTS.md): **Pixi-first**.

- Full gates before pushing changes: `pixi run ci-local`
- Jupyter: `pixi run -e default -- jupyter lab`

## Notebook order

Run from the **repository root** so `data/` and `torchsky` imports resolve.

| Notebook | Goal |
|----------|------|
| [examples/notebooks/01_field_fit_synthetic.ipynb](../../examples/notebooks/01_field_fit_synthetic.ipynb) | Single-band synthetic fit on CPU (~minutes). |
| [examples/notebooks/02_multiband_hyperspectral_synthetic.ipynb](../../examples/notebooks/02_multiband_hyperspectral_synthetic.ipynb) | Tabulated throughput, `render_field_cube`, `write_cube_fits`, `ExposurePose` demo. |
| [examples/notebooks/03_smacs0723_real_minicutout.ipynb](../../examples/notebooks/03_smacs0723_real_minicutout.ipynb) | Real UPdec JWST cutouts; needs local data (see below). |

## Real-data mini-cutout (optional but recommended)

The repo does **not** commit FITS cutouts. Build a small multi-band fixture:

```bash
pixi run -e default -- python scripts/build_smacs0723_minicutout.py --download
```

This pulls UPdec JWST mosaic references and writes cropped FITS files under
`data/smacs0723_minicutout/` (or `$TORCHSKY_DATA/smacs0723_minicutout/`) plus
`manifest.json` checksums.

**Note:** the bundled SMACS scripts target **JWST NIRCam** public products.
Adding **HST ACS** cutouts is the same pipeline once FITS + WCS headers are on disk.

## Where code should live

| Kind of work | Location |
|--------------|----------|
| Reusable primitives (pose, writers, small API glue) | `src/torchsky/` |
| Training orchestration, logging, paper experiments | `examples/notebooks/`, `examples/`, or a private research repo |
| Student thesis-only analysis | Same as experiments; avoid widening public API without review |

See [field.md](../field.md), [implicit_forward.md](../implicit_forward.md), [spectral.md](../spectral.md).

## Success criteria (term-scale)

1. End-to-end fit of a spectral field to **≥2** calibrated bands on a shared sky patch.
2. Differentiable alignment experiment (**`ExposurePose`** or successor) with a clear gauge-fix story.
3. Exported **hyperspectral FITS cube** (linear λ grid) openable in DS9 or similar.
4. Comparison baseline (e.g. weighted co-add via `coadd_image_plane` or external drizzle) on the same pixels.

## Research extensions (beyond the onboarding path)

- Spatially varying PSF / ePSF (`PSFSpec` extensions).
- Full spherical pose or tying poses across visits.
- HEALPix or full-mosaic field outputs at scale.
- Joint noise / outlier modeling (`implicit_forward` losses or custom loops).
