# WCS-aware multi-wavelength INR for SMACS J0723

This implementation trains directly on native JWST exposure pixels. It does
not first coadd or resample the SCI arrays. The fixed per-exposure WCS maps
native detector pixels into one continuous union canvas inside the forward
model. Fourier and spherical positional encodings are both implemented, with
trainable feature-space alignment (alignment option A).

## Data layout

```text
/arc/projects/ots/jwst_smacs0723/
├── input/
│   ├── F090W/*_crf.fits
│   ├── F150W/*_crf.fits
│   ├── F200W/*_crf.fits
│   ├── F277W/*_crf.fits
│   ├── F356W/*_crf.fits
│   ├── F444W/*_crf.fits
│   └── calibration/
│       └── throughput/
│           └── detector-specific official NIRCam throughput tables
└── results/
    └── wcs_aware_multiband_inr/
```

Each CRF must contain `SCI` and `ERR`; `DQ` is used for the valid-pixel mask.
`UPDCMASK` is also respected when present.

## Dependencies

Install STPSF in the CUDA environment:

```bash
pixi run -e cuda python -m pip install -r requirements/wcs-aware-inr.txt
```

STPSF needs its separate instrument data. By default it may install data under
the home directory. To keep calibration data on the project volume, point
`STPSF_PATH` at a project directory before the first PSF-generation run:

```bash
export STPSF_PATH=/arc/projects/ots/jwst_smacs0723/input/calibration/stpsf-data
```

## Production run

Production defaults are:

- official tabulated detector-specific throughput;
- nine native TorchSky bandpass quadrature samples;
- an exposure- and wavelength-specific monochromatic STPSF kernel;
- `DET_DIST` detector-sampled PSF output;
- Fourier positional encoding;
- trainable feature-space alignment;
- ERR inverse-variance weighted robust loss.

```bash
pixi run -e cuda -- python scripts/smacs0723_wcs_aware_multiband_inr.py \
  --exposure-dir /arc/projects/ots/jwst_smacs0723/input \
  --output-dir /arc/projects/ots/jwst_smacs0723/results/wcs_aware_multiband_inr \
  --bands F090W,F150W,F200W,F277W,F356W,F444W \
  --module b \
  --bandpass-mode official \
  --throughput-dir /arc/projects/ots/jwst_smacs0723/input/calibration/throughput \
  --bandpass-samples 9 \
  --psf-mode stpsf \
  --psf-size 51 \
  --stpsf-extension DET_DIST \
  --position-encoding fourier \
  --device cuda
```

Use `--run-both-encodings` to train separate Fourier and spherical models.

The first STPSF run builds and caches a `[E,L,K,K]` PSF bank under the output
calibration cache. Later runs reuse the cache when the wavelengths and kernel
configuration match.

## Preserved basic modes

The simpler implementations remain available for debugging and ablation.

Bandpass:

```text
--bandpass-mode center
--bandpass-mode top_hat
--bandpass-mode gaussian
--bandpass-mode official
```

PSF:

```text
--psf-mode none
--psf-mode gaussian
--psf-mode npy
--psf-mode stpsf
```

Examples:

```bash
# Geometry/alignment smoke test without instrumental PSF.
pixi run -e cuda -- python scripts/smacs0723_wcs_aware_multiband_inr.py \
  --bandpass-mode top_hat \
  --bandpass-samples 5 \
  --psf-mode none \
  --steps 100

# Gaussian PSF ablation.
pixi run -e cuda -- python scripts/smacs0723_wcs_aware_multiband_inr.py \
  --bandpass-mode gaussian \
  --bandpass-samples 5 \
  --psf-mode gaussian \
  --psf-size  nine \
  --psf-fwhm-pix F090W=1.1,F150W=1.3,F200W=1.6,F277W=2.1,F356W=2.6,F444W=3.1
```

For `npy` mode, a kernel can be supplied as one `[K,K]` array, which is reused
at all quadrature wavelengths, or as `[L,K,K]`. `.npz` files may additionally
store a `wavelength` array for validation.

## Throughput manifest

Automatic file-name matching recognizes detector names such as `NRCB1`, `B1`,
`NRCB5`, and `NRCBLONG`. For deterministic selection, provide a JSON manifest:

```json
{
  "NRCB1/F090W": "F090W_NRCB1_system_throughput.txt",
  "NRCB2/F150W": "F150W_NRCB2_system_throughput.txt",
  "NRCB3/F200W": "F200W_NRCB3_system_throughput.txt",
  "NRCB5/F277W": "F277W_NRCB5_system_throughput.txt",
  "NRCB5/F356W": "F356W_NRCB5_system_throughput.txt",
  "NRCB5/F444W": "F444W_NRCB5_system_throughput.txt"
}
```

Pass it with `--throughput-manifest`. Relative paths are resolved relative to
the manifest file.

## Observation operator

For exposure `e`, native detector pixel `p`, and quadrature sample `j`, the
implemented prediction is

```text
I_hat_e(p) = sum_j w_ej [K_ej * S_theta(., lambda_ej)](p) + b_e.
```

The PSF is applied inside the wavelength loop, before the bandpass sum. The
stored PSF kernels are flipped once on ingestion so the native-neighbor gather
implements mathematical convolution rather than cross-correlation.
