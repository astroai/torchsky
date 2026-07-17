# WCS-aware multi-wavelength INR for SMACS J0723

This implementation trains directly on native JWST exposure pixels. It does
not first coadd or resample the SCI arrays. The fixed per-exposure WCS maps
native detector pixels into one continuous union canvas inside the forward
model. Fourier and spherical positional encodings are both implemented, with
trainable feature-space alignment.

## Data

The current SMACS J0723 input contains 135 CRF products. They consistently use
JWST pipeline `CAL_VER=1.20.2` and `CRDS_CTX=jwst_1464.pmap`.

Each CRF must contain `SCI` and `ERR`; `DQ` is used for the valid-pixel mask.
`UPDCMASK` is also respected when present.

## Dependencies

```bash
pixi run -e cuda python -m pip install -r requirements/wcs-aware-inr.txt
```

`astroquery` is required for the official SVO bandpass path. STPSF and its
instrument data are needed only when an explicit STPSF model is requested.

## Official bandpass

`--bandpass-mode official` queries the SVO Filter Profile Service through
`astroquery.svo_fps`. The default collection is `JWST/NIRCam2025`, whose filter
IDs include `JWST/NIRCam2025.F090W` through
`JWST/NIRCam2025.F444W`. These curves are detector-averaged total-system
throughputs, not detector-specific curves.

The first request for each band is saved as an NPZ snapshot. Later runs reuse
the snapshot unless `--svo-force-refresh` is given. The tabulated response is
interpolated onto the configured wavelength samples and converted to normalized
trapezoidal quadrature weights.

## First no-explicit-PSF run

```bash
pixi run -e cuda -- python scripts/smacs0723_wcs_aware_multiband_inr.py \
  --exposure-dir /arc/projects/ots/jwst_smacs0723/input \
  --output-dir /arc/projects/ots/jwst_smacs0723/results/wcs_aware_multiband_inr \
  --input-product crf \
  --bands F090W,F150W,F200W,F277W,F356W,F444W \
  --module b \
  --bandpass-mode official \
  --svo-collection JWST/NIRCam2025 \
  --svo-cache-dir /arc/projects/ots/jwst_smacs0723/input/calibration/svo \
  --bandpass-samples 9 \
  --psf-mode none \
  --position-encoding fourier \
  --device cuda \
  --steps 10000
```

## Preserved ablations

Bandpass modes: `center`, `top_hat`, `gaussian`, and `official`.

PSF modes: `none`, `gaussian`, `npy`, and `stpsf`.

## Observation operator

For exposure `e`, native detector pixel `p`, and quadrature sample `j`, the
general prediction is

```text
I_hat_e(p) = sum_j w_ej [K_ej * S_theta(., lambda_ej)](p) + b_e.
```

With `--psf-mode none`, `K_ej` is the identity and the decoder learns an
observed-resolution shared spectral representation. With an explicit PSF, the
PSF is applied inside the wavelength loop before the bandpass sum.
