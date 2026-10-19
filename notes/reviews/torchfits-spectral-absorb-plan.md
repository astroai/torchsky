# Torchfits spectral-transform ownership decision

Status: **accepted 2026-07-20**.

## Context

torchfits 1.0 **hard-deleted** (no deprecation):

- `torchfits.transforms.spectral` — ContinuumNormalize, ContinuumRemoval,
  DopplerShift, SpectralBinning, BandMath, PhaseFold
- `torchfits.transforms.continuum` — SavitzkyGolayFilter, RunningPercentile,
  UpperEnvelopeContinuum, WaveletDecompose, AsymmetricLeastSquares,
  AlphaShapeContinuum

torchfits keeps FITS I/O + image/ML preprocess (stretch, normalize, clip,
fits_meta, lupton). Spectroscopy analysis does not belong there.

torchsky already owns:

- `torchsky.spectral` — `Spectrum1D`, `DataCube`, resample helpers
- `torchsky.wcs.spectral` — WAVE/FREQ/VELO transforms
- `torchsky.spectrophotometry` — passbands
- Map cube spectral affine/NUFFT operators
- `torchsky.transforms.RedshiftShift` — lightweight index interp

## Decision

- Torchsky owns low-level, tensor-native spectral geometry: flux-conserving
  rebinning and wavelength-aware relativistic Doppler shifts.
- These are free functions in `torchsky.spectral` so batched tensor users do not
  need to construct `Spectrum1D` objects.
- The index-based random `RedshiftShift` transform is removed rather than kept
  as a competing physical API.
- Continuum estimators, phase folding, band arithmetic, and application-level
  preprocessing are not copied from Torchfits without a second demonstrated
  consumer and benchmark.
- Torchfits remains the required FITS/table I/O boundary; it does not own these
  transforms.

This deliberately absorbs primitives, not the deleted Torchfits module tree.
