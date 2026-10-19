# Frequency-Adaptive Field Candidates

BACON and SASNet remain experimental bakeoff candidates. They are not production
presets and were implemented independently rather than merging the external
script branch wholesale.

## BACON

`BaconField` follows the multiplicative-filter construction with frozen sine
frequencies. Its configured bandwidths describe output angular support. The
latent filters receive half that budget because the non-negative sum-of-squares
readout doubles Fourier support. This replaces a Softplus readout, which would
destroy a strict compact-support claim. Multiscale outputs, gradients, devices,
and tensor-only state have focused tests. The remaining gate is equal-budget
held-out scientific recovery against the registered field zoo.

Reference: [BACON: Band-Limited Coordinate Networks for Multiscale Scene
Representation](https://openaccess.thecvf.com/content/CVPR2022/html/Lindell_BACON_Band-Limited_Coordinate_Networks_for_Multiscale_Scene_Representation_CVPR_2022_paper.html).

## SASNet

`SASNetField` is described as spatially localized and frequency-adaptive, not
strictly band-limited: learned spatial masks modulate a frozen frequency
embedding before nonlinear decoding. The finest frozen frequency and hash-grid
resolution are capped from the supplied pixel footprint and finite PSF support,
and the resolved caps are recorded. A caller must supply the physical pixel
footprint; the candidate does not infer one from an arbitrary tensor shape.

Reference: [SASNet: Spatially-Adaptive Sinusoidal Networks for
INRs](https://openaccess.thecvf.com/content/CVPR2026/papers/Feng_SASNet_Spatially-Adaptive_Sinusoidal_Networks_for_INRs_CVPR_2026_paper.pdf).

## Initialization and training disposition

The existing sine field already uses the variance-preserving SIREN-scale
initialization recovered by VI3NR for sinusoidal activations. SASNet's frozen
frequency dictionary adopts the central TUNER idea, but Torchsky does not claim
a complete TUNER implementation yet. Any additional frequency tuning must first
show FP32 parity and a held-out gain under equal parameters, optimizer steps,
wall time, and input information.

References: [TUNER](https://openaccess.thecvf.com/content/CVPR2025/html/Novello_Tuning_the_Frequencies_Robust_Training_for_Sinusoidal_Neural_Networks_CVPR_2025_paper.html)
and [VI3NR](https://openaccess.thecvf.com/content/CVPR2025/html/Koneputugodage_VI3NR_Variance_Informed_Initialization_for_Implicit_Neural_Representations_CVPR_2025_paper.html).

## Admission gate

Both candidates must pass the same serialization/device/gradient contract,
shared WCS/passband/PSF/noise operator tests, equal-budget exposure- and
sky-held-out benchmarks, and calibrated scientific recovery. A visual sharpness
or PSNR win is insufficient. Only a benchmark winner may enter a production
preset; a null result leaves the candidate experimental.
