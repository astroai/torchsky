# Performance highlights

Headline results from the ADASS XXXVI (2026) evaluation, run on real JWST/NIRCam
and CFHT/MegaCam data. Every number on this page is copied from the harness
record, not re-measured here. The full job sources, per-run provenance
(scripts, `launch.env`, sha256 sums, `pip freeze`), and result files are in
[`torchsky-harness/campaigns/adass2026`](https://github.com/astroai/torchsky-harness/tree/main/campaigns/adass2026).

!!! note "How to read these numbers"
    They are wall-clock times on one machine class for specific inputs, not
    general speed claims. Kernel-level snapshots are in
    [Kernel benchmarks](benchmarks.md); correctness against trusted references is
    in [Parity evidence](parity.md); tuning advice is in
    [Scaling and tuning](scaling.md).

## Environment as recorded

| Item | Value |
|---|---|
| Subject | torchsky `main`, installed from source tarballs (speed runs at commits `82522bf` and `df37916`; science runs at `9749c80`, `14c9a6d`, `3bd7b79`) |
| Software | PyTorch 2.14, torchfits 1.3.1, weightmask 0.3.0 |
| Platform | CANFAR `astroai/base:latest` (SWarp benchmark: `astroai/improc:latest`) |
| Compute | 8 CPU threads and an H100 MIG 1g.12gb slice for the NIRCam speed benchmark; 16 CPU threads for the UNIONS profiling runs |
| Comparators | SWarp 2.41 and STScI drizzle 3.0 |

## NIRCam coadd against SWarp and STScI drizzle

135 NIRCam exposures (566 Mpix) in six bands onto 0.04 arcsec TAN grids.
Wall clock including reads. SWarp receives the SIP distortion converted to TPV.

| Output grid | SWarp 2.41 | torchsky Lanczos-3, CPU / GPU | STScI drizzle 3.0 | torchsky drizzle, CPU / GPU |
|---|---|---|---|---|
| 512 x 512 | 30.1 s | 14.3 s / 5.2 s | 171 s | 71 s / 27 s |
| 5025 x 5025 | 144 s | 221 s / 65 s | 245 s | 149 s / 148 s |

What the table says honestly: on the small grid torchsky is faster than both
legacy codes on CPU and much faster on a GPU slice; on the 5025 x 5025 grid
torchsky Lanczos-3 on CPU (221 s) is slower than SWarp (144 s), while the GPU
run (65 s) is faster, and drizzle is faster on CPU and about equal on GPU.

Outputs agree with the legacy codes:

- torchsky drizzle matches STScI drizzle to better than 2e-6 sigma.
- torchsky Lanczos-3 against SWarp: median difference below 0.1 sigma, scatter
  0.1 to 0.9 sigma.
- CPU against GPU: drizzle is bitwise identical; Lanczos-3 differs by 1e-7 sigma.

The speed-ups over the earlier code (Lanczos-3 784 s to 221 s on CPU and an
out-of-memory failure to 65 s on GPU at 5025 x 5025; drizzle 1849 s to 149 s on
CPU and 584 s to 148 s on GPU) come from footprint cropping per observation, a
verified coarse-lattice target-to-source mapping for SIP and TPV, a fused chunked
separable sampler, memory-budget automatic tiling with host accumulators, an
exact axis-aligned drizzle overlap with a native CPU kernel, and deterministic
CUDA accumulation. Evidence: `results/bench_speed.json` and the
`bench_coadd_82522bf` run.

## UNIONS tile: 280 MegaCam CCDs

Individual MegaCam exposures are reduced with weightmask and coadded with a
torchsky Lanczos-3 weighted coadd onto the official 10k x 10k TAN grid of tile
CFIS.000.229 (flux semantics, automatic tiling).

| Run | Time |
|---|---|
| v5 baseline, 2500 x 2500 tiles, CPU | 2292 s |
| Current code, CPU (automatic tiles) | 711 s |
| Current code, H100 MIG 1g.12gb | 660 s (no out-of-memory) |

Against the official tile, the pixel, photometry, and astrometry statistics are
identical in the CPU and GPU runs: 2 arcsec aperture photometry within 2.1 mmag
(r band) in every magnitude bin and centroids within 1 mas. Against Gaia DR3
propagated to the exposure epoch, the mean offset is (-0.2, +2.9) mas
(east, north) for torchsky and (+0.2, +1.6) mas for the official tile.

The MegaCam headers keep `RADECSYS = 'FK5'` although the astrometry is Gaia
based; read literally this applies a (-29, -9) mas frame bias. torchsky now
emits `CelestialFrameBiasWarning` on such conversions
([Coordinates](../coordinates.md)). Evidence: `unions_coadd_v9` (CPU) and
`unions_coadd_v10_gpu`, with comparison files `unions_compare_v9_cpu.json` and
`unions_compare_v10_gpu.json`.

## SIP evaluation

Nested Horner evaluation of the SIP polynomials replaced materialized power
stacks. For JWST order-5 SIP on 4 million points the forward distortion is 15x
faster and the Newton inverse 4x faster, with results equal to 1e-12 pixel.
Recorded in the [changelog](../changelog.md).

## Forward-model fits

The same evaluation fits SMACS J0723 NIRCam exposures with a chromatic PSF field
and compares sky parameterizations against coadd baselines on held-out exposures
(for example, flux-error NMAD 2.1 sigma for an explicit grid versus 2.8 sigma for
the SWarp-like Lanczos-3 coadd). The protocol and full table are in the
[evaluation README](https://github.com/astroai/torchsky-harness/blob/main/campaigns/adass2026/README.md);
the method is described in [Field formulation](../field-formulation.md).

## Reproducing

Job scripts and launchers live under `campaigns/adass2026/jobs` in the harness.
The harness records that some jobs install torchsky from a source tarball and use
`astropy.io.fits` for legacy-code inputs; migration to torchfits and Pixi
environments is pending, so treat them as evidence of the reported numbers rather
than as a release gate. Harness methodology:
[benchmark guide](https://github.com/astroai/torchsky-harness/blob/main/docs/benchmarks.md).
