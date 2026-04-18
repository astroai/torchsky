# BENCHMARK_HISTORY.md

This document tracks the historical performance gates and competitive benchmarks for `torchsky`. It serves as a record of accuracy, throughput, and parity against established astronomy tools.

---

## 1. Map Fitting and Field Convergence

### Field Fit Unified Regimes
Measurements of RMSE (Root Mean Square Error) for various field models across different observational regimes.

| Regime | Model | Full RMSE | Support RMSE | Uniform RMSE |
| :--- | :--- | :---: | :---: | :---: |
| **Local Exact-Band TPV** | `coadd_bilinear` | `0.03934` | `0.03934` | `0.76263` |
| | `sky_encoder_mlp` | `0.14483` | `0.13497` | `0.14483` |
| | `operator_constrained` | `0.31319` | `0.26526` | `0.31319` |
| **Local Broad-Band TPV** | `coadd_bilinear` | `0.04009` | `0.04009` | `0.79481` |
| | `sky_encoder_resmlp` | `0.12374` | `0.07799` | `0.12374` |
| **Sparse Exact 3-band** | `wire_implicit` | `0.006924` | `0.006924` | `0.005201` |

### INR Status & Wins
- **Exact-band TAN/TPV:** `torchsky` consistently beats bilinear coadd on uniform target grids.
- **Broad-band Local:** Feature fields achieve superior coherence across exposures compared to baseline stacking.

---

## 2. Competitive Parity (Ground Truth vs Stacks)

Benchmarks comparing `torchsky` against `swarp`, `montage`, `dfreproject`, and `drizzle` on large union mosaics.

### Large Union Rectangular Masked (1024x768)
| Library | Variant | Time (ms) | RMSE | Flux Error | SNR (dB) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **torchsky** | bilinear | 21466 | **1.2000** | 2.9371 | -1.069 |
| **swarp** | bilinear | 3224 | 1.3353 | 3.4429 | -1.997 |
| **dfreproject** | bilinear | 2183 | 2.6345 | 6.7581 | -7.898 |
| **drizzle** | point | 31923 | 2.0336 | 4.6182 | -5.621 |

### Performance Summary (Aggregate)
| Size | Fastest | Best Accuracy | Best SNR | Best Flux |
| :--- | :--- | :--- | :--- | :--- |
| **1024x768** | `dfreproject` | `torchsky` | `torchsky` | `torchsky` |
| **2048x1536** | `swarp` | `dfreproject` | `dfreproject` | `dfreproject` |
| **512x384** | `dfreproject` | `dfreproject` | `dfreproject` | `dfreproject` |

---

## 3. WCS Inversion Performance

Throughput for inverse WCS (world-to-pixel) kernels on 1M points.

| Case | Method | Time (ms) | Throughput (Mpts/s) | Max Pixel Error |
| :--- | :--- | :---: | :---: | :---: |
| **TPV Affine** | `torchsky:fused` | 239.4 | **4.17** | 2.8e-07 |
| | `torchsky:tensor` | 344.3 | 2.90 | 2.8e-07 |
| **TPV Nonlinear** | `torchsky:fused` | 1404.8 | 0.71 | 1.6e-06 |
| | `torchsky:tensor` | 1226.8 | **0.81** | 2.0e-06 |

---

## 4. Multi-field & Mosaic Regression Gates

| Case | Best Accuracy | Payload | Fastest Time |
| :--- | :--- | :--- | :---: |
| **TAN Multifield Cube** | `coadd:bilinear` | 0.08298 RMSE | 4.41 ms |
| **TPV Multifield Cube** | `coadd:bilinear` | 0.1735 RMSE | 44.33 ms |
| **TAN Multires Noise** | `coadd:bilinear` | 0.0497 RMSE | 2.24 ms |
| **TPV Flux Multires** | `coadd:bilinear` | 0.1358 RMSE | 36.30 ms |

---

## 5. Map Compatibility Gate

Verification against standard Astropy/SIP fixtures.

| Case | Fastest Backend | Disagreement (vs reproject) |
| :--- | :--- | :---: |
| **astropy_sip_fixture** | `reproject_interp` | 0.0 |
| **astropy_sip_mef** | `dfreproject` | 0.0 |
| **shifted_wcs** | `torchsky:bilinear` | 0.0 |
