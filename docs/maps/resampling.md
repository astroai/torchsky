# Resampling
<!-- TODO(0.5.0 docs): intro -->


## Kernels

| Kernel family | Typical analogue | Notes |
|---|---|---|
| `nearest`, `bilinear` | Nearest and linear regridding | Baseline for compatibility checks |
| `bicubic`, `catmull_rom`, `mitchell_netravali` | Cubic families | Polynomial definitions differ across tools |
| `lanczos2` to `lanczos5` | Lanczos mosaickers (SWarp) | Wider support, higher cost |
| `fft_shift`, `fft_diagonal_affine`, `fft_linear_affine`, `periodic_linear_affine` | Restricted FFT remappers | Same-shape linear frames with explicit boundaries |
| `drizzle_point`, `drizzle_square`, `drizzle_turbo`, `drizzle_gaussian` | STScI drizzle drop kernels | Native; square drizzle uses exact box overlap |
| `conservative_polygon` | Conservative overlap | Native overlap remapper |
| `splat_point`, `splat_box` | Forward deposition | Approximate differentiable splats |

Drizzle conventions. `drizzle_square` and `conservative_polygon` overlap each
projected source-pixel polygon exactly with the target pixels; `drizzle_turbo` and
`splat_box` use the separable box overlap when the local source-to-target Jacobian
maps axes onto axes (to float32 resolution) and the exact polygon overlap for rotated
or sheared pairs; IFU cube drizzle scales each voxel footprint by the local
detector-to-cube Jacobian in x, y and wavelength. `drizzle_point` differs from STScI
`drizzle` 3.0 at the grid edge: torchsky keeps drops whose centre lies in the outer
half-pixel rim (target coordinates in `[-0.5, n - 0.5]`), whereas STScI 3.0 drops those
outside `[0, n - 1]`; this is intentional so no drop overlapping the grid is discarded.

Target-to-source mapping uses a verified interpolated lattice for SIP and TPV;
`mapping_mode("exact")` selects the per-pixel solve.

## Drizzle
<!-- TODO(0.5.0 docs): drizzle as a resampling method (alongside nearest, bilinear, Lanczos, FFT, conservative) -->
