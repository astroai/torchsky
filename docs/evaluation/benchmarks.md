# Reference benchmarks

These tables are a bounded, single-machine snapshot of operations `torchsky`
owns. They answer "what does this kernel cost on this class of input", not
"is torchsky scientifically correct" and not "is it faster than another
package". Upstream parity, survey-scale validation, real-data checks, and
comparable scorecards belong to
[`torchsky-harness`](https://github.com/astroai/torchsky-harness).
Torchsky retains production implementations and the bounded kernel snapshots below.

## Environment

| Field | Value |
|---|---|
| Machine | `Apple M2 (arm64)` |
| GPU | `not used` |
| Operating system | `macOS-14.8.2-arm64-arm-64bit-Mach-O` |
| PyTorch | `2.13.0` |
| torchfits | `1.0.0` |
| torchsky | `0.4.0` |
| Threads | `1` |
| Measured | `2026-10-23T00:12:17+00:00` |
| Commit | `8f18ebdfd584e743a9a09d6b6471860675dfdb47` |

## WCS

| Case | Parameters | Median | Throughput |
|---|---|---|---|
| `wcs_pixel_to_world_tan` | n=10,000, float64, TAN | 0.000636 s | 15.726 Mpoints/s |
| `wcs_world_to_pixel_tan` | n=10,000, float64, TAN | 0.000464 s | 21.573 Mpoints/s |
| `wcs_pixel_to_world_tan` | n=200,000, float64, TAN | 0.01517 s | 13.185 Mpoints/s |
| `wcs_world_to_pixel_tan` | n=200,000, float64, TAN | 0.01195 s | 16.737 Mpoints/s |

## HEALPix

| Case | Parameters | Median | Throughput |
|---|---|---|---|
| `ang2pix` | n=10,000, nside=1024, NESTED | 0.000244 s | 41.061 Mpoints/s |
| `ang2pix` | n=200,000, nside=1024, NESTED | 0.004118 s | 48.569 Mpoints/s |
| `get_interp_weights` | n=50,000, nside=512, bilinear | 0.004848 s | 10.313 Mpoints/s |
| `query_disc` | nside=512, radius=0.5 deg | 0.008405 s | 22.908 Mpixels/s |

## Spherical harmonics

| Case | Parameters | Median | Throughput |
|---|---|---|---|
| `anafast` | nside=32, lmax=64 | 0.001564 s | 7.859 Mpixels/s |
| `anafast` | nside=128, lmax=256 | 0.03108 s | 6.325 Mpixels/s |

## Catalogs

| Case | Parameters | Median | Throughput |
|---|---|---|---|
| `crossmatch_sky (best)` | left=20,000, right=20,000, radius=2 arcsec | 1.028 s | 0.019 Mleft rows/s |

## Mapmaking

| Case | Parameters | Median | Throughput |
|---|---|---|---|
| `coadd (bilinear)` | 4 x 128x128, TAN | 0.02393 s | 2.738 Minput pixels/s |
| `coadd (bilinear)` | 4 x 512x512, TAN | 0.09392 s | 11.164 Minput pixels/s |

## How these were measured

- One process, one thread (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`),
  CPU unless the environment table names a GPU.
- One warm-up call, then the median of three timed calls (`anafast`,
  `crossmatch_sky`, and `coadd` use two timed calls to keep the run bounded).
- Inputs are synthetic, float64 for coordinate geometry and float32 for image
  coadds. Sizes are chosen to expose scaling, not to represent a survey.
- Throughput counts the operation's natural unit: transformed points, analysed
  map pixels, left catalog rows, or input image pixels.

Regenerate the artifact and this page on the reference machine:

```bash
pixi run docs-bench
pixi run docs-bench-tables
```

The JSON snapshot is
[`docs/data/reference_benchmarks.json`](../data/reference_benchmarks.json).
Treat numbers from different machines as different measurements; do not compare
them without recording both environments.

## What the numbers do and do not show

- They show the cost of the current Python/Torch path and of the optional
  compiled kernels when installed. A snapshot taken with
  `TORCHSKY_DISABLE_CPP=1` must say so; the generator records the device but
  not that flag, so regenerate without it.
- They do not gate CI: shared runners are too noisy, and a regression gate
  belongs in the bounded performance smoke tests (`tests/test_perf_gate.py`).
- They are not a substitute for the harness scorecards. See
  [parity and correctness evidence](parity.md) for what is checked against
  trusted references.

See also: [performance guide](performance.md) and the
[runtime contract](../reference/runtime-contract.md).
