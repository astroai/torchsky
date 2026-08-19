# Spin SHT after in-process polar FFT (2026-08-19)

Linux CANFAR jobs at SHA `425af888` restored in-process polar RING FFTs
(Cooley-Tukey/Bluestein). Equatorial belts stay batched ATen. Do not claim
Linux SHT. Mac rows from the same SHA still stand (`alm2map_spin` ~1.6x
healpy at nside=128; `map2alm_spin` leads).

Artifacts:

| Run | Session | Result |
|---|---|---|
| `sphere-spectral-20260819-071402` | `wcyt3m58` | `rc=0`. Gist https://gist.github.com/sfabbro/11d787c07c4a2ab36d8323ba61f86ee5 |
| `sphere-spin-20260819-071411` | `r83mixp2` | `rc=0`. Rel L2 failures empty. Throughput gate off. |

Healpy 1.20.0, torch 2.10.0. Ratio is healpy/torchsky (higher is better for
torchsky). Times are medians from `sphere_spectral_nside{32,64,128}.json`.

## Spectral vs healpy (Linux CPU, SHA `425af888`)

| nside | lmax | map2alm_spin torch / healpy | ratio | alm2map_spin torch / healpy | ratio |
|---|---:|---|---:|---|---:|
| 32 | 32 | 3.27 ms / 0.301 ms | 0.092 | 6.44 ms / 0.258 ms | 0.040 |
| 64 | 63 | 4.44 ms / 1.04 ms | 0.23 | 11.6 ms / 0.979 ms | 0.084 |
| 128 | 128 | 6.68 ms / 4.81 ms | 0.72 | 193 ms / 3.74 ms | 0.019 |

Same-shape historical Linux nside=128 rows (not same machine):

| SHA | Polar FFT | map2alm ratio | alm2map ratio |
|---|---|---:|---:|
| `9e65cbc` | in-process | 0.66 (1.5x) | 0.33 (3.0x) |
| `934b02fa` | per-nph ATen | 0.14 (7.3x) | 0.088 (11x) |
| `425af888` | in-process | 0.72 (1.4x) | 0.019 (52x) |

Analysis is back in the `9e65cbc` class. Synthesis is much worse than both
prior rows. Polar FFT is not the 193 ms: `map2alm_spin` uses the matching
analysis FFTs and is 6.7 ms. Working hypothesis: `SpinMPool` sized from
host `hardware_concurrency()` while the job requested 8 CPUs. Interpolate
is store-heavy; oversubscribe thrashes. Integrate is a reduction and stays
cheap.

## Spin matrix (SHA `425af888`, `--no-throughput-gate`)

`ratio_vs_healpy` = healpy_time / torch_time. Extended large bucket is
nside=64, lmax=48.

| case set | bucket | map2alm | alm2map |
|---|---|---:|---:|
| default | small | 0.068 | 0.051 |
| extended | small | 0.077 | 0.065 |
| extended | medium | 0.141 | 0.046 |
| extended | large | 0.337 | 0.081 |

Versus `934b02fa` extended large: map2alm 0.118 to 0.337; alm2map 0.107 to 0.081.
Rel L2 failures empty.

## Not claimed

- Linux CPU SHT vs healpy.
- Fused CUDA/MPS Ishioka. CUDA/MPS SHT stays on the ATen ring path.
- cuHPX as a backend.
