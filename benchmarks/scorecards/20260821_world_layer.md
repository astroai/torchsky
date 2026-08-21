# World-layer CANFAR ingest (2026-08-21)

Commit `775762c22edf4d970c7507b630afaad534e7de3d` on `origin/main`.
Image `images.canfar.net/astroai/notebook:latest`. Server **production**
(`ws-uv.canfar.net`); staging returned HTTP 503. Requested 8 CPU / 32 GB
(CUDA job: +1 GPU). Astropy 8.0.1, torch 2.10.0, Python 3.13.14.
Fetch: `bash scripts/fetch_canfar_wcs_sphere_run.sh <run_id>`.

Do not treat these rows as a claimed replacement of astropy.wcs, mocpy, or
healsparse. The 17 Aug main WCS suite (CPU 87/90, CUDA 90/90 large-N on SHA
`9e65cbc`) is unchanged; this ingest is the Paper II / TPV / spectral /
tabular / observed / moc / CUDA device follow-up.

## Sessions

| Mode | Run ID | Session | Wall (HEARTBEAT) | rc | Peak RSS seen |
|---|---|---|---|---:|---|
| wcs-paper2 | `wcs-paper2-20260821-011146` | `dkva8vuk` | 01:12:44–01:14:11Z (87 s) | 0 | ~1.5 GB |
| wcs-tpv | `wcs-tpv-20260821-011146` | `r7tvx584` | 01:12:47–01:15:31Z (164 s) | 0 | ~0.9 GB |
| wcs-spectral | `wcs-spectral-20260821-011146` | `o6clsacf` | 01:12:43–01:14:05Z (82 s) | 0 | ~0.7 GB |
| wcs-tabular | `wcs-tabular-20260821-011146` | `zkst0ke2` | 01:12:42–01:20:11Z (7.5 min) | 0 | ~0.7 GB |
| moc | `moc-20260821-011146` | `lmthrc9u` | 01:12:47–01:14:26Z (99 s) | 0 | ~0.7 GB |
| observed | `observed-20260821-011146` | `e1sga5p8` | 01:12:47–01:14:06Z (79 s) | 0 | ~0.7 GB |
| world-cuda | `world-cuda-20260821-011146` | `shzangza` | 01:12:57–01:50:20Z (37 min) | 0 | ~1.6 GB |
| catalog | `catalog-20260821-011146` | `gk4wzdv3` | still running at scorecard write | — | ~3.4 GB |

Gists (secret HEARTBEAT fallback):
[paper2](https://gist.github.com/sfabbro/773663c45da7d3418a9101d8157a6024),
[spectral](https://gist.github.com/sfabbro/70eb5a7c01c6d86cffe8042c1b1f5605),
[observed](https://gist.github.com/sfabbro/daa08c0a01ba4fdde410aeb6ebdf8790),
[moc](https://gist.github.com/sfabbro/9162209f8cb926e183ee21c6ac0aab21),
[tpv](https://gist.github.com/sfabbro/ac93a3306546c8c5065a46fd217db990).

## Paper II CPU (`bench_wcs.py`, N=1e5 mixed, vs astropy CPU)

All 6 codes 100000/100000 valid.

| Code | fw_x | inv_x | max ang | max inv px |
|---|---:|---:|---:|---:|
| PCO | 1.91 | 1.57 | 5.1e-10″ | 6.2e-8 |
| XPH | 0.92 | 1.38 | 1.6e-8″ | 3.3e-12 |
| TSC | 0.53 | 1.87 | 3.7e-10″ | 0 |
| CSC | 1.48 | 2.87 | 0.011″ | 1.7e-5 |
| QSC | 0.73 | 2.55 | 4.6e-9″ | 1.0e-12 |
| NCP | 3.63 | 17.3 | 3.9e-8″ | 8.2e-10 |

Forward first: PCO, CSC, NCP. Behind: XPH, TSC, QSC. CSC angular error is the
Chan & O'Neill polynomial vs WCSLIB, not a fused-kernel bug. Not claimed as a
Paper II suite win.

## Paper II / TPV CUDA (`world-cuda`, same N, vs **CPU** astropy)

CUDA pytest `-k cuda`: 15 passed, 0 failed, 0 skipped (29 s).

| Code | fw_x | inv_x | max ang |
|---|---:|---:|---:|
| PCO | 27.5 | 17.8 | 3.7e-10″ |
| XPH | 16.7 | 19.4 | 1.6e-8″ |
| TSC | 5.11 | 5.68 | 1.8e-10″ |
| CSC | 4.90 | 6.68 | 0.011″ |
| QSC | 4.93 | 6.22 | 4.6e-9″ |
| NCP | 23.8 | 40.4 | 4.0e-8″ |
| TPV N=2e5 | 45.9 | 48.2 | 2.0e-10″ |

These speedups are GPU torchsky vs CPU astropy. Do not claim a same-device
CUDA peer win. Mixed TPV roundtrip p99 is ~1700 px off-axis (same as CPU).

## TPV CPU

`bench_wcs.py` N=2e5 mixed: forward **0.92×** astropy (43.0 ms / 46.8 ms). Not
claimed. Inverse 13.5× vs astropy wall, but mixed roundtrip p99 is ~1700 px.

Dedicated `bench_wcs_tpv_inverse.py` N=1e6, CPU:

| Case | tensor_core | astropy | max inv px (torchsky) |
|---|---:|---:|---:|
| affine | 9.3 ms | 3427 ms | 2.9e-7 |
| nonlinear | 124 ms | 3360 ms | 2.0e-6 |

CUDA tensor_core: affine 0.5 ms, nonlinear 6.5 ms (vs CPU astropy 0.67 s / 0.98 s).

Small-N N=256: TAN/SIN/ARC max |dpx| < 6e-8. TPV compile=ok but max |dy| = 1110 px
(same mixed-FOV inverse hole). Small-N CUDA is launch tax vs CPU astropy.

## Spectral / temporal vs WCSLIB, N=1e5

CPU: 55 rows, **26 first / 7 tie / 22 behind** (1.05 / 0.95 cuts). Linear
FREQ/WAVE ~0.89–0.91×. `FREQ-GRA` leftover is relative float noise (~0.6 Hz
on ~1e14 Hz), not 29 Hz. Time-scale conversions are slower than astropy
(CPU UTC→TAI 8.7×). CUDA: 49/55 first vs CPU WCSLIB; same relative `FREQ-GRA`.

## Coupled `-TAB`

CPU N=10k grid=8: forward 1.4 ms, err 0. Inverse **27.5 s**, err 5.5e-11 px.
CUDA: forward 1.1 ms, inverse **165 s**, same err. Correct and not a throughput
claim.

## Observed place (pinned EOP, N=4096)

CPU pytest: 40 passed, 3 skipped. CIRS max sep **2.13e-9″**, TETE **5.36e-9″**,
NPB resid 2.4e-14. Medians: GCRS 0.33 ms, CIRS 2.02 ms, TETE 1.55 ms, AltAz
1.62 ms. AltAz has no ERFA oracle. CUDA seps match CPU; CIRS 3.92 ms / AltAz
4.58 ms (small-N device tax). No C++ kernel.

## MOC vs mocpy (order 8)

| Op | torchsky | mocpy | ratio ts/mocpy |
|---|---:|---:|---:|
| cone | 6.48 ms | 0.304 ms | 0.047 |
| contains 50k | 3.98 ms | 3.95 ms | 0.99 |
| union | 1.42 ms | 1.19 µs | 0.00084 |

Contains is a tie. Cone and union are not claimed.

## CUDA geom / sparse (torchsky-only; peers missing on `cuda-bench`)

Contains 1.03e8 pts/s at N=2e5. Sparse interpolate nearest 9.3e7 qps, bilinear
3.6e7 qps, nside 256, 200k queries. Not claimed vs spherical-geometry or
healsparse (those packages were not installed on this CUDA image).

## Catalog

CPU N=1e6 + N=1e7 `--skip-decl-pruned` was still running when this card was
written. CUDA N=20k×70k `crossmatch_sky` 0.212 s on device (matches coarse
HEALPix 0.212 s; 10718 matches).

## Not claimed / still out

- Paper II CPU suite first; TPV mixed forward; spectral vs WCSLIB suite first
- Linux SHT (unchanged)
- healsparse `get_values` / interleaved `ud_grade` (unchanged)
- mocpy cone/union
- Live GWCS graphs in core (optional `gwcs` extra freezes then samples)
- SkyCoord class (optional `astropy_interop` copies CatalogState)
- fused CUDA Ishioka (blocked on cuHPX `go-scalar-cuda`)
- Linux SHT, mocpy cone/union, healsparse `get_values`, Paper II suite-first
