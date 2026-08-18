# World-layer scorecard (2026-08-17)

Linux CANFAR trees fetched from `vos:sfabbro/torchsky-runs/` after SHA
`9e65cbc` (morning), then gist fallback after SHA `934b02fa` (evening ATen
polar FFT). Mac CPU rows from
[`20260816_world_layer.md`](20260816_world_layer.md) still stand. Do not
treat the Linux SHT or `get_values` rows as claimed replacement.

Artifacts:

| Run | Session | Result |
|---|---|---|
| `wcs-sphere-cuda-20260817-143713` | `wnweyt2u` | Completed. CUDA pytest 5/5. Trees `acceptance_wcs_cuda`, `acceptance_sphere_cuda`. |
| `wcs-sphere-cpu-20260817-143713` | `zbmq77z8` | `cpu_rc=1` after pytest 996/0 fail: Mac-tuned spin ratio gate (`small` median 0.054 < 0.150). WCS + sphere trees uploaded. |
| `sphere-spectral-20260817-144235` | `bz1h7l4n` | Completed. nside 32/64/128 JSON. |
| `sphere-sparse-20260817-144235` | `kaqq63c6` | Completed. nside 256/512 JSON. 50k-query cap. Separate medians, not interleaved. |
| `sphere-spin-20260817-144235` | `e9sknnjh` | Failed Mac-tuned small-N ratio gate. Rel L2 not the failure. |
| `sphere-geometry-20260817-144235` | `zh61gg5a` | Failed Mac-tuned geometry median-ratio gate. |
| `sphere-only-20260817-144235` | `djwe8eiz` | Clone GitHub 503. Heartbeat only. |
| `sphere-spectral-20260817-224511` | `p32plf3x` | `rc=0` on `934b02fa`. VOS registry empty in the container. Gist [5551db4f](https://gist.github.com/5551db4f802794840897a3d592ba35d7). |
| `sphere-sparse-20260817-224511` | `biy3ghbw` | `rc=0`. Gist [ccf7c46a](https://gist.github.com/ccf7c46ab57b90e9646468174820b551). Interleaved, 200k/500k queries. |
| `sphere-spin-20260817-224511` | `i8og8vh0` | `rc=0`. Gist [682ae966](https://gist.github.com/682ae9667ee6c0dc5a245b7bb034ff92). Rel L2 clean; throughput gate off. |

## WCS (Linux)

| Device | Smart first | Large-N first | Claim |
|---|---:|---:|---|
| CPU (`acceptance_wcs_full`) | 140/150 (93.3%) | 87/90 (96.7%) | Large-N first except TPV forward (1.05–1.20× astropy). Not claimed for TPV. |
| CUDA (`acceptance_wcs_cuda`) | 122/150 (81.3%) | 90/90 (100%) | Large-N CUDA first vs `astropy_torch` on that job. Small-N is CUDA launch tax vs CPU astropy (up to ~3.7× at N=1k). Do not claim small-N CUDA WCS. |

Coupled `-TAB` inverse stays fail-loud unless `pixel_n_dim == world_n_dim`.

## Spin SHT vs healpy (Linux CPU, dedicated spectral job)

Times are medians from `sphere_spectral_nside{32,64,128}.json`. Ratio is
healpy/torchsky (higher is better for torchsky).

| nside | lmax | map2alm_spin torch / healpy | ratio | alm2map_spin torch / healpy | ratio |
|---|---:|---|---:|---|---:|
| 32 | 32 | 1.75 ms / 0.289 ms | 0.17 | 1.97 ms / 0.245 ms | 0.12 |
| 64 | 63 | 4.14 ms / 1.03 ms | 0.25 | 4.46 ms / 0.937 ms | 0.21 |
| 128 | 128 | 6.31 ms / 4.17 ms | 0.66 | 9.36 ms / 3.11 ms | 0.33 |

Fat CPU suite (`acceptance_sphere_full`) agrees at small N: `alm2map_spin`
9.6× healpy, `map2alm_spin` 7.7×. Equatorial rings already used ATen
pocketfft; polar rings used a scalar Cooley-Tukey/Bluestein fallback. Polar
rings now use the same ATen FFT as the equator (PyTorch pocketfft on CPU,
cuFFT on CUDA, MPS FFT on Apple GPU). **These rows are pre-ATen-polar.**
Do not claim Linux SHT. Do not start a fused CUDA SHT until a post-fix
Linux CPU row is competitive.

Mac (2026-08-16): Ishioka `map2alm_spin` leads healpy; `alm2map_spin` still
~1.2–1.4×. Unchanged.

## Spin SHT vs healpy (Linux CPU, SHA `934b02fa`, ATen polar FFT)

Same dedicated spectral script. Artifacts are the gists above (VOS PUT failed:
`ivo://cadc.nrc.ca/vault` missing from the container registry). Ratio is still
healpy/torchsky.

| nside | lmax | map2alm_spin torch / healpy | ratio | alm2map_spin torch / healpy | ratio |
|---|---:|---|---:|---|---:|
| 32 | 32 | 1.22 ms / 0.326 ms | 0.27 | 1.81 ms / 0.256 ms | 0.14 |
| 64 | 63 | 4.30 ms / 1.01 ms | 0.24 | 8.66 ms / 0.910 ms | 0.11 |
| 128 | 128 | 23.2 ms / 3.18 ms | 0.14 | 30.7 ms / 2.70 ms | 0.088 |

nside=128 torch is slower than the morning `9e65cbc` row, and healpy on this
job is also faster, so this is not a same-machine ATen delta. It is the
post-ATen Linux row. Healpy stays the SHT speed peer. Do not claim Linux SHT.
Do not start a fused CUDA SHT.

Spin matrix (`--no-throughput-gate`): small-N median ratios ~0.07–0.10;
extended large-N `map2alm_spin` 0.118, `alm2map_spin` 0.107. Rel L2 failures
empty.

## Sparse maps vs healsparse (Linux CPU, SHA `934b02fa`, interleaved)

10% random coverage, nest-aligned queries, **interleaved** timing, full
`--n-queries` (200k at nside=256, 500k at nside=512). Ratio is
healsparse_time / torchsky_time.

| nside | queries | get_values torch / hs | ratio | ud_grade torch / hs | ratio |
|---|---:|---|---:|---|---:|
| 256 | 200k | 1.37 ms / 1.15 ms | 0.84 | 5.42 ms / 19.2 ms | 3.54 |
| 512 | 500k | 5.77 ms / 5.50 ms | 0.95 | 33.3 ms / 90.8 ms | 2.73 |

Linux `get_values` is still behind; do not claim it. Coverage-block `ud_grade`
leads this interleaved healsparse job. Valid-pixel values remain `maxabs 0.0`.
FITS `.hs` is still a hole.

## Sparse maps vs healsparse (Linux CPU, dedicated sparse job)

10% random coverage, nest-aligned queries, **50k cap**, separate medians
(not interleaved).

| nside | get_values torch / hs | ratio (hs/ts) | ud_grade torch / hs | ratio (hs/ts) |
|---|---|---:|---|---:|
| 256 | 0.342 ms / 0.264 ms | 0.77 | 10.4 ms / 21.4 ms | 2.06 |
| 512 | 0.357 ms / 0.288 ms | 0.80 | 24.9 ms / 82.5 ms | 3.31 |

Mac (2026-08-16, interleaved, 50k): `get_values` ~1.03×, `ud_grade`
~1.02–1.09×. Linux `get_values` is behind at this N; do not claim it.
Linux `ud_grade` on this morning job used separate medians; the evening
`934b02fa` job is interleaved (see above). FITS `.hs` is still a hole.

## Geometry (Linux CPU, fat sphere job)

`ang2pix` still leads. Behind healpy at N=200k: `nest2ring` 1.29×,
`ring2nest` 1.21×, `pix2ang_*` 1.12–1.18×, nested interp weights 1.60×.
Those are throughput deficits, not correctness failures. CANFAR geometry
jobs must record ratios without Mac-tuned min-ratio exits.

## Not claimed / out of program

- Fused SHT on CUDA/MPS: host copy of GPU tensors into CPU C++.
- CUDA SHT, cuHPX as a default backend.
- SkyCoord clone, live GWCS graph, STMOC.
- FITS I/O (torchfits), map stackers.
