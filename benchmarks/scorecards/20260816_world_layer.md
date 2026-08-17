# World-layer scorecard (2026-08-16)

Mac CPU evidence after Ishioka 2018 skip-2 (spin 1/2 fused RING). Linux CANFAR
VOS trees were still missing at write time (`vos:sfabbro/torchsky-runs/` had
`bin/` only). Do not claim Linux or CUDA SHT leadership from these rows.

| Job | Machine | Result | Claim |
|---|---|---|---|
| `map2alm_spin` fused Ishioka vs healpy | Mac aarch64 | Ishioka faster than Wigner and healpy; rel L2 ~1e-13 | Mac CPU claimed; Linux pending VOS |
| `alm2map_spin` fused Ishioka vs healpy | Mac aarch64 | Beats Wigner ~15–18%; still ~1.16–1.38× healpy | Not claimed |
| Sparse `get_values` vs healsparse | Mac aarch64 | Fused CPU fill-block gather. Nest-aligned nside=256, 50k interleaved: matches healsparse (~0.20 ms median, ratio ~1.03). Valid pixels `maxabs 0.0`. Linux unconfirmed. | Mac CPU match claimed |
| Sparse `ud_grade` vs healsparse | Mac aarch64 | Coverage-block degrade when coverage nside is unchanged. Nest 10% coverage, interleaved: nside=256→128 ratio ~1.02; nside=512→256 ratio ~1.09; dense `maxabs ~4e-16`. Linux unconfirmed. | Mac CPU match claimed |
| Fused SHT CUDA/MPS | — | Host copy of GPU tensors into CPU C++ | Not a GPU SHT |
| Polygons | CPU | Exact area atol 1e-12 sr vs spherical-geometry | CPU only |
| WCS CUDA | CANFAR | Requires astropy installed before pytest collection | Runner fixed; artifacts pending |
| Coupled `-TAB` inverse | CPU | Fail-loud unless `pixel_n_dim == world_n_dim` | Documented |

Git SHA at snapshot: local `main` including this world-layer work. Refresh from
`scripts/fetch_canfar_wcs_sphere_run.sh` when a run tree appears under VOS.
