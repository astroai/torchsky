# World-layer scorecard (2026-08-16)

Mac CPU evidence after Ishioka 2018 skip-2 (spin 1/2 fused RING). Linux CANFAR
VOS trees were still missing at write time (`vos:sfabbro/torchsky-runs/` had
`bin/` only). Do not claim Linux or CUDA SHT leadership from these rows.

| Job | Machine | Result | Claim |
|---|---|---|---|
| `map2alm_spin` fused Ishioka vs healpy | Mac aarch64 | Ishioka faster than Wigner and healpy; rel L2 ~1e-13 | Mac CPU claimed; Linux pending VOS |
| `alm2map_spin` fused Ishioka vs healpy | Mac aarch64 | Beats Wigner ~15–18%; still ~1.16–1.38× healpy | Not claimed |
| Sparse `get_values` vs healsparse | Mac aarch64 | Coverage-block path, nest-aligned, nside=256, 50k queries: still ~5–6× healsparse | Not claimed |
| Fused SHT CUDA/MPS | — | Host copy of GPU tensors into CPU C++ | Not a GPU SHT |
| Polygons | CPU | Exact area atol 1e-12 sr vs spherical-geometry | CPU only |
| WCS CUDA | CANFAR | Requires astropy installed before pytest collection | Runner fixed; artifacts pending |
| Coupled `-TAB` inverse | CPU | Fail-loud unless `pixel_n_dim == world_n_dim` | Documented |

Git SHA at snapshot: local `main` including this world-layer work. Refresh from
`scripts/fetch_canfar_wcs_sphere_run.sh` when a run tree appears under VOS.
