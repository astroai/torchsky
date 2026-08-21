# World-layer CANFAR jobs (2026-08-20) — pending sign-off

Launch matrix in `scripts/launch_canfar_wcs_sphere.sh`:

- Existing: `cpu` / `cuda` / `suite` / `sht` / `spectral` / `sparse` / `spin` / `geometry` / `cuhpx`
- World-layer (or `world` to submit them together): `wcs-paper2`, `wcs-tpv`,
  `wcs-spectral`, `wcs-tabular`, `sht-profile`, `moc`, `catalog`, `observed`,
  `world-cuda`. `geom-cuda` remains a polygon-only CUDA rerun.

Fetch with `scripts/fetch_canfar_wcs_sphere_run.sh <run_id>`.

Local smokes (tiny N, not Linux evidence):

```bash
pixi run bench-wcs-paper2
pixi run bench-wcs-spectral-temporal
pixi run bench-wcs-tabular
pixi run bench-wcs-small-n-paths
pixi run bench-observed-place
pixi run sphere-bench-moc
# device (CUDA if present, else skip; MPS for observed/geom tests):
pixi run -e test -- pytest tests/test_wcs_observed_frames.py tests/test_sphere_polygons.py tests/test_wcs_core.py -q -k "cuda or mps"
```

**Not claimed** until a Linux/CUDA artifact (commit, image, device, command,
wall, peak memory) is on VOS and xhigh signs `docs/compatibility.md`:

- Paper II large-N (`PCO`, `XPH`, `TSC`, `CSC`, `QSC`, `NCP`) including `--device cuda`
- TPV forward 1.05–1.20× vs astropy; small-N two-arg vs stacked vs `compile()`
- Spectral/temporal WCS vs WCSLIB (`bench_wcs_spectral_temporal.py --device cuda`)
- Coupled `-TAB` (`bench_wcs_tabular.py --device cuda`)
- SHT nside=128 8-CPU `TORCHSKY_SHT_PROFILE` vs healpy (SHA `389f9dfd`+ equatorial in-process FFT)
- CUDA ATen ring SHT and cuHPX spike (`scripts/run_canfar_cuhpx_sht_spike.sh`)
- mocpy cone / contains / union (`bench_sphere_core.py --moc-only`)
- healsparse `ud_grade` / `get_values` interleaved
- `crossmatch_sky` N=1e6 (with pruned) and N=1e7 (`--skip-decl-pruned`); CUDA N=2e4 (`catalog_cuda.json`)
- pinned-EOP observed place including AltAz (`benchmarks/bench_observed_place.py --device cuda`)
- polygon contains on CUDA (`world-cuda` / `geom-cuda --device cuda`)
- sparse interpolate / ud_grade on CUDA (`sphere_sparse_cuda.json`)

Fused CUDA Ishioka stays off unless the ATen + cuHPX rows say the leftover cost is the Ishioka kernel on device.
