# World-layer CANFAR jobs (2026-08-20) — pending sign-off

Superseded by [`20260821_world_layer.md`](20260821_world_layer.md) (SHA
`775762c2`, production CANFAR). Launch matrix remains in
`scripts/launch_canfar_wcs_sphere.sh`.

**Still unsigned after that ingest:**

- CPU catalog N=1e6 / N=1e7 (job still running when 20260821 was written)
- SHT nside=128 8-CPU `TORCHSKY_SHT_PROFILE` vs healpy
- CUDA ATen ring SHT and cuHPX spike (`scripts/run_canfar_cuhpx_sht_spike.sh`)
- healsparse `ud_grade` / `get_values` interleaved (no new row on this SHA)

Fused CUDA Ishioka stays off unless the ATen + cuHPX rows say the leftover cost is the Ishioka kernel on device.
