# WCS/sphere claim-or-close ingest (SHA `d8d706f6`, relaunch `7564070c`)

Linux CANFAR `astroai/notebook:latest`, 8 CPU, astropy 8.0.1. Mac numbers
never count.

## Relaunch SHA `7564070c` stamp `20260821-1852*`

`world-cuda` `adc196ju` / `world-cuda-20260821-185223` **rc=0**, pytest
**18/18** (`-k cuda`). CSC CUDA without autograd host-copies to the CPU
kernel: mixed max vs astropy is **8.35 mas** (same CPU ceiling); do **not**
claim CUDA CSC accuracy or throughput (CSC forward 1.21× CPU astropy is the
host-copy path). Other Paper II CUDA vs CPU astropy stays first (PCO 26.6×,
NCP 22.9×, TSC 4.78×). Coupled `-TAB` CUDA inverse 12.6 ms / 5.2e-11 px.

`catalog` `va27ywnu` / `catalog-20260821-185211` uploaded
`catalog_compat.json` then continued to N=1e6. Synthetic 10k×85k
`crossmatch_sky` vs SkyCoord is **0.116×** (0.363 s vs 0.042 s); max sep
diff 1.2e-10″. Real VOTable N=18 is ~1.03×. FITS roundtrip **SKIPPED**:
bench called `.to()` on a pyarrow `ChunkedArray` from `torchsky.table.read`
(torchsky bench, not torchfits). Do **not** claim vs SkyCoord. N=1e6 / N=1e7
JSON still pending.

## CSC accuracy (SHA `d8d706f6`)

CPU mixed N=2e4 fused=0/1: median ~1e-12″ vs astropy, **max 8.42 mas /
p99 5.70 mas**. Same ceiling as the old 8.4 mas row. Fused=1 is slower
than fused=0 at N=2e4 (0.48× vs 1.05× astropy). Do **not** claim Linux
CPU CSC vs WCSLIB.

CUDA Paper II N=1e5 CSC vs CPU astropy was 1.83e-10″ on SHA `d8d706f6`
ATen, but `test_paper2_cuda_matches_cpu[CSC]` failed at 8.3e-7°. Superseded
by SHA `7564070c` host-copy (pytest 18/18, mixed max 8.35 mas). Do **not**
claim CUDA CSC.

## FREQ-GRA

CPU and CUDA `max_abs_error` 0.625 Hz on ~1e14 Hz. Relative float noise,
not 29 Hz.

## Paper III / IV

Named WCSLIB-valid chains in tests: `FREQ-W2F`, `WAVE-F2W`, `AWAV-F2A`,
linear `ENER`/`WAVN`/`VOPT`, `WAVE-V2W`. `ENER-F2E` / `WAVN-F2N` /
`VOPT-W2V` are not WCSLIB codes. `SPECSYS` stored, not applied.

## Throughput (this SHA)

| Job | Linux row | Claim |
|---|---|---|
| SHT nside=128 / 8 CPU | `map2alm_spin` 0.41× healpy (10.9 ms vs 4.5 ms); `alm2map_spin` 0.32× (10.7 ms vs 3.3 ms). Profile: synthesize ~6 ms vs interpolate ~2.1 ms. | **not claimed** |
| cuHPX | `verdict: no-go` — `cuhpx` import failed | **no-go**; no fused CUDA Ishioka |
| HealSparse `get_values` | 0.89× nside=256 / 200k; 0.92× nside=512 / 500k | **not claimed** |
| HealSparse `ud_grade` | 2.86× (256→128); 4.63× (512→256) | **claimed** on this stamp |
| MOC order-8 | contains 0.98×; cone 0.061×; union 8.7e-4× | contains unsigned (~1×); cone/union **not claimed** |
| Paper II N=1e5 CPU | PCO 1.89× / NCP 2.55× first; XPH 0.88× / CSC 0.61× / QSC 0.64× / TSC 0.53× | **per-code only** |
| Paper II CUDA vs CPU astropy | SHA `d8d706f6` all first (CSC 3.9× … PCO 32×); job **rc=1** on CSC CUDA-vs-CPU. SHA `7564070c` pytest **18/18** rc=0; CSC mixed max 8.35 mas (host-copy). | throughput except CSC; do **not** claim CUDA CSC |
| Spectral vs WCSLIB N=1e5 | 13 first / 1 tie / 41 behind. Linear FREQ 0.40×, WAVE 0.47×. Time scales first (UTC→TAI 8.3×). CUDA 47/4/4 vs CPU WCSLIB. | **not suite-first**; claim WAVN-LOG, WAVE-GRI, time scales |
| Coupled `-TAB` N=10k | CPU inverse 53 ms / 5e-11 px; CUDA inverse 9.1 ms (`d8d706f6`) / 12.6 ms (`7564070c`) / same err | **correct**; old 27 s / 165 s line is dead |
| Small-N N=256 CUDA | 0.07–0.12× CPU astropy (launch tax) | large-N only; CPU small-N not remasured (`wcs-tpv` not in this stamp) |
| Catalog vs SkyCoord | SHA `7564070c` `catalog_compat.json`: synthetic 10k×85k **0.116×**; real VOTable N=18 ~1.03×; FITS roundtrip skipped (Arrow `.to()`). n1e6/n1e7 still running. | **not claimed** |

## Device residency

`query_disc` length-3 host copy 2.9% of nside=512 wall. Left on the CPU
kernel.

## Optional extras

Unchanged: no `SkyCoord` class; DE/TLE/live GWCS stay extras. Fused CUDA
Ishioka blocked on cuHPX `no-go`.

## Sessions

| Mode | Session | Run | Exit |
|---|---|---|---|
| sht-profile | `ajrnrk3q` | `sht-profile-20260821-174344` | 0 |
| healsparse | `d5tw4p9b` | `sphere-sparse-20260821-174356` | 0 |
| cuhpx | `r20bzc5d` | `cuhpx-sht-20260821-174402` | 0 (no-go) |
| moc | `r4zgt32q` | `moc-20260821-174408` | 0 |
| wcs-paper2 | `q27hjczu` | `wcs-paper2-20260821-174413` | 0 |
| wcs-spectral | `olsngo0v` | `wcs-spectral-20260821-174422` | 0 |
| wcs-tabular | `lifav3to` | `wcs-tabular-20260821-174427` | 0 |
| world-cuda | `adc196ju` | `world-cuda-20260821-185223` | 0 (pytest 18/18) |
| catalog | `va27ywnu` | `catalog-20260821-185211` | running (compat JSON in; n1e6/n1e7 pending) |
| catalog (deleted hung) | `r8sdn6rh` | `catalog-20260821-174431` | killed after `P14A()` write |
| catalog (old) | `gk4wzdv3` / `ahw1eca2` / `b210atfp` | `*-011146` / `*-041934` / `*-044023` | running, VOS HEARTBEAT |
