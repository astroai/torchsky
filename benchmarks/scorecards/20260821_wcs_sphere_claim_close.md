# WCS/sphere claim-or-close (2026-08-21)

Mac numbers never count. Linux rows below are either a claimed CANFAR
artifact or a documented ceiling. Design cuts are optional extras: no
`SkyCoord` class, no `gwcs` / `jplephem` / `sgp4` on `import torchsky`.

## CSC accuracy

- Fused C++ Chan–O’Neill (CPU `wcs_csc_x2s` / `wcs_csc_s2x`) matches
  WCSLIB/astropy at ~3e-10″.
- `TORCHSKY_WCS_FUSED=0` still uses that scalar C++ polynomial on CPU
  float64 (batched ATen `addcmul` Horner is SIMD-contracted and sits at
  mas). CUDA/MPS keep ATen; **do not claim CUDA CSC** until `world-cuda`
  on a SHA that includes `wcs_csc_x2s` is green at 1e-10°.
- The old 8.4 mas Paper II row was the ATen SIMD path, not fused C++.

## FREQ-GRA

Leftover ~0.6 Hz on ~1e14 Hz is float noise after the Edlén fix. Gate
relative error (`<1e-11`). The stale 29 Hz scorecard line is wrong.

## Paper III / IV

Named Paper III chains: `FREQ-W2F`, `WAVE-F2W`, `AWAV-F2A`, `ENER-F2E`,
`WAVN-F2N`, `VOPT-W2V`. Unknown algorithms fail closed. `SPECSYS` is
stored, not applied. Paper IV `TREFPOS` fixtures: TOPOCENTER, GEOCENTER,
BARYCENTER, HELIOCENTER, EMBARYCENTER (plus optional `TREFDIR`).

## Throughput ceilings (unsigned until a HEAD CANFAR JSON)

| Job | Status |
|---|---|
| Linux SHT | Not claimed. Launch `sht-profile` on HEAD. Healpy stays the speed peer. |
| cuHPX | Spike `scripts/run_canfar_cuhpx_sht_spike.sh`. No fused CUDA Ishioka until `go-scalar-cuda`. |
| HealSparse `get_values` | SHA `934b02fa`: 0.84–0.95× (not claimed). `ud_grade` 2.7–3.5× **leads** on that stamp. |
| MOC cone / union | SHA `775762c2` leaf-era: cone 0.047×, union 8e-4×. Remeasure hierarchical `from_circle` on HEAD before claiming. |
| Paper II XPH/TSC/QSC | Claim per-code, never 6/6. Old stamp: PCO/CSC/NCP first; XPH/QSC/TSC behind. |
| Spectral vs WCSLIB | 26/55 first at N=1e5. Claim the codes that lead; fail loud on linear FREQ/WAVE and time scales. |
| Coupled `-TAB` inverse | **Correct, not fast**: 27 s CPU / 165 s CUDA at N=10k. Do not CUDA-optimize. |
| Small-N zenithal | Large-N claim only if N=256 stays ~2–3× astropy after serial `n≤8192`. |
| Catalog | Indexed-vs-coarse is torchsky-vs-torchsky. Astropy SkyCoord peer is `bench_catalog_compat.py` on the CANFAR `catalog` mode. |

## Device residency

CUDA `query_disc` copies a length-3 `vec` to the CPU kernel. Local nside=512
timing: host copy is 2.9% of query wall (below the 10% port threshold). The
MPS on-device path is not ported. Exact `NativeExactSphericalRegion` stays CPU.

## Optional extras (Wave 4)

- Frames: ecliptic-of-date (TT MJD + IAU 2006 NPB), Galactocentric
  (IAU Galactic + explicit R0/z_sun), LSR (explicit Schönrich UVW).
  Not a SkyCoord graph.
- `torchsky.wcs.astropy_interop`: `from_astropy` / `to_astropy`.
- `torchsky.wcs.ephemeris.get_body`: `jplephem` extra + DE kernel path.
- `torchsky.wcs.tle.catalog_state_from_tle`: `sgp4` extra, frame `TEME`.
- `torchsky.wcs.gwcs_live.freeze_live_gwcs`: `gwcs` extra, then freeze.
  Core `gwcs_sampling.py` still has no `import gwcs`.
