# Benchmark scorecard index

Committed snapshots for tracking throughput leadership over time. Ephemeral run artifacts remain under `benchmarks_results/` (gitignored).

Archive a new snapshot after a full domain sweep:

```bash
pixi run bench-archive-scorecard -- \
  --label main_domains \
  --runs fits=<fits_run_id>,fitstable=<fitstable_run_id>,wcs=<wcs_run_id>,sphere=<sphere_run_id>
```

| Date | Label | FITS smart | Table smart | WCS large-N | Sphere specialized | Deficits | Git SHA | Snapshot |
|---|---|---:|---:|---:|---:|---:|---|---|
| 20260612 | main_domains | 84/84 | 70/70 | 90/90 | 30/30 | 24 | `193e62eec875` | [20260612_main_domains](20260612_main_domains.md) |
| 20260815 | wcs_sphere_spectral_audit | - | - | 0/0 | 2/3 | 4 | `5f911d00c30c` | [20260815_wcs_sphere_spectral_audit](20260815_wcs_sphere_spectral_audit.md) |
| 20260816 | world_layer | - | - | pending VOS | Mac Ishioka map2alm leads; alm2map ~1.2–1.4×; sparse get_values matches healsparse | 1 | local main | [20260816_world_layer](20260816_world_layer.md) |
| 20260817 | world_layer | - | - | CPU 87/90; CUDA 90/90 large-N | Linux SHT not claimed (`934b02fa` nside=128 ~7–11× healpy). Linux `get_values` 0.84–0.95× hs interleaved. Linux `ud_grade` 2.7–3.5× interleaved | see card | `9e65cbc` + `934b02fa` | [20260817_world_layer](20260817_world_layer.md) |
| 20260819 | sht_polar_fft | - | - | - | Linux SHT not claimed. `425af888` nside=128 alm2map ~52×; `b1f02069` 8-thread cap: interpolate 2.5 ms, polar FFT ~280 ms | see card | `425af888` + `b1f02069` | [20260819_sht_polar_fft](20260819_sht_polar_fft.md) |
| 20260820 | world_layer_pending | - | - | superseded by 20260821 | — | — | local | [20260820_world_layer_pending](20260820_world_layer_pending.md) |
| 20260821 | world_layer | - | - | Paper II mixed; TPV fw 0.92×; CUDA vs CPU astropy | moc contains 0.99×; cone/union not claimed | see card | `775762c2` | [20260821_world_layer](20260821_world_layer.md) |
| 20260831 | map_overlap_background | - | - | - | Global overlap graph: 1.635 projected Mpix/s, 244 MB RSS delta, 83.9 MB exact scratch | compact built-ins; custom tiled estimators fail closed | `85010fda` + working tree | [20260831_map_overlap_background](20260831_map_overlap_background.md) |
