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
| 20260816 | world_layer | - | - | pending VOS | Mac Ishioka map2alm leads; alm2map ~1.2–1.4× | 2 | local main | [20260816_world_layer](20260816_world_layer.md) |
