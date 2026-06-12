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
