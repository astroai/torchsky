# Benchmark scorecard history

Committed snapshots of multi-domain throughput scorecards. Use these to track
progress and regressions across releases without committing full
`benchmarks_results/` trees (those stay local and gitignored).

## Workflow

1. Run the four main domains (or `pixi run bench-all`).
2. Note the `run_id` folders under `benchmarks_results/`.
3. Archive a snapshot:

```bash
pixi run bench-archive-scorecard -- \
  --label main_domains \
  --runs fits=api_audit_fits_full,fitstable=scorecard_fitstable,wcs=scorecard_wcs,sphere=scorecard_sphere \
  --notes "Optional context for this snapshot"
```

4. Commit `benchmarks/scorecards/<date>_<label>.{json,md}` and the updated
   `index.md`.

## Files

| File | Purpose |
|---|---|
| `index.md` | Chronological table of archived scorecards |
| `YYYYMMDD_<label>.json` | Machine-readable snapshot (diff-friendly) |
| `YYYYMMDD_<label>.md` | Human-readable summary |

Older narrative optimization notes remain in
[`docs/archive/benchmark-history.md`](../../docs/archive/benchmark-history.md).
Release-style prose snapshots remain in [`docs/benchmarks.md`](../../docs/benchmarks.md).

**Open investigation:** WCS small-N zenithal `pixel_to_world(x,y)` vs Astropy —
attempt log, baseline, and future directions in
[`docs/projects/wcs-small-n-investigation.md`](../../docs/projects/wcs-small-n-investigation.md).
