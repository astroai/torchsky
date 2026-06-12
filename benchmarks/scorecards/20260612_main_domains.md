# Benchmark scorecard: main_domains

- Archived: `2026-06-12T21:12:59Z`
- Profile: `user`
- Platform: `macOS-14.8.2-arm64-arm-64bit-Mach-O`
- Git SHA: `193e62eec87590fc9dbc508a9d0d5c3c3b02fb8d`

## Summary

| Domain | Run ID | Families first | Deficits |
|---|---|---:|---:|
| fits | `api_audit_fits_full` | 249/249 | 0 |
| fitstable | `scorecard_fitstable` | 140/140 | 0 |
| sphere | `scorecard_sphere` | 32/32 | 0 |
| wcs | `scorecard_wcs` | 126/150 | 24 |

## By family

### fits (`api_audit_fits_full`)

- `smart`: 84/84 (100.0%)
- `specialized`: 165/165 (100.0%)
- Deficits: none

### fitstable (`scorecard_fitstable`)

- `smart`: 70/70 (100.0%); large-N 30/30
- `specialized`: 70/70 (100.0%); large-N 30/30
- Deficits: none

### sphere (`scorecard_sphere`)

- `geometry`: 2/2 (100.0%); large-N 2/2
- `specialized`: 30/30 (100.0%); large-N 24/24
- Deficits: none

### wcs (`scorecard_wcs`)

- `smart`: 126/150 (84.0%); large-N 90/90
- Deficits: 24 rows

## Notes

Post API-audit four-domain sweep; sphere bench path fixes applied same session.

Local artifacts (gitignored): `benchmarks_results/api_audit_fits_full/`, `benchmarks_results/scorecard_fitstable/`, `benchmarks_results/scorecard_wcs/`, `benchmarks_results/scorecard_sphere/`
