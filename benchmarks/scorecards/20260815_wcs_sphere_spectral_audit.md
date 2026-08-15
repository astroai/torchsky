# Benchmark scorecard: wcs_sphere_spectral_audit

- Archived: `2026-08-15T14:03:25Z`
- Profile: `user`
- Platform: `macOS-14.8.2-arm64-arm-64bit-Mach-O`
- Git SHA: `5f911d00c30cb0aa6a8a648677c7cfde3654f16d`

## Summary

| Domain | Run ID | Families first | Deficits |
|---|---|---:|---:|
| sphere | `acceptance_sphere` | 2/3 | 1 |
| wcs | `acceptance_wcs` | 3/6 | 3 |

## By family

### sphere (`acceptance_sphere`)

- `specialized`: 2/3 (66.7%)
- Deficits: 1 rows

### wcs (`acceptance_wcs`)

- `smart`: 3/6 (50.0%)
- Deficits: 3 rows

## Notes

WCS/sphere/spectral audit. SFL stacked projector, extra-axis CD inverse, LONPOLE=180 fused/Python, SpectralAxis via wcs.spectral. mocpy 0.20.0 and healsparse 1.12.2 oracle rows in sparse/core benches. Fused default kept ON (TAN/SFL/HPX faster at N=1e5). Spin map2alm_spin still ~6x vs healpy. Sparse get_values still behind healsparse after removing host sync.

Local artifacts (gitignored): `benchmarks_results/acceptance_wcs/`, `benchmarks_results/acceptance_sphere/`
