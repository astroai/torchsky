# Unified Field Regime Snapshot

Updated: 2026-03-27

This note records a focused four-regime spot check for the current
`operator_constrained_feature_field` after adding:
- a width-gated support-refinement path
- a coarse/fine spatial split for that refinement path
- an exact-case sparse benchmark contract that disables extra low-resolution
  context supervision on the fully supported 3-band gate

## Local exact-band TPV

- case: `tpv_field_fit_deep_multiband_exact`
- source run: `/tmp/torchsky_unified_local_exact_v3.json`

| model | full RMSE | support RMSE | uniform RMSE |
| --- | ---: | ---: | ---: |
| `coadd_bilinear` | `0.03934` | `0.03934` | `0.76263` |
| `sky_encoder_mlp` | `0.14483` | `0.13497` | `0.14483` |
| `operator_constrained_feature_field` | `0.31319` | `0.26526` | `0.31319` |

Conclusion:
- the unified field is not competitive on the local exact TPV gate yet

## Local broad-band TPV

- case: `tpv_field_fit_deep_multiband_broad`
- source run: `/tmp/torchsky_unified_local_broad_v3.json`

| model | full RMSE | support RMSE | uniform RMSE |
| --- | ---: | ---: | ---: |
| `coadd_bilinear` | `0.04009` | `0.04009` | `0.79481` |
| `sky_encoder_resmlp` | `0.12374` | `0.07799` | `0.12374` |
| `operator_constrained_feature_field` | `0.22327` | `0.17372` | `0.22327` |

Conclusion:
- the unified field is also not competitive on the local broad TPV gate yet

## Sparse exact 3-band TPV

- case: `tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit`
- source run: `/tmp/torchsky_opcon_exact32_v7.json`
- profile: `sparse_mosaic_frontier`

| model | full RMSE | support RMSE | uniform RMSE |
| --- | ---: | ---: | ---: |
| `coadd_bilinear` | `0.03827` | `0.03827` | `0.03827` |
| `sky_lambda_feature_field` | `0.03589` | `0.03589` | `0.03589` |
| `sky_encoder_resmlp` | `0.02844` | `0.02844` | `0.02844` |
| `operator_constrained_feature_field` | `0.03319` | `0.03319` | `0.03319` |
| `explicit_spectral_grid_field` | `0.02435` | `0.02435` | `0.02435` |

Conclusion:
- the unified field now beats `coadd_bilinear` on the real sparse exact 3-band gate
- it is still behind the current best INR and the stronger explicit-grid baseline

## Sparse broad dense-overlap TPV

- case: `tpv_field_fit_sparse_large_mosaic_broad_denseoverlap`
- source run: `/tmp/torchsky_opcon_broad32_v6.json`
- profile: `sparse_mosaic_frontier`

| model | full RMSE | support RMSE | uniform RMSE |
| --- | ---: | ---: | ---: |
| `coadd_bilinear` | `0.05574` | `0.05574` | `0.60507` |
| `sky_lambda_feature_field` | `0.03375` | `0.02428` | `0.03375` |
| `operator_constrained_feature_field` | `0.11820` | `0.11677` | `0.11820` |

Conclusion:
- the unified field is still clearly behind on the sparse broad frontier
- `sky_lambda_feature_field` remains the current sparse broad winner

## Readout

- The coarse/fine support-refinement step improved the local exact, local broad, and sparse broad rows relative to the earlier width-gated-only checkpoint.
- It gave back a little on sparse exact, though that gate still remains ahead of `coadd_bilinear`.
- The coarse/fine support-refinement path materially improved the sparse broad frontier while keeping the sparse exact gate ahead of coadd.
- The current regime split is now clearer:
  - local exact and local broad: unified family loses badly
  - sparse exact: unified family is now viable
  - sparse broad: unified family still needs a larger architectural step
