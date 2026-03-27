# INR Status

Updated: 2026-03-27

## Implemented formulations

### Coordinate / classic INRs
- `sine_implicit`
- `wire_implicit`
- `gaussian_basis`
- `fourier_mlp`

### Spectral / factorized field variants
- `spectral_basis_field`
- `learned_spectral_basis_field`
- `conditioned_spectral_field`
- `bandpass_conditioned_field`
- `sky_encoder_bandpass_field`

### Learned sky encoders
- `sky_encoder_mlp`
- `sky_encoder_resmlp`

### Hash / grid variants
- `hashgrid_conditioned_field`
- `hashgrid_mlp`
- `hashgrid_spectral_field`

### Structured sky-lambda models
- `sky_lambda_planes_field`
- `sky_lambda_kplanes_field`
- `sky_lambda_feature_field`
- `operator_constrained_feature_field`
- `sky_lambda_atlas_field`

## Implemented training / rendering features
- transmitted rendering with exact known bandpasses:
  - `I(ra,dec) = integral T(lambda) f(ra,dec,lambda) d lambda`
- coordinate modes:
  - `local_tangent`
  - `sphere_xyz`
- training spectral sampling:
  - deterministic quadrature
  - `bandpass_mc`
  - `mixture_mc`
- fit losses:
  - weighted `mse`
  - `charbonnier`
  - `huber`
- reduction modes:
  - `global`
  - `per_band`
  - `per_observation`
- support-aware field-fit metrics and recommendation logic

## Current benchmark-backed wins

### Exact-band same-field TAN
Source: `benchmark_results/map_field_fit_exact_support.json`

- `spectral_basis_field`
  - full RMSE: `0.02086`
  - support RMSE: `0.02070`
- `coadd_bilinear`
  - full/support RMSE: `0.04105`

Conclusion:
- exact-band TAN is solved in favor of INR
- current winning formulation: `spectral_basis_field`

### Exact-band same-field TPV
Source: `benchmark_results/map_field_fit_exact_support.json`

- `sky_encoder_mlp`
  - full RMSE: `0.02110`
  - support RMSE: `0.01730`
- `coadd_bilinear`
  - full/support RMSE: `0.03934`

Conclusion:
- exact-band TPV is solved in favor of INR
- current winning formulation: `sky_encoder_mlp`

### Broad-band same-field local deep-field
Source: `benchmark_results/map_field_fit_broad_support.json`

- `tan_sip_field_fit_deep_multiband_broad`
  - `sky_encoder_bandpass_field`: full/support RMSE `0.03650 / 0.03641`
  - `coadd_bilinear`: `0.04758 / 0.04758`
- `tpv_field_fit_deep_multiband_broad`
  - `sky_encoder_bandpass_field`: full/support RMSE `0.03748 / 0.02513`
  - `coadd_bilinear`: `0.04009 / 0.04009`

Conclusion:
- same-field local broad-band rows are benchmark-backed INR wins
- current saved winning formulation: `sky_encoder_bandpass_field`

## Current frontier

### Staged stronger non-INR baseline
Source:
- `/tmp/torchsky_explicit_grid_exact32.json`
- `/tmp/torchsky_explicit_grid_broad32.json`

- `tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit @ 32, 8 steps`
  - `coadd_bilinear`: `0.03827 / 0.03827`
  - `explicit_spectral_grid_field`: `0.02435 / 0.02435`
- `tpv_field_fit_sparse_large_mosaic_broad_denseoverlap @ 32, 8 steps`
  - `coadd_bilinear`: `0.05574 / 0.05574`
  - `explicit_spectral_grid_field`: `1.02647 / 1.00658`

Conclusion:
- a coadd-initialized explicit spectral grid is now staged as a stronger non-INR multiframe refinement baseline
- it is genuinely stronger than bilinear coadd on the sparse exact 3-band gate
- it is dramatically worse on the sparse broad-overlap gate
- this reinforces that the broad sparse regime is not just a weak-baseline problem

### Sparse TPV exact mosaic, smaller scale
Source: recent focused probe at `size=64`, `steps=8`

- `coadd_bilinear`: full/support RMSE `0.05917 / 0.05917`
- `sky_lambda_feature_field`: `0.07640 / 0.03606`

Conclusion:
- structured feature-field is the first clear sparse-mosaic INR win
- current winning formulation: `sky_lambda_feature_field`

### Sparse TPV exact mosaic, larger scale
Source: `/tmp/torchsky_sparse_tpv128_mix.json`

- `coadd_bilinear`
  - full/support RMSE: `0.06104 / 0.06104`
- `sky_lambda_feature_field`, `16` steps, mixed `band_global_mix` reduction
  - `0.13904 / 0.06219`

Conclusion:
- `sky_lambda_feature_field` is the best current structured branch on this gate
- the mixed `band_global_mix` reduction is a real but modest improvement over the earlier pure `per_band` default
- it gets very close to coadd on support-aware RMSE at `16` steps
- it still loses on support and still loses badly on full-grid RMSE

### Focused sparse-mosaic frontier profile
Source: `benchmark_results/map_field_fit_sparse_frontier_profile.json`

- `tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit`
  - `sky_encoder_resmlp @ 32 steps`: full/support RMSE `0.02844 / 0.02844`
  - `coadd_bilinear @ 32 steps`: `0.03827 / 0.03827`
- `tpv_field_fit_sparse_large_mosaic_broad_denseoverlap`
  - `sky_lambda_feature_field @ 16 steps`: full/support RMSE `0.03375 / 0.02428`
  - `coadd_bilinear @ 16 steps`: `0.05574 / 0.05574`

Conclusion:
- case-specific step budgets make the current sparse frontier much more honest
- the best sparse exact 3-band branch at this smaller profile size is now `sky_encoder_resmlp`
- the best sparse broad-overlap branch remains `sky_lambda_feature_field`

### Unified field regime snapshot
Source: `benchmark_results/map_field_fit_unified_regimes.md`

- local exact TPV
  - `coadd_bilinear`: `0.03934 / 0.03934`
  - `operator_constrained_feature_field`: `0.31319 / 0.26526`
- local broad TPV
  - `coadd_bilinear`: `0.04009 / 0.04009`
  - `operator_constrained_feature_field`: `0.22327 / 0.17372`
- sparse exact 3-band TPV
  - `coadd_bilinear`: `0.03827 / 0.03827`
  - `operator_constrained_feature_field`: `0.03319 / 0.03319`
- sparse broad dense-overlap TPV
  - `coadd_bilinear`: `0.05574 / 0.05574`
  - `operator_constrained_feature_field`: `0.11820 / 0.11677`

Conclusion:
- the unified family is no longer purely aspirational: it now beats `coadd_bilinear` on the sparse exact 3-band frontier
- but it is still clearly not a one-family solution yet
- it remains weak on both local regimes and on the sparse broad frontier

### Sparse TAN-SIP broad mosaic, larger scale
Sources:
- earlier sparse frontier probes
- current `sky_lambda_feature_field` `128` broad run exited with code `137`

Known reference points from earlier probes:
- `coadd_bilinear @ 128`: support RMSE about `0.05267`
- `sky_lambda_kplanes_field @ 128, 24 steps`: support RMSE about `0.11350`
- `sky_encoder_mlp @ 128, 24 steps`: support RMSE about `0.11940`

Current conclusion:
- large sparse broad-band mosaics remain unsolved
- current structured feature-field is not yet practical or stable enough there

### Sparse TPV broad mosaic, larger scale
Source: `/tmp/torchsky_broad128_uniform.json`

- `coadd_bilinear`
  - native/support RMSE: `0.06658 / 0.06658`
  - uniform-grid RMSE: `0.78836`
  - single-overlap support RMSE: `0.07775`
  - multi-overlap support RMSE: `0.04788`
  - fringe/far extrapolated RMSE: `1.07008 / 0.98735`
- `sky_lambda_feature_field`
  - native/support RMSE: `0.12177 / 0.12423`
  - uniform-grid RMSE: `0.12177`
  - single-overlap support RMSE: `0.11496`
  - multi-overlap support RMSE: `0.13557`
  - fringe/far extrapolated RMSE: `0.11390 / 0.12122`

Conclusion:
- on the large sparse TPV broad row, the coadd still wins on observed support
- but the structured feature field is already dramatically better on the full target grid once both methods are scored uniformly
- the remaining gap is now clearly support-region fidelity, not extrapolation

## Best current formulation by regime
- exact-band TAN, same field: `spectral_basis_field`
- exact-band TPV, same field: `sky_encoder_mlp`
- broad-band local deep-field: `sky_encoder_bandpass_field`
- sparse TPV exact mosaic: `sky_lambda_feature_field`
- sparse TPV exact mosaic, 3 weakly coupled bands with 15 visits each: `sky_encoder_resmlp` on the focused sparse profile, `sky_lambda_feature_field` on the harder larger-size frontier
- sparse TPV exact mosaic, strongest current non-INR baseline: `explicit_spectral_grid_field`
- sparse broad larger mosaic: `sky_lambda_feature_field` on the focused sparse profile
- sparse broad larger mosaic, uniform full-grid metric: `sky_lambda_feature_field`
- unified operator-aware follow-up family: `operator_constrained_feature_field` now beats `coadd_bilinear` on the sparse exact 3-band gate, but still loses badly on local exact, local broad, and sparse broad gates

## Where we are still lacking
- full-grid extrapolation on sparse larger mosaics
- broad-band sparse larger mosaics
- practical scaling at `128+` sparse mosaic sizes
- a single INR family that wins across all regimes
- PSF-aware observation modeling
- calibration-aware latent fitting
- redshift / radial-velocity aware spectral modeling

## Working interpretation
- We no longer have one INR frontier.
- We have a split frontier:
  - local same-field products are already good with simpler learned encoders and spectral-basis models
  - sparse larger mosaics want structured sky-lambda feature fields
- The most promising next branch for the deepest-sky direction is still `sky_lambda_feature_field`, but only on the sparse-mosaic path and only after we improve full-grid behavior and scaling.
- The unified operator-aware family now looks like a plausible sparse-exact scaffold, but not yet a replacement for the current best regime-specific winners.
- The new 3-band multivisit sparse TPV limiting case is the first sign that the
  structured feature field can exploit "many visits per band, few bands overall"
  once the bandpass latent is given enough capacity.
  On the current `32x32`, `8`-step smoke:
  - `coadd_bilinear`: `RMSE 0.03827`
  - `sky_lambda_feature_field`: `RMSE 0.03364`
- On the harder `128x128` version of that same case, the plain full-frame fit is
  still worse than the coadd, but a band-balanced patch/minibatch training
  contract is the first thing that improves the feature field itself:
  - full-frame, `8` steps: `RMSE 0.15618`, support `0.08777`
  - patch/minibatch, `32` steps: `RMSE 0.15198`, support `0.08616`
- A tiled `sky_lambda_feature_atlas_field` is now implemented as an experimental
  locality branch, but on the harder `128x128` three-band frontier it is not
  yet competitive with the simpler global feature field.
- The new chunked sparse-training contract also helps the sparse TPV broad
  overlap frontier on support-aware accuracy, but not enough to beat the coadd:
  on the current `32x32`, `8`-step smoke it moves `sky_lambda_feature_field`
  from roughly `0.086` support RMSE into the `0.075` range while the coadd
  remains at `0.0557`.
- With a larger step budget on that same chunked broad sparse TPV row, the
  feature field does cross the line: at `32x32`, `16` steps it reaches
  `RMSE 0.03375`, support `0.02428`, ahead of the coadd's `0.05574`.
- On the harder `128x128` sparse TPV broad row, the new uniform-grid and
  region-split diagnostics show that the current feature field is already the
  better extrapolator by a wide margin, while the coadd remains stronger on the
  actually observed support. That means the current scientific blocker is no
  longer “can the field extrapolate?” but “can we sharpen supported-pixel
  fidelity without sacrificing that extrapolation advantage?”
