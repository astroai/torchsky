# Planned work

The only tracked planning document in the package. Implemented behavior lives in
`docs/`; experimental research, reviews, and archives are untracked on purpose.
Scientific campaigns, comparator evidence, and acceptance status live in
[`torchsky-harness`](https://github.com/astroai/torchsky-harness).

## Durable engineering priorities

1. Fix correctness and boundary defects in shared geometry, WCS, observation
   operators, and data interchange before adding features.
2. Keep algorithms viable at survey scale through indexing, tiling, chunking,
   streaming, and sparse execution before considering native kernels.
3. Preserve tensor-native execution, differentiability, and claimed
   CPU/CUDA/MPS behavior with bounded tests.
4. Keep public APIs narrow; every exported surface needs a documented use and a
   real consumer.
5. Support performance claims only with reproducible harness measurements.

## Release 0.4.0 queue

- **Documentation audit:** finish the directory-by-directory review of
  `src/torchsky` (map, sphere, wcs, catalogs, encoders, coords, transforms,
  cli, viz) and the harness; remove dead code, duplicate helpers, and stale
  scripts.
- **Visualization (`torchsky.viz`):** in-depth review for 0.4.0. Audit the
  FastAPI server, routers, static frontend, cache, and renderer paths for
  obsolete endpoints, unused assets, and untested branches; decide whether it
  ships as an optional extra or is split out.
- **Experimental field surfaces:** clarify or rename `native_physical`,
  `joint_native`, `neural_inverse`, and the lensing/magnification operators so
  the public names state what they do; drop any that lack a demonstrated
  consumer or acceptance role for 0.4.0.
- **Harness evidence refresh:** rebuild the wheel and rerun the declared gates
  and GPU validation whenever the subject changes; regenerate the evidence
  bundle and baseline subject identity before tagging.
- **Docs accuracy:** refresh stale gate numbers and counts in the harness
  release docs; keep the package docs site, examples, and link gate green.
- **Release mechanics:** reconcile `main` with the release branch, configure
  PyPI trusted publishers and GitHub Pages, tag `v0.4.0` / `v0.1.0`, then run
  post-release verification from PyPI.
- **Publishing workflows overlap:** `.github/workflows/build_wheels.yml` and
  `.github/workflows/publish.yml` both trigger on `v*` tags and both upload to
  PyPI. Pick one canonical multi-platform trusted-publishing workflow (the
  release checklist and README already point at `build_wheels.yml`; the newer
  `publish.yml` adds tag/version checks), merge the useful verification steps,
  and delete the other before tagging.

## Calibrated-field work (post-0.4.0 unless promoted)

- Adaptive/controlled chromatic response, default training/renderer
  integration, and physical radiometric/flat calibration beyond the opt-in
  native source-node route.
- Photon/charge/read additive-stage renderer integration, normalized native
  likelihoods, and nonlinear/history detector laws.
- End-to-end observation lineage and joint nuisance propagation across all
  field/coadd callers.
- Query/UQ reports and calibration lifecycle across all solvers.
- Native-data science gates: confirmed stellar references, independent
  calibration/audit sources, PSF/background/selection validation, and blind
  cross-instrument prediction. Current HST/ACS, JWST/NIRCam, and CFHT/MegaCam
  products remain `training_ready=false`.
- Broader CUDA/MPS device evidence; skipped device tests are not validation.

## Engineering audit backlog

- Renderer/coadd/solver nuisance and lifecycle propagation.
- Legacy band-grid MPS footprint-bound issue.
- Conservative NLL derivative-order/device contract before widening its range.
- Keep observed curvature distinct from expected Fisher information and
  calibration uncertainty distinct from fixed product scaling.
