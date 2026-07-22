# Phase-space consumer audit (2026-07-21)

status: proposal only; no production consumer was changed

## Decision

Build the Torchsky 6D primitive as an independently tested lower-level API
first.  Do **not** replace either downstream 4D call site on its arrival:
neither consumer currently supplies a complete 6D state, and silently treating
missing distance/RV as zero would make an apparently more accurate path no more
scientific than the present angular propagation.

The smallest scientifically meaningful downstream integration is Xmatch's
existing Gaia catalogue route, after *additive* metadata fields map the already
available Gaia columns.  It keeps the public matching request and result schema
unchanged: `_apply_proper_motion()` remains the sole coordinate-rewriting seam,
and `sky_match()` still sees only propagated RA/Dec plus its current 2x2
positional covariance.

Required additive `CatalogueSource`/YAML metadata:

- `parallax_column` (mas)
- `radial_velocity_column` (km/s)

For the bundled Gaia routes, request `radial_velocity` along with the currently
requested `parallax`; use the 6D primitive only for rows where both are finite
and parallax is positive.  Send every other row through the current 4D
`propagate_proper_motion` path.  This is an explicit data-availability split,
not a distance/RV imputation heuristic.  Position-error propagation is a
separate later change: today the matcher intentionally consumes only a 2x2
RA/Dec covariance.

## Xmatch inventory

### Existing state and call seam

- `src/xmatch/sources.py:21-40`: `CatalogueSource` has RA/Dec, a catalogue or
  per-row Julian epoch, `pm_ra_column`, `pm_dec_column`, and 2D position-error
  metadata.  It has no parallax, distance, radial-velocity, PM-error, or
  astrometric-covariance fields.
- `src/xmatch/matchers.py:431-496`: `_apply_proper_motion()` extracts those
  arrays and calls the stable wrapper in `astro_utils`; this is the only
  production coordinate-propagation seam.
- `src/xmatch/astro_utils.py:120-166`: wrapper sanitizes NaN PM to zero and a
  NaN source epoch to the target epoch, then prefers Torchsky.  Its public
  contract is deliberately 4D angular only and must remain so for existing
  callers.
- `src/xmatch/matchers.py:278-337`: `skyellipse` carries a tangent-plane 2x2
  RA/Dec covariance from `ra_err_column`, `dec_err_column`, and optional
  correlation.  PM uncertainty is only the independent heuristic drift prior
  in `matchers.py:508-604`; no `pmra_error`, `pmdec_error`, parallax error, or
  cross-term is represented.
- `src/xmatch/out_of_core.py:260-267`: bounded-memory matching presently
  rejects target-epoch/PM paths, so the first integration is in-memory only.

### Real catalogue/configuration availability

- `src/xmatch/xmatch.yaml:104-123` (`gaia_cds`) already requests `Plx` with
  PM and `Epoch`, but no radial velocity or uncertainty/correlation columns.
- `src/xmatch/xmatch.yaml:280-297` (`gaia_noao`) and `:341-357` (`gaia_esa`)
  already request `parallax`, `pmra`, `pmdec`, and `ref_epoch`, but no
  `radial_velocity`.
- The YAML has RA/Dec correlation (`RADECor`/`ra_dec_corr`) only.  It cannot
  express Gaia's full five-parameter covariance, so covariance propagation is
  not a no-churn follow-up.
- `tests/test_real_catalogues.py:248-272` builds a cached Gaia source with
  position errors; it currently omits epoch, PM, parallax, and RV mappings.

### Exact existing test leverage

- `tests/test_astro_utils.py:90-121`: backend preference and current NaN PM /
  epoch normalization contract.  Keep unchanged for the 4D branch.
- `tests/test_astro_utils.py:123-146`: Astropy high-declination/pole reference
  fixture; a 6D wrapper test can copy it and add parallax/RV inputs.
- `tests/test_matchers.py:236-428`: `pm_frames` plus both-side, one-side,
  catalogue-epoch, missing-epoch, NaN-PM, and end-to-end match fixtures.  Add
  one finite-6D Gaia-like row and one missing-RV row to prove the two branches
  preserve matching semantics.
- `tests/test_real_catalogues.py:248-272`: optional cached real Gaia fixture;
  use only after the fetch/query fixture includes radial velocity.  It is not a
  deterministic CI prerequisite today.

## Cosmodist inventory

- `src/cosmodist/data/labels.py:439-522`: `StarRegistry` retains RA/Dec,
  magnitude, PM components, and one registry-level MJD reference epoch.  It
  calls Torchsky's current angular primitive in `positions_deg()`; no parallax,
  distance, RV, astrometric errors, or covariance is retained.
- `src/cosmodist/data/labels.py:382-437`: `GaiaParallaxRegistry` separately
  retains parallax only long enough to form a distance-modulus label and its
  scalar uncertainty.  It is not connected to `StarRegistry` and has no PM,
  epoch, or RV.
- `src/cosmodist/data/scene_patch_dataset.py:681-765` constructs those two
  registries independently from config.  `config_scene.yaml:63-68` likewise
  defines separate Gaia-parallax and stars inputs.  Combining them would add a
  new label-data contract and is larger than the Xmatch metadata addition.
- No Cosmodist production call site currently needs radial velocity.  Its
  immediate correct consumer remains 4D PM propagation for image-epoch star
  anchors.

### Exact existing test leverage

- `tests/data/test_labels.py:37-112`: 10-year PM displacement, pole parity
  against Astropy, single-PM-component behavior, and epoch-aware query fixture.
  It is the right regression suite for retaining the 4D path.
- `tests/data/test_labels.py:214-233`: tiny CSV fixture for the separate Gaia
  parallax registry.  It verifies only distance-modulus conversion and SNR/
  minimum-parallax filtering; it cannot validate phase-space propagation.

## Torchsky primitive and test boundary

The present `torchsky.wcs.propagate_proper_motion` explicitly documents its
unit-distance angular contract in `src/torchsky/wcs/frames.py:299-357`.
Preserve it unchanged.  Add a separately named 6D primitive accepting ICRS
RA/Dec, `pm_ra_cosdec`, `pm_dec`, parallax (mas), radial velocity (km/s), and
source/target Julian years; its result may return at least RA/Dec and should
define missing/invalid-distance behavior explicitly rather than inheriting
Xmatch's NaN normalization.

The immediate Torchsky fixture base is
`tests/test_wcs_frames.py:40-90`: it already has vectorized epochs, extreme
declinations, Astropy `SkyCoord.apply_space_motion` parity, broadcasting, and
autograd coverage.  Extend it with a mixed nearby/high-PM and distant/low-PM
6D sample, scalar and vector epochs, long-baseline perspective acceleration,
and a finite-difference/autograd check.  Build covariance/Jacobian tests only
when a covariance-returning API is selected; neither downstream schema can
consume that output yet.

## Deferred work (intentionally not bundled)

1. Full 5x5/6x6 covariance mapping and Jacobian propagation in Xmatch.
   This requires explicit units and Gaia correlation-column mappings, changes
   `CatalogueSource`, YAML, CLI doctor classification, remote column planning,
   and the matcher covariance representation.
2. Combining Cosmodist `StarRegistry` and `GaiaParallaxRegistry`.  That changes
   supervision/calibration data ownership and is not justified solely by a
   coordinate primitive.
3. Enabling target-epoch propagation in Xmatch's out-of-core engine.  It is
   currently rejected by design and needs bounded-memory semantics first.
