# Torchsky Planning and Migration History

> Historical note: this file records early migration planning from `torchfits`.
> It is not the current public API contract. For the active API surface, start
> with [README.md](../../README.md), [docs/index.md](../../docs/index.md), and
> [api.md](../api.md).

This document captures the planning, design decisions, and implementation history from the legacy `torchfits` thread before work moved fully into `torchsky`.

It is intended as a durable handoff/history note inside the `torchsky` repository.

## Origin

- `torchfits` shipped a handoff release at `v0.3.1`.
- `torchsky` was then created as a new private repository seeded from the `torchfits` `v0.3.1` history.
- The goal was not to preserve `torchfits` compatibility. The goal was to create a stronger long-term base for a broader astronomy platform.

## Core Strategic Decisions

### 1. Clean break from `torchfits`

- `torchsky` is a clean-break package and repository.
- There is no compatibility obligation to old `torchfits` imports.
- Existing `torchfits` code should only be reused when it is already the best implementation.
- Imported git history is provenance, not a promise of API continuity.

### 2. Canonical public package shape

The canonical public namespaces are:

- `torchsky.io`
- `torchsky.coords`
- `torchsky.geom`

Supporting namespaces added in phase 1:

- `torchsky.map`
- `torchsky.encoders`
- `torchsky.catalogs`

Top-level convenience exports should remain thin and limited to high-frequency entry points.

### 3. `geom` naming and scope

- The geometry namespace is `geom`, not `geometry`.
- A low-level internal geometry/kernel layer is a conditional decision, not a guaranteed one.
- It should only be introduced if it clearly improves:
  - usability
  - functionality
  - performance
- If introduced, it should stay internal first and remain narrow.
- C++ is justified only where benchmarked hot paths remain after algorithmic cleanup.

### 4. Unified mapmaking product

- There should be one public mapmaking product family: `torchsky map`.
- It should not split the user-facing interface by radio/X-ray/optical/infrared/visible.
- Different observation operators can exist internally, but the public contract stays unified.

### 5. Phase-1 ML-native justification

Phase 1 should not be only a rename. It should include one capability that justifies the PyTorch/ML direction.

The chosen flagship direction is:

- image-plane-first
- multi-exposure
- multi-band
- WCS-aware
- differentiable
- optionally augmented by a coordinate-aware neural residual field

The latent field concept discussed for phase 1 was:

- `I(ra, dec, log_lambda)` or `I(ra, dec, log_nu)`

### 6. Coordinate-aware encoding direction

Phase-1 encoder direction agreed in-thread:

- sphere-aware harmonic sky encoding
- SIREN-style implicit field
- wavelength/frequency encoding on log scale

Seminal baselines retained for comparison:

- Fourier features
- SIREN

More recent approaches should be evaluated as optional backends or benchmarks, not as default architecture. The discussion explicitly called out avoiding overcommitting to old location-encoding literature while still keeping seminal baselines.

### 7. Catalog and crossmatching direction

- Catalog work is distinct from sensor-data mapmaking.
- `torchsky` should support SOTA catalog crossmatching.
- HATS/LSDB should be leveraged where they are already strong:
  - partitioned storage
  - scalable region filtering
  - coarse candidate generation
  - streaming/partition execution patterns
- `torchsky` should own tensor-native refinement on top:
  - precise angular scoring
  - astrometric refinement
  - proper-motion / epoch-aware matching
  - optional probabilistic or learned scoring later

### 8. Long-range platform ambition

The thread made explicit that `torchsky` is intended to become a much broader scientific platform over time, including eventual replacement-quality capabilities across:

- mapmaking and mosaicking
- drizzle-like reconstruction
- radio/CMB mapmaking
- spectral products
- astrometry
- photometry
- PSF modeling
- source extraction
- differential image analysis
- interferometric imaging
- radial velocity and broader spectroscopy

The strategic framing behind this was readiness for combined analysis across missions/surveys such as:

- SKA
- SPHEREx
- LSST
- Euclid
- Roman
- eROSITA
- future CMB experiments

## Phase Plan Agreed in Thread

### Phase 0: final `torchfits` handoff

- Ship final `torchfits` release.
- Communicate that the next major step is `torchsky`.
- Avoid doing partial `torchsky` architecture under the `torchfits` name.

### Phase 1: `torchsky` bootstrap

- Create private `torchsky` repository from `torchfits` `v0.3.1`.
- Rename package/repo/build metadata to `torchsky`.
- Establish `io`, `coords`, and `geom` as primary namespaces.
- Add the first `torchsky map` baseline.
- Add `encoders` and `catalogs` scaffolding to support the phase-1 direction.

### Later phases

- Expand mapmaking operators without splitting the public `torchsky map` product line.
- Expand catalog infrastructure and crossmatch sophistication.
- Revisit internal low-level geom or native kernels only where profiling justifies it.

## Repository Transition That Was Implemented

The transition plan that was actually carried out:

- A new private GitHub repo `sfabbro/torchsky` was created.
- It was seeded from the `torchfits` history and anchored from `v0.3.1`.
- `origin` in the private clone points to the private `torchsky` repo.
- The old public `torchfits` repo remains only as a reference `upstream`.
- Development was initially bootstrapped from a temporary branch and then fast-forwarded into `torchsky` `main`.
- The final decision was to stop using the bootstrap branch and continue directly on `torchsky` `main`.

## Implementation History Carried Out During the Thread

### Bootstrap and rename

Initial bootstrap work created:

- package rename from `torchfits` to `torchsky`
- canonical namespace packages:
  - `src/torchsky/io`
  - `src/torchsky/coords`
  - `src/torchsky/geom`
- phase-1 scaffolding:
  - `src/torchsky/map`
  - `src/torchsky/encoders`
  - `src/torchsky/catalogs`
  - CLI entrypoint for `torchsky map`

### Early phase-1 baseline

Implemented in the bootstrap:

- image-plane coadd baseline for `torchsky map`
- harmonic sky encoding / spectral encoding / SIREN-style implicit field scaffolding
- catalog stream dataset and chunked angular crossmatch baseline

### Hardening pass 1

Main work:

- tightened phase-1 contracts
- fixed `torchsky.map` namespace exposure
- removed stale `fits` namespace exposure from the root package
- hardened `coadd_image_plane`:
  - proper target-shape handling from target headers
  - non-finite pixel handling
  - consistent 2D validation
  - weight validation
- hardened catalog streaming:
  - worker-safe iterable splitting
  - partition factory support
- added regression coverage for the new phase-1 surface

### Hardening pass 2: `geom`

Main work:

- profiled `geom`/region infrastructure before attempting native code
- found that `PixelizedRegion` set operations were a good Python-level optimization target
- replaced `torch.isin`-based region `intersection()` / `difference()` with sorted `searchsorted`-based logic
- preserved sorted/unique contracts with direct tests

Measured result discussed in-thread:

- large-region `PixelizedRegion` intersection and difference improved by about 2x to 3x end-to-end

### Hardening pass 3: `map`

Main work:

- audited the image-plane reprojection path
- identified repeated construction of the same target world grid as wasted work
- introduced prepared target-grid reuse within each coadd run
- added a regression test that explicitly checks the target grid is built only once per run

Measured result discussed in-thread:

- on a shifted-WCS `6 x 512 x 512` synthetic coadd, runtime dropped from about `0.65s` to about `0.42s`
- roughly a 36% reduction on the path that actually uses reprojection

### Hardening pass 4: `catalogs`

Main work:

- audited `crossmatch_sky`
- identified the brute-force evaluation of all right chunks as the dominant waste
- added declination-sorted candidate pruning before the expensive angular-distance refinement
- added a wraparound regression at the RA `0/360` boundary

Measured result discussed in-thread:

- on a synthetic `1000 x 10000` benchmark, runtime dropped from about `7.09s` to about `0.95s`
- roughly a 7.5x reduction with the same match count

## Commits Produced During the Thread

These commits were explicitly created and pushed onto `torchsky` `main` during the work:

- `39461ba` bootstrap: rename package to `torchsky` and add phase-1 scaffolding
- `db86f2e` hardening: improve catalog streaming and map baseline
- `8dbc21d` hardening: tighten phase1 map and catalog contracts
- `4af967d` geom: speed up pixel region set operations
- `3831407` map: reuse prepared target grids across reprojections
- `c822740` catalogs: prune crossmatch candidates by declination

## What Was Explicitly Deferred

The thread also made several things explicit by deferring them:

- no `torchfits` compatibility layer
- no commitment yet to a generalized public low-level geom API
- no premature move to C++ where Python-level algorithmic fixes still deliver strong wins
- no modality-specific public mapmaking split
- no immediate implementation of the full long-range replacement roadmap
- no commitment yet to a full HDF5 strategy inside phase 1

## Current Working Agreement

Current state agreed at the end of the thread:

- `torchsky` is the canonical repo
- `torchsky` `main` is the active development branch
- work should continue directly on `main` for now
- the old `torchfits` thread is treated as historical planning context
- this file exists to preserve that planning context inside the `torchsky` repo

## Recommended Next Technical Focus

The next likely areas to attack after this document was requested were:

- deeper spherical polygon/intersection kernel work in `geom`
- stronger large-scale catalog crossmatch infrastructure beyond declination pruning
- eventual native kernels only where profiling still justifies them after algorithmic cleanup

