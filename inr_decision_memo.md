# INR Decision Memo

This memo turns the current knowledge base into a concrete implementation thesis for `torchsky`.

## Current blocker

The most important benchmark fact is now:

- on large sparse broad mosaics, the current field can already beat coadd strongly on the uniform target grid
- but it still loses on supported pixels

So the next problem is **supported-pixel fidelity under sparse heterogeneous observations**, not generic extrapolation and not PSF-awareness yet.

## Close-read synthesis

### `2506.09949` Sampling Theory for Super-Resolution with Implicit Neural Representations

- Method family: theory for when INR super-resolution is identifiable from sampling.
- Exact mechanism that matters: reconstruction quality depends on whether the sampling operator actually constrains the represented degrees of freedom.
- Transfers: do not let the spectral model represent detail finer than the bandpass operator can support.
- Does not transfer: abstract sampling theory does not tell us the best spatial backbone.
- Informs: spectral scaffold.

### `2307.06207` Local Conditional Neural Fields for Large-Scale Reconstructions

- Method family: local conditional neural fields for scalable reconstruction.
- Exact mechanism that matters: separate large-scale reconstruction into locally conditioned field evaluations without giving up global consistency.
- Transfers: sparse large mosaics likely want local support-sensitive refinement over a shared latent scaffold.
- Does not transfer: our spectral and astronomical forward model is different from generic computational imaging.
- Informs: support-refinement and systems/scaling strategy.

### `2602.15155` Refine Now, Query Fast

- Method family: decoupled representation refinement.
- Exact mechanism that matters: use a richer representation/refinement stage, then a cheap query path.
- Transfers: strengthens the case for explicit multiscale features plus lightweight decoding, not deeper pure-coordinate MLPs.
- Does not transfer: its exact surrogate-learning setup is not our mapmaking problem.
- Informs: spatial backbone.

### `2203.09517` TensoRF

- Method family: low-rank tensor factorization for neural fields.
- Exact mechanism that matters: factorized features can replace large MLPs while keeping continuous query.
- Transfers: explicit feature factors should remain the backbone bias rather than pure coordinate networks.
- Does not transfer: 4D radiance-field specifics.
- Informs: spatial backbone.

### `2503.21507` F-INR

- Method family: functional tensor decomposition for INRs.
- Exact mechanism that matters: stronger factorized continuous representations beyond early tensor/plane methods.
- Transfers: supports building one stronger structured family instead of many ad hoc branches.
- Does not transfer: task suite is broader than astronomy mapmaking.
- Informs: spatial backbone and continuous spectral design.

### `2302.01226` Factor Fields

- Method family: unified factorized field representation.
- Exact mechanism that matters: represent signals through factorized components rather than one monolithic field.
- Transfers: encourages a shared field plus small correction paths instead of pure monolithic continuous fields.
- Does not transfer: generic factorization alone does not solve sparse support fidelity.
- Informs: spatial backbone and hybrid decomposition.

### `2301.05187` WIRE

- Method family: wavelet implicit representation.
- Exact mechanism that matters: multiscale wavelet bias improves local detail and scale behavior.
- Transfers: the next backbone should be explicitly multiscale in space.
- Does not transfer: wavelet INRs alone do not answer the spectral operator question.
- Informs: spatial backbone.

### `2209.10684` Attention Beats Concatenation for Conditioning Neural Fields

- Method family: conditioning mechanism study for neural fields.
- Exact mechanism that matters: attention/modulation can preserve conditioning signal better than naive concatenation.
- Transfers: use FiLM/attention-style conditioning in bandpass residual or support-refinement paths.
- Does not transfer: does not tell us the correct spectral representation itself.
- Informs: support-refinement.

### `2502.17177` Joint multiband deconvolution for Euclid and Rubin images

- Method family: joint multiband deconvolution baseline.
- Exact mechanism that matters: correlated information across bands can improve reconstruction without fully continuous latent spectra.
- Transfers: strong baseline reference and evidence that multiband coupling is real even outside INR.
- Does not transfer: instrument-specific formulation and deconvolution framing.
- Informs: baseline design.

### `2510.16110` Linear Image Regridding and Coaddition with Oversampled Point Spread Functions

- Method family: careful linear analysis of regridding and coaddition.
- Exact mechanism that matters: explains why resampling/coaddition can remain extremely competitive if the operator is treated correctly.
- Transfers: strengthens the requirement that INR must beat strong linear baselines, not just weak stacks.
- Does not transfer: not a latent-field architecture.
- Informs: baseline design.

### `2511.09881` fiDrizzle-MU

- Method family: iterative drizzle-style reconstruction.
- Exact mechanism that matters: iterative classical methods can close part of the gap to learned models.
- Transfers: benchmark bar must be higher than bilinear coadd.
- Does not transfer: does not solve continuous spectrospatial modeling.
- Informs: baseline design.

### `2512.09115` SuperF

- Method family: multi-image super-resolution with implicit fields.
- Exact mechanism that matters: many-frame fusion into one latent continuous representation can work, but the architecture must protect observation fidelity.
- Transfers: directly relevant to the “many visits per band vs one field” use case.
- Does not transfer: generic image SR assumptions and task losses.
- Informs: support-refinement and baseline comparison.

## Decision

The next family should be:

`shared multiscale spatial field + continuous low-DOF spectral scaffold + small bandpass residual + operator-aware spectral regularization`

This should replace another round of:

- pure-coordinate MLP variants
- atlas routing tweaks
- generic optimizer churn

## Architecture matrix

| Paper | Spatial representation | Spectral representation | Support-fidelity strategy | Systems/scaling | Baseline implications |
| --- | --- | --- | --- | --- | --- |
| `2506.09949` | no | yes | no | no | no |
| `2307.06207` | yes | no | yes | yes | no |
| `2602.15155` | yes | no | indirect | indirect | no |
| `2203.09517` | yes | no | indirect | yes | no |
| `2503.21507` | yes | yes | indirect | yes | no |
| `2302.01226` | yes | yes | yes | indirect | no |
| `2301.05187` | yes | no | indirect | no | no |
| `2209.10684` | no | indirect | yes | no | no |
| `2502.17177` | no | yes | indirect | no | yes |
| `2510.16110` | no | no | no | no | yes |
| `2511.09881` | no | no | no | indirect | yes |
| `2512.09115` | yes | indirect | yes | indirect | yes |

## Concrete implementation thesis

1. Use an explicit multiscale spatial feature field as the shared trunk.
2. Predict a low-dimensional continuous spectral coefficient vector from the shared spatial latent.
3. Evaluate a fixed spectral basis at wavelength or integrated bandpass samples.
4. Penalize poorly observed spectral modes according to the actual bandpass operator.
5. Add a tiny bandpass-conditioned residual path with zero initialization.
6. Optimize sparse benchmark cases with patch-based band-balanced training and low-memory settings.
7. Evaluate against stronger iterative baselines, not only simple bilinear coadd.

**Status (code):** The multiscale spatial trunk for hash-grid fields is implemented in `torchsky.map.field` as `_HashGridEncoder` (dense coarse voxels + per-level hash sizes + optional smoothstep) with `HashGridResMLPField` for the residual refinement head; see `inr_knowledge_base.md` for the paper-to-knob mapping.
