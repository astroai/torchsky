# INR Knowledge Base

Shortlist of papers worth keeping in active working memory for `torchsky` mapmaking.

The goal is not to collect every INR paper. It is to keep the few papers that most directly inform:

- latent mapmaking from heterogeneous observations
- sparse large mosaics
- multiband or spectrospatial coupling
- low-memory scalable fitting
- strong classical baselines we still need to beat

## Previously shortlisted

### Core INR and representation papers

- Zip-NeRF. Anti-aliased multiscale representation for scale-aware neural fields.
  - Why it matters: our spatial and spectral detail should not exceed what the observation operator can resolve.
  - Link: https://arxiv.org/abs/2304.06706

- K-Planes. Factorized multiscale feature planes across dimensions.
  - Why it matters: closest clean template for structured `sky-lambda` factorization.
  - Link: https://arxiv.org/abs/2301.10241

- 3D Gaussian Splatting.
  - Why it matters: explicit primitives and fast rendering mindset; less direct than INR, but still relevant as a representation contrast.
  - Link: https://arxiv.org/abs/2308.04079

### Most relevant recent papers already reviewed

- Scalable and High-Quality Neural Implicit Representation for 3D Reconstruction.
  - Why it matters: overlapping local fields / scalability ideas for large sparse mosaics.
  - Link: https://arxiv.org/abs/2501.08577

- Sampling Theory for Super-Resolution with Implicit Neural Representations.
  - Why it matters: strongest support for operator-aware spectral and spatial capacity limits.
  - Link: https://arxiv.org/abs/2506.09949

- Deep Spectral Prior.
  - Why it matters: inspiration for regularizing spectral modes instead of letting the model invent unconstrained detail.
  - Link: https://arxiv.org/abs/2505.19873

- Physics-Guided Dual Implicit Neural Representations for Source Separation.
  - Why it matters: useful template for shared-signal plus nuisance/residual decomposition.
  - Link: https://arxiv.org/abs/2507.05249

- Where Do We Stand with Implicit Neural Representations? A Technical and Performance Survey.
  - Why it matters: compact survey and checklist to keep us from rediscovering known failure modes.
  - Link: https://arxiv.org/abs/2411.03688

## New additions from Session Buddy export

These are the additional papers from
[session-buddy-export-2026-03-27-17-07-05.json](/Users/seb/src/torchsky/session-buddy-export-2026-03-27-17-07-05.json)
that look most worth checking next.

### Highest priority

- Joint multiband deconvolution for Euclid and Vera C. Rubin images.
  - Why it matters: directly about joint multiband fusion across different resolutions and overlapping filters.
  - Relevance to us: strong near-neighbor to our multiband latent-field goal, even though it is not an INR paper.
  - Link: https://arxiv.org/abs/2502.17177

- Unified Spectrospatial Forward Models: Spatially Continuous Maps of Weak Emission Lines in the Rosette Nebula with SDSS-V LVM.
  - Why it matters: probably the closest astronomy-native paper in spirit to what we want, namely spatially continuous spectrospatial forward modeling with shared information across neighboring positions.
  - Relevance to us: good reference for how to think about continuous sky-plus-spectrum modeling without falling back to per-pixel fitting.
  - Link: https://arxiv.org/abs/2510.07395

- SuperF: Neural Implicit Fields for Multi-Image Super-Resolution.
  - Why it matters: direct INR-based multi-image super-resolution from multiple shifted observations.
  - Relevance to us: very relevant for the "many exposures per band versus one latent field" question.
  - Link: https://arxiv.org/abs/2512.09115

- Linear Image Regridding and Coaddition with Oversampled Point Spread Functions: Lessons from 1D.
  - Why it matters: this is exactly the kind of paper that can explain why simple resample-plus-coadd remains hard to beat.
  - Relevance to us: needed to understand the baseline properly, not just the fancy model.
  - Link: https://arxiv.org/abs/2510.16110

- fiDrizzle-MU: A Fast Iterative Drizzle with Multiplicative Updates.
  - Why it matters: modern iterative drizzle-style baseline.
  - Relevance to us: if INR is to be credible, it has to beat strong iterative coaddition baselines too, not only a simple bilinear stack.
  - Link: https://arxiv.org/abs/2511.09881

### Second priority

- ampere: A tool to fit heterogeneous observations consistently.
  - Why it matters: not an INR paper, but directly about consistent fitting of heterogeneous observations with flexible likelihoods.
  - Relevance to us: useful for likelihood design, nuisance handling, and uncertainty treatment.
  - Link: https://arxiv.org/abs/2510.26333

- BrahMap: A scalable and modular map-making framework for the CMB experiments.
  - Why it matters: a mapmaking systems paper, not a representation paper.
  - Relevance to us: important for large-scale streaming/chunking ideas once we push beyond local patches and sparse `128+` mosaics.
  - Link: https://arxiv.org/abs/2501.16122

- Physics Constrained Learning for Data-driven Inverse Modeling from Sparse Observations.
  - Why it matters: older but clean inverse-problem paper about training under hard physical constraints from sparse data.
  - Relevance to us: useful for thinking about stronger forward-model contracts beyond penalty-only fitting.
  - Link: https://arxiv.org/abs/2002.10521

### Optional background papers

- Cosmological Spectral Deconvolution.
  - Why it matters: classic operator-thinking reference for how broad observations constrain latent spectra.
  - Relevance to us: helps motivate observability-limited spectral parameterizations.
  - Link: https://arxiv.org/abs/1403.3727

- Broadband Intensity Tomography: Spectral Tagging of the Cosmic UV Background.
  - Why it matters: broad-band observations extracting latent spectral information.
  - Relevance to us: useful conceptual support for the claim that overlapping filters can recover more latent spectral structure than naive per-band treatment suggests.
  - Link: https://arxiv.org/abs/1810.00885

## Additional theory and methods shortlist

This second pass is intentionally less astronomy-specific and more about transferable methods from CV, ML, inverse problems, and uncertainty.

### Representation and conditioning

- WIRE: Wavelet Implicit Neural Representations.
  - Why it matters: explicit multiscale wavelet bias inside an INR.
  - Relevance to us: one of the clearest candidates for improving spatial detail without losing large-scale coherence.
  - Link: https://arxiv.org/abs/2301.05187

- Factor Fields: A Unified Framework for Neural Fields and Beyond.
  - Why it matters: structured factorization beyond plain MLP coordinate models.
  - Relevance to us: useful reference for building a stronger unified field family instead of keeping a patchwork of winners by regime.
  - Link: https://arxiv.org/abs/2302.01226

- Attention Beats Concatenation for Conditioning Neural Fields.
  - Why it matters: conditioning strategy can matter as much as backbone choice.
  - Relevance to us: directly relevant to bandpass or band-conditioned field heads, where simple concatenation may be leaving performance on the table.
  - Link: https://arxiv.org/abs/2209.10684

- Robustifying Fourier Features Embeddings for Implicit Neural Representations.
  - Why it matters: directly about making the standard positional encoding path behave better.
  - Relevance to us: especially relevant because we already saw coordinate scaling and precision matter in our fits.
  - Link: https://arxiv.org/abs/2502.05482

- Grids Often Outperform Implicit Neural Representations.
  - Why it matters: important reality check against INR aesthetic bias.
  - Relevance to us: if explicit multiscale grids or hybrid grid-plus-decoder models are just better in our regime, we should know that early.
  - Link: https://arxiv.org/abs/2506.11139

### Inverse problems and multi-image fusion

- Handheld Multi-Frame Super-Resolution.
  - Why it matters: strong classical CV reference for combining many shifted noisy frames without hallucinating detail.
  - Relevance to us: one of the best non-astronomy analogs to "many exposures, one latent image".
  - Link: https://arxiv.org/abs/1905.03277

- Model-Based Deep Learning.
  - Why it matters: broad framework for mixing forward operators with learned components.
  - Relevance to us: directly supports moving toward operator-aware latent mapmaking instead of pure black-box INRs.
  - Link: https://arxiv.org/abs/2012.08405

- Local Conditional Neural Fields for Versatile and Generalizable Large-Scale Reconstructions in Computational Imaging.
  - Why it matters: large-scale reconstruction with local conditional fields is very close to our sparse large-mosaic pain point.
  - Relevance to us: probably the most promising generic methods paper for the "large sparse support, local fidelity, global consistency" problem.
  - Link: https://arxiv.org/abs/2307.06207

### Uncertainty and calibration

- Density-Calibrated Conformal Quantile Regression.
  - Why it matters: stronger distribution-aware uncertainty calibration than naive heteroscedastic heads.
  - Relevance to us: promising next layer once we want uncertainty maps for INR that are actually comparable to coadd inverse-variance products.
  - Link: https://arxiv.org/abs/2411.19523

- Conditional Coverage Diagnostics for Conformal Prediction.
  - Why it matters: good uncertainty methods often fail in the regions that matter most.
  - Relevance to us: very aligned with our support/fringe/far-extrapolated metric split and the need to diagnose where uncertainty is or is not trustworthy.
  - Link: https://arxiv.org/abs/2512.11779

## Additional picks from `inr_a`, `inr_b`, and `inr_c`

After revisiting
[inr_a.md](/Users/seb/src/torchsky/inr_a.md),
[inr_b.md](/Users/seb/src/torchsky/inr_b.md), and
[inr_c.md](/Users/seb/src/torchsky/inr_c.md),
the worthwhile additions are mostly from `inr_c`.

### Add these

- Refine Now, Query Fast: A Decoupled Refinement Paradigm for Implicit Neural Fields.
  - Why it matters: this is the clearest articulation of the design pattern we keep circling back to, namely explicit feature representation plus lightweight query plus optional refinement.
  - Relevance to us: directly supports building one stronger structured field family instead of accumulating many ad hoc branches.
  - Link: https://arxiv.org/abs/2602.15155

- TensoRF: Tensorial Radiance Fields.
  - Why it matters: compact low-rank tensor factorization as a practical alternative to large MLP fields.
  - Relevance to us: strong reference for moving beyond plain MLPs while keeping a continuous latent field.
  - Link: https://arxiv.org/abs/2203.09517

- F-INR: Functional Tensor Decomposition for Implicit Neural Representations.
  - Why it matters: updated factorized-INR direction beyond the early tensor/plane methods.
  - Relevance to us: especially relevant if we want one family that can cover both local and larger sparse regimes without becoming a huge decoder.
  - Link: https://arxiv.org/abs/2503.21507

- INR-Bench: A Unified Benchmark for Implicit Neural Representations in Multi-Domain Regression and Reconstruction.
  - Why it matters: benchmark paper that helps separate universal claims from architecture hype.
  - Relevance to us: useful for deciding when a fancy new INR family is likely to transfer and when a simpler Fourier-encoded MLP remains the right baseline.
  - Link: https://arxiv.org/abs/2510.10188

### Do not prioritize from those notes yet

- `inr_a.md` mentions MoE-FINER, NSTR, and HC-INR as promising, but the note does not pin them to robust source references in the repo, so they should stay in the "maybe later" bucket until we verify them properly.
- `inr_b.md` is still dominated by surface, lighting, and dynamic-scene reconstruction papers that are too far from our 2D-on-sky latent mapmaking problem.

## Targeted search follow-up

This is the implementation-focused follow-up search for the current bottlenecks:

- support-fidelity under sparse multiframe fusion
- operator-aware spectral capacity
- hybrid continuous plus residual representations
- stronger non-INR baselines

### New short list

- ImageMM: Joint multi-frame image restoration and super-resolution.
  - Label: `baseline reference`
  - Problem solved: joint restoration and super-resolution from multiple frames.
  - Transfer to torchsky: strong analogue to many exposures per band with one latent target image.
  - Not transferable: camera/image pipeline assumptions and generic RGB restoration losses.
  - Implementation implication: add one stronger benchmark-local multiframe iterative baseline beyond bilinear stack.
  - Link: https://arxiv.org/abs/2501.03002

- Towards Super-resolution via Iterative multi-exposure Coaddition.
  - Label: `baseline reference`
  - Problem solved: repeated exposure fusion through an explicitly iterative coaddition scheme.
  - Transfer to torchsky: likely the closest classical baseline family for our “many visits, one latent image” setting.
  - Not transferable: not designed for continuous spectrospatial latent fields.
  - Implementation implication: stage an iterative coadd-style benchmark baseline before claiming broad INR wins.
  - Link: https://arxiv.org/abs/2209.07673

- Mixture of neural fields for heterogeneous reconstruction in cryo-EM.
  - Label: `keep for later`
  - Problem solved: heterogeneous reconstruction with a mixture of fields rather than one single field.
  - Transfer to torchsky: relevant if one unified field still cannot cover all regimes or sky components cleanly.
  - Not transferable: cryo-EM geometry and data model are different from sky mapmaking.
  - Implementation implication: only revisit if the unified family still fails badly on mixed regimes.
  - Link: https://arxiv.org/abs/2412.09420

- Joint Bayesian calibration and map-making for intensity mapping experiments.
  - Label: `keep for later`
  - Problem solved: coupled calibration plus map-making under a principled Bayesian formulation.
  - Transfer to torchsky: strong later reference once we move beyond pure reconstruction and into calibration nuisance parameters.
  - Not transferable: current blocker is still supported-pixel fidelity, not calibration.
  - Implementation implication: defer until after the next unified field phase.
  - Link: https://arxiv.org/abs/2509.10992

- Distribution-free uncertainty quantification for inverse problems: application to weak lensing mass mapping.
  - Label: `keep for later`
  - Problem solved: uncertainty quantification with coverage guarantees in an inverse problem setting.
  - Transfer to torchsky: highly relevant once INR uncertainty needs to be competitive with coadd inverse-variance products.
  - Not transferable: does not directly improve current support-fidelity bottlenecks.
  - Implementation implication: use after the new field family stabilizes, not before.
  - Link: https://arxiv.org/abs/2410.08831

- Solving ill-posed inverse problems using iterative deep neural networks.
  - Label: `keep for later`
  - Problem solved: unrolled iterative inversion with learned updates.
  - Transfer to torchsky: useful reference for moving from “one-shot INR fit” toward alternating update schemes if needed.
  - Not transferable: PDE-style framing and inversion machinery are more general than our immediate need.
  - Implementation implication: possible future direction if direct latent fitting saturates again.
  - Link: https://arxiv.org/abs/1704.04058

- R2D2 image reconstruction with model uncertainty quantification in radio astronomy.
  - Label: `keep for later`
  - Problem solved: reconstruction plus uncertainty in a hard astronomical inverse problem.
  - Transfer to torchsky: useful for later uncertainty benchmarking and scientific trust.
  - Not transferable: different forward operator and imaging regime.
  - Implementation implication: later uncertainty reference, not a next-step architecture driver.
  - Link: https://arxiv.org/abs/2403.18052

- Spotting Hallucinations in Inverse Problems with Data-Driven Priors.
  - Label: `implement now`
  - Problem solved: diagnostics for when learned inverse methods invent structure unsupported by data.
  - Transfer to torchsky: directly relevant to our support vs uniform-grid split and to evaluating residual correction heads safely.
  - Not transferable: not a representation recipe by itself.
  - Implementation implication: add a benchmark/diff diagnostic pass for supported vs unsupported regions when evaluating the new unified family.
  - Link: https://arxiv.org/abs/2306.13272

## `torchsky.map.field` hash-grid encoder (implementation alignment)

The shared `_HashGridEncoder` used by `HashGridMLPField`, `HashGridResMLPField`, `HashGridSpectralField`, and `HashGridConditionedField` is shaped by the INR notes in this repo and the decision memo:

- **Explicit multiscale trunk** (`inr_decision_memo.md`, `inr_c.md`, Zip-NeRF / WIRE mentions in this file): logarithmically spaced resolutions with per-level feature vectors.
- **Collision mitigation on coarse scales** (`inr_b.md`): when `resolution ** coord_dim <= hash_max_dense_cells` (default 4096), the level uses a **dense** voxel table instead of hashing, so large-scale structure is not corrupted by hash collisions.
- **Finer levels, larger tables** (`inr_b.md`): `log2_hashmap_size` may be an `int` (shared) or a **per-level** sequence so capacity grows with resolution.
- **Smoother query footprint** (Zip-NeRF-style motivation in this file): optional `hash_interpolation="smoothstep"` replaces linear trilinear weights with smoothstep for slightly low-pass interpolation.
- **Decoupled refinement** (`inr_c.md` “Refine Now, Query Fast”, `inr_decision_memo.md`): `HashGridResMLPField` (`hashgrid_resmlp`) stacks the same encoder with a **residual MLP** decoder instead of a plain feedforward head.

Optional `hash_level_gain` supplies learnable per-level scalars on encoder outputs (useful when mixing many resolutions). Set `hash_dense_coarse=False` to recover the older “always hashed” behavior for reproducibility with legacy checkpoints.

## What these papers suggest for `torchsky`

The current strongest directions are:

1. Keep one continuous latent sky model, but constrain spectral degrees of freedom by the actual transmission operator.
2. Improve support-region fidelity without giving up the full-grid coherence that the feature field already achieves.
3. Benchmark against stronger classical baselines than simple bilinear coadd when possible.
4. Treat large sparse mosaics as a systems problem too: streaming, chunking, and scalable mapmaking matter.
5. Consider hybrid models where shared latent structure is combined with small residual or nuisance components, instead of forcing one pure representation everywhere.
6. Take uncertainty seriously as a region-conditional property, not a single scalar quality score.
7. Bias toward explicit multiscale feature fields, tensor factorizations, and decoupled refinement before trying more pure-coordinate MLP variants.

## What not to prioritize yet

- Generic view-synthesis papers that depend on camera rays, occlusion, or 3D geometry.
- More generic "better MLP" papers without a clear inverse-problem or multiband angle.
- Diffusion or flow papers that do not help with fitting heterogeneous observations under a known forward operator.
