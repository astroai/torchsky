# Torchsky: calibrated fields, instrument response, and discovery

Research and development plan · implementation in progress · 12 October 2026

This document defines research and implementation goals, not a claim of achieved performance. It reflects the working tree, including provisional untracked solvers, and primary literature/documentation available on the date above. The architecture and experiments are our synthesis; cited performance results belong to their authors and their stated datasets. The user authorized implementation after accepting revision 2; the [implementation status](field_response_discovery_status.md) distinguishes tested components from outstanding milestone and science gates.

Revision 2 incorporates the accepted [review and amendments](field_response_discovery_amendments.md): non-Bayesian inference by default, independently usable exposure workflows, query-specific uncertainty, and calibration guarantees with explicit assumptions.

## The recommendation

Make torchsky a **calibrated sky-field inference engine using native likelihoods**, with non-Bayesian inference as the default. PSF estimation, background separation, detection, segmentation, and catalogs cooperate through shared response and evidence contracts. Its defining scientific product should be a queryable spatial–spectral–temporal sky with uncertainty and an explicit account of which observations support each result. Reusable native likelihood factors and calibrated operators preserve the evidence behind every reconstructed view.

The strongest opportunity is not a single universally better PSF model. It is a system that can combine native measurements from very different instruments without losing their response, noise, history, or calibration information, then demonstrate improved scientific recovery under controlled tests.

Ten decisions should govern the work:

1. **Keep the celestial field independent of the instrument.** Condition instrument response on instrument state; do not let an instrument identifier silently change the same celestial sky.
2. **Fit native measurements.** Start with minimally altered calibrated exposures; support raw reads when nonlinear or history-dependent detector effects require them. Avoid making resampled mosaics the inference substrate.
3. **Separate optical transport from detector response.** A spatially varying, chromatic optical response can still be linear in incident intensity. Brighter–fatter, saturation, persistence, coincidence loss, and pile-up require additional mechanisms.
4. **Treat the additive term as a component graph.** Celestial diffuse emission, changing environmental illumination, instrument emission, electronic offsets, and stochastic noise enter at different stages.
5. **Use physics, empirical models, and learned residuals on one continuum.** Small datasets should work with constrained models; increased capacity must earn its place on independent evidence.
6. **Derive discovery and catalog views from the field and the measurement likelihood.** Do not force every emitter into a blob, point source, common multiband morphology, or galaxy class.
7. **Publish resolution and uncertainty honestly.** Fine sampling, precise localization, recovered angular detail, and regularized/learned prediction are different claims. Measurement prediction, latent-parameter confidence and expected segmentation risk need separate contracts.
8. **Treat all measurements as evidence, with appropriate semantics.** Poor seeing does not justify discarding a useful exposure; corrupted samples do not become celestial signal merely because they were recorded.
9. **Keep exposure workflows independently useful.** PSF/background calibration, detection, forced photometry and transient queries must work without converging a global field. They share the same operators and evidence conventions as field reconstruction.
10. **Use likelihood-based estimates and requested uncertainty queries by default.** Preserve full information operators where quadratic compression is valid; declare regularization and representation bias. Reuse optional torchregress calibrators under audited assumptions. Bayesian methods remain available for bounded problems where they earn their cost and assumptions.

## 1. What the existing methods teach us

### PSF models: inherit their strengths, test their assumptions

**DAOPHOT / ALLFRAME.** These remain essential crowded-field references: local background estimation, empirical stellar profiles, simultaneous fitting of neighbors, and repeated observations. Their natural comparison task is stellar astrometry/photometry and crowding, not arbitrary diffuse-field reconstruction. Torchsky should preserve an efficient explicit-source view for calibration stars and severe crowding even when the principal sky representation is continuous. [Stetson, DAOPHOT](https://articles.adsabs.harvard.edu/pdf/1987PASP...99..191S), [crowded-field photometric recovery with DAOPHOT/ALLFRAME](https://arxiv.org/abs/astro-ph/9911193).

**PSFEx.** It fits sampled basis representations and smooth context dependence, explicitly addressing undersampling; it is not merely a Gaussian or spatially constant PSF. It supplies a strong, inexpensive empirical baseline. A JWST NIRCam study found PSFEx performed best among the three construction methods it tested, and demonstrated material AGN/host biases from PSF mismatch. That result is specific to that comparison, not a universal ranking. [PSFEx algorithm](https://psfex.readthedocs.io/en/latest/Working.html), [Bertin 2011](https://adsabs.harvard.edu/pdf/2011ASPC..442..435B), [Zhuang & Shen 2023](https://arxiv.org/abs/2304.13776).

**Piff.** Its model/interpolator separation, WCS handling, reserve stars, and diagnostics are useful architecture precedents. DES Y6 added color-dependent PSF modeling, cleaner stellar selection using external photometry, and fourth-order moment diagnostics. Torchsky therefore needs to compete against current chromatic Piff, not a caricature of traditional achromatic fitting. [Piff](https://rmjarvis.github.io/Piff/), [DES Y3](https://arxiv.org/abs/2011.03409), [DES Y6, 2025](https://arxiv.org/abs/2501.05781).

The DES Y6 paper also describes ongoing composite optics/atmosphere/diffusion modeling. Physics-guided decomposition is consequently a competitive baseline and research direction, not sufficient novelty by itself. [Piff future improvements](https://arxiv.org/html/2501.05781v3#S5).

**CosmoStat: RCA, MCCD, WaveDiff.** I interpret “cosmostats” as the CosmoStat PSF program. RCA uses constrained low-rank recovery and spatial structure. MCCD couples global focal-plane modes with local detector structure; its paper reports a 22% pixel RMSE improvement over PSFEx on its CFIS data. WaveDiff shifts the learnable representation into wavefront space and differentiates through optical propagation; its original gains were established in a simplified space-telescope setting. These motivate two complementary model families, empirical focal-plane response and physical wavefront response, rather than one mandatory parameterization. [RCA](https://www.cosmostat.org/publications/constraint-matrix-factorization-for-space-variant-psfs-field-restoration), [MCCD](https://arxiv.org/abs/2011.09835), [WaveDiff](https://arxiv.org/abs/2203.04908).

**Recent work strengthens the identifiability requirement.** A 2026 WaveDiff study reports that excellent image-space PSF recovery in the original model coexisted with much poorer wavefront recovery; its modified optimization improves wavefront recovery in its tested setting. A learned response can predict measurements correctly without uniquely recovering its physical causes. Treat physical parameter interpretation as a separate validation task. [Centofanti et al. 2026](https://arxiv.org/abs/2607.01032).

**Effective PSFs and physical simulators.** Anderson & King show why the effective response of pixels to a point source is central to undersampled photometry and astrometry. STPSF supplies wavelength-, position-, and optical-state-aware JWST/Roman simulations, including conventions for detector effects. Use these as calibration products and comparators; record whether an input response already includes pixel integration or detector effects. [Anderson & King 2000](https://arxiv.org/abs/astro-ph/0006325), [STPSF](https://stpsf.readthedocs.io/en/latest/), [recent ePSF modeling study](https://arxiv.org/abs/2512.16764).

**The neglected wide-angle response matters.** Elderflower fits scattered light from bright stars jointly with background. Core-only PSF scoring can miss the dominant error in faint halos and diffuse emission. Torchsky needs a calibrated core, intermediate structure, and long-range wings, including sources outside a reconstruction tile. [Liu et al., wide-angle PSF modeling](https://doi.org/10.3847/1538-4357/ac32c6).

**Telemetry-guided differentiable optics is a research lane.** Recent MagAO-X laboratory work uses a differentiable physical model and wavefront telemetry to avoid companion self-subtraction; VLT/ERIS work also illustrates sensitivity to jitter differences between telemetry and the science channel. These are reasons to retain telemetry and source-dependent throughput in the response graph, not evidence that survey-wide inference is solved. [MagAO-X, 2026](https://arxiv.org/abs/2608.24445), [VLT/ERIS, 2026](https://arxiv.org/abs/2607.29524).

### The closest field and discovery precedents

The central direction already has important precedents. D4PO reconstructs multi-domain photon fields and components with uncertainty. Latent-space Field Tension uses model tension to propose additional emission components. Aim-resolve couples radio field reconstruction with component identification and catalog uncertainty. These methods make field-derived discovery a credible direction, but prevent a claim that the idea itself is new. [D4PO](https://arxiv.org/abs/1802.02013), [Field Tension, 2025](https://arxiv.org/abs/2506.20758), [aim-resolve, 2025](https://arxiv.org/abs/2512.04840).

The proposed contribution is the combination of **cross-instrument response calibration, nonlinear detector history, open morphology/time structure, scalable field inference, and auditable scientific validation**. Each part needs its own comparison and failure criteria.

**AstroPhot and modern parametric fitting are close comparators too.** AstroPhot already uses PyTorch/autodifferentiation for simultaneous sky, source and PSF inference across multiple images, with parameter covariance and sampling options. Torch-native joint optimization alone is therefore not a differentiator. PySersic supplies an additional gradient-based Bayesian parametric reference. SourceXtractor++ is the operational model-fitting engine used by COSMOS2025 on non-PSF-homogenized measurement images; its source catalog and configured background/parametric assumptions make it a necessary flagship comparator. [AstroPhot](https://arxiv.org/abs/2308.01957), [PySersic](https://arxiv.org/abs/2306.05454), [COSMOS2025 measurement methodology](https://doi.org/10.1051/0004-6361/202555799), [SourceXtractor++ background model](https://sourcextractorplusplus.readthedocs.io/en/stable/Background.html).

## 2. The mathematical substrate

### A radiation field, not a universal image

Let \(I(\boldsymbol\omega,\lambda,t)\) denote incident photon spectral radiance, for example in photons s\(^{-1}\) m\(^{-2}\) sr\(^{-1}\) μm\(^{-1}\), at a declared observer/reference convention. \(\boldsymbol\omega\) is a sky direction; \(t\) includes a declared time scale and exposure interval. Energy radiance and \(I_\nu\) representations need explicit Jacobians and photon-energy conversions, not a silent unit switch.

The scalar intensity is nonnegative. Optional polarization uses Stokes fields or a positive semidefinite coherency representation. Signed polarization components, absorption contrasts, and signed interferometric image products do not justify constraining every public field to be positive. Preserve nonnegative *total physical intensity* while allowing appropriate signed derived quantities.

The overarching contract is

\[
p(\{y_e\}\mid I,\eta,h)
= \prod_e p_e(y_e\mid I,\eta_e,h_e)
\]

only when observations are conditionally independent given the shared calibration variables and detector histories. Otherwise use the appropriate joint factor. \(\eta\) includes response/calibration variables; \(h\) includes detector state and previous illumination. Default inference estimates fields, calibration variables and requested scientific functionals through likelihoods or justified estimating equations, explicit physical constraints, and declared regularization where needed. A posterior is an optional interpretation or downstream method, not the defining product.

Preserve native likelihood factors, calibrated response operators, processing/history provenance and identifiable modes. Fixed linear-Gaussian blocks additionally admit exact quadratic compression. Neither deterministic optimization nor conformal calibration removes the assumptions of a restricted basis, positivity constraint, learned model or penalty. Signed measurement estimators can be appropriate even though the underlying total physical intensity is nonnegative.

For a linear photon-counting imager before nonlinear detector effects, a useful special case is

\[
\Lambda_{ep}=
\int_{\Delta t_e}\!dt\int d\lambda\int d\Omega\,
A_e(\lambda,t)\,q_e(\lambda,\boldsymbol\omega,t)
\,K_{ep}(\boldsymbol\omega,\lambda,t)
\,[I(\boldsymbol\omega,\lambda,t)+F_e(\boldsymbol\omega,\lambda,t)]
+\Lambda^{\rm internal}_{ep}.
\]

Here \(A\) is collecting area, \(q\) is photon detection efficiency, and \(K_{ep}\) is the fraction reaching pixel \(p\), including its footprint and any declared intrapixel response. \(F_e\) represents environmental photon foregrounds. The integral gives expected detected photoelectrons under a one-electron-per-detected-photon convention; instruments with other conversion mechanisms must specify them. Dark current and electronic offsets belong at their own stages. Spectral/time integration, geometric Jacobians, quantum efficiency, exposure duration, and pixel area must each appear exactly once.

An optical point-response density is **source-position conditioned**:

\[
K_e(\mathbf u\mid\boldsymbol\omega,\lambda,t,\eta_e).
\]

Convolution is the special case where the shape depends only on displacement. A chromatic, spatially varying \(K\) does not by itself break linearity. Signal-dependent charge transport does.

For nonlinear detectors, model latent arrivals/charge and their readout distribution. Writing \(D(\Lambda)\) as the mean is an approximation: generally \(E[D(Q)]\ne D(E[Q])\). Keep the distinction in the likelihood contract and quantify approximation error where it matters.

### Nine low-level primitives worth building

1. **A field protocol.** Batched evaluation and integration over sky regions, spectral intervals, and time intervals; explicit units, support, reference frame, and optional polarization. Explicit grids, adaptive finite elements/Foam, sparse source measures, Gaussian-process fields, and neural implicit fields should implement the same scientific contract. A point source is a measure-valued component; do not require an MLP to approximate a delta function.
2. **A measurement-response protocol.** Predict native measurements. Provide an exact adjoint for a linear map, and Jacobian-vector/vector-Jacobian products for nonlinear maps. Declare domains, codomains, validity, losses, support, and whether pixel/detector effects are already included. An adjoint of a nonlinear response means the adjoint of its local Jacobian, not a fictitious global linear operator.
3. **Controlled integration and sampling.** Pixel footprints, intrapixel sensitivity, bandpass, exposure-time averaging, spectral redistribution, and read times, with convergence/error controls. Different sampling grids must not change the underlying physical scene.
4. **A state-conditioned response model.** Optical state, atmosphere, field position, wavelength, focal-plane detector, temperature, read mode, pointing/jitter, and telemetry. Separate persistent instrument parameters, exposure-level states, and calibration-supported residuals.
5. **A detector-state transition.** Charge accumulation and transport, nonlinear readout, saturation/censoring, persistence/traps, event coincidence/pile-up, read covariance, and reset/read history. Start with individually validated mechanisms, not one unconstrained learned state machine.
6. **A typed additive-component graph.** Each component declares its coordinate frame, physical stage, spectral/time dependence, units, stochastic contribution, uncertainty, and transfer through later response stages.
7. **A likelihood/noise protocol.** Gaussian with covariance; Poisson; Poisson–Gaussian; joint ramp likelihood; marked point process; complex visibility likelihood; and calibrated contamination mixtures. Include normalization when estimating variance, so uncertainty cannot grow freely to erase residuals.
8. **A calibration/inference graph.** Independent reference evidence, shared nuisance parameters, gauges, sparse overlap structure, cross-fitting, declared penalties/constraints, and uncertainty propagation. Distinguish profiling from calibrated random-effect uncertainty. Catalog measurements are optional evidence factors, not the definition of the sky.
9. **A field-query and discovery protocol.** Region integrals, spectral/time queries, local hypotheses, component proposals, support/membership, associations, covariance queries, calibration/selection reports, and provenance. Catalogs become compact indexed views of these results.

These are proposed contracts, not instructions to create nine speculative packages. Extend the existing scene/map/operator interfaces only as a concrete implementation demonstrates a need.

The response is a graph, not a universal fixed ordering of sampling and detector physics. Charge-dependent collection can alter effective pixel boundaries *during* integration; diffusion can precede sampling, while IPC/readout effects can follow it. Represent coupled collection/state evolution and the actual read sequence instead of attaching every detector correction after one fixed pixel-integration block.

## 3. Raw or detrended images?

**Default: native, minimally altered calibrated exposures with traceable response and uncertainty. Ultimate precision: move upstream only for the information a science goal needs.** Raw is not automatically more informative in a usable inference system if its calibration model is wrong or missing.

For optical CCD imaging, begin with validated bias/dark/gain/nonlinearity handling and native pixel geometry. Fit the effective response appropriate to that processing state. Record flat illumination/spectrum, throughput, source solid angles, detector footprints, pixel-area treatment and any point-source correction. A sky-flat response and point-source throughput are not interchangeable; one binary normalization flag cannot describe every instrument. Preserve sky levels, variance components, masks, and applied transformations. Add joint raw-electron inference when residual charge transport, readout, or calibration errors dominate a measured endpoint.

For IR arrays, calibrated single-exposure count rates are a practical first lane, but ramps contain timing and unsaturated-read information that rate images can discard. Brighter–fatter/charge migration, persistence, jumps, and read covariance may require group/read-level modeling. Do not fit a flux-conditioned PSF on a detrended rate product and assume it recovers all raw detector physics. [JWST pipeline stages](https://jwst.readthedocs.io/en/latest/jwst/user_documentation/available_pipelines.html), [JWST science products](https://jwst.readthedocs.io/en/latest/jwst/data_products/science_products.html).

For UV/X-ray photon instruments, calibrated events with attitude, timing, energy/grade, exposure/live-time, and response files are usually the appropriate native representation. Raw telemetry is unnecessary unless a validated reprocessing question requires it. Binning events into one broadband image discards useful spectral/time information.

For radio interferometers, retain calibrated visibilities, flags, weights, antenna/beam metadata, and calibration uncertainty where available. A restored CLEAN image is a derived estimate with complicated transfer and residual/noise conventions; it is not interchangeable with the visibility likelihood. Legacy maps may still be useful under a documented approximate response and covariance.

PSF calibration and science fitting must use **the same response convention**, or an explicitly modeled transformation between them. An ePSF calibrated on detrended data is appropriate for those data; a pre-pixel optical PSF is appropriate before pixel integration. Do not deconvolve detector effects twice.

Avoid untracked high-pass filtering, background subtraction, cosmetic inpainting, clipping, or PSF homogenization before inference. If a legacy product has already undergone these steps, retain its transfer function, covariance, lost modes, and provenance when recoverable. Label missing information. Never count a mosaic and its constituent exposures as independent evidence.

## 4. A PSF program that improves with data

### Capacity should grow along identifiable directions

Maintain three interchangeable families:

- **Constrained analytic/empirical response:** Gaussian/Moffat and richer profiles, ePSF grids, and small basis expansions. Useful for sparse calibration and fast baselines.
- **Physical response:** pupil amplitude and optical-path/wavefront fields, propagation, atmosphere, jitter, scattering, and geometry. Use appropriate optical models, ray tracing, or measured response for each instrument; a Zernike pupil is not a universal X-ray/radio model.
- **Hybrid response:** calibrated physics plus structured empirical residuals, with detector/focal-plane/exposure hierarchy and learned spatial–spectral–temporal conditioning.

A wavefront residual can be highly expressive without relinquishing propagation physics. An empirical response should remain available when physical state is unknown, the optical model is incomplete, or the data do not identify wavefront parameters. Compare the two on predictions and science, not aesthetic realism.

For an appropriate incoherent far-field optical model, the monochromatic intensity response derives from a pupil amplitude \(P\) and optical-path field \(W\): \(K\propto|\mathcal F[P\exp(2\pi i W/\lambda)]|^2\), with coordinates, throughput and normalization supplied by the optical geometry. A constrained modal \(W\) plus a learned residual is a concrete first physical family. Remove unobservable piston, handle pointing/tilt gauges, and test wavefront interpretation separately from intensity prediction. Fresnel propagation, coronagraphic transfer, scattering, atmosphere and detector blocks extend the response where that simplified model is invalid. [optical formulation and PSF review](https://arxiv.org/abs/2306.07996).

For empirical mixtures, source-dependent coefficients with normalized component kernels can conserve emitted flux in a lossless domain. Output-position blending can preserve a constant image while changing the integrated flux of a point source. In an equal-cell dimensionless transport matrix, column sums of one conserve point-source flux and row sums of one preserve a constant vector; both require a doubly stochastic matrix. A source-conditioned spatially varying PSF generally supplies only the first property. Test a point measure and uniform radiance with their correct source solid angles, detector footprints and calibrated throughput. A constant vector of integrated fluxes is not necessarily uniform radiance. Do not force both sums to one by renormalization. Finite images, vignetting, masks, coronagraphic attenuation, and truncated wings are legitimate losses: track them as throughput/support, rather than renormalizing missing photons into the footprint.

Add core and wing capacity separately. Determine finite support from a declared truncation-error budget for the intended photometry/surface-brightness endpoint. Handle long-range wings with coarse global fields, hierarchical source aggregation, or another measured scalable method; do not expand every local convolution to the whole survey.

### Break these assumptions explicitly

- **Homogeneity:** continuous focal-plane dependence plus detector-local residuals and real detector boundaries; atmospheric anisoplanatism when supported.
- **Achromatic broad filters:** propagate each spectral component through its wavelength-dependent response before integrating the band. A broad-band effective stellar PSF depends on the star's spectrum; internal source color gradients can defeat one effective PSF for an extended object. [color gradients](https://arxiv.org/abs/1105.5595), [atmospheric chromatic effects](https://arxiv.org/abs/1409.6273).
- **One response per exposure:** integrate changing optical state, pointing, and atmosphere over the exposure if these changes are measurable; infer uncertainty otherwise.
- **Flux-independent detector response:** model brighter–fatter as charge transport influenced by accumulated charge and surrounding illumination. Flat-field correlations and flux-dependent stars provide complementary evidence. CCD and MIRI mechanisms require instrument-specific models. [CCD brighter–fatter](https://arxiv.org/abs/1402.0725), [MIRI brighter–fatter](https://arxiv.org/abs/2303.13517).
- **Memoryless response:** retain prior illumination, trap state, read/reset timing, and persistence uncertainty. A crop needs relevant detector history outside that crop.
- **Saturation as a dead hole:** recover information from unsaturated reads, wings, and other exposures. Use a censored likelihood only when the readout actually supplies a known threshold; blooming, clipping, and grade rejection need their corresponding response. Fully saturated unresolved cores cannot yield arbitrary detail.
- **All PSF parameters are physical:** expose predictive equivalence and degeneracy. Phase-retrieval ambiguities, pupil gauges, centroid/WCS shifts, stellar SED/passband uncertainty, and blur/scene tradeoffs need constraints.

### Small data to large data

Initialize from validated instrument calibration and a small model. Pool stable properties across detectors/visits with explicit uncertainty; fit transient exposure states locally. Learn residuals from native calibration-star pixels and a likelihood appropriate to their processing state. In a Gaussian stamp model, analytically profile linear stellar flux and local additive coefficients; optimize continuous centroid, SED and PSF parameters with uncertainty from neighbors and external calibration. External astrometry is a calibrated measurement constraint with epoch/proper-motion uncertainty, without requiring Bayesian framing. Fitting pre-estimated normalized kernels alone loses some calibration uncertainty and couples star-selection errors to the model.

Evaluate a point source's ePSF directly at its continuous detector position/subpixel phase instead of rasterizing a delta source or bilinearly resampling noisy stamps. Distinguish pre-pixel optical response from an ePSF that already includes pixel integration, and apply that integration exactly once. Accuracy depends on response calibration, intra-pixel sensitivity, chromatic state, finite support and interpolation/quadrature; direct evaluation is not automatically exact. Extended fields use integration with a measured error budget rather than universal oversampling factors. Point measures are optional internal components or catalog anchors, not a compulsory scaffold for the field.

Reserve stars by identity, detector region, color, brightness, and subpixel phase. Increase model rank, spatial scales, spectral freedom, temporal detail, or history complexity only when independent prediction and science transfer improve. Archive calibration manifests and model provenance. When data are sparse or outside coverage, uncertainty should increase; a confident extrapolated neural kernel is a failure.

Good outcomes to measure include flux/centroid bias, encircled energy and wings, second/fourth moments, residual correlations, and science-specific errors. Small pixel RMSE alone can reward the wrong wings or conceal consequential shape errors.

## 5. Background: additive physics without subtracting the sky away

Use “background” as a role in a chosen analysis, not a single class of pixels.

1. **Celestial emission of interest:** unresolved populations, nebulosity, dust/cirrus, halos, streams, diffuse radio/X-ray emission, and any other real sky component. It belongs in the celestial field even when it complicates object photometry.
2. **Observer-dependent photon foregrounds:** zodiacal light, airglow/OH emission, thermal atmosphere, or time-varying foreground illumination. These enter before the appropriate photon response and contribute shot noise.
3. **Instrument photon emission and stray paths:** thermal emission, ghosts, scattered light, and detector illumination patterns. Their location in the optical graph determines which subsequent operators act on them.
4. **Detector charge and electronic terms:** dark current, pedestal/bias, amplifier structures, and readout effects. A stochastic read process is not merely a smooth additive mean.

Fringing, flat response, charge migration, and brightness-linked scattering may depend on illumination or wavelength. Putting every residual into a flexible additive image hides the mechanism and can absorb sky signal. A signed residual around a calibrated mean is useful; the physical photon/charge rates generating it still need valid total rates and noise.

Fit smooth terms, detector templates, amplifier/time modes, low-rank conditional fields, and physical foreground spectra as competing components. Admit finer structure when calibration/blank-sky evidence supports it; low frequency is not synonymous with instrumental background. Preserve abrupt electronics boundaries and real celestial discontinuities.

The central degeneracy is that a sky monopole can trade with exposure offsets, and diffuse sky can trade with PSF wings or foreground structure. Overlap graphs, dithers, rotation, observer geometry, blank-sky observations, external absolute measurements, and calibrated bright-star wings help, but cannot identify every mode. A zero-mean background constraint chooses a gauge; it does not measure an absolute celestial zero point.

Use joint inference with a declared nuisance treatment. For the Gaussian block \(y=AI+B\beta+\epsilon\), \(\epsilon\sim\mathcal N(0,C)\), distinguish two contracts:

1. **Fixed unknown background, profiled.** With \(W_0=C^{-1}\), eliminate linear coefficients using

\[
W_{\rm prof}=W_0-W_0B(B^TW_0B)^\dagger B^TW_0.
\]

The projection can remove celestial modes that overlap the background basis. A gauge chooses a convention; it does not restore their information.

2. **Externally calibrated random-effect uncertainty.** A justified Gaussian calibration mean/covariance for \(\beta\), independent of the declared read/noise term, gives

\[
y\mid I \sim \mathcal N(AI+B\bar\beta,\ C+B\Sigma_\beta B^T),
\]

with matrix-free Woodbury precision

\[
W_{\rm cal}=W_0-W_0B(\Sigma_\beta^{-1}+B^TW_0B)^{-1}B^TW_0.
\]

This need not impose a prior on the sky. A deterministic penalty on \(\beta\) can produce the same quadratic algebra without automatically supplying the same uncertainty interpretation. Use factorizations/solves and rank-revealing methods rather than explicit inverses. Cache small factors only while basis, mask, covariance and calibration remain fixed; never materialize a dense pixel covariance. Per-application work scales with the basis size, but factor construction has additional cost. Amplifier/fringe/time structure may need more than a universally prescribed low-rank basis. Fitting parameter-dependent covariance requires its likelihood normalization.

Keep projected information loss in the product rather than announcing a clean reconstruction of it. When calibration variables correlate with scene/noise variables, use the full joint covariance rather than adding independent error bars. Profiling is a valid frequentist operation; a Laplace determinant term belongs to an integration approximation and is scene-independent in the fixed linear-Gaussian case.

Blank-sky selection is a useful initialization, not proof of zero astrophysical emission. Undetected source flux can bias masking/modeling and dither-derived sky estimates; a realistic sky-subtraction study demonstrates this directly. PCAT-DE provides a useful joint diffuse/point-source comparator. [Watkins et al. 2024](https://arxiv.org/abs/2401.12297), [PCAT-DE](https://arxiv.org/abs/2307.10385).

Validate with paired injections into real measurements *before* the component being tested, measuring attenuation and leakage between diffuse sky, PSF wings, detector patterns, and source flux. Include blank fields, bright-star surroundings, and large structures crossing tile boundaries. Low residual RMS after fitting background can mean that genuine signal was removed.

## 6. Cross-wavelength response: share inference, preserve measurement physics

### Optical and near-IR imagers

Compose atmosphere/transmission/DCR, optical wavefront or empirical response, jitter, intrapixel sensitivity, charge diffusion/transport, and readout in their correct order. DCR acts through geometry and the source spectrum; it is not just a broadening coefficient. Share telescope/focal-plane structure while allowing detector boundaries and visit-dependent atmosphere/focus. Gaia-like astrometric anchors require reference errors, proper motion, parallax where relevant, and epoch propagation.

Global repeated-star calibration is an important precedent: FGCM models exposure-dependent atmosphere and passbands to obtain chromatic photometric corrections. Implement the needed tensor-native calibration factors rather than making FGCM a runtime backend. [Burke et al., FGCM](https://arxiv.org/abs/1706.01542).

### Infrared arrays and bolometers

For JWST-like arrays, distinguish charge diffusion, IPC after sampling, charge migration/brighter–fatter, nonlinearity, trap/persistence state, amplifier offsets, read covariance, and read/group selection. STPSF's simplified detector effects do not include every illumination/time-dependent effect. Pin the actual pipeline and calibration context, including what was already corrected. [STPSF detector effects](https://stpsf.readthedocs.io/en/stable/jwst_detector_effects.html), [charge migration](https://jwst-pipeline.readthedocs.io/en/latest/jwst/charge_migration/description.html), [NIRCam persistence](https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-persistence), [1/f noise](https://jwst-docs.stsci.edu/known-issues/1-f-noise).

Zodiacal/Galactic light, thermal observatory emission, and stray-light paths have different transfer functions. For far-IR/sub-mm bolometers, observations are often **scanning time streams** with pointing trajectories, temporal response, drifts, atmosphere, and correlated noise. Effective beam depends on scan and processing. Some high-pass PACS products remove extended emission; prefer timelines or maps with measured transfer. [JWST background model](https://jwst-docs.stsci.edu/jwst-general-support/jwst-background-model), [Planck HFI beam/time response](https://arxiv.org/abs/1303.5068), [PACS product definitions](https://archives.esac.esa.int/hsa/legacy/HPDP/PACS/PACS-P/PPSC/HPPSC_Explanatory_Supplement_v2.2.pdf), [Scanamorphos](https://arxiv.org/abs/1205.2576).

FIR confusion is partly source-count/association ambiguity and partly unresolved emission; optical catalog positions cannot be assumed exhaustive. XID+ is a strong comparator for posterior flux inference at prior positions, but a position-prior list cannot discover sources absent from it. [XID+](https://arxiv.org/abs/1606.05770).

### Ultraviolet

GALEX-like data naturally support calibrated photon-event models with time, detector position, aspect, sensitivity, and exposure. Global dead time and local MCP gain sag affect counts differently; a correction to the mean does not restore lost information. UVOT coincidence loss needs event footprint/frame occupancy and background dependence. Retain dust-scattered/line emission as sky signal while separately modeling airglow, zodiacal foreground, and detector patterns. [GALEX pipeline](https://www.galex.caltech.edu/researcher/techdoc-ch3.html), [GALEX dead time/gain sag](https://www.galex.caltech.edu/researcher/faq.html), [UVOT calibration](https://arxiv.org/abs/1004.2448), [diffuse UV foregrounds](https://arxiv.org/abs/1307.5232).

### X-ray

Use a marked-event response \(P(\mathbf u,c,g\mid\boldsymbol\omega,E,t,\eta)\): incident energy is redistributed into measured channels/grades. Include effective area/ARF, redistribution/RMF, energy/off-axis/azimuth response, attitude/dither, live time and selection. Cosmic photons, particle events, fluorescence and soft-proton flares need separate components. Particle background does not receive the celestial ARF/PSF. [Chandra PSF Central](https://cxc.harvard.edu/ciao4.18/PSFs/psf_central.html), [CIAO RMF](https://cxc.harvard.edu/ciao4.17/ahelp/mkrmf.html), [XMM off-axis PSF](https://xmm-tools.cosmos.esa.int/external/xmm_user_support/documentation/uhb/offaxisxraypsf.html), [ACIS background](https://cxc.harvard.edu/ciao/dictionary/acisbg.html).

Pile-up merges or rejects incident photons and changes event energy/grade; it is a nonlinear occupancy/selection process, not simply a wider kernel. A marked Poisson likelihood is appropriate only in its independent-arrival/acceptance regime. Do not promise to recover events discarded onboard. Low-count detection should use count likelihoods, with energy-dependent templates and nuisance uncertainty. [CXC pile-up](https://cxc.harvard.edu/ciao/why/pileup_intro.html), [Poisson matched filtering](https://arxiv.org/abs/1709.01524), [X-Sifter](https://arxiv.org/abs/2412.07858).

### Radio and millimetre interferometry

Predict complex visibilities from sky brightness coherency through Jones responses and baseline phases. Include primary beams, non-coplanar baselines, direction/time/frequency gains, averaging, polarization leakage, atmosphere/ionosphere, and RFI/flagging semantics. At fixed gains this is linear in sky coherency; jointly estimating sky and gains is nonlinear. A signed dirty beam is not a normalized positive photon kernel. [Smirnov RIME I](https://arxiv.org/abs/1101.1764), [RIME II](https://arxiv.org/abs/1101.1765).

Jointly fit array data and single-dish/total-power observations where available. Missing short spacings are missing spatial information, not a background to remove. Optical morphology cannot measure missing radio flux without an explicit cross-band prior. Keep gain–flux, phase–position, and leakage–polarization gauges visible. [ALMA extended structures](https://almascience.nrao.edu/about-alma/alma-basics), [joint direction-dependent imaging/calibration](https://arxiv.org/abs/2305.05489), [joint polarized calibration, 2026 revision](https://arxiv.org/abs/2511.16556).

Frequency-dependent morphology, curvature, spectral lines, and Faraday rotation require more than a single spectral index map. Current comparators include fast-resolve for scale, CHIBI for hierarchical multifrequency synchrotron inference, and aim-resolve for component discovery. Their useful domain priors should be optional families, not laws imposed on every field. [fast-resolve](https://arxiv.org/abs/2406.09144), [CHIBI, 2026](https://arxiv.org/abs/2606.04094).

## 7. The field representation and inference engine

### Flexible structure without compulsory common morphology

A pragmatic representation is a **sum of fields and sparse measures**, with different correlation and support behavior. The sum is useful without declaring each mathematical component to be a physical object. Allow:

- a multiresolution spatial–spectral field for persistent emission;
- sparse unresolved components where point-like structure is justified;
- localized residual fields for complex knots, arcs, rings, filaments, jets and discontinuities;
- temporal variation and moving/expanding support;
- physical absorption/transfer or signed contrast views where needed;
- optional polarization and spectral-line components.

Spatial–spectral factorization can be an economical low-data model. Permit local rank growth, wavelength-dependent support/centroids, independent residual channels, and discontinuities at spectral features. Do not demand \(I(\omega,\lambda)=M(\omega)S(\lambda)\), impose a global galaxy SED, or interpolate a smooth spectrum through wholly unconstrained frequency decades.

Couple physically related channels through learned, uncertainty-aware cross-covariances or hierarchical priors. Preserve an independent-channel limit. Shared latent factors must not turn an optical point source into a compulsory X-ray/radio counterpart or erase a radio-only lobe. Spectral resolution, like angular resolution, is bounded by the responses and data.

Keep one explicit-grid/finite-element statistical reference implementation. Compare Foam/adaptive fields, INR/hash-grid models, and hybrids against it under the same likelihood, response, priors, compute budget and output queries. Continuous coordinates are not evidence that a neural representation is scientifically superior.

### Twenty years of observations require time

Provide persistent sky plus temporal components, trajectories with reference epoch, and exposure-time integration. Proper motion/parallax, Solar-System motion, variable stars, transients, changing jets and expanding remnants cannot be represented faithfully by one timeless sharp image. Motion and variability need competing hypotheses so neither silently becomes a bad-pixel rejection rule.

A view of COSMOS should specify wavelength and epoch or a documented persistent/temporal summary. Do not imply that all photons from twenty years depict one instantaneous scene. Short-timescale phenomena need read/event/time-stream timing, not just exposure midpoint conditioning.

### Scalable joint inference

Provide two composable execution routes. **Exposure workflows** fit native PSFs/backgrounds, detect candidates, perform forced photometry and evaluate transient queries without requiring a converged field. **Field workflows** consume the same operators and evidence to reconstruct adaptive/tiled continuous emission, optional point measures, spectral/time structure and discovery views. A known catalog can accelerate queries but is not a prerequisite for either route.

Use spatial indices and observation-footprint graphs, bounded tiles with physically justified overlap, sparse local support and matrix-free response/Jacobian actions. Local scene updates should communicate with shared global calibration, background and wing variables. Share coarse large-scale sky/foreground modes across tiles to prevent seams and loss of extended structure.

For fixed linear-Gaussian blocks, let \(z_e\) be data centered by the calibrated additive mean and \(W_e\) the precision under the declared nuisance treatment. Preserve the likelihood's quadratic:

\[
b=\sum_e A_e^TW_ez_e,\qquad H=\sum_e A_e^TW_eA_e,
\qquad -2\ell(I)=I^THI-2I^Tb+c.
\]

Retain the full action of \(H\), using matrix-free factors or controlled sparse/structured compression. \((b,\operatorname{diag}H)\) is generally lossy; \(E[b]=HI\), not \(I\). Constants, noise normalization and native data remain needed for absolute predictive checks and some model comparisons. Shared calibration or correlated observations require a joint factor rather than independent per-exposure marginalizations.

Accumulating \(b\) can stream; applying matrix-free \(H\) can revisit exposure operators/caches and solving may require iterations. A single accumulation pass is not a single-pass unbiased reconstruction. Changes to calibration or scene-dependent covariance invalidate the fixed compression. Poisson events, censored observations, nonlinear detector/read history and other nonquadratic objectives retain their appropriate likelihoods.

Use PCG/LSQR and measured preconditioners for Gaussian blocks. For nonlinear blocks, use profiled variable projection or alternating/joint Gauss–Newton-type updates where justified. Cache geometry, bandpass quadrature and stable response states; invalidate caches when calibration changes. Stochastic sample objectives need inclusion probabilities, so data sampling does not change the likelihood weights. Preconditioning addresses convergence, not bias from a field basis that cannot represent the observations; test representation and optimization error separately.

Coordinate partitioning alone does not localize visibilities: radio Fourier measurements couple large sky regions. Start that lane with a global matrix-free Fourier operator, then introduce controlled approximations/domain decomposition validated against it. Detector memory also links observations outside a spatial tile.

Neural inference should first accelerate initialization, proposals, preconditioning or validated field representation. Approximate posterior inference is an optional research alternative. Recent BLISS/NPE work is encouraging, including varying PSF/background and DC2 catalog experiments, but its LSST result explicitly qualifies evidence under absence of model misspecification. Benchmark domain shift and unknown objects before replacing likelihood-based refinement. [varying-response NPE](https://arxiv.org/abs/2503.00156), [DC2 NPE](https://arxiv.org/abs/2510.15315).

Avoid dense sky × spectrum × time cubes and dense pixel covariance matrices as the default. Store covariance as operators/low-rank factors/conditional blocks and expose selected functionals. Optimize native kernels only after the algorithm and measured hotspot are established. CPU is the numerical reference; CUDA/MPS support follows explicit dtype/device capabilities.

For scale perspective, a hypothetical dense rendering of 2 deg² at 0.03 arcsec sampling contains 28.8 billion pixels: 115.2 GB for just one float32 channel, before uncertainty, gradients, bands or epochs (decimal GB; simple area/sampling calculation). This is an illustrative storage calculation, not a measured COSMOS workload. Adaptive storage, on-demand queries, bounded activations and distributed observation accumulation are architectural requirements at that scale.

### Uncertainty follows the scientific query

For an estimable linear functional \(g^TI\), the unregularized linear estimator has fixed-model sampling variance \(g^TH^{-1}g\) on the estimable, gauge-fixed subspace. A selected solve can evaluate it without producing an all-sky variance map. \(1/\operatorname{diag}H\) is a conditional sensitivity proxy or preconditioner, not generally \(\operatorname{diag}(H^{-1})\).

For a symmetric quadratic penalty \(P\) and estimate \((H+P)^{-1}b\), report mean transfer \(R=(H+P)^{-1}H\), sampling covariance \((H+P)^{-1}H(H+P)^{-1}\), and possible bias \((R-\mathrm{Id})I\). Inverse penalized curvature \((H+P)^{-1}\) is not this frequentist sampling covariance. Variance alone does not capture regularization bias. Nonnegativity constraints and learned/restricted bases also require bias/coverage checks.

For nonlinear source/blend queries, use local curvature with nuisance treatment and validate linearization. With query/source variables \(\theta\) and nuisance variables \(\eta\), the locally Gaussian effective precision is

\[
F_{\rm eff}=F_{\theta\theta}-F_{\theta\eta}F_{\eta\eta}^{-1}F_{\eta\theta},
\]

including legitimate calibration information and explicit singular gauges. Inverting only \(F_{\theta\theta}\) conditions on omitted parameters. Shared response, long wings and diffuse backgrounds may require shared low-rank coupling, boundary-response operators or larger budgeted solves. A small block is exact only for its declared closed model; conditional approximations need a coupled reference.

Make global inverse-diagonal maps optional with accuracy/cost budgets. Stochastic probing is a numerical technique usable in either frequentist or Bayesian calculations. For independent real Rademacher probes and symmetric \(M\), the estimated diagonal has per-pixel variance \(\sum_{j\ne i}M_{ij}^2/N\). Its relative error depends on coupling; a diagonal matrix is exact in one probe. Do not claim a universal 18% error or runtime multiplier from 32 probes. Keep global matrix-free solves available for reconstruction and selected coupled queries.

### Calibration with a declared target

Use different uncertainty contracts for different outputs:

- **Future native measurements:** predictive intervals/regions include photon/read noise and calibrated response uncertainty. Held-out stamps/exposures are noisy observed labels, not noiseless PSF or background truth.
- **Latent photometry and field functionals:** analytic/profile uncertainty, robust covariance or block resampling where justified; injection/reference-calibrated sets cover the declared target population under their assumptions.
- **Discovery and supports:** selected-hypothesis FDR and expected-loss calibration with their own null, selection and component definitions, as specified in Section 8.
- **Unidentified modes:** wide/unbounded intervals or unresolved alternatives. A finite numerical floor must not turn a nullspace into a measured quantity.

Split conformal supplies marginal prediction coverage under exchangeability of calibration and test units. Fixed Mondrian groups can supply within-group marginal coverage under within-group exchangeability and sufficient calibration samples. Spatial localization alone does not supply conditional coverage at each position. Covariate-shift weighting needs defensible density ratios, overlap and invariant conditional target law; estimated weights introduce error. General drift results have a distributional-discrepancy penalty that is not automatically known. [Weighted conformal](https://arxiv.org/abs/1904.06019), [beyond exchangeability](https://arxiv.org/abs/2202.13415), [conditional coverage limits](https://arxiv.org/abs/1903.04684).

Fisher-shaped scalar vector scores can calibrate joint regions without requiring Gaussian residuals, provided their shape model is frozen independently of calibration labels and the sampling assumptions hold. The Fisher orientation is an efficiency choice, not necessarily exact error geometry. Calibrate estimable coordinates; leave unmeasured directions unbounded. Coordinate-wise intervals do not imply joint vector coverage. Distributional scores can accommodate skew/asymmetry, but do not establish an exact full conditional CDF under arbitrary misspecification. [Distributional conformal prediction](https://arxiv.org/abs/1909.07889).

Known injections give recovery labels for their injected population, not the unknown real scene. Sharp/deep references have noise, selection, spectral differences and response uncertainty. Keep generator development, threshold calibration and final audit separate; include mismatch generators and rare/crowded regimes. Calibration may become uninformative outside support. A conformal multiplier does not correct point-estimate bias or restore lost spatial frequencies. The CosmoStat weak-lensing study is a relevant inverse-UQ precedent with population-dependent limitations. [Leterme, Fadili and Starck](https://arxiv.org/abs/2410.08831).

A complementary non-Bayesian research lane constructs confidence sets by inverting valid measurement-space tests jointly over sky and calibration variables, then minimizing/maximizing selected field functionals over that set. This can bound latent quantities without latent training labels. Coverage still requires defensible noise/response assumptions, and bounds may be costly or unbounded. Strict inverse-problem bounds are a useful precedent; monotonicity/convexity of particle spectra must not become universal sky assumptions. [Kuusela and Stark](https://arxiv.org/abs/1512.00905).

Attach a linked uncertainty report to each query family: estimand/units; confidence, prediction or expected-risk target; level and marginal/conditional/joint scope; method; calibration revision/unit/group/support/sample size and weight diagnostics; nuisance and regularization treatment; selection rule; assumptions; independent coverage/width; and extrapolation status. Keep large covariance/support products and shared metadata outside the lean object table. Post-hoc calibration is not universally constant-cost; include calibration, scoring, inversion and null-tail budgets.

### Reuse torchregress through an optional adapter

The current sibling tree supplies scalar/grouped/weighted predictors, `CQR`, `DistributionalConformal`, joint `MultivariateScoreConformal` regions, variance/PIT calibration and a `PredictiveBatch` container. Reuse these rather than duplicating generic calibrators. Dispatch separate vector calibrators for fixed groups where its current vector API lacks `groups=`; preserve squared-score semantics and infinite thresholds. `MultiTargetConformal` has coordinate-wise semantics and is not a substitute for a joint region. Astronomy response/resolution, null construction, selection and calibration provenance remain torchsky responsibilities. [Current conformal source](../../../torchregress/src/torchregress/losses/conformal.py), [post-hoc calibration](../../../torchregress/src/torchregress/calibration/posthoc.py).

Pin an audited revision and device/dtype contracts. Its current non-exchangeable coverage helper measures calibration weights' discrepancy from uniform; that alone cannot bound arbitrary test-score drift. Do not expose its bound as certified astronomical coverage until the needed distributional connection is established. Shift/CDF-matching diagnostics and variance scaling do not acquire guarantees merely by being called calibration. Dedicated FDR/AdaDetect and CRC implementations were not found in the targeted audit and remain proposed work.

Keep torchregress an optional adapter/extra or harness dependency: its SciPy/TorchMetrics requirements should not silently expand torchsky's lean default runtime. General UQ methods belong there; native astronomical injection campaigns, real-data transfer validation and scientific scorecards belong in torchsky-harness.

## 8. Detection, segmentation and lean catalogs

### Discovery is a hypothesis query

Search for an additional emission component, a coherent extended structure, a temporal change, a trajectory, an absorption feature, or evidence that the current model is inadequate. Candidate generators can use point templates, multiscale wavelets, ridges/curves, spectral features, change points, motion tracks, and learned proposals. Use their **union**, including individual-band candidates. A joint detector should not veto a real single-band source.

Rank/evaluate candidates in native measurement space while comparing source, PSF, background, artifact and motion explanations. A high-SNR peak in a fitted/denoised field is not independent detection evidence: the fit, regularization, search and selection affect its null distribution.

For a fixed linear Gaussian model with known precision \(W\), template \(q\), stacked response \(A\), and null residual \(r\), let \(v=Aq\) and let \(J\) contain the already-fitted null scene/background modes. The efficient nuisance-adjusted score uses

\[
W_\perp=W-WJ(J^TWJ)^\dagger J^TW,
\qquad s(q)=\frac{v^TW_\perp r}{\sqrt{v^TW_\perp v}}.
\]

With no fitted nuisance modes this reduces to the usual matched-filter score. Its simple standard-normal null applies only under the fixed linear assumptions. Estimated covariance, nonlinear fitting and constraints need further calibration. A candidate with zero projected information is unidentifiable. Using an unprojected denominator on fitted residuals can make scores conservative rather than correctly standardized. Searching location, scale, morphology, spectrum or motion introduces multiplicity. Calibrate the *whole* fitting/proposal/refinement procedure, or use genuinely independent confirmatory measurements when available.

For independent Poisson counts and a fixed nonnegative expected-count increment \(a p_i\) above positive null means \(\mu_{0i}\), the log likelihood ratio is

\[
\log\frac{L_1}{L_0}
=\sum_i\left[y_i\log\left(1+\frac{a p_i}{\mu_{0i}}\right)-a p_i\right].
\]

Evaluate the positive-rate expression with `log1p` where appropriate. Handle zero null rates by their probability support: zero counts are valid under a zero-rate null and a positive source increment contributes a negative log-ratio; positive counts have zero probability under that null. Do not invent a background floor merely to avoid division by zero. Unknown backgrounds, amplitude/position scans, event selection and pile-up change the inference/test. These are useful recovery limits, not universal detection formulas.

Conformal rank p-values are an optional calibration route when null and candidate scores have the required exchangeability/selection semantics. Blank cutouts can contain unresolved emission; sign flipping needs a symmetric null, and phase randomization need not preserve it. Peak finding and learned proposal selection change the tested population. AdaDetect has a specific conditional-null exchangeability and score-symmetry construction supporting its FDR theorem; arbitrary scores followed by BH do not inherit it. Use proven constructions when applicable, or dependence-appropriate procedures with valid selected-hypothesis p-values. A calibration bank of \(m\) nulls has minimum rank p-value \(1/(m+1)\); budget null-tail resolution against the full search multiplicity. Measure completeness at achieved FDR. [AdaDetect](https://arxiv.org/abs/2208.06685).

Classical baselines remain necessary: SExtractor/SEP for operational source extraction; NoiseChisel for faint irregular/diffuse detection; ProFound for segmentation photometry; and Poisson/temporal detectors in their own regimes. [SExtractor](https://aas.aanda.org/articles/aas/pdf/1996/08/ds1060.pdf), [SEP](https://www.theoj.org/joss-papers/joss.00058/10.21105.joss.00058.pdf), [NoiseChisel](https://arxiv.org/abs/1505.01664), [ProFound](https://arxiv.org/abs/1802.00937), [TRANSLIENT](https://arxiv.org/abs/2403.09771).

### Segmentation is a conditional interpretation

Keep hierarchical supports and alternative associations. A lobe, its host, a star-forming knot and its parent, overlapping clouds, a blend, or a stream may require different scientific groupings. The underlying radiance field cannot determine a unique count of “objects” without an ontology or prior.

Produce soft sky memberships and per-observation component responsibilities when the model supports them; their interpretation depends on wavelength, epoch, target resolution, and component definition. Leave diffuse/unassigned emission in the field. Hard segmentation is an exported view. Absorption and polarization may need different attribution rules from positive emission mixtures.

Conformal Risk Control can tune a support parameter for a declared bounded monotone loss under exchangeable calibration/test units, with the finite-sample correction. Missed-flux fraction can decrease monotonically as support expands; contamination fraction need not. Inner high-purity and outer containment supports are useful exports, but require separately valid losses and a feasible tradeoff. Expected-risk control is not guaranteed containment for every object. If a blend admits no support meeting both goals, report ambiguity rather than force two certified masks. Calibration needs explicit component labels/flux definitions or declared injection/reference targets. [Conformal Risk Control](https://arxiv.org/abs/2208.02814).

Compare current scarlet2, not only scarlet's older factorization. Scarlet2 already supports multiple instruments/epochs, learned priors and sampling. Tractor/Legacy Surveys supplies a native-exposure parametric fitting reference; multiband PCAT supplies uncertain source-count/blend alternatives; current multiband crowdsource is a crowded-star reference. [scarlet](https://arxiv.org/abs/1802.10157), [scarlet2, 2026](https://joss.theoj.org/papers/10.21105/joss.09646), [scarlet2 multiresolution](https://scarlet2.readthedocs.io/en/stable/howto/multiresolution.html), [Tractor](https://thetractor.org/doc/intro.html), [Legacy Surveys DR10](https://www.legacysurvey.org/dr10/description/), [multiband PCAT](https://arxiv.org/abs/1907.04929), [multiband crowdsource, 2026](https://arxiv.org/abs/2604.24875).

### Keep the catalog small by storing measurements properly

A compact object/structure index can contain ID, support or reference position, reference epoch, interpretation/evidence summary, ambiguity/quality flags, and references to field, uncertainty and provenance products. An existence probability is conditional on a model/prior; a detection statistic is not automatically that probability.

Store requested photometry, spectral/temporal summaries and shape measurements in a long measurement relation keyed by object/query, band, epoch and definition. Keep large covariance, light curves, morphology fields, uncertainty/calibration reports and alternative interpretations in linked products. This keeps the default catalog lean without discarding information or multiplying columns for every instrument and aperture. Catalog entries need evidence/measurement semantics, without requiring Bayesian existence probabilities.

Preserve IDs through explicit split/merge lineage rather than forcing the same object count after new data arrive. Probabilistic association uses astrometry, motion, spectra, morphology, uncertainty and selection; geometric crossmatch only generates candidates.

Use catalogs in three roles: calibration anchors with uncertainties; proposal/forced-measurement queries; and independent comparison surfaces with their selection functions. Propagate reference errors, temporal variability and shared calibration covariance. Do not fit the field to COSMOS2025 and then use agreement with that same catalog as independent validation.

## 9. Why adding poorer observations should not blur the result

For a fixed, correct linear Gaussian model with independent new data,

\[
H_{\rm data}=\sum_e A_e^T C_e^{-1}A_e,
\qquad
\Delta H=A_{\rm new}^T C_{\rm new}^{-1}A_{\rm new}\succeq0.
\]

For an unregularized estimator on a fixed estimable subspace with nonsingular information, \(H^{-1}\) cannot increase in positive-semidefinite order as independent information is added. In the optional fixed Gaussian-prior model, \((P+H)^{-1}\) has the same monotonicity. Neither statement guarantees that every realized estimate, regularized reconstruction, nonlinear fit or empirical interval improves. Do not confuse inverse penalized curvature with the sampling covariance in Section 7.

A blurred/noisy exposure chiefly contributes to modes it actually measures; it need not set the finest response of the inferred field. It can constrain integrated flux, low surface-brightness extent, colors, time evolution, or calibration. Never homogenize all observations to the poorest resolution as a prerequisite to inference.

New nuisance parameters and uncertain calibration matter. In a locally Gaussian joint scene/calibration problem, the scene's effective precision is the Schur complement

\[
H_{\rm eff}=H_{II}-H_{I\eta}H_{\eta\eta}^{-1}H_{\eta I},
\]

with legitimate calibration information included and singular gauges handled explicitly. Eliminating fresh exposure-specific nuisance parameters yields a PSD Schur increment in a valid Gaussian block. Shared nuisance variables require a joint update rather than independent per-exposure marginalization. Counting conditional information while treating a fitted PSF/background/gain as exact overstates what the new exposure adds.

Model shared calibration and overlap between data products. A hundred exposures with the same flat-field error do not provide a hundred independent calibrations. More low-quality exposures with a coherent unmodeled bias can pull a solution away from truth. Robust contamination models, explicit discrepancy factors and uncertainty may reduce their contribution; arbitrary survey-balancing weights can change the scientific target and should not be disguised as likelihoods.

Diagnose practical degradation separately as response/noise misspecification, changed regularization or insufficient representation capacity, optimization error, or nuisance/incidental-parameter problems in the specified model. Joint fitting is not universally inconsistent and a Laplace volume term is not a general correction for profiling. Preconditioning cannot cure a representation incapable of fitting all valid observations.

Add a cross-exposure tension diagnostic on shared estimable modes, with null-space handling and cross-covariance when estimates share data/calibration. The sum of two inverse Hessians is not generally sufficient. Use reserve measurements/calibrations to determine whether declared response uncertainty is under-dispersed. Route inconsistent observations into a revised response/error model or a documented unresolved state while retaining their evidence. Real temporal change or an incomplete sky model must not be mistaken for a defective exposure.

Preserve every observation and its provenance, including uncertain or currently unusable samples. An archived observation can contribute calibration/history information even when it cannot constrain a celestial pixel. Some measurements may have zero usable sky information or only a bound; no positive weight floor is justified merely by “all data are signal.”

For the special static, adequately modeled, background-dominated imaging case, proper coadds can preserve sufficient statistics; native inference must recover that classical limit before claiming a gain. Neural reconstruction cannot beat an information-preserving estimator solely by changing representation. A win must come from better response/likelihood, validated constraints/regularization, richer data, or approximation/computational advantages. [proper coaddition I](https://arxiv.org/abs/1512.06872), [proper coaddition II](https://arxiv.org/abs/1512.06879), [ZOGY](https://arxiv.org/abs/1601.02655).

Publish an **incremental evidence ledger** for each observing campaign: measurement/footprint/epoch/calibration provenance, changes in identifiable modes and selected query uncertainties, held-out prediction, and science bias/coverage from injection suites. Distinguish quality, depth and resolution. Freeze a best-data reference and test best-plus-older data in multiple orders; failures are response/model investigations, not a reason to hide the observation.

## 10. What a super-resolved view can legitimately claim

Provide at least two clearly labeled products:

- **A data-supported field view:** rendered at a declared target response, with spatial/spectral/time-dependent resolution/support and uncertainty. Fine pixels are an output sampling choice.
- **An optional regularized/learned prediction:** finer or cross-band-inferred structure with alternative interpretations and constraint/model sensitivity. Identify details that are not constrained by measurements. Bayesian predictions can be an explicitly selected instance of this product.

Analyze local linearized response singular modes, frequency support and uncertainty after the declared calibration nuisance treatment. A regularized linear estimator has a resolution operator, such as \((H+P)^{-1}H\); its fine grid is not an identity transfer. Use suitable local/tiled approximations and functional probes rather than allocating a giant dense matrix. A calibrated interval width does not establish support for an unconstrained frequency.

Dithers can recover information lost to detector undersampling when the combined response has the requisite rank. Subpixel localization of an isolated source is different from resolving two arbitrary structures. Optical-transfer zeros, absent radio baselines, temporal gaps and missing spectral channels leave nullspaces. Exact noiseless band-limited recovery has conditions; it is not arbitrary sub-diffraction recovery. [Lauer on dithered reconstruction](https://arxiv.org/abs/astro-ph/9810394), [Drizzle](https://arxiv.org/abs/astro-ph/9808087).

Withhold genuinely sharper observations where possible and predict them. Withholding only broad-beam images cannot validate newly created fine detail. Measure flux, morphology, contrast, ring/filament recovery and transfer functions, including cases where the correct result is “unresolved.” Do not report higher SNR solely because regularization smoothed away uncertainty.

Cross-band priors can improve estimates of correlated structure, but a sharp NIR image does not directly measure an equally sharp X-ray or radio image. Allow spectral/morphological novelty and make unsupported extrapolation visible.

## 11. COSMOS as the flagship, with broader controls

Use COSMOS as a **federated observation campaign**, not one fixed mosaic. The archive spans UV through radio and many instruments. COSMOS2020's independent Classic/Farmer reductions and COSMOS2025/COSMOS-Web provide valuable comparison views, while their masks and selection limit their role as truth. Pin release/version and overlap footprint. The COSMOS-Web release site reports a catalog update to v1.1 on 4 May 2026. [COSMOS archive](https://irsa.ipac.caltech.edu/data/COSMOS/overview.html), [COSMOS2020](https://arxiv.org/abs/2110.13923), [COSMOS2025](https://arxiv.org/abs/2506.03243), [release site](https://cosmos2025.iap.fr/).

**Pilot choice, confirmed 12 October 2026:** query public native HST and JWST observations by footprint across all programs and dates, including visits beyond the original COSMOS and COSMOS-Web releases. Include a ground-based lane, initially CFHT MegaCam, with Subaru HSC and Suprime-Cam inventories alongside it. Archive instrument names and processing levels must be checked rather than inferred from camera nicknames. Begin with a bounded overlap; retain the query needed to expand to the complete campaign. Inventory pagination, public release, footprint overlap, product lineage and calibration history are part of the manifest contract.

Fit single native exposures as independent factors only where their noise/calibration permits. Released stacks provide comparison, initialization and legacy evidence when native inputs cannot be recovered. Do not sum a stack and its constituent exposures as independent information. Raw files/reads remain available for later detector/history studies; the first calibrated-image lane must preserve its processing and radiometric conventions. These choices retain the older and poorer observations without homogenizing every instrument to their resolution.

Start with a bounded overlap region, then scale by scientific stress cases:

1. **Native optical/NIR pilot:** HST ACS, JWST NIRCam, and older Subaru/ground-based exposures where available. Include an unusually complex source, bright stars and faint surroundings, sparse-calibrator detector regions, and realistic empty-looking areas. Fit bands without compulsory common morphology.
2. **Depth and diffuse campaign:** add earlier ground-based epochs, UltraVISTA and IRAC/MIRI data as available. Score faint extended emission and color gradients, not just new galaxy counts. Add scanning/FIR data only with appropriate response/transfer. Include surrounding regions for PSF wings and large-scale background anchors.
3. **Native-event campaign:** Chandra/XMM and GALEX-like UV, with independently validated calibration and no forced counterparts. Joint modeling is accepted only if it improves calibrated endpoints over standalone modal fits.
4. **Visibility campaign:** VLA/ALMA and total-power/short-spacing information where publicly available and relevant. Audit whether the archive exposes original measurements or only derived products. A campaign manifest should record that limitation before promising “all data.”
5. **Twenty-year temporal campaign:** persistent view, reference-epoch views, motion, variability and transient products. Old shallow observations may be uniquely useful here.

Do a metadata/rights/provenance inventory before large downloads. Record raw availability, duplicates, observation times, footprint, processing, calibration versions and available reference data. Use `torchfits` at FITS boundaries and appropriate optional adapters for other native formats; keep acquisition and large campaigns in `torchsky-harness`.

COSMOS cannot establish all-object performance by itself. Add crowded stellar fields, Galactic dust/nebulosity, resolved nearby galaxies, Solar-System/moving-source fields, supernova remnants and radio lobes/filaments as independent controls. Include held-out sky regions and instruments absent from learned-prior training.

## 12. A scorecard that makes “SOTA” falsifiable

Define a claim per task, observation regime and compute budget. No evidence currently establishes one method as best for all modalities and morphologies. Publish preregistered primary endpoints and failures alongside wins.

**Response/calibration:** held-out raw-stamp/native-event prediction; centroid/flux bias; encircled energy including wings; second/fourth moments and spatial correlations; brightness/color/phase dependence; nonlinear/history response; and coverage of calibration intervals. Propagate errors into chosen scientific queries. Do not tune on reserve stars and retain the “reserve” label.

**Background/diffuse preservation:** zero-level ambiguity, signed bias, spatial/spectral transfer, injected extended flux retention, cross-talk with PSF wings/foregrounds, seam amplitudes, and conditional uncertainty. Test real blank fields, crowding, dust and bright stars. Report mode-level failure when absolute levels are not identified.

**Detection:** completeness at matched false-discovery rate, astrometry/flux conditional on selection, calibrated low-count tests, faint/diffuse/irregular/line-only/moving populations, and end-to-end selection functions. Negative-image tests can be useful controls in a symmetric Gaussian setting; they are not a universal null for positive photon counts or asymmetric artifacts. Detection/shape coupling is already known to affect scientific selection, as metadetection demonstrates in its shear regime. [metadetection](https://arxiv.org/abs/1911.02505).

**Segmentation/association:** recovered component/region flux, blend covariance, split/merge calibration, spectral/time-dependent support, scientifically meaningful associations, and expected missed/contaminated flux under declared loss definitions. Hard-mask IoU is a useful synthetic metric only when its “truth” partition has a stated definition. A different legitimate physical grouping is not automatically a wrong segment. Include blends where requested purity and containment are simultaneously infeasible.

**Field and resolution:** blind native-observation predictive likelihood; count/read/visibility residual diagnostics; independent high-resolution prediction; mode transfer/nullspace; constraint/regularization/model sensitivity; flux and morphology errors; predictive and functional uncertainty coverage. Fitting residuals should be interpreted with effective model degrees of freedom and selection, not forced to look white by inflating variance.

**Uncertainty/calibration:** distinguish measurement prediction from latent parameter confidence, marginal from joint-vector coverage, and expected support risk from individual containment. Measure coverage together with width/usefulness in isolated/crowded, SNR, phase/undersampling and background regimes. Compare local conditional errors against a coupled-nuisance reference; report insufficient calibration samples, missing support and infinite/unbounded results. A shift diagnostic or a temperature multiplier alone is not a guarantee.

**Added observations:** paired best-only versus best-plus-older/worse-data campaigns, random and adversarial addition order, calibration shifts, shared errors, unknown history, seeing/depth differences, time variability and spectral mismatch. Require noninferiority for preregistered high-resolution/flux/bias endpoints and demonstrate useful gains where the added data carry information. Set scientific margins before unblinding; do not invent a universal percent threshold.

**Scale:** peak host/device memory, I/O volume, measurement/query throughput, iteration and convergence cost, uncertainty cost, and time to a fixed scientific accuracy. Measure small/medium/large workloads and local-to-global coupling. Report CPU/CUDA/MPS contracts and optional-kernel crossovers. Accuracy and runtime form a Pareto comparison; do not trade unreported bias for throughput.

### Evidence design

Use independent synthetic generators plus paired injections into real native observations. Include flux/contrast/size/crowding/color/phase/time/brightness axes and rare structure. A fit model generating its own injections only tests self-consistency. Inject at the relevant physical stage; post-calibration additions cannot validate raw brighter–fatter/persistence behavior. Use detector simulation or real calibration experiments for those endpoints.

Split full visits/nights/instruments, calibration-star identities/regions and sky fields, not only neighboring pixels. Freeze hyperparameters before blind tests. Exposure holdout measures prediction of the same sky through different measurements; sky-field holdout measures learned-prior generalization. These are different claims.

Make injections an active calibration engine, with distinct development/training, threshold-calibration and final-audit partitions. Model/pipeline selection on the final audit invalidates its independence. Known injected increments certify recovery only for the declared injection population; retain independent generators, uncertain sharp-reference comparisons and mismatch controls. Thousands of correlated pixels are not automatically thousands of independent conformal units. Predeclare the group definitions and sampling unit, and downgrade the claimed guarantee when support or exchangeability assumptions fail.

Declare whether held-out-exposure calibration uses independent calibration pixels/telemetry available at deployment or must be predicted from other exposures. Neither regime may tune on held-out science outcomes. Uncertain reference catalogs or repeated calibration data need dependence modeling.

Use benchmark bootstrap/uncertainty units that respect dependence (sky region, visit, calibration group), and simulation-based calibration/coverage where the assumptions permit it. Validate approximate uncertainty under model mismatch and uncommon sources, not only under the training simulator. Require wins against a correctly configured comparator with equal calibration/data access and a documented budget.

All exhaustive parity, external comparator execution, real-data acquisition, science scorecards and CANFAR campaigns belong in **`torchsky-harness`**. Torchsky retains bounded regression, unit, adjoint, gradient, device and API contracts. FITS implementations belong in **`torchfits`**.

## 13. Current torchsky foundations and immediate gaps

The audit is of the present working tree, not a released-feature scorecard. Detailed references are in the [audit note](../../.cursor/harness/trajectories/2026-10-12-psf-field-audit.md).

Useful existing foundations include `ObservationOperatorBundle`; source-position/wavelength/time-conditioned `ConditionedKernelPSF`; bounded `ExtendedResidualKernelPSF`; signed learned backgrounds; calibration evidence splits and score reports; structured detector-noise models; WCS/pixel/bandpass rendering; spectral field families; and matrix-free linear operators/solvers. The untracked grid inverse and neural PCG work is provisional and must pass its own review before becoming a planning dependency.

The highest-impact implementation gaps are:

1. **Pre-pixel spatial response.** `map/field/rendering.py` samples/integrates the scene in native pixels and then applies a discrete PSF. Existing pixel-response guards are useful, but do not establish the full continuous optical integral needed for precision superresolution.
2. **PSF spatial convention.** The newer source-side conditioned mixture is stronger than older output-side low-rank blending. Unify conservation/support semantics before widening PSF models.
3. **Physical additive placement.** Current field backgrounds add after the PSF. Add photon illumination, dark charge, read offsets and calibrated residual stages with their corresponding noise semantics.
4. **Native calibration likelihood.** Current kernel-based calibration uses normalized, registered stamps. Native star pixels plus flux/centroid/SED/neighbor uncertainty should become the precision lane.
5. **Detector state and native modalities.** Generic learned residuals are not validated brighter–fatter/persistence/pile-up mechanisms. Image-bin adapters do not replace event, timeline or visibility likelihoods.
6. **Joint uncertainty and dynamic field queries.** Conditional covariance estimates at fixed response do not marginalize calibration or certify resolution. Standard fields need an explicit temporal/dynamic contract.
7. **Discovery product semantics.** Geometric catalog indexing/crossmatch is useful groundwork, but does not implement calibrated discovery evidence, segmentation, associations or field-query catalogs.
8. **Query calibration adapters.** Torchregress supplies reusable UQ methods, but astronomy targets, shared nuisance propagation, calibration units, selection/FDR and shift guarantees require explicit adapters and independent evidence.

Preserve current ownership and unrelated changes. Production implementation is now authorized by the user's subsequent request. Keep changes aligned with the accepted goals and current repository ownership; historical benchmark queues and heavy comparator installation in the core environment remain outside that authorization.

## 14. Execution sequence and acceptance gates

Use milestone gates rather than declaring “SOTA” on a calendar. Effort estimates below are planning estimates for a focused small team with instrument/domain support; they exclude uncertain archive acquisition and research iterations.

### M0 — response and evidence contracts

**Status: in progress.** Tested bounded components below have ✓; the full M0 gate remains open. The [API guide](../native-likelihood.md) specifies their supported assumptions.

- ✓ **M0.1 — Gaussian nuisance precision:** matrix-free profiled/calibrated contracts, rank/gauge handling, calibrated mean and determinant correction; [tests](../../tests/test_nuisance_precision.py).
- ✓ **M0.2 — native point-response reference:** continuous pixel-integrated Gaussian, unresampled stamps, signed flux/background profiling, conditional covariance and retained response; [tests](../../tests/test_native_epsf.py).
- ✓ **M0.3 — likelihood and functional evidence:** full normal quadratic, optional Gaussian normalization, lineage/shared-elimination guards and sampling variance/mean transfer; [tests](../../tests/test_gaussian_likelihood.py).
- ✓ **M0.4 — calibrated additive placement:** typed sampled-input/output deterministic planes, standalone and supported field routes, crops/halos, legacy parity and unsupported-path rejection; [tests](../../tests/test_additive_stage_contract.py).
- ✓ **M0.5 — small native queries:** signed forced/blend fits, full small information and sampling covariance, penalty transfer, numerical rank and shared-nuisance contracts without a global scene; [tests](../../tests/test_linear_query.py).
- ✓ **M0.6 — legacy inverse-probe correction:** actual residual checks and signed inverse-curvature diagnostics, singular/inadequate-solve rejection, no positive floor; [tests](../../tests/test_inverse_information_diagonal.py).
- ✓ **M0.7 — detector-grid PSF transport:** image-shaped asymmetric calibration/render convention, boundary losses, source-flux versus flat-sky distinction and coherent explicit inverse adjoints; [tests](../../tests/test_psf_transport_invariants.py), [migration](../psf-response-conventions.md).
- ✓ **M0.8 — evidence/calibration lifecycle:** declared tensor versions and revision tokens guard fixed factors, systems, cached normal evidence and query/functional reports; [tests](../../tests/test_evidence_lifecycle.py). Hidden callback and unversioned state require explicit caller revisions.
- ✓ **M0.9 — native spectral/source quadrature:** source-side measures/weights, continuous-position already integrated response, explicit sparse support, exact discrete transpose and first/second input/response derivatives; [tests](../../tests/test_continuous_response.py). This standalone primitive does not yet replace field rendering or prove physical optics/flat calibration.
- ✓ **M0.10 — linear physical additive/noise reference:** optical photon routing/thinning, dark-charge collection, linear charge mixing, post-charge read terms and full covariance/gain/subtraction; [tests](../../tests/test_detector_response.py). Normalized mixed likelihoods, nonlinear/history and renderer wiring remain open.
- ✓ **M0.11 — joint native Gaussian field binding:** bounded same-band exposure groups, shared nuisance profiling/calibrated covariance and normalization, typed calibrated means, native lineage and dependency guards; [tests](../../tests/test_field_gaussian_likelihood.py). First-order direct-residual and normal gradients agree; survey-wide shared coupling and other native laws remain open.
- ✓ **M0.12 — pixel radiometry/flat transfer:** explicit detector-product scale, units/reference provenance, exact transpose/full covariance and lifecycle; [tests](../../tests/test_pixel_calibration.py). Sky-flat correction retains its point-source transfer; physical flat/color calibration and scale uncertainty remain open.
- ✓ **M0.13 — opt-in source-node native field route:** source sky/wavelength evaluation before transport/calibration; exact sampled-state Gaussian binding, native support/normalization and conditional query evidence; [tests](../../tests/test_native_field_response.py). Controlled/adaptive integration and training defaults remain open.
- ✓ **M0.14 — joint source-node nuisance binding:** one profiled/calibrated nuisance elimination across bounded native responses with different bands, detector charts and product units on a common registered source state; [tests](../../tests/test_joint_native.py). Full shared covariance, support/normalization, gauges, lineage and lifecycle are retained. Fixed additive spans do not provide exact nonlinear calibration propagation or whole-survey coupling.
- ✓ **M0.15 — Poisson–Gaussian zero boundary and CDF tails:** analytic right first/second and mixed zero-rate derivatives, reverse/forward AD and inference semantics, stable complementary tail probabilities/scores; [tests](../../tests/test_poisson_gaussian_boundary.py). Its four small-positive CDF curvature diagnostics are resolved by M0.16; finite-support approximation, higher zero-rate derivatives, coupled detector laws and field-stage wiring remain open.
- ✓ **M0.16 — positive-rate finite-mixture CDF curvature:** mode-scaled detached coefficient recurrences and explicit rate powers fix tiny-rate cancellation without a physical cutoff or changed law. Original strict assertions, additional subnormal rates, mixed Hessian/JVP, heterogeneous/nonzero-lower supports and independent wide-support SciPy values pass in [tests](../../tests/test_poisson_gaussian_boundary.py). M0.17 addresses its separate NLL gap; full physical law and hardware/renderer integration remain open.
- ✓ **M0.17 — bounded positive-rate NLL derivatives:** separate prior/joint modal polynomial sums, standardization, stable anchors and pre-cast support checks; [tests](../../tests/test_poisson_gaussian_positive_nll.py). Independent closed-form/polynomial/SciPy values, mixed Hessians/JVP and range counterexamples pass. Conservative derivative-domain rejection remains, including some finite final derivatives; no universal physical-law or covariance guarantee.
- ✓ **M0.18 — signed detector charts and GWCS wrapper:** preserve reflected physical frames with consistent sampled Jacobian sign, absolute determinant floors and parity-preserving composite calibration; correct optional bounds forwarding. [Geometry tests](../../tests/test_wcs_detector_astrometry.py) and [optional wrapper tests](../../tests/test_wcs_optional_extras.py), plus actual embedded JWST signed sampling, pass. Sampled validation is not proof of global injectivity or independent astrometry.
- ✓ **M0.19 — explicit native physical integrated-read likelihood:** nonnegative source-node illumination through physical photon routing, independent post-optics dark charge, signed read/output offsets and electron/output density normalization; direct-state and scalar-field clients need no global solve. Original evidence/masks and callbacks are guarded, unsupported legacy/mixing contracts reject, and extreme unit conversion avoids premature squares. [Tests](../../tests/test_native_physical.py): 32 passed, 4 device skips; dependency integration: 305 passed, 28 skips. Fixed, independent-pixel finite-support law on bounded tiles; no correlated/history physics, shared-nuisance elimination or calibrated UQ claim.
- ✓ **M0.20 — checkpointed physical-mixture chunks:** opt-in kept-pixel chunking on the direct-state and scalar-field clients preserves the finite law, masks, density normalization and physical-parameter gradients. Ordinary reverse AD recomputes bounded mixture blocks; evidence guards also run during backward. Independent review, saved-tensor/call-size assertions, zero/range limits, mixed first/second AD and Float32 MPS-to-CPU references pass. Source/response and higher-order graphs remain resident; forward AD bypasses checkpointing. Its earlier dependency run exposed the shared-nuisance MPS eigvalsh failure subsequently resolved by M0.21.
- ✓ **M0.21 — compact spectral analysis on MPS:** nuisance Gram spectra/singular inversion and small-query spectral norms/PSD/rank checks use documented compact CPU Torch transfers; science arrays/actions/contractions and native Cholesky solves stay on MPS. Rank policies, roundoff rejection/repair, gauges and calibrated determinant/transfer/covariance retain independent dense/closed-form checks, including first/second AD. Actual Float32 MPS copy-shape and solver-placement tests pass. Final native/nuisance/query integration: **437 passed, 30 device skips**, with fast/type gates passing. This resolves M0.20's nuisance eigvalsh failure and the newly exercised small-query SVD failure; a separate legacy grid-footprint bound failure remains open, as do CUDA and scientific calibration gates.

These foundations do not close controlled/adaptive/default continuous-field rendering, physical flat/color calibration, general correlated photon/charge/read likelihoods and nonlinear/history physics, all legacy-caller lineage/cache propagation, hardware evidence or the native COSMOS science gate. Exact counts and remaining work are in the [status record](field_response_discovery_status.md). After the user's DNS recovery report, bounded CANFAR API/image/tool/mount readiness passed; acquisition planning proceeds in torchsky-harness. Platform readiness does not close scientific gates or authorize an automatic full benchmark suite.

**First 4–6 weeks:** one owner for contracts/calibration plus one for independent harness evidence. Stabilize the exposure and field execution routes, radiometric measures/flat convention, pre-pixel versus ePSF semantics, additive stages, observation processing/history provenance, exact linear adjoint and nonlinear derivative contracts. Specify profiled versus calibrated background treatment, full normal-operator compression, shared nuisance coupling and query/UQ reports. Build independent reference cases and a bounded COSMOS manifest; keep an explicit-grid solver as a statistical reference. Audit provisional inverse modules for duplicate binding/solver policy and ownership before depending on them; preserve unrelated untracked work.

**Gate:** analytic/unit/conservation/adjoint checks pass; quadrature and native response agree with independent reference integrals; known gauges fail closed or are reported; no duplicate measurement likelihoods. Recover single-exposure and proper-coadd limits where their assumptions hold. Include counterexamples distinguishing diagonal information from covariance, conditional from nuisance-aware errors, and regularized curvature from sampling covariance. Standalone exposure tasks must not require a converged field.

### M1 — precision optical/NIR PSF and background

**Next 8–12 weeks:** graduate continuous-phase ePSF/pre-pixel integration, native calibration-star fitting with linear variable projection, centroid/SED handling, spatial/chromatic instrument hierarchy, core/wing support, typed background inference and shared-nuisance query uncertainty for a bounded instrument pair. Add audited scalar/Mondrian and joint-vector torchregress calibration baselines before flows or newer named variants. Use independent exposure/star/region/time calibration units. Compare PSFEx, current Piff, ePSF, MCCD and WaveDiff in suitable tasks; no package must win every task to remain useful.

**Gate:** blind calibration and science-transfer improvement or statistically demonstrated noninferiority against the best relevant comparator; retained diffuse flux and calibrated uncertainties; no gain bought by changing calibration access or the reference target response. If the hybrid loses to the empirical baseline, retain the empirical option and diagnose before adding capacity.

### M2 — field inference and discovery

**Next 8–16 weeks, partially overlapping M1:** fixed-response grid/Foam/INR comparisons; shared-nuisance field/calibration queries; multiscale/open-morphology detection with complete selection/FDR validation; wavelength/epoch-specific supports and expected-loss calibration where feasible; compact catalogs and probabilistic associations. Compare local errors against a coupled reference and validate the field independently of catalogs. Keep development, calibration and blind audit separate. Compare scarlet2, AstroPhot, SourceXtractor++, Tractor, PCAT, NoiseChisel/ProFound and current crowded-field tools in their respective domains.

**Gate:** better completeness at achieved matched false-discovery rate and controlled bias/coverage for declared populations; useful interval widths; retained structures outside catalogs; correct unresolved/ambiguous outputs; survey-scale cost at fixed accuracy. Include calibration-support and model-mismatch failures that must become uninformative or downgrade guarantees. A prettier field or a larger catalog is not an acceptance criterion.

### M3 — one detector family with nonlinear history

**A dedicated 3–6 month lane selected after M1/M2 evidence:** choose a calibrated JWST detector/problem with real read/history evidence; prioritize the effect demonstrably limiting flux/morphology (charge migration, persistence, ramp/read noise or saturation). Fit calibration data separately before joint science refinement. Add a learned residual only after mechanism and state are identifiable enough. Inject at the relevant physical stage; after-readout addition cannot validate charge transport or persistence. Retain missing read/history information as a limitation.

**Gate:** independently calibrated signal/fluence/history transfer and recovered science with correct uncertainty, including extrapolation failures and missing history. Explicitly compare against current mission reduction; do not claim that an old pipeline defect remains current.

### M4 — event, timeline and visibility lanes

**Interface validation until M1/M2 pass:** bounded event, timeline and Fourier/visibility examples prove non-image measurement shapes, likelihoods and adjoint/derivative contracts. Full native domain stacks are outside the first-quarter engineering scope. After the optical/NIR gates, select one staffed modality lane with native data and domain partners: X-ray/UV marked events, bolometer timelines, or radio visibilities plus total power and residual calibration. Begin with standalone inference before cross-wavelength coupling. Reuse domain/inference primitives while retaining different invariants and likelihoods. Set effort estimates from that lane's data/model gaps rather than a universal per-domain schedule.

**Gate:** domain-specific parity and science recovery, joint-calibration coverage, open-morphology tests, and demonstrated useful cross-band gains without compulsory counterparts. Compare X-Sifter/D4PO/field-tension, scan mapmakers, WSClean/CASA and resolve-family methods using pinned configurations. These are harness comparators, not production backends.

### M5 — the federated COSMOS view

**Federate incrementally as validated lanes support the actual archived measurements.** Publish versioned persistent and epoch-specific spatial–spectral field products, resolution/support/uncertainty queries, a compact catalog/association view, and the incremental evidence ledger. Include every available observation through an honest native or documented legacy-product likelihood; disclose inaccessible/lost measurements and current unsupported responses. Preserve inconsistent observations for revisable response/sky modeling rather than silently burying them in a reconstruction.

**Gate:** blind cross-instrument prediction; better selected scientific endpoints than latest-only and classical all-data references; poorer-data noninferiority campaign; complete provenance; broad control fields. Full multi-modal maturity is plausibly an 18–36 month program for a sustained small interdisciplinary team, not a promise for one quarter.

### The first 90 days

Deliver one decisive optical/NIR experiment: a bounded HST/ACS–JWST/NIRCam overlap and a CFHT MegaCam ground-based lane, with Subaru HSC/Suprime-Cam archive queries available for controlled expansion. Use native samples, pre-pixel/chromatic ePSF response, a controlled background nuisance model, independent development/calibration/audit folds, and a best-only versus best-plus-older/poorer-data scorecard with tension diagnostics. Complete standalone instrument checks before joint fitting. Fit both an explicit statistical field baseline and one adaptive/implicit alternative. Include isolated/crowded stars, complex extended or ridge-like emission, bright-star wings, diffuse emission and a temporal-change control. Broader stellar/crowded and diffuse control fields accompany the COSMOS overlap.

Measure flux/astrometric bias, completeness at achieved FDR, diffuse/ridge transfer, identifiable-mode resolution, conditional and joint query coverage, interval usefulness, native holdout prediction and end-to-end time/memory/data passes. Set science budgets and noninferiority margins before the blind audit. Keep event/timeline/visibility work to interface proofs during this experiment.

The immediate result should answer: **does our response/inference system preserve more supported information with less measured bias than the strongest matched baseline?** Only then choose the next model family or nonlinear effect. A focused failure here is more valuable than implementing six modalities with unverified likelihoods.

## 15. Research bets and decisions that remain open

**High-confidence architecture choices:** native likelihoods; non-Bayesian inference by default; instrument-independent sky; independent exposure and field routes; typed response/additive stages; physical radiometry; full information operators where valid; query-specific uncertainty/gauge handling; independent comparator evidence; catalogs as views.

**Promising research bets:** hierarchical physics plus learned residuals; adaptive local spectral/morphological rank; joint scene/calibration inference with scalable functional uncertainty; audited empirical/conformal calibration; confidence-set bounds for selected latent queries; field-tension/component proposals; uncertainty-aware cross-band coupling; nonlinear raw-read inference in selected instruments.

**High-risk bets:** fully implicit response with little calibration; broad modality-to-modality superresolution; universal learned morphology priors; real-time full posterior inference at survey scale; recovery of lost saturated/piled-up/filtered information. Treat these as experiments with explicit failure criteria, not required default behavior.

Choose the exact HST/JWST/ground overlap and calibration folds after the all-program COSMOS inventory, and choose nonlinear mechanisms from measured residuals/calibration evidence. Establish science budgets for flux/astrometry/shape/diffuse transfer and computational budgets before blind testing. Data and calibration diversity may be the limiting resource before network architecture or GPU time.

The durable advantage to pursue is **a sky estimate that can explain, calibrate, and be challenged by every observation—while keeping unsupported detail, ambiguous objects, and missing information visible.**
