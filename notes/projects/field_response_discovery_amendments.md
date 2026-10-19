# Proposed amendments: likelihood-first calibrated fields

Review date: 2026-10-12. This records the evaluation and amendments incorporated into [the revised plan](field_response_discovery_plan.md), revision 2. The review text preserves the reasoning behind that decision. Initial approval covered the plan and overview; the user subsequently authorized production implementation, tracked in the [status record](field_response_discovery_status.md). Local code observations describe the working trees at review time, not released capabilities.

## Recommendation

Keep the field-centered scientific goal and native response graph. Make non-Bayesian, likelihood-based inference the default. Allow exposure/stamp tasks to run independently of a field reconstruction. Preserve an executable likelihood representation, compute uncertainty for requested scientific quantities, and add empirical/conformal calibration where its target and assumptions are defensible. Use torchregress for reusable calibration machinery through an optional adapter.

The governing contract should be **native measurements → calibrated response and likelihood → estimable field/query → uncertainty with declared scope**. Fixed linear-Gaussian sufficient statistics are an important specialization. They are not the universal engine for Poisson events, nonlinear detectors, changing calibration, or arbitrary structured noise.

Full-field sampling is unnecessary for the default product. Neither deterministic inference nor conformal calibration automatically makes a reconstruction unbiased, prior-free in its effective assumptions, or identifiable below the instrument's supported resolution.

## Evaluation of the architectural review

**Accept:** independently usable native exposure operators; native-pixel calibration; explicit radiometry and response stages; matrix-free nuisance elimination; a poorer-data tension experiment; a narrower optical/NIR implementation scope; an ownership audit of provisional solvers.

The field should be the common scientific representation, while PSF calibration, background fitting, detection, forced photometry and difference-image queries remain useful without converging a global field. These two routes must share units, likelihoods and calibration provenance.

**Refine the PSF normalization argument.** For a dimensionless equal-cell transport matrix, column sums of one conserve point-source flux, while row sums of one preserve a constant vector. Both require a doubly stochastic matrix. A spatially varying source-conditioned PSF generally supplies only the first property. However, a constant vector of integrated fluxes is not necessarily a constant radiance: source solid angles, detector footprints, throughput, masks, and flat calibration enter the physical test. Do not force both sums to one by numerical renormalization. Test the declared response to a point measure and to a uniform radiance separately, with the correct integration measures and boundary losses.

Record the actual flat calibration convention, spectral illumination, pixel-area treatment, throughput and point-source correction. A single sky-flat/point-source flag is insufficient for every instrument. This extends the plan's existing conservation warning into an explicit radiometric contract.

**Accept direct continuous-phase point response, with a qualified accuracy claim.** An ePSF can evaluate native star pixels without rasterizing a delta source or interpolating noisy observations. Its accuracy still depends on its definition, intra-pixel response, wavelength, centroid, finite support and interpolation/quadrature. An ePSF already integrating detector pixels must not undergo pixel integration again. Extended emission needs a controlled integration error budget; neither universal 2× oversampling nor universal 16–256× cost follows from the physics.

Point measures can be internal computational components or optional catalog anchors. They must not become a compulsory catalog scaffold for diffuse, filamentary, moving, blended or unfamiliar objects.

**Correct two proposed remedies.** A zero Poisson null rate does not justify inventing a background floor. Under a zero-rate null, zero observed counts have a valid likelihood, and positive counts have zero null probability. Implement the support limits explicitly; use `log1p` where the positive-rate ratio is valid. Also, preconditioning improves numerical convergence but cannot repair a finite-capacity field's approximation bias. Test representation capacity separately from optimization error.

## Evaluation of the non-Bayesian review

**Accept the default inference choice and the computational concern.** Survey products should not require global MCMC/VI or a stochastic inverse-diagonal map. Profile linear nuisance blocks, use scalable deterministic solvers, and spend uncertainty computation on science queries. Retain Bayesian methods as optional, bounded alternatives when their assumptions and cost are useful.

Several supporting claims in the review are too strong:

1. **Full normal operators matter.** For fixed linear Gaussian data, `(b, H)` captures the likelihood's scene-dependent quadratic. `(b, diag(H))` generally does not. The dirty map has expectation `H I`, rather than `I`. A proper coadd is sufficient under its particular imaging assumptions, not for arbitrary spatial/chromatic response and detector history. [Proper coaddition II](https://arxiv.org/abs/1512.06879).
2. **Information is not variance.** `1 / diag(H)` is a conditional sensitivity proxy in a coupled system. It is generally different from `diag(H⁻¹)`. Local source-block inversion likewise conditions on everything omitted; shared calibration, background and neighboring field modes can materially enlarge uncertainty.
3. **Hutchinson probing is not intrinsically Bayesian.** It estimates the diagonal of an inverse operator, which also arises in frequentist covariance calculations. The current helper performs one solve per probe, but neither a universal 32× wall-time factor nor 18% relative pixel error follows from 32 probes. For independent real Rademacher probes and a symmetric matrix `M`, the estimator's per-pixel variance is `sum_{j≠i} M_ij² / N`. A diagonal `M` is recovered exactly with one probe. Solver error and final clipping introduce additional considerations.
4. **Positivity and regularization bias cross statistical schools.** The Gaussian-lognormal identity compares an intensity mean with the exponential of a log-intensity mean; it does not by itself prove frequentist bias against the true sky. Nonnegative least squares also has boundary bias. A penalty or restricted learned basis can bias a deterministic estimate. The transfer operator must be reported regardless of whether the penalty is called a prior.
5. **Joint optimization is not universally inconsistent.** Incidental-parameter problems require an actual asymptotic model. Profiling is a legitimate frequentist operation. A Laplace volume term belongs to an integration approximation, rather than being a universal correction required for profiling. In a fixed linear Gaussian nuisance block, its determinant is independent of the scene and does not change the scene optimum.
6. **Analytic errors do not always underestimate uncertainty.** Model misspecification can move uncertainty in either direction. Robust/sandwich covariance, independent recovery tests and calibration help diagnose it, subject to their own assumptions.

Reject categorical runtime multipliers and universal claims that one statistical family necessarily creates halos or that another guarantees unbiased superresolution. Compare methods at fixed data access, scientific error and computational budget.

## Evaluation of the conformal review

**Accept conformal calibration as a useful layer. Reject the claim that it universally replaces inverse-problem uncertainty at constant cost.** It calibrates a defined prediction set or risk under defined sampling conditions. It does not restore lost modes or correct a biased point estimate. Calibration, vector scoring, density inversion, candidate selection and null-tail estimation all have costs.

**Observed measurements are useful labels, but not latent truth.** Held-out star stamps and exposures supply noisy photon/read observations. They do not directly supply noiseless PSF encircled energy, a pure background image, an uncontaminated deblended flux, or the true high-resolution sky. Predictive measurement-space coverage and parameter/field confidence must have separate names and evidence.

**Spatial localization does not remove exchangeability assumptions.** Fixed Mondrian groups can give within-group marginal coverage when calibration and test units are exchangeable within each group. Arbitrary local weights do not establish conditional coverage at every focal-plane position. Covariate-shift weighting needs appropriate density ratios, support overlap and an invariant conditional target law; estimated ratios add uncertainty. General drift results have a coverage penalty involving distributional discrepancy, which is not automatically known. [Weighted conformal prediction](https://arxiv.org/abs/1904.06019), [beyond exchangeability](https://arxiv.org/abs/2202.13415), [conditional coverage limits](https://arxiv.org/abs/1903.04684).

**Detection needs valid nulls and a complete selection contract.** Blank sky can contain unresolved emission. Sign flipping requires a symmetric null, and phase randomization need not preserve the relevant null distribution. Searching for peaks before testing changes the score distribution. AdaDetect has a specific conditional-null exchangeability and score-symmetry construction supporting its FDR theorem; arbitrary neural scores plus shared nulls plus BH do not inherit it. Use the proven procedure when its assumptions apply, or a dependence-appropriate procedure with valid selected-hypothesis p-values. Compare completeness empirically at the achieved FDR. The rank formula also has minimum p-value `1/(m+1)`, a serious null-sample budget constraint for survey searches. [AdaDetect](https://arxiv.org/abs/2208.06685).

**CRC controls expected task risk, not each source's containment.** Its basic guarantee uses exchangeable calibration/test loss functions, a bounded monotone loss and a finite-sample correction. Missed-flux fraction can be monotone as support expands; contamination fraction need not be. Inner/outer supports are useful exports, but purity and containment require separate valid losses and a feasible tradeoff. An unresolved blend may have no support satisfying both targets. Report that ambiguity instead of promising two guaranteed masks. [Conformal Risk Control](https://arxiv.org/abs/2208.02814).

**Vector calibration is appropriate for declared estimable queries.** A Fisher-shaped region can be conformalized without a Gaussian residual law if calibration labels and sampling assumptions are suitable. Its orientation need not be the true error geometry; better geometry improves efficiency. Singular information must leave unmeasured directions unbounded or remove them from the declared query, rather than gaining artificial certainty through a numerical floor. Coordinate-wise coverage differs from joint vector coverage. Distributional conformal can use asymmetric predictive models, but it does not establish the entire conditional CDF exactly under arbitrary misspecification. [Distributional conformal prediction](https://arxiv.org/abs/1909.07889).

**Promote injections to active calibration with independent audit.** Known injected increments give recovery targets for the injected population. They do not reveal the unknown real scene or guarantee transfer to unseen morphology, detector errors or crowding. Sharp reference observations are noisy, incomplete and spectrally different proxies. Keep generator development, threshold calibration and blind auditing separate; include mismatch generators and reference uncertainty. The CosmoStat weak-lensing study supports the value of calibration while also discussing population-dependent coverage limitations. [Leterme, Fadili and Starck](https://arxiv.org/abs/2410.08831).

## Proposed replacement for Sections 2 and 7

> Torchsky represents incident sky radiation and predicts native observations through calibrated physical response graphs. Its default inference estimates fields and requested scientific functionals using likelihoods or justified estimating equations, explicit physical constraints, and declared regularization when needed. A posterior distribution is an optional interpretation or downstream method, rather than the defining product.
>
> The primary reusable evidence is native likelihood factors, calibration/provenance and executable response operators. Fixed linear-Gaussian blocks additionally support exact quadratic compression. Every reconstructed view declares its gauge, data-supported modes, transfer/resolution, approximation error and assumptions. Catalogs and segments are query products; a catalog is not required to fit or validate a field.

For a fixed linear-Gaussian block, let `W` be the declared precision after valid nuisance treatment and `z` the data centered by the calibrated nuisance mean. Then

\[
b=\sum_e A_e^T W_e z_e,\qquad
H=\sum_e A_e^T W_e A_e,
\qquad
-2\ell(I)=I^T H I-2I^T b+c.
\]

Retain the full action of `H`, through matrix-free factors or controlled sparse/structured compression, rather than only its diagonal. Constants, noise normalization and native data remain necessary for absolute predictive diagnostics and some model comparisons. Cross-observation dependence requires a joint factor; shared calibration cannot be integrated independently once per exposure and then counted repeatedly.

Accumulating `b` can stream. Applying a matrix-free `H` may revisit exposures or caches, and inversion can require iterations. One-pass accumulation is not one-pass unbiased reconstruction. Calibration or scene-dependent covariance changes invalidate the fixed compression. Poisson, censored, read-history and visibility objectives retain their appropriate likelihoods.

Provide two composable execution routes:

- **Exposure route:** standalone native PSF/background calibration, matched-filter or event detection, forced photometry and transient queries.
- **Field route:** tiled/adaptive joint reconstruction, optional point measures, continuous/diffuse emission, spectral/time variation, and field-derived discovery.

Use variable projection where nuisance coefficients are actually linear under the likelihood. Flux and local additive coefficients have a closed-form Gaussian least-squares step; centroid, color and PSF parameters usually require optimization. External astrometry enters as a calibrated measurement constraint, with proper motion, epoch and uncertainty, without requiring Bayesian framing.

Keep matrix-free global solves available for field reconstruction and selected coupled queries. Make global inverse-diagonal maps optional, with explicit accuracy/cost budgets. Neural models may initialize, propose, precondition or represent the field; validate the approximation bias and uncommon-object behavior of each use.

## Proposed replacement for Section 5: nuisance treatments

Support two explicitly different Gaussian contracts for `y = A I + B beta + noise`:

1. **Fixed unknown nuisance, profiled:** with `W₀=C⁻¹`, use
   \[
   W_{\rm prof}=W_0-W_0B(B^TW_0B)^\dagger B^TW_0.
   \]
   The corresponding sky/background modes may be unidentifiable. A chosen gauge does not restore their information.
2. **Externally calibrated random-effect uncertainty:** if `beta` has a justified Gaussian calibration mean/covariance independent of the declared noise, center by `B beta_bar` and use
   \[
   W_{\rm cal}=(C+B\Sigma_\beta B^T)^{-1}
   =W_0-W_0B(\Sigma_\beta^{-1}+B^TW_0B)^{-1}B^TW_0.
   \]
   This is a matrix-free Woodbury computation, not a compulsory sky prior. A deterministic penalty on `beta` can yield the same quadratic algebra but does not automatically have the same uncertainty interpretation.

Use factorization/linear solves and rank-revealing methods rather than explicit inverses. Cache the small factor only while its basis, covariance, mask and calibration are fixed. Per-application work scales with the basis size, but factor construction and coupled calibration have additional costs. Do not prescribe 10–50 modes universally: amplifier boundaries, fringes and temporal structure may require different sparse or hierarchical bases. Parameter-dependent covariance requires its likelihood normalization when fitting those parameters.

## Proposed new subsection: query uncertainty and calibration

For a linear functional `gᵀ I`, the unregularized linear estimator's fixed-model sampling variance involves `gᵀ H⁻¹ g` on the estimable, gauge-fixed subspace. A selected adjoint solve can compute it without an all-sky variance map. For nonlinear source/blend queries, use local curvature with shared nuisance treatment, and validate linearization.

For a quadratic penalty `P` and estimator `(H+P)⁻¹ b`, the mean transfer is `R=(H+P)⁻¹ H` and its sampling covariance is `(H+P)⁻¹ H (H+P)⁻¹` for symmetric operators. The inverse penalized curvature `(H+P)⁻¹` is not that sampling covariance. Variance alone also omits the regularization bias `(R−Id) I`. Keep these quantities distinct when calibrating scientific queries or comparing a regularized display with an unbiased estimable functional.

With source/query parameters `theta` and omitted nuisance `eta`, the relevant Gaussian local precision is

\[
F_{\rm eff}=F_{\theta\theta}
-F_{\theta\eta}F_{\eta\eta}^{-1}F_{\eta\theta},
\]

where the nuisance block includes its legitimate calibration information. A local inverse is exact only for the declared closed model. For wide PSF wings, diffuse backgrounds or global calibration, use shared low-rank coupling, boundary-response operators or budgeted larger solves. Label a conditional approximation as conditional and validate it against a coupled reference.

Choose the UQ method by target:

- **Future native measurements:** predictive intervals/regions including observation noise, held out in independent or justified exchangeable blocks.
- **Latent photometry and field functionals:** analytic/profile uncertainty, robust covariance or block resampling where justified; injection/reference-calibrated regions only for the declared target population.
- **Discovery:** selected-hypothesis testing and FDR procedures with valid null/selection/dependence assumptions.
- **Supports/segmentation:** calibrated expected losses or empirical quality metrics tied to an explicit component/flux definition.
- **Unidentified modes:** wide/unbounded intervals, alternative interpretations or an explicit unresolved result.

A complementary non-Bayesian research lane is confidence-set inversion in measurement space. Construct a jointly valid acceptance set for sky and calibration parameters under the declared noise/response uncertainty; minimize/maximize a requested field functional over it. This can bound a latent quantity without observing latent labels. It is model-based frequentist UQ, not assumption-free inference, and may be costly or unbounded. Strict-bound inverse-problem methods are relevant precedents; their monotonicity/convexity constraints must not be assumed for arbitrary skies. [Kuusela and Stark](https://arxiv.org/abs/1512.00905).

Attach a compact, linked uncertainty report to queries: estimand and units; confidence versus prediction versus expected-risk target; method and nominal level; conditional/marginal/joint scope; calibration revision and unit; group/support/sample size and weight diagnostics; nuisance and regularization treatment; selection rule; assumptions; independent coverage/width results; and extrapolation status. Keep this metadata and covariance/support products outside the default lean object table.

## Torchregress reuse and boundaries

The current sibling working tree already contains relevant machinery:

- `SplitConformal`, `CQR`, `DistributionalConformal`, grouped calibration and weighted thresholds in [`losses/conformal.py`](../../../torchregress/src/torchregress/losses/conformal.py). The common predictor accepts `groups=`, `weights=` and test weights. Finite-sample thresholds can be infinite; preserve that result.
- `MultivariateScoreConformal` in the same file uses a scalar Mahalanobis or Gaussian-NLL score with full/shared/diagonal covariance inputs and joint-region membership. This is directly relevant to small multiband/blend query vectors. It does not currently accept Mondrian groups; an astronomy adapter can dispatch separate fitted calibrators for fixed groups. Use squared-score threshold semantics correctly. Treat covariance as a shape model, not proof of a Gaussian error law.
- `MultiTargetConformal` supplies coordinate-wise intervals; those are not interchangeable with a joint vector region.
- [`VarianceTemperatureScaler`](../../../torchregress/src/torchregress/calibration/posthoc.py) fits variance scaling and an optional floor, and can account for noisy reference labels. `PITCalibrator` and calibration metrics support distribution diagnostics. These transforms do not carry conformal guarantees merely because they are calibration tools.
- [`PredictiveBatch`](../../../torchregress/src/torchregress/prediction.py) is a useful point/quantile/distribution interchange container. Astronomy-specific resolution, provenance and selection semantics still belong in torchsky.
- Weighted/non-exchangeable and shift utilities exist, but require a method-specific assumption audit. The OT module describes a score-CDF matching surrogate, not a general exact transport theorem.

Relevant code tests exist, including [`test_multivariate_score_cp.py`](../../../torchregress/tests/test_multivariate_score_cp.py), [`test_nexc_conformal.py`](../../../torchregress/tests/test_nexc_conformal.py) and calibration edge-case tests. They were inspected for this review; the sibling test suite was not run, and their existence is not astronomy transfer evidence.

**Specific audit item:** `NonExchangeableConformalRegressor.two_sided_coverage_bounds()` computes bounds from a calibration-weight discrepancy against uniform weights. That statistic alone cannot upper-bound arbitrary population score drift. Uniform weights give zero discrepancy even if every future residual shifts outside the interval. The theorem's distributional discrepancy needs a defensible connection to the application; do not advertise this helper's output as certified coverage for evolving instruments/skies without resolving that connection. Invalid-weight fallback to uniform also cannot preserve a covariate-shift guarantee just by warning. This review proposes an audit, rather than changing sibling code.

No dedicated AdaDetect/FDR or CRC implementation was found by the targeted source/docs search. These are proposed additions, not existing torchregress capabilities. Prefer generic calibration methods in torchregress; keep astronomy null construction, native injections, response coupling and query semantics in torchsky and its harness.

Make torchregress an optional calibration adapter/extra or harness dependency. It currently brings SciPy and TorchMetrics as dependencies; importing it should not silently widen torchsky's lean default runtime. Pin an audited revision and supported device/dtype paths. Do not duplicate its general-purpose calibrators in torchsky.

## Proposed edits to Sections 8–10

For Gaussian discovery, write the nuisance-adjusted score explicitly. With candidate response `v=Aq`, null residual `r`, and fitted null-mode Jacobian `J`, use the appropriate projected precision

\[
W_\perp=W-WJ(J^TWJ)^\dagger J^TW,
\qquad
s=\frac{v^TW_\perp r}{\sqrt{v^TW_\perp v}}.
\]

In the fixed linear Gaussian null this gives the efficient score normalization. Estimated covariance, nonlinear fitting, constraints and a searched candidate require additional calibration. If the projected candidate has zero information, report the hypothesis as unidentifiable. The unprojected denominator on fitted residuals can make a score conservative; simply calling every mismatch anti-conservative would also be wrong.

Expand beyond source templates through multiscale field functionals, wavelets, ridges, coherent spectral/time changes and optional learned proposals. Calibrate the complete selection pipeline. Conformal calibration is an option, not the only discovery method.

Segments remain wavelength/time/query-dependent interpretations with ambiguous blend groups allowed. A catalog entry records a measured entity or hypothesis and links to evidence, rather than requiring a Bayesian existence probability. Keep flux vectors, support alternatives and uncertainty products in linked relations.

For adding poorer data, retain the fixed-model PSD information theorem and state its conditions. Fresh exposure-specific nuisance elimination gives a PSD Schur increment in a valid Gaussian block. Shared nuisance parameters need a joint update; the result is not an independent sum of per-exposure marginalized increments.

Test four distinct causes of practical degradation: response/noise misspecification; changed regularization or incomplete field representation; optimization error; and genuine nuisance/incidental-parameter problems in the specified model. None follows merely from a new exposure having poor seeing.

Compare old/new data predictions on shared estimable modes. A tension statistic needs null-space handling and cross-covariance when estimates share data/calibration; the sum of two inverse Hessians is not generally sufficient. Use reserve observations/calibrations to diagnose the mismatch. Route an inconsistent exposure into a revised uncertainty/response model or a documented unresolved state, retaining its evidence. Do not automatically reject an exposure because it contains a real transient or because it initially disagrees with an incomplete sky model.

Rename the superresolution products to a **data-supported reconstruction** and an **optional regularized/learned prediction**. Report the resolution/transfer of both. Permit signed unbiased measurement estimators even though underlying physical total intensity is nonnegative. Increasing data should expand supported detail where identifiable; a conformal width cannot establish support for an unconstrained spatial frequency.

## Proposed edits to Sections 12–14 and the first 90 days

**M0: contracts and bounded solver audit.** Add the two execution routes, radiometric measures/flat convention, pre-pixel versus ePSF semantics, profiled versus calibrated background contracts, full normal-operator compression, shared nuisance treatment, and a query/UQ report. Audit provisional inverse modules for ownership and duplicate binding/solver policy before making them a dependency. Refactor only the parts needed by authorized implementation; do not prune unrelated untracked work. Require diagonal-versus-full information counterexamples and null-space cases in the bounded mathematical contracts.

**M1: precision optical/NIR and calibration baseline.** Native star pixels; analytic Gaussian flux/background profiling; continuous centroid/SED handling; wings and diffuse-sky retention; two known instruments. Add simple scalar/Mondrian and joint-vector torchregress baselines before experimenting with flows or newer named CP variants. Split by meaningful exposures, star groups, detector/time blocks and independent fields; thousands of correlated pixels do not automatically count as thousands of independent calibration units.

**M2: field queries and complete discovery calibration.** Explicit-field baseline versus adaptive/implicit alternatives, shared-nuisance query uncertainty, complete candidate-selection/FDR validation, expected-loss supports where feasible, compact catalog relations and catalog-independent field validation. Compare conditional local error bars with a coupled reference. Use separate training/development, calibration and final-audit partitions; tuning against the audit invalidates its role.

**M3: one measured nonlinear/history limitation, after M1/M2 evidence.** Keep the nonlinear detector ambition and choose it from residual/calibration evidence. Direct injections must occur at the correct physical stage: after-readout addition cannot validate charge transport, persistence or photon counting statistics. Missing read/history information remains an explicit limitation.

**M4: interface proofs initially.** Before M1/M2 pass, limit events, timelines and visibilities to bounded protocol/adjoint/likelihood examples. Do not schedule three full domain stacks into the first quarter. After the optical/NIR gate, choose a single staffed modality lane with native data and appropriate domain partners. The architecture must remain non-image-capable throughout.

**M5: incremental federation.** Retain the long-term all-available-data COSMOS ambition, with versioned response evidence and epoch-dependent skies. Incorporate old/poor-quality data at native support and measured information content; do not homogenize sharp data to the worst PSF. Preserve unsupported or inconsistent measurements for revisable modeling rather than silently burying them in a reconstruction.

The first 90 days should deliver a bounded optical/NIR COSMOS comparison: strongest matched baseline versus torchsky; best data alone versus best plus older/poorer data; an explicit field versus one implicit/adaptive alternative; isolated and crowded stars, extended/ridge-like emission, bright-star wings, a diffuse component and a temporal-change control. Broader control fields must include stellar/crowded and diffuse regimes; COSMOS alone cannot validate all astronomical objects.

Measure photometric/astrometric bias, completeness at achieved FDR, diffuse/ridge transfer, identifiable-mode resolution, conditional and joint query coverage, interval width/usefulness, native holdout prediction, and end-to-end time/memory/data passes. Include mismatch cases where calibration must fail or become uninformative. Establish noninferiority margins and science budgets before the blind audit. Failure of a coverage assumption must downgrade the claimed guarantee, rather than only trigger a larger multiplier.

## Evidence recorded for this proposal

Reviewed the original plan, current solver helper and relevant sibling UQ source/docs/tests; verified primary conformal and inverse-UQ references. Ran a bounded Pixi analytical check in [review-uq-identities.py](../../.cursor/harness/trajectories/2026-10-12-review-uq-identities.py). It checks loss from diagonal compression, conditional versus nuisance-aware variance, exact Rademacher diagonal-estimator variance, and failure of weight-only drift certification. These are counterexamples to general claims, not scientific performance benchmarks. The initial review made no production, plan or sibling changes. After acceptance, its amendments were incorporated into the plan and interactive overview. Subsequent production work is tracked in the implementation status; sibling repositories and comparator environments were not changed by the review.
