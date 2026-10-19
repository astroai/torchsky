# Native likelihood and calibration foundations

These primitives support exposure tasks without a global field and retain full
information for field queries. They are bounded, tested components; they do not
establish SOTA, full raw detector physics, or calibrated real-sky coverage. The
status section at the end of this page records what is implemented and which
gates remain open.

## Native star pixels

`torchsky.map.operators.native_pixel_stamp` slices existing detector pixels and records their zero-based pixel centers. It does not interpolate noisy measurements, normalize an aperture, pad missing detector edges or require an object catalog. Inputs are one independent stamp, its fixed diagonal Gaussian variance and an optional keep mask. Float32/float64 inputs must share dtype/device; masked measurements can be nonfinite while retained measurements must be valid.

`PixelIntegratedResponse` callables evaluate per-native-pixel response at continuous `(pixel_center - source_center)` offsets. The response already contains pixel integration. Never integrate it again or force a cropped footprint to sum to one. `GaussianPixelResponse` is an exact rectangle-integrated axis-aligned Gaussian reference with uniform intra-pixel sensitivity and infinite full-domain unit normalization. It intentionally does not model color, charge transport, history or arbitrary optics.

`profile_native_star(stamp, response, centroid_xy, background_basis=...)` fits signed stellar flux and linear additive coefficients with weighted QR. The default additive basis is a constant; an explicit empty basis fits flux alone. The centroid and response remain differentiable nonlinear inputs. Numerical deficient rank raises. The result reports conditional coefficient covariance, conditional degrees of freedom and both footprint/kept response fractions. Those errors condition on the supplied response, position, variance and neighbor treatment; they are not joint PSF/centroid uncertainty or conformal coverage.

The response protocol admits richer Torch callables. It does not certify their flux normalization or calibration. Parameter graphs remain connected; rebuild a response from current parameters inside each optimization step. Shared-star, chromatic, wings, neighbor and real-reference validation remain subsequent work.

For the existing discrete-grid models, kernels have the orientation of their unit-source image stamp. The renderer performs the `conv2d` correlation-to-convolution conversion. See [PSF response conventions](psf-response-conventions.md) for source-flux versus uniform-sky behavior, detector boundaries and migration of old asymmetric arrays/checkpoints.

## Background and other linear nuisances

`ProfiledNuisancePrecision(B, apply_precision=W0, ...)` applies

\[
W_0-W_0B(B^TW_0B)^\dagger B^TW_0.
\]

Its small numerical-rank policy is explicit and scale-sensitive; no jitter restores removed information. Keep the resulting sky/background gauges unresolved. Rank checks in the nuisance span cannot prove that a general callback is positive semidefinite everywhere.

`CalibratedNuisancePrecision(B, apply_precision=W0, covariance_factor=L, mean=...)` represents externally calibrated Gaussian uncertainty with covariance `L Lᵀ`. It uses a small factor of `Id + (B L)ᵀ W0 (B L)` rather than a dense pixel covariance or an inverse calibration covariance. This handles singular or zero calibration covariance. Call `center(data)` before using its precision for likelihood assembly. `logdet_covariance_correction` supplies the relative determinant term; the base covariance normalization stays the caller's responsibility.

Both primitives accept arbitrary measurement shapes and real float32/float64 data; MPS requires float32. The explicit `spectral_device` policy uses CPU on MPS, and the data device on CPU/CUDA. On MPS, only the small nuisance Gram matrix and scalar roundoff bound move to CPU for eigenvalue/rank checks and singular Hermitian inversion. The resulting pseudoinverse returns through a differentiable Torch transfer. Measurement arrays, base precision actions, contractions, Cholesky factors/solves, determinant correction and outputs remain on MPS. This does not enable global MPS CPU fallback, add jitter or relax the rank/PSD contract. It makes no speed claim. Fixed-rank first/second gradients are checked against independent dense/closed-form references, including actual Float32 MPS execution; unexecuted CUDA cases remain skips.

Inputs, callbacks and calibration are fixed for each constructed precision. Rebuild after changing them. Crossing rank boundaries is not a smooth operation. These bounded numerical/device checks do not calibrate an instrument's actual noise or establish scientific coverage.

For correlated noise, the callback must already be the precision of the observed support. Masking both sides of a full-data precision is generally a conditional/weighted restriction, not marginalization over missing measurements. These objects do not infer a noise model from labels or equate a deterministic penalty with calibrated random-effect uncertainty.

## Full likelihood evidence and queries

`torchsky.map.likelihood.GaussianLikelihoodFactor` wraps an existing `BoundObservationOperator`, centered native data and explicit measurement/calibration/provenance identifiers. Native measurement identifiers must refer to disjoint sample groups; derived products include their constituent groups. The caller owns truthful lineage. A `GaussianLikelihoodSystem` rejects repeated groups and separately eliminated shared nuisance identifiers. Shared uncertain calibration/background terms require a single joint factor, or explicit coupled treatment outside independent summation.

The operator may return vectors, timelines or complex arrays; it need not return a 2D image. Response/adjoint and precision callbacks must satisfy their declared fixed linear contracts. This does not implement radio RIME, photon events or nonlinear reads merely by accepting their tensor shapes. Real/complex noise conventions and units belong to the operator and density normalization.

`normal_equations()` preserves `b`, full matrix-free `H`, `c` and measurement lineage for

\[
Q(x)=x^THx-2x^Tb+c.
\]

A diagonal of `H` is not substituted for this information. Statistics have a fixed-model lifetime; rebuild them after calibration, covariance, data or response changes. Native factors remain available for direct residual evaluation. Neither arbitrary nonlinear nor Poisson likelihoods admit this Gaussian compression.

Supply a real scalar tensor `log_normalization` to each conditional/calibrated factor for absolute Gaussian density evaluation. In real coordinates it is half of `n log(2 pi) + logdet(C)` on observed support. `negative_log_likelihood()` rejects missing constants and profiled nuisance objectives as normalized predictive densities. Fitting covariance without this term permits artificially inflated noise.

`solve()` uses the existing matrix-free CG engine and returns its diagnostics; its result does not itself certify convergence. `functional_uncertainty(g, damping=d, ...)` solves `K u=g`, where `K=H+d Id`, checks the actual residual and returns

\[
\operatorname{Var}(g^T\hat x)=u^THu,
\qquad R^Tg=Hu.
\]

For an unregularized estimable query this agrees with `gᵀ H⁻¹ g`. For ridge regularization it is the sampling variance rather than inverse penalized curvature. The difference of `mean_response` from `g` describes possible bias; a small variance or successful residual check does not bound that bias or establish identifiability of all sky modes. Failed/inconsistent query solves raise. Singular/ill-conditioned modes need explicit estimability and further diagnostics; residual tolerance alone does not bound forward error.

This is fixed-model analytic UQ. It supplies no automatic misspecification, selection, conformal or distribution-shift guarantee. Those require the separate calibration and scientific evidence lanes in the plan. Source/blend queries must include the nuisance and field coupling they intend to measure.

## Evidence and calibration lifetime

`torchsky.map.evidence.EvidenceSnapshot.capture({"response": A, "noise": C, ...}, revision=...)` declares the dependencies of a fixed likelihood. Ordinary Torch in-place writes invalidate the snapshot, including writes through shared views. The optional revision getter returns an immutable scalar or tuple and must change for mutable metadata, replaced tensor bindings, hidden callback state, or writes that bypass Torch mutation versions. Inference tensors require an explicit caller-maintained revision. Identifiers and content hashes alone do not establish calibration validity.

Factors automatically track data and normalization tensors and reject replacement of their bound callbacks, operator bundle or target. Pass `dependencies=snapshot` to cover precision/response matrices, module parameters, masks, nuisance calibration and other captured state. Arbitrary callback closures are not inspected. Rebuild factors and derived statistics after changing dependencies; in optimization, rebuild parameter-dependent graphs for each step.

Systems check evidence before and after callback evaluation. Normal-equation outputs, small-query numeric reports and functional-uncertainty reports reject stale evidence on public reads. Query reports additionally track supplied templates, offsets, penalties or the functional direction; `dependencies` covers their hidden state. A tensor already retrieved from a report remains an ordinary tensor whose later use cannot be intercepted. These guards scan declared versions/metadata without copying or hashing science arrays; they do not prove that an undeclared callback implements its claimed physics.

## Small forced-photometry and blend queries

`torchsky.map.query.fit_linear_query(system, templates, ...)` fits real, signed coefficients directly to native measurement-space templates. Each factor has a tensor shaped `(n_parameters, *factor.data.shape)`; each template includes that observation's response. Optional `offset_predictions` fix the complementary scene, after calibrated mean centering. Complex measurements/templates use the factor's declared real-coordinate precision convention. No global field representation, scene callback or inverse-field solve is required.

The query accumulates the full small information matrix and right-hand side, then uses Cholesky. It retains blend and band coupling, applies the same profiled/calibrated nuisance precision as the likelihood factors, and rejects deficient systems without adding jitter. Shared nuisances must still be handled jointly. Templates and complementary-scene estimates are conditional inputs; omitted neighboring or calibration uncertainty does not enter the reported errors. A Jacobian of a nonlinear model gives a local approximation rather than exact nonlinear coverage.

An optional explicit PSD `penalty_precision=P` produces coefficient transfer `(F+P)⁻¹F` and sampling covariance `(F+P)⁻¹F(F+P)⁻¹`, alongside the numerical data rank, information eigenvalues and rank cutoff. A finite regularized covariance in a data-null mode is not a confidence bound on its true value. Numerical rank is scale-dependent; rescale heterogeneous coefficient units deliberately and report the chosen tolerances. Rank tolerances do not relax symmetry/PSD validation: only negative eigenvalues within the separate floating-point allowance are repaired to zero. The API imposes no source positivity or morphology constraint.

Per factor, a bounded query uses `p+1` precision actions, `O(N p²)` contractions and an `O(p³)` small factorization. The supplied precision callback may have additional or superlinear costs. Supply local stamps or bounded groups, rather than dense templates for the whole survey. This algorithmic structure is not a measured performance claim. Tests compare correlated-noise blends against independent SciPy GLS, full nuisance fits, shared-calibration error floors, complex real-coordinate references, gradients and singular-mode failures.

On MPS Float32, only the compact `p × p` query matrix moves to CPU for spectral norms, symmetry/PSD/rank checks and any already-declared roundoff repair. Reported eigenvalues, cutoffs and repaired matrices return to MPS through differentiable Torch copies. Measurement-space work and Cholesky solves remain on MPS; CPU/CUDA placement and all numerical thresholds stay unchanged. This avoids unsupported native MPS spectral kernels without enabling global CPU fallback. Independent shared-covariance tests check first/second query derivatives, sampling covariance and transfer on the available hardware. This bounded route does not resolve the separate legacy band-grid footprint-bound failure or establish universal device support.

## Typed calibrated additive placement

`AdditiveComponent` declares a plane's stage, frame, units, value semantics, processing state and stochastic contribution. Two deterministic calibrated-image placements currently execute:

- `SAMPLED_RESPONSE_INPUT`: an already sampled/band-integrated native-grid plane receives the existing discrete achromatic response once.
- `CALIBRATED_OUTPUT`: a residual in observed product units bypasses that response.

`apply_calibrated_additive_response` composes them in a standalone exposure path. `FieldObservation.additive_components` uses the same contract in supported scalar/distribution rendering, moves and crops. Tiled rendering retains response halos. Deterministic additions leave predictive variance unchanged; they do not add photon shot noise or nuisance covariance.

Legacy `sky_background`, `instrument_background` and `detector_background` retain their existing post-response detector-product semantics. Photon illumination, dark-charge and read stages still reject in the legacy field renderer. The opt-in native physical likelihood below handles them through a separate explicit contract. Chromatic/learned-PSF input planes, pixel-sample/exact-Foam paths and component anchors reject unsupported combinations rather than silently drop them. Units and processing state must match; no area/time/gain/flat conversion is inferred. The new joint Gaussian binding supports the two calibrated placements while the older provisional inverse binding still rejects typed components.

## Continuous positions and source-side spectral transport

`NativeQuadratureResponseOp` transports sampled source measures directly into native, already integrated detector pixels. A `NativeSpectralPixelResponse` callable sees detector/source coordinates, wavelength in Angstrom and explicit footprint metadata. It may vary with source position and wavelength. `AchromaticPixelResponse` adapts the existing unit-square response reference without another pixel integral.

The finite matrix is `A[p,n,l] = response[p,n,l] * source_measure[n] * wavelength_measure[l] * source_spectral_weight[n,l]`. Use source weights one for point spectral flux density, or source-node solid angles for spectral radiance. Use Angstrom spectral widths for densities per Angstrom, or ones for bin-integrated flux. Optional band/throughput weights act before transport; separate operators represent separate bands. Area, energy-to-photon conversion, duration, gain, QE and flat calibration are explicit responsibilities, never inferred from labels. A detector footprint integral belongs inside the response exactly once; a measured PSF need not have a unit cropped-aperture sum.

`forward` maps `[..., N, L]` to `[..., P]`; `adjoint` is its exact discrete Euclidean transpose. Both permit signed linear inputs and retain gradients with respect to continuous positions, measures and response parameters. Independent Gaussian-CDF integrals check chromatic centroids, extended emission, source solid angles and cropped tails. Masks zero rows without flux renormalization. Quadrature accuracy and physical response calibration remain separate scientific questions.

The optional `interaction_pairs` graph lists detector/source pairs and requires an explicit `interaction_support` declaration. Its response work scales as `O(E L)`; every forward/adjoint call also validates tensors and graph uniqueness. Omitted edges must be zero in the represented model, or belong to an explicitly separate wing/long-range operator. A support string is provenance rather than a proof; infinite tails must not be silently cut. Without a graph the chunked all-pairs path is a bounded `O(P N L)` reference. Chunks bound response blocks, not resident arrays, graph validation or autograd training memory. This primitive is standalone and does not yet replace the existing discrete field renderer or certify continuous-field resolution.

## Physical additive stages: a linear integrated-exposure reference

`integrated_photoelectron_mean` accepts incident and foreground photon rates in the same already integrated spatial/spectral bins. A positive linear `optical_mean_response` maps their sum into pixel photon rates through independent routing/thinning of Poisson arrivals. Its coefficients must describe non-amplifying routing probabilities over disjoint incident photon streams, with total probability at most one per incident bin. Those callback semantics cannot be proved from a single evaluation. Duration and a detection probability in `[0,1]` produce independent collected-photoelectron means. This is distinct from deterministically mixing charge after collection. If using the quadrature operator here, integrate densities with their measures exactly once; do not supply pre-integrated bin rates and also multiply by density quadrature weights.

`integrated_detector_moments` adds independent dark-electron means before a fixed linear charge response `D`, then read offset and Gaussian read noise with factor `R`, then gain conversion `G`. It retains the full covariance action

\[
C v = G\left[D\,\operatorname{diag}(\mu_{\mathrm{photo}}+\mu_{\mathrm{dark}})D^T
       +R\,\operatorname{diag}(v_{\mathrm{read}})R^T\right]G^T v.
\]

`LinearDetectorResponse` supplies each map and its exact transpose. Their fixed linearity/adjoint semantics are caller contracts. Signed read offsets and deterministic subtraction change the mean; subtraction does not remove shot noise. Off-diagonal covariance is retained, with no diagonal variance plane, dense covariance allocation or automatic inverse. Borrowed inputs and callback state must remain fixed until rebuilding moments. This exact linear moment reference is not a normalized Poisson–Gaussian likelihood, a fitted calibration, or a nonlinear/history model. Multi-electron photon conversion, saturation, brighter-fatter effects, persistence and read ramps require further primitives and validation.

## Bounded Poisson–Gaussian law and zero-rate derivatives

`poisson_gaussian_nll` and `poisson_gaussian_cdf` describe an independent Poisson electron count plus continuous Gaussian read noise. `expected_electrons` is the nonnegative integrated count mean, including independent photon and dark contributions. `gain` is output units per electron; `background` is a signed deterministic output offset; read variance is in output units squared. This gain convention is reciprocal to electrons per ADU. Observations require real Float32/Float64; other parameters follow the existing observation dtype/device coercion.

The retained count support is finite, selected outside autograd and explicitly renormalized. This is a normalized approximation rather than the full physical law; support changes can change the density discontinuously. Support bounds and width are checked before integer conversion, and counts outside the observation dtype's exact integer range reject. CDF and survival sums use complementary-error functions and select the better-conditioned probability half. This numerical choice retains representable tail values and rate scores without selecting a different physical model.

At exactly zero rate, the helpers supply analytic right-hand first/second rate and mixed derivatives, including support that increases immediately to the right. Reverse-mode, public forward AD/JVP, no-grad and inference evaluations are tested. Higher rate derivatives at zero are unsupported. Unrepresentable boundary score/curvature rejects when rate derivatives are requested; finite value-only evaluations remain available. No artificial positive count floor is added.

The CDF uses mode-scaled finite probabilities with explicit rate powers and detached coefficient recurrences. For retained count `k` and detached mode `m = floor(rate)`, its unnormalized weight is proportional to `(rate**k/k!) / (rate**m/m!)`. Scaling above/below the mode keeps active bases and coefficients at most one. Constant/linear powers are handled explicitly to avoid unselected inverse-power overflow in tiny-rate Hessians. No intensity cutoff, positive-rate floor or alternate distribution is introduced by this CDF construction.

The four former CDF expected failures now pass their unchanged first/second-derivative assertions at rate `1e-6`, values `±10`, variance/gain one and `tail_sigma=0.5`. Additional strict checks cover rates `1e-40`/`1e-30`, finite-polynomial mixed Hessians/JVPs, heterogeneous supports and independent SciPy values through rate `10000.3`. These bounded checks do not certify every rate/tail combination or higher derivative order.

The NLL now uses separate prior and observation-conditioned modal polynomial sums. Explicit rate powers avoid tiny-rate log-weight Hessian cancellation; the observation-conditioned reference preserves material joint terms even when individual Poisson probabilities and absolute Gaussian densities underflow. A frozen Gaussian reference supplies stable values while a live conditional-density ratio preserves read/gain/background/observation derivatives. Adjacent factorial ratios avoid subtracting large log-factorials. Residuals and gain are standardized before squaring. No positive-rate floor, intensity cutoff, Bayesian prior or alternate physical distribution is introduced.

Derivative requests fail closed when modal reference scales or degree-multiplied curvature intermediates exceed the dtype range. This contract is conservative: at rate `1e-6`, observed values `±1000` and unit gain/read variance, a finite final derivative still rejects because a reference scale underflows. A tested Float32 shifted-mode case also rejects despite a representable final Hessian; Float64 evaluates it. In these domains, no-grad/inference values remain available from the same retained law's log sums. This is not a universally certified derivative engine. Forward JVP under no-grad still counts as a derivative request.

The observed NLL Hessian need not be positive semidefinite: the two-count law at rate `0.2`, observation `0.3` and unit gain/variance has negative rate curvature. Do not substitute it directly for an expected Fisher information or claim calibrated covariance from its inverse. [Positive-rate tests](../tests/test_poisson_gaussian_positive_nll.py) independently check finite-polynomial mixed Hessians/JVPs, closed-form tiny-rate scores, shifted/wide supports and range failures.

## Native physical single-read likelihood

`torchsky.map.field.bind_native_integrated_read_likelihood` binds a native `FieldObservation`, a `NativeFieldResponse` and an electron-to-output `PixelCalibrationOp`. Call `nll(sampled_values)` directly on nonnegative `[N,L]` source states, or `nll_field(model)` to sample a scalar field on registered source nodes. Neither requires a converged global field or a catalog. This opt-in route requires `processing_state="raw_reads"`, `value_semantics="single_integrated_read"` and `detector_law="poisson_gaussian"`: one continuous integrated read per native pixel. It does not interpret a nondestructive ramp or a processed HST FLC, JWST CAL or MegaCam Elixir exposure as a raw read.

The photon response already integrates the source and spectral measures into native pixels, in photon/s. Its calibration declares `flat_convention="photon_routing_qe_excluded"`; the caller must establish physical independent photon routing/thinning, throughput and area conversion. A declaration cannot certify those properties or turn a sky-flat response into routing probabilities. Supply positive scalar `exposure_time_s`, `electrons_per_photon` in `[0,1]` and positive kept-pixel `read_variance_electrons2`. Here electrons per photon is a detection probability producing at most one electron, with remaining QE applied once; higher charge yields require another detector law. Rates are constant over the supplied duration.

Pass typed `additive_components` separately from the observation's legacy additive fields. Source-node `PHOTON_ILLUMINATION` is nonnegative spectral density in source units and receives the same response as the source. Detector `DARK_CHARGE` is a nonnegative electron/s rate added after optics. Signed `READ_OFFSET` is in electrons; signed `CALIBRATED_OUTPUT` is in observed units. The latter two change means without deleting shot noise. For duration `t`, QE `Q`, photon response `R`, dark rate `d`, read offset `r`, output offset `o` and output scale `D`, the conditional parameters are

\[
\lambda=t\,[Q\,R(I+I_{\rm illumination})+d],\quad
g=D,\quad v=D^2v_{\rm read},\quad b=Dr+o.
\]

The normalized retained Poisson–Gaussian density is evaluated in observed units, including the diagonal density Jacobian: `NLL_output(y) = NLL_electron((y-o)/D-r) + log(D)` per kept pixel. Subtracting a known sky mean leaves its photon noise in `lambda`. Observation and response masks exclude measurements without renormalizing the PSF. `reduction="sum"` is the independent-pixel NLL; `"mean"` is a reporting objective; `"none"` returns an image with zero excluded rows. `expected_output` reports the full physical conditional mean, distinct from the finite retained law's mean.

Legacy weights, uncertainty planes, backgrounds, second PSFs and ambiguous observation-level time/gain conversion reject. General charge/read mixing maps also reject, including claimed identity maps: diagonal mean/covariance checks do not establish an independent normalized likelihood. Original data, masks, physical terms and parent calibration/response are guarded before and after evaluation. Hidden callback state, inference tensors and alias writes require truthful caller revisions; rebind after changes or for a new fixed-evidence parameter graph.

This route conditions on fixed calibration and physical additive parameters. It inherits the finite-support and conservative derivative-range limits above, and supplies no Gaussian normal-statistic compression, shared-nuisance elimination, expected Fisher information or calibrated uncertainty. Correlated/nonlinear/history laws, exposure-time integration, streamed survey training and native science calibration remain separate gates.

Both `nll(values, mixture_chunk_size=C)` and `nll_field(model, mixture_chunk_size=C)` optionally evaluate at most `C` kept pixels per mixture call. With ordinary reverse AD, non-reentrant checkpointing recomputes mixture internals during backward: first-backward mixture workspace is `O(C K)`, with `O(P)` law inputs/results, instead of retaining all `O(P K)` mixture activations. `C` must be a positive integer; the default `None` retains the original path. Masks, per-pixel finite count support, density normalization, reductions and physical-parameter gradients are preserved. Evidence must remain fixed until backward finishes; changes before or during recomputation reject. Float32/Float64 values, first/second gradients, JVPs and actual saved-tensor shapes have bounded checks.

Forward AD uses direct chunks to avoid checkpoint transform incompatibilities; combining it with reverse AD retains the mixture graph. Higher-order reverse graphs can also retain recomputed intermediates. Field sampling and response graphs stay resident under every option, and `reduction="none"` still returns the complete image. Source-node chunking alone does not bound likelihood training memory. This component bounds the ordinary first-order mixture activations, not total survey training memory, and makes no runtime speedup claim.

Correlated charge/read noise, ADC quantization, nonlinear detector history and physical additive-stage field wiring require further laws. Photon moments are not this density, and a calibrated image's units/processing cannot be inferred to satisfy it. Bounded tests compare direct finite probabilities, independent SciPy density/tails, mixed derivatives and normalized integrals.

## Joint nuisance binding for existing band-grid fields

`torchsky.map.field.gaussian.bind_linear_band_grid_likelihood` builds one Gaussian factor from a bounded group of `FieldObservation` objects without creating a global `FieldModel`. It reuses the frozen discrete band-grid response and requires matching band metadata, units, dtype/device and fixed independent Gaussian noise. It does not implement chromatic pre-pixel rendering. Explicit non-Gaussian detector laws reject.

Fixed legacy output means and supported typed input/output means are subtracted at their declared stages. Optional `nuisance_bases[e]` share one ordered coefficient set across exposures. With no covariance factor, coefficients are profiled jointly and zero-level gauges remain unresolved. An independently calibrated factor `L` and mean give joint covariance `B L Lᵀ Bᵀ`, centering and the density determinant correction. Shared calibration error therefore does not average down as independent exposure noise. All exposures sharing eliminated coefficients belong in this factor; the system rejects independently eliminated repeats.

Required native sample-group identifiers prevent input/stack double counting when truthful lineage is supplied. Original tensors and response metadata are guarded. Custom geometry/passband objects and inference-mode data require an explicit revision getter. Binding packs `O(N)` data and `O(N k)` nuisance values, with no dense pixel covariance. Use bounded exposure/tile groups; this is not a whole-survey coupled-calibration solver. Independent dense GLS and determinant checks cover masks and shared errors. The discrete inverse adjoint now uses an exact bilinear scatter/PSF transpose so first-order gradients of the normal quadratic agree with direct native residuals, including tiled paths and zero duals. No arbitrary second-order geometry-gradient claim follows from that change.

## Pixel calibration and sky-flat transfer

`PixelCalibrationOp(scale[P], input_unit=..., output_unit=..., calibration_id=..., provenance_id=..., flat_convention=..., reference_convention=...)` applies an explicit positive diagonal transform after native detector integration. The reference declares its illuminating spatial/spectral convention. Unit and flat labels do not calculate area, time, QE, gain or normalization. Signed product means are allowed. This fixed calibration does not estimate uncertainty in its scale.

Mean and Euclidean transpose both multiply by `D = diag(scale)`; the transpose is not reciprocal calibration. `apply_covariance(v, covariance_action=C)` applies `D C Dᵀ v`, retaining correlations, and `transform_covariance` supports small dense references. Supply callback dependencies through `EvidenceSnapshot` to detect their changes. Ordinary scale writes invalidate the transform; inference tensors and alias writes need explicit revision policies. Retrieved result tensors remain ordinary tensors.

A sky-flat reference divides by the measured response to its illumination. For a spatially varying response this can flatten row sums while changing point-source column sums. The primitive preserves that transfer; it does not apply a second normalization to hide it. Point-source photometry must use the complete calibrated response and appropriate reference/color/area corrections. The independent finite-matrix counterexample in `tests/test_pixel_calibration.py` also checks full covariance transformation and first/second derivatives; it is not a measured telescope flat.

## Source-node field rendering and conditional native evidence

The opt-in `torchsky.map.field.native_response.NativeFieldResponse` combines native source quadrature, explicit source sky registration, pixel calibration and declared source/response units. Sky coordinates and the response chart are separate, registered by `geometry_id`; the caller supplies that mapping, source solid-angle measures and any needed Jacobians. Detector rows must already be in the observation image's row-major order. Matching row counts cannot prove coordinate registration. `response_id`, `band_name` and dependency snapshots retain the declared fixed response evidence.

`sample_field` calls the ordinary scalar field interface at source sky nodes for each scalar wavelength, then forms `[N,L]` without calling `render_bandpass`. Chunking bounds coordinates per model call, while the sampled state and autograd graphs remain resident. Source nodes may represent continuous radiance or optional point measures; a catalog or global field solve is not required. Point spectral-flux states require source measures one. Scalar field sampling supports densities per Angstrom; already integrated bins use direct sampled states, without an implicit bin-width conversion. An epoch in `ForwardContext` can condition a time-aware model; it does not integrate variability over the exposure.

`render_native_field_observation(model, observation, response)` transports monochromatic values before native pixel integration, then applies calibration and output means once. It rejects a second PSF, sampled-input additive planes, raw reads and non-Gaussian detector laws in this bounded calibrated route. Supported typed calibrated-output components are revalidated at use; legacy sky/instrument/detector planes keep their existing observed-product residual meaning. Predictions do not depend on observed image values or data validity. The observation's bandpass/area/time/gain metadata is retained as provenance and is not applied numerically again: the declared response/calibration owns all conversions and spectral integration. No legacy normalized spectral average is reused as a physical integral. Existing render/training defaults remain unchanged.

`bind_native_field_likelihood` takes finite sampled `[N,L]` states through the same response and exact transpose into a `GaussianLikelihoodFactor`, independently of any `nn.Module`. It centers output means once and uses fixed diagonal inverse variance in calibrated output units, on the intersection of observation and response support. The Gaussian determinant constant counts that same support. Full spectral/spatial normal information is retained; native factors from different instruments may add on a common sampled state when their physical registration and units agree. Truthful native sample IDs reject input/stack duplicates. Independent dense tests include a lower-weight broad response and verify the PSD information increment in the fixed, correctly specified model; this is not a guarantee against misspecification bias.

Uncertainty in this binding is **conditional on the fixed response, calibration and output means**. It does not propagate shared calibration, source-coordinate uncertainty, state-dependent photon covariance or fitted neighboring emission automatically. The joint native binding below handles declared additive nuisance modes; the existing same-band joint binding is a separate route. Treating photon moments as fixed Gaussian precision would violate this contract. Original applied tensors, scalar metadata and conversion/passband tensor bindings are guarded; arbitrary metadata object internals need a caller revision if their freshness matters. Rebuild response/factors/queries after changes or new parameter graphs. Adaptive quadrature, default training integration, physical sky-flat/color calibration and native-data validation remain open.

## Shared nuisances across native source-node responses

`torchsky.map.field.joint_native_likelihood.bind_joint_native_likelihood` combines a bounded tuple of observations and `NativeFieldResponse` objects into one factor. The sampled source state must share exactly ordered sky and wavelength nodes, source units, measure kinds, dtype and device. Native detector charts, bands, response weights and output product units can differ. It does not infer registration, interpolation or unit conversion, and it needs no global field model.

For `K` ordered, standardized dimensionless nuisance coordinates, supply a `[P_e,K]` basis for each observation in its own output units, plus matching `nuisance_basis_units`. Physical conversion between products belongs in these bases. Calibrated mode uses one independent mean `[K]` and covariance factor `[K,R]` across the entire group. It centers the mean once, applies the full joint Woodbury precision and retains the determinant correction, including cross-exposure covariance. Profiled mode eliminates the joint span once, retains its sky/background gauges and rejects normalized predictive-density queries. Every exposure sharing these eliminated coordinates belongs in the same factor; native measurement identifiers must still be disjoint.

Conditional measurement noise is independent, fixed and diagonal in each native product. Other correlated base noise needs a different joint precision. The declared nuisance span is additive and fixed: a response or photometric-scale derivative represented by a basis is only a local linearization, not exact nonlinear calibration propagation. Shared covariance does not average down as independent exposure noise. Bounded dense references compare mixed electron/ADU products and different bands against independent SciPy GLS, normalized density, masked support and shared-error floors.

Parent response/data evidence and nuisance bases, mean and covariance are guarded at use and in cached normal/query reports. Declare hidden dependencies and genuine caller revisions for inference tensors or alias changes; rebuild after changes or new autograd graphs. Packed data and bases remain resident, with matrix-free response/precision actions and no dense pixel covariance. This is a bounded group primitive, not a whole-survey sparse calibration solver or a measured throughput claim.

## Run an exposure example

Run `pixi run -e default -- python examples/native_exposure_calibration.py`. It demonstrates native star profiling, calibrated background precision, normalized likelihood and a requested flux uncertainty with no global field. It generates a controlled Gaussian reference and is not independent validation. Bounded tests use independent SciPy Gaussian integrals, dense GLS/covariance references, gradient/curvature checks and regression cases; real-data science comparisons belong in torchsky-harness.

`pixi run -e default -- python examples/native_response_moments.py` demonstrates continuous source transport and ordered photon/dark/charge/read moments with explicit units. It does not approximate those moments as a calibrated Gaussian likelihood.

`pixi run -e default -- python examples/native_field_response.py` demonstrates source-node radiance, response/calibration ownership, an opt-in native image prediction and a conditional field-template query. Its small planar quadrature and chosen Gaussian noise are illustrative, not a calibrated instrument or accuracy certificate.

## Implementation status and open gates

Implemented and covered by independent dense/SciPy references, gradient checks,
and bounded device tests:

- Gaussian nuisance precision, native point-response profiling, fixed Gaussian
  likelihood/query evidence, typed calibrated additive placement;
- small native linear queries, legacy inverse-probe correctness, detector-grid
  PSF transport orientation, fixed-evidence calibration lifecycle;
- continuous-position spectral quadrature and ordered photon/dark/charge/read
  detector moments;
- joint Gaussian field-observation bridge, explicit pixel radiometric/flat
  transfer, opt-in native source-node field route, and the joint source-node
  calibration/nuisance binding;
- Poisson-Gaussian zero boundary/CDF tails with stable positive-rate NLL
  curvature, signed detector charts with optional GWCS freezing, the native
  physical integrated-read likelihood, and checkpointed physical-mixture
  chunks;
- compact nuisance/query spectral placement on MPS and bounded metadata-query
  tooling for native-product acquisition.

Open gates (none are implied by the bounded tests above):

- adaptive/controlled chromatic response, default training/renderer integration,
  and physical radiometric/flat calibration beyond the opt-in source-node route;
- photon/charge/read additive-stage renderer integration, normalized native
  likelihoods, and nonlinear/history detector laws;
- end-to-end observation lineage and joint nuisance propagation across every
  field/coadd caller;
- query/UQ reports and calibration lifecycle across all solvers;
- verified real native-data science references: HST/ACS, JWST/NIRCam, and
  CFHT/MegaCam products currently have bounded byte/header/geometry/calibration
  checks only (`training_ready=false`);
- broader CUDA/MPS execution evidence and independent chromatic PSF/background
  calibration (M1), field-reconstruction discovery gates (M2), nonlinear
  detector evidence (M3), interface proofs (M4), and all-available-data
  federation (M5).
