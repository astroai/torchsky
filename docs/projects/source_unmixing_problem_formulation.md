# Source unmixing with known or unknown sources

## Purpose

This note formulates spatial-spectral source unmixing for two demanding
applications:

1. full-sky absolute spectroscopy and component separation for FOSSIL-like CMB
   spectral-distortion measurements;
2. sky, telluric, and source separation in a multi-object spectrograph (MOS).

It assumes only basic familiarity with supervised learning. The central idea is
that unmixing is not primarily a classification problem. It is an inverse
problem: multiple latent signals pass through a calibrated measurement operator,
which may include corrections independently learned and validated on calibration
evidence, and contribute to the same detector samples. A catalogue can constrain
part of the latent geometry, but it does not replace the instrument model or
make the decomposition automatically identifiable.

The same Torchsky spine can support both applications:

> explicit physical axes and measurement operators; optional known target or
> source constraints; flexible nuisance models; scientifically independent
> cross-fitting; paired injections; and an exportable unexplained residual.

The intended product is not a model that assigns every datum to a closed list
of components. It is a measurement with quantified source attenuation,
cross-talk, unresolved degeneracies, and stable residual structure.

**Reader's path.** Sections 1--9 define the shared inverse problem,
identifiability limits, protected estimation, and injections. Section 10 applies
that framework to FOSSIL-like absolute spectroscopy, and Section 11 applies it
to MOS sky and telluric subtraction. Sections 12--17 describe the Torchsky
boundary, validation tiers, invariants, and scientific claim gates.

## 1. What does “known source” mean?

Three different kinds of knowledge are often collapsed into one phrase.

### 1.1 Known support or geometry

We may know where an object is and how it enters an observation:

- a catalogue supplies celestial coordinates and perhaps morphology;
- a MOS configuration says which target was assigned to each fibre or slit;
- a PSF and aperture model predict the fraction of source flux in each sample;
- a moving-object ephemeris predicts position as a function of time.

The spectrum can remain completely unknown. This is **known-support,
unknown-spectrum unmixing**.

### 1.2 Known spectral law or template

We may know the spectral shape but not its amplitude or sky distribution. In a
CMB spectral-distortion analysis, the non-relativistic temperature, \(y\), and
standard \(\mu\) shapes are examples of declared target templates. Their
monopoles or anisotropy maps are unknown.

This is **known-template, unknown-amplitude unmixing**. It does not require a
catalogue of discrete sources.

### 1.3 Known measurement response

The WCS, beam or PSF, fibre aperture, wavelength solution, spectral response,
line-spread function (LSF), Fourier-transform spectrometer (FTS) kernel,
throughput, and detector sampling describe how a latent signal becomes data.
These are not astronomical sources, but they determine whether sources can be
separated.

The response is never perfectly known. The practical contract is therefore
**calibrated response with declared uncertainty**, not an implicit assumption
that a neural network will discover the instrument by itself.

### 1.4 Unknown-source unmixing

In the most open case, the number, support, spectra, or physical identity of
components are not fixed. Examples include an uncatalogued object in a slit, an
unexpected emission line, a foreground spectral mode missing from the sky
model, or an exotic CMB distortion outside the template bank.

Unknown-source unmixing needs an explicit complexity control and an open
residual. Otherwise a flexible decomposition can invent arbitrary components
or force every anomaly into the nearest familiar class.

### 1.5 Mechanistic hybrids: flexible models on explicit physical contracts

Torchsky should permit a mechanistic hybrid when a useful physical
approximation is too rigid for the calibration evidence. A fixed analytic PSF,
low-order telluric approximation, or nominal blackbody-reference model can
remain an initializer or baseline without limiting the final model class.

In this architecture:

1. **Mechanistic contract:** The causal ordering
   \(\mathcal H[\mathcal T I_{\rm cel}+E]+B\), radiometric value semantics,
   passband and pixel integration, and matched linear adjoints remain explicit.
   The declared WCS is evaluated to numerical tolerance while its physical
   calibration uncertainty is propagated. Conservation or throughput
   constraints are imposed only where physically appropriate.
2. **Flexible learned substrate:** Focal-plane- and wavelength-dependent LSF
   distortions, airglow variation, telluric fine structure, stray light,
   detector common modes, and diffuse foreground variation may use continuous
   neural fields, adaptive coordinate encoders, learned spectral bases, Power
   Foams, or hybrid models.
3. **Calibration and identifiability tests:** Physical priors, standard stars,
   lamps, reference states, repeated observations, cross-fitting, and protected
   target projections constrain which variation the flexible components can
   explain. They measure and limit target attenuation; they cannot guarantee
   that nuisance and signal are identifiable from the available data.

This approach can reduce misspecification relative to a rigid model and reduce
the freedom of an unconstrained black box. It does not eliminate
non-identifiability. Residual target/nuisance overlap, unsupported calibration
domains, and architecture dependence remain part of the scientific result.

## 2. A common continuous scene model

Let \(\boldsymbol x\) denote sky position, \(\nu\) frequency, and \(t\) time.
The incident celestial specific intensity can be decomposed as

\[
I_{\rm cel}(\boldsymbol x,\nu,t)
= \sum_{q=1}^{Q}
M_q(\boldsymbol x,t;\gamma_q)S_q(\nu,t)
+ \sum_{c=1}^{C}
A_c(\boldsymbol x,t)F_c(\nu,\boldsymbol x,t)
+ U(\boldsymbol x,\nu,t).
\]

The first sum represents discrete or morphologically specified sources.
\(M_q\) is a normalized spatial profile satisfying
\(\int_{\rm sky} M_q(\boldsymbol x,t;\gamma_q)\,\mathrm d\Omega_{\boldsymbol x} = 1\)
(with units of sr\(^{-1}\)) and \(S_q\) is its flux-density spectrum (e.g. in Jy).
The product \(M_q S_q\) evaluates to surface brightness (e.g. Jy sr\(^{-1}\) or
W m\(^{-2}\) Hz\(^{-1}\) sr\(^{-1}\)). The second sum represents diffuse components
whose spatial amplitude \(A_c\) and spectral law \(F_c\) may both vary. Whatever
factor convention is chosen, each product \(A_cF_c\) and the open component
\(U\) must have the same surface-brightness units as \(I_{\rm cel}\). \(U\) is not
initially assigned to a named source family; it is the candidate open component.

This equation is conceptual rather than unique. A component can be represented
by a parametric law, a basis expansion, an INR, a Power Foam, spherical
harmonics or needlets, a sparse source dictionary, or a hybrid. What matters is
that each term has declared units and enters the same observation operator.

For surface brightness, \(I_\nu\) may have units such as
W m\(^{-2}\) Hz\(^{-1}\) sr\(^{-1}\). A point-source spectrum \(S_q\) instead
has flux-density units such as Jy. A normalized PSF or morphology converts that
flux density into surface brightness before spatial integration. Adding a Jy
spectrum directly to an MJy sr\(^{-1}\) map is dimensionally invalid even when
the arrays have the same shape.

## 3. The common measurement equation

Index measured samples by \(a\). A sample can be an image pixel, a MOS detector
pixel, a calibrated spectral channel, a time sample, or an interferogram delay
sample. Write

\[
\mu_a
= \mathcal H_a(\psi_a)
\left[
\mathcal T_a I_{\rm cel}
+ E_a
\right]
+ B_a,
\qquad
y_a=\mu_a+\epsilon_a.
\]

Here:

- \(\mathcal T_a\) is a multiplicative transmission operator, such as the
  atmosphere in ground-based spectroscopy;
- \(E_a\) is emission added before some or all of the instrumental response,
  such as atmospheric sky emission;
- \(\mathcal H_a\) composes geometry, spatial response, spectral response,
  differencing, calibration, and detector integration;
- \(B_a\) is emission or offset added at a later detector/instrument stage;
- \(\psi_a\) contains calibrated response parameters;
- \(\epsilon_a\) is noise.

The ordering is physical. Telluric absorption multiplies extraterrestrial
source light; airglow is additive. Instrumental thermal emission may be added
after atmospheric transmission. A differential FTS measures a transformed sky
minus reference signal. Treating every nuisance as an additive post-processing
vector gives the wrong derivatives and can bias line or distortion amplitudes.

The ordering and units are physical contracts; the functional form of each
stage need not be fixed. For example, a spatial response may be a convolution,
a field-dependent beam or PSF operator, a differentiable optics/detector model,
a neural operator learned from calibration data, or a physical model with a
learned residual. The same freedom applies to transmission, spectral response,
background, and noise. A flexible stage must declare its conditioning variables,
support, invariants, training evidence, and valid calibration domain.

If \(\boldsymbol y\) is itself a coadd, cube, INR rendering, Foam rendering, or
other reconstructed product, its upstream transfer function and correlated
covariance are part of \(\mathcal H\) and \(\boldsymbol N\). Such a product is
not a noiseless label. Native detector samples are preferable when the upstream
compression is not sufficient for the target statistic or when its transfer
and covariance cannot be propagated faithfully.

When the calibrated operator is linear in the latent scene, stack all samples:

\[
\boldsymbol y
= \boldsymbol H(\psi)\boldsymbol z
+\boldsymbol b+\boldsymbol\epsilon,
\]

where \(\boldsymbol z\) contains the latent components. The scene
parameterization, transmission, or learned calibration corrections may make the
complete problem nonlinear, but the linear operator and its adjoint remain
valuable building blocks.

## 4. Likelihood and complete objective

For Gaussian noise with covariance \(\boldsymbol N\) fixed independently of the
fitted parameters, the negative log-likelihood, up to an additive constant, is

\[
\mathcal L_{\rm data}
=\frac12
(\boldsymbol y-\boldsymbol\mu)^\mathsf T
\boldsymbol N^{-1}
(\boldsymbol y-\boldsymbol\mu).
\]

If a positive-definite covariance \(\boldsymbol N(\eta)\) is learned jointly,
the parameter-dependent normalization cannot be dropped. On the valid-data
subspace the Gaussian objective is

\[
\mathcal L_{\rm data}(\theta,\psi,\eta)
=\frac12\boldsymbol r^\mathsf T
\boldsymbol N(\eta)^{-1}\boldsymbol r
+\frac12\log\det\boldsymbol N(\eta),
\qquad
\boldsymbol r=\boldsymbol y-\boldsymbol\mu(\theta,\psi).
\]

Without the log-determinant term, a joint fit can lower the quadratic term by
inflating the covariance. A distribution-agnostic `NoiseLearner` need not define
a normalized likelihood, but its output must then be described as a weighting
or whitening rule, trained on independent or cross-fitted evidence and
calibrated on held-out residuals.

Diagonal inverse variance is a special case, not the general MOS or
time-stream noise model. FTS samples, resampled spectra, detector common modes,
and sky-subtracted products can have substantial covariance.

For diagnostic evaluation, let
\(\boldsymbol N=\boldsymbol C_N\boldsymbol C_N^\mathsf T\) on the valid-data
subspace and define the whitened residual

\[
\boldsymbol r_{\rm white}
= \boldsymbol C_N^{-1}(\boldsymbol y-\boldsymbol\mu).
\]

If the residual is multivariate Gaussian, the covariance is known, and no
parameters or normalizations were estimated from these samples, then
\(Q=\|\boldsymbol r_{\rm white}\|_2^2\sim\chi^2_{r_N}\), where \(r_N\) is the
rank of the valid-data covariance. For an ordinary fitted linear model, a reduced
chi-square also requires the fitted degrees of freedom. That quantity is not
automatic for a regularized or nonlinear learned model. In that case report the
mean squared whitened residual and calibrate its distribution with held-out
folds or simulation. Autocorrelation \(C_r(\Delta a)\) with null intervals
diagnoses remaining correlation; it does not prove that the noise is white. For
heavy-tailed outliers use a declared robust loss such as Charbonnier, while low
photon counts require an appropriate count or Poisson-Gaussian likelihood.

The complete estimator is

\[
(\widehat\theta,\widehat\psi,\widehat\eta)
=\underset{\theta,\psi,\eta}{\operatorname{argmin}}
\left[
\mathcal L_{\rm data}(\theta,\psi,\eta)
+\mathcal R_{\rm source}(\theta)
+\mathcal R_{\rm nuisance}(\theta)
+\mathcal C_{\rm calibration}(\psi)
+\mathcal C_{\rm noise}(\eta)
+\mathcal C_{\rm protection}(\theta)
\right].
\]

The source and nuisance models may be closed-form, non-parametric, neural, or
hybrid. Their regularization can encode sparsity, positivity, smoothness,
spectral rank, temporal stability, or physical priors. Calibration terms anchor
the response to lamps, standards, sky lines, reference loads, repeat
observations, or other evidence. Protection terms restrict how a nuisance model
can overlap a target subspace. A physical approximation is therefore a useful
inductive bias and baseline, not a mandatory ceiling when the data support a
more flexible model. When the covariance is fixed, omit \(\eta\) and
\(\mathcal C_{\rm noise}\). When it is learned, \(\mathcal C_{\rm noise}\) records
any structural constraint or independent calibration evidence and does not
replace the likelihood normalization above.

Every term changes the estimator. A strong smoothness prior can suppress a
narrow line; a low-rank spectral prior can erase an exotic distortion; a hard
target projection can transfer a foreground overlap into the target estimate.
Regularization must therefore be selected using held-out data and injections,
not reconstruction loss alone.

## 5. Known-support source unmixing

Assume the positions or spatial profiles of \(Q\) sources are supplied. Applying
the WCS, PSF/beam, aperture, and trace model produces a mixing coefficient
\(K_{aq}(\nu)\): the contribution of unit flux from source \(q\) to sample
\(a\). The prediction becomes

\[
\mu_a
=\sum_q\int\!\mathrm d\nu\,
R_a(\nu)K_{aq}(\nu)S_q(\nu)
+n_a(\theta_{\rm nuisance}),
\]

where \(R_a\) includes spectral response and \(n_a\) contains diffuse sky,
background, or other components.

After discretizing the spectra, this is a block linear inverse problem:

\[
\boldsymbol y
=\boldsymbol K\boldsymbol s
+\boldsymbol G\boldsymbol c
+\boldsymbol\epsilon.
\]

\(\boldsymbol s\) contains source spectra and \(\boldsymbol c\) nuisance
coefficients. Solving each source or wavelength independently discards useful
constraints from repeated exposures, overlapping apertures, common sky, and
spectral continuity. Conditional on fixed \(\boldsymbol c\), a Tikhonov source
solve with \(\boldsymbol W=\boldsymbol N^{-1}\) and a coordinate-aware
second-derivative operator \(\boldsymbol D_2\) satisfies

\[
\left(
\boldsymbol K^\mathsf T \boldsymbol W \boldsymbol K
+ \lambda \boldsymbol D_2^\mathsf T \boldsymbol D_2
\right)\widehat{\boldsymbol s}
=\boldsymbol K^\mathsf T\boldsymbol W
(\boldsymbol y-\boldsymbol G\boldsymbol c).
\]

The coordinate-aware construction matters for irregular channels and changes
of frequency or wavelength units. Its normalization and the units of \(\lambda\)
must make the data and penalty terms commensurate and must transform with the
spectral coordinate convention. The regularizer can stabilize an
ill-conditioned deblend, but it selects a prior-dependent solution rather than
resolving an exact spatial degeneracy. If \(\boldsymbol c\) is also unknown,
solve the joint regularized block system for \([\boldsymbol s,\boldsymbol c]\)
or marginalize the nuisance model. Blended spectra are identifiable only when
the relevant columns of \([\boldsymbol K\ \boldsymbol G]\), calibration evidence,
or declared priors distinguish them under the noise metric.

### 5.1 What a catalogue buys us

A good catalogue can fix or constrain:

- source count and identity;
- position and proper motion;
- morphology or point-source status;
- fibre/slit assignment;
- expected trace location;
- optional photometric anchors or spectral priors.

It does not guarantee:

- accurate aperture losses under seeing and differential atmospheric
  refraction;
- correct deblending in crowded apertures;
- an uncontaminated sky fibre;
- the right source spectrum or radial velocity;
- complete coverage of faint, transient, moving, or serendipitous sources.

Catalogue uncertainty should enter as a prior or a limited pose/morphology
refinement, not as a silently exact label.

### 5.2 Known templates as a second form of known source

If the spectral shape \(\boldsymbol t_q\) is known but its amplitude
\(\alpha_q\) is not, then

\[
\boldsymbol s_q=\alpha_q\boldsymbol t_q.
\]

This reduces a high-dimensional spectrum to a small target-amplitude problem.
It is the natural formulation for \(\mu\), \(y\), temperature-shift, or a
specified dark-sector distortion template. The spatial amplitude can be a
single monopole, a full-sky field, or a small harmonic basis.

Known templates are probes, not necessarily an exhaustive ontology. Their
subtraction must leave an inspectable residual.

## 6. Catalogue-free and unknown-source unmixing

Without a source list, the data must constrain both **what exists** and **what
its signal is**. This problem is harder because source number, support,
morphology, spectrum, and component identity can trade against one another.

There is no universal need to construct a detection catalogue before fitting.
Several parameterizations are possible.

### 6.1 Low-rank or basis decomposition

Arrange spatial samples by spectral channels in a matrix \(\boldsymbol Y\). A
basic model is

\[
\boldsymbol Y\approx\boldsymbol A\boldsymbol S^\mathsf T,
\]

where columns of \(\boldsymbol A\) are spatial amplitudes and columns of
\(\boldsymbol S\) are spectra. PCA constrains orthogonal variance directions;
NMF constrains non-negative factors; ICA seeks statistical independence;
learned bases replace fixed algebraic constraints.

These are useful nuisance models, but the factorization is not unique. For any
invertible matrix \(\boldsymbol Q\),

\[
\boldsymbol A\boldsymbol S^\mathsf T
= (\boldsymbol A\boldsymbol Q)
(\boldsymbol S\boldsymbol Q^{-\mathsf T})^\mathsf T
= (\boldsymbol A\boldsymbol Q)
(\boldsymbol Q^{-1}\boldsymbol S^\mathsf T)
= \boldsymbol A(\boldsymbol Q\boldsymbol Q^{-1})\boldsymbol S^\mathsf T
= \boldsymbol A\boldsymbol S^\mathsf T.
\]

Physical interpretation therefore requires anchors, constraints, or external
evidence. Selecting the component count only for low reconstruction error can
absorb more signal as capacity grows.

### 6.2 Sparse overcomplete scene

Place candidate source atoms on a fine spatial grid or continuous proposal set
and solve for sparse spectra or amplitudes:

\[
\widehat{\boldsymbol s}
=\arg\min_{\boldsymbol s}
\frac12\|\boldsymbol y-\boldsymbol K\boldsymbol s\|_{\boldsymbol N^{-1}}^2
+\lambda\|\boldsymbol s\|_1.
\]

Groups across wavelength can encourage one spatial source to retain a coherent
spectrum. This avoids committing to a catalogue, but resolution and source
separation remain limited by the PSF/aperture response and dictionary
coherence. Positivity is appropriate for emission flux, but not for
differenced, continuum-subtracted, or absorption signals.

### 6.3 Continuous fields, INRs, and Power Foam

A continuous field \(F_\theta(\boldsymbol x,\nu,t)\) can represent the total
sky or one component without an object list. An INR uses a coordinate-conditioned
network or learned feature field. A Power Foam uses adaptive bounded spatial
cells with spectral coefficients. Both can be embedded in a differentiable
observation model. INR parameters and Foam appearance coefficients can be
optimized directly; hard Foam geometry instead requires a declared smooth
surrogate or another non-gradient optimization method.

These representations answer “what intensity field explains the data?” before
answering “how many catalogue objects does it contain?” A catalogue can be
derived later by a common detector if object-level products are needed.

Their reconstruction transfer is part of the scientific result. A downstream
component fit must propagate the field's support, target-response Jacobian,
correlated covariance, and response/nuisance uncertainty, preferably by
composing the original detector operator. Treating rendered INR or Foam pixels
as independent exact measurements can turn regularization attenuation into a
spurious component constraint.

The limitation is interpretability: one field can contain sources, diffuse
emission, and residual systematics unless roles, context dependence, and
validation splits constrain it. The hard/smooth distinction for Foam also
applies: source-response claims must use the declared final hard rendering, not
only its geometry-training surrogate.

### 6.4 Explicit unknown objects

A model can infer a finite set of positions, morphologies, motions, and spectra
directly. This is useful for uncatalogued or moving sources, but unknown source
count introduces a model-selection or birth/death problem. A practical sequence
is safer than an unconstrained end-to-end object generator:

1. fit calibrated diffuse and known-source components;
2. form a cross-fitted whitened residual;
3. propose compact or line-like candidates from that residual;
4. refit all accepted candidates jointly through the original detector model;
5. retain unmatched stable residual structure as an open product.

Candidate proposal and final measurement must use different evidence folds or
correct for selection. Otherwise the reported flux and significance inherit a
look-elsewhere and noise-selection bias.

### 6.5 The open residual

Unknown-source analysis must be allowed to say “unexplained.” Define

\[
\boldsymbol r_{\rm open}
=\boldsymbol y
-\widehat{\boldsymbol y}_{\rm target}
-\widehat{\boldsymbol y}_{\rm predictable\ nuisance}.
\]

An open residual product should retain:

- the observed and predicted values;
- target, known-source, and predictable-nuisance contributions;
- unwhitened and whitened unexplained residuals;
- masks, weights, covariance or whitening metadata;
- sky, detector, wavelength/frequency, time, scan, and reference context;
- model, calibration, split, and input provenance.

It is not automatically new physics or a new astronomical source. It is a
ranked candidate product that can be tested for celestial consistency,
cross-view stability, calibration sensitivity, and external correlations.

### 6.6 Discovery, characterization, and multiplicity

A flexible residual basis is selected from the data, so the largest fitted mode
does not have the null distribution of one prespecified template. Discovery and
characterization must use separate evidence. One valid design uses discovery
folds to propose a spectral-spatial subspace, freezes that subspace and its
selection rule, then estimates amplitudes and uncertainties on independent
characterization folds. When modes are relearned in several folds, compare
subspaces with principal angles or canonical correlations rather than matching
arbitrary component labels.

The discovery statistic must be calibrated by rerunning the complete selection
pipeline on null simulations, sign/scan/detector jackknives, or exchangeable
surrogates. Use a maximum-statistic or family-wise-error gate for a small
candidate set, or a declared false-discovery-rate procedure for an atlas. The
null ensemble must include architecture and capacity selection, not only the
final frozen fit. Follow-up spectra withheld from both nuisance training and the
simulator family test sensitivity to unknown shapes. A thermalization-manifold
test must propagate theoretical interpolation and numerical uncertainty in the
manifold; distance from one deterministic grid is not a discovery significance.

## 7. Identifiability and unavoidable gauges

Unmixing is identifiable only when the data and constraints distinguish the
latent contributions. Important degeneracies include:

### 7.1 Scale and permutation

In a factored component \(A_cF_c\), multiplying \(A_c\) by \(k\) and dividing
\(F_c\) by \(k\) changes nothing. Blind component labels can also be permuted.
A reported component needs a scale convention, such as unit spectral norm,
unit response in a reference band, or an anchored amplitude.

Torchsky's existing field and harmonic component decompositions expose this
scale gauge and can canonicalize it without changing predictions. A canonical
gauge improves reproducibility; it does not create physical identifiability.

### 7.2 Additive sky/background ambiguity

A spatially smooth celestial component can be exchanged with an additive
background unless detector motion, sky/reference differences, blank regions,
or absolute calibration distinguish them. Random pixel splits do not break this
degeneracy because neighbouring pixels share the same nuisance state.

### 7.3 Multiplicative transmission/source ambiguity

The product \(T(\nu)S(\nu)\) cannot be separated from one spectrum alone.
Telluric standards, airmass/time diversity, repeated sources, atmospheric
physics, or externally anchored continua are needed. Treating telluric
absorption as an additive negative sky spectrum is generally wrong.

### 7.4 PSF, aperture, morphology, and source flux

An error in seeing, beam, centring, differential atmospheric refraction, or
fibre acceptance changes apparent source flux and colour. PSF/aperture
response may be learned flexibly from calibration or repeated geometry, and
need not reduce to a convolution or a fixed parametric PSF. It cannot, however, be
identified freely from only the target whose morphology or spectrum is being
measured. Independent stars, dithers, detector positions, wavelengths, times,
or physical constraints must break the response/source degeneracy.

### 7.5 Spectral response and astrophysical lines

Wavelength shifts, LSF asymmetry, FTS phase, and response-curve errors can mimic
line centroids, widths, or residual distortion modes. Calibration-state and
scan-direction splits are necessary in addition to an overall residual score.

### 7.6 Target/nuisance overlap

If a target template \(\boldsymbol t\) lies partly in the nuisance span
\(\mathcal N\), no algorithm can infer the overlapping amplitude without
additional information. The relevant quantities are the weighted overlap,
effective target norm after nuisance projection, and condition number—not the
nominal target norm before foreground fitting.

## 8. Protected targets and orthogonal estimation

Let \(\boldsymbol T\) contain protected target templates and
\(\boldsymbol W=\boldsymbol N^{-1}\). The weighted projector onto their span is

\[
\boldsymbol\Pi_T
=\boldsymbol T
(\boldsymbol T^\mathsf T\boldsymbol W\boldsymbol T)^{+}
\boldsymbol T^\mathsf T\boldsymbol W,
\]

using a rank-revealing factorization when the template bank is rank deficient
or linearly dependent. In computation, prefer a weighted QR or SVD of
\(\boldsymbol W^{1/2}\boldsymbol T\) over explicitly forming the Gram matrix,
which squares its condition number. The Gram pseudoinverse shown above covers
the rank-deficient case. A hard protected nuisance model can be restricted to
\((\boldsymbol I-\boldsymbol\Pi_T)\mathcal N\), or a soft penalty can discourage
weighted overlap.

Hard protection guarantees that the fitted nuisance cannot algebraically
subtract the declared target. It does **not** guarantee an unbiased target:
real foreground power parallel to \(\boldsymbol T\) is then forced into the
target estimate. The report must include target/nuisance overlap, effective
target norm, condition number, and the change between hard, soft, and
unprotected fits.

Protection does not require a rigid signal model. The signal of interest may be
a fixed template, a template family with learned deformations, a flexible
source field, or an initially unspecified open component. Likewise, the
nuisance may be parametric, low-rank, sparse, an INR/Foam, a large neural model,
or an ensemble. The scientific comparison should follow a capacity path, a
sequence of models from simple to flexible, rather than evaluate one chosen
architecture. Increase nuisance and signal flexibility, select it using
out-of-fold prediction, and measure end-to-end target response with paired
injections. A more flexible nuisance is accepted only while target attenuation,
cross-talk, false positives, fold instability, and architecture sensitivity
remain within declared limits. Structure that is stable across independent
views but unstable under nuisance architecture belongs in the open residual,
not automatically in either a discovery claim or a nuisance label.

If no target span can be declared, hard projection cannot guarantee signal
preservation. Open searches must instead use scientifically independent folds,
blinded challenge injections spanning plausible but withheld morphologies,
capacity-path stability, and cross-view reproducibility. Even then, attenuation
outside the injected signal family remains an explicit limitation rather than
something model flexibility can eliminate.

For cross-fitting, partition observations into scientifically meaningful folds
\(k\). Fit the nuisance model on all folds except \(k\), predict fold \(k\), and
estimate a target amplitude from the held-out residual:

\[
\widehat\alpha_k
=\frac{
\boldsymbol t_k^\mathsf T\boldsymbol W_k
(\boldsymbol y_k-\widehat{\boldsymbol n}_{-k})
}{
\boldsymbol t_k^\mathsf T\boldsymbol W_k\boldsymbol t_k
}.
\]

This simple expression is valid only when the target response and nuisance
prediction are fixed for the held-out fold. More general estimators should
orthogonalize against the fitted nuisance tangent and propagate calibration
uncertainty. Fold-to-fold variation is part of the result, not merely an
optimization diagnostic. Because cross-fit estimates share overlapping
training data, the folds are not automatically independent replicates.
Uncertainty must be based on the underlying independent observational units or
on repeated non-overlapping partitions, not on a naive standard error over the
\(K\) fold estimates.

Cross-fitting prevents direct reuse of a held-out sample, but it does not make
the nuisance prediction known. Let \(\boldsymbol C_{n,k}\) be the predictive
covariance of \(\widehat{\boldsymbol n}_{-k}\), and let
\(\boldsymbol J_{\psi,k}\boldsymbol C_\psi
\boldsymbol J_{\psi,k}^{\mathsf T}\) propagate independently calibrated
response uncertainty into fold \(k\). When these errors are conditionally
independent of held-out detector noise, use

\[
\boldsymbol V_k
=\boldsymbol N_k+\boldsymbol C_{n,k}
+\boldsymbol J_{\psi,k}\boldsymbol C_\psi
 \boldsymbol J_{\psi,k}^{\mathsf T}
\]

in place of \(\boldsymbol W_k^{-1}\). If independence does not hold, include the
cross-covariance terms in
\(\operatorname{Cov}(\boldsymbol y_k-\widehat{\boldsymbol n}_{-k})\).
For a fixed target response, the conditional generalized least-squares variance
is \((\boldsymbol t_k^{\mathsf T}\boldsymbol V_k^{-1}
\boldsymbol t_k)^{-1}\). Joint nonlinear fits should instead profile or
marginalize nuisance and calibration parameters, or use an independently
validated influence-function or sandwich approximation. In every case,
injections and repeated independent units must demonstrate interval coverage;
fold scatter alone is not an uncertainty estimate.

## 9. Paired injections and target response

For target signal \(\boldsymbol s\) and amplitude \(a\), define the symmetric
pipeline response

\[
\mathcal R(a)
=\frac{
\widehat\alpha(\boldsymbol d+a\boldsymbol s)
-\widehat\alpha(\boldsymbol d-a\boldsymbol s)
}{2a}.
\]

Normalize \(\boldsymbol s\) so that \(a\) has declared physical units and
\(\widehat\alpha\) estimates the same amplitude. Under that convention
\(\mathcal R\) is dimensionless and \(1-\mathcal R\) is attenuation. Otherwise
report the response as a dimensional Jacobian rather than calling it an
attenuation fraction.

Positive/negative pairing cancels much of the fixed background contribution and
tests the complete estimator, including candidate selection, nuisance fitting,
calibration refinement, and final measurement. Required outputs are:

- attenuation \(1-\mathcal R\);
- response nonlinearity with amplitude;
- cross-talk into other targets/components;
- dependence on sky region, detector, fibre/slit, scan, time, and spectral
  morphology;
- detection efficiency and false-positive rate;
- comparison of scene-level and measurement-level injections.

Injection after the measurement operator does not test WCS, PSF/beam,
aperture, response, or FTS errors. Both locations can be useful, but they answer
different questions and must be labelled.

## 10. Application I: FOSSIL full-sky spectral unmixing

### 10.1 Mission context carried from the FOSSIL science note

The July 2026 FOSSIL science document records FOSSIL (FTS fOr CMB Spectral
diStortIon expLoration) as an ESA M8 concept for a full-sky absolute
spectrometric survey from L2. Its public June 2026 baseline is a differential
FTS comparing the sky with an internal blackbody reference, with approximately
130 bands over 50--2000 GHz, about 15 GHz spectral resolution, and roughly 1.6
degree angular resolution; public material also described a goal extending
toward 30 GHz. The source note says that the Phase-2 proposal was submitted in
April 2026 and treats subsequent selection milestones as prospective.

The same document records forecast uncertainties of about
\(\sigma(\mu)=1.5\times10^{-8}\) from the monopole and about
\(5\times10^{-9}\) when spatial information is included. These values are a
dated concept baseline, not new Torchsky results.

The standard primordial signal is expected at order
\(\mu\simeq2\times10^{-8}\). It probes dissipation of small-scale acoustic
perturbations over roughly
\(k\sim50\)--\(10^4\ {\rm Mpc}^{-1}\), depending on the observable and window
convention. This extends primordial-power measurements far beyond ordinary CMB
temperature and polarization anisotropies.

### 10.2 FTS measurement equation

For optical-path difference \(x\), an idealized differential interferogram is

\[
d(x)
=\int\!\mathrm d\nu\,R(\nu)
\left[I_{\rm sky}(\nu)-I_{\rm ref}(\nu)\right]
\cos\!\left(\frac{2\pi\nu x}{c}+\phi(\nu,x)\right)
+n(x).
\]

A production operator must also represent finite maximum delay and instrument
line shape, single- or double-sided scans, irregular or missing delay samples,
zero-path-difference errors, phase, detector response, throughput mismatch,
flags, scan direction, beams, pointing, and calibration. In a simple phase
model, a ZPD offset contributes \(2\pi\nu\Delta x_{\rm zpd}/c\), while additional
dispersion can contribute \(\phi_{\rm disp}(\nu)\); scan-dependent effects may
retain further \(x\) dependence. In double-sided data, an uncorrected phase
produces an odd interferogram contribution; more generally it appears as an
imaginary component after complex spectral transformation.

#### Sampling and end-to-end transfer functions

Under the OPD convention above, uniform sampling with spacing \(\Delta x\) has
Nyquist frequency \(\nu_{\rm Nyq}=c/(2\Delta x)\). For a symmetric unapodized
scan extending to maximum absolute delay \(x_{\max}\), the first-zero spectral
resolution is approximately \(\Delta\nu=c/(2x_{\max})\). Sidedness, apodization,
irregular sampling, missing samples, phase correction and the chosen estimator
modify the instrument line shape and channel covariance. The exact convention
and measured transfer function must therefore replace these ideal formulas in a
science analysis.

An end-to-end signal-transfer report must inject monopole and anisotropic
spectral modes before OPD sampling and propagate them through flagging, phase
correction, spectral estimation, calibration and mapmaking. It must measure
aliasing from above the modeled band, attenuation and mixing between spectral
templates, angular/spectral mode coupling, correlated output noise and any
monopole removed by baseline or mapmaking operations.

If the internal reference is modelled at \(T_0\) but is actually
\(T_0+\Delta T\), its first-order contribution to the sky-minus-reference
residual is

\[
\Delta I_{\nu,{\rm ref}}
=-\frac{\partial B_\nu(T_0)}{\partial T}\,\Delta T.
\]

Dividing this expression by \(B_\nu(T_0)\) gives a fractional intensity error,
not a chemical-potential amplitude. Its bias on a fitted scalar \(\mu\) must be
computed by propagating \(\Delta I_{\nu,{\rm ref}}\) through the instrument and
projecting it through the complete nuisance-aware \(\mu\) estimator. The
differential sign and every linear adjoint require explicit tests.

#### Required FOSSIL systematic-error budget

A scalar reference-temperature perturbation is only one calibration mode. A
FOSSIL precision analysis must propagate a parameterized or empirical budget
covering at least:

- internal-reference temperature, emissivity, reflectance and spatial/temporal
  thermal gradients;
- OPD metrology, frequency-scale error, ZPD drift, phase and instrument-line-
  shape uncertainty;
- detector gain, offset, nonlinearity, temporal response, saturation and
  correlated readout modes;
- chromatic and asymmetric beams, far sidelobes, detector beam mismatch,
  pointing and scan-to-map transfer;
- sky/reference throughput mismatch, internal emission, polarization leakage
  where relevant, and scan-synchronous signals;
- signal-dependent photon noise from the total sky, reference and instrument
  loading, together with detector and correlated noise;
- flagging, gap filling, filtering, spectral estimation and mapmaking transfer
  functions.

Each mode needs a prior or independent calibration source, a derivative or
nonlinear response through the science estimator, an allocated bias and
variance contribution, and a null test tied to an instrument symmetry or
observing redundancy. Detector, scan direction, reference state, calibrator
temperature and time/half-mission jackknives must be exercised on the complete
time-ordered or interferogram pipeline, not only on final maps.
The PIXIE calibration and systematic-error studies in references 20--21 provide
relevant worked examples, although the final FOSSIL budget must follow its own
hardware and scan design.

The raw-to-science chain is

```text
latent full-sky spectral components
    -> beam, pointing, spectral response and sky/reference difference
    -> detector sequence / interferogram
    -> calibrated spectra
    -> full-sky spectral maps and monopole
    -> target estimates + open residual
```

Mission packet decoding remains outside Torchsky. Torchsky should accept decoded
samples, aligned flags, housekeeping, optical delay, detector and scan context,
and provenance.

### 10.3 What is being unmixed?

A FOSSIL-like sky can contain:

- the CMB blackbody and temperature-shift mode;
- \(y\) and relativistic-\(y\) distortions;
- the standard \(\mu\) distortion;
- recombination and broader residual-distortion modes;
- Galactic dust, synchrotron, free-free, spinning-dust/AME, and line emission;
- CIB and other extragalactic backgrounds;
- spatially and temporally structured instrumental/reference residuals;
- dark-matter decay, annihilation, scattering, photon--dark-photon conversion,
  extra-radiation, or other exotic distortions;
- a deliberately unclassified residual.

Torchsky core should not hard-code this complete ontology. It should provide a
`SpectralTemplate`-like interface, response-aware component fields, protected
subspaces, full-sky products, splits, and validation. FOSSIL/BISOU extensions
should own mission channel tables, scan configurations, telemetry adapters,
sensitivity assumptions, cosmological and dark-sector template libraries, and
publication estimators.

### 10.4 Known-target FOSSIL analysis

The first precision case is the standard \(\mu\) template. Its low-dimensional
spectral law makes it suitable for testing whether a high-capacity foreground
model preserves a weak target. Required comparisons include:

- parametric foreground laws;
- moment expansions;
- PCA/NMF or robust low-rank models;
- learned spectral bases;
- continuous spatial-spectral fields;
- protected hybrid models.

All receive the same frequency axis, radiometry, beams, masks, noise,
sky/reference and FTS operators, and split manifest. The result reports
\(\mu\) bias and attenuation, effective sensitivity, nuisance overlap, split
stability, foreground-prediction error, null failures, residual covariance,
throughput, and memory.

Protection applies to response-convolved templates, not to analytic spectra
before the instrument. The precision target bank should jointly include the
CMB temperature-shift mode, non-relativistic and relativistic \(y\) modes, the
standard \(\mu\) mode, recombination/residual modes required at the achieved
sensitivity, and any other declared cosmological component. Calibration and
beam derivatives belong in a jointly fitted or marginalized nuisance tangent
space. Report both the conditional \(\mu\) constraint with other modes fixed and
the marginalized constraint with foreground, calibration and neighboring
spectral modes free. Protecting only one nominal \(\mu\) vector can transfer
power parallel to omitted modes into the fitted \(\mu\) amplitude.

The \(\mu\) monopole and a \(\mu\)-anisotropy map are distinct observables. The
monopole requires preserved absolute zero level and sky/reference calibration.
An anisotropy reconstruction requires an angular transfer function, beam and
mask treatment, and split cross-spectra; its correlations with CMB \(T/E\) or
large-scale-structure tracers do not substitute for a monopole measurement.
Sensitivity gains from spatial information must state whether they arise from
foreground control of the monopole, direct anisotropy information, or both.

The physical questions include whether primordial power remains nearly scale
invariant over additional decades, whether inflationary features or running
exist, whether enhanced small-scale power or isocurvature is present, and
whether \(\mu\) anisotropies/correlations reveal primordial non-Gaussianity or
statistical anisotropy.

### 10.5 Unknown-shape and dark-sector analysis

Theory templates should be optional probes rather than an exhaustive partition
of the data. Dark-matter energy injection, hidden-sector photon conversion,
extra radiation, and altered thermalization can share leading \(\mu/y\)
amplitudes while differing in residual spectral structure or anisotropy
correlations.

The open-discovery sequence is:

1. fit explicitly declared targets and only held-out-predictable nuisance;
2. retain an open residual at detector, interferogram, spectrum, and full-sky
   levels;
3. search it first with known templates, then a broad residual basis;
4. rank modes by detector/scan/time reproducibility, celestial consistency,
   calibration stability, spectrum, morphology, and external correlations;
5. interpret stable candidates only after null and shift tests.

Unknown-shape injections must be withheld from nuisance-model development.
Detection efficiency should be reported as a function of spectral morphology
and nuisance overlap, not only particle parameters.

Candidate proposal, statistical confirmation and physical characterization
must use the discovery/characterization separation in Section 6.6. Calibrate a
global maximum-mode or false-discovery statistic by rerunning basis selection,
capacity selection and candidate ranking on the complete null ensemble. Align
relearned residual subspaces across splits by their noise-weighted principal
angles, not by arbitrary mode number. A candidate is not assigned physical
significance until it survives this global gate, the instrument-systematic
budget, independent science splits and simulator-family holdout.

The corresponding fundamental questions are whether the dark sector exchanged
energy or photon number with the Standard Model during the distortion era,
whether light mediators or tiny kinetic mixings are visible outside collider
and direct-detection reach, whether spectral shape distinguishes energy
injection from photon conversion or modified thermalization, and whether a
stable component lies outside both known astrophysical/instrumental modes and
the available particle-template bank.

### 10.6 Three open-discovery products

The two FOSSIL notes define three ordered scientific directions.

#### Open full-sky residual atlas

Publish or retain collaboration-wide unexplained spectral modes with their
full-sky morphology, spectra, detector/scan/time stability, calibration
sensitivity, and external-tracer correlations. The residual is a first-class
product, not a failed fit.

#### Joint monopole--anisotropy residual tomography

Determine which residual modes are isotropic, trace Galactic structure,
correlate with CMB \(T/E/B\), or correlate with CIB, line-intensity, and
large-scale-structure tracers. These measurements can help discriminate a
primordial origin from late-time or Galactic emission when interpreted through
a joint response and covariance model with foreground and systematic nulls.
They do not identify an origin by correlation alone or require every residual
to follow a predeclared law.

#### Falsification of the standard thermalization manifold

Construct the observable manifold generated by broad energy and photon-number
injection histories evolved through standard Compton scattering, double
Compton emission, and bremsstrahlung. Ask whether a calibration-stable celestial
residual lies inside that manifold. A component that is stable across declared
detector, scan, time, calibration, and nuisance-architecture splits and is
orthogonal under the declared noise-weighted metric would first motivate
re-examination of calibration and missing astrophysics, and only then new photon
kinetics or non-standard thermalization.

The recommended order is the residual atlas with standard \(\mu\) injections,
then monopole--anisotropy tomography, then the thermalization-manifold test with
a specialist once the measurement pipeline is trusted.

### 10.7 Scientifically independent FOSSIL splits

Random frequency pixels are not independent folds, and not every useful split
is a repeated measurement of the same estimand. The split manifest must assign
each partition one of the following roles:

- **Repeated-science splits:** detector or detector group, appropriate
  time/half-mission blocks, scan direction and odd/even FTS sweeps can provide
  approximate repeated measurements after their different response and noise
  are propagated. These support consistency tests and split cross-statistics.
- **Nuisance-generalization splits:** sky regions test foreground prediction and
  domain shift. They are not independent measurements of a global monopole,
  which is present in every region and coupled through common calibration.
- **Spectral-prediction splits:** held-out frequency ranges test interpolation
  or extrapolation. Their target response and nuisance degeneracy differ, so
  they are not exchangeable replicates and must be compared through the full
  observation model.
- **Calibration/null contrasts:** sky versus internal-reference state,
  calibrator temperature/state, instrument symmetry and shuffled housekeeping
  test calibration or systematic models. A reference state contains no
  celestial target and is not a science fold.
- **Distribution-shift tests:** a held-out instrument, hardware configuration or
  simulator family measures portability beyond the development distribution;
  it does not increase the nominal experiment's sample size.

Target estimates must agree across applicable repeated-science splits within
their propagated covariance. A component that follows detector, scan,
calibrator or housekeeping state but not celestial coordinates is not an
astrophysical discovery. Shared calibration parameters and overlapping training
sets induce cross-split covariance and must be retained when combining results.

## 11. Application II: MOS sky and telluric subtraction

### 11.1 MOS measurement equation

Index exposure by \(e\), fibre/slit by \(f\), and detector spectral pixel by
\(p\). A useful staged model is

\[
\mu_{efp}
=\mathcal A_e\!\left[
\mathcal D^{\rm src}_{efp}\!\left\{
T_e(\lambda,\boldsymbol x_f,t)
\left(
\sum_q K_{efq}(\lambda,t)S_q(\lambda,t)
+\Omega_{ef}(\lambda,t)I_{\rm ext}(\lambda,\boldsymbol x_f,t)
\right)\right\}
+\mathcal D^{\rm sky}_{efp}\!\left\{
\Omega_{ef}(\lambda,t)E_{\rm atm}(\lambda,\boldsymbol x_f,t)
\right\}
+\mathcal D^{\rm inst}_{efp}\!\left\{
E_{\rm inst}(\lambda,f,t)
\right\}
\right]
+B_{efp}.
\]

\(\mathcal A_e\) integrates over the exposure for count-like data or averages
over it for count-rate data; the declared calibration determines which. This
operation is required when airglow, water vapour, seeing, guiding or flexure
changes appreciably within one exposure. \(I_{\rm ext}\) is extraterrestrial
diffuse emission and therefore receives the same atmospheric transmission as
compact sources.

\(K_{efq}(\lambda)\) includes target position, seeing/PSF, slit or fibre aperture,
differential atmospheric refraction (DAR), guiding and morphology.
Wavelength-dependent DAR induces a spatial centroid offset

\[
\Delta\boldsymbol x_f(\lambda)
=\tan z\,[n(\lambda)-n(\lambda_0)]
\widehat{\boldsymbol e}_{\rm zenith},
\]

where \(z\) is the zenith distance, not the airmass
\(X\simeq\sec z\), and \(\widehat{\boldsymbol e}_{\rm zenith}\) is the
unit tangent-plane direction toward the zenith, set by the parallactic angle.
The expression is an angular displacement in radians under the small-refraction
approximation; its sign follows the declared offset convention.
\(\Omega_{ef}\) is the effective sky solid angle. The physical transmission
satisfies \(0\leq T_e\leq1\). Nearly saturated telluric intervals contain little
or no recoverable source information and must be flagged or reported with their
large model dependence rather than repaired by unconstrained division.
\(E_{\rm atm}\) is emergent atmospheric radiance at the instrument entrance,
including atmospheric self-absorption; a more detailed radiative-transfer
operator may replace this separation, but it must not apply atmospheric
attenuation twice. It contains airglow lines, scattered light and atmospheric
continuum. \(E_{\rm inst}\) contains pre-detection instrumental/thermal emission.

The three \(\mathcal D\) operators apply the appropriate throughput, trace
geometry, wavelength solution, fibre-dependent line-spread response, detector
integration, cross-talk and calibration for each illumination path. They may
coincide in an idealized instrument, but fibre near/far-field illumination,
focal-ratio degradation, slit illumination, pupil dependence and the optical
stage where emission enters can make source, extended-sky and instrument
responses measurably different. Non-local scattered light, ghosts, persistence
and detector cross-talk must be explicit operators or late background terms,
not absorbed silently into a local sky spectrum. LSF variations across the
focal plane and between illumination paths must be included to avoid residual
sky-line subtraction spikes.

In photoelectron-count units, photon noise depends on the total incident source,
sky and instrument loading before subtraction. Read noise, dark current and
correlated detector modes add further covariance, and uncertainty in the fitted
sky/telluric prediction must be propagated as in Section 8. Subtracting a mean
sky model removes its expectation, not its shot noise or prediction variance.

This ordering distinguishes four operations that conventional “sky
subtraction” can mix together:

1. deblend astronomical sources entering the same aperture;
2. estimate additive atmospheric sky emission;
3. correct multiplicative telluric absorption;
4. remove instrumental/detector backgrounds and cross-talk.

### 11.2 MOS with known sources

When an imaging catalogue and fibre/slit assignment are available, construct
\(K_{efq}(\lambda)\) for every plausible contributor, not only the nominal
target. Jointly infer source spectra and common/local sky rather than treating
each extracted 1D spectrum independently.

Useful constraints include:

- astrometric positions and proper motions;
- source morphology from imaging;
- exposure-dependent seeing and guiding;
- wavelength-dependent differential atmospheric refraction;
- repeated visits and overlapping fibre/slit apertures;
- photometric flux anchors with their uncertainties;
- optional rest-frame or stellar spectral priors.

The source spectrum must remain flexible enough for emission lines, absorption
features, redshift, radial velocity, and unusual objects. A photometric prior
should constrain broad flux scale, not force a template that erases the
discovery signal.

### 11.3 MOS without a catalogue

A detection list is not required for the initial fit. Use all calibrated 2D
detector data or extracted fibre spectra to learn:

- a shared, spatially/time-varying sky-emission field;
- detector/fibre-private residual components;
- a multiplicative telluric field anchored by standards or repeated sources;
- sparse or continuous astronomical residuals.

After cross-fitted sky/telluric prediction, compact traces or coherent spectral
features in the whitened residual can propose sources. Refitting those sources
through the original 2D trace/PSF/LSF operator provides fluxes and
uncertainties. The final catalogue is therefore an output, not a prerequisite.

This is particularly useful for serendipitous sources, blended objects,
transients, moving objects, and lines from objects below the direct-imaging
catalogue threshold.

### 11.4 Learning the sky without “blank” fibres

Dedicated sky fibres are valuable but need not be assumed perfectly blank.
One model can treat source contributions as sparse or context-specific while
the atmospheric sky is shared predictably across many fibres, positions, and
times. A fibre with a source can still constrain sky lines at wavelengths where
the source model and cross-fibre evidence permit the source and sky
contributions to be separated.

The model must respect different variability scales:

- narrow airglow lines can vary rapidly in amplitude while keeping related line
  structure;
- sky continuum varies more smoothly in wavelength and position;
- telluric bands vary multiplicatively with airmass, water vapour and time;
- fibre LSF and throughput vary across the focal plane and calibration state;
- astronomical spectra remain fixed or evolve according to their own time and
  velocity model.

Torchsky's adaptive spatio-spectral-temporal encoders are a useful local
nuisance primitive because their active scales can be learned within
calibration-derived bounds. They are not evidence by themselves that source and
sky are separated. Cross-fibre/time prediction and source injections provide
that evidence.

### 11.5 A staged MOS solve

1. **Calibrate geometry and illumination-dependent response.** Fit trace,
   WCS/assignment, wavelength, source/extended-sky LSF and throughput, fibre
   cross-talk, scattered light and detector covariance from independent
   calibration evidence.
2. **Initialize additive components.** Use sky fibres, source-excluded regions,
   robust cross-fibre structure, and instrument backgrounds without assuming
   every control is blank.
3. **Initialize telluric transmission.** Use standards, atmospheric metadata,
   repeated sources and radiative-transfer constraints with
   \(0\leq T\leq1\); flag saturated intervals with inadequate information.
4. **Fit known or sparse source components.** Propagate every component through
   the same aperture, PSF, trace and LSF operators.
5. **Cross-fit nuisance models.** Hold out fibres, exposures, time blocks,
   detector regions, or targets rather than random detector pixels, and retain
   predictive covariance for the held-out nuisance estimate.
6. **Jointly refine within calibration priors.** Prevent the source target from
   redefining its own LSF, sky or telluric model.
7. **Propagate uncertainty.** Combine source/sky photon noise, detector
   covariance, nuisance predictive covariance and response/calibration
   uncertainty; validate interval coverage with independent injections.
8. **Export target spectra and open residuals.** Preserve 2D and 1D residual
   context for line searches and systematic diagnosis.

### 11.6 Required MOS comparators

The strongest baseline depends on the instrument and observing mode. Validation
must select and tune conventional comparators without test-set access and must
not use a per-fibre median as the only reference. The minimum candidate set is:

- the instrument pipeline's sky-fibre, offset-sky, nod-and-shuffle or dedicated-
  sky procedure, including its standard extraction;
- native, unrectified 2D wavelength/trace-aware B-spline sky modeling where raw
  detector geometry is available;
- line-family or vibrational-group OH rescaling with wavelength/flexure and LSF
  correction for near-infrared data;
- a segmented PCA residual method such as ZAP where the data geometry supports
  it;
- a physical telluric fit such as Molecfit, with its fitted molecular columns,
  wavelength and LSF degrees of freedom;
- the best validated hybrid of these steps used operationally by the
  instrument team.

All methods must receive the same raw or calibrated detector samples, masks,
calibrations and allowed external metadata. Compare both native-detector and
rectified/extracted paths when resampling is part of the scientific question,
and propagate each path's covariance and transfer function to the same final
science scorer. References 16--19 describe the conventional native-2D, OH,
telluric and PCA families named above.

### 11.7 MOS validation endpoints

The primary metric depends on the downstream science. A complete report should
include:

- injected emission- and absorption-line flux bias and completeness;
- equivalent-width bias;
- continuum and colour bias;
- line centroid and width bias;
- redshift and radial-velocity bias where applicable;
- residuals around bright and variable sky lines;
- false-line rate in blank and shuffled controls;
- held-out fibre/exposure predictive likelihood or whitened residual score;
- covariance and autocorrelation of the subtracted spectrum;
- performance versus fibre, focal-plane position, time, airmass, source
  brightness, crowding and spectral morphology;
- performance versus distance from bright sky lines in local LSF-width units,
  sky brightness and within-exposure sky variability;
- performance versus telluric transmission, including separately reported
  saturated or masked intervals;
- sensitivity to wavelength/flexure error, source/sky illumination mismatch,
  LSF uncertainty, scattered light, persistence and rectification;
- calibrated confidence-interval coverage after nuisance and response
  uncertainty propagation;
- for radial-velocity work, dependence on barycentric correction, spectral-
  template mismatch, wavelength zero point and LSF uncertainty;
- throughput, transfer volume, and peak memory.

A low RMS can be scientifically worse if it attenuates narrow lines. Line
response and false-positive rate must be measured at the resolutions and
wavelengths used by the intended science.

## 12. Shared Torchsky primitives

The two applications differ in scale and instrument, but require the same
mission-neutral abstractions.

| Primitive | FOSSIL use | MOS use |
| --- | --- | --- |
| typed tensor/axis semantics | frequency, sky pixel, detector, scan, delay | wavelength, detector pixel, fibre/slit, exposure, time |
| spectral response and radiometry | FTS band response, absolute intensity, reference | throughput, LSF, flux/surface brightness, fibre aperture |
| executable operator + adjoint | beam/pointing/FTS/difference/mapmaking | PSF/aperture/trace/LSF/cross-talk/extraction |
| continuous or basis components | full-sky foregrounds and residual modes | sky field, telluric field, source spectra |
| shared/private context mixer | celestial versus detector/scan residuals | common sky versus fibre/instrument/source components |
| protected subspace | \(\mu\), \(y\), dark-sector templates | lines, continua, RV/redshift templates |
| scientific partitions | detector/scan/time/reference/sky region | fibre/exposure/night/detector/target |
| paired injections | distortion attenuation and cross-talk | source/line attenuation and false detections |
| open residual | residual atlas and exotic searches | serendipitous lines/sources and systematic diagnosis |
| reconstruction transfer and covariance | propagate map/INR/Foam regularization into distortion inference | propagate extraction/reconstruction uncertainty into source and sky inference |

These primitives should be differentiable where scientifically coherent,
chunkable over large axes, executable on CPU/CUDA and eventually other devices,
and testable against tiny explicit operators. Known measurement transforms must
not be hidden inside a learned scene.

## 13. Current Torchsky substrate and missing pieces

### 13.1 Already present

- torch-native WCS, HEALPix, sparse maps, spherical harmonics and mapmaking;
- image-plane and full-sky observation operators;
- `ObservationOperatorBundle` separation of geometry, spectral response, PSF,
  radiometry, background, masks/weights and noise;
- matrix-free multi-observation execution and exact-adjoint tests on promoted
  operators;
- continuous spatial-spectral field families, including INRs and spectral
  Power Foam;
- learned spectral bases and context-conditioned fields;
- field and harmonic shared/survey-private/instrument-private/interaction
  component decompositions with explicit scale-gauge handling and leakage
  diagnostics;
- adaptive detector-x/y, wavelength and MJD nuisance encoders with
  calibration-evidence bounds;
- evidence-space calibration terms plus explicit scene/background/PSF joint
  training stages;
- fill-distance interval bounds for rectangular focal-plane calibration
  coverage;
- staged learned PSF/background calibration and held-out diagnostics;
- a hardened `SpectralAxis` for frequency, wavelength, energy and wavenumber;
- residual and support-aware field diagnostics plus bounded execution.

These are building blocks, not a completed FOSSIL or MOS source-unmixing
pipeline.

### 13.2 Still required

- a minimal immutable `TensorSpec` that rejects shape-compatible but physically
  incompatible tensors;
- reusable spectral-response/channel-mixing and radiometric semantics;
- one typed linear-operator protocol with domain/codomain, forward, adjoint,
  chunk/device and normal-operator contracts;
- full-sky spectral maps with explicit monopole, dipole, anisotropy, zero-level,
  beam, mask, weight and split semantics;
- decoded observation-sequence, sample, detector, scan and delay contracts;
- experimental interferogram and FTS operators with dense oracles;
- distribution-agnostic nuisance/noise learner protocols;
- weighted protected subspaces \(\boldsymbol\Pi_T\) and orthogonal target
  estimators;
- scientific cross-fit manifests and paired-injection records;
- open residual products \(\boldsymbol r_{\rm open}\) at detector, sequence,
  interferogram, spectrum, map and full-sky levels;
- sphere-native and delay-native adaptive encoders;
- catalogue-free sparse/unknown-object proposal and refit machinery;
- predictive covariance and calibrated uncertainty propagation through learned
  non-stationary nuisance and response operators;
- FTS alias, spectral/angular transfer-function and instrument-systematic budget
  reports;
- illumination-path-aware MOS source/sky/instrument operators with
  within-exposure integration and photon-loading covariance;
- selection-aware discovery statistics with subspace alignment and global
  false-positive control;
- reproducible FOSSIL and MOS comparator profiles with one common scorer.

The implementation order from the FOSSIL roadmap remains appropriate:

1. minimal tensor semantics and typed operator adapters;
2. spectral response, channel mixing and radiometry;
3. weighted target protection;
4. cross-fit partitions and paired injections;
5. full-sky spectral/monopole/open-residual products;
6. decoded sequences and experimental FTS;
7. uncertainty propagation, transfer-function and global-discovery reports;
8. MOS and FOSSIL end-to-end profiles built on those shared primitives.

## 14. Benchmark hierarchy

### 14.1 Exact mathematical fixtures

- spectral-coordinate and radiometric round trips;
- analytic response integrals;
- operator forward/adjoint identities;
- differential signs and reference orientation;
- constant-source aperture and flux conservation;
- identity-transmission and zero-emission limits;
- FTS delta lines and blackbodies;
- FTS Nyquist aliases, finite-OPD line shape and apodization transfer;
- monopole preservation/removal contracts;
- identical-illumination MOS limits in which source, sky and instrument
  response operators reduce to the declared common operator;
- Poisson source-plus-sky variance before and after sky estimation;
- permutation and scale-gauge invariance.

### 14.2 Controlled stress tests

- low-rank plus sparse sources;
- spatially varying spectral curvature;
- blended sources and coherent source/sky spectra;
- correlated and non-stationary noise;
- missing samples, fibres and channels;
- response, wavelength, phase, gain and beam/LSF shifts;
- calibration drift and focal-plane variation;
- calibrator emissivity/thermal gradients, chromatic beams, sidelobes, detector
  nonlinearity and signal-dependent photon noise;
- source/extended-sky illumination mismatch, within-exposure airglow/telluric
  variation, saturated telluric regions, scattered light and persistence;
- target/nuisance overlap;
- withheld source or distortion morphologies;
- unsupported sky, detector, delay or atmospheric regimes.

### 14.3 Real-background injections

Inject known distortions, spectra, or lines into observed multi-frequency and
spectroscopic nuisance backgrounds. This tests the estimator against structure
not generated by the same simulator used to design it. Preserve the original
control observation and use paired identifiers so differences are attributable
to the injection. Check empirical confidence-interval coverage, not only point
recovery, and rerun adaptive candidate selection on null backgrounds when a
global discovery statistic is claimed.

## 15. Required invariants

1. **Units:** every additive term has identical radiometric semantics at the
   point of addition.
2. **Flux:** a normalized PSF/beam plus complete aperture conserves point-source
   flux away from true boundaries.
3. **Identity atmosphere:** \(T=1\), \(E_{\rm atm}=0\), zero diffuse celestial
   emission, and zero instrumental background reduce the MOS operator to source
   response plus noise.
4. **No source:** zero source amplitudes reduce to the nuisance-only prediction
   without an invented catalogue component.
5. **Known isolated source:** an identity spatial/spectral response recovers its
   spectrum under the declared noise model.
6. **Adjoint:** each linear operator satisfies the weighted or Euclidean
   dot-product identity appropriate to its domain and codomain.
7. **Chunk parity:** chunked and resident execution accumulate the same normal
   equations within numerical tolerance.
8. **Gauge:** rescaling a component and inversely rescaling its mixer leaves the
   prediction unchanged.
9. **Permutation:** relabelling unconstrained blind components does not change
   the aggregate scene or likelihood.
10. **Protection:** a protected template cannot enter the hard-projected
    nuisance span, while foreground overlap and condition loss remain reported.
11. **Injection oddness:** in the locally linear regime, positive and negative
    injection differences reverse sign and give consistent response.
12. **Split isolation:** no sample from a held-out detector/target/time/scan unit
    influences nuisance fitting or normalization for that fold.
13. **Open accounting:** observed equals target plus known source plus
    predictable nuisance plus open residual under one declared convention.
14. **FTS sampling:** a tone above the uniform-OPD Nyquist frequency aliases to
    its predicted in-band frequency, while the declared anti-alias/band model
    either excludes or propagates that power.
15. **Uncertainty:** a sky or nuisance estimate with nonzero predictive variance
    increases, rather than decreases, the propagated target variance.
16. **MOS illumination limit:** setting the source, extended-sky and instrument
    response operators equal reproduces the simpler common-operator equation.
17. **Discovery calibration:** the complete adaptive selection pipeline attains
    its declared family-wise error or false-discovery rate on null ensembles.
18. **Reconstruction handoff:** replacing native samples by a coadd, INR, Foam,
    cube, or extracted spectrum reproduces the native-sample target estimate
    only when the declared transfer and covariance make that compression
    sufficient for the target; otherwise the discrepancy is measured and
    reported.

## 16. Scientific objectives and claim gates

### 16.1 FOSSIL

The ordered programme is:

1. recover standard \(\mu\) injections near the expected signal scale while
   quantifying attenuation and nuisance overlap;
2. show whether spatial information improves effective sensitivity without
   introducing split-dependent bias;
3. produce an open full-sky residual atlas;
4. add monopole--anisotropy modes and external cross-correlations;
5. evaluate blinded dark-sector and unknown-shape spectra;
6. test stable residuals against the standard thermalization manifold.

A weak-target result is incomplete unless it reports target bias, attenuation,
effective sensitivity, target/nuisance overlap, fold and architecture
stability, nuisance/calibration uncertainty and interval coverage, end-to-end
spectral/angular transfer functions, the allocated systematic-error budget,
null failures, residual covariance, unsupported regimes, deterministic
provenance, throughput and peak memory. An open-discovery claim additionally
requires a global selection-aware significance or false-discovery gate and
independent characterization evidence.

### 16.2 MOS

The primary claim should be tied to a science measurement, for example:

- lower line-flux bias at fixed false-line rate;
- improved completeness at fixed purity for faint emission lines;
- lower RV or redshift error without line attenuation;
- improved continuum/telluric fidelity on held-out targets;
- recovery of serendipitous sources absent from the input catalogue.

The comparison must use identical raw/calibrated samples, masks, response
metadata, split units, injections, target extraction, and final scorer. Beating
a per-fibre median sky model is useful engineering evidence; a scientific claim
requires the applicable conventional suite in Section 11.6, illumination-aware
response checks, measured covariance including nuisance prediction uncertainty,
and calibrated interval coverage. Report performance near bright sky lines,
through low-transmission telluric regions and under flexure/LSF shifts, not only
an average over easy wavelengths.

### 16.3 FOSSIL research deliverables

The science note proposes two concrete paper-scale outcomes:

1. **Discovery-preserving recovery of primordial CMB spectral distortions with
   cross-fitted spatial-spectral nuisance models.** Compare parametric/moment,
   PCA/NMF, learned-basis, continuous-field, and protected hybrid estimators on
   FOSSIL-like simulations and real multi-frequency backgrounds with injected
   \(\mu\) signals.
2. **Open-world searches for exotic CMB spectral distortions with learned
   nuisance structure and cross-view stability tests.** Withhold published and
   unknown-shape dark-sector spectra during method development, then require
   consistency across detector, scan, time, sky/reference, and
   map/interferogram views.

The associated delivery sequence is a trustworthy full-sky/FTS substrate,
standard \(\mu\) validation under foreground and instrument misspecification, a
blinded dark-sector challenge set, and a science/white-paper contribution that
presents targeted precision inference and preservation of unexplained spectral
structure as complementary mission capabilities.

## 17. What failure means

A decomposition can fail in several scientifically distinct ways:

- **underfit nuisance:** residual foreground/sky dominates the target;
- **overfit nuisance:** the target or unknown source is attenuated;
- **response error:** calibration structure mimics a source or distortion;
- **non-identifiability:** target and nuisance spans are inseparable with the
  available views;
- **distribution shift:** a detector, night, sky region, or source morphology
  lies outside training support;
- **closed-world failure:** a stable real component is forced into an existing
  label or discarded as nuisance.

The remedy depends on the failure. More neural-network capacity is not a
universal answer. Additional independent views, calibration, better physical
operators, different split units, or a more honest unresolved degeneracy may be
required.

## Glossary

- **Source:** a discrete or diffuse latent contributor to the measured signal.
- **Component:** a mathematical term in the decomposition; it need not
  correspond one-to-one with a physical object.
- **Mixing:** mapping component amplitudes/spectra into shared measurements.
- **Unmixing:** inference of latent contributors from their mixture.
- **Known support:** supplied position, morphology, aperture assignment, or
  other spatial constraint.
- **Known template:** supplied spectral shape with unknown amplitude or spatial
  distribution.
- **Blind/semi-blind separation:** decomposition with few/no supplied component
  laws, or a mixture of known and learned components.
- **Nuisance:** structure required to explain the measurement but not the target
  of the current scientific estimate.
- **Protected subspace:** target span that a nuisance model is constrained not
  to absorb under a declared metric.
- **Cross-fitting:** fitting nuisance structure on independent folds and
  evaluating target estimates on held-out folds.
- **Open residual:** unexplained measurement-space structure retained with
  context and provenance.
- **FTS:** Fourier-transform spectrometer.
- **ZPD:** zero path difference in an interferogram.
- **LSF:** line-spread function.
- **MOS:** multi-object spectrograph.

## Source notes and scientific references

This formulation integrates the user-authored `fossil_primitives_plan.md` and
`fossil_science.docx`, while preserving both untracked source files unchanged.
The current mission-neutral implementation checklist is tracked in
[FOSSIL/BISOU Mission-Neutral Torchsky Track](fossil.md).

The FOSSIL science note's scientific grounding includes the public FOSSIL/BISOU
material and reviews of CMB spectral distortions, thermalization, foregrounds,
\(\mu\)-anisotropy, dark-sector energy injection, and general distortion
parametrizations listed there. Mission values above were checked against the
cited June 2026 public slides and remain dated concept forecasts rather than
Torchsky measurements.

For traceability, the reference set inherited from that source note and the
methodological sources used in this formulation are listed below. The public
mission slides, 2026 *Physical Review D* entries, and references 16--21 were
rechecked in August 2026; the remaining bibliography has not been exhaustively
re-verified for this formulation.

1. B. Maffei, “Measuring spectral distortions with FOSSIL and BISOU,”
   Challenges at the CMB Frontier (2026),
   [public slides](https://indico.in2p3.fr/event/39689/contributions/172737/attachments/102686/159782/CMB-FRONTIER_Maffei.pdf).
2. Institut d'Astrophysique Spatiale,
   [BISOU project page](https://www.ias.universite-paris-saclay.fr/en/content/bisou).
3. J. Chluba et al., “New horizons in cosmology with spectral distortions of
   the cosmic microwave background,” *Experimental Astronomy* 51, 1515--1554
   (2021), [doi:10.1007/s10686-021-09729-5](https://doi.org/10.1007/s10686-021-09729-5).
4. J. Chluba and R. A. Sunyaev, “The evolution of CMB spectral distortions in
   the early Universe,” *MNRAS* 419, 1294--1314 (2012),
   [doi:10.1111/j.1365-2966.2011.19786.x](https://doi.org/10.1111/j.1365-2966.2011.19786.x).
5. J. Chluba and D. Jeong, “Teasing bits of information out of the CMB energy
   spectrum,” *MNRAS* 438, 2065--2082 (2014),
   [doi:10.1093/mnras/stt2327](https://doi.org/10.1093/mnras/stt2327).
6. M. H. Abitbol et al., “Prospects for measuring CMB spectral distortions in
   the presence of foregrounds,” *MNRAS* 471, 1126--1140 (2017),
   [doi:10.1093/mnras/stx1653](https://doi.org/10.1093/mnras/stx1653).
7. F. Bianchini and G. Fabbian, “CMB spectral distortions revisited: a new take
   on mu distortions and primordial non-Gaussianities from FIRAS data,”
   *Physical Review D* 106, 063527 (2022),
   [doi:10.1103/PhysRevD.106.063527](https://doi.org/10.1103/PhysRevD.106.063527).
8. J. Chluba et al., “Evolution of CMB spectral distortion anisotropies and
   tests of primordial non-Gaussianity,” *MNRAS* 466, 2390--2401 (2017),
   [doi:10.1093/mnras/stw3230](https://doi.org/10.1093/mnras/stw3230).
9. M. Lucca et al., “The synergy between CMB spectral distortions and
   anisotropies,” *JCAP* 02, 026 (2020),
   [doi:10.1088/1475-7516/2020/02/026](https://doi.org/10.1088/1475-7516/2020/02/026).
10. X. Coulon, B. Maffei and N. Aghanim, “Towards measurements of CMB spectral
    distortions,” *EPJ Web of Conferences* 293, 00012 (2024),
    [doi:10.1051/epjconf/202429300012](https://doi.org/10.1051/epjconf/202429300012).
11. D. K. Adams et al., “CMB spectral distortions from resonant conversions in
    atomic dark sectors,” *Physical Review D* 114, 023537 (2026),
    [doi:10.1103/gpbc-6kr6](https://doi.org/10.1103/gpbc-6kr6).
12. S.-P. Li and J. Chluba, “Neutrinogenic CMB spectral distortions,”
    *Physical Review D* 113, 043504 (2026),
    [doi:10.1103/gl24-7pqt](https://doi.org/10.1103/gl24-7pqt).
13. H. Liu et al., “Exotic energy injection in the early universe II: CMB
    spectral distortions and constraints on light dark matter,”
    [arXiv:2303.07370](https://arxiv.org/abs/2303.07370).
14. G. Barenboim et al., “Parametrizations of spectral distortions and
    applications in cosmology,” *Physical Review D* 113, 063566 (2026),
    [doi:10.1103/dq6j-fzz6](https://doi.org/10.1103/dq6j-fzz6).
15. CNES/IAS doctoral project description, “Preparing for CMB spectral
    distortion measurement with space missions” (2026),
    [project description](https://recrutement.cnes.fr/en/annonce/4148993-26-212-preparing-for-cmb-spectral-distortion-measurement-with-space-missions-91405-orsay).
16. D. D. Kelson, “Optimal Techniques in Two-dimensional Spectroscopy:
    Background Subtraction for the 21st Century,” *PASP* 115, 688--699 (2003),
    [doi:10.1086/375502](https://doi.org/10.1086/375502).
17. R. I. Davies, “A method to remove residual OH emission from near-infrared
    spectra,” *MNRAS* 375, 1099--1105 (2007),
    [doi:10.1111/j.1365-2966.2006.11383.x](https://doi.org/10.1111/j.1365-2966.2006.11383.x).
18. A. Smette et al., “Molecfit: A general tool for telluric absorption
    correction. I. Method and application to ESO instruments,” *A&A* 576, A77
    (2015), [doi:10.1051/0004-6361/201423932](https://doi.org/10.1051/0004-6361/201423932).
19. K. T. Soto et al., “ZAP---Enhanced PCA Sky Subtraction for Integral Field
    Spectroscopy,” *MNRAS* 458, 3210--3220 (2016),
    [doi:10.1093/mnras/stw474](https://doi.org/10.1093/mnras/stw474).
20. A. Kogut and D. J. Fixsen, “Calibration Method and Uncertainty for the
    Primordial Inflation Explorer (PIXIE),” *JCAP* 05, 041 (2020),
    [doi:10.1088/1475-7516/2020/05/041](https://doi.org/10.1088/1475-7516/2020/05/041).
21. A. Kogut et al., “Systematic error mitigation for the PIXIE Fourier
    transform spectrometer,” *JCAP* 07, 057 (2023),
    [doi:10.1088/1475-7516/2023/07/057](https://doi.org/10.1088/1475-7516/2023/07/057).

## Related Torchsky documents

- [INR and Power Foam sky reconstruction](inr_foam_problem_formulation.md):
  detailed continuous-field and exact Foam formulation.
- [Continuous fields](../field.md): implemented field models, shared/private
  decompositions, observation operators and training diagnostics.
- [Implicit forward modelling](../implicit_forward.md): compact multi-survey
  prediction and robust-loss scaffold.
- [FOSSIL/BISOU Mission-Neutral Torchsky Track](fossil.md): implementation
  phases and current checkbox status.
