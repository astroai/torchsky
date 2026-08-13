# INR and Power Foam sky reconstruction: problem formulation and objectives

## Purpose

This note defines the inverse problem that Torchsky's implicit neural
representations (INRs) and spectral Power Foam models are intended to solve. It
is written for astronomers who know the basic supervised-learning pattern—fit a
model from inputs and labelled targets—but have not used continuous neural
fields or Power Foam geometry.

The central point is simple:

> An INR and a Power Foam are two different parameterizations of the same
> continuous sky. They should be compared through the same instrument model,
> the same detector samples, and the same scientific endpoints.

Our immediate experiment asks whether fitting all exposures and wavelengths
jointly can recover fainter sources than conventional per-band coaddition. The
claim is not that a model creates information. The hypothesis is that a shared
spatial-spectral representation can use real correlations between bands while
avoiding some information loss from resampling and early coaddition.

**Reader's path.** Sections 1--5 define the inverse problem and observation
model. Section 6 introduces INR and Foam representations. Sections 7--11 cover
calibration, flexibility, and identifiability. Sections 12--15 specify the
current experiment, claim gates, and numerical checks. Sections 16--17 describe
the longer-term component-separation generalization and the interpretation of
success or failure.

## 1. From supervised learning to an astronomical inverse problem

In ordinary supervised learning, a model receives an input \(x\), predicts a
target \(y\), and is trained on many known pairs \((x,y)\). Here the desired
high-resolution, noise-free, monochromatic sky is not available as a label.
Instead, we have many noisy observations produced from that sky by different
filters, point-spread functions (PSFs), detector pixels, dithers, backgrounds,
and masks.

We therefore use **analysis by synthesis**:

1. propose a continuous sky model;
2. pass it through a differentiable model of each observation;
3. compare the predicted detector pixels with the measured detector pixels;
4. update the sky and, only where identifiable, calibrated nuisance parameters.

This is self-supervised in the narrow sense that the observations supervise
their own reconstruction. It is more usefully described as a likelihood-based
inverse problem, because the forward physics and noise model determine what the
loss means.

No source catalogue is required to fit the scene. A catalogue can be useful
later for photometry, validation, or scientific selection. For a controlled
depth experiment, injected sources provide known positions and fluxes, while a
common detection procedure is applied to every reconstructed product.

## 2. Data, coordinates, and unknowns

Index observations by \(n=1,\ldots,N\) and pixels in observation \(n\) by
\(p\). Let

- \(y_{np}\) be the calibrated measured pixel value;
- \(m_{np}\in[0,1]\) be its hard or soft validity mask;
- \(w_{np}\geq 0\) be its statistical weight, normally inverse variance;
- \(\boldsymbol u\) be a continuous detector-plane coordinate;
- \(\boldsymbol x\) be a two-dimensional sky coordinate in a local tangent
  plane, or a point on the celestial sphere for a wide field;
- \(\lambda\) be wavelength;
- \(t_n\) and \(c_n\) be the exposure time and other observing context.

The principal unknown is a continuous specific-intensity field

\[
I(\boldsymbol x,\lambda),
\]

or, when variability is part of the science,

\[
I(\boldsymbol x,\lambda,t).
\]

For the current stacking experiment the astronomical scene is static and time
dependence belongs to the observing operator. The field may be expressed in
surface-brightness units such as MJy sr\(^{-1}\), provided the response and
calibration conventions are consistent. Raw electron or count-rate modelling
is also possible, but it requires the corresponding gain, exposure-time,
throughput, and noise terms.

We denote all scene parameters by \(\theta\), all calibrated or learnable
observation parameters by \(\psi\), and the prediction for an observation by

\[
\boldsymbol\mu_n = \mathcal A_n(I_\theta;\psi_n)+\boldsymbol B_n(\psi_n).
\]

Here \(\mathcal A_n\) is the calibrated observation operator. It contains the
WCS, spectral and spatial response, pixel footprint, photometric calibration,
and any other effect that acts before the declared additive background
\(\boldsymbol B_n\). Its components may be analytic, learned from calibration
and repeated observations, or hybrids of both. They must retain explicit units,
coordinate dependence, ordering, and validation contracts.

## 3. The continuous observation model

### 3.1 A physically explicit linear special case

Let \(W_n(\boldsymbol u)\) map detector coordinates to sky coordinates, and let
\(\Omega^{\rm sky}_{np}=W_n(\Omega^{\rm det}_{np})\) be the corresponding sky
footprint with solid angle \(|\Omega^{\rm sky}_{np}|\). Let
\(R_n(\lambda)\) be the complete spectral response, including any factors
needed to convert a source spectrum to the convention used by the calibrated
image. Let \(\mathcal P_n(\lambda,t_n,c_n)\) be a spatial-response operator acting
on the two-dimensional intensity slice \(I_\theta(\cdot,\lambda)\). It may be a
shift-invariant convolution, a position-dependent integral operator, a
differentiable optical/detector model, a learned operator, or a hybrid.
A surface-brightness prediction can be written schematically as

\[
\mu_{np} =
\frac{g_n}{|\Omega^{\rm sky}_{np}|}
\int_{\Omega^{\rm sky}_{np}}\!\mathrm d\Omega_{\boldsymbol x}
\int\!\mathrm d\lambda\,R_n(\lambda)
\left[\mathcal P_n(\lambda,t_n,c_n) I_\theta(\cdot,\lambda)\right]
\!(\boldsymbol x)
+ B_{np}(t_n,c_n).
\]

The division by sky solid angle is appropriate when predicting a pixel-averaged
surface brightness. It is omitted when \(y_{np}\) is an integrated pixel flux.
Equivalently, the integral can be performed in detector coordinates
\(\boldsymbol u=(u_x,u_y)\) by including the solid-angle Jacobian. If
\(\widehat{\boldsymbol n}_n(\boldsymbol u)\) is the unit sky direction, its
coordinate-invariant form is

\[
J_{\Omega,n}(\boldsymbol u)
=\left\|
\frac{\partial\widehat{\boldsymbol n}_n}{\partial u_x}
\mathbin{\times}
\frac{\partial\widehat{\boldsymbol n}_n}{\partial u_y}
\right\|.
\]

Away from coordinate singularities, for right ascension \(\alpha\) and
declination \(\delta\) expressed in radians, this is equivalently

\[
J_{\Omega,n}(\boldsymbol u)
=|\cos\delta(\boldsymbol u)|
\left|\det
\frac{\partial(\alpha,\delta)}{\partial(u_x,u_y)}
\right|.
\]

The pixel solid angle is therefore

\[
|\Omega^{\rm sky}_{np}|
=\int_{\Omega^{\rm det}_{np}}
J_{\Omega,n}(\boldsymbol u)\,\mathrm d^2\boldsymbol u.
\]

This is not a cosmetic convention: data, weights, injected fluxes, WCS
distortions such as TAN-SIP or TPV, and the forward operator must all use the
same semantics. The compact \(B_{np}\) term denotes the correspondingly
integrated or averaged additive background.

The placement of \(B_{np}\) is part of the model, not notation for every
unwanted signal. Diffuse celestial light and optical stray light that pass
through the spatial response belong inside the appropriate response operator;
detector offsets added later may belong in \(B_{np}\). In count space, photon
variance depends on the total incident source, sky, telescope, and instrument
loading before any fitted mean background is subtracted. A background model can
remove an expectation value, but not the photons' shot noise or the uncertainty
of the fitted background.

The response may be normalized so that
\(\int R_n(\lambda)\,\mathrm d\lambda=1\), in which case the spectral integral
is a band-averaged intensity. For photon-counting data, factors such as
\(\lambda/(hc)\) can instead be included in \(R_n\). Torchsky should record the
chosen convention rather than silently mix the two.

This form assumes that \(\mathcal P_n\) is linear in incident intensity when its
calibration parameters are fixed. The general operator \(\mathcal A_n\) need not
be: signal-dependent detector response, saturation, persistence, charge
transport, or other nonlinear effects can be represented by differentiable
analytic or learned stages. Calling \(\mathcal P_n\) a spatial response rather
than a convolution is intentional; a single compact kernel is only one possible
approximation.

### 3.2 What each operator contributes

The forward model prevents the scene representation from absorbing known
instrument effects:

- **WCS:** associates every detector footprint with a sky region. It must
  preserve the adopted pixel-centre and axis conventions.
- **Passband:** integrates one common spectral field through each filter. A
  nominal central wavelength alone is generally insufficient for a broad band.
- **Spatial response:** maps incident sky structure into detector structure
  before pixel integration. A conventional PSF convolution is the
  shift-invariant case. Field-position-, wavelength-, time-, or detector-state-
  dependent responses require a more general operator, with flux or throughput
  accounting stated explicitly.
- **Pixel integration:** averages or integrates over the finite detector
  footprint; point sampling is biased for undersampled structure.
- **Calibration:** converts the physical prediction into the units of the data.
- **Background:** adds sky, telescope/instrument, and detector terms in their
  appropriate coordinate frames.
- **Mask and noise:** state which residuals carry information and how much.

The ideal equation above is one useful scientific contract, not a requirement
that every response have a closed form. A practical implementation may use
quadrature, a compact PSF kernel, a wave-optics calculation, a learned
non-stationary response, or a physical model with a learned residual correction.
Its approximation class, training evidence, support, and invariants must be
declared and tested against calibration and held-out observations. In
particular, Torchsky's current exact
fixed-geometry Foam solver can compose a supplied linear pixel-domain PSF
forward/adjoint pair. That is exact for the declared discrete operator, but is
not by itself proof of an exact continuous spatial response.

## 4. Likelihood and training objective

For independent Gaussian pixel errors with known variances, the negative log
likelihood, up to constants, is

\[
\mathcal L_{\mathrm{data}}(\theta,\psi)
= \frac{1}{2}\sum_{n,p}
m_{np}w_{np}\left(y_{np}-\mu_{np}(\theta,\psi)\right)^2.
\]

If the weights are true inverse variances, this is a \(\chi^2/2\) objective.
If they have been rescaled or clipped, it remains a weighted least-squares
objective but no longer has an automatic probabilistic calibration. Robust
Charbonnier or Huber losses can reduce sensitivity to unmasked outliers. A
Poisson-Gaussian likelihood is preferable when working close to raw counts.
Likewise, a soft mask is an analysis weight rather than missing-data
conditioning unless a probabilistic selection model gives it that meaning.

For correlated Gaussian noise with a fixed positive-definite covariance
\(\boldsymbol N\) on the valid-pixel subspace, replace the diagonal sum by

\[
\mathcal L_{\rm data}
=\frac12\boldsymbol r^{\mathsf T}\boldsymbol N^{-1}\boldsymbol r,
\qquad
\boldsymbol r=\boldsymbol y-\boldsymbol\mu(\theta,\psi).
\]

If \(\boldsymbol N(\eta)\) is fitted from the same data, a normalized Gaussian
objective must also contain
\(\tfrac12\log\det\boldsymbol N(\eta)\); otherwise the fit can reduce the
quadratic term by inflating its covariance. A learned whitening rule that does
not define a normalized likelihood must instead be trained on independent or
cross-fitted evidence and calibrated on held-out residuals.

The complete objective, allowing an optional learned noise model, is

\[
(\hat\theta,\hat\psi,\hat\eta)
= \underset{\theta,\psi,\eta}{\operatorname{argmin}}
\left[
\mathcal L_{\mathrm{data}}(\theta,\psi,\eta)
+ \sum_q \alpha_q\mathcal R_q(\theta)
+ \sum_r \beta_r\mathcal C_r(\psi)
+ \mathcal C_{\rm noise}(\eta)
\right].
\]

The \(\mathcal R_q\) terms encode declared scene priors or regularization, such
as positivity, limited spectral complexity, spatial smoothness, or proximity
to an initializer. The \(\mathcal C_r\) terms anchor nuisance models to
calibration evidence. Every regularizer changes the estimator and can bias
faint-source flux, so its coefficient must be selected on validation data, not
on the blind scientific endpoint. When the covariance is fixed, omit \(\eta\)
and \(\mathcal C_{\rm noise}\); when it is learned, the calibration term does
not replace the likelihood normalization above.

Resampled images usually have correlated noise even if their input detector
pixels did not. One reason to fit native pixels is to postpone that correlation
and information-loss problem. It does not make the calibrated response or
nuisance prediction exact; their uncertainty is addressed in Section 10.

## 5. What is an implicit neural representation?

An **implicit neural representation** stores a signal in the parameters of a
function rather than in a fixed array of pixel values. In its simplest form,

\[
I_\theta(\boldsymbol x,\lambda)
= f_\theta\!\left(E(\boldsymbol x,\lambda)\right),
\]

where \(E\) encodes continuous coordinates and \(f_\theta\) is a neural
network. Querying the function at arbitrary coordinates produces an intensity.
There is no intrinsic output pixel grid.

Raw coordinates are difficult for a small multilayer perceptron to use for
fine structure. Practical encoders therefore provide multi-scale features,
for example sinusoidal/Fourier features, learned feature planes, or hash-grid
features. Torchsky's INR family includes both conventional coordinate MLPs and
models with learned multi-resolution spatial-spectral feature planes followed
by a compact decoder. They share the continuous query contract even though
their internal parameterizations differ.

### 5.1 Joint and independent INR controls

A joint INR receives spatial and spectral coordinates and shares parameters
across bands. An independent control must be able to represent arbitrary
per-band appearances without cross-band spectral constraints. Merely training
the joint model for fewer steps is not an independent control.

Useful comparisons include:

- independent per-band models;
- one model with an explicit full-rank band embedding;
- a joint model with a lower-dimensional continuous spectral representation.

Capacity, training samples, optimization steps, and compute should be reported
separately. Matching parameter count alone does not match optimization cost or
spatial resolution.

### 5.2 INR advantages and risks

An INR can represent smooth continuous structure with a relatively small
number of parameters, evaluate at arbitrary output coordinates, and couple
space and wavelength flexibly. Automatic differentiation makes it easy to
optimize through the observation model.

Its risks are equally important:

- the network's frequency bias can suppress compact or sharp structure;
- excessive capacity can fit noise or detector artefacts;
- insufficient spectral flexibility can force unusual colours toward common
  ones;
- training is non-convex and can depend on initialization and scheduling;
- a smooth-looking reconstruction is not evidence of correct photometry;
- uncertainty is not supplied automatically by a deterministic network.

These risks are why injected-source response, flux bias, and held-out exposure
prediction are required in addition to reconstruction loss.

## 6. What is a spectral Power Foam?

A **Power Foam** represents the sky with adaptive geometric cells rather than
with a regular pixel grid or a global neural decoder. The cells are a bounded
form of a power diagram, which is a weighted Voronoi diagram.

### 6.1 Hard bounded-power geometry

For site \(\boldsymbol s_j\) and radius \(r_j\geq0\), define the power distance

\[
d_j(\boldsymbol x)
= \|\boldsymbol x-\boldsymbol s_j\|_2^2-r_j^2.
\]

The hard bounded cell is

\[
C_j = \left\{
\boldsymbol x:
d_j(\boldsymbol x)\leq d_k(\boldsymbol x)\ \text{for every }k,
\quad d_j(\boldsymbol x)\leq0
\right\}.
\]

The first condition assigns a point to its smallest power distance. The second
intersects that ordinary power cell with the site's radius-\(r_j\) disk. The
union need not cover the entire plane; uncovered regions have zero Foam signal
and must not be mistaken for measured blank sky. Production geometry should
cover the declared target region, and support must be propagated to outputs.

The radii affect both support and the boundaries between cells. Unlike a
standard Voronoi tessellation, a geometrically larger weight can win territory
even when its site is not the nearest in Euclidean distance.

### 6.2 Spatial-spectral appearance inside a cell

Let \(\phi_k(\lambda)\), \(k=1,\ldots,K\), be spectral basis functions. The
current spectral affine Foam uses

\[
a_j(\lambda)=\sum_{k=1}^{K}c_{jk}\phi_k(\lambda)
\]

for the value at site \(j\), and optionally

\[
\boldsymbol g_j(\lambda)
=\sum_{k=1}^{K}\boldsymbol h_{jk}\phi_k(\lambda)
\]

for a local spatial gradient. The hard field is

\[
I(\boldsymbol x,\lambda)
= a_j(\lambda)
+ \boldsymbol g_j(\lambda)^\mathsf T
(\boldsymbol x-\boldsymbol s_j),
\qquad \boldsymbol x\in C_j.
\]

With \(\boldsymbol g_j=0\), the field is piecewise constant. The affine term is
the minimum local extension that can preserve a slope across a cell. It does
not make the field continuous at cell boundaries; continuity would be an
additional modelling constraint, not an automatic property.

Torchsky currently uses normalized local-tangent coordinates and a Legendre
spectral basis. Normalization keeps the numerical scales of spatial and
spectral coefficients controlled. Other bases are possible when the data
justify them.

### 6.3 Shared and independent spectral appearance

For observation \(n\), define its integrated spectral design row

\[
b_{nk}=\int R_n(\lambda)\phi_k(\lambda)\,\mathrm d\lambda.
\]

A low-rank choice \(K<B\), where \(B\) is the number of bands, forces the band
appearances to share a lower-dimensional spectral description. This is the
mechanism by which a spectral Foam can borrow information across wavelengths.

A spectrally complete independent-appearance control on the same geometry uses
\(K=B\) and requires the response matrix \(\boldsymbol B=[b_{nk}]\) to have
full numerical rank over the distinct bands. Whitening the columns can improve
conditioning without reducing their span. A six-mode model is not independent
for six bands if its integrated response matrix has rank below six. This rank
condition establishes representational completeness in band space, not general
statistical identifiability. The latter also depends on the complete
noise-weighted observation operator, its conditioning, the shared geometry, and
overlap with nuisance components.

This control isolates **spectral sharing**, but it still shares the cell
geometry. A completely independent control would also fit separate geometry in
each band. Both comparisons are informative and should be named accurately.

### 6.4 Differentiable geometry training

Hard cell membership changes discontinuously when a boundary crosses a query
point. To train sites and radii with gradient descent, Torchsky uses a smooth
surrogate. In simplified notation,

\[
q_j(\boldsymbol x)
=\operatorname{softmax}_j
\left[-\tau_p d_j(\boldsymbol x)
+\log\sigma\!\left(-\tau_s d_j(\boldsymbol x)\right)\right],
\]

\[
u(\boldsymbol x)
=1-\prod_j\sigma\!\left(\tau_s d_j(\boldsymbol x)\right),
\qquad
\widetilde w_j(\boldsymbol x)=u(\boldsymbol x)q_j(\boldsymbol x).
\]

Here \(\sigma\) is the logistic function. The factor \(q_j\) gives smooth
conditional ownership and \(u\) gives the smooth occupancy of the union of
bounded cells. For fixed signed distances and in the limit
\(\tau_s\to\infty\), a query point inside cell \(j\)
(\(d_j(\boldsymbol x)<0\)) has \(\sigma(\tau_s d_j(\boldsymbol x))\to0\), so
\(u(\boldsymbol x)\to1\). A point outside every cell
(\(d_k(\boldsymbol x)>0\) for all \(k\)) has every logistic factor tending to
one, so \(u(\boldsymbol x)\to0\). At finite temperature these are smooth
approximations rather than exact equalities. Increasing the positive
temperatures \(\tau_p\) and \(\tau_s\) makes transitions progressively sharper.

This surrogate exists for optimization. Because both the softmax and union
occupancy couple all cells, changing \(d_j\) changes every
\(\widetilde w_k\), not only \(\widetilde w_j\). The complete geometry gradients
are

\[
\frac{\partial \mathcal L}{\partial \boldsymbol s_j}
= \sum_{\boldsymbol x}\sum_k
\frac{\partial \mathcal L}
{\partial \widetilde w_k(\boldsymbol x)}
\frac{\partial \widetilde w_k(\boldsymbol x)}
{\partial d_j(\boldsymbol x)}
\left[-2(\boldsymbol x-\boldsymbol s_j)\right],
\qquad
\frac{\partial \mathcal L}{\partial r_j}
= \sum_{\boldsymbol x}\sum_k
\frac{\partial \mathcal L}
{\partial \widetilde w_k(\boldsymbol x)}
\frac{\partial \widetilde w_k(\boldsymbol x)}
{\partial d_j(\boldsymbol x)}
\left[-2r_j\right].
\]

Equivalently, automatic differentiation first accumulates
\(\partial\mathcal L/\partial d_j\) through every ownership weight and then
applies the two derivatives of \(d_j\) shown above.

Final products must use detached hard geometry and exact or explicitly
controlled pixel integration, because a soft boundary can change source flux
and effective resolution.

### 6.5 Exact hard pixel operator

For a sky-projected pixel footprint \(\Omega_{np}\), define the intersection
area and first moment

\[
A_{npj}=\int_{\Omega_{np}\cap C_j}\!\mathrm d^2\boldsymbol x,
\qquad
\boldsymbol M_{npj}
=\int_{\Omega_{np}\cap C_j}
(\boldsymbol x-\boldsymbol s_j)\,\mathrm d^2\boldsymbol x.
\]

In practice, Torchsky first clips a convex local-tangent pixel polygon by the
cell's radical half-planes. It then integrates the remaining polygon against
the radius-\(r_j\) disk using analytic straight-segment and circular-arc area
and first-moment contributions. Ordinary polygon clipping alone is not exact
because the bounded cell can have circular arcs. Curved WCS pixel edges must be
adaptively tessellated to a declared error budget; only the resulting
tangent-plane polygon/disk intersection is analytic.

At fixed geometry, the integrated, pre-PSF Foam prediction is linear in the
appearance coefficients:

\[
z_{np}
=\sum_j\left[
A_{npj}\sum_k b_{nk}c_{jk}
+\boldsymbol M_{npj}^{\mathsf T}
\sum_k b_{nk}\boldsymbol h_{jk}
\right].
\]

Dividing \(z_{np}\) by the projected pixel area gives pixel-averaged surface
brightness. The sparse intersections are the exact geometry component of the
operator. A PSF or other linear pixel operator can then be composed through a
matched forward/adjoint pair.

For fixed sites and radii, appearance refinement is a regularized linear least-
squares problem:

\[
\hat{\boldsymbol c}
=\underset{\boldsymbol c}{\operatorname{argmin}}
\left\{
\frac12\sum_n
\left\|\boldsymbol W_n^{1/2}
(\boldsymbol y_n-\boldsymbol H_n\boldsymbol F_n\boldsymbol c)
\right\|_2^2
+\frac{\gamma}{2}
\left\|\boldsymbol D(\boldsymbol c-\boldsymbol c_0)\right\|_2^2
\right\}.
\]

The resulting normal equations can be solved matrix-free with preconditioned
conjugate gradients (PCG) when the regularized normal operator is positive
definite:

\[
\left(
\sum_n \boldsymbol F_n^\mathsf T\boldsymbol H_n^\mathsf T
\boldsymbol W_n\boldsymbol H_n\boldsymbol F_n
+\gamma\boldsymbol D^\mathsf T\boldsymbol D
\right)\boldsymbol c
= \sum_n \boldsymbol F_n^\mathsf T\boldsymbol H_n^\mathsf T
\boldsymbol W_n\boldsymbol y_n
+\gamma\boldsymbol D^\mathsf T\boldsymbol D\boldsymbol c_0.
\]

Here \(\boldsymbol F_n\) contains the exact footprint and passband terms,
\(\boldsymbol H_n\) is the declared linear spatial-response operator,
\(\boldsymbol W_n\) contains pixel weights, and \(\boldsymbol D\) scales value
and slope parameters comparably. A cold solve uses \(\boldsymbol c_0=0\). A
post-fit refinement uses the supplied initializer as \(\boldsymbol c_0\), so
the ridge does not pull source flux toward zero merely because refinement was
requested.

With \(\gamma\geq0\) and positive-semidefinite weights, the normal operator is
symmetric and positive semidefinite. Standard PCG additionally requires the
operator to be positive definite. For \(\gamma>0\), this is equivalent to

\[
\ker\!\left(\begin{bmatrix}
\boldsymbol W_1^{1/2}\boldsymbol H_1\boldsymbol F_1\\
\vdots\\
\boldsymbol W_N^{1/2}\boldsymbol H_N\boldsymbol F_N
\end{bmatrix}\right)
\cap\ker(\boldsymbol D)=\{\boldsymbol 0\}
\]

when the displayed operators have the declared finite-dimensional domains. For
\(\gamma=0\), the stacked data operator itself must have a trivial null space.
PCG also requires a symmetric positive-definite preconditioner on the solved
space. If the applicable condition fails, the implementation must constrain the
null space or solve the augmented least-squares problem with a method valid for
a rank-deficient system, such as LSQR or LSMR. A small residual alone must not
be interpreted as a unique coefficient solution.

Sparse exposure operators can remain on the host and be transferred in bounded
observation batches. In exact arithmetic and at convergence, a valid diagonal
or per-cell block preconditioner changes the iteration path, convergence rate,
and memory use, but not the unique solution. At finite tolerance and precision,
the returned coefficients can differ. Reports must therefore include the
stopping rule and residual norm and verify that scientific metrics are stable
under a tighter tolerance.

### 6.6 Foam advantages and risks

Power Foam offers explicit local support, adaptive spatial allocation, exact
polygon/cell integration at fixed geometry, and a fixed-geometry appearance
problem that is linear. Its geometry can concentrate degrees of freedom where
the scene needs them rather than everywhere on a regular grid.

Its main risks are:

- poor site placement can leave holes or waste most cells outside the target;
- cells that are too large attenuate compact sources;
- soft training and hard rendering can disagree;
- affine cells can still have discontinuous boundaries;
- joint geometry optimization is non-convex;
- exact candidate geometry and sparse operators can dominate memory or setup
  time on very large fields;
- current binary support is not a propagated uncertainty map.

The representation should therefore be judged by source response and exact
hard products, not by a low smooth-surrogate loss alone.

## 7. INR and Foam solve the same inverse problem

The following pieces must be identical in a fair comparison:

| Shared element | Why it must be shared |
| --- | --- |
| input exposures and frozen train/validation/test split | prevents data leakage and unequal information |
| masks and inverse variances | gives each detector pixel the same statistical role |
| WCS and finite pixel footprints | prevents one method receiving easier resampled geometry |
| passbands and spectral convention | makes “joint wavelength modelling” mean the same thing |
| PSF contract | prevents resolution or flux-calibration differences masquerading as depth |
| background contract | prevents the scene from winning by fitting a different nuisance model |
| calibration evidence and uncertainty treatment | prevents one method receiving a better-known response or treating fitted nuisance as exact |
| injected sources and null controls | makes completeness and purity paired measurements |
| output grid and detection/photometry code | prevents the scorer from favouring one representation |

Only the parameterization of \(I_\theta\), its declared priors, and its
optimization algorithm should differ. Results should report parameter count,
sampled detector pixels, optimization steps, wall time, host memory, and
accelerator memory. There is no single perfect notion of “matched compute,” so
we use both capacity-matched and spatial-resolution-matched controls.

## 8. Classical coadds and what joint fields change

Drizzle and SWarp first map each exposure to an output grid and combine the
resampled values. They are strong operational baselines, not straw figures.
Drizzle can retain much of the dither information when its kernel, scale, and
`pixfrac` are chosen well. SWarp provides mature interpolation and weighted-
coaddition choices. A multiband \(\chi^2\) or matched-filter detector can then
combine their per-band products. A properly constructed linear coadd
accompanied by its effective PSF and covariance can also be sufficient for some
linear, static-sky measurements. The comparison must therefore include both
the named operational products and a native-detector or proper-coadd
information reference. Any gain over Drizzle or SWarp alone could otherwise
measure suboptimal resampling, PSF handling, or detection rather than an
advantage of INR or Foam.

INR and Foam instead estimate one latent scene directly from all selected
detector pixels:

\[
\{\boldsymbol y_n\}_{n=1}^{N}
\longrightarrow I_\theta(\boldsymbol x,\lambda)
\longrightarrow \{\widehat{\boldsymbol y}_n\}_{n=1}^{N}.
\]

They can delay choosing an output grid, avoid repeated interpolation, and use a
shared spectral prior during reconstruction rather than only during detection.
These are possible advantages, not guaranteed ones. A well-tuned classical
pipeline may be statistically preferable when the learned prior is wrong or
the forward model is misspecified.

## 9. Why a joint spectral model might reach deeper

Suppose a faint source is individually below the detection threshold in every
band, but its spectral energy distribution lies near a low-dimensional family.
A joint model can use all compatible photons to estimate a smaller set of
shared latent quantities. In bias-variance language, the spectral constraint
can lower estimator variance at the cost of some bias.

The expected gain depends on four conditions:

1. the bands contain genuinely correlated astronomical signal;
2. the spectral representation includes the source population being tested;
3. the WCS, PSF, calibration, background, and noise models are accurate enough;
4. regularization reduces noise more than it attenuates source flux.

The model does **not** create photons, and “using six bands” does not by itself
imply a six-fold gain. Sources with strong emission lines, breaks, unusual
colours, or band-limited variability can be harmed by a smooth low-rank prior.
This is why independent-appearance controls and source-population strata are
part of the experiment.

### 9.1 Information reference and effective sensitivity

For a locally linear Gaussian problem with a fixed unit-amplitude detector
template \(\boldsymbol q\) and complete fixed predictive covariance
\(\boldsymbol N\) (or \(\boldsymbol V\) when Section 10's nuisance and response
terms apply), the Fisher information for its amplitude is

\[
\mathcal I_a
=\boldsymbol q^{\mathsf T}\boldsymbol N^{-1}\boldsymbol q,
\qquad
\operatorname{Var}(\widehat a)\geq \mathcal I_a^{-1}
\]

for an unbiased estimator under the declared model. This is an information
reference, not a promise that a real nonlinear reconstruction attains the
bound. A regularized estimator can reduce raw variance by attenuating the
target. If paired injections give local response
\(R_a=\partial\mathbb E[\widehat a]/\partial a\), compare methods using the
response-corrected sensitivity

\[
\sigma_{a,{\rm eff}}
=\frac{\sqrt{\operatorname{Var}(\widehat a)}}{|R_a|}
\]

together with bias and nonlinearity. A method that reports smaller variance
only because \(|R_a|<1\) has not reached deeper. For a population-level spectral
prior, the classical detector must receive the same preregistered template
family or mixture weights; otherwise the experiment confounds reconstruction
with prior information available only to the learned method.

## 10. Flexible mechanistic hybrids

PSF and background models may vary spatially, spectrally, and temporally on
different scales. Torchsky provides adaptive coordinate encoders and
conditioned neural PSF/background models for this purpose. Their flexibility
does not remove the need for calibration.

A central Torchsky design goal is to turn useful but incomplete physical
approximations into mechanistic hybrid models. Instead of requiring a static
Gaussian or Moffat PSF, or a low-order background polynomial, a hybrid keeps
the physical contract while learning the structure that calibration data can
identify:

- **Mechanistic foundation:** evaluate the declared WCS and passband operators
  to numerical tolerance; preserve radiometric semantics, causal ordering,
  pixel-footprint integration, and matched linear adjoints; and enforce the
  appropriate conservation or throughput law for each stage. The physical WCS
  and calibration parameters still carry uncertainty.
- **Flexible learned substrate:** represent calibration-supported
  spatial-spectral-temporal response variation, focal-plane aberrations, stray
  light, backgrounds, and detector non-uniformities with continuous neural
  fields, adaptive coordinate encoders, or conditioned kernel banks.

Simple physical approximations serve as useful initializers and baselines, not a
ceiling on model fidelity. With sufficiently diverse calibration, repeated
sources, dithers, wavelengths, times, and detector states, the data may support
a learned spatial response, background, spectral response, noise model, or
residual correction that is more accurate than a fixed functional form. A
hybrid retains the known geometry, radiometry, conservation laws, and
causal ordering while learning calibration-supported residual structure.

Model flexibility is therefore selected from evidence rather than fixed in
advance. Capacity sweeps should compare closed-form, hybrid, and learned models
on held-out calibration and exposure folds. Increasing capacity is accepted
only when it improves out-of-fold prediction without violating unit,
positivity, support, flux/throughput, adjoint, or injection-response checks
applicable to that operator.

Unrestricted joint learning is non-identifiable:

- a broad source can be exchanged between scene morphology and PSF width;
- diffuse sky emission can be exchanged with additive background;
- an astrometric offset can be exchanged with source position;
- passband or gain errors can be exchanged with source colour.

For a Gaussian-width PSF special case, one possible calibration anchor is

\[
\mathcal C_{\rm width}(\psi)
=\frac12\sum_n
\frac{[\sigma_{{\rm PSF},n}(\psi)-\sigma_{{\rm cal},n}]^2}
{v_{\sigma,n}},
\]

where \(v_{\sigma,n}\) is the calibration variance. This constrains one width
summary; it does not identify a general PSF or break every source/response
degeneracy. A flexible spatial response instead needs direct evidence-space
terms for calibrated kernels or detector responses, with held-out centroid,
encircled-energy, asymmetry, halo, wavelength, time, and focal-plane checks.

The signal model and nuisance model need not have the same form or capacity.
The astronomical signal may use a known template family, a flexible INR/Foam,
or an open residual, while nuisance terms may use anything from a physical
formula to a large neural field. What matters is which evidence identifies each
term. A nuisance model trained on the same samples used to measure the signal
can erase that signal as its capacity grows. Conversely, an underfit nuisance
model can leave structured residuals that mimic a discovery. Report the full
capacity path: held-out likelihood, target attenuation and cross-talk from
paired injections, target/nuisance tangent overlap where defined, architecture
sensitivity, and the stable unexplained residual.

The safe sequence is therefore staged:

1. initialize the WCS, passbands, detector scale, and PSF support from
   instrument metadata;
2. fit background variation to source-excluded or blank-sky evidence;
3. fit PSF variation to isolated calibration stars or injected calibration
   responses;
4. fit the astronomical scene with nuisance models fixed or strongly anchored;
5. allow joint refinement, including high-capacity learned corrections, only
   when held-out calibration and exposure metrics improve and target-response
   gates remain acceptable.

Calibration anchoring does not make a fitted response known exactly. Let
\(\boldsymbol C_n\) be the predictive covariance of a held-out background or
other nuisance prediction, and let
\(\boldsymbol J_{\psi,n}\boldsymbol C_\psi
\boldsymbol J_{\psi,n}^{\mathsf T}\) propagate independently estimated response
uncertainty. When those terms are conditionally independent of detector noise,
the held-out residual covariance is

\[
\boldsymbol V_n
=\boldsymbol N_n+\boldsymbol C_n
+\boldsymbol J_{\psi,n}\boldsymbol C_\psi
 \boldsymbol J_{\psi,n}^{\mathsf T}.
\]

If they are not independent, the corresponding cross-covariances are required.
Joint nonlinear fits may instead profile or marginalize calibration and
nuisance parameters, or use a validated approximation. Optimizer seeds or a
network ensemble probe fit instability; they do not by themselves represent
calibration uncertainty. Predictive intervals and flux intervals must achieve
their declared coverage on held-out calibration data, paired injections, and,
where available, repeated independent fields.

Adaptive scales should be inferred from calibration-coordinate coverage and
held-out evidence, not from the blind depth sample. The current UPDec endpoint
deliberately uses a fixed, mildly misspecified Gaussian fitted PSF against a
known injected PSF. This isolates reconstruction depth before introducing a
more ambitious learned-PSF result.

## 11. Identifiability and validation splits

A low training residual is insufficient because a flexible model can absorb
noise, backgrounds, or calibration errors. We need data that did not update the
model.

For the current SMACS J0723 experiment, the frozen dither split is

- training: `PATT_NUM={1,2,4,5,7}`;
- validation: `PATT_NUM={3,8}`;
- blind test: `PATT_NUM={6,9}`.

Validation exposures select architecture, spectral rank, atom density,
learning rate, temperature schedule, regularization, stopping rule, and
detection threshold. Test exposures and blind injection identifiers are used
once for the final scientific measurement. A failure on the blind test changes
the claim, not the configuration.

These exposure folds share one celestial field and often common calibration.
They test prediction into withheld detector realizations conditional on that
field; they are not independent sky realizations and do not establish field-to-
field generalization. Pixels, normalization statistics, calibration updates,
background fits, PSF fits, and learned whitening parameters associated with a
held-out exposure must all remain outside the corresponding training fold. If
several architectures or method variants are inspected on the blind test, it
is no longer a one-use test; the complete selection rule must be frozen or its
multiplicity calibrated.

Useful diagnostics include:

- weighted residuals in native held-out exposures;
- residual autocorrelation and structure around sources;
- flux conservation under known injections;
- response to negative, shuffled-band, and blank controls;
- forward/adjoint dot-product checks for every linear operator;
- results stratified by band, magnitude, colour, crowding, and field quadrant.

## 12. Our immediate quantitative objective

The current primary endpoint is detector-level source recovery in the six JWST
NIRCam bands F090W, F150W, F200W, F277W, F356W, and F444W. All methods receive
the same aligned CRF observations, masks, inverse variances, WCS metadata,
passbands, PSF contract, injections, and output grids.

The scalar depth coordinate must be declared before generating the blind
injections. Use a reference-band AB magnitude \(m_{\rm ref}\), or an equivalent
amplitude normalization of a preregistered spectral template, and define every
other band through the injection's colour or spectral parameters. There is no
single multiband AB magnitude without such a convention. Report separate
recovery curves for declared colour/SED, crowding, and morphology strata rather
than hiding a population-dependent gain in one mixture average.

Inject sources in the latent scene and propagate them through passband, WCS,
spatial response, pixel integration, calibration, and masking before adding
them to detector data. When count-level source noise is material, positive
injections also require the corresponding signal-dependent noise draw; paired
negative injections remain local-response probes rather than physical skies.
To measure reconstruction response, rerun every fitting and selection stage
that would respond to the added source. Adding a template only to an already
frozen reconstruction measures the detector/scorer response, not the INR or
Foam pipeline response, and must be labelled separately.

Freeze the validation-selected detection threshold before examining the blind
test. Apply the same threshold-selection rule separately to each method; do not
force one numerical score threshold onto differently calibrated scorers. At the
resulting threshold, chosen to target 95% validation purity, define
\(m_{50}\) on the test set as the value of \(m_{\rm ref}\) at which the fitted
per-source recovery probability is 50%. The recovery model, monotonicity
constraint, matching radius or probabilistic association rule, blend policy,
and treatment of genuine but uncatalogued objects must be frozen with the
threshold. If the fitted curve does not bracket 50% over the injected range,
report a bound rather than extrapolating an \(m_{50}\). Do not recalibrate the
threshold on test detections. Because a larger AB magnitude is fainter, the
primary contrast must name one preregistered SED/morphology stratum or a fixed
population mixture and is

\[
\Delta m_{50}
=m_{50}(\text{shared spectral Foam})
-m_{50}(\text{best classical multiband pipeline}).
\]

The desired claim requires all of the following, not just
\(\Delta m_{50}>0\):

- the paired 95% injection-batch bootstrap interval for \(\Delta m_{50}\)
  excludes zero;
- measured test purity is at least 95%, with a lower 95% cluster-bootstrap
  bound of at least 93%;
- median absolute bright-source flux bias is at most 2% in every band and no
  more than one percentage point worse than the best classical method;
- adjacent-band median colour bias is at most 0.02 mag for the declared
  smooth-continuum population;
- held-out exposure predictive score improves; for a loss-like score where
  smaller is better, the alternative non-inferiority gate is that the upper
  95% confidence bound on shared minus independent score does not exceed a
  preregistered margin \(\delta_{\rm pred}\);
- the gain appears in at least two of four blind 256-pixel quadrants as a
  heterogeneity diagnostic, not as independent replication;
- blank, negative-injection, and shuffled-band controls pass.

Use the same injection-batch identifiers for every method and resample those
identifiers jointly 10,000 times, refitting the declared recovery curve in each
replicate; never resample individual detections as if they were independent.
Keep the four spatial quadrants as fixed strata and report their results
separately. The resulting intervals quantify injection-realization uncertainty
conditional on this field, its background, the fitted reconstruction, and the
frozen test exposures. They do not include field-to-field variance, calibration
uncertainty, or optimizer instability unless those quantities are explicitly
resampled. Report a separate calibration/seed sensitivity analysis and require
coverage checks for any combined interval. Generalization beyond this
conditioning requires independent fields or exposure sets. Compute the purity
lower bound from the same paired cluster bootstrap and report bootstrap-fit
failures. Unmatched detections in a real field can be genuine uncatalogued
sources, so negative images, blank simulations, and adjudicated real-source
matches must accompany any false-positive estimate. Wall time, parameter/atom
count, peak resident memory, and peak accelerator memory are reported with the
scientific metrics. A 0.2 mag improvement is a useful planning target, not a
success threshold that can be chosen after seeing the data.

The strongest classical comparator is not a single-band stack. It includes
validation-selected Drizzle and SWarp products followed by PSF-aware multiband
detection with the same spectral prior, plus the native-detector linear or
proper-coadd information reference. Independent and shared INR and Foam rows
determine whether any gain comes from the representation, spectral sharing, or
simply more capacity.

## 13. Model matrix and ablations

The minimum defensible method matrix is:

| Family | Required row | Scientific question |
| --- | --- | --- |
| classical | weighted Torchsky per-band stack | internal mapmaking control |
| classical | upstream square Drizzle | standard dithered-imaging comparator |
| classical | validation-selected SWarp | standard survey-stacking comparator |
| classical | native-detector linear fit or proper coadd with effective PSF/covariance | information-preserving linear reference |
| classical | PSF-aware whitened multiband detection over the best stacks with the same SED/template family | strongest classical use of the same bands and prior |
| INR | independent appearance | what does a flexible continuous spatial model do without spectral sharing? |
| INR | shared spectral appearance | does INR spectral coupling add depth without unacceptable bias? |
| Foam | independent appearance on shared geometry | what does Foam geometry do without spectral sharing? |
| Foam | shared low-rank spectral appearance | primary adaptive spectral candidate |

Estimate any whitening operator from training or calibration evidence only,
and include the inter-band and resampling covariance supported by those data.
Apply the same transform to data, templates, and injected signals. If a diagonal
or otherwise approximate covariance is used for scale, label the detector
approximate and validate its null distribution on held-out blank data.

For Foam, the planned shared row uses a four-mode spectral basis. The
six-mode, rank-six whitened row spans arbitrary appearances in six bands and is
the response-aware independent control. Both a near-parameter-matched grid and
a spatial-resolution-matched grid are required because these controls answer
different questions.

For both INR and Foam, the final scored product must be generated by the same
declared observation model used for evaluation. Foam is scored from detached
hard geometry, never from the smooth ownership surrogate.

## 14. Optimization procedures

### 14.1 INR

A minimal INR fit is:

1. normalize sky and wavelength coordinates;
2. initialize the coordinate encoder and decoder;
3. sample valid detector pixels or manageable exposure crops;
4. render predictions through passband, WCS, pixel, PSF, and background
   operators;
5. evaluate the weighted likelihood and declared regularizers;
6. update parameters with a gradient optimizer;
7. select the checkpoint by frozen validation metrics;
8. render final products and predict held-out exposures.

Mini-batching changes the stochastic optimization path but must not change the
meaning of the objective. Sampling probabilities and loss reweighting must be
recorded if detector pixels are not sampled uniformly. Architecture selection
uses the frozen validation folds; repeated seeds quantify optimization
stability, and the rule for choosing or averaging them is fixed before the
blind test.

### 14.2 Foam

A robust Foam fit separates geometry from exact appearance inference:

1. initialize sites, radii, spectral coefficients, and optional slopes without
   using blind source outcomes;
2. if geometry is adaptive, optimize sites and radii through smooth ownership
   with an annealed temperature schedule;
3. detach and freeze the selected hard geometry;
4. construct exact sparse pixel/cell area and first-moment operators;
5. solve the fixed-geometry spectral affine appearance in native detector
   pixels with the declared PSF and weights;
6. verify PCG convergence, native-pixel residuals, support, and flux response;
7. hard-render final products for the common scorer.

For deterministic fixed geometry, smooth appearance optimization may be
unnecessary if the exact linear solve is both more faithful and computationally
viable. We should delete that redundant stage if controlled evidence confirms
it adds no value.

## 15. Mathematical and software invariants

The following limiting cases are small but powerful correctness checks:

1. **Constant sky:** with identity PSF, exact pixel-integrated flux equals
   intensity times covered pixel area.
2. **Surface-brightness semantics:** dividing the same integral by pixel area
   returns the constant intensity for pixels of different size.
3. **Single band:** \(K=1\) reduces the spectral model to a spatial field times
   one response scalar.
4. **No affine term:** setting all \(\boldsymbol h_{jk}=0\) reproduces the
   piecewise-constant Foam.
5. **Independent spectral limit:** a full-rank \(K=B\) response basis can
   reproduce arbitrary values in all \(B\) bands on fixed geometry.
6. **Unit change:** an affine change of wavelength units leaves the physical
   band prediction invariant after the domain and passbands are transformed
   consistently.
7. **Adjoint:** every linear pair satisfies
   \(\langle\boldsymbol A\boldsymbol x,\boldsymbol y\rangle
   =\langle\boldsymbol x,\boldsymbol A^\mathsf T\boldsymbol y\rangle\) to the
   declared numerical tolerance.
8. **Solver:** PCG and a dense solve agree on a small positive-definite problem;
   a rank-deficient fixture either constrains the null space or uses the
   declared least-squares solver, and tighter PCG tolerance leaves scientific
   metrics stable.
9. **Hard/soft boundary:** scientific products use hard cells; any smooth/hard
   difference is measured rather than assumed negligible.
10. **Zero-information pixels:** zero mask or weight contributes neither to the
    objective nor to the normal equations.
11. **Response correction:** an estimator with attenuation \(|R_a|<1\) cannot
    claim improved sensitivity from its uncorrected output variance.
12. **Nuisance uncertainty:** adding nonzero predictive background or response
    covariance cannot reduce the propagated target variance.
13. **Split isolation:** held-out detector samples influence neither scene or
    nuisance fitting nor normalization, whitening, calibration refinement, or
    checkpoint selection for that fold.

Torchsky's focused Foam tests exercise exact areas and moments, forward/adjoint
agreement, heterogeneous pixel tessellations, spectral response, and the
matrix-free solve. Real-data injection tests are still required because
algebraic correctness cannot establish astronomical fidelity.

## 16. Longer-term generalization

The same formulation extends beyond broadband stacking. A component-separated
sky can be written as

\[
I(\boldsymbol x,\lambda,t)
=\sum_{q=1}^{Q}
S_q(\boldsymbol x,t)\,G_q(\lambda;\eta_q)
+ I_{\mathrm{residual}}(\boldsymbol x,\lambda,t),
\]

where components can represent astrophysical emission, foregrounds, telluric
or sky emission, or moving/transient sources. The component fields may
themselves be INRs, Foams, or hybrids. The observation operator then maps their
sum into CMB frequency maps, 21-cm cubes, multi-object spectra, or time-domain
images.

An INR or Foam reconstruction is not automatically a component separation. If
it is used as an upstream product, downstream inference must retain its
measurement operator, target transfer function, support, correlated covariance,
and nuisance/calibration uncertainty. Rendering the field to a regular grid and
then treating those pixels as independent noiseless labels discards the inverse
problem that gave the reconstruction scientific meaning.

This is relevant to FOSSIL and component separation, but it introduces new
identifiability constraints: spectral components need anchors, mixing laws or
priors; moving sources need a kinematic model; and instrumental backgrounds
need detector-frame evidence. The immediate depth experiment should first
establish the shared low-level WCS, passband, PSF, background, likelihood, and
exact-rendering primitives on a controlled imaging problem.

## 17. What success and failure mean

Success is not a visually smoother image or a lower training loss. It is a
pre-registered improvement in faint-source recovery at fixed purity, with
acceptable photometry, held-out prediction, controls, and computational cost.

A null result is scientifically useful. It can show that classical coadds are
already near the available information limit, that the chosen spectral prior
does not match the source population, that the forward model is the limiting
factor, or that the representation/optimizer needs improvement. The claim must
follow the result:

- if shared Foam passes every gate, report the quantitative Foam depth gain;
- if joint INR passes and Foam does not, report the joint continuous-field
  result and describe Foam as experimental;
- if neither has a positive blind interval, report the common detector-level
  benchmark and its measured limits.

That claim ladder is part of the problem formulation. It prevents model
development from turning into post-selection of the most favourable image or
metric.

## Glossary

- **Coadd/stack:** an output image formed by resampling and combining multiple
  observations.
- **Forward model:** the calculation that predicts observed detector values
  from a latent sky and instrument state.
- **INR:** implicit neural representation; a learned function queried at
  continuous coordinates.
- **Power diagram:** a weighted Voronoi tessellation based on squared distance
  minus a site weight.
- **Power Foam:** Torchsky's bounded, potentially adaptive power-cell field with
  spatial-spectral appearance coefficients.
- **Hard rendering:** evaluation with exact discrete cell ownership and pixel
  intersections.
- **Smooth surrogate:** differentiable approximation to cell ownership used
  while optimizing geometry.
- **Passband design:** the spectral basis integrated through each observation's
  response curve.
- **Held-out exposure:** an observation excluded from parameter fitting and
  used to test prediction.
- **Purity:** fraction of detections that correspond to injected/real targets
  under the declared matching rule.
- **Completeness:** fraction of injected/real targets recovered under the same
  rule.
- **\(m_{50}\):** magnitude where fitted recovery probability is 50%.
- **PCG:** preconditioned conjugate gradients, an iterative solver for the
  regularized fixed-geometry normal equations.

## Related Torchsky documents

- [Continuous fields](../field.md): current field APIs, models, observation
  operators, and training workflow.
- [Implicit forward modelling](../implicit_forward.md): compact multi-survey
  prediction and robust-loss scaffold.
- [ADASS UPDec depth sprint](adass_updec_depth_week.md): frozen data split,
  method matrix, claim gates, and current execution status.
- [Source unmixing with known or unknown sources](source_unmixing_problem_formulation.md):
  component identifiability, protected targets, cross-fitting, and uncertainty
  after reconstruction.
- [FOSSIL primitives](fossil.md): longer-term differentiable
  spatial-spectral component-separation use cases.
