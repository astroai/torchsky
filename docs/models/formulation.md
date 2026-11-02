# Formulation

!!! note "Background for experimental models"
    This page describes the inverse problem behind the experimental
    [`torchsky.field`](fields.md) models. The mathematics is stable; the
    model kinds, solver options, and names it refers to may change between
    minor releases, as described on the [Fields](fields.md) page.

**In short.**

- A field model estimates one continuous sky \(I(\boldsymbol x, \lambda)\)
  from many exposures in many bands, by predicting every detector pixel
  through a model of the instrument and comparing with the data (analysis by
  synthesis). No training labels or source catalogue are needed.
- The instrument model has explicit, separately testable parts: WCS, passband,
  spatial response (PSF), pixel integration, calibration, background, and
  noise. Their units and ordering must be consistent with the data.
- The loss is a likelihood. With known Gaussian noise it is \(\chi^2/2\);
  fitted noise needs the normalization term, and regularizers bias faint
  fluxes.
- An **INR** (a neural network of coordinates) and a **spectral Power Foam**
  (adaptive cells with spectra) are two parameterizations of the same sky.
  They must be compared through the same data, instrument model, and scorer.
- At fixed cells the Foam is linear in its coefficients; its exact
  pixel-integrated operator is solved by preconditioned conjugate gradients.
- A joint spectral model can reach fainter sources than per-band coadds only
  if bands are truly correlated, the spectral model fits the sources, and the
  instrument model is accurate. It does not create photons.
- Flexible PSF and background models are degenerate with the scene; they need
  calibration data and staged fits, and every claim needs held-out data.

## The Inverse Problem

In supervised learning a model is trained on pairs of inputs and known
targets. Here the target, a noise-free sky at high resolution in every
wavelength, is never observed. What is observed are noisy images of it, each
made through a different filter, PSF, detector, dither position, background,
and mask. The fit therefore runs **analysis by synthesis**:

1. propose a continuous sky model;
2. pass it through a differentiable model of each observation;
3. compare the predicted pixels with the measured pixels;
4. update the sky and, where the data identify them, instrument parameters.

The observations supervise their own reconstruction, and the forward physics
and noise model define what the loss means. A source catalogue is not needed
to fit the scene; it can be used afterwards for photometry and validation, and
injected sources with known flux give controlled tests.

## Data and Unknowns

Index observations by \(n = 1, \dots, N\) and their pixels by \(p\):

- \(y_{np}\): calibrated pixel value;
- \(m_{np} \in [0, 1]\): mask (hard or soft);
- \(w_{np} \ge 0\): statistical weight, normally inverse variance;
- \(\boldsymbol u\): continuous detector coordinate;
- \(\boldsymbol x\): sky coordinate, in a local tangent plane or on the sphere;
- \(\lambda\): wavelength; \(t_n\), \(c_n\): exposure time and observing context.

The main unknown is the specific intensity \(I(\boldsymbol x, \lambda)\), or
\(I(\boldsymbol x, \lambda, t)\) when variability is part of the science. It is
expressed in surface-brightness units (for example MJy sr⁻¹) or, with gain,
exposure time, and throughput terms, in counts. With scene parameters
\(\theta\) and instrument parameters \(\psi\), each observation is predicted as

\[
\boldsymbol\mu_n = \mathcal A_n(I_\theta; \psi_n) + \boldsymbol B_n(\psi_n),
\]

where \(\mathcal A_n\) is the calibrated observation operator (WCS, spectral
and spatial response, pixel footprint, photometric calibration) and
\(\boldsymbol B_n\) the additive background.

## The Observation Model

Let \(W_n\) map detector coordinates to the sky, so pixel \(p\) covers the sky
region \(\Omega^\mathrm{sky}_{np} = W_n(\Omega^\mathrm{det}_{np})\) with solid
angle \(|\Omega^\mathrm{sky}_{np}|\). With the spectral response \(R_n(\lambda)\),
a spatial-response operator \(\mathcal P_n\) acting on each monochromatic slice
of the sky, and a calibration factor \(g_n\), a pixel-averaged surface
brightness is

\[
\mu_{np} =
\frac{g_n}{|\Omega^{\rm sky}_{np}|}
\int_{\Omega^{\rm sky}_{np}}\!\mathrm d\Omega_{\boldsymbol x}
\int\!\mathrm d\lambda\,R_n(\lambda)
\left[\mathcal P_n(\lambda,t_n,c_n) I_\theta(\cdot,\lambda)\right]
\!(\boldsymbol x)
+ B_{np}.
\]

Drop the division by solid angle when \(y_{np}\) is an integrated flux per
pixel. In detector coordinates the solid angle of a pixel is the integral of
the Jacobian

\[
J_{\Omega,n}(\boldsymbol u)
=\left\|
\frac{\partial\widehat{\boldsymbol n}_n}{\partial u_x}
\times
\frac{\partial\widehat{\boldsymbol n}_n}{\partial u_y}
\right\|
= |\cos\delta| \left|\det \frac{\partial(\alpha,\delta)}{\partial(u_x,u_y)}\right|,
\]

with \(\widehat{\boldsymbol n}_n\) the unit sky direction and \(\alpha, \delta\)
in radians. Data, weights, injected fluxes, WCS distortions (SIP, TPV), and the
forward operator must all use the same convention.

Each part of the operator keeps the scene from absorbing an instrument effect:

| Part | Role |
|---|---|
| WCS | Associates each detector footprint with a sky region, with fixed pixel-centre and axis conventions |
| Passband | Integrates the one spectral field through each filter; a central wavelength alone is not enough for a broad band |
| Spatial response | Maps sky structure into detector structure before pixel integration; a shift-invariant PSF convolution is the simplest case |
| Pixel integration | Averages over the finite pixel; point sampling is biased for undersampled structure |
| Calibration | Converts the physical prediction into the units of the data |
| Background | Adds sky, telescope, and detector terms in their own frames |
| Mask and noise | Say which residuals carry information, and how much |

**Background placement is part of the model.** Diffuse sky light and stray
light that pass through the optics belong inside the response; detector offsets
belong in \(B_{np}\). Subtracting a background model removes its mean, not its
photon noise.

**Spectral normalization.** The response may be normalized,
\(\int R_n\,\mathrm d\lambda = 1\), giving a band-averaged intensity, or
include \(\lambda/hc\) for photon counting. Record the choice; do not mix them.

**Linearity.** The equation assumes \(\mathcal P_n\) is linear in intensity at
fixed calibration. The full operator need not be: saturation, nonlinearity, or
charge transport can be differentiable stages of \(\mathcal A_n\). A practical
\(\mathcal P_n\) may be a compact kernel, a position-dependent operator, a
wave-optics calculation, or a learned model; its accuracy must be declared and
checked against calibration data. The exact Foam solver, for example, composes
a supplied linear pixel-domain PSF exactly, which is exact for that discrete
operator but not a proof of the continuous response.

## Likelihood and Objective

For independent Gaussian pixel errors with known variances, the negative log
likelihood, up to a constant, is

\[
\mathcal L_{\mathrm{data}}(\theta,\psi)
= \frac{1}{2}\sum_{n,p}
m_{np}w_{np}\left(y_{np}-\mu_{np}(\theta,\psi)\right)^2 .
\]

With true inverse-variance weights this is \(\chi^2/2\); with rescaled or
clipped weights it is a weighted least-squares objective without probabilistic
calibration. Charbonnier and Huber losses reduce the effect of unmasked
outliers; a Poisson–Gaussian likelihood ([Likelihoods](likelihoods.md)) is
better near raw counts. A soft mask is a weight, not a model of missing data.

For correlated Gaussian noise with covariance \(\boldsymbol N\) on the valid
pixels,
\(\mathcal L_\mathrm{data} = \tfrac12 \boldsymbol r^\mathsf{T} \boldsymbol N^{-1} \boldsymbol r\)
with \(\boldsymbol r = \boldsymbol y - \boldsymbol\mu\). If
\(\boldsymbol N(\eta)\) is fitted to the same data, the objective must include
\(\tfrac12 \ln\det \boldsymbol N(\eta)\), or the fit lowers the loss by
inflating the noise.

The complete objective adds scene regularizers \(\mathcal R_q\) and
calibration anchors \(\mathcal C_r\):

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

Regularizers (positivity, spectral simplicity, smoothness) change the
estimator and can bias faint-source flux; choose their weights on validation
data. Resampled images have correlated noise even when the detector pixels do
not, which is one reason to fit native pixels.

## Implicit Neural Representations

An **implicit neural representation (INR)** stores the sky in the weights of
a function rather than in an array of pixels:

\[
I_\theta(\boldsymbol x,\lambda) = f_\theta\!\left(E(\boldsymbol x,\lambda)\right),
\]

where \(E\) encodes the coordinates and \(f_\theta\) is a neural network. The
field can be queried at any coordinate and has no output grid. Because small
networks learn fine structure poorly from raw coordinates, \(E\) supplies
multi-scale features: Fourier features, learned feature planes, or hash grids
([Encoders](encoders.md)).

A **joint** INR shares parameters across wavelength; an **independent**
control must be able to give each band an arbitrary appearance (separate
per-band models, or a full-rank band embedding). Training a joint model for
fewer steps is not an independent control.

| Strengths | Risks |
|---|---|
| Compact for smooth structure | Frequency bias can suppress compact or sharp sources |
| Evaluates at any coordinate | Excess capacity fits noise and detector artefacts |
| Couples space and wavelength flexibly | Too little spectral freedom pulls unusual colours toward common ones |
| Gradients through the whole instrument model | Non-convex training depends on initialization and schedule; no built-in uncertainty |

A smooth-looking reconstruction is not evidence of correct photometry, which
is why injected sources and held-out exposures are required.

## Spectral Power Foam

A **Power Foam** represents the sky with adaptive cells instead of a pixel
grid or a global network. The cells form a bounded **power diagram**, a
weighted Voronoi diagram.

**Cells.** For site \(\boldsymbol s_j\) and radius \(r_j \ge 0\), the power
distance is \(d_j(\boldsymbol x) = \|\boldsymbol x-\boldsymbol s_j\|^2 - r_j^2\),
and the bounded cell is

\[
C_j = \left\{
\boldsymbol x:
d_j(\boldsymbol x)\leq d_k(\boldsymbol x)\ \text{for every }k,
\quad d_j(\boldsymbol x)\leq0
\right\}.
\]

A point belongs to the site of smallest power distance, within that site's
disk. A larger radius can win territory from a nearer site. The union of cells
need not cover the plane; uncovered regions have zero Foam signal, which is not
the same as measured blank sky.

**Appearance.** With spectral basis functions \(\phi_k(\lambda)\) (Legendre
polynomials in normalized wavelength), each cell has a value and optionally a
spatial gradient:

\[
I(\boldsymbol x,\lambda)
= \sum_k c_{jk}\phi_k(\lambda)
+ \Big(\sum_k \boldsymbol h_{jk}\phi_k(\lambda)\Big)^{\mathsf T}
(\boldsymbol x-\boldsymbol s_j),
\qquad \boldsymbol x\in C_j .
\]

Without gradients the field is piecewise constant. The affine term keeps a
slope across a cell but does not make the field continuous between cells.

**Sharing across bands.** Observation \(n\) sees the basis through its
response, \(b_{nk} = \int R_n(\lambda) \phi_k(\lambda)\,\mathrm d\lambda\). With
fewer basis functions \(K\) than bands \(B\), the bands share a
low-dimensional spectral description; that is how the Foam borrows information
across wavelength. The matching independent control on the same cells uses
\(K = B\) with a full-rank response matrix \([b_{nk}]\); it removes spectral
sharing but still shares the cell geometry.

**Training the geometry.** Hard cell membership is not differentiable, so
sites and radii are trained through a smooth surrogate with temperatures
\(\tau_p, \tau_s\):

\[
q_j(\boldsymbol x)
=\operatorname{softmax}_j
\left[-\tau_p d_j(\boldsymbol x)
+\log\sigma\!\left(-\tau_s d_j(\boldsymbol x)\right)\right],
\qquad
u(\boldsymbol x)
=1-\prod_j\sigma\!\left(\tau_s d_j(\boldsymbol x)\right),
\]

with \(\widetilde w_j = u\,q_j\) the soft ownership of point \(\boldsymbol x\)
by cell \(j\) and \(\sigma\) the logistic function. Raising the temperatures
sharpens the cells toward the hard partition. Because the surrogate can move
flux across soft boundaries, final products use the detached hard geometry.

**Exact pixel operator.** For a pixel footprint \(\Omega_{np}\), the
intersection area \(A_{npj}\) and first moment \(\boldsymbol M_{npj}\) with cell
\(j\) are computed exactly: the pixel polygon is clipped by the cell's
half-planes and integrated against its disk with analytic segment and arc
terms (curved WCS pixel edges are tessellated to a stated tolerance). At fixed
geometry the integrated prediction is linear in the coefficients:

\[
z_{np}
=\sum_j\left[
A_{npj}\sum_k b_{nk}c_{jk}
+\boldsymbol M_{npj}^{\mathsf T}
\sum_k b_{nk}\boldsymbol h_{jk}
\right].
\]

**Solving for the coefficients.** With \(\boldsymbol F_n\) the footprint and
passband terms, \(\boldsymbol H_n\) a linear PSF, weights \(\boldsymbol W_n\),
and a ridge \(\gamma\) toward an initializer \(\boldsymbol c_0\) scaled by
\(\boldsymbol D\), the coefficients solve

\[
\left(
\sum_n \boldsymbol F_n^\mathsf T\boldsymbol H_n^\mathsf T
\boldsymbol W_n\boldsymbol H_n\boldsymbol F_n
+\gamma\boldsymbol D^\mathsf T\boldsymbol D
\right)\boldsymbol c
= \sum_n \boldsymbol F_n^\mathsf T\boldsymbol H_n^\mathsf T
\boldsymbol W_n\boldsymbol y_n
+\gamma\boldsymbol D^\mathsf T\boldsymbol D\boldsymbol c_0,
\]

by preconditioned conjugate gradients (PCG), matrix-free. PCG needs the left
side to be positive definite: the stacked data operator and \(\boldsymbol D\)
must have no common null direction (for \(\gamma = 0\), the data operator alone
must have none). Otherwise constrain the null space or use LSQR/LSMR. A small
residual does not by itself show a unique solution; report the stopping rule
and check that results are stable under a tighter tolerance.

| Strengths | Risks |
|---|---|
| Local support and adaptive resolution | Poorly placed sites leave holes or waste cells |
| Exact pixel integration at fixed geometry | Large cells attenuate compact sources |
| Linear appearance problem with an exact solve | Soft training and hard rendering can disagree |
| | Geometry optimization is non-convex; sparse operators can dominate memory on large fields |

## Comparing Representations Fairly

An INR and a Foam are two parameterizations of the same sky, so a comparison
must share everything except the parameterization, its priors, and its
optimizer:

| Shared element | Why |
|---|---|
| Exposures and the train/validation/test split | Equal information, no leakage |
| Masks and inverse variances | Same statistical role for each pixel |
| WCS and pixel footprints | No method gets easier resampled geometry |
| Passbands and spectral convention | "Joint wavelength modelling" means the same thing |
| PSF, background, and calibration treatment | Resolution or nuisance differences cannot pose as depth |
| Injected sources and null controls | Completeness and purity are paired |
| Output grid and detection/photometry code | The scorer favours neither |

Report parameter count, sampled pixels, optimizer steps, wall time, and memory;
there is no single notion of matched compute, so use both capacity-matched and
resolution-matched controls.

## Coadds, Joint Fields, and Depth

Drizzle and SWarp resample each exposure to an output grid and combine them.
They are strong baselines: a well-tuned drizzle keeps much of the dither
information, and a linear coadd with its effective PSF and covariance is
sufficient for some static linear measurements. Field models instead estimate
one scene directly from all detector pixels, postponing the output grid,
avoiding repeated interpolation, and using a shared spectral model during
reconstruction. These are possible advantages, not guaranteed ones; when the
learned prior or the forward model is wrong, the classical pipeline can be
better.

A joint spectral model can detect a source that is below threshold in every
band if its spectrum lies near a low-dimensional family: fewer shared
quantities are estimated from all compatible photons, trading some bias for
lower variance. The gain requires that

1. the bands carry genuinely correlated signal;
2. the spectral model includes the sources tested;
3. WCS, PSF, calibration, background, and noise are accurate enough;
4. regularization reduces noise more than it attenuates flux.

Sources with strong lines, breaks, or unusual colours can be harmed by a
smooth low-rank prior.

**Effective sensitivity.** For a linear Gaussian problem with unit-amplitude
template \(\boldsymbol q\) and covariance \(\boldsymbol N\), the Fisher
information for the amplitude is
\(\mathcal I_a = \boldsymbol q^\mathsf{T} \boldsymbol N^{-1} \boldsymbol q\), and an
unbiased estimator has \(\operatorname{Var}(\hat a) \ge \mathcal I_a^{-1}\). A
regularized estimator can lower its variance by attenuating the source. With
the response to injections \(R_a = \partial\,\mathbb E[\hat a]/\partial a\),
compare methods by

\[
\sigma_{a,\mathrm{eff}} = \frac{\sqrt{\operatorname{Var}(\hat a)}}{|R_a|},
\]

together with bias. A smaller variance obtained with \(|R_a| < 1\) is not a
deeper detection.

## Learned Instrument Models and Identifiability

PSFs and backgrounds vary across the detector, with wavelength, and with time,
on different scales. torchsky provides adaptive coordinate encoders and
conditioned neural PSF and background models for this, built as **mechanistic
hybrids**: the WCS, passband, radiometry, pixel integration, conservation
laws, and adjoints stay explicit, and a learned component models only the
residual structure that calibration data can identify.

Unrestricted joint learning is not identifiable:

- a broad source can be traded for a wider PSF;
- diffuse emission can be traded for background;
- an astrometric offset can be traded for source position;
- passband or gain errors can be traded for source colour.

A nuisance model trained on the pixels that measure the signal can absorb the
signal as its capacity grows; an underfit one leaves residuals that look like
sources. Fit in stages:

1. initialize WCS, passbands, detector scale, and PSF support from instrument
   metadata;
2. fit background variation on source-free data;
3. fit PSF variation on isolated stars or injected calibration sources;
4. fit the scene with nuisance models fixed or strongly anchored;
5. refine jointly only while held-out calibration and exposure metrics improve.

A calibrated nuisance is still uncertain. When background prediction
covariance \(\boldsymbol C_n\) and response uncertainty
\(\boldsymbol J_{\psi,n}\boldsymbol C_\psi\boldsymbol J_{\psi,n}^\mathsf{T}\) are
independent of detector noise, the held-out residual covariance is

\[
\boldsymbol V_n
=\boldsymbol N_n+\boldsymbol C_n
+\boldsymbol J_{\psi,n}\boldsymbol C_\psi
 \boldsymbol J_{\psi,n}^{\mathsf T}.
\]

Seeds and ensembles probe optimization instability; they do not represent
calibration uncertainty.

**Validation.** A low training residual is not evidence. Freeze train,
validation, and test exposures before fitting; select architecture, rank, cell
density, schedule, and regularization on validation data only; use the test
exposures once. Exposure folds of one field test prediction into withheld
detector realizations of that field, not generalization to other fields.
Useful diagnostics are weighted residuals in held-out exposures, residual
autocorrelation around sources, flux recovery of injections, responses to
blank, shuffled-band, and negative controls, and adjoint dot-product tests of
every linear operator.

## Fitting Procedures

**INR.**

1. Normalize sky and wavelength coordinates (the `Domain`).
2. Initialize the encoder and decoder.
3. Sample valid pixels or exposure crops.
4. Render predictions through passband, WCS, pixel, PSF, and background.
5. Evaluate the weighted likelihood and regularizers; take a gradient step.
6. Select the checkpoint on validation metrics; render products and held-out
   exposures.

Mini-batching changes the optimization path, not the objective; record
sampling probabilities if pixels are not sampled uniformly.

**Foam.**

1. Initialize sites, radii, and coefficients without using test outcomes.
2. If the geometry is adaptive, optimize sites and radii through the smooth
   surrogate with an annealed temperature.
3. Freeze the hard geometry and build the exact area and moment operators.
4. Solve the fixed-geometry appearance in native pixels with the declared PSF
   and weights.
5. Check PCG convergence, native residuals, support, and flux response; render
   hard products.

## Correctness Checks

Small limiting cases catch most implementation errors:

1. **Constant sky:** with an identity PSF, integrated flux equals intensity
   times covered pixel area.
2. **Surface brightness:** dividing by pixel area returns the constant
   intensity for pixels of any size.
3. **Single band:** \(K = 1\) reduces to a spatial field times one response
   scalar.
4. **No slope:** \(\boldsymbol h_{jk} = 0\) reproduces the piecewise-constant
   Foam.
5. **Independent spectra:** a full-rank \(K = B\) basis reproduces arbitrary
   values in all bands on fixed geometry.
6. **Units:** changing wavelength units, with domain and passbands transformed
   consistently, leaves predictions unchanged.
7. **Adjoint:** \(\langle A\boldsymbol x, \boldsymbol y\rangle =
   \langle \boldsymbol x, A^\mathsf{T}\boldsymbol y\rangle\) for every linear
   pair.
8. **Solver:** PCG matches a dense solve on a small positive-definite problem.
9. **Zero information:** masked or zero-weight pixels change neither the
   objective nor the normal equations.
10. **Response correction:** an estimator with \(|R_a| < 1\) cannot claim
    sensitivity from its uncorrected variance.
11. **Split isolation:** held-out pixels influence no fitted parameter,
    normalization, or checkpoint choice.

The Foam tests in the repository cover exact areas and moments,
forward–adjoint agreement, heterogeneous pixel tessellations, spectral
response, and the matrix-free solve. Algebraic correctness does not establish
astronomical fidelity; that needs real-data injection tests, which run in
[torchsky-harness](https://github.com/astroai/torchsky-harness).

## Glossary

- **Coadd (stack):** an image formed by resampling and combining exposures.
- **Forward model:** the calculation that predicts detector values from a sky
  and instrument state.
- **INR:** implicit neural representation, a learned function queried at
  continuous coordinates.
- **Power diagram:** a weighted Voronoi tessellation using squared distance
  minus a site weight.
- **Power Foam:** torchsky's bounded, adaptive power-cell field with spectral
  coefficients per cell.
- **Hard rendering:** evaluation with exact cell ownership and pixel
  intersections.
- **Smooth surrogate:** the differentiable approximation of cell ownership
  used to train the geometry.
- **Held-out exposure:** an observation excluded from fitting and used to test
  prediction.
- **Completeness / purity:** fraction of true sources recovered / fraction of
  detections that are true, under one matching rule.
- **PCG:** preconditioned conjugate gradients, the iterative solver for the
  fixed-geometry normal equations.

## See Also

- [Fields](fields.md): the field API, models, and fitting workflow.
- [Scenes](scenes.md): linear observation operators and normal-equation
  solves.
- [Likelihoods](likelihoods.md): Gaussian and Poisson–Gaussian noise models.
- [PSF conventions](psf-conventions.md): kernel orientation and flux
  conservation.
