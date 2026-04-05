# Implicit forward modeling and INR for survey-scale inference

**Math:** Equations use GitHub-style delimiters: **`$…$` for inline** and **`$$…$$` on their own lines for display**. Enable math in VS Code with a Markdown+Math extension, or view on GitHub.com.

This note specifies how we think about **end-to-end differentiable forward models** for astronomical surveys, with **as few structural assumptions as possible**, explicit **time dependence**, **kinematics** (redshift, bulk and transverse motion, **peculiar velocities relative to an arbitrary reference field**, rotation, acceleration, high-precision astrometric / radial-velocity regimes, and external kinematic summaries with uncertainties), and a concrete split between **the sky (away from detectors)** and **everything the instrument and environment do to the data**.

It extends and generalizes the earlier “global inverse problem” picture: we keep tractability ideas (sparse rendering, tracts, anchors) but drop commitments to additive Gaussian noise, separable PSFs, or a purely discrete source list where a continuous scene is more natural. **§8** spells out how the same shell scales to **many heterogeneous surveys** (imaging, spectroscopy, X-ray, CMB) without pretending one job fits every raw pixel at once.

---

## 1. Rationale and problem statement

Classical pipelines stage WCS, photometry, background, and coadds. That sequencing tends to:

1. **Lose coupled uncertainties** between astrometry, throughput, and scene estimates.
2. **Distort noise models** after resampling and stacking.
3. **Replace a continuous sky** with discrete catalog operations where blending and low-S/N structure matter.

**Direction:** Treat survey inference as **one inverse problem**: find latent state and instrument law such that predicted raw (or minimally processed) data agree with observations under a chosen likelihood or scoring rule.

**Minimal forward assumption (only this is “the model”):**


$$
Y \sim \mathbb{P}_{\psi}\big(\,\cdot \mid S,\, c\,\big)
$$


- $S$: latent **scene** (see §2)—what we want the sky to be *before* or *at* the boundary we choose for “detector-side” effects. When kinematics are explicit, $S = (F_{\theta}, \zeta)$ with $\zeta$ the block in §4.
- $c$: **context** for one readout unit (exposure, detector segment, integration, pointing metadata, temperature, etc.).
- $\psi$: parameters of the **instrument–environment–processing** mechanism (tables, polynomials, or large neural maps).
- $Y$: actual data vector (pixels, bins, visibilities, event bins, …).

**Deterministic forward** is $\mathbb{P}_{\psi} = \delta_{G_{\psi}(S,c)}$. **Non-additive noise, correlations, bright-fatter, CTE, nonlinearity, fringing, unknown unknowns** are not separate axioms: they are **properties of** $\mathbb{P}_{\psi}$. With enough diverse data, **$\psi$ can be learned** (regularized) rather than hand-derived.

The rest of this document **instantiates** that shell for surveys and for torchsky-style continuous fields.

---

## 2. Latent state: sky away from detectors vs instrument

We deliberately factor **conceptually** (implementation may fuse pieces for efficiency):

### 2.1 Scene $S$ — “sky away from detectors”

The scene is whatever must be specified so that, together with $c$ and $\psi$, the data law is determined. **It need not be physically unique**; it must be **identifiable** given data and any constraints (anchors, regularization, or design choices that fix gauge).

**Target for INR (recommended default):** a continuous **surface brightness field** (or specific intensity) on domain $\mathcal{U}$:


$$
F_{\theta}: \mathcal{U} \to \mathbb{R}^{d_f}
$$


Typical coordinates:

- **Spatial:** unit sphere or tangent patch $(\alpha,\delta)$ or 3D direction $\omega$.
- **Spectral:** wavelength $\lambda$, log-wavelength, or energy $E$ (one consistent linear axis per analysis).
- **Time:** $t$ (e.g. mid-exposure MJD, or per-read time tag for time-resolved work).

So $\mathcal{U} \subseteq \mathbb{S}^2 \times \mathbb{R}_{\lambda} \times \mathbb{R}_{t}$ (possibly with some dimensions collapsed when unused).

**INR:** $F_{\theta} = g_{\theta}$ (MLP / grid / hybrid as in `torchsky.map.field`).

**Discrete sources** are a **special case**: sum of $\delta$-like or compact kernels in $(\omega,\lambda,t)$, or a hybrid **field + catalog residual**.

**Boundary choice:** In practice we fix a **radiometric boundary** (e.g. top-of-atmosphere specific intensity, or “telescope entrance pupil” radiance) and put everything downstream (optics, detector, electronics) in $\mathbb{P}_{\psi}$. The boundary should match how $\mathbb{P}_{\psi}$ is trained or calibrated. Changing the boundary is a **coordinate change** on $S$, not a change to the abstract shell.

### 2.2 Instrument–environment law $\mathbb{P}_{\psi}(\cdot \mid S, c)$

Everything that maps from the scene at the chosen boundary to the data:

- Optics, filter, dispersion, integration over pixel or spaxel.
- PSF/LSF (spatially and spectrally varying), distortion, distortion of the PSF with flux and background.
- Gain, nonlinearity, persistence, CTE, read noise structure, correlations.
- Pipeline steps you refuse to peel off (e.g. certain debias patterns) can be absorbed if $Y$ is defined at that stage.

**Implicit neural surrogates** (as in the original spec) remain valid **inside** this block: WCS-like maps, effective throughput fields, neural ePSFs conditioned on $(u,v,f,\text{local context})$, learned background/structured noise predictors, etc. They are **not** the sky; they are **part of $\psi$** or of a deterministic $G_{\psi}$.

---

## 3. Time dependence

Time enters in two distinct ways; both can appear simultaneously.

### 3.1 Time in the scene $F_{\theta}(\omega, \lambda, t)$

Use when the **physical sky** changes during the survey: variable sources, moving objects, weather-induced diffuse emission (if modeled in $S$), etc.

- Static limit: $\partial_t F \approx 0$ enforced by regularization or architecture (e.g. no $t$ input).
- Quasi-static: $t$ as low-frequency embedding or piecewise-constant segments.
- Rapid: full $t$ coordinate or sequence models coupled to readout indices.

### 3.2 Time (and epoch) in context $c$

Always present at least as **exposure identity** and timestamps:


$$
c = \big(\, t_{\text{exp}},\, \text{pointing},\,\text{band},\,\text{detector id},\,\text{environment metadata},\, \ldots \,\big)
$$


The law $\mathbb{P}_{\psi}(\cdot \mid S, c)$ can **depend on $c$** even if $S$ is static: drifting PSF, temperature-dependent gain, rotating instrumental polarization, etc.

**Composition:**


$$
Y \sim \mathbb{P}_{\psi}\big(\,\cdot \mid F_{\theta},\, c(t)\,\big)
$$


Read: “given the **same** intrinsic field $F_{\theta}$, different epochs produce different data distributions because **$\psi$ is conditioned on $c$** or because **$F_{\theta}$ is evaluated at different $t$**.”

---

## 4. Kinematics in the latent model

Kinematics ties **where** and **when** emission appears in observed coordinates to **intrinsic** (rest-frame or comoving) descriptions. We keep the forward shell unchanged:


$$
Y \sim \mathbb{P}_{\psi}\big(\,\cdot \mid S,\, c\,\big)
$$


and **enlarge** what sits inside $S$ (and optionally $c$) with **kinematic variables** $\zeta$. Those may be **inferred from data**, **fixed from external summaries**, or **hybrid** (fixed with an adjustable slack). Nothing below assumes a particular inference paradigm—only **what the generative map may depend on** and how **external kinematic information** can be attached.

### 4.1 Kinematic degrees of freedom $\zeta$

Write the scene as depending on a (possibly empty) kinematic block:


$$
S = \big(F_{\theta},\, \zeta\big), \quad \zeta = \{\zeta^{(1)}, \zeta^{(2)}, \ldots\}
$$


Examples of components (not all required in any one model):

| Component | Typical role | Example parameters |
|-----------|--------------|---------------------|
| **Line-of-sight / redshift** | Map rest ↔ observed spectrum | Empirical $z$, $v_{\mathrm{rad}}$, or user-chosen $(z,\,d_L,\ldots)$ if a distance–redshift map is part of the **specified** model |
| **Transverse motion** | Map sky position vs time | Proper motion $(\mu_\alpha,\mu_\delta)$, parallax $\varpi$, reference epoch $t_0$ |
| **Acceleration / curvature** | Higher-order sky paths | Acceleration in plane of sky, orbital phase in binaries |
| **Bulk fields** | Extended media | Line-of-sight velocity field $v_{\parallel}(\omega)$, bulk flow of gas |
| **Residual / “peculiar” motion** | Offset from **any** user-chosen reference | Per-object $\mathbf{v}_{\mathrm{pec}}$, cell values, or field $ \mathbf{v}_{\mathrm{pec}}(\omega,t)$ (see §4.9) |
| **Rotation / shear** | Extended resolved bodies | Angular velocity $\boldsymbol{\Omega}$, $v\sin i$, differential rotation curve inside a galaxy model |
| **Instrument / platform** | Non-celestial motion | Satellite boresight drift, scan velocity—often better as **context** $c$, but the same mathematics applies |

$\zeta$ may be **global** (one $z$ for a quasar), **per source** ($\zeta_k$ for each compact object), **per voxel or coefficient** (a field of line-of-sight velocities), or **shared** across a population (template + offsets).

### 4.2 Rest-frame scene vs observed coordinates (spectral)

A natural split is to define intrinsic specific intensity in **emitter rest** spectral coordinate $\lambda_{\mathrm{emit}}$ (or frequency $\nu_{\mathrm{emit}}$) and map to **observed** $\lambda_{\mathrm{obs}}$ via a **Doppler / cosmological** rule encoded in $\zeta$.

**Non-relativistic line-of-sight (illustrative):**


$$
\lambda_{\mathrm{obs}} = \lambda_{\mathrm{emit}}\,\big(1 + z\big), \qquad
z \approx \frac{v_{\mathrm{rad}}}{c} \quad (\text{sign convention fixed per implementation})
$$


**Relativistic / cosmological:** use the mapping appropriate to the science case (special relativistic Doppler, cosmological redshift with distance–duality if relevant). The **modeling commitment** is only: **$\lambda_{\mathrm{obs}}$ is a deterministic (or learned) function of $(\lambda_{\mathrm{emit}}, \zeta)$** inside $\mathcal{A}_{\psi}$ or inside the definition of $F_{\theta}$.

Two equivalent placements:

1. **Rest-frame INR:** $F_{\theta}(\omega,\lambda_{\mathrm{emit}}, t)$; operator $\mathcal{A}_{\psi}$ applies the $\lambda$ map before bandpass integration.
2. **Observed-frame INR:** $F_{\theta}(\omega,\lambda_{\mathrm{obs}}, t)$; kinematics already folded into training targets—less interpretable for $z$, fewer constraints.

### 4.3 Transverse motion and the sky path $\omega(t)$

For a compact emitter, observed direction can be modeled as a **path on the sphere** (or tangent approximation):


$$
\omega(t) = \mathcal{W}\big(\omega_0,\, \boldsymbol{\mu},\, \mathbf{a},\, \varpi,\, t;\, c\big)
$$


- $\omega_0$: reference position at epoch $t_0$.
- $\boldsymbol{\mu}$: proper motion (and optionally parallax $\varpi$ tying heliocentric baseline to $c$).
- $\mathbf{a}$: acceleration in the chosen parametrization (e.g. orbital segment of a binary).

**Extended rotating body:** $\omega$ is not a single point; each surface element carries a **velocity field** $\mathbf{v}(\mathbf{r})$ in the body frame, inducing differential Doppler shifts and line broadening. For unresolved objects, a **scalar or low-D summary** ($v\sin i$, velocity dispersion $\sigma$) may collapse the detail.

### 4.4 Line profiles, broadening, and rotation

Kinematics does not only shift centroids:

- **Thermal / microturbulent broadening:** width parameters in $\zeta$ convolve rest-frame spectra (inside $\mathcal{A}_{\psi}$ or inside $F_{\theta}$).
- **Rotation:** broadening kernel derived from $\boldsymbol{\Omega}$ and inclination, or learned **equivalent kernel** tied to $\zeta$ if physics is approximate.
- **Instrument LSF:** remains in $\psi$; **astrophysical broadening** is part of $S$ or the map from $S$ to pre-instrument specific intensity.

### 4.5 Accelerations and time-varying velocities

Use **acceleration** when parabolic sky motion, tides, or pulsar / orbital models need a second derivative in time. Symbolically:


$$
\omega(t) = \omega_0 + (t-t_0)\,\dot{\omega} + \tfrac{1}{2}(t-t_0)^2\,\ddot{\omega} + \cdots
$$


or a **numerical ephemeris** $t \mapsto \omega(t)$ supplied as data in $c$ (spacecraft targets) with **residual learnable parameters** in $\zeta$ if the ephemeris is imperfect.

### 4.6 External kinematic knowledge (uncertainty-aware, paradigm-agnostic)

Often **independent estimates** $(\hat{\zeta}_j,\, \sigma_j)$ exist (spectroscopic $z$, Gaia proper motions, RV from another survey). In modeling terms this is an **additional objective contribution** that pulls $\zeta_j$ toward $\hat{\zeta}_j$ with **strength set by $\sigma_j$**—without committing to Bayesian vs frequentist interpretation:


$$
\mathcal{L}_{\text{kin}} = \sum_{j \in \mathcal{J}} \rho_j\!\left(\frac{\zeta_j - \hat{\zeta}_j}{\sigma_j}\right)
$$


- $\rho_j$: any convex or robust penalty (quadratic, Huber, truncated, …).
- **Hard constraint:** limit $\|\zeta_j - \hat{\zeta}_j\| \le k\sigma_j$ via projection or barrier.
- **Ignore uncertainty:** $\sigma_j \to \infty$ drops the term; **fix kinematics:** take $\zeta_j \equiv \hat{\zeta}_j$ and remove from free parameters.

The same formalism covers **informative** and **uninformative** external summaries; the word “prior” is avoided here on purpose.

### 4.7 Composition with the forward law

The minimal update is:


$$
Y \sim \mathbb{P}_{\psi}\big(\,\cdot \mid F_{\theta},\, \zeta,\, c\,\big)
$$


Deterministic mean:


$$
\hat{Y} = G_{\psi}\big(F_{\theta},\, \zeta,\, c\big)
$$


**Identification:** $F_{\theta}$, $\zeta$, and parts of $\psi$ can trade off (e.g. wrong $z$ compensated by distorted $F_{\theta}$). **Kinematic anchors** $\mathcal{L}_{\text{kin}}$, multi-epoch astrometry, and **physically motivated smoothness** on $F_{\theta}$ reduce that degeneracy.

### 4.8 Relation to time §3

- **§3.1** $t$ in $F_{\theta}$: intrinsic variability (SN light curves).
- **§4** $\zeta$: **geometric / spectral mapping** of the same emitter across $t$ (proper motion, $z$, orbit).
- Both can appear: $F_{\theta}(\omega,\lambda_{\mathrm{emit}}, t)$ for variability **in the rest frame**, while $\omega(t)$ moves the footpoint on the sky.

### 4.9 Peculiar velocities without physical assumptions

“**Peculiar**” here means only: **motion relative to a reference model the user chooses**, not a commitment to Friedmann–Lemaître flow, a specific $H_0$, or any other cosmological or Galactic potential.

1. **Pick a reference velocity law** $\mathbf{v}_{\mathrm{ref}}(\omega,\,t;\, \kappa)$, parameterized by $\kappa$ (possibly empty: $\mathbf{v}_{\mathrm{ref}} \equiv \mathbf{0}$; possibly a low-order polynomial, spline table, mean field from another fit, or even a second INR). This object is **part of the model specification**, not “truth.”

2. **Represent the total kinematic state** either as:
   - **Split:** $\mathbf{v}_{\mathrm{tot}} = \mathbf{v}_{\mathrm{ref}}(\cdot;\kappa) + \mathbf{v}_{\mathrm{pec}}$ with **learned or anchored** $\mathbf{v}_{\mathrm{pec}}$ stored in $\zeta$; or  
   - **Direct:** learn $\mathbf{v}_{\mathrm{tot}}$ only, and define $\mathbf{v}_{\mathrm{pec}} := \mathbf{v}_{\mathrm{tot}} - \mathbf{v}_{\mathrm{ref}}$ **after the fact** for interpretation.

3. **Spectral side:** the line-of-sight contribution is whatever scalar or vector projection the implementation uses, e.g. $\delta\lambda / \lambda \sim c^{-1}\,\hat{\mathbf{n}} \cdot \mathbf{v}$ in a chosen sign convention—again as a **mapping inside** $\mathcal{A}_{\psi}$, not as a claim about which frame is “correct.”

4. **Scales:** $\mathbf{v}_{\mathrm{pec}}$ may be **sparse** (few tracers), **dense on a grid**, **harmonic / B-spline coefficients**, or **continuous** via a small field network—whatever the data support.

This is the hook for **cosmic-flow / bulk-flow / supernova–distance** style analyses: they correspond to particular choices of $\mathbf{v}_{\mathrm{ref}}$, $\mathbf{v}_{\mathrm{pec}}$, and anchors $\mathcal{L}_{\text{kin}}$, not a change to the abstract forward shell.

### 4.10 High-precision kinematics and reference analyses

Some use cases require **kinematic residuals far below typical pixel or LSF scales** (e.g. microarcsecond-class astrometry over long baselines, line-of-sight shifts at cm/s equivalent in wavelength space when the instrument model allows). The forward model still reads $Y \sim \mathbb{P}_{\psi}(\cdot \mid F_{\theta},\zeta,c)$; the following are **implementation and modeling choices**, not new axioms:

- **Numerics:** use angular representations that avoid polar singularities for accumulated proper motion (unit vectors, quaternions, or tangent-plane offsets with careful patch handoff); use **sufficient numeric precision** when differencing nearly equal angles or propagating $\omega(t)$ over long time spans.
- **Time tagging:** $\mathcal{W}$ and $\mathbf{v}(\cdot)$ should consume **timestamps at the resolution the data support** (exposure mid-time, shutter-open intervals, or per-read times); sub-exposure structure belongs in $c$ or in an extended $\mathcal{A}_{\psi}$.
- **Propagation / aberration / deflection:** any relativistic or solar-system correction can be inserted as an **explicit, swappable map** $u_{\mathrm{obs}} = \mathcal{C}_{\mathrm{prop}}(u_{\mathrm{emit}}, t, c)$ **before** the instrument—treated as part of $G_{\psi}$ or $\mathcal{A}_{\psi}$, optionally turned **off** or **replaced** by a learned residual.
- **Joint constraints:** analyses that combine **accurate distances or standardizable candles**, **wide-field astrometry / proper motions**, and **line-of-sight velocities** to infer flows or corrected redshifts are compatible with this spec: they fix **which components** live in $\zeta$, which in $F_{\theta}$, and which external $(\hat{\zeta}, \sigma)$ enter $\mathcal{L}_{\text{kin}}$.

**Reference papers (examples of the regimes this architecture is meant to accommodate—not prescriptive):**

- [MNRAS **490**, 2948 (2019)](https://academic.oup.com/mnras/article/490/2/2948/5572488?login=false)—illustrative of **distance/flow/kinematics** combinations in the literature.  
- [MNRAS **538**, 76 (2024)](https://academic.oup.com/mnras/article/538/1/76/8019798?guestAccessKey=)—illustrative of **high-precision** astrometric / spectroscopic–kinematic inference.

The spec does **not** adopt the physics conclusions of any particular paper; it only requires that **$\zeta$**, **$\mathcal{A}_{\psi}$**, and **anchors** can be set up so that similar **data products and precision goals** are representable.

---

## 5. Forward map: from abstract law to a tractable graph

The abstract object is $\mathbb{P}_{\psi}(Y \mid S, c)$. Implementations choose a **structure** for identifiability and compute:

### 5.1 Linear-functional form (optional specialization)

When a mean $\hat{Y}$ exists:


$$
\hat{Y} = \mathcal{A}_{\psi}(F_{\theta}, c), \quad Y \sim \mathbb{P}_{\psi}\big(\,\cdot \mid \hat{Y}, c\big)
$$


$\mathcal{A}_{\psi}$ is the **continuous-discrete** operator: bandpass integrals, PSF convolution, binning, Fourier sampling for interferometry, etc. **Nonlinear** detector maps fold into $\mathbb{P}_{\psi}(\cdot \mid \hat{Y}, c)$ or into $\mathcal{A}_{\psi}$ via $h(\hat{Y})$.

This aligns with `torchsky`’s **spectral quadrature** and **render_field_observation**: $\mathcal{A}_{\psi}$ is approximated with quadrature rules; $\psi$ collects PSF, WCS, and throughput.

### 5.2 Discrete-source rendering (optional specialization)

When the scene is a sum of compact sources:


$$
F_{\theta}(\omega,\lambda,t) \approx \sum_k \Phi_k(\omega,\lambda,t \mid \eta_k)
$$


the original **sparse stamp + neural ePSF** graph is a **fast $\mathcal{A}_{\psi}$** for that parameterization. It coexists with a **field-based** $F_{\theta}$ for diffuse structure (hybrid models).

### 5.3 Relation to torchsky today

- **Scene INR:** `torchsky.map.field` continuous models over sky + wavelength (time can be added as an extra input or conditioning feature).
- **Observation model:** `FieldObservation` + `render_field_observation` / `fit_field_model` implement a particular $\mathcal{A}_{\psi}$ with bandpass metadata and quadrature; `observation_builders` and `spectral_data_budget` help choose resolution along $\lambda$.
- **Detector-side INR:** not yet a single module; the spec here reserves $\psi$ for per-exposure neural fields (WCS, throughput, ePSF, structured noise) **separate** from $F_{\theta}$.
- **Kinematics:** $\zeta$ is not yet first-class in `map.field`; redshift / motion enter today mainly by **pre-shifting** observations or by choosing $F_{\theta}$ in observed $\lambda$. A natural extension is **rest-frame** $F_{\theta}$ plus explicit $\lambda_{\mathrm{obs}}(\lambda_{\mathrm{emit}},\zeta)$ inside the renderer.

---

## 6. Objective and identification

### 6.1 Data term

Replace “Gaussian pixels only” with the choice that matches $\mathbb{P}_{\psi}$:

- Heteroscedastic Gaussian on means (current simple form).
- Poisson / compound for counts.
- Student-t or Huber on residuals for heavy tails.
- Full **neural likelihood** or flow on $Y \mid \hat{Y}, c$ if $\psi$ is large enough and data support it.

**Cross-exposure linking:** Shared $F_{\theta}$ (and shared source list if used) ties epochs; **$c$** carries exposure-specific $\psi_e$ or shared $\psi$ with $c$-conditioning.

### 6.2 Constraints and anchors (critical for “sky away from detectors”)

Without anchors, **$F_{\theta}$ and $\psi$ can trade off** (e.g. ripples in “sky” vs ripples in throughput).

Use:

- **External astrometry / photometry** (e.g. Gaia) on bright objects.
- **Standard stars / spectrophotometric standards** where available.
- **Smoothness regularization** on $F_{\theta}$ in $\lambda$ and $t$ (appropriate scale from `spectral_data_budget`-style heuristics).
- **Instrument-side tying** of $\psi_e$ across time (slow drift) so variability is not absorbed entirely by $F_{\theta}$.
- **Kinematic anchors** as in §4.6: spectroscopic redshifts, RVs, proper motions with stated scales $\sigma_j$.

### 6.3 Total objective (schematic)


$$
\mathcal{L} = -\log \mathbb{P}_{\psi}(Y_{\text{obs}} \mid F_{\theta}, \zeta, c) + \mathcal{R}_{\text{scene}}(\theta) + \mathcal{R}_{\text{inst}}(\psi) + \mathcal{R}_{\text{kin}}(\zeta) + \mathcal{L}_{\text{anchors}} + \mathcal{L}_{\text{kin}}
$$


Here $\mathcal{R}_{\text{kin}}(\zeta)$ collects **generic** regularizers on kinematic parameters (e.g. smoothness of a velocity field); $\mathcal{L}_{\text{kin}}$ is reserved for **external** $(\hat{\zeta}, \sigma)$ summaries. Either term may be zero.

---

## 7. Computational notes (unchanged in spirit)

1. **Tracts / batches:** Load cutouts and a subset of parameters per sky patch.
2. **Sparse rendering:** Only evaluate where $\mathcal{A}_{\psi}$ has support (stamps for point sources; tiles for fields).
3. **Optimization:** Scale-aware optimizers; possibly different schedules for $\theta$ vs $\psi$.
4. **Uncertainties:** local curvature, bootstrap, simulation-based calibration, or sampling around $(\theta,\psi,\zeta)$—exact Hessian at survey scale is rarely the whole story.

5. **Survey-scale feasibility:** see §8—**no** claim that one optimization pass ingests every raw pixel from every facility below; **yes** claim that a **staged, modular** use of the same forward shell is realistic.

---

## 8. Feasibility: many facilities, one modeling shell

The following names are **examples** of the diversity we want to absorb—not a promise to ingest every catalog at full depth in v0.1:

**Optical / NIR imaging (wide or deep):** Euclid, LSST, UNIONS, Pan-STARRS, DES, HSC-SSP, SkyMapper, SDSS (imaging), J-PAS, GALEX, UKIDSS, WISE, SPHEREx (spectral pixels).

**Spectroscopy:** DESI, SDSS (spectra), SPHEREx (again, as integral-field-style samples), Euclid (spectro-photometric modes where applicable).

**X-ray:** eROSITA.

**CMB / millimetre (low spatial resolution, different $\mathcal{A}_{\psi}$):** Planck, ACT, SPT, WMAP.

**Astrometry / 6D phase-space anchors:** Gaia (and cross-survey proper-motion ties).

### 8.1 What is “not feasible” vs what still *is* global (honest scope)

**Clarification:** A **single global objective** with **shared latents** $(F_{\theta}, \zeta)$ and survey-specific $\psi^{(s)}$ is exactly what §8.2 recommends. Nothing in that formula forbids “global”: the latent scene **is** the device that ties Euclid, LSST, Gaia, etc. **Mathematically**, one could write one joint likelihood over all data.

What we rule out as **practically unrealistic** in the naive form is:

- **Monolithic single job:** one optimization loop that, each step, loads **every** raw pixel/photon from **every** listed archive into device memory, runs one forward–backward over the union, and updates **all** parameters at once. **Data volume, memory, and wall time** make that a non-starter at full depth; **latents shrink what you must infer, not how many bytes you condition on** unless you **change** $Y$ (compress to summaries, pre-calibrated products, etc.).
- **One uniform high-res pixelization** for the entire sky and all wavelengths from **arcsec imaging through degree-scale CMB** without **multi-resolution** or **harmonic** branches (§8.2)—again compute and conditioning, not “latents can’t help.”
- **Blind** fusion with **no** overlap, anchors, or gauge fixes: even with rich latents, **identifiability** can fail (degenerate trade-offs between $F_{\theta}$ and $\psi$) unless coupling is designed (§6.2, §4.7).

**What remains feasible and still “global” in spirit:**

- **Same** $\mathcal{L} = \sum_s \mathcal{L}_s + \text{anchors}$ optimized until stationarity, with **shared** $(\theta,\zeta)$ across surveys, using **distributed** data loading (map over tracts/exposures), **stochastic** subsampling of exposures, **consensus** or **ADMM-style** splits, or **hierarchical** latents (low-dimensional global + tract-local refinements). That is a **global fit** implemented as **many passes** over data—not one tensor holding the whole sky archive at once.

So: **latents enable global consistency**; **engineering** dictates chunked, modular, or distributed optimization, not abandonment of a single underlying joint model.

### 8.2 What is feasible (architecture)

1. **Modular likelihoods (same scene, many instruments).**  
   Write the training or inference objective as a **sum (or product) of survey-specific terms** that share **only** the pieces of $(F_{\theta}, \zeta)$ the data actually constrain:

   $$
   \mathcal{L} = \sum_{s \in \mathcal{S}} \mathcal{L}_s\big(Y^{(s)};\, F_{\theta},\, \zeta,\, \psi^{(s)},\, c^{(s)}\big) + \text{anchors}.
   $$

   Each survey $s$ carries its own **instrument block** $\psi^{(s)}$ (or $\psi^{(s)}_e$ per exposure). **$F_{\theta}$** is shared **where footprints and spectral support overlap**; elsewhere it is extrapolated or left unconstrained (masked out of $\mathcal{L}_s$).

2. **Tracts, tiles, and streaming.**  
   The sky is processed in **overlapping patches** (and, for all-sky, **hierarchical healpix-style** tiling). Only **cutouts and parameters touching those patches** load into device memory. Full-survey runs are **loops / map-reduce** over tracts, not one tensor.

3. **Multi-resolution scene or coupled fields.**  
   - **High-res branch:** $F^{(\mathrm{hi})}_{\theta}(\omega,\lambda)$ for arcsec–scale optical/NIR/X-ray cores.  
   - **Low-res branch:** $F^{(\mathrm{lo})}_{\theta}(\omega,\lambda)$ or spherical-harmonic coefficients for CMB-relevant scales; **linked** to $F^{(\mathrm{hi})}$ by a **fixed linear downprojection** (beam convolution, harmonic truncation) inside $\mathcal{A}_{\psi}$, *or* treated as **separate** latent fields with **tie** regularization where both see the same sky (e.g. large-scale dust).  
   CMB **primary** anisotropy is often modeled as **not** the same $F_{\theta}$ as stellar optical light; **feasible** fusion is **secondary** (lensing, dust, SZ) as **context** in $c$, **templates**, or a **dedicated** low-res component—not pretending one INR replaces ΛCDM Boltzmann.

4. **Spectral coverage without one giant MLP input.**  
   Use **multi-branch** INR, **spectral basis** (Fourier / PCA / learned dictionary), or **piecewise** $F_{\theta}$ per broad band family; `spectral_data_budget`-style rules set quadrature and capacity so $\lambda$ or $E$ does not explode compute.

5. **Progressive / curriculum fusion.**  
   Sensible order in practice: **(i)** anchors (Gaia astrometry/PM; photometric redshift or spec $z$ where needed); **(ii)** one well-calibrated wide band (e.g. WISE or LSST $i$) for **geometry and coarse SED**; **(iii)** add surveys by **decreasing depth or increasing systematics**; **(iv)** add spectroscopy (DESI, SDSS) to break SED–$z$ degeneracy; **(v)** add eROSITA and CMB **last** as **separate** $\mathcal{L}_s$ with clear gauge (templates, masks).

6. **Volume and operations.**  
   Petabyte-scale archives imply **epoch subsampling**, **stratified sky fractions**, and **pre-compressed** representations (HEALPix, lossy masks)—still compatible with the forward shell if $Y^{(s)}$ is defined as **what enters $\mathcal{L}_s$**, not necessarily every Level-0 file.

### 8.3 Role of each survey class (schematic)

| Class | Typical $\mathcal{A}_{\psi}$ | Coupling to shared $F_{\theta}$ |
|-------|------------------------------|-----------------------------------|
| **Gaia** | Often enters as **anchors** $\mathcal{L}_{\text{kin}}$, $\mathcal{L}_{\text{anchors}}$ on $\omega,\mu,\varpi$; can be imaging-like for XP spectra. | Ties **$\zeta$** and bright stars; stabilizes **global astrometric frame**. |
| **Wide imaging** (LSST, DES, UNIONS, Pan-STARRS, SkyMapper, SDSS img, HSC-SSP patches, J-PAS, GALEX, UKIDSS, WISE, Euclid VIS/NIR) | Pixel/spaxel integration, bandpass, PSF; per-exposure $\psi^{(s)}_e$. | **Primary** driver of $F_{\theta}(\omega,\lambda)$ on matched $\lambda$ bins. |
| **Spectroscopy** (DESI, SDSS spec, SPHEREx voxels) | Fiber or slit LSF, sky subtraction residuals in $\psi$. | **Strong** constraints along $\lambda$; breaks color–redshift degeneracy when $\zeta$ includes $z$ or $v$. |
| **eROSITA** | Energy-bin ARF/RMF-style response (approximate in $\psi$). | Same $F_{\theta}$ if defined in **energy** or mapped $\lambda\leftrightarrow E$; **separate** $\psi^{(\mathrm{X})}$. |
| **CMB** (Planck, ACT, SPT, WMAP) | Harmonic analysis, large beam, often **anisotropy + foreground** separation. | **Weak** direct coupling to arcsec $F_{\theta}$; **feasible** via **shared large-scale foreground** INR or **$c$** (lensing maps, $\tau$) feeding **both** branches. |

### 8.4 Euclid, SPHEREx, and “hard” cases

- **Euclid:** weak lensing **shear** and **photo-$z$** are not automatically a single scalar $F_{\theta}$; **feasible** path is **extra tensor / distortion parameters** in $\zeta$ or $\psi$ acting on $F_{\theta}$, or a **second field** for lensing deflection potential at **low resolution** tied to mass tracers—not claiming one RGB-style INR does all science products.
- **SPHEREx:** all-sky × many $\lambda$ bins → **tracts + spectral sparsity / bases**; same formula $\mathcal{L}_s$, different batching.
- **UNIONS / regional surveys:** smaller footprint → good **pilot** regions to validate **multi-survey** $\mathcal{L}_s$ before LSST-scale runs.

### 8.5 Bottom line

The **plan is feasible** if we treat **“Euclid + LSST + … + WMAP”** not as one undifferentiated tensor, but as **many terms** $\mathcal{L}_s$ sharing **where the sky is jointly observed**, with **survey-specific** $\mathcal{A}_{\psi^{(s)}}$, **tract-wise** compute, **multi-resolution** links, and **anchors** (especially **Gaia**) so $F_{\theta}$ and $\psi$ do not collapse into each other.

---

## 9. Plan: INR for the sky away from detectors (roadmap)

**Goal:** A **single continuous scene** $F_{\theta}(\omega, \lambda, t)$ (or patchwise collection with overlap) that explains multi-epoch, multi-band data **without** baking detector defects into the sky.

### Phase A — Static sky field (current torchsky core)

- **$F_{\theta}$:** `FieldDomain` + field MLP / encoder fields over $(\alpha,\delta,\lambda)$ on a tangent patch or sphere.
- **$\mathcal{A}_{\psi}$:** `FieldObservation` bandpass integrals + PSF if present; use `spectral_data_budget` / `suggest_num_spectral_samples` for quadrature.
- **Deliverable:** Joint fit across bands for a patch; scene held fixed in $t$.

### Phase B — Spectral resolution and builders

- Ingest **binned spectra / IFU / X-ray channels** via `observation_builders` so $F_{\theta}$ sees consistent $\lambda$ (or $E$) support.
- **Capacity:** tie `suggest_spectral_model_capacity` to architecture width for $\lambda$.

### Phase C — Time in the scene

- Add **$t$** input to $F_{\theta}$ (Fourier features in MJD, per-night embedding, or piecewise-constant segments).
- **Regularization:** temporal smoothness or sparse variability (only some pixels/volumes time-varying).
- **Context $c$:** still carries exposure time for $\mathbb{P}_{\psi}$ even when $F_{\theta}$ is static (instrument drift).

### Phase D — Explicit detector-side INR block $\psi$

- Per-exposure (or per-night) **neural maps** for throughput, background structure, and **ePSF** conditioned on local data statistics—not on sky parameters $\theta$.
- **Training constraint:** freeze $F_{\theta}$ on a subset of exposures or regularize $\psi$ so **high-frequency features** preferentially live in $\psi$, not in $F_{\theta}$ (except where physically sky-like).

### Phase E — Rich likelihood $\mathbb{P}_{\psi}(Y \mid \hat{Y}, c)$

- Move beyond diagonal Gaussian: correlated noise models, per-amplifier effects, optional learned residual distribution.
- **Validation:** simulations + null tests on residuals vs $c$.

### Phase F — Hybrid scene

- **$F_{\theta}$** for diffuse / crowded / structured background; **discrete** $\sum_k \Phi_k$ for bright point-like objects; shared $\mathcal{A}_{\psi}$.

### Phase G — Kinematics in the loop

- **Rest-frame** $F_{\theta}(\omega,\lambda_{\mathrm{emit}},t)$ with learnable or anchored **$z_k$ / $v_{\mathrm{rad},k}$** per source (or a low-D field for gas).
- **Multi-epoch astrometry:** joint $\omega_0, \boldsymbol{\mu}, \varpi$ with optional **$\mathcal{L}_{\text{kin}}$** from Gaia (or competitors).
- **Peculiar / residual velocities:** implement §4.9—reference field $\mathbf{v}_{\mathrm{ref}}(\cdot;\kappa)$ plus learned $\mathbf{v}_{\mathrm{pec}}$ (sparse tracers, grid, or field); no fixed cosmology.
- **High-precision path:** stable $\omega(t)$ integrators, fine time tags, optional $\mathcal{C}_{\mathrm{prop}}$ (§4.10); target regimes such as [MNRAS 490, 2948](https://academic.oup.com/mnras/article/490/2/2948/5572488?login=false) and [MNRAS 538, 76](https://academic.oup.com/mnras/article/538/1/76/8019798?guestAccessKey=).
- **Rotation / broadening:** add $\zeta$ that widen rest-frame lines or morphologies before $\mathcal{A}_{\psi}$.
- **Platform motion:** prefer **$c$** for satellite scan law; put **residual ephemeris error** in $\zeta$ when calibrating science products.

### Phase H — Multi-survey feasibility (§8)

- Implement **$\mathcal{L} = \sum_s \mathcal{L}_s$** with **per-survey** $\psi^{(s)}$ and **shared** $F_{\theta}$ only on overlaps; tract / healpix outer loop.
- Pilot **overlap regions** (e.g. DES+SDSS+WISE, or UNIONS-like patches) before full LSST depth.
- Add **low-res CMB / foreground** branch or **$c$**-only coupling before claiming joint CMB–arcsec scene recovery.
- **Euclid / SPHEREx:** spectral–spatial batching and, for lensing, **explicit** extensions beyond scalar intensity $F_{\theta}$.

---

## 10. Summary

| Layer | Role | Typical parameterization |
|-------|------|---------------------------|
| **Scene $F_{\theta}$** | Sky away from detector (at chosen boundary) | INR over $(\omega,\lambda,t)$ or rest $\lambda_{\mathrm{emit}}$; optional + discrete sources |
| **Kinematics $\zeta$** | Redshift, transverse motion, rotation, acceleration, bulk flows, **peculiar (reference-split) velocities** | Learned, fixed, or anchored with external $(\hat{\zeta},\sigma)$; $\mathbf{v}_{\mathrm{ref}}$ user-defined (§4.9–4.10) |
| **Context $c$** | When / where / how data were taken | Metadata + time stamps + platform ephemeris |
| **Law $\mathbb{P}_{\psi}$** | Instrument + environment + noise | Physics + neural corrections |
| **Data $Y$** | Raw or early pipeline products | Pixels, bins, events, visibilities |

**One line (display):**

$$
Y \sim \mathbb{P}_{\psi}\big(\,\cdot \mid F_{\theta},\, \zeta,\, c(t)\,\big)
$$

with $F_{\theta}$ the sky INR, $\zeta$ the kinematic latents, and $\mathbb{P}_{\psi}$ the detector–world model (learned or explicit).

This is the architecture torchsky’s field modules should **grow toward**: **$F_{\theta}$** in `map.field` (and sphere), **$\mathcal{A}_{\psi}$** and **$\mathbb{P}_{\psi}$** as separate, swappable layers that consume `FieldObservation`-shaped bindings to the data.

**Feasibility at ensemble scale (Euclid, LSST, eROSITA, UNIONS, Pan-STARRS, DESI, SDSS, HSC-SSP, DES, SkyMapper, Gaia, J-PAS, GALEX, UKIDSS, WISE, SPHEREx, Planck, ACT, SPT, WMAP, …):** modular $\mathcal{L}_s$, tracts, multi-resolution branches, and Gaia/CMB-appropriate coupling—§8—not a single monolithic fit over every raw pixel.
