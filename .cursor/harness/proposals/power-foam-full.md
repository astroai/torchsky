# Full Power Foam for Torchsky: primary-source audit and implementation plan

Date: 2026-07-22
Status: bounded proposal; no public API commitment
Scope: exact Power Foam audit, astronomy translation, staged Torch-native implementation

## Decision

Torchsky should implement a **domain-adapted Power Foam sky field**, not copy the
reference renderer and not relabel a Gaussian or softmax residual field as Foam.
The method deserves the Power Foam name only after it has all of the following:

1. hard bounded power-cell geometry;
2. an overlap-safe sparse adjacency/candidate structure;
3. a flux-conserving, coverage-aware renderer whose geometry gradients do not rely
   only on point-sampled `argmin` ownership;
4. local detail sites decoupled from coarse cell geometry;
5. learned per-cell radii and an evidence-driven densify/prune lifecycle;
6. the existing Torchsky WCS, pixel-footprint, bandpass, PSF, background, mask,
   weight, and exposure-pose observation operators;
7. a multi-wavelength appearance model with shared spatial geometry; and
8. correctness, scaling, and scientific-depth evidence against strong classical,
   single-band, and INR baselines.

The first implementation should be experimental and patch-local. A sphere-native
weighted-cap implementation is a later required milestone for all-sky use; a
single tangent plane must not be advertised as full-sky Foam.

## Primary sources and provenance

This audit used only the authors' primary materials:

- Govindarajan et al., *Power Foam: Unifying Real-Time Differentiable Ray Tracing
  and Rasterization*, arXiv:2604.24994 (2026):
  <https://arxiv.org/abs/2604.24994>
- Author project page: <https://powerfoam.github.io/>
- Official implementation: <https://github.com/theialab/powerfoam>
- Code snapshot audited on 2026-07-22:
  [`96392252ebd0059fe6ca98881b62e12295d9242f`](https://github.com/theialab/powerfoam/tree/96392252ebd0059fe6ca98881b62e12295d9242f)
- Official code license at that snapshot: Apache-2.0:
  <https://github.com/theialab/powerfoam/blob/96392252ebd0059fe6ca98881b62e12295d9242f/LICENSE>

The official repository explicitly describes itself as non-final research code.
Its reference environment is Linux/CUDA with PyTorch 2.9.1 and Warp 1.10.0. That
is not compatible with Torchsky's CPU/CUDA/MPS tensor-native runtime contract, so
Torchsky should implement the published mathematics independently in PyTorch. Do
not copy the Warp kernels. If code is ever copied or translated closely, retain
the Apache-2.0 notices, state modifications, and add the required third-party
license material; Torchsky itself is MIT-licensed.

This proposal uses “reference Power Foam” for the paper's literal 3-D graphics
method and “Power Foam sky field” for the astronomy-specific 2-D/2+spectral
method below. Documentation must preserve that distinction.

## What the reference algorithm actually is

### 1. Bounded power cells

For site center \(p_i\in\mathbb{R}^d\) and learned radius \(r_i\geq0\), the
power distance is

\[
\pi_i(x)=\|x-p_i\|_2^2-r_i^2.
\]

The ordinary power cell is

\[
P_i=\{x:\pi_i(x)\leq\pi_j(x)\ \forall j\},
\]

and the rendered bounded cell is

\[
B_i=P_i\cap\{x:\|x-p_i\|_2\leq r_i\}.
\]

For neighboring sites \(i,j\), the radical plane is

\[
(p_j-p_i)^\top x =
\tfrac12\left(\|p_j\|^2-\|p_i\|^2+r_i^2-r_j^2\right).
\]

The same learned radius therefore moves both the inter-cell faces and the
spherical bound. This coupling is essential. An ordinary Voronoi partition
multiplied by an unrelated compact window is not the paper's construction.

### 2. Čech adjacency as an exact rendering superset

The minimal dual adjacency of the bounded power diagram is the weighted
alpha-complex. Rebuilding the full regular triangulation during optimization is
expensive, so the paper uses the Čech graph: connect \(i,j\) when

\[
\|p_i-p_j\|_2 < r_i+r_j.
\]

This graph contains every necessary bounded-cell face. Extra Čech edges add
radical planes that cannot cut the true bounded cell, so they cost tests but do
not change the rendered result. The official code builds the graph by BVH AABB
queries followed by exact sphere-overlap tests and stores it in CSR form.

This is not optional for a scalable implementation. A chunked all-pairs matrix
is a useful oracle, not a survey-scale renderer.

### 3. Exact cell intersection and rendering

The reference ray renderer intersects a ray with a cell's bounding sphere, then
clips the resulting interval against the radical plane of every Čech neighbor.
The rasterizer culls bounding spheres to image tiles, sorts cells globally by
power distance from the camera, performs the same interval clipping per pixel,
and evaluates the exact piecewise-constant volume-rendering integral. The paper
proves that power-distance camera ordering is pop-free.

For a clipped interval \([t_0,t_1]\) and density \(\sigma_i\), the official code
uses

\[
\Delta\log T_i=-\sigma_i(t_1-t_0),\qquad
\alpha_i=1-\exp(\Delta\log T_i),
\]

then standard front-to-back transmittance compositing. It implements explicit
derivatives of ray-sphere, ray-radical-plane, and ray-oriented-plane
intersections with respect to centers and radii. Consequently, the learned
geometry receives boundary gradients. A hard `argmin` evaluated only at pixel
centres does **not** reproduce this differentiability.

### 4. Oriented-point or dipole surface

Each 3-D cell has an oriented point \((p_i,n_i)\). Its internal plane separates a
learned high-density half from a fixed zero-density half. This replaces the two
high/low-density cells needed by Radiant Foam to form one surface and is important
for 3-D geometry efficiency.

### 5. Local detail sites

The reference method decouples coarse cell geometry from high-frequency surface
appearance. Each oriented plane carries \(K\) local detail sites \(s_{ik}\). At a
surface point \(x\), displacement and appearance are normalized local mixtures.
The paper writes weights proportional to `exp(-tau * distance)`; the audited code
uses the squared, radius-normalized form

\[
a_{ik}(x)=
\frac{\exp[-\tau\|x-s_{ik}\|^2/r_i^2]}
{\sum_l\exp[-\tau\|x-s_{il}\|^2/r_i^2]},
\]

with \(\tau=10\), eight detail sites per cell, a scalar normal displacement per
site, and a Spherical Voronoi directional-radiance function per detail site. This
soft Voronoi texture is **inside a hard bounded power cell**. It is not soft power
ownership between cells.

### 6. Optimization lifecycle and losses

The paper initializes sites from a sparse Structure-from-Motion point cloud,
learns centers/radii/orientation/density/detail positions/displacement/appearance,
and uses:

\[
\mathcal L=\mathcal L_{RGB}+\lambda_1\mathcal L_{SSIM}
+\lambda_2\mathcal L_{normal}+\lambda_3\mathcal L_{sparse}
+\lambda_4\mathcal L_{connect}.
\]

The three specialized regularizers are:

- normal: contribution-weighted squared positive dot product between face normal
  and camera ray;
- sparsity: accumulated opacity-times-transmittance contribution; and
- connectivity:

  \[
  \mathcal L_{connect}(i)=\sum_{j\in\mathrm{Cech}(i)}
  \max(r_i+r_j-\|p_i-p_j\|,0)^2.
  \]

Per-cell photometric error and contribution are tracked by exponential moving
averages. Every 100 steps in the densification interval, low-contribution cells
are pruned and high-error survivors are sampled for duplication. Duplicates are
perturbed tangentially to the oriented surface and optimizer moments are carried
forward. The Čech graph is rebuilt increasingly sparsely, capped at every 20
iterations. The paper's reported training grows to roughly 500,000 cells for
indoor scenes and 1.2 million for outdoor scenes.

Steiner points are added only for efficient traversal of otherwise empty 3-D
space after training. They are not learned radiance primitives.

## Essential versus graphics-specific components

| Reference component | Sky-field decision | Reason |
|---|---|---|
| Learned centers and radii | Keep exactly | They define both power faces and compact support. |
| Hard bounded power cells | Keep exactly | This is the representation's identity and gives non-overlap/locality. |
| Čech overlap graph | Keep exactly | Safe sparse superset of required faces/candidates. |
| Tile culling | Keep, in detector/WCS form | Required for survey-scale rendering. |
| Boundary-aware geometry gradients | Keep | Otherwise centers/radii cannot be learned faithfully. |
| Local soft Voronoi detail sites | Keep, spectralize values | Decouples coarse geometry from fine sky appearance. |
| Learned radius schedule | Keep | The paper's strongest ablation failure is fixed radii. |
| Contribution/error EMA densification and pruning | Keep, change evidence | Adaptive allocation is part of the usable representation. |
| 3-D opacity/transmittance | Replace | Astronomical coaddition observes 2-D specific intensity, not an occluding NeRF volume. |
| Oriented dipole plane and normal loss | Omit for 2-D sky | There is no front/back volumetric surface in ordinary sky maps. |
| Normal displacement | Omit initially | No latent 3-D surface; subpixel morphology belongs in 2-D detail texture. |
| Directional RGB/Spherical Voronoi view model | Replace with continuous SED | Wavelength/bandpass, not view direction, is the relevant appearance variable. |
| Painter ordering | Omit | Additive sky intensity has no depth occlusion. |
| Ray adjacency walk and Steiner cells | Omit | No empty-volume traversal in a 2-D celestial field. |
| SfM initialization | Replace | Use cross-exposure sky residual evidence and/or coarse sky intensity. |
| L2+SSIM | Replace as default | Use the existing mask/weight/noise likelihood; SSIM is not a calibrated noise model. |

The omissions above do not make the sky field a partial implementation. They are
literal consequences of changing the physical domain from an occluding 3-D scene
to a radiometric field on the celestial sphere. What must not be omitted is the
bounded power geometry, sparse exact semantics, local detail texture, adaptive
lifecycle, and boundary-aware renderer.

## Mathematical contract for a multi-wavelength sky field

### Patch-local geometry

For a tangent-plane coordinate \(u\in\mathbb R^2\), use the exact bounded cells
\(B_i\) above. Let `owner(u)` be `i` when \(u\in B_i\), otherwise `-1`.

Each cell has \(K\) local detail sites \(s_{ik}\in\mathbb R^2\), expressed in
radius-normalized local coordinates. Their spatial weights are

\[
a_{ik}(u)=\frac{\exp[-\tau\|(u-s_{ik})/r_i\|^2]}
{\sum_l\exp[-\tau\|(u-s_{il})/r_i\|^2]}.
\]

Use a shared continuous spectral basis \(\phi_m(\lambda)\), learned or fixed, and
per-detail coefficients \(c_{ikm}\):

\[
S_{ik}(\lambda)=b_k+\sum_{m=1}^{M}c_{ikm}\phi_m(\lambda),
\qquad
F_i(u,\lambda)=\sum_{k=1}^{K}a_{ik}(u)S_{ik}(\lambda).
\]

The initial composite field should be

\[
I(u,\lambda)=g\!\left(I_{base}(u,\lambda)+
\mathbf 1[owner(u)=i]F_i(u,\lambda)\right),
\]

where `I_base` is any existing `_BaseFieldModel` and `g` is explicit policy:
identity for a signed residual model, or a documented positive transform for a
physical intensity model. Do not silently clamp negative predictions; that hides
likelihood bias and destroys gradients.

Shared spatial cells are the correct first multi-wavelength inductive bias. A
power tessellation in `(x, y, wavelength)` mixes unlike units, changes spatial
topology across broad bandpass integration, and gives capacity where broadband
data may have no spectral rank. Wavelength-dependent geometry can be considered
later as a low-rank perturbation only after identifiability tests.

The existing `render_field_observation` path then computes, for each exposure and
pixel, the same WCS/pixel/bandpass/PSF/background prediction as other Torchsky
fields. This is the mechanism by which faint spatial evidence can be shared
across bands without pretending that all bands have equal flux or PSFs.

### Sphere-native geometry

For all-sky use, sites are unit vectors \(v_i\in\mathbb S^2\) with angular support
radii \(\rho_i\). Define chord weights \(w_i=4\sin^2(\rho_i/2)\) and

\[
\pi_i(u)=\|u-v_i\|^2-w_i=2(1-u^\top v_i)-w_i,
\qquad u\in\mathbb S^2.
\]

The bounded spherical cell is the minimum-power region clipped to the angular
cap `acos(u dot v_i) <= rho_i`. Equality of two power distances is the
intersection of the sphere with a plane; the local tangent limit recovers the
Euclidean formulation. Candidate caps overlap when their centre separation is
less than \(\rho_i+\rho_j\).

This weighted-cap/Laguerre construction must be implemented and tested directly,
or the sky must be covered by an explicit overlap-safe tangent atlas. Treating
unit vectors as unconstrained ordinary 3-D Foam sites creates Euclidean balls off
the sphere and is not a valid all-sky construction.

## Staged implementation

### Stage 0 — provenance and terminology gate

- Keep this primary-source audit with a pinned official commit.
- Add a short user-facing reference note before exposing a model name.
- Call the current Gaussian residual wrapper `ResidualTileBasisField`; never call
  it Foam.
- Keep all new APIs under `torchsky.map.field` experimental until Stage 5.

Exit gate: docs distinguish exact cells, soft optimization surrogates, local soft
texture, and graphics-specific parts.

### Stage 1 — exact dense geometry oracle

Recommended minimal surface: one `foam.py` module plus one focused test file.

- `power_distance(points, sites, radii)`.
- `bounded_power_ownership(...)` with owner `-1` outside all bounds.
- `power_face_planes(...)` using the exact radical-plane equation.
- a chunked all-pairs `cech_adjacency(...)` oracle.
- hard and explicitly named soft-surrogate ownership modes.
- strict device/dtype/finite/non-negative-radius validation.

The dense implementation is a reference oracle only. Chunking bounds memory but
does not change its \(O(QN)\) ownership or \(O(N^2)\) adjacency work.

Exit gate: brute-force randomized parity in 2-D and 3-D; equal radii reduce to
Voronoi; translation/rotation/uniform-scale equivariance; exact inside/outside
semantics; chunked parity; finite gradients away from topology changes.

### Stage 2 — sparse Čech graph and query index

Tangent patches:

1. map each bounding disk AABB to integer uniform-grid bins;
2. sort `(bin_key, site_id)` pairs with Torch operations;
3. form CSR bin ranges;
4. generate unique candidate pairs only within shared/neighboring bins;
5. apply the exact disk-overlap predicate; and
6. store symmetric site adjacency as CSR.

Detector queries use a second CSR mapping from output tiles to candidate disks.
Build topology under `no_grad`; centers/radii remain differentiable in subsequent
intersection calculations. Rebuild after optimizer steps on an explicit schedule
or when a conservative displacement/radius-change bound invalidates the index.

All-sky indexing uses a coarse nested HEALPix level and exact angular-cap overlap
tests. A cap is inserted into every coarse pixel it intersects, including boundary
neighbors. The HEALPix level is chosen from the largest support radius and a
measured occupancy target; it is not a correctness threshold because final
angular tests remain exact.

Exit gates:

- sparse edge set equals the dense Čech oracle on randomized/adversarial inputs;
- no missed candidate under index rebuild thresholds;
- deterministic CSR ordering;
- measured candidate reduction and peak memory;
- no default dependency on SciPy, Warp, Astropy, or CUDA.

### Stage 3 — coverage-aware differentiable 2-D renderer

For cell \(i\), construct

\[
B_i=D(p_i,r_i)\cap\bigcap_{j\in\mathrm{Cech}(i)}
\{u:\pi_i(u)\leq\pi_j(u)\}.
\]

Rasterization must cull disks to detector tiles through Torchsky WCS bounds, then
integrate cell contribution over detector pixel footprints. Two complementary
modes are required:

- `hard`: exact cell ownership with adaptive area quadrature or clipped-cell
  coverage, used for evaluation and product generation;
- `train`: coverage-aware differentiable integration, or a documented annealed
  surrogate whose hard-render discrepancy is measured every validation epoch.

A practical Torch-native path is convex half-plane clipping followed by
disk-boundary tessellation with a user-visible maximum sagitta/area error. This
is a mathematically controlled approximation, unlike fixed supersampling. The
cell/pixel intersection can then use polygon clipping and area-weighted local
texture quadrature. Exact circular-arc integration may replace this after the
profile identifies clipping as the bottleneck.

Do not claim Stage 3 complete if geometry is learned only through hard point
samples: `argmin` has no useful boundary derivative. At minimum, finite-difference
tests must show centre and radius derivatives of integrated pixel flux on cases
where a boundary crosses a pixel.

The renderer must preserve:

- total flux for constant cell texture;
- current 2x2 Gauss-Legendre pixel integration as the cheap baseline;
- arbitrary Torchsky WCS including distorted sampling;
- PSF/background application exactly once;
- tiling parity and bounded memory; and
- CPU, CUDA, and MPS semantics.

Exit gate: hard render matches a dense reference within the declared integration
error; train/hard mismatch is bounded; finite-difference geometry gradients pass;
constant-flux and dither invariance pass; no full `Q x N` allocation.

### Stage 4 — spectral detail field and existing observation operators

- Implement `PowerFoamSpectralField(_BaseFieldModel)` with a base field, shared
  centers/radii, `K` local detail sites, and low-rank continuous SED values.
- Reuse `FieldDomain`, `FieldObservation`, `ForwardContext`, bandpass quadrature,
  exposure pose, spatial quadrature, PSF/background bundles, and field training.
- Provide `render_bandpass` so the spectral basis is evaluated/integrated in one
  batched call rather than repeating ownership work per wavelength sample.
- Bound memory by detector tiles, spectral chunks, and candidate count.
- Use `spectral_data_budget` to cap spectral rank. Broadband data must not silently
  fit more independent SED modes than the response matrix constrains.
- Add the sphere-native geometry backend before enabling all-sky domains.

Exit gate: monochromatic and band-integrated analytic spectra match known truth;
shared-geometry multi-band gradients are finite; held-out bands expose rather
than hide spectral extrapolation failure; tangent and spherical local-limit parity
passes.

### Stage 5 — adaptive optimization lifecycle

Initialization must use sky-consistent evidence:

1. fit or provide a diffuse base INR;
2. render all exposures through their own operators;
3. backproject whitened residual evidence to a common sky grid;
4. require repeat support across independent exposures before creating a sky cell;
5. initialize radii from local neighbor/support scales, clipped by PSF/pixel
   identifiability bounds; and
6. initialize detail spectra from multi-band weighted residuals.

Maintain per-cell EMAs of:

- contribution to predicted detector likelihood;
- whitened residual improvement, by exposure and band;
- support count across independent exposures; and
- local hard-versus-training-render discrepancy.

Lifecycle operations:

- **densify/split:** sample high-evidence cells and duplicate along the principal
  tangent direction of backprojected residual structure, not from one detector
  tile alone;
- **prune:** remove cells with consistently negligible cross-exposure likelihood
  contribution, after a grace interval;
- **merge (astronomy extension):** merge near-degenerate neighboring cells only
  when held-out likelihood does not worsen and spectral textures agree;
- **rebuild:** update the sparse index/Čech CSR after topology changes and when
  conservative movement bounds require it.

Carry optimizer state through split/prune deterministically. Save lifecycle events
and random-generator states in checkpoints and map plan JSON.

Use the observation likelihood already supported by Torchsky (`gaussian_nll` when
calibrated inverse variance is available, otherwise an explicitly chosen robust
loss). Add named regularizers:

- Čech overlap/connectivity, using the paper's equation;
- contribution sparsity;
- spectral smoothness or basis shrinkage only where the bandpass rank requires it;
- optional cell-count/description-length penalty; and
- train/hard renderer consistency during surrogate training.

Do not port the graphics normal loss. Do not use one fixed paper threshold across
surveys. Defaults must be scale-normalized and justified by validation curves;
all lifecycle thresholds remain explicit configuration.

Exit gate: deterministic resume, optimizer-state correctness, no single-exposure
artifact promotion, and ablations showing learned radii, local detail sites,
connectivity, densification, and pruning each have the expected effect.

### Stage 6 — product and CLI integration

Only after Stages 1–5:

- add a concise experimental `torchsky map` mode for joint-band field fitting;
- include method, geometry backend, integration tolerance, spectral rank,
  lifecycle schedule, random seed, input hashes/HDUs, and operator summaries in
  `--plan-json`;
- write rendered intensity, inverse-variance/uncertainty, support, residual, cell
  id/contribution, and provenance products through `torchfits`;
- make dry-run report estimated cell count, candidate occupancy, device, memory,
  and expected operator work; and
- never make the Foam path the default until the scientific benchmark gate passes.

## Validation and benchmark programme

### Geometry correctness

- randomized dense owner and Čech parity in 2-D/3-D;
- equal-radii Voronoi reduction;
- radical-plane points have equal power distance;
- every rendered point lies inside its owner's disk and all owner halfspaces;
- Čech supersets every active clipped-cell face;
- duplicates, zero radii, tangent disks, ties, empty cells, huge dynamic range,
  float32/float64, and seam/pole cases;
- translation, rotation, and uniform-scale equivariance;
- sparse versus dense parity and deterministic topology rebuilds.

### Renderer and autodiff correctness

- analytic disk/pixel and two-cell boundary cases;
- flux conservation for constant and linear local texture;
- adjoint dot-product checks whenever the fixed-geometry renderer is linear in
  cell values;
- finite differences for centers, radii, detail sites, spectral coefficients, and
  base-field parameters away from true topology singularities;
- hard-versus-training renderer error as a function of integration tolerance;
- tiled/untiled parity and WCS seam/distortion cases;
- PSF and background applied once; dithered exposures predict consistently.

### Performance and scaling

Report, separately for CPU/CUDA/MPS where available:

- site count, query/pixel count, edges, mean/max degree, candidates per tile;
- index build/rebuild time, render forward/backward time, and peak memory;
- dense-oracle versus sparse crossover;
- small, medium, and survey-like scales, including at least 1e5 and 1e6 sites on
  hardware that can hold them;
- sensitivity to support-radius distribution and clustered sources; and
- comparison with the existing Gaussian residual field and strongest INR presets.

No native extension is justified until Torch-native sparse structure is correct
and profiling identifies a stable hotspot.

### Multi-wavelength “go deeper” experiment

The central scientific claim is not that combining bands magically increases
flux in every band. It is that shared spatial structure plus learned continuous
SEDs can improve detection and flux inference for sources whose morphology is
coherent across wavelengths, while preserving honest uncertainty for unusual
colors.

Construct a preregistered injection benchmark with:

- point sources, compact galaxies, extended low-surface-brightness galaxies, and
  diffuse structure;
- realistic source blends and a distribution of smooth SEDs;
- deliberately adverse populations: dropouts, strong emission-line sources,
  sharp color gradients, and morphology changing with wavelength;
- independent exposures per band with different dithers, WCS distortions, PSFs,
  masks, backgrounds, and calibrated heteroscedastic noise;
- a held-out exposure in every represented band and at least one held-out band
  when the spectral setup permits it; and
- blank-sky injections to measure false positives.

Compare at minimum:

1. per-band inverse-variance coadds;
2. per-band drizzle/conservative coadds;
3. a strong classical multi-band detection image or joint likelihood baseline;
4. single-band INR fits;
5. single-band Power Foam sky fields;
6. joint multi-band INR with the same observation operators; and
7. joint multi-band INR + Power Foam with matched parameter/compute budgets.

Primary metrics:

- 50% and 90% completeness depth at fixed false-discovery rate;
- forced-photometry flux bias, scatter, and interval coverage by band;
- color bias and catastrophic-color rate;
- deblending completeness/purity versus separation;
- surface-brightness recovery and total-flux conservation;
- held-out-exposure Gaussian NLL/chi-square and residual spatial whiteness;
- morphology error after PSF matching; and
- throughput, peak memory, and parameter count.

Report metrics by source/SED class, not only an average. A joint method that helps
ordinary continuum sources but hallucinates dropout flux is not a win. Declare a
depth gain only at a fixed false-positive rate and with calibrated uncertainty.
Use identical train/validation/test injections and operator metadata across all
methods; tune on validation only.

After the synthetic gate, run the same blinded-injection protocol on a real
multi-band exposure set. Inject sources before resampling into individual detector
images, retain untouched sky cutouts for false-positive measurement, and keep the
truth catalog hidden from optimization.

## Dependencies and architecture constraints

- Core runtime remains Torch + NumPy + existing justified kernels; no `warp-lang`,
  SciPy, Astropy, CGAL, or CUDA-only dependency.
- FITS I/O remains in `torchfits`; Foam consumes tensors/headers and writes through
  existing product/CLI boundaries.
- Geometry accepts in-memory tensors and does not know about files.
- Dense reference functions stay available for tests and tiny inputs but are
  documented as quadratic.
- Public field APIs remain thin; lifecycle/index objects stay internal until two
  independent callers establish a stable interface.
- Checkpoints carry a schema version and enough topology/provenance state for
  deterministic resume.

## Definition of done

Torchsky can call this a proper Power Foam sky implementation when:

1. dense and sparse bounded-power semantics agree exactly;
2. there is no all-pairs allocation in the survey renderer;
3. detector-pixel flux is coverage integrated with validated geometry gradients;
4. local spectral detail is continuous and bandpass integrated through the shared
   observation operators;
5. centers and radii are learned with deterministic split/prune/rebuild;
6. tangent and sphere-native geometry both pass seam/local-limit tests;
7. CPU/CUDA/MPS behavior is tested to the surrounding module's contract;
8. full products and plans are reproducible; and
9. the blinded multi-wavelength benchmark shows where joint Foam goes deeper,
   where it is neutral, and where spectral/morphological mismatch limits it.

Before these gates, releases should use wording such as “experimental bounded
power-diagram primitives” or “Power-Foam-derived prototype,” not “full Power
Foam.”
