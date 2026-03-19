# Architectural Specification: End-to-End Differentiable Forward Modeling for Astronomical Surveys

## 1. Rationale & Problem Statement
Traditional astronomical survey pipelines operate sequentially: they estimate backgrounds, extract sources on single images, cross-match catalogs, derive astrometric/photometric calibrations, and finally co-add data. 

This sequential paradigm suffers from fundamental mathematical flaws:
1.  **Loss of Covariance:** Errors in spatial calibration (WCS) propagate into photometric extraction, but the cross-terms of the covariance matrix are permanently lost between pipeline stages.
2.  **Correlated Noise:** Resampling and co-adding images correlates pixel noise and destroys the pristine Poisson statistics of the raw data.
3.  **Heuristic Cross-matching:** Assigning detections across different exposures or filters via geometric distance (e.g., nearest-neighbor matching) breaks down in crowded fields and ignores the continuous nature of the data.

**The Proposed Solution:** We abandon the pipeline. Instead, we formulate the entire survey as a single, global **Inverse Problem**. We construct a fully differentiable generative forward model that simulates the raw Level-0 pixel data from a latent state of the universe. By optimizing this model via gradient descent against the raw pixels, we simultaneously solve for global astrometry, forced multi-band photometry, and complex instrumental calibrations.

---

## 2. The Global Parameter Space (Latent State)
We define the system not by what we measure, but by the physical state we wish to infer. The total state vector $\Theta = \{\mathcal{U}, \Phi\}$ is divided into the Universe ($\mathcal{U}$) and the Instrument ($\Phi$).

### 2.1 The Multi-Band Universe ($\mathcal{U}$)
The universe consists of $K$ discrete sources. A single source $k$ possesses exactly one physical position, but multiple intrinsic fluxes corresponding to the survey's filter bands, denoted by $\mathcal{B}$.
* $\mathbf{p}_k = (\alpha_k, \delta_k)$: The true celestial coordinates (RA, Dec).
* $\mathbf{F}_k = \{F_{k,b}\}_{b \in \mathcal{B}}$: A vector of intrinsic, calibrated fluxes for every band.

### 2.2 The Implicit Instrument State ($\Phi$)
Simulating complete semiconductor physics (Brighter-Fatter electron repulsion, Charge Transfer Inefficiency) requires full-image, sequential convolutions, which destroys the ability to parallelize rendering on GPUs. 

To achieve computational tractability while maintaining supreme accuracy, we replace explicit physical calibrations with **Implicit Neural Fields** for each exposure $e$:
* **Geometric WCS Mapping ($\mathcal{M}_e$):** Maps true celestial coordinates to pixel coordinates $(u, v) = \mathcal{M}_e(\alpha, \delta \mid \mathbf{W}_e)$, parameterized by polynomial weights $\mathbf{W}_e$.
* **Effective Throughput Field ($\mathcal{T}_{\text{eff}, e}$):** A spatially varying scaling factor $\mathcal{T}_{\text{eff}, e}(u, v)$ that implicitly absorbs atmospheric extinction, zero-points, and flat-field (quantum efficiency) variations.
* **Effective Background Field ($\mathcal{B}_{\text{eff}, e}$):** A smoothly varying spatial map $\mathcal{B}_{\text{eff}, e}(u, v)$ that implicitly absorbs the true sky background, master bias, dark current, and fringe patterns.
* **Neural Effective PSF (ePSF, $\psi_{\text{eff}, e}$):** A non-linear function (parameterized by a small Multi-Layer Perceptron) that dictates the shape of a point source. 

Critically, the ePSF is conditioned on the source's flux and the local background:
$$\psi_{\text{eff}, e} \Big( \Delta u, \Delta v \mid u, v, f_{\text{inst}}, \mathcal{B}_{\text{eff}, e}(u, v) \Big)$$
*By making the ePSF flux-dependent and background-dependent, the neural network can implicitly learn and reproduce the widening of bright stars (Brighter-Fatter) and readout smearing (CTE) without requiring sequential pixel simulation.*

---

## 3. The Forward Model ($G$)
To generate the predicted pixel values $\hat{I}_e$ for exposure $e$ (which operates in band $b$), the model executes a highly parallelizable operations graph:

**Step 1: Coordinate Projection**
Project the shared, latent universal coordinates of source $k$ into the pixel space of the specific exposure:
$$(u_{k,e}, v_{k,e}) = \mathcal{M}_e(\alpha_k, \delta_k \mid \mathbf{W}_e)$$

**Step 2: Flux Attenuation**
Scale the intrinsic flux for the specific band $b$ through the implicit throughput field to get the instrumental flux:
$$f_{\text{inst}, k, e} = F_{k, b} \cdot \mathcal{T}_{\text{eff}, e}(u_{k,e}, v_{k,e})$$

**Step 3: Sparse Rendering & Assembly**
Evaluate the Neural ePSF only within a localized bounding box around $(u_{k,e}, v_{k,e})$. Sum all rendered stamps and add the effective background field:
$$\hat{I}_e(x, y) = \mathcal{B}_{\text{eff}, e}(x, y) + \sum_{k=1}^K f_{\text{inst}, k, e} \cdot \psi_{\text{eff}, e} \Big( x - u_{k,e}, y - v_{k,e} \mid u_{k,e}, v_{k,e}, f_{\text{inst}, k, e}, \mathcal{B}_{\text{eff}, e} \Big)$$

---

## 4. The Global Objective Function ($\mathcal{L}$)
Optimization is driven by a single loss function evaluated via Automatic Differentiation.

### 4.1 The Pixel Likelihood (Data Loss)
This term evaluates the reconstruction error against the raw observed pixels $I_e$, weighted by the raw pixel variance $V_e$ (read noise + Poisson noise). Crucially, this sum is computed across *all* exposures and *all* bands simultaneously.
$$\mathcal{L}_{\text{data}} = \sum_{e=1}^E \sum_{x, y} \frac{\Big( I_e(x, y) - \hat{I}_e(x, y) \Big)^2}{2 V_e(x, y)}$$
*Note on Cross-matching:* Because a source $k$ has only one set of coordinates $(\alpha_k, \delta_k)$, overlapping pixels from different epochs or bands inherently constrain the same latent variable. The pixels mathematically "cross-match" themselves via the forward projection.

### 4.2 The Astrometric Anchor Prior
To prevent the latent coordinates from drifting arbitrarily due to geometric degeneracies between the WCS polynomials and the coordinates, we anchor high-SNR sources $j$ to an absolute reference frame (e.g., Gaia DR3):
$$\mathcal{L}_{\text{astrom}} = \sum_{j \in \text{Gaia}} \left( \frac{(\alpha_j - \alpha_{j, \text{Gaia}})^2}{2\sigma_{\alpha}^2} + \frac{(\delta_j - \delta_{j, \text{Gaia}})^2}{2\sigma_{\delta}^2} \right)$$
A similar photometric anchor prior ($\mathcal{L}_{\text{photom}}$) can be applied to standard stars to lock the absolute global zero-points.

### 4.3 Total Optimization
The system updates all parameters simultaneously to find the minimum:
$$\Theta^* = \arg\min_{\mathcal{U}, \Phi} \left[ \mathcal{L}_{\text{data}}(\mathcal{U}, \Phi) + \lambda \mathcal{L}_{\text{astrom}}(\mathcal{U}) \right]$$

---

## 5. Computational Feasibility & Implementation
Building this framework requires modern machine learning infrastructure, specifically frameworks designed for vectorization and automatic differentiation (e.g., **JAX**).

1.  **Memory Batching (Tracts):** The sky must be chunked into overlapping geometric tracts (e.g., 1 square degree). Only raw exposure cutouts intersecting that tract, along with the relevant subset of $\mathcal{U}$, are loaded into GPU memory.
2.  **Sparse Scatter-Add:** The summation in Step 3 must be implemented using sparse scatter-add tensor operations. Computing the ePSF over blank sky pixels is a fatal waste of compute; rendering must be strictly confined to small bounding boxes around $u_{k,e}, v_{k,e}$.
3.  **Optimization Engine:** First-order solvers (SGD) will fail due to the vastly different scales of the parameters (tiny radians for RA/Dec vs. massive flux values). Optimization requires algorithms like **AdamW** with carefully tuned learning rate schedules, or second-order approximations like **L-BFGS**.
4.  **Rigorous Uncertainties:** Upon convergence, the ultimate uncertainties ($\sigma_{\alpha}, \sigma_{\delta}, \sigma_{\text{flux}}$) are extracted by computing the inverse of the Hessian matrix $H$ at the minimum: $\Sigma_{\Theta} = H^{-1}$. This provides mathematically perfect error bars that fully account for blending, PSF uncertainty, and WCS covariance.


