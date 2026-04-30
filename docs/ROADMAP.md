# ROADMAP.md

This document outlines the vision, design thesis, and implementation roadmap for `torchsky`, focusing on end-to-end differentiable forward models and survey-scale inference using Neural Fields (INR).

For implementation status, test-gate results, and current replacement gaps, see [Code-Grounded Roadmap Status](roadmap_status.md). That file is the primary place to update after roadmap implementation slices.

---

## 1. Vision: Implicit Forward Modeling

### The Core Thesis
Classical mapping pipelines stage WCS, photometry, and coadds in isolation, which tends to lose coupled uncertainties and distort noise models. `torchsky` treats survey inference as **one inverse problem**: finding the latent scene and instrument law such that predicted data agree with observations.

The generative map follows the relation:
$$Y \sim \mathbb{P}_{\psi}\big(\,\cdot \mid F_{\theta},\, \zeta,\, c(t)\,\big)$$

- **$F_{\theta}$:** Continuous latent specific intensity (the sky).
- **$\zeta$:** Kinematic latents (redshift, proper motion, acceleration).
- **$c(t)$:** Observational context (pointing, platform ephemeris).
- **$\mathbb{P}_{\psi}$:** Instrument-environment-noise model (convolutions, gains, detector defects).

### The "Sky Away From Detectors" Goal
The ultimate target is a **single continuous scene** that explains multi-epoch, multi-band data without baking detector defects into the sky representation.

---

## 2. Implementation Roadmap

### Phase 1: Static Sky Fields (Current Core)
- **$F_{\theta}$:** Field-like MLPs/encoders over sky + wavelength on tangent patches or spheres.
- **$\mathcal{A}_{\psi}$:** Bandpass quadrature and PSF convolution via `render_field_observation`.
- **Status:** Implemented in `torchsky.map.field`.

### Phase 2: Pretrained Trunks (STRAINER)
- **Main Decision:** Implement transferable encoder pretraining (inspired by the *Surprising Effectiveness of Noise Pretraining* paper).
- **Mechanism:** Pretrain a shared "trunk" (e.g., `StrainerSkyEncoderMLPField`) on synthetic noise and astronomy-shaped fields, then freeze or fine-tune it for specific survey patches.
- **Objective:** Improve supported-pixel fidelity especially in sparse, heterogeneous observation regimes.
- **Status:** Phase 1 (API and format) and Phase 2 (multi-noise pretraining) are implemented in tree.

### Phase 3: Kinematics and Dynamic Skies
- **Rest-frame Modeling:** Define $F_{\theta}$ in rest-frame $\lambda_{emit}$, with explicit Doppler/redshift shifts $z \in \zeta$.
- **Transverse Motion:** Support proper motion $\boldsymbol{\mu}$, parallax $\varpi$, and sky paths $\omega(t)$ for high-precision astrometry.
- **Peculiar Velocities:** Support motion relative to a reference field (v-ref + v-pec) without assuming a specific cosmology.

### Phase 4: Hybrid Scenes and Multi-Survey Fusion
- **Composite Models:** Shared latent fields for diffuse structure combined with discrete source models for bright objects.
- **Multi-Survey Likelihood:** $\mathcal{L} = \sum_s \mathcal{L}_s$. Link Euclid, LSST, eROSITA, and CMB data through shared sky patches while using survey-specific instrument modules $\psi^{(s)}$.

---

## 3. Beyond Maps: Strategic Directions

- **Generative Physics (Improved MeanFlow):** Moving from "fitting a map" to "generative modeling of the latent physics." This allows for rigorous non-Gaussian cosmology constraints.
- **Any-to-Any Modality:** Cross-wavelength field representations as central building blocks, not side products.
- **Uncertainty Quantification:** Taking region-conditional property seriously, incorporating distributional conformal prediction (VSPS) for rigorous constraints.

---

## 4. Feasibility and Scalability

Large-scale survey fusion is not a monolithic single-job problem, but a **systems and engineering challenge**:
1. **Modular Likelihoods:** Surveys share only what the data actually constrain.
2. **Tracts and Tiling:** Processing the sky in overlapping patches and healpix tiles to cap peak memory.
3. **Multi-resolution Branches:** Separate branches for high-res imaging (arcsec) and low-res/harmonic CMB (degree-scale).
4. **Curriculum Fusion:** Starting with Gaia/LSST anchors and progressively adding spectroscopy and high-systematics data.
