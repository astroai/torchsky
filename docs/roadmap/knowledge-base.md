# Research Knowledge Base

> Roadmap note: this is research context, not a public API or workflow contract.
> For current shipped surfaces, start with [api.md](../api.md).

This document provides a consolidated survey of research and paper summaries that inform the design of implicit neural representations (INR), neural fields, and latent sky modeling in `torchsky`.

---

## Implicit Neural Representations (INR) Survey

### 1. The State of the Art (as of early 2026)

The short answer is that **hash-based feature grids** have decisively won the representation war. Pure activation-based MLPs (like the original NeRF) suffer from spectral bias—meaning they naturally over-smooth high-frequency details—and are far too computationally expensive for dense reconstruction.

The meta has evolved past first-generation hash grids (like Instant-NGP or Neuralangelo). The primary bottleneck with standard hash grids was *hash collisions*, where different parts of a 3D space share the same memory, causing artifacting in fine geometric details.

#### Classification of SOTA Models:
- **Surface Accuracy:** **AniSDF** (ICLR 2025). Uses fused-granularity neural surfaces alongside anisotropic encoding to solve thin-structure artifacts.
- **Lighting & Materials:** **5D Hash-Grid-Sphere** (March 2026). Extends 3D hash grids into a 5D spatio-directional domain using a hierarchical geodesic grid.
- **Dynamic Scenes:** **MoAngelo** (3DV 2026). High-fidelity reconstruction for deforming objects.
- **Design Pattern:** **Explicit feature field + tiny decoder + refinement**. Decouple representation from refinement (e.g., hash-grid/tri-plane followed by a refiner).
- **Static Fidelity:** **Zip-NeRF**. Combines mip-NeRF 360 anti-aliasing with grid-based methods like Instant-NGP. Reported lower error rates while training significantly faster.

---

## Literature Review & Paper Summaries

### Core Representations
- **Zip-NeRF:** Anti-aliased multiscale representation. Essential for ensuring spatial/spectral detail does not exceed what the observation operator can resolve. ([2304.06706](https://arxiv.org/abs/2304.06706))
- **WIRE:** Implicit Neural Representations with Wavelet-inspired architectures for better high-frequency capture. ([2301.05187](https://arxiv.org/abs/2301.05187))
- **SIREN:** Periodic activation functions (sine) for better derivative capture in coordinate MLPs.
- **TensoRF:** Low-rank 4D tensor factorization for compact, fast-training radiance fields. ([ecva.net](https://www.ecva.net/papers/eccv_2022/papers_ECCV_2022/html/3431_ECCV_2022_paper.php))
- **K-Planes:** Space + Time + Appearance factorization for non-static scenes. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2023/html/Fridovich-Keil_K-Planes_Explicit_2D_Plane_Factorizations_for_T-Spatio-Temporal_Neural_Fields_CVPR_2023_paper.html))

### Inverse Problems & Calibration
- **Sampling Theory for Super-Resolution (INR):** Reconstruction quality depends on whether the sampling operator constraints the represented degrees of freedom. ([2506.09949](https://arxiv.org/abs/2506.09949))
- **Self-Calibrating INRs:** Important for learning nuisance parameters (PSF variations, astrometry shifts) alongside the latent sky model. ([2108.13826](https://arxiv.org/abs/2108.13826))
- **Spotting Hallucinations:** Diagnostics for when learned inverse methods invent structure unsupported by data. ([2306.13272](https://arxiv.org/abs/2306.13272))

---

## Implementation Alignment in `torchsky`

The shared `_HashGridEncoder` in `torchsky.map.field` implements several of these research findings:

1. **Explicit Multiscale Trunk:** Logarithmically spaced resolutions with per-level feature vectors (inspired by Zip-NeRF/WIRE).
2. **Collision Mitigation:** Uses **dense** voxel tables for coarse scales where `resolution ** dim <= 4096`, preventing corrupting large-scale structure with hash collisions.
3. **Smoother Footprints:** Optional `smoothstep` interpolation for low-pass filtering.
4. **Decoupled Refinement:** `HashGridResMLPField` stacks the encoder with a residual MLP decoder rather than a plain feedforward head.

### Scalability Considerations
- **Query Cost:** Billing-pixel queries are expensive. `render_field_cube` uses **chunking** and windowing to cap peak activation memory.
- **Capacity:** Pure MLPs have fixed parameters; hash grids scale memory with resolution. **Tiling** or multi-field mosaics are the practical path for all-sky coverage at arcsec resolution.
- **Native Execution:** Large-map paths utilize HEALPix évaluation (`render_field_healpix`) or `torch.func` vmapping for JAX-style point batching.

---

## Recommendations for `torchsky`
1. Constrain spectral degrees of freedom by the actual transmission operator.
2. Improve support-region fidelity without sacrificing full-grid coherence.
3. Treat large sparse mosaics as a systems problem (streaming, chunking).
4. Use hybrid models: shared latent structure + small residual/nuisance components.
5. Prioritize explicit multiscale feature fields over pure activation-only MLPs.
