Beyond map operations, the next big fronts are:

**1. Latent-scene decomposition science**
Not just better remappers, but actual shared-vs-private sky/instrument decomposition across surveys. We started that on the spherical side, but the real next step is lifting it into richer latent models:
- INR / field-based shared-private decomposition
- learned nuisance fields, not only learned mixture weights
- identifiability controls and leakage benchmarks

There is now a first exact identifiability control on both the harmonic and field sides: component-scale gauge canonicalization can rescale learned component amplitudes and their matching mixer weights without changing predictions. On the field side this is wired through explicit per-component gains inside `FieldComponentDecomposition`, so the decomposition wrapper can be gauge-fixed exactly even for train-time field workflows. There is also now a first native diagnostics surface for decomposition quality on both the harmonic and field sides: shared-to-private leakage, private-to-shared leakage, and optional calibration-transfer stability against an anchor subspace, with field diagnostics evaluated in background-stripped rendered observation space. A first explicit `FieldComponentDecomposition` wrapper and train-time field decomposition workflow now exist via native builder/fit helpers, and there is now a first explicit solve-time anchor observation family (`FieldComponentAnchorTerm`) for standards / calibration-transfer supervision of selected decomposition roles, including rendered image anchors, direct coordinate/spectral sample anchors, and pixel-frame astrometric sample anchors resolved through each observation WCS. Richer anchor surfaces, real-data diagnostics, and stronger INR decomposition programs are still open.

**2. PSF / transfer / detector modeling**
This is probably the biggest science gap after geometry.
- spatially varying PSF / ePSF / beam models
- cross-instrument transfer operators
- detector residual operators: nonlinearity, persistence, brighter-fatter, CTE-like effects
- learned residual operator surfaces where explicit models are too weak

**3. Multi-survey / multi-modal inference**
The architecture now points toward joint inference across surveys, but the actual science layer is still early.
- SPHEREx + WISE + GALEX + CMB-style shared latent models
- mixed spectral laws and component separation
- survey-private nuisance structure
- real-data benchmark surfaces, not only synthetic ones

**4. WCS / spherical / spectral core depth**
Map operations touched a lot of this, but the lower-level geometry stack is still a major standalone frontier.
- more exact / faster WCS kernels
- more complete spherical transforms and projections
- higher-performance native kernels in `cpp_src`
- stronger comparator coverage and benchmark leadership

**5. Source/catalog layer**
Eventually you need the bridge from continuous fields and maps to catalogs and measurements.
- differentiable source extraction / detection
- joint field + source models
- photometry / morphology / cross-band measurement on top of latent fields
- scalable catalog inference surfaces

**6. I/O / data systems / scale**
If this is meant to work at survey scale, systems work matters as much as inference.
- chunked / distributed map and cube workflows
- better table-image-map interoperability
- stronger memory-aware planning
- real pipeline ergonomics for large datasets

**7. Real benchmark and product surfaces**
A lot of the internal machinery is getting strong, but the strongest remaining need is proving it on realistic science cases.
- real-data multi-survey decomposition benchmarks
- PSF-aware heterogeneous stacks
- calibration transfer tests
- end-to-end product evaluations users can trust

If you want the shortest honest answer: after map operations, the next highest-value areas are `PSF/instrument modeling`, `shared-vs-private latent decomposition`, and `multi-survey real-data inference`.

If you want, I can turn that into a concrete repo roadmap with:
- `near-term`
- `mid-term`
- `research-grade`
