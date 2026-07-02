# Architecture

`torchsky` is meant to be the backbone of a differentiable, any-device,
torch-native stack for modelling observations of the sky. The target is not a
thin wrapper around legacy astronomy packages. The core should make it practical
to model foregrounds, instruments, backgrounds, explicit physical components,
learned implicit components, and hybrid scenes jointly across wavelength,
instrument type, and data modality.

The guiding model is:

```text
observations ~ instrument/noise law(scene, kinematics, context)
```

The architecture should help users extract maximal information from all relevant
data jointly: learned models when enough data can beat assumption-heavy physical
models, physical models where their inductive structure remains unbeatable, and
hybrid models where that is the honest representation of the problem.

## Public Pillars

Public surfaces are organized around three architectural pillars. The current
code still has compatibility namespaces and historical layout, but new public
work should move toward these pillars while keeping the advertised public API
small and namespace-oriented.

| Pillar | Purpose | Current surfaces |
|---|---|---|
| `world` | Coordinates, WCS, frames, spherical geometry, HEALPix, MOC, sparse maps, and harmonics | `torchsky.wcs.WCS`, `torchsky.sphere` |
| `scene` | Observations, operators, instruments, latent scenes, products, coadds, objectives, renderers, and fits | promoted `torchsky.map` paths, `torchsky.scene` facade, and field/INR modelling while it remains experimental |
| `viz` | Browser and notebook-facing visualization | `torchsky.viz` |

Use `torchsky.surfaces` to inspect this contract from code.

FITS I/O is an external boundary owned by `torchfits`. File-oriented
applications load tensors, header mappings, masks/weights, or streamed table
batches there and pass those values into torchsky. Core computation must not
open files or depend on CFITSIO.

## Refactor Rules

- Preserve the lean public contract first; keep implementation helpers in their
  owning namespaces.
- Keep core runtime dependencies lean: PyTorch, NumPy, and benchmark-justified
  geometry/WCS extensions.
- Do not duplicate FITS/HDU/table/cache implementations or link CFITSIO; those
  belong to `torchfits`.
- Use Astropy, fitsio, healpy, hpgeom, healsparse, reproject, drizzle, and
  similar packages as references, fixtures, comparators, or optional
  compatibility targets, not as hidden core backends.
- Prefer torch-native tensor paths that preserve differentiability and device
  placement.
- Prefer survey-scale designs: chunking, streaming, tiling, sparse
  representations, explicit operators, and benchmarked kernels.
- Prioritize exact operator semantics, asymptotic improvements, bounded memory,
  and end-to-end throughput before convenience APIs or micro-optimizations.
- Make user-facing APIs and CLIs clean, low-jargon, and easy to adopt.
- Avoid cosmetic churn unless it materially improves architecture, usability,
  scalability, differentiability, or correctness.
