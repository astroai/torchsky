# Torchsky Architecture & API Review

## Executive Summary

This document reviews the `torchsky` source code architecture against the 2026 goals for a differentiable, any-device, torch-native foundational library. The analysis reveals that the library is currently hindered by excessive namespace aliasing, "God module" design patterns, fragmented dispatch functions, and an API heavy with domain jargon. A clean, modular separation is required to reach the target "clean, lean, easy to adopt" user experience.

## 1. Directory Structure Creep & Redundancies

The repository structure has evolved into a heavily nested set of namespaces that overlap significantly in concept, leading to cognitive overload:

- **Shallow Alias Layers:** Modules like `src/torchsky/coords/` and `src/torchsky/geom/` are almost entirely empty shell namespaces. Their `__init__.py` files exclusively import and re-export implementations from `wcs/` and `sphere/` (e.g., `coords` re-exports `WCS`, `healpix`, and `wcs.frames`; `geom` re-exports `MOC` and `SphericalPolygon` from `sphere`). This forces the user to guess whether an object lives in the domain module (`wcs`/`sphere`) or the alias module (`coords`/`geom`).
- **Reverse Aliasing in I/O:** `src/torchsky/io/__init__.py` imports its implementations *backwards* from the root `__init__.py` (`from .. import read`, etc.), creating a circular dependency loop and confusing the source of truth for the I/O engine.

## 2. API Intuitiveness & The "God Module"

### The Root `__init__.py` Bottleneck

The root `src/torchsky/__init__.py` is currently over 3,100 lines long. Rather than acting as a clean, declarative API router, it implements the entire FITS core handling inline (caching, header parsing, multiple dispatch logic). This results in slow import times and partial failure of its own lazy-loading mechanisms.

### Fragmented Dispatch

Instead of a single, intuitive, polymorphic entry point, the API is bloated with fragmented reading utilities exposed globally. `io/__init__.py` reveals at least seven variants:

- `read()`
- `read_fast()`
- `read_image()`
- `read_table()`
- `read_hdus()`
- `read_subset()`
- `read_large_table()`

### Jargon-Heavy and Verbose Naming

The API leaks complex project internals and raw mathematical terminology directly to the user:

- **Implicit Models (INRs):** Internal project names are leaking into public classes, such as `StrainerSkyEncoderMLPField`, and operations like `render_field_observation_distribution`. (The term "Strainer" appears tied to an internal architecture rather than an intuitive user concept).
- **Low-Level Mapping:** Function names are overly descriptive to the point of being intimidating, such as `accumulate_image_on_healpix_map_adjoint` or `resample_linear_affine_periodic_bilinear_2d`.

## 3. Non-Efficient Architectural Parts

- **Monolithic Files:** Core engine files are growing unmanageably large. `src/torchsky/map/image_plane.py` is ~114KB, `src/torchsky/sphere/spectral.py` is ~144KB, and `src/torchsky/table.py` is ~121KB. While they may be functionally efficient for `torch.compile`, they represent significant technical debt for maintainability.
- **Coupled Domains:** The library currently couples low-level, high-performance FITS/Arrow dataset loading tightly with complex, differentiable implicit physics models (`map/field/`). A foundational tensor library should keep these cleanly decoupled.

---

## Proposed Restructuring Plan

To align with the ambitious goals of the 2026 Cosmapper architecture, the following restructuring plan is proposed:

### Phase 1: Dismantle the God Module and Fix I/O

1. **Extract Root Logic:** Move the 3,100 lines of FITS parsing and dispatch logic from `src/torchsky/__init__.py` into a dedicated backend module, such as `src/torchsky/io/fits.py` and `src/torchsky/core/`.
2. **Unify Dispatch:** Deprecate the fragmented `read_`* functions in favor of a single `torchsky.read()` function that infers behavior from the file (returning an image tensor, a table frame, or a lazy dataset), with optional `mode=` kwargs.

### Phase 2: Collapse the Namespace Redundancies

1. **Deprecate Aliases:** Remove the `coords` and `geom` directories.
2. **Consolidate Domains:** Establish `torchsky.wcs` as the single source of truth for all coordinate and affine transformations, and `torchsky.sphere` as the single source of truth for all spherical pixelization (HEALPix, MOC) and harmonic representations.

### Phase 3: De-Jargon the Public API

1. **Functional Parity:** Rename long, verbose mapping functions to match standard PyTorch paradigms. E.g., replace `resample_linear_affine_periodic_bilinear_2d` with a cleaner `torchsky.functional.resample(..., mode='bilinear')`.
2. **Abstract Internal Models:** Hide "Strainer" and heavily specific "MLPField" classes behind a standard abstraction layer, e.g., `torchsky.models.ImplicitSkyModel`.

### Phase 4: Modularize the Monoliths

1. **Split Large Modules:** Break down `table.py`, `sphere/spectral.py`, and `map/image_plane.py` into internal packages (e.g., `src/torchsky/sphere/spectral/sht.py`, `src/torchsky/sphere/spectral/transforms.py`) while keeping the public import path unchanged via clean `__init__.py` routers.