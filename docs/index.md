---
template: home.html
title: Astro compute for PyTorch
---

<div class="ts-below" markdown>

**torchsky** is the torch-native compute layer for sky data: coordinates and WCS
transforms, spherical geometry and HEALPix, mapmaking, observation operators,
scene composition, and differentiable sky models. It runs on CPU, CUDA, and
Apple Silicon as ordinary PyTorch tensor code.

File I/O belongs to [`torchfits`](https://github.com/astroai/torchfits);
scientific evaluation and comparisons belong to
[`torchsky-harness`](https://github.com/astroai/torchsky-harness). `torchsky` is
the science layer in between: it takes tensors and metadata and returns
tensors, on whatever device you ask for.

---

## Core capabilities

<div class="grid cards" markdown>

-   :material-compass-outline:{ .lg .middle } **Coordinates and WCS**

    ---

    Tensor-native FITS WCS: analytic and distorted projections, SIP/TPV,
    spectral and time axes, frames, alignment, and compiled small-N paths.

    [:octicons-arrow-right-24: WCS guide](wcs.md)

-   :material-globe-model:{ .lg .middle } **Sphere and HEALPix**

    ---

    HEALPix indexing, neighbours and interpolation, MOC, sparse maps, vector
    geometry, and harmonic transforms.

    [:octicons-arrow-right-24: Sphere guide](sphere.md)

-   :material-image-filter-center-focus:{ .lg .middle } **Mapmaking**

    ---

    Plan target grids, project exposures, coadd images, spectral cubes, and
    HEALPix maps with explicit value semantics and memory bounds.

    [:octicons-arrow-right-24: Map guide](map.md)

-   :material-math-integral:{ .lg .middle } **Fields and scenes**

    ---

    Composable observation operators, differentiable field models, rendering,
    and calibration-set fitting behind an experimental facade.

    [:octicons-arrow-right-24: Field guide](field.md)

</div>

---

## At a glance

=== "Coordinates and WCS"

    ```python
    import torch
    from torchsky.wcs import WCS

    wcs = WCS(header)               # header is a torchfits mapping
    ra, dec = wcs.pixel_to_world(x, y)
    x_back, y_back = wcs.world_to_pixel(ra, dec)
    ```

    Device moves, dtype, and differentiability follow the tensors you pass in.
    Compiled small-N paths keep per-object inference cheap.

=== "HEALPix"

    ```python
    import torch
    import torchsky.sphere as sphere

    pix = sphere.ang2pix(1024, lon_deg, lat_deg, lonlat=True)
    neighbours = sphere.neighbors(1024, pix, nest=False)
    ```

    Same kernels on CPU or GPU; no NumPy round trips inside a loop.

=== "Coadd images"

    ```python
    from torchsky.map import Observation, coadd, plan

    observations = [Observation("exposure.fits")]
    target = plan(observations, projection="TAN")
    result = coadd(observations, target)
    ```

    `result.data`, `result.weight`, and `result.header` are tensors and metadata;
    write them with `torchfits`.

=== "Spectral cubes"

    ```python
    from torchsky.map import ResampleConfig, coadd_cube

    cube = coadd_cube(
        observations,
        target,
        wavelengths=wavelengths,
        resample=ResampleConfig(spectral_kernel="drizzle"),
    )
    ```

    Coadd 2D filter-band images into a cube, or assemble 3D IFU detector frames
    with per-pixel wavelengths.

---

## Why torchsky

<div class="grid cards" markdown>

-   **One tensor API**

    ---

    Coordinates, geometry, maps, and models accept and return PyTorch tensors.
    No hidden NumPy or Astropy round trips inside hot loops.

-   **Explicit semantics**

    ---

    Flux versus surface brightness, weights and inverse variance, kernels, and
    spectral units are declared and validated, not inferred.

-   **Memory-bounded by design**

    ---

    Tiling, chunking, streaming, sparse HEALPix, and exposure-order accumulation
    keep survey-scale products inside a fixed budget.

-   **Measured, not asserted**

    ---

    Correctness and performance claims are evaluated in
    [`torchsky-harness`](https://github.com/astroai/torchsky-harness) against
    analytic truth and reference implementations.

</div>

---

## Start here

<div class="grid cards" markdown>

-   **New users**

    ---

    [Quick start](quickstart.md) · [Install](install.md) ·
    [Getting started](getting-started.md) · [Examples](examples.md)

-   **Domain guides**

    ---

    [WCS](wcs.md) · [Sphere](sphere.md) · [Map](map.md) ·
    [Catalogs](catalogs.md) · [Spectral](spectral.md) · [Scene](scene.md) ·
    [Field](field.md)

-   **API and architecture**

    ---

    [API reference](reference/index.md) · [Architecture](architecture.md) ·
    [Compatibility](compatibility.md) ·
    [PSF conventions](psf-response-conventions.md)

-   **Project**

    ---

    [Changelog](changelog.md) · [Roadmap](roadmap.md) ·
    [Release process](release.md) · [Contributing](contributing.md)

</div>

---

`torchsky` is under active development toward its 0.4.0 release. It is not yet
published on PyPI; installation instructions cover the available source and wheel
builds.

</div>
