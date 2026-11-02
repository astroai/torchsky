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

    [:octicons-arrow-right-24: WCS guide](coordinates/index.md)

-   :material-globe-model:{ .lg .middle } **Sphere and HEALPix**

    ---

    HEALPix indexing, neighbours and interpolation, MOC, sparse maps, vector
    geometry, and harmonic transforms.

    [:octicons-arrow-right-24: Sphere guide](sphere/index.md)

-   :material-image-filter-center-focus:{ .lg .middle } **Mapmaking**

    ---

    Plan target grids, project exposures, coadd images, spectral cubes, and
    HEALPix maps with explicit value semantics and memory bounds.

    [:octicons-arrow-right-24: Map guide](maps/index.md)

-   :material-math-integral:{ .lg .middle } **Fields and scenes**

    ---

    Composable observation operators, differentiable field models, rendering,
    and calibration-set fitting behind an experimental facade.

    [:octicons-arrow-right-24: Field guide](models/fields.md)

</div>

---

## At a glance

=== "Coordinates and WCS"

    <!-- docs: run -->
    ```python
    import torch
    from torchsky.wcs import WCS

    header = {
        "NAXIS": 2, "NAXIS1": 256, "NAXIS2": 256,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRPIX1": 128.5, "CRPIX2": 128.5, "CRVAL1": 150.0, "CRVAL2": 2.0,
        "CDELT1": -2.8e-4, "CDELT2": 2.8e-4,
    }
    x = torch.tensor([10.0, 200.0], dtype=torch.float64)
    y = torch.tensor([20.0, 240.0], dtype=torch.float64)

    wcs = WCS(header)
    ra, dec = wcs.pixel_to_world(x, y)
    x_back, y_back = wcs.world_to_pixel(ra, dec)
    torch.testing.assert_close(x_back, x, atol=1e-6, rtol=0.0)
    ```

    Device moves, dtype, and differentiability follow the tensors you pass in.

=== "HEALPix"

    <!-- docs: run -->
    ```python
    import torch
    import torchsky.sphere as sphere

    lon_deg = torch.tensor([10.0, 120.0], dtype=torch.float64)
    lat_deg = torch.tensor([5.0, -40.0], dtype=torch.float64)

    pix = sphere.ang2pix(1024, lon_deg, lat_deg, lonlat=True, nest=True)
    neighbours = sphere.neighbors(1024, pix, nest=True)
    assert neighbours.shape == (2, 8)
    ```

    Same kernels on CPU or GPU; no NumPy round trips inside a loop.

=== "Coadd images"

    <!-- docs: run -->
    ```python
    import torch
    from torchsky.map import Observation, coadd, plan

    header = {
        "NAXIS": 2, "NAXIS1": 64, "NAXIS2": 64,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRPIX1": 32.5, "CRPIX2": 32.5, "CRVAL1": 150.0, "CRVAL2": 2.0,
        "CDELT1": -1e-3, "CDELT2": 1e-3,
    }
    observations = [
        Observation(data=torch.randn(64, 64), wcs={**header, "CRVAL1": c})
        for c in (150.0, 150.01)
    ]
    target = plan(observations, projection="TAN")
    result = coadd(observations, target, kernel="bilinear")
    assert result.data.shape == target.shape
    ```

    `result.data`, `result.weight`, and `result.header` are tensors and metadata;
    write them with `torchfits`.

=== "Spectral cubes"

    <!-- docs: run -->
    ```python
    import torch
    from torchsky.map import Observation, ResampleConfig, coadd_cube

    header = {
        "NAXIS": 2, "NAXIS1": 16, "NAXIS2": 16,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRPIX1": 8.5, "CRPIX2": 8.5, "CRVAL1": 150.0, "CRVAL2": 2.0,
        "CDELT1": -1e-3, "CDELT2": 1e-3,
    }
    wavelengths = [5000.0, 6000.0, 7000.0]
    bands = [
        Observation(data=torch.full((16, 16), 1.0 + i), wcs=header, wavelength=w)
        for i, w in enumerate(wavelengths)
    ]
    cube = coadd_cube(
        bands,
        wavelengths=torch.tensor(wavelengths),
        resample=ResampleConfig(spectral_kernel="linear"),
    )
    assert cube.data.ndim == 3
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

    [Install](introduction/install.md) · [Quick start](introduction/quickstart.md) ·
    [Concepts](introduction/concepts.md) · [Examples](examples/index.md)

-   **User guide**

    ---

    [WCS](coordinates/index.md) · [Sphere](sphere/index.md) · [Catalogs](catalogs/index.md) ·
    [Coadds](maps/index.md) · [Spectra](spectra/index.md) · [Fields](models/fields.md)

-   **Performance**

    ---

    [Highlights](performance/highlights.md) ·
    [Scaling and tuning](performance/scaling.md) ·
    [Parity evidence](performance/parity.md)

-   **Project**

    ---

    [API reference](reference/index.md) · [Changelog](project/changelog.md) ·
    [Roadmap](project/roadmap.md) · [Contributing](project/contributing.md)

</div>

---

torchsky 0.5.0rc1 is a GitHub pre-release (not on PyPI until 0.5.0);
[Install](introduction/install.md) covers source builds. Coming from Astropy, healpy, or
reproject? See [the porting guide](introduction/coming-from.md).

</div>
