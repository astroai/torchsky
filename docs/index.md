---
template: home.html
title: Sky computing for PyTorch
---

<div class="ts-below" markdown>

!!! danger "These docs have not yet been verified by a human"
    The guides, examples and reference text for this first release were
    drafted with AI assistance. Automated tests run every code snippet and
    example, but no person has yet reviewed the prose, the numbers or the
    recommendations. Treat them as a draft, check results that matter, and
    [report anything wrong](https://github.com/astroai/torchsky/issues).

**torchsky** does astronomy's sky-domain computations as PyTorch tensor code:
coordinates and WCS, HEALPix and spherical geometry, catalog crossmatches,
image coadds and HEALPix maps, spectra and synthetic photometry, and
differentiable sky models. Inputs are tensors and header dicts, outputs are
tensors on the same device (CPU, CUDA, or Apple MPS), and the autograd graph
stays intact. The kernels batch, index, and stream, so the call that handles a
hundred rows also handles a survey.

[`torchfits`](https://github.com/astroai/torchfits) reads and writes the FITS
files; torchsky starts from the tensors it returns.

---

## Real Data

Each example fetches a small public sample once, runs in seconds on a CPU, and
prints quantitative checks of its result. Select a figure to open the example.

<div class="grid cards ts-gallery" markdown>

-   [![Gaia, 2MASS and AllWISE sources in M67 and their match separations](assets/gallery/examples/crossmatch_gaia_2mass_allwise.png)](examples/index.md#gaia-dr3-x-2mass-x-allwise-in-the-field-of-m67)

    **[Gaia × 2MASS × AllWISE](examples/index.md#gaia-dr3-x-2mass-x-allwise-in-the-field-of-m67)**
    Best, one-to-one, error-radius, and group matches of three catalogs in M67.

-   [![Planck Commander synchrotron map, Galactic mask and angular power spectra](assets/gallery/examples/planck_power_spectrum.png)](examples/index.md#apodised-mask-power-spectrum-of-the-commander-synchrotron-map)

    **[Planck power spectrum](examples/index.md#apodised-mask-power-spectrum-of-the-commander-synchrotron-map)**
    An apodised Galactic mask and the angular power spectrum of a HEALPix map.

-   [![SDSS quasar spectrum in the rest frame with the ugriz passbands](assets/gallery/examples/sdss_spectrum_photometry.png)](examples/index.md#a-quasar-spectrum-rest-frame-dust-and-ugriz-magnitudes)

    **[SDSS spectrum photometry](examples/index.md#a-quasar-spectrum-rest-frame-dust-and-ugriz-magnitudes)**
    Rest frame, dust correction, and ugriz magnitudes that match the SDSS pipeline.

-   [![MaNGA H-alpha flux and velocity maps of a galaxy](assets/gallery/examples/manga_ifu_cube.png)](examples/index.md#a-manga-ifu-cube-h-map-rotation-and-reprojection)

    **[MaNGA IFU cube](examples/index.md#a-manga-ifu-cube-h-map-rotation-and-reprojection)**
    Hα flux and velocity maps, and a flux-conserving reprojection of the cube.

-   [![Horsehead Nebula plate with its WCS grid and footprint](assets/gallery/examples/horsehead_wcs_tour.png)](examples/index.md#horsehead-nebula-wcs-footprint-and-reprojection)

    **[Horsehead WCS tour](examples/index.md#horsehead-nebula-wcs-footprint-and-reprojection)**
    A photographic plate: pixel to sky, footprint, MOC, and reprojection.

-   [![Chandra events near Sgr A* as a sparse HEALPix map with its MOC](assets/gallery/examples/chandra_events_on_sphere.png)](examples/index.md#sgr-a-events-as-a-sparse-healpix-map-and-moc)

    **[Chandra events on the sphere](examples/index.md#sgr-a-events-as-a-sparse-healpix-map-and-moc)**
    X-ray events near Sgr A* as a sparse HEALPix counts map and a MOC of the chips.

-   [![Four MegaPipe cutouts and their Lanczos3 and drizzle mosaic](assets/gallery/examples/megapipe_mosaic.png)](examples/index.md#four-megapipe-cutouts-coadded-with-lanczos3-and-drizzle)

    **[CFHT MegaPipe mosaic](examples/index.md#four-megapipe-cutouts-coadded-with-lanczos3-and-drizzle)**
    Four cutouts on four TAN projections, registered and coadded.

-   [![Legacy Surveys images of the Cartwheel and the fitted continuous field](assets/gallery/examples/legacy_field_fit.png)](examples/index.md#one-continuous-field-fitted-to-griz-images-of-the-cartwheel)

    **[Cartwheel field fit](examples/index.md#one-continuous-field-fitted-to-griz-images-of-the-cartwheel)**
    One continuous sky model fitted to g, r, i, and z images at once.

-   [![Legacy Surveys image of NGC 288, rendered scene and residuals](assets/gallery/examples/legacy_forced_photometry.png)](examples/index.md#forced-photometry-of-143-gaia-stars-near-ngc-288)

    **[Gaia forced photometry](examples/index.md#forced-photometry-of-143-gaia-stars-near-ngc-288)**
    Fluxes of 143 Gaia stars in NGC 288, compared with the Legacy Surveys catalog.

</div>

---

## Areas

<div class="grid cards" markdown>

-   :material-compass-outline:{ .lg .middle } **Coordinates and WCS**

    ---

    Pixels to sky and back with FITS WCS, SIP and TPV distortions, celestial
    frames, proper motion, and observed place.

    [:octicons-arrow-right-24: Coordinates](coordinates/index.md)

-   :material-globe-model:{ .lg .middle } **Celestial sphere**

    ---

    HEALPix indexing and queries, exact spherical regions, MOC coverage,
    sparse maps, and spherical harmonic transforms.

    [:octicons-arrow-right-24: Celestial sphere](sphere/index.md)

-   :material-table-search:{ .lg .middle } **Catalogs**

    ---

    Catalogs as tensors with frame and epoch; cone and footprint selection;
    exact crossmatches from a few rows to billions.

    [:octicons-arrow-right-24: Catalogs](catalogs/index.md)

-   :material-image-filter-center-focus:{ .lg .middle } **Images and maps**

    ---

    Plan a target grid, resample with declared kernels or drizzle, and coadd
    images, spectral cubes, and HEALPix maps.

    [:octicons-arrow-right-24: Images and maps](maps/index.md)

-   :material-chart-bell-curve:{ .lg .middle } **Spectra and photometry**

    ---

    Spectral axes, redshift and Doppler shifts, dust laws, passbands,
    synthetic magnitudes, and cosmological distances.

    [:octicons-arrow-right-24: Spectra and photometry](spectra/index.md)

-   :material-math-integral:{ .lg .middle } **Models** (experimental)

    ---

    Forward models of the sky through each exposure's WCS, PSF, and bandpass,
    fitted by gradient descent with exact adjoints.

    [:octicons-arrow-right-24: Models](models/index.md)

</div>

---

## Start Here

<div class="grid cards" markdown>

-   **New users**

    ---

    [What torchsky is](introduction/index.md) · [Install](introduction/install.md) ·
    [Quick start](introduction/quickstart.md) · [Concepts](introduction/concepts.md)

-   **Examples and evidence**

    ---

    [Examples](examples/index.md) · [Performance](performance/highlights.md) ·
    [Parity](performance/parity.md)

-   **Reference**

    ---

    [API reference](reference/index.md) · [CLI](introduction/cli.md) ·
    [Coming from Astropy and healpy](introduction/coming-from.md)

-   **Project**

    ---

    [Changelog](project/changelog.md) · [Roadmap](project/roadmap.md) ·
    [Contributing](project/contributing.md)

</div>

---

torchsky 0.5.0rc1 is a GitHub pre-release and is not on PyPI until 0.5.0;
[Install](introduction/install.md) covers wheels and source builds.

</div>
