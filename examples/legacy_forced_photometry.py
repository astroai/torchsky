"""Forced photometry of Gaia stars on Legacy Surveys g/r/z images near NGC 288.

Science question: given star positions from a catalogue, how well can a
scene of point sources on a real image recover their fluxes in three bands?
A 100 arcsec Legacy Surveys DR10 g/r/z cutout (native 0.262 arcsec DECam
pixels, with inverse variance) 4 arcmin west of the globular cluster NGC 288
is fetched with the DR10 Tractor catalogue of the same field: 143 stars with
Gaia positions and Tractor fluxes. The positions are held fixed in a
``torchsky.scene.SourceCatalogScene`` with one flux per star and band, and
``torchsky.catalogs.fit_forced_photometry`` groups stars whose PSF footprints
overlap and solves each group for its fluxes and a local constant background
by weighted linear least squares on the native pixels. The PSF is a Gaussian
whose width is fitted per band: the forced fit is repeated over a grid of
widths and a parabola through the chi-square minimum gives the best one. The
fitted fluxes are compared with the Tractor fluxes, which come from fits to
the individual exposures with pixelized PSF models. The ratio is constant
star to star within about 3 percent, and its median, about 0.85, is the
aperture correction of a Gaussian core that misses the PSF wings. The figure
shows the r image with the catalogue positions, the rendered scene, the
residual, and forced against Tractor fluxes. Runtime about 20 s on CPU once
the 4 MB of samples are cached.

Data: Legacy Surveys DR10 (Dey et al. 2019, AJ 157, 168), public, with Gaia
DR3 astrometry (Gaia Collaboration 2023).

    pixi run -e test -- python examples/legacy_forced_photometry.py [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from _sample_data import fetch_sample, skip_if_unavailable

import torchfits
from torchsky.catalogs import fit_forced_photometry
from torchsky.field import FieldObservation, PSFSpec
from torchsky.scene import SourceCatalogScene
from torchsky.wcs import WCS

BANDS = ("g", "r", "z")
WIDTHS = torch.arange(1.2, 2.45, 0.1, dtype=torch.float64)  # Gaussian sigma, pixels
WCS_KEYS = (
    "CTYPE1",
    "CTYPE2",
    "CRVAL1",
    "CRVAL2",
    "CRPIX1",
    "CRPIX2",
    "CD1_1",
    "CD1_2",
    "CD2_1",
    "CD2_2",
)


def observations_for(images, invvar, header, sigmas) -> list[FieldObservation]:
    return [
        FieldObservation(
            image=images[k],
            header=header,
            weight=invvar[k],
            band_name=band,
            psf_spec=PSFSpec.gaussian(float(sigmas[k])),
        )
        for k, band in enumerate(BANDS)
    ]


def forced_fit(images, invvar, header, ra, dec, sigmas):
    observations = observations_for(images, invvar, header, sigmas)
    scene = SourceCatalogScene(
        ra,
        dec,
        coefficients=torch.zeros((ra.numel(), len(BANDS)), dtype=torch.float64),
        band_labels=BANDS,
    )
    result = fit_forced_photometry(observations, scene, background="constant")
    chi2 = torch.stack(
        [
            ((obs.image - pred) ** 2 * obs.weight).sum()
            for obs, pred in zip(observations, result.predictions)
        ]
    )
    return result, chi2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    with skip_if_unavailable():
        image_path = str(fetch_sample("legacy_ngc288_grz"))
        catalog_path = str(fetch_sample("legacy_ngc288_catalog"))
    images = torchfits.read(image_path, hdu=0).to(torch.float64)
    invvar = torchfits.read(image_path, hdu=1).to(torch.float64)
    card = dict(torchfits.read_header(image_path, 0))
    header = {key: card[key] for key in WCS_KEYS}
    header.update(NAXIS=2, NAXIS1=images.shape[2], NAXIS2=images.shape[1])
    pixel_arcsec = abs(float(card["CD2_2"])) * 3600.0

    catalog = torchfits.read(catalog_path, hdu=1)
    ra, dec = catalog["ra"].to(torch.float64), catalog["dec"].to(torch.float64)
    x, y = WCS(header).world_to_pixel(ra, dec)
    ny, nx = images.shape[1:]
    inside = (x > 4) & (x < nx - 5) & (y > 4) & (y < ny - 5)
    ra, dec, x, y = ra[inside], dec[inside], x[inside], y[inside]
    tractor = torch.stack(
        [catalog[f"flux_{b}"].to(torch.float64)[inside] for b in BANDS], 1
    )
    tractor_ivar = torch.stack(
        [catalog[f"flux_ivar_{b}"].to(torch.float64)[inside] for b in BANDS], 1
    )
    print(
        f"{nx} x {ny} px at {pixel_arcsec:.3f} arcsec; {int(inside.sum())} catalogue stars on the image"
    )

    # Gaussian PSF width per band: chi-square over a grid, refined by a parabola.
    chi2 = torch.stack(
        [forced_fit(images, invvar, header, ra, dec, [w] * 3)[1] for w in WIDTHS]
    )
    best = []
    step = float(WIDTHS[1] - WIDTHS[0])
    for k, band in enumerate(BANDS):
        i = min(max(int(chi2[:, k].argmin()), 1), len(WIDTHS) - 2)
        c0, c1, c2 = (float(v) for v in chi2[i - 1 : i + 2, k])
        best.append(float(WIDTHS[i]) + 0.5 * step * (c0 - c2) / (c0 - 2.0 * c1 + c2))
        print(
            f"{band}: PSF sigma {best[-1]:.2f} px, FWHM {best[-1] * 2.3548 * pixel_arcsec:.2f} arcsec"
        )
    result, _ = forced_fit(images, invvar, header, ra, dec, best)
    print(f"blend groups: {len(result.grouping.blend_groups)} for {ra.numel()} stars")

    ratio = result.flux / tractor
    snr = tractor * tractor_ivar.sqrt()
    for k, band in enumerate(BANDS):
        good = snr[:, k] > 50
        q = ratio[good, k]
        median = float(q.median())
        scatter = float(1.4826 * (q - q.median()).abs().median())
        print(
            f"{band}: forced / Tractor flux, {int(good.sum())} stars at S/N > 50: median {median:.3f}, robust scatter {scatter:.3f}"
        )
        assert 0.7 < median < 1.05 and scatter < 0.06

    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-forced-"))
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), constrained_layout=True)
    r_image, r_model = images[1], result.predictions[1]
    lo, hi = (
        float(v)
        for v in torch.quantile(
            r_image.flatten(), torch.tensor([0.05, 0.995], dtype=torch.float64)
        )
    )
    axes[0].imshow(r_image, origin="lower", cmap="gray_r", vmin=lo, vmax=hi)
    axes[0].scatter(x, y, s=30, facecolors="none", edgecolors="C3", linewidths=0.6)
    axes[0].set_title("DR10 r band and Gaia positions", fontsize=9)
    axes[1].imshow(r_model, origin="lower", cmap="gray_r", vmin=lo, vmax=hi)
    axes[1].set_title("rendered scene: point sources x Gaussian PSF", fontsize=9)
    span = 0.3 * hi
    axes[2].imshow(
        r_image - r_model, origin="lower", cmap="RdBu_r", vmin=-span, vmax=span
    )
    axes[2].set_title(
        "residual: faint non-Gaia stars, per-group backgrounds", fontsize=9
    )
    for ax in axes[:3]:
        ax.set(xticks=[], yticks=[])
    for k, band in enumerate(BANDS):
        axes[3].loglog(
            tractor[:, k], result.flux[:, k], ".", ms=4, color=f"C{k}", label=band
        )
    lim = (float(tractor.min()) * 0.7, float(tractor.max()) * 1.4)
    axes[3].plot(lim, lim, "k-", lw=0.6)
    median_ratio = float(ratio[:, 1].median())
    axes[3].plot(
        lim,
        [v * median_ratio for v in lim],
        "k--",
        lw=0.6,
        label=f"x {median_ratio:.2f} (r median)",
    )
    axes[3].set(
        xlim=lim,
        ylim=lim,
        xlabel="Tractor flux (nanomaggies)",
        ylabel="torchsky forced flux",
        title="Forced photometry vs DR10 Tractor",
    )
    axes[3].legend(fontsize=8)
    fig.savefig(out / "legacy_forced_photometry.png", dpi=90)
    print(f"figure: {out / 'legacy_forced_photometry.png'}")


if __name__ == "__main__":
    main()
