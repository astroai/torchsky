#!/usr/bin/env python3
"""Generate the demonstration gallery figures.

Committed figures live under ``docs/assets/gallery`` and are embedded by
``docs/gallery.md``. Regenerate deliberately; do not run this during the docs
build (it imports torch and matplotlib).

    pixi run docs-gallery
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "gallery"
DPI = 130


def _tan_header(
    shape: tuple[int, int], crval=(150.0, 2.0), cdelt=0.02
) -> dict[str, object]:
    height, width = shape
    return {
        "NAXIS": 2,
        "NAXIS1": width,
        "NAXIS2": height,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": width / 2.0 + 0.5,
        "CRPIX2": height / 2.0 + 0.5,
        "CRVAL1": crval[0],
        "CRVAL2": crval[1],
        "CDELT1": -cdelt,
        "CDELT2": cdelt,
    }


def _coadd_figure(generator: torch.Generator) -> None:
    from torchsky.map import Observation, coadd, plan

    size = 96
    header = _tan_header((size, size))
    observations = []
    for index in range(3):
        data = torch.zeros((size, size), dtype=torch.float32)
        for _ in range(4):
            x = int(torch.randint(12, size - 12, (1,), generator=generator))
            y = int(torch.randint(12, size - 12, (1,), generator=generator))
            data[y - 1 : y + 2, x - 1 : x + 2] += 1.0
        data = data + 0.02 * torch.randn((size, size), generator=generator)
        observations.append(
            Observation(
                data=data,
                wcs={**header, "CRVAL1": 150.0 + 0.02 * index},
            )
        )
    target = plan(observations, projection="TAN")
    result = coadd(observations, target, kernel="bilinear")

    figure, axes = plt.subplots(1, 4, figsize=(11, 3), constrained_layout=True)
    for axis, observation, title in zip(
        axes[:3], observations, ["exposure 1", "exposure 2", "exposure 3"]
    ):
        axis.imshow(observation.data.detach(), origin="lower", cmap="magma")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    image = axes[3].imshow(result.data.detach(), origin="lower", cmap="magma")
    axes[3].set_title("coadd (bilinear)")
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    figure.colorbar(image, ax=axes[3], fraction=0.046, pad=0.02)
    figure.savefig(OUT / "coadd.png", dpi=DPI)
    plt.close(figure)


def _healpix_figure(generator: torch.Generator) -> None:
    import torchsky.sphere as sphere

    nside = 64
    ell = torch.arange(1, 3 * nside + 1, dtype=torch.float64)
    cl = 1.0 / (ell * (ell + 1.0)) ** 1.2
    torch.manual_seed(11)
    healpix_map = sphere.synfast(cl, nside, lmax=3 * nside)
    npix = 12 * nside**2
    theta, phi = sphere.pix2ang(nside, torch.arange(npix))
    lon = torch.rad2deg(phi) - 180.0
    lat = 90.0 - torch.rad2deg(theta)

    figure, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    scatter = axis.scatter(
        lon.numpy(),
        lat.numpy(),
        c=healpix_map.numpy(),
        s=2.0,
        cmap="viridis",
        linewidths=0,
    )
    axis.set_xlabel("lon [deg]")
    axis.set_ylabel("lat [deg]")
    axis.set_title(f"synthetic HEALPix map (nside={nside}, lmax={3 * nside})")
    figure.colorbar(scatter, ax=axis, fraction=0.03, pad=0.02, label="value")
    figure.savefig(OUT / "healpix.png", dpi=DPI)
    plt.close(figure)


def _sht_figure(generator: torch.Generator) -> None:
    import torchsky.sphere as sphere

    nside = 64
    lmax = 3 * nside
    ell = torch.arange(1, lmax + 1, dtype=torch.float64)
    input_cl = 1.0 / (ell * (ell + 1.0)) ** 1.1
    torch.manual_seed(5)
    healpix_map = sphere.synfast(input_cl, nside, lmax=lmax)
    recovered = sphere.anafast(healpix_map, lmax=lmax, pol=False)

    figure, axis = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    axis.loglog(ell.numpy(), input_cl.numpy(), label="input $C_\\ell$")
    axis.loglog(
        ell.numpy(),
        recovered[:lmax].numpy(),
        label="anafast(synfast($C_\\ell$))",
        alpha=0.8,
    )
    axis.set_xlabel("multipole $\\ell$")
    axis.set_ylabel("$C_\\ell$")
    axis.set_title(f"spherical harmonic round trip (nside={nside})")
    axis.legend()
    figure.savefig(OUT / "sht.png", dpi=DPI)
    plt.close(figure)


def _photometry_figure(generator: torch.Generator) -> None:
    from torchsky.spectrophotometry import Passband, synthetic_photometry

    wavelength = torch.linspace(3500.0, 9000.0, 400, dtype=torch.float64)
    flux = 1.0e-14 * (wavelength / 5000.0) ** -1.0
    bands = [
        Passband.from_samples(
            torch.tensor([4200.0, 4800.0, 5600.0, 6200.0], dtype=torch.float64),
            torch.tensor([0.0, 0.8, 1.0, 0.0], dtype=torch.float64),
            name="g",
        ),
        Passband.from_samples(
            torch.tensor([5600.0, 6500.0, 7500.0, 8500.0], dtype=torch.float64),
            torch.tensor([0.0, 0.7, 1.0, 0.0], dtype=torch.float64),
            name="r",
        ),
    ]
    results = [synthetic_photometry(wavelength, flux, band) for band in bands]

    figure, axis = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    axis.plot(wavelength.numpy(), flux.numpy(), label="input $F_\\lambda$")
    for band, result in zip(bands, results):
        axis.plot(
            band.wavelength.numpy(), band.throughput.numpy(), label=f"{band.name} band"
        )
        axis.axvline(
            float(result.value) * 0.0 + band.pivot_wavelength.item(), ls=":", lw=1
        )
    axis.set_xlabel("wavelength [Angstrom]")
    axis.set_ylabel("flux / throughput")
    axis.set_title("synthetic photometry through two passbands")
    axis.legend()
    figure.savefig(OUT / "photometry.png", dpi=DPI)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    _coadd_figure(generator)
    _healpix_figure(generator)
    _sht_figure(generator)
    _photometry_figure(generator)
    print(f"wrote {len(list(OUT.glob('*.png')))} figures to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
