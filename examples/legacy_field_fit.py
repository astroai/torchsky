"""Fit one continuous field to Legacy Surveys g/r/i/z images of the Cartwheel galaxy.

Science question: can a single continuous model of the sky, a function of
(RA, Dec, wavelength), explain images of a real galaxy taken through four
filters and sampled on two different pixel grids? Six Legacy Surveys DR10
images of the Cartwheel ring galaxy are fetched from the legacysurvey.org
cutout service with their inverse-variance maps. Four bands (g, r, i, z) are on
a 128 x 128 grid of 0.524 arcsec pixels; g and z are also on a 160 x 160 grid
of 0.35 arcsec pixels with a shifted centre, so the model is constrained
through two WCS at once. All six cutouts are resampled from the same DECam
coadds, so they are not independent exposures. The cutout service preserves
surface brightness (nanomaggies per 0.262 arcsec DECam pixel) on every grid,
which is the value semantics the field model uses for both. Each image becomes a
``torchsky.field.FieldObservation`` with its header, inverse-variance weight,
the DECam total throughput of its filter (SVO Filter Profile Service), and a
Gaussian PSF of 1.1 arcsec FWHM. A small coordinate network (the ``fast``
preset) is fitted to all six images jointly with ``field.fit``, then rendered
back through each observation. The example prints, per image, the flux ratio
of model to data and the fraction of the image variance the model explains,
and it shows data, model and residual. The ring, the nucleus and the colour
change between them are recovered. The residuals keep the knots of the ring
and the field stars: this compact network is smoother than the data, and
``field`` is experimental. Runtime about 60 s on CPU once the 1 MB of cutouts
and filter curves are cached.

Data: Legacy Surveys DR10 (Dey et al. 2019, AJ 157, 168), public; DECam
filter curves from the SVO Filter Profile Service (Rodrigo & Solano 2020).

    pixi run -e test -- python examples/legacy_field_fit.py [--device cuda] [--out DIR]
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
from torchsky import field
from torchsky.filter_profiles import (
    bandpass_from_transmission_file,
    field_observation_spectral_kwargs,
)

GRIDS = (("legacy_cartwheel_griz", "griz"), ("legacy_cartwheel_gz_fine", "gz"))
PSF_FWHM_ARCSEC = 1.1
STEPS = 240
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


def load_observations() -> list[field.FieldObservation]:
    with skip_if_unavailable():
        cutouts = {name: fetch_sample(name) for name, _ in GRIDS}
        curves = {band: fetch_sample(f"svo_decam_{band}") for band in "griz"}
    observations = []
    for name, bands in GRIDS:
        path = str(cutouts[name])
        images = torchfits.read(path, hdu=0).to(
            torch.float32
        )  # (band, y, x), nanomaggies
        invvar = torchfits.read(path, hdu=1).to(torch.float32)
        card = dict(torchfits.read_header(path, 0))
        header = {key: card[key] for key in WCS_KEYS}
        header.update(NAXIS=2, NAXIS1=images.shape[2], NAXIS2=images.shape[1])
        pixel_arcsec = abs(float(card["CD2_2"])) * 3600.0
        sigma_px = PSF_FWHM_ARCSEC / 2.3548 / pixel_arcsec
        for index, band in enumerate(bands):
            spectral = field_observation_spectral_kwargs(
                bandpass_from_transmission_file(curves[band], band_name=band)
            )
            spectral["band_name"] = f'{band} {pixel_arcsec:.2f}"'
            observations.append(
                field.FieldObservation(
                    image=images[index],
                    header=header,
                    weight=invvar[index],
                    psf_spec=field.PSFSpec.gaussian(sigma_px),
                    **spectral,
                )
            )
    return observations


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    torch.manual_seed(0)

    observations = load_observations()
    for obs in observations:
        lo, hi = obs.spectral_min, obs.spectral_max
        print(
            f"{obs.band_name:9s} {tuple(obs.image.shape)} px, passband {lo:.0f}-{hi:.0f} A, total {float(obs.image.sum()):7.1f} nmgy"
        )

    domain = field.estimate_domain(observations)
    model = field.preset("fast", domain)
    result = field.fit(
        observations,
        model,
        steps=STEPS,
        lr=5.0e-3,
        num_spectral_samples=3,
        training_patch_size=32,
        training_patches_per_observation=2,
        device=args.device,
    )
    result.model.to("cpu")
    print(
        f"fit: {STEPS} steps, loss {result.loss_history[0]:.4f} -> {result.final_loss:.4f}"
    )
    assert result.final_loss < 0.2 * result.loss_history[0]

    rows = []
    with torch.no_grad():
        for obs in observations:
            model_image = field.render(result.model, obs, num_spectral_samples=3)
            residual = obs.image - model_image
            explained = 1.0 - float(residual.var() / obs.image.var())
            ratio = float(model_image.sum() / obs.image.sum())
            rows.append((obs, model_image, residual))
            print(
                f"{obs.band_name:9s} model/data flux {ratio:.3f}, variance explained {explained:.3f}"
            )
            assert 0.8 < ratio < 1.2 and explained > 0.6

    # Colour of the ring and the nucleus from the model on the coarse grid.
    g_model, z_model = rows[0][1], rows[3][1]
    yy, xx = torch.meshgrid(torch.arange(128.0), torch.arange(128.0), indexing="ij")
    nucleus = (xx - 63.5) ** 2 + (yy - 63.5) ** 2 < 4.0**2
    ring = ((xx - 63.5) ** 2 + (yy - 63.5) ** 2).sqrt().sub(48.0).abs() < 8.0

    def colour(
        mask: torch.Tensor, model_g: torch.Tensor, model_z: torch.Tensor
    ) -> float:
        return float(-2.5 * torch.log10(model_g[mask].sum() / model_z[mask].sum()))

    print(
        f"model g-z colour: nucleus {colour(nucleus, g_model, z_model):+.2f} mag, "
        f"ring {colour(ring, g_model, z_model):+.2f} mag (the ring is bluer)"
    )

    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-legacy-field-"))
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        3, len(rows), figsize=(2.1 * len(rows), 6.6), constrained_layout=True
    )
    for col, (obs, model_image, residual) in enumerate(rows):
        hi = float(torch.quantile(obs.image.flatten(), 0.995))
        noise = float(1.4826 * (obs.image - obs.image.median()).abs().median())
        for row, (img, label, kw) in enumerate(
            (
                (obs.image, "data", {"vmin": -noise, "vmax": hi}),
                (model_image, "field model", {"vmin": -noise, "vmax": hi}),
                (
                    residual,
                    "residual",
                    {"vmin": -0.3 * hi, "vmax": 0.3 * hi, "cmap": "RdBu_r"},
                ),
            )
        ):
            ax = axes[row, col]
            ax.imshow(img, origin="lower", cmap=kw.pop("cmap", "magma"), **kw)
            ax.set(xticks=[], yticks=[])
            ax.set_title(f"{obs.band_name}: {label}", fontsize=8)
    fig.savefig(out / "legacy_field_fit.png", dpi=90)
    print(f"figure: {out / 'legacy_field_fit.png'}; device={args.device}")


if __name__ == "__main__":
    main()
