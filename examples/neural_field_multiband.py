"""Fit a continuous multiband neural field to three noisy images.

Three 24x24 images of one synthetic two-core galaxy (blue, green, red bands,
different noise levels, masked edge rows) are fitted by a small coordinate
network that is a function of (RA, Dec, wavelength). The fitted field is then
rendered back through each band and compared with the noiseless truth and the
noisy data; the example prints residual SNR and sharpness from
``field.quality``. The field API is experimental. Runtime about 5 s on CPU.

    pixi run -e test -- python examples/neural_field_multiband.py [--device cuda] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from torchsky import field


def _tan_header(size: int) -> dict[str, float | int | str]:
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 0.5 * (size + 1),
        "CRPIX2": 0.5 * (size + 1),
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CDELT1": -0.08 / 3600.0,
        "CDELT2": 0.08 / 3600.0,
    }


def _synthetic_band(size: int, wavelength: float) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    blue_core = torch.exp(-((xx + 0.22) ** 2 + (yy - 0.08) ** 2) / 0.045)
    red_core = torch.exp(-((xx - 0.18) ** 2 + (yy + 0.14) ** 2) / 0.030)
    ridge = torch.exp(-((yy - 0.18 * torch.sin(3.0 * xx)) ** 2) / 0.018)
    spectral_t = (float(wavelength) - 5600.0) / 1200.0
    image = (1.0 - 0.25 * spectral_t) * blue_core
    image = image + (0.7 + 0.35 * spectral_t) * red_core
    image = image + 0.20 * ridge
    return image.to(dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-field-"))
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    size = 24
    header = _tan_header(size)
    observations: list[field.FieldObservation] = []
    for name, wavelength, noise_sigma in (
        ("blue", 5000.0, 0.025),
        ("green", 5600.0, 0.020),
        ("red", 6800.0, 0.030),
    ):
        clean = _synthetic_band(size, wavelength)
        image = clean + noise_sigma * torch.randn_like(clean)
        mask = torch.ones_like(image, dtype=torch.bool)
        mask[:2, :] = False
        mask[-2:, :] = False
        weight = torch.full_like(image, 1.0 / (noise_sigma * noise_sigma))
        observations.append(
            field.FieldObservation(
                image=image,
                header=dict(header),
                weight=weight,
                mask=mask,
                spectral_min=wavelength - 60.0,
                spectral_max=wavelength + 60.0,
                transmission_center=wavelength,
                transmission_sigma=24.0,
                band_name=name,
            )
        )

    domain = field.estimate_domain(observations)
    model = field.preset("fast", domain, hidden_dim=48, hidden_layers=2)
    result = field.fit(
        observations,
        model,
        steps=60,
        lr=5.0e-3,
        loss_kind="charbonnier",
        charbonnier_eps=1.0e-3,
        num_spectral_samples=5,
        gradient_residual_weight=0.05,
        device=args.device,
    )
    result.model.to("cpu")  # fit on --device, evaluate on CPU
    quality = field.quality(
        result.model,
        observations,
        num_spectral_samples=5,
        residual_scales=(1, 2, 4),
    )["aggregate"]

    print(f"final loss: {result.final_loss:.6f}")
    print(f"residual SNR dB: {quality['residual_snr_db']:.3f}")
    print(f"sharpness ratio: {quality['sharpness_ratio']:.3f}")
    print(f"scale-4 residual RMSE: {quality['residual_rmse_scale_4']:.6f}")
    assert quality["residual_snr_db"] > 10.0

    fig, axes = plt.subplots(3, 3, figsize=(6.5, 6.5), constrained_layout=True)
    with torch.no_grad():
        for row, obs in enumerate(observations):
            clean = _synthetic_band(size, obs.transmission_center)
            fit_img = field.render(result.model, obs).detach().cpu()
            for ax, (title, im) in zip(
                axes[row],
                [("data", obs.image), ("field fit", fit_img), ("truth", clean)],
            ):
                ax.imshow(im.cpu(), origin="lower", cmap="magma", vmin=0, vmax=1.3)
                ax.set_title(f"{obs.band_name}: {title}", fontsize=8)
                ax.axis("off")
    fig.savefig(out / "neural_field_multiband.png", dpi=90)
    print(f"device={args.device}  figure: {out / 'neural_field_multiband.png'}")


if __name__ == "__main__":
    main()
