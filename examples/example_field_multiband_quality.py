#!/usr/bin/env python
"""Fit a tiny multiband neural field and report reconstruction quality."""

from __future__ import annotations

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
    )
    quality = field.quality(
        result.model,
        observations,
        num_spectral_samples=5,
        residual_scales=(1, 2, 4),
    )["aggregate"]

    print("Multiband field quality example")
    print(f"final loss: {result.final_loss:.6f}")
    print(f"residual SNR dB: {quality['residual_snr_db']:.3f}")
    print(f"sharpness ratio: {quality['sharpness_ratio']:.3f}")
    print(f"scale-4 residual RMSE: {quality['residual_rmse_scale_4']:.6f}")


if __name__ == "__main__":
    main()
