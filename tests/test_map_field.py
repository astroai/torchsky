from __future__ import annotations

import torch

from torchsky.map import (
    FieldDomain,
    FieldObservation,
    SineImplicitField,
    estimate_field_domain,
    fit_field_model,
    render_field_cube,
    render_field_observation,
)
from torchsky.spectral import SpectralAxis


def _tan_header(
    size: int,
    *,
    crval1: float,
    crval2: float,
    scale_deg: float,
) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": size / 2.0,
        "CRPIX2": size / 2.0,
        "CRVAL1": crval1,
        "CRVAL2": crval2,
        "CD1_1": -scale_deg,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": scale_deg,
    }


def test_render_field_observation_changes_with_bandpass() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    model = SineImplicitField(domain, hidden_dim=24, hidden_layers=2)
    header = _tan_header(16, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    low = FieldObservation(
        image=torch.zeros((16, 16), dtype=torch.float32),
        header=header,
        spectral_min=4700.0,
        spectral_max=5000.0,
        transmission_center=4820.0,
        transmission_sigma=80.0,
    )
    high = FieldObservation(
        image=torch.zeros((16, 16), dtype=torch.float32),
        header=header,
        spectral_min=6200.0,
        spectral_max=6600.0,
        transmission_center=6400.0,
        transmission_sigma=120.0,
    )
    low_render = render_field_observation(model, low, num_spectral_samples=7)
    high_render = render_field_observation(model, high, num_spectral_samples=7)

    assert low_render.shape == (16, 16)
    assert high_render.shape == (16, 16)
    assert not torch.allclose(low_render, high_render)


def test_fit_field_model_reduces_loss_and_renders_cube() -> None:
    torch.manual_seed(1234)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
    )
    truth_model = SineImplicitField(truth_domain, hidden_dim=24, hidden_layers=2)
    header_a = _tan_header(20, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    header_b = _tan_header(20, crval1=150.01, crval2=2.01, scale_deg=1.2 / 3600.0)
    template_a = FieldObservation(
        image=torch.zeros((20, 20), dtype=torch.float32),
        header=header_a,
        spectral_min=4700.0,
        spectral_max=5050.0,
        transmission_center=4840.0,
        transmission_sigma=100.0,
        psf_sigma_pix=0.45,
    )
    template_b = FieldObservation(
        image=torch.zeros((20, 20), dtype=torch.float32),
        header=header_b,
        spectral_min=6200.0,
        spectral_max=6650.0,
        transmission_center=6420.0,
        transmission_sigma=110.0,
        psf_sigma_pix=0.6,
    )
    obs_a = render_field_observation(truth_model, template_a, num_spectral_samples=7).detach()
    obs_b = render_field_observation(truth_model, template_b, num_spectral_samples=7).detach()
    observations = [
        FieldObservation(
            image=obs_a,
            header=header_a,
            weight=torch.ones_like(obs_a),
            spectral_min=template_a.spectral_min,
            spectral_max=template_a.spectral_max,
            transmission_center=template_a.transmission_center,
            transmission_sigma=template_a.transmission_sigma,
            psf_sigma_pix=template_a.psf_sigma_pix,
        ),
        FieldObservation(
            image=obs_b,
            header=header_b,
            weight=torch.ones_like(obs_b),
            spectral_min=template_b.spectral_min,
            spectral_max=template_b.spectral_max,
            transmission_center=template_b.transmission_center,
            transmission_sigma=template_b.transmission_sigma,
            psf_sigma_pix=template_b.psf_sigma_pix,
        ),
    ]
    fit_domain = estimate_field_domain(observations)
    model = SineImplicitField(fit_domain, hidden_dim=24, hidden_layers=2)
    fit = fit_field_model(
        observations,
        model,
        steps=35,
        lr=1.0e-2,
        num_spectral_samples=7,
    )
    cube = render_field_cube(
        fit.model,
        target_header=header_a,
        spectral_axis=SpectralAxis(
            values=torch.tensor([4800.0, 5600.0, 6400.0], dtype=torch.float64),
            unit="Angstrom",
            type="WAVE",
        ),
    )

    assert fit.final_loss < fit.initial_loss
    assert cube.shape == (3, 20, 20)
