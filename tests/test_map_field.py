from __future__ import annotations

import math

import pytest
import torch

from torchsky.map import (
    FieldDomain,
    FieldObservation,
    SineImplicitField,
    estimate_field_domain,
    fit_field_model,
    make_field_model,
    render_field_cube,
    render_field_observation,
    render_field_observation_distribution,
)
from torchsky.spectral import SpectralAxis
from torchsky.map.field import _noise_pretrain_field_model
from torchsky.map.field.core import _HashGridEncoder, _bandpass_quadrature


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

    assert math.isfinite(fit.final_loss)
    assert fit.final_loss < 1.0e-2
    assert cube.shape == (3, 20, 20)


def test_fit_field_model_supports_per_band_reduction() -> None:
    torch.manual_seed(1234)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
    )
    truth_model = SineImplicitField(truth_domain, hidden_dim=24, hidden_layers=2)
    header = _tan_header(16, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    templates = [
        FieldObservation(
            image=torch.zeros((16, 16), dtype=torch.float32),
            header=header,
            spectral_min=4700.0,
            spectral_max=4700.0,
            transmission_center=4700.0,
            transmission_sigma=None,
            band_name="b0",
        ),
        FieldObservation(
            image=torch.zeros((16, 16), dtype=torch.float32),
            header=header,
            spectral_min=4700.0,
            spectral_max=4700.0,
            transmission_center=4700.0,
            transmission_sigma=None,
            band_name="b0",
        ),
        FieldObservation(
            image=torch.zeros((16, 16), dtype=torch.float32),
            header=header,
            spectral_min=6200.0,
            spectral_max=6200.0,
            transmission_center=6200.0,
            transmission_sigma=None,
            band_name="b1",
        ),
    ]
    observations = []
    for idx, template in enumerate(templates):
        image = render_field_observation(truth_model, template, num_spectral_samples=1).detach()
        weight = torch.full_like(image, 10.0 if idx < 2 else 1.0)
        observations.append(
            FieldObservation(
                image=image,
                header=template.header,
                weight=weight,
                spectral_min=template.spectral_min,
                spectral_max=template.spectral_max,
                transmission_center=template.transmission_center,
                transmission_sigma=template.transmission_sigma,
                band_name=template.band_name,
            )
        )
    fit_domain = estimate_field_domain(observations)
    model = SineImplicitField(fit_domain, hidden_dim=24, hidden_layers=2)
    fit = fit_field_model(
        observations,
        model,
        steps=10,
        lr=1.0e-2,
        num_spectral_samples=1,
        loss_reduction="per_band",
    )
    assert fit.final_loss < fit.initial_loss


def test_fit_field_model_ignores_nonfinite_masked_pixels() -> None:
    torch.manual_seed(1234)
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
    )
    truth_model = SineImplicitField(domain, hidden_dim=24, hidden_layers=2)
    header = _tan_header(16, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    template = FieldObservation(
        image=torch.zeros((16, 16), dtype=torch.float32),
        header=header,
        spectral_min=5600.0,
        spectral_max=5600.0,
        transmission_center=5600.0,
        transmission_sigma=None,
        band_name="b0",
    )
    image = render_field_observation(truth_model, template, num_spectral_samples=1).detach()
    image_with_nan = image.clone()
    image_with_nan[0, 0] = float("nan")
    weight = torch.ones_like(image)
    mask = torch.ones_like(image)
    mask[0, 0] = 0.0
    observations = [
        FieldObservation(
            image=image_with_nan,
            header=header,
            weight=weight,
            mask=mask,
            spectral_min=template.spectral_min,
            spectral_max=template.spectral_max,
            transmission_center=template.transmission_center,
            transmission_sigma=template.transmission_sigma,
            band_name=template.band_name,
        )
    ]
    fit_domain = estimate_field_domain(observations)
    model = SineImplicitField(fit_domain, hidden_dim=24, hidden_layers=2)
    fit = fit_field_model(
        observations,
        model,
        steps=4,
        lr=1.0e-3,
        num_spectral_samples=1,
        loss_reduction="global",
    )
    pred = render_field_observation(fit.model, template, num_spectral_samples=1)

    assert torch.isfinite(torch.tensor(fit.loss_history)).all()
    assert torch.isfinite(pred).all()


def test_fit_field_model_supports_band_balanced_patch_training() -> None:
    torch.manual_seed(2234)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("sky_lambda_feature_field", truth_domain)
    header = _tan_header(24, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    wavelengths = (4800.0, 4800.0, 6200.0, 6200.0, 7600.0, 7600.0)
    observations = []
    for idx, wavelength in enumerate(wavelengths):
        template = FieldObservation(
            image=torch.zeros((24, 24), dtype=torch.float32),
            header=header,
            spectral_min=wavelength,
            spectral_max=wavelength,
            transmission_center=wavelength,
            transmission_sigma=None,
            band_name=f"b{idx // 2}",
        )
        image = render_field_observation(truth_model, template, num_spectral_samples=1).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=template.header,
                weight=torch.ones_like(image),
                spectral_min=template.spectral_min,
                spectral_max=template.spectral_max,
                transmission_center=template.transmission_center,
                transmission_sigma=template.transmission_sigma,
                band_name=template.band_name,
            )
        )
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    model = make_field_model("sky_lambda_feature_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=8,
        lr=2.0e-3,
        num_spectral_samples=1,
        loss_kind="mse",
        loss_reduction="global",
        observations_per_step=3,
        observation_sampling="band_balanced",
        training_patch_size=16,
    )
    assert math.isfinite(fit.final_loss)
    assert fit.final_loss < 5.0e-3


def test_noise_pretraining_updates_trunk_but_restores_decoder_head() -> None:
    torch.manual_seed(3210)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
    )
    truth_model = make_field_model(
        "sky_encoder_mlp",
        truth_domain,
        hidden_dim=48,
        hidden_layers=2,
        num_fourier_frequencies=4,
        spectral_frequencies=3,
    )
    header = _tan_header(20, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    templates = [
        FieldObservation(
            image=torch.zeros((20, 20), dtype=torch.float32),
            header=header,
            spectral_min=4700.0,
            spectral_max=4700.0,
            transmission_center=4700.0,
            transmission_sigma=None,
            band_name="b0",
        ),
        FieldObservation(
            image=torch.zeros((20, 20), dtype=torch.float32),
            header=header,
            spectral_min=6500.0,
            spectral_max=6500.0,
            transmission_center=6500.0,
            transmission_sigma=None,
            band_name="b1",
        ),
    ]
    observations = []
    for template in templates:
        image = render_field_observation(
            truth_model,
            template,
            num_spectral_samples=1,
        ).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=template.header,
                weight=torch.ones_like(image),
                spectral_min=template.spectral_min,
                spectral_max=template.spectral_max,
                transmission_center=template.transmission_center,
                transmission_sigma=template.transmission_sigma,
                band_name=template.band_name,
            )
        )
    fit_domain = estimate_field_domain(observations)

    torch.manual_seed(777)
    model = make_field_model(
        "sky_encoder_mlp",
        fit_domain,
        hidden_dim=48,
        hidden_layers=2,
        num_fourier_frequencies=4,
        spectral_frequencies=3,
    )
    initial_hidden_weight = model.hidden[0].weight.detach().clone()
    initial_output_weight = model.output.weight.detach().clone()
    initial_output_bias = model.output.bias.detach().clone()
    _noise_pretrain_field_model(
        observations,
        model,
        noise_kind="gaussian",
        steps=12,
        lr=5.0e-3,
        observations_per_step=2,
        pixel_samples_per_observation=128,
        num_terms=32,
        max_frequency=6.0,
    )

    assert not torch.allclose(model.hidden[0].weight, initial_hidden_weight)
    assert torch.allclose(model.output.weight, initial_output_weight)
    assert torch.allclose(model.output.bias, initial_output_bias)


def test_fit_field_model_supports_noise_pretraining_for_sky_encoder_mlp() -> None:
    torch.manual_seed(3210)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
    )
    truth_model = make_field_model(
        "sky_encoder_mlp",
        truth_domain,
        hidden_dim=48,
        hidden_layers=2,
        num_fourier_frequencies=4,
        spectral_frequencies=3,
    )
    header = _tan_header(20, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    template = FieldObservation(
        image=torch.zeros((20, 20), dtype=torch.float32),
        header=header,
        spectral_min=5600.0,
        spectral_max=5600.0,
        transmission_center=5600.0,
        transmission_sigma=None,
        band_name="b0",
    )
    image = render_field_observation(
        truth_model,
        template,
        num_spectral_samples=1,
    ).detach()
    observations = [
        FieldObservation(
            image=image,
            header=template.header,
            weight=torch.ones_like(image),
            spectral_min=template.spectral_min,
            spectral_max=template.spectral_max,
            transmission_center=template.transmission_center,
            transmission_sigma=template.transmission_sigma,
            band_name=template.band_name,
        )
    ]
    fit_domain = estimate_field_domain(observations)
    torch.manual_seed(777)
    model = make_field_model(
        "sky_encoder_mlp",
        fit_domain,
        hidden_dim=48,
        hidden_layers=2,
        num_fourier_frequencies=4,
        spectral_frequencies=3,
    )
    fit = fit_field_model(
        observations,
        model,
        steps=6,
        lr=5.0e-3,
        num_spectral_samples=1,
        loss_kind="mse",
        loss_reduction="global",
        noise_pretrain_kind="gaussian",
        noise_pretrain_steps=24,
        noise_pretrain_observations_per_step=2,
        noise_pretrain_pixel_samples_per_observation=128,
        noise_pretrain_num_terms=32,
        noise_pretrain_max_frequency=6.0,
    )

    assert fit.final_loss < fit.initial_loss


def test_fit_field_model_supports_streaming_band_balanced_patch_training() -> None:
    torch.manual_seed(2235)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("sky_lambda_feature_field", truth_domain)
    header = _tan_header(24, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    wavelengths = (4700.0, 4700.0, 6200.0, 6200.0, 7800.0, 7800.0)
    observations = []
    for idx, wavelength in enumerate(wavelengths):
        template = FieldObservation(
            image=torch.zeros((24, 24), dtype=torch.float32),
            header=header,
            spectral_min=wavelength,
            spectral_max=wavelength,
            transmission_center=wavelength,
            transmission_sigma=None,
            band_name=f"b{idx // 2}",
        )
        image = render_field_observation(truth_model, template, num_spectral_samples=1).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=template.header,
                weight=torch.ones_like(image),
                spectral_min=template.spectral_min,
                spectral_max=template.spectral_max,
                transmission_center=template.transmission_center,
                transmission_sigma=template.transmission_sigma,
                band_name=template.band_name,
            )
        )
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    model = make_field_model("sky_lambda_feature_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=8,
        lr=2.0e-3,
        num_spectral_samples=1,
        loss_kind="mse",
        loss_reduction="band_global_mix",
        observations_per_step=6,
        observation_sampling="band_balanced",
        training_patch_size=12,
        streaming_accumulation=True,
        full_observations_per_step=1,
    )
    assert fit.final_loss < fit.initial_loss


def test_fit_field_model_supports_shared_sky_patch_sampling() -> None:
    torch.manual_seed(2237)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("sky_lambda_feature_field", truth_domain)
    headers = (
        _tan_header(24, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        _tan_header(20, crval1=150.0004, crval2=1.9998, scale_deg=1.3 / 3600.0),
        _tan_header(28, crval1=149.9997, crval2=2.0002, scale_deg=0.9 / 3600.0),
    )
    wavelengths = (4700.0, 4700.0, 6200.0, 6200.0, 7800.0, 7800.0)
    observations = []
    for idx, wavelength in enumerate(wavelengths):
        header = headers[idx % len(headers)]
        template = FieldObservation(
            image=torch.zeros(
                (int(header["NAXIS2"]), int(header["NAXIS1"])), dtype=torch.float32
            ),
            header=header,
            spectral_min=wavelength,
            spectral_max=wavelength,
            transmission_center=wavelength,
            transmission_sigma=None,
            band_name=f"b{idx // 2}",
        )
        image = render_field_observation(
            truth_model, template, num_spectral_samples=1
        ).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=template.header,
                weight=torch.ones_like(image),
                spectral_min=template.spectral_min,
                spectral_max=template.spectral_max,
                transmission_center=template.transmission_center,
                transmission_sigma=template.transmission_sigma,
                band_name=template.band_name,
            )
        )
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    model = make_field_model("sky_lambda_feature_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=8,
        lr=2.0e-3,
        num_spectral_samples=1,
        loss_kind="mse",
        loss_reduction="global",
        observations_per_step=6,
        observation_sampling="band_balanced",
        training_patch_size=12,
        shared_sky_patch_sampling=True,
    )
    assert fit.final_loss < fit.initial_loss


def test_fit_field_model_supports_pixel_sampled_training() -> None:
    torch.manual_seed(2238)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("spectral_basis_field", truth_domain)
    headers = (
        _tan_header(24, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        _tan_header(20, crval1=150.0003, crval2=2.0001, scale_deg=1.2 / 3600.0),
        _tan_header(28, crval1=149.9998, crval2=1.9997, scale_deg=0.9 / 3600.0),
    )
    wavelengths = (4700.0, 4700.0, 6200.0, 6200.0, 7800.0, 7800.0)
    observations = []
    for idx, wavelength in enumerate(wavelengths):
        header = headers[idx % len(headers)]
        template = FieldObservation(
            image=torch.zeros(
                (int(header["NAXIS2"]), int(header["NAXIS1"])), dtype=torch.float32
            ),
            header=header,
            spectral_min=wavelength,
            spectral_max=wavelength,
            transmission_center=wavelength,
            transmission_sigma=None,
            band_name=f"b{idx // 2}",
        )
        image = render_field_observation(
            truth_model, template, num_spectral_samples=1
        ).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=template.header,
                weight=torch.ones_like(image),
                spectral_min=template.spectral_min,
                spectral_max=template.spectral_max,
                transmission_center=template.transmission_center,
                transmission_sigma=template.transmission_sigma,
                band_name=template.band_name,
            )
        )
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    model = make_field_model("spectral_basis_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=8,
        lr=2.0e-3,
        num_spectral_samples=1,
        loss_kind="mse",
        loss_reduction="global",
        observations_per_step=6,
        observation_sampling="band_balanced",
        pixel_samples_per_observation=64,
    )
    assert math.isfinite(fit.final_loss)
    assert fit.final_loss < 1.0e-4


def test_spectral_basis_field_render_bandpass_respects_weights() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
        coord_mode="sphere_xyz",
    )
    torch.manual_seed(2240)
    model = make_field_model("spectral_basis_field", domain)
    ra = torch.full((3, 3), 150.0, dtype=torch.float32)
    dec = torch.full((3, 3), 2.0, dtype=torch.float32)
    wavelengths = torch.tensor([5000.0, 7000.0], dtype=torch.float32)
    first_only = model.render_bandpass(
        ra,
        dec,
        wavelengths,
        torch.tensor([1.0, 0.0], dtype=torch.float32),
    )
    second_only = model.render_bandpass(
        ra,
        dec,
        wavelengths,
        torch.tensor([0.0, 1.0], dtype=torch.float32),
    )
    mixed = model.render_bandpass(
        ra,
        dec,
        wavelengths,
        torch.tensor([0.5, 0.5], dtype=torch.float32),
    )

    assert not torch.allclose(first_only, second_only)
    assert not torch.allclose(mixed, first_only)
    assert not torch.allclose(mixed, second_only)


def test_fit_field_model_supports_downsampled_context_training() -> None:
    torch.manual_seed(2236)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("sky_lambda_feature_field", truth_domain)
    header = _tan_header(24, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    wavelengths = (4700.0, 6200.0, 7800.0)
    observations = []
    for wavelength in wavelengths:
        template = FieldObservation(
            image=torch.zeros((24, 24), dtype=torch.float32),
            header=header,
            spectral_min=wavelength,
            spectral_max=wavelength,
            transmission_center=wavelength,
            transmission_sigma=None,
            band_name=f"b{int(wavelength)}",
        )
        image = render_field_observation(truth_model, template, num_spectral_samples=1).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=template.header,
                weight=torch.ones_like(image),
                spectral_min=template.spectral_min,
                spectral_max=template.spectral_max,
                transmission_center=template.transmission_center,
                transmission_sigma=template.transmission_sigma,
                band_name=template.band_name,
            )
        )
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    model = make_field_model("sky_lambda_feature_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=8,
        lr=2.0e-3,
        num_spectral_samples=1,
        loss_kind="mse",
        loss_reduction="global",
        observations_per_step=1,
        training_patch_size=12,
        context_observations_per_step=1,
        context_downsample_factor=4,
        context_loss_weight=0.25,
    )
    assert fit.final_loss < fit.initial_loss


def test_fit_field_model_supports_spatial_smoothness_regularization() -> None:
    torch.manual_seed(1234)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
    )
    truth_model = SineImplicitField(truth_domain, hidden_dim=24, hidden_layers=2)
    header = _tan_header(16, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    template = FieldObservation(
        image=torch.zeros((16, 16), dtype=torch.float32),
        header=header,
        spectral_min=4700.0,
        spectral_max=5050.0,
        transmission_center=4840.0,
        transmission_sigma=100.0,
    )
    image = render_field_observation(truth_model, template, num_spectral_samples=5).detach()
    observations = [
        FieldObservation(
            image=image,
            header=template.header,
            weight=torch.ones_like(image),
            spectral_min=template.spectral_min,
            spectral_max=template.spectral_max,
            transmission_center=template.transmission_center,
            transmission_sigma=template.transmission_sigma,
        )
    ]
    fit_domain = estimate_field_domain(observations)
    model = SineImplicitField(fit_domain, hidden_dim=24, hidden_layers=2)
    fit = fit_field_model(
        observations,
        model,
        steps=8,
        lr=1.0e-2,
        num_spectral_samples=5,
        spatial_smoothness_weight=1.0e-3,
        spatial_smoothness_grid_size=6,
    )
    assert len(fit.loss_history) == 8
    assert torch.isfinite(torch.tensor(fit.final_loss))


def test_fit_field_model_supports_bandpass_mc_training() -> None:
    torch.manual_seed(2468)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1200.0,
    )
    truth_model = SineImplicitField(truth_domain, hidden_dim=24, hidden_layers=2)
    header = _tan_header(16, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    templates = [
        FieldObservation(
            image=torch.zeros((16, 16), dtype=torch.float32),
            header=header,
            spectral_min=4700.0,
            spectral_max=5300.0,
            transmission_center=5000.0,
            transmission_sigma=120.0,
            band_name="blue",
        ),
        FieldObservation(
            image=torch.zeros((16, 16), dtype=torch.float32),
            header=header,
            spectral_min=6100.0,
            spectral_max=6900.0,
            transmission_center=6500.0,
            transmission_sigma=180.0,
            band_name="red",
        ),
    ]
    observations = []
    for template in templates:
        image = render_field_observation(
            truth_model,
            template,
            num_spectral_samples=7,
        ).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=template.header,
                weight=torch.ones_like(image),
                spectral_min=template.spectral_min,
                spectral_max=template.spectral_max,
                transmission_center=template.transmission_center,
                transmission_sigma=template.transmission_sigma,
                band_name=template.band_name,
            )
        )
    fit_domain = estimate_field_domain(observations)
    model = SineImplicitField(fit_domain, hidden_dim=24, hidden_layers=2)
    fit = fit_field_model(
        observations,
        model,
        steps=8,
        lr=1.0e-2,
        num_spectral_samples=5,
        training_spectral_sampling="bandpass_mc",
        training_spectral_sampling_grid_size=33,
    )
    pred = render_field_observation(
        fit.model,
        observations[0],
        num_spectral_samples=7,
    )
    assert fit.final_loss < fit.initial_loss
    assert pred.shape == observations[0].image.shape
    assert torch.isfinite(pred).all()


def test_fit_field_model_supports_mixture_mc_training() -> None:
    torch.manual_seed(8642)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5800.0,
        wavelength_scale=1400.0,
    )
    truth_model = SineImplicitField(truth_domain, hidden_dim=24, hidden_layers=2)
    header = _tan_header(16, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    templates = [
        FieldObservation(
            image=torch.zeros((16, 16), dtype=torch.float32),
            header=header,
            spectral_min=4500.0,
            spectral_max=5400.0,
            transmission_center=4950.0,
            transmission_sigma=160.0,
            band_name="g",
        ),
        FieldObservation(
            image=torch.zeros((16, 16), dtype=torch.float32),
            header=header,
            spectral_min=5600.0,
            spectral_max=7000.0,
            transmission_center=6300.0,
            transmission_sigma=240.0,
            band_name="r",
        ),
    ]
    observations = []
    for template in templates:
        image = render_field_observation(
            truth_model,
            template,
            num_spectral_samples=7,
        ).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=template.header,
                weight=torch.ones_like(image),
                spectral_min=template.spectral_min,
                spectral_max=template.spectral_max,
                transmission_center=template.transmission_center,
                transmission_sigma=template.transmission_sigma,
                band_name=template.band_name,
            )
        )
    fit_domain = estimate_field_domain(observations)
    model = SineImplicitField(fit_domain, hidden_dim=24, hidden_layers=2)
    fit = fit_field_model(
        observations,
        model,
        steps=8,
        lr=1.0e-2,
        num_spectral_samples=5,
        training_spectral_sampling="mixture_mc",
        training_spectral_sampling_grid_size=33,
        spectral_anchor_observations_per_step=1,
        spectral_anchor_num_samples=11,
        spectral_anchor_loss_weight=0.1,
    )
    assert fit.final_loss < fit.initial_loss


def test_fit_field_model_supports_sphere_xyz_coordinates() -> None:
    torch.manual_seed(4321)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("wire_implicit", truth_domain)
    header = _tan_header(18, crval1=150.0, crval2=2.0, scale_deg=1.1 / 3600.0)
    template = FieldObservation(
        image=torch.zeros((18, 18), dtype=torch.float32),
        header=header,
        spectral_min=5000.0,
        spectral_max=6200.0,
        transmission_center=5600.0,
        transmission_sigma=180.0,
        psf_sigma_pix=0.5,
    )
    image = render_field_observation(truth_model, template, num_spectral_samples=5).detach()
    observations = [
        FieldObservation(
            image=image,
            header=header,
            weight=torch.ones_like(image),
            spectral_min=template.spectral_min,
            spectral_max=template.spectral_max,
            transmission_center=template.transmission_center,
            transmission_sigma=template.transmission_sigma,
            psf_sigma_pix=template.psf_sigma_pix,
        )
    ]
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    model = make_field_model("wire_implicit", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=20,
        lr=1.0e-2,
        num_spectral_samples=5,
    )
    cube = render_field_cube(
        fit.model,
        target_header=header,
        spectral_axis=SpectralAxis(
            values=torch.tensor([5200.0, 5600.0, 6000.0], dtype=torch.float64),
            unit="Angstrom",
            type="WAVE",
        ),
    )

    assert fit_domain.coord_mode == "sphere_xyz"
    assert fit.final_loss < fit.initial_loss
    assert cube.shape == (3, 18, 18)


def test_fourier_mlp_field_supports_charbonnier_fit() -> None:
    torch.manual_seed(2468)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("fourier_mlp", truth_domain)
    header = _tan_header(18, crval1=150.0, crval2=2.0, scale_deg=0.9 / 3600.0)
    template = FieldObservation(
        image=torch.zeros((18, 18), dtype=torch.float32),
        header=header,
        spectral_min=5000.0,
        spectral_max=6200.0,
        transmission_center=5600.0,
        transmission_sigma=180.0,
        psf_sigma_pix=0.45,
    )
    image = render_field_observation(truth_model, template, num_spectral_samples=5).detach()
    observations = [
        FieldObservation(
            image=image,
            header=header,
            weight=torch.ones_like(image),
            spectral_min=template.spectral_min,
            spectral_max=template.spectral_max,
            transmission_center=template.transmission_center,
            transmission_sigma=template.transmission_sigma,
            psf_sigma_pix=template.psf_sigma_pix,
        )
    ]
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    model = make_field_model("fourier_mlp", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=16,
        lr=1.0e-2,
        num_spectral_samples=5,
        loss_kind="charbonnier",
    )

    assert all(math.isfinite(loss) for loss in fit.loss_history)
    assert fit.final_loss < 2.0e-2


def test_spectral_basis_field_supports_exact_wavelength_fit() -> None:
    torch.manual_seed(5312)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("spectral_basis_field", truth_domain)
    header = _tan_header(18, crval1=150.0, crval2=2.0, scale_deg=0.9 / 3600.0)
    wavelengths = (4800.0, 5600.0, 6400.0)
    observations: list[FieldObservation] = []
    for wavelength in wavelengths:
        template = FieldObservation(
            image=torch.zeros((18, 18), dtype=torch.float32),
            header=header,
            wavelength=wavelength,
        )
        image = render_field_observation(
            truth_model,
            template,
            num_spectral_samples=1,
        ).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=header,
                weight=torch.ones_like(image),
                wavelength=wavelength,
            )
        )
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    torch.manual_seed(5313)
    model = make_field_model("spectral_basis_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=12,
        lr=4.0e-3,
        num_spectral_samples=1,
        loss_kind="charbonnier",
        lr_schedule="cosine",
        grad_clip_norm=1.0,
    )

    assert all(math.isfinite(loss) for loss in fit.loss_history)
    assert fit.final_loss < 1.5e-2


def test_conditioned_spectral_field_supports_exact_wavelength_fit() -> None:
    torch.manual_seed(5314)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("conditioned_spectral_field", truth_domain)
    header = _tan_header(18, crval1=150.0, crval2=2.0, scale_deg=0.9 / 3600.0)
    wavelengths = (4800.0, 5600.0, 6400.0)
    observations: list[FieldObservation] = []
    for wavelength in wavelengths:
        template = FieldObservation(
            image=torch.zeros((18, 18), dtype=torch.float32),
            header=header,
            wavelength=wavelength,
        )
        image = render_field_observation(
            truth_model,
            template,
            num_spectral_samples=1,
        ).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=header,
                weight=torch.ones_like(image),
                wavelength=wavelength,
            )
        )
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    torch.manual_seed(5315)
    model = make_field_model("conditioned_spectral_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=12,
        lr=4.0e-3,
        num_spectral_samples=1,
        loss_kind="charbonnier",
        lr_schedule="cosine",
        grad_clip_norm=1.0,
    )

    assert min(fit.loss_history) < fit.initial_loss


def test_learned_spectral_basis_field_supports_exact_wavelength_fit() -> None:
    torch.manual_seed(5316)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("learned_spectral_basis_field", truth_domain)
    header = _tan_header(18, crval1=150.0, crval2=2.0, scale_deg=0.9 / 3600.0)
    wavelengths = (4800.0, 5600.0, 6400.0)
    observations: list[FieldObservation] = []
    for wavelength in wavelengths:
        template = FieldObservation(
            image=torch.zeros((18, 18), dtype=torch.float32),
            header=header,
            wavelength=wavelength,
        )
        image = render_field_observation(
            truth_model,
            template,
            num_spectral_samples=1,
        ).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=header,
                weight=torch.ones_like(image),
                wavelength=wavelength,
            )
        )
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    torch.manual_seed(5317)
    model = make_field_model("learned_spectral_basis_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=12,
        lr=4.0e-3,
        num_spectral_samples=1,
        loss_kind="charbonnier",
        lr_schedule="cosine",
        grad_clip_norm=1.0,
    )

    assert min(fit.loss_history) < fit.initial_loss


def test_sky_lambda_planes_field_supports_exact_wavelength_fit() -> None:
    torch.manual_seed(5318)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("sky_lambda_planes_field", truth_domain)
    header = _tan_header(18, crval1=150.0, crval2=2.0, scale_deg=0.9 / 3600.0)
    wavelengths = (4800.0, 5600.0, 6400.0)
    observations: list[FieldObservation] = []
    for wavelength in wavelengths:
        template = FieldObservation(
            image=torch.zeros((18, 18), dtype=torch.float32),
            header=header,
            wavelength=wavelength,
        )
        image = render_field_observation(
            truth_model,
            template,
            num_spectral_samples=1,
        ).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=header,
                weight=torch.ones_like(image),
                wavelength=wavelength,
            )
        )
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    torch.manual_seed(5319)
    model = make_field_model("sky_lambda_planes_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=12,
        lr=4.0e-3,
        num_spectral_samples=1,
        loss_kind="charbonnier",
        lr_schedule="cosine",
        grad_clip_norm=1.0,
    )

    assert min(fit.loss_history) < fit.initial_loss


def test_sky_lambda_kplanes_field_supports_exact_wavelength_fit() -> None:
    torch.manual_seed(5418)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("sky_lambda_kplanes_field", truth_domain)
    header = _tan_header(18, crval1=150.0, crval2=2.0, scale_deg=0.9 / 3600.0)
    wavelengths = (4800.0, 5600.0, 6400.0)
    observations: list[FieldObservation] = []
    for wavelength in wavelengths:
        template = FieldObservation(
            image=torch.zeros((18, 18), dtype=torch.float32),
            header=header,
            wavelength=wavelength,
        )
        image = render_field_observation(
            truth_model,
            template,
            num_spectral_samples=1,
        ).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=header,
                weight=torch.ones_like(image),
                wavelength=wavelength,
            )
        )
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    torch.manual_seed(5419)
    model = make_field_model("sky_lambda_kplanes_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=12,
        lr=4.0e-3,
        num_spectral_samples=1,
        loss_kind="charbonnier",
        lr_schedule="cosine",
        grad_clip_norm=1.0,
    )

    assert min(fit.loss_history) < fit.initial_loss


def test_sky_lambda_atlas_field_supports_exact_wavelength_fit() -> None:
    torch.manual_seed(5420)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("sky_lambda_atlas_field", truth_domain)
    header = _tan_header(18, crval1=150.0, crval2=2.0, scale_deg=0.9 / 3600.0)
    wavelengths = (4800.0, 5600.0, 6400.0)
    observations: list[FieldObservation] = []
    for wavelength in wavelengths:
        template = FieldObservation(
            image=torch.zeros((18, 18), dtype=torch.float32),
            header=header,
            wavelength=wavelength,
        )
        image = render_field_observation(
            truth_model,
            template,
            num_spectral_samples=1,
        ).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=header,
                weight=torch.ones_like(image),
                wavelength=wavelength,
            )
        )
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    torch.manual_seed(5421)
    model = make_field_model("sky_lambda_atlas_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=12,
        lr=4.0e-3,
        num_spectral_samples=1,
        loss_kind="charbonnier",
        lr_schedule="cosine",
        grad_clip_norm=1.0,
    )

    assert min(fit.loss_history) < fit.initial_loss


def test_sky_lambda_structured_fields_bandpass_single_wavelength_matches_forward() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
        coord_mode="sphere_xyz",
    )
    header = _tan_header(8, crval1=150.0, crval2=2.0, scale_deg=0.9 / 3600.0)
    template = FieldObservation(
        image=torch.zeros((8, 8), dtype=torch.float32),
        header=header,
        wavelength=5600.0,
    )
    for kind in (
        "spectral_basis_field",
        "sky_lambda_kplanes_field",
        "sky_lambda_feature_field",
        "operator_constrained_feature_field",
        "sky_lambda_feature_atlas_field",
        "sky_lambda_atlas_field",
    ):
        torch.manual_seed(7001)
        model = make_field_model(kind, domain)
        direct = render_field_observation(model, template, num_spectral_samples=1)
        bandpass = render_field_observation(
            model,
            FieldObservation(
                image=template.image,
                header=template.header,
                spectral_min=5600.0,
                spectral_max=5600.0,
                transmission_center=5600.0,
                transmission_sigma=None,
            ),
            num_spectral_samples=1,
        )
        assert torch.allclose(direct, bandpass, atol=1.0e-5, rtol=1.0e-5)


def test_sky_lambda_feature_field_supports_exact_wavelength_fit() -> None:
    torch.manual_seed(5422)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("sky_lambda_feature_field", truth_domain)
    header = _tan_header(18, crval1=150.0, crval2=2.0, scale_deg=0.9 / 3600.0)
    wavelengths = (4800.0, 5600.0, 6400.0)
    observations: list[FieldObservation] = []
    for wavelength in wavelengths:
        template = FieldObservation(
            image=torch.zeros((18, 18), dtype=torch.float32),
            header=header,
            wavelength=wavelength,
        )
        image = render_field_observation(
            truth_model,
            template,
            num_spectral_samples=1,
        ).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=header,
                weight=torch.ones_like(image),
                wavelength=wavelength,
            )
        )
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    torch.manual_seed(5423)
    model = make_field_model("sky_lambda_feature_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=12,
        lr=4.0e-3,
        num_spectral_samples=1,
        loss_kind="charbonnier",
        lr_schedule="cosine",
        grad_clip_norm=1.0,
    )

    assert min(fit.loss_history) < fit.initial_loss


def test_operator_constrained_feature_field_supports_exact_wavelength_fit() -> None:
    torch.manual_seed(5428)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("operator_constrained_feature_field", truth_domain)
    header = _tan_header(18, crval1=150.0, crval2=2.0, scale_deg=0.9 / 3600.0)
    wavelengths = (4800.0, 5600.0, 6400.0)
    observations: list[FieldObservation] = []
    for wavelength in wavelengths:
        template = FieldObservation(
            image=torch.zeros((18, 18), dtype=torch.float32),
            header=header,
            wavelength=wavelength,
        )
        image = render_field_observation(
            truth_model,
            template,
            num_spectral_samples=1,
        ).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=header,
                weight=torch.ones_like(image),
                wavelength=wavelength,
            )
        )
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    torch.manual_seed(5429)
    model = make_field_model("operator_constrained_feature_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=12,
        lr=4.0e-3,
        num_spectral_samples=1,
        loss_kind="mse",
        lr_schedule="cosine",
        grad_clip_norm=1.0,
        spectral_observability_weight=1.0e-3,
        residual_head_penalty_weight=1.0e-5,
    )

    assert min(fit.loss_history) < fit.initial_loss


def test_operator_constrained_feature_field_observability_weights_track_overlap() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=6500.0,
        wavelength_scale=2500.0,
        coord_mode="sphere_xyz",
    )
    model = make_field_model("operator_constrained_feature_field", domain)
    header = _tan_header(8, crval1=150.0, crval2=2.0, scale_deg=0.9 / 3600.0)
    sparse_observations = [
        FieldObservation(
            image=torch.zeros((8, 8), dtype=torch.float32),
            header=header,
            spectral_min=4500.0,
            spectral_max=5000.0,
            transmission_center=4750.0,
            transmission_sigma=120.0,
            band_name="b0",
        ),
        FieldObservation(
            image=torch.zeros((8, 8), dtype=torch.float32),
            header=header,
            spectral_min=8000.0,
            spectral_max=8500.0,
            transmission_center=8250.0,
            transmission_sigma=120.0,
            band_name="b1",
        ),
    ]
    dense_observations = [
        FieldObservation(
            image=torch.zeros((8, 8), dtype=torch.float32),
            header=header,
            spectral_min=float(center - 450.0),
            spectral_max=float(center + 450.0),
            transmission_center=float(center),
            transmission_sigma=180.0,
            band_name=f"b{idx}",
        )
        for idx, center in enumerate((4700.0, 5400.0, 6100.0, 6800.0, 7500.0, 8200.0))
    ]
    sparse_weights = model._observability_mode_weights(
        sparse_observations,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    dense_weights = model._observability_mode_weights(
        dense_observations,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    widths = model.spectral_basis_widths.to(dtype=torch.float32)
    sparse_effective_width = float(
        torch.sum(sparse_weights * widths).item() / torch.sum(sparse_weights).item()
    )
    dense_effective_width = float(
        torch.sum(dense_weights * widths).item() / torch.sum(dense_weights).item()
    )

    assert sparse_effective_width < dense_effective_width


def test_fit_field_model_supports_operator_constrained_penalties() -> None:
    torch.manual_seed(5430)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("operator_constrained_feature_field", truth_domain)
    header = _tan_header(12, crval1=150.0, crval2=2.0, scale_deg=0.9 / 3600.0)
    wavelengths = (4700.0, 6200.0, 7800.0)
    observations: list[FieldObservation] = []
    for wavelength in wavelengths:
        template = FieldObservation(
            image=torch.zeros((12, 12), dtype=torch.float32),
            header=header,
            spectral_min=wavelength,
            spectral_max=wavelength,
            transmission_center=wavelength,
            transmission_sigma=None,
            band_name=f"b{int(wavelength)}",
        )
        image = render_field_observation(truth_model, template, num_spectral_samples=1).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=template.header,
                weight=torch.ones_like(image),
                spectral_min=template.spectral_min,
                spectral_max=template.spectral_max,
                transmission_center=template.transmission_center,
                transmission_sigma=template.transmission_sigma,
                band_name=template.band_name,
            )
        )
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    model = make_field_model("operator_constrained_feature_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=8,
        lr=2.0e-3,
        num_spectral_samples=1,
        loss_kind="mse",
        loss_reduction="global",
        observations_per_step=3,
        observation_sampling="band_balanced",
        training_patch_size=8,
        spectral_observability_weight=1.0e-3,
        residual_head_penalty_weight=1.0e-5,
    )
    assert len(fit.loss_history) == 8
    assert all(torch.isfinite(torch.tensor(v)) for v in fit.loss_history)
    assert torch.isfinite(torch.tensor(fit.final_loss))


def test_sky_lambda_feature_atlas_field_supports_exact_wavelength_fit() -> None:
    torch.manual_seed(5422)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("sky_lambda_feature_atlas_field", truth_domain)
    header = _tan_header(18, crval1=150.0, crval2=2.0, scale_deg=0.9 / 3600.0)
    wavelengths = (4800.0, 5600.0, 6400.0)
    observations: list[FieldObservation] = []
    for wavelength in wavelengths:
        template = FieldObservation(
            image=torch.zeros((18, 18), dtype=torch.float32),
            header=header,
            wavelength=wavelength,
        )
        image = render_field_observation(
            truth_model,
            template,
            num_spectral_samples=1,
        ).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=header,
                weight=torch.ones_like(image),
                wavelength=wavelength,
            )
        )
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    torch.manual_seed(5423)
    model = make_field_model("sky_lambda_feature_atlas_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=12,
        lr=4.0e-3,
        num_spectral_samples=1,
        loss_kind="charbonnier",
        lr_schedule="cosine",
        grad_clip_norm=1.0,
    )

    assert min(fit.loss_history) < fit.initial_loss


def test_sky_lambda_feature_field_can_predict_distribution() -> None:
    torch.manual_seed(5424)
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
        coord_mode="sphere_xyz",
    )
    model = make_field_model(
        "sky_lambda_feature_field",
        domain,
        predict_variance=True,
    )
    header = _tan_header(10, crval1=150.0, crval2=2.0, scale_deg=0.9 / 3600.0)
    template = FieldObservation(
        image=torch.zeros((10, 10), dtype=torch.float32),
        header=header,
        wavelength=5600.0,
    )
    ra = torch.full((10, 10), 150.0, dtype=torch.float32)
    dec = torch.full((10, 10), 2.0, dtype=torch.float32)
    mean, logvar = model.forward_distribution(
        ra,
        dec,
        5600.0,
    )
    rendered, rendered_logvar = render_field_observation_distribution(
        model,
        template,
        num_spectral_samples=1,
    )

    assert mean.shape == logvar.shape == (10, 10)
    assert rendered.shape == rendered_logvar.shape == (10, 10)
    assert bool(torch.isfinite(logvar).all())
    assert bool(torch.isfinite(rendered_logvar).all())


def test_sky_lambda_feature_field_supports_feature_scale_curriculum() -> None:
    torch.manual_seed(5425)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("sky_lambda_feature_field", truth_domain)
    header = _tan_header(16, crval1=150.0, crval2=2.0, scale_deg=0.9 / 3600.0)
    wavelengths = (5000.0, 6200.0, 7400.0)
    observations: list[FieldObservation] = []
    for wavelength in wavelengths:
        template = FieldObservation(
            image=torch.zeros((16, 16), dtype=torch.float32),
            header=header,
            wavelength=wavelength,
        )
        image = render_field_observation(
            truth_model,
            template,
            num_spectral_samples=1,
        ).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=header,
                weight=torch.ones_like(image),
                wavelength=wavelength,
            )
        )
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    model = make_field_model("sky_lambda_feature_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=10,
        lr=4.0e-3,
        num_spectral_samples=1,
        loss_kind="charbonnier",
        lr_schedule="cosine",
        grad_clip_norm=1.0,
        curriculum="feature_scale_ramp",
    )

    assert min(fit.loss_history) < fit.initial_loss


def test_sky_lambda_feature_field_supports_band_global_mix_reduction() -> None:
    torch.manual_seed(5426)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("sky_lambda_feature_field", truth_domain)
    header = _tan_header(14, crval1=150.0, crval2=2.0, scale_deg=0.9 / 3600.0)
    wavelengths = (5000.0, 6200.0, 7400.0)
    observations: list[FieldObservation] = []
    for wavelength in wavelengths:
        template = FieldObservation(
            image=torch.zeros((14, 14), dtype=torch.float32),
            header=header,
            wavelength=wavelength,
        )
        image = render_field_observation(
            truth_model,
            template,
            num_spectral_samples=1,
        ).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=header,
                weight=torch.ones_like(image),
                wavelength=wavelength,
            )
        )
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    model = make_field_model("sky_lambda_feature_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=12,
        lr=4.0e-3,
        num_spectral_samples=1,
        loss_kind="mse",
        loss_reduction="band_global_mix",
        lr_schedule="cosine",
        grad_clip_norm=1.0,
    )

    assert fit.final_loss >= 0.0
    assert len(fit.loss_history) == 12


def test_bandpass_conditioned_field_supports_exact_wavelength_fit() -> None:
    torch.manual_seed(5320)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("bandpass_conditioned_field", truth_domain)
    header = _tan_header(18, crval1=150.0, crval2=2.0, scale_deg=0.9 / 3600.0)
    wavelengths = (4800.0, 5600.0, 6400.0)
    observations: list[FieldObservation] = []
    for wavelength in wavelengths:
        template = FieldObservation(
            image=torch.zeros((18, 18), dtype=torch.float32),
            header=header,
            wavelength=wavelength,
        )
        image = render_field_observation(
            truth_model,
            template,
            num_spectral_samples=1,
        ).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=header,
                weight=torch.ones_like(image),
                wavelength=wavelength,
            )
        )
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    torch.manual_seed(5321)
    model = make_field_model("bandpass_conditioned_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=12,
        lr=4.0e-3,
        num_spectral_samples=1,
        loss_kind="charbonnier",
        lr_schedule="cosine",
        grad_clip_norm=1.0,
    )

    assert min(fit.loss_history) < fit.initial_loss


def test_sky_encoder_bandpass_field_supports_broad_band_fit() -> None:
    torch.manual_seed(5322)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=9000.0,
        wavelength_scale=7000.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("sky_encoder_bandpass_field", truth_domain)
    header = _tan_header(18, crval1=150.0, crval2=2.0, scale_deg=0.9 / 3600.0)
    template = FieldObservation(
        image=torch.zeros((18, 18), dtype=torch.float32),
        header=header,
        spectral_min=5400.0,
        spectral_max=7200.0,
        transmission_center=6200.0,
        transmission_sigma=280.0,
        psf_sigma_pix=0.45,
    )
    image = render_field_observation(truth_model, template, num_spectral_samples=7).detach()
    observations = [
        FieldObservation(
            image=image,
            header=header,
            weight=torch.ones_like(image),
            spectral_min=template.spectral_min,
            spectral_max=template.spectral_max,
            transmission_center=template.transmission_center,
            transmission_sigma=template.transmission_sigma,
            psf_sigma_pix=template.psf_sigma_pix,
        )
    ]
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    torch.manual_seed(5323)
    model = make_field_model("sky_encoder_bandpass_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=12,
        lr=3.0e-3,
        num_spectral_samples=7,
        loss_kind="charbonnier",
        loss_reduction="global",
        lr_schedule="cosine",
        grad_clip_norm=1.0,
    )

    assert min(fit.loss_history) < fit.initial_loss


def test_sky_encoder_mlp_field_supports_charbonnier_fit() -> None:
    torch.manual_seed(1357)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("sky_encoder_mlp", truth_domain)
    header = _tan_header(18, crval1=150.0, crval2=2.0, scale_deg=0.95 / 3600.0)
    template = FieldObservation(
        image=torch.zeros((18, 18), dtype=torch.float32),
        header=header,
        spectral_min=5000.0,
        spectral_max=6200.0,
        transmission_center=5600.0,
        transmission_sigma=180.0,
        psf_sigma_pix=0.45,
    )
    image = render_field_observation(truth_model, template, num_spectral_samples=5).detach()
    observations = [
        FieldObservation(
            image=image,
            header=header,
            weight=torch.ones_like(image),
            spectral_min=template.spectral_min,
            spectral_max=template.spectral_max,
            transmission_center=template.transmission_center,
            transmission_sigma=template.transmission_sigma,
            psf_sigma_pix=template.psf_sigma_pix,
        )
    ]
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    model = make_field_model("sky_encoder_mlp", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=12,
        lr=5.0e-3,
        num_spectral_samples=5,
        loss_kind="charbonnier",
    )

    assert min(fit.loss_history) < fit.initial_loss


def test_hashgrid_mlp_field_supports_charbonnier_fit() -> None:
    torch.manual_seed(9753)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("hashgrid_mlp", truth_domain)
    header = _tan_header(16, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    template = FieldObservation(
        image=torch.zeros((16, 16), dtype=torch.float32),
        header=header,
        spectral_min=5000.0,
        spectral_max=6200.0,
        transmission_center=5600.0,
        transmission_sigma=180.0,
        psf_sigma_pix=0.4,
    )
    image = render_field_observation(truth_model, template, num_spectral_samples=5).detach()
    observations = [
        FieldObservation(
            image=image,
            header=header,
            weight=torch.ones_like(image),
            spectral_min=template.spectral_min,
            spectral_max=template.spectral_max,
            transmission_center=template.transmission_center,
            transmission_sigma=template.transmission_sigma,
            psf_sigma_pix=template.psf_sigma_pix,
        )
    ]
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    torch.manual_seed(9754)
    model = make_field_model("hashgrid_mlp", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=10,
        lr=5.0e-3,
        num_spectral_samples=5,
        loss_kind="charbonnier",
    )

    assert min(fit.loss_history) < fit.initial_loss


def test_hashgrid_spectral_field_supports_exact_wavelength_fit() -> None:
    torch.manual_seed(9755)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("hashgrid_spectral_field", truth_domain)
    header = _tan_header(16, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    wavelengths = (5000.0, 5600.0, 6200.0)
    observations: list[FieldObservation] = []
    for wavelength in wavelengths:
        template = FieldObservation(
            image=torch.zeros((16, 16), dtype=torch.float32),
            header=header,
            wavelength=wavelength,
        )
        image = render_field_observation(
            truth_model,
            template,
            num_spectral_samples=1,
        ).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=header,
                weight=torch.ones_like(image),
                wavelength=wavelength,
            )
        )
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    torch.manual_seed(9756)
    model = make_field_model("hashgrid_spectral_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=10,
        lr=4.0e-3,
        num_spectral_samples=1,
        loss_kind="charbonnier",
        lr_schedule="cosine",
        grad_clip_norm=1.0,
    )

    assert min(fit.loss_history) < fit.initial_loss


def test_hashgrid_conditioned_field_supports_exact_wavelength_fit() -> None:
    torch.manual_seed(9757)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("hashgrid_conditioned_field", truth_domain)
    header = _tan_header(16, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    wavelengths = (5000.0, 5600.0, 6200.0)
    observations: list[FieldObservation] = []
    for wavelength in wavelengths:
        template = FieldObservation(
            image=torch.zeros((16, 16), dtype=torch.float32),
            header=header,
            wavelength=wavelength,
        )
        image = render_field_observation(
            truth_model,
            template,
            num_spectral_samples=1,
        ).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=header,
                weight=torch.ones_like(image),
                wavelength=wavelength,
            )
        )
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    torch.manual_seed(9758)
    model = make_field_model("hashgrid_conditioned_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=10,
        lr=4.0e-3,
        num_spectral_samples=1,
        loss_kind="charbonnier",
        lr_schedule="cosine",
        grad_clip_norm=1.0,
    )

    assert min(fit.loss_history) < fit.initial_loss


def test_hashgrid_encoder_rejects_mismatched_log2_sequence() -> None:
    with pytest.raises(ValueError, match="log2_hashmap_size"):
        _HashGridEncoder(2, num_levels=4, log2_hashmap_size=(10, 11, 12))


def test_hashgrid_encoder_coarse_dense_fine_hash() -> None:
    enc = _HashGridEncoder(
        2,
        num_levels=5,
        base_resolution=8,
        finest_resolution=256,
        max_dense_cells=1024,
        dense_coarse=True,
    )
    assert any(enc._is_dense)
    assert not all(enc._is_dense)


def test_hashgrid_resmlp_field_supports_charbonnier_fit() -> None:
    torch.manual_seed(9759)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("hashgrid_resmlp", truth_domain)
    header = _tan_header(16, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    template = FieldObservation(
        image=torch.zeros((16, 16), dtype=torch.float32),
        header=header,
        spectral_min=5000.0,
        spectral_max=6200.0,
        transmission_center=5600.0,
        transmission_sigma=180.0,
        psf_sigma_pix=0.4,
    )
    image = render_field_observation(truth_model, template, num_spectral_samples=5).detach()
    observations = [
        FieldObservation(
            image=image,
            header=header,
            weight=torch.ones_like(image),
            spectral_min=template.spectral_min,
            spectral_max=template.spectral_max,
            transmission_center=template.transmission_center,
            transmission_sigma=template.transmission_sigma,
            psf_sigma_pix=template.psf_sigma_pix,
        )
    ]
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    torch.manual_seed(9760)
    model = make_field_model("hashgrid_resmlp", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=12,
        lr=4.0e-3,
        num_spectral_samples=5,
        loss_kind="charbonnier",
    )

    assert min(fit.loss_history) < fit.initial_loss


def test_sky_encoder_resmlp_field_supports_charbonnier_fit() -> None:
    torch.manual_seed(24601)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("sky_encoder_resmlp", truth_domain)
    header = _tan_header(16, crval1=150.0, crval2=2.0, scale_deg=0.95 / 3600.0)
    template = FieldObservation(
        image=torch.zeros((16, 16), dtype=torch.float32),
        header=header,
        spectral_min=5000.0,
        spectral_max=6200.0,
        transmission_center=5600.0,
        transmission_sigma=180.0,
        psf_sigma_pix=0.45,
    )
    image = render_field_observation(truth_model, template, num_spectral_samples=5).detach()
    observations = [
        FieldObservation(
            image=image,
            header=header,
            weight=torch.ones_like(image),
            spectral_min=template.spectral_min,
            spectral_max=template.spectral_max,
            transmission_center=template.transmission_center,
            transmission_sigma=template.transmission_sigma,
            psf_sigma_pix=template.psf_sigma_pix,
        )
    ]
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    torch.manual_seed(24602)
    model = make_field_model("sky_encoder_resmlp", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=10,
        lr=3.0e-3,
        num_spectral_samples=5,
        loss_kind="charbonnier",
        lr_schedule="cosine",
        grad_clip_norm=0.75,
    )

    assert min(fit.loss_history) < fit.initial_loss


def test_fit_field_model_supports_exact_wavelength_observations_with_cosine_schedule() -> None:
    torch.manual_seed(8642)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("sky_encoder_mlp", truth_domain)
    header = _tan_header(16, crval1=150.0, crval2=2.0, scale_deg=0.95 / 3600.0)
    wavelengths = (4800.0, 5600.0, 6400.0)
    observations: list[FieldObservation] = []
    for wavelength in wavelengths:
        template = FieldObservation(
            image=torch.zeros((16, 16), dtype=torch.float32),
            header=header,
            wavelength=wavelength,
        )
        image = render_field_observation(
            truth_model,
            template,
            num_spectral_samples=1,
        ).detach()
        observations.append(
            FieldObservation(
                image=image,
                header=header,
                weight=torch.ones_like(image),
                wavelength=wavelength,
            )
        )
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    model = make_field_model("sky_encoder_mlp", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=12,
        lr=4.0e-3,
        num_spectral_samples=1,
        loss_kind="charbonnier",
        loss_reduction="global",
        lr_schedule="cosine",
        grad_clip_norm=1.0,
        curriculum="spectral_frequency_ramp",
    )

    assert fit_domain.coord_mode == "sphere_xyz"
    assert all(math.isfinite(loss) for loss in fit.loss_history)
    assert fit.final_loss < 5.0e-3


def test_bandpass_quadrature_trapezoid_closer_than_uniform_for_quadratic() -> None:
    obs = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=_tan_header(4, crval1=0.0, crval2=0.0, scale_deg=1.0 / 3600.0),
        spectral_min=0.0,
        spectral_max=1.0,
    )
    device = torch.device("cpu")
    samples, weights = _bandpass_quadrature(
        obs, num_spectral_samples=6, device=device
    )
    lam2 = torch.square(samples)
    trap_est = float(torch.sum(weights * lam2).item())
    uniform_est = float(torch.mean(lam2).item())
    truth = 1.0 / 3.0
    assert abs(trap_est - truth) < abs(uniform_est - truth)


def test_estimate_field_domain_uses_pc_cdelt_when_cd_absent() -> None:
    scale = 0.5 / 3600.0
    header_pc = {
        "NAXIS1": 100,
        "NAXIS2": 100,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 50.0,
        "CRPIX2": 50.0,
        "CRVAL1": 10.0,
        "CRVAL2": -5.0,
        "CDELT1": -scale,
        "CDELT2": scale,
        "PC1_1": 1.0,
        "PC1_2": 0.0,
        "PC2_1": 0.0,
        "PC2_2": 1.0,
    }
    header_cd = {
        **header_pc,
        "CD1_1": -scale,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": scale,
    }
    del header_cd["CDELT1"]
    del header_cd["CDELT2"]
    del header_cd["PC1_1"]
    del header_cd["PC1_2"]
    del header_cd["PC2_1"]
    del header_cd["PC2_2"]
    obs_pc = FieldObservation(
        image=torch.zeros((100, 100), dtype=torch.float32),
        header=header_pc,
        wavelength=5000.0,
    )
    obs_cd = FieldObservation(
        image=torch.zeros((100, 100), dtype=torch.float32),
        header=header_cd,
        wavelength=5000.0,
    )
    d_pc = estimate_field_domain([obs_pc])
    d_cd = estimate_field_domain([obs_cd])
    assert math.isclose(
        d_pc.spatial_scale_deg,
        d_cd.spatial_scale_deg,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )


def test_render_field_cube_spectral_chunk_matches_per_plane_loop() -> None:
    torch.manual_seed(0)
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
    )
    model = SineImplicitField(domain, hidden_dim=16, hidden_layers=2)
    header = _tan_header(12, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    axis = SpectralAxis(
        values=torch.tensor([4800.0, 5200.0, 6000.0], dtype=torch.float64),
        unit="Angstrom",
        type="WAVE",
    )
    c1 = render_field_cube(
        model, target_header=header, spectral_axis=axis, spectral_chunk=1
    )
    c_all = render_field_cube(
        model, target_header=header, spectral_axis=axis, spectral_chunk=0
    )
    assert torch.allclose(c1, c_all, rtol=0.0, atol=1.0e-5)
