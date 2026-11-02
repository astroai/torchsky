from __future__ import annotations
import copy
import math
import pytest
import torch
from torchsky.map.field import (
    FieldDomain,
    FieldObservation,
    SineImplicitField,
    estimate_field_domain,
    fit_field_model,
    make_field_model,
    render_field_cube,
    render_field_observation,
)
from torchsky.spectral import SpectralAxis
from torchsky.map.field.training import _noise_pretrain_field_model
from map_field_support import (
    _tan_header,
)


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
    obs_a = render_field_observation(
        truth_model, template_a, num_spectral_samples=7
    ).detach()
    obs_b = render_field_observation(
        truth_model, template_b, num_spectral_samples=7
    ).detach()
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
        image = render_field_observation(
            truth_model, template, num_spectral_samples=1
        ).detach()
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
    image = render_field_observation(
        truth_model, template, num_spectral_samples=1
    ).detach()
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


@pytest.mark.needs_float64
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
            psf_sigma_pix=0.5,
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
                psf_sigma_pix=template.psf_sigma_pix,
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
        spatial_sampling="gauss_legendre_2x2",
    )
    assert fit.final_loss < fit.initial_loss


@pytest.mark.needs_float64
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
    image = render_field_observation(
        truth_model, template, num_spectral_samples=5
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
    truth_images = []
    for template in templates:
        image = render_field_observation(
            truth_model,
            template,
            num_spectral_samples=7,
        ).detach()
        truth_images.append(image)
    # The untrained truth field renders at ~1e-3; at that scale the initial loss
    # (~3e-6) is already at the optimizer's step noise and "training improves the
    # fit" is ill-posed on any device. Normalise to unit amplitude.
    scale = torch.cat([im.reshape(-1) for im in truth_images]).abs().max()
    for template, image in zip(templates, truth_images, strict=True):
        image = image / scale
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
    initial_model = copy.deepcopy(model)

    def _deterministic_loss(field: torch.nn.Module) -> float:
        # The reported fit losses are evaluated on different Monte-Carlo bandpass
        # draws (device RNG streams differ), so at ~1e-6 they are noise-limited.
        # Compare the trained and initial models on the fixed quadrature instead.
        total = 0.0
        for obs in observations:
            rendered = render_field_observation(field, obs, num_spectral_samples=7)
            total += float(
                torch.mean((rendered - obs.image.to(rendered.device)) ** 2).detach()
            )
        return total

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
    assert _deterministic_loss(fit.model) < _deterministic_loss(initial_model)
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


def test_fit_field_model_mixture_mc_falls_back_for_uncovered_bandpass() -> None:
    torch.manual_seed(8643)
    header = _tan_header(4, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    observations = [
        FieldObservation(
            image=torch.ones((4, 4), dtype=torch.float32),
            header=header,
            weight=torch.ones((4, 4), dtype=torch.float32),
            spectral_min=lower,
            spectral_max=upper,
            band_name=name,
        )
        for lower, upper, name in (
            (4_000.0, 4_100.0, "blue"),
            (8_000.0, 8_100.0, "red"),
        )
    ]
    model = SineImplicitField(
        estimate_field_domain(observations), hidden_dim=8, hidden_layers=1
    )

    fit = fit_field_model(
        observations,
        model,
        steps=1,
        lr=1.0e-3,
        num_spectral_samples=1,
        training_spectral_sampling="mixture_mc",
    )

    assert math.isfinite(fit.final_loss)


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
    image = render_field_observation(
        truth_model, template, num_spectral_samples=5
    ).detach()
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
