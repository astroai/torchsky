from __future__ import annotations
import math
import pytest
import torch
from torchsky.map.field import (
    FieldDomain,
    FieldObservation,
    estimate_field_domain,
    fit_field_model,
    make_field_model,
    render_field_observation,
    render_field_observation_distribution,
)
from torchsky.map.field.hashgrid import HashGridEncoder
from map_field_support import (
    _tan_header,
)


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


@pytest.mark.needs_float64
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
    image = render_field_observation(
        truth_model, template, num_spectral_samples=7
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


@pytest.mark.needs_float64
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
        HashGridEncoder(2, num_levels=4, log2_hashmap_size=(10, 11, 12))


def test_hashgrid_encoder_coarse_dense_fine_hash() -> None:
    enc = HashGridEncoder(
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


@pytest.mark.needs_float64
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


@pytest.mark.needs_float64
def test_fit_field_model_supports_exact_wavelength_observations_with_cosine_schedule() -> (
    None
):
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
