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
)
from map_field_support import (
    _tan_header,
)


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


@pytest.mark.needs_float64
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


@pytest.mark.needs_float64
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


def test_sky_lambda_structured_fields_bandpass_single_wavelength_matches_forward() -> (
    None
):
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


def test_operator_constrained_feature_field_observability_weights_track_overlap() -> (
    None
):
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
    model_kwargs = {
        "base_resolution": 4,
        "num_scales": 2,
        "features_per_plane": 2,
        "scale_hidden_dim": 8,
        "scale_num_blocks": 1,
        "decoder_hidden_dim": 8,
        "decoder_num_blocks": 1,
        "refine_hidden_dim": 8,
        "bandpass_hidden_dim": 8,
    }
    truth_model = make_field_model(
        "sky_lambda_feature_atlas_field", truth_domain, **model_kwargs
    )
    header = _tan_header(8, crval1=150.0, crval2=2.0, scale_deg=0.9 / 3600.0)
    wavelengths = (5000.0, 6200.0)
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
    model = make_field_model(
        "sky_lambda_feature_atlas_field", fit_domain, **model_kwargs
    )
    fit = fit_field_model(
        observations,
        model,
        steps=6,
        lr=4.0e-3,
        num_spectral_samples=1,
        loss_kind="charbonnier",
        lr_schedule="cosine",
        grad_clip_norm=1.0,
    )

    assert min(fit.loss_history) < fit.initial_loss
