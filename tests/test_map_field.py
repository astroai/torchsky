from __future__ import annotations

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

    assert min(fit.loss_history) < fit.initial_loss


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

    assert min(fit.loss_history) < fit.initial_loss


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
    assert min(fit.loss_history) < fit.initial_loss
