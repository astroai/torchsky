from __future__ import annotations
import math
import pytest
import torch
from torchsky import WCS
from torchsky.map.field import (
    FieldComponentDecomposition,
    FieldDomain,
    FieldObservation,
    ForwardContext,
    SineImplicitField,
    estimate_field_domain,
    render_field_cube,
    render_field_observation,
)
from torchsky.spectral import SpectralAxis
from torchsky.map.field.bandpass import bandpass_quadrature
from map_field_support import (
    _ConstantField,
    _tan_header,
)


def test_bandpass_quadrature_trapezoid_closer_than_uniform_for_quadratic() -> None:
    obs = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=_tan_header(4, crval1=0.0, crval2=0.0, scale_deg=1.0 / 3600.0),
        spectral_min=0.0,
        spectral_max=1.0,
    )
    device = torch.device("cpu")
    samples, weights = bandpass_quadrature(obs, num_spectral_samples=6, device=device)
    lam2 = torch.square(samples)
    trap_est = float(torch.sum(weights * lam2).item())
    uniform_est = float(torch.mean(lam2).item())
    truth = 1.0 / 3.0
    assert abs(trap_est - truth) < abs(uniform_est - truth)


def test_field_observation_rejects_partial_transmission_table() -> None:
    with pytest.raises(ValueError, match="both be set"):
        FieldObservation(
            image=torch.zeros((2, 2), dtype=torch.float32),
            header=_tan_header(2, crval1=0.0, crval2=0.0, scale_deg=1.0 / 3600.0),
            spectral_min=1.0,
            spectral_max=2.0,
            transmission_wavelength=torch.tensor([1.0, 2.0], dtype=torch.float32),
            transmission_throughput=None,
        )


def test_field_observation_rejects_zero_mass_transmission_table() -> None:
    with pytest.raises(ValueError, match="positive total throughput"):
        FieldObservation(
            image=torch.zeros((2, 2), dtype=torch.float32),
            header=_tan_header(2, crval1=0.0, crval2=0.0, scale_deg=1.0 / 3600.0),
            spectral_min=1.0,
            spectral_max=2.0,
            transmission_wavelength=torch.tensor([1.0, 2.0], dtype=torch.float32),
            transmission_throughput=torch.zeros(2, dtype=torch.float32),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tabulated_bandpass_quadrature_runs_on_cuda() -> None:
    device = torch.device("cuda")
    wl_tab = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32, device="cpu")
    tt_tab = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device="cpu")
    obs = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=_tan_header(4, crval1=0.0, crval2=0.0, scale_deg=1.0 / 3600.0),
        spectral_min=0.0,
        spectral_max=1.0,
        transmission_wavelength=wl_tab,
        transmission_throughput=tt_tab,
    )
    samples, weights = bandpass_quadrature(obs, num_spectral_samples=128, device=device)
    assert samples.device.type == device.type
    assert weights.device.type == device.type
    assert torch.isfinite(samples).all() and torch.isfinite(weights).all()


def test_bandpass_quadrature_tabulated_triangle_weighted_mean() -> None:
    """Tabulated T(λ) peak 1 at 0.5, 0 at endpoints ⇒ ∫λT dλ / ∫T dλ = 0.5."""
    wl_tab = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    tt_tab = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    obs = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=_tan_header(4, crval1=0.0, crval2=0.0, scale_deg=1.0 / 3600.0),
        spectral_min=0.0,
        spectral_max=1.0,
        transmission_center=0.5,
        transmission_sigma=100.0,
        transmission_wavelength=wl_tab,
        transmission_throughput=tt_tab,
    )
    device = torch.device("cpu")
    samples, weights = bandpass_quadrature(
        obs, num_spectral_samples=2001, device=device
    )
    est = float(torch.sum(weights * samples).item())
    assert abs(est - 0.5) < 2.0e-3


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


def test_estimate_field_domain_uses_actual_shifted_footprints() -> None:
    scale = 1.0 / 3600.0
    headers = [
        _tan_header(20, crval1=150.0, crval2=2.0, scale_deg=scale),
        _tan_header(20, crval1=150.02, crval2=2.0, scale_deg=scale),
    ]
    observations = [
        FieldObservation(
            image=torch.zeros((20, 20), dtype=torch.float32),
            header=header,
            wavelength=5000.0,
        )
        for header in headers
    ]

    domain = estimate_field_domain(observations)
    model = SineImplicitField(domain, hidden_dim=8, hidden_layers=1)
    for observation in observations:
        x = torch.tensor([-0.5, 19.5, 19.5, -0.5], dtype=torch.float64)
        y = torch.tensor([-0.5, -0.5, 19.5, 19.5], dtype=torch.float64)
        ra, dec = WCS(observation.header).pixel_to_world(x, y)
        encoded = model._spatial_coords(ra, dec, 5000.0)
        assert float(torch.max(torch.abs(encoded))) <= 1.0
    assert abs(domain.center_ra_deg - 150.01) < 1.0e-3


def test_estimate_field_domain_is_ra_wrap_safe() -> None:
    scale = 1.0 / 3600.0
    observations = [
        FieldObservation(
            image=torch.zeros((20, 20), dtype=torch.float32),
            header=_tan_header(
                20,
                crval1=ra,
                crval2=-10.0,
                scale_deg=scale,
            ),
            wavelength=5000.0,
        )
        for ra in (359.99, 0.01)
    ]

    domain = estimate_field_domain(observations)

    assert min(domain.center_ra_deg, 360.0 - domain.center_ra_deg) < 1.0e-3
    assert domain.spatial_scale_deg < 0.03


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


def test_render_field_observation_rejects_non_positive_spatial_tile() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
    )
    model = SineImplicitField(domain, hidden_dim=8, hidden_layers=1)
    obs = FieldObservation(
        image=torch.zeros((8, 8), dtype=torch.float32),
        header=_tan_header(8, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        wavelength=5600.0,
    )
    with pytest.raises(ValueError, match="spatial_tile_hw entries must be positive"):
        render_field_observation(model, obs, spatial_tile_hw=(0, 4))


def test_render_field_cube_rejects_non_positive_spatial_tile() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
    )
    model = SineImplicitField(domain, hidden_dim=8, hidden_layers=1)
    axis = SpectralAxis(
        values=torch.tensor([5600.0], dtype=torch.float64),
        unit="Angstrom",
        type="WAVE",
    )
    with pytest.raises(ValueError, match="spatial_tile_hw entries must be positive"):
        render_field_cube(
            model,
            target_header=_tan_header(
                8, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0
            ),
            spectral_axis=axis,
            spatial_tile_hw=(4, 0),
        )


def test_decomposition_diagnostics_accept_truth_and_anchor_on_other_device(
    compute_device,
) -> None:
    """A device-moved fitted model is compared with a CPU truth/anchor model."""
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    header = _tan_header(4, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)

    def _model(value: float) -> FieldComponentDecomposition:
        return FieldComponentDecomposition(
            [_ConstantField(domain, value), _ConstantField(domain, 1.0)],
            component_roles=("shared", "survey"),
            surveys=("A",),
            instruments=("cam1",),
        )

    estimate = _model(2.0).to(compute_device)
    truth = _model(3.0)
    observation = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=header,
        spectral_min=5600.0,
        spectral_max=5600.0,
        transmission_center=5600.0,
        transmission_sigma=None,
    )
    diagnostics = estimate.decomposition_diagnostics(
        [observation],
        truth,
        observation_forward_contexts=[ForwardContext(survey="A", instrument="cam1")],
        num_spectral_samples=1,
        calibration_anchor_model=truth,
        calibration_anchor_roles="shared",
    )
    assert diagnostics.calibration_transfer_stability is not None
    assert math.isfinite(float(diagnostics.calibration_transfer_stability))


def test_fourier_mlp_features_build_on_device(compute_device) -> None:
    """Fourier frequency scales must be built with ops available on every backend."""
    from torchsky.map.field.models_implicit import FourierMLPField

    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    field = FourierMLPField(
        domain, hidden_dim=8, hidden_layers=1, num_frequencies=3, max_frequency=8.0
    ).to(compute_device)
    coords = torch.linspace(-1.0, 1.0, 12).reshape(4, 3)
    features = field._fourier_features(coords.to(compute_device))
    reference = field.cpu()._fourier_features(coords)
    assert features.device.type == compute_device.type
    torch.testing.assert_close(features.cpu(), reference, rtol=1e-5, atol=1e-5)
