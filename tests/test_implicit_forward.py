from __future__ import annotations

import pytest
import torch
from torch import nn

from torchsky.map import (
    FieldDomain,
    FieldObservation,
    ForwardContext,
    KinematicState,
    SineImplicitField,
    SurveyDataTerm,
    fit_field_model,
    multi_survey_charbonnier_loss,
    predict_observation_mean,
    render_field_observation,
    scale_observation_for_rest_frame_scene,
)


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


def test_scale_observation_divides_spectral_metadata_by_one_plus_z() -> None:
    header = _tan_header(8, crval1=10.0, crval2=20.0, scale_deg=1.0 / 3600.0)
    obs = FieldObservation(
        image=torch.zeros((8, 8)),
        header=header,
        spectral_min=4000.0,
        spectral_max=5000.0,
        transmission_center=4500.0,
        transmission_sigma=100.0,
        wavelength=4800.0,
    )
    scaled = scale_observation_for_rest_frame_scene(obs, one_plus_z=2.0)
    assert scaled.spectral_min == pytest.approx(2000.0)
    assert scaled.spectral_max == pytest.approx(2500.0)
    assert scaled.transmission_center == pytest.approx(2250.0)
    assert scaled.transmission_sigma == pytest.approx(50.0)
    assert scaled.wavelength == pytest.approx(2400.0)


def test_scale_observation_scales_tabulated_wavelength_axis() -> None:
    header = _tan_header(8, crval1=10.0, crval2=20.0, scale_deg=1.0 / 3600.0)
    wl = torch.tensor([4000.0, 5000.0], dtype=torch.float32)
    tt = torch.tensor([0.5, 1.0], dtype=torch.float32)
    obs = FieldObservation(
        image=torch.zeros((8, 8)),
        header=header,
        spectral_min=4000.0,
        spectral_max=5000.0,
        transmission_wavelength=wl,
        transmission_throughput=tt,
    )
    scaled = scale_observation_for_rest_frame_scene(obs, one_plus_z=2.0)
    assert scaled.transmission_wavelength is not None
    assert torch.allclose(
        scaled.transmission_wavelength, torch.tensor([2000.0, 2500.0])
    )
    assert torch.allclose(scaled.transmission_throughput, tt)


def test_predict_observation_mean_matches_render_without_rest_frame() -> None:
    torch.manual_seed(0)
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    model = SineImplicitField(domain, hidden_dim=16, hidden_layers=2)
    header = _tan_header(12, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    obs = FieldObservation(
        image=torch.zeros((12, 12), dtype=torch.float32),
        header=header,
        spectral_min=4700.0,
        spectral_max=5000.0,
        transmission_center=4820.0,
        transmission_sigma=80.0,
    )
    pred = predict_observation_mean(model, obs, num_spectral_samples=5)
    ref = render_field_observation(model, obs, num_spectral_samples=5)
    assert torch.allclose(pred, ref)


def test_predict_observation_mean_rest_frame_matches_manual_scale() -> None:
    torch.manual_seed(1)
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    model = SineImplicitField(domain, hidden_dim=16, hidden_layers=2)
    header = _tan_header(10, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    obs = FieldObservation(
        image=torch.zeros((10, 10), dtype=torch.float32),
        header=header,
        spectral_min=4700.0,
        spectral_max=5000.0,
        transmission_center=4820.0,
        transmission_sigma=80.0,
    )
    opz = 1.15
    kin = KinematicState(one_plus_z=opz)
    pred = predict_observation_mean(
        model,
        obs,
        kinematics=kin,
        rest_frame_scene=True,
        num_spectral_samples=5,
    )
    manual = render_field_observation(
        model,
        scale_observation_for_rest_frame_scene(obs, one_plus_z=opz),
        num_spectral_samples=5,
    )
    assert torch.allclose(pred, manual)


def test_multi_survey_charbonnier_loss_runs_backward() -> None:
    torch.manual_seed(2)
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
    )
    model = SineImplicitField(domain, hidden_dim=14, hidden_layers=2)
    header = _tan_header(8, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    img = torch.randn((8, 8), dtype=torch.float32) * 0.1
    obs_a = FieldObservation(
        image=img,
        header=header,
        spectral_min=4700.0,
        spectral_max=5000.0,
        transmission_center=4820.0,
        transmission_sigma=80.0,
    )
    obs_b = FieldObservation(
        image=img * 0.5,
        header=header,
        spectral_min=6000.0,
        spectral_max=6400.0,
        transmission_center=6200.0,
        transmission_sigma=100.0,
    )
    terms = (
        SurveyDataTerm("a", (obs_a,), loss_weight=1.0),
        SurveyDataTerm("b", (obs_b,), loss_weight=0.5),
    )
    loss = multi_survey_charbonnier_loss(model, terms, num_spectral_samples=4)
    assert loss.ndim == 0
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_kinematic_state_rejects_invalid_one_plus_z() -> None:
    with pytest.raises(ValueError, match="positive"):
        KinematicState(one_plus_z=0.0).effective_one_plus_z()
    with pytest.raises(ValueError, match="positive"):
        KinematicState(one_plus_z=-1.0).effective_one_plus_z()


def test_survey_data_term_forward_context_length_must_match() -> None:
    header = _tan_header(4, crval1=0.0, crval2=0.0, scale_deg=1.0 / 3600.0)
    o1 = FieldObservation(
        image=torch.zeros((4, 4)),
        header=header,
        spectral_min=4000.0,
        spectral_max=4100.0,
        transmission_center=4050.0,
        transmission_sigma=10.0,
    )
    o2 = FieldObservation(
        image=torch.zeros((4, 4)),
        header=header,
        spectral_min=5000.0,
        spectral_max=5100.0,
        transmission_center=5050.0,
        transmission_sigma=10.0,
    )
    with pytest.raises(ValueError, match="forward_context"):
        SurveyDataTerm(
            "t",
            (o1, o2),
            forward_context_per_observation=(ForwardContext(survey="a"),),
        )


def test_survey_data_term_kinematics_length_must_match() -> None:
    header = _tan_header(4, crval1=0.0, crval2=0.0, scale_deg=1.0 / 3600.0)
    o1 = FieldObservation(
        image=torch.zeros((4, 4)),
        header=header,
        spectral_min=4000.0,
        spectral_max=4100.0,
        transmission_center=4050.0,
        transmission_sigma=10.0,
    )
    o2 = FieldObservation(
        image=torch.zeros((4, 4)),
        header=header,
        spectral_min=5000.0,
        spectral_max=5100.0,
        transmission_center=5050.0,
        transmission_sigma=10.0,
    )
    with pytest.raises(ValueError, match="length"):
        SurveyDataTerm(
            "t",
            (o1, o2),
            kinematics_per_observation=(KinematicState(1.1),),
        )


class _SurveyScaleWrapper(nn.Module):
    """Test helper: scales output when ``forward_context.survey`` is ``\"B\"``."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(
        self,
        ra_deg: torch.Tensor,
        dec_deg: torch.Tensor,
        wavelength: torch.Tensor | float,
        forward_context: ForwardContext | None = None,
    ) -> torch.Tensor:
        y = self.inner(ra_deg, dec_deg, wavelength)
        if forward_context is not None and forward_context.survey == "B":
            return y * 2.0
        return y


def test_render_field_observation_passes_forward_context_when_supported() -> None:
    torch.manual_seed(0)
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    base = SineImplicitField(domain, hidden_dim=16, hidden_layers=2)
    model = _SurveyScaleWrapper(base)
    header = _tan_header(8, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    obs = FieldObservation(
        image=torch.zeros((8, 8), dtype=torch.float32),
        header=header,
        spectral_min=4700.0,
        spectral_max=5000.0,
        transmission_center=4820.0,
        transmission_sigma=80.0,
    )
    p0 = render_field_observation(model, obs, num_spectral_samples=4)
    p_a = render_field_observation(
        model,
        obs,
        num_spectral_samples=4,
        forward_context=ForwardContext(survey="A"),
    )
    p_b = render_field_observation(
        model,
        obs,
        num_spectral_samples=4,
        forward_context=ForwardContext(survey="B"),
    )
    assert torch.allclose(p0, p_a)
    assert torch.allclose(p_b, p0 * 2.0)


def test_render_field_observation_use_func_vmap_rejects_forward_context() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    model = SineImplicitField(domain, hidden_dim=8, hidden_layers=1)
    header = _tan_header(4, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    obs = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=header,
        spectral_min=4700.0,
        spectral_max=5000.0,
        transmission_center=4820.0,
        transmission_sigma=80.0,
    )
    with pytest.raises(ValueError, match="use_func_vmap"):
        render_field_observation(
            model,
            obs,
            num_spectral_samples=3,
            use_func_vmap=True,
            forward_context=ForwardContext(survey="x"),
        )


def test_fit_field_model_threads_observation_forward_contexts() -> None:
    torch.manual_seed(3)
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
    )
    model = _SurveyScaleWrapper(SineImplicitField(domain, hidden_dim=20, hidden_layers=2))
    header = _tan_header(12, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    target = torch.ones((12, 12), dtype=torch.float32)
    obs_a = FieldObservation(
        image=target,
        header=header,
        spectral_min=4700.0,
        spectral_max=5000.0,
        transmission_center=4820.0,
        transmission_sigma=80.0,
    )
    obs_b = FieldObservation(
        image=target * 2.0,
        header=header,
        spectral_min=6000.0,
        spectral_max=6400.0,
        transmission_center=6200.0,
        transmission_sigma=100.0,
    )
    ctxs = (ForwardContext(survey="A"), ForwardContext(survey="B"))
    fit_field_model(
        (obs_a, obs_b),
        model,
        steps=30,
        lr=1.0e-2,
        observations_per_step=2,
        observation_forward_contexts=ctxs,
    )
    pred_a = render_field_observation(
        model.inner,
        obs_a,
        num_spectral_samples=5,
    )
    pred_b_wrapped = render_field_observation(
        model,
        obs_b,
        num_spectral_samples=5,
        forward_context=ctxs[1],
    )
    assert torch.allclose(pred_a, target, atol=0.15, rtol=0.15)
    assert torch.allclose(pred_b_wrapped, obs_b.image, atol=0.2, rtol=0.2)


def test_multi_survey_requires_at_least_one_observation() -> None:
    model = SineImplicitField(
        FieldDomain(
            center_ra_deg=0.0,
            center_dec_deg=0.0,
            spatial_scale_deg=1.0,
            wavelength_center=5000.0,
            wavelength_scale=500.0,
        ),
        hidden_dim=8,
        hidden_layers=1,
    )
    with pytest.raises(ValueError, match="at least one"):
        multi_survey_charbonnier_loss(model, (SurveyDataTerm("empty", ()),))
