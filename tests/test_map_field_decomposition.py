from __future__ import annotations
import math
import pytest
import torch
from conftest import assert_close_on_device
from torchsky import WCS
from torchsky.map.field import (
    FieldCentroidAnchorTerm,
    FieldComponentAnchorTerm,
    FieldComponentDecomposition,
    FieldDomain,
    FieldObservation,
    ForwardContext,
    field_component_anchor_loss,
    field_component_centroid_anchor_loss,
    fit_field_component_decomposition,
    make_field_component_decomposition,
    render_field_observation,
)
from torchsky.map.solvers import ComponentDecompositionDiagnostics
from torchsky.map.field.core import _BaseFieldModel
from map_field_support import (
    _ConstantField,
    _GaussianSpotField,
    _tan_header,
)


class _AffineSpectralField(_BaseFieldModel):
    def __init__(self, domain: FieldDomain, offset: float, slope: float) -> None:
        super().__init__(domain)
        self.offset = torch.nn.Parameter(
            torch.tensor(float(offset), dtype=torch.float32)
        )
        self.slope = torch.nn.Parameter(torch.tensor(float(slope), dtype=torch.float32))

    def forward(
        self,
        ra_deg: torch.Tensor,
        dec_deg: torch.Tensor,
        wavelength_angstrom: torch.Tensor | float,
        *,
        forward_context: ForwardContext | None = None,
    ) -> torch.Tensor:
        del dec_deg, forward_context
        wavelength = torch.as_tensor(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=ra_deg.dtype,
        )
        if wavelength.shape != ra_deg.shape:
            wavelength = torch.broadcast_to(wavelength, ra_deg.shape)
        return (
            torch.zeros_like(ra_deg)
            + self.offset
            + self.slope
            * (wavelength - float(self.domain.wavelength_center))
            / float(self.domain.wavelength_scale)
        )


class _CoordinateField(_BaseFieldModel):
    def forward(
        self,
        ra_deg: torch.Tensor,
        dec_deg: torch.Tensor,
        wavelength_angstrom: torch.Tensor | float,
        *,
        forward_context: ForwardContext | None = None,
    ) -> torch.Tensor:
        del wavelength_angstrom, forward_context
        return (
            100.0 * (ra_deg - float(self.domain.center_ra_deg))
            + 200.0 * (dec_deg - float(self.domain.center_dec_deg))
        ).to(dtype=ra_deg.dtype)


def test_field_component_decomposition_respects_contextual_roles() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    model = FieldComponentDecomposition(
        [
            _ConstantField(domain, 1.0),
            _ConstantField(domain, 2.0),
            _ConstantField(domain, 3.0),
        ],
        component_roles=("shared", "survey", "instrument"),
        surveys=("A",),
        instruments=("cam1",),
    )
    with torch.no_grad():
        model.mixer.shared_weights[:] = torch.tensor([1.0, 0.0, 0.0])
        model.mixer.survey_weights[1] = torch.tensor([0.0, 0.5, 0.0])
        model.mixer.instrument_weights[1] = torch.tensor([0.0, 0.0, 2.0])

    ra = torch.zeros((2, 2), dtype=torch.float32)
    dec = torch.zeros((2, 2), dtype=torch.float32)
    context = ForwardContext(survey="A", instrument="cam1")

    full = model(ra, dec, 5600.0, forward_context=context)
    private = model.component_prediction(
        ra,
        dec,
        5600.0,
        roles=("survey", "instrument"),
        forward_context=context,
    )
    parts = model.component_prediction(
        ra,
        dec,
        5600.0,
        roles=("survey", "instrument"),
        forward_context=context,
        aggregate=False,
    )

    assert_close_on_device(full, torch.full((2, 2), 8.0, dtype=torch.float32))
    assert_close_on_device(private, torch.full((2, 2), 7.0, dtype=torch.float32))
    assert_close_on_device(parts[0], torch.full((2, 2), 1.0, dtype=torch.float32))
    assert_close_on_device(parts[1], torch.full((2, 2), 6.0, dtype=torch.float32))


def test_field_component_render_ignores_observation_backgrounds() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    model = FieldComponentDecomposition(
        [_ConstantField(domain, 2.5)],
        component_roles=("shared",),
    )
    with torch.no_grad():
        model.mixer.shared_weights[:] = torch.tensor([1.0])
    observation = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=_tan_header(4, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        spectral_min=5600.0,
        spectral_max=5600.0,
        transmission_center=5600.0,
        transmission_sigma=None,
        sky_background=torch.full((4, 4), 7.0, dtype=torch.float32),
        instrument_background=torch.full((4, 4), 11.0, dtype=torch.float32),
        detector_background=torch.full((4, 4), 13.0, dtype=torch.float32),
        background_offset=17.0,
    )

    rendered = model.render_component_observation(
        observation,
        roles="shared",
        num_spectral_samples=1,
    )

    assert_close_on_device(rendered, torch.full((4, 4), 2.5, dtype=torch.float32))


def test_field_component_fix_scale_gauge_preserves_predictions() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    model = FieldComponentDecomposition(
        [
            _ConstantField(domain, 2.0),
            _ConstantField(domain, 3.0),
        ],
        component_roles=("shared", "survey"),
        surveys=("A",),
    )
    with torch.no_grad():
        model.component_gains[:] = torch.tensor([4.0, -2.0])
        model.mixer.shared_weights[:] = torch.tensor([0.5, 0.0])
        model.mixer.survey_weights[1] = torch.tensor([0.0, 1.5])
    ra = torch.zeros((2, 2), dtype=torch.float32)
    dec = torch.zeros((2, 2), dtype=torch.float32)
    context = ForwardContext(survey="A")

    before = model(ra, dec, 5600.0, forward_context=context)
    gains = model.fix_component_scale_gauge_()
    after = model(ra, dec, 5600.0, forward_context=context)

    assert_close_on_device(before, after)
    assert_close_on_device(gains, torch.ones_like(gains))
    assert_close_on_device(
        model.mixer.shared_weights,
        torch.tensor([2.0, 0.0], dtype=torch.float32),
    )
    assert_close_on_device(
        model.mixer.survey_weights[1],
        torch.tensor([0.0, -3.0], dtype=torch.float32),
    )


def test_field_component_anchor_loss_matches_shared_target() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    model = FieldComponentDecomposition(
        [
            _ConstantField(domain, 3.0),
            _ConstantField(domain, 1.0),
        ],
        component_roles=("shared", "survey"),
        surveys=("A",),
    )
    with torch.no_grad():
        model.mixer.shared_weights[:] = torch.tensor([1.0, 0.0])
        model.mixer.survey_weights[1] = torch.tensor([0.0, 1.0])
    observation = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=_tan_header(4, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        spectral_min=5600.0,
        spectral_max=5600.0,
        transmission_center=5600.0,
        transmission_sigma=None,
    )
    matched_term = FieldComponentAnchorTerm(
        observation=observation,
        target_image=torch.full((4, 4), 3.0, dtype=torch.float32),
        roles="shared",
        forward_context=ForwardContext(survey="A"),
    )
    shifted_term = FieldComponentAnchorTerm(
        observation=observation,
        target_image=torch.full((4, 4), 2.0, dtype=torch.float32),
        roles="shared",
        forward_context=ForwardContext(survey="A"),
        loss_weight=2.0,
    )

    matched_loss = field_component_anchor_loss(
        model,
        [matched_term],
        num_spectral_samples=1,
    )
    shifted_loss = field_component_anchor_loss(
        model,
        [shifted_term],
        num_spectral_samples=1,
    )

    assert_close_on_device(matched_loss, torch.zeros((), dtype=torch.float32))
    assert_close_on_device(shifted_loss, torch.tensor(2.0, dtype=torch.float32))


def test_field_component_anchor_loss_supports_sample_spectral_targets() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    model = FieldComponentDecomposition(
        [_AffineSpectralField(domain, 1.0, 2.0)],
        component_roles=("shared",),
    )
    with torch.no_grad():
        model.mixer.shared_weights[:] = torch.tensor([1.0])
    observation = FieldObservation(
        image=torch.zeros((1, 1), dtype=torch.float32),
        header=_tan_header(1, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        spectral_min=5600.0,
        spectral_max=5600.0,
        transmission_center=5600.0,
        transmission_sigma=None,
    )
    sample_term = FieldComponentAnchorTerm(
        observation=observation,
        target_values=torch.tensor([1.6, 0.6], dtype=torch.float32),
        roles="shared",
        sample_ra_deg=torch.tensor([150.0, 150.01], dtype=torch.float32),
        sample_dec_deg=torch.tensor([2.0, 2.01], dtype=torch.float32),
        sample_weight=torch.tensor([1.0, 2.0], dtype=torch.float32),
        override_wavelengths=torch.tensor([5900.0], dtype=torch.float32),
        override_bandpass_weights=torch.tensor([1.0], dtype=torch.float32),
    )

    loss = field_component_anchor_loss(
        model,
        [sample_term],
        num_spectral_samples=1,
    )

    assert_close_on_device(loss, torch.tensor(2.0 / 3.0, dtype=torch.float32))


def test_field_component_anchor_loss_supports_astrometric_pixel_targets() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    model = FieldComponentDecomposition(
        [_CoordinateField(domain)],
        component_roles=("shared",),
    )
    with torch.no_grad():
        model.mixer.shared_weights[:] = torch.tensor([1.0])
    observation = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=_tan_header(4, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        spectral_min=5600.0,
        spectral_max=5600.0,
        transmission_center=5600.0,
        transmission_sigma=None,
    )
    x_pix = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y_pix = torch.tensor([1.0, 2.0], dtype=torch.float32)
    ra_deg, dec_deg = WCS(observation.header).pixel_to_world(
        x_pix.to(dtype=torch.float64),
        y_pix.to(dtype=torch.float64),
    )
    target = 100.0 * (
        ra_deg.to(dtype=torch.float32) - float(domain.center_ra_deg)
    ) + 200.0 * (dec_deg.to(dtype=torch.float32) - float(domain.center_dec_deg))
    sample_term = FieldComponentAnchorTerm(
        observation=observation,
        target_values=target,
        roles="shared",
        sample_x_pix=x_pix,
        sample_y_pix=y_pix,
    )

    loss = field_component_anchor_loss(
        model,
        [sample_term],
        num_spectral_samples=1,
    )

    assert_close_on_device(
        loss, torch.zeros((), dtype=torch.float32), atol=1.0e-6, rtol=0.0
    )


def test_field_component_centroid_anchor_loss_matches_world_target() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    observation = FieldObservation(
        image=torch.zeros((9, 9), dtype=torch.float32),
        header=_tan_header(9, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        spectral_min=5600.0,
        spectral_max=5600.0,
        transmission_center=5600.0,
        transmission_sigma=None,
    )
    target_x = torch.tensor(6.0, dtype=torch.float64)
    target_y = torch.tensor(3.0, dtype=torch.float64)
    target_ra, target_dec = WCS(observation.header).pixel_to_world(target_x, target_y)
    model = FieldComponentDecomposition(
        [
            _GaussianSpotField(
                domain,
                amplitude=3.0,
                center_ra_deg=float(target_ra),
                center_dec_deg=float(target_dec),
                sigma_deg=0.7 / 3600.0,
            )
        ],
        component_roles=("shared",),
    )
    with torch.no_grad():
        model.mixer.shared_weights[:] = torch.tensor([1.0])
    matched_term = FieldCentroidAnchorTerm(
        observation=observation,
        target_ra_deg=float(target_ra),
        target_dec_deg=float(target_dec),
        roles="shared",
        window_sigma_pix=1.5,
    )
    shifted_term = FieldCentroidAnchorTerm(
        observation=observation,
        target_x_pix=2.5,
        target_y_pix=6.0,
        roles="shared",
        window_sigma_pix=1.5,
    )

    matched_loss = field_component_centroid_anchor_loss(
        model,
        [matched_term],
        num_spectral_samples=1,
    )
    shifted_loss = field_component_centroid_anchor_loss(
        model,
        [shifted_term],
        num_spectral_samples=1,
    )

    assert float(matched_loss.item()) < 1.0e-2
    assert float(shifted_loss.item()) > float(matched_loss.item()) + 1.0


def test_field_component_decomposition_diagnostics_measure_exact_leakage() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    truth_model = FieldComponentDecomposition(
        [
            _ConstantField(domain, 4.0),
            _ConstantField(domain, 2.0),
            _ConstantField(domain, 0.0),
        ],
        component_roles=("shared", "survey", "instrument"),
        surveys=("A",),
        instruments=("cam1",),
    )
    estimate_model = FieldComponentDecomposition(
        [
            _ConstantField(domain, 5.0),
            _ConstantField(domain, 3.0),
            _ConstantField(domain, 0.0),
        ],
        component_roles=("shared", "survey", "instrument"),
        surveys=("A",),
        instruments=("cam1",),
    )
    with torch.no_grad():
        truth_model.mixer.shared_weights[:] = torch.tensor([1.0, 0.0, 0.0])
        truth_model.mixer.survey_weights[1] = torch.tensor([0.0, 1.0, 0.0])
        estimate_model.mixer.shared_weights[:] = torch.tensor([1.0, 0.0, 0.0])
        estimate_model.mixer.survey_weights[1] = torch.tensor([0.0, 1.0, 0.0])
    observation = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=_tan_header(4, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        spectral_min=5600.0,
        spectral_max=5600.0,
        transmission_center=5600.0,
        transmission_sigma=None,
    )
    diagnostics = estimate_model.decomposition_diagnostics(
        [observation],
        truth_model,
        observation_forward_contexts=[ForwardContext(survey="A")],
        num_spectral_samples=1,
        calibration_anchor_model=truth_model,
        calibration_anchor_roles="shared",
    )

    assert isinstance(diagnostics, ComponentDecompositionDiagnostics)
    assert_close_on_device(
        torch.tensor(diagnostics.shared_to_private_leakage),
        torch.tensor(0.5),
    )
    assert_close_on_device(
        torch.tensor(diagnostics.private_to_shared_leakage),
        torch.tensor(0.25),
    )
    assert_close_on_device(
        torch.tensor(diagnostics.calibration_transfer_stability),
        torch.tensor(0.25),
    )


def test_make_field_component_decomposition_infers_context_axes() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    model = make_field_component_decomposition(
        component_roles=("shared", "survey", "instrument", "interaction"),
        domain=domain,
        component_models=[
            _ConstantField(domain, 1.0),
            _ConstantField(domain, 1.0),
            _ConstantField(domain, 1.0),
            _ConstantField(domain, 1.0),
        ],
        observation_forward_contexts=[
            ForwardContext(survey="survey_b", instrument="cam2"),
            ForwardContext(survey="survey_a", instrument="cam1"),
            ForwardContext(survey="survey_b", instrument="cam1"),
            None,
        ],
    )

    assert model.mixer.surveys == ("survey_b", "survey_a")
    assert model.mixer.instruments == ("cam2", "cam1")


def test_fit_field_component_decomposition_trains_contextual_split() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    header = _tan_header(4, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    observations = [
        FieldObservation(
            image=torch.full((4, 4), 4.0, dtype=torch.float32),
            header=header,
            spectral_min=5600.0,
            spectral_max=5600.0,
            transmission_center=5600.0,
            transmission_sigma=None,
        ),
        FieldObservation(
            image=torch.full((4, 4), 2.0, dtype=torch.float32),
            header=header,
            spectral_min=5600.0,
            spectral_max=5600.0,
            transmission_center=5600.0,
            transmission_sigma=None,
        ),
    ]
    contexts = [
        ForwardContext(survey="A", instrument="cam1"),
        ForwardContext(survey="B", instrument="cam1"),
    ]

    result = fit_field_component_decomposition(
        observations,
        component_roles=("shared", "survey"),
        component_models=[
            _ConstantField(domain, 1.0),
            _ConstantField(domain, 1.0),
        ],
        observation_forward_contexts=contexts,
        steps=200,
        lr=1.0e-1,
        num_spectral_samples=1,
    )

    assert isinstance(result.model, FieldComponentDecomposition)
    assert result.model.mixer.surveys == ("A", "B")
    pred_a = render_field_observation(
        result.model,
        observations[0],
        num_spectral_samples=1,
        forward_context=contexts[0],
    )
    pred_b = render_field_observation(
        result.model,
        observations[1],
        num_spectral_samples=1,
        forward_context=contexts[1],
    )

    assert_close_on_device(pred_a, observations[0].image, atol=5.0e-2, rtol=0.0)
    assert_close_on_device(pred_b, observations[1].image, atol=5.0e-2, rtol=0.0)
    assert result.final_loss < 5.0e-3


def test_component_anchor_terms_improve_shared_calibration_stability() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    header = _tan_header(4, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    contexts = [
        ForwardContext(survey="A", instrument="cam1"),
        ForwardContext(survey="B", instrument="cam1"),
    ]
    truth_model = FieldComponentDecomposition(
        [
            _ConstantField(domain, 3.0),
            _ConstantField(domain, 1.0),
        ],
        component_roles=("shared", "survey"),
        surveys=("A", "B"),
        instruments=("cam1",),
    )
    with torch.no_grad():
        truth_model.mixer.shared_weights[:] = torch.tensor([1.0, 0.0])
        truth_model.mixer.survey_weights[1] = torch.tensor([0.0, 1.0])
        truth_model.mixer.survey_weights[2] = torch.tensor([0.0, -1.0])
    template = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=header,
        spectral_min=5600.0,
        spectral_max=5600.0,
        transmission_center=5600.0,
        transmission_sigma=None,
    )
    observations = [
        FieldObservation(
            image=render_field_observation(
                truth_model,
                template,
                num_spectral_samples=1,
                forward_context=context,
            ).detach(),
            header=header,
            spectral_min=5600.0,
            spectral_max=5600.0,
            transmission_center=5600.0,
            transmission_sigma=None,
        )
        for context in contexts
    ]

    def _make_init_model() -> FieldComponentDecomposition:
        model = make_field_component_decomposition(
            component_roles=("shared", "survey"),
            domain=domain,
            component_models=[
                _ConstantField(domain, 1.0),
                _ConstantField(domain, 1.0),
            ],
            observation_forward_contexts=contexts,
        )
        with torch.no_grad():
            model.component_gains[:] = torch.tensor([1.8, 0.7])
            model.mixer.shared_weights[:] = torch.tensor([0.1, 0.0])
            model.mixer.survey_weights[1] = torch.tensor([0.0, 2.5])
            model.mixer.survey_weights[2] = torch.tensor([0.0, 0.5])
        return model

    plain = fit_field_component_decomposition(
        observations,
        component_roles=("shared", "survey"),
        model=_make_init_model(),
        observation_forward_contexts=contexts,
        steps=120,
        lr=7.0e-2,
        num_spectral_samples=1,
    )
    anchor_term = FieldComponentAnchorTerm(
        observation=template,
        target_image=torch.full((4, 4), 3.0, dtype=torch.float32),
        roles="shared",
        forward_context=contexts[0],
        loss_weight=1.0,
    )
    anchored = fit_field_component_decomposition(
        observations,
        component_roles=("shared", "survey"),
        model=_make_init_model(),
        observation_forward_contexts=contexts,
        component_anchor_terms=[anchor_term],
        steps=120,
        lr=7.0e-2,
        num_spectral_samples=1,
    )
    plain.model.fix_component_scale_gauge_()
    anchored.model.fix_component_scale_gauge_()
    plain_diag = plain.model.decomposition_diagnostics(
        observations,
        truth_model,
        observation_forward_contexts=contexts,
        num_spectral_samples=1,
        calibration_anchor_model=truth_model,
        calibration_anchor_roles="shared",
    )
    anchored_diag = anchored.model.decomposition_diagnostics(
        observations,
        truth_model,
        observation_forward_contexts=contexts,
        num_spectral_samples=1,
        calibration_anchor_model=truth_model,
        calibration_anchor_roles="shared",
    )

    assert (
        anchored_diag.calibration_transfer_stability
        < plain_diag.calibration_transfer_stability
    )


def test_sample_anchor_terms_improve_shared_spectral_calibration_stability() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    header = _tan_header(4, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    contexts = [
        ForwardContext(survey="A", instrument="cam1"),
        ForwardContext(survey="B", instrument="cam1"),
    ]
    truth_model = FieldComponentDecomposition(
        [
            _AffineSpectralField(domain, 3.0, 1.0),
            _ConstantField(domain, 1.0),
        ],
        component_roles=("shared", "survey"),
        surveys=("A", "B"),
        instruments=("cam1",),
    )
    with torch.no_grad():
        truth_model.mixer.shared_weights[:] = torch.tensor([1.0, 0.0])
        truth_model.mixer.survey_weights[1] = torch.tensor([0.0, 1.0])
        truth_model.mixer.survey_weights[2] = torch.tensor([0.0, -1.0])
    template = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=header,
        spectral_min=5600.0,
        spectral_max=5600.0,
        transmission_center=5600.0,
        transmission_sigma=None,
    )
    observations = [
        FieldObservation(
            image=render_field_observation(
                truth_model,
                template,
                num_spectral_samples=1,
                forward_context=context,
            ).detach(),
            header=header,
            spectral_min=5600.0,
            spectral_max=5600.0,
            transmission_center=5600.0,
            transmission_sigma=None,
        )
        for context in contexts
    ]

    def _make_init_model() -> FieldComponentDecomposition:
        model = make_field_component_decomposition(
            component_roles=("shared", "survey"),
            domain=domain,
            component_models=[
                _AffineSpectralField(domain, 0.5, 0.1),
                _ConstantField(domain, 1.0),
            ],
            observation_forward_contexts=contexts,
        )
        with torch.no_grad():
            model.component_gains[:] = torch.tensor([1.5, 0.8])
            model.mixer.shared_weights[:] = torch.tensor([0.2, 0.0])
            model.mixer.survey_weights[1] = torch.tensor([0.0, 2.2])
            model.mixer.survey_weights[2] = torch.tensor([0.0, 0.4])
        return model

    plain = fit_field_component_decomposition(
        observations,
        component_roles=("shared", "survey"),
        model=_make_init_model(),
        observation_forward_contexts=contexts,
        steps=120,
        lr=7.0e-2,
        num_spectral_samples=1,
    )
    sample_anchor = FieldComponentAnchorTerm(
        observation=template,
        target_values=torch.tensor([3.4], dtype=torch.float32),
        roles="shared",
        sample_ra_deg=torch.tensor([150.0], dtype=torch.float32),
        sample_dec_deg=torch.tensor([2.0], dtype=torch.float32),
        override_wavelengths=torch.tensor([6000.0], dtype=torch.float32),
        override_bandpass_weights=torch.tensor([1.0], dtype=torch.float32),
        forward_context=contexts[0],
        loss_weight=1.0,
    )
    anchored = fit_field_component_decomposition(
        observations,
        component_roles=("shared", "survey"),
        model=_make_init_model(),
        observation_forward_contexts=contexts,
        component_anchor_terms=[sample_anchor],
        steps=120,
        lr=7.0e-2,
        num_spectral_samples=1,
    )
    plain.model.fix_component_scale_gauge_()
    anchored.model.fix_component_scale_gauge_()
    plain_diag = plain.model.decomposition_diagnostics(
        observations,
        truth_model,
        observation_forward_contexts=contexts,
        num_spectral_samples=1,
        calibration_anchor_model=truth_model,
        calibration_anchor_roles="shared",
    )
    anchored_diag = anchored.model.decomposition_diagnostics(
        observations,
        truth_model,
        observation_forward_contexts=contexts,
        num_spectral_samples=1,
        calibration_anchor_model=truth_model,
        calibration_anchor_roles="shared",
    )

    assert (
        anchored_diag.calibration_transfer_stability
        < plain_diag.calibration_transfer_stability
    )


@pytest.mark.needs_float64
def test_component_centroid_anchor_terms_improve_shared_geometry() -> None:
    torch.manual_seed(0)
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    template = FieldObservation(
        image=torch.zeros((9, 9), dtype=torch.float32),
        header=_tan_header(9, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        spectral_min=5600.0,
        spectral_max=5600.0,
        transmission_center=5600.0,
        transmission_sigma=None,
        weight=torch.full((9, 9), 1.0e-12, dtype=torch.float32),
    )
    truth_x = torch.tensor(6.0, dtype=torch.float64)
    truth_y = torch.tensor(3.0, dtype=torch.float64)
    truth_ra, truth_dec = WCS(template.header).pixel_to_world(truth_x, truth_y)
    off_ra, off_dec = WCS(template.header).pixel_to_world(
        torch.tensor(1.5, dtype=torch.float64),
        torch.tensor(6.5, dtype=torch.float64),
    )
    contexts = [ForwardContext(survey="A")]
    centroid_term = FieldCentroidAnchorTerm(
        observation=template,
        target_ra_deg=float(truth_ra),
        target_dec_deg=float(truth_dec),
        roles="shared",
        forward_context=contexts[0],
        loss_weight=20.0,
        window_sigma_pix=1.5,
    )

    def _make_init_model() -> FieldComponentDecomposition:
        model = FieldComponentDecomposition(
            [
                _GaussianSpotField(
                    domain,
                    amplitude=1.5,
                    center_ra_deg=float(off_ra),
                    center_dec_deg=float(off_dec),
                    sigma_deg=0.8 / 3600.0,
                )
            ],
            component_roles=("shared",),
        )
        with torch.no_grad():
            model.mixer.shared_weights[:] = torch.tensor([1.0])
        model.component_models[0].log_amplitude.requires_grad_(False)
        return model

    plain = fit_field_component_decomposition(
        [template],
        component_roles=("shared",),
        model=_make_init_model(),
        observation_forward_contexts=contexts,
        steps=50,
        lr=5.0e-5,
        num_spectral_samples=1,
    )
    anchored = fit_field_component_decomposition(
        [template],
        component_roles=("shared",),
        model=_make_init_model(),
        observation_forward_contexts=contexts,
        component_centroid_anchor_terms=[centroid_term],
        steps=50,
        lr=5.0e-5,
        num_spectral_samples=1,
    )
    plain_x, plain_y = WCS(template.header).world_to_pixel(
        plain.model.component_models[0].center_ra_deg.detach().to(dtype=torch.float64),
        plain.model.component_models[0].center_dec_deg.detach().to(dtype=torch.float64),
    )
    anchored_x, anchored_y = WCS(template.header).world_to_pixel(
        anchored.model.component_models[0]
        .center_ra_deg.detach()
        .to(dtype=torch.float64),
        anchored.model.component_models[0]
        .center_dec_deg.detach()
        .to(dtype=torch.float64),
    )
    target_xf = float(truth_x.item())
    target_yf = float(truth_y.item())
    plain_distance = math.hypot(
        float(plain_x.item()) - target_xf,
        float(plain_y.item()) - target_yf,
    )
    anchored_distance = math.hypot(
        float(anchored_x.item()) - target_xf,
        float(anchored_y.item()) - target_yf,
    )

    assert anchored_distance < plain_distance * 0.3
