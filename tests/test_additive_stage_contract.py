"""Bounded calibrated-image placement, without claims of raw detector physics."""

from dataclasses import replace

import pytest
import torch

from torchsky.map.field.observation_ops import crop_field_observation
from torchsky.map.field.rendering import (
    render_field_observation,
    render_field_observation_distribution,
    render_field_samples,
    render_field_samples_distribution,
    render_spectral_power_foam_observation_exact,
)
from torchsky.map.field.types import FieldObservation, PSFSpec
from torchsky.map.operators.background import (
    AdditiveComponent,
    AdditiveFrame,
    AdditiveProcessingState,
    AdditiveStage,
    apply_calibrated_additive_response,
)
from torchsky.map.operators.bundle import ObservationOperatorBundle
from torchsky.map.operators.psf import LearnedPSFOp, apply_psf_spec


def test_component_anchor_cannot_silently_strip_typed_background():
    from torchsky.map.field.core import _observation_without_background

    observation = _observation(_component(torch.ones(9, 9)))
    with pytest.raises(NotImplementedError, match="typed additive"):
        _observation_without_background(observation)


def test_provisional_inverse_cannot_ignore_typed_components():
    from torchsky.map.field.inverse import (
        bind_linear_band_grid_observation_operator,
        prepare_linear_band_grid_observation_data,
    )
    from torchsky.map.field.types import FieldDomain

    observation = _observation(_component(torch.ones(9, 9)))
    domain = FieldDomain(180.0, 0.0, 0.1, 10000.0, 1000.0)
    with pytest.raises(NotImplementedError, match="typed additive"):
        prepare_linear_band_grid_observation_data(observation)
    with pytest.raises(NotImplementedError, match="typed additive"):
        bind_linear_band_grid_observation_operator(
            observation,
            domain,
            spatial_resolution=9,
            spatial_bounds_local_deg=(-0.1, 0.1, -0.1, 0.1),
        )


class _ConstantField(torch.nn.Module):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.level = torch.nn.Parameter(torch.tensor(0.0, device=device))

    def forward(self, ra, dec, wavelength):
        return torch.ones_like(ra, dtype=torch.float32) * self.level

    def forward_distribution(self, ra, dec, wavelength):
        return self(ra, dec, wavelength), torch.zeros_like(ra, dtype=torch.float32)


def _component(values, stage=AdditiveStage.SAMPLED_RESPONSE_INPUT, **kwargs):
    return AdditiveComponent(
        values=values,
        stage=stage,
        coordinate_frame=(
            AdditiveFrame.SAMPLED_SKY
            if stage == AdditiveStage.SAMPLED_RESPONSE_INPUT
            else AdditiveFrame.DETECTOR
        ),
        value_unit="electron/s",
        value_semantics="integrated_flux_per_pixel",
        processing_state=AdditiveProcessingState.CALIBRATED_IMAGE,
        **kwargs,
    )


def _observation(*components, side=9, **kwargs):
    header = {
        "NAXIS": 2,
        "NAXIS1": side,
        "NAXIS2": side,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": side / 2,
        "CRPIX2": side / 2,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }
    defaults = dict(
        image=torch.zeros(side, side),
        header=header,
        wavelength=5000.0,
        psf_spec=PSFSpec.kernel(torch.tensor([[0.25, 0.5, 0.25]])),
        additive_components=components,
        value_unit="electron/s",
        value_semantics="integrated_flux_per_pixel",
        processing_state=AdditiveProcessingState.CALIBRATED_IMAGE,
    )
    defaults.update(kwargs)
    return FieldObservation(**defaults)


def test_input_response_and_output_additions_do_not_commute():
    point = torch.zeros(9, 9)
    point[4, 4] = 8.0
    before = render_field_observation(_ConstantField(), _observation(_component(point)))
    after = render_field_observation(
        _ConstantField(),
        _observation(_component(point, AdditiveStage.CALIBRATED_OUTPUT)),
    )
    expected = torch.zeros_like(point)
    expected[4, 3:6] = torch.tensor([2.0, 4.0, 2.0])
    torch.testing.assert_close(before, expected)
    torch.testing.assert_close(after, point)
    assert not torch.equal(before, after)
    assert before.sum() == after.sum() == point.sum()


def test_standalone_exposure_response_needs_no_global_field():
    point = torch.zeros(9, 9)
    point[4, 4] = 8.0
    output_plane = torch.full_like(point, -1.0)
    spec = PSFSpec.kernel(torch.tensor([[0.25, 0.5, 0.25]]))
    components = (
        _component(point),
        _component(output_plane, AdditiveStage.CALIBRATED_OUTPUT),
    )
    kwargs = dict(
        value_unit="electron/s",
        value_semantics="integrated_flux_per_pixel",
        processing_state=AdditiveProcessingState.CALIBRATED_IMAGE,
    )
    output = apply_calibrated_additive_response(
        torch.zeros_like(point),
        components,
        response=lambda image: apply_psf_spec(image, spec),
        **kwargs,
    )
    expected = output_plane.clone()
    expected[4, 3:6] += torch.tensor([2.0, 4.0, 2.0])
    torch.testing.assert_close(output, expected)
    with pytest.raises(ValueError, match="preserve shape"):
        apply_calibrated_additive_response(
            point,
            components,
            response=lambda image: image[:1],
            **kwargs,
        )


def test_flat_response_and_edge_flux_loss_are_not_renormalized():
    flat = torch.ones(9, 9)
    rendered = render_field_observation(
        _ConstantField(), _observation(_component(flat))
    )
    torch.testing.assert_close(rendered[:, 1:-1], flat[:, 1:-1])
    torch.testing.assert_close(rendered[:, [0, -1]], torch.full((9, 2), 0.75))
    point = torch.zeros(9, 9)
    point[4, 0] = 8.0
    edge = render_field_observation(_ConstantField(), _observation(_component(point)))
    assert edge.sum() == 6.0
    assert edge[4, 0] == 4.0 and edge[4, 1] == 2.0


def test_legacy_named_backgrounds_remain_post_response_products():
    point = torch.zeros(9, 9)
    point[4, 4] = 8.0
    observation = _observation(
        sky_background=point,
        instrument_background=2 * point,
        detector_background=3 * point,
        background_offset=1.0,
    )
    torch.testing.assert_close(
        render_field_observation(_ConstantField(), observation), 6 * point + 1
    )


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_tensor_device_dtype_and_gradient_contract(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    values = torch.zeros(9, 9, requires_grad=True)
    residual = torch.zeros(9, 9, requires_grad=True)
    observation = _observation(
        _component(values), _component(residual, AdditiveStage.CALIBRATED_OUTPUT)
    ).to(device=device, dtype=torch.float32)
    output = render_field_observation(_ConstantField(device), observation)
    assert output.device.type == device
    assert output.dtype == observation.image.dtype
    grad_in, grad_out = torch.autograd.grad(output[4, 4], (values, residual))
    expected = torch.zeros_like(values)
    expected[4, 3:6] = torch.tensor([0.25, 0.5, 0.25])
    torch.testing.assert_close(grad_in, expected)
    expected_out = torch.zeros_like(residual)
    expected_out[4, 4] = 1.0
    torch.testing.assert_close(grad_out, expected_out)


def test_move_and_crop_preserve_gradients_and_placement():
    values = torch.ones(9, 9, dtype=torch.float64, requires_grad=True)
    original = _observation(_component(values)).to(dtype=torch.float64)
    moved = original.to(dtype=torch.float32)
    cropped = crop_field_observation(moved, x0=2, y0=3, width=4, height=3)
    component = cropped.additive_components[0]
    assert component.stage == AdditiveStage.SAMPLED_RESPONSE_INPUT
    assert component.values.shape == (3, 4)
    assert component.values.dtype == torch.float32
    component.values.sum().backward()
    expected = torch.zeros_like(values)
    expected[3:6, 2:6] = 1
    torch.testing.assert_close(values.grad, expected)


def test_tiled_response_preserves_halo_additions_and_gradients():
    values = torch.arange(81, dtype=torch.float32).reshape(9, 9).requires_grad_()
    output_values = (-values * 0.2).detach().requires_grad_()
    observation = _observation(
        _component(values), _component(output_values, AdditiveStage.CALIBRATED_OUTPUT)
    )
    whole = render_field_observation(_ConstantField(), observation)
    tiled = render_field_observation(
        _ConstantField(), observation, spatial_tile_hw=(3, 4)
    )
    torch.testing.assert_close(tiled, whole)
    whole_grad = torch.autograd.grad(whole.square().sum(), (values, output_values))
    tiled_grad = torch.autograd.grad(tiled.square().sum(), (values, output_values))
    for actual, expected in zip(tiled_grad, whole_grad, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("sampling", ["center", "gauss_legendre_2x2"])
def test_distribution_adds_means_without_inventing_shot_noise(sampling):
    values = torch.ones(9, 9)
    observation = _observation(_component(values))
    base = replace(observation, additive_components=())
    mean, logvar = render_field_observation_distribution(
        _ConstantField(), observation, spatial_sampling=sampling
    )
    _, base_logvar = render_field_observation_distribution(
        _ConstantField(), base, spatial_sampling=sampling
    )
    expected = render_field_observation(
        _ConstantField(), observation, spatial_sampling=sampling
    )
    torch.testing.assert_close(mean, expected)
    torch.testing.assert_close(logvar, base_logvar)


@pytest.mark.parametrize(
    "stage",
    [
        AdditiveStage.PHOTON_ILLUMINATION,
        AdditiveStage.DARK_CHARGE,
        AdditiveStage.READ_OFFSET,
    ],
)
def test_physical_stages_fail_closed(stage):
    with pytest.raises(ValueError, match="unsupported physical additive stage"):
        _observation(_component(torch.ones(9, 9), stage))


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"value_unit": "ADU"}, "units"),
        ({"value_semantics": "surface_brightness"}, "value_semantics"),
        ({"processing_state": AdditiveProcessingState.RAW_READS}, "processing_state"),
        ({"processing_state": None}, "processing_state"),
    ],
)
def test_observation_contract_mismatches_fail_closed(changes, match):
    with pytest.raises(ValueError, match=match):
        _observation(_component(torch.ones(9, 9)), **changes)


def test_unknown_frame_shape_noise_and_nonfinite_values_fail_closed():
    valid = _component(torch.ones(9, 9))
    with pytest.raises(ValueError, match="coordinate_frame"):
        _observation(replace(valid, coordinate_frame=AdditiveFrame.DETECTOR))
    with pytest.raises(ValueError, match="shape"):
        _observation(_component(torch.ones(1, 1)))
    with pytest.raises(ValueError, match="stochastic physics"):
        _observation(
            _component(torch.ones(9, 9), stochastic_contribution="photon_shot")
        )
    with pytest.raises(ValueError, match="finite"):
        _component(torch.full((9, 9), float("nan")))
    with pytest.raises(ValueError):
        replace(valid, stage="unrecognized_stage")
    with pytest.raises(TypeError, match="floating point"):
        _component(torch.ones(9, 9, dtype=torch.int64))


def test_mutated_component_values_are_revalidated_at_rendering():
    values = torch.ones(9, 9)
    observation = _observation(_component(values))
    values[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        render_field_observation(_ConstantField(), observation)


def test_unspecified_chromatic_additive_response_fails_closed():
    observation = _observation(_component(torch.ones(9, 9)))
    bundle = ObservationOperatorBundle(psf=LearnedPSFOp())
    with pytest.raises(ValueError, match="achromatic explicit PSF"):
        render_field_observation(_ConstantField(), observation, operator_bundle=bundle)


def test_unsupported_rendering_paths_do_not_drop_components():
    observation = _observation(_component(torch.ones(9, 9)), psf_spec=PSFSpec())
    coordinates = dict(ra_deg=torch.ones(3), dec_deg=torch.ones(3))
    for render in (render_field_samples, render_field_samples_distribution):
        with pytest.raises(ValueError, match="typed additive planes"):
            render(_ConstantField(), observation, **coordinates)
    with pytest.raises(ValueError, match="typed additive planes"):
        render_spectral_power_foam_observation_exact(_ConstantField(), observation)


def test_coherent_psf_variance_propagates_std_linearly():
    from torchsky.map.field.rendering import render_field_observation_distribution

    model = _ConstantField()  # predictive std = 1 everywhere, fully correlated
    observation = _observation(side=15)
    _, coherent = render_field_observation_distribution(
        model, observation, num_spectral_samples=1, variance_mode="coherent"
    )
    _, independent = render_field_observation_distribution(
        model,
        observation,
        num_spectral_samples=1,
        variance_mode="independent_aleatoric",
    )
    # Normalised kernel on a uniform fully-correlated std keeps variance at 1.
    assert torch.exp(coherent[7, 7]).item() == pytest.approx(1.0, rel=1e-5)
    # Independent pixels: variance shrinks by sum(K**2) = 0.375 for this kernel.
    assert torch.exp(independent[7, 7]).item() == pytest.approx(0.375, rel=1e-5)
