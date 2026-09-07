"""Bound detector PSF work without changing the observation likelihood."""

import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from torchsky.map.field import rendering
from torchsky.map.field.training import fit_field_model
from torchsky.map.field.types import FieldObservation, PSFSpec


class Scene(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, ra, dec, wavelength):
        wavelength = torch.broadcast_to(wavelength, ra.shape)
        return self.net(
            torch.stack(((ra - 1) * 10, dec * 10, wavelength / 2), -1)
        ).squeeze(-1)


def observation(side=33):
    image = torch.linspace(0.1, 0.9, side * (side + 3)).reshape(side, side + 3)
    return FieldObservation(
        image=image,
        header={
            "NAXIS1": side + 3,
            "NAXIS2": side,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRVAL1": 1.0,
            "CRVAL2": 0.0,
            "CRPIX1": 1.0,
            "CRPIX2": 1.0,
            "CDELT1": 0.001,
            "CDELT2": 0.001,
        },
        weight=1 + image,
        mask=(image > 0.2),
        spectral_min=1.0,
        spectral_max=2.0,
        psf_spec=PSFSpec.kernel(
            torch.tensor([[0.0, 1.0, 0.0], [1.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
        ),
        detector_background=torch.full_like(image, 0.04),
    )


def test_psf_tile_iterator_matches_full_predictions_and_gradients():
    iterator = getattr(rendering, "iter_field_observation_tiles", None)
    assert callable(iterator), "bounded PSF tile iterator is missing"
    torch.manual_seed(16)
    model = Scene()
    obs = observation()
    kwargs = {"num_spectral_samples": 3, "spatial_sampling": "gauss_legendre_2x2"}
    full = rendering.render_field_observation(model, obs, **kwargs)
    full.square().sum().backward()
    expected_gradients = [parameter.grad.clone() for parameter in model.parameters()]
    model.zero_grad()
    assembled = torch.empty_like(full)
    for ys, xs, prediction in iterator(model, obs, spatial_tile_hw=(8, 11), **kwargs):
        assembled[ys, xs] = prediction.detach()
        prediction.square().sum().backward()
    torch.testing.assert_close(assembled, full, rtol=2e-5, atol=2e-6)
    for parameter, expected in zip(model.parameters(), expected_gradients, strict=True):
        torch.testing.assert_close(parameter.grad, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "reduction", ["per_observation", "per_band", "global", "band_global_mix"]
)
def test_streamed_tile_fit_preserves_full_loss_and_gradient(reduction):
    torch.manual_seed(17)
    full_model = Scene()
    tiled_model = copy.deepcopy(full_model)
    first = observation()
    second = replace(
        first, image=first.image * 1.3, weight=first.weight * 0.7, band_name="second"
    )
    kwargs = dict(
        steps=1,
        lr=0.0,
        num_spectral_samples=2,
        loss_reduction=reduction,
        spatial_sampling="gauss_legendre_2x2",
        device="cpu",
        streaming_accumulation=True,
        diagnostic_spatial_tile_hw=(8, 11),
    )
    full_fit = fit_field_model([first, second], full_model, **kwargs)
    tiled_fit = fit_field_model(
        [first, second], tiled_model, render_spatial_tile_hw=(8, 11), **kwargs
    )
    assert tiled_fit.final_loss == pytest.approx(full_fit.final_loss, rel=2e-5)
    for full, tiled in zip(
        full_model.parameters(), tiled_model.parameters(), strict=True
    ):
        torch.testing.assert_close(tiled.grad, full.grad, rtol=3e-5, atol=2e-6)


def test_streamed_tile_training_has_bounded_saved_activation_memory():
    class Saved:
        def __init__(self, tensor, counter):
            self.tensor, self.counter = tensor, counter
            self.size = tensor.numel() * tensor.element_size()
            counter[0] += self.size
            counter[1] = max(counter)

        def __del__(self):
            self.counter[0] -= self.size

    def measure(side, tiled):
        counter = [0, 0]
        torch.manual_seed(18)
        with torch.autograd.graph.saved_tensors_hooks(
            lambda tensor: Saved(tensor, counter), lambda saved: saved.tensor
        ):
            fit_field_model(
                [observation(side)],
                Scene(),
                steps=1,
                lr=0.0,
                num_spectral_samples=1,
                streaming_accumulation=True,
                render_spatial_tile_hw=(16, 16) if tiled else None,
                diagnostic_spatial_tile_hw=(16, 16),
                device="cpu",
            )
        assert counter[0] == 0
        return counter[1]

    small = measure(40, True)
    large = measure(80, True)
    full = measure(80, False)
    print(
        f"saved activation peak bytes: tiled40={small}, tiled80={large}, full80={full}"
    )
    assert large <= small * 1.05
    assert large < full / 4


def test_training_diagnostics_inherit_requested_tile_size():
    class BoundedScene(Scene):
        def forward(self, ra, dec, wavelength):
            assert ra.numel() <= 100, "diagnostics unexpectedly rendered a full image"
            return super().forward(ra, dec, wavelength)

    fit_field_model(
        [observation()], BoundedScene(), steps=1, lr=0.0,
        num_spectral_samples=1, streaming_accumulation=True,
        render_spatial_tile_hw=(8, 8), device="cpu",
    )


def test_zero_pixel_samples_retains_full_quadrature_and_negative_rejected():
    torch.manual_seed(207)
    full, zero = Scene(), Scene()
    zero.load_state_dict(full.state_dict())
    obs = replace(observation(), psf_spec=PSFSpec())
    kwargs = dict(steps=1, lr=0.0, num_spectral_samples=1,
                  spatial_sampling="gauss_legendre_2x2", device="cpu")
    fit_field_model([obs], full, **kwargs)
    fit_field_model([obs], zero, pixel_samples_per_observation=0, **kwargs)
    for left, right in zip(full.parameters(), zero.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad)
    with pytest.raises(ValueError, match="non-negative"):
        fit_field_model([obs], zero, pixel_samples_per_observation=-1, **kwargs)
    with pytest.raises(ValueError, match="context and anchor"):
        fit_field_model([obs], zero, streaming_accumulation=True,
                        render_spatial_tile_hw=(8, 8), context_observations_per_step=1,
                        context_loss_weight=1.0, **kwargs)


def test_incompatible_sampling_and_correlated_tile_loss_fail_explicitly():
    model, obs = Scene(), observation()
    with pytest.raises(ValueError, match="pixel-sampled"):
        fit_field_model([obs], model, steps=1, pixel_samples_per_observation=20)
    with pytest.raises(ValueError, match="pixel-sampled"):
        fit_field_model(
            [replace(obs, psf_spec=PSFSpec())],
            model,
            steps=1,
            pixel_samples_per_observation=20,
            spatial_sampling="gauss_legendre_2x2",
        )
    with pytest.raises(ValueError, match="structured_nll"):
        fit_field_model(
            [obs],
            model,
            steps=1,
            loss_kind="structured_nll",
            streaming_accumulation=True,
            render_spatial_tile_hw=(8, 8),
        )


def test_effective_psf_rejects_double_pixel_integration():
    spec = PSFSpec(
        mode="kernel2d", kernel2d=torch.ones(3, 3), pixel_response_included=True
    )
    assert spec.to("cpu").pixel_response_included is True
    obs = replace(observation(), psf_spec=spec)
    with pytest.raises(ValueError, match="pixel response"):
        rendering.render_field_observation(
            Scene(), obs, spatial_sampling="gauss_legendre_2x2"
        )


def test_zero_pixel_samples_disable_sampling_and_negatives_fail():
    class RecordingScene(Scene):
        def __init__(self):
            super().__init__()
            self.training_sizes = []

        def forward(self, ra, dec, wavelength):
            if torch.is_grad_enabled():
                self.training_sizes.append(ra.numel())
            return super().forward(ra, dec, wavelength)

    obs = replace(observation(9), psf_spec=PSFSpec(), detector_background=None)
    model = RecordingScene()
    fit_field_model(
        [obs], model, steps=1, num_spectral_samples=1,
        pixel_samples_per_observation=0, spatial_sampling="gauss_legendre_2x2",
        device="cpu",
    )
    assert model.training_sizes == [obs.image.numel()] * 4
    with pytest.raises(ValueError, match="non-negative"):
        fit_field_model([obs], Scene(), steps=1, pixel_samples_per_observation=-1)


@pytest.mark.parametrize("unsupported", [
    {"context_observations_per_step": 1, "context_loss_weight": 1.0},
    {"training_spectral_sampling": "mixture_mc", "spectral_anchor_observations_per_step": 1,
     "spectral_anchor_loss_weight": 1.0},
])
def test_streamed_tiles_reject_full_image_auxiliary_losses(unsupported):
    with pytest.raises(ValueError, match="context and anchor"):
        fit_field_model(
            [observation()], Scene(), steps=1, streaming_accumulation=True,
            render_spatial_tile_hw=(8, 8), **unsupported,
        )


def test_disabled_psf_bundle_does_not_reject_pixel_quadrature():
    from torchsky.map.operators.bundle import ObservationOperatorBundle

    obs = observation(9)
    obs.psf_spec.pixel_response_included = True
    explicit_none = ObservationOperatorBundle()
    prediction = rendering.render_field_observation(
        Scene(), obs, operator_bundle=explicit_none,
        spatial_sampling="gauss_legendre_2x2", spatial_tile_hw=(4, 5),
    )
    assert prediction.shape == obs.image.shape
