"""Bound detector PSF work without changing the observation likelihood."""

import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from torchsky.map.field import rendering
from torchsky.map.field.training import fit_field_model
from torchsky.map.field.types import FieldObservation, PSFSpec
from torchsky.map.operators.bundle import (
    ObservationOperatorBundle,
    field_observation_operator_bundle,
)
from torchsky.map.operators.background import LearnedBackgroundOp
from torchsky.map.operators.noise import StructuredDetectorNoise, StructuredNoiseOp
from torchsky.map.operators.psf import LearnedPSFOp


class Scene(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, ra, dec, wavelength):
        wavelength = torch.broadcast_to(wavelength, ra.shape)
        return self.net(
            torch.stack(((ra - 1) * 10, dec * 10, wavelength / 2), -1).to(
                self.net[0].weight.dtype
            )
        ).squeeze(-1)


class IdentityLearnedPSF(nn.Module):
    kernel_size = 1

    def forward(self, image, **_kwargs):
        return image


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
        [observation()],
        BoundedScene(),
        steps=1,
        lr=0.0,
        num_spectral_samples=1,
        streaming_accumulation=True,
        render_spatial_tile_hw=(8, 8),
        device="cpu",
    )


def test_zero_pixel_samples_retains_full_quadrature_and_negative_rejected():
    torch.manual_seed(207)
    full, zero = Scene(), Scene()
    zero.load_state_dict(full.state_dict())
    obs = replace(observation(), psf_spec=PSFSpec())
    kwargs = dict(
        steps=1,
        lr=0.0,
        num_spectral_samples=1,
        spatial_sampling="gauss_legendre_2x2",
        device="cpu",
    )
    fit_field_model([obs], full, **kwargs)
    fit_field_model([obs], zero, pixel_samples_per_observation=0, **kwargs)
    for left, right in zip(full.parameters(), zero.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad)
    with pytest.raises(ValueError, match="non-negative"):
        fit_field_model([obs], zero, pixel_samples_per_observation=-1, **kwargs)
    with pytest.raises(ValueError, match="context and anchor"):
        fit_field_model(
            [obs],
            zero,
            streaming_accumulation=True,
            render_spatial_tile_hw=(8, 8),
            context_observations_per_step=1,
            context_loss_weight=1.0,
            **kwargs,
        )


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


@pytest.mark.parametrize(
    "reduction", ["per_observation", "per_band", "global", "band_global_mix"]
)
def test_recomputed_structured_loss_preserves_full_image_and_noise_gradients(reduction):
    class Background(nn.Module):
        def __init__(self):
            super().__init__()
            self.slope = nn.Parameter(torch.tensor(0.01))

        def forward(self, x, y, wavelength, time):
            return self.slope * (x + 0.3 * y)

    torch.manual_seed(238)
    models = [Scene(), None]
    models[1] = copy.deepcopy(models[0])
    first = observation(13)
    observations = [first, replace(observation(17), band_name="other")]
    noises = [StructuredDetectorNoise(num_amplifiers=2), None]
    noises[1] = copy.deepcopy(noises[0])
    backgrounds = [Background(), Background()]
    results = []
    for index, model in enumerate(models):
        bundles = [
            replace(
                field_observation_operator_bundle(obs),
                noise=StructuredNoiseOp(noises[index], frozen=False),
                background=LearnedBackgroundOp(backgrounds[index]),
            )
            for obs in observations
        ]
        results.append(
            fit_field_model(
                observations,
                model,
                steps=1,
                lr=0.0,
                num_spectral_samples=2,
                spatial_sampling="gauss_legendre_2x2",
                loss_kind="structured_nll",
                loss_reduction=reduction,
                observation_operator_bundles=bundles,
                streaming_accumulation=bool(index),
                render_spatial_tile_hw=(6, 7) if index else None,
                device="cpu",
            )
        )
    assert results[1].final_loss == pytest.approx(results[0].final_loss, rel=2e-5)
    for left, right in [models, noises, backgrounds]:
        for (name, full), tiled in zip(
            left.named_parameters(), right.parameters(), strict=True
        ):
            if full.grad is None:
                assert tiled.grad is None, name
            else:
                torch.testing.assert_close(
                    tiled.grad,
                    full.grad,
                    rtol=3e-4,
                    atol=2e-6,
                    msg=lambda message: f"{name}: {message}",
                )


def test_recomputed_structured_loss_replays_random_draws():
    class RandomScene(Scene):
        def forward(self, ra, dec, wavelength):
            return super().forward(ra, dec, wavelength) * torch.rand_like(ra)

    torch.manual_seed(241)
    full = RandomScene()
    recomputed = copy.deepcopy(full)
    obs = observation(13)
    noise = StructuredDetectorNoise(num_amplifiers=2).requires_grad_(False)
    bundle = replace(
        field_observation_operator_bundle(obs), noise=StructuredNoiseOp(noise)
    )
    rng_states = []
    for model, streaming in [(full, False), (recomputed, True)]:
        torch.manual_seed(243)
        fit_field_model(
            [obs],
            model,
            steps=1,
            lr=0.0,
            num_spectral_samples=2,
            training_spectral_sampling="bandpass_mc",
            loss_kind="structured_nll",
            observation_operator_bundles=[bundle],
            streaming_accumulation=streaming,
            render_spatial_tile_hw=(6, 7),
            diagnostic_spatial_tile_hw=(6, 7),
            device="cpu",
        )
        rng_states.append(torch.get_rng_state())
    for left, right in zip(full.parameters(), recomputed.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=2e-5, atol=2e-6)
    assert torch.equal(*rng_states)


def test_recomputed_structured_loss_bounds_field_activation_memory():
    class Saved:
        def __init__(self, tensor, counter):
            self.tensor, self.counter = tensor, counter
            self.size = tensor.numel() * tensor.element_size()
            counter[0] += self.size
            counter[1] = max(counter)

        def __del__(self):
            self.counter[0] -= self.size

    class MeasuredScene(Scene):
        def __init__(self):
            super().__init__()
            self.counter = [0, 0]

        def forward(self, ra, dec, wavelength):
            # Count field activations separately: the exact noise likelihood
            # necessarily retains full-image tensors outside this scope.
            with torch.autograd.graph.saved_tensors_hooks(
                lambda tensor: Saved(tensor, self.counter), lambda saved: saved.tensor
            ):
                return super().forward(ra, dec, wavelength)

    def measure(side, streaming):
        model, obs = MeasuredScene(), observation(side)
        bundle = replace(
            field_observation_operator_bundle(obs),
            noise=StructuredNoiseOp(StructuredDetectorNoise(num_amplifiers=2)),
        )
        fit_field_model(
            [obs],
            model,
            steps=1,
            lr=0.0,
            num_spectral_samples=1,
            observation_operator_bundles=[bundle],
            loss_kind="structured_nll",
            streaming_accumulation=streaming,
            render_spatial_tile_hw=(8, 8),
            diagnostic_spatial_tile_hw=(8, 8),
            device="cpu",
        )
        assert model.counter[0] == 0
        return model.counter[1]

    small, large, retained = measure(20, True), measure(40, True), measure(40, False)
    assert large <= small * 1.05
    assert large < retained / 4


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


def test_active_learned_effective_psf_rejects_double_pixel_integration():
    obs = replace(
        observation(9),
        psf_spec=PSFSpec(
            mode="kernel2d",
            kernel2d=torch.ones(3, 3),
            pixel_response_included=False,
        ),
    )
    bundle = ObservationOperatorBundle(
        psf=LearnedPSFOp(
            IdentityLearnedPSF(), pixel_response_included=True, frozen=True
        )
    )
    with pytest.raises(ValueError, match="pixel response"):
        rendering.render_field_observation(
            Scene(),
            obs,
            operator_bundle=bundle,
            spatial_sampling="gauss_legendre_2x2",
        )
    prediction = rendering.render_field_observation(
        Scene(), obs, operator_bundle=bundle, spatial_sampling="center"
    )
    assert prediction.shape == obs.image.shape


def test_active_learned_psf_requires_known_pixel_response_for_quadrature():
    obs = replace(observation(9), psf_spec=None)
    unknown = ObservationOperatorBundle(
        psf=LearnedPSFOp(IdentityLearnedPSF(), frozen=True)
    )
    with pytest.raises(ValueError, match="pixel response convention is unknown"):
        rendering.render_field_observation(
            Scene(),
            obs,
            operator_bundle=unknown,
            spatial_sampling="gauss_legendre_2x2",
        )
    optical = ObservationOperatorBundle(
        psf=LearnedPSFOp(
            IdentityLearnedPSF(), pixel_response_included=False, frozen=True
        )
    )
    prediction = rendering.render_field_observation(
        Scene(),
        obs,
        operator_bundle=optical,
        spatial_sampling="gauss_legendre_2x2",
    )
    assert prediction.shape == obs.image.shape


def test_learned_psf_spec_requires_known_pixel_response_for_quadrature():
    spec = PSFSpec.learned(IdentityLearnedPSF(), base_sigma_pix=1.0)
    obs = replace(observation(9), psf_spec=spec)
    with pytest.raises(ValueError, match="pixel response convention is unknown"):
        rendering.render_field_observation(
            Scene(), obs, spatial_sampling="gauss_legendre_2x2"
        )


def test_exact_foam_render_rejects_effective_psf_after_pixel_integration():
    obs = replace(observation(9), psf_spec=None)
    bundle = ObservationOperatorBundle(
        psf=LearnedPSFOp(
            IdentityLearnedPSF(), pixel_response_included=True, frozen=True
        )
    )
    with pytest.raises(ValueError, match="pixel response"):
        rendering.render_spectral_power_foam_observation_exact(
            Scene(), obs, operator_bundle=bundle
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
        [obs],
        model,
        steps=1,
        num_spectral_samples=1,
        pixel_samples_per_observation=0,
        spatial_sampling="gauss_legendre_2x2",
        device="cpu",
    )
    assert model.training_sizes == [obs.image.numel()] * 4
    with pytest.raises(ValueError, match="non-negative"):
        fit_field_model([obs], Scene(), steps=1, pixel_samples_per_observation=-1)


@pytest.mark.parametrize(
    "unsupported",
    [
        {"context_observations_per_step": 1, "context_loss_weight": 1.0},
        {
            "training_spectral_sampling": "mixture_mc",
            "spectral_anchor_observations_per_step": 1,
            "spectral_anchor_loss_weight": 1.0,
        },
    ],
)
def test_streamed_tiles_reject_full_image_auxiliary_losses(unsupported):
    with pytest.raises(ValueError, match="context and anchor"):
        fit_field_model(
            [observation()],
            Scene(),
            steps=1,
            streaming_accumulation=True,
            render_spatial_tile_hw=(8, 8),
            **unsupported,
        )


def test_disabled_psf_bundle_does_not_reject_pixel_quadrature():
    obs = observation(9)
    obs.psf_spec.pixel_response_included = True
    explicit_none = ObservationOperatorBundle()
    prediction = rendering.render_field_observation(
        Scene(),
        obs,
        operator_bundle=explicit_none,
        spatial_sampling="gauss_legendre_2x2",
        spatial_tile_hw=(4, 5),
    )
    assert prediction.shape == obs.image.shape
