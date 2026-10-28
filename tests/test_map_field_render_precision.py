"""Subpixel sky structure must survive rendering at large absolute RA/Dec."""

import math

import pytest
import torch

from torchsky.map.field.observation_ops import coordinate_grid
from torchsky.map.field.rendering import (
    render_field_observation,
    render_field_observation_distribution,
    render_field_samples,
    render_field_samples_distribution,
)
from torchsky.map.field.types import FieldObservation


class _NarrowSource(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.amplitude = torch.nn.Parameter(torch.tensor(1.3))
        self.x_center = torch.nn.Parameter(torch.tensor(0.007))

    def forward(self, ra, dec, wavelength):
        # Inverse TAN projection in double; only local arcseconds become float32.
        dra = torch.deg2rad(ra - 110.8375)
        lat = torch.deg2rad(dec)
        lat0 = math.radians(-73.4392)
        denominator = (
            lat.sin() * math.sin(lat0) + lat.cos() * math.cos(lat0) * dra.cos()
        )
        x = torch.rad2deg(lat.cos() * dra.sin() / denominator) * 3600
        y = (
            torch.rad2deg(
                (lat.sin() * math.cos(lat0) - lat.cos() * math.sin(lat0) * dra.cos())
                / denominator
            )
            * 3600
        )
        return self.profile(x.float(), y.float())

    def profile(self, x, y):
        return self.amplitude * torch.exp(
            -((x - self.x_center).square() + (y + 0.003).square()) / (2 * 0.016**2)
        )

    def forward_distribution(self, ra, dec, wavelength):
        mean = self(ra, dec, wavelength)
        return mean, torch.full_like(mean, -2.0)


class _BandpassSource(_NarrowSource):
    def render_bandpass(self, ra, dec, wavelengths, weights):
        return self(ra, dec, wavelengths[0]) * weights.sum()

    def render_bandpass_distribution(self, ra, dec, wavelengths, weights):
        mean = self.render_bandpass(ra, dec, wavelengths, weights)
        return mean, torch.full_like(mean, -2.0)


class _PreciseConstant(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(1.0 + 1.0e-10, dtype=torch.float64))

    def forward(self, ra, dec, wavelength):
        ra, dec, wavelength = torch.broadcast_tensors(ra, dec, wavelength)
        return torch.zeros_like(ra, dtype=torch.float64) + self.value


class _WavelengthDistribution(torch.nn.Module):
    predictive_distribution_available = True

    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def forward(self, ra, dec, wavelength):
        ra, dec, wavelength = torch.broadcast_tensors(ra, dec, wavelength)
        return ra + dec + wavelength * 0.0 + self.offset

    def forward_distribution(self, ra, dec, wavelength):
        mean = self(ra, dec, wavelength)
        variance = torch.where(wavelength < 5500.0, 1.0, 9.0)
        return mean, torch.log(torch.zeros_like(mean) + variance)


def _observation():
    return FieldObservation(
        image=torch.zeros(9, 11),
        wavelength=5000.0,
        header={
            "NAXIS": 2,
            "NAXIS1": 11,
            "NAXIS2": 9,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": 6.0,
            "CRPIX2": 5.0,
            "CRVAL1": 110.8375,
            "CRVAL2": -73.4392,
            "CD1_1": -0.01 / 3600,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": 0.01 / 3600,
        },
    )


@pytest.mark.parametrize("model_type", [_NarrowSource, _BandpassSource])
@pytest.mark.parametrize("sampling", ["center", "gauss_legendre_2x2"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_fine_pixels_preserve_source_and_parameter_gradients(
    model_type, sampling, device
):
    model, obs = model_type().to(device), _observation()
    y, x = torch.meshgrid(
        torch.arange(9, device=device), torch.arange(11, device=device), indexing="ij"
    )
    offsets = (
        (0.0,) if sampling == "center" else (-0.5 / math.sqrt(3), 0.5 / math.sqrt(3))
    )
    expected = (
        sum(
            model.profile(-(x - 5 + dx) * 0.01, (y - 4 + dy) * 0.01)
            for dx in offsets
            for dy in offsets
        )
        / len(offsets) ** 2
    )
    expected_grad = torch.autograd.grad(
        expected.square().sum(), tuple(model.parameters())
    )
    for tiles in (None, (4, 3)):
        actual = render_field_observation(
            model, obs, spatial_sampling=sampling, spatial_tile_hw=tiles
        )
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        gradients = torch.autograd.grad(
            actual.square().sum(), tuple(model.parameters())
        )
        for actual_grad, analytic_grad in zip(gradients, expected_grad, strict=True):
            torch.testing.assert_close(actual_grad, analytic_grad, rtol=2e-4, atol=2e-4)
        mean, _ = render_field_observation_distribution(
            model, obs, spatial_sampling=sampling, spatial_tile_hw=tiles
        )
        torch.testing.assert_close(mean, expected, rtol=2e-5, atol=2e-6)
    if sampling == "center":
        ra, dec = coordinate_grid(
            obs.header, shape=(9, 11), device=torch.device(device)
        )
        kwargs = {"ra_deg": ra.flatten(), "dec_deg": dec.flatten()}
        samples = render_field_samples(model, obs, **kwargs)
        mean, _ = render_field_samples_distribution(model, obs, **kwargs)
        torch.testing.assert_close(samples, expected.flatten(), rtol=2e-5, atol=2e-6)
        assert mean.shape == (99,)
        torch.testing.assert_close(
            mean.flatten(), expected.flatten(), rtol=2e-5, atol=2e-6
        )


@pytest.mark.parametrize(
    ("ra", "dec", "shape"),
    [
        (torch.tensor(1.0), torch.tensor(2.0), ()),
        (torch.arange(4.0), torch.tensor(2.0), (4,)),
        (torch.arange(6.0).reshape(2, 3).T, torch.arange(3.0)[:, None], (3, 2)),
        (
            torch.arange(6.0).reshape(2, 1, 3),
            torch.arange(4.0).reshape(1, 4, 1),
            (2, 4, 3),
        ),
        (torch.empty(0, 2), torch.empty(0, 1), (0, 2)),
    ],
)
def test_sample_rendering_preserves_broadcast_shape(ra, dec, shape) -> None:
    observation = _observation()
    model = _WavelengthDistribution()
    wavelengths = torch.tensor([5000.0, 6000.0], dtype=torch.float64)
    weights = torch.tensor([0.25, 0.75], dtype=torch.float64)
    values = render_field_samples(
        model,
        observation,
        ra_deg=ra,
        dec_deg=dec,
        override_wavelengths=wavelengths,
        override_weights=weights,
    )
    mean, logvar = render_field_samples_distribution(
        model,
        observation,
        ra_deg=ra,
        dec_deg=dec,
        override_wavelengths=wavelengths,
        override_weights=weights,
    )
    assert values.shape == mean.shape == logvar.shape == shape
    torch.testing.assert_close(values, mean)


def test_sample_distribution_defaults_to_coherent_quadrature_variance() -> None:
    observation = _observation()
    model = _WavelengthDistribution()
    kwargs = {
        "ra_deg": torch.tensor([1.0, 2.0], dtype=torch.float64),
        "dec_deg": torch.tensor(0.0, dtype=torch.float64),
        "override_wavelengths": torch.tensor([5000.0, 6000.0]),
        "override_weights": torch.tensor([0.25, 0.75]),
    }
    _, coherent_logvar = render_field_samples_distribution(model, observation, **kwargs)
    _, independent_logvar = render_field_samples_distribution(
        model,
        observation,
        variance_mode="independent_aleatoric",
        **kwargs,
    )
    torch.testing.assert_close(
        coherent_logvar.exp(), torch.full((2,), 6.25, dtype=torch.float64)
    )
    torch.testing.assert_close(
        independent_logvar.exp(), torch.full((2,), 5.125, dtype=torch.float64)
    )


def test_float64_sample_rendering_retains_sub_float32_precision_and_gradients() -> None:
    observation = _observation()
    observation.image = observation.image.to(dtype=torch.float64)
    model = _PreciseConstant()
    ra = torch.arange(6, dtype=torch.float64).reshape(2, 3).T
    actual = render_field_samples(
        model,
        observation,
        ra_deg=ra,
        dec_deg=torch.zeros((3, 1), dtype=torch.float64),
        override_wavelengths=torch.tensor([5000.0, 6000.0]),
        override_weights=torch.tensor([0.25, 0.75]),
    )
    assert actual.dtype == torch.float64
    torch.testing.assert_close(
        actual,
        torch.full((3, 2), 1.0 + 1.0e-10, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    )
    actual.sum().backward()
    torch.testing.assert_close(model.value.grad, torch.tensor(6.0, dtype=torch.float64))
    rendered = render_field_observation(
        model,
        observation,
        override_wavelengths=torch.tensor([5000.0, 6000.0]),
        override_weights=torch.tensor([0.25, 0.75]),
    )
    assert rendered.dtype == torch.float64
    torch.testing.assert_close(
        rendered,
        torch.full((9, 11), 1.0 + 1.0e-10, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    )
