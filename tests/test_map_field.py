from __future__ import annotations

import math

import pytest
import torch

from torchsky import WCS
from torchsky.map import (
    ComponentDecompositionDiagnostics,
    FieldComponentAnchorTerm,
    FieldCentroidAnchorTerm,
    FieldDomain,
    FieldComponentDecomposition,
    FieldObservation,
    ForwardContext,
    PSFSpec,
    SineImplicitField,
    estimate_field_domain,
    field_component_anchor_loss,
    field_component_centroid_anchor_loss,
    fit_field_component_decomposition,
    fit_field_model,
    make_field_component_decomposition,
    make_field_model,
    render_field_cube,
    render_field_observation,
    render_field_observation_distribution,
)
from torchsky.spectral import SpectralAxis
from torchsky.map.field import _noise_pretrain_field_model
from torchsky.map.field.bandpass import bandpass_quadrature
from torchsky.map.field.core import _BaseFieldModel
from torchsky.map.field.hashgrid import HashGridEncoder
from torchsky.map.field.observation_ops import (
    crop_field_observation,
    crop_field_observation_for_training,
)
from torchsky.map.field import rendering as field_rendering
from torchsky.map.operators import apply_kernel2d_variance, gaussian_kernel1d


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


class _ConstantField(_BaseFieldModel):
    def __init__(self, domain: FieldDomain, value: float) -> None:
        super().__init__(domain)
        self.value = torch.nn.Parameter(torch.tensor(float(value), dtype=torch.float32))

    def forward(
        self,
        ra_deg: torch.Tensor,
        dec_deg: torch.Tensor,
        wavelength_angstrom: torch.Tensor | float,
        *,
        forward_context: ForwardContext | None = None,
    ) -> torch.Tensor:
        del dec_deg, wavelength_angstrom, forward_context
        return torch.zeros_like(ra_deg) + self.value


class _ConstantDistributionField(_ConstantField):
    def __init__(self, domain: FieldDomain, value: float, variance: float) -> None:
        super().__init__(domain, value)
        self.log_variance = torch.nn.Parameter(
            torch.tensor(math.log(float(variance)), dtype=torch.float32)
        )

    def forward_distribution(
        self,
        ra_deg: torch.Tensor,
        dec_deg: torch.Tensor,
        wavelength_angstrom: torch.Tensor | float,
        *,
        forward_context: ForwardContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.forward(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
            forward_context=forward_context,
        )
        return mean, torch.zeros_like(mean) + self.log_variance


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


class _GaussianSpotField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        amplitude: float,
        center_ra_deg: float,
        center_dec_deg: float,
        sigma_deg: float,
    ) -> None:
        super().__init__(domain)
        sign = 1.0 if amplitude >= 0.0 else -1.0
        magnitude = max(abs(float(amplitude)), 1.0e-6)
        self.log_amplitude = torch.nn.Parameter(
            torch.tensor(math.log(magnitude), dtype=torch.float32)
        )
        self.amplitude_sign = float(sign)
        self.center_ra_deg = torch.nn.Parameter(
            torch.tensor(float(center_ra_deg), dtype=torch.float32)
        )
        self.center_dec_deg = torch.nn.Parameter(
            torch.tensor(float(center_dec_deg), dtype=torch.float32)
        )
        self.sigma_deg = float(sigma_deg)

    def forward(
        self,
        ra_deg: torch.Tensor,
        dec_deg: torch.Tensor,
        wavelength_angstrom: torch.Tensor | float,
        *,
        forward_context: ForwardContext | None = None,
    ) -> torch.Tensor:
        del wavelength_angstrom, forward_context
        sigma = torch.tensor(self.sigma_deg, device=ra_deg.device, dtype=ra_deg.dtype)
        dx = (
            ra_deg - self.center_ra_deg.to(device=ra_deg.device, dtype=ra_deg.dtype)
        ) / sigma
        dy = (
            dec_deg - self.center_dec_deg.to(device=dec_deg.device, dtype=dec_deg.dtype)
        ) / sigma
        amplitude = self.log_amplitude.to(
            device=ra_deg.device, dtype=ra_deg.dtype
        ).exp()
        return (
            self.amplitude_sign
            * amplitude
            * torch.exp(-0.5 * (dx.square() + dy.square()))
        )


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


def test_render_field_observation_accepts_explicit_operator_bundle() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    model = _ConstantField(domain, 2.0)
    observation = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=_tan_header(4, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        wavelength=5600.0,
        background_offset=3.0,
    )

    implicit = render_field_observation(model, observation, num_spectral_samples=1)
    explicit = render_field_observation(
        model,
        observation,
        num_spectral_samples=1,
        operator_bundle=field_rendering.ObservationOperatorBundle(),
    )

    torch.testing.assert_close(implicit, torch.full((4, 4), 5.0))
    torch.testing.assert_close(explicit, torch.full((4, 4), 2.0))


def test_observation_crop_preserves_physical_background_components() -> None:
    values = torch.arange(56, dtype=torch.float32).reshape(7, 8)
    observation = FieldObservation(
        image=values,
        header=_tan_header(8, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        wavelength=5600.0,
        sky_background=values + 100.0,
        instrument_background=values + 200.0,
        detector_background=values + 300.0,
    )

    cropped = crop_field_observation(
        observation,
        x0=2,
        y0=1,
        width=4,
        height=3,
    )

    torch.testing.assert_close(cropped.image, values[1:4, 2:6])
    torch.testing.assert_close(cropped.sky_background, values[1:4, 2:6] + 100.0)
    torch.testing.assert_close(
        cropped.instrument_background, values[1:4, 2:6] + 200.0
    )
    torch.testing.assert_close(
        cropped.detector_background, values[1:4, 2:6] + 300.0
    )


def test_psf_halo_training_crop_matches_full_render_and_gradient() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    model = _ConstantField(domain, 2.0)
    observation = FieldObservation(
        image=torch.zeros((20, 20), dtype=torch.float32),
        header=_tan_header(20, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        wavelength=5600.0,
        psf_sigma_pix=1.2,
    )
    x0, y0, width, height = 6, 7, 5, 4
    naive = crop_field_observation(
        observation,
        x0=x0,
        y0=y0,
        width=width,
        height=height,
    )
    halo = crop_field_observation_for_training(
        observation,
        x0=x0,
        y0=y0,
        width=width,
        height=height,
    )

    full = render_field_observation(model, observation, num_spectral_samples=1)
    naive_render = render_field_observation(model, naive, num_spectral_samples=1)
    halo_render = render_field_observation(model, halo, num_spectral_samples=1)
    expected = full[y0 : y0 + height, x0 : x0 + width]
    assert halo.mask is not None
    assert int(torch.count_nonzero(halo.mask)) == width * height
    assert float(torch.max(torch.abs(naive_render - expected)).detach()) > 0.1
    torch.testing.assert_close(halo_render[halo.mask], expected.reshape(-1))

    model.zero_grad(set_to_none=True)
    expected.sum().backward()
    assert model.value.grad is not None
    full_gradient = model.value.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    render_field_observation(model, halo, num_spectral_samples=1)[halo.mask].sum().backward()
    torch.testing.assert_close(model.value.grad, full_gradient)

    edge_halo = crop_field_observation_for_training(
        observation,
        x0=0,
        y0=0,
        width=width,
        height=height,
    )
    edge_render = render_field_observation(
        model, edge_halo, num_spectral_samples=1
    )
    assert edge_halo.mask is not None
    torch.testing.assert_close(
        edge_render[edge_halo.mask], full[:height, :width].reshape(-1)
    )


def test_kernel_psf_training_crop_uses_kernel_support() -> None:
    observation = FieldObservation(
        image=torch.zeros((20, 20), dtype=torch.float32),
        header=_tan_header(20, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        wavelength=5600.0,
        psf_spec=PSFSpec.kernel(torch.ones((5, 7), dtype=torch.float32)),
    )
    cropped = crop_field_observation_for_training(
        observation,
        x0=8,
        y0=8,
        width=4,
        height=3,
    )
    assert cropped.image.shape == (7, 10)
    assert cropped.mask is not None
    assert int(torch.count_nonzero(cropped.mask)) == 12


def test_field_spatial_quadrature_preserves_constant_and_tiles() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    model = _ConstantField(domain, 2.0)
    observation = FieldObservation(
        image=torch.zeros((9, 9), dtype=torch.float32),
        header=_tan_header(9, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        wavelength=5600.0,
        background_offset=3.0,
    )

    center = render_field_observation(model, observation, num_spectral_samples=1)
    quadrature = render_field_observation(
        model,
        observation,
        num_spectral_samples=1,
        spatial_sampling="gauss_legendre_2x2",
    )
    tiled = render_field_observation(
        model,
        observation,
        num_spectral_samples=1,
        spatial_sampling="gauss_legendre_2x2",
        spatial_tile_hw=(4, 5),
    )

    torch.testing.assert_close(center, torch.full((9, 9), 5.0))
    torch.testing.assert_close(quadrature, center)
    torch.testing.assert_close(tiled, quadrature)
    quadrature.sum().backward()
    torch.testing.assert_close(model.value.grad, torch.tensor(81.0))


def test_field_spatial_quadrature_reduces_subpixel_gaussian_error() -> None:
    size = 7
    scale_deg = 1.0 / 3600.0
    header = _tan_header(
        size,
        crval1=150.0,
        crval2=2.0,
        scale_deg=scale_deg,
    )
    observation = FieldObservation(
        image=torch.zeros((size, size), dtype=torch.float32),
        header=header,
        wavelength=5600.0,
    )
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    wcs = WCS(header)
    dense_offsets = (
        torch.arange(48, dtype=torch.float64) + 0.5
    ) / 48.0 - 0.5
    dy, dx, y, x = torch.meshgrid(
        dense_offsets,
        dense_offsets,
        torch.arange(size, dtype=torch.float64),
        torch.arange(size, dtype=torch.float64),
        indexing="ij",
    )
    dense_ra, dense_dec = wcs.pixel_to_world(
        (x + dx).reshape(-1),
        (y + dy).reshape(-1),
    )
    center_error = 0.0
    quadrature_error = 0.0
    for phase_x, phase_y in ((-0.3, -0.2), (0.0, 0.25), (0.3, -0.25)):
        spot_ra, spot_dec = wcs.pixel_to_world(
            torch.tensor(3.0 + phase_x, dtype=torch.float64),
            torch.tensor(3.0 + phase_y, dtype=torch.float64),
        )
        model = _GaussianSpotField(
            domain,
            amplitude=1.0,
            center_ra_deg=float(spot_ra),
            center_dec_deg=float(spot_dec),
            sigma_deg=0.45 * scale_deg,
        )
        center = render_field_observation(
            model,
            observation,
            num_spectral_samples=1,
        )
        quadrature = render_field_observation(
            model,
            observation,
            num_spectral_samples=1,
            spatial_sampling="gauss_legendre_2x2",
        )
        dense = model(dense_ra, dense_dec, 5600.0).reshape(
            48, 48, size, size
        ).mean(dim=(0, 1))
        center_error += float(torch.mean(torch.abs(center - dense)).detach())
        quadrature_error += float(
            torch.mean(torch.abs(quadrature - dense)).detach()
        )

    assert quadrature_error < center_error


def test_field_spatial_sampling_rejects_unknown_mode() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    observation = FieldObservation(
        image=torch.zeros((3, 3), dtype=torch.float32),
        header=_tan_header(3, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        wavelength=5600.0,
    )

    with pytest.raises(ValueError, match="unsupported spatial_sampling"):
        render_field_observation(
            _ConstantField(domain, 1.0),
            observation,
            spatial_sampling="unknown",  # type: ignore[arg-type]
        )


def test_render_field_observation_distribution_fallback_applies_operators_once() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    model = _ConstantField(domain, 2.0)
    observation = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=_tan_header(4, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        wavelength=5600.0,
        background_offset=3.0,
    )

    mean, logvar = render_field_observation_distribution(
        model,
        observation,
        num_spectral_samples=1,
    )

    torch.testing.assert_close(mean, torch.full((4, 4), 5.0))
    torch.testing.assert_close(logvar, torch.zeros((4, 4), dtype=torch.float32))
    quadrature_mean, quadrature_logvar = render_field_observation_distribution(
        model,
        observation,
        num_spectral_samples=1,
        spatial_sampling="gauss_legendre_2x2",
    )
    torch.testing.assert_close(quadrature_mean, mean)
    torch.testing.assert_close(
        quadrature_logvar,
        torch.full_like(logvar, math.log(0.25)),
    )


def test_distribution_tiling_and_psf_variance_match_full_render() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    model = _ConstantDistributionField(domain, value=2.0, variance=3.0)
    observation = FieldObservation(
        image=torch.zeros((11, 11), dtype=torch.float32),
        header=_tan_header(11, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        wavelength=5600.0,
        psf_sigma_pix=1.0,
    )

    full_mean, full_logvar = render_field_observation_distribution(
        model, observation, num_spectral_samples=1
    )
    kernel = gaussian_kernel1d(
        1.0, device=torch.device("cpu"), dtype=torch.float32
    )
    expected_center_variance = 3.0 * torch.square(kernel).sum() ** 2
    torch.testing.assert_close(
        torch.exp(full_logvar[5, 5]), expected_center_variance
    )
    full_logvar.sum().backward()
    assert model.log_variance.grad is not None
    full_gradient = model.log_variance.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    observation.clear_coordinate_cache()
    tiled_mean, tiled_logvar = render_field_observation_distribution(
        model,
        observation,
        num_spectral_samples=1,
        spatial_tile_hw=(4, 5),
    )
    torch.testing.assert_close(tiled_mean, full_mean)
    torch.testing.assert_close(tiled_logvar, full_logvar)
    assert observation._cached_ra_deg is None
    assert observation._cached_dec_deg is None
    tiled_logvar.sum().backward()
    torch.testing.assert_close(model.log_variance.grad, full_gradient)


def test_psf_kernel_requires_a_physical_odd_support() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PSFSpec.kernel(torch.tensor([[1.0, -1.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="dimensions must be odd"):
        PSFSpec.kernel(torch.ones((2, 3), dtype=torch.float32))


def test_kernel_psf_variance_uses_squared_normalized_weights() -> None:
    variance = torch.ones((9, 9), dtype=torch.float32, requires_grad=True)
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    propagated = apply_kernel2d_variance(variance, kernel)
    normalized = kernel / kernel.sum()
    torch.testing.assert_close(propagated[4, 4], torch.square(normalized).sum())
    propagated.sum().backward()
    assert variance.grad is not None
    assert bool(torch.isfinite(variance.grad).all())


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

    torch.testing.assert_close(full, torch.full((2, 2), 8.0, dtype=torch.float32))
    torch.testing.assert_close(private, torch.full((2, 2), 7.0, dtype=torch.float32))
    torch.testing.assert_close(parts[0], torch.full((2, 2), 1.0, dtype=torch.float32))
    torch.testing.assert_close(parts[1], torch.full((2, 2), 6.0, dtype=torch.float32))


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

    torch.testing.assert_close(rendered, torch.full((4, 4), 2.5, dtype=torch.float32))


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

    torch.testing.assert_close(before, after)
    torch.testing.assert_close(gains, torch.ones_like(gains))
    torch.testing.assert_close(
        model.mixer.shared_weights,
        torch.tensor([2.0, 0.0], dtype=torch.float32),
    )
    torch.testing.assert_close(
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

    torch.testing.assert_close(matched_loss, torch.zeros((), dtype=torch.float32))
    torch.testing.assert_close(shifted_loss, torch.tensor(2.0, dtype=torch.float32))


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

    torch.testing.assert_close(loss, torch.tensor(2.0 / 3.0, dtype=torch.float32))


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

    torch.testing.assert_close(
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
    torch.testing.assert_close(
        torch.tensor(diagnostics.shared_to_private_leakage),
        torch.tensor(0.5),
    )
    torch.testing.assert_close(
        torch.tensor(diagnostics.private_to_shared_leakage),
        torch.tensor(0.25),
    )
    torch.testing.assert_close(
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

    torch.testing.assert_close(pred_a, observations[0].image, atol=5.0e-2, rtol=0.0)
    torch.testing.assert_close(pred_b, observations[1].image, atol=5.0e-2, rtol=0.0)
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
    obs_a = render_field_observation(
        truth_model, template_a, num_spectral_samples=7
    ).detach()
    obs_b = render_field_observation(
        truth_model, template_b, num_spectral_samples=7
    ).detach()
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

    assert math.isfinite(fit.final_loss)
    assert fit.final_loss < 1.0e-2
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
        image = render_field_observation(
            truth_model, template, num_spectral_samples=1
        ).detach()
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


def test_fit_field_model_ignores_nonfinite_masked_pixels() -> None:
    torch.manual_seed(1234)
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
    )
    truth_model = SineImplicitField(domain, hidden_dim=24, hidden_layers=2)
    header = _tan_header(16, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    template = FieldObservation(
        image=torch.zeros((16, 16), dtype=torch.float32),
        header=header,
        spectral_min=5600.0,
        spectral_max=5600.0,
        transmission_center=5600.0,
        transmission_sigma=None,
        band_name="b0",
    )
    image = render_field_observation(
        truth_model, template, num_spectral_samples=1
    ).detach()
    image_with_nan = image.clone()
    image_with_nan[0, 0] = float("nan")
    weight = torch.ones_like(image)
    mask = torch.ones_like(image)
    mask[0, 0] = 0.0
    observations = [
        FieldObservation(
            image=image_with_nan,
            header=header,
            weight=weight,
            mask=mask,
            spectral_min=template.spectral_min,
            spectral_max=template.spectral_max,
            transmission_center=template.transmission_center,
            transmission_sigma=template.transmission_sigma,
            band_name=template.band_name,
        )
    ]
    fit_domain = estimate_field_domain(observations)
    model = SineImplicitField(fit_domain, hidden_dim=24, hidden_layers=2)
    fit = fit_field_model(
        observations,
        model,
        steps=4,
        lr=1.0e-3,
        num_spectral_samples=1,
        loss_reduction="global",
    )
    pred = render_field_observation(fit.model, template, num_spectral_samples=1)

    assert torch.isfinite(torch.tensor(fit.loss_history)).all()
    assert torch.isfinite(pred).all()


def test_fit_field_model_supports_band_balanced_patch_training() -> None:
    torch.manual_seed(2234)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("sky_lambda_feature_field", truth_domain)
    header = _tan_header(24, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    wavelengths = (4800.0, 4800.0, 6200.0, 6200.0, 7600.0, 7600.0)
    observations = []
    for idx, wavelength in enumerate(wavelengths):
        template = FieldObservation(
            image=torch.zeros((24, 24), dtype=torch.float32),
            header=header,
            spectral_min=wavelength,
            spectral_max=wavelength,
            transmission_center=wavelength,
            transmission_sigma=None,
            band_name=f"b{idx // 2}",
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
    model = make_field_model("sky_lambda_feature_field", fit_domain)
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
        training_patch_size=16,
    )
    assert math.isfinite(fit.final_loss)
    assert fit.final_loss < 5.0e-3


def test_noise_pretraining_updates_trunk_but_restores_decoder_head() -> None:
    torch.manual_seed(3210)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
    )
    truth_model = make_field_model(
        "sky_encoder_mlp",
        truth_domain,
        hidden_dim=48,
        hidden_layers=2,
        num_fourier_frequencies=4,
        spectral_frequencies=3,
    )
    header = _tan_header(20, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    templates = [
        FieldObservation(
            image=torch.zeros((20, 20), dtype=torch.float32),
            header=header,
            spectral_min=4700.0,
            spectral_max=4700.0,
            transmission_center=4700.0,
            transmission_sigma=None,
            band_name="b0",
        ),
        FieldObservation(
            image=torch.zeros((20, 20), dtype=torch.float32),
            header=header,
            spectral_min=6500.0,
            spectral_max=6500.0,
            transmission_center=6500.0,
            transmission_sigma=None,
            band_name="b1",
        ),
    ]
    observations = []
    for template in templates:
        image = render_field_observation(
            truth_model,
            template,
            num_spectral_samples=1,
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
    fit_domain = estimate_field_domain(observations)

    torch.manual_seed(777)
    model = make_field_model(
        "sky_encoder_mlp",
        fit_domain,
        hidden_dim=48,
        hidden_layers=2,
        num_fourier_frequencies=4,
        spectral_frequencies=3,
    )
    initial_hidden_weight = model.hidden[0].weight.detach().clone()
    initial_output_weight = model.output.weight.detach().clone()
    initial_output_bias = model.output.bias.detach().clone()
    _noise_pretrain_field_model(
        observations,
        model,
        noise_kind="gaussian",
        steps=12,
        lr=5.0e-3,
        observations_per_step=2,
        pixel_samples_per_observation=128,
        num_terms=32,
        max_frequency=6.0,
    )

    assert not torch.allclose(model.hidden[0].weight, initial_hidden_weight)
    assert torch.allclose(model.output.weight, initial_output_weight)
    assert torch.allclose(model.output.bias, initial_output_bias)


def test_fit_field_model_supports_noise_pretraining_for_sky_encoder_mlp() -> None:
    torch.manual_seed(3210)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
    )
    truth_model = make_field_model(
        "sky_encoder_mlp",
        truth_domain,
        hidden_dim=48,
        hidden_layers=2,
        num_fourier_frequencies=4,
        spectral_frequencies=3,
    )
    header = _tan_header(20, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    template = FieldObservation(
        image=torch.zeros((20, 20), dtype=torch.float32),
        header=header,
        spectral_min=5600.0,
        spectral_max=5600.0,
        transmission_center=5600.0,
        transmission_sigma=None,
        band_name="b0",
    )
    image = render_field_observation(
        truth_model,
        template,
        num_spectral_samples=1,
    ).detach()
    observations = [
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
    ]
    fit_domain = estimate_field_domain(observations)
    torch.manual_seed(777)
    model = make_field_model(
        "sky_encoder_mlp",
        fit_domain,
        hidden_dim=48,
        hidden_layers=2,
        num_fourier_frequencies=4,
        spectral_frequencies=3,
    )
    fit = fit_field_model(
        observations,
        model,
        steps=6,
        lr=5.0e-3,
        num_spectral_samples=1,
        loss_kind="mse",
        loss_reduction="global",
        noise_pretrain_kind="gaussian",
        noise_pretrain_steps=24,
        noise_pretrain_observations_per_step=2,
        noise_pretrain_pixel_samples_per_observation=128,
        noise_pretrain_num_terms=32,
        noise_pretrain_max_frequency=6.0,
    )

    assert fit.final_loss < fit.initial_loss


def test_fit_field_model_supports_streaming_band_balanced_patch_training() -> None:
    torch.manual_seed(2235)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("sky_lambda_feature_field", truth_domain)
    header = _tan_header(24, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    wavelengths = (4700.0, 4700.0, 6200.0, 6200.0, 7800.0, 7800.0)
    observations = []
    for idx, wavelength in enumerate(wavelengths):
        template = FieldObservation(
            image=torch.zeros((24, 24), dtype=torch.float32),
            header=header,
            spectral_min=wavelength,
            spectral_max=wavelength,
            transmission_center=wavelength,
            transmission_sigma=None,
            band_name=f"b{idx // 2}",
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
    model = make_field_model("sky_lambda_feature_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=8,
        lr=2.0e-3,
        num_spectral_samples=1,
        loss_kind="mse",
        loss_reduction="band_global_mix",
        observations_per_step=6,
        observation_sampling="band_balanced",
        training_patch_size=12,
        streaming_accumulation=True,
        full_observations_per_step=1,
    )
    assert fit.final_loss < fit.initial_loss


def test_fit_field_model_supports_shared_sky_patch_sampling() -> None:
    torch.manual_seed(2237)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("sky_lambda_feature_field", truth_domain)
    headers = (
        _tan_header(24, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        _tan_header(20, crval1=150.0004, crval2=1.9998, scale_deg=1.3 / 3600.0),
        _tan_header(28, crval1=149.9997, crval2=2.0002, scale_deg=0.9 / 3600.0),
    )
    wavelengths = (4700.0, 4700.0, 6200.0, 6200.0, 7800.0, 7800.0)
    observations = []
    for idx, wavelength in enumerate(wavelengths):
        header = headers[idx % len(headers)]
        template = FieldObservation(
            image=torch.zeros(
                (int(header["NAXIS2"]), int(header["NAXIS1"])), dtype=torch.float32
            ),
            header=header,
            spectral_min=wavelength,
            spectral_max=wavelength,
            transmission_center=wavelength,
            transmission_sigma=None,
            psf_sigma_pix=0.5,
            band_name=f"b{idx // 2}",
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
                psf_sigma_pix=template.psf_sigma_pix,
                band_name=template.band_name,
            )
        )
    fit_domain = estimate_field_domain(observations, coord_mode="sphere_xyz")
    model = make_field_model("sky_lambda_feature_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=8,
        lr=2.0e-3,
        num_spectral_samples=1,
        loss_kind="mse",
        loss_reduction="global",
        observations_per_step=6,
        observation_sampling="band_balanced",
        training_patch_size=12,
        shared_sky_patch_sampling=True,
        spatial_sampling="gauss_legendre_2x2",
    )
    assert fit.final_loss < fit.initial_loss


def test_fit_field_model_supports_pixel_sampled_training() -> None:
    torch.manual_seed(2238)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("spectral_basis_field", truth_domain)
    headers = (
        _tan_header(24, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        _tan_header(20, crval1=150.0003, crval2=2.0001, scale_deg=1.2 / 3600.0),
        _tan_header(28, crval1=149.9998, crval2=1.9997, scale_deg=0.9 / 3600.0),
    )
    wavelengths = (4700.0, 4700.0, 6200.0, 6200.0, 7800.0, 7800.0)
    observations = []
    for idx, wavelength in enumerate(wavelengths):
        header = headers[idx % len(headers)]
        template = FieldObservation(
            image=torch.zeros(
                (int(header["NAXIS2"]), int(header["NAXIS1"])), dtype=torch.float32
            ),
            header=header,
            spectral_min=wavelength,
            spectral_max=wavelength,
            transmission_center=wavelength,
            transmission_sigma=None,
            band_name=f"b{idx // 2}",
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
    model = make_field_model("spectral_basis_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=8,
        lr=2.0e-3,
        num_spectral_samples=1,
        loss_kind="mse",
        loss_reduction="global",
        observations_per_step=6,
        observation_sampling="band_balanced",
        pixel_samples_per_observation=64,
    )
    assert math.isfinite(fit.final_loss)
    assert fit.final_loss < 1.0e-4


def test_spectral_basis_field_render_bandpass_respects_weights() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
        coord_mode="sphere_xyz",
    )
    torch.manual_seed(2240)
    model = make_field_model("spectral_basis_field", domain)
    ra = torch.full((3, 3), 150.0, dtype=torch.float32)
    dec = torch.full((3, 3), 2.0, dtype=torch.float32)
    wavelengths = torch.tensor([5000.0, 7000.0], dtype=torch.float32)
    first_only = model.render_bandpass(
        ra,
        dec,
        wavelengths,
        torch.tensor([1.0, 0.0], dtype=torch.float32),
    )
    second_only = model.render_bandpass(
        ra,
        dec,
        wavelengths,
        torch.tensor([0.0, 1.0], dtype=torch.float32),
    )
    mixed = model.render_bandpass(
        ra,
        dec,
        wavelengths,
        torch.tensor([0.5, 0.5], dtype=torch.float32),
    )

    assert not torch.allclose(first_only, second_only)
    assert not torch.allclose(mixed, first_only)
    assert not torch.allclose(mixed, second_only)


def test_fit_field_model_supports_downsampled_context_training() -> None:
    torch.manual_seed(2236)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    truth_model = make_field_model("sky_lambda_feature_field", truth_domain)
    header = _tan_header(24, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    wavelengths = (4700.0, 6200.0, 7800.0)
    observations = []
    for wavelength in wavelengths:
        template = FieldObservation(
            image=torch.zeros((24, 24), dtype=torch.float32),
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
    model = make_field_model("sky_lambda_feature_field", fit_domain)
    fit = fit_field_model(
        observations,
        model,
        steps=8,
        lr=2.0e-3,
        num_spectral_samples=1,
        loss_kind="mse",
        loss_reduction="global",
        observations_per_step=1,
        training_patch_size=12,
        context_observations_per_step=1,
        context_downsample_factor=4,
        context_loss_weight=0.25,
    )
    assert fit.final_loss < fit.initial_loss


def test_fit_field_model_supports_spatial_smoothness_regularization() -> None:
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
    template = FieldObservation(
        image=torch.zeros((16, 16), dtype=torch.float32),
        header=header,
        spectral_min=4700.0,
        spectral_max=5050.0,
        transmission_center=4840.0,
        transmission_sigma=100.0,
    )
    image = render_field_observation(
        truth_model, template, num_spectral_samples=5
    ).detach()
    observations = [
        FieldObservation(
            image=image,
            header=template.header,
            weight=torch.ones_like(image),
            spectral_min=template.spectral_min,
            spectral_max=template.spectral_max,
            transmission_center=template.transmission_center,
            transmission_sigma=template.transmission_sigma,
        )
    ]
    fit_domain = estimate_field_domain(observations)
    model = SineImplicitField(fit_domain, hidden_dim=24, hidden_layers=2)
    fit = fit_field_model(
        observations,
        model,
        steps=8,
        lr=1.0e-2,
        num_spectral_samples=5,
        spatial_smoothness_weight=1.0e-3,
        spatial_smoothness_grid_size=6,
    )
    assert len(fit.loss_history) == 8
    assert torch.isfinite(torch.tensor(fit.final_loss))


def test_fit_field_model_supports_bandpass_mc_training() -> None:
    torch.manual_seed(2468)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1200.0,
    )
    truth_model = SineImplicitField(truth_domain, hidden_dim=24, hidden_layers=2)
    header = _tan_header(16, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    templates = [
        FieldObservation(
            image=torch.zeros((16, 16), dtype=torch.float32),
            header=header,
            spectral_min=4700.0,
            spectral_max=5300.0,
            transmission_center=5000.0,
            transmission_sigma=120.0,
            band_name="blue",
        ),
        FieldObservation(
            image=torch.zeros((16, 16), dtype=torch.float32),
            header=header,
            spectral_min=6100.0,
            spectral_max=6900.0,
            transmission_center=6500.0,
            transmission_sigma=180.0,
            band_name="red",
        ),
    ]
    observations = []
    for template in templates:
        image = render_field_observation(
            truth_model,
            template,
            num_spectral_samples=7,
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
    fit_domain = estimate_field_domain(observations)
    model = SineImplicitField(fit_domain, hidden_dim=24, hidden_layers=2)
    fit = fit_field_model(
        observations,
        model,
        steps=8,
        lr=1.0e-2,
        num_spectral_samples=5,
        training_spectral_sampling="bandpass_mc",
        training_spectral_sampling_grid_size=33,
    )
    pred = render_field_observation(
        fit.model,
        observations[0],
        num_spectral_samples=7,
    )
    assert fit.final_loss < fit.initial_loss
    assert pred.shape == observations[0].image.shape
    assert torch.isfinite(pred).all()


def test_fit_field_model_supports_mixture_mc_training() -> None:
    torch.manual_seed(8642)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5800.0,
        wavelength_scale=1400.0,
    )
    truth_model = SineImplicitField(truth_domain, hidden_dim=24, hidden_layers=2)
    header = _tan_header(16, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    templates = [
        FieldObservation(
            image=torch.zeros((16, 16), dtype=torch.float32),
            header=header,
            spectral_min=4500.0,
            spectral_max=5400.0,
            transmission_center=4950.0,
            transmission_sigma=160.0,
            band_name="g",
        ),
        FieldObservation(
            image=torch.zeros((16, 16), dtype=torch.float32),
            header=header,
            spectral_min=5600.0,
            spectral_max=7000.0,
            transmission_center=6300.0,
            transmission_sigma=240.0,
            band_name="r",
        ),
    ]
    observations = []
    for template in templates:
        image = render_field_observation(
            truth_model,
            template,
            num_spectral_samples=7,
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
    fit_domain = estimate_field_domain(observations)
    model = SineImplicitField(fit_domain, hidden_dim=24, hidden_layers=2)
    fit = fit_field_model(
        observations,
        model,
        steps=8,
        lr=1.0e-2,
        num_spectral_samples=5,
        training_spectral_sampling="mixture_mc",
        training_spectral_sampling_grid_size=33,
        spectral_anchor_observations_per_step=1,
        spectral_anchor_num_samples=11,
        spectral_anchor_loss_weight=0.1,
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
    assert samples.device == device
    assert weights.device == device
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
