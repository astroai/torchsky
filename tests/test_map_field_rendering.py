from __future__ import annotations
import math
import pytest
import torch
from conftest import assert_close_on_device
from torchsky import WCS
from torchsky.map.field import (
    FieldDomain,
    FieldObservation,
    ForwardContext,
    PSFSpec,
    SineImplicitField,
    fit_field_model,
    render_field_observation,
    render_field_observation_distribution,
)
from torchsky.map.field.core import _BaseFieldModel
from torchsky.map.field.observation_ops import (
    coordinate_grid,
    crop_field_observation,
    crop_field_observation_for_training,
)
from torchsky.map.field import rendering as field_rendering
from torchsky.map.operators import (
    AdaptiveSpatioSpectralTemporalEncoder,
    BackgroundCalibrationTerm,
    ConditionedBackgroundField,
    ConditionedKernelPSF,
    LearnedBackgroundOp,
    LearnedPSFOp,
    ObservationOperatorBundle,
    apply_kernel2d_variance,
    gaussian_kernel1d,
)
from map_field_support import (
    _ConstantField,
    _GaussianSpotField,
    _tan_header,
)


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


def test_fit_field_model_rejects_amp_off_cuda() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.01,
        wavelength_center=5000.0,
        wavelength_scale=500.0,
    )
    observation = FieldObservation(
        image=torch.ones((4, 4), dtype=torch.float32),
        header=_tan_header(4, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        weight=torch.ones((4, 4), dtype=torch.float32),
        spectral_min=4500.0,
        spectral_max=5500.0,
        transmission_center=5000.0,
    )

    with pytest.raises(ValueError, match="CUDA"):
        fit_field_model(
            [observation],
            _ConstantField(domain, 0.0),
            steps=1,
            device="cpu",
            amp_dtype="float16",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fit_field_model_amp_skips_overflowing_step() -> None:
    class OverflowField(_BaseFieldModel):
        def __init__(self, domain: FieldDomain) -> None:
            super().__init__(domain)
            self.linear = torch.nn.Linear(3, 1)
            torch.nn.init.constant_(self.linear.weight, 0.1)
            torch.nn.init.constant_(self.linear.bias, 0.1)

        def forward(self, ra_deg, dec_deg, wavelength_angstrom):
            wavelength = torch.as_tensor(
                wavelength_angstrom, device=ra_deg.device, dtype=ra_deg.dtype
            ).expand_as(ra_deg)
            features = torch.stack(
                (
                    (ra_deg - 150.0) / 0.01,
                    (dec_deg - 2.0) / 0.01,
                    wavelength / 5000.0,
                ),
                dim=-1,
            ).to(self.linear.weight.dtype)
            return self.linear(features).squeeze(-1) * 1.0e6

    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.01,
        wavelength_center=5000.0,
        wavelength_scale=500.0,
    )
    model = OverflowField(domain)
    initial = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    observation = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=_tan_header(4, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        weight=torch.ones((4, 4), dtype=torch.float32),
        spectral_min=4500.0,
        spectral_max=5500.0,
        transmission_center=5000.0,
    )

    result = fit_field_model(
        [observation],
        model,
        steps=1,
        lr=1.0e-3,
        grad_clip_norm=1.0,
        device="cuda",
        amp_dtype="float16",
    )

    assert result.diagnostics is not None
    assert result.diagnostics["amp_dtype"] == "float16"
    assert result.diagnostics["amp_scaler_enabled"] is True
    assert result.diagnostics["amp_overflow_steps"] == 1
    assert result.amp_scaler_state is not None
    for name, parameter in result.model.named_parameters():
        assert_close_on_device(parameter.cpu(), initial[name])


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

    assert_close_on_device(implicit, torch.full((4, 4), 5.0))
    assert_close_on_device(explicit, torch.full((4, 4), 2.0))


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

    assert_close_on_device(cropped.image, values[1:4, 2:6])
    assert_close_on_device(cropped.sky_background, values[1:4, 2:6] + 100.0)
    assert_close_on_device(cropped.instrument_background, values[1:4, 2:6] + 200.0)
    assert_close_on_device(cropped.detector_background, values[1:4, 2:6] + 300.0)
    assert cropped.detector_x0_pix == 2.0
    assert cropped.detector_y0_pix == 1.0


def test_native_detector_transform_uses_full_detector_coordinates_after_crop() -> None:
    class LinearDetectorSky(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shift = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))

        def pixel_to_world(
            self, x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return 150.0 + (x + self.shift) / 3600.0, 2.0 + y / 3600.0

    transform = LinearDetectorSky()
    observation = FieldObservation(
        image=torch.zeros((5, 6)),
        header=_tan_header(6, crval1=10.0, crval2=-20.0, scale_deg=0.01),
        wavelength=5600.0,
        detector_x0_pix=100.0,
        detector_y0_pix=200.0,
        detector_to_sky=transform,
    )
    cropped = crop_field_observation(observation, x0=3, y0=2, width=2, height=2)
    ra_deg, dec_deg = coordinate_grid(cropped, shape=(2, 2), device=torch.device("cpu"))

    assert_close_on_device(
        ra_deg[0, 0], torch.tensor(150.0 + 103.25 / 3600.0, dtype=torch.float64)
    )
    assert_close_on_device(
        dec_deg[0, 0], torch.tensor(2.0 + 202.0 / 3600.0, dtype=torch.float64)
    )
    ra_deg.sum().backward()
    assert transform.shift.grad is not None
    assert float(transform.shift.grad) > 0.0


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
    assert_close_on_device(halo_render[halo.mask], expected.reshape(-1))

    model.zero_grad(set_to_none=True)
    expected.sum().backward()
    assert model.value.grad is not None
    full_gradient = model.value.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    render_field_observation(model, halo, num_spectral_samples=1)[
        halo.mask
    ].sum().backward()
    assert_close_on_device(model.value.grad, full_gradient)

    edge_halo = crop_field_observation_for_training(
        observation,
        x0=0,
        y0=0,
        width=width,
        height=height,
    )
    edge_render = render_field_observation(model, edge_halo, num_spectral_samples=1)
    assert edge_halo.mask is not None
    assert_close_on_device(
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


def test_training_crop_accepts_learned_psf_support_override() -> None:
    observation = FieldObservation(
        image=torch.zeros((20, 20), dtype=torch.float32),
        header=_tan_header(20, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        wavelength=5600.0,
        detector_x0_pix=100.0,
        detector_y0_pix=200.0,
    )

    cropped = crop_field_observation_for_training(
        observation,
        x0=8,
        y0=8,
        width=4,
        height=3,
        psf_halo=(3, 4),
    )

    assert cropped.image.shape == (9, 12)
    assert cropped.mask is not None
    assert int(torch.count_nonzero(cropped.mask)) == 12
    assert cropped.detector_x0_pix == 104.0
    assert cropped.detector_y0_pix == 205.0


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

    assert_close_on_device(center, torch.full((9, 9), 5.0))
    assert_close_on_device(quadrature, center)
    assert_close_on_device(tiled, quadrature)
    quadrature.sum().backward()
    assert_close_on_device(model.value.grad, torch.tensor(81.0))


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
    dense_offsets = (torch.arange(48, dtype=torch.float64) + 0.5) / 48.0 - 0.5
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
        dense = (
            model(dense_ra, dense_dec, 5600.0)
            .reshape(48, 48, size, size)
            .mean(dim=(0, 1))
        )
        center_error += float(torch.mean(torch.abs(center - dense)).detach())
        quadrature_error += float(torch.mean(torch.abs(quadrature - dense)).detach())

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


def test_render_field_observation_distribution_rejects_missing_uncertainty() -> None:
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

    with pytest.raises(ValueError, match="explicit predictive distribution"):
        render_field_observation_distribution(
            model,
            observation,
            num_spectral_samples=1,
        )
    with pytest.raises(ValueError, match="explicit predictive distribution"):
        render_field_observation_distribution(
            model,
            observation,
            num_spectral_samples=1,
            spatial_sampling="gauss_legendre_2x2",
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
    kernel = gaussian_kernel1d(1.0, device=torch.device("cpu"), dtype=torch.float32)
    # Default variance_mode is coherent: std propagates linearly through the PSF.
    expected_center_variance = 3.0 * kernel.sum() ** 2
    assert_close_on_device(torch.exp(full_logvar[5, 5]), expected_center_variance)
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
    assert_close_on_device(tiled_mean, full_mean)
    assert_close_on_device(tiled_logvar, full_logvar)
    assert observation._cached_ra_deg is None
    assert observation._cached_dec_deg is None
    tiled_logvar.sum().backward()
    assert_close_on_device(model.log_variance.grad, full_gradient)


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
    assert_close_on_device(propagated[4, 4], torch.square(normalized).sum())
    propagated.sum().backward()
    assert variance.grad is not None
    assert bool(torch.isfinite(variance.grad).all())


def test_conditioned_psf_background_render_is_tiled_exact_and_contextual() -> None:
    torch.manual_seed(73)
    size = 17
    scale_deg = 1.0 / 3600.0
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=7000.0,
        wavelength_scale=1000.0,
    )
    model = _GaussianSpotField(
        domain,
        amplitude=2.0,
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        sigma_deg=0.8 * scale_deg,
    )
    psf_encoder = AdaptiveSpatioSpectralTemporalEncoder(
        spatial_center=(8.0, 8.0),
        spatial_scale=8.0,
        spatial_length_bounds=(2.0, 34.0),
        wavelength_center=7000.0,
        spectral_scale=1000.0,
        spectral_length_bounds=(100.0, 2000.0),
        time_center_mjd=60000.0,
        temporal_scale_days=20.0,
        temporal_length_bounds_days=(0.5, 80.0),
        spatial_bands=2,
        spectral_bands=2,
        temporal_bands=1,
    )
    background_encoder = AdaptiveSpatioSpectralTemporalEncoder(
        spatial_center=(8.0, 8.0),
        spatial_scale=16.0,
        spatial_length_bounds=(8.0, 68.0),
        wavelength_center=7000.0,
        spectral_scale=500.0,
        spectral_length_bounds=(50.0, 3000.0),
        time_center_mjd=60000.0,
        temporal_scale_days=40.0,
        temporal_length_bounds_days=(2.0, 160.0),
        spatial_bands=2,
        spectral_bands=3,
        temporal_bands=2,
    )
    psf_model = ConditionedKernelPSF(
        psf_encoder,
        n_basis=3,
        kernel_size=7,
        init_fwhm_pix=2.0,
        hidden_dim=12,
    )
    background_model = ConditionedBackgroundField(
        background_encoder,
        hidden_dim=12,
        hidden_layers=1,
    )
    with torch.no_grad():
        psf_model.mixer[-1].weight.normal_(std=0.15)
        background_model.network[-1].weight.normal_(std=0.01)
    bundle = ObservationOperatorBundle(
        psf=LearnedPSFOp(psf_model),
        background=LearnedBackgroundOp(background_model, coordinate_frame="detector"),
    )
    observation = FieldObservation(
        image=torch.zeros((size, size), dtype=torch.float32),
        header=_tan_header(size, crval1=150.0, crval2=2.0, scale_deg=scale_deg),
        spectral_min=6500.0,
        spectral_max=7500.0,
        transmission_center=7000.0,
        transmission_sigma=250.0,
        detector_x0_pix=100.0,
        detector_y0_pix=200.0,
    )
    context = ForwardContext(time_mjd=60010.0)

    with pytest.raises(ValueError, match="covariance propagation"):
        render_field_observation_distribution(
            model,
            observation,
            num_spectral_samples=3,
            forward_context=context,
            operator_bundle=bundle,
        )

    full = render_field_observation(
        model,
        observation,
        num_spectral_samples=3,
        forward_context=context,
        operator_bundle=bundle,
    )
    full.square().sum().backward()
    full_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in (
            *model.named_parameters(),
            *((f"psf.{name}", value) for name, value in psf_model.named_parameters()),
            *(
                (f"bg.{name}", value)
                for name, value in background_model.named_parameters()
            ),
        )
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    psf_model.zero_grad(set_to_none=True)
    background_model.zero_grad(set_to_none=True)
    tiled = render_field_observation(
        model,
        observation,
        num_spectral_samples=3,
        forward_context=context,
        operator_bundle=bundle,
        spatial_tile_hw=(6, 5),
    )
    assert_close_on_device(tiled, full, rtol=2e-5, atol=2e-6)
    tiled.square().sum().backward()
    tiled_gradients = {
        name: parameter.grad
        for name, parameter in (
            *model.named_parameters(),
            *((f"psf.{name}", value) for name, value in psf_model.named_parameters()),
            *(
                (f"bg.{name}", value)
                for name, value in background_model.named_parameters()
            ),
        )
        if parameter.grad is not None
    }
    assert full_gradients.keys() == tiled_gradients.keys()
    for name, expected in full_gradients.items():
        assert_close_on_device(tiled_gradients[name], expected, rtol=2e-4, atol=2e-5)

    later = render_field_observation(
        model,
        observation,
        num_spectral_samples=3,
        forward_context=ForwardContext(time_mjd=60030.0),
        operator_bundle=bundle,
        spatial_tile_hw=(6, 5),
    )
    assert not torch.allclose(later, tiled)
    assert not torch.equal(
        psf_encoder.spectral_axis.length_scales,
        background_encoder.spectral_axis.length_scales,
    )


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.needs_float64
def test_fit_field_model_optimizes_conditioned_background_bundle(
    streaming: bool,
) -> None:
    torch.manual_seed(74)
    size = 9
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=7000.0,
        wavelength_scale=1000.0,
    )
    field_model = _ConstantField(domain, 2.0)
    encoder = AdaptiveSpatioSpectralTemporalEncoder(
        spatial_center=(4.0, 4.0),
        spatial_scale=4.0,
        spatial_length_bounds=(2.0, 18.0),
        wavelength_center=7000.0,
        spectral_scale=1000.0,
        spectral_length_bounds=(100.0, 4000.0),
        time_center_mjd=60000.0,
        temporal_scale_days=10.0,
        temporal_length_bounds_days=(0.25, 40.0),
        spatial_bands=2,
        spectral_bands=0,
        temporal_bands=0,
    )
    background_model = ConditionedBackgroundField(
        encoder, hidden_dim=16, hidden_layers=1
    )
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    target = 2.0 + 0.15 * (x.to(dtype=torch.float32) - 4.0) / 4.0
    observation = FieldObservation(
        image=target,
        header=_tan_header(size, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0),
        wavelength=7000.0,
    )
    bundle = ObservationOperatorBundle(
        background=LearnedBackgroundOp(background_model, coordinate_frame="detector")
    )
    calibration_term = BackgroundCalibrationTerm(
        background_model,
        detector_x=torch.tensor([1.0, 4.0, 7.0]),
        detector_y=torch.tensor([4.0, 4.0, 4.0]),
        wavelength=torch.full((3,), 7000.0),
        time_mjd=torch.full((3,), 60000.0),
        target_values=torch.tensor([-0.1125, 0.0, 0.1125]),
        loss_weight=1.0,
    )

    result = fit_field_model(
        [observation],
        field_model,
        steps=30,
        lr=3.0e-2,
        num_spectral_samples=1,
        observation_operator_bundles=[bundle],
        operator_calibration_terms=[calibration_term],
        observation_forward_contexts=[ForwardContext(time_mjd=60000.0)],
        operator_scale_gate_weight=1.0e-4,
        operator_scale_shift_weight=1.0e-4,
        streaming_accumulation=streaming,
    )

    assert result.final_loss < 0.2 * result.initial_loss
    assert background_model.network[-1].weight.grad is not None
    assert encoder.x_axis.gate_logits.grad is not None
    assert encoder.y_axis.gate_logits.grad is not None
    assert result.observation_operator_bundles == (bundle,)
    assert result.diagnostics is not None
    operator_report = result.diagnostics["operator_scale_diagnostics"][0]
    assert operator_report["operator"] == "background"
    scale_report = operator_report["axes"]
    assert scale_report["spatial_x"]["length_bounds"] == [2.0, 18.0]
    assert len(scale_report["spatial_x"]["gates"]) == 2
    calibration_report = result.diagnostics["operator_calibration_diagnostics"][0]
    assert calibration_report["kind"] == "background_value_mse"
