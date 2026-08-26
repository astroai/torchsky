"""Scientific invariants for the shared native-observation foundation."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from torchsky.map.field.implicit_forward import observation_charbonnier_loss
from torchsky.map.field.types import FieldObservation, ForwardContext, PSFSpec
from torchsky.map.operators.noise import (
    StructuredDetectorNoise,
    reduce_detector_nll,
)
from torchsky.map.operators.radiometry import (
    expected_electrons_from_photon_spectral_density,
)
from torchsky.map.operators.resampling import (
    sample_image_plane_batched_channels,
    sample_image_plane_channels,
)
from torchsky.map.product import MapBandpassSpec, MapProductSpec
from torchsky.map.scenes import target_to_field_observation
from torchsky.map.splat import splat_bilinear
from torchsky.spectrophotometry import Passband


def _header(height: int = 3, width: int = 4) -> dict[str, object]:
    return {
        "NAXIS": 2,
        "NAXIS1": width,
        "NAXIS2": height,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 2.0,
        "CRPIX2": 2.0,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CD1_1": -1.0 / 3600.0,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 1.0 / 3600.0,
    }


def _passband() -> Passband:
    return Passband(
        torch.tensor([5000.0, 5100.0, 5200.0]),
        torch.tensor([0.1, 1.0, 0.1]),
        name="test",
    )


def _certified_observation() -> tuple[FieldObservation, ForwardContext]:
    image = torch.ones((3, 4), dtype=torch.float32)
    mask = torch.ones_like(image, dtype=torch.bool)
    mask[0, 0] = False
    weight = torch.full_like(image, 2.0)
    weight[~mask] = 0.0
    observation = FieldObservation(
        image=image,
        header=_header(),
        weight=weight,
        mask=mask,
        amplifier_index=torch.zeros_like(image, dtype=torch.int64),
        passband=_passband(),
        psf_spec=PSFSpec.kernel(torch.ones((3, 3)) / 9.0),
        spectral_quantity="wavelength",
        spectral_unit="Angstrom",
        value_unit="electron",
        value_semantics="detector_counts",
        detector_law="gaussian",
        gain_electrons_per_adu=torch.full_like(image, 1.5),
        gain_provenance_id="gain-v1",
        exposure_time_s=30.0,
        collecting_area_m2=10.0,
    )
    context = ForwardContext(
        survey="survey",
        instrument="camera",
        detector="detector-1",
        exposure_id="exposure-1",
        time_mjd=60000.0,
        time_scale="utc",
        calibration_id="cal-v1",
        provenance_id="raw-sha256",
    )
    return observation, context


def test_certified_observation_moves_without_losing_semantic_dtypes() -> None:
    observation, context = _certified_observation()
    assert observation.validate_certified(context) is observation

    moved = observation.to(dtype=torch.float64)
    assert moved.image.dtype == torch.float64
    assert moved.weight is not None and moved.weight.dtype == torch.float64
    assert moved.mask is not None and moved.mask.dtype == torch.bool
    assert moved.amplifier_index is not None
    assert moved.amplifier_index.dtype == torch.int64
    assert (
        moved.passband is not None and moved.passband.wavelength.dtype == torch.float64
    )
    assert moved.psf_spec is not None and moved.psf_spec.kernel2d is not None
    assert moved.psf_spec.kernel2d.dtype == torch.float64
    assert moved.validate_certified(context) is moved


def test_certified_observation_rejects_ambiguous_mask_and_ivar() -> None:
    observation, context = _certified_observation()
    with pytest.raises(TypeError, match="mask must be boolean"):
        replace(
            observation,
            mask=observation.mask.to(dtype=torch.float32),
        ).validate_certified(context)

    assert observation.weight is not None and observation.mask is not None
    bad_weight = observation.weight.clone()
    bad_weight[~observation.mask] = 1.0
    with pytest.raises(ValueError, match="zero inverse variance"):
        replace(observation, weight=bad_weight).validate_certified(context)


def test_passband_propagates_from_map_product_to_target() -> None:
    passband = _passband()
    product = MapProductSpec(
        target_header=_header(),
        bands=(MapBandpassSpec(label="test", passband=passband),),
    )
    target = product.target_render_spec()
    assert target.passband is passband
    assert target.spectral_unit == "Angstrom"
    observation = target_to_field_observation(target)
    assert observation.passband is passband
    assert observation.value_semantics == "surface_brightness"
    assert observation.spectral_unit == "Angstrom"


def test_passband_rejects_non_wavelength_units() -> None:
    with pytest.raises(ValueError, match="Angstrom"):
        FieldObservation(
            image=torch.ones((2, 2)),
            header=_header(height=2, width=2),
            passband=_passband(),
            spectral_unit="Hz",
        )

    with pytest.raises(ValueError, match="mutually exclusive"):
        FieldObservation(
            image=torch.ones((2, 2)),
            header=_header(height=2, width=2),
            passband=_passband(),
            transmission_center=5100.0,
            transmission_sigma=50.0,
        )


def test_expected_electrons_is_nonuniform_grid_accurate_and_differentiable() -> None:
    density = torch.full((3,), 2.0, dtype=torch.float64, requires_grad=True)
    wavelength = torch.tensor([5000.0, 5001.0, 5003.0], dtype=torch.float64)
    expected = expected_electrons_from_photon_spectral_density(
        density,
        wavelength,
        torch.ones_like(wavelength),
        collecting_area_m2=4.0,
        exposure_time_s=5.0,
        spatial_measure=0.5,
    )
    torch.testing.assert_close(expected, torch.tensor(60.0, dtype=torch.float64))
    expected.backward()
    assert density.grad is not None and bool(torch.isfinite(density.grad).all())


def test_masked_detector_reductions_do_not_admit_nan_leakage() -> None:
    nll = torch.tensor([1.0, float("nan"), 3.0])
    mask = torch.tensor([True, False, True])
    torch.testing.assert_close(
        reduce_detector_nll(nll, mask=mask),
        torch.tensor(2.0),
    )
    torch.testing.assert_close(
        reduce_detector_nll(
            nll,
            mask=mask,
            sample_weight=torch.tensor([1.0, float("nan"), 2.0]),
            reduction="sum",
        ),
        torch.tensor(7.0),
    )

    prediction = torch.tensor([[0.0, float("nan")]], requires_grad=True)
    observation = FieldObservation(
        image=torch.tensor([[1.0, float("nan")]]),
        header=_header(height=1, width=2),
        weight=torch.tensor([[1.0, 0.0]]),
        mask=torch.tensor([[True, False]]),
        wavelength=5000.0,
    )
    loss = observation_charbonnier_loss(prediction, observation, eps=1.0e-3)
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert prediction.grad is not None
    torch.testing.assert_close(prediction.grad[0, 1], torch.tensor(0.0))


def test_structured_noise_ignores_nan_only_outside_valid_support() -> None:
    residual = torch.tensor([[0.1, float("nan")], [0.2, -0.1]])
    err = torch.tensor([[1.0, float("nan")], [1.0, 1.0]])
    valid = torch.tensor([[True, False], [True, True]])
    model = StructuredDetectorNoise(num_amplifiers=1)
    nll = model.negative_log_likelihood(residual, err=err, valid=valid)
    assert bool(torch.isfinite(nll))
    inlier, outlier = model.contamination_responsibilities(
        residual,
        err=err,
        valid=valid,
    )
    torch.testing.assert_close(inlier[~valid], torch.zeros(1))
    torch.testing.assert_close(outlier[~valid], torch.zeros(1))


@pytest.mark.parametrize("kernel", ["bilinear", "catmull_rom"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched_channel_resampling_matches_per_batch_and_backpropagates(
    kernel: str,
    dtype: torch.dtype,
) -> None:
    image = torch.arange(80, dtype=dtype).reshape(2, 2, 4, 5).requires_grad_()
    x = torch.stack(
        [
            torch.linspace(0.4, 3.6, 6, dtype=dtype).reshape(2, 3),
            torch.linspace(0.7, 3.2, 6, dtype=dtype).reshape(2, 3),
        ]
    )
    y = torch.stack(
        [
            torch.linspace(0.3, 2.7, 6, dtype=dtype).reshape(2, 3),
            torch.linspace(0.8, 2.2, 6, dtype=dtype).reshape(2, 3),
        ]
    )
    grid = torch.stack((2.0 * x / 4.0 - 1.0, 2.0 * y / 3.0 - 1.0), dim=-1)
    combined = sample_image_plane_batched_channels(
        image,
        x_src=x,
        y_src=y,
        grid=grid,
        kernel=kernel,
    )
    separate = torch.stack(
        [
            sample_image_plane_channels(
                image[index],
                x_src=x[index],
                y_src=y[index],
                grid=grid[index],
                kernel=kernel,
            )
            for index in range(2)
        ]
    )
    torch.testing.assert_close(combined, separate)
    combined.sum().backward()
    assert image.grad is not None and bool(torch.isfinite(image.grad).all())


def test_splat_reports_edge_loss_and_retains_coordinate_gradients() -> None:
    flux = torch.tensor([[[2.0]]], dtype=torch.float64, requires_grad=True)
    xy = torch.tensor(
        [[[[-0.25, 0.5]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    native, diagnostics = splat_bilinear(
        flux,
        xy,
        (2, 2),
        return_diagnostics=True,
    )
    torch.testing.assert_close(
        diagnostics.retained_weight,
        torch.tensor([[[0.75]]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        diagnostics.deposited_flux,
        torch.tensor([1.5], dtype=torch.float64),
    )
    torch.testing.assert_close(
        diagnostics.lost_flux,
        torch.tensor([0.5], dtype=torch.float64),
    )
    native.sum().backward()
    assert flux.grad is not None and bool(torch.isfinite(flux.grad).all())
    assert xy.grad is not None and bool(torch.isfinite(xy.grad).all())
