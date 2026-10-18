"""Bounded electron-unit, ordering, and exact covariance reference checks."""

import pytest
import torch

from torchsky.map.operators.detector_response import (
    LinearDetectorResponse,
    integrated_detector_moments,
    integrated_photoelectron_mean,
)


def _linear(matrix: torch.Tensor, shape: tuple[int, ...]) -> LinearDetectorResponse:
    return LinearDetectorResponse(
        lambda value: (matrix @ value.reshape(-1)).reshape(shape),
        lambda value: (matrix.T @ value.reshape(-1)).reshape(shape),
    )


def test_photons_route_before_collection_and_dark_bypasses_optics() -> None:
    source = torch.tensor([2.0, 5.0, 0.0], dtype=torch.float64)
    foreground = torch.tensor([1.0, 0.0, 4.0], dtype=source.dtype)
    optical = torch.tensor([[0.5, 0.1, 0.0], [0.0, 0.2, 0.25]], dtype=source.dtype)
    qe = torch.tensor([0.8, 0.6], dtype=source.dtype)
    photo = integrated_photoelectron_mean(
        source,
        foreground_photon_rate=foreground,
        optical_mean_response=lambda value: optical @ value,
        exposure_time_s=10.0,
        electrons_per_photon=qe,
    )
    torch.testing.assert_close(photo, torch.tensor([16.0, 12.0], dtype=source.dtype))
    pixel_moments = integrated_detector_moments(photo)
    actual_pixel_covariance = torch.stack(
        [pixel_moments.covariance_action(v) for v in torch.eye(2, dtype=source.dtype)],
        dim=1,
    )
    torch.testing.assert_close(actual_pixel_covariance, torch.diag(photo))
    # Optical Poisson routing/thinning is not deterministic charge mixing.
    deterministic_photon_map = torch.diag(qe * 10) @ optical
    wrong_covariance = (
        deterministic_photon_map
        @ torch.diag(source + foreground)
        @ deterministic_photon_map.T
    )
    assert not torch.allclose(actual_pixel_covariance, wrong_covariance)
    dark = torch.tensor([3.0, 7.0], dtype=source.dtype)
    charge = torch.tensor([[0.9, 0.2], [0.1, 0.8]], dtype=source.dtype)
    read_offset = torch.tensor([-2.0, 5.0], dtype=source.dtype)
    result = integrated_detector_moments(
        photo,
        dark_electron_mean=dark,
        charge_response=_linear(charge, (2,)),
        read_offset_electrons=read_offset,
    )
    torch.testing.assert_close(result.mean, charge @ (photo + dark) + read_offset)
    assert not torch.allclose(result.mean, charge @ photo + dark + read_offset)
    assert not torch.allclose(result.mean, charge @ (photo + dark + read_offset))


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_covariance_matches_independent_dense_poisson_and_gaussian_moments(
    dtype,
) -> None:
    shape = (1, 2, 2)
    photo = torch.tensor([0.0, 0.2, 4.0, 8.0], dtype=dtype).reshape(shape)
    dark = torch.tensor([1.0, 0.0, 2.0, 0.0], dtype=dtype).reshape(shape)
    charge = torch.tensor(
        [[0.8, 0.2, 0, 0], [0.2, 0.7, 0.1, 0], [0, 0.1, 0.8, 0.1], [0, 0, 0.1, 0.9]],
        dtype=dtype,
    )
    read = torch.tensor(
        [[1, 0, 0, 0], [0.3, 1, 0, 0], [0, 0, 1, 0.2], [0, 0, 0, 1]],
        dtype=dtype,
    )
    read_variance = torch.tensor([0.5, 0.2, 0.3, 0.8], dtype=dtype).reshape(shape)
    gain = torch.tensor([1.0, 2.0, 4.0, 3.0], dtype=dtype).reshape(shape)
    result = integrated_detector_moments(
        photo,
        dark_electron_mean=dark,
        charge_response=_linear(charge, shape),
        read_variance_electrons2=read_variance,
        read_noise_response=_linear(read, shape),
        electrons_per_output_unit=gain,
    )
    # Independent laws give variance in collected electron counts; a dense
    # transformation of their covariance is the small reference oracle.
    shot_variance = (
        torch.distributions.Poisson(photo).variance
        + torch.distributions.Poisson(dark).variance
    )
    gaussian_variance = torch.distributions.Normal(
        torch.zeros_like(read_variance), read_variance.sqrt()
    ).variance
    output_transform = torch.diag(gain.reshape(-1).reciprocal())
    reference = (
        output_transform
        @ (
            charge @ torch.diag(shot_variance.reshape(-1)) @ charge.T
            + read @ torch.diag(gaussian_variance.reshape(-1)) @ read.T
        )
        @ output_transform.T
    )
    basis = torch.eye(4, dtype=dtype)
    actual = torch.stack(
        [
            result.covariance_action(vector.reshape(shape)).reshape(-1)
            for vector in basis
        ],
        dim=1,
    )
    torch.testing.assert_close(actual, reference)
    assert actual[0, 1] > 0  # off-diagonal terms must survive
    wrong_diagonal = torch.diag(
        (charge @ shot_variance.reshape(-1)) / gain.reshape(-1).square()
    )
    assert not torch.allclose(actual, wrong_diagonal)
    torch.testing.assert_close(actual, actual.T)
    assert bool((torch.linalg.eigvalsh(actual) >= 0).all())
    v = torch.tensor([1.0, -3.0, 2.0, 5.0], dtype=dtype).reshape(shape)
    w = torch.tensor([-4.0, 0.3, 1.0, 2.0], dtype=dtype).reshape(shape)
    torch.testing.assert_close(
        (v * result.covariance_action(w)).sum(),
        (result.covariance_action(v) * w).sum(),
    )


def test_deterministic_subtraction_and_read_offsets_leave_shot_noise() -> None:
    photo = torch.tensor([0.3, 2.0], dtype=torch.float64)
    baseline = integrated_detector_moments(photo, dark_electron_mean=0.2)
    residual = integrated_detector_moments(
        photo,
        dark_electron_mean=0.2,
        read_offset_electrons=-3.0,
        subtracted_mean_output_units=7.0,
    )
    assert bool((residual.mean < 0).all())
    vector = torch.tensor([2.0, -1.0], dtype=photo.dtype)
    torch.testing.assert_close(residual.mean, baseline.mean - 10)
    torch.testing.assert_close(
        residual.covariance_action(vector), baseline.covariance_action(vector)
    )


@pytest.mark.parametrize("shape", ((), (3,), (2, 3), (1, 2, 3)))
def test_zero_flux_dark_read_and_identity_limits(shape) -> None:
    zero = torch.zeros(shape, dtype=torch.float64)
    photons = integrated_photoelectron_mean(zero, exposure_time_s=3.0)
    result = integrated_detector_moments(photons)
    torch.testing.assert_close(result.mean, zero)
    torch.testing.assert_close(result.covariance_action(torch.ones_like(zero)), zero)
    result = integrated_detector_moments(zero, dark_electron_mean=2.0)
    torch.testing.assert_close(result.mean, zero + 2)
    torch.testing.assert_close(
        result.covariance_action(torch.ones_like(zero)), zero + 2
    )
    read_only = integrated_detector_moments(zero, read_variance_electrons2=0.7)
    torch.testing.assert_close(read_only.mean, zero)
    torch.testing.assert_close(
        read_only.covariance_action(torch.ones_like(zero)), zero + 0.7
    )
    torch.testing.assert_close(
        integrated_photoelectron_mean(zero + 10, exposure_time_s=0.0), zero
    )


def test_mean_and_covariance_are_differentiable_in_physical_calibration() -> None:
    dtype = torch.float64
    source = torch.tensor([0.2, 1.1], dtype=dtype, requires_grad=True)
    qe = torch.tensor([0.4, 0.8], dtype=dtype, requires_grad=True)
    duration = torch.tensor(3.0, dtype=dtype, requires_grad=True)
    dark = torch.tensor([0.1, 0.5], dtype=dtype, requires_grad=True)
    read_var = torch.tensor([0.3, 0.2], dtype=dtype, requires_grad=True)
    gain = torch.tensor([1.2, 2.0], dtype=dtype, requires_grad=True)
    matrix = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=dtype, requires_grad=True)
    read_offset = torch.tensor([-1.0, 0.2], dtype=dtype, requires_grad=True)

    def evaluate(s, q, t, d, r, g, m, b):
        photo = integrated_photoelectron_mean(
            s, exposure_time_s=t, electrons_per_photon=q
        )
        result = integrated_detector_moments(
            photo,
            dark_electron_mean=d,
            charge_response=_linear(m, (2,)),
            read_variance_electrons2=r,
            electrons_per_output_unit=g,
            read_offset_electrons=b,
        )
        return result.mean, result.covariance_action(
            torch.tensor([1.0, -0.5], dtype=dtype)
        )

    assert torch.autograd.gradcheck(
        evaluate, (source, qe, duration, dark, read_var, gain, matrix, read_offset)
    )


@pytest.mark.parametrize("value", (-1.0, float("nan"), float("inf")))
@pytest.mark.parametrize(
    "name", ("photoelectron_mean", "dark_electron_mean", "read_variance_electrons2")
)
def test_invalid_physical_means_fail_closed(name, value) -> None:
    photo = torch.ones(2)
    if name == "photoelectron_mean":
        photo[0] = value
        kwargs = {}
    else:
        kwargs = {name: value}
    with pytest.raises(ValueError):
        integrated_detector_moments(photo, **kwargs)


@pytest.mark.parametrize("value", (0.0, -1.0, float("inf"), float("nan")))
def test_gain_invalid(value) -> None:
    with pytest.raises(ValueError):
        integrated_detector_moments(torch.ones(2), electrons_per_output_unit=value)


def test_invalid_tensor_shapes_dtypes_and_callbacks_fail_closed() -> None:
    photo = torch.ones(2, 3)
    for invalid in (torch.ones(3), torch.ones(2, 3, dtype=torch.float64)):
        with pytest.raises(ValueError):
            integrated_detector_moments(photo, dark_electron_mean=invalid)
    for invalid in (
        torch.ones(2, dtype=torch.int64),
        torch.ones(2, dtype=torch.complex64),
    ):
        with pytest.raises(TypeError):
            integrated_detector_moments(invalid)
    with pytest.raises(ValueError, match="nonempty"):
        integrated_detector_moments(torch.empty(0))
    bad_forward = LinearDetectorResponse(lambda value: value[0], lambda value: value)
    with pytest.raises(ValueError, match="shape"):
        integrated_detector_moments(photo, charge_response=bad_forward)
    bad_adjoint = LinearDetectorResponse(
        lambda value: value, lambda value: value.double()
    )
    result = integrated_detector_moments(photo, charge_response=bad_adjoint)
    with pytest.raises(ValueError, match="dtype and device"):
        result.covariance_action(photo)
    with pytest.raises(ValueError, match="shape"):
        integrated_detector_moments(photo).covariance_action(torch.ones(3))
    with pytest.raises(TypeError, match="LinearDetectorResponse"):
        integrated_detector_moments(photo, charge_response=lambda value: value)
    with pytest.raises(TypeError, match="callable"):
        LinearDetectorResponse(None, None)


def test_invalid_photon_conversion_and_overflow_fail_closed() -> None:
    source = torch.ones(2)
    for kwargs in (
        {"electrons_per_photon": 1.1},
        {"electrons_per_photon": -0.1},
        {"exposure_time_s": -1.0},
        {"foreground_photon_rate": -source},
        {"foreground_photon_rate": torch.ones(3)},
        {"optical_mean_response": lambda value: -value},
        {"optical_mean_response": lambda value: value.double()},
    ):
        options = {"exposure_time_s": 1.0, **kwargs}
        with pytest.raises(ValueError):
            integrated_photoelectron_mean(source, **options)
    huge = torch.full((2,), torch.finfo(source.dtype).max)
    with pytest.raises(ValueError, match="finite"):
        integrated_photoelectron_mean(huge, exposure_time_s=2.0)
    with pytest.raises(ValueError, match="finite"):
        integrated_detector_moments(huge, dark_electron_mean=huge)


@pytest.mark.parametrize("device", ("cpu", "cuda", "mps"))
def test_device_dtype_preservation_and_mismatches(device) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    source = torch.tensor([0.5, 4.0], device=device, dtype=torch.float32)
    photo = integrated_photoelectron_mean(
        source, exposure_time_s=2.0, electrons_per_photon=0.5
    )
    matrix = torch.tensor([[0.9, 0.2], [0.1, 0.8]], device=device, dtype=source.dtype)
    result = integrated_detector_moments(
        photo,
        charge_response=_linear(matrix, (2,)),
        dark_electron_mean=0.1,
        read_variance_electrons2=0.5,
        electrons_per_output_unit=2.0,
    )
    output = result.covariance_action(source)
    assert output.dtype == source.dtype and output.device == source.device
    assert result.mean.dtype == source.dtype and result.mean.device == source.device
    if device != "cpu":
        with pytest.raises(ValueError, match="dtype and device"):
            integrated_detector_moments(photo, dark_electron_mean=torch.ones(2))
