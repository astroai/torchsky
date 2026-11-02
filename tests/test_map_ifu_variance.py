from __future__ import annotations

import pytest
import torch

from torchsky.map import Observation, Target, coadd_cube, coadd_healpix
from torchsky.wcs import WCS


def _header(shape: tuple[int, int] = (1, 1)) -> dict[str, float | int | str]:
    height, width = shape
    return {
        "NAXIS": 2,
        "NAXIS1": width,
        "NAXIS2": height,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CDELT1": -0.05,
        "CDELT2": 0.05,
    }


def _ifu_observation(
    value: float,
    weight: float,
    *,
    inverse_variance: bool,
    scale: float = 1.0,
    mask: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> Observation:
    data = torch.tensor(
        [[value]], dtype=dtype, device=device, requires_grad=requires_grad
    )
    return Observation(
        data=data,
        header=_header(),
        wcs=WCS(_header()),
        wavelength_map=torch.tensor([[5001.0]], dtype=dtype, device=device),
        weight_map=torch.tensor([[weight]], dtype=dtype, device=device),
        weight_is_inverse_variance=inverse_variance,
        photometric_scale=scale,
        mask=torch.tensor([[mask]], dtype=torch.bool, device=device),
    )


def _coadd(observations: list[Observation], *, device: torch.device | str = "cpu"):
    return coadd_cube(
        observations,
        Target(wcs=WCS(_header()), shape=(1, 1)),
        spectral_axis=[5000.0, 5002.0],
        spectral_kernel="linear",
        device=device,
    )


def test_ifu_half_bin_uses_squared_coefficients_for_diagonal_variance() -> None:
    result = _coadd([_ifu_observation(3.0, 4.0, inverse_variance=True)])

    torch.testing.assert_close(result.data[:, 0, 0], torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(result.weight[:, 0, 0], torch.tensor([2.0, 2.0]))
    assert result.inverse_variance is not None
    torch.testing.assert_close(
        result.inverse_variance[:, 0, 0], torch.tensor([4.0, 4.0])
    )
    assert result.correlation_factor is not None
    torch.testing.assert_close(
        result.correlation_factor[:, 0, 0], torch.tensor([2.0, 2.0])
    )


def test_ifu_multiple_inverse_variance_contributions_and_mask() -> None:
    observations = [
        _ifu_observation(2.0, 4.0, inverse_variance=True),
        _ifu_observation(6.0, 1.0, inverse_variance=True),
    ]
    result = _coadd(observations)

    torch.testing.assert_close(result.data[:, 0, 0], torch.tensor([2.8, 2.8]))
    torch.testing.assert_close(result.weight[:, 0, 0], torch.tensor([2.5, 2.5]))
    assert result.inverse_variance is not None
    torch.testing.assert_close(
        result.inverse_variance[:, 0, 0], torch.tensor([5.0, 5.0])
    )

    masked = _coadd(
        [observations[0], _ifu_observation(6.0, 1.0, inverse_variance=True, mask=False)]
    )
    torch.testing.assert_close(masked.data[:, 0, 0], torch.tensor([2.0, 2.0]))
    assert masked.inverse_variance is not None
    torch.testing.assert_close(
        masked.inverse_variance[:, 0, 0], torch.tensor([4.0, 4.0])
    )


def test_ifu_generic_weights_are_not_reported_as_inverse_variance() -> None:
    result = _coadd([_ifu_observation(4.0, 3.0, inverse_variance=False)])

    torch.testing.assert_close(result.data[:, 0, 0], torch.tensor([4.0, 4.0]))
    torch.testing.assert_close(result.weight[:, 0, 0], torch.tensor([1.5, 1.5]))
    assert result.inverse_variance is None
    assert result.correlation_factor is None


def test_ifu_photometric_scale_is_applied_once() -> None:
    result = _coadd([_ifu_observation(3.0, 4.0, inverse_variance=True, scale=2.0)])

    torch.testing.assert_close(result.data[:, 0, 0], torch.tensor([6.0, 6.0]))
    torch.testing.assert_close(result.weight[:, 0, 0], torch.tensor([0.5, 0.5]))
    assert result.inverse_variance is not None
    torch.testing.assert_close(
        result.inverse_variance[:, 0, 0], torch.tensor([1.0, 1.0])
    )


def test_healpix_photometric_scale_is_applied_once() -> None:
    shape = (32, 32)
    header = _header(shape)
    observation = Observation(
        data=torch.full(shape, 3.0),
        header=header,
        wcs=WCS(header),
        weight_map=torch.full(shape, 4.0),
        weight_is_inverse_variance=True,
        photometric_scale=2.0,
    )

    result = coadd_healpix([observation], nside=64, nest=True, kernel="nearest")
    covered = result.weight > 0.0
    assert bool(covered.any())
    torch.testing.assert_close(
        result.data[covered], torch.full_like(result.data[covered], 6.0)
    )
    torch.testing.assert_close(
        result.weight[covered], torch.full_like(result.weight[covered], 1.0)
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ),
                reason="MPS not available",
            ),
        ),
    ],
)
def test_ifu_cube_dtype_device_and_data_gradient(device: str) -> None:
    dtype = torch.float32 if device == "mps" else torch.float64
    observation = _ifu_observation(
        3.0,
        4.0,
        inverse_variance=True,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    assert observation.data is not None

    result = _coadd([observation], device=device)
    result.data.sum().backward()

    assert result.data.dtype == torch.float32
    assert result.data.device.type == device
    assert result.inverse_variance is not None
    assert result.inverse_variance.dtype == torch.float32
    assert result.inverse_variance.device.type == device
    assert observation.data.grad is not None
    torch.testing.assert_close(
        observation.data.grad, torch.tensor([[2.0]], dtype=dtype, device=device)
    )
