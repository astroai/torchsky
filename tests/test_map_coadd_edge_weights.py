"""Edge renormalisation of truncated resampling kernels (coadd weight semantics).

Near an input's edge the taps of an interpolation kernel fall off the image. The
resampled weight must stay the kernel-weighted mean of the input weights (so a
constant inverse-variance map stays constant), and the propagated inverse
variance must stay the exact ``1 / sum(c_i**2 / w_i)`` of the linear estimate.
"""

from __future__ import annotations

import pytest
import torch

from torchsky.map import (
    CubeTarget,
    Observation,
    blot,
    coadd,
    coadd_cube,
    coadd_healpix,
)
from torchsky.map.operators.resampling import (
    _axis_interpolation_coefficients,
    in_bounds_kernel_sum,
)

pytest.importorskip("torchsky._cpp")

_GRID_KERNELS = (
    "bicubic",
    "lanczos2",
    "lanczos3",
    "lanczos4",
    "catmull_rom",
    "mitchell_netravali",
)


def _header(n: int, *, crpix_shift: tuple[float, float] = (0.0, 0.0)) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": n,
        "NAXIS2": n,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CRPIX1": (n + 1) / 2 + crpix_shift[0],
        "CRPIX2": (n + 1) / 2 + crpix_shift[1],
        "CDELT1": -1.0e-4,
        "CDELT2": 1.0e-4,
    }


def _exposure(
    n: int, w: float, data: torch.Tensor | None = None, **shift
) -> Observation:
    image = torch.ones((n, n)) if data is None else data
    return Observation(
        data=image,
        wcs=_header(n, **shift),
        weight_map=torch.full((n, n), w),
        weight_is_inverse_variance=True,
    )


@pytest.mark.parametrize("kernel", ["nearest", "bilinear", *_GRID_KERNELS])
def test_in_bounds_kernel_sum_matches_tap_coefficients(kernel: str) -> None:
    x = torch.linspace(-0.4, 9.4, 97, dtype=torch.float64)
    y = torch.linspace(9.3, -0.3, 97, dtype=torch.float64)
    expected = []
    for coord, size in ((x, 10), (y, 7)):
        _, coefficients = _axis_interpolation_coefficients(
            coord, size=size, kernel=kernel
        )
        expected.append(coefficients.sum(dim=0))
    total = in_bounds_kernel_sum(x, y, height=7, width=10, kernel=kernel)
    torch.testing.assert_close(total, expected[0] * expected[1])


@pytest.mark.parametrize("kernel", _GRID_KERNELS)
def test_constant_inverse_variance_weight_is_preserved_to_the_edge(
    kernel: str,
) -> None:
    w = 4.0
    exposure = _exposure(12, w, crpix_shift=(0.37, -0.21))
    result = coadd([exposure], _header(14), kernel=kernel)
    covered = result.weight > 0
    assert int(covered.sum()) > 100
    # Before the fix Lanczos3 gave up to ~1.2 w next to the edge.
    torch.testing.assert_close(
        result.weight[covered], torch.full_like(result.weight[covered], w)
    )
    torch.testing.assert_close(
        result.data[covered], torch.ones_like(result.data[covered])
    )


@pytest.mark.parametrize("kernel", ["bilinear", "lanczos3", "catmull_rom"])
def test_inverse_variance_is_exact_at_the_edge(kernel: str) -> None:
    """Brute-force the linear map with impulse images: ivar = 1/sum(c^2/w)."""
    n, w = 6, 2.5
    target = _header(8)
    shift = {"crpix_shift": (0.31, 0.43)}
    base = coadd([_exposure(n, w, **shift)], target, kernel=kernel)
    covered = base.weight > 0
    assert base.inverse_variance is not None
    coefficients = []
    for index in range(n * n):
        impulse = torch.zeros(n * n)
        impulse[index] = 1.0
        result = coadd(
            [_exposure(n, w, data=impulse.reshape(n, n), **shift)],
            target,
            kernel=kernel,
        )
        coefficients.append(torch.where(covered, result.data, 0.0))
    c = torch.stack(coefficients).double()
    expected = 1.0 / (c.square().sum(dim=0)[covered] / w)
    torch.testing.assert_close(
        base.inverse_variance[covered].double(), expected, rtol=2e-5, atol=0.0
    )


def test_two_exposure_weight_never_exceeds_the_inverse_variance_sum() -> None:
    w_a, w_b = 1.0 / 1.76**2, 1.0 / 1.43**2
    exposures = [
        _exposure(11, w_a, crpix_shift=(2.3, 1.7)),
        _exposure(11, w_b, crpix_shift=(-2.6, -1.4)),
    ]
    result = coadd(exposures, _header(16), kernel="lanczos3")
    covered = result.weight > 0
    assert float(result.weight.max()) <= (w_a + w_b) * (1.0 + 1.0e-5)
    both = result.overlap_count == 2 if result.overlap_count is not None else None
    if both is not None and bool(both.any()):
        torch.testing.assert_close(
            result.weight[both], torch.full_like(result.weight[both], w_a + w_b)
        )
    single = covered & ~both if both is not None else covered
    values = result.weight[single]
    assert bool(
        (
            torch.isclose(values, torch.tensor(w_a))
            | torch.isclose(values, torch.tensor(w_b))
            | torch.isclose(values, torch.tensor(w_a + w_b))
        ).all()
    )


def test_cube_weight_is_preserved_to_the_edge() -> None:
    w = 3.0
    observations = [
        Observation(
            data=torch.full((12, 12), 1.0 + i),
            wcs=_header(12, crpix_shift=(0.37, -0.21)),
            wavelength=wavelength,
            weight_map=torch.full((12, 12), w),
            weight_is_inverse_variance=True,
        )
        for i, wavelength in enumerate((5000.0, 6000.0))
    ]
    target = CubeTarget(_header(14), [4500.0, 5500.0, 6500.0], shape=(14, 14))
    cube = coadd_cube(observations, target, kernel="lanczos3")
    covered = cube.weight > 0
    assert bool(covered.any())
    torch.testing.assert_close(
        cube.weight[covered], torch.full_like(cube.weight[covered], w)
    )


def test_healpix_weight_is_preserved_to_the_edge() -> None:
    w = 2.0
    exposure = _exposure(24, w, crpix_shift=(0.37, -0.21))
    # Cells (0.2 arcsec) finer than the 0.36 arcsec pixels: the sampling path.
    result = coadd_healpix([exposure], nside=1 << 20, sparse=True, kernel="lanczos3")
    sparse_weight = result.weight
    weight = sparse_weight.get_values(sparse_weight.valid_pixels)
    weight = weight[weight > 0]
    assert weight.numel() > 50
    torch.testing.assert_close(weight, torch.full_like(weight, w))


def test_blot_of_constant_coadd_is_constant_to_the_edge() -> None:
    coadded = coadd([_exposure(12, 1.0)], _header(12), kernel="lanczos3")
    blotted = blot(coadded, _header(12, crpix_shift=(0.37, -0.21)), kernel="lanczos3")
    valid = blotted != 0
    assert int(valid.sum()) > 80
    torch.testing.assert_close(blotted[valid], torch.ones_like(blotted[valid]))
