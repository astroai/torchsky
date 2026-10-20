import pytest
import torch

from torchsky.wcs.detector_astrometry import (
    AffineFreeResidualGrid2D,
    DetectorToSkyTransform,
    DetectorToTangentTransform,
    RegularGridTransform2D,
    TangentPlaneAffine,
    validate_sampled_transform,
)


def _sample_table(
    height: int,
    width: int,
    function,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, dtype=dtype),
        torch.arange(width, dtype=dtype),
        indexing="ij",
    )
    return function(x, y)


@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
def test_regular_grid_identity_is_exact_and_differentiable(mode: str) -> None:
    table = _sample_table(5, 6, lambda x, y: torch.stack((x, y), dim=-1))
    transform = RegularGridTransform2D(table, mode=mode)
    points = torch.tensor(
        [[0.0, 0.0], [1.25, 2.5], [5.0, 4.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    result = transform(points)
    jacobian = transform.jacobian(points)
    result.sum().backward()

    torch.testing.assert_close(result, points.detach(), atol=2e-14, rtol=0)
    torch.testing.assert_close(
        jacobian,
        torch.eye(2, dtype=torch.float64).expand(3, 2, 2),
        atol=3e-14,
        rtol=0,
    )
    torch.testing.assert_close(points.grad, torch.ones_like(points))
    assert result.dtype == points.dtype
    assert result.device == points.device


@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
def test_regular_grid_reproduces_affine_map_with_origin_and_spacing(mode: str) -> None:
    origin = torch.tensor([10.0, -4.0], dtype=torch.float64)
    spacing = torch.tensor([2.0, 0.5], dtype=torch.float64)
    matrix = torch.tensor([[1.2, -0.3], [0.25, 0.8]], dtype=torch.float64)
    offset = torch.tensor([7.0, -2.0], dtype=torch.float64)
    table = _sample_table(
        5,
        6,
        lambda x, y: (
            torch.stack(
                (origin[0] + spacing[0] * x, origin[1] + spacing[1] * y),
                dim=-1,
            )
            @ matrix.T
            + offset
        ),
    )
    transform = RegularGridTransform2D(
        table,
        origin=origin,
        spacing=spacing,
        mode=mode,
    )
    points = torch.tensor(
        [[10.0, -4.0], [13.2, -2.7], [20.0, -2.0]], dtype=torch.float64
    )

    torch.testing.assert_close(
        transform(points), points @ matrix.T + offset, atol=5e-14, rtol=0
    )
    torch.testing.assert_close(
        transform.jacobian(points), matrix.expand(3, 2, 2), atol=8e-14, rtol=0
    )


def test_regular_grid_rejects_folded_table_and_out_of_bounds_points() -> None:
    table = _sample_table(
        4,
        4,
        lambda x, y: torch.stack((x * (2.0 - x), y), dim=-1),
    )
    with pytest.raises(ValueError, match="folded or singular"):
        RegularGridTransform2D(table)

    identity = RegularGridTransform2D(
        _sample_table(4, 4, lambda x, y: torch.stack((x, y), dim=-1))
    )
    with pytest.raises(ValueError, match="outside"):
        identity(torch.tensor([[4.0, 1.0]], dtype=torch.float64))


@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_regular_grid_reflected_nonlinear_map_preserves_values_inverse_and_ad(
    mode: str, dtype: torch.dtype
) -> None:
    def reference(points: torch.Tensor) -> torch.Tensor:
        x, y = points.unbind(-1)
        return torch.stack(
            (-1.2 * x + 0.03 * x * y + 7, 0.8 * y + 0.02 * x * y - 2), -1
        )

    table = _sample_table(
        5, 6, lambda x, y: reference(torch.stack((x, y), -1)), dtype=dtype
    )
    transform = RegularGridTransform2D(table, mode=mode)
    points = torch.tensor(
        [[0.25, 0.4], [2.3, 1.2], [4.8, 2.5]], dtype=dtype, requires_grad=True
    )
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(
        transform(points), reference(points), atol=tolerance, rtol=tolerance
    )
    assert bool((transform.assert_invertible() < 0).all())
    expected_gradient = torch.autograd.grad(
        reference(points).sum(), points, create_graph=True
    )[0]
    gradient = torch.autograd.grad(transform(points).sum(), points, create_graph=True)[
        0
    ]
    torch.testing.assert_close(
        gradient, expected_gradient, atol=tolerance, rtol=tolerance
    )
    curvature = torch.autograd.grad(gradient.sum(), points)[0]
    expected_curvature = torch.autograd.grad(expected_gradient.sum(), points)[0]
    torch.testing.assert_close(
        curvature, expected_curvature, atol=tolerance, rtol=tolerance
    )
    back = transform.inverse(transform(points), tol=tolerance)
    torch.testing.assert_close(back, points, atol=tolerance, rtol=tolerance)
    roundtrip_gradient = torch.autograd.grad(back.sum(), points, create_graph=True)[0]
    torch.testing.assert_close(
        roundtrip_gradient, torch.ones_like(points), atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        torch.autograd.grad(roundtrip_gradient.sum(), points)[0],
        torch.zeros_like(points),
        atol=tolerance,
        rtol=0,
    )


def test_regular_grid_rejects_opposite_parities_without_sampled_zero() -> None:
    # Bilinear cell determinants are +1, -1, +1: checking abs(det) alone misses it.
    table = _sample_table(4, 4, lambda x, y: torch.stack((x.remainder(2), y), -1))
    with pytest.raises(ValueError, match="consistent sign"):
        RegularGridTransform2D(table)


@pytest.mark.parametrize("scale", [0.0, -1e-4])
def test_regular_grid_absolute_determinant_floor_rejects_singular_and_small_reflection(
    scale: float,
) -> None:
    table = _sample_table(4, 4, lambda x, y: torch.stack((scale * x, y), -1))
    with pytest.raises(ValueError, match="folded or singular"):
        RegularGridTransform2D(table, determinant_floor=1e-3)


def test_tangent_plane_affine_is_learnable_and_orientation_preserving() -> None:
    matrix = torch.tensor([[1.1, 0.2], [-0.1, 0.9]], dtype=torch.float64)
    offset = torch.tensor([0.3, -0.4], dtype=torch.float64)
    affine = TangentPlaneAffine(matrix, offset)
    points = torch.tensor([[2.0, 3.0], [-1.0, 0.5]], dtype=torch.float32)

    result = affine(points)
    result.square().sum().backward()

    torch.testing.assert_close(
        result,
        points @ matrix.to(torch.float32).T + offset.to(torch.float32),
    )
    assert result.dtype == torch.float32
    assert affine.matrix.grad is not None
    assert affine.offset.grad is not None
    assert float(affine.assert_invertible().detach()) > 0
    with pytest.raises(ValueError, match="positive determinant"):
        TangentPlaneAffine(torch.diag(torch.tensor([-1.0, 1.0])), torch.zeros(2))


def test_residual_grid_exactly_removes_constant_and_affine_modes() -> None:
    control = _sample_table(
        6,
        5,
        lambda x, y: torch.stack(
            (2.0 + 0.3 * x - 0.7 * y, -1.0 + 0.2 * x + 0.4 * y),
            dim=-1,
        ),
    )
    residual = AffineFreeResidualGrid2D(control)
    projected = residual.projected_control
    flat = projected.reshape(-1, 2)
    basis = residual.affine_basis

    torch.testing.assert_close(projected, torch.zeros_like(projected), atol=2e-14, rtol=0)
    torch.testing.assert_close(basis.T @ flat, torch.zeros(3, 2, dtype=flat.dtype), atol=2e-14, rtol=0)


def test_two_by_two_residual_grid_has_finite_zero_bending_energy() -> None:
    residual = AffineFreeResidualGrid2D(
        torch.zeros((2, 2, 2), dtype=torch.float64), mode="bilinear"
    )
    energy = residual.bending_energy()
    assert torch.isfinite(energy)
    torch.testing.assert_close(energy, torch.zeros_like(energy))


def test_grid_inverse_tolerances_are_in_output_units_and_tol_is_alias() -> None:
    table = _sample_table(
        2, 2, lambda x, y: torch.stack((1000.0 * x, 0.001 * y), dim=-1)
    )
    transform = RegularGridTransform2D(table, spacing=(10.0, 20.0))
    target = torch.tensor([[1000.0 + 1.0e-6, 0.001]], dtype=torch.float64)

    expected = torch.tensor([[10.0, 20.0]], dtype=torch.float64)
    torch.testing.assert_close(transform.inverse(target, atol=2.0e-6), expected)
    torch.testing.assert_close(transform.inverse(target, tol=2.0e-6), expected)
    with pytest.raises(ValueError, match="atol"):
        transform.inverse(target, atol=5.0e-7)
    with pytest.raises(ValueError, match="aliases"):
        transform.inverse(target, tol=2.0e-6, atol=2.0e-6)


@pytest.mark.parametrize("parity", [1.0, -1.0])
def test_non_affine_residual_and_composite_jacobian_are_consistent(
    parity: float,
) -> None:
    table = RegularGridTransform2D(
        _sample_table(8, 8, lambda x, y: torch.stack((parity * x, y), dim=-1)),
        mode="bicubic",
    )
    control = _sample_table(
        8,
        8,
        lambda x, y: 1.0e-3
        * torch.stack((torch.sin(x) * torch.sin(y), torch.cos(x) * torch.sin(y)), dim=-1),
    )
    residual = AffineFreeResidualGrid2D(control, mode="bicubic")
    composite = DetectorToTangentTransform(
        table,
        TangentPlaneAffine.identity(dtype=torch.float64),
        residual,
    )
    point = torch.tensor([[2.3, 4.2]], dtype=torch.float64, requires_grad=True)

    output, analytic = composite.evaluate_with_jacobian(point)
    autograd_rows = []
    for output_axis in range(2):
        gradient = torch.autograd.grad(
            output[0, output_axis], point, retain_graph=True
        )[0]
        autograd_rows.append(gradient[0])
    autograd_jacobian = torch.stack(autograd_rows)

    assert not torch.allclose(output, point)
    torch.testing.assert_close(analytic[0], autograd_jacobian, atol=2e-12, rtol=2e-12)
    assert bool((composite.jacobian_determinant(point) * parity > 0).all())
    assert bool((composite.assert_invertible() * parity > 0).all())
    assert float(residual.bending_energy().detach()) > 0
    recovered = composite.inverse(output, tol=1e-10)
    torch.testing.assert_close(recovered, point, atol=1e-10, rtol=0)
    roundtrip_gradient = torch.autograd.grad(recovered.sum(), point, create_graph=True)[
        0
    ]
    torch.testing.assert_close(
        roundtrip_gradient, torch.ones_like(point), atol=1e-10, rtol=0
    )
    torch.testing.assert_close(
        torch.autograd.grad(roundtrip_gradient.sum(), point)[0],
        torch.zeros_like(point),
        atol=1e-10,
        rtol=0,
    )


@pytest.mark.parametrize("parity", [1.0, -1.0])
def test_composite_rejects_residual_that_changes_base_parity(parity: float) -> None:
    table = RegularGridTransform2D(
        _sample_table(4, 4, lambda x, y: torch.stack((parity * x, y), -1))
    )
    residual = AffineFreeResidualGrid2D(
        _sample_table(
            4, 4, lambda x, y: torch.stack((100 * x.sin(), torch.zeros_like(y)), -1)
        )
    )
    with pytest.raises(ValueError, match="folded or singular"):
        DetectorToTangentTransform(
            table, TangentPlaneAffine.identity(dtype=torch.float64), residual
        )


def test_validate_sampled_transform_reports_mas_thresholds() -> None:
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.5], [359.9, -0.2]], dtype=torch.float64
    )

    exact = validate_sampled_transform(lambda value: value, lambda value: value, points)
    assert exact.passed
    assert exact.max_mas < 1.0e-9

    arcsec_offset = 1.0e-4
    failed = validate_sampled_transform(
        lambda value: value + torch.tensor([arcsec_offset, 0.0], dtype=value.dtype),
        lambda value: value,
        points,
        coordinates="plane_arcsec",
    )
    assert failed.max_mas == pytest.approx(0.1)
    assert not failed.passed
    with pytest.raises(ValueError, match="misses tolerance"):
        failed.assert_valid()


def test_detector_to_sky_forward_and_pixel_to_world_freeze_attitude() -> None:
    class Attitude(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.zero = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

        def forward(self, tangent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            longitude = torch.remainder(tangent[..., 0] + 359.0 + self.zero, 360.0)
            latitude = tangent[..., 1] - 2.0 + self.zero
            return longitude, latitude

    table = RegularGridTransform2D(
        _sample_table(4, 4, lambda x, y: torch.stack((x, y), dim=-1))
    )
    tangent = DetectorToTangentTransform(
        table, TangentPlaneAffine.identity(dtype=torch.float64)
    )
    attitude = Attitude()
    transform = DetectorToSkyTransform(tangent, attitude)
    x = torch.tensor([0.5, 2.0], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([1.5, 3.0], dtype=torch.float64)

    longitude, latitude = transform.pixel_to_world(x, y)
    points_result = transform(torch.stack((x, y), dim=-1))
    longitude.sum().backward()

    torch.testing.assert_close(points_result, torch.stack((longitude, latitude), dim=-1))
    torch.testing.assert_close(longitude, torch.tensor([359.5, 1.0], dtype=torch.float64))
    torch.testing.assert_close(latitude, torch.tensor([-0.5, 1.0], dtype=torch.float64))
    assert x.grad is not None
    assert not attitude.zero.requires_grad
    assert bool((transform.jacobian_determinant(torch.stack((x, y), dim=-1)) > 0).all())
