from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
import torch

from torchsky.map import AdaptiveResampleConfig, adaptive_reprojection_response
from torchsky.map.operators.adaptive import AdaptiveValueSemantics
from torchsky.map.operators import apply_response_products
from torchsky.map.quality import DQPolicy


class _TargetWCS:
    celestial_frame = "ICRS"
    celestial_equinox = None

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)

    def pixel_to_world(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x, y


class _SourceWCS:
    celestial_frame = "ICRS"
    celestial_equinox = None

    def __init__(
        self,
        mapping: Callable[
            [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ],
        device: str | torch.device = "cpu",
    ) -> None:
        self.mapping = mapping
        self.device = torch.device(device)

    def world_to_pixel(
        self, lon: torch.Tensor, lat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mapping(lon, lat)


def _config(**overrides: object) -> AdaptiveResampleConfig:
    values: dict[str, object] = {
        "max_edge_error_px": 1.0e-6,
        "max_jacobian_variation": 1.0e-6,
        "max_source_step_px": 0.5,
        "max_depth": 8,
        "planning_chunk_size": 2,
        "default_chunk_size": 3,
    }
    values.update(overrides)
    return AdaptiveResampleConfig(**values)


def _bilinear_dense_sample(
    image: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    x0 = torch.floor(x).to(torch.int64)
    y0 = torch.floor(y).to(torch.int64)
    output = torch.zeros_like(x)
    for offset_y, offset_x in ((0, 0), (0, 1), (1, 0), (1, 1)):
        xi = x0 + offset_x
        yi = y0 + offset_y
        valid = (xi >= 0) & (xi < image.shape[1]) & (yi >= 0) & (yi < image.shape[0])
        coefficient = (1.0 - torch.abs(x - xi)) * (1.0 - torch.abs(y - yi))
        output[valid] += image[yi[valid], xi[valid]] * coefficient[valid]
    return output


def _dense_oracle(
    image: torch.Tensor,
    mapping: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    target_shape: tuple[int, int],
    *,
    samples: int = 192,
) -> torch.Tensor:
    offset = (torch.arange(samples, dtype=image.dtype) + 0.5) / samples - 0.5
    values = []
    for row in range(target_shape[0]):
        row_values = []
        for column in range(target_shape[1]):
            yy, xx = torch.meshgrid(row + offset, column + offset, indexing="ij")
            x_source, y_source = mapping(xx.reshape(-1), yy.reshape(-1))
            row_values.append(_bilinear_dense_sample(image, x_source, y_source).mean())
        values.append(torch.stack(row_values))
    return torch.stack(values)


def test_adaptive_config_and_public_opt_in_surface() -> None:
    assert AdaptiveResampleConfig().value_semantics == "surface_brightness"
    with pytest.raises(ValueError, match="value_semantics"):
        AdaptiveResampleConfig(value_semantics=cast(AdaptiveValueSemantics, "flux"))
    with pytest.raises(ValueError, match="max_edge_error_px"):
        AdaptiveResampleConfig(max_edge_error_px=0.0)


@pytest.mark.parametrize(
    "semantics, scale", [("surface_brightness", 1.0), ("integrated_flux", 4.0)]
)
def test_affine_anisotropic_kernel_constant_and_flux_semantics(
    semantics: str, scale: float
) -> None:
    def mapping(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return 2.0 * x + 3.5, 2.0 * y + 3.5

    plan = adaptive_reprojection_response(
        _SourceWCS(mapping),
        _TargetWCS(),
        (10, 10),
        (2, 3),
        config=_config(value_semantics=cast(AdaptiveValueSemantics, semantics)),
        dtype=torch.float64,
    )
    constant = torch.ones((10, 10), dtype=torch.float64)
    zero = torch.zeros_like(constant)
    torch.testing.assert_close(
        plan.apply(constant), torch.full((2, 3), scale, dtype=torch.float64)
    )
    torch.testing.assert_close(
        plan.apply(zero), torch.zeros((2, 3), dtype=torch.float64)
    )
    assert torch.allclose(
        plan.coefficients.reshape(-1).new_tensor(scale * 6.0),
        plan.apply(constant).sum(),
    )


def test_affine_truth_response_products_adjoint_gradient_and_noncontiguous() -> None:
    def mapping(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return 1.5 * x + 2.0, 0.75 * y + 1.5

    plan = adaptive_reprojection_response(
        _SourceWCS(mapping),
        _TargetWCS(),
        (8, 8),
        (3, 4),
        config=_config(),
        dtype=torch.float64,
    )
    yy, xx = torch.meshgrid(
        torch.arange(8, dtype=torch.float64),
        torch.arange(8, dtype=torch.float64),
        indexing="ij",
    )
    contiguous = 1.0 + 2.0 * xx - 0.5 * yy
    source = contiguous.mT.contiguous().mT.detach().requires_grad_()
    assert not source.is_contiguous()
    expected_y, expected_x = torch.meshgrid(
        torch.arange(3, dtype=torch.float64),
        torch.arange(4, dtype=torch.float64),
        indexing="ij",
    )
    expected = 1.0 + 2.0 * (1.5 * expected_x + 2.0) - 0.5 * (0.75 * expected_y + 1.5)
    output = plan.apply(source, chunk_size=1)
    torch.testing.assert_close(output, expected, atol=1.0e-12, rtol=1.0e-12)
    target = torch.linspace(-1.0, 2.0, 12, dtype=torch.float64).reshape(3, 4)
    torch.testing.assert_close(
        torch.sum(output * target),
        torch.sum(source * plan.adjoint(target, chunk_size=2)),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    output.square().sum().backward()
    torch.testing.assert_close(source.grad, plan.adjoint(2.0 * output))

    dq = torch.zeros((8, 8), dtype=torch.int64)
    dq[2:5, 2:7] = 4
    products = apply_response_products(
        plan,
        source.detach(),
        variance_components={"detector": torch.ones_like(source)},
        dq=dq,
        dq_policy=DQPolicy(reject_bits=0, unknown_bit_policy="preserve"),
        chunk_size=2,
    )
    torch.testing.assert_close(products.science, output.detach())
    assert products.dq is not None and bool((products.dq == 4).any())
    assert products.context.shape[1:] == (3, 4)
    assert bool((products.nexp > 0).all())


def test_distorted_truth_matches_dense_high_resolution_oracle() -> None:
    def mapping(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = x - 1.5
        v = y - 1.0
        return 4.0 + 1.2 * x + 0.045 * u * v, 3.5 + 0.9 * y + 0.035 * u.square()

    plan = adaptive_reprojection_response(
        _SourceWCS(mapping),
        _TargetWCS(),
        (12, 12),
        (3, 4),
        config=_config(max_edge_error_px=2.0e-3, max_jacobian_variation=5.0e-3),
        dtype=torch.float64,
    )
    yy, xx = torch.meshgrid(
        torch.arange(12, dtype=torch.float64),
        torch.arange(12, dtype=torch.float64),
        indexing="ij",
    )
    image = torch.sin(0.7 * xx) + torch.cos(0.45 * yy) + 0.03 * xx * yy
    oracle = _dense_oracle(image, mapping, (3, 4))
    torch.testing.assert_close(plan.apply(image), oracle, atol=2.0e-3, rtol=2.0e-3)


def test_tighter_declared_bounds_monotonically_improve_distorted_truth() -> None:
    def mapping(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return 4.0 + x + 0.12 * (x - 1.0).square(), 4.0 + y + 0.08 * (x - 1.0) * (
            y - 1.0
        )

    yy, xx = torch.meshgrid(
        torch.arange(12, dtype=torch.float64),
        torch.arange(12, dtype=torch.float64),
        indexing="ij",
    )
    image = torch.sin(1.1 * xx) * torch.cos(0.8 * yy)
    oracle = _dense_oracle(image, mapping, (3, 3), samples=256)
    loose = adaptive_reprojection_response(
        _SourceWCS(mapping),
        _TargetWCS(),
        (12, 12),
        (3, 3),
        config=_config(
            max_edge_error_px=0.2,
            max_jacobian_variation=0.2,
            max_source_step_px=10.0,
        ),
        dtype=torch.float64,
    )
    tight = adaptive_reprojection_response(
        _SourceWCS(mapping),
        _TargetWCS(),
        (12, 12),
        (3, 3),
        config=_config(
            max_edge_error_px=2.0e-3,
            max_jacobian_variation=2.0e-3,
            max_source_step_px=10.0,
        ),
        dtype=torch.float64,
    )
    loose_error = torch.max(torch.abs(loose.apply(image) - oracle))
    tight_error = torch.max(torch.abs(tight.apply(image) - oracle))
    assert float(tight_error) <= float(loose_error)
    assert tight.source_indices.numel() >= loose.source_indices.numel()


@pytest.mark.parametrize(
    "mapping,error",
    [
        (lambda x, y: (x.square() + 3.0, y + 3.0), "fold or singular"),
        (lambda x, y: (torch.zeros_like(x) + 3.0, y + 3.0), "fold or singular"),
    ],
)
def test_fold_and_singular_jacobians_fail_closed(
    mapping: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    error: str,
) -> None:
    with pytest.raises(RuntimeError, match=error):
        adaptive_reprojection_response(
            _SourceWCS(mapping),
            _TargetWCS(),
            (8, 8),
            (1, 1),
            config=_config(max_source_step_px=10.0),
            dtype=torch.float64,
        )


def test_unresolved_subdivision_and_memory_bounds_fail_closed() -> None:
    def mapping(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return 3.0 + x + 0.2 * x.square(), 3.0 + y

    with pytest.raises(RuntimeError, match="could not resolve"):
        adaptive_reprojection_response(
            _SourceWCS(mapping),
            _TargetWCS(),
            (8, 8),
            (1, 1),
            config=_config(
                max_edge_error_px=1.0e-8,
                max_jacobian_variation=1.0e-8,
                max_source_step_px=10.0,
                max_depth=0,
            ),
            dtype=torch.float64,
        )
    with pytest.raises(RuntimeError, match="max_plan_entries"):
        adaptive_reprojection_response(
            _SourceWCS(lambda x, y: (x + 2.0, y + 2.0)),
            _TargetWCS(),
            (8, 8),
            (2, 2),
            config=_config(max_plan_entries=1),
            dtype=torch.float64,
        )


def test_planning_and_execution_chunking_are_identical() -> None:
    def mapping(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x + 2.25, y + 1.75

    first = adaptive_reprojection_response(
        _SourceWCS(mapping),
        _TargetWCS(),
        (8, 8),
        (3, 4),
        config=_config(planning_chunk_size=1),
        dtype=torch.float64,
    )
    second = adaptive_reprojection_response(
        _SourceWCS(mapping),
        _TargetWCS(),
        (8, 8),
        (3, 4),
        config=_config(planning_chunk_size=64),
        dtype=torch.float64,
    )
    image = torch.arange(64, dtype=torch.float64).reshape(8, 8)
    torch.testing.assert_close(
        first.apply(image, chunk_size=1), second.apply(image, chunk_size=99)
    )
    torch.testing.assert_close(
        first.apply_squared(image, chunk_size=2),
        second.apply_squared(image, chunk_size=7),
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_adaptive_response_device_and_dtype(device: str) -> None:
    dtype = torch.float32 if device == "mps" else torch.float64

    def mapping(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x + 2.0, y + 2.0

    plan = adaptive_reprojection_response(
        _SourceWCS(mapping, device),
        _TargetWCS(device),
        (6, 6),
        (2, 2),
        config=_config(),
        dtype=dtype,
        device=device,
    )
    image = torch.ones((6, 6), dtype=dtype, device=device, requires_grad=True)
    output = plan.apply(image)
    assert output.device.type == device
    assert output.dtype == dtype
    output.sum().backward()
    assert image.grad is not None and image.grad.device.type == device
