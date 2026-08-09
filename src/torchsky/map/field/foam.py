"""Bounded power-diagram primitives for experimental sky-field models.

These tensor-native kernels implement the geometry underlying Power Foam.  They
are representation primitives, not a complete renderer: exact rendering also
needs sparse candidate construction and pixel/cell intersection integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any, Callable, Literal

import torch
from torch import Tensor

PowerValueSemantics = Literal["surface_brightness", "integrated_flux"]


@dataclass(frozen=True)
class SparseCechGraph:
    """Experimental exact Čech graph plus its sparse-index diagnostics."""

    candidate_pairs: Tensor
    edges: Tensor
    indptr: Tensor
    indices: Tensor
    total_pairs: int
    site_bin_memberships: int
    occupied_bins: int
    bin_size: float


@dataclass(frozen=True)
class TileDiskCSR:
    """Exact disk candidates for a regular tangent-plane tile grid."""

    indptr: Tensor
    indices: Tensor
    tile_ids: Tensor
    aabb_memberships: int
    total_pairs: int
    tile_shape: tuple[int, int]
    tile_origin: tuple[float, float]
    tile_size: tuple[float, float]


@dataclass(frozen=True)
class PolygonDiskCSR:
    """Conservative disk candidates for arbitrary convex pixel polygons."""

    indptr: Tensor
    indices: Tensor
    polygon_ids: Tensor
    polygon_bin_memberships: int
    site_bin_memberships: int
    raw_candidate_pairs: int
    total_pairs: int
    bin_size: float


@dataclass(frozen=True)
class PowerPixelOperator:
    """Exact sparse bounded-power-cell to convex-pixel area operator.

    ``areas`` is pixel-major and ``transpose_areas`` is the same data in
    cell-major order. Outputs from :meth:`matvec` are always pixel-integrated.
    Surface-brightness inputs are multiplied by intersection area; integrated
    cell-flux inputs are multiplied by the fraction of their bounded cell area.
    """

    indptr: Tensor
    indices: Tensor
    pixel_ids: Tensor
    areas: Tensor
    transpose_indptr: Tensor
    transpose_indices: Tensor
    transpose_areas: Tensor
    cell_areas: Tensor
    pixel_count: int
    cell_count: int
    candidate_pairs: int

    def _weights(self, semantics: PowerValueSemantics, *, transposed: bool) -> Tensor:
        if semantics not in ("surface_brightness", "integrated_flux"):
            raise ValueError(
                "semantics must be 'surface_brightness' or 'integrated_flux'"
            )
        areas = self.transpose_areas if transposed else self.areas
        if semantics == "surface_brightness":
            return areas
        cell_ids = (
            torch.repeat_interleave(
                torch.arange(
                    self.cell_count,
                    dtype=torch.long,
                    device=self.cell_areas.device,
                ),
                self.transpose_indptr[1:] - self.transpose_indptr[:-1],
            )
            if transposed
            else self.indices
        )
        normalizers = self.cell_areas[cell_ids]
        if bool(torch.any(normalizers <= 0)):
            raise RuntimeError("referenced bounded cells must have positive area")
        return areas / normalizers

    def matvec(
        self,
        cell_values: Tensor,
        *,
        semantics: PowerValueSemantics = "surface_brightness",
    ) -> Tensor:
        """Map ``(..., cells)`` values to pixel-integrated ``(..., pixels)``."""
        if not cell_values.is_floating_point() or cell_values.shape[-1:] != (
            self.cell_count,
        ):
            raise ValueError(
                "cell_values must be floating point with shape (..., cells)"
            )
        if (
            cell_values.device != self.areas.device
            or cell_values.dtype != self.areas.dtype
        ):
            raise ValueError("cell_values must match the operator dtype and device")
        flat = cell_values.reshape(-1, self.cell_count)
        contributions = flat[:, self.indices] * self._weights(
            semantics,
            transposed=False,
        )
        output = flat.new_zeros((flat.shape[0], self.pixel_count))
        output.scatter_add_(
            1,
            self.pixel_ids.expand(flat.shape[0], -1),
            contributions,
        )
        return output.reshape(cell_values.shape[:-1] + (self.pixel_count,))

    def rmatvec(
        self,
        pixel_values: Tensor,
        *,
        semantics: PowerValueSemantics = "surface_brightness",
    ) -> Tensor:
        """Apply the exact transpose to ``(..., pixels)`` values."""
        if not pixel_values.is_floating_point() or pixel_values.shape[-1:] != (
            self.pixel_count,
        ):
            raise ValueError(
                "pixel_values must be floating point with shape (..., pixels)"
            )
        if (
            pixel_values.device != self.areas.device
            or pixel_values.dtype != self.areas.dtype
        ):
            raise ValueError("pixel_values must match the operator dtype and device")
        cell_ids = torch.repeat_interleave(
            torch.arange(
                self.cell_count,
                dtype=torch.long,
                device=self.areas.device,
            ),
            self.transpose_indptr[1:] - self.transpose_indptr[:-1],
        )
        flat = pixel_values.reshape(-1, self.pixel_count)
        contributions = flat[:, self.transpose_indices] * self._weights(
            semantics,
            transposed=True,
        )
        output = flat.new_zeros((flat.shape[0], self.cell_count))
        output.scatter_add_(
            1,
            cell_ids.expand(flat.shape[0], -1),
            contributions,
        )
        return output.reshape(pixel_values.shape[:-1] + (self.cell_count,))


def render_power_spectral_samples(
    operator: PowerPixelOperator,
    cell_samples: Tensor,
    response_weights: Tensor,
    *,
    semantics: PowerValueSemantics = "surface_brightness",
) -> Tensor:
    """Integrate sampled cell spectra and render them with shared geometry.

    ``cell_samples`` has shape ``(..., samples, cells)``. Response weights have
    shape ``(samples,)`` for one channel or ``(channels, samples)`` for many.
    The weights are linear quadrature/response coefficients supplied by the
    existing spectral operator; they are not renormalized here.
    """
    if (
        not cell_samples.is_floating_point()
        or cell_samples.ndim < 2
        or cell_samples.shape[-1] != operator.cell_count
    ):
        raise ValueError(
            "cell_samples must be floating point with shape (..., samples, cells)"
        )
    if (
        cell_samples.device != operator.areas.device
        or cell_samples.dtype != operator.areas.dtype
    ):
        raise ValueError("cell_samples must match the operator dtype and device")
    if not response_weights.is_floating_point() or response_weights.ndim not in (1, 2):
        raise ValueError(
            "response_weights must have shape (samples,) or (channels, samples)"
        )
    if response_weights.shape[-1] != cell_samples.shape[-2]:
        raise ValueError("response_weights and cell_samples must share sample count")
    if not bool(torch.isfinite(cell_samples).all()) or not bool(
        torch.isfinite(response_weights).all()
    ):
        raise ValueError("cell_samples and response_weights must be finite")

    squeeze_channel = response_weights.ndim == 1
    responses = response_weights.reshape(-1, response_weights.shape[-1]).to(
        device=cell_samples.device,
        dtype=cell_samples.dtype,
    )
    channel_cells = torch.einsum("bs,...sc->...bc", responses, cell_samples)
    rendered = operator.matvec(channel_cells, semantics=semantics)
    return rendered.squeeze(-2) if squeeze_channel else rendered


def render_power_spectral_basis(
    operator: PowerPixelOperator,
    cell_coefficients: Tensor,
    basis_samples: Tensor,
    response_weights: Tensor,
    *,
    semantics: PowerValueSemantics = "surface_brightness",
) -> Tensor:
    """Evaluate compact cell spectra, integrate responses, and render pixels.

    ``cell_coefficients`` has shape ``(..., cells, basis)`` and
    ``basis_samples`` has shape ``(samples, basis)``.
    """
    if (
        not cell_coefficients.is_floating_point()
        or cell_coefficients.ndim < 2
        or cell_coefficients.shape[-2] != operator.cell_count
    ):
        raise ValueError(
            "cell_coefficients must be floating point with shape (..., cells, basis)"
        )
    if (
        cell_coefficients.device != operator.areas.device
        or cell_coefficients.dtype != operator.areas.dtype
    ):
        raise ValueError("cell_coefficients must match the operator dtype and device")
    if (
        not basis_samples.is_floating_point()
        or basis_samples.ndim != 2
        or basis_samples.shape[1] != cell_coefficients.shape[-1]
    ):
        raise ValueError("basis_samples must have shape (samples, basis)")
    if (
        basis_samples.device != cell_coefficients.device
        or basis_samples.dtype != cell_coefficients.dtype
    ):
        raise ValueError("basis_samples must match cell_coefficients dtype and device")
    cell_samples = torch.einsum(
        "...ck,sk->...sc",
        cell_coefficients,
        basis_samples,
    )
    return render_power_spectral_samples(
        operator,
        cell_samples,
        response_weights,
        semantics=semantics,
    )


def solve_spectral_cell_coefficients(
    basis_values: Tensor,
    cell_values: Tensor,
    cell_weights: Tensor | None = None,
    *,
    ridge: float = 1.0e-6,
) -> Tensor:
    """Solve independent weighted spectral coefficients for fixed cells.

    ``basis_values`` has shape ``(bands, basis)`` and ``cell_values`` has shape
    ``(bands, cells)``. Cells with zero total weight remain exactly zero.
    """
    if (
        not basis_values.is_floating_point()
        or basis_values.ndim != 2
        or basis_values.shape[0] == 0
        or basis_values.shape[1] == 0
    ):
        raise ValueError("basis_values must have floating shape (bands, basis)")
    if (
        not cell_values.is_floating_point()
        or cell_values.ndim != 2
        or cell_values.shape[0] != basis_values.shape[0]
    ):
        raise ValueError("cell_values must have floating shape (bands, cells)")
    if cell_values.device != basis_values.device or cell_values.dtype != basis_values.dtype:
        raise ValueError("basis_values and cell_values must share dtype and device")
    if not bool(torch.isfinite(basis_values).all()) or not bool(
        torch.isfinite(cell_values).all()
    ):
        raise ValueError("basis_values and cell_values must be finite")
    if cell_weights is None:
        weights = torch.ones_like(cell_values)
    else:
        if (
            not cell_weights.is_floating_point()
            or cell_weights.shape != cell_values.shape
            or cell_weights.device != cell_values.device
            or cell_weights.dtype != cell_values.dtype
        ):
            raise ValueError("cell_weights must match cell_values")
        if not bool(torch.isfinite(cell_weights).all()) or bool(
            torch.any(cell_weights < 0)
        ):
            raise ValueError("cell_weights must be finite and non-negative")
        weights = cell_weights
    penalty = float(ridge)
    if not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("ridge must be finite and positive")

    normal = torch.einsum("bc,bk,bl->ckl", weights, basis_values, basis_values)
    rhs = torch.einsum("bc,bk,bc->ck", weights, basis_values, cell_values)
    scale = torch.diagonal(normal, dim1=-2, dim2=-1).mean(dim=-1)
    regularization = penalty * torch.where(
        scale > 0,
        scale,
        torch.ones_like(scale),
    )
    identity = torch.eye(
        basis_values.shape[1],
        dtype=basis_values.dtype,
        device=basis_values.device,
    )
    coefficients = torch.linalg.solve(
        normal + regularization[:, None, None] * identity,
        rhs.unsqueeze(-1),
    ).squeeze(-1)
    observed = torch.sum(weights, dim=0) > 0
    return torch.where(observed[:, None], coefficients, torch.zeros_like(coefficients))


def solve_power_pixel_least_squares(
    operator: PowerPixelOperator,
    pixel_values: Tensor,
    pixel_weights: Tensor | None = None,
    *,
    semantics: PowerValueSemantics = "surface_brightness",
    ridge: float = 1.0e-6,
    max_iterations: int = 50,
    tolerance: float = 1.0e-6,
    initial: Tensor | None = None,
    pixel_matvec: Callable[[Tensor], Tensor] | None = None,
    pixel_rmatvec: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Solve a weighted sparse pixel-to-cell inverse with Jacobi PCG.

    ``pixel_values`` has shape ``(..., pixels)`` in the pixel-integrated units
    returned by :meth:`PowerPixelOperator.matvec`. The result has shape
    ``(..., cells)``. Pixels with zero weight do not constrain the solution.
    Optional pixel-domain callables compose another linear operator after the
    power-pixel map and must be supplied as a forward/adjoint pair.
    """
    if (
        not pixel_values.is_floating_point()
        or pixel_values.shape[-1:] != (operator.pixel_count,)
        or pixel_values.device != operator.areas.device
        or pixel_values.dtype != operator.areas.dtype
    ):
        raise ValueError(
            "pixel_values must match the operator dtype and device with shape "
            "(..., pixels)"
        )
    if not bool(torch.isfinite(pixel_values).all()):
        raise ValueError("pixel_values must be finite")
    if pixel_weights is None:
        weights = torch.ones_like(pixel_values)
    else:
        if (
            not pixel_weights.is_floating_point()
            or pixel_weights.shape != pixel_values.shape
            or pixel_weights.device != pixel_values.device
            or pixel_weights.dtype != pixel_values.dtype
        ):
            raise ValueError("pixel_weights must match pixel_values")
        if not bool(torch.isfinite(pixel_weights).all()) or bool(
            torch.any(pixel_weights < 0)
        ):
            raise ValueError("pixel_weights must be finite and non-negative")
        weights = pixel_weights
    penalty = float(ridge)
    threshold = float(tolerance)
    iterations = int(max_iterations)
    if not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("ridge must be finite and positive")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("tolerance must be finite and positive")
    if iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if (pixel_matvec is None) != (pixel_rmatvec is None):
        raise ValueError("pixel_matvec and pixel_rmatvec must be supplied together")
    output_shape = pixel_values.shape[:-1] + (operator.cell_count,)
    if initial is None:
        solution = pixel_values.new_zeros(output_shape)
    else:
        if (
            not initial.is_floating_point()
            or initial.shape != output_shape
            or initial.device != pixel_values.device
            or initial.dtype != pixel_values.dtype
            or not bool(torch.isfinite(initial).all())
        ):
            raise ValueError("initial must be finite and match the output shape")
        solution = initial.clone()

    flat_values = pixel_values.reshape(-1, operator.pixel_count)
    flat_weights = weights.reshape_as(flat_values)
    solution = solution.reshape(-1, operator.cell_count)
    edge_weights = operator._weights(semantics, transposed=False)
    diagonal = flat_values.new_zeros((flat_values.shape[0], operator.cell_count))
    diagonal.scatter_add_(
        1,
        operator.indices.expand(flat_values.shape[0], -1),
        flat_weights[:, operator.pixel_ids] * edge_weights.square(),
    )
    positive_count = torch.sum(diagonal > 0, dim=1).clamp(min=1)
    scale = torch.sum(diagonal, dim=1) / positive_count
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    regularization = penalty * scale
    preconditioner = diagonal + regularization[:, None]

    def apply_normal(cell_values: Tensor) -> Tensor:
        projected = operator.matvec(cell_values, semantics=semantics)
        if pixel_matvec is not None:
            projected = pixel_matvec(projected)
        weighted = flat_weights * projected
        if pixel_rmatvec is not None:
            weighted = pixel_rmatvec(weighted)
        return operator.rmatvec(
            weighted,
            semantics=semantics,
        ) + regularization[:, None] * cell_values

    weighted_values = flat_weights * flat_values
    if pixel_rmatvec is not None:
        weighted_values = pixel_rmatvec(weighted_values)
    rhs = operator.rmatvec(weighted_values, semantics=semantics)
    residual = rhs - apply_normal(solution)
    preconditioned = residual / preconditioner
    direction = preconditioned.clone()
    residual_dot = torch.sum(residual * preconditioned, dim=1)
    rhs_norm = torch.linalg.vector_norm(rhs, dim=1)
    residual_norm = torch.linalg.vector_norm(residual, dim=1)
    reference_norm = torch.where(rhs_norm > 0, rhs_norm, residual_norm)
    active = residual_norm > threshold * reference_norm
    tiny = torch.finfo(pixel_values.dtype).tiny
    # ponytail: fixed-iteration PCG avoids a solver dependency; the Jacobi
    # preconditioner ignores optional pixel mixing and can be upgraded if a
    # broad PSF makes convergence profiling justify it.
    for _ in range(iterations):
        if not bool(torch.any(active)):
            break
        normal_direction = apply_normal(direction)
        denominator = torch.sum(direction * normal_direction, dim=1)
        safe_denominator = torch.where(
            torch.abs(denominator) > tiny,
            denominator,
            torch.ones_like(denominator),
        )
        step = torch.where(active, residual_dot / safe_denominator, 0.0)
        solution = solution + step[:, None] * direction
        residual = residual - step[:, None] * normal_direction
        next_active = active & (
            torch.linalg.vector_norm(residual, dim=1) > threshold * reference_norm
        )
        preconditioned = residual / preconditioner
        next_residual_dot = torch.sum(residual * preconditioned, dim=1)
        safe_residual_dot = torch.where(
            torch.abs(residual_dot) > tiny,
            residual_dot,
            torch.ones_like(residual_dot),
        )
        momentum = torch.where(
            next_active,
            next_residual_dot / safe_residual_dot,
            0.0,
        )
        direction = preconditioned + momentum[:, None] * direction
        residual_dot = next_residual_dot
        active = next_active
    return solution.reshape(output_shape)


def _validate_sites_and_radii(
    sites: Tensor,
    radii: Tensor,
    *,
    allow_empty: bool,
) -> None:
    if not sites.is_floating_point():
        raise TypeError("sites must be a floating-point tensor")
    if sites.ndim != 2:
        raise ValueError("sites must have shape (N, D)")
    if not allow_empty and sites.shape[0] == 0:
        raise ValueError("at least one power site is required")
    if radii.ndim != 1 or radii.shape[0] != sites.shape[0]:
        raise ValueError("radii must have shape (N,) matching sites")
    if sites.device != radii.device:
        raise ValueError("sites and radii must be on the same device")
    if sites.dtype != radii.dtype:
        raise ValueError("sites and radii must have the same dtype")
    if not bool(torch.isfinite(sites).all()):
        raise ValueError("sites must be finite")
    if not bool(torch.isfinite(radii).all()) or bool(torch.any(radii < 0)):
        raise ValueError("radii must be finite and non-negative")


def _validate_power_inputs(points: Tensor, sites: Tensor, radii: Tensor) -> None:
    _validate_sites_and_radii(sites, radii, allow_empty=False)
    if not points.is_floating_point():
        raise TypeError("points must be a floating-point tensor")
    if points.ndim < 1:
        raise ValueError("points must end in coordinates")
    if points.shape[-1] != sites.shape[-1]:
        raise ValueError("points and sites must have the same coordinate dimension")
    if points.device != sites.device:
        raise ValueError("points, sites, and radii must be on the same device")
    if points.dtype != sites.dtype:
        raise ValueError("points, sites, and radii must have the same dtype")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("points must be finite")


def power_distance(points: Tensor, sites: Tensor, radii: Tensor) -> Tensor:
    """Return ``||x - p_i||² - r_i²`` for every point/site pair.

    ``points`` has shape ``(..., D)``, ``sites`` has shape ``(N, D)``, and the
    result has shape ``(..., N)``.  Use :func:`bounded_power_ownership` for
    bounded-memory ownership queries over many points.
    """
    _validate_power_inputs(points, sites, radii)
    delta = points.unsqueeze(-2) - sites
    return torch.sum(delta * delta, dim=-1) - radii * radii


def bounded_power_ownership(
    points: Tensor,
    sites: Tensor,
    radii: Tensor,
    *,
    point_chunk_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Return exact bounded-power-cell ids and minimum power distances.

    Cell ``i`` is its ordinary power cell intersected with its radius-``r_i``
    ball.  Points outside the union of those balls receive owner ``-1``.  The
    optional chunk size bounds the temporary point-by-site distance matrix.
    """
    _validate_power_inputs(points, sites, radii)
    flat = points.reshape(-1, points.shape[-1])
    if flat.shape[0] == 0:
        shape = points.shape[:-1]
        return (
            torch.empty(shape, dtype=torch.long, device=points.device),
            torch.empty(shape, dtype=points.dtype, device=points.device),
        )
    chunk_size = flat.shape[0] if point_chunk_size is None else int(point_chunk_size)
    if chunk_size <= 0:
        raise ValueError("point_chunk_size must be positive")

    owner_parts: list[Tensor] = []
    distance_parts: list[Tensor] = []
    for start in range(0, flat.shape[0], chunk_size):
        distances = power_distance(flat[start : start + chunk_size], sites, radii)
        minimum, owners = torch.min(distances, dim=-1)
        owners = torch.where(minimum <= 0, owners, torch.full_like(owners, -1))
        owner_parts.append(owners)
        distance_parts.append(minimum)
    shape = points.shape[:-1]
    return torch.cat(owner_parts).reshape(shape), torch.cat(distance_parts).reshape(
        shape
    )


def soft_bounded_power_weights(
    points: Tensor,
    sites: Tensor,
    radii: Tensor,
    *,
    temperature: float = 10.0,
) -> Tensor:
    """Return a differentiable, compactly supported power-ownership surrogate.

    Weights are a softmax of negative power distance over balls containing each
    query.  They sum to one inside the union of bounds and to zero outside it.
    This is useful during optimization but is not an exact power-cell renderer.
    """
    _validate_power_inputs(points, sites, radii)
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    distances = power_distance(points, sites, radii)
    support = distances <= 0
    logits = -float(temperature) * distances
    masked_logits = torch.where(support, logits, torch.full_like(logits, -torch.inf))
    any_support = support.any(dim=-1, keepdim=True)
    safe_logits = torch.where(
        any_support, masked_logits, torch.zeros_like(masked_logits)
    )
    weights = torch.softmax(safe_logits, dim=-1)
    return torch.where(any_support, weights, torch.zeros_like(weights))


def smooth_bounded_power_weights(
    points: Tensor,
    sites: Tensor,
    radii: Tensor,
    *,
    power_temperature: float = 10.0,
    support_temperature: float = 10.0,
) -> Tensor:
    """Return smooth bounded-power weights for geometry optimization.

    The result sums to a smooth probability that each query lies inside the
    union of the site bounds; the missing mass is background. Conditional
    ownership combines power distance with each site's smooth support
    probability. This is a training surrogate, not exact power-cell inference
    or exact pixel-area integration.
    """
    _validate_power_inputs(points, sites, radii)
    power_scale = float(power_temperature)
    support_scale = float(support_temperature)
    if not math.isfinite(power_scale) or power_scale <= 0:
        raise ValueError("power_temperature must be finite and positive")
    if not math.isfinite(support_scale) or support_scale <= 0:
        raise ValueError("support_temperature must be finite and positive")

    distances = power_distance(points, sites, radii)
    log_membership = torch.nn.functional.logsigmoid(-support_scale * distances)
    conditional = torch.softmax(
        -power_scale * distances + log_membership,
        dim=-1,
    )
    log_outside_union = torch.nn.functional.logsigmoid(support_scale * distances).sum(
        dim=-1, keepdim=True
    )
    union_occupancy = -torch.expm1(log_outside_union)
    return conditional * union_occupancy


class SpectralPowerFoam(torch.nn.Module):
    """Trainable shared geometry with caller-defined spectral basis values.

    Smooth point queries are the optimization path. :meth:`render_exact` reuses
    the learned appearance coefficients with a hard ``PowerPixelOperator`` built
    from detached geometry for scientific inference.
    """

    def __init__(
        self,
        sites: Tensor,
        radii: Tensor,
        cell_coefficients: Tensor,
        *,
        minimum_radius: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_sites_and_radii(sites, radii, allow_empty=False)
        if sites.shape[1] != 2:
            raise ValueError("SpectralPowerFoam requires 2-D sites")
        if (
            not cell_coefficients.is_floating_point()
            or cell_coefficients.ndim != 2
            or cell_coefficients.shape[0] != sites.shape[0]
            or cell_coefficients.shape[1] == 0
        ):
            raise ValueError("cell_coefficients must have shape (cells, basis>=1)")
        if (
            cell_coefficients.device != sites.device
            or cell_coefficients.dtype != sites.dtype
        ):
            raise ValueError(
                "sites, radii, and cell_coefficients must share dtype and device"
            )
        if not bool(torch.isfinite(cell_coefficients).all()):
            raise ValueError("cell_coefficients must be finite")
        radius_floor = float(minimum_radius)
        if not math.isfinite(radius_floor) or radius_floor < 0.0:
            raise ValueError("minimum_radius must be finite and non-negative")
        shifted = radii.detach() - radius_floor
        if bool(torch.any(shifted < 0)):
            raise ValueError("initial radii must be at least minimum_radius")
        scale = torch.clamp(radii.detach().abs().max(), min=1.0)
        shifted = torch.clamp(
            shifted,
            min=torch.finfo(radii.dtype).eps * scale,
        )
        raw_radii = shifted + torch.log(-torch.expm1(-shifted))

        self.sites = torch.nn.Parameter(sites.detach().clone())
        self.raw_radii = torch.nn.Parameter(raw_radii)
        self.cell_coefficients = torch.nn.Parameter(cell_coefficients.detach().clone())
        self.register_buffer(
            "_minimum_radius",
            radii.new_tensor(radius_floor),
        )

    @property
    def radii(self) -> Tensor:
        """Strictly positive trainable radii in the geometry coordinate unit."""
        return self._minimum_radius + torch.nn.functional.softplus(self.raw_radii)

    @property
    def basis_count(self) -> int:
        return int(self.cell_coefficients.shape[1])

    def forward(
        self,
        points: Tensor,
        basis_values: Tensor,
        *,
        power_temperature: float = 10.0,
        support_temperature: float = 10.0,
        compact_fixed_geometry: bool = False,
    ) -> Tensor:
        """Evaluate point samples; uncovered union mass contributes zero.

        ``compact_fixed_geometry`` uses exact bounded support and is intended for
        appearance-only optimization. It avoids the dense point-by-all-cells
        matrix, but deliberately rejects trainable sites or radii.
        """
        if (
            not basis_values.is_floating_point()
            or basis_values.ndim < 1
            or basis_values.shape[-1] != self.basis_count
        ):
            raise ValueError("basis_values must have shape (..., basis)")
        if (
            basis_values.device != self.sites.device
            or basis_values.dtype != self.sites.dtype
        ):
            raise ValueError("basis_values must match model dtype and device")
        if basis_values.ndim > 1 and basis_values.shape[:-1] != points.shape[:-1]:
            raise ValueError("non-scalar basis_values leading shape must match points")
        selected: Tensor | None = None
        if compact_fixed_geometry:
            if self.sites.requires_grad or self.raw_radii.requires_grad:
                raise ValueError(
                    "compact_fixed_geometry requires frozen sites and radii"
                )
            support_scale = float(support_temperature)
            if not math.isfinite(support_scale) or support_scale <= 0:
                raise ValueError("support_temperature must be finite and positive")
            flat_points = points.reshape(-1, points.shape[-1])
            if flat_points.shape[0] == 0:
                return points[..., 0] * self.cell_coefficients.sum()
            with torch.no_grad():
                lower = flat_points.amin(dim=0)
                upper = flat_points.amax(dim=0)
                # ponytail: this O(cells) scan is cheaper than maintaining a
                # spatial index here; switch to cached tile CSR if profiling says so.
                closest = torch.minimum(torch.maximum(self.sites, lower), upper)
                selected = torch.nonzero(
                    torch.sum((self.sites - closest) ** 2, dim=1)
                    <= self.radii**2,
                    as_tuple=False,
                ).flatten()
            if selected.numel() == 0:
                return points[..., 0] * 0.0 + self.cell_coefficients.sum() * 0.0
            weights = soft_bounded_power_weights(
                points,
                self.sites[selected],
                self.radii[selected],
                temperature=power_temperature,
            )
        else:
            weights = smooth_bounded_power_weights(
                points,
                self.sites,
                self.radii,
                power_temperature=power_temperature,
                support_temperature=support_temperature,
            )
        coefficients = (
            self.cell_coefficients
            if selected is None
            else self.cell_coefficients[selected]
        )
        cell_values = torch.einsum(
            "...k,ck->...c",
            basis_values,
            coefficients,
        )
        return torch.sum(weights * cell_values, dim=-1)

    def render_exact(
        self,
        operator: PowerPixelOperator,
        basis_samples: Tensor,
        response_weights: Tensor,
        *,
        semantics: PowerValueSemantics = "surface_brightness",
    ) -> Tensor:
        """Render hard pixels with an operator built from detached model geometry."""
        if operator.cell_count != self.sites.shape[0]:
            raise ValueError("operator cell count must match the model")
        return render_power_spectral_basis(
            operator,
            self.cell_coefficients,
            basis_samples,
            response_weights,
            semantics=semantics,
        )


def cech_adjacency(
    sites: Tensor,
    radii: Tensor,
    *,
    site_chunk_size: int = 1024,
) -> Tensor:
    """Return undirected edges between all overlapping bounding balls.

    The result has shape ``(2, E)`` with each edge stored once as ``i < j``.
    This exact, bounded-memory reference still performs quadratic pair tests;
    survey-scale callers should use a spatial candidate index before this test.
    """
    _validate_power_inputs(sites[:1], sites, radii)
    chunk_size = int(site_chunk_size)
    if chunk_size <= 0:
        raise ValueError("site_chunk_size must be positive")
    edges: list[Tensor] = []
    count = sites.shape[0]
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        delta = sites[start:stop, None, :] - sites[None, :, :]
        distance_squared = torch.sum(delta * delta, dim=-1)
        radius_sum = radii[start:stop, None] + radii[None, :]
        local_i = torch.arange(start, stop, device=sites.device)[:, None]
        global_j = torch.arange(count, device=sites.device)[None, :]
        overlap = (local_i < global_j) & (distance_squared <= radius_sum * radius_sum)
        ij = torch.nonzero(overlap, as_tuple=False)
        if ij.numel() != 0:
            ij[:, 0] += start
            edges.append(ij)
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=sites.device)
    return torch.cat(edges, dim=0).transpose(0, 1).contiguous()


def _empty_sparse_cech_graph(
    sites: Tensor,
    *,
    bin_size: float,
    site_bin_memberships: int = 0,
    occupied_bins: int = 0,
) -> SparseCechGraph:
    empty_pairs = torch.empty((2, 0), dtype=torch.long, device=sites.device)
    return SparseCechGraph(
        candidate_pairs=empty_pairs,
        edges=empty_pairs,
        indptr=torch.zeros(
            sites.shape[0] + 1,
            dtype=torch.long,
            device=sites.device,
        ),
        indices=torch.empty((0,), dtype=torch.long, device=sites.device),
        total_pairs=sites.shape[0] * (sites.shape[0] - 1) // 2,
        site_bin_memberships=site_bin_memberships,
        occupied_bins=occupied_bins,
        bin_size=bin_size,
    )


def sparse_cech_graph(
    sites: Tensor,
    radii: Tensor,
    *,
    bin_size: float | None = None,
    max_site_bin_memberships: int = 10_000_000,
    max_candidate_pairs: int = 50_000_000,
) -> SparseCechGraph:
    """Build an exact 2-D Čech graph through a Torch-native uniform-grid index.

    Each disk is inserted into every grid bin touched by its axis-aligned
    bounding box. Candidate pairs share at least one bin, then the exact
    disk-overlap predicate is applied. ``edges`` stores each overlap once as
    ``i < j``; ``indptr`` and ``indices`` store the same graph symmetrically.
    """
    _validate_sites_and_radii(sites, radii, allow_empty=True)
    if sites.shape[1] != 2:
        raise ValueError("sparse_cech_graph currently requires 2-D disk sites")
    membership_limit = int(max_site_bin_memberships)
    candidate_limit = int(max_candidate_pairs)
    if membership_limit <= 0 or candidate_limit <= 0:
        raise ValueError("sparse Čech allocation limits must be positive")

    if bin_size is None:
        max_radius = float(radii.max().item()) if radii.numel() else 0.0
        width = 2.0 * max_radius if max_radius > 0.0 else 1.0
    else:
        width = float(bin_size)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("bin_size must be finite and positive")
    if sites.shape[0] < 2:
        return _empty_sparse_cech_graph(sites, bin_size=width)

    with torch.no_grad():
        index_dtype = torch.float32 if sites.device.type == "mps" else torch.float64
        index_sites = sites.detach().to(dtype=index_dtype)
        index_radii = radii.detach().to(dtype=index_dtype).unsqueeze(-1)
        lower = index_sites - index_radii
        upper = index_sites + index_radii
        origin = lower.amin(dim=0)
        lower_bin = torch.floor((lower - origin) / width).to(dtype=torch.long)
        upper_bin = torch.floor((upper - origin) / width).to(dtype=torch.long)
        lower_bin = lower_bin.clamp_min(0)
        upper_bin = torch.maximum(upper_bin, lower_bin)
        spans = upper_bin - lower_bin + 1
        membership_counts = spans[:, 0] * spans[:, 1]
        membership_count = int(membership_counts.sum().item())
        if membership_count > membership_limit:
            raise ValueError(
                f"uniform-grid index needs {membership_count} site-bin memberships, "
                f"exceeding max_site_bin_memberships={membership_limit}; increase "
                "bin_size or the explicit limit"
            )

        site_ids = torch.repeat_interleave(
            torch.arange(sites.shape[0], device=sites.device),
            membership_counts,
        )
        membership_offsets = torch.cumsum(membership_counts, dim=0) - membership_counts
        local = torch.arange(membership_count, device=sites.device) - (
            torch.repeat_interleave(membership_offsets, membership_counts)
        )
        span_x = spans[site_ids, 0]
        bin_x = lower_bin[site_ids, 0] + torch.remainder(local, span_x)
        bin_y = lower_bin[site_ids, 1] + torch.div(
            local,
            span_x,
            rounding_mode="floor",
        )
        bin_columns = int(upper_bin[:, 0].max().item()) + 1
        max_bin_y = int(upper_bin[:, 1].max().item())
        if max_bin_y > (torch.iinfo(torch.long).max - bin_columns + 1) // bin_columns:
            raise ValueError("uniform-grid key range exceeds int64")
        bin_keys = bin_y * bin_columns + bin_x

        order = torch.argsort(site_ids, stable=True)
        order = order[torch.argsort(bin_keys[order], stable=True)]
        sorted_keys = bin_keys[order]
        sorted_sites = site_ids[order]
        _, bin_counts = torch.unique_consecutive(sorted_keys, return_counts=True)
        occupied_bins = int(bin_counts.numel())
        pair_counts = bin_counts * (bin_counts - 1) // 2
        raw_pair_count = int(pair_counts.sum().item())
        if raw_pair_count > candidate_limit:
            raise ValueError(
                f"uniform-grid bins produce {raw_pair_count} raw candidate pairs, "
                f"exceeding max_candidate_pairs={candidate_limit}; decrease "
                "occupancy with a smaller bin_size or raise the explicit limit"
            )
        if raw_pair_count == 0:
            return _empty_sparse_cech_graph(
                sites,
                bin_size=width,
                site_bin_memberships=membership_count,
                occupied_bins=occupied_bins,
            )

        group_ids = torch.repeat_interleave(
            torch.arange(occupied_bins, device=sites.device),
            pair_counts,
        )
        pair_offsets = torch.cumsum(pair_counts, dim=0) - pair_counts
        pair_rank = torch.arange(raw_pair_count, device=sites.device) - (
            torch.repeat_interleave(pair_offsets, pair_counts)
        )
        group_size = bin_counts[group_ids]
        solve_dtype = torch.float32 if sites.device.type == "mps" else torch.float64
        diagonal = (2 * group_size - 1).to(dtype=solve_dtype)
        row = torch.floor(
            0.5
            * (
                diagonal
                - torch.sqrt(
                    diagonal * diagonal - 8.0 * pair_rank.to(dtype=solve_dtype)
                )
            )
        ).to(dtype=torch.long)
        row_start = row * (2 * group_size - row - 1) // 2
        row = torch.where(row_start > pair_rank, row - 1, row)
        row_start = row * (2 * group_size - row - 1) // 2
        next_start = (row + 1) * (2 * group_size - row - 2) // 2
        row = torch.where(next_start <= pair_rank, row + 1, row)
        row_start = row * (2 * group_size - row - 1) // 2
        column = row + 1 + pair_rank - row_start
        group_start = (torch.cumsum(bin_counts, dim=0) - bin_counts)[group_ids]
        pairs = torch.stack(
            (
                sorted_sites[group_start + row],
                sorted_sites[group_start + column],
            ),
            dim=1,
        )
        pairs = torch.unique(torch.sort(pairs, dim=1).values, dim=0, sorted=True)
        pair_order = torch.argsort(pairs[:, 1], stable=True)
        pairs = pairs[pair_order]
        pairs = pairs[torch.argsort(pairs[:, 0], stable=True)]
        candidate_pairs = pairs.transpose(0, 1).contiguous()

        left, right = candidate_pairs
        delta = sites[left] - sites[right]
        radius_sum = radii[left] + radii[right]
        overlap = torch.sum(delta * delta, dim=-1) <= radius_sum * radius_sum
        edges = candidate_pairs[:, overlap]
        if edges.shape[1] == 0:
            indptr = torch.zeros(
                sites.shape[0] + 1,
                dtype=torch.long,
                device=sites.device,
            )
            indices = torch.empty((0,), dtype=torch.long, device=sites.device)
        else:
            rows = torch.cat((edges[0], edges[1]))
            columns = torch.cat((edges[1], edges[0]))
            csr_order = torch.argsort(columns, stable=True)
            csr_order = csr_order[torch.argsort(rows[csr_order], stable=True)]
            rows = rows[csr_order]
            indices = columns[csr_order]
            row_counts = torch.bincount(rows, minlength=sites.shape[0])
            indptr = torch.cat(
                (
                    torch.zeros(1, dtype=torch.long, device=sites.device),
                    torch.cumsum(row_counts, dim=0),
                )
            )

    return SparseCechGraph(
        candidate_pairs=candidate_pairs,
        edges=edges,
        indptr=indptr,
        indices=indices,
        total_pairs=sites.shape[0] * (sites.shape[0] - 1) // 2,
        site_bin_memberships=membership_count,
        occupied_bins=occupied_bins,
        bin_size=width,
    )


def tile_disk_candidate_csr(
    sites: Tensor,
    radii: Tensor,
    *,
    tile_shape: tuple[int, int],
    tile_origin: tuple[float, float] = (0.0, 0.0),
    tile_size: tuple[float, float] = (1.0, 1.0),
    max_aabb_memberships: int = 50_000_000,
) -> TileDiskCSR:
    """Return exact disk-to-tile candidates for a regular 2-D tile grid.

    Coordinates use a local tangent plane with ``(x, y)`` sites, an ``(x, y)``
    lower-left grid origin, an ``(x, y)`` tile size, and a ``(rows, columns)``
    tile shape. Tiles are flattened in row-major order. A disk touching a tile
    boundary is included in every adjacent tile sharing that boundary.

    The index enumerates only the tiles covered by each disk's AABB, then
    applies the exact closed disk/AABB intersection predicate. It never
    allocates a ``number_of_tiles × number_of_sites`` matrix.
    """
    _validate_sites_and_radii(sites, radii, allow_empty=True)
    if sites.shape[1] != 2:
        raise ValueError("tile_disk_candidate_csr requires 2-D disk sites")
    if len(tile_shape) != 2:
        raise ValueError("tile_shape must contain (rows, columns)")
    rows, columns = (int(tile_shape[0]), int(tile_shape[1]))
    if rows <= 0 or columns <= 0:
        raise ValueError("tile_shape entries must be positive")
    if len(tile_origin) != 2 or len(tile_size) != 2:
        raise ValueError("tile_origin and tile_size must contain (x, y)")
    origin_x, origin_y = (float(tile_origin[0]), float(tile_origin[1]))
    size_x, size_y = (float(tile_size[0]), float(tile_size[1]))
    if not all(math.isfinite(value) for value in (origin_x, origin_y)):
        raise ValueError("tile_origin entries must be finite")
    if not all(math.isfinite(value) and value > 0.0 for value in (size_x, size_y)):
        raise ValueError("tile_size entries must be finite and positive")
    membership_limit = int(max_aabb_memberships)
    if membership_limit <= 0:
        raise ValueError("max_aabb_memberships must be positive")
    tile_count = rows * columns
    if tile_count > torch.iinfo(torch.long).max:
        raise ValueError("tile grid exceeds int64 indexing")

    empty = torch.empty((0,), dtype=torch.long, device=sites.device)
    if sites.shape[0] == 0:
        return TileDiskCSR(
            indptr=torch.zeros(tile_count + 1, dtype=torch.long, device=sites.device),
            indices=empty,
            tile_ids=empty,
            aabb_memberships=0,
            total_pairs=0,
            tile_shape=(rows, columns),
            tile_origin=(origin_x, origin_y),
            tile_size=(size_x, size_y),
        )

    with torch.no_grad():
        index_dtype = torch.float32 if sites.device.type == "mps" else torch.float64
        index_sites = sites.detach().to(dtype=index_dtype)
        index_radii = radii.detach().to(dtype=index_dtype)
        grid_origin = torch.tensor(
            [origin_x, origin_y],
            dtype=index_dtype,
            device=sites.device,
        )
        grid_size = torch.tensor(
            [size_x, size_y],
            dtype=index_dtype,
            device=sites.device,
        )
        lower_scaled = (index_sites - index_radii[:, None] - grid_origin) / grid_size
        upper_scaled = (index_sites + index_radii[:, None] - grid_origin) / grid_size
        # Closed support includes the tile immediately left/below an exact boundary.
        lower_tile = torch.ceil(lower_scaled).to(torch.long) - 1
        upper_tile = torch.floor(upper_scaled).to(torch.long)
        lower_tile[:, 0].clamp_(0, columns - 1)
        upper_tile[:, 0].clamp_(0, columns - 1)
        lower_tile[:, 1].clamp_(0, rows - 1)
        upper_tile[:, 1].clamp_(0, rows - 1)

        grid_upper = grid_origin + grid_size * torch.tensor(
            [columns, rows],
            dtype=index_dtype,
            device=sites.device,
        )
        outside = torch.any(
            (index_sites + index_radii[:, None] < grid_origin)
            | (index_sites - index_radii[:, None] > grid_upper),
            dim=1,
        )
        spans = upper_tile - lower_tile + 1
        membership_counts = spans[:, 0] * spans[:, 1]
        membership_counts = torch.where(
            outside,
            torch.zeros_like(membership_counts),
            membership_counts,
        )
        membership_count = int(membership_counts.sum().item())
        if membership_count > membership_limit:
            raise ValueError(
                f"disk AABBs cover {membership_count} tile memberships, exceeding "
                f"max_aabb_memberships={membership_limit}; increase tile size or "
                "the explicit limit"
            )
        if membership_count == 0:
            return TileDiskCSR(
                indptr=torch.zeros(
                    tile_count + 1,
                    dtype=torch.long,
                    device=sites.device,
                ),
                indices=empty,
                tile_ids=empty,
                aabb_memberships=0,
                total_pairs=tile_count * sites.shape[0],
                tile_shape=(rows, columns),
                tile_origin=(origin_x, origin_y),
                tile_size=(size_x, size_y),
            )

        site_ids = torch.repeat_interleave(
            torch.arange(sites.shape[0], device=sites.device),
            membership_counts,
        )
        membership_offsets = torch.cumsum(membership_counts, dim=0) - membership_counts
        local = torch.arange(membership_count, device=sites.device) - (
            torch.repeat_interleave(membership_offsets, membership_counts)
        )
        span_x = spans[site_ids, 0]
        tile_x = lower_tile[site_ids, 0] + torch.remainder(local, span_x)
        tile_y = lower_tile[site_ids, 1] + torch.div(
            local,
            span_x,
            rounding_mode="floor",
        )

        tile_lower = grid_origin + grid_size * torch.stack((tile_x, tile_y), dim=1)
        tile_upper = tile_lower + grid_size
        centers = index_sites[site_ids]
        closest = torch.minimum(torch.maximum(centers, tile_lower), tile_upper)
        intersects = torch.sum((centers - closest) ** 2, dim=1) <= (
            index_radii[site_ids] ** 2
        )
        site_ids = site_ids[intersects]
        tile_ids = (tile_y[intersects] * columns + tile_x[intersects]).to(torch.long)

        order = torch.argsort(site_ids, stable=True)
        order = order[torch.argsort(tile_ids[order], stable=True)]
        tile_ids = tile_ids[order]
        indices = site_ids[order]
        tile_counts = torch.bincount(tile_ids, minlength=tile_count)
        indptr = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=sites.device),
                torch.cumsum(tile_counts, dim=0),
            )
        )

    return TileDiskCSR(
        indptr=indptr,
        indices=indices,
        tile_ids=tile_ids,
        aabb_memberships=membership_count,
        total_pairs=tile_count * sites.shape[0],
        tile_shape=(rows, columns),
        tile_origin=(origin_x, origin_y),
        tile_size=(size_x, size_y),
    )


def polygon_disk_candidate_csr(
    pixel_polygons: Tensor,
    sites: Tensor,
    radii: Tensor,
    *,
    bin_size: float | None = None,
    max_bin_memberships: int = 50_000_000,
    max_candidate_pairs: int = 10_000_000,
) -> PolygonDiskCSR:
    """Return conservative disk candidates for arbitrary convex polygons.

    Polygon and disk AABBs are joined through a uniform grid, then candidate
    pairs that fail the exact disk/AABB intersection predicate are removed.
    The result can contain false positives for rotated polygons, but it cannot
    omit a disk that intersects a polygon.
    """
    _validate_sites_and_radii(sites, radii, allow_empty=True)
    if sites.shape[1] != 2:
        raise ValueError("polygon_disk_candidate_csr requires 2-D disk sites")
    if (
        not pixel_polygons.is_floating_point()
        or pixel_polygons.ndim != 3
        or pixel_polygons.shape[-1] != 2
        or pixel_polygons.shape[1] < 3
    ):
        raise ValueError("pixel_polygons must have floating shape (P, V>=3, 2)")
    if pixel_polygons.device != sites.device or pixel_polygons.dtype != sites.dtype:
        raise ValueError("pixels, sites, and radii must share dtype and device")
    if not bool(torch.isfinite(pixel_polygons).all()):
        raise ValueError("pixel polygons must be finite")
    membership_limit = int(max_bin_memberships)
    candidate_limit = int(max_candidate_pairs)
    if membership_limit <= 0 or candidate_limit <= 0:
        raise ValueError("sparse allocation limits must be positive")
    if bin_size is None:
        max_radius = float(radii.max().item()) if radii.numel() else 0.0
        width = max_radius if max_radius > 0.0 else 1.0
    else:
        width = float(bin_size)
    if not math.isfinite(width) or width <= 0:
        raise ValueError("bin_size must be finite and positive")

    polygon_count = pixel_polygons.shape[0]
    site_count = sites.shape[0]
    empty = torch.empty((0,), dtype=torch.long, device=sites.device)
    if polygon_count == 0 or site_count == 0:
        return PolygonDiskCSR(
            indptr=torch.zeros(
                polygon_count + 1, dtype=torch.long, device=sites.device
            ),
            indices=empty,
            polygon_ids=empty,
            polygon_bin_memberships=0,
            site_bin_memberships=0,
            raw_candidate_pairs=0,
            total_pairs=polygon_count * site_count,
            bin_size=width,
        )

    with torch.no_grad():
        index_dtype = torch.float32 if sites.device.type == "mps" else torch.float64
        polygons = pixel_polygons.detach().to(dtype=index_dtype)
        index_sites = sites.detach().to(dtype=index_dtype)
        index_radii = radii.detach().to(dtype=index_dtype)
        polygon_lower = polygons.amin(dim=1)
        polygon_upper = polygons.amax(dim=1)
        site_lower = index_sites - index_radii[:, None]
        site_upper = index_sites + index_radii[:, None]
        origin = torch.minimum(
            polygon_lower.amin(dim=0),
            site_lower.amin(dim=0),
        )

        def memberships(
            lower: Tensor,
            upper: Tensor,
        ) -> tuple[Tensor, Tensor, int]:
            lower_bin = torch.floor((lower - origin) / width).to(torch.long)
            upper_bin = torch.floor((upper - origin) / width).to(torch.long)
            spans = upper_bin - lower_bin + 1
            counts = spans[:, 0] * spans[:, 1]
            count = int(counts.sum().item())
            if count > membership_limit:
                raise ValueError(
                    f"uniform-grid index needs {count} memberships, exceeding "
                    f"max_bin_memberships={membership_limit}"
                )
            object_ids = torch.repeat_interleave(
                torch.arange(lower.shape[0], device=sites.device),
                counts,
            )
            starts = torch.cumsum(counts, dim=0) - counts
            local = torch.arange(count, device=sites.device) - torch.repeat_interleave(
                starts, counts
            )
            span_x = spans[object_ids, 0]
            bin_x = lower_bin[object_ids, 0] + torch.remainder(local, span_x)
            bin_y = lower_bin[object_ids, 1] + torch.div(
                local, span_x, rounding_mode="floor"
            )
            return object_ids, torch.stack((bin_x, bin_y), dim=1), count

        polygon_ids, polygon_bins, polygon_memberships = memberships(
            polygon_lower, polygon_upper
        )
        site_ids, site_bins, site_memberships = memberships(site_lower, site_upper)
        all_bins = torch.cat((polygon_bins, site_bins))
        min_bins = all_bins.amin(dim=0)
        all_bins = all_bins - min_bins
        columns = int(all_bins[:, 0].max().item()) + 1
        polygon_keys = (
            all_bins[:polygon_memberships, 1] * columns
            + all_bins[:polygon_memberships, 0]
        )
        site_keys = (
            all_bins[polygon_memberships:, 1] * columns
            + all_bins[polygon_memberships:, 0]
        )

        polygon_order = torch.argsort(polygon_ids, stable=True)
        polygon_order = polygon_order[
            torch.argsort(polygon_keys[polygon_order], stable=True)
        ]
        site_order = torch.argsort(site_ids, stable=True)
        site_order = site_order[torch.argsort(site_keys[site_order], stable=True)]
        polygon_keys = polygon_keys[polygon_order]
        polygon_ids = polygon_ids[polygon_order]
        site_keys = site_keys[site_order]
        site_ids = site_ids[site_order]
        unique_polygon_keys, polygon_counts = torch.unique_consecutive(
            polygon_keys, return_counts=True
        )
        unique_site_keys, site_counts = torch.unique_consecutive(
            site_keys, return_counts=True
        )
        polygon_starts = torch.cumsum(polygon_counts, dim=0) - polygon_counts
        site_starts = torch.cumsum(site_counts, dim=0) - site_counts
        site_groups = {
            int(key): (int(start), int(count))
            for key, start, count in zip(
                unique_site_keys.tolist(),
                site_starts.tolist(),
                site_counts.tolist(),
                strict=True,
            )
        }
        pair_pixels: list[Tensor] = []
        pair_sites: list[Tensor] = []
        raw_count = 0
        for key, start, count in zip(
            unique_polygon_keys.tolist(),
            polygon_starts.tolist(),
            polygon_counts.tolist(),
            strict=True,
        ):
            site_group = site_groups.get(int(key))
            if site_group is None:
                continue
            site_start, site_count_in_bin = site_group
            group_count = int(count) * site_count_in_bin
            raw_count += group_count
            if raw_count > candidate_limit:
                raise ValueError(
                    f"uniform-grid join needs {raw_count} raw candidate pairs, "
                    f"exceeding max_candidate_pairs={candidate_limit}"
                )
            pair_pixels.append(
                polygon_ids[int(start) : int(start) + int(count)].repeat_interleave(
                    site_count_in_bin
                )
            )
            pair_sites.append(
                site_ids[site_start : site_start + site_count_in_bin].repeat(int(count))
            )
        if not pair_pixels:
            candidate_pixels = empty
            candidate_sites = empty
        else:
            candidate_pixels = torch.cat(pair_pixels)
            candidate_sites = torch.cat(pair_sites)
            keys = torch.unique(
                candidate_pixels * site_count + candidate_sites,
                sorted=True,
            )
            candidate_pixels = torch.div(keys, site_count, rounding_mode="floor")
            candidate_sites = torch.remainder(keys, site_count)
            closest = torch.minimum(
                torch.maximum(
                    index_sites[candidate_sites], polygon_lower[candidate_pixels]
                ),
                polygon_upper[candidate_pixels],
            )
            intersects = (
                torch.sum((index_sites[candidate_sites] - closest) ** 2, dim=1)
                <= index_radii[candidate_sites] ** 2
            )
            candidate_pixels = candidate_pixels[intersects]
            candidate_sites = candidate_sites[intersects]
        counts = torch.bincount(candidate_pixels, minlength=polygon_count)
        indptr = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=sites.device),
                torch.cumsum(counts, dim=0),
            )
        )
    return PolygonDiskCSR(
        indptr=indptr,
        indices=candidate_sites,
        polygon_ids=candidate_pixels,
        polygon_bin_memberships=polygon_memberships,
        site_bin_memberships=site_memberships,
        raw_candidate_pairs=raw_count,
        total_pairs=polygon_count * site_count,
        bin_size=width,
    )


def power_face_planes(
    sites: Tensor,
    radii: Tensor,
    edges: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return radical planes ``normal · x = offset`` for site-pair edges."""
    _validate_power_inputs(sites[:1], sites, radii)
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long:
        raise ValueError("edges must be a torch.long tensor with shape (2, E)")
    if edges.device != sites.device:
        raise ValueError("edges and sites must be on the same device")
    if edges.numel() and (
        bool(torch.any(edges < 0)) or bool(torch.any(edges >= sites.shape[0]))
    ):
        raise ValueError("edges contain an out-of-range site index")
    left, right = edges[0], edges[1]
    left_sites = sites[left]
    right_sites = sites[right]
    normals = right_sites - left_sites
    offsets = 0.5 * (
        torch.sum(right_sites * right_sites, dim=-1)
        - torch.sum(left_sites * left_sites, dim=-1)
        + radii[left] * radii[left]
        - radii[right] * radii[right]
    )
    return normals, offsets


def _polygon_signed_area(vertices: Tensor) -> Tensor:
    following = torch.roll(vertices, shifts=-1, dims=0)
    return 0.5 * torch.sum(
        vertices[:, 0] * following[:, 1] - vertices[:, 1] * following[:, 0]
    )


def _clip_polygon_half_plane(
    vertices: Tensor,
    normal: Tensor,
    offset: Tensor,
) -> Tensor:
    """Clip one convex polygon by ``normal · x <= offset``."""
    if vertices.shape[0] == 0:
        return vertices
    values = vertices @ normal - offset
    scale = (
        torch.abs(offset)
        + torch.linalg.vector_norm(normal)
        * torch.max(torch.linalg.vector_norm(vertices, dim=1))
        + 1.0
    )
    tolerance = 64.0 * torch.finfo(vertices.dtype).eps * scale
    inside = values <= tolerance
    output: list[Tensor] = []
    for index in range(vertices.shape[0]):
        following = (index + 1) % vertices.shape[0]
        start = vertices[index]
        stop = vertices[following]
        start_inside = bool(inside[index].item())
        stop_inside = bool(inside[following].item())
        if start_inside:
            output.append(start)
        if start_inside != stop_inside:
            denominator = values[index] - values[following]
            fraction = torch.clamp(values[index] / denominator, 0.0, 1.0)
            output.append(start + fraction * (stop - start))
    if not output:
        return vertices.new_empty((0, 2))
    return torch.stack(output)


def _segment_circle_area(start: Tensor, stop: Tensor, radius: Tensor) -> Tensor:
    """Return the signed disk intersection area swept by one polygon edge."""
    delta = stop - start
    quadratic = torch.dot(delta, delta)
    if bool(quadratic == 0):
        return start.new_zeros(())
    linear = 2.0 * torch.dot(start, delta)
    constant = torch.dot(start, start) - radius * radius
    discriminant = linear * linear - 4.0 * quadratic * constant
    parameters = [start.new_zeros(()), start.new_ones(())]
    tolerance = (
        64.0
        * torch.finfo(start.dtype).eps
        * (torch.abs(linear * linear) + torch.abs(4.0 * quadratic * constant) + 1.0)
    )
    if bool(discriminant >= -tolerance):
        root = torch.sqrt(torch.clamp_min(discriminant, 0.0))
        for parameter in (
            (-linear - root) / (2.0 * quadratic),
            (-linear + root) / (2.0 * quadratic),
        ):
            if bool((parameter > 0.0) & (parameter < 1.0)):
                parameters.append(parameter)
    parameters.sort(key=lambda value: float(value.item()))

    area = start.new_zeros(())
    for lower, upper in zip(parameters[:-1], parameters[1:], strict=True):
        first = start + lower * delta
        second = start + upper * delta
        midpoint = 0.5 * (first + second)
        cross = first[0] * second[1] - first[1] * second[0]
        if bool(torch.dot(midpoint, midpoint) <= radius * radius + tolerance):
            area = area + 0.5 * cross
        else:
            area = area + 0.5 * radius * radius * torch.atan2(
                cross,
                torch.dot(first, second),
            )
    return area


def _polygon_disk_intersection_area(
    vertices: Tensor,
    center: Tensor,
    radius: Tensor,
) -> Tensor:
    """Analytic area of a polygon intersected with a closed disk."""
    if vertices.shape[0] < 3 or bool(radius <= 0):
        return vertices.new_zeros(())
    shifted = vertices - center
    squared_distance = torch.sum(shifted * shifted, dim=1)
    containment_tolerance = (
        64.0
        * torch.finfo(vertices.dtype).eps
        * (radius * radius + torch.max(squared_distance) + 1.0)
    )
    if bool(torch.all(squared_distance <= radius * radius + containment_tolerance)):
        return torch.abs(_polygon_signed_area(vertices))
    area = vertices.new_zeros(())
    for index in range(shifted.shape[0]):
        area = area + _segment_circle_area(
            shifted[index],
            shifted[(index + 1) % shifted.shape[0]],
            radius,
        )
    return torch.abs(area)


def _cell_clip_polygon(
    vertices: Tensor,
    *,
    cell: int,
    sites: Tensor,
    radii: Tensor,
    neighbors: Tensor,
) -> Tensor:
    clipped = vertices
    for raw_neighbor in neighbors:
        neighbor = int(raw_neighbor.item())
        if neighbor == cell:
            continue
        normal = sites[neighbor] - sites[cell]
        offset = 0.5 * (
            torch.dot(sites[neighbor], sites[neighbor])
            - torch.dot(sites[cell], sites[cell])
            + radii[cell] * radii[cell]
            - radii[neighbor] * radii[neighbor]
        )
        if bool(torch.all(normal == 0)):
            if bool(offset < 0) or (bool(offset == 0) and neighbor < cell):
                return vertices.new_empty((0, 2))
            continue
        clipped = _clip_polygon_half_plane(clipped, normal, offset)
        if clipped.shape[0] == 0:
            break
    return clipped


def _validate_csr(
    indptr: Tensor,
    indices: Tensor,
    *,
    rows: int,
    columns: int,
    name: str,
    device: torch.device,
) -> None:
    if (
        indptr.dtype != torch.long
        or indices.dtype != torch.long
        or indptr.device != device
        or indices.device != device
    ):
        raise ValueError(f"{name} CSR tensors must be torch.long on the input device")
    if indptr.shape != (rows + 1,) or indices.ndim != 1:
        raise ValueError(f"{name} CSR has an invalid shape")
    if (
        int(indptr[0].item()) != 0
        or int(indptr[-1].item()) != indices.numel()
        or bool(torch.any(indptr[1:] < indptr[:-1]))
    ):
        raise ValueError(f"{name} indptr must be monotonic and span indices")
    if indices.numel() and (
        bool(torch.any(indices < 0)) or bool(torch.any(indices >= columns))
    ):
        raise ValueError(f"{name} indices contain an out-of-range entry")


def bounded_power_pixel_operator(
    pixel_polygons: Tensor,
    sites: Tensor,
    radii: Tensor,
    *,
    candidate_indptr: Tensor | None = None,
    candidate_indices: Tensor | None = None,
    neighbor_indptr: Tensor | None = None,
    neighbor_indices: Tensor | None = None,
    max_candidate_pairs: int = 1_000_000,
) -> PowerPixelOperator:
    """Build an exact sparse operator from bounded power cells to pixels.

    Pixels are finite convex polygons with shape ``(P, V, 2)``. Candidate CSR
    may conservatively include zero-area disk/pixel pairs; those are removed
    after exact radical-half-plane clipping and analytic disk/polygon
    integration. Neighbor CSR may use :func:`sparse_cech_graph`; omitted
    neighbors use every site and are intended only as a small correctness path.
    Curved WCS edges must first be adaptively tessellated to an error budget;
    that controlled approximation is intentionally outside this exact
    straight-edge operator and must not be reported as analytic pixel area.
    """
    _validate_sites_and_radii(sites, radii, allow_empty=True)
    if sites.shape[1] != 2:
        raise ValueError("bounded_power_pixel_operator requires 2-D sites")
    if (
        not pixel_polygons.is_floating_point()
        or pixel_polygons.ndim != 3
        or pixel_polygons.shape[-1] != 2
        or pixel_polygons.shape[1] < 3
    ):
        raise ValueError("pixel_polygons must have floating shape (P, V>=3, 2)")
    if pixel_polygons.device != sites.device or pixel_polygons.dtype != sites.dtype:
        raise ValueError("pixels, sites, and radii must share dtype and device")
    if not bool(torch.isfinite(pixel_polygons).all()):
        raise ValueError("pixel polygons must be finite")
    signed_pixel_areas = (
        torch.stack([_polygon_signed_area(polygon) for polygon in pixel_polygons])
        if pixel_polygons.shape[0]
        else pixel_polygons.new_empty((0,))
    )
    if pixel_polygons.shape[0]:
        if bool(torch.any(torch.abs(signed_pixel_areas) == 0)):
            raise ValueError("pixel polygons must have non-zero area")
    edge_a = torch.roll(pixel_polygons, shifts=-1, dims=1) - pixel_polygons
    edge_b = torch.roll(edge_a, shifts=-1, dims=1)
    turns = edge_a[..., 0] * edge_b[..., 1] - edge_a[..., 1] * edge_b[..., 0]
    positive = turns > 0
    negative = turns < 0
    if bool(torch.any(positive.any(dim=1) & negative.any(dim=1))):
        raise ValueError("pixel polygons must be convex and consistently ordered")

    pixel_count = pixel_polygons.shape[0]
    cell_count = sites.shape[0]
    candidate_limit = int(max_candidate_pairs)
    if candidate_limit <= 0:
        raise ValueError("max_candidate_pairs must be positive")
    if (candidate_indptr is None) != (candidate_indices is None):
        raise ValueError(
            "candidate_indptr and candidate_indices must be supplied together"
        )
    if candidate_indptr is None:
        candidate_count = pixel_count * cell_count
        if candidate_count > candidate_limit:
            raise ValueError(
                f"dense fallback needs {candidate_count} candidate pairs, exceeding "
                f"max_candidate_pairs={candidate_limit}; supply sparse candidate CSR"
            )
        candidate_indptr = (
            torch.arange(
                pixel_count + 1,
                dtype=torch.long,
                device=sites.device,
            )
            * cell_count
        )
        candidate_indices = torch.arange(
            cell_count,
            dtype=torch.long,
            device=sites.device,
        ).repeat(pixel_count)
    assert candidate_indices is not None
    _validate_csr(
        candidate_indptr,
        candidate_indices,
        rows=pixel_count,
        columns=cell_count,
        name="candidate",
        device=sites.device,
    )
    if candidate_indices.numel() > candidate_limit:
        raise ValueError("candidate CSR exceeds max_candidate_pairs")

    if (neighbor_indptr is None) != (neighbor_indices is None):
        raise ValueError(
            "neighbor_indptr and neighbor_indices must be supplied together"
        )
    if neighbor_indptr is not None:
        assert neighbor_indices is not None
        _validate_csr(
            neighbor_indptr,
            neighbor_indices,
            rows=cell_count,
            columns=cell_count,
            name="neighbor",
            device=sites.device,
        )
    all_cells = torch.arange(cell_count, dtype=torch.long, device=sites.device)

    def neighbors(cell: int) -> Tensor:
        if neighbor_indptr is None:
            return all_cells
        assert neighbor_indices is not None
        return neighbor_indices[neighbor_indptr[cell] : neighbor_indptr[cell + 1]]

    active_cells = torch.unique(candidate_indices, sorted=True)
    cell_areas = sites.new_zeros((cell_count,))
    for raw_cell in active_cells:
        cell = int(raw_cell.item())
        radius = radii[cell]
        lower = sites[cell] - radius
        upper = sites[cell] + radius
        bounds = torch.stack(
            (
                lower,
                torch.stack((upper[0], lower[1])),
                upper,
                torch.stack((lower[0], upper[1])),
            )
        )
        cell_polygon = _cell_clip_polygon(
            bounds,
            cell=cell,
            sites=sites,
            radii=radii,
            neighbors=neighbors(cell),
        )
        cell_areas[cell] = _polygon_disk_intersection_area(
            cell_polygon,
            sites[cell],
            radius,
        )

    kept_pixels: list[int] = []
    kept_cells: list[int] = []
    kept_areas: list[Tensor] = []
    row_counts: list[int] = []
    for pixel in range(pixel_count):
        row = torch.unique(
            candidate_indices[candidate_indptr[pixel] : candidate_indptr[pixel + 1]],
            sorted=True,
        )
        before = len(kept_areas)
        for raw_cell in row:
            cell = int(raw_cell.item())
            clipped = _cell_clip_polygon(
                pixel_polygons[pixel],
                cell=cell,
                sites=sites,
                radii=radii,
                neighbors=neighbors(cell),
            )
            area = _polygon_disk_intersection_area(
                clipped,
                sites[cell],
                radii[cell],
            )
            if clipped.shape[0] >= 3:
                squared_distance = torch.sum(
                    (clipped - sites[cell]) ** 2,
                    dim=1,
                )
                containment_tolerance = (
                    64.0
                    * torch.finfo(area.dtype).eps
                    * (
                        radii[cell] * radii[cell]
                        + torch.max(squared_distance)
                        + 1.0
                    )
                )
                fully_contained = bool(
                    torch.all(
                        squared_distance
                        <= radii[cell] * radii[cell] + containment_tolerance
                    )
                )
            else:
                fully_contained = False
            area_tolerance = (
                64.0
                * torch.finfo(area.dtype).eps
                * (
                    torch.abs(signed_pixel_areas[pixel])
                    + math.pi * radii[cell] * radii[cell]
                )
            )
            pixel_tolerance = (
                64.0
                * torch.finfo(area.dtype).eps
                * torch.abs(signed_pixel_areas[pixel])
            )
            if bool(
                area > area_tolerance
                or (fully_contained and area > pixel_tolerance)
            ):
                kept_pixels.append(pixel)
                kept_cells.append(cell)
                kept_areas.append(area)
        row_counts.append(len(kept_areas) - before)

    indices = torch.tensor(kept_cells, dtype=torch.long, device=sites.device)
    pixel_ids = torch.tensor(kept_pixels, dtype=torch.long, device=sites.device)
    areas = torch.stack(kept_areas) if kept_areas else sites.new_empty((0,))
    counts = torch.tensor(row_counts, dtype=torch.long, device=sites.device)
    indptr = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=sites.device),
            torch.cumsum(counts, dim=0),
        )
    )
    order = torch.argsort(pixel_ids, stable=True)
    order = order[torch.argsort(indices[order], stable=True)]
    transpose_indices = pixel_ids[order]
    transpose_areas = areas[order]
    transpose_counts = torch.bincount(indices, minlength=cell_count)
    transpose_indptr = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=sites.device),
            torch.cumsum(transpose_counts, dim=0),
        )
    )
    return PowerPixelOperator(
        indptr=indptr,
        indices=indices,
        pixel_ids=pixel_ids,
        areas=areas,
        transpose_indptr=transpose_indptr,
        transpose_indices=transpose_indices,
        transpose_areas=transpose_areas,
        cell_areas=cell_areas,
        pixel_count=pixel_count,
        cell_count=cell_count,
        candidate_pairs=int(candidate_indices.numel()),
    )


def _padded_cell_neighbors(
    cell_ids: Tensor,
    *,
    neighbor_indptr: Tensor | None,
    neighbor_indices: Tensor | None,
    cell_count: int,
    max_neighbors: int,
) -> Tensor:
    if cell_ids.numel() == 0:
        return cell_ids.new_empty((0, max_neighbors))
    if neighbor_indptr is None:
        return torch.arange(
            cell_count,
            dtype=torch.long,
            device=cell_ids.device,
        ).expand(cell_ids.shape[0], -1)
    assert neighbor_indices is not None
    counts = neighbor_indptr[cell_ids + 1] - neighbor_indptr[cell_ids]
    rows = torch.repeat_interleave(
        torch.arange(cell_ids.shape[0], device=cell_ids.device),
        counts,
    )
    starts = torch.repeat_interleave(neighbor_indptr[cell_ids], counts)
    offsets = torch.arange(rows.shape[0], device=cell_ids.device) - (
        torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
    )
    padded = torch.full(
        (cell_ids.shape[0], max_neighbors),
        -1,
        dtype=torch.long,
        device=cell_ids.device,
    )
    padded[rows, offsets] = neighbor_indices[starts + offsets]
    return padded


def _batched_clip_power_cells(
    polygons: Tensor,
    cell_ids: Tensor,
    padded_neighbors: Tensor,
    sites: Tensor,
    radii: Tensor,
) -> tuple[Tensor, Tensor]:
    """Clip equal-vertex-count polygons without candidate-wise synchronization."""
    batch, initial_vertices, _ = polygons.shape
    capacity = initial_vertices + padded_neighbors.shape[1]
    vertices = polygons.new_zeros((batch, capacity, 2))
    vertices[:, :initial_vertices] = polygons
    counts = torch.full(
        (batch,),
        initial_vertices,
        dtype=torch.long,
        device=polygons.device,
    )
    positions = torch.arange(capacity, device=polygons.device).expand(batch, -1)
    cell_sites = sites[cell_ids]
    cell_radii = radii[cell_ids]

    for plane in range(padded_neighbors.shape[1]):
        neighbor_ids = padded_neighbors[:, plane]
        present = neighbor_ids >= 0
        safe_neighbors = neighbor_ids.clamp_min(0)
        neighbor_sites = sites[safe_neighbors]
        neighbor_radii = radii[safe_neighbors]
        normals = neighbor_sites - cell_sites
        offsets = 0.5 * (
            torch.sum(neighbor_sites * neighbor_sites, dim=1)
            - torch.sum(cell_sites * cell_sites, dim=1)
            + cell_radii * cell_radii
            - neighbor_radii * neighbor_radii
        )
        coincident = torch.all(normals == 0, dim=1)
        dominated = (
            present
            & coincident
            & ((offsets < 0) | ((offsets == 0) & (neighbor_ids < cell_ids)))
        )
        active = present & ~coincident & (counts > 0)

        safe_counts = counts.clamp_min(1)
        following = torch.remainder(positions + 1, safe_counts[:, None])
        stop = torch.gather(vertices, 1, following[..., None].expand(-1, -1, 2))
        start_values = (
            torch.sum(vertices * normals[:, None, :], dim=2) - offsets[:, None]
        )
        stop_values = torch.sum(stop * normals[:, None, :], dim=2) - offsets[:, None]
        vertex_scale = torch.max(
            torch.linalg.vector_norm(vertices, dim=2), dim=1
        ).values
        tolerance = (
            64.0
            * torch.finfo(vertices.dtype).eps
            * (
                torch.abs(offsets)
                + torch.linalg.vector_norm(normals, dim=1) * vertex_scale
                + 1.0
            )
        )
        valid_edges = positions < counts[:, None]
        start_inside = start_values <= tolerance[:, None]
        stop_inside = stop_values <= tolerance[:, None]
        emit_start = valid_edges & start_inside
        emit_intersection = valid_edges & (start_inside != stop_inside)
        denominator = start_values - stop_values
        fraction = torch.where(
            emit_intersection,
            start_values
            / torch.where(
                denominator != 0,
                denominator,
                torch.ones_like(denominator),
            ),
            torch.zeros_like(start_values),
        ).clamp(0.0, 1.0)
        intersection = vertices + fraction[..., None] * (stop - vertices)
        emitted = torch.stack((vertices, intersection), dim=2).reshape(
            batch,
            2 * capacity,
            2,
        )
        emit = torch.stack((emit_start, emit_intersection), dim=2).reshape(
            batch,
            2 * capacity,
        )
        destinations = (torch.cumsum(emit, dim=1) - 1).clamp_min(0)
        clipped = vertices.new_zeros(vertices.shape)
        clipped.scatter_add_(
            1,
            destinations[..., None].expand(-1, -1, 2),
            emitted * emit[..., None],
        )
        clipped_counts = torch.sum(emit, dim=1)
        vertices = torch.where(active[:, None, None], clipped, vertices)
        counts = torch.where(active, clipped_counts, counts)
        counts = torch.where(dominated, torch.zeros_like(counts), counts)
    return vertices, counts


def _batched_polygon_disk_area(
    vertices: Tensor,
    counts: Tensor,
    centers: Tensor,
    radii: Tensor,
) -> Tensor:
    """Analytic polygon/disk areas for a padded polygon batch."""
    batch, capacity, _ = vertices.shape
    positions = torch.arange(capacity, device=vertices.device).expand(batch, -1)
    safe_counts = counts.clamp_min(1)
    following = torch.remainder(positions + 1, safe_counts[:, None])
    start = vertices - centers[:, None, :]
    stop = (
        torch.gather(vertices, 1, following[..., None].expand(-1, -1, 2))
        - centers[:, None, :]
    )
    delta = stop - start
    quadratic = torch.sum(delta * delta, dim=2)
    safe_quadratic = torch.where(
        quadratic > 0,
        quadratic,
        torch.ones_like(quadratic),
    )
    linear = 2.0 * torch.sum(start * delta, dim=2)
    constant = torch.sum(start * start, dim=2) - radii[:, None] ** 2
    discriminant = linear * linear - 4.0 * quadratic * constant
    tolerance = (
        64.0
        * torch.finfo(vertices.dtype).eps
        * (torch.abs(linear * linear) + torch.abs(4.0 * quadratic * constant) + 1.0)
    )
    root = torch.sqrt(torch.clamp_min(discriminant, 0.0))
    lower_root = (-linear - root) / (2.0 * safe_quadratic)
    upper_root = (-linear + root) / (2.0 * safe_quadratic)
    roots_exist = discriminant >= -tolerance
    lower_valid = roots_exist & (lower_root > 0) & (lower_root < 1)
    upper_valid = roots_exist & (upper_root > 0) & (upper_root < 1)
    parameters = torch.stack(
        (
            torch.zeros_like(lower_root),
            torch.where(lower_valid, lower_root, torch.ones_like(lower_root)),
            torch.where(upper_valid, upper_root, torch.ones_like(upper_root)),
            torch.ones_like(lower_root),
        ),
        dim=2,
    )
    parameters = torch.sort(parameters, dim=2).values
    lower = parameters[..., :-1]
    upper = parameters[..., 1:]
    first = start[:, :, None, :] + lower[..., None] * delta[:, :, None, :]
    second = start[:, :, None, :] + upper[..., None] * delta[:, :, None, :]
    midpoint = 0.5 * (first + second)
    cross = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    triangle = 0.5 * cross
    sector = (
        0.5
        * radii[:, None, None] ** 2
        * torch.atan2(cross, torch.sum(first * second, dim=3))
    )
    inside = torch.sum(midpoint * midpoint, dim=3) <= (
        radii[:, None, None] ** 2 + tolerance[:, :, None]
    )
    intervals = torch.where(inside, triangle, sector)
    intervals = torch.where(
        upper > lower,
        intervals,
        torch.zeros_like(intervals),
    )
    valid_edges = (positions < counts[:, None]) & (counts[:, None] >= 3)
    intersection_area = torch.abs(
        torch.sum(intervals * valid_edges[..., None], dim=(1, 2))
    )
    vertex_distance_squared = torch.sum(start * start, dim=2)
    containment_tolerance = (
        64.0
        * torch.finfo(vertices.dtype).eps
        * (
            radii * radii
            + torch.max(
                torch.where(
                    valid_edges,
                    vertex_distance_squared,
                    torch.zeros_like(vertex_distance_squared),
                ),
                dim=1,
            ).values
            + 1.0
        )
    )
    fully_contained = (counts >= 3) & torch.all(
        (~valid_edges)
        | (
            vertex_distance_squared
            <= radii[:, None] * radii[:, None] + containment_tolerance[:, None]
        ),
        dim=1,
    )
    polygon_area = torch.abs(torch.sum(triangle[..., 0] * valid_edges, dim=1))
    return torch.where(fully_contained, polygon_area, intersection_area)


@lru_cache(maxsize=1)
def _compiled_batched_power_kernels() -> tuple[Any, Any]:
    """Create compiled numeric kernels lazily; validation and assembly stay eager."""
    return (
        torch.compile(_batched_clip_power_cells, fullgraph=True),
        torch.compile(_batched_polygon_disk_area, fullgraph=True),
    )


def batched_bounded_power_pixel_operator(
    pixel_polygons: Tensor,
    sites: Tensor,
    radii: Tensor,
    *,
    candidate_indptr: Tensor,
    candidate_indices: Tensor,
    neighbor_indptr: Tensor,
    neighbor_indices: Tensor,
    max_candidate_pairs: int = 1_000_000,
    max_packed_vertices: int = 64,
    candidate_chunk_size: int = 4096,
    kernel_execution: Literal["eager", "compile"] = "eager",
) -> PowerPixelOperator:
    """Build the exact operator with batched Torch clipping and area integration.

    This production candidate requires both sparse CSR directions. The scalar
    :func:`bounded_power_pixel_operator` remains the small-input oracle and dense
    fallback.
    """
    _validate_sites_and_radii(sites, radii, allow_empty=True)
    if sites.shape[1] != 2:
        raise ValueError("batched_bounded_power_pixel_operator requires 2-D sites")
    if (
        not pixel_polygons.is_floating_point()
        or pixel_polygons.ndim != 3
        or pixel_polygons.shape[-1] != 2
        or pixel_polygons.shape[1] < 3
    ):
        raise ValueError("pixel_polygons must have floating shape (P, V>=3, 2)")
    if pixel_polygons.device != sites.device or pixel_polygons.dtype != sites.dtype:
        raise ValueError("pixels, sites, and radii must share dtype and device")
    if not bool(torch.isfinite(pixel_polygons).all()):
        raise ValueError("pixel polygons must be finite")
    signed_pixel_areas = (
        torch.stack([_polygon_signed_area(polygon) for polygon in pixel_polygons])
        if pixel_polygons.shape[0]
        else pixel_polygons.new_empty((0,))
    )
    if bool(torch.any(torch.abs(signed_pixel_areas) == 0)):
        raise ValueError("pixel polygons must have non-zero area")
    edge_a = torch.roll(pixel_polygons, shifts=-1, dims=1) - pixel_polygons
    edge_b = torch.roll(edge_a, shifts=-1, dims=1)
    turns = edge_a[..., 0] * edge_b[..., 1] - edge_a[..., 1] * edge_b[..., 0]
    if bool(torch.any((turns > 0).any(dim=1) & (turns < 0).any(dim=1))):
        raise ValueError("pixel polygons must be convex and consistently ordered")
    pixel_count = pixel_polygons.shape[0]
    cell_count = sites.shape[0]
    _validate_csr(
        candidate_indptr,
        candidate_indices,
        rows=pixel_count,
        columns=cell_count,
        name="candidate",
        device=sites.device,
    )
    _validate_csr(
        neighbor_indptr,
        neighbor_indices,
        rows=cell_count,
        columns=cell_count,
        name="neighbor",
        device=sites.device,
    )
    if candidate_indices.numel() > int(max_candidate_pairs):
        raise ValueError("candidate CSR exceeds max_candidate_pairs")
    chunk_size = int(candidate_chunk_size)
    if chunk_size <= 0:
        raise ValueError("candidate_chunk_size must be positive")
    if kernel_execution == "eager":
        clip_cells = _batched_clip_power_cells
        polygon_disk_area = _batched_polygon_disk_area
    elif kernel_execution == "compile":
        clip_cells, polygon_disk_area = _compiled_batched_power_kernels()
    else:
        raise ValueError("kernel_execution must be 'eager' or 'compile'")

    candidate_pixels = torch.repeat_interleave(
        torch.arange(pixel_count, device=sites.device),
        candidate_indptr[1:] - candidate_indptr[:-1],
    )
    if cell_count:
        keys = candidate_pixels * cell_count + candidate_indices
        keys = torch.unique(keys, sorted=True)
        candidate_pixels = torch.div(keys, cell_count, rounding_mode="floor")
        candidate_cells = torch.remainder(keys, cell_count)
    else:
        candidate_cells = candidate_indices
    active_cells = torch.unique(candidate_cells, sorted=True)
    neighbor_counts = neighbor_indptr[1:] - neighbor_indptr[:-1]
    max_neighbors = (
        int(neighbor_counts[active_cells].max().item()) if active_cells.numel() else 0
    )
    if pixel_polygons.shape[1] + max_neighbors > int(max_packed_vertices):
        raise ValueError(
            "batched clipping exceeds max_packed_vertices; raise the explicit "
            "limit or use the scalar correctness oracle"
        )

    cell_areas = sites.new_zeros((cell_count,))
    for start in range(0, active_cells.shape[0], chunk_size):
        chunk_cells = active_cells[start : start + chunk_size]
        chunk_radii = radii[chunk_cells]
        lower = sites[chunk_cells] - chunk_radii[:, None]
        upper = sites[chunk_cells] + chunk_radii[:, None]
        bounds = torch.stack(
            (
                lower,
                torch.stack((upper[:, 0], lower[:, 1]), dim=1),
                upper,
                torch.stack((lower[:, 0], upper[:, 1]), dim=1),
            ),
            dim=1,
        )
        chunk_neighbors = _padded_cell_neighbors(
            chunk_cells,
            neighbor_indptr=neighbor_indptr,
            neighbor_indices=neighbor_indices,
            cell_count=cell_count,
            max_neighbors=max_neighbors,
        )
        cell_vertices, cell_counts = clip_cells(
            bounds,
            chunk_cells,
            chunk_neighbors,
            sites,
            radii,
        )
        cell_areas[chunk_cells] = polygon_disk_area(
            cell_vertices,
            cell_counts,
            sites[chunk_cells],
            chunk_radii,
        )

    kept_pixels: list[Tensor] = []
    kept_cells: list[Tensor] = []
    kept_areas: list[Tensor] = []
    for start in range(0, candidate_cells.shape[0], chunk_size):
        chunk_cells = candidate_cells[start : start + chunk_size]
        chunk_pixels = candidate_pixels[start : start + chunk_size]
        chunk_neighbors = _padded_cell_neighbors(
            chunk_cells,
            neighbor_indptr=neighbor_indptr,
            neighbor_indices=neighbor_indices,
            cell_count=cell_count,
            max_neighbors=max_neighbors,
        )
        vertices, counts = clip_cells(
            pixel_polygons[chunk_pixels],
            chunk_cells,
            chunk_neighbors,
            sites,
            radii,
        )
        chunk_areas = polygon_disk_area(
            vertices,
            counts,
            sites[chunk_cells],
            radii[chunk_cells],
        )
        area_tolerance = (
            64.0
            * torch.finfo(chunk_areas.dtype).eps
            * (
                torch.abs(signed_pixel_areas[chunk_pixels])
                + math.pi * radii[chunk_cells] ** 2
            )
        )
        positions = torch.arange(vertices.shape[1], device=vertices.device)
        valid_vertices = positions[None, :] < counts[:, None]
        squared_distance = torch.sum(
            (vertices - sites[chunk_cells, None, :]) ** 2,
            dim=2,
        )
        containment_tolerance = (
            64.0
            * torch.finfo(chunk_areas.dtype).eps
            * (
                radii[chunk_cells] * radii[chunk_cells]
                + torch.max(
                    torch.where(
                        valid_vertices,
                        squared_distance,
                        torch.zeros_like(squared_distance),
                    ),
                    dim=1,
                ).values
                + 1.0
            )
        )
        fully_contained = (counts >= 3) & torch.all(
            (~valid_vertices)
            | (
                squared_distance
                <= radii[chunk_cells, None] ** 2 + containment_tolerance[:, None]
            ),
            dim=1,
        )
        pixel_tolerance = (
            64.0
            * torch.finfo(chunk_areas.dtype).eps
            * torch.abs(signed_pixel_areas[chunk_pixels])
        )
        keep = (chunk_areas > area_tolerance) | (
            fully_contained & (chunk_areas > pixel_tolerance)
        )
        kept_pixels.append(chunk_pixels[keep])
        kept_cells.append(chunk_cells[keep])
        kept_areas.append(chunk_areas[keep])
    pixel_ids = (
        torch.cat(kept_pixels) if kept_pixels else candidate_pixels.new_empty((0,))
    )
    indices = torch.cat(kept_cells) if kept_cells else candidate_cells.new_empty((0,))
    areas = torch.cat(kept_areas) if kept_areas else sites.new_empty((0,))
    row_counts = torch.bincount(pixel_ids, minlength=pixel_count)
    indptr = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=sites.device),
            torch.cumsum(row_counts, dim=0),
        )
    )
    order = torch.argsort(pixel_ids, stable=True)
    order = order[torch.argsort(indices[order], stable=True)]
    transpose_indices = pixel_ids[order]
    transpose_areas = areas[order]
    transpose_counts = torch.bincount(indices, minlength=cell_count)
    transpose_indptr = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=sites.device),
            torch.cumsum(transpose_counts, dim=0),
        )
    )
    return PowerPixelOperator(
        indptr=indptr,
        indices=indices,
        pixel_ids=pixel_ids,
        areas=areas,
        transpose_indptr=transpose_indptr,
        transpose_indices=transpose_indices,
        transpose_areas=transpose_areas,
        cell_areas=cell_areas,
        pixel_count=pixel_count,
        cell_count=cell_count,
        candidate_pairs=int(candidate_indices.numel()),
    )


def sparse_bounded_power_pixel_operator(
    pixel_polygons: Tensor,
    sites: Tensor,
    radii: Tensor,
    *,
    bin_size: float | None = None,
    max_candidate_pairs: int = 10_000_000,
    max_bin_memberships: int = 50_000_000,
    max_site_bin_memberships: int = 10_000_000,
    max_packed_vertices: int = 64,
    candidate_chunk_size: int = 4096,
    kernel_execution: Literal["eager", "compile"] = "eager",
) -> PowerPixelOperator:
    """Build the production sparse exact operator for arbitrary sky pixels."""
    candidates = polygon_disk_candidate_csr(
        pixel_polygons,
        sites,
        radii,
        bin_size=bin_size,
        max_bin_memberships=max_bin_memberships,
        max_candidate_pairs=max_candidate_pairs,
    )
    neighbors = sparse_cech_graph(
        sites,
        radii,
        bin_size=bin_size,
        max_site_bin_memberships=max_site_bin_memberships,
        max_candidate_pairs=max_candidate_pairs,
    )
    return batched_bounded_power_pixel_operator(
        pixel_polygons,
        sites,
        radii,
        candidate_indptr=candidates.indptr,
        candidate_indices=candidates.indices,
        neighbor_indptr=neighbors.indptr,
        neighbor_indices=neighbors.indices,
        max_candidate_pairs=max_candidate_pairs,
        max_packed_vertices=max_packed_vertices,
        candidate_chunk_size=candidate_chunk_size,
        kernel_execution=kernel_execution,
    )


__all__ = [
    "PolygonDiskCSR",
    "PowerPixelOperator",
    "PowerValueSemantics",
    "SpectralPowerFoam",
    "SparseCechGraph",
    "TileDiskCSR",
    "batched_bounded_power_pixel_operator",
    "bounded_power_pixel_operator",
    "bounded_power_ownership",
    "cech_adjacency",
    "power_distance",
    "power_face_planes",
    "polygon_disk_candidate_csr",
    "render_power_spectral_basis",
    "render_power_spectral_samples",
    "smooth_bounded_power_weights",
    "soft_bounded_power_weights",
    "sparse_bounded_power_pixel_operator",
    "sparse_cech_graph",
    "tile_disk_candidate_csr",
]
