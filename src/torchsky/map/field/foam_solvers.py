"""Bounded power-diagram primitives for experimental sky-field models.

These tensor-native kernels implement the geometry underlying Power Foam.  They
are representation primitives, not a complete renderer: exact rendering also
needs sparse candidate construction and pixel/cell intersection integration.
"""

from __future__ import annotations

from .foam_common import PowerValueSemantics

from dataclasses import dataclass
import math
import time
from typing import Callable, Literal, Sequence

import torch
from torch import Tensor


def spectral_basis_whitening_transform(
    response_basis: Tensor,
    *,
    rcond: float | None = None,
) -> Tensor:
    """Return a right transform with orthonormal band-response columns.

    ``response_basis`` has shape ``(bands, basis)`` and contains each
    passband-integrated spectral basis vector.  The returned square transform
    preserves the represented continuous spectral subspace while making
    ``response_basis @ transform`` orthonormal.  This is useful both as an
    optimizer preconditioner and for full-rank, per-band appearance controls.
    """
    if (
        not response_basis.is_floating_point()
        or response_basis.ndim != 2
        or response_basis.shape[0] < response_basis.shape[1]
        or response_basis.shape[1] == 0
    ):
        raise ValueError(
            "response_basis must have floating shape (bands>=basis, basis>=1)"
        )
    if not bool(torch.isfinite(response_basis).all()):
        raise ValueError("response_basis must be finite")
    if rcond is None:
        relative_cutoff = torch.finfo(response_basis.dtype).eps * max(
            response_basis.shape
        )
    else:
        relative_cutoff = float(rcond)
        if not math.isfinite(relative_cutoff) or relative_cutoff < 0.0:
            raise ValueError("rcond must be finite and non-negative")

    _, singular_values, vh = torch.linalg.svd(response_basis, full_matrices=False)
    cutoff = relative_cutoff * singular_values[0]
    if bool(torch.any(singular_values <= cutoff)):
        rank = int(torch.sum(singular_values > cutoff))
        raise ValueError(
            "response_basis is numerically rank deficient "
            f"(rank {rank} < {response_basis.shape[1]})"
        )
    return vh.mT / singular_values.unsqueeze(0)


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
    centered_first_moments: Tensor
    transpose_indptr: Tensor
    transpose_indices: Tensor
    transpose_areas: Tensor
    cell_areas: Tensor
    cell_centered_first_moments: Tensor
    pixel_count: int
    cell_count: int
    candidate_pairs: int

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> PowerPixelOperator:
        """Move the sparse operator tensors without changing its geometry."""
        target = torch.device(device)
        if self.areas.device == target:
            return self
        return PowerPixelOperator(
            indptr=self.indptr.to(device=target, non_blocking=non_blocking),
            indices=self.indices.to(device=target, non_blocking=non_blocking),
            pixel_ids=self.pixel_ids.to(device=target, non_blocking=non_blocking),
            areas=self.areas.to(device=target, non_blocking=non_blocking),
            centered_first_moments=self.centered_first_moments.to(
                device=target, non_blocking=non_blocking
            ),
            transpose_indptr=self.transpose_indptr.to(
                device=target, non_blocking=non_blocking
            ),
            transpose_indices=self.transpose_indices.to(
                device=target, non_blocking=non_blocking
            ),
            transpose_areas=self.transpose_areas.to(
                device=target, non_blocking=non_blocking
            ),
            cell_areas=self.cell_areas.to(device=target, non_blocking=non_blocking),
            cell_centered_first_moments=self.cell_centered_first_moments.to(
                device=target, non_blocking=non_blocking
            ),
            pixel_count=self.pixel_count,
            cell_count=self.cell_count,
            candidate_pairs=self.candidate_pairs,
        )

    @property
    def storage_bytes(self) -> int:
        """Number of bytes held by the operator's tensor payload."""
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.indptr,
                self.indices,
                self.pixel_ids,
                self.areas,
                self.centered_first_moments,
                self.transpose_indptr,
                self.transpose_indices,
                self.transpose_areas,
                self.cell_areas,
                self.cell_centered_first_moments,
            )
        )

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

    def matvec_affine(
        self,
        cell_values: Tensor,
        cell_gradients: Tensor,
    ) -> Tensor:
        """Integrate cell-local affine surface brightness over every pixel.

        ``cell_gradients[..., cell, :]`` is the gradient with respect to the
        same two coordinates used to build the power diagram, centered on each
        cell site. The stored exact first moments make this a linear operator,
        not a point-sampled approximation.
        """
        if (
            not cell_values.is_floating_point()
            or cell_values.shape[-1:] != (self.cell_count,)
            or cell_values.device != self.areas.device
            or cell_values.dtype != self.areas.dtype
        ):
            raise ValueError(
                "cell_values must match the operator with shape (..., cells)"
            )
        if (
            not cell_gradients.is_floating_point()
            or cell_gradients.shape != cell_values.shape + (2,)
            or cell_gradients.device != self.areas.device
            or cell_gradients.dtype != self.areas.dtype
        ):
            raise ValueError(
                "cell_gradients must match cell_values with shape (..., cells, 2)"
            )
        flat_values = cell_values.reshape(-1, self.cell_count)
        flat_gradients = cell_gradients.reshape(-1, self.cell_count, 2)
        contributions = flat_values[:, self.indices] * self.areas
        contributions = contributions + torch.sum(
            flat_gradients[:, self.indices] * self.centered_first_moments,
            dim=-1,
        )
        output = flat_values.new_zeros((flat_values.shape[0], self.pixel_count))
        output.scatter_add_(
            1,
            self.pixel_ids.expand(flat_values.shape[0], -1),
            contributions,
        )
        return output.reshape(cell_values.shape[:-1] + (self.pixel_count,))

    def rmatvec_affine(self, pixel_values: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the exact transpose of :meth:`matvec_affine`."""
        if (
            not pixel_values.is_floating_point()
            or pixel_values.shape[-1:] != (self.pixel_count,)
            or pixel_values.device != self.areas.device
            or pixel_values.dtype != self.areas.dtype
        ):
            raise ValueError(
                "pixel_values must match the operator with shape (..., pixels)"
            )
        flat = pixel_values.reshape(-1, self.pixel_count)
        edge_values = flat[:, self.pixel_ids]
        value_output = flat.new_zeros((flat.shape[0], self.cell_count))
        value_output.scatter_add_(
            1,
            self.indices.expand(flat.shape[0], -1),
            edge_values * self.areas,
        )
        gradient_output = flat.new_zeros((flat.shape[0], self.cell_count, 2))
        gradient_output.scatter_add_(
            1,
            self.indices[None, :, None].expand(flat.shape[0], -1, 2),
            edge_values[..., None] * self.centered_first_moments,
        )
        leading = pixel_values.shape[:-1]
        return (
            value_output.reshape(leading + (self.cell_count,)),
            gradient_output.reshape(leading + (self.cell_count, 2)),
        )


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


def render_power_spectral_affine_basis(
    operator: PowerPixelOperator,
    cell_coefficients: Tensor,
    cell_gradient_coefficients: Tensor,
    basis_samples: Tensor,
    response_weights: Tensor,
) -> Tensor:
    """Render a spectral basis with an exact cell-local affine spatial basis."""
    if (
        not cell_coefficients.is_floating_point()
        or cell_coefficients.ndim != 2
        or cell_coefficients.shape[0] != operator.cell_count
        or cell_coefficients.device != operator.areas.device
        or cell_coefficients.dtype != operator.areas.dtype
    ):
        raise ValueError(
            "cell_coefficients must match the operator with shape (cells, basis)"
        )
    if (
        not cell_gradient_coefficients.is_floating_point()
        or cell_gradient_coefficients.shape != cell_coefficients.shape + (2,)
        or cell_gradient_coefficients.device != cell_coefficients.device
        or cell_gradient_coefficients.dtype != cell_coefficients.dtype
    ):
        raise ValueError(
            "cell_gradient_coefficients must match coefficients with trailing 2"
        )
    if (
        not basis_samples.is_floating_point()
        or basis_samples.ndim != 2
        or basis_samples.shape[1] != cell_coefficients.shape[-1]
        or basis_samples.device != cell_coefficients.device
        or basis_samples.dtype != cell_coefficients.dtype
    ):
        raise ValueError("basis_samples must have shape (samples, basis)")
    if (
        not response_weights.is_floating_point()
        or response_weights.ndim not in (1, 2)
        or response_weights.device != cell_coefficients.device
        or response_weights.dtype != cell_coefficients.dtype
        or not bool(torch.isfinite(cell_coefficients).all())
        or not bool(torch.isfinite(cell_gradient_coefficients).all())
        or not bool(torch.isfinite(basis_samples).all())
        or not bool(torch.isfinite(response_weights).all())
    ):
        raise ValueError("affine spectral inputs must be finite and share dtype/device")
    responses = response_weights.reshape(-1, response_weights.shape[-1]).to(
        device=cell_coefficients.device,
        dtype=cell_coefficients.dtype,
    )
    if responses.shape[1] != basis_samples.shape[0]:
        raise ValueError("response_weights and basis_samples must share sample count")
    response_basis = responses @ basis_samples
    cell_values = torch.einsum("bk,ck->bc", response_basis, cell_coefficients)
    cell_gradients = torch.einsum(
        "bk,ckd->bcd",
        response_basis,
        cell_gradient_coefficients,
    )
    rendered = operator.matvec_affine(cell_values, cell_gradients)
    return rendered.squeeze(0) if response_weights.ndim == 1 else rendered


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
    if (
        cell_values.device != basis_values.device
        or cell_values.dtype != basis_values.dtype
    ):
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


def _batched_jacobi_pcg(
    apply_normal: Callable[[Tensor], Tensor],
    solution: Tensor,
    residual: Tensor,
    preconditioner: Tensor,
    *,
    iterations: int,
    threshold: float,
    reference_norm: Tensor,
) -> Tensor:
    """Multi-RHS Jacobi PCG. ``reference_norm`` is the per-row stopping scale."""
    preconditioned = residual / preconditioner
    direction = preconditioned.clone()
    residual_dot = torch.sum(residual * preconditioned, dim=1)
    active = torch.linalg.vector_norm(residual, dim=1) > threshold * reference_norm
    tiny = torch.finfo(solution.dtype).tiny
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
    return solution


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
        return (
            operator.rmatvec(
                weighted,
                semantics=semantics,
            )
            + regularization[:, None] * cell_values
        )

    weighted_values = flat_weights * flat_values
    if pixel_rmatvec is not None:
        weighted_values = pixel_rmatvec(weighted_values)
    rhs = operator.rmatvec(weighted_values, semantics=semantics)
    residual = rhs - apply_normal(solution)
    rhs_norm = torch.linalg.vector_norm(rhs, dim=1)
    residual_norm = torch.linalg.vector_norm(residual, dim=1)
    reference_norm = torch.where(rhs_norm > 0, rhs_norm, residual_norm)
    # ponytail: Jacobi PCG ignores optional pixel mixing; upgrade the shared
    # loop if a broad PSF makes convergence profiling justify a block preconditioner.
    solution = _batched_jacobi_pcg(
        apply_normal,
        solution,
        residual,
        preconditioner,
        iterations=iterations,
        threshold=threshold,
        reference_norm=reference_norm,
    )
    return solution.reshape(output_shape)


def solve_power_pixel_affine_least_squares(
    operator: PowerPixelOperator,
    pixel_values: Tensor,
    pixel_weights: Tensor | None = None,
    *,
    ridge: float = 1.0e-6,
    max_iterations: int = 50,
    tolerance: float = 1.0e-6,
    pixel_matvec: Callable[[Tensor], Tensor] | None = None,
    pixel_rmatvec: Callable[[Tensor], Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """Solve cell values and local x/y gradients with the exact affine map."""
    if (
        not pixel_values.is_floating_point()
        or pixel_values.shape[-1:] != (operator.pixel_count,)
        or pixel_values.device != operator.areas.device
        or pixel_values.dtype != operator.areas.dtype
        or not bool(torch.isfinite(pixel_values).all())
    ):
        raise ValueError(
            "pixel_values must be finite and match the operator with shape (..., pixels)"
        )
    if pixel_weights is None:
        weights = torch.ones_like(pixel_values)
    elif (
        not pixel_weights.is_floating_point()
        or pixel_weights.shape != pixel_values.shape
        or pixel_weights.device != pixel_values.device
        or pixel_weights.dtype != pixel_values.dtype
        or not bool(torch.isfinite(pixel_weights).all())
        or bool(torch.any(pixel_weights < 0))
    ):
        raise ValueError("pixel_weights must match pixel_values and be non-negative")
    else:
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

    flat_values = pixel_values.reshape(-1, operator.pixel_count)
    flat_weights = weights.reshape_as(flat_values)
    batch = flat_values.shape[0]
    cells = operator.cell_count
    positive_cell_areas = operator.cell_areas[operator.cell_areas > 0]
    fallback_cell_area = (
        torch.median(positive_cell_areas)
        if positive_cell_areas.numel()
        else operator.cell_areas.new_tensor(1.0)
    )
    cell_scales = torch.sqrt(
        torch.where(operator.cell_areas > 0, operator.cell_areas, fallback_cell_area)
    )
    edge_weight = flat_weights[:, operator.pixel_ids]
    value_diagonal = flat_values.new_zeros((batch, cells))
    value_diagonal.scatter_add_(
        1,
        operator.indices.expand(batch, -1),
        edge_weight * operator.areas.square(),
    )
    gradient_diagonal = flat_values.new_zeros((batch, cells, 2))
    gradient_diagonal.scatter_add_(
        1,
        operator.indices[None, :, None].expand(batch, -1, 2),
        edge_weight[..., None]
        * torch.square(
            operator.centered_first_moments / cell_scales[operator.indices, None]
        ),
    )
    diagonal = torch.cat((value_diagonal, gradient_diagonal.flatten(1)), dim=1)
    positive_count = torch.sum(diagonal > 0, dim=1).clamp(min=1)
    scale = torch.sum(diagonal, dim=1) / positive_count
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    regularization = penalty * scale
    preconditioner = diagonal + regularization[:, None]

    def unpack(parameters: Tensor) -> tuple[Tensor, Tensor]:
        scaled_gradients = parameters[:, cells:].reshape(batch, cells, 2)
        return parameters[:, :cells], scaled_gradients / cell_scales[None, :, None]

    def apply_normal(parameters: Tensor) -> Tensor:
        values, gradients = unpack(parameters)
        projected = operator.matvec_affine(values, gradients)
        if pixel_matvec is not None:
            projected = pixel_matvec(projected)
        weighted = flat_weights * projected
        if pixel_rmatvec is not None:
            weighted = pixel_rmatvec(weighted)
        value_adjoint, gradient_adjoint = operator.rmatvec_affine(weighted)
        packed = torch.cat(
            (
                value_adjoint,
                (gradient_adjoint / cell_scales[None, :, None]).flatten(1),
            ),
            dim=1,
        )
        return packed + regularization[:, None] * parameters

    weighted_values = flat_weights * flat_values
    if pixel_rmatvec is not None:
        weighted_values = pixel_rmatvec(weighted_values)
    rhs_values, rhs_gradients = operator.rmatvec_affine(weighted_values)
    rhs = torch.cat(
        (
            rhs_values,
            (rhs_gradients / cell_scales[None, :, None]).flatten(1),
        ),
        dim=1,
    )
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    rhs_norm = torch.linalg.vector_norm(rhs, dim=1)
    # ponytail: the block-Jacobi diagonal ignores value/slope cross-terms; use
    # a small per-cell block preconditioner only if real convergence requires it.
    solution = _batched_jacobi_pcg(
        apply_normal,
        solution,
        residual,
        preconditioner,
        iterations=iterations,
        threshold=threshold,
        reference_norm=rhs_norm,
    )
    solved_values, solved_gradients = unpack(solution)
    leading = pixel_values.shape[:-1]
    return (
        solved_values.reshape(leading + (cells,)),
        solved_gradients.reshape(leading + (cells, 2)),
    )


def solve_spectral_power_pixel_affine_least_squares(
    operators: Sequence[PowerPixelOperator],
    response_basis: Tensor,
    pixel_values: Sequence[Tensor],
    pixel_weights: Sequence[Tensor] | None = None,
    *,
    ridge: float = 1.0e-6,
    max_iterations: int = 50,
    tolerance: float = 1.0e-6,
    pixel_matvecs: Sequence[Callable[[Tensor], Tensor] | None] | None = None,
    pixel_rmatvecs: Sequence[Callable[[Tensor], Tensor] | None] | None = None,
    initial_coefficients: Tensor | None = None,
    initial_gradient_coefficients: Tensor | None = None,
    solve_device: torch.device | str | None = None,
    observation_batch_size: int | Literal["auto"] | None = None,
    preconditioner: Literal["diagonal", "cell_block", "auto"] = "diagonal",
    preconditioner_max_bytes: int | None = None,
    diagnostics: dict[str, float] | None = None,
) -> tuple[Tensor, Tensor]:
    """Jointly solve a spectral affine Foam through heterogeneous pixel grids.

    Each row of ``response_basis`` is the passband-integrated basis for one
    observation. Pixel values use the integrated units returned by the matching
    operator. Optional forward/adjoint pairs compose a linear pixel-domain PSF.
    Geometry is shared, while WCS footprints, masks, weights, and PSFs may vary.
    Operators and pixels may be stored off-device; ``observation_batch_size``
    bounds how many are resident on ``solve_device`` at once without changing
    the accumulated normal equations. ``cell_block`` preconditioning captures
    within-cell spectral and affine-slope coupling at O(cells * basis^2)
    storage; the default diagonal remains appropriate for very large grids.
    """
    observation_count = len(operators)
    if observation_count == 0:
        raise ValueError("operators must not be empty")
    if (
        not response_basis.is_floating_point()
        or response_basis.ndim != 2
        or response_basis.shape[0] != observation_count
        or response_basis.shape[1] == 0
        or not bool(torch.isfinite(response_basis).all())
    ):
        raise ValueError(
            "response_basis must be finite with shape (observations, basis>=1)"
        )
    if len(pixel_values) != observation_count:
        raise ValueError("pixel_values must match operators")
    weights = (
        [torch.ones_like(value) for value in pixel_values]
        if pixel_weights is None
        else list(pixel_weights)
    )
    if len(weights) != observation_count:
        raise ValueError("pixel_weights must match operators")
    forwards = (
        [None] * observation_count if pixel_matvecs is None else list(pixel_matvecs)
    )
    adjoints = (
        [None] * observation_count if pixel_rmatvecs is None else list(pixel_rmatvecs)
    )
    if len(forwards) != observation_count or len(adjoints) != observation_count:
        raise ValueError("pixel transforms must match operators")
    if any(
        (forward is None) != (adjoint is None)
        for forward, adjoint in zip(forwards, adjoints, strict=True)
    ):
        raise ValueError("pixel forward and adjoint must be supplied in pairs")

    first = operators[0]
    cells = first.cell_count
    basis_count = int(response_basis.shape[1])
    device = first.areas.device if solve_device is None else torch.device(solve_device)
    if response_basis.dtype != first.areas.dtype:
        raise ValueError("response_basis must match operator dtype")
    basis_on_device = response_basis.to(device=device)
    observation_transfer_bytes = [
        operator.storage_bytes
        + values.numel() * values.element_size()
        + weight.numel() * weight.element_size()
        for operator, values, weight in zip(
            operators, pixel_values, weights, strict=True
        )
    ]
    batch_auto = observation_batch_size == "auto"
    transfer_budget_bytes = sum(observation_transfer_bytes)
    if batch_auto and device.type == "cuda":
        try:
            free_device_bytes, _ = torch.cuda.mem_get_info(device)
            # ponytail: reserve 80% of currently free CUDA memory for the model,
            # workspaces, and allocator fragmentation. Upgrade to measured peak
            # telemetry if heterogeneous real exposures defeat this bound.
            transfer_budget_bytes = min(2 * 1024**3, free_device_bytes // 5)
        except RuntimeError:
            # torch.cuda.mem_get_info routes through NVML and asserts on some
            # H100 MIG slices; fall back to the conservative 2 GiB cap.
            transfer_budget_bytes = min(transfer_budget_bytes, 2 * 1024**3)
    max_observation_bytes = max(observation_transfer_bytes)
    batch_size = (
        max(
            1,
            min(
                observation_count,
                transfer_budget_bytes // max(max_observation_bytes, 1),
            ),
        )
        if batch_auto
        else (
            observation_count
            if observation_batch_size is None
            else int(observation_batch_size)
        )
    )
    if batch_size <= 0:
        raise ValueError("observation_batch_size must be positive")
    for operator, values, weight in zip(operators, pixel_values, weights, strict=True):
        if operator.cell_count != cells or operator.areas.dtype != first.areas.dtype:
            raise ValueError("operators must share cell count and dtype")
        if (
            not values.is_floating_point()
            or values.shape != (operator.pixel_count,)
            or values.dtype != first.areas.dtype
            or not bool(torch.isfinite(values).all())
        ):
            raise ValueError("pixel values must be finite and match each operator")
        if (
            not weight.is_floating_point()
            or weight.shape != values.shape
            or weight.dtype != values.dtype
            or not bool(torch.isfinite(weight).all())
            or bool(torch.any(weight < 0))
        ):
            raise ValueError(
                "pixel weights must be finite, non-negative, and match values"
            )
    penalty = float(ridge)
    threshold = float(tolerance)
    iterations = int(max_iterations)
    if not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("ridge must be finite and positive")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("tolerance must be finite and positive")
    if iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if preconditioner not in ("diagonal", "cell_block", "auto"):
        raise ValueError("preconditioner must be 'diagonal', 'cell_block', or 'auto'")
    if preconditioner_max_bytes is not None and preconditioner_max_bytes <= 0:
        raise ValueError("preconditioner_max_bytes must be positive")

    def diagnostic_time() -> float:
        if diagnostics is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    setup_started = diagnostic_time()

    union_cell_areas = torch.stack(
        [operator.cell_areas.to(device=device) for operator in operators]
    ).amax(dim=0)
    positive_areas = union_cell_areas[union_cell_areas > 0]
    fallback_area = (
        torch.median(positive_areas)
        if positive_areas.numel()
        else union_cell_areas.new_tensor(1.0)
    )
    cell_scales = torch.sqrt(
        torch.where(union_cell_areas > 0, union_cell_areas, fallback_area)
    )
    value_diagonal = torch.zeros(
        (cells, basis_count), dtype=first.areas.dtype, device=device
    )
    gradient_diagonal = torch.zeros(
        (cells, basis_count, 2), dtype=first.areas.dtype, device=device
    )
    block_storage_bytes = cells * (3 * basis_count) ** 2 * first.areas.element_size()
    response_condition = torch.linalg.cond(basis_on_device)
    # ponytail: cap automatic block construction at 512 MiB and 10% of free
    # CUDA memory; replace this heuristic with allocator telemetry if real
    # fragmented-device runs show that free bytes overestimate usable memory.
    block_budget_bytes = 512 * 1024**2
    try:
        if device.type == "cuda":
            free_device_bytes, _ = torch.cuda.mem_get_info(device)
            block_budget_bytes = min(block_budget_bytes, free_device_bytes // 10)
        else:
            block_budget_bytes = min(block_budget_bytes, 256 * 1024**2)
    except RuntimeError:
        pass  # keep the 512 MiB cap when NVML is unavailable (MIG slices)
    if preconditioner_max_bytes is not None:
        block_budget_bytes = min(block_budget_bytes, int(preconditioner_max_bytes))
    use_cell_block = preconditioner == "cell_block" or (
        preconditioner == "auto"
        and 2 * block_storage_bytes <= block_budget_bytes
        and bool(response_condition >= 2.0)
    )
    block_normal = (
        torch.zeros(
            (cells, 3 * basis_count, 3 * basis_count),
            dtype=first.areas.dtype,
            device=device,
        )
        if use_cell_block
        else None
    )

    def observation_batches():
        for start in range(0, observation_count, batch_size):
            stop = min(start + batch_size, observation_count)
            yield [
                (
                    operators[index].to(device),
                    pixel_values[index].to(device=device),
                    weights[index].to(device=device),
                    basis_on_device[index],
                    forwards[index],
                    adjoints[index],
                )
                for index in range(start, stop)
            ]

    for batch in observation_batches():
        for operator, _values, weight, basis, _forward, _adjoint in batch:
            edge_weight = weight[operator.pixel_ids]
            value_base = weight.new_zeros(cells)
            value_base.scatter_add_(
                0, operator.indices, edge_weight * operator.areas.square()
            )
            gradient_base = weight.new_zeros((cells, 2))
            gradient_base.scatter_add_(
                0,
                operator.indices[:, None].expand(-1, 2),
                edge_weight[:, None]
                * torch.square(
                    operator.centered_first_moments
                    / cell_scales[operator.indices, None]
                ),
            )
            basis_square = basis.square()
            value_diagonal += value_base[:, None] * basis_square[None, :]
            gradient_diagonal += gradient_base[:, None, :] * basis_square[None, :, None]
            if block_normal is not None:
                spatial_features = torch.cat(
                    (
                        operator.areas[:, None],
                        operator.centered_first_moments
                        / cell_scales[operator.indices, None],
                    ),
                    dim=1,
                )
                spatial_normal = weight.new_zeros((cells, 3, 3))
                for row in range(3):
                    for column in range(row, 3):
                        entry = weight.new_zeros(cells)
                        entry.scatter_add_(
                            0,
                            operator.indices,
                            edge_weight
                            * spatial_features[:, row]
                            * spatial_features[:, column],
                        )
                        spatial_normal[:, row, column] = entry
                        spatial_normal[:, column, row] = entry
                block_normal += torch.einsum(
                    "k,l,cij->ckilj", basis, basis, spatial_normal
                ).reshape(cells, 3 * basis_count, 3 * basis_count)
    diagonal = torch.cat((value_diagonal.flatten(), gradient_diagonal.flatten()))
    positive_count = torch.sum(diagonal > 0).clamp(min=1)
    scale = torch.sum(diagonal) / positive_count
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    regularization = penalty * scale
    diagonal_preconditioner = diagonal + regularization
    block_cholesky: Tensor | None = None
    block_fallback_cells = 0
    if block_normal is not None:
        block_normal.diagonal(dim1=-2, dim2=-1).add_(regularization)
        block_cholesky, cholesky_info = torch.linalg.cholesky_ex(block_normal)
        failed = cholesky_info != 0
        block_fallback_cells = int(torch.sum(failed))
        if block_fallback_cells:
            local_diagonal = torch.cat(
                (value_diagonal[:, :, None], gradient_diagonal), dim=2
            ).reshape(cells, 3 * basis_count)
            block_normal[failed] = torch.diag_embed(
                (local_diagonal + regularization)[failed]
            )
            block_cholesky = torch.linalg.cholesky(block_normal)
    setup_seconds = diagnostic_time() - setup_started

    coefficient_size = cells * basis_count

    def pack_parameters(coefficients: Tensor, gradients: Tensor) -> Tensor:
        return torch.cat(
            (
                coefficients.flatten(),
                (gradients * cell_scales[:, None, None]).flatten(),
            )
        )

    def unpack(parameters: Tensor) -> tuple[Tensor, Tensor]:
        coefficients = parameters[:coefficient_size].reshape(cells, basis_count)
        scaled_gradients = parameters[coefficient_size:].reshape(cells, basis_count, 2)
        return coefficients, scaled_gradients / cell_scales[:, None, None]

    def apply_preconditioner(residual: Tensor) -> Tensor:
        if block_cholesky is None:
            return residual / diagonal_preconditioner
        coefficient_residual = residual[:coefficient_size].reshape(cells, basis_count)
        gradient_residual = residual[coefficient_size:].reshape(cells, basis_count, 2)
        local_residual = torch.cat(
            (coefficient_residual[:, :, None], gradient_residual), dim=2
        ).reshape(cells, 3 * basis_count)
        local_solution = torch.cholesky_solve(
            local_residual[:, :, None], block_cholesky
        )[:, :, 0].reshape(cells, basis_count, 3)
        return torch.cat(
            (
                local_solution[:, :, 0].flatten(),
                local_solution[:, :, 1:].flatten(),
            )
        )

    def pack_adjoint(coefficients: Tensor, gradients: Tensor) -> Tensor:
        return torch.cat(
            (
                coefficients.flatten(),
                (gradients / cell_scales[:, None, None]).flatten(),
            )
        )

    def adjoint_observation(
        operator: PowerPixelOperator,
        values: Tensor,
        basis: Tensor,
    ) -> Tensor:
        value_adjoint, gradient_adjoint = operator.rmatvec_affine(values)
        return pack_adjoint(
            value_adjoint[:, None] * basis[None, :],
            gradient_adjoint[:, None, :] * basis[None, :, None],
        )

    normal_operator_applications = 0

    def apply_normal(parameters: Tensor) -> Tensor:
        nonlocal normal_operator_applications
        normal_operator_applications += 1
        coefficients, gradients = unpack(parameters)
        output = torch.zeros_like(parameters)
        for batch in observation_batches():
            for operator, _values, weight, basis, forward, adjoint in batch:
                cell_values = coefficients @ basis
                cell_gradients = torch.einsum("ckd,k->cd", gradients, basis)
                projected = operator.matvec_affine(cell_values, cell_gradients)
                if forward is not None:
                    projected = forward(projected)
                weighted = weight * projected
                if adjoint is not None:
                    weighted = adjoint(weighted)
                output += adjoint_observation(operator, weighted, basis)
        return output + regularization * parameters

    rhs_started = diagnostic_time()
    rhs = torch.zeros(diagonal.shape, dtype=first.areas.dtype, device=device)
    for batch in observation_batches():
        for operator, values, weight, basis, _forward, adjoint in batch:
            weighted = weight * values
            if adjoint is not None:
                weighted = adjoint(weighted)
            rhs += adjoint_observation(operator, weighted, basis)
    if (initial_coefficients is None) != (initial_gradient_coefficients is None):
        raise ValueError("initial coefficients and gradients must be supplied together")
    if initial_coefficients is None:
        solution = torch.zeros_like(rhs)
        ridge_center = torch.zeros_like(rhs)
    else:
        assert initial_gradient_coefficients is not None
        if (
            not initial_coefficients.is_floating_point()
            or not initial_gradient_coefficients.is_floating_point()
            or initial_coefficients.shape != (cells, basis_count)
            or initial_gradient_coefficients.shape != (cells, basis_count, 2)
            or initial_coefficients.dtype != rhs.dtype
            or initial_gradient_coefficients.dtype != rhs.dtype
            or not bool(torch.isfinite(initial_coefficients).all())
            or not bool(torch.isfinite(initial_gradient_coefficients).all())
        ):
            raise ValueError("initial coefficient shapes must match cells and basis")
        solution = pack_parameters(
            initial_coefficients.to(device=device),
            initial_gradient_coefficients.to(device=device),
        )
        ridge_center = solution
        rhs = rhs + regularization * ridge_center
    rhs_seconds = diagnostic_time() - rhs_started
    initial_residual_started = diagnostic_time()
    residual = rhs - apply_normal(solution)
    preconditioned = apply_preconditioner(residual)
    direction = preconditioned.clone()
    residual_dot = torch.sum(residual * preconditioned)
    rhs_norm = torch.linalg.vector_norm(rhs)
    initial_relative_residual = torch.linalg.vector_norm(residual) / torch.clamp(
        rhs_norm, min=torch.finfo(rhs.dtype).tiny
    )
    initial_residual_seconds = diagnostic_time() - initial_residual_started
    tiny = torch.finfo(rhs.dtype).tiny
    completed_iterations = 0
    pcg_started = diagnostic_time()
    for _ in range(iterations):
        residual_norm = torch.linalg.vector_norm(residual)
        if bool(residual_norm <= threshold * rhs_norm):
            break
        normal_direction = apply_normal(direction)
        denominator = torch.dot(direction, normal_direction)
        if bool(torch.abs(denominator) <= tiny):
            break
        step = residual_dot / denominator
        solution = solution + step * direction
        residual = residual - step * normal_direction
        completed_iterations += 1
        preconditioned = apply_preconditioner(residual)
        next_residual_dot = torch.sum(residual * preconditioned)
        if bool(torch.abs(residual_dot) <= tiny):
            break
        direction = preconditioned + (next_residual_dot / residual_dot) * direction
        residual_dot = next_residual_dot
    pcg_seconds = diagnostic_time() - pcg_started
    final_residual_norm = torch.linalg.vector_norm(residual)
    relative_residual = final_residual_norm / torch.clamp(
        rhs_norm, min=torch.finfo(rhs.dtype).tiny
    )
    if diagnostics is not None:
        diagnostics.update(
            {
                "iterations": float(completed_iterations),
                "normal_operator_applications": float(normal_operator_applications),
                "initial_relative_residual": float(initial_relative_residual),
                "ridge_centered_on_initial": float(initial_coefficients is not None),
                "relative_residual": float(relative_residual),
                "converged": float(relative_residual <= threshold),
                "preconditioner_setup_seconds": setup_seconds,
                "rhs_seconds": rhs_seconds,
                "initial_residual_seconds": initial_residual_seconds,
                "pcg_seconds": pcg_seconds,
                "preconditioner_cell_block": float(block_cholesky is not None),
                "preconditioner_block_fallback_cells": float(block_fallback_cells),
                "preconditioner_auto": float(preconditioner == "auto"),
                "preconditioner_response_condition": float(response_condition),
                "preconditioner_auto_condition_threshold": 2.0,
                "preconditioner_auto_budget_bytes": float(block_budget_bytes),
                "preconditioner_block_construction_peak_bytes": float(
                    2 * block_storage_bytes
                ),
                "preconditioner_storage_bytes": float(
                    diagonal_preconditioner.numel()
                    * diagonal_preconditioner.element_size()
                    if block_cholesky is None
                    else block_cholesky.numel() * block_cholesky.element_size()
                ),
                "observation_batch_size": float(batch_size),
                "observation_batch_auto": float(batch_auto),
                "observation_batch_transfer_budget_bytes": float(transfer_budget_bytes),
                "max_observation_transfer_bytes": float(max_observation_bytes),
                "operator_storage_bytes": float(
                    sum(operator.storage_bytes for operator in operators)
                ),
                "peak_operator_batch_bytes": float(
                    max(
                        sum(
                            operator.storage_bytes
                            for operator in operators[start : start + batch_size]
                        )
                        for start in range(0, observation_count, batch_size)
                    )
                ),
            }
        )
    return unpack(solution)
