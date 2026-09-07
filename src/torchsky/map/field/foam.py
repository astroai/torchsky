"""Bounded power-diagram primitives for experimental sky-field models.

These tensor-native kernels implement the geometry underlying Power Foam.  They
are representation primitives, not a complete renderer: exact rendering also
needs sparse candidate construction and pixel/cell intersection integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import time
from typing import Any, Callable, Literal, Sequence

import torch
from torch import Tensor

PowerValueSemantics = Literal["surface_brightness", "integrated_flux"]


@dataclass(frozen=True)
class PersistentFoamSiteInitialization:
    """Deterministic sites selected from coherent multi-exposure evidence."""

    sites: Tensor
    radii: Tensor
    scores: Tensor
    exposure_counts: Tensor
    bin_ids: Tensor


@dataclass(frozen=True)
class FoamAdaptationResult:
    """Cell-identity mapping from one deterministic Foam adaptation step."""

    densified_cell_ids: tuple[int, ...]
    pruned_cell_ids: tuple[int, ...]
    old_to_new: tuple[int, ...]


def persistent_foam_site_initialization(
    exposure_coordinates: Sequence[Tensor],
    exposure_whitened_residuals: Sequence[Tensor],
    *,
    num_sites: int,
    min_exposures: int,
    bin_shape: tuple[int, int],
    lower: tuple[float, float],
    upper: tuple[float, float],
    initial_radius: float,
    exposure_contributions: Sequence[Tensor] | None = None,
) -> PersistentFoamSiteInitialization:
    """Select sites from positive residual evidence coherent across exposures.

    Each exposure contributes one weighted mean whitened residual per spatial
    bin, so dense exposures cannot dominate sparse ones. Candidate score is the
    number of covered exposures times the squared coherent positive mean.
    ``min_exposures`` is an explicit persistence gate; ties resolve by bin id.
    """
    coordinates = tuple(exposure_coordinates)
    residuals = tuple(exposure_whitened_residuals)
    if not coordinates or len(coordinates) != len(residuals):
        raise ValueError("coordinates and residuals require matching exposures")
    if exposure_contributions is None:
        contributions: tuple[Tensor | None, ...] = (None,) * len(coordinates)
    else:
        contributions = tuple(exposure_contributions)
        if len(contributions) != len(coordinates):
            raise ValueError("exposure_contributions must match exposures")
    requested_sites = int(num_sites)
    required_exposures = int(min_exposures)
    bins_y, bins_x = map(int, bin_shape)
    if requested_sites < 1:
        raise ValueError("num_sites must be positive")
    if required_exposures < 1 or required_exposures > len(coordinates):
        raise ValueError("min_exposures must be within the exposure count")
    if bins_y < 1 or bins_x < 1:
        raise ValueError("bin_shape dimensions must be positive")
    radius = float(initial_radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("initial_radius must be finite and positive")
    first = coordinates[0]
    if not first.is_floating_point():
        raise ValueError("exposure coordinates must be floating tensors")
    device, dtype = first.device, first.dtype
    lower_t = torch.tensor(lower, device=device, dtype=dtype)
    upper_t = torch.tensor(upper, device=device, dtype=dtype)
    if (
        lower_t.shape != (2,)
        or upper_t.shape != (2,)
        or not bool(torch.isfinite(lower_t).all())
        or not bool(torch.isfinite(upper_t).all())
        or bool(torch.any(upper_t <= lower_t))
    ):
        raise ValueError("lower and upper must define a finite 2-D box")

    bin_count = bins_y * bins_x
    signed_sum = torch.zeros(bin_count, device=device, dtype=dtype)
    covered_count = torch.zeros(bin_count, device=device, dtype=torch.int64)
    positive_count = torch.zeros_like(covered_count)
    for exposure_index, (coords, values, weights) in enumerate(
        zip(coordinates, residuals, contributions, strict=True)
    ):
        flat_coords = coords.reshape(-1, 2)
        flat_values = values.reshape(-1).to(device=device, dtype=dtype)
        if (
            coords.device != device
            or coords.dtype != dtype
            or coords.ndim < 2
            or coords.shape[-1] != 2
            or flat_values.numel() != flat_coords.shape[0]
            or not bool(torch.isfinite(flat_coords).all())
            or not bool(torch.isfinite(flat_values).all())
        ):
            raise ValueError(
                f"exposure {exposure_index} coordinates/residuals are incompatible"
            )
        if bool(torch.any(flat_coords < lower_t)) or bool(
            torch.any(flat_coords > upper_t)
        ):
            raise ValueError("exposure coordinates must lie within lower/upper")
        if weights is None:
            flat_weights = torch.ones_like(flat_values)
        else:
            flat_weights = weights.reshape(-1).to(device=device, dtype=dtype)
            if (
                flat_weights.numel() != flat_values.numel()
                or not bool(torch.isfinite(flat_weights).all())
                or bool(torch.any(flat_weights < 0.0))
            ):
                raise ValueError(
                    f"exposure {exposure_index} contributions must be finite and non-negative"
                )
        unit = (flat_coords - lower_t) / (upper_t - lower_t)
        bin_xy = torch.floor(
            unit * torch.tensor([bins_x, bins_y], device=device, dtype=dtype)
        ).to(dtype=torch.int64)
        bin_xy[:, 0].clamp_(0, bins_x - 1)
        bin_xy[:, 1].clamp_(0, bins_y - 1)
        bin_ids = bin_xy[:, 1] * bins_x + bin_xy[:, 0]
        weighted_sum = torch.zeros_like(signed_sum).index_add(
            0,
            bin_ids,
            flat_values * flat_weights,
        )
        weight_sum = torch.zeros_like(signed_sum).index_add(
            0,
            bin_ids,
            flat_weights,
        )
        covered = weight_sum > 0.0
        exposure_mean = torch.where(
            covered,
            weighted_sum / torch.clamp(weight_sum, min=torch.finfo(dtype).tiny),
            torch.zeros_like(weighted_sum),
        )
        signed_sum.add_(exposure_mean)
        covered_count.add_(covered.to(dtype=torch.int64))
        positive_count.add_((covered & (exposure_mean > 0.0)).to(dtype=torch.int64))

    coherent_mean = signed_sum / torch.clamp(covered_count, min=1).to(dtype=dtype)
    scores = (
        covered_count.to(dtype=dtype) * torch.clamp_min(coherent_mean, 0.0).square()
    )
    valid_ids = torch.nonzero(
        (positive_count >= required_exposures) & (scores > 0.0),
        as_tuple=False,
    ).flatten()
    if valid_ids.numel() < requested_sites:
        raise ValueError(
            "insufficient persistent positive-evidence bins for requested num_sites"
        )
    order = torch.argsort(scores[valid_ids], descending=True, stable=True)
    selected = valid_ids[order[:requested_sites]]
    selected_y = torch.div(selected, bins_x, rounding_mode="floor")
    selected_x = torch.remainder(selected, bins_x)
    step = (upper_t - lower_t) / torch.tensor(
        [bins_x, bins_y], device=device, dtype=dtype
    )
    sites = (
        lower_t
        + (torch.stack([selected_x, selected_y], dim=-1).to(dtype=dtype) + 0.5) * step
    )
    return PersistentFoamSiteInitialization(
        sites=sites,
        radii=torch.full((requested_sites,), radius, device=device, dtype=dtype),
        scores=scores[selected],
        exposure_counts=positive_count[selected],
        bin_ids=selected,
    )


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
    preconditioned = residual / preconditioner
    direction = preconditioned.clone()
    residual_dot = torch.sum(residual * preconditioned, dim=1)
    rhs_norm = torch.linalg.vector_norm(rhs, dim=1)
    active = torch.linalg.vector_norm(residual, dim=1) > threshold * rhs_norm
    tiny = torch.finfo(pixel_values.dtype).tiny
    # ponytail: the block-Jacobi diagonal ignores value/slope cross-terms; use
    # a small per-cell block preconditioner only if real convergence requires it.
    for _ in range(iterations):
        if not bool(torch.any(active)):
            break
        normal_direction = apply_normal(direction)
        denominator = torch.sum(direction * normal_direction, dim=1)
        safe_denominator = torch.where(
            torch.abs(denominator) > tiny, denominator, torch.ones_like(denominator)
        )
        step = torch.where(active, residual_dot / safe_denominator, 0.0)
        solution = solution + step[:, None] * direction
        residual = residual - step[:, None] * normal_direction
        next_active = active & (
            torch.linalg.vector_norm(residual, dim=1) > threshold * rhs_norm
        )
        preconditioned = residual / preconditioner
        next_residual_dot = torch.sum(residual * preconditioned, dim=1)
        safe_residual_dot = torch.where(
            torch.abs(residual_dot) > tiny,
            residual_dot,
            torch.ones_like(residual_dot),
        )
        momentum = torch.where(next_active, next_residual_dot / safe_residual_dot, 0.0)
        direction = preconditioned + momentum[:, None] * direction
        residual_dot = next_residual_dot
        active = next_active
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
        cell_gradient_coefficients: Tensor | None = None,
        minimum_radius: float = 0.0,
        log_intensity_spectra: bool = False,
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
        if cell_gradient_coefficients is not None and (
            not cell_gradient_coefficients.is_floating_point()
            or cell_gradient_coefficients.shape != cell_coefficients.shape + (2,)
            or cell_gradient_coefficients.device != sites.device
            or cell_gradient_coefficients.dtype != sites.dtype
            or not bool(torch.isfinite(cell_gradient_coefficients).all())
        ):
            raise ValueError(
                "cell_gradient_coefficients must match coefficients with trailing 2"
            )
        if log_intensity_spectra and cell_gradient_coefficients is not None:
            raise ValueError(
                "log-intensity Foam does not yet support exact affine spatial spectra"
            )
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
        self.cell_gradient_coefficients = (
            torch.nn.Parameter(cell_gradient_coefficients.detach().clone())
            if cell_gradient_coefficients is not None
            else None
        )
        self.log_intensity_spectra = bool(log_intensity_spectra)
        self.register_buffer(
            "_minimum_radius",
            radii.new_tensor(radius_floor),
        )
        self.register_buffer(
            "heldout_residual_ema",
            torch.zeros_like(radii),
        )
        self.register_buffer(
            "contribution_ema",
            torch.zeros_like(radii),
        )
        self.register_buffer(
            "residual_position_ema",
            sites.detach().clone(),
        )
        self.register_buffer(
            "evidence_updates",
            torch.zeros(sites.shape[0], device=sites.device, dtype=torch.int64),
        )

    @property
    def radii(self) -> Tensor:
        """Strictly positive trainable radii in the geometry coordinate unit."""
        return self._minimum_radius + torch.nn.functional.softplus(self.raw_radii)

    @property
    def basis_count(self) -> int:
        return int(self.cell_coefficients.shape[1])

    @torch.no_grad()
    def update_adaptation_statistics(
        self,
        cell_indices: Tensor,
        whitened_residuals: Tensor,
        contributions: Tensor,
        positions: Tensor,
        *,
        split: str,
        decay: float = 0.9,
    ) -> None:
        """Update per-cell held-out residual, contribution, and position EMAs."""
        if split != "held_out":
            raise ValueError("Foam adaptation statistics require split='held_out'")
        rate = float(decay)
        if not math.isfinite(rate) or rate < 0.0 or rate >= 1.0:
            raise ValueError("decay must be finite in [0, 1)")
        indices = cell_indices.reshape(-1).to(
            device=self.sites.device,
            dtype=torch.int64,
        )
        residual = whitened_residuals.reshape(-1).to(
            device=self.sites.device,
            dtype=self.sites.dtype,
        )
        contribution = contributions.reshape(-1).to(
            device=self.sites.device,
            dtype=self.sites.dtype,
        )
        coords = positions.reshape(-1, 2).to(
            device=self.sites.device,
            dtype=self.sites.dtype,
        )
        sample_count = indices.numel()
        if (
            sample_count == 0
            or residual.numel() != sample_count
            or contribution.numel() != sample_count
            or coords.shape[0] != sample_count
        ):
            raise ValueError(
                "adaptation statistic samples must be non-empty and aligned"
            )
        if (
            bool(torch.any(indices < 0))
            or bool(torch.any(indices >= self.sites.shape[0]))
            or not bool(torch.isfinite(residual).all())
            or not bool(torch.isfinite(contribution).all())
            or bool(torch.any(contribution < 0.0))
            or not bool(torch.isfinite(coords).all())
        ):
            raise ValueError("adaptation statistic values are invalid")

        cell_count = int(self.sites.shape[0])
        counts = torch.zeros(
            cell_count,
            device=self.sites.device,
            dtype=self.sites.dtype,
        ).index_add_(0, indices, torch.ones_like(residual))
        residual_energy = residual.square()
        residual_sum = torch.zeros_like(counts).index_add_(0, indices, residual_energy)
        contribution_sum = torch.zeros_like(counts).index_add_(0, indices, contribution)
        weighted_position = torch.zeros_like(self.residual_position_ema).index_add_(
            0,
            indices,
            coords * residual_energy.unsqueeze(-1),
        )
        ordinary_position = torch.zeros_like(self.residual_position_ema).index_add_(
            0,
            indices,
            coords,
        )
        active = counts > 0.0
        batch_residual = residual_sum / torch.clamp(
            counts, min=torch.finfo(counts.dtype).tiny
        )
        batch_contribution = contribution_sum / torch.clamp(
            counts, min=torch.finfo(counts.dtype).tiny
        )
        batch_position = torch.where(
            (residual_sum > 0.0).unsqueeze(-1),
            weighted_position
            / torch.clamp(
                residual_sum.unsqueeze(-1), min=torch.finfo(counts.dtype).tiny
            ),
            ordinary_position
            / torch.clamp(counts.unsqueeze(-1), min=torch.finfo(counts.dtype).tiny),
        )
        first = self.evidence_updates == 0
        blend = active & ~first
        initialize = active & first
        self.heldout_residual_ema[initialize] = batch_residual[initialize]
        self.contribution_ema[initialize] = batch_contribution[initialize]
        self.residual_position_ema[initialize] = batch_position[initialize]
        self.heldout_residual_ema[blend] = (
            rate * self.heldout_residual_ema[blend]
            + (1.0 - rate) * batch_residual[blend]
        )
        self.contribution_ema[blend] = (
            rate * self.contribution_ema[blend]
            + (1.0 - rate) * batch_contribution[blend]
        )
        self.residual_position_ema[blend] = (
            rate * self.residual_position_ema[blend]
            + (1.0 - rate) * batch_position[blend]
        )
        self.evidence_updates[active] += 1

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
                    torch.sum((self.sites - closest) ** 2, dim=1) <= self.radii**2,
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
        if self.cell_gradient_coefficients is not None:
            gradient_coefficients = (
                self.cell_gradient_coefficients
                if selected is None
                else self.cell_gradient_coefficients[selected]
            )
            gradients = torch.einsum(
                "...k,ckd->...cd",
                basis_values,
                gradient_coefficients,
            )
            selected_sites = self.sites if selected is None else self.sites[selected]
            cell_values = cell_values + torch.sum(
                gradients * (points[..., None, :] - selected_sites),
                dim=-1,
            )
        if self.log_intensity_spectra:
            cell_values = torch.exp(cell_values)
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
        if self.cell_gradient_coefficients is not None:
            if semantics != "surface_brightness":
                raise ValueError(
                    "affine Foam exact rendering requires surface_brightness semantics"
                )
            return render_power_spectral_affine_basis(
                operator,
                self.cell_coefficients,
                self.cell_gradient_coefficients,
                basis_samples,
                response_weights,
            )
        if self.log_intensity_spectra:
            cell_samples = torch.exp(
                torch.einsum(
                    "ck,sk->sc",
                    self.cell_coefficients,
                    basis_samples,
                )
            )
            return render_power_spectral_samples(
                operator,
                cell_samples,
                response_weights,
                semantics=semantics,
            )
        return render_power_spectral_basis(
            operator,
            self.cell_coefficients,
            basis_samples,
            response_weights,
            semantics=semantics,
        )


def _rank_cell_ids(scores: Tensor, eligible: Tensor, *, descending: bool) -> Tensor:
    cell_ids = torch.nonzero(eligible, as_tuple=False).flatten()
    return cell_ids[torch.argsort(scores[cell_ids], descending=descending, stable=True)]


def _replace_optimizer_parameter(
    optimizer: torch.optim.Optimizer,
    old: torch.nn.Parameter,
    new: torch.nn.Parameter,
    *,
    keep_ids: Tensor,
    child_count: int,
) -> None:
    found = False
    for group in optimizer.param_groups:
        parameters = group["params"]
        for index, parameter in enumerate(parameters):
            if parameter is old:
                parameters[index] = new
                found = True
    old_state = optimizer.state.pop(old, None)
    if old_state is None:
        return
    if not found:
        raise ValueError("optimizer state contains a parameter outside its groups")
    new_state: dict[Any, Any] = {}
    for name, value in old_state.items():
        if isinstance(value, Tensor) and value.shape == old.shape:
            retained = value.index_select(0, keep_ids.to(device=value.device))
            children = torch.zeros(
                (child_count, *value.shape[1:]),
                device=value.device,
                dtype=value.dtype,
            )
            new_state[name] = torch.cat([retained, children], dim=0)
        elif isinstance(value, Tensor):
            new_state[name] = value.clone()
        else:
            new_state[name] = value
    optimizer.state[new] = new_state


def _replace_cell_parameter(
    model: SpectralPowerFoam,
    optimizer: torch.optim.Optimizer,
    name: str,
    values: Tensor,
    *,
    keep_ids: Tensor,
    child_count: int,
) -> None:
    old = getattr(model, name)
    if not isinstance(old, torch.nn.Parameter):
        raise TypeError(f"Foam {name} is not a parameter")
    new = torch.nn.Parameter(values, requires_grad=old.requires_grad)
    setattr(model, name, new)
    _replace_optimizer_parameter(
        optimizer,
        old,
        new,
        keep_ids=keep_ids,
        child_count=child_count,
    )


def _inverse_radius_parameter(radii: Tensor, minimum_radius: Tensor) -> Tensor:
    shifted = radii - minimum_radius
    if bool(torch.any(shifted <= 0.0)):
        raise ValueError("adapted radii must remain above minimum_radius")
    return shifted + torch.log(-torch.expm1(-shifted))


@torch.no_grad()
def adapt_spectral_power_foam(
    model: SpectralPowerFoam,
    optimizer: torch.optim.Optimizer,
    *,
    densify_count: int,
    prune_count: int,
    min_updates: int = 1,
) -> FoamAdaptationResult:
    """Deterministically split high-error cells and prune low contributors.

    Densification proposals are the held-out residual-position EMAs. A split
    copies appearance, assigns equal-area radii to parent and child, preserves
    optimizer state for retained cells, and starts child optimizer moments at
    zero. Pruning and densification counts are explicit—there are no hidden
    score thresholds.
    """
    add_count = int(densify_count)
    remove_count = int(prune_count)
    required_updates = int(min_updates)
    cell_count = int(model.sites.shape[0])
    if add_count < 0 or remove_count < 0:
        raise ValueError("densify_count and prune_count must be non-negative")
    if required_updates < 1:
        raise ValueError("min_updates must be positive")
    if remove_count >= cell_count:
        raise ValueError("pruning must leave at least one Foam cell")
    if add_count == 0 and remove_count == 0:
        return FoamAdaptationResult((), (), tuple(range(cell_count)))

    updated = model.evidence_updates >= required_updates
    split_radii = model.radii.detach() / math.sqrt(2.0)
    movable = torch.any(model.residual_position_ema != model.sites.detach(), dim=-1) & (
        split_radii > model._minimum_radius
    )
    densify_ranked = _rank_cell_ids(
        model.heldout_residual_ema,
        updated & movable,
        descending=True,
    )
    if densify_ranked.numel() < add_count:
        raise ValueError("insufficient held-out evidence for requested densification")
    densified = densify_ranked[:add_count]
    densify_mask = torch.zeros(cell_count, dtype=torch.bool, device=model.sites.device)
    densify_mask[densified] = True

    prune_ranked = _rank_cell_ids(
        model.contribution_ema,
        updated & ~densify_mask,
        descending=False,
    )
    if prune_ranked.numel() < remove_count:
        raise ValueError("insufficient held-out evidence for requested pruning")
    pruned = prune_ranked[:remove_count]
    keep_mask = torch.ones(cell_count, dtype=torch.bool, device=model.sites.device)
    keep_mask[pruned] = False
    keep_ids = torch.nonzero(keep_mask, as_tuple=False).flatten()
    keep_list = keep_ids.tolist()
    kept_position = {old_id: new_id for new_id, old_id in enumerate(keep_list)}

    old_sites = model.sites.detach()
    old_radii = model.radii.detach()
    old_coefficients = model.cell_coefficients.detach()
    new_sites = torch.cat(
        [old_sites[keep_ids], model.residual_position_ema[densified]], dim=0
    )
    new_radii = torch.cat([old_radii[keep_ids], split_radii[densified]], dim=0)
    for cell_id in densified.tolist():
        new_radii[kept_position[cell_id]] = split_radii[cell_id]
    new_coefficients = torch.cat(
        [old_coefficients[keep_ids], old_coefficients[densified]], dim=0
    )
    new_raw_radii = _inverse_radius_parameter(new_radii, model._minimum_radius)

    _replace_cell_parameter(
        model,
        optimizer,
        "sites",
        new_sites,
        keep_ids=keep_ids,
        child_count=add_count,
    )
    _replace_cell_parameter(
        model,
        optimizer,
        "raw_radii",
        new_raw_radii,
        keep_ids=keep_ids,
        child_count=add_count,
    )
    _replace_cell_parameter(
        model,
        optimizer,
        "cell_coefficients",
        new_coefficients,
        keep_ids=keep_ids,
        child_count=add_count,
    )
    if model.cell_gradient_coefficients is not None:
        old_gradients = model.cell_gradient_coefficients.detach()
        new_gradients = torch.cat(
            [old_gradients[keep_ids], old_gradients[densified]], dim=0
        )
        _replace_cell_parameter(
            model,
            optimizer,
            "cell_gradient_coefficients",
            new_gradients,
            keep_ids=keep_ids,
            child_count=add_count,
        )

    residual_stats = torch.cat(
        [model.heldout_residual_ema[keep_ids], model.heldout_residual_ema[densified]]
    )
    contribution_stats = torch.cat(
        [model.contribution_ema[keep_ids], model.contribution_ema[densified]], dim=0
    )
    for child_index, cell_id in enumerate(densified.tolist()):
        parent_index = kept_position[cell_id]
        half = 0.5 * model.contribution_ema[cell_id]
        contribution_stats[parent_index] = half
        contribution_stats[len(keep_ids) + child_index] = half
    position_stats = torch.cat(
        [
            model.residual_position_ema[keep_ids],
            model.residual_position_ema[densified],
        ],
        dim=0,
    )
    update_stats = torch.cat(
        [model.evidence_updates[keep_ids], model.evidence_updates[densified]], dim=0
    )
    model.heldout_residual_ema = residual_stats
    model.contribution_ema = contribution_stats
    model.residual_position_ema = position_stats
    model.evidence_updates = update_stats

    old_to_new = [-1] * cell_count
    for new_id, old_id in enumerate(keep_list):
        old_to_new[old_id] = new_id
    return FoamAdaptationResult(
        densified_cell_ids=tuple(int(value) for value in densified.tolist()),
        pruned_cell_ids=tuple(int(value) for value in pruned.tolist()),
        old_to_new=tuple(old_to_new),
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
    omit a disk that intersects a polygon. ``max_candidate_pairs`` bounds both
    each temporary join and the retained result, not duplicate hits across bins.
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
        polygon_groups = list(
            zip(
                unique_polygon_keys.tolist(),
                polygon_starts.tolist(),
                polygon_counts.tolist(),
                strict=True,
            )
        )
        raw_count = sum(
            int(count) * site_groups[int(key)][1]
            for key, _, count in polygon_groups
            if int(key) in site_groups
        )
        stream_join = raw_count > candidate_limit
        retained_count = 0
        if stream_join:
            polygon_first_bin = torch.floor((polygon_lower - origin) / width).long()
            site_first_bin = torch.floor((site_lower - origin) / width).long()
        for key, start, count in polygon_groups:
            site_group = site_groups.get(int(key))
            if site_group is None:
                continue
            site_start, site_count_in_bin = site_group
            group_count = int(count) * site_count_in_bin
            if stream_join:
                for offset in range(0, group_count, candidate_limit):
                    local = torch.arange(
                        offset,
                        min(offset + candidate_limit, group_count),
                        device=sites.device,
                    )
                    pixels = polygon_ids[
                        int(start)
                        + torch.div(local, site_count_in_bin, rounding_mode="floor")
                    ]
                    disks = site_ids[
                        site_start + torch.remainder(local, site_count_in_bin)
                    ]
                    # Each AABB pair belongs to its lexically first shared bin.
                    # This removes duplicate bin hits before retaining any pairs.
                    first = (
                        torch.maximum(polygon_first_bin[pixels], site_first_bin[disks])
                        - min_bins
                    )
                    owned = first[:, 1] * columns + first[:, 0] == int(key)
                    pixels, disks = pixels[owned], disks[owned]
                    closest = torch.minimum(
                        torch.maximum(index_sites[disks], polygon_lower[pixels]),
                        polygon_upper[pixels],
                    )
                    intersects = (index_sites[disks] - closest).square().sum(
                        dim=1
                    ) <= index_radii[disks].square()
                    pixels, disks = pixels[intersects], disks[intersects]
                    retained_count += pixels.numel()
                    if retained_count > candidate_limit:
                        raise ValueError(
                            f"polygon/disk result needs {retained_count} candidate pairs, "
                            f"exceeding max_candidate_pairs={candidate_limit}"
                        )
                    if pixels.numel():
                        pair_pixels.append(pixels)
                        pair_sites.append(disks)
                continue
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


def _segment_circle_area_moment(
    start: Tensor,
    stop: Tensor,
    radius: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return signed disk-intersection area and first moment for one edge."""
    delta = stop - start
    quadratic = torch.dot(delta, delta)
    if bool(quadratic == 0):
        return start.new_zeros(()), start.new_zeros((2,))
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
    moment = start.new_zeros((2,))
    for lower, upper in zip(parameters[:-1], parameters[1:], strict=True):
        first = start + lower * delta
        second = start + upper * delta
        midpoint = 0.5 * (first + second)
        cross = first[0] * second[1] - first[1] * second[0]
        if bool(torch.dot(midpoint, midpoint) <= radius * radius + tolerance):
            triangle_area = 0.5 * cross
            area = area + triangle_area
            moment = moment + triangle_area * (first + second) / 3.0
        else:
            area = area + 0.5 * radius * radius * torch.atan2(
                cross,
                torch.dot(first, second),
            )
            first_unit = first / torch.linalg.vector_norm(first)
            second_unit = second / torch.linalg.vector_norm(second)
            moment = moment + (radius**3 / 3.0) * torch.stack(
                (
                    second_unit[1] - first_unit[1],
                    first_unit[0] - second_unit[0],
                )
            )
    return area, moment


def _segment_circle_area(start: Tensor, stop: Tensor, radius: Tensor) -> Tensor:
    """Return the signed disk intersection area swept by one polygon edge."""
    return _segment_circle_area_moment(start, stop, radius)[0]


def _polygon_disk_intersection_area_moment(
    vertices: Tensor,
    center: Tensor,
    radius: Tensor,
) -> tuple[Tensor, Tensor]:
    """Analytic area and global first moment of polygon intersected with a disk."""
    if vertices.shape[0] < 3 or bool(radius <= 0):
        return vertices.new_zeros(()), vertices.new_zeros((2,))
    shifted = vertices - center
    squared_distance = torch.sum(shifted * shifted, dim=1)
    containment_tolerance = (
        64.0
        * torch.finfo(vertices.dtype).eps
        * (radius * radius + torch.max(squared_distance) + 1.0)
    )
    area = vertices.new_zeros(())
    local_moment = vertices.new_zeros((2,))
    if bool(torch.all(squared_distance <= radius * radius + containment_tolerance)):
        following = torch.roll(shifted, shifts=-1, dims=0)
        triangle_areas = 0.5 * (
            shifted[:, 0] * following[:, 1] - shifted[:, 1] * following[:, 0]
        )
        area = torch.sum(triangle_areas)
        local_moment = torch.sum(
            triangle_areas[:, None] * (shifted + following) / 3.0,
            dim=0,
        )
    else:
        for index in range(shifted.shape[0]):
            edge_area, edge_moment = _segment_circle_area_moment(
                shifted[index],
                shifted[(index + 1) % shifted.shape[0]],
                radius,
            )
            area = area + edge_area
            local_moment = local_moment + edge_moment
    orientation = torch.where(area < 0, -torch.ones_like(area), torch.ones_like(area))
    positive_area = torch.abs(area)
    return positive_area, orientation * local_moment + center * positive_area


def _polygon_disk_intersection_area(
    vertices: Tensor,
    center: Tensor,
    radius: Tensor,
) -> Tensor:
    """Analytic area of a polygon intersected with a closed disk."""
    return _polygon_disk_intersection_area_moment(vertices, center, radius)[0]


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
    cell_centered_first_moments = sites.new_zeros((cell_count, 2))
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
        cell_area, cell_first_moment = _polygon_disk_intersection_area_moment(
            cell_polygon,
            sites[cell],
            radius,
        )
        cell_areas[cell] = cell_area
        cell_centered_first_moments[cell] = cell_first_moment - sites[cell] * cell_area

    kept_pixels: list[int] = []
    kept_cells: list[int] = []
    kept_areas: list[Tensor] = []
    kept_centered_first_moments: list[Tensor] = []
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
            area, first_moment = _polygon_disk_intersection_area_moment(
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
                    * (radii[cell] * radii[cell] + torch.max(squared_distance) + 1.0)
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
                area > area_tolerance or (fully_contained and area > pixel_tolerance)
            ):
                kept_pixels.append(pixel)
                kept_cells.append(cell)
                kept_areas.append(area)
                kept_centered_first_moments.append(first_moment - sites[cell] * area)
        row_counts.append(len(kept_areas) - before)

    indices = torch.tensor(kept_cells, dtype=torch.long, device=sites.device)
    pixel_ids = torch.tensor(kept_pixels, dtype=torch.long, device=sites.device)
    areas = torch.stack(kept_areas) if kept_areas else sites.new_empty((0,))
    centered_first_moments = (
        torch.stack(kept_centered_first_moments)
        if kept_centered_first_moments
        else sites.new_empty((0, 2))
    )
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
        centered_first_moments=centered_first_moments,
        transpose_indptr=transpose_indptr,
        transpose_indices=transpose_indices,
        transpose_areas=transpose_areas,
        cell_areas=cell_areas,
        cell_centered_first_moments=cell_centered_first_moments,
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


def _batched_polygon_disk_area_moment(
    vertices: Tensor,
    counts: Tensor,
    centers: Tensor,
    radii: Tensor,
) -> tuple[Tensor, Tensor]:
    """Analytic polygon/disk areas and global first moments for a batch."""
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
    triangle_moment = triangle[..., None] * (first + second) / 3.0
    sector = (
        0.5
        * radii[:, None, None] ** 2
        * torch.atan2(cross, torch.sum(first * second, dim=3))
    )
    first_norm = torch.linalg.vector_norm(first, dim=3).clamp_min(
        torch.finfo(vertices.dtype).tiny
    )
    second_norm = torch.linalg.vector_norm(second, dim=3).clamp_min(
        torch.finfo(vertices.dtype).tiny
    )
    first_unit = first / first_norm[..., None]
    second_unit = second / second_norm[..., None]
    sector_moment = (radii[:, None, None, None] ** 3 / 3.0) * torch.stack(
        (
            second_unit[..., 1] - first_unit[..., 1],
            first_unit[..., 0] - second_unit[..., 0],
        ),
        dim=3,
    )
    inside = torch.sum(midpoint * midpoint, dim=3) <= (
        radii[:, None, None] ** 2 + tolerance[:, :, None]
    )
    intervals = torch.where(inside, triangle, sector)
    interval_moments = torch.where(
        inside[..., None],
        triangle_moment,
        sector_moment,
    )
    intervals = torch.where(
        upper > lower,
        intervals,
        torch.zeros_like(intervals),
    )
    interval_moments = torch.where(
        (upper > lower)[..., None],
        interval_moments,
        torch.zeros_like(interval_moments),
    )
    valid_edges = (positions < counts[:, None]) & (counts[:, None] >= 3)
    signed_intersection_area = torch.sum(intervals * valid_edges[..., None], dim=(1, 2))
    intersection_orientation = torch.where(
        signed_intersection_area < 0,
        -torch.ones_like(signed_intersection_area),
        torch.ones_like(signed_intersection_area),
    )
    intersection_area = torch.abs(signed_intersection_area)
    intersection_local_moment = intersection_orientation[:, None] * torch.sum(
        interval_moments * valid_edges[..., None, None],
        dim=(1, 2),
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
    polygon_signed_area = torch.sum(triangle[..., 0] * valid_edges, dim=1)
    polygon_orientation = torch.where(
        polygon_signed_area < 0,
        -torch.ones_like(polygon_signed_area),
        torch.ones_like(polygon_signed_area),
    )
    polygon_area = torch.abs(polygon_signed_area)
    polygon_local_moment = polygon_orientation[:, None] * torch.sum(
        triangle_moment[..., 0, :] * valid_edges[..., None],
        dim=1,
    )
    area = torch.where(fully_contained, polygon_area, intersection_area)
    local_moment = torch.where(
        fully_contained[:, None],
        polygon_local_moment,
        intersection_local_moment,
    )
    return area, local_moment + centers * area[:, None]


@lru_cache(maxsize=1)
def _compiled_batched_power_kernels() -> tuple[Any, Any]:
    """Create compiled numeric kernels lazily; validation and assembly stay eager."""
    return (
        torch.compile(_batched_clip_power_cells, fullgraph=True),
        torch.compile(_batched_polygon_disk_area_moment, fullgraph=True),
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
        polygon_disk_area_moment = _batched_polygon_disk_area_moment
    elif kernel_execution == "compile":
        clip_cells, polygon_disk_area_moment = _compiled_batched_power_kernels()
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
    cell_centered_first_moments = sites.new_zeros((cell_count, 2))
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
        chunk_areas, chunk_first_moments = polygon_disk_area_moment(
            cell_vertices,
            cell_counts,
            sites[chunk_cells],
            chunk_radii,
        )
        cell_areas[chunk_cells] = chunk_areas
        cell_centered_first_moments[chunk_cells] = (
            chunk_first_moments - sites[chunk_cells] * chunk_areas[:, None]
        )

    kept_pixels: list[Tensor] = []
    kept_cells: list[Tensor] = []
    kept_areas: list[Tensor] = []
    kept_centered_first_moments: list[Tensor] = []
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
        chunk_areas, chunk_first_moments = polygon_disk_area_moment(
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
        kept_centered_first_moments.append(
            (chunk_first_moments - sites[chunk_cells] * chunk_areas[:, None])[keep]
        )
    pixel_ids = (
        torch.cat(kept_pixels) if kept_pixels else candidate_pixels.new_empty((0,))
    )
    indices = torch.cat(kept_cells) if kept_cells else candidate_cells.new_empty((0,))
    areas = torch.cat(kept_areas) if kept_areas else sites.new_empty((0,))
    centered_first_moments = (
        torch.cat(kept_centered_first_moments)
        if kept_centered_first_moments
        else sites.new_empty((0, 2))
    )
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
        centered_first_moments=centered_first_moments,
        transpose_indptr=transpose_indptr,
        transpose_indices=transpose_indices,
        transpose_areas=transpose_areas,
        cell_areas=cell_areas,
        cell_centered_first_moments=cell_centered_first_moments,
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
    "FoamAdaptationResult",
    "PolygonDiskCSR",
    "PersistentFoamSiteInitialization",
    "PowerPixelOperator",
    "PowerValueSemantics",
    "SpectralPowerFoam",
    "SparseCechGraph",
    "TileDiskCSR",
    "batched_bounded_power_pixel_operator",
    "adapt_spectral_power_foam",
    "bounded_power_pixel_operator",
    "bounded_power_ownership",
    "cech_adjacency",
    "power_distance",
    "power_face_planes",
    "polygon_disk_candidate_csr",
    "persistent_foam_site_initialization",
    "render_power_spectral_affine_basis",
    "render_power_spectral_basis",
    "render_power_spectral_samples",
    "solve_power_pixel_affine_least_squares",
    "smooth_bounded_power_weights",
    "soft_bounded_power_weights",
    "sparse_bounded_power_pixel_operator",
    "sparse_cech_graph",
    "tile_disk_candidate_csr",
]
