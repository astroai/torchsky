"""Torch-native sampled detector astrometry with explicit calibration gauges.

The fixed detector table maps ``(x, y)`` detector coordinates to a two-
dimensional intermediate frame such as JWST V2/V3.  Exposure pose is owned by
``TangentPlaneAffine``; detector-shared residuals are constrained to the
orthogonal complement of constant and affine control-grid modes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

InterpolationMode = Literal["bilinear", "bicubic"]
AngularCoordinates = Literal["sky_degrees", "plane_degrees", "plane_arcsec"]


def _finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all().detach().cpu().item()):
        raise ValueError(f"{name} must contain only finite values")


def _as_pair(
    value: Any,
    name: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    pair = torch.as_tensor(value, dtype=dtype, device=device)
    if pair.shape != (2,):
        raise ValueError(f"{name} must have shape (2,)")
    _finite(pair, name)
    return pair


def _lagrange4(t: Tensor) -> tuple[Tensor, Tensor]:
    """Cubic Lagrange weights and derivatives on nodes 0, 1, 2, 3."""
    w0 = -(t - 1) * (t - 2) * (t - 3) / 6
    w1 = t * (t - 2) * (t - 3) / 2
    w2 = -t * (t - 1) * (t - 3) / 2
    w3 = t * (t - 1) * (t - 2) / 6
    d0 = -(
        (t - 2) * (t - 3) + (t - 1) * (t - 3) + (t - 1) * (t - 2)
    ) / 6
    d1 = ((t - 2) * (t - 3) + t * (t - 3) + t * (t - 2)) / 2
    d2 = -((t - 1) * (t - 3) + t * (t - 3) + t * (t - 1)) / 2
    d3 = ((t - 1) * (t - 2) + t * (t - 2) + t * (t - 1)) / 6
    return torch.stack((w0, w1, w2, w3), dim=-1), torch.stack(
        (d0, d1, d2, d3), dim=-1
    )


def _interpolate_regular_grid(
    points: Tensor,
    table: Tensor,
    origin: Tensor,
    spacing: Tensor,
    mode: InterpolationMode,
) -> tuple[Tensor, Tensor]:
    """Interpolate values and return ``d(output) / d(x, y)``."""
    if points.shape[-1:] != (2,):
        raise ValueError("points last dimension must be 2")
    shape = points.shape[:-1]
    flat = points.reshape(-1, 2)
    scaled = (flat - origin) / spacing
    height, width = table.shape[:2]
    outside = (
        (scaled[:, 0] < 0)
        | (scaled[:, 0] > width - 1)
        | (scaled[:, 1] < 0)
        | (scaled[:, 1] > height - 1)
    )
    if bool(outside.any().detach().cpu().item()):
        raise ValueError("points fall outside the regular-grid domain")

    u, v = scaled.unbind(dim=-1)
    if mode == "bilinear":
        ix = torch.clamp(torch.floor(u).to(torch.long), max=width - 2)
        iy = torch.clamp(torch.floor(v).to(torch.long), max=height - 2)
        tx = u - ix
        ty = v - iy
        f00 = table[iy, ix]
        f10 = table[iy, ix + 1]
        f01 = table[iy + 1, ix]
        f11 = table[iy + 1, ix + 1]
        values = (
            (1 - tx)[:, None] * (1 - ty)[:, None] * f00
            + tx[:, None] * (1 - ty)[:, None] * f10
            + (1 - tx)[:, None] * ty[:, None] * f01
            + tx[:, None] * ty[:, None] * f11
        )
        du = (1 - ty)[:, None] * (f10 - f00) + ty[:, None] * (f11 - f01)
        dv = (1 - tx)[:, None] * (f01 - f00) + tx[:, None] * (f11 - f10)
    else:
        start_x = torch.clamp(torch.floor(u).to(torch.long) - 1, 0, width - 4)
        start_y = torch.clamp(torch.floor(v).to(torch.long) - 1, 0, height - 4)
        offsets = torch.arange(4, device=points.device)
        ix = start_x[:, None] + offsets
        iy = start_y[:, None] + offsets
        wx, dwx = _lagrange4(u - start_x)
        wy, dwy = _lagrange4(v - start_y)
        samples = table[iy[:, :, None], ix[:, None, :]]
        weights = wy[:, :, None] * wx[:, None, :]
        values = torch.sum(weights[..., None] * samples, dim=(1, 2))
        du = torch.sum(
            (wy[:, :, None] * dwx[:, None, :])[..., None] * samples,
            dim=(1, 2),
        )
        dv = torch.sum(
            (dwy[:, :, None] * wx[:, None, :])[..., None] * samples,
            dim=(1, 2),
        )

    jacobian = torch.stack((du / spacing[0], dv / spacing[1]), dim=-1)
    return values.reshape(*shape, 2), jacobian.reshape(*shape, 2, 2)


def _determinant(jacobian: Tensor) -> Tensor:
    return (
        jacobian[..., 0, 0] * jacobian[..., 1, 1]
        - jacobian[..., 0, 1] * jacobian[..., 1, 0]
    )


def _inverse_tolerances(
    *,
    target: Tensor,
    output_scale: Tensor,
    tol: float | None,
    atol: float | None,
    rtol: float,
) -> tuple[float, float]:
    if tol is not None and atol is not None:
        raise ValueError("tol and atol are aliases; specify only one")
    absolute = tol if tol is not None else atol
    if absolute is None:
        eps = torch.finfo(target.dtype).eps
        scale = float(output_scale.detach().abs().cpu().item())
        absolute = max(1.0e-8, 64.0 * eps) * scale
    absolute = float(absolute)
    relative = float(rtol)
    if not math.isfinite(absolute) or absolute < 0.0:
        raise ValueError("atol must be finite and non-negative")
    if not math.isfinite(relative) or relative < 0.0:
        raise ValueError("rtol must be finite and non-negative")
    return absolute, relative


def _inverse_residual_ok(
    residual: Tensor,
    target: Tensor,
    *,
    atol: float,
    rtol: float,
) -> bool:
    limit = atol + rtol * target.abs()
    return bool((residual.abs() <= limit).all().detach().cpu().item())


class RegularGridTransform2D(nn.Module):
    """Frozen regular-grid detector transform with consistent sampled parity.

    ``table`` has shape ``(height, width, 2)`` and samples detector coordinates
    ``origin + spacing * (x_index, y_index)``. Bicubic mode uses local
    four-point Lagrange interpolation, which reproduces affine maps exactly,
    including at the detector boundary.
    Sampled Jacobians may be consistently positive or consistently negative;
    a reflected coordinate frame is preserved without changing its values.
    The determinant floor applies to their absolute magnitude. These sampled
    diagnostics do not prove global injectivity between validation points.
    """

    table: Tensor
    origin: Tensor
    spacing: Tensor
    mode: InterpolationMode

    def __init__(
        self,
        table: Any,
        *,
        origin: Any = (0.0, 0.0),
        spacing: Any = (1.0, 1.0),
        mode: InterpolationMode = "bilinear",
        validation_subdivisions: int = 4,
        determinant_floor: float = 0.0,
    ) -> None:
        super().__init__()
        values = torch.as_tensor(table)
        if not values.is_floating_point() or values.is_complex():
            raise ValueError("table must be a real floating-point tensor")
        if values.ndim != 3 or values.shape[-1] != 2:
            raise ValueError("table must have shape (height, width, 2)")
        minimum_size = 4 if mode == "bicubic" else 2
        if mode not in {"bilinear", "bicubic"}:
            raise ValueError("mode must be 'bilinear' or 'bicubic'")
        if min(values.shape[:2]) < minimum_size:
            raise ValueError(f"{mode} tables need at least {minimum_size} samples per axis")
        if validation_subdivisions < 1:
            raise ValueError("validation_subdivisions must be positive")
        if determinant_floor < 0:
            raise ValueError("determinant_floor must be non-negative")
        _finite(values, "table")
        table_t = values.detach().clone()
        origin_t = _as_pair(
            origin, "origin", dtype=table_t.dtype, device=table_t.device
        )
        spacing_t = _as_pair(
            spacing, "spacing", dtype=table_t.dtype, device=table_t.device
        )
        if not bool((spacing_t > 0).all().detach().cpu().item()):
            raise ValueError("spacing must be strictly positive")
        self.mode = mode
        self.validation_subdivisions = int(validation_subdivisions)
        self.determinant_floor = float(determinant_floor)
        self.register_buffer("table", table_t)
        self.register_buffer("origin", origin_t)
        self.register_buffer("spacing", spacing_t)
        self.assert_invertible()

    @property
    def grid_shape(self) -> tuple[int, int]:
        return int(self.table.shape[0]), int(self.table.shape[1])

    def _coerce_points(self, points: Any) -> Tensor:
        pts = torch.as_tensor(points)
        if not pts.is_floating_point() or pts.is_complex():
            raise ValueError("points must be a real floating-point tensor")
        if pts.shape[-1:] != (2,):
            raise ValueError("points last dimension must be 2")
        _finite(pts, "points")
        return pts

    def evaluate_with_jacobian(self, points: Any) -> tuple[Tensor, Tensor]:
        pts = self._coerce_points(points)
        table = self.table.to(device=pts.device, dtype=pts.dtype)
        origin = self.origin.to(device=pts.device, dtype=pts.dtype)
        spacing = self.spacing.to(device=pts.device, dtype=pts.dtype)
        return _interpolate_regular_grid(pts, table, origin, spacing, self.mode)

    def forward(self, points: Any) -> Tensor:
        return self.evaluate_with_jacobian(points)[0]

    def jacobian(self, points: Any) -> Tensor:
        return self.evaluate_with_jacobian(points)[1]

    def jacobian_determinant(self, points: Any) -> Tensor:
        return _determinant(self.jacobian(points))

    def diagnostic_points(self, subdivisions: int | None = None) -> Tensor:
        factor = self.validation_subdivisions if subdivisions is None else subdivisions
        if factor < 1:
            raise ValueError("subdivisions must be positive")
        height, width = self.grid_shape
        x = torch.linspace(
            0,
            width - 1,
            (width - 1) * factor + 1,
            dtype=self.table.dtype,
            device=self.table.device,
        )
        y = torch.linspace(
            0,
            height - 1,
            (height - 1) * factor + 1,
            dtype=self.table.dtype,
            device=self.table.device,
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        index_points = torch.stack((xx, yy), dim=-1)
        return self.origin + index_points * self.spacing

    def assert_invertible(
        self,
        *,
        subdivisions: int | None = None,
        determinant_floor: float | None = None,
    ) -> Tensor:
        """Return sampled determinants or reject a non-finite/folded table."""
        floor = self.determinant_floor if determinant_floor is None else determinant_floor
        if floor < 0:
            raise ValueError("determinant_floor must be non-negative")
        with torch.no_grad():
            determinant = self.jacobian_determinant(
                self.diagnostic_points(subdivisions)
            )
            _finite(determinant, "table Jacobian determinant")
            positive = bool((determinant > floor).all().detach().cpu().item())
            negative = bool((determinant < -floor).all().detach().cpu().item())
            if not (positive or negative):
                minimum = float(determinant.abs().min().detach().cpu().item())
                raise ValueError(
                    "table is folded or singular: sampled Jacobian determinants "
                    "must have a consistent sign and absolute magnitude above "
                    f"{float(floor):.8g} (minimum magnitude {minimum:.8g})"
                )
        return determinant

    def inverse(
        self,
        values: Any,
        *,
        max_iter: int = 25,
        tol: float | None = None,
        atol: float | None = None,
        rtol: float = 0.0,
        fail_closed: bool = True,
    ) -> Tensor:
        """Newton inverse with output-unit ``atol`` and dimensionless ``rtol``.

        ``tol`` remains a compatibility alias for ``atol``.
        """
        tgt = self._coerce_points(values)
        origin = self.origin.to(device=tgt.device, dtype=tgt.dtype)
        spacing = self.spacing.to(device=tgt.device, dtype=tgt.dtype)
        table = self.table.to(device=tgt.device, dtype=tgt.dtype)
        height, width = self.grid_shape
        p00 = table[0, 0]
        dx = (table[0, -1] - table[0, 0]) / max(width - 1, 1)
        dy = (table[-1, 0] - table[0, 0]) / max(height - 1, 1)
        affine = torch.stack((dx, dy), dim=-1)
        delta = (tgt - p00).unsqueeze(-1)
        idx = torch.linalg.solve(affine, delta).squeeze(-1)
        x = origin + idx * spacing
        x0, y0 = origin[0], origin[1]
        x1 = origin[0] + (width - 1) * spacing[0]
        y1 = origin[1] + (height - 1) * spacing[1]
        x = torch.stack(
            (
                torch.clamp(x[..., 0], min=x0, max=x1),
                torch.clamp(x[..., 1], min=y0, max=y1),
            ),
            dim=-1,
        )
        output_extent = (table.amax(dim=(0, 1)) - table.amin(dim=(0, 1))).amax()
        atol_value, rtol_value = _inverse_tolerances(
            target=tgt,
            output_scale=output_extent,
            tol=tol,
            atol=atol,
            rtol=rtol,
        )
        for _ in range(int(max_iter)):
            pred, jac = self.evaluate_with_jacobian(x)
            resid = pred - tgt
            step = torch.linalg.solve(jac, resid.unsqueeze(-1)).squeeze(-1)
            x = x - step
            x = torch.stack(
                (
                    torch.clamp(x[..., 0], min=x0, max=x1),
                    torch.clamp(x[..., 1], min=y0, max=y1),
                ),
                dim=-1,
            )
        pred, _ = self.evaluate_with_jacobian(x)
        residual = pred - tgt
        max_resid = float(residual.abs().amax().detach().cpu().item())
        if fail_closed and not _inverse_residual_ok(
            residual, tgt, atol=atol_value, rtol=rtol_value
        ):
            raise ValueError(
                f"grid inverse residual {max_resid:.8g} exceeds "
                f"atol={atol_value:.8g}, rtol={rtol_value:.8g}"
            )
        return x


class TangentPlaneAffine(nn.Module):
    """Learnable full affine correction in a two-dimensional tangent plane."""

    matrix: nn.Parameter
    offset: nn.Parameter

    def __init__(self, matrix: Any, offset: Any) -> None:
        super().__init__()
        matrix_t = torch.as_tensor(matrix)
        if (
            not matrix_t.is_floating_point()
            or matrix_t.is_complex()
            or matrix_t.shape != (2, 2)
        ):
            raise ValueError("matrix must be a real floating-point tensor of shape (2, 2)")
        offset_t = _as_pair(
            offset,
            "offset",
            dtype=matrix_t.dtype,
            device=matrix_t.device,
        )
        _finite(matrix_t, "matrix")
        determinant = _determinant(matrix_t)
        if float(determinant.detach().cpu().item()) <= 0:
            raise ValueError("affine matrix must have positive determinant")
        self.matrix = nn.Parameter(matrix_t.detach().clone())
        self.offset = nn.Parameter(offset_t.detach().clone())

    @classmethod
    def identity(
        cls,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> TangentPlaneAffine:
        return cls(
            torch.eye(2, dtype=dtype, device=device),
            torch.zeros(2, dtype=dtype, device=device),
        )

    def forward(self, points: Any) -> Tensor:
        pts = torch.as_tensor(points)
        if not pts.is_floating_point() or pts.shape[-1:] != (2,):
            raise ValueError("points must be floating point with last dimension 2")
        matrix = self.matrix.to(device=pts.device, dtype=pts.dtype)
        offset = self.offset.to(device=pts.device, dtype=pts.dtype)
        return torch.matmul(pts, matrix.T) + offset

    def inverse(self, points: Any) -> Tensor:
        pts = torch.as_tensor(points)
        if not pts.is_floating_point() or pts.shape[-1:] != (2,):
            raise ValueError("points must be floating point with last dimension 2")
        matrix = self.matrix.to(device=pts.device, dtype=pts.dtype)
        offset = self.offset.to(device=pts.device, dtype=pts.dtype)
        return torch.matmul(pts - offset, torch.linalg.inv(matrix).T)

    def jacobian(self, points: Tensor) -> Tensor:
        matrix = self.matrix.to(device=points.device, dtype=points.dtype)
        return matrix.expand(*points.shape[:-1], 2, 2)

    def jacobian_determinant(self) -> Tensor:
        return _determinant(self.matrix)

    def assert_invertible(self, determinant_floor: float = 0.0) -> Tensor:
        if determinant_floor < 0:
            raise ValueError("determinant_floor must be non-negative")
        determinant = self.jacobian_determinant()
        _finite(determinant, "affine Jacobian determinant")
        if float(determinant.detach().cpu().item()) <= determinant_floor:
            raise ValueError("affine correction is folded or singular")
        return determinant


class AffineFreeResidualGrid2D(nn.Module):
    """Smooth detector-coordinate residual with constant/affine modes removed."""

    control: nn.Parameter
    origin: Tensor
    spacing: Tensor
    affine_basis: Tensor
    mode: InterpolationMode

    def __init__(
        self,
        control: Any,
        *,
        origin: Any = (0.0, 0.0),
        spacing: Any = (1.0, 1.0),
        mode: InterpolationMode = "bicubic",
    ) -> None:
        super().__init__()
        values = torch.as_tensor(control)
        if (
            not values.is_floating_point()
            or values.is_complex()
            or values.ndim != 3
            or values.shape[-1] != 2
        ):
            raise ValueError("control must be a real floating tensor of shape (H, W, 2)")
        minimum_size = 4 if mode == "bicubic" else 2
        if mode not in {"bilinear", "bicubic"} or min(values.shape[:2]) < minimum_size:
            raise ValueError(f"{mode} control grids need at least {minimum_size} samples per axis")
        _finite(values, "control")
        origin_t = _as_pair(
            origin, "origin", dtype=values.dtype, device=values.device
        )
        spacing_t = _as_pair(
            spacing, "spacing", dtype=values.dtype, device=values.device
        )
        if not bool((spacing_t > 0).all().detach().cpu().item()):
            raise ValueError("spacing must be strictly positive")
        height, width = values.shape[:2]
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, dtype=values.dtype, device=values.device),
            torch.linspace(-1, 1, width, dtype=values.dtype, device=values.device),
            indexing="ij",
        )
        design = torch.stack((torch.ones_like(x), x, y), dim=-1).reshape(-1, 3)
        orthogonal, _ = torch.linalg.qr(design, mode="reduced")
        self.mode = mode
        self.control = nn.Parameter(values.detach().clone())
        self.register_buffer("origin", origin_t)
        self.register_buffer("spacing", spacing_t)
        self.register_buffer("affine_basis", orthogonal)

    @property
    def projected_control(self) -> Tensor:
        flat = self.control.reshape(-1, 2)
        basis = self.affine_basis.to(device=flat.device, dtype=flat.dtype)
        projected = flat - basis @ (basis.T @ flat)
        return projected.reshape_as(self.control)

    def evaluate_with_jacobian(self, points: Any) -> tuple[Tensor, Tensor]:
        pts = torch.as_tensor(points)
        if not pts.is_floating_point() or pts.is_complex() or pts.shape[-1:] != (2,):
            raise ValueError("points must be real floating point with last dimension 2")
        _finite(pts, "points")
        control = self.projected_control.to(device=pts.device, dtype=pts.dtype)
        origin = self.origin.to(device=pts.device, dtype=pts.dtype)
        spacing = self.spacing.to(device=pts.device, dtype=pts.dtype)
        return _interpolate_regular_grid(pts, control, origin, spacing, self.mode)

    def forward(self, points: Any) -> Tensor:
        return self.evaluate_with_jacobian(points)[0]

    def jacobian(self, points: Any) -> Tensor:
        return self.evaluate_with_jacobian(points)[1]

    def bending_energy(self) -> Tensor:
        """Mean squared second difference of the projected control grid."""
        grid = self.projected_control
        dxx = grid[:, 2:] - 2 * grid[:, 1:-1] + grid[:, :-2]
        dyy = grid[2:] - 2 * grid[1:-1] + grid[:-2]
        energy = grid.new_zeros(())
        if dxx.numel() > 0:
            energy = energy + dxx.square().mean()
        if dyy.numel() > 0:
            energy = energy + dyy.square().mean()
        return energy


class DetectorToTangentTransform(nn.Module):
    """Compose a fixed table, exposure affine, and optional detector residual."""

    table: RegularGridTransform2D
    affine: TangentPlaneAffine
    residual: AffineFreeResidualGrid2D | None

    def __init__(
        self,
        table: RegularGridTransform2D,
        affine: TangentPlaneAffine,
        residual: AffineFreeResidualGrid2D | None = None,
    ) -> None:
        super().__init__()
        self.table = table
        self.affine = affine
        self.residual = residual
        self.assert_invertible()

    def evaluate_with_jacobian(self, points: Any) -> tuple[Tensor, Tensor]:
        pts = torch.as_tensor(points)
        base, base_jacobian = self.table.evaluate_with_jacobian(pts)
        affine = self.affine(base)
        affine_matrix = self.affine.matrix.to(device=pts.device, dtype=pts.dtype)
        jacobian = torch.matmul(affine_matrix, base_jacobian)
        if self.residual is not None:
            correction, correction_jacobian = self.residual.evaluate_with_jacobian(pts)
            affine = affine + correction
            jacobian = jacobian + correction_jacobian
        return affine, jacobian

    def inverse(
        self,
        values: Any,
        *,
        max_iter: int = 25,
        tol: float | None = None,
        atol: float | None = None,
        rtol: float = 0.0,
    ) -> Tensor:
        tgt = torch.as_tensor(values)
        if not tgt.is_floating_point() or tgt.shape[-1:] != (2,):
            raise ValueError("values must be floating point with last dimension 2")
        mid = self.affine.inverse(tgt)
        if self.residual is None:
            return self.table.inverse(
                mid,
                max_iter=max_iter,
                tol=tol,
                atol=atol,
                rtol=rtol,
            )
        x = self.table.inverse(mid, max_iter=max_iter, fail_closed=False)
        origin = self.table.origin.to(device=tgt.device, dtype=tgt.dtype)
        spacing = self.table.spacing.to(device=tgt.device, dtype=tgt.dtype)
        height, width = self.table.grid_shape
        x0, y0 = origin[0], origin[1]
        x1 = origin[0] + (width - 1) * spacing[0]
        y1 = origin[1] + (height - 1) * spacing[1]
        with torch.no_grad():
            diagnostic_output = self(self.table.diagnostic_points())
            output_extent = (
                diagnostic_output.amax(dim=(0, 1))
                - diagnostic_output.amin(dim=(0, 1))
            ).amax()
        atol_value, rtol_value = _inverse_tolerances(
            target=tgt,
            output_scale=output_extent,
            tol=tol,
            atol=atol,
            rtol=rtol,
        )
        for _ in range(int(max_iter)):
            pred, jac = self.evaluate_with_jacobian(x)
            resid = pred - tgt
            step = torch.linalg.solve(jac, resid.unsqueeze(-1)).squeeze(-1)
            x = x - step
            x = torch.stack(
                (
                    torch.clamp(x[..., 0], min=x0, max=x1),
                    torch.clamp(x[..., 1], min=y0, max=y1),
                ),
                dim=-1,
            )
        pred, _ = self.evaluate_with_jacobian(x)
        residual = pred - tgt
        max_resid = float(residual.abs().amax().detach().cpu().item())
        if not _inverse_residual_ok(
            residual, tgt, atol=atol_value, rtol=rtol_value
        ):
            raise ValueError(
                f"detector inverse residual {max_resid:.8g} exceeds "
                f"atol={atol_value:.8g}, rtol={rtol_value:.8g}"
            )
        return x

    def forward(self, points: Any) -> Tensor:
        return self.evaluate_with_jacobian(points)[0]

    def jacobian_determinant(self, points: Any) -> Tensor:
        return _determinant(self.evaluate_with_jacobian(points)[1])

    def assert_invertible(
        self,
        *,
        subdivisions: int | None = None,
        determinant_floor: float = 0.0,
    ) -> Tensor:
        if determinant_floor < 0:
            raise ValueError("determinant_floor must be non-negative")
        self.affine.assert_invertible(determinant_floor)
        base_determinant = self.table.assert_invertible(subdivisions=subdivisions)
        with torch.no_grad():
            determinant = self.jacobian_determinant(
                self.table.diagnostic_points(subdivisions)
            )
            _finite(determinant, "composite Jacobian determinant")
            signed_magnitude = determinant * base_determinant.sign()
            if bool(
                (signed_magnitude <= determinant_floor).any().detach().cpu().item()
            ):
                raise ValueError("composite detector transform is folded or singular")
        return determinant


class DetectorToSkyTransform(nn.Module):
    """Forward-only detector-to-sky composition with a frozen attitude.

    The attitude callable receives tangent-plane ``(..., 2)`` points and may
    return either an ``(..., 2)`` tensor or an ``(longitude, latitude)`` tuple.
    Its parameters are frozen while derivatives with respect to coordinates
    remain available. Invertibility diagnostics intentionally apply only to the
    detector-to-tangent transform, before any spherical longitude wrapping.
    """

    detector_to_tangent: DetectorToTangentTransform
    attitude: Any

    def __init__(
        self,
        detector_to_tangent: DetectorToTangentTransform,
        attitude: Any,
    ) -> None:
        super().__init__()
        if not callable(attitude):
            raise TypeError("attitude must be callable")
        self.detector_to_tangent = detector_to_tangent
        self.attitude = attitude
        if isinstance(attitude, nn.Module):
            attitude.requires_grad_(False)

    def forward(self, points: Any) -> Tensor:
        tangent = self.detector_to_tangent(points)
        result = self.attitude(tangent)
        if isinstance(result, tuple):
            if len(result) != 2:
                raise ValueError("attitude tuple output must contain longitude, latitude")
            world = torch.stack(result, dim=-1)
        else:
            world = torch.as_tensor(result)
        if world.shape != tangent.shape:
            raise ValueError("attitude must return the same (..., 2) shape it receives")
        if world.dtype != tangent.dtype or world.device != tangent.device:
            raise ValueError("attitude must preserve coordinate dtype and device")
        _finite(world, "attitude output")
        return world

    def pixel_to_world(self, x: Any, y: Any) -> tuple[Tensor, Tensor]:
        x_tensor = torch.as_tensor(x)
        if not x_tensor.is_floating_point() or x_tensor.is_complex():
            raise ValueError("x must be a real floating-point tensor")
        y_tensor = torch.as_tensor(y, dtype=x_tensor.dtype, device=x_tensor.device)
        if not y_tensor.is_floating_point() or y_tensor.is_complex():
            raise ValueError("y must be a real floating-point tensor")
        x_tensor, y_tensor = torch.broadcast_tensors(x_tensor, y_tensor)
        world = self(torch.stack((x_tensor, y_tensor), dim=-1))
        return world[..., 0], world[..., 1]

    def jacobian_determinant(self, points: Any) -> Tensor:
        """Return the detector-to-tangent determinant, not a sky-chart value."""
        return self.detector_to_tangent.jacobian_determinant(points)

    def assert_invertible(
        self,
        *,
        subdivisions: int | None = None,
        determinant_floor: float = 0.0,
    ) -> Tensor:
        return self.detector_to_tangent.assert_invertible(
            subdivisions=subdivisions,
            determinant_floor=determinant_floor,
        )


@dataclass(frozen=True)
class SampledTransformValidation:
    """Angular residual summary for independent transform samples."""

    errors_mas: Tensor
    p99_mas: float
    max_mas: float
    p99_limit_mas: float
    max_limit_mas: float

    @property
    def passed(self) -> bool:
        return self.p99_mas <= self.p99_limit_mas and self.max_mas <= self.max_limit_mas

    def assert_valid(self) -> None:
        if not self.passed:
            raise ValueError(
                f"sampled transform misses tolerance: p99={self.p99_mas:.8g} mas "
                f"(limit {self.p99_limit_mas:.8g}), max={self.max_mas:.8g} mas "
                f"(limit {self.max_limit_mas:.8g})"
            )


def _sky_separation_mas(actual: Tensor, reference: Tensor) -> Tensor:
    degrees_to_radians = torch.pi / 180.0

    def unit(points: Tensor) -> Tensor:
        longitude = points[..., 0] * degrees_to_radians
        latitude = points[..., 1] * degrees_to_radians
        cos_latitude = torch.cos(latitude)
        return torch.stack(
            (
                cos_latitude * torch.cos(longitude),
                cos_latitude * torch.sin(longitude),
                torch.sin(latitude),
            ),
            dim=-1,
        )

    a = unit(actual)
    b = unit(reference)
    cross = torch.linalg.vector_norm(torch.linalg.cross(a, b, dim=-1), dim=-1)
    dot = torch.sum(a * b, dim=-1)
    radians = torch.atan2(cross, dot)
    return radians * (180.0 / torch.pi) * 3_600_000.0


def validate_sampled_transform(
    transform: Any,
    reference: Any,
    points: Any,
    *,
    coordinates: AngularCoordinates = "sky_degrees",
    p99_limit_mas: float = 0.01,
    max_limit_mas: float = 0.05,
    raise_on_failure: bool = False,
) -> SampledTransformValidation:
    """Compare a torch transform with an independent sampled reference."""
    samples = torch.as_tensor(points)
    if not samples.is_floating_point() or samples.shape[-1:] != (2,):
        raise ValueError("points must be floating point with last dimension 2")
    predicted = torch.as_tensor(transform(samples))
    expected = torch.as_tensor(reference(samples), device=predicted.device, dtype=predicted.dtype)
    if predicted.shape != expected.shape or predicted.shape[-1:] != (2,):
        raise ValueError("transform and reference must return matching (..., 2) tensors")
    _finite(predicted, "transform output")
    _finite(expected, "reference output")
    if coordinates == "sky_degrees":
        errors = _sky_separation_mas(predicted, expected)
    elif coordinates == "plane_degrees":
        errors = torch.linalg.vector_norm(predicted - expected, dim=-1) * 3_600_000.0
    elif coordinates == "plane_arcsec":
        errors = torch.linalg.vector_norm(predicted - expected, dim=-1) * 1_000.0
    else:
        raise ValueError(
            "coordinates must be 'sky_degrees', 'plane_degrees', or 'plane_arcsec'"
        )
    flat_errors = errors.reshape(-1)
    if flat_errors.numel() == 0:
        raise ValueError("at least one validation point is required")
    result = SampledTransformValidation(
        errors_mas=errors,
        p99_mas=float(torch.quantile(flat_errors, 0.99).detach().cpu().item()),
        max_mas=float(flat_errors.max().detach().cpu().item()),
        p99_limit_mas=float(p99_limit_mas),
        max_limit_mas=float(max_limit_mas),
    )
    if raise_on_failure:
        result.assert_valid()
    return result
