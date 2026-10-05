from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

from .stats import huber_location, tukey_location


def _chebyshev_basis_2d(
    H: int,
    W: int,
    order: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Build 2D Chebyshev polynomial basis [H*W, (order+1)*(order+2)/2]."""
    return _chebyshev_basis_2d_window(
        H,
        W,
        order,
        y0=0,
        y1=H,
        x0=0,
        x1=W,
        device=device,
        dtype=dtype,
    )


def _chebyshev_basis_2d_window(
    H: int,
    W: int,
    order: int,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Build the global-coordinate Chebyshev basis for one output window."""
    if not (0 <= y0 < y1 <= H and 0 <= x0 < x1 <= W):
        raise ValueError("polynomial render window is out of bounds")
    y = torch.linspace(-1, 1, H, device=device, dtype=dtype)[y0:y1]
    x = torch.linspace(-1, 1, W, device=device, dtype=dtype)[x0:x1]
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    T = [torch.ones_like(gx), gx]
    for n in range(2, order + 1):
        T.append(2 * gx * T[n - 1] - T[n - 2])
    U = [torch.ones_like(gy), gy]
    for n in range(2, order + 1):
        U.append(2 * gy * U[n - 1] - U[n - 2])

    basis = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            basis.append(U[i] * T[j])
    return torch.stack(basis, dim=-1).reshape(-1, len(basis))


DifferenceTile = tuple[int, int, int, int, Tensor, Tensor]
DifferenceTileFactory = Callable[[], Iterable[DifferenceTile]]


@dataclass(frozen=True)
class CompactPolynomialSkyModel:
    """Polynomial coefficients that render bounded output windows."""

    coefficients: Tensor
    image_shape: tuple[int, int]
    order: int

    def render_window(self, y0: int, y1: int, x0: int, x1: int) -> Tensor:
        height, width = self.image_shape
        basis = _chebyshev_basis_2d_window(
            height,
            width,
            self.order,
            y0=int(y0),
            y1=int(y1),
            x0=int(x0),
            x1=int(x1),
            device=self.coefficients.device,
            dtype=self.coefficients.dtype,
        )
        return (basis @ self.coefficients).reshape(y1 - y0, x1 - x0)


def _validate_row_major_tiling(
    windows: list[tuple[int, int, int, int]],
    image_shape: tuple[int, int],
) -> None:
    """Require exact row-major rectangle coverage without a pixel-sized bitmap."""
    height, width = image_shape
    index = 0
    expected_y = 0
    while index < len(windows):
        y0, y1, _, _ = windows[index]
        if y0 != expected_y or y1 <= y0:
            raise ValueError("compact polynomial tiles must cover rows contiguously")
        expected_x = 0
        while index < len(windows) and windows[index][:2] == (y0, y1):
            _, _, x0, x1 = windows[index]
            if x0 != expected_x or x1 <= x0:
                raise ValueError(
                    "compact polynomial tiles must cover columns contiguously"
                )
            expected_x = x1
            index += 1
        if expected_x != width:
            raise ValueError("compact polynomial tile row does not cover the image")
        expected_y = y1
    if expected_y != height:
        raise ValueError("compact polynomial tiles do not cover the image")


class PolynomialSkyModel:
    """2D polynomial sky background via one-step robust WLS.

    Fits a Chebyshev polynomial of given *order* to image pixels using
    weighted least squares with robust per-pixel weights from the
    chosen *statistic* (``"huber"`` or ``"tukey"``).

    Parameters
    ----------
    order
        Maximum polynomial degree (total degree, not per-axis).  Order
        2 fits ``1, x, y, x*y, x^2, y^2`` (6 terms).
    statistic
        Robust weight function: ``"huber"`` (default) or ``"tukey"``.
    """

    def __init__(
        self,
        *,
        order: int = 3,
        statistic: str = "huber",
    ):
        if statistic not in ("huber", "tukey"):
            raise ValueError(f"unknown statistic '{statistic}'; choose 'huber' or 'tukey'")
        if order < 0:
            raise ValueError(f"order must be >= 0, got {order}")
        self.order = order
        self.statistic = statistic

    def fit_compact(
        self,
        tile_factory: DifferenceTileFactory,
        *,
        image_shape: tuple[int, int],
        scratch: Tensor,
    ) -> CompactPolynomialSkyModel:
        """Fit robust coefficients from replayable tiles using one spill plane.

        The tile factory is replayed for the robust-location, scale, and normal-
        equation passes. Only polynomial-sized matrices and one tile are live in
        memory; ``scratch`` supplies exact upper-median selection without a
        rendered model plane.
        """

        height, width = int(image_shape[0]), int(image_shape[1])
        if height <= 0 or width <= 0:
            raise ValueError("compact polynomial image shape must be positive")
        if tuple(scratch.shape) != (height, width) or not scratch.is_floating_point():
            raise ValueError(
                "compact polynomial scratch must be a floating image-sized tensor"
            )
        if scratch.device.type != "cpu" or not scratch.is_contiguous():
            raise ValueError(
                "compact polynomial scratch must be contiguous CPU storage"
            )
        layout: list[tuple[int, int, int, int]] = []

        def checked_tiles(*, record_layout: bool) -> Iterable[DifferenceTile]:
            seen = 0
            for row in tile_factory():
                if len(row) != 6:
                    raise ValueError("compact polynomial tile rows require six values")
                y0, y1, x0, x1, difference, support = row
                window = (int(y0), int(y1), int(x0), int(x1))
                y0, y1, x0, x1 = window
                if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
                    raise ValueError("compact polynomial tile is out of bounds")
                window_shape = (y1 - y0, x1 - x0)
                if (
                    tuple(difference.shape) != window_shape
                    or tuple(support.shape) != window_shape
                ):
                    raise ValueError(
                        "compact polynomial tile data must match its window"
                    )
                if (
                    difference.device != scratch.device
                    or support.device != scratch.device
                    or difference.dtype != scratch.dtype
                ):
                    raise ValueError(
                        "compact polynomial tiles and scratch must share device/dtype"
                    )
                if record_layout:
                    layout.append(window)
                elif seen >= len(layout) or layout[seen] != window:
                    raise ValueError(
                        "compact polynomial tile factory changed between passes"
                    )
                seen += 1
                yield y0, y1, x0, x1, difference, support
            if record_layout:
                _validate_row_major_tiling(layout, (height, width))
            elif seen != len(layout):
                raise ValueError(
                    "compact polynomial tile factory changed between passes"
                )

        def fill_scratch(
            transform: Callable[[Tensor], Tensor],
            *,
            record_layout: bool = False,
        ) -> int:
            scratch.fill_(float("inf"))
            valid_count = 0
            for y0, y1, x0, x1, difference, support in checked_tiles(
                record_layout=record_layout
            ):
                valid = support.to(dtype=torch.bool)
                if bool((~torch.isfinite(difference[valid])).any()):
                    raise ValueError(
                        "compact polynomial supported differences must be finite"
                    )
                transformed = transform(difference)
                if transformed.shape != difference.shape:
                    raise ValueError(
                        "compact polynomial scratch transform changed tile shape"
                    )
                scratch[y0:y1, x0:x1].copy_(
                    torch.where(
                        valid,
                        transformed,
                        torch.full_like(transformed, float("inf")),
                    )
                )
                valid_count += int(valid.sum().item())
            return valid_count

        def upper_median(valid_count: int) -> Tensor:
            if valid_count <= 0:
                raise ValueError("overlap support must contain at least one pixel")
            index = valid_count // 2
            # NumPy's in-place 1-D introselect operates directly on the
            # disk-backed scratch mapping with bounded auxiliary memory.
            values = scratch.detach().reshape(-1).numpy()
            values.partition(index)
            return scratch.new_tensor(values[index].item())

        valid_count = fill_scratch(lambda values: values, record_layout=True)
        median = upper_median(valid_count)
        fill_scratch(lambda values: torch.abs(values - median))
        initial_mad = upper_median(valid_count).clamp(min=1.0e-10)
        initial_scale = initial_mad * 1.4826
        location_numerator = scratch.new_zeros(())
        location_denominator = scratch.new_zeros(())
        for _, _, _, _, difference, support in checked_tiles(record_layout=False):
            valid = support.to(dtype=torch.bool)
            values = difference[valid]
            residual = values - median
            if self.statistic == "huber":
                threshold = 1.345 * initial_scale
                weights = torch.where(
                    residual.abs() <= threshold,
                    torch.ones_like(residual),
                    threshold / residual.abs(),
                )
            else:
                scaled = residual / (4.685 * initial_scale)
                scaled_square = scaled.square()
                weights = torch.where(
                    scaled_square < 1.0,
                    (1.0 - scaled_square).square(),
                    torch.zeros_like(scaled_square),
                )
            location_numerator.add_((weights * values).sum())
            location_denominator.add_(weights.sum())
        location = location_numerator / location_denominator.clamp(min=1.0e-10)

        fill_scratch(lambda values: torch.abs(values - location))
        regression_mad = upper_median(valid_count).clamp(min=1.0e-10)
        regression_scale = regression_mad * 1.4826
        n_parameters = (self.order + 1) * (self.order + 2) // 2
        if valid_count < n_parameters:
            raise ValueError(
                f"not enough valid pixels ({valid_count}) for "
                f"{n_parameters} polynomial parameters"
            )
        normal = torch.zeros(
            (n_parameters, n_parameters),
            dtype=scratch.dtype,
            device=scratch.device,
        )
        rhs = torch.zeros(
            (n_parameters,),
            dtype=scratch.dtype,
            device=scratch.device,
        )
        for y0, y1, x0, x1, difference, support in checked_tiles(
            record_layout=False
        ):
            valid = support.to(dtype=torch.bool).reshape(-1)
            values = difference.reshape(-1)[valid]
            basis = _chebyshev_basis_2d_window(
                height,
                width,
                self.order,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                device=scratch.device,
                dtype=scratch.dtype,
            )[valid]
            residual = values - location
            if self.statistic == "huber":
                scaled = residual / regression_scale
                threshold = torch.tensor(
                    1.345, dtype=scratch.dtype, device=scratch.device
                )
                weights = torch.where(
                    scaled.abs() <= threshold,
                    torch.ones_like(scaled),
                    threshold / scaled.abs(),
                )
            else:
                scaled_square = (residual / (4.685 * regression_scale)).square()
                weights = torch.where(
                    scaled_square < 1.0,
                    (1.0 - scaled_square).square(),
                    torch.zeros_like(scaled_square),
                )
            weight_sqrt = weights.sqrt().clamp(min=1.0e-10)
            weighted_basis = basis * weight_sqrt.unsqueeze(-1)
            weighted_values = values * weight_sqrt
            normal.add_(weighted_basis.T @ weighted_basis)
            rhs.add_(weighted_basis.T @ weighted_values)
        try:
            coefficients = torch.linalg.solve(normal, rhs)
        except torch.linalg.LinAlgError:
            coefficients = torch.linalg.lstsq(normal, rhs, driver="gelsd").solution
        return CompactPolynomialSkyModel(
            coefficients=coefficients,
            image_shape=(height, width),
            order=self.order,
        )

    def __call__(
        self,
        image: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        dtype = image.dtype
        H, W = image.shape[-2:]
        device = image.device

        basis = _chebyshev_basis_2d(H, W, self.order, device, dtype)
        y = image.reshape(-1)

        if mask is not None:
            valid = ~mask.reshape(-1)
            fit_basis = basis[valid]
            fit_y = y[valid]
        else:
            fit_basis = basis
            fit_y = y

        n_param = fit_basis.shape[-1]
        if fit_y.shape[0] < n_param:
            raise ValueError(
                f"not enough valid pixels ({fit_y.shape[0]}) for {n_param} polynomial parameters"
            )

        if self.statistic == "huber":
            loc_fn = huber_location
        else:
            loc_fn = tukey_location

        theta, _ = self._fit_irls(fit_basis, fit_y, loc_fn)
        sky = (basis @ theta).reshape(H, W)
        return sky.to(dtype=dtype)

    @staticmethod
    def _fit_irls(
        basis: Tensor,
        y: Tensor,
        loc_fn,
    ) -> tuple[Tensor, Tensor]:
        init = loc_fn(y, dim=-1, keepdim=True)
        r = y - init
        mad = torch.sort(r.abs(), dim=-1).values[r.shape[-1] // 2].clamp(min=1e-10)
        scale = (mad * 1.4826).clamp(min=1e-10)
        u = r / scale
        if loc_fn is huber_location:
            thresh = torch.tensor(1.345, device=y.device, dtype=y.dtype)
            w = torch.where(
                u.abs() <= thresh,
                torch.ones_like(u),
                thresh / u.abs().clamp_min(1e-10),
            )
        else:
            delta = torch.tensor(4.685, device=y.device, dtype=y.dtype)
            u_sq = (u / delta) ** 2
            w = torch.where(u_sq < 1.0, (1.0 - u_sq) ** 2, torch.zeros_like(u))

        w_sqrt = w.sqrt().clamp(min=1e-10)
        Xw = basis * w_sqrt.unsqueeze(-1)
        yw = y * w_sqrt
        XtX = Xw.T @ Xw
        Xty = Xw.T @ yw
        try:
            theta = torch.linalg.solve(XtX, Xty)
        except torch.linalg.LinAlgError:
            theta = torch.linalg.lstsq(XtX, Xty, driver="gelsd").solution
        return theta, w

    def estimate(
        self,
        image: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        return self.__call__(image, mask)
