from __future__ import annotations

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
    y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
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
                thresh / u.abs(),
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
