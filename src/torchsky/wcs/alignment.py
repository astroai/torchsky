from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

AffineGeometry = Literal["shift", "rshift", "rscale", "general"]


@dataclass(frozen=True)
class AffineFitResult:
    """Weighted 2D affine correction fit."""

    matrix: Tensor
    offset: Tensor
    residuals: Tensor
    rms: Tensor
    geometry: str
    inlier_mask: Tensor
    iterations: int

    def apply(self, points: Any) -> Tensor:
        pts = torch.as_tensor(
            points, dtype=self.matrix.dtype, device=self.matrix.device
        )
        return torch.matmul(pts, self.matrix.T) + self.offset


@dataclass(frozen=True)
class TangentPlaneFrame:
    """Gnomonic tangent plane centered on a sky position."""

    ra0: float
    dec0: float

    @classmethod
    def from_points(cls, ra: Any, dec: Any) -> "TangentPlaneFrame":
        ra_t = torch.as_tensor(ra, dtype=torch.float64)
        dec_t = torch.as_tensor(dec, dtype=torch.float64)
        if ra_t.shape != dec_t.shape:
            raise ValueError("ra and dec must have the same shape")
        valid = torch.isfinite(ra_t) & torch.isfinite(dec_t)
        if not bool(valid.any().detach().cpu().item()):
            raise ValueError("at least one finite sky point is required")
        unit = _radec_to_unit(ra_t[valid], dec_t[valid])
        mean = unit.mean(dim=0)
        norm = torch.linalg.norm(mean)
        if float(norm.detach().cpu().item()) <= 1e-30:
            raise ValueError("cannot determine tangent point for antipodal sky points")
        center = mean / norm
        ra0, dec0 = _unit_to_radec(center)
        return cls(float(ra0.detach().cpu().item()), float(dec0.detach().cpu().item()))

    def world_to_plane(self, ra: Any, dec: Any) -> tuple[Tensor, Tensor]:
        """Project sky coordinates in degrees to tangent-plane degrees."""
        ra_t = torch.as_tensor(ra)
        dtype = (
            ra_t.dtype
            if ra_t.is_floating_point() and ra_t.element_size() >= 4
            else torch.float64
        )
        ra_t = ra_t.to(dtype=dtype)
        dec_t = torch.as_tensor(dec, dtype=dtype, device=ra_t.device)
        if ra_t.shape != dec_t.shape:
            raise ValueError("ra and dec must have the same shape")

        d2r = torch.as_tensor(torch.pi / 180.0, dtype=dtype, device=ra_t.device)
        r2d = torch.as_tensor(180.0 / torch.pi, dtype=dtype, device=ra_t.device)
        ra_rad = ra_t * d2r
        dec_rad = dec_t * d2r
        ra0_rad = torch.as_tensor(self.ra0, dtype=dtype, device=ra_t.device) * d2r
        dec0_rad = torch.as_tensor(self.dec0, dtype=dtype, device=ra_t.device) * d2r

        dra = torch.remainder(ra_rad - ra0_rad + torch.pi, 2.0 * torch.pi) - torch.pi
        sin_dec = torch.sin(dec_rad)
        cos_dec = torch.cos(dec_rad)
        sin_dec0 = torch.sin(dec0_rad)
        cos_dec0 = torch.cos(dec0_rad)
        cosc = sin_dec0 * sin_dec + cos_dec0 * cos_dec * torch.cos(dra)
        xi = cos_dec * torch.sin(dra) / cosc
        eta = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * torch.cos(dra)) / cosc
        invalid = cosc <= 0.0
        xi = torch.where(invalid, torch.full_like(xi, torch.nan), xi * r2d)
        eta = torch.where(invalid, torch.full_like(eta, torch.nan), eta * r2d)
        return xi, eta

    def plane_to_world(self, xi: Any, eta: Any) -> tuple[Tensor, Tensor]:
        """Invert tangent-plane degrees to sky coordinates in degrees."""
        xi_t = torch.as_tensor(xi)
        dtype = (
            xi_t.dtype
            if xi_t.is_floating_point() and xi_t.element_size() >= 4
            else torch.float64
        )
        xi_t = xi_t.to(dtype=dtype)
        eta_t = torch.as_tensor(eta, dtype=dtype, device=xi_t.device)
        if xi_t.shape != eta_t.shape:
            raise ValueError("xi and eta must have the same shape")

        d2r = torch.as_tensor(torch.pi / 180.0, dtype=dtype, device=xi_t.device)
        r2d = torch.as_tensor(180.0 / torch.pi, dtype=dtype, device=xi_t.device)
        x = xi_t * d2r
        y = eta_t * d2r
        ra0_rad = torch.as_tensor(self.ra0, dtype=dtype, device=xi_t.device) * d2r
        dec0_rad = torch.as_tensor(self.dec0, dtype=dtype, device=xi_t.device) * d2r

        rho = torch.sqrt(x * x + y * y)
        c = torch.atan(rho)
        sin_c = torch.sin(c)
        cos_c = torch.cos(c)
        sin_dec0 = torch.sin(dec0_rad)
        cos_dec0 = torch.cos(dec0_rad)
        rho_safe = torch.clamp(rho, min=1e-30)

        dec_rad = torch.asin(
            torch.clamp(cos_c * sin_dec0 + (y * sin_c * cos_dec0) / rho_safe, -1.0, 1.0)
        )
        ra_rad = ra0_rad + torch.atan2(
            x * sin_c,
            rho * cos_dec0 * cos_c - y * sin_dec0 * sin_c,
        )
        ra_rad = torch.where(
            rho <= 1e-30, torch.full_like(ra_rad, ra0_rad.item()), ra_rad
        )
        dec_rad = torch.where(
            rho <= 1e-30, torch.full_like(dec_rad, dec0_rad.item()), dec_rad
        )
        return torch.remainder(ra_rad * r2d, 360.0), dec_rad * r2d


@dataclass(frozen=True)
class SkyAffineFitResult:
    """Affine correction fitted in a local tangent plane."""

    frame: TangentPlaneFrame
    affine: AffineFitResult

    @property
    def rms(self) -> Tensor:
        return self.affine.rms

    @property
    def inlier_mask(self) -> Tensor:
        return self.affine.inlier_mask

    def apply(self, ra: Any, dec: Any) -> tuple[Tensor, Tensor]:
        xi, eta = self.frame.world_to_plane(ra, dec)
        plane = torch.stack([xi, eta], dim=-1)
        corrected = self.affine.apply(plane)
        return self.frame.plane_to_world(corrected[..., 0], corrected[..., 1])


def _radec_to_unit(ra: Tensor, dec: Tensor) -> Tensor:
    d2r = torch.as_tensor(torch.pi / 180.0, dtype=torch.float64, device=ra.device)
    ra_rad = ra * d2r
    dec_rad = dec * d2r
    cos_dec = torch.cos(dec_rad)
    return torch.stack(
        [cos_dec * torch.cos(ra_rad), cos_dec * torch.sin(ra_rad), torch.sin(dec_rad)],
        dim=-1,
    )


def _unit_to_radec(unit: Tensor) -> tuple[Tensor, Tensor]:
    r2d = torch.as_tensor(180.0 / torch.pi, dtype=torch.float64, device=unit.device)
    x, y, z = unit.unbind(dim=-1)
    ra = torch.remainder(torch.atan2(y, x) * r2d, 360.0)
    dec = torch.asin(torch.clamp(z, -1.0, 1.0)) * r2d
    return ra, dec


def _as_points(value: Any, name: str) -> Tensor:
    pts = torch.as_tensor(value, dtype=torch.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    if pts.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one point")
    return pts


def _weighted_mean(x: Tensor, w: Tensor) -> Tensor:
    return torch.sum(x * w[:, None], dim=0) / torch.sum(w)


def _solve_weighted_linear(design: Tensor, target: Tensor, weights: Tensor) -> Tensor:
    sqrt_w = torch.sqrt(weights)[:, None]
    a = design * sqrt_w
    b = target * sqrt_w
    return torch.linalg.lstsq(a, b).solution


def _fit_once(
    source: Tensor,
    target: Tensor,
    weights: Tensor,
    geometry: AffineGeometry,
) -> tuple[Tensor, Tensor]:
    src_cent = _weighted_mean(source, weights)
    tgt_cent = _weighted_mean(target, weights)
    src0 = source - src_cent
    tgt0 = target - tgt_cent

    if geometry == "shift":
        matrix = torch.eye(2, dtype=torch.float64, device=source.device)
        offset = _weighted_mean(target - source, weights)
        return matrix, offset

    if geometry == "general":
        design = torch.cat(
            [
                source,
                torch.ones(
                    (source.shape[0], 1), dtype=source.dtype, device=source.device
                ),
            ],
            dim=1,
        )
        sol = _solve_weighted_linear(design, target, weights)
        matrix = sol[:2, :].T
        offset = sol[2, :]
        return matrix, offset

    z_src = torch.complex(src0[:, 0], src0[:, 1])
    z_tgt = torch.complex(tgt0[:, 0], tgt0[:, 1])
    denom = torch.sum(weights * (z_src.real * z_src.real + z_src.imag * z_src.imag))
    if torch.abs(denom) < 1e-30:
        raise ValueError("degenerate source points for rotation/scale fit")
    complex_num = torch.sum(weights * z_tgt * torch.conj(z_src)) / denom
    if geometry == "rshift":
        complex_num = complex_num / torch.clamp(torch.abs(complex_num), min=1e-30)
    elif geometry != "rscale":
        raise ValueError(f"unsupported affine geometry {geometry!r}")

    a = complex_num.real
    b = complex_num.imag
    matrix = torch.stack(
        [torch.stack([a, -b]), torch.stack([b, a])],
        dim=0,
    ).to(dtype=torch.float64)
    offset = tgt_cent - torch.matmul(src_cent, matrix.T)
    return matrix, offset


def fit_affine_2d(
    source: Any,
    target: Any,
    *,
    weights: Any | None = None,
    geometry: AffineGeometry = "general",
    sigma_clip: float | None = None,
    max_iter: int = 3,
) -> AffineFitResult:
    """Fit a weighted 2D correction from ``source`` to ``target`` points.

    Geometry modes mirror common tweakwcs alignment families:
    ``shift`` (translation), ``rshift`` (rotation + translation),
    ``rscale`` (uniform scale + rotation + translation), and ``general``
    (full affine + translation).

    Args:
        source: Input points with shape ``(N, 2)``.
        target: Corresponding target points with shape ``(N, 2)``.
        weights: Optional nonnegative per-point weights.
        geometry: Constrained affine transformation family.
        sigma_clip: Optional residual clipping threshold in robust sigmas.
        max_iter: Maximum clipping-and-refit iterations.

    Returns:
        Fitted matrix, offset, residuals, mask, and fit diagnostics.

    Examples:
        >>> import torch
        >>> from torchsky.wcs import fit_affine_2d
        >>> fit = fit_affine_2d(torch.tensor([[0.0, 0.0]]), torch.tensor([[1.0, 2.0]]), geometry="shift")
        >>> fit.offset.tolist()
        [1.0, 2.0]
    """
    src = _as_points(source, "source")
    tgt = _as_points(target, "target").to(device=src.device)
    if src.shape != tgt.shape:
        raise ValueError("source and target must have the same shape")
    if src.shape[0] < {"shift": 1, "rshift": 2, "rscale": 2, "general": 3}[geometry]:
        raise ValueError(f"not enough points for {geometry!r} affine fit")

    if weights is None:
        w = torch.ones(src.shape[0], dtype=torch.float64, device=src.device)
    else:
        w = torch.as_tensor(weights, dtype=torch.float64, device=src.device)
        if w.shape != (src.shape[0],):
            raise ValueError("weights must have shape (N,)")
    valid = (
        torch.isfinite(src).all(dim=1)
        & torch.isfinite(tgt).all(dim=1)
        & torch.isfinite(w)
        & (w > 0)
    )
    if not bool(valid.any()):
        raise ValueError("no valid weighted point pairs")

    inliers = valid.clone()
    matrix = torch.eye(2, dtype=torch.float64, device=src.device)
    offset = torch.zeros(2, dtype=torch.float64, device=src.device)
    iterations = 0
    for iterations in range(1, max(int(max_iter), 1) + 1):
        matrix, offset = _fit_once(src[inliers], tgt[inliers], w[inliers], geometry)
        fitted = torch.matmul(src, matrix.T) + offset
        residuals = tgt - fitted
        norm = torch.linalg.norm(residuals, dim=1)
        if sigma_clip is None:
            break
        norm_in = norm[inliers]
        median = torch.median(norm_in)
        mad = torch.median(torch.abs(norm_in - median))
        robust_sigma = mad
        if float(robust_sigma.detach().cpu().item()) <= 1e-30:
            rms_clip = torch.sqrt(
                torch.sum(w[inliers] * norm_in * norm_in) / torch.sum(w[inliers])
            )
            limit = float(sigma_clip) * torch.clamp(rms_clip, min=1e-30)
        else:
            limit = median + float(sigma_clip) * robust_sigma
        new_inliers = valid & (norm <= limit)
        if bool(torch.equal(new_inliers, inliers)):
            break
        if (
            int(new_inliers.sum().item())
            < {"shift": 1, "rshift": 2, "rscale": 2, "general": 3}[geometry]
        ):
            break
        inliers = new_inliers

    fitted = torch.matmul(src, matrix.T) + offset
    residuals = tgt - fitted
    norm = torch.linalg.norm(residuals[inliers], dim=1)
    rms = torch.sqrt(torch.sum(w[inliers] * norm * norm) / torch.sum(w[inliers]))
    return AffineFitResult(
        matrix=matrix,
        offset=offset,
        residuals=residuals,
        rms=rms,
        geometry=geometry,
        inlier_mask=inliers,
        iterations=iterations,
    )


def fit_sky_affine(
    source_ra: Any,
    source_dec: Any,
    target_ra: Any,
    target_dec: Any,
    *,
    frame: TangentPlaneFrame | None = None,
    weights: Any | None = None,
    geometry: AffineGeometry = "general",
    sigma_clip: float | None = None,
    max_iter: int = 3,
) -> SkyAffineFitResult:
    """Fit a local sky-coordinate affine correction in a gnomonic plane."""
    src_ra = torch.as_tensor(source_ra, dtype=torch.float64)
    src_dec = torch.as_tensor(source_dec, dtype=torch.float64)
    tgt_ra = torch.as_tensor(target_ra, dtype=torch.float64)
    tgt_dec = torch.as_tensor(target_dec, dtype=torch.float64)
    if (
        src_ra.shape != src_dec.shape
        or src_ra.shape != tgt_ra.shape
        or src_ra.shape != tgt_dec.shape
    ):
        raise ValueError(
            "source and target sky coordinate arrays must have matching shapes"
        )
    if frame is None:
        frame = TangentPlaneFrame.from_points(
            torch.cat([src_ra.reshape(-1), tgt_ra.reshape(-1)]),
            torch.cat([src_dec.reshape(-1), tgt_dec.reshape(-1)]),
        )
    src_x, src_y = frame.world_to_plane(src_ra, src_dec)
    tgt_x, tgt_y = frame.world_to_plane(tgt_ra, tgt_dec)
    source = torch.stack([src_x.reshape(-1), src_y.reshape(-1)], dim=-1)
    target = torch.stack([tgt_x.reshape(-1), tgt_y.reshape(-1)], dim=-1)
    affine = fit_affine_2d(
        source,
        target,
        weights=weights,
        geometry=geometry,
        sigma_clip=sigma_clip,
        max_iter=max_iter,
    )
    return SkyAffineFitResult(frame=frame, affine=affine)
