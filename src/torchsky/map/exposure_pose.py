"""Learnable per-exposure astrometric residual in a local tangent plane.

Patches are modeled in Torchsky's exact gnomonic tangent chart with four scalars:

- ``delta_ra_deg``, ``delta_dec_deg``: translation in tangent-plane coordinates
  mapped back to RA/Dec. ``delta_ra_deg`` is the local RA-coordinate offset at
  the tangent point (its plane translation is ``delta_ra_deg * cos(dec0)``).
- ``delta_theta_rad``: in-plane rotation about the patch center.
- ``delta_log_scale``: isotropic log-scale about the center.

The shared :class:`~torchsky.wcs.TangentPlaneFrame` owns the spherical
projection, avoiding a second small-angle chart convention.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from ..wcs import TangentPlaneFrame, WCS

if TYPE_CHECKING:
    from .field.types import FieldObservation


class ExposurePose(nn.Module):
    """Identity at initialization (all deltas zero)."""

    def __init__(self) -> None:
        super().__init__()
        self.delta_ra_deg = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.delta_dec_deg = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.delta_theta_rad = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.delta_log_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def apply_to_world(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        *,
        center_ra_deg: float,
        center_dec_deg: float,
    ) -> tuple[Tensor, Tensor]:
        """Apply pose in an exact gnomonic chart, then map back to the sphere."""
        ra0 = float(center_ra_deg)
        dec0 = float(center_dec_deg)
        cos_dec0 = math.cos(math.radians(dec0))
        if abs(cos_dec0) < 1e-6:
            raise ValueError(
                "ExposurePose.apply_to_world is ill-conditioned near the pole "
                "(|cos(dec0)| too small); pick a different patch center."
            )

        if ra_deg.shape != dec_deg.shape:
            raise ValueError("ra_deg and dec_deg must have matching shapes")
        if ra_deg.device != dec_deg.device or ra_deg.dtype != dec_deg.dtype:
            raise ValueError("ra_deg and dec_deg must share dtype and device")
        if not ra_deg.is_floating_point():
            raise TypeError("ra_deg and dec_deg must have floating dtype")
        dtype, device = ra_deg.dtype, ra_deg.device
        frame = TangentPlaneFrame(ra0=ra0, dec0=dec0)
        xi, eta = frame.world_to_plane(ra_deg, dec_deg)

        theta = self.delta_theta_rad.to(device=device, dtype=dtype)
        scale = torch.exp(self.delta_log_scale.to(device=device, dtype=dtype))
        c = torch.cos(theta)
        st = torch.sin(theta)

        xi_r = scale * (c * xi - st * eta)
        eta_r = scale * (st * xi + c * eta)

        d_ra = self.delta_ra_deg.to(device=device, dtype=dtype)
        d_dec = self.delta_dec_deg.to(device=device, dtype=dtype)
        xi_out = xi_r + d_ra * cos_dec0
        eta_out = eta_r + d_dec

        return frame.plane_to_world(xi_out, eta_out)

    def compose_with_wcs(
        self,
        wcs: WCS,
        ra_deg: Tensor,
        dec_deg: Tensor,
        *,
        center_ra_deg: float,
        center_dec_deg: float,
    ) -> tuple[Tensor, Tensor]:
        """Apply :meth:`apply_to_world`, then differentiable ``wcs.world_to_pixel``."""
        ra_p, dec_p = self.apply_to_world(
            ra_deg,
            dec_deg,
            center_ra_deg=center_ra_deg,
            center_dec_deg=center_dec_deg,
        )
        world = torch.stack((ra_p, dec_p), dim=-1)
        pixels = wcs.world_to_pixel_differentiable(world)
        return pixels[..., 0], pixels[..., 1]


def freeze_pose(pose: ExposurePose) -> None:
    """Disable gradients on all pose parameters (e.g. anchor one exposure)."""
    for p in pose.parameters():
        p.requires_grad = False


def create_exposure_poses(
    observations: list[FieldObservation],
    *,
    anchor_first: bool = True,
) -> list[ExposurePose]:
    """Create a list of ExposurePose objects, one for each observation.

    Args:
        observations: List of observations to create poses for.
        anchor_first: If True, the first pose will have ``requires_grad=False``
            to fix the absolute astrometric gauge (preventing global drift).
    """
    poses = [ExposurePose() for _ in observations]
    if anchor_first and poses:
        freeze_pose(poses[0])
    return poses


def fit_pose_correction(
    src_ra_deg: Tensor,
    src_dec_deg: Tensor,
    dst_ra_deg: Tensor,
    dst_dec_deg: Tensor,
    *,
    center_ra_deg: float,
    center_dec_deg: float,
    model: str = "rshift",
) -> ExposurePose:
    """Fit a rigid tangent-plane correction mapping matched ``src`` onto ``dst``.

    Pairs are matched sky positions (e.g. from
    :func:`torchsky.catalogs.crossmatch_sky`): ``src`` in the exposure's
    current frame, ``dst`` in the reference frame. The closed-form
    Kabsch/Procrustes solution is computed in the same tangent plane used by
    the renderer. Callers fitting an exposure must therefore pass that
    exposure's WCS reference point (normally ``CRVAL1``/``CRVAL2``). Rotation
    and translation are origin-dependent, so silently choosing a matched
    catalogue source as the center would produce the wrong correction when
    the renderer later applies the pose about CRVAL.

    Models:
    - ``"shift"``: translation only.
    - ``"rshift"``: translation plus in-plane rotation about the tangent pole.

    Inputs may require grad; the computation runs as tensor ops on the
    inputs' device/dtype, but the returned :class:`ExposurePose` parameters
    are fresh leaves initialized from the closed-form solution (refine them
    with the usual optimizers, not through this function).
    """
    if model not in ("shift", "rshift"):
        raise ValueError(f"unsupported pose-correction model {model!r}")

    src_ra = torch.as_tensor(src_ra_deg).reshape(-1)
    src_dec = torch.as_tensor(src_dec_deg).reshape(-1)
    dst_ra = torch.as_tensor(dst_ra_deg).reshape(-1)
    dst_dec = torch.as_tensor(dst_dec_deg).reshape(-1)
    if not (src_ra.shape == src_dec.shape == dst_ra.shape == dst_dec.shape):
        raise ValueError("src/dst coordinate tensors must have matching shapes")
    if not (
        src_ra.device == src_dec.device == dst_ra.device == dst_dec.device
        and src_ra.dtype == src_dec.dtype == dst_ra.dtype == dst_dec.dtype
    ):
        raise ValueError("src/dst coordinate tensors must share dtype and device")
    if not src_ra.is_floating_point():
        raise TypeError("src/dst coordinate tensors must have floating dtype")
    n = int(src_ra.numel())
    if n == 0:
        raise ValueError("at least one matched pair is required")
    if model == "rshift" and n < 2:
        raise ValueError("rshift requires at least two non-degenerate matched pairs")
    if not all(
        bool(torch.isfinite(values).all())
        for values in (src_ra, src_dec, dst_ra, dst_dec)
    ):
        raise ValueError("src/dst coordinate tensors must contain only finite values")

    ra0 = float(center_ra_deg)
    dec0 = float(center_dec_deg)
    if not math.isfinite(ra0) or not math.isfinite(dec0):
        raise ValueError("pose center coordinates must be finite")
    cos_dec0 = math.cos(math.radians(dec0))
    if abs(cos_dec0) < 1e-6:
        raise ValueError(
            "fit_pose_correction is ill-conditioned near the pole "
            "(|cos(dec0)| too small); pick a different tangent center."
        )

    frame = TangentPlaneFrame(ra0=ra0, dec0=dec0)

    def _tangent(ra: Tensor, dec: Tensor) -> tuple[Tensor, Tensor]:
        return frame.world_to_plane(ra, dec)

    xi_s, eta_s = _tangent(src_ra, src_dec)
    xi_d, eta_d = _tangent(dst_ra, dst_dec)

    sx = xi_s - xi_s.mean()
    sy = eta_s - eta_s.mean()
    dx = xi_d - xi_d.mean()
    dy = eta_d - eta_d.mean()
    if model == "shift":
        theta = torch.zeros((), dtype=src_ra.dtype, device=src_ra.device)
    else:
        src_power = sx.square().sum() + sy.square().sum()
        dst_power = dx.square().sum() + dy.square().sum()
        scale = torch.stack(
            (xi_s.abs().max(), eta_s.abs().max(), xi_d.abs().max(), eta_d.abs().max())
        ).max()
        tolerance = (
            torch.finfo(src_ra.dtype).eps
            * torch.maximum(
                scale.square(), torch.full_like(scale, torch.finfo(src_ra.dtype).tiny)
            )
            * n
        )
        if bool(src_power <= tolerance) or bool(dst_power <= tolerance):
            raise ValueError("rshift matched pairs are degenerate in the tangent plane")
        # Maximize sum d @ R(theta) s with R = [[c, -s], [s, c]]:
        # theta = atan2(sum(dy*sx - dx*sy), sum(dx*sx + dy*sy)).
        sin_num = (dy * sx - dx * sy).sum()
        cos_den = (dx * sx + dy * sy).sum()
        theta = torch.atan2(sin_num, cos_den)
    c = torch.cos(theta)
    s = torch.sin(theta)
    tx = xi_d.mean() - (c * xi_s.mean() - s * eta_s.mean())
    ty = eta_d.mean() - (s * xi_s.mean() + c * eta_s.mean())

    pose = ExposurePose().to(device=src_ra.device, dtype=src_ra.dtype)
    with torch.no_grad():
        pose.delta_theta_rad.copy_(theta)
        pose.delta_log_scale.zero_()
        # apply_to_world adds delta_ra_deg * cos(dec0) to the xi coordinate.
        pose.delta_ra_deg.copy_(tx / cos_dec0)
        pose.delta_dec_deg.copy_(ty)
    return pose
