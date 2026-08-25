"""Learnable per-exposure astrometric residual in a local tangent plane.

Small patches (typical HST/JWST cutouts) are modeled with four scalars:

- ``delta_ra_deg``, ``delta_dec_deg``: translation in tangent-plane coordinates
  mapped back to RA/Dec. ``delta_ra_deg`` is the RA offset **at the tangent
  point** (declination = *center_dec_deg*); the applied RA shift varies per
  pixel as ``delta_ra_deg * cos(dec0) / cos(dec_out)`` to correctly account
  for the convergence of RA meridians.
- ``delta_theta_rad``: in-plane rotation about the patch center.
- ``delta_log_scale``: isotropic log-scale about the center.

The forward transform uses a first-order tangent-plane expansion with
per-point ``cos(dec)`` scaling for RA offsets. This is more accurate than a
constant ``cos(dec0)`` for mid-latitude fields up to several degrees, but
remains an approximation. Poles and very large fields may need a spherically
exact formulation (research extension).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from ..wcs import WCS

if TYPE_CHECKING:
    from .field.types import FieldObservation


def _wrapped_delta_ra_deg(ra_a_deg: Tensor, ra_b_deg: float) -> Tensor:
    return ((ra_a_deg - float(ra_b_deg) + 180.0) % 360.0) - 180.0


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
        """Apply pose in tangent coordinates at ``(center_ra_deg, center_dec_deg)``, then map back.

        The tangent-plane projection uses per-point ``cos(dec)`` to scale
        RA offsets, which is more accurate than a constant ``cos(dec0)`` for
        mid-latitude fields. Near the pole (``|cos(dec)| < 1e-12``) the
        per-point scaling saturates to avoid division by zero; the global
        polar guard at ``center_dec_deg`` remains for early rejection of
        ill-conditioned inputs.
        """
        ra0 = float(center_ra_deg)
        dec0 = float(center_dec_deg)
        cos_dec0 = math.cos(math.radians(dec0))
        if abs(cos_dec0) < 1e-6:
            raise ValueError(
                "ExposurePose.apply_to_world is ill-conditioned near the pole "
                "(|cos(dec0)| too small); pick a different patch center."
            )

        dtype = ra_deg.dtype
        device = ra_deg.device

        cos_dec = torch.cos(torch.deg2rad(dec_deg))
        xi = _wrapped_delta_ra_deg(ra_deg, ra0) * cos_dec
        eta = dec_deg - dec0

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

        dec_out = dec0 + eta_out
        cos_dec_out = torch.cos(torch.deg2rad(dec_out))
        cos_dec_out = torch.where(
            torch.abs(cos_dec_out) < 1e-12,
            torch.tensor(1e-12, dtype=dtype, device=device),
            cos_dec_out,
        )
        ra_out = ra0 + xi_out / cos_dec_out

        ra_out = torch.remainder(ra_out + 180.0, 360.0) - 180.0
        return ra_out, dec_out

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
    model: str = "rshift",
    anchor_index: int = 0,
) -> ExposurePose:
    """Fit a rigid tangent-plane correction mapping matched ``src`` onto ``dst``.

    Pairs are matched sky positions (e.g. from
    :func:`torchsky.catalogs.crossmatch_sky`): ``src`` in the exposure's
    current frame, ``dst`` in the reference frame. The closed-form
    Kabsch/Procrustes solution is computed in the tangent plane centered on
    the anchor pair ``src[anchor_index]``, which acts as the gauge: the
    rotation pole sits at the anchor source position and an exposure already
    in the reference frame (self-pairs, ``src is dst``) fits to the identity
    pose, so the anchor exposure stays identity-frozen while all other
    exposures receive corrections relative to it.

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
    n = int(src_ra.numel())
    if n == 0:
        raise ValueError("at least one matched pair is required")
    anchor = int(anchor_index)
    if not 0 <= anchor < n:
        raise ValueError(f"anchor_index {anchor} out of range for {n} pairs")

    # Gauge point: the anchor source position. Kept as python floats because
    # ExposurePose.apply_to_world anchors its tangent plane on floats too.
    ra0 = float(src_ra[anchor].detach())
    dec0 = float(src_dec[anchor].detach())
    cos_dec0 = math.cos(math.radians(dec0))
    if abs(cos_dec0) < 1e-6:
        raise ValueError(
            "fit_pose_correction is ill-conditioned near the pole "
            "(|cos(dec0)| too small); pick a different anchor."
        )

    def _tangent(ra: Tensor, dec: Tensor) -> tuple[Tensor, Tensor]:
        return (
            _wrapped_delta_ra_deg(ra, ra0) * torch.cos(torch.deg2rad(dec)),
            dec - dec0,
        )

    xi_s, eta_s = _tangent(src_ra, src_dec)
    xi_d, eta_d = _tangent(dst_ra, dst_dec)

    sx = xi_s - xi_s.mean()
    sy = eta_s - eta_s.mean()
    dx = xi_d - xi_d.mean()
    dy = eta_d - eta_d.mean()
    if model == "shift":
        theta = torch.zeros((), dtype=src_ra.dtype, device=src_ra.device)
    else:
        # Maximize sum d @ R(theta) s with R = [[c, -s], [s, c]]:
        # theta = atan2(sum(dy*sx - dx*sy), sum(dx*sx + dy*sy)).
        sin_num = (dy * sx - dx * sy).sum()
        cos_den = (dx * sx + dy * sy).sum()
        theta = torch.atan2(sin_num, cos_den)
    c = torch.cos(theta)
    s = torch.sin(theta)
    tx = xi_d.mean() - (c * xi_s.mean() - s * eta_s.mean())
    ty = eta_d.mean() - (s * xi_s.mean() + c * eta_s.mean())

    pose = ExposurePose()
    with torch.no_grad():
        pose.delta_theta_rad.copy_(theta.to(pose.delta_theta_rad.device))
        pose.delta_log_scale.copy_(
            torch.zeros_like(theta, device=pose.delta_log_scale.device)
        )
        # apply_to_world adds delta_ra_deg * cos(dec0) to the xi coordinate.
        pose.delta_ra_deg.copy_((tx / cos_dec0).to(pose.delta_ra_deg.device))
        pose.delta_dec_deg.copy_(ty.to(pose.delta_dec_deg.device))
    return pose.to(device=src_ra.device, dtype=src_ra.dtype)
