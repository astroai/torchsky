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
