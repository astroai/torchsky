"""Learnable per-exposure astrometric residual in a local tangent plane.

Small patches (typical HST/JWST cutouts) are modeled with four scalars:

- ``delta_ra_deg``, ``delta_dec_deg``: translation in plate coordinates mapped back to RA/Dec.
- ``delta_theta_rad``: in-plane rotation about the patch center.
- ``delta_log_scale``: isotropic log-scale about the center.

This is an approximate BA hook; it does not replace full spherical bundle adjustment.
Poles and very large fields may need a spherically exact formulation (research extension).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from ..wcs import WCS


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
        """Apply pose in tangent coordinates at ``(center_ra_deg, center_dec_deg)``, then map back."""
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

        xi = _wrapped_delta_ra_deg(ra_deg, ra0) * cos_dec0
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

        ra_out = ra0 + xi_out / cos_dec0
        dec_out = dec0 + eta_out

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
        """Apply :meth:`apply_to_world`, then ``wcs.world_to_pixel`` (FITS pixel convention)."""
        ra_p, dec_p = self.apply_to_world(
            ra_deg,
            dec_deg,
            center_ra_deg=center_ra_deg,
            center_dec_deg=center_dec_deg,
        )
        # CFITSIO-backed ``world_to_pixel`` does not accept tensors that require grad.
        return wcs.world_to_pixel(ra_p.detach(), dec_p.detach())


def freeze_pose(pose: ExposurePose) -> None:
    """Disable gradients on all pose parameters (e.g. anchor one exposure)."""
    for p in pose.parameters():
        p.requires_grad = False
